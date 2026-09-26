"""Audit the saved single-head baseline and fixed score-neutralization variants.

Run from repository root. Does not retrain, change historical outputs, or tune on test.
Industry variants use a current classification snapshot and are retrospective only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.style import STYLE_COLUMNS, style_panel, design_matrix, neutralize
from quant_mvp.long_only import long_only_backtest, performance

LABELS = {'raw': '原始单头', 'styles_half': '行情风格剔除50%', 'styles': '行情风格剔除100%',
          'industry': '仅行业中性化（回溯）', 'combined_half': '行业+行情剔除50%（回溯）',
          'combined': '行业+行情剔除100%（回溯）'}
FACTORS = dict(zip(STYLE_COLUMNS, ['近1周涨跌', '12至1周动量', '52至4周动量',
    '20周波动率', '52周市场Beta', '20周成交金额', '20周非流动性']))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block_interval(values, block=4, samples=3000):
    """Paired circular moving-block bootstrap CI for mean weekly differences."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(values), size=(samples, int(np.ceil(len(values)/block))))
    indices = ((starts[..., None] + np.arange(block)) % len(values)).reshape(samples, -1)
    estimates = values[indices[:, :len(values)]].mean(axis=1)
    return np.quantile(estimates, [.025, .975]).tolist()


def analyze(pred, exposures, split):
    pred = pred.copy()
    pred.date = pd.to_datetime(pred.date)
    data = pred.merge(exposures, on=['date', 'symbol'], how='left', validate='one_to_one', indicator=True)
    assert data._merge.eq('both').all()
    assert data.score.notna().all() and data.groupby('date').size().eq(500).all()
    frames = {key: [] for key in LABELS}
    weekly, coefficients, industry_weights, ic_rows = [], [], [], []
    for date, group in data.groupby('date', sort=True):
        group = group.sort_values('symbol').reset_index(drop=True)
        raw = group.score.to_numpy()
        designs = {key: design_matrix(group, columns=cols, industry=industry)[0]
                   for key, cols, industry in [('styles', STYLE_COLUMNS, False),
                                               ('industry', [], True), ('combined', STYLE_COLUMNS, True)]}
        scores = {'raw': raw}
        r2s = {}
        for key, design in designs.items():
            scores[key], r2s[key] = neutralize(raw, design)
        scores['styles_half'] = (raw + scores['styles']) / 2
        scores['combined_half'] = (raw + scores['combined']) / 2
        z = design_matrix(group)[1]
        for method, values in scores.items():
            p = group[['date', 'symbol', 'realized_return']].copy()
            p['score'] = values
            frames[method].append(p)
            selected = p.sort_values(['score', 'symbol'], ascending=[False, True]).head(20).index
            industry_share = group.loc[selected, 'industry'].value_counts(normalize=True)
            bench_share = group.industry.value_counts(normalize=True)
            post_r2 = neutralize(values, designs['combined'])[1]
            style_r2 = neutralize(values, designs['styles'])[1]
            ic = p.score.corr(p.realized_return, method='spearman')
            ic_rows.append(dict(split=split, date=date, method=method, rank_ic=ic,
                                normal_ic=p.score.corr(p.realized_return)))
            weekly.append(dict(split=split, date=date, method=method,
                raw_style_r2=r2s['styles'], raw_industry_r2=r2s['industry'], raw_combined_r2=r2s['combined'],
                remaining_combined_r2=post_r2, remaining_style_r2=style_r2,
                top20_max_industry_weight=industry_share.max(),
                top20_industry_active_l1=(industry_share.reindex(bench_share.index, fill_value=0)-bench_share).abs().sum(),
                raw_top20_overlap=len(set(selected) & set(p.assign(score=raw).sort_values(
                    ['score', 'symbol'], ascending=[False, True]).head(20).index))/20))
            for factor in STYLE_COLUMNS:
                coefficients.append(dict(split=split, date=date, method=method, factor=factor,
                    spearman=pd.Series(values).corr(group[factor], method='spearman'),
                    pearson_rank_style=pd.Series(values).corr(z[factor]),
                    top20_exposure_sigma=z.loc[selected, factor].mean()))
            for industry in bench_share.index:
                industry_weights.append(dict(split=split, date=date, method=method, industry=industry,
                    weight=industry_share.get(industry, 0.), benchmark_weight=bench_share[industry]))
    curves, summary, predictions = {}, [], {}
    for method, parts in frames.items():
        p = pd.concat(parts, ignore_index=True)
        p.date = p.date.dt.strftime('%Y-%m-%d')
        curve, metrics = long_only_backtest(p, top_n=20, cost_bps=10, periods_per_year=52)
        curves[method] = curve
        predictions[method] = p
        ics = [row for row in ic_rows if row['method'] == method]
        summary.append(dict(split=split, method=method, label=LABELS[method],
            rank_ic=float(np.mean([row['rank_ic'] for row in ics])),
            normal_ic=float(np.mean([row['normal_ic'] for row in ics])),
            **metrics['net'], relative_return=metrics['relative_return'],
            mean_turnover=metrics['mean_turnover'], missing_held_returns=metrics['missing_held_returns'],
            periods=metrics['periods']))
    return pd.DataFrame(summary), curves, predictions, pd.DataFrame(weekly), pd.DataFrame(coefficients), pd.DataFrame(industry_weights), pd.DataFrame(ic_rows)


def make_html(out, manifest, table, exposure, weekly, curves, deltas, industry, halves):
    import plotly.graph_objects as go
    import plotly.io as pio
    test = table[table.split.eq('test')].copy()
    val = table[table.split.eq('validation')].copy()
    def display_table(frame):
        result = frame[['label', 'rank_ic', 'total_return', 'annualized_return', 'sharpe',
                        'max_drawdown', 'relative_return', 'mean_turnover']].copy()
        result.columns = ['方法', 'RankIC', '累计净收益', '年化净收益', '净Sharpe', '最大回撤', '相对股票池收益', '周均双边换手']
        for col in ['累计净收益', '年化净收益', '最大回撤', '相对股票池收益', '周均双边换手']:
            result[col] = result[col].map(lambda x: f'{x:.2%}')
        return result.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0)
    curve_fig = go.Figure()
    for key in LABELS:
        curve = curves['test'][key]
        curve_fig.add_trace(go.Scatter(x=curve.date, y=curve.long_net_equity, name=LABELS[key],
                                       visible='legendonly' if key in ['industry', 'combined_half'] else True))
    curve_fig.add_trace(go.Scatter(x=curve.date, y=curve.benchmark_net_equity, name='500股等权基准', line=dict(dash='dash')))
    curve_fig.update_layout(title='测试期 Top20 净值 · 点击图例切换方案', template='plotly_white', height=470)
    corr = exposure[exposure.split.eq('test')].groupby(['factor', 'method']).spearman.mean().unstack().reindex(STYLE_COLUMNS)
    heat = go.Figure(go.Heatmap(z=corr[list(LABELS)].to_numpy(),
        x=list(LABELS.values()), y=[FACTORS[x] for x in corr.index], zmid=0, colorscale='RdBu',
        text=np.round(corr[list(LABELS)].to_numpy(), 3), texttemplate='%{text}'))
    heat.update_layout(title='平均每周 Spearman：score 与行情风格', template='plotly_white', height=480)
    exp = exposure[exposure.split.eq('test')].groupby(['factor', 'method']).top20_exposure_sigma.mean().unstack().reindex(STYLE_COLUMNS)
    exp_fig = go.Figure()
    for method in ['raw', 'styles_half', 'styles', 'combined']:
        exp_fig.add_trace(go.Bar(x=[FACTORS[x] for x in exp.index], y=exp[method], name=LABELS[method]))
    exp_fig.update_layout(title='Top20持仓风格偏离股票池 · 横截面排名标准差单位', template='plotly_white', height=430)
    w = weekly[(weekly.split == 'test') & (weekly.method == 'raw')]
    diagnostics = {name: float(w[name].mean()) for name in ['raw_style_r2', 'raw_industry_r2', 'raw_combined_r2']}
    delta_text = ''.join(f'<li>{LABELS[d["method"]]}：相对原始组合的周均收益差 {d["weekly_mean_difference"]:.3%}，'
                         f'95%区间 [{d["ci95"][0]:.3%}, {d["ci95"][1]:.3%}]</li>' for d in deltas)
    top_ind = industry[(industry.split=='test') & (industry.method=='raw')].groupby('industry')[['weight', 'benchmark_weight']].mean().sort_values('weight', ascending=False).head(8)
    top_ind['active_weight'] = top_ind.weight-top_ind.benchmark_weight
    limitations = '''<ul><li>只检验7类行情代理和申万行业，不含历史市值、账面市值比、盈利、成长、杠杆等基本面，不能声称消除全部风格。</li>
<li>行情代理在信号当周收盘可见；成交金额沿用Yahoo缓存估计值，非交易所核验成交额。市场Beta基准是500股等权收益。</li>
<li>行业来自2026年分类快照回溯；缺失归为Unknown，不剔除股票。包含行业的方案仅作后验解释，不能视为无前视的交易验证。</li>
<li>R²是每周横截面对分数的解释比例，不是策略收益中风格贡献比例；各项R²不能相加。低相关也不等于不存在非线性风格。RankIC统一用并列值平均排名，旧引擎用双argsort打破并列，基线差约0.000004。</li>
<li>OLS使残差分数与所用设计矩阵线性正交；再取Top20后持仓仍可暴露于风格，并不等于组合风险中性。50%表示减半拟合成分，R²不一定减半。</li>
<li>复用epoch8和原有验证期，测试期此前已查看，多方案对照为探索性分析，置信区间不修正多重比较。</li>
<li>使用原预测中的±25%截断对数收益以严格复现基线；原股票池有存续筛选偏差。每周收盘到收盘、10bps双边成交成本，无涨跌停/停牌成交限制和末期清仓。</li></ul>'''
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>500股单头score风格分析</title>
<style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:35px auto;max-width:1280;padding:0 24px;color:#19324b;line-height:1.7}}h1{{font-size:28px}}h2{{margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dce4ed;text-align:right}}td:first-child,th:first-child{{text-align:left}}.note{{background:#eef4fa;padding:18px;border-radius:10px}}li{{margin:6px 0}}</style>
<h1>500股 · 单头 score：风格暴露与中性化对照</h1>
<p>复用 real_weekly_500_full_rankic_10ep / best.pt（epoch8），不重训。测试信号 {manifest['test_start']} ～ {manifest['test_end']}，77周；原500股、Top20等权。</p>
<div class="note">按验证期净Sharpe预先选定的行情方案：<b>{LABELS[manifest['validation_selected_method']]}</b>。
原分数的每周平均解释度：行情7因子 <b>{diagnostics['raw_style_r2']:.1%}</b>，行业 <b>{diagnostics['raw_industry_r2']:.1%}</b>，合并 <b>{diagnostics['raw_combined_r2']:.1%}</b>。行业覆盖 {manifest['industry_coverage']}/500。</div>
<h2>测试期：同一基线、相同交易成本</h2>{display_table(test)}
{pio.to_html(curve_fig, full_html=False, include_plotlyjs=True)}
<h2>验证期：2023-09-01 ～ 2025-03-07</h2>{display_table(val)}
<p>只在原始、行情剔除50%、行情剔除100%三者中按验证净Sharpe比较；行业快照方案不参与选择。方案及强度固定，不按测试表现继续搜参。</p>
{pio.to_html(heat, full_html=False, include_plotlyjs=False)}
{pio.to_html(exp_fig, full_html=False, include_plotlyjs=False)}
<h2>原Top20平均行业权重</h2>{top_ind.to_html(float_format=lambda x:f'{x:.2%}', border=0)}
<h2>收益差是否稳定</h2><p>配对4周循环分块bootstrap，3000次，随机种子42；区间针对周均净收益差，不是累计收益。</p><ul>{delta_text}</ul>
<p>各半段独立复利，沿用完整回测的当期换手成本：</p>{halves.to_html(index=False, float_format=lambda x:f'{x:.4f}', border=0)}
<h2>定义与边界</h2><p>每周仅用当时行情构造7因子，横截面转百分位排名并标准化，缺失以当周中位数补齐；OLS拟合score，减去50%或100%的拟合偏离，再按新分数选Top20。任何收益标签都不参与中性化或选股。</p>
<p>近1周涨跌；过去12至1周动量；52至4周动量；20周简单收益波动率；52周Beta；20周平均成交金额取对数；20周平均绝对收益/成交金额。窗口按缓存中的周观测计数。</p>{limitations}
<p>复现：<code>python scripts/analyze_single_score_style.py --output outputs/另一个新目录</code>。数据、源权重hash、逐周暴露、分数、回测和汇总保存在本报告同目录。</p></html>'''
    (out/'report.html').write_text(body, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='outputs/single_score_500_style_analysis')
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Use a new output directory; historical analyses are preserved')
    source = Path('outputs/real_weekly_500_full_rankic_10ep')
    cfg = yaml.safe_load((source/'config.yaml').read_text())
    assert cfg['n_stocks'] == 500 and cfg['n_score_heads'] == 1 and not cfg.get('stock_weight_path')
    validation_path = Path('data/reliable500_top150/teacher_validation.parquet')
    validation_manifest = json.loads(Path('data/reliable500_top150/manifest.json').read_text())
    assert validation_manifest['teacher_sha256'] == sha(source/'best.pt')
    classification_path = Path('data/sw_file_20/classification_audit.csv')
    mapping = pd.read_csv(classification_path, dtype={'code': str})
    mapping['symbol'] = mapping.code + np.where(mapping.code.str.startswith('6'), '.SS', '.SZ')
    mapping = mapping.rename(columns={'industry_l1': 'industry'})[['symbol', 'industry']]
    assert not mapping.symbol.duplicated().any()
    panel = pd.read_parquet(cfg['cache_path'])
    exposures = style_panel(panel).merge(mapping, on='symbol', how='left', validate='many_to_one')
    coverage = exposures.drop_duplicates('symbol').industry.notna().sum()
    exposures.industry = exposures.industry.fillna('Unknown')
    manifest = dict(source=str(source), checkpoint_sha256=sha(source/'best.pt'),
        cache_sha256=sha(cfg['cache_path']), validation_sha256=sha(validation_path),
        test_sha256=sha(source/'predictions.parquet'), classification_sha256=sha(classification_path),
        industry_coverage=int(coverage), industry_policy='current snapshot, retrospective exploratory only',
        styles=STYLE_COLUMNS, fixed_methods=list(LABELS), selection_candidates=['raw', 'styles_half', 'styles'],
        score_normalization='none; subtract OLS fitted component minus cross-sectional score mean',
        factor_normalization='cross-sectional percentile rank z-score; same-date median imputation',
        full_style_neutralization=False, retrained=False)
    results = {}
    # Select the candidate on validation before loading or evaluating test predictions.
    results['validation'] = analyze(pd.read_parquet(validation_path), exposures, 'validation')
    eligible = results['validation'][0]
    eligible = eligible[eligible.method.isin(manifest['selection_candidates'])]
    manifest['validation_selected_method'] = eligible.sort_values('sharpe', ascending=False, kind='stable').iloc[0].method
    print('Validation candidate frozen:', manifest['validation_selected_method'], flush=True)
    test = pd.read_parquet(source/'predictions.parquet')
    manifest.update(test_start=test.date.min(), test_end=test.date.max())
    results['test'] = analyze(test, exposures, 'test')
    baseline = json.loads((source/'metrics.json').read_text())
    raw_row = results['test'][0].query('method == "raw"').iloc[0]
    for key in ['total_return', 'sharpe', 'max_drawdown']:
        np.testing.assert_allclose(raw_row[key], baseline['long_only']['net'][key], rtol=1e-10)
    from quant_mvp.engine import rank_ic
    legacy_ic = np.mean([rank_ic(g.score.to_numpy(), g.realized_return.to_numpy())
                         for _, g in test.groupby('date')])
    np.testing.assert_allclose(legacy_ic, baseline['rank_ic_mean'], rtol=1e-10)
    manifest['legacy_rank_ic'] = float(legacy_ic)
    manifest['standard_spearman_rank_ic'] = float(raw_row.rank_ic)
    np.testing.assert_allclose(results['validation'][0].query('method == "raw"').iloc[0].sharpe,
                               baseline['selection_value'], rtol=1e-9)
    summary = pd.concat([r[0] for r in results.values()], ignore_index=True)
    weekly = pd.concat([r[3] for r in results.values()], ignore_index=True)
    exposure = pd.concat([r[4] for r in results.values()], ignore_index=True)
    industry = pd.concat([r[5] for r in results.values()], ignore_index=True)
    curves = {split: r[1] for split, r in results.items()}
    base_returns = curves['test']['raw'].long_net_return.to_numpy()
    deltas, half_rows = [], []
    for method in LABELS:
        returns = curves['test'][method].long_net_return.to_numpy()
        if method != 'raw':
            diff = returns - base_returns
            deltas.append(dict(method=method, weekly_mean_difference=float(diff.mean()), ci95=block_interval(diff)))
        for half, series in [('前38周', returns[:38]), ('后39周', returns[38:])]:
            half_rows.append(dict(method=LABELS[method], half=half, **performance(series)))
    halves = pd.DataFrame(half_rows)
    out.mkdir(parents=True)
    summary.to_csv(out/'summary.csv', index=False, encoding='utf-8-sig')
    weekly.to_csv(out/'weekly_diagnostics.csv', index=False)
    exposure.to_csv(out/'weekly_style_exposures.csv', index=False)
    industry.to_csv(out/'weekly_industry_weights.csv', index=False, encoding='utf-8-sig')
    exposure.groupby(['split', 'method', 'factor'])[['spearman', 'pearson_rank_style', 'top20_exposure_sigma']].mean().to_csv(out/'mean_style_exposures.csv')
    halves.to_csv(out/'test_half_metrics.csv', index=False, encoding='utf-8-sig')
    exposures.to_parquet(out/'style_inputs.parquet', index=False)
    for split, result in results.items():
        for method in LABELS:
            result[1][method].to_csv(out/f'{split}_{method}_curve.csv', index=False)
        pd.concat([p.assign(method=method) for method, p in result[2].items()], ignore_index=True).to_parquet(out/f'{split}_predictions.parquet', index=False)
        result[6].to_csv(out/f'{split}_weekly_ic.csv', index=False)
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'paired_bootstrap.json').write_text(json.dumps(deltas, indent=2), encoding='utf-8')
    make_html(out, manifest, summary, exposure, weekly, curves, deltas, industry, halves)
    print(summary.drop(columns='label').to_string(index=False))
    print('Report:', out/'report.html')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
