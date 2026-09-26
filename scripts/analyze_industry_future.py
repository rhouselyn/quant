"""Descriptive horizon/industry dependence and temporal-feature persistence.

No model fitting, trading decisions, or hyperparameter selection use these future
returns. Market residuals are explanatory statistics, never available inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.data import FEATURES, dataset_arrays

HORIZONS = [1, 2, 3, 4, 5, 8]
SPLITS = {'train': (0, 364), 'validation': (364, 442), 'test': (442, 520)}


def fit_market(values, market, end):
    """Fit alpha/beta per column using training dates and valid observations only."""
    coefficients = []
    for name in values.columns:
        v = values[name].iloc[:end].to_numpy()
        m = market.iloc[:end].to_numpy()
        valid = np.isfinite(v) & np.isfinite(m)
        if valid.sum() < 30:
            coefficients.append([np.nan, np.nan])
        else:
            coefficients.append(np.linalg.lstsq(np.column_stack([np.ones(valid.sum()), m[valid]]), v[valid], rcond=None)[0])
    coef = np.asarray(coefficients)
    predicted = coef[:, 0][None] + market.to_numpy()[:, None] * coef[:, 1][None]
    return pd.DataFrame(values.to_numpy()-predicted, index=values.index, columns=values.columns), coef


def group_returns(stock_returns, groups):
    result = {}
    for name, symbols in groups.items():
        g = stock_returns[symbols]
        # Require at least 18 of the original 20 stocks. Never fill missing returns.
        result[name] = g.mean(axis=1).where(g.notna().sum(axis=1) >= 18)
    return pd.DataFrame(result)


def correlation_stats(frame):
    # Common dates yield a PSD correlation matrix, essential for PCA interpretation.
    common = frame.dropna()
    corr = common.corr()
    off = corr.to_numpy()[np.triu_indices(len(corr), 1)]
    eigenvalues = np.linalg.eigvalsh(corr.to_numpy())
    return corr, dict(n_periods=len(common), mean_corr=float(off.mean()),
        mean_abs_corr=float(np.abs(off).mean()), median_corr=float(np.median(off)),
        positive_pair_fraction=float((off > 0).mean()),
        pc1_variance_share=float(eigenvalues[-1]/eigenvalues.sum()))


def pair_stats(frame, industry_ids):
    corr = frame.corr(min_periods=20).to_numpy()
    a, b = np.triu_indices(len(industry_ids), 1)
    same = industry_ids[a] == industry_ids[b]
    pairs = corr[a, b]
    return dict(same_industry_stock_corr=float(np.nanmean(pairs[same])),
                other_industry_stock_corr=float(np.nanmean(pairs[~same])),
                same_industry_pair_count=int(np.isfinite(pairs[same]).sum()),
                other_industry_pair_count=int(np.isfinite(pairs[~same]).sum()))


def lag_stats(weekly, start, end, lag):
    correlations = []
    for name in weekly.columns:
        a = weekly[name].iloc[start:end-lag].reset_index(drop=True)
        b = weekly[name].iloc[start+lag:end].reset_index(drop=True)
        correlations.append(a.corr(b))
    return float(np.nanmean(correlations))


def future_step_matrix(weekly, start, end, steps=5):
    matrices = []
    for name in weekly.columns:
        f = pd.DataFrame({f'未来{k}周': weekly[name].shift(-k).iloc[start:end-steps].to_numpy()
                          for k in range(1, steps+1)})
        matrices.append(f.corr().to_numpy())
    return np.nanmean(matrices, axis=0).tolist()


def feature_persistence(panel, cfg):
    X, _, dates, symbols = dataset_arrays(panel, cfg['lookback'], cfg['n_features'], cfg['train_end'])
    raw = panel.set_index(['date', 'symbol']).reindex(pd.MultiIndex.from_product([dates, symbols]))[FEATURES].to_numpy().reshape(X.shape)
    rows = []
    for split, (start, end) in SPLITS.items():
        for f, feature in enumerate(FEATURES):
            correlations, input_correlations = [], []
            for t in range(max(start, 60), end-5):
                observed_input = pd.Series(X[t, :, f])
                future_input = pd.Series(X[t+1:t+6, :, f].mean(axis=0))
                if observed_input.std() > 1e-8 and future_input.std() > 1e-8:
                    input_correlations.append(observed_input.corr(future_input, method='spearman'))
                valid = np.isfinite(raw[t:t+6, :, f]).all(axis=0)
                if valid.sum() < 100:
                    continue
                current = pd.Series(X[t, valid, f])
                future = pd.Series(X[t+1:t+6, valid, f].mean(axis=0))
                corr = current.corr(future, method='spearman')
                if np.isfinite(corr):
                    correlations.append(corr)
            rows.append(dict(split=split, feature=feature, n_anchors=len(correlations),
                current_vs_future5_mean_rank_corr=float(np.mean(correlations)) if correlations else np.nan,
                actual_input_n_anchors=len(input_correlations),
                actual_input_rank_corr=float(np.nanmean(input_correlations)) if input_correlations else np.nan,
                raw_missing_fraction=float((~np.isfinite(raw[max(start, 60):end, :, f])).mean())))
    return pd.DataFrame(rows)


def report_html(out, summary, matrices, step_matrices, lag, persistence, pairs, manifest):
    from plotly.offline import get_plotlyjs
    table = summary[summary.split.eq('train')][['horizon', 'n_periods', 'mean_corr',
        'residual_mean_corr', 'residual_mean_abs_corr', 'pc1_variance_share',
        'same_industry_stock_corr', 'other_industry_stock_corr']].copy()
    table.columns = ['未来累计周数', '训练观测数', '行业间相关', '去市场后相关', '去市场后绝对相关', '首主成分比例', '同业个股相关', '跨业个股相关']
    display = table.to_html(index=False, border=0, float_format=lambda x: f'{x:.3f}')
    chosen = persistence[persistence.feature.isin(['ret1', 'ret5', 'ret60', 'vol20', 'vol60', 'vlog'])]
    feature_table = chosen.pivot(index='feature', columns='split', values='current_vs_future5_mean_rank_corr').to_html(border=0, float_format=lambda x:f'{x:.3f}', na_rep='有效样本不足')
    missing_table = chosen.pivot(index='feature', columns='split', values='raw_missing_fraction').to_html(border=0, float_format=lambda x:f'{x:.1%}')
    data = dict(matrices=matrices, future_steps=step_matrices, labels=manifest['industries'])
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>580股行业相关性与InfoNCE分析</title>
<style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1300px;margin:30px auto;padding:0 24px;color:#20394e;line-height:1.7}h1{font-size:28px}h2{margin-top:30px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #dae3eb;text-align:right}th:first-child,td:first-child{text-align:left}select{padding:7px;margin:5px 12px 5px 0}.note{background:#eef4fa;padding:18px;border-radius:10px}.plot{width:100%;min-height:560px}</style>
<h1>580股 · 29个行业：未来收益相关性与InfoNCE诊断</h1>
<div class="note">本页统计真实行情，不训练模型、不改变默认损失。行业每类20股；行业收益为原20股未来累计简单收益等权均值，至少18股有效。未来累计收益只要求起终价格有效，区间内缺失周不补价。不是行业指数。行业相关性高不等于可预测性强，也不能据此证明InfoNCE有益或有害。</div>
<h2>训练期：不同未来累计持有期</h2>TABLE
<p>行业相关矩阵使用29行业共同有效日期。去市场：各行业未来收益对29行业等权未来收益回归，alpha/beta仅以训练期估计；验证/测试固定系数。残差用到了实际未来市场收益，仅用于事后解释，不能作为实时输入。共同市场由这些行业构成，因此残差相关存在机械约束；应结合绝对相关和强相关行业对判断。</p>
<h2>29×29行业相关热图</h2>
<label>区间 <select id="split"><option value="train">训练期</option><option value="validation">验证期</option><option value="test">测试期（探索）</option></select></label>
<label>累计期限 <select id="horizon">HORIZONS</select></label>
<label>口径 <select id="mode"><option value="raw">原始收益</option><option value="market_residual">扣除训练期估计的市场成分</option></select></label>
<div id="heat" class="plot"></div><p id="matrixMeta"></p>
<h2>未来第1～5周，单周收益之间是否相似</h2>
<p>横纵轴是不同未来单周，而非累计期限；对每个行业先计算跨日期相关，再取29行业均值。使用上方区间与口径选择。</p><div id="steps" class="plot"></div>
<h2>当周行业收益 → 后续单周行业收益</h2>LAG
<p>均值为29个行业自身收益的跨期Pearson相关；仅用于描述线性持续性，没有进行预测模型测试。未来累计收益采用重叠窗口，文件另存按h个不同起点抽取的不重叠结果；不把重叠周或行业对当作独立样本，未报告朴素显著性。</p>
<h2>当前InfoNCE可能利用什么</h2>
<p>当前任务是识别同一股票的未来5周特征窗口，其他579股为负例，其中19只是同行（3.28%）。输入窗口虽不重叠，滚动特征依赖的历史却大量重叠。下表是“当前特征”与“未来5周平均特征”在同一横截面上的Spearman相关再按日期平均，使用当前训练输入的标准化。它不是收益预测IC，也不是实际embedding相关。</p>FEATURES
<p>例如ret60/vol60和成交量水平可能非常持久；InfoNCE因此可能学习身份、波动和量能指纹，而非下一周收益排序。长期/短期任务不一致并不证明辅助任务无效，但收益相关统计不能代替损失消融。</p>
<h2>输入质量：特征原值的缺失比例</h2>MISSING
<p>训练统计剔除前60周预热。长窗口rolling默认要求窗口内每周都有数据，日历中的空周会向后传播缺失。模型随后填0再标准化；验证期部分60周特征没有满足至少100股连续6周有效的相关性样本，不能把这些位置解释为市场上真实的零动量/零波动。CSV同时保存实际填值输入的持续性和有效样本数。此处仅诊断，未修改缓存或特征工程。</p>
<h2>架构判断</h2><p>最小方案建议 L→G→L→G→L→Score：L为20股+本行业代表的self-attention，G为29行业代表的self-attention。末尾L已把全局更新传回股票，股票cross-attention全部代表不是必要条件。初版无需市场token；市场状态可作为后续小门控输入单独检验。重复纯L、没有G，则没有显式跨行业消息。</p>
<p>InfoNCE不是数学必需。建议先以Score+0.5RankIC建立关系模型基线，再以相同结构、数据和训练预算比较时间InfoNCE权重0、0.1、1。若保留，先只作用于关系层之前的个股表示。未进行580股同条件训练消融，不能宣称关闭会提高收益；当前默认仍为1。</p>
<h2>边界</h2><p>训练2016-09-30～2023-09-15；验证2023-09-22～2025-03-14；测试2025-03-21～2026-09-11。每个未来终点都在其自身区间内；因此长期收益可用样本更少。当前行业快照回溯且股票有存续偏差；测试期已用于探索。本报告不改变原始缓存和默认模型。</p>
<script>PLOTLY</script><script>const d=DATA;
const selectors=['split','horizon','mode'];
function draw(){const split=document.getElementById('split').value,h=document.getElementById('horizon').value,mode=document.getElementById('mode').value;const m=d.matrices[split+'_'+h+'_'+mode];
Plotly.react('heat',[{type:'heatmap',x:d.labels,y:d.labels,z:m.corr,zmin:-1,zmax:1,colorscale:'RdBu',reversescale:true,hovertemplate:'%{y} × %{x}<br>相关=%{z:.3f}<extra></extra>'}],{height:850,margin:{l:110,r:30,t:25,b:140},xaxis:{tickangle:-55},yaxis:{autorange:'reversed'}},{responsive:true,displaylogo:false});
document.getElementById('matrixMeta').textContent='共同有效日期 '+m.n_periods+'；行业对平均相关 '+m.mean_corr.toFixed(3)+'；平均绝对相关 '+m.mean_abs_corr.toFixed(3)+'。';
const names=['未来第1周','未来第2周','未来第3周','未来第4周','未来第5周'];Plotly.react('steps',[{type:'heatmap',x:names,y:names,z:d.future_steps[split+'_'+mode],zmin:-1,zmax:1,colorscale:'RdBu',reversescale:true,texttemplate:'%{z:.3f}'}],{height:490,margin:{l:100,r:30,t:25,b:70},yaxis:{autorange:'reversed'}},{responsive:true,displaylogo:false});}
selectors.forEach(x=>document.getElementById(x).addEventListener('change',draw));draw();</script></html>'''
    lag_table = lag[lag.split.eq('train')].pivot(index='lag_weeks', columns='mode', values='own_industry_mean_corr').to_html(border=0, float_format=lambda x:f'{x:.3f}')
    html = html.replace('TABLE', display).replace('HORIZONS', ''.join(f'<option value="{h}">{h}周</option>' for h in HORIZONS))
    html = html.replace('FEATURES', feature_table).replace('MISSING', missing_table).replace('LAG', lag_table).replace('DATA', json.dumps(data, ensure_ascii=False)).replace('PLOTLY', get_plotlyjs())
    (out/'report.html').write_text(html, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='outputs/industry_580_future_analysis')
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Choose a new directory; preserve existing analyses')
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    assert cfg['n_stocks'] == 580 and cfg['train_end'] == 364 and cfg['valid_end'] == 442
    panel = pd.read_parquet(cfg['cache_path'])
    universe = pd.read_csv(cfg['universe_path']).sort_values(['industry_id', 'symbol'])
    groups = {name: g.symbol.tolist() for name, g in universe.groupby('industry_l1', sort=False)}
    assert len(groups) == 29 and all(len(g) == 20 for g in groups.values())
    close = panel.pivot(index='date', columns='symbol', values='close').sort_index()
    assert close.shape == (520, 580)
    ids = universe.set_index('symbol').loc[close.columns].industry_id.to_numpy()
    summary, matrices, nonoverlap, pairs, industry_values = [], {}, [], [], []
    for h in HORIZONS:
        stock = close.shift(-h)/close-1
        industry = group_returns(stock, groups)
        market = industry.mean(axis=1).where(industry.notna().all(axis=1))
        residual, coef = fit_market(industry, market, cfg['train_end']-h)
        stock_residual, _ = fit_market(stock, market, cfg['train_end']-h)
        for split, (start, end) in SPLITS.items():
            frame = industry.iloc[start:end-h]
            res = residual.iloc[start:end-h]
            raw_corr, stats = correlation_stats(frame)
            res_corr, res_stats = correlation_stats(res)
            row = dict(split=split, horizon=h, **stats,
                **{f'residual_{k}': v for k, v in res_stats.items()},
                **pair_stats(stock.iloc[start:end-h], ids),
                **{f'residual_{k}':v for k,v in pair_stats(stock_residual.iloc[start:end-h], ids).items()})
            summary.append(row)
            for mode, corr, s, values in [('raw', raw_corr, stats, frame), ('market_residual', res_corr, res_stats, res)]:
                matrices[f'{split}_{h}_{mode}'] = dict(corr=corr.to_numpy().tolist(), **s)
                for phase in range(h):
                    sample = values.iloc[phase::h].dropna()
                    if len(sample) >= 8:
                        _, phase_stats = correlation_stats(sample)
                        nonoverlap.append(dict(split=split, horizon=h, mode=mode, phase=phase, **phase_stats))
                a, b = np.triu_indices(len(corr), 1)
                entries = pd.DataFrame({'industry_a':corr.index[a], 'industry_b':corr.index[b], 'corr':corr.to_numpy()[a, b]})
                strongest = pd.concat([entries.nlargest(5, 'corr').assign(tail='highest'), entries.nsmallest(5, 'corr').assign(tail='lowest')])
                pairs.extend(strongest.assign(split=split, horizon=h, mode=mode).to_dict('records'))
            industry_values.append(frame.assign(split=split, horizon=h).reset_index())
        print('Completed cumulative horizon', h, flush=True)
    weekly = group_returns(close.pct_change(fill_method=None), groups)
    weekly_market = weekly.mean(axis=1).where(weekly.notna().all(axis=1))
    weekly_residual, _ = fit_market(weekly, weekly_market, cfg['train_end'])
    lag, step_matrices = [], {}
    for split, (start, end) in SPLITS.items():
        for mode, frame in [('raw', weekly), ('market_residual', weekly_residual)]:
            for k in HORIZONS:
                lag.append(dict(split=split, mode=mode, lag_weeks=k,
                    own_industry_mean_corr=lag_stats(frame, start, end, k)))
            step_matrices[f'{split}_{mode}'] = future_step_matrix(frame, start, end)
    persistence = feature_persistence(panel, cfg)
    manifest = dict(n_stocks=580, n_industries=29, industries=list(groups), horizons=HORIZONS,
        source_cache=cfg['cache_path'], source_sha256=hashlib.sha256(Path(cfg['cache_path']).read_bytes()).hexdigest(),
        split_indices=SPLITS, same_industry_negatives=19, total_negatives=579,
        stock_pair_min_periods=20, industry_min_valid_stocks=18,
        market_regression='alpha/beta train-only per horizon; future market realized returns used only for descriptive residuals',
        future_endpoint_policy='t+h must be strictly before split end; no padding of prices or returns',
        comparison='descriptive only; no model retraining or InfoNCE ablation', default_temporal_weight=cfg['temporal_weight'])
    out.mkdir(parents=True)
    summary = pd.DataFrame(summary)
    lag, pairs = pd.DataFrame(lag), pd.DataFrame(pairs)
    summary.to_csv(out/'horizon_summary.csv', index=False)
    pd.DataFrame(nonoverlap).to_csv(out/'nonoverlapping_horizon_summary.csv', index=False)
    pairs.to_csv(out/'strongest_industry_pairs.csv', index=False, encoding='utf-8-sig')
    lag.to_csv(out/'lag_correlations.csv', index=False)
    persistence.to_csv(out/'feature_persistence.csv', index=False)
    pd.concat(industry_values, ignore_index=True).to_parquet(out/'industry_forward_returns.parquet', index=False)
    for name, data in [('manifest',manifest), ('industry_matrices',matrices), ('future_step_matrices',step_matrices)]:
        (out/f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    report_html(out, summary, matrices, step_matrices, lag, persistence, pairs, manifest)
    print(summary[['split','horizon','n_periods','mean_corr','residual_mean_corr','residual_mean_abs_corr','pc1_variance_share','same_industry_stock_corr','other_industry_stock_corr']].round(4).to_string(index=False))
    print('Report:', out/'report.html')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
