"""Render completed trials and descriptive style diagnostics; never select a model."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from threadpoolctl import threadpool_limits
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.style import neutralize

ROOT = Path('outputs/hierarchical_580_comparison')
NAMES = {'off': '关闭 InfoNCE', 'h2': '未来两周 InfoNCE', 'h5': '未来五周 InfoNCE'}
PROXIES = {'ret1': '近1观测周收益', 'ret20': '20观测周动量', 'ret60': '60观测周动量',
           'vol20': '20观测周波动率', 'vlog': '对数成交股数', 'amount_z20': '相对成交金额'}


def diagnose(predictions, panel, universe, method, split):
    predictions = predictions.copy()
    predictions['date'] = pd.to_datetime(predictions.date)
    merged = predictions.merge(panel[['date', 'symbol', *PROXIES]], on=['date', 'symbol'],
                               validate='one_to_one').merge(universe[['symbol', 'industry_l1']],
                                                           on='symbol', validate='many_to_one')
    if len(merged) != len(predictions):
        raise ValueError('Diagnostics must retain every prediction row')
    records = []
    for date, group in merged.groupby('date', sort=True):
        group = group[np.isfinite(group.score)].sort_values(['score', 'symbol'], ascending=[False, True])
        if len(group) < 40:
            continue
        ranks = group[list(PROXIES)].rank(pct=True)
        if ranks.isna().any().any():
            raise ValueError('Current valid scores require available diagnostic features')
        industry = pd.get_dummies(group.industry_l1, dtype=float).to_numpy()
        styles = np.column_stack([np.ones(len(group)), ranks.to_numpy()])
        raw = group.score.to_numpy()
        record = dict(method=method, split=split, date=str(date.date()), stocks=len(group),
                      industry_r2=neutralize(raw, industry)[1],
                      proxy_r2=neutralize(raw, styles)[1],
                      combined_r2=neutralize(raw, np.column_stack([industry, ranks]))[1],
                      top20_max_industry_weight=group.head(20).industry_l1.value_counts(normalize=True).max())
        record.update({f'spearman_{feature}': group.score.corr(group[feature], method='spearman')
                       for feature in PROXIES})
        records.append(record)
    return pd.DataFrame(records)


def result_table(rows):
    columns = {'method': '方案', 'best_epoch': '最佳轮次', 'validation_sharpe': '验证净 Sharpe',
               'rank_ic': '测试 RankIC', 'total_return': '测试累计净收益',
               'annualized_return': '测试年化净收益', 'sharpe': '测试净 Sharpe',
               'max_drawdown': '测试最大回撤', 'turnover': '周均双边换手'}
    table = pd.DataFrame(rows)[list(columns)].copy()
    table['method'] = table.method.map(NAMES)
    for column in ['total_return', 'annualized_return', 'max_drawdown', 'turnover']:
        table[column] = table[column].map(lambda value: f'{value:.2%}')
    return table.rename(columns=columns).to_html(index=False, border=0, float_format=lambda value: f'{value:.4f}')


def render():
    report = json.loads((ROOT / 'comparison.json').read_text(encoding='utf-8'))
    selection = json.loads((ROOT / 'selection.json').read_text(encoding='utf-8'))
    if selection['test_consulted'] or selection['selected'] != report['selected']:
        raise ValueError('Expected a frozen validation-only selection')
    diagnostics = []
    for row in report['rows']:
        out = Path('outputs') / row['run']
        cfg = yaml.safe_load((out / 'config.yaml').read_text(encoding='utf-8'))
        panel = pd.read_parquet(cfg['cache_path'])
        panel['date'] = pd.to_datetime(panel.date)
        universe = pd.read_csv(cfg['universe_path'])
        for split, name in [('validation', 'validation_predictions.parquet'), ('test', 'predictions.parquet')]:
            predictions = pd.read_parquet(out / name)
            # Audit the realized returns used for evaluation against the original data.
            original = panel[['date', 'symbol', 'target']].copy()
            check = predictions.copy(); check['date'] = pd.to_datetime(check.date)
            check = check.merge(original, on=['date', 'symbol'], validate='one_to_one')
            np.testing.assert_allclose(check.realized_return, check.target, rtol=1e-6, atol=1e-8, equal_nan=True)
            diagnostics.append(diagnose(predictions, panel, universe, row['method'], split))
    diagnostics = pd.concat(diagnostics, ignore_index=True)
    diagnostics.to_csv(ROOT / 'style_diagnostics_weekly.csv', index=False)
    summary = diagnostics.drop(columns=['date', 'stocks']).groupby(['split', 'method'], sort=False).mean().reset_index()
    summary.to_csv(ROOT / 'style_diagnostics.csv', index=False)
    curves = pd.read_csv(ROOT / 'curves.csv')
    figure = go.Figure([go.Scatter(x=curves.date, y=curves[key], name=NAMES[key]) for key in NAMES])
    figure.update_layout(template='plotly_white', height=440, margin=dict(t=30,l=55,r=20,b=45),
                         yaxis_title='Top20 净值', hovermode='x unified', legend=dict(orientation='h'))
    winner = NAMES[report['selected']]
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>580股分层模型 · InfoNCE对照</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1250px;margin:30px auto;padding:0 24px;color:#20354b;line-height:1.75;background:#f6f8fb}
h1{font-size:28px}h2{font-size:21px}.panel{background:#fff;padding:22px;margin:18px 0;border:1px solid #dce5ee;border-radius:12px;overflow:auto}
table{border-collapse:collapse;font-size:14px;width:100%}th,td{text-align:right;padding:10px;border-bottom:1px solid #e1e7ee;white-space:nowrap}th:first-child,td:first-child{text-align:left}
.muted{color:#697b8b}a{color:#246fb4}.flow{font-size:18px;font-weight:600;color:#246fb4}code{background:#edf2f7;padding:2px 5px}</style>
<h1>580股 · 行业分层注意力＋一个市场 token</h1>'''
    html += f'<div class="panel"><strong>已完成三组各10轮；验证集选定：{winner}。</strong><p>默认配置已切换至验证胜出方案。测试结果用于对照展示，不改变选择。所有组使用相同580股、数据处理、初始种子42、训练日期、Top20策略和单边10bps成交成本。</p></div>'
    html += '<div class="panel"><h2>验证选择与测试表现</h2>' + result_table(report['rows']) + '</div>'
    html += '<div class="panel"><h2>测试期 Top20 净值</h2>' + figure.to_html(full_html=False, include_plotlyjs=True) + '</div>'
    for split, label in [('validation', '验证期'), ('test', '测试期')]:
        table = summary[summary.split.eq(split)].copy()
        table['method'] = table.method.map(NAMES)
        cols = {'method': '方案', 'industry_r2': '行业解释比例', 'proxy_r2': '行情代理解释比例',
                'combined_r2': '两者联合解释比例', 'top20_max_industry_weight': 'Top20最大行业占比'}
        table = table[list(cols)].rename(columns=cols)
        for col in list(cols.values())[1:]:table[col] = table[col].map(lambda value: f'{value:.2%}')
        html += f'<div class="panel"><h2>{label}分数暴露诊断</h2>' + table.to_html(index=False, border=0)
        html += '<p class="muted">逐周横截面回归的样本内R²及持仓行业集中度，表中为周均值。行情代理包括近1周收益、20/60周动量、20周波动、对数成交股数、相对成交金额；不是完整风险模型，也不能将解释比例视作有害风格的比例。行业与行情分别回归，解释比例有重叠，不能相加。当前行业分类为回溯快照。对比的是各组验证选定的模型，最佳轮次可能不同，不能完全隔离训练轮次的影响。</p></div>'
    html += '''<div class="panel"><h2>实际模型与两周目标的含义</h2>
<p class="flow">40周 × 24特征 → 时序编码 → 行业内 → 行业间 → 行业内 → 行业间 → 行业内 → 单头 score</p>
<p>每个行业20股与一个可学习代表做self-attention；29个行业代表与一个市场token做self-attention。代表是端到端学习的参数加动态均值，并未单独预训练。股票通过本行业代表接收其他行业信息，最后一次行业内更新将最近的全局信息传回股票。</p>
<p>市场token由可学习向量、行业均值与六个当时可知的市场统计初始化。每个日期重新计算，不跨日期保留隐藏状态。所有方案均含市场token，因此这次对照不能单独判断市场token的增益。</p>
<p>InfoNCE作用于关系层之前的个股时序表示，经独立投影头计算。H2把t+1、t+2两个未来周的24维特征一起编码成一个正样本；H5对应五周。同日期其他有效股票作为负例，目标编码器使用EMA。预测阶段不读取未来行情。</p>
<p>未来周的24维特征仍含60周动量等历史滚动统计，所以“两周key”并非只含未来两周新增收益。它更贴近下一周任务，但也可能更容易匹配持续的股票风格；是否有用需要看对照结果，不能只看InfoNCE下降。</p></div>
<div class="panel"><h2>数据与判断边界</h2><p>600股中剔除不足20股的综合、美容护理两类，保留29×20＝580股。以真实观测周计算滚动特征，修复节假日空周破坏60周特征的问题；只前向携带历史特征。训练标签可截断，验证与测试回测使用原始未截断收益，报告生成时再次逐条核对。</p>
<p>同一数据版本下的单种子探索；测试期此前已有分析用途，不是全新独立留出集。现有股票池存在存续筛选和行业快照回溯限制。回测为周收盘到收盘、收益缺失记平并计数，不代表实际下单执行。与旧500股模型同时改变了股票池和特征处理，不能据此单独归因于注意力架构。</p></div>
<p><a href="/industry-analysis">行业相关性分析</a> · <a href="/">全部实验</a></p></html>'''
    (ROOT / 'report.html').write_text(html, encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        render()
