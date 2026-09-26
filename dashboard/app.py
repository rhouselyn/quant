"""Interactive dashboard for quant MVP experiment outputs.

Run from the repository root with::

    python dashboard/app.py --output-root outputs

The app uses Plotly from its public CDN and Flask, so it needs no frontend
build step. It discovers every output directory containing metrics.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template_string, send_file
try:
    from quant_mvp.engine import signal_quality_metrics
except Exception:
    # Running ``python dashboard/app.py`` from the repository root does not
    # automatically add ``src`` to sys.path; make the optional lazy metrics
    # calculation work in that invocation too.
    _src = str(Path(__file__).resolve().parents[1] / 'src')
    if _src not in sys.path:
        sys.path.insert(0, _src)
    try:
        from quant_mvp.engine import signal_quality_metrics
    except Exception:
        signal_quality_metrics = None

PAGE = r'''
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quant MVP · 回测监控台</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0b1220;--panel:#111c2e;--muted:#9fb0c5;--text:#edf4ff;--line:#22344d;--accent:#4aa8ff;--good:#43d19e;--warn:#ffc857}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#09111e,#0d1728 60%,#101c30);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}.wrap{max-width:1500px;margin:0 auto;padding:26px 28px 40px}header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:22px}h1{margin:0 0 7px;font-size:29px;letter-spacing:.2px}.subtitle{color:var(--muted);font-size:14px}.toolbar{display:flex;align-items:center;gap:9px;color:var(--muted);font-size:13px}select,button{border:1px solid var(--line);border-radius:8px;background:#16253a;color:var(--text);padding:9px 11px;font-size:13px}button{cursor:pointer;background:#1c5487;border-color:#2b76b2}button:hover{background:#236ba7}.notice{display:none;border:1px solid #694e21;color:#ffd884;background:#3b2d14;border-radius:9px;padding:10px 13px;margin-bottom:16px;font-size:13px}.cards{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:11px;margin:0 0 17px}.card{background:rgba(17,28,46,.93);border:1px solid var(--line);border-radius:11px;padding:14px 15px;min-height:78px}.label{color:var(--muted);font-size:12px;margin-bottom:8px}.value{color:var(--text);font-size:22px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.value.good{color:var(--good)}.value.warn{color:var(--warn)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.panel{background:rgba(17,28,46,.94);border:1px solid var(--line);border-radius:11px;padding:15px 16px 10px;min-width:0}.panel h2{font-size:15px;margin:0 0 7px;font-weight:600}.plot{height:350px}.wide{grid-column:1/-1}.small{height:310px}.meta{margin-top:12px;color:var(--muted);font-size:12px;line-height:1.7}.tables{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid var(--line);padding:7px 6px;text-align:right}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}td.up{color:var(--good)}td.down{color:#ff7882}.empty{display:flex;align-items:center;justify-content:center;height:260px;color:var(--muted);font-size:13px}footer{color:#72849c;font-size:11px;padding:20px 2px 0}@media (max-width:950px){.cards{grid-template-columns:repeat(3,1fr)}.grid{grid-template-columns:1fr}.wide{grid-column:auto}.tables{grid-template-columns:1fr}header{align-items:flex-start;flex-direction:column}}@media (max-width:520px){.wrap{padding:18px 12px}.cards{grid-template-columns:repeat(2,1fr)}}
</style></head>
<body><main class="wrap">
<header><div><h1>Quant MVP · 回测监控台</h1><div class="subtitle">训练过程、横截面排序和样本外净值一览</div></div><div class="toolbar"><a href="/style-analysis" target="_blank" style="color:#72b7ff;white-space:nowrap">单头风格分析</a><label for="experiment">实验</label><select id="experiment"></select><label for="dateSelector">查看周期</label><select id="dateSelector" title="查看最近 10 个交易周期的预测"></select><button id="refresh">刷新</button></div></header>
<div id="notice" class="notice"></div>
<section id="longOnlySection" hidden>
<h2 id="longOnlyTitle">Top-20 纯多头</h2>
<div id="longCards" class="cards"></div>
<div class="grid"><div class="panel"><h2>做多净值与基准</h2><div id="longEquityPlot" class="plot"></div></div><div class="panel"><h2>相对基准表现</h2><div id="longExcessPlot" class="plot"></div></div></div>
<p id="longAssumptions" class="meta"></p><div class="panel"><h2>验证集选模</h2><p id="selectionMeta" class="meta"></p><div id="selectionTable"></div></div>
</section>
<h2>因子与多空辅助分析</h2><section class="cards"><div class="card"><div class="label">RankIC 均值</div><div id="rankic" class="value">—</div></div><div class="card"><div class="label">Normal IC</div><div id="normalIc" class="value">—</div></div><div class="card"><div class="label">ICIR</div><div id="icir" class="value">—</div></div><div class="card"><div class="label">IC t-stat / p</div><div id="icSig" class="value">—</div></div><div class="card"><div class="label">Top-Bottom Spread</div><div id="spread" class="value">—</div></div><div class="card"><div class="label">净多空最终净值</div><div id="equity" class="value good">—</div></div><div class="card"><div class="label">净 Sharpe</div><div id="sharpe" class="value">—</div></div><div class="card"><div class="label">最大回撤</div><div id="drawdown" class="value warn">—</div></div><div class="card"><div id="universeLabel" class="label">股票 / 周数</div><div id="universe" class="value">—</div></div><div class="card"><div class="label">模型 / 设备</div><div id="backend" class="value" style="font-size:15px">—</div></div></section>
<section class="grid"><div class="panel wide"><h2>样本外净值曲线</h2><div id="equityPlot" class="plot"></div></div><div class="panel"><h2>训练损失与验证 RankIC</h2><div id="trainingPlot" class="plot"></div></div><div class="panel" id="gatePanel" hidden><h2>关系层门控 tanh(θ) 与验证 ICIR</h2><div id="gatePlot" class="plot"></div></div><div class="panel" id="etaPanel" hidden><h2>分数的行业暴露 η² 与验证 RankIC</h2><div id="etaPlot" class="plot"></div></div><div class="panel"><h2 id="rankicTitle">每周 RankIC</h2><div id="rankicPlot" class="plot"></div></div><div class="panel"><h2>因子信号质量</h2><div id="qualityPlot" class="plot"></div><div id="qualityMeta" class="meta"></div></div><div class="panel wide"><h2 id="scatterTitle">最新交易周：模型分数与实际收益</h2><div id="scatterPlot" class="plot small"></div><div id="latestMeta" class="meta"></div></div><div class="panel wide"><h2 id="rankingTitle">最新交易周排名</h2><div class="tables"><div><div class="label">做多候选（分数最高）</div><div id="longTable"></div></div><div><div class="label">做空候选（分数最低）</div><div id="shortTable"></div></div></div></div></section>
<footer>数据由本地 outputs 目录自动发现；图表只展示实验产物，不代表未来收益。选择其他实验或重新运行训练后点击“刷新”。</footer>
</main>
<script>
const plotConfig={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']};const layoutBase={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#dbe8f8',size:11},margin:{l:48,r:20,t:16,b:40},hovermode:'x unified',xaxis:{gridcolor:'#22344d',zerolinecolor:'#22344d'},yaxis:{gridcolor:'#22344d',zerolinecolor:'#22344d'}};const pct=(v,digits=2)=>Number.isFinite(v)?(v*100).toFixed(digits)+'%':'—';const num=(v,digits=3)=>Number.isFinite(v)?Number(v).toFixed(digits):'—';function setText(id,val,cls=''){const e=document.getElementById(id);e.textContent=val;e.className='value '+cls}function layout(extra={}){return Object.assign({},layoutBase,extra)}function table(rows){if(!rows||!rows.length)return '<div class="empty">暂无数据</div>';let h='<table><thead><tr><th>股票</th><th>模型分数</th><th>实际收益</th></tr></thead><tbody>';rows.forEach(r=>{const c=(r.realized_return||0)>=0?'up':'down';h+=`<tr><td>${r.symbol}</td><td>${num(r.score,4)}</td><td class="${c}">${pct(r.realized_return)}</td></tr>`});return h+'</tbody></table>'}async function api(url){const r=await fetch(url);if(!r.ok)throw new Error(await r.text());return await r.json()}let experimentData=null;let weeklyMode=true;function renderSnapshot(lp){if(!lp)return;drawScatter(lp);document.getElementById('longTable').innerHTML=table(lp.long);document.getElementById('shortTable').innerHTML=table(lp.short);document.getElementById('latestMeta').textContent=lp.date?`日期：${lp.date} · 股票数：${lp.count} · RankIC：${num(lp.rank_ic,3)}`:'暂无预测数据';document.getElementById('scatterTitle').textContent=weeklyMode?`${lp.date||'最新交易周'}：模型分数与实际收益`:`${lp.date||'最新交易日'}：模型分数与实际收益`;document.getElementById('rankingTitle').textContent=weeklyMode?`${lp.date||'最新交易周'}排名`:`${lp.date||'最新交易日'}排名`}function populateDateSelector(dates){const s=document.getElementById('dateSelector');const old=s.value;s.innerHTML='';(dates||[]).forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=d;s.appendChild(o)});if(old&&(dates||[]).includes(old))s.value=old;else if(s.options.length)s.value=s.options[0].value;s.disabled=!s.options.length}async function loadExperiments(){const x=await api('/api/experiments');const s=document.getElementById('experiment');const old=s.value||new URLSearchParams(location.search).get('experiment');s.innerHTML='';x.experiments.forEach(e=>{const o=document.createElement('option');o.value=e.name;o.textContent=e.name;s.appendChild(o)});if(x.experiments.some(e=>e.name===old))s.value=old;else if(x.default)s.value=x.default;if(s.value)await loadExperiment(s.value)}async function loadExperiment(name){const n=document.getElementById('notice');n.style.display='none';try{const d=await api('/api/experiment/'+encodeURIComponent(name));experimentData=d;const m=d.metrics||{};const net=m.net||{};weeklyMode=m.frequency==='weekly';document.getElementById('universeLabel').textContent=`股票 / ${weeklyMode?'周数':'交易日'}`;const q=m.signal_quality||{};setText('rankic',num(m.rank_ic_mean,3),m.rank_ic_mean>0?'good':'warn');setText('normalIc',num(q.normal_ic_mean,3),q.normal_ic_mean>0?'good':'warn');setText('icir',num(q.icir,2),q.icir>0.5?'good':'warn');setText('icSig',Number.isFinite(q.ic_tstat)?`${num(q.ic_tstat,2)} / ${Number.isFinite(q.ic_pvalue)?q.ic_pvalue.toFixed(3):'—'}`:'—',q.ic_tstat&&Math.abs(q.ic_tstat)>2?'good':'warn');setText('spread',pct(q.top_bottom_spread_mean),q.top_bottom_spread_mean>0?'good':'warn');setText('equity',num(net.final_equity,3),net.final_equity>=1?'good':'warn');setText('sharpe',num(net.sharpe,2),net.sharpe>0?'good':'warn');setText('drawdown',pct(net.max_drawdown),net.max_drawdown>=-0.1?'good':'warn');setText('universe',`${m.n_stocks??'—'} / ${m.n_days??'—'}`);setText('backend',`${m.backend??'—'} · ${m.device??'—'}`);document.getElementById('rankicTitle').textContent=weeklyMode?'每周 RankIC':'每日 RankIC';drawLongOnly(d);drawEquity(d.equity);drawTraining(d.training);drawGate(d.training);drawEta(d.training);drawRankic(d.rankic);drawQuality(q);populateDateSelector(d.recent_dates||[]);const lp=(d.recent_predictions||[])[0]||d.latest_predictions;renderSnapshot(lp);const source=m.data_source||m.source||'';if(String(source).toLowerCase().includes('synthetic')){n.textContent='当前实验标记为合成数据；收益仅用于验证训练与回测链路。';n.style.display='block'}}catch(e){n.textContent='加载实验失败：'+e.message;n.style.display='block'}}function drawEquity(a){if(!a||!a.length){Plotly.purge('equityPlot');return}const x=a.map(r=>r.date);const traces=[{x,y:a.map(r=>r.gross_equity),name:'Gross 多空',line:{color:'#72b7ff',width:1.6}},{x,y:a.map(r=>r.net_equity),name:'Net 多空（含成本）',line:{color:'#43d19e',width:2.2}}];if(a.some(r=>Number.isFinite(r.top5_equity))){traces.push({x,y:a.map(r=>r.top5_equity),name:'Top5 多头 Gross',line:{color:'#ffc857',width:1.8,dash:'dot'}})}if(a.some(r=>Number.isFinite(r.top5_net_equity))){traces.push({x,y:a.map(r=>r.top5_net_equity),name:'Top5 多头 Net',line:{color:'#ff7f50',width:2.1}})}Plotly.react('equityPlot',traces,layout({yaxis:{title:'净值',gridcolor:'#22344d'},legend:{orientation:'h',y:1.12}}),plotConfig)}function drawTraining(a){if(!a||!a.length){Plotly.purge('trainingPlot');return}const x=a.map(r=>r.epoch);const l=layout({yaxis:{title:'loss',gridcolor:'#22344d'},xaxis:{title:'epoch',gridcolor:'#22344d'},legend:{orientation:'h',y:1.12}});l.yaxis2={title:'验证 RankIC',overlaying:'y',side:'right',gridcolor:'#22344d'};const traces=[{x,y:a.map(r=>r.loss),name:'总损失',line:{color:'#72b7ff'}},{x,y:a.map(r=>r.score_loss),name:'Score tail',line:{color:'#b08cff'}},{x,y:a.map(r=>r.temporal_loss),name:'Temporal InfoNCE',line:{color:'#ff9f68'}},{x,y:a.map(r=>r.rankic_loss),name:'RankIC loss',line:{color:'#c45cff'}},{x,y:a.map(r=>r.block_rank_loss),name:'四分位块间 loss',line:{color:'#ffc857',dash:'dot'}},{x,y:a.map(r=>r.val_rank_ic),name:'Val RankIC',yaxis:'y2',line:{color:'#43d19e',width:2}}];if(a.some(r=>Number.isFinite(r.group_loss))){traces.push({x,y:a.map(r=>r.group_loss),name:'Group regularizer',line:{color:'#ffc857',dash:'dot'}})}if(a.some(r=>Number.isFinite(r.industry_loss))){traces.push({x,y:a.map(r=>r.industry_loss),name:'Industry excess rankIC loss',line:{color:'#8de8c0',dash:'dot'}})}if(a.some(r=>Number.isFinite(r.exposure_loss))){traces.push({x,y:a.map(r=>r.exposure_loss),name:'行业暴露 η² 惩罚',line:{color:'#ff5c8d',dash:'dot'}})}if(a.some(r=>Number.isFinite(r.control_loss))){traces.push({x,y:a.map(r=>r.control_loss),name:'周度分数粘性惩罚',line:{color:'#ffd166',dash:'dot'}})}Plotly.react('trainingPlot',traces,l,plotConfig)}function drawGate(a){const p=document.getElementById('gatePanel');if(!a||!a.length||!a.some(r=>Number.isFinite(r.gate_mean))){p.hidden=true;Plotly.purge('gatePlot');return}p.hidden=false;const x=a.map(r=>r.epoch);const l=layout({yaxis:{title:'tanh(θ) 门控',gridcolor:'#22344d'},xaxis:{title:'epoch',gridcolor:'#22344d'},legend:{orientation:'h',y:1.12}});l.yaxis2={title:'Val ICIR',overlaying:'y',side:'right',gridcolor:'#22344d'};const traces=[{x,y:a.map(r=>r.gate_mean),name:'门均值',line:{color:'#43d19e',width:2}},{x,y:a.map(r=>r.gate_max),name:'门最大值',line:{color:'#72b7ff',dash:'dot'}},{x,y:a.map(r=>r.gate_min),name:'门最小值',line:{color:'#ff9f68',dash:'dot'}}];if(a.some(r=>Number.isFinite(r.val_icir)))traces.push({x,y:a.map(r=>r.val_icir),name:'Val ICIR',yaxis:'y2',line:{color:'#c45cff',width:2}});if(a.some(r=>Number.isFinite(r.val_industry_ic)))traces.push({x,y:a.map(r=>r.val_industry_ic),name:'Val 行业超额 IC',yaxis:'y2',line:{color:'#8de8c0',width:2,dash:'dashdot'}});Plotly.react('gatePlot',traces,l,plotConfig)}function drawEta(a){const p=document.getElementById('etaPanel');if(!a||!a.length||!a.some(r=>Number.isFinite(r.val_score_eta2_industry))){p.hidden=true;Plotly.purge('etaPlot');return}p.hidden=false;const x=a.map(r=>r.epoch);const l=layout({yaxis:{title:'η²（分数被行业解释的方差占比）',gridcolor:'#22344d'},xaxis:{title:'epoch',gridcolor:'#22344d'},legend:{orientation:'h',y:1.12}});l.yaxis2={title:'Val RankIC',overlaying:'y',side:'right',gridcolor:'#22344d'};const traces=[{x,y:a.map(r=>r.val_score_eta2_industry),name:'验证分数 η²',line:{color:'#ff5c8d',width:2}}];if(a.some(r=>Number.isFinite(r.val_score_eta2_neutral)))traces.push({x,y:a.map(r=>r.val_score_eta2_neutral),name:'中性分数期望值',line:{color:'#8899aa',width:1.6,dash:'dash'}});if(a.some(r=>Number.isFinite(r.exposure_loss)))traces.push({x,y:a.map(r=>r.exposure_loss),name:'训练周 η²',line:{color:'#b08cff',width:1.6,dash:'dot'}});if(a.some(r=>Number.isFinite(r.val_rank_ic)))traces.push({x,y:a.map(r=>r.val_rank_ic),name:'Val RankIC',yaxis:'y2',line:{color:'#43d19e',width:2}});Plotly.react('etaPlot',traces,l,plotConfig)}
function drawRankic(a){if(!a||!a.length){Plotly.purge('rankicPlot');return}Plotly.react('rankicPlot',[{x:a.map(r=>r.date),y:a.map(r=>r.rank_ic),name:'RankIC',mode:'lines+markers',line:{color:'#43d19e',width:1.4},marker:{size:4}}],layout({yaxis:{title:'RankIC',gridcolor:'#22344d'},xaxis:{gridcolor:'#22344d'}}),plotConfig)}function drawQuality(q){if(!q||!q.group_returns||!q.group_returns.length){Plotly.purge('qualityPlot');return}const labels=q.group_returns.map((_,i)=>'G'+(i+1));const tr=[{x:labels,y:q.group_returns,name:'分组平均收益',type:'bar',marker:{color:q.group_returns,colorscale:'RdYlGn',cmid:0}}];Plotly.react('qualityPlot',tr,layout({yaxis:{title:'下一期收益',tickformat:'.1%',gridcolor:'#22344d'},xaxis:{title:'因子分组（低→高）',gridcolor:'#22344d'},showlegend:false}),plotConfig);document.getElementById('qualityMeta').textContent=`单调性相关：${num(q.monotonicity,3)} · 上升比例：${pct(q.monotonicity_ratio)} · 年化 Top-Bottom：${pct(q.top_bottom_spread_annualized)}`;}function drawScatter(p){if(!p||!p.rows||!p.rows.length){Plotly.purge('scatterPlot');return}Plotly.react('scatterPlot',[{x:p.rows.map(r=>r.score),y:p.rows.map(r=>r.realized_return),mode:'markers',text:p.rows.map(r=>r.symbol),hovertemplate:'%{text}<br>score=%{x:.4f}<br>收益=%{y:.2%}<extra></extra>',marker:{color:p.rows.map(r=>r.realized_return),colorscale:'RdYlGn',showscale:true,colorbar:{title:'收益'},size:7,opacity:.8}}],layout({xaxis:{title:'模型分数',gridcolor:'#22344d'},yaxis:{title:'下一周期实际收益',tickformat:'.1%',gridcolor:'#22344d'},hovermode:'closest'}),plotConfig)}document.getElementById('experiment').addEventListener('change',e=>loadExperiment(e.target.value));document.getElementById('dateSelector').addEventListener('change',e=>{const p=(experimentData?.recent_predictions||[]).find(x=>x.date===e.target.value);if(p)renderSnapshot(p)});document.getElementById('refresh').addEventListener('click',loadExperiments);loadExperiments();
</script></body></html>
'''

def _json_safe(value: Any) -> Any:
    if value is None:return None
    if isinstance(value,float) and (math.isnan(value) or math.isinf(value)):return None
    if hasattr(value,'item'):
        try:return _json_safe(value.item())
        except Exception:pass
    if hasattr(value,'isoformat'):return value.isoformat()
    return value

def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k:_json_safe(v) for k,v in row.items()} for row in df.to_dict(orient='records')]

def _prediction_snapshot(frame: pd.DataFrame, date: str, top_n: int = 10) -> dict[str, Any]:
    """Build the rows shown for one cross-sectional prediction date."""
    g=frame[frame['date']==date].copy().sort_values('score',ascending=False)
    return {'date':date,'count':int(len(g)),
            'rank_ic':_json_safe(g['rank_ic'].iloc[0]) if len(g) and 'rank_ic' in g else None,
            'rows':_records(g),'long':_records(g.head(top_n)),
            'short':_records(g.tail(top_n).sort_values('score'))}

def _add_top5_equity(equity: pd.DataFrame, pred: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    """Reconstruct the top-five equal-weight long-only curve for old runs."""
    if pred.empty:return equity
    rows=[]
    for date,g in pred.groupby('date',sort=True):
        g=g.dropna(subset=['score','realized_return']).sort_values('score',ascending=False)
        if not g.empty: rows.append({'date':str(date),'top5_return':float(g.head(5)['realized_return'].mean())})
    if not rows:return equity
    top=pd.DataFrame(rows).sort_values('date')
    equity=equity.copy()
    if 'top5_return' not in equity.columns:
        equity=top if equity.empty else equity.merge(top,on='date',how='outer').sort_values('date').reset_index(drop=True)
    # Long-only top-five pays one side of the transaction cost.  The
    # long-short curve pays two sides in the engine, so do not reuse that
    # convention here.
    cost=float(cost_bps)/10000.0
    if 'top5_net_return' not in equity.columns:equity['top5_net_return']=equity['top5_return']-cost
    if 'top5_equity' not in equity.columns:equity['top5_equity']=(1+equity['top5_return'].fillna(0.0)).cumprod()
    if 'top5_net_equity' not in equity.columns:equity['top5_net_equity']=(1+equity['top5_net_return'].fillna(0.0)).cumprod()
    return equity

def _experiments(root: Path) -> list[dict[str,str]]:
    if not root.exists():return []
    return [{"name":p.name,"path":str(p)} for p in sorted(root.iterdir()) if p.is_dir() and (p/'metrics.json').exists()]

def create_app(output_root: str|Path='outputs') -> Flask:
    root=Path(output_root).resolve();app=Flask(__name__);app.config['OUTPUT_ROOT']=root
    @app.get('/')
    def index():return render_template_string(PAGE.replace('</script></body>', Path(__file__).with_name('long_only.js').read_text(encoding='utf-8') + '</script></body>'))
    @app.get('/style-analysis')
    def style_analysis():
        report = root/'single_score_500_style_analysis'/'report.html'
        if not report.exists():
            return 'Run scripts/analyze_single_score_style.py to generate the report.', 404
        return send_file(report)
    @app.get('/industry-analysis')
    def industry_analysis():
        report = root/'industry_580_future_analysis'/'report.html'
        if not report.exists():
            return 'Run scripts/analyze_industry_future.py to generate the report.', 404
        return send_file(report)
    @app.get('/api/hierarchy-progress')
    def hierarchy_progress():
        folder=root/'hierarchical_580_comparison'
        progress=json.loads((folder/'progress.json').read_text(encoding='utf-8')) if (folder/'progress.json').exists() else {'status':'not_started'}
        runs=[]
        for key in ['off','h2','h5']:
            p=root/f'real_hierarchical_580_{key}_10ep'
            if (p/'metrics.json').exists():
                m=json.loads((p/'metrics.json').read_text(encoding='utf-8'))
                history=_records(pd.read_csv(p/'training_metrics.csv')) if (p/'training_metrics.csv').exists() else []
                runs.append({'method':key,'status':m.get('status'),'best_epoch':m.get('best_epoch'),'validation_sharpe':m.get('selection_value'),'history':history})
        return jsonify(_json_safe({'progress':progress,'runs':runs}))
    @app.get('/hierarchy-comparison')
    def hierarchy_comparison():
        report=root/'hierarchical_580_comparison'/'report.html'
        if report.exists():return send_file(report)
        return Path(__file__).with_name('hierarchy_progress.html').read_text(encoding='utf-8')
    @app.get('/api/experiments')
    def list_experiments():
        exps=_experiments(root);names=[e['name'] for e in exps]
        # Prefer the latest real-data run when one is available, otherwise use
        # the latest full experiment.
        preferred = ['real_weekly_500_full_rankic_10ep', 'real_weekly_500_heads4_decorr_8ep', 'real_reliable500_weight2_8ep', 'real_reliable500_top150_5ep', 'real_teacher500_top30_5ep', 'real_sw_file_20_mixed_rankic_8ep', 'real_weekly_500_mixed_rankic_8ep', 'real_csi800_tail_rankic_8ep', 'real_csi800_equal_score_8ep', 'real_csi800_mixed_rankic_8ep', 'real_weekly_500_top20_sharpe',
                     'real_weekly_100_520_h5_rankic',
                     'real_contrastive_weekly_100_yahoo',
                     'real_single_score_100_520_h5',
                     'real_single_score_100_520_h1',
                     'real_multi_score_rankic_100_520_h1',
                     'real_contrastive_weekly_100_yahoo_v5',
                     'real_contrastive_weekly_100_yahoo_v3',
                     'real_contrastive_weekly_100_yahoo_original',
                     'real_contrastive_weekly_100_yahoo_v2',
                     'upgrade_weekly_smoke',
                     'real_contrastive_weekly_100_yahoo']
        contrastive = next((n for n in preferred if n in names), None)
        contrastive = contrastive or next((n for n in names if 'weekly' in n and n.startswith('real_contrastive')), None) or next((n for n in names if n.startswith('real_contrastive')), None)
        default=contrastive or ('real_temporal_cs' if 'real_temporal_cs' in names else ('real_upgraded' if 'real_upgraded' in names else ('real_long20' if 'real_long20' in names else ('real_medium' if 'real_medium' in names else ('real_smoke2' if 'real_smoke2' in names else ('full2' if 'full2' in names else (names[-1] if names else None)))))))
        comparison=root/'hierarchical_580_comparison'/'progress.json'
        if comparison.exists():
            progress=json.loads(comparison.read_text(encoding='utf-8'))
            if progress.get('status')=='complete':
                selected=f"real_hierarchical_580_{progress['selected']}_10ep"
                if selected in names:default=selected
        return jsonify({'experiments':exps,'default':default})
    @app.get('/api/experiment/<name>')
    def experiment(name: str):
        p=root/name
        if not p.is_dir() or not (p/'metrics.json').exists():return jsonify({'error':'experiment not found'}),404
        with (p/'metrics.json').open('r',encoding='utf-8') as f:metrics=json.load(f)
        training=pd.read_csv(p/'training_metrics.csv') if (p/'training_metrics.csv').exists() else pd.DataFrame();equity=pd.read_csv(p/'equity_curve.csv') if (p/'equity_curve.csv').exists() else pd.DataFrame();pred=pd.read_parquet(p/'predictions.parquet') if (p/'predictions.parquet').exists() else pd.DataFrame()
        rankic=pd.DataFrame();latest={'date':None,'count':0,'rank_ic':None,'rows':[],'long':[],'short':[]};recent=[];recent_dates=[]
        if not pred.empty:
            pred['date']=pred['date'].astype(str);rankic=pred.groupby('date',as_index=False)['rank_ic'].first();all_dates=sorted(pred['date'].unique());recent_dates=list(reversed(all_dates[-10:]));latest_date=all_dates[-1];latest=_prediction_snapshot(pred,latest_date,int(metrics.get('long_only',{}).get('top_n',10)));recent=[_prediction_snapshot(pred,d,int(metrics.get('long_only',{}).get('top_n',10))) for d in recent_dates]
            cost=metrics.get('transaction_cost_bps',metrics.get('cost_bps',10.0)) if isinstance(metrics,dict) else 10.0
            equity=_add_top5_equity(equity,pred,float(cost))
            # Older runs predate the quality report; calculate it lazily so
            # the dashboard presents a consistent set of diagnostics.
            if 'signal_quality' not in metrics and signal_quality_metrics is not None:
                ppy = 52 if metrics.get('frequency') == 'weekly' else 252
                metrics['signal_quality'] = signal_quality_metrics(
                    pred, n_groups=int(metrics.get('quality_groups', 10)),
                    periods_per_year=ppy)
        long_curve=pd.read_csv(p/'long_only_curve.csv') if (p/'long_only_curve.csv').exists() else pd.DataFrame()
        universe=pd.read_csv(p/'universe.csv',dtype=str).fillna('') if (p/'universe.csv').exists() else pd.DataFrame()
        manifest=json.loads((p/'manifest.json').read_text(encoding='utf-8')) if (p/'manifest.json').exists() else {}
        composition={'counts':universe['index_name'].value_counts().to_dict() if 'index_name' in universe else {}, 'classified':int(universe['citic_l1'].ne('').sum()) if 'citic_l1' in universe else 0, 'total':len(universe), 'manifest':manifest, 'industry_counts':universe['industry_l1'].value_counts().to_dict() if 'industry_l1' in universe else {}}
        comparison=json.loads((p/'rankic_selected_comparison.json').read_text()) if (p/'rankic_selected_comparison.json').exists() else None
        score_comparison=json.loads((p/'score_weight_comparison.json').read_text(encoding='utf-8')) if (p/'score_weight_comparison.json').exists() else None
        tail_comparison=json.loads((p/'tail_rankic_comparison.json').read_text(encoding='utf-8')) if (p/'tail_rankic_comparison.json').exists() else None
        experiment_comparison=json.loads((p/'comparison.json').read_text()) if (p/'comparison.json').exists() else []
        checkpoint_comparison=json.loads((p/'checkpoint_comparison.json').read_text()) if (p/'checkpoint_comparison.json').exists() else None
        reliability_report=json.loads((p/'reliability_report.json').read_text(encoding='utf-8')) if (p/'reliability_report.json').exists() else None
        reliability_curves=pd.read_csv(p/'reliability_curves.csv') if (p/'reliability_curves.csv').exists() else pd.DataFrame()
        reliability_ranking=pd.read_csv(p/'reliability_ranking.csv') if (p/'reliability_ranking.csv').exists() else pd.DataFrame()
        weight_sweep=json.loads((p/'weight_sweep_report.json').read_text(encoding='utf-8')) if (p/'weight_sweep_report.json').exists() else None
        weight_curves=pd.read_csv(p/'weight_sweep_curves.csv') if (p/'weight_sweep_curves.csv').exists() else pd.DataFrame()
        if (p/'stock_weights.csv').exists():
            weights=pd.read_csv(p/'stock_weights.csv')
            metrics['stock_weight_summary']=[{'weight':float(w),'stocks':int(n)} for w,n in weights.groupby('weight').size().items()]
        if (p/'stock_weight_evaluation.json').exists():
            metrics['stock_weight_evaluation']=json.loads((p/'stock_weight_evaluation.json').read_text(encoding='utf-8'))
        if (p/'head_comparison.json').exists():
            metrics['head_comparison']=json.loads((p/'head_comparison.json').read_text(encoding='utf-8'))
            metrics['head_comparison_curves']=_records(pd.read_csv(p/'head_comparison_curves.csv'))
        return jsonify({'weight_sweep':weight_sweep,'weight_sweep_curves':_records(weight_curves),'reliability_report':reliability_report,'reliability_curves':_records(reliability_curves),'reliability_ranking':_records(reliability_ranking),'checkpoint_comparison':checkpoint_comparison,'experiment_comparison':experiment_comparison,'tail_rankic_comparison':tail_comparison,'score_weight_comparison':score_comparison,'universe_info':composition,'rankic_selected_comparison':comparison,'long_only_curve':_records(long_curve),'metrics':metrics,'training':_records(training),'equity':_records(equity),'rankic':_records(rankic),'latest_predictions':latest,'recent_dates':recent_dates,'recent_predictions':recent})
    return app

def main() -> None:
    parser=argparse.ArgumentParser(description='Serve the quant MVP dashboard');parser.add_argument('--output-root',default='outputs');parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8501);args=parser.parse_args();app=create_app(args.output_root);print(f'Dashboard: http://{args.host}:{args.port}/  (outputs: {Path(args.output_root).resolve()})');app.run(host=args.host,port=args.port,debug=False)

if __name__=='__main__':main()

