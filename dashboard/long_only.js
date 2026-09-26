function drawLongOnly(d) {
  const panel = document.getElementById('longOnlySection');
  const m = d.metrics, q = m.long_only;
  let provenance = document.getElementById('universeProvenance');
  if (!provenance) {
    provenance = document.createElement('p'); provenance.id = 'universeProvenance';
    panel.before(provenance);
  }
  let objective = document.getElementById('objectiveSummary');
  if (!objective) {
    objective = document.createElement('p'); objective.id = 'objectiveSummary';
    provenance.after(objective);
  }
  objective.textContent = m.rankic_tail_mix != null ? `训练：${m.epochs} epochs · query ${m.lookback}周 / key ${m.temporal_horizon}周 · 单score head；Score 头尾各${pct(m.extreme_frac,0)}，权重${m.score_top_weight}/${m.score_bottom_weight}；RankIC 总权重${m.rankic_weight}，其中${pct(m.rankic_tail_mix,0)}为头尾各${pct(m.rankic_tail_fraction,0)}子集、${pct(1-m.rankic_tail_mix,0)}为全量。` : '';
  if(m.rankic_tail_mix == null && m.rankic_weight > 0){
    objective.textContent=`训练：${m.epochs} epochs · query ${m.lookback}周 / key ${m.temporal_horizon}周 · ${m.n_score_heads}个score头；Score头尾各${pct(m.extreme_frac,0)}，权重${m.score_top_weight}/${m.score_bottom_weight}；RankIC权重${m.rankic_weight}，${m.rankic_tail_fraction != null ? '收益头尾各'+pct(m.rankic_tail_fraction,0)+'合并子集' : '仅全量横截面，无极端子集项'}。`;
  }
  const u = d.universe_info;
  let headsPanel=document.getElementById('headComparison');
  if(!headsPanel){headsPanel=document.createElement('section');headsPanel.id='headComparison';provenance.after(headsPanel);}
  const headReport=m.head_comparison;
  headsPanel.hidden=!headReport;
  if(headReport){
    const hd=headReport.diagnostics;
    headsPanel.innerHTML='<h2>4个Score头取平均＋输出去相关 · 对照</h2><p>恢复原500股等权训练与回测，没有150股筛选或加权。Score和全量RankIC作用于4头平均值；每周单独计算4头相关矩阵，非对角项平方和乘0.001加入损失，不做warm-up。旧Group去相关仍关闭。</p><table><tr><th>模型</th><th>训练轮数</th><th>最佳epoch</th><th>验证净Sharpe</th><th>测试净收益</th><th>测试净Sharpe</th><th>最大回撤</th><th>RankIC</th><th>Normal IC</th></tr>'+headReport.rows.map(r=>`<tr><td>${r.label}</td><td>${r.epochs}</td><td>${r.best_epoch}</td><td>${num(r.val_sharpe)}</td><td>${pct(r.long_only.net.total_return)}</td><td>${num(r.long_only.net.sharpe)}</td><td>${pct(r.long_only.net.max_drawdown)}</td><td>${num(r.rank_ic)}</td><td>${num(r.normal_ic)}</td></tr>`).join('')+'</table><div id="headEquityPlot" style="height:350px"></div><div class="grid"><div><h3>验证期：每周相关矩阵的均值</h3><div id="headValCorr" style="height:310px"></div></div><div><h3>测试期：每周相关矩阵的均值</h3><div id="headTestCorr" style="height:310px"></div></div></div><p id="headStats"></p><table><tr><th>Head</th><th>验证平均横截面标准差</th><th>测试平均横截面标准差</th><th>测试单头RankIC</th></tr>'+[0,1,2,3].map(i=>`<tr><td>${i+1}</td><td>${num(hd.validation.head_std_mean[i],5)}</td><td>${num(hd.test.head_std_mean[i],5)}</td><td>${num(hd.test.head_rank_ic_mean[i],4)}</td></tr>`).join('')+'</table><div id="outputDecorrPlot" style="height:270px"></div><p id="headNote"></p>';
    const hc=m.head_comparison_curves||[];
    Plotly.react('headEquityPlot',headReport.rows.map(r=>({x:hc.map(c=>c.date),y:hc.map(c=>c[r.key]),name:r.label})),layout({yaxis:{title:'Top20测试净值'},legend:{orientation:'h',y:1.12}}),plotConfig);
    const labels=['Head1','Head2','Head3','Head4'];
    [['headValCorr','validation'],['headTestCorr','test']].forEach(([id,key])=>Plotly.react(id,[{z:hd[key].correlation_mean,x:labels,y:labels,type:'heatmap',zmin:-1,zmax:1,zmid:0,colorscale:'RdBu',reversescale:true}],layout({margin:{l:65,r:25,t:10,b:40}}),plotConfig));
    document.getElementById('headStats').textContent=`平均绝对头间相关：验证 ${num(hd.validation.mean_abs_offdiag,4)}，测试 ${num(hd.test.mean_abs_offdiag,4)}；平均非对角平方和：验证 ${num(hd.validation.mean_squared_frobenius,4)}，测试 ${num(hd.test.mean_squared_frobenius,4)}。平均分的横截面标准差：验证 ${num(hd.validation.mean_score_std,5)}，测试 ${num(hd.test.mean_score_std,5)}。低相关不等于有效因子，也需检查分数是否趋于常数。`;
    const hist=d.training||[];
    Plotly.react('outputDecorrPlot',[{x:hist.map(r=>r.epoch),y:hist.map(r=>r.output_decorr_loss),name:'原始输出去相关loss'}],layout({xaxis:{title:'Epoch'},yaxis:{title:'非对角相关平方和'}}),plotConfig);
    document.getElementById('headNote').textContent=headReport.note;
  }
  let scopePanel=document.getElementById('evaluationScope');
  if(!scopePanel){scopePanel=document.createElement('section');scopePanel.id='evaluationScope';provenance.after(scopePanel);}
  scopePanel.hidden=!m.evaluation_only;
  if(m.evaluation_only){
    scopePanel.innerHTML='<h2>仅在已选150股内回测 · 同权重对照</h2><p>训练仍为500股、其中150股收益监督权重2倍；沿用第8轮best，不重新训练或选择权重。此页预测列表、RankIC、信号分析与交易均限制在固定150股，每周选Top20。基准改为150股等权。</p><table><tr><th>候选池</th><th>净收益</th><th>净Sharpe</th><th>最大回撤</th><th>对应池基准收益</th></tr>'+(m.scope_comparison||[]).map(r=>`<tr><td>${r.label}</td><td>${pct(r.net.total_return)}</td><td>${num(r.net.sharpe)}</td><td>${pct(r.net.max_drawdown)}</td><td>${pct(r.benchmark.total_return)}</td></tr>`).join('')+'</table>';
  }
  if(m.stock_weight_summary){
    objective.textContent+=' 股票训练权重：'+m.stock_weight_summary.map(r=>r.stocks+'股 × '+r.weight+'倍').join('；')+'。Score与全量RankIC相关矩按股票权重归一化，时间InfoNCE不变；交易候选仍是全部500股。';
  }
  provenance.textContent = u?.total && Object.keys(u.counts).length ? `股票池：${Object.entries(u.counts).map(([k,v])=>k+' '+v+'支').join(' + ')}；成分日期：${(u.manifest.asof||[]).join(', ')}。中信一级分类：${u.classified}/${u.total} 已有来源，${u.classified < u.total ? '待补完整可核验映射' : '已覆盖'}。当前成分回溯历史，非历史逐期成分。` : (m.universe_name ? `股票池：${m.universe_name} · ${m.n_stocks} 支` : '');

  let screening=document.getElementById('screeningNote');
  if(!screening){screening=document.createElement('p');screening.id='screeningNote';provenance.after(screening);}
  const teacher=u?.manifest?.teacher_best_epoch;
  screening.textContent=teacher ? `二阶段筛选：原500股第${teacher}轮权重，在${u.manifest.screen_date}按score取前30%固定150股，再从头训练5轮。筛选未使用测试期收益，但第一阶段使用训练数据拟合并用验证集选模，第二阶段再次使用相同验证集；属于探索实验，不能视为独立验证。高score表示该时点预测收益排序高，不代表股票长期更符合因子。` : '';
  let reliablePanel=document.getElementById('reliabilityPanel');
  let fixedWeight=document.getElementById('fixedStockWeight');
  if(!fixedWeight){fixedWeight=document.createElement('section');fixedWeight.id='fixedStockWeight';provenance.after(fixedWeight);}
  const fixedReport=m.stock_weight_evaluation;
  fixedWeight.hidden=!fixedReport;
  if(fixedReport){
    fixedWeight.innerHTML='<h2>固定2倍权重 · 与已有原500股模型对照</h2><p>保留全部500股，仅提高固定150股的收益监督权重；本次只训练2倍、8轮。以下为同一测试期，涨跌指标均为平衡准确率。</p><table><tr><th>模型</th><th>训练轮数</th><th>最佳轮次</th><th>全部500股准确率</th><th>选中150股准确率</th><th>其余350股准确率</th><th>Top20净收益</th><th>净Sharpe</th><th>最大回撤</th></tr>'+fixedReport.rows.map(r=>`<tr><td>${r.label}</td><td>${r.epochs}</td><td>${r.best_epoch}</td><td>${pct(r.direction.balanced_accuracy)}</td><td>${pct(r.selected.balanced_accuracy)}</td><td>${pct(r.remaining.balanced_accuracy)}</td><td>${pct(r.net.total_return)}</td><td>${num(r.net.sharpe)}</td><td>${pct(r.net.max_drawdown)}</td></tr>`).join('')+'</table><p id="fixedWeightCaveat"></p>';
    document.getElementById('fixedWeightCaveat').textContent=fixedReport.caveat;
  }
  let weightPanel=document.getElementById('stockWeightSweep');
  if(!weightPanel){weightPanel=document.createElement('section');weightPanel.id='stockWeightSweep';provenance.after(weightPanel);}
  const sweep=d.weight_sweep;
  weightPanel.hidden=!sweep;
  if(sweep){
    const wr=sweep.runs;
    weightPanel.innerHTML='<h2>保留500股 · 可靠150股训练加权对照</h2><p>每组同种子从头训练5轮；选中150股的Score及全量RankIC相关矩加权并归一化，其他350股权重为1，InfoNCE不变。交易仍从全部500股中选Top20。倍数和epoch只按验证集净Sharpe选择。</p><p id="weightChoice"></p><div style="overflow-x:auto"><table><tr><th>倍数</th><th>选中股理论权重占比</th><th>最佳epoch</th><th>验证净Sharpe</th><th>测试净收益</th><th>测试净Sharpe</th><th>最大回撤</th><th>全500股平衡准确率</th><th>选中150股准确率</th><th>其余350股准确率</th></tr>'+wr.map(r=>`<tr style="${r.chosen?'background:#194236;font-weight:600':''}"><td>${r.multiplier}×${r.chosen?' · 验证选中':''}</td><td>${pct(150*r.multiplier/(350+150*r.multiplier))}</td><td>${r.best_epoch}</td><td>${num(r.val_sharpe)}</td><td>${pct(r.long_only.net.total_return)}</td><td>${num(r.long_only.net.sharpe)}</td><td>${pct(r.long_only.net.max_drawdown)}</td><td>${pct(r.direction.balanced_accuracy)}</td><td>${pct(r.selected_direction.balanced_accuracy)}</td><td>${pct(r.remaining_direction.balanced_accuracy)}</td></tr>`).join('')+'</table></div><div id="weightValidationPlot" style="height:290px"></div><div id="weightEquityPlot" style="height:370px"></div><p>理论权重占比按全部500股计算；Score只计算收益头尾20%，实际占比随每周标签变化。涨跌准确率均为平衡准确率，使用验证期校准后冻结的上涨概率，交易不使用概率择时。150股名单与本轮调参复用了验证期，测试期此前也已查看：本页属于探索对照，不是独立验证。</p>';
    document.getElementById('weightChoice').textContent=`验证集选择：${sweep.chosen_multiplier}倍。1倍是同样5轮的新训练基线；不能与之前10轮的基线混作单变量比较。`;
    Plotly.react('weightValidationPlot',[{x:wr.map(r=>r.multiplier+'×'),y:wr.map(r=>r.val_sharpe),type:'bar',name:'验证净Sharpe',marker:{color:wr.map(r=>r.chosen?'#43d19e':'#72b7ff')}}],layout({yaxis:{title:'验证净Sharpe'},xaxis:{title:'150股相对权重'}}),plotConfig);
    const curves=d.weight_sweep_curves||[];
    Plotly.react('weightEquityPlot',wr.map((r,i)=>({x:curves.map(c=>c.date),y:curves.map(c=>c[r.experiment]),name:r.multiplier+'×'+(r.chosen?'（验证选中）':''),line:{width:r.chosen?3:1.5}})),layout({yaxis:{title:'Top20测试净值'},legend:{orientation:'h',y:1.12}}),plotConfig);
  }
  if(!reliablePanel){reliablePanel=document.createElement('section');reliablePanel.id='reliabilityPanel';provenance.after(reliablePanel);}
  const reliability=d.reliability_report;
  reliablePanel.hidden=!reliability;
  if(reliability){
    screening.textContent='可靠性筛股：先在验证期前26周校准上涨概率，后52周逐周只使用过去数据更新校准；后52周分成前后两半，分别对上涨/下跌召回率做Beta(5,5)收缩，按较差半段排名取150股（每半段至少5涨5跌）。原epoch8曾用全验证期选模，筛股和重训再次使用验证期；本页为探索结果，不是独立验证。';
    const rr=reliability.comparisons;
    const rows=rr.map(r=>`<tr><td>${r.label}</td><td>${r.n_stocks}</td><td>${pct(r.direction.balanced_accuracy)}</td><td>${pct(r.direction.up_recall)}</td><td>${pct(r.direction.down_recall)}</td><td>${num(r.direction.brier,4)}</td><td>${pct(r.long_only.net.total_return)}</td><td>${num(r.long_only.net.sharpe)}</td><td>${pct(r.long_only.net.max_drawdown)}</td><td>${pct(r.long_only.benchmark.total_return)}</td></tr>`).join('');
    reliablePanel.innerHTML='<h2>预测可靠性筛选 · 三组测试对照</h2><p>平衡准确率 = (上涨召回率 + 下跌召回率) / 2；一直猜涨的基线为50%。Brier越小越好。绝对涨跌使用校准概率≥0.5判断，交易仍按原score选Top20，没有使用涨跌判断择时。</p><div style="overflow-x:auto"><table><tr><th>方法</th><th>股票数</th><th>平衡准确率</th><th>涨召回</th><th>跌召回</th><th>Brier</th><th>净收益</th><th>净Sharpe</th><th>最大回撤</th><th>各自池基准</th></tr>'+rows+'</table></div><div id="reliabilityPlot" style="height:370px"></div><p id="reliabilityVal"></p><p id="reliabilityVerdict"></p><details><summary>查看可靠性前20支及筛选分数</summary><div id="reliabilityNames"></div></details>';
    const cr=d.reliability_curves||[];
    Plotly.react('reliabilityPlot',rr.map((r,i)=>({x:cr.map(v=>v.date),y:cr.map(v=>v[r.key]),name:r.label,line:{color:['#9fb0c5','#72b7ff','#43d19e'][i]}})),layout({yaxis:{title:'Top20做多净值',gridcolor:'#22344d'},legend:{orientation:'h',y:1.12}}),plotConfig);
    document.getElementById('reliabilityVal').textContent='用于筛股的验证期平衡准确率：'+reliability.validation_selection.map(r=>r.label+' '+pct(r.balanced_accuracy)).join('；')+'。这些分数参与了筛选，不等同测试表现。测试期未选350股平衡准确率：'+pct(reliability.test_unselected.balanced_accuracy)+'。';
    document.getElementById('reliabilityVerdict').textContent=reliability.conclusion||'检查三组结果是否同时改善预测可靠性和交易收益；不能仅凭验证期提高得出结论。';
    const ranks=d.reliability_ranking||[];
    const nameTable=document.getElementById('reliabilityNames');
    const tbl=document.createElement('table');const header=document.createElement('tr');
    ['股票','原验证平衡准确率','前半收缩准确率','后半收缩准确率','筛选值（较低半段）'].forEach(k=>{const th=document.createElement('th');th.textContent=k;header.append(th);});tbl.append(header);
    ranks.filter(r=>r.selected).slice(0,20).forEach(r=>{const tr=document.createElement('tr');[r.symbol,pct(r.val_balanced_accuracy),pct(r.half1_shrunk),pct(r.half2_shrunk),pct(r.reliability)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td);});tbl.append(tr);});nameTable.replaceChildren(tbl);
  }
  let industryPanel=document.getElementById('industryPanel');
  if(!industryPanel){industryPanel=document.createElement('div');industryPanel.id='industryPanel';provenance.after(industryPanel);}
  const industries=Object.entries(u?.industry_counts||{}).sort((a,b)=>a[0].localeCompare(b[0],'zh'));
  industryPanel.hidden=!industries.length;
  if(industries.length){
    provenance.textContent=`股票池：用户提供申万分类文件 · ${u.total}支 · ${industries.length}个一级行业；每行业最多20支，不足取全部。按当前正常上市股票和长期行情覆盖筛选，当前分类回溯历史。`;
    industryPanel.innerHTML='<h3>申万一级行业样本分布</h3><div id="industryCountPlot" style="height:360px"></div><p>已剔除当前ST/退市及非沪深股票，并检查历史覆盖和近期成交；筛选是当前时点规则，不代表历史上一直非ST或可交易。</p>';
    Plotly.react('industryCountPlot',[{x:industries.map(r=>r[0]),y:industries.map(r=>r[1]),type:'bar',name:'样本数',marker:{color:'#72b7ff'}}],layout({xaxis:{tickangle:-50},yaxis:{title:'股票数',dtick:5},margin:{l:45,r:20,t:15,b:110}}),plotConfig);
  }
  panel.hidden = !q;
  if (!q) return;
  document.getElementById('longOnlyTitle').textContent = `Top-${q.top_n} 纯多头 · 样本外表现`;
  const cards = [
    ['净累计收益', pct(q.net.total_return)], ['净 Sharpe', num(q.net.sharpe)],
    ['最大回撤', pct(q.net.max_drawdown)], ['等权基准净收益', pct(q.benchmark.total_return)],
    ['相对基准收益', pct(q.relative_return)], ['平均买卖总换手', pct(q.mean_turnover)]
  ];
  document.getElementById('longCards').innerHTML = cards.map(([k,v]) => `<div class="card"><div class="label">${k}</div><div class="value">${v}</div></div>`).join('');
  document.getElementById('longAssumptions').textContent = `每周等权持有 ${q.top_n} 支；周收盘到下周收盘的研究回测，尚未模拟下一开盘成交及涨跌停限制。单边综合成本 ${q.cost_bps} bps × 买卖总换手，含首次建仓、不含期末清仓；缺失持仓收益按 0 估值并计数：${q.missing_held_returns} 笔。`;
  const a = d.long_only_curve || [], x = a.map(r=>r.date);
  Plotly.react('longEquityPlot', [
    {x,y:a.map(r=>r.long_net_equity),name:`Top-${q.top_n} 净值`,line:{color:'#43d19e',width:2.5}},
    {x,y:a.map(r=>r.benchmark_net_equity),name:'同股票池等权基准（含成本）',line:{color:'#9fb0c5',width:2}}
  ], layout({yaxis:{title:'净值',gridcolor:'#22344d'},legend:{orientation:'h',y:1.12}}),plotConfig);
  Plotly.react('longExcessPlot', [{x,y:a.map(r=>r.long_relative_equity),name:'策略净值 / 基准净值',line:{color:'#72b7ff',width:2}}],layout({yaxis:{title:'相对净值（1 为基准）',gridcolor:'#22344d'}}),plotConfig);
  document.getElementById('selectionMeta').textContent = `选模规则：验证集 Top-${q.top_n} 纯多头净 Sharpe 最大；相等时取较早轮次。最佳 epoch：${m.best_epoch}，验证净 Sharpe：${num(m.selection_value)}。下表只使用验证集，以上曲线只使用样本外测试区间。`;
  if(m.evaluation_only){document.getElementById('selectionMeta').textContent+=' 本页仅改变测试候选池；下表及选模值仍是原500股验证口径，没有用150股重新选epoch。';}
  if(m.selection_metric==='manual_latest_epoch') {
    document.getElementById('selectionMeta').textContent=`本页为手动评估最后一轮 newest.pt（epoch ${m.evaluated_epoch}），未重新训练，也不是验证集选出的 best。原验证选模保留 epoch ${m.validation_selected_best_epoch}；最后一轮验证净 Sharpe ${num(m.selection_value)}。`;
  }
  let checkpointCompare=document.getElementById('checkpointComparison');
  if(!checkpointCompare){checkpointCompare=document.createElement('p');checkpointCompare.id='checkpointComparison';document.getElementById('selectionMeta').after(checkpointCompare);}
  const ck=d.checkpoint_comparison;
  checkpointCompare.textContent=ck ? `同测试集对照：best（epoch ${ck.best_epoch}）净收益 ${pct(ck.best.net.total_return)} / Sharpe ${num(ck.best.net.sharpe)} / 回撤 ${pct(ck.best.net.max_drawdown)}；最后一轮（epoch ${ck.latest_epoch}）净收益 ${pct(ck.latest.net.total_return)} / Sharpe ${num(ck.latest.net.sharpe)} / 回撤 ${pct(ck.latest.net.max_drawdown)}。` : '';
  let scoreCompare = document.getElementById('scoreWeightComparison');
  if (!scoreCompare) {
    scoreCompare = document.createElement('p'); scoreCompare.id = 'scoreWeightComparison';
    document.getElementById('selectionMeta').after(scoreCompare);
  }
  const previous = d.score_weight_comparison;
  scoreCompare.textContent = previous ? `同800股、同数据区间的旧 Score 0.7/0.3 对照：测试净收益 ${pct(previous.net.total_return)}、净 Sharpe ${num(previous.net.sharpe)}、最大回撤 ${pct(previous.net.max_drawdown)}；本次等权分别为 ${pct(q.net.total_return)}、${num(q.net.sharpe)}、${pct(q.net.max_drawdown)}。两次分别按验证集选模，测试比较不参与选择。` : '';
  let tailCompare = document.getElementById('tailRankicComparison');
  if (!tailCompare) {
    tailCompare = document.createElement('p'); tailCompare.id = 'tailRankicComparison';
    document.getElementById('selectionMeta').after(tailCompare);
  }
  const mixed = d.tail_rankic_comparison;
  tailCompare.textContent = mixed ? `同800股、Score等权对照：之前0.8头尾/0.2全量 RankIC 的测试净收益 ${pct(mixed.net.total_return)}、净 Sharpe ${num(mixed.net.sharpe)}、最大回撤 ${pct(mixed.net.max_drawdown)}；本次纯头尾 RankIC 分别为 ${pct(q.net.total_return)}、${num(q.net.sharpe)}、${pct(q.net.max_drawdown)}。两次各自按验证净 Sharpe 选模，测试结果不参与选模。` : '';
  let experimentCompare = document.getElementById('experimentComparison');
  if (!experimentCompare) {
    experimentCompare = document.createElement('div'); experimentCompare.id = 'experimentComparison';
    document.getElementById('selectionMeta').after(experimentCompare);
  }
  experimentCompare.innerHTML = (d.experiment_comparison||[]).length ? '<h3>实验对照</h3><p>各实验按自己的验证集选模；各实验股票池或训练轮数可能不同，不能作为仅改变损失的严格对照。</p><table><tr><th>实验</th><th>轮数</th><th>最佳轮次</th><th>净收益</th><th>净Sharpe</th><th>最大回撤</th></tr>' + d.experiment_comparison.map(r=>`<tr><td>${r.experiment}</td><td>${r.epochs}</td><td>${r.best_epoch}</td><td>${pct(r.total_return)}</td><td>${num(r.sharpe)}</td><td>${pct(r.max_drawdown)}</td></tr>`).join('')+'</table>' : '';
  const comparison = d.rankic_selected_comparison;
  if (comparison) document.getElementById('selectionMeta').textContent += ` 原 RankIC 最优模型在同一 Top-20 口径下：测试净收益 ${pct(comparison.net.total_return)}、净 Sharpe ${num(comparison.net.sharpe)}、最大回撤 ${pct(comparison.net.max_drawdown)}。比较结果不参与选模。`;
  const header = '<tr><th>Epoch</th><th>净 Sharpe</th><th>净收益</th><th>最大回撤</th><th>相对基准收益</th><th>RankIC</th></tr>';
  document.getElementById('selectionTable').innerHTML = '<table>'+header+(d.training||[]).map(r=>`<tr style="${r.epoch===m.best_epoch?'background:#194236;font-weight:600':''}"><td>${r.epoch}${r.epoch===m.best_epoch?' · 已选':''}</td><td>${num(r.val_long_net_sharpe)}</td><td>${pct(r.val_long_net_return)}</td><td>${pct(r.val_long_max_drawdown)}</td><td>${pct(r.val_long_relative_return)}</td><td>${num(r.val_rank_ic)}</td></tr>`).join('')+'</table>';
}
