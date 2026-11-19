"""Publish all registered transfer cells and the authorized missing-data accounting."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from replay_kit_risk import paired

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/bce_native40_transfer_v1'
DOC=ROOT/'docs/research_20261119_bce_native40_transfer_results.md'
EVIDENCE=ROOT/'docs/evidence/20261119_bce_native40_transfer'

def table(frame):
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines.extend('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)

def main():
    m=pd.read_csv(OUT/'metrics.csv');w=pd.read_parquet(OUT/'weekly.parquet')
    rm=pd.read_csv(OUT/'reference_metrics.csv');rw=pd.read_parquet(OUT/'reference_weekly.parquet')
    s=pd.read_csv(OUT/'signals.csv',parse_dates=['date']).set_index('date')
    miss=pd.read_csv(OUT/'missing_returns.csv',parse_dates=['date'])
    coverage=pd.read_csv(OUT/'coverage.csv');aug=pd.read_csv(OUT/'augmentation.csv')
    lock=json.loads((OUT/'validation_lock.json').read_text(encoding='utf8'))
    verify=json.loads((OUT/'verification.json').read_text(encoding='utf8'))
    identities=pd.read_csv(OUT/'identity.csv');trace=pd.read_csv(OUT/'training_loss.csv')
    assert len(m)==56 and len(w)==16520 and len(rm)==12 and len(rw)==3540 and len(trace)==1800 and len(identities)==18
    assert len(s)==295 and len(lock['train_dates'])==473 and len(lock['validation_dates'])==123
    assert not w.duplicated(['book','arm','rule','fee','date']).any()
    np.testing.assert_allclose(w.net,w.long_net+w.short_net,atol=1e-12,rtol=0)
    for family in ('old','B0','P6'):
        for suffix in ('plain','fixed5'):
            q=identities.loc[identities.arm.eq(family+'_'+suffix)]
            assert q.labels.nunique()==1 and q.weights.nunique()==1
            if family!='old':assert q.inputs.nunique()==1 and q.normalization.nunique()==1
    rows=[]
    for book,groups in (('old',('B0','P6')),('B0',('B0',)),('P6',('P6',))):
        for rule in ('wide','full'):
            for fee in (10,30):
                q=w.loc[w.book.eq(book)&w.rule.eq(rule)&w.fee.eq(fee)].pivot(index='date',columns='arm',values='net')
                for family in groups:
                    for suffix in ('plain','fixed5'):
                        rows.append(dict(book=book,rule=rule,fee=fee,arm=family+'_'+suffix,control='old_'+suffix,
                                         **paired(q[family+'_'+suffix].to_numpy(),q['old_'+suffix].to_numpy())))
                for family in (('old','B0','P6') if book=='old' else ('old',book)):
                    rows.append(dict(book=book,rule=rule,fee=fee,arm=family+'_fixed5',control=family+'_plain',
                                     **paired(q[family+'_fixed5'].to_numpy(),q[family+'_plain'].to_numpy())))
    pairs=pd.DataFrame(rows);pairs.to_csv(OUT/'paired.csv',index=False)
    attribution=[]
    for book,families in (('old',('B0','P6')),('B0',('B0',)),('P6',('P6',))):
        base=w.loc[w.book.eq(book)&w.fee.eq(10)&w.arm.eq('old_plain')]
        full=base.loc[base.rule.eq('full')].set_index('date').reindex(s.index)
        wide=base.loc[base.rule.eq('wide')].set_index('date').reindex(s.index)
        direction=s.old_plain.gt(0)
        long=np.where(direction,full.gross_ret,(wide.gross_ret-.6*full.gross_ret)/.4)
        short=np.where(direction,(wide.gross_ret-.7*full.gross_ret)/.3,full.gross_ret)
        gap=long-short
        for family in families:
            for suffix in ('plain','fixed5'):
                arm=family+'_'+suffix;control='old_'+suffix
                dw=s[arm].gt(0).astype(float)-s[control].gt(0).astype(float)
                exposure=float(dw.mean()*gap.mean());gross=float((dw*gap).mean())
                for fee in (10,30):
                    aa=w.loc[w.book.eq(book)&w.fee.eq(fee)&w.rule.eq('full')&w.arm.eq(arm)].set_index('date').reindex(s.index)
                    bb=w.loc[w.book.eq(book)&w.fee.eq(fee)&w.rule.eq('full')&w.arm.eq(control)].set_index('date').reindex(s.index)
                    np.testing.assert_allclose(aa.gross_ret-bb.gross_ret,dw*gap,atol=1e-12,rtol=0)
                    tc=float((aa.trading_cost-bb.trading_cost).mean());bc=float((aa.borrow_cost-bb.borrow_cost).mean());net=float((aa.net-bb.net).mean())
                    np.testing.assert_allclose(net,gross-tc-bc,atol=1e-12,rtol=0)
                    attribution.append(dict(book=book,arm=arm,control=control,fee=fee,net_delta_bp=net*1e4,gross_delta_bp=gross*1e4,
                                            exposure_bp=exposure*1e4,covariance_bp=(gross-exposure)*1e4,extra_trading_bp=tc*1e4,extra_borrow_bp=bc*1e4))
    pd.DataFrame(attribution).to_csv(OUT/'attribution.csv',index=False)
    yearly=[]
    for key,g in w.assign(year=w.date.dt.year).groupby(['book','arm','rule','fee','year']):
        r=g.net.to_numpy();nav=np.r_[1.,np.cumprod(1+r)]
        yearly.append(dict(zip(['book','arm','rule','fee','year'],key))|dict(weeks=len(g),total_return=float(nav[-1]-1),
            max_drawdown=float(np.max(1-nav/np.maximum.accumulate(nav))),sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(52)),
            long_bp=float(g.long_net.mean()*1e4),short_bp=float(g.short_net.mean()*1e4)))
    pd.DataFrame(yearly).to_csv(OUT/'yearly.csv',index=False)
    directions=[]
    for family in ('old','B0','P6'):
        for suffix in ('plain','fixed5'):
            arm=family+'_'+suffix;old='old_'+suffix
            directions.append(dict(arm=arm,long_weeks=int(s[arm].gt(0).sum()),changed_vs_old=int(s[arm].gt(0).ne(s[old].gt(0)).sum())))
    pd.DataFrame(directions).to_csv(OUT/'directions.csv',index=False)
    valdays=pd.DatetimeIndex(lock['validation_dates']);missing_summary=[]
    for family in ('old','B0','P6'):
        for phase,kind in (('train','base'),('train','augmentation'),('validation','base')):
            q=miss.loc[miss.family.eq(family)&miss.phase.eq(phase)&miss.kind.eq(kind)]
            if phase=='validation':q=q.loc[q.date.isin(valdays)]
            missing_summary.append(dict(family=family,phase=phase,kind=kind,affected_weeks=q.date.nunique(),
                affected_baskets=len(q),missing_long=int(q.missing_long.sum()),missing_short=int(q.missing_short.sum())))
    ms=pd.DataFrame(missing_summary);ms.to_csv(OUT/'missing_summary.csv',index=False)
    def metric_table(book,rule):
        rows=[]
        for family in (('old','B0','P6') if book=='old' else ('old',book)):
            for suffix in ('plain','fixed5'):
                arm=family+'_'+suffix;q=m.loc[m.book.eq(book)&m.rule.eq(rule)&m.arm.eq(arm)].set_index('fee')
                def pair(col,scale=1,digits=2):return ' / '.join(f'{q.at[f,col]*scale:.{digits}f}'+('%' if scale==100 else '') for f in (10,30))
                rows.append([family,'无增强' if suffix=='plain' else '固定5%',pair('annual_calendar',100),pair('max_drawdown',100),pair('sharpe',digits=3),pair('worst10_avg',100),pair('long_bp'),pair('short_bp')])
        return table(pd.DataFrame(rows,columns=['40/10信号来源','增强','净复合年化10/30bp','最大回撤10/30bp','夏普10/30bp','最差10%周均10/30bp','多腿bp/周10/30bp','空腿bp/周10/30bp']))
    steps=[]
    for family in ('old','B0','P6'):
        for suffix in ('plain','fixed5'):
            arm=family+'_'+suffix;rr=[r for r in lock['selections'] if r['arm']==arm]
            assert [r['seed'] for r in rr]==[42,43,44]
            steps.append([family,suffix,' / '.join(str(r['step']) for r in rr),' / '.join('是' if r['step']>10 else '否' for r in rr),
                          ' / '.join(f"{r['val_bce']:.6f}" for r in rr)])
    reference=[]
    for book in ('old','B0','P6'):
        for rule in ('wide','full'):
            a=m.loc[m.book.eq(book)&m.arm.eq('old_fixed5')&m.rule.eq(rule)].set_index('fee')
            b=rm.loc[rm.book.eq(book)&rm.rule.eq(rule)].set_index('fee')
            reference.append([book,rule,' / '.join(f"{a.at[f,'annual_calendar']*100:.2f}%" for f in (10,30)),
                              ' / '.join(f"{b.at[f,'annual_calendar']*100:.2f}%" for f in (10,30))])
    p=pairs.query("rule=='full' and control.str.startswith('old_')",engine='python')
    pairtable=table(pd.DataFrame([[r.book,r.arm,r.control,r.fee,f'{r.mean_delta_bps:.2f}',f'[{r.mean_lo_bps:.2f}, {r.mean_hi_bps:.2f}]'] for r in p.itertuples()],
                                 columns=['实际20/10','信号','对照','费档','净均差bp/周','95%块区间']))
    results='\n\n'.join(f"## 实际{book}20/10：{'全多/全空' if rule=='full' else '40%/70%宽档'}\n\n{metric_table(book,rule)}" for book in ('old','B0','P6') for rule in ('full','wide'))
    resource=pd.read_csv(OUT/'resource.csv')
    forward=pd.read_csv(OUT/'forward_checks.csv')
    assert len(forward)==6 and forward.max_score_error.max()<2e-5 and forward.max_vector_error.max()<2e-5
    compare=[]
    for family,book in (('B0','B0'),('B0','old'),('P6','P6'),('P6','old')):
        for suffix in ('plain','fixed5'):
            arm=family+'_'+suffix;old='old_'+suffix
            q=m.loc[m.book.eq(book)&m.rule.eq('full')&m.fee.eq(10)].set_index('arm')
            compare.append([family,book,suffix,f"{q.at[arm,'annual_calendar']*100:.2f}%",f"{q.at[old,'annual_calendar']*100:.2f}%",
                            f"{(q.at[arm,'annual_calendar']-q.at[old,'annual_calendar'])*100:+.2f}"])
    text=f"""# B0/P6自选40/10信号→自身及旧20/10：473周统一验收

2026-11-19，Asia/Shanghai，complete。[协议及用户授权](research_20261119_bce_native40_transfer_protocol.md)，父研究[等价初始化](research_20261119_bce_equiv_init_results.md)。本批实际新训18个配比头，四组迁移全部完成；不重训股票主干、不部署。轻量证据见[目录](evidence/20261119_bce_native40_transfer/README.md)。

## 结论如何读

四组迁移在本批两增强、两规则、两费档的32个对应比较中，净复合年化均低于同一本实际20/10账本的统一旧信号。B0固定5%相对无增强改善，P6固定5%退步；这不是所有风险指标被旧信号严格支配，需逐项看回撤和尾部。多数净均差区间仍跨零，不将历史未胜出解释为普适不可能。

本批比较的是各自主干评分生成的40/10输入及名单监督，对旧与自身20/10的迁移；不是只换编码器、也不是各自20/10专训。全矩阵固定原473/123/295，但经用户明确授权，原生覆盖门槛改旧门槛、缺收益记平；不能称为保留原生覆盖门槛的完整原生训练。

10bp全切换的主比较如下，末列为净复合年化百分点差；所有0/5%均预登记，未按测试择优。

{table(pd.DataFrame(compare,columns=['新信号来源','实际20/10','增强','新信号年化','统一旧信号年化','新减旧百分点']))}

收益与尾部、回撤、费用需结合下方全表判断；本批没有对市场/行业暴露做alpha归因。比较使用已读历史，更多seed或改名不会恢复独立盲测。

## 固定项及实际改变项

原473训练周2001-11-23至2017-12-15、原123验证监督周（133原验证推理范围）、295测试决策周2021-01-08至2026-10-09全部保留，日期逐项与父锁对齐。测试首入场2021-01-11至末退出2026-10-19。所有473周的基础样本权重至少.95、周总权重1，没有以缺标签、原生不可用或标签极端删周。

三个信号来源均为129→32投影、分腿均值、LayerNorm、32→64→32残差FFN、64→1，100步全批Adam lr=.001、weight_decay=.01；第11步起冻结投影，其余层继续更新，最低验证BCE选步后平均三个logit。无注意力分支、无新损失、无温度搜索。

旧主干是三个abl2000_base best3/2/6与相应头seed42/43/44；B0/P6分别使用已登记的单个research_B0_F0_s42_a1/research_P6_F0_s42_a1主干，加三个初始化seed42/43/44。旧三编码器的多样性与新单编码器三头不同，不把整体差异说成只改变某一个权重张量。

新B0/P6各自冻结原生预处理和Q1训练/标准验证/历史特征，评分轮次仍B0多1–3空6–8、P6多6–8空3–5；B0逐轮截面z融合、P6原始decision_score融合。向量分别为第3/8轮128维。仅当前available用旧缓存事前可知门槛，不读未来valid决定选股。信号构建沿旧研究筛选规则：多头取自身Top50，空头从排除Top50后的候选按自身空头分数升序，2018后用旧季度资格，基础取40/10。相同分数输入时新候选算法已在473周逐项复现原groups_for。

“自己的标签”是自己名单内股票收益的等权均值，收益源、区间和券费公式仍原旧定义，不是同时切换Q1标签。每股收益r=expm1(y_old)，在valid=False/非有限处按用户授权置0：

\\[
g_t=\\frac1{{40}}\\sum_{{i\\in L_t}}r_{{i,t}}+\\frac1{{10}}\\sum_{{i\\in S_t}}r_{{i,t}}+0.08/52,\\qquad y_t=\\mathbf1\\{{g_t>0\\}}.
\\]

该目标比较多腿与扣固定券费后的空腿，不是预测市场绝对涨跌。缺报价记0属于估值假设，不表示真实赚亏为0或能按期平仓。

每个来源/seed的mu/sd只在473周基础40/10输入拟合，0/5%共享；不把表示语义不同的B0/P6输入硬套等价初始化。旧无增强基础和123验证标签逐位未变，测试logit复现父批。

## 473周的缺失与增强审计

旧门槛修订前原生30周无股票可选；此处按授权保留，原生与旧门槛逐日数量见coverage.csv。最终各阶段覆盖摘要：

{table(coverage.groupby(['arm','phase']).agg(weeks=('date','size'),native_empty_weeks=('native_available',lambda x:int((x==0).sum())),minimum_legacy_available=('legacy_available','min'),newly_allowed_stock_weeks=('newly_allowed','sum')).reset_index())}

基础和变体收益缺失如下。验证这里只统计原123个参与损失的周，不把另外10周计作新监督。基础训练全部473周，缺失股票仍在原名单并记0，不以未来结果换股：

{table(ms)}

标签正负分布及相对旧标签翻向：

{table(pd.read_csv(OUT/'label_summary.csv'))}

固定5%共用原473索引里的236启用周，61118、SeedSequence([61118,原日期索引,1])，各自Top50/Bottom20池最多3次抽40/10，无放回、输入仍按原排名排列。只拒绝重复或非法名单，不按未来valid丢变体。有效周.95原损失+.05变体平均；无变体周原损失1。这里5%是启用周内损失权重，不是5%训练周或5%资金。

{table(aug)}

原旧5%会跳过未来标签不完整的变体，本批统一规则会把它们按记平保留。因此旧5%也必须重训。原旧5%保留为参考，不能与新信号的统一规则混作主对照：

{table(pd.DataFrame(reference,columns=['实际20/10','规则','本批统一旧5%年化10/30bp','原始旧5%年化10/30bp']))}

## 验证选步

{table(pd.DataFrame(steps,columns=['来源','增强','选步42/43/44','是否过第10步','验证BCE42/43/44']))}

规则本身不变，检查点可在冻结前被选中；所有18条轨迹完整训练100步。不能把较晚/较早选中自动解释为拟合好坏。训练损失共1800行，18份输入/监督/权重/统计指纹；同一新来源的三seed原始数据完全相同，仅初始化不同。

## 执行口径与全指标

实际交易仍各自原锁定B0/P6或旧三主干20/10，用同一执行面板、成交/券源、流动性、受阻退出代理与净值漂移；不把信号构建的旧覆盖门槛偷偷换入实际交易规则。每个来源的同一测试logit复用于两套交易名单，没有按交易对象另训。每格按10bp+8.35%年券费 / 30bp+12%年券费顺序。

全切换多头比例0/100%，宽档40/70%，总名义预算1、同一NAV分母；受阻开仓可留现金。复合年化按实际首入场到末退出的日历时长；最大回撤为周估值；最差10%是295周最差30周净收益均值。两腿为对同一总NAV的净周贡献bp/周，二者和为净周均，不能相加成复合年化。

{results}

## 配对差与年度结果

四组对照均在同日期、同实际20/10、同增强及费用下比较新减旧，8周移动块5000次自助，95%区间。全切换净周均差：

{pairtable}

完整配对及尾部差见paired.csv；逐年区间收益、年内回撤、夏普及两腿贡献见yearly.csv。多重比较未校正；区间跨零不是严格等效。未按295周表现改阈值、轮次、增强比例或交易名单。

同一本账本按全切换与宽档毛收益恢复每周满额多腿L、空腿S，令新减旧的判多指示差为Delta w，则毛收益差=Delta w*(L−S)。平均毛差可恒等分成mean(Delta w)*mean(L−S)与cov(Delta w,L−S)，再扣交易费用差与券费差得到净差，全部逐周断言闭合。该分解只是已读历史金额记账，不识别独立alpha。全切换归因，单位bp/周：

{table(pd.DataFrame(attribution).round(3))}

## 验收与执行

- 18头实训、1800条轨迹；473/123/295日期与原锁逐项对齐；没有删周、缺值股票没有从名单中剔除。缺收益记平是用户明确授权的三来源统一规则。
- 旧无增强473及123标签逐位等于原标签，295周logit在1e-6内复现；旧/P6实际20/10无增强共{verify['archived_weekly_rows_reproduced']}行与原独立账本逐周净收益相同，容差1e-12。
- 六个模型/阶段各抽一个日期，独立CPU完整前向核对GPU缓存分数和向量，误差见forward_checks.csv；新候选算法的旧分数473周复现检查通过。
- 主指标56行、周账本16520行，另原5%参考12行/3540周账本；唯一键、295周、long_net+short_net=net均通过。四组新对旧及增强差共{len(pairs)}个配对比较。
- 推理以小批次GPU在原rolling_anchor_seeds_v1 CPU回放阶段执行，其GPU导出阶段暂停并empty_cache，不改其锁和队列。36个模型/阶段导出，最大PyTorch分配{resource.peak_gpu_bytes.max()/2**20:.1f}MiB；实际暂停和耗时见prepare.log/resource.csv。资源等待含在秒数里，不作为纯算力基准。
- 首次读取含中文锁时使用Windows默认GBK报错，尚未推理/训练；统一显式UTF-8后从头准备，失败日志prepare_attempt1.log保留。训练/选步没有因该错误变化。
- P6训练期第8轮推理开始时，状态文件被并发读取造成Windows原子替换PermissionError，保留prepare_attempt2.log及prepare_before_io_retry.py。仅增加有限重试和恢复记账；缓存签名迁移只允许这一份已保存源码哈希变化，其余模型/数据/日期依赖全部相同才可复用，未改数值路径。已完成但尚未写出耗时的5个P6缓存导出标为reused、时间缺失，不伪造实测耗时。配比头在恢复后才开始训练。
- 命令：`$env:PYTHONPATH='scripts;src'; python -u outputs/bce_native40_transfer_v1/prepare.py`，然后`python -u outputs/bce_native40_transfer_v1/train_replay.py`，最后`python outputs/bce_native40_transfer_v1/report.py`。当前state/verification为complete，无本批待跑任务，不改其它在跑任务状态。

主干样本内训练名单和向量、原473由旧标签完整性挑出的历史、固定幸存池、最新修订数据、季度资格代理、记平估值和真实券源未知等限制继承。全部历史已读，本批不能声称独立样本外确认或可成交实盘收益。原始行情、向量、主干及18新头仅本机，未核实异地备份；轻量证据不支持脱离这些资产全量独立重训。
"""
    DOC.write_text(text,encoding='utf8')
    hashes=json.loads((OUT/'inputs.json').read_text(encoding='utf8'))
    for p in (Path(__file__),OUT/'check_forward.py','scripts/replay_kit_risk.py'):
        p=Path(p).resolve()
        with p.open('rb') as f:hashes[str(p)]=hashlib.file_digest(f,'sha256').hexdigest()
    (OUT/'inputs.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    verify.update(report_complete=True,paired_comparisons=len(pairs),training_trace_rows=len(trace),max_gpu_allocated_mib=float(resource.peak_gpu_bytes.max()/2**20))
    (OUT/'verification.json').write_text(json.dumps(verify,indent=2),encoding='utf8')
    (OUT/'state.json').write_text(json.dumps(verify,indent=2),encoding='utf8')
    files=['prepare.py','prepare_before_io_retry.py','train_replay.py','report.py','check_forward.py','prepare_attempt1.log','prepare_attempt2.log','prepare.log','train.log','dates.json','coverage.csv','resource.csv',
           'augmentation.csv','missing_returns.csv','missing_summary.csv','label_summary.csv','signal_baskets.csv','identity.csv','training_loss.csv',
           'validation_lock.json','signals.csv','metrics.csv','reference_metrics.csv','paired.csv','attribution.csv','forward_checks.csv','yearly.csv','directions.csv','inputs.json','verification.json','state.json']
    EVIDENCE.mkdir(parents=True,exist_ok=True);manifest=[]
    for name in files:
        src=OUT/name;dst=EVIDENCE/name;shutil.copyfile(src,dst)
        with src.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        with dst.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha
        manifest.append(dict(source=str(src.relative_to(ROOT)),copy=name,bytes=src.stat().st_size,sha256=sha))
    (EVIDENCE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    with (EVIDENCE/'manifest.json').open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    (EVIDENCE/'README.md').write_text(f'''# B0/P6自选40/10迁移：轻量证据

2026-11-19，Asia/Shanghai。[结果](../../research_20261119_bce_native40_transfer_results.md)。{len(manifest)}份逐字节复制件，来源及SHA256见[manifest](manifest.json)，manifest SHA256为`{sha}`。保留用户授权记平的覆盖和缺失审计、18头训练轨迹/锁、56行主指标、12行原5%参考、配对和年度结果。

全量行情、推理向量/评分、股票主干、18个配比权重及周账本仍仅本机原data/outputs，未核实异地备份；本目录不是独立全量重训包。源码复制件按原工作区依赖审阅，不在证据目录直接运行。未自动Git提交或推送。
''',encoding='utf8')
    print(table(pd.DataFrame(compare,columns=['source','book','augmentation','new_cagr','old_cagr','delta_pp'])))
    print('Published',DOC,'with',len(manifest),'verified evidence copies; manifest',sha)

if __name__=='__main__':main()
