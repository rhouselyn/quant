"""Verify independent retraining against the current pipeline and publish evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from replay_kit_risk import paired

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/bce_old_retrain_audit_v1'
PARENT=ROOT/'outputs/bce_native40_transfer_v1'
DOC=ROOT/'docs/research_20261119_bce_old_retrain_audit_results.md'
EVIDENCE=ROOT/'docs/evidence/20261119_bce_old_retrain_audit'

def table(frame):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows.extend('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(rows)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    m=pd.read_csv(OUT/'metrics.csv');w=pd.read_parquet(OUT/'weekly.parquet')
    sm=pd.read_csv(OUT/'single_metrics.csv');sw=pd.read_parquet(OUT/'single_weekly.parquet')
    parentw=pd.read_parquet(PARENT/'weekly.parquet');parentt=pd.read_csv(PARENT/'training_loss.csv')
    ps=pd.read_csv(PARENT/'signals.csv',parse_dates=['date']).set_index('date')
    s=pd.read_csv(OUT/'signals.csv',parse_dates=['date']).set_index('date')
    single=pd.read_csv(OUT/'single_signals.csv',parse_dates=['date']).set_index('date')
    t=pd.read_csv(OUT/'training_loss.csv');encoding=pd.read_csv(OUT/'encoding_checks.csv')
    full=pd.read_csv(OUT/'full_forward_checks.csv');native=pd.read_csv(OUT/'native_training_checks.csv')
    native_prices=pd.read_csv(OUT/'native_price_label_checks.csv');prices=pd.read_csv(OUT/'raw_label_checks.csv')
    losses=pd.read_csv(OUT/'loss_checks.csv');ident=pd.read_csv(OUT/'identity.csv').set_index(['arm','seed'])
    parent_ident=pd.read_csv(PARENT/'identity.csv').set_index(['arm','seed'])
    lock=json.loads((OUT/'validation_lock.json').read_text(encoding='utf8'))
    slock=json.loads((OUT/'single_validation_lock.json').read_text(encoding='utf8'))
    assert len(m)==48 and len(w)==14160 and len(sm)==24 and len(sw)==7080 and len(t)==1200
    assert len(pd.read_csv(OUT/'single_training_loss.csv'))==400 and len(native)==12 and len(full)==12
    assert len(encoding)==2673 and len(prices)==1304 and len(native_prices)==2608
    for q in (prices,native_prices):assert ((q.cache_gap>0)==(q.price_gap>0)).all() if 'price_gap' in q else ((q.cache_gap>0)==(q.arithmetic_gap>0)).all()
    assert native.batch_fingerprints_match.all()
    assert native[['initial_loss_error','first_val_error','selected_val_error']].abs().to_numpy().max()<2e-6
    checks=[];directions=[];tracechecks=[]
    for arm in s:
        suffix=arm.split('_')[1];control='old_'+suffix
        direction=int(s[arm].gt(0).ne(ps[control].gt(0)).sum())
        directions.append(dict(arm=arm,changed_directions=direction,max_logit_difference=float((s[arm]-ps[control]).abs().max())))
        assert direction==0,(arm,direction)
        for seed in (42,43,44):
            cp=torch.load(OUT/f'{arm}_{seed}.pt',weights_only=True)
            reference=torch.load(PARENT/f'{control}_{seed}.pt',weights_only=True)
            maxparam=max(float((v-reference['state'][k]).abs().max()) for k,v in cp['state'].items())
            entry=next(r for r in lock['selections'] if r['arm']==arm and r['seed']==seed)
            if arm.startswith('cached_'):
                for key in ('inputs','weights','normalization'):assert ident.at[(arm,seed),key]==parent_ident.at[(control,seed),key],(arm,seed,key)
                a=t.loc[t.arm.eq(arm)&t.seed.eq(seed)].set_index('step')
                b=parentt.loc[parentt.arm.eq(control)&parentt.seed.eq(seed)].set_index('step')
                delta=float((a[['train_bce','val_bce']]-b[['train_bce','val_bce']]).abs().to_numpy().max())
                assert delta<2e-6
                tracechecks.append(dict(arm=arm,seed=seed,max_loss_difference=delta))
            checks.append(dict(arm=arm,seed=seed,selected_step=entry['step'],max_parameter_difference=maxparam))
    reproduced=0
    for arm in s:
        control='old_'+arm.split('_')[1]
        a=w.loc[w.arm.eq(arm)]
        b=parentw.loc[parentw.arm.eq(control)]
        merged=a.merge(b,on=['book','date','rule','fee'],validate='one_to_one')
        assert len(merged)==3540
        np.testing.assert_allclose(merged.net_x,merged.net_y,atol=1e-12,rtol=0);reproduced+=len(merged)
    pd.DataFrame(checks).to_csv(OUT/'parameter_checks.csv',index=False)
    pd.DataFrame(tracechecks).to_csv(OUT/'trace_checks.csv',index=False)
    pd.DataFrame(directions).to_csv(OUT/'direction_checks.csv',index=False)
    enc=encoding.groupby(['seed','phase']).agg(stocks=('stocks','sum'),changed_windows=('changed_windows','sum'),
        max_cache_error=('max_cache_vector_error','max'),max_current_change=('max_current_vector_change','max')).reset_index()
    enc.to_csv(OUT/'encoding_summary.csv',index=False)
    assert encoding.max_cache_vector_error.max()<2e-6 and full.max_error.max()<2e-6
    mainrows=[];singlerows=[]
    for book in ('old','B0','P6'):
        for suffix in ('plain','fixed5'):
            for rule in ('wide','full'):
                a=m.loc[m.book.eq(book)&m.rule.eq(rule)&m.arm.eq('current_'+suffix)].set_index('fee')
                b=sm.loc[sm.book.eq(book)&sm.rule.eq(rule)&sm.arm.eq('single_'+suffix)].set_index('fee')
                def values(frame,col,scale=1,digits=2):return ' / '.join(f'{frame.at[f,col]*scale:.{digits}f}'+('%' if scale==100 else '') for f in (10,30))
                mainrows.append([book,suffix,rule,values(a,'annual_calendar',100),values(a,'max_drawdown',100),values(a,'sharpe',digits=3),
                                 values(a,'worst10_avg',100),values(a,'long_bp'),values(a,'short_bp')])
                singlerows.append([book,suffix,rule,values(a,'annual_calendar',100),values(b,'annual_calendar',100),values(b,'max_drawdown',100),values(b,'sharpe',digits=3)])
    pairs=[]
    for book in ('old','B0','P6'):
        for suffix in ('plain','fixed5'):
            for rule in ('wide','full'):
                for fee in (10,30):
                    a=w.loc[w.book.eq(book)&w.arm.eq('current_'+suffix)&w.rule.eq(rule)&w.fee.eq(fee)].sort_values('date')
                    b=sw.loc[sw.book.eq(book)&sw.arm.eq('single_'+suffix)&sw.rule.eq(rule)&sw.fee.eq(fee)].sort_values('date')
                    pairs.append(dict(book=book,suffix=suffix,rule=rule,fee=fee,**paired(b.net.to_numpy(),a.net.to_numpy())))
    pd.DataFrame(pairs).to_csv(OUT/'single_paired.csv',index=False)
    fair=[];native_metrics=pd.read_csv(PARENT/'metrics.csv')
    for family,book in (('B0','B0'),('B0','old'),('P6','P6'),('P6','old')):
        for suffix in ('plain','fixed5'):
            for rule in ('wide','full'):
                for fee in (10,30):
                    a=parentw.loc[parentw.book.eq(book)&parentw.arm.eq(family+'_'+suffix)&parentw.rule.eq(rule)&parentw.fee.eq(fee)].sort_values('date')
                    b=sw.loc[sw.book.eq(book)&sw.arm.eq('single_'+suffix)&sw.rule.eq(rule)&sw.fee.eq(fee)].sort_values('date')
                    nm=native_metrics.loc[native_metrics.book.eq(book)&native_metrics.arm.eq(family+'_'+suffix)&native_metrics.rule.eq(rule)&native_metrics.fee.eq(fee)].iloc[0]
                    om=sm.loc[sm.book.eq(book)&sm.arm.eq('single_'+suffix)&sm.rule.eq(rule)&sm.fee.eq(fee)].iloc[0]
                    fair.append(dict(family=family,book=book,suffix=suffix,rule=rule,fee=fee,native_cagr=nm.annual_calendar,
                                     old_single_cagr=om.annual_calendar,**paired(a.net.to_numpy(),b.net.to_numpy())))
    fair=pd.DataFrame(fair);fair.to_csv(OUT/'native_vs_single.csv',index=False)
    fairrows=[]
    for family,book in (('B0','B0'),('B0','old'),('P6','P6'),('P6','old')):
        for suffix in ('plain','fixed5'):
            q=fair.loc[fair.family.eq(family)&fair.book.eq(book)&fair.suffix.eq(suffix)&fair.rule.eq('full')].set_index('fee')
            fairrows.append([family,book,suffix,' / '.join(f"{q.at[f,'old_single_cagr']*100:.2f}%" for f in (10,30)),
                            ' / '.join(f"{q.at[f,'native_cagr']*100:.2f}%" for f in (10,30)),
                            f"{q.at[10,'mean_delta_bps']:.2f} [{q.at[10,'mean_lo_bps']:.2f}, {q.at[10,'mean_hi_bps']:.2f}]"])
    steps=[]
    for label,items in (('main',lock['selections']),('single',slock['selections'])):
        for arm in dict.fromkeys(r['arm'] for r in items):
            rr=sorted([r for r in items if r['arm']==arm],key=lambda r:r['seed'])
            steps.append([label,arm,' / '.join(str(r['step']) for r in rr)])
    single_flips=[dict(arm=arm,changed_vs_three=int(single[arm].gt(0).ne(s['current_'+arm.split('_')[1]].gt(0)).sum())) for arm in single]
    pd.DataFrame(single_flips).to_csv(OUT/'single_directions.csv',index=False)
    text=f"""# 旧40/10按当前流程重训：训练器、输入、标签与集成审计

2026-11-19，Asia/Shanghai，complete。[协议及解释性增量](research_20261119_bce_old_retrain_audit_protocol.md)，父批[三来源473周](research_20261119_bce_native40_transfer_results.md)。主复现实际12头，单编码器解释对照另训4头、复用本批头42共2个；总新训16头，不重训股票主干。轻量证据见[目录](evidence/20261119_bce_old_retrain_audit/README.md)。

## 结论与可排除的问题

旧信号重新训练后仍复现高收益。独立训练器的cached_rebuild臂与当前Q1特征的fresh臂，295周都与父批旧信号0换向，48行主复现指标及14160条周净收益一致。因此当前共同训练流程并没有一接入就把旧信号训坏，旧高收益也不依赖直接沿用旧配比头权重。本次没有发现日期/股票错位、BCE符号反转、归一化样本串用、投影未冻结或验证选步接错。

需要修正此前对巨大差距的解释：旧三个头分别使用三个编码器，新B0/P6三个头共用一个编码器。统一旧向量到一个编码器后，B0实际20/10的固定5%年化由42.33%/25.14%降为29.02%/13.07%，接近B0专训27.80%/12.96%。这个条件对照支持向量多样性贡献，不能将整个旧新差距都说成B0专训或架构质量差。P6单旧编码器5%仍39.57%/16.63%，高于P6专训12.64%/−5.68%；其差额仍未唯一解释。

这是一项有范围的复现结论，不是证明全项目绝无bug，也不能唯一解释旧与B0/P6的全部差额。额外单编码器对照见下方，避免把三编码器对单编码器的差异忽略后统称“架构优劣”。

## 具体重做什么

原473周2001-11-23至2017-12-15、原123监督验证周（原133推理范围）、295测试决策周2021-01-08至2026-10-09，及各自实际旧/B0/P6的20/10账本均不变。原旧融合40/10名单和其标签固定；不是在Q1特征下重新排序一套旧名单。用户授权旧覆盖门槛、缺收益记平同时用于基础和变体，236启用周、708固定变体、周内.95/.05及总权重1不变。

cached_rebuild从旧三主干缓存重建输入；current_features让旧best3/2/6重新编码当前Q1训练、标准验证和历史特征，沿用每个旧模型自己的冻结预处理。先对原旧特征完整重编码，与缓存逐股票核对；若当前窗口与原窗口逐值相同可复用这次刚重算的向量。没有把旧缓存冒充已重算。三套编码器处理共{int(encoding.stocks.sum())}个所需股票窗口、2673个编码器日期行。

选择性编码前断言跨股关系门均为零，cross_summary/market_film/cross_stock均未启用；在三个seed的首训练日、末测试日对原/当前特征做12次完整forward校验，证明子集编码等价。最大缓存向量误差{encoding.max_cache_vector_error.max():.3g}，完整forward误差{full.max_error.max():.3g}。

当前特征输入并非没变化。每个seed的训练28071个所需股票窗口变化0；验证6150个窗口有3300个变化，测试14750个有4500个变化，具体向量幅度见表。这些改变没有让集成logit越过0阈值。

{table(enc.round(8))}

{table(pd.DataFrame(directions))}

## 独立标签及损失核对

直接读原weekly开盘/收盘及板块阈值，按下周开盘入场、再下周开盘退出，缺开盘或低开触发原回退条件时用当周收盘，独立构造算术收益；不从缓存log收益反推主标签。缺收益仍按授权记0。旧基础473、验证123及708变体共1304个篮子，和B0/P6各1304个篮子的二值方向，全部与原缓存监督一致，共3912例。独立算术价格与float32 log/exp会有细小幅度差，BCE使用的0/1目标没有差异，不把原始浮点gap哈希不同当作标签错误。

新训练循环不调用父批fit，不加载旧配比头初始化。手工种子重置后构造LegPool，仍129→32、两腿均值、残差FFN32→64→32、64→1，Adam .001/.01、100步、第11步开始冻结投影、最低验证BCE选步。独立计算softplus(z)−y*z与标准BCE，并用解析梯度(sigmoid(z)−y)*weight/473核对。

cached臂的输入、权重、归一化指纹与父批旧分支一致，100步训练/验证轨迹最大差见trace_checks.csv，全部低于2e−6。加权sum的浮点结合顺序可造成极小数值差，不影响选步或方向。

{table(pd.DataFrame(tracechecks))}

全部选步：

{table(pd.DataFrame(steps,columns=['登记','臂','seed42/43/44步']))}

## B0/P6有没有喂错或训错

不追加完整重训B0/P6，而是独立重建其473/123周的原始训练与验证批次：两套各708变体、三seed共12种配置，其输入/标签/权重/归一化指纹与父批完全匹配；独立价格监督方向一致。重置同初始化、重算首次训练损失和第1步Adam后的验证损失，并直接加载每个已选检查点重算验证BCE。首步及检查点误差均低于2e−6，实际为下表数值。这排除了“给新模型串了另一组名单/标签/标准化”“保存了不对应验证成绩的头”等已查机制；不是只凭训练日志判断。

{table(native)}

B0/P6的原生股票向量与分数在父批已有三阶段共6次独立完整CPU forward核对；本批不假装又完整重算了所有新主干，也不根据已读测试调其超参。原源的收益融合单位检查确认：B0逐轮截面z融合；P6金融score按原始单位平均，与fuse_probability_epochs的score聚合一致，不能因函数名含probability就误认为应对涨跌概率取平均。

## 主复现的效果

cached与current两臂的下表相同，金额均为10bp+8.35%年券费 / 30bp+12%年券费。全切换为多头0/100%，宽档40/70%，受阻开仓可留现金；复合年化按首入场2021-01-11至末退出2026-10-19日历跨度。最大回撤为周估值，最差10%是最差30周净收益均值；两腿为同NAV的净bp/周贡献。

{table(pd.DataFrame(mainrows,columns=['实际20/10','增强','规则','净复合年化10/30bp','最大回撤10/30bp','夏普10/30bp','最差10%周均10/30bp','多腿bp/周10/30bp','空腿bp/周10/30bp']))}

固定5%使用本批统一记平的708变体，因此指导旧20/10为51.41%/31.97%，不是原634变体的52.14%/32.73%。这项差异继承用户授权规则，原结果未覆盖。

## 只改编码器向量多样性的解释性对照

把三个头都改用旧seed42编码器的当前特征向量，头初始化仍42/43/44，名单、标签和实际交易对象不变。复用本轮新训头42，另外4头完整100步，先按原验证锁步再读新臂测试表现。该控制只改变向量来源，没有将旧融合排名改成单seed选股。

{table(pd.DataFrame(singlerows,columns=['实际20/10','增强','规则','三编码器年化10/30bp','单编码器年化10/30bp','单编码器最大回撤10/30bp','单编码器夏普10/30bp']))}

{table(pd.DataFrame(single_flips))}

与B0/P6实际单编码器路线的对照，均为同一实际账本、同增强、全切换；最后一列为新减单旧编码器的10bp净周均差与95%块区间，单位bp/周：

{table(pd.DataFrame(fairrows,columns=['新信号','实际20/10','增强','旧单编码器年化10/30bp','新单编码器年化10/30bp','10bp净均差及95%区间']))}

这是统一编码器向量数量后的解释性比较，旧融合名单及标签与新自己的名单/标签仍不同，故不能称所有数据变量已经一致。B0固定5%的点估计接近，不是统计上已证明等效；完整两费率、两规则区间见native_vs_single.csv。

其净周均/尾部差的8周块5000次配对区间见single_paired.csv。不同向量会改变验证选步，故此处是同一算法/预算下的输入来源控制，不是锁死各头同一更新阶段的纯参数效应；已读历史不作独立确认，也不把年化百分点差当作可相加的原因份额。

## 大差距仍有哪些解释边界

已查数据接线与训练复现没有发现具体错误，不等于新路线已经找到最优训练方案。旧信号三编码器集成、新信号单编码器集成；各自榜单和标签任务也不同。原验证选择出的新头较早只是结果，不能据此认定没有训练或擅自改成冻结后最优。上批B0/P6的训练标签相对旧标签分别翻向37/36周、验证翻向10/15周，不能把它们当成完全相同的监督任务。

原473周本就是按旧篮子未来标签完整性形成的固定历史集合；新篮子在55/69周存在部分记平，固定共同周也没有把这种历史条件变成中性采样。过去反复读取测试、幸存股票池、历史修订、记平、退出代理和真实券源未知仍存在。训练器审计不能证明旧高年化可持续，不能将所有差额简单宣布为新架构差或旧seed质量更高。

## 验收与复现

主12头×100步=1200行；增量4头×100步=400行，另复用本轮2头。48主指标/14160周与父批一一复现，24增量指标/7080周；二值标签3912例、原向量全所需窗口、12完整forward及12新路线首步/所选验证核对通过。所有主训练选步在测试回放前锁定，所有新文件置独立目录，旧原批未改。

命令：`$env:PYTHONPATH='scripts;src'; python -u outputs/bce_old_retrain_audit_v1/run.py`；新路线核对`python outputs/bce_old_retrain_audit_v1/check_native.py`；增量`python outputs/bce_old_retrain_audit_v1/single_encoder.py`；报告`python outputs/bce_old_retrain_audit_v1/report.py`。全CPU4线程，没有抢用现有GPU。主协议执行时原SHA256在validation_lock中，增量登记后原文本逐字节保留protocol_at_main_run.md，不改原锁。

当前本批complete，无待跑任务。原行情、输入数组、向量及权重仍仅本机，未核实异地备份；轻量证据保存源码、审计、锁、损失和核心表，不能脱离原资产全量独立重训。未部署、未自动Git提交或推送。
"""
    DOC.write_text(text,encoding='utf8')
    hashes=json.loads((OUT/'inputs.json').read_text(encoding='utf8'))
    protocol=ROOT/'docs/research_20261119_bce_old_retrain_audit_protocol.md'
    old_hash=lock['hashes'][str(protocol.resolve())]
    hashes.pop(str(protocol.resolve()),None)
    assert sha(OUT/'protocol_at_main_run.md')==old_hash
    hashes[str((OUT/'protocol_at_main_run.md').resolve())]=old_hash
    for p in (protocol,OUT/'report.py',OUT/'single_encoder.py',OUT/'check_native.py',PARENT/'signals.csv',PARENT/'weekly.parquet',PARENT/'training_loss.csv',PARENT/'identity.csv'):
        hashes[str(p.resolve())]=sha(p)
    for row in slock['selections']:hashes[str(Path(row['path']).resolve())]=sha(row['path'])
    (OUT/'inputs.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    verification=dict(status='complete',new_heads_trained=16,reused_new_heads_for_single=2,main_training_rows=1200,single_training_rows=400,
        original_price_basket_labels_checked=3912,original_cache_encoder_date_rows=2673,full_forward_checks=12,native_training_cases=12,
        main_weekly_rows_reproduced=reproduced,main_metric_rows=48,single_metric_rows=24,main_direction_changes=0)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf8')
    (OUT/'state.json').write_text(json.dumps(verification,indent=2),encoding='utf8')
    files=['run.py','check_native.py','single_encoder.py','report.py','protocol_at_main_run.md','run.log','native_check.log','single.log',
           'raw_label_checks.csv','native_price_label_checks.csv','encoding_checks.csv','encoding_summary.csv','full_forward_checks.csv',
           'loss_checks.csv','native_training_checks.csv','training_loss.csv','single_training_loss.csv','identity.csv','validation_lock.json',
           'single_validation_lock.json','signals.csv','single_signals.csv','metrics.csv','single_metrics.csv','single_paired.csv',
           'trace_checks.csv','parameter_checks.csv','direction_checks.csv','single_directions.csv','native_vs_single.csv','inputs.json','verification.json','state.json']
    EVIDENCE.mkdir(parents=True,exist_ok=True);manifest=[]
    for name in files:
        p=OUT/name;dest=EVIDENCE/name;shutil.copyfile(p,dest);digest=sha(p);assert sha(dest)==digest
        manifest.append(dict(source=str(p.relative_to(ROOT)),copy=name,bytes=p.stat().st_size,sha256=digest))
    (EVIDENCE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');manifest_sha=sha(EVIDENCE/'manifest.json')
    (EVIDENCE/'README.md').write_text(f'''# 旧40/10独立重训与接线审计证据

2026-11-19，Asia/Shanghai。[结果](../../research_20261119_bce_old_retrain_audit_results.md)。{len(manifest)}份逐字节复制件，来源、大小及SHA256见[manifest](manifest.json)，manifest SHA256为`{manifest_sha}`。保存原执行协议、16头训练记录、两条主复现与单编码器增量、3912例价格标签核对和新路线首步/已选检查点验证。

原始行情、输入数组、股票主干、配比头权重及完整周账本仍仅本机，未核实异地备份。本目录不是独立全量重训包，源码按原工作区路径审阅。未自动Git提交或推送。
''',encoding='utf8')
    print('Verified',reproduced,'main weekly rows,',len(native),'native training cases; published',DOC)
    print(sm.query("book=='old' and rule=='full'")[['arm','fee','annual_calendar','max_drawdown','sharpe']].to_string(index=False))

if __name__=='__main__':main()
