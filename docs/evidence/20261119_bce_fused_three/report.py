"""Verify and archive fused-roster training versus vectors-only and old BCE."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from attention_followup import LegPool
from replay_kit_risk import paired

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/bce_fused_three_v1'
PARENT=ROOT/'outputs/bce_three_encoder_v1'
DOC=ROOT/'docs/research_20261119_bce_fused_three_results.md'
EVIDENCE=ROOT/'docs/evidence/20261119_bce_fused_three'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def table(frame):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows.extend('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(rows)

def main():
    sys.stdout.reconfigure(encoding='utf8')
    torch.set_num_threads(4)
    m=pd.read_csv(OUT/'metrics.csv');w=pd.read_parquet(OUT/'weekly.parquet')
    s=pd.read_csv(OUT/'signals.csv',parse_dates=['date']).set_index('date');members=pd.read_csv(OUT/'member_logits.csv',parse_dates=['date']).set_index('date')
    lock=json.loads((OUT/'validation_lock.json').read_text(encoding='utf8'));trace=pd.read_csv(OUT/'training_loss.csv')
    ident=pd.read_csv(OUT/'identity.csv');full=pd.read_csv(OUT/'full_forward_checks.csv');rosters=pd.read_csv(OUT/'roster_changes.csv')
    actual=pd.read_csv(OUT/'actual_baskets.csv',parse_dates=['date']).set_index(['book','date'])
    assert len(m)==136 and len(w)==40120 and len(trace)==1200 and len(full)==18
    assert w.groupby(['book','arm','rule','fee']).size().eq(295).all()
    assert len(actual)==1475 and not w.duplicated(['book','arm','rule','fee','date']).any()
    np.testing.assert_allclose(w.long_net+w.short_net,w.net,atol=1e-12,rtol=0)
    assert full.max_score_error.max()<2e-5 and full.max_vector_error.max()<2e-5
    years=(pd.Timestamp('2026-10-19')-pd.Timestamp('2021-01-11')).days/365.2425
    for row in m.itertuples():
        q=w.loc[w.book.eq(row.book)&w.arm.eq(row.arm)&w.rule.eq(row.rule)&w.fee.eq(row.fee)].sort_values('date');r=q.net.to_numpy();nav=np.r_[1.,np.cumprod(1+r)]
        values=[nav[-1]**(1/years)-1,np.max(1-nav/np.maximum.accumulate(nav)),r.mean()/r.std(ddof=1)*np.sqrt(52),np.sort(r)[:30].mean(),q.long_net.mean()*1e4,q.short_net.mean()*1e4]
        np.testing.assert_allclose(values,[row.annual_calendar,row.max_drawdown,row.sharpe,row.worst10_avg,row.long_bp,row.short_bp],atol=1e-12,rtol=0)
    steps=[];labels=[];directions=[];bookchanges=[];validation=[]
    for family in ('B0','P6'):
        for suffix in ('plain','fixed5'):
            arm=family+'_fused_'+suffix;rows=[];valpreds=[]
            q=ident.loc[ident.family.eq(family)&ident.suffix.eq(suffix)]
            assert len(q)==3 and q.inputs.nunique()==3 and q.labels.nunique()==q.weights.nunique()==1
            cols=[arm+'_s'+str(seed) for seed in (42,43,44)]
            np.testing.assert_allclose(members[cols].mean(axis=1),s[arm],atol=1e-7,rtol=0)
            for seed in (42,43,44):
                r=next(x for x in lock['selections'] if x['arm']==arm and x['encoder_seed']==seed);rows.append(r)
                cp=torch.load(r['path'],weights_only=True);head=LegPool(False).eval();head.load_state_dict(cp['state'])
                with np.load(OUT/f'{family}_s{seed}_data.npz') as a:
                    flat=a['base'].reshape(-1,129);np.testing.assert_array_equal(cp['mu'].numpy(),flat.mean(0));np.testing.assert_array_equal(cp['sd'].numpy(),flat.std(0)+1e-6)
                    with torch.no_grad():
                        pred=head((torch.from_numpy(a['vx'])-cp['mu'])/cp['sd']).flatten();target=torch.tensor(a['vy']>0,dtype=torch.float32)
                        loss=F.binary_cross_entropy_with_logits(pred,target).item();valpreds.append(pred)
                np.testing.assert_allclose(loss,r['val_bce'],atol=2e-6,rtol=0)
            steps.append([arm,' / '.join(str(x['step']) for x in rows),' / '.join('是' if x['step']>10 else '否' for x in rows),' / '.join(f"{x['val_bce']:.6f}" for x in rows)])
            validation.append(dict(family=family,augmentation=suffix,ensemble_validation_bce=float(F.binary_cross_entropy_with_logits(torch.stack(valpreds).mean(0),target)),constant_half_bce=float(np.log(2)),positive_labels=int(target.sum())))
            for control in (family+'_vector_'+suffix,'old_'+suffix,family+'_fused_plain'):
                if control==arm:continue
                changed=s[arm].gt(0).ne(s[control].gt(0))
                directions.append(dict(arm=arm,control=control,long_weeks=int(s[arm].gt(0).sum()),changed_weeks=int(changed.sum()),changed_dates=';'.join(d.strftime('%Y-%m-%d') for d in s.index[changed])))
        with np.load(OUT/f'{family}_s42_data.npz') as a,np.load(PARENT/f'{family}_s42_data.npz') as b:
            for phase,col in [('train','y'),('validation','vy')]:labels.append(dict(family=family,phase=phase,weeks=len(a[col]),changed_labels=int(((a[col]>0)!=(b[col]>0)).sum()),new_positive=int((a[col]>0).sum())))
        for day in s.index:
            a=actual.loc[(family,day)];b=actual.loc[(family+'_fused',day)]
            bookchanges.append(dict(family=family,date=day,long_overlap=len(set(a.longs.split('|'))&set(b.longs.split('|'))),short_overlap=len(set(a.shorts.split('|'))&set(b.shorts.split('|')))))
    for name,rows in [('directions',directions),('label_changes',labels),('actual_roster_changes',bookchanges)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    pd.DataFrame(steps,columns=['信号','seed42/43/44所选步','是否过第10步','验证BCE']).to_csv(OUT/'selected_steps.csv',index=False)
    pd.DataFrame(validation).to_csv(OUT/'validation_ensemble.csv',index=False)
    pairs=[]
    for family in ('B0','P6'):
        for book in ('old',family,family+'_fused'):
            for rule in ('wide','full'):
                for fee in (10,30):
                    q=w.loc[w.book.eq(book)&w.rule.eq(rule)&w.fee.eq(fee)].pivot(index='date',columns='arm',values='net')
                    for suffix in ('plain','fixed5'):
                        arm=family+'_fused_'+suffix
                        for control in (family+'_vector_'+suffix,'old_'+suffix):pairs.append(dict(kind='signal',book=book,arm=arm,control=control,rule=rule,fee=fee,**paired(q[arm].to_numpy(),q[control].to_numpy())))
                    pairs.append(dict(kind='augmentation',book=book,arm=family+'_fused_fixed5',control=family+'_fused_plain',rule=rule,fee=fee,**paired(q[family+'_fused_fixed5'].to_numpy(),q[family+'_fused_plain'].to_numpy())))
        for rule in ('wide','full'):
            for fee in (10,30):
                for prefix in ('old',family+'_vector',family+'_fused'):
                    for suffix in ('plain','fixed5'):
                        arm=prefix+'_'+suffix
                        a=w.loc[w.book.eq(family+'_fused')&w.arm.eq(arm)&w.rule.eq(rule)&w.fee.eq(fee)].sort_values('date')
                        b=w.loc[w.book.eq(family)&w.arm.eq(arm)&w.rule.eq(rule)&w.fee.eq(fee)].sort_values('date')
                        pairs.append(dict(kind='execution_book',book=family+'_fused',arm=arm,control=family,rule=rule,fee=fee,**paired(a.net.to_numpy(),b.net.to_numpy())))
    pairs=pd.DataFrame(pairs);assert len(pairs)==168;pairs.to_csv(OUT/'paired.csv',index=False)
    yearly=[]
    for key,g in w.assign(year=w.date.dt.year).groupby(['book','arm','rule','fee','year']):
        g=g.sort_values('date');r=g.net.to_numpy();nav=np.r_[1.,np.cumprod(1+r)]
        yearly.append(dict(zip(['book','arm','rule','fee','year'],key))|dict(weeks=len(g),total_return=float(nav[-1]-1),max_drawdown=float(np.max(1-nav/np.maximum.accumulate(nav))),
            sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(52)),long_bp=float(g.long_net.mean()*1e4),short_bp=float(g.short_net.mean()*1e4)))
    pd.DataFrame(yearly).to_csv(OUT/'yearly.csv',index=False)
    roster_summary=rosters.groupby(['family','phase']).agg(weeks=('date','size'),mean_long_overlap=('long_overlap','mean'),mean_short_overlap=('short_overlap','mean')).reset_index()
    roster_summary.to_csv(OUT/'roster_summary.csv',index=False)
    actual_summary=pd.DataFrame(bookchanges).groupby('family').agg(weeks=('date','size'),mean_long_overlap=('long_overlap','mean'),mean_short_overlap=('short_overlap','mean')).reset_index()
    actual_summary.to_csv(OUT/'actual_roster_summary.csv',index=False)
    def values(q,col,scale=1,digits=2):return ' / '.join(f'{q.at[f,col]*scale:.{digits}f}'+('%' if scale==100 else '') for f in (10,30))
    def metric_table(book,rule):
        rows=[]
        for arm in m.loc[m.book.eq(book)&m.rule.eq(rule),'arm'].unique():
            q=m.loc[m.book.eq(book)&m.rule.eq(rule)&m.arm.eq(arm)].set_index('fee')
            rows.append([arm,values(q,'annual_calendar',100),values(q,'worst10_avg',100),values(q,'long_bp'),values(q,'short_bp'),values(q,'sharpe',digits=3),values(q,'max_drawdown',100)])
        return table(pd.DataFrame(rows,columns=['信号','净复合年化10/30bp','最差10%周10/30bp','多腿净bp/周10/30bp','空腿净bp/周10/30bp','夏普10/30bp','最大回撤10/30bp']))
    summary=[]
    for family in ('B0','P6'):
        for book in ('old',family,family+'_fused'):
            for suffix in ('plain','fixed5'):
                rows=[]
                for prefix in ('old',family+'_vector',family+'_fused'):
                    q=m.loc[m.book.eq(book)&m.rule.eq('full')&m.arm.eq(prefix+'_'+suffix)].set_index('fee');rows.append(values(q,'annual_calendar',100))
                summary.append([family,book,suffix,*rows])
    compare=pd.DataFrame(summary,columns=['家族','实际20/10','增强','旧BCE年化10/30bp','只换向量年化10/30bp','融合名单年化10/30bp'])
    compare.to_csv(OUT/'comparison.csv',index=False)
    contrasts=[]
    for family in ('B0','P6'):
        for book in ('old',family,family+'_fused'):
            for suffix in ('plain','fixed5'):
                for rule in ('wide','full'):
                    for fee in (10,30):
                        q=m.loc[m.book.eq(book)&m.rule.eq(rule)&m.fee.eq(fee)].set_index('arm')
                        candidate=q.at[family+'_fused_'+suffix,'annual_calendar']
                        contrasts.append(dict(family=family,book=book,suffix=suffix,rule=rule,fee=fee,fused_cagr=candidate,
                            vector_cagr=q.at[family+'_vector_'+suffix,'annual_calendar'],old_cagr=q.at['old_'+suffix,'annual_calendar']))
    contrasts=pd.DataFrame(contrasts);contrasts.to_csv(OUT/'cagr_contrasts.csv',index=False)
    fixed=contrasts.loc[~contrasts.book.str.endswith('_fused')]
    verdict=f"原锁定交易名单的32个年化配对中，本次融合信号高于只换向量{int(fixed.fused_cagr.gt(fixed.vector_cagr).sum())}个，高于旧BCE{int(fixed.fused_cagr.gt(fixed.old_cagr).sum())}个。这里是历史点估计计数，风险指标和配对区间仍须逐项看，不能据此宣布统计上的全面优越性。"
    results='\n\n'.join(f"## 实际{book}20/10：{'全多/全空' if rule=='full' else '40%/70%宽档'}\n\n{metric_table(book,rule)}" for book in ('old','B0','P6','B0_fused','P6_fused') for rule in ('full','wide'))
    p=pairs.loc[pairs.kind.eq('signal')&pairs.rule.eq('full')&pairs.fee.eq(10)]
    pairrows=[[x.book,x.arm,x.control,f'{x.mean_delta_bps:.2f}',f'[{x.mean_lo_bps:.2f}, {x.mean_hi_bps:.2f}]'] for x in p.itertuples()]
    DOC.write_text(f'''# 三主干融合公共40/10名单：两类20/10执行的完整对照

2026-11-19，Asia/Shanghai，complete。[协议与用户范围确认](research_20261119_bce_fused_three_protocol.md)；父批[只换编码器向量](research_20261119_bce_three_encoder_results.md)。六真实股票主干复用已有8轮训练，12个BCE头实际新训各100步。轻量证据见[目录](evidence/20261119_bce_fused_three/README.md)。

## 结论读法与主比较

{verdict}

下表是同一实际账本、同增强、全多空的直接对照。先在原锁定B0/P6账本看融合名单训练的信号是否改善，再在融合账本固定信号看交易名单效应；不能把同时换信号和下单名单的差额全归于信号训练。测试已经反复读取，本批为探索历史复检，不根据结果再选规则。

{table(compare)}

old为旧三编码器40/10信号，vector为前批同家族seed42公共名单/标签、三个真实编码器，fused为本次三主干分数融合后的公共名单/标签、三个真实编码器。plain无增强，fixed5固定5%。实际book无_fused后缀是前批原锁定20/10名单，有_fused后缀是本次新融合20/10名单。旧BCE也直接套到融合账本，避免拿不同名单不配对比较。

具体结论须分开：本次B0融合名单专训信号指导原B0全多空，0/5%年化为4.25%/−10.22%、1.14%/−12.95%，低于只换向量；P6同配置为10.45%/−8.25%、10.82%/−7.87%，相对只换向量有所恢复，仍低于旧BCE。保留旧5%信号、仅换实际交易名单，B0由42.33%/25.14%升为51.90%/32.84%，而P6由53.01%/28.37%降为43.84%/20.59%。这区分了选股分数融合和专训择时信号；B0执行名单的净均差区间仍跨零，未确认稳定提升。

## 融合、训练与不变项

B0先按原long1–3/short6–8窗口逐轮截面z后平均，P6先按原long6–8/short3–5平均原金融score；所有seed固定这些原窗口，没有重新搜索。随后各seed、各腿在共同事前候选池标准化（sd+1e−6），42/43/44等权平均分数，按父native_groups生成公共40/10。不是平均排名，也没有平均跨编码器的原始向量。不同家族原始分数的单位不混用。各seed原生锁可能不同，本批刻意固定前批窗口以控制名单融合这一变化，不能称为各seed重新独立选轮的系统。

训练、验证、测试分别固定原473/123/295日期，标签、候选与收益源定义沿旧BCE；新公共名单自然改变相应输入及标签，不再声称只换向量。训练仍2001-11-23至2017-12-15，验证监督位于2018-04-06至2020-12-18，测试决策2021-01-08至2026-10-09，实际持有2021-01-11至2026-10-19。没有按新标签完整性删周；旧覆盖门槛及缺股票收益记0是用户既有授权，缺失逐股票留痕。

向量B0仍第3轮，P6仍第8轮；每头读取对应seed128维向量加side，129→32投影、分腿均值、LayerNorm/残差32→64→32/最终LayerNorm，拼接64→1。每seed在473基础输入拟合mu/sd，0/5%共用统计，不读验证/测试拟合。监督仍1[mean(simple_long40)+mean(simple_short10)+.08/52>0]。固定61118的236启用周，在新Top50/Bottom20池固定抽40/10共708变体；启用周基础.95、变体合计.05，其余1，每周权重1。

加权BCE之和/473，Adam lr=.001/weight_decay=.01、100步、第11步冻结投影；原123周最低BCE选检查点，不强制冻结后。12头全部锁定后才推理新测试，每家族三logit等权平均、0阈值判向，再映射宽档多头40/70%或全切换0/100%。同一信号复用所有交易对象，不按交易收益重新训练或挑头。

{table(pd.DataFrame(steps,columns=['信号','seed42/43/44所选步','是否过第10步','验证BCE']))}

已锁检查点平均logit后的原验证集BCE：

{table(pd.DataFrame(validation).round(6))}

四组集成BCE均高于恒定预测0.5的log(2)=.693147，说明在这个固定验证任务上，原训练算法没有得到更好的概率拟合。该诊断没有用于另挑规则或删成员；BCE与交易收益并非同一目标，也不能单凭此表证明未拟合、过拟合或唯一根因。新名单同时改变输入、监督和由训练拟合的统计，二值标签翻向较少也不保证输出和择时稳定。

原40/10名单重合数（满额40/10）及标签翻向：

{table(roster_summary.round(3))}

{table(pd.DataFrame(labels))}

原实际20/10与新融合20/10名单平均重合数（满额20/10）：

{table(actual_summary.round(3))}

## 执行边界与完整136行指标

实际融合20/10使用模型原生可用门槛，与构建信号的旧门槛分开。由于可用股票会影响P6行业均值及分解，不能直接屏蔽旧门槛的分数冒充原生推理；对两mask不同的测试日期另做原生完整forward，其余日期沿数值相同分数。成员支持须一致，未取交集或缺成员重配权。空侧仍沿实际流动性、融券资格及排除规则，受阻开仓可留现金。

每格10bp+8.35%年券费 / 30bp+12%年券费，同NAV名义预算1。复合年化按实际首入场至末退出日历；周Sharpe按sqrt(52)，最大回撤为周估值，最差10%周为最差30周净收益均值。多/空腿为同总NAV净bp/周贡献，合计净周均，不可相加成复合年化。完整换手和费用在metrics.csv，逐年收益、年内回撤、夏普、两腿在yearly.csv。

{results}

## 判向与配对不确定性

{table(pd.DataFrame(directions).drop(columns='changed_dates'))}

全部换向日期在directions.csv，各成员与集成logit在member_logits.csv/signals.csv。净周均配对差：10bp全多空，本次融合信号减对应对照，单位bp/周：

{table(pd.DataFrame(pairrows,columns=['实际账本','新信号','对照','净均差','8周块95%区间']))}

完整168项包括两费用、两规则下融合对只换向量/旧BCE、5%对无增强，以及同信号融合交易名单对原名单。8周移动块5000次自助区间；多重比较未校正，跨零不是严格等效，不能从历史点估计推出稳定未来增益。

## 验收与复现

六主干真实参数与旧数据指纹复核；18个模型/阶段独立完整CPU forward核对GPU分数和向量，最大分数误差{full.max_score_error.max():.3g}、向量误差{full.max_vector_error.max():.3g}。每家族各增强跨seed标签/权重指纹相同、输入不同；473/123/295完整，708变体不按未来有效性过滤。12头1200行训练、原验证选步锁、选中检查点验证BCE和标准化统计重算通过。

136指标/40120周账本、五套实际名单共1475行、唯一键、两腿净和、平均logit及136行独立汇总核验通过；原旧/原B0/原P6账本的旧BCE和只换向量对照16520条周收益复现，容差1e−12。新增输出不覆盖前批。GPU按正常独占锁等待当前批次后运行，不抢锁、不改其它队列；实际导出耗时及峰值见resource.csv。

最初预计GPU可用后10–25分钟，低估了完整多轮截面的导出。84项实测导出累计约56.4分钟，峰值PyTorch分配50.5MiB，另有数据组装、独立CPU前向、训练和回放时间；没有因此改变预算、实验集合或抢用前序锁。训练、验证、测试特征导出不使用未来收益筛选，缺标签及资格/旧输入链的41项既有指纹亦复核。

项目根目录设置PYTHONPATH=scripts;src后，依次python -u outputs/bce_fused_three_v1/prepare.py、python -u outputs/bce_fused_three_v1/train_replay.py、python outputs/bce_fused_three_v1/report.py。各epoch/阶段签名和完成哈希用于准备阶段恢复；配比训练不冒充可精确中断续训。当前complete，无本批待跑任务；不部署、不自动Git提交。

未重新训练股票主干或重新搜索各seed原生评分窗口、冻结时间及融合权重。继承样本内主干名单/向量、原473由旧完整标签形成的历史集合、幸存池/修订数据、记平与退出估值、真实券源未知限制。原行情、向量、主干和12头仅本机，未核实异地备份，轻量证据不是独立全量重训包。
''',encoding='utf8')
    hashes=json.loads((OUT/'inputs.json').read_text(encoding='utf8'))
    hashes.update(json.loads((OUT/'frozen_auxiliary.json').read_text(encoding='utf8')))
    for p,h in hashes.items():
        if Path(p).resolve()!=Path(__file__).resolve():assert sha(p)==h,('changed input',p)
    hashes[str(Path(__file__).resolve())]=sha(__file__);hashes[str(ROOT/'scripts/replay_kit_risk.py')]=sha(ROOT/'scripts/replay_kit_risk.py')
    (OUT/'inputs.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    verify=json.loads((OUT/'verification.json').read_text(encoding='utf8'));verify.update(report_complete=True,independent_metric_rows=136,paired_comparisons=168,
        full_forward_checks=18,input_hashes_verified=len(hashes),selected_validation_bce_verified=True,mean_logit_verified=True)
    for name in ('verification.json','state.json'):(OUT/name).write_text(json.dumps(verify,indent=2),encoding='utf8')
    names=['prepare.py','train_replay.py','report.py','finish.py','queue_entry.json','frozen_auxiliary.json','prepare.log','train.log','model_inventory.json','dates.json','full_forward_checks.csv','resource.csv','signal_baskets.csv',
        'roster_changes.csv','roster_summary.csv','missing_returns.csv','identity.csv','augmentation.csv','coverage.csv','training_loss.csv','validation_lock.json','member_logits.csv',
        'signals.csv','metrics.csv','actual_baskets.csv','directions.csv','label_changes.csv','selected_steps.csv','validation_ensemble.csv','actual_roster_changes.csv','actual_roster_summary.csv','paired.csv',
        'yearly.csv','comparison.csv','cagr_contrasts.csv','inputs.json','verification.json','state.json']
    EVIDENCE.mkdir(parents=True,exist_ok=True);manifest=[]
    sources=[(OUT/name,name) for name in names]+[(PARENT/'train_replay.py','fit_source.py'),(ROOT/'docs/research_20261119_bce_fused_three_protocol.md','protocol.md')]
    for src,name in sources:
        dest=EVIDENCE/name;shutil.copyfile(src,dest);h=sha(src);assert sha(dest)==h
        manifest.append(dict(source=str(src.relative_to(ROOT)),copy=name,bytes=src.stat().st_size,sha256=h))
    (EVIDENCE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');h=sha(EVIDENCE/'manifest.json')
    (EVIDENCE/'README.md').write_text(f'''# 三主干融合公共名单BCE证据

2026-11-19，Asia/Shanghai。[结果](../../research_20261119_bce_fused_three_results.md)。{len(manifest)}份源码、锁、完整训练轨迹、136指标、168配对及名单/标签/年度证据复制件，来源和SHA256见[manifest](manifest.json)，manifest SHA256为`{h}`。

全量行情、向量、股票主干、配比头和周账本仅本机，未核实异地备份。非独立全量重训包，源码按原工作区依赖运行；未自动Git提交或推送。
''',encoding='utf8')
    index=ROOT/'docs/INDEX.md';content=index.read_text(encoding='utf8')
    entry=next(line for line in content.splitlines() if line.startswith('三主干融合公共40/10名单BCE（'))
    replacement='三主干融合公共40/10名单BCE（2026-11-19完成）：[协议](research_20261119_bce_fused_three_protocol.md) · [完整结果](research_20261119_bce_fused_three_results.md) · [证据](evidence/20261119_bce_fused_three/README.md)。12头实训，原473/123/295、原锁定及融合20/10两类执行，136指标/40120周完成。'+verdict.split('这里')[0]
    index.write_text(content.replace(entry,replacement,1),encoding='utf8')
    ledger=ROOT/'docs/experiment_ledger.md';content=ledger.read_text(encoding='utf8')
    paragraph='2026-11-19 三主干分数融合公共40/10名单BCE：[协议](research_20261119_bce_fused_three_protocol.md) · [结果](research_20261119_bce_fused_three_results.md)。六真实编码器，固定原评分窗口、473/123/295及0/5%增强，新训12头各100步，比较旧/只换向量/融合名单信号；原20/10与融合20/10同时配对。136指标/40120周、18独立完整forward、16520冻结对照周复现与168配对验收通过。'+verdict+'已读历史，记平/原473选周/券源限制保留。'
    if 'research_20261119_bce_fused_three_results.md' not in content:
        anchor='配套阅读：架构与实现见 [实现与实验记录](hierarchy_implementation.md)。'
        assert anchor in content;ledger.write_text(content.replace(anchor,anchor+'\n\n'+paragraph,1),encoding='utf8')
    print(compare.to_string(index=False));print('Published complete report and',len(manifest),'verified evidence copies; manifest',h)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        (OUT/'state.json').write_text(json.dumps(dict(status='failed',phase='report',error=repr(exc)),indent=2),encoding='utf8')
        raise
