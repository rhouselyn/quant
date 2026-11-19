"""Verify and publish the fixed-roster representation comparison."""
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
OUT=ROOT/'outputs/bce_old_roster_vectors_v1'
DOC=ROOT/'docs/research_20261119_bce_old_roster_vectors_results.md'
EVIDENCE=ROOT/'docs/evidence/20261119_bce_old_roster_vectors'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def table(frame):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows.extend('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(rows)

def main():
    sys.stdout.reconfigure(encoding='utf8');torch.set_num_threads(4)
    m=pd.read_csv(OUT/'metrics.csv');w=pd.read_parquet(OUT/'weekly.parquet')
    s=pd.read_csv(OUT/'signals.csv',parse_dates=['date']).set_index('date');members=pd.read_csv(OUT/'member_logits.csv',parse_dates=['date']).set_index('date')
    trace=pd.read_csv(OUT/'training_loss.csv');ident=pd.read_csv(OUT/'identity.csv');loss=pd.read_csv(OUT/'loss_checks.csv')
    lock=json.loads((OUT/'validation_lock.json').read_text(encoding='utf8'));models=json.loads((OUT/'model_inventory.json').read_text(encoding='utf8'))['models']
    assert len(m)==24 and len(w)==7080 and len(trace)==1800 and len(lock['selections'])==18
    assert len(models)==9 and len({x['encoder_sha256'] for x in models})==9
    assert len(pd.read_csv(OUT/'common_baskets.csv'))==1599
    for suffix,g in ident.groupby('suffix'):assert len(g)==9 and g.labels.nunique()==g.binary_labels.nunique()==g.weights.nunique()==1 and g.inputs.nunique()==9
    assert not w.duplicated(['arm','rule','fee','date']).any() and w.groupby(['arm','rule','fee']).size().eq(295).all()
    np.testing.assert_allclose(w.net,w.long_net+w.short_net,atol=1e-12,rtol=0)
    assert loss.closed_form_error.max()<2e-6 and loss.gradient_error.max()<1e-8
    for row in loss.itertuples():
        first=trace.loc[trace.arm.eq(row.arm)&trace.head_seed.eq(row.seed)&trace.step.eq(1),'train_bce'].item()
        np.testing.assert_allclose(first,row.initial_loss,atol=2e-6,rtol=0)
    years=(pd.Timestamp('2026-10-19')-pd.Timestamp('2021-01-11')).days/365.2425
    for row in m.itertuples():
        g=w.loc[w.arm.eq(row.arm)&w.rule.eq(row.rule)&w.fee.eq(row.fee)].sort_values('date');r=g.net.to_numpy();nav=np.r_[1.,np.cumprod(1+r)]
        v=[nav[-1]**(1/years)-1,np.max(1-nav/np.maximum.accumulate(nav)),r.mean()/r.std(ddof=1)*np.sqrt(52),np.sort(r)[:30].mean(),g.long_net.mean()*1e4,g.short_net.mean()*1e4]
        np.testing.assert_allclose(v,[row.annual_calendar,row.max_drawdown,row.sharpe,row.worst10_avg,row.long_bp,row.short_bp],atol=1e-12,rtol=0)
    steps=[];validation=[];directions=[]
    for family in ('old','B0','P6'):
        for suffix in ('plain','fixed5'):
            arm=family+'_'+suffix;selected=sorted([x for x in lock['selections'] if x['arm']==arm],key=lambda x:x['encoder_seed']);ps=[]
            assert len(selected)==3
            for row in selected:
                seed=row['encoder_seed'];cp=torch.load(row['path'],weights_only=True);h=LegPool(False).eval();h.load_state_dict(cp['state'])
                with np.load(OUT/f'{family}_s{seed}_data.npz') as a:
                    flat=a['base'].reshape(-1,129);np.testing.assert_array_equal(cp['mu'],flat.mean(0));np.testing.assert_array_equal(cp['sd'],flat.std(0)+1e-6)
                    y=torch.tensor(a['vy']>0,dtype=torch.float32)
                    with torch.no_grad():pred=h((torch.from_numpy(a['vx'])-cp['mu'])/cp['sd']).flatten();value=float(F.binary_cross_entropy_with_logits(pred,y))
                np.testing.assert_allclose(value,row['val_bce'],atol=2e-6,rtol=0);ps.append(pred)
            cols=[arm+'_s'+str(seed) for seed in (42,43,44)];np.testing.assert_allclose(s[arm],members[cols].mean(1),atol=1e-7,rtol=0)
            steps.append([family,suffix,' / '.join(str(x['step']) for x in selected),' / '.join('是' if x['step']>10 else '否' for x in selected),' / '.join(f"{x['val_bce']:.6f}" for x in selected)])
            validation.append(dict(family=family,suffix=suffix,ensemble_bce=float(F.binary_cross_entropy_with_logits(torch.stack(ps).mean(0),y)),constant_half_bce=float(np.log(2)),positive_labels=int(y.sum())))
            for control in ('old_'+suffix,family+'_plain'):
                if control==arm:continue
                changed=s[arm].gt(0).ne(s[control].gt(0));directions.append(dict(arm=arm,control=control,long_weeks=int(s[arm].gt(0).sum()),changed_weeks=int(changed.sum()),changed_dates=';'.join(d.strftime('%Y-%m-%d') for d in s.index[changed])))
    pd.DataFrame(steps,columns=['向量来源','增强','seed42/43/44选步','是否过第10步','验证BCE']).to_csv(OUT/'selected_steps.csv',index=False)
    pd.DataFrame(validation).to_csv(OUT/'validation_ensemble.csv',index=False);pd.DataFrame(directions).to_csv(OUT/'directions.csv',index=False)
    pairs=[]
    for rule in ('wide','full'):
        for fee in (10,30):
            q=w.loc[w.rule.eq(rule)&w.fee.eq(fee)].pivot(index='date',columns='arm',values='net')
            for family in ('B0','P6'):
                for suffix in ('plain','fixed5'):
                    arm=family+'_'+suffix;control='old_'+suffix;pairs.append(dict(arm=arm,control=control,rule=rule,fee=fee,**paired(q[arm].to_numpy(),q[control].to_numpy())))
            for family in ('old','B0','P6'):
                arm=family+'_fixed5';control=family+'_plain';pairs.append(dict(arm=arm,control=control,rule=rule,fee=fee,**paired(q[arm].to_numpy(),q[control].to_numpy())))
    pairs=pd.DataFrame(pairs);assert len(pairs)==28;pairs.to_csv(OUT/'paired.csv',index=False)
    base=w.loc[w.arm.eq('old_plain')&w.fee.eq(10)];ff=base.loc[base.rule.eq('full')].set_index('date').reindex(s.index);ww=base.loc[base.rule.eq('wide')].set_index('date').reindex(s.index)
    direction=s.old_plain.gt(0);long=np.where(direction,ff.gross_ret,(ww.gross_ret-.6*ff.gross_ret)/.4);short=np.where(direction,(ww.gross_ret-.7*ff.gross_ret)/.3,ff.gross_ret);gap=long-short
    attribution=[]
    for family in ('B0','P6'):
        for suffix in ('plain','fixed5'):
            arm=family+'_'+suffix;control='old_'+suffix;dw=s[arm].gt(0).astype(float)-s[control].gt(0).astype(float)
            gross=float((dw*gap).mean());exposure=float(dw.mean()*gap.mean())
            for fee in (10,30):
                a=w.loc[w.arm.eq(arm)&w.rule.eq('full')&w.fee.eq(fee)].set_index('date').reindex(s.index)
                b=w.loc[w.arm.eq(control)&w.rule.eq('full')&w.fee.eq(fee)].set_index('date').reindex(s.index)
                np.testing.assert_allclose(a.gross_ret-b.gross_ret,dw*gap,atol=1e-12,rtol=0)
                tc=float((a.trading_cost-b.trading_cost).mean());bc=float((a.borrow_cost-b.borrow_cost).mean());net=float((a.net-b.net).mean());np.testing.assert_allclose(net,gross-tc-bc,atol=1e-12,rtol=0)
                attribution.append(dict(arm=arm,control=control,fee=fee,net_delta_bp=net*1e4,gross_delta_bp=gross*1e4,exposure_bp=exposure*1e4,covariance_bp=(gross-exposure)*1e4,extra_trading_bp=tc*1e4,extra_borrow_bp=bc*1e4))
    pd.DataFrame(attribution).to_csv(OUT/'attribution.csv',index=False)
    yearly=[]
    for key,g in w.assign(year=w.date.dt.year).groupby(['arm','rule','fee','year']):
        g=g.sort_values('date');r=g.net.to_numpy();nav=np.r_[1.,np.cumprod(1+r)]
        yearly.append(dict(zip(['arm','rule','fee','year'],key))|dict(weeks=len(g),total_return=float(nav[-1]-1),max_drawdown=float(np.max(1-nav/np.maximum.accumulate(nav))),sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(52)),long_bp=float(g.long_net.mean()*1e4),short_bp=float(g.short_net.mean()*1e4)))
    pd.DataFrame(yearly).to_csv(OUT/'yearly.csv',index=False)
    def values(q,col,k=1,digits=2):return ' / '.join(f'{q.at[f,col]*k:.{digits}f}'+('%' if k==100 else '') for f in (10,30))
    def metric_table(rule):
        rows=[]
        for family in ('old','B0','P6'):
            for suffix in ('plain','fixed5'):
                q=m.loc[m.arm.eq(family+'_'+suffix)&m.rule.eq(rule)].set_index('fee')
                rows.append([family,suffix,values(q,'annual_calendar',100),values(q,'worst10_avg',100),values(q,'long_bp'),values(q,'short_bp'),values(q,'sharpe',digits=3),values(q,'max_drawdown',100)])
        return table(pd.DataFrame(rows,columns=['向量来源','增强','净复合年化10/30bp','最差10%周10/30bp','多腿净bp/周10/30bp','空腿净bp/周10/30bp','夏普10/30bp','最大回撤10/30bp']))
    previous=pd.read_csv(ROOT/'outputs/bce_three_encoder_v1/metrics.csv');comparisons=[]
    for family in ('B0','P6'):
        for suffix in ('plain','fixed5'):
            old=m.loc[m.arm.eq('old_'+suffix)&m.rule.eq('full')].set_index('fee')
            now=m.loc[m.arm.eq(family+'_'+suffix)&m.rule.eq('full')].set_index('fee')
            before=previous.loc[previous.book.eq('old')&previous.arm.eq(family+'_three_'+suffix)&previous.rule.eq('full')].set_index('fee')
            comparisons.append([family,suffix,values(before,'annual_calendar',100),values(now,'annual_calendar',100),values(old,'annual_calendar',100)])
    pd.DataFrame(comparisons,columns=['新向量','增强','此前自身名单标签→旧20/10','本次旧名单标签→旧20/10','旧向量旧任务→旧20/10']).to_csv(OUT/'task_comparison.csv',index=False)
    pp=pairs.loc[pairs.rule.eq('full')&pairs.control.str.startswith('old_')&pairs.arm.str.startswith(('B0_','P6_'))];pr=[[r.arm,r.fee,f'{r.mean_delta_bps:.2f}',f'[{r.mean_lo_bps:.2f}, {r.mean_hi_bps:.2f}]'] for r in pp.itertuples()]
    DOC.write_text(f'''# 固定旧40/10名单标签：旧/B0/P6三向量来源实训对照

2026-11-19，Asia/Shanghai，complete。[冻结协议](research_20261119_bce_old_roster_vectors_protocol.md)；父批[旧重训审计](research_20261119_bce_old_retrain_audit_results.md)、[三编码器](research_20261119_bce_three_encoder_results.md)。18个BCE头全部本轮新训，各100步；三来源共同旧20/10执行。轻量证据见[目录](evidence/20261119_bce_old_roster_vectors/README.md)。

## 结论与比较边界

固定旧名单和标签后，更换向量来源仍足以改变这套BCE的结果。全多空旧向量0/5%年化50.01%/30.91%、51.41%/31.97%，B0为34.11%/14.03%、34.45%/14.42%，P6两组均19.54%/6.67%。因此此前差距确实包含表征来源及其在固定学习算法中的适配差异，不能全部归于换名单或标签。

但此前监督任务变化也有影响：同样B0三向量、同旧20/10执行，改回旧名单/标签后由24.87%/9.55%恢复至34.11%/14.03%；P6由9.08%/−3.19%恢复至19.54%/6.67%。这些是条件桥接，不能将CAGR差额线性拆成向量和名单标签的因果百分比。旧来源在本次同标签验证集的集成BCE约.68595，B0约.69016，P6约.69347；验证排序也支持旧表征更适配当前头。多数净均差块区间跨零，不主张未来收益差已被严格确认。

这是同一监督任务、同一实际交易账本下更换向量来源的对照，不再混用B0/P6自己选的40/10名单或标签。先直接看下面两费档、两规则全24指标，再用与前批的桥接表判断过去差额是否还包含任务变化。不能把年化百分点相加成原因份额，也不能把区间跨零称为严格等效。

{table(pd.DataFrame(comparisons,columns=['新向量','增强','此前自身名单标签→旧20/10','本次旧名单标签→旧20/10','旧向量旧任务→旧20/10']))}

同一新编码器在此前和本次的比较，同时改变了被选股票、对应监督及随训练输入拟合的统计，因此只识别这组任务变化的整体影响。当前三来源之间则固定这些股票/标签/权重，均按同一算法各自拟合归一化并验证选步。

## 固定项、向量身份与归一化

原473训练日期2001-11-23至2017-12-15；原123验证监督周（133推理周范围2018-04-06至2020-12-18）；295测试决策周2021-01-08至2026-10-09，实际持仓2021-01-11至2026-10-19。全部日期与原锁逐项一致，固定旧基础40/10、236启用周、61118和原日期索引流的708固定变体。缺股票收益记平，不因新来源可用性、未来valid或缺收益换股删周。仅一份公共名单：891个基础篮子加708个增强篮子共1599行。

标签直接使用旧审计的同一数组；另按原价格/收益缓存路径重建，基础473、验证123及708变体的二值方向逐位核对。所有9个来源/seed在每增强下标签及周权重指纹完全相同、输入指纹9个均不同。基础.95/变体合计.05只在236启用周生效，其余原名单1，周总权重1；无增强全部1。

旧来源为abl2000_base真实三seed的best3/2/6缓存，已由父审计完整重编码/价格监督复核；B0为三seed第3轮，P6为三seed第8轮，复用上一批完整580股票向量并核验其SHA及18项完整forward证据。没有本轮新增GPU推理或股票主干训练。三组共9个不同编码器参数SHA，18个BCE头全部新训，不能混为仅改变MLP初始化。

均值/标准差公式与拟合范围相同，各坐标系只在473基础周拟合mu/sd+1e−6，0/5%共享；禁止拿旧坐标的统计硬套新坐标。不同向量自然产生不同统计数值，因此比较的是固定学习算法更换表征后的效果，不是固定同一个仿射变换的坐标实验。源主干既定特征和预处理边界继承原批，旧/Q1原始输入及股票主干训练目标差异包含在“向量来源”整体中，不能进一步宣称已唯一识别纯架构优劣或纯信息量差。

## 训练与验证锁

模型LegPool(False)：128维股票向量加side，129→32投影，分腿均值、LayerNorm、残差32→64→32及最终LayerNorm，拼接64→1。损失=sum(weight*BCEWithLogits)/473，目标1[mean(simple_long40)+mean(simple_short10)+.08/52>0]；Adam .001/.01，100全批更新，第11步起冻结投影，最低原123周验证BCE选步，允许冻结前。三seed头各锁后平均原始logit、阈值0，不平均跨编码器原始向量。

{table(pd.DataFrame(steps,columns=['向量来源','增强','seed42/43/44选步','是否过第10步','验证BCE']))}

原123周同标签任务上的已锁集成验证BCE，未用于新增选规则：

{table(pd.DataFrame(validation).round(6))}

每个配置独立核对softplus(z)−y*z与标准BCE，并验解析梯度(sigmoid(z)−y)*weight/473；其初始损失和实际训练第1步前损失匹配。投影冻结断言在实际fit内；18个已选检查点重新计算验证BCE及训练mu/sd通过。旧6头参数及统计复现容差2e−6，旧两条信号295周判向相同、2360条周收益逐周误差≤1e−12。

## 全多/全空：共同旧20/10

每格顺序10bp+8.35%年券费 / 30bp+12%年券费，预算1、同NAV，执行/现金/退出代理/净值漂移全部同原旧20/10。净复合年化按真实首入场至末退出日历；最差10%是最差30周净收益均值，多/空腿是对总NAV的净bp/周贡献（两腿和为净周均，不可相加成复合年化），Sharpe按周收益sqrt(52)，回撤为周估值。

{metric_table('full')}

## 宽档40%/70%：共同旧20/10

{metric_table('wide')}

## 换向、费用与配对

{table(pd.DataFrame(directions).drop(columns='changed_dates'))}

换向具体日期见directions.csv，成员和集成logit见member_logits.csv/signals.csv。全切换新向量减同增强旧向量，净均差及8周移动块5000次95%区间，单位bp/周：

{table(pd.DataFrame(pr,columns=['信号','费档','净周均差','95%块区间']))}

完整28项包括两规则两费用的新减旧及5%减0、净均差及尾部差。多重比较未校正，历史复检不独立。逐年收益、年内回撤、夏普和两腿见yearly.csv，2026为部分年度。输入/监督全固定后，差额通过方向与仓位路径影响同一交易账本；不可能再归于换了一套股票下单。

B0无增强与旧方向不同51周，P6不同81周；P6判多276/295周。5%只让B0再换1周、P6换0周，后者参数/logit不同但所有离散仓位相同。10bp下B0相对旧的净差−21.51bp/周，毛差−16.91，另多交易/券费约4.60；P6净差−44.97，毛差−49.48，费用节省约4.51。因而这次主要损失来自方向选择对应的毛收益，并非把更多费用或不同股票误算给新向量。

同一旧账本恢复每周满额多腿L、空腿S，dw为新减旧判多指示差，则毛收益差=dw*(L−S)。其均值分为mean(dw)*mean(L−S)与cov(dw,L−S)，再扣交易/券费差得到净差，逐周及均值恒等式已断言；这是金额归因，不识别唯一经济alpha：

{table(pd.DataFrame(attribution).round(3))}

## 验收、复现与限制

18头1800条训练、9真实编码器、共同1599篮子/708变体、每增强9份标签/权重一致，原473/123/295完整；初始BCE/梯度、所有已选验证和平均logit通过。24指标/7080周唯一键、两腿净和及24行独立汇总通过，旧2360周复现。源码和所有引用缓存/权重/预处理/资格输入哈希逐项验收；复用的完整forward原证据位于父批，本轮未声称又做完整主干推理。

项目根目录设置PYTHONPATH=scripts;src，python -u outputs/bce_old_roster_vectors_v1/run.py，再python outputs/bce_old_roster_vectors_v1/report.py。不覆盖旧批、未复用旧BCE权重作为初始化、不改其它队列、不部署、不自动Git提交。当前complete，无本批待跑任务。

继承主干样本内表征和名单、原473按旧标签完整性形成的历史选择、幸存池/修订数据、记平与退出估值、真实券源未知限制。全部测试历史已经读取，不能将本轮称为新独立样本外确认。原数据、向量、9主干及18头仅本机，未核实异地备份；轻量证据不是脱离原资产的全量重训包。
''',encoding='utf8')
    hashes=json.loads((OUT/'inputs.json').read_text(encoding='utf8'))
    for p,h in hashes.items():
        if Path(p).resolve()!=Path(__file__).resolve():assert sha(p)==h,('changed input',p)
    for p in (Path(__file__),ROOT/'outputs/bce_three_encoder_v1/metrics.csv',ROOT/'scripts/replay_kit_risk.py'):hashes[str(p.resolve())]=sha(p)
    (OUT/'inputs.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    verify=json.loads((OUT/'verification.json').read_text(encoding='utf8'));verify.update(report_complete=True,independent_metric_rows=24,paired_comparisons=28,input_hashes_verified=len(hashes),selected_validation_verified=True,mean_logit_verified=True)
    for name in ('verification.json','state.json'):(OUT/name).write_text(json.dumps(verify,indent=2),encoding='utf8')
    names=['run.py','report.py','run.log','model_inventory.json','dates.json','common_baskets.csv','missing_returns.csv','identity.csv','loss_checks.csv','old_parameter_reproduction.csv',
        'training_loss.csv','validation_lock.json','signals.csv','member_logits.csv','metrics.csv','selected_steps.csv','validation_ensemble.csv','directions.csv',
        'paired.csv','attribution.csv','yearly.csv','task_comparison.csv','inputs.json','verification.json','state.json']
    EVIDENCE.mkdir(parents=True,exist_ok=True);manifest=[]
    sources=[(OUT/name,name) for name in names]+[(ROOT/'outputs/bce_three_encoder_v1/train_replay.py','fit_source.py'),(ROOT/'docs/research_20261119_bce_old_roster_vectors_protocol.md','protocol.md')]
    for src,name in sources:
        dest=EVIDENCE/name;shutil.copyfile(src,dest);h=sha(src);assert sha(dest)==h
        manifest.append(dict(source=str(src.relative_to(ROOT)),copy=name,bytes=src.stat().st_size,sha256=h))
    (EVIDENCE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');h=sha(EVIDENCE/'manifest.json')
    (EVIDENCE/'README.md').write_text(f'''# 固定旧名单三向量来源BCE证据

2026-11-19，Asia/Shanghai。[结果](../../research_20261119_bce_old_roster_vectors_results.md)。{len(manifest)}份源码、锁、18头训练/梯度/身份、公共名单及24指标/28配对的逐字节复制件，见[manifest](manifest.json)，manifest SHA256为`{h}`。原数据、向量、主干、BCE权重及完整周账本仅本机，未核异地备份；非独立全量重训包，源码按原工作区依赖复现，未自动Git提交。
''',encoding='utf8')
    print(table(pd.DataFrame(comparisons,columns=['source','augmentation','previous_own_task','current_legacy_task','old_legacy_task'])))
    print('Published',DOC,'with',len(manifest),'verified evidence copies; manifest',h)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        (OUT/'state.json').write_text(json.dumps(dict(status='failed',phase='report',error=repr(exc)),indent=2),encoding='utf8')
        raise
