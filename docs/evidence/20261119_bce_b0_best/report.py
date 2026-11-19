"""Independently verify checkpoint ablations, diagnose directions, and archive."""
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
from replay_equal_weight_model_books import legacy_inputs
from replay_kit_risk import paired

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/bce_b0_best_v1'
PARENT = ROOT/'outputs/bce_old_roster_vectors_v1'
DOC = ROOT/'docs/research_20261119_bce_b0_best_results.md'
EVIDENCE = ROOT/'docs/evidence/20261119_bce_b0_best'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write(p, x):
    p.write_text(json.dumps(x,indent=2),encoding='utf8')


def table(columns, rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def diagnostic(z,y):
    z=np.asarray(z,dtype=float); y=np.asarray(y,dtype=bool); p=1/(1+np.exp(-np.clip(z,-700,700))); q=z>0
    tp=int((q&y).sum()); tn=int((~q&~y).sum()); pos=int(y.sum()); neg=len(y)-pos
    return dict(n=len(y),positives=pos,predicted_long=int(q.sum()),accuracy=float((q==y).mean()),balanced_accuracy=.5*(tp/pos+tn/neg),bce=float((np.logaddexp(0,z)-y*z).mean()),brier=float(((p-y)**2).mean()),logit_mean=float(z.mean()),logit_sd=float(z.std()),true_positive=tp,true_negative=tn)


def main():
    sys.stdout.reconfigure(encoding='utf8');torch.set_num_threads(4)
    m=pd.read_csv(OUT/'metrics.csv');w=pd.read_parquet(OUT/'weekly.parquet');s=pd.read_csv(OUT/'signals.csv',parse_dates=['date']).set_index('date')
    trace=pd.read_csv(OUT/'training_loss.csv');ident=pd.read_csv(OUT/'identity.csv');checks=pd.read_csv(OUT/'loss_checks.csv')
    lock=json.loads((OUT/'validation_lock.json').read_text());models=json.loads((OUT/'model_inventory.json').read_text())['models']
    dates={k:pd.DatetimeIndex(v) for k,v in json.loads((OUT/'dates.json').read_text()).items()}
    assert len(m)==64 and len(w)==18880 and len(trace)==3000 and len(lock['selections'])==30
    assert not w.duplicated(['arm','rule','fee','date']).any() and w.groupby(['arm','rule','fee']).size().eq(295).all()
    np.testing.assert_allclose(w.net,w.long_net+w.short_net,atol=1e-12,rtol=0)
    years=(pd.Timestamp('2026-10-19')-pd.Timestamp('2021-01-11')).days/365.2425
    for row in m.itertuples():
        g=w.loc[w.arm.eq(row.arm)&w.rule.eq(row.rule)&w.fee.eq(row.fee)].sort_values('date');r=g.net.to_numpy();nav=np.r_[1.,np.cumprod(1+r)]
        a=[nav[-1]**(1/years)-1,np.max(1-nav/np.maximum.accumulate(nav)),r.mean()/r.std(ddof=1)*np.sqrt(52),np.sort(r)[:30].mean(),g.long_net.mean()*1e4,g.short_net.mean()*1e4]
        np.testing.assert_allclose(a,[row.annual_calendar,row.max_drawdown,row.sharpe,row.worst10_avg,row.long_bp,row.short_bp],atol=1e-12,rtol=0)
    for _,g in ident.groupby('suffix'):
        assert g.labels.nunique()==g.weights.nunique()==1
    assert checks.closed_form_error.max()<2e-6 and checks.gradient_error.max()<1e-8
    for row in checks.itertuples():
        first=trace.loc[trace.arm.eq(row.arm)&trace.head_seed.eq(row.seed)&trace.step.eq(1),'train_bce'].item()
        np.testing.assert_allclose(first,row.initial_loss,atol=2e-6,rtol=0)
    sn=pd.read_csv(OUT/'parameter_snapshots.csv')
    for (family,suffix,seed),g in sn.groupby(['family','suffix','seed']):
        q=g.set_index(['mode','step'])
        for mode in g['mode'].unique():
            after={'f10':10,'f30':30,'free':None}[mode]
            if after is not None:assert q.at[(mode,after),'projection_sha256']==q.at[(mode,100),'projection_sha256']
            else:assert q.at[(mode,30),'projection_sha256']!=q.at[(mode,100),'projection_sha256']
    source_checks=[]
    for seed in (42,43,44):
        with np.load(OUT/f'best_s{seed}_data.npz') as a,np.load(OUT/f'rawold_s{seed}_data.npz') as b:
            np.testing.assert_array_equal(a['base'],b['base']);np.testing.assert_array_equal(a['aug_x'],b['aug_x'])
        for suffix in ('plain','fixed5'):
            a=trace.loc[trace.arm.eq(f'best_{suffix}_f10')&trace.head_seed.eq(seed)].set_index('step')
            b=trace.loc[trace.arm.eq(f'rawold_{suffix}_f10')&trace.head_seed.eq(seed)].set_index('step')
            np.testing.assert_array_equal(a.train_bce,b.train_bce)
            source_checks.append(dict(seed=seed,suffix=suffix,train_inputs_exact=True,training_loss_all100_exact=True))
    pd.DataFrame(source_checks).to_csv(OUT/'source_training_checks.csv',index=False)
    d,dd,_,gap,_,_=legacy_inputs()
    ty=gap[dd.get_indexer(dates['test'])]>0
    targets={};members={phase:{} for phase in dates};steps=[]
    for row in lock['selections']:
        family,seed=row['family'],row['encoder_seed']
        p=PARENT/f'B0_s{seed}_data.npz' if family=='e3' else OUT/f'{family}_s{seed}_data.npz'
        cp=torch.load(row['path'],weights_only=True);h=LegPool(False).eval();h.load_state_dict(cp['state'])
        with np.load(p) as a:
            flat=a['base'].reshape(-1,129);np.testing.assert_array_equal(cp['mu'],flat.mean(0));np.testing.assert_array_equal(cp['sd'],flat.std(0)+1e-6)
            targets['train']=a['y']>0;targets['validation']=a['vy']>0;targets['test']=ty
            for phase,key in [('train','base'),('validation','vx'),('test','hx')]:
                with torch.no_grad():z=h((torch.from_numpy(a[key])-cp['mu'])/cp['sd']).flatten().numpy()
                members[phase][row['arm']+'_s'+str(seed)]=z
        val=diagnostic(members['validation'][row['arm']+'_s'+str(seed)],targets['validation'])['bce']
        np.testing.assert_allclose(val,row['val_bce'],atol=2e-6,rtol=0)
        t=trace.loc[trace.arm.eq(row['arm'])&trace.head_seed.eq(seed)]
        selected=int(t.loc[t.val_bce.idxmin(),'step']);assert selected==row['step']
        steps.append(dict(arm=row['arm'],seed=seed,step=selected,past10=selected>10,val_bce=val,encoder_epoch=3 if family=='e3' else next(x['best_epoch'] for x in models if x['encoder_seed']==seed)))
    oldlock=json.loads((PARENT/'validation_lock.json').read_text())
    for row in oldlock['selections']:
        if not row['arm'].startswith('old_'):continue
        seed=row['encoder_seed'];cp=torch.load(row['path'],weights_only=True);h=LegPool(False).eval();h.load_state_dict(cp['state'])
        with np.load(PARENT/f'old_s{seed}_data.npz') as a:
            for phase,key in [('train','base'),('validation','vx'),('test','hx')]:
                with torch.no_grad():z=h((torch.from_numpy(a[key])-cp['mu'])/cp['sd']).flatten().numpy()
                members[phase][row['arm']+'_s'+str(seed)]=z
    predictions={}
    for phase,values in members.items():
        out={}
        for arm in s:
            if arm.startswith('swap'):
                swapped=int(arm[4:6]);suffix=arm.split('_')[1]
                out[arm]=np.mean([values[f"{'best' if seed==swapped else 'e3'}_{suffix}_f10_s{seed}"] for seed in (42,43,44)],axis=0)
            else:out[arm]=np.mean([values[arm+'_s'+str(seed)] for seed in (42,43,44)],axis=0)
        predictions[phase]=pd.DataFrame(out,index=dates[phase])
    np.testing.assert_allclose(predictions['test'][s.columns],s,atol=1e-6,rtol=0)
    assert predictions['test'].gt(0).eq(s.gt(0)).all().all()
    member_rows=[];diag=[]
    prior=targets['train'].mean();priorlogit=np.log(prior/(1-prior))
    for phase,pred in predictions.items():
        pred.index.name='date';pred.to_csv(OUT/f'{phase}_logits.csv')
        for arm in pred:diag.append(dict(phase=phase,arm=arm,**diagnostic(pred[arm],targets[phase])))
        for name,z in [('constant_half',np.zeros(len(pred))),('training_prior',np.full(len(pred),priorlogit))]:diag.append(dict(phase=phase,arm=name,**diagnostic(z,targets[phase])))
        for arm,z in members[phase].items():member_rows.append(dict(phase=phase,member=arm,**diagnostic(z,targets[phase])))
    pd.DataFrame(diag).to_csv(OUT/'prediction_diagnostics.csv',index=False);pd.DataFrame(member_rows).to_csv(OUT/'member_diagnostics.csv',index=False);pd.DataFrame(steps).to_csv(OUT/'selected_steps.csv',index=False)
    for suffix in ('plain','fixed5'):
        for phase in predictions:
            np.testing.assert_array_equal(members[phase][f'best_{suffix}_f10_s44'],members[phase][f'e3_{suffix}_f10_s44'])
    val={r['arm']:r['ensemble_bce'] for r in pd.read_csv(OUT/'validation_ensemble.csv').to_dict('records')}
    for choice in lock['freeze_choices']:
        candidates=[f"best_{choice['suffix']}_{mode}" for mode in ('f10','f30','free')]
        assert choice['arm']==min(candidates,key=lambda a:(val[a],candidates.index(a)))
    directions=[];pairs=[];attribution=[];yearly=[]
    comparisons=[]
    for suffix in ('plain','fixed5'):
        for family in ('e3','best','rawold','swap42','swap43'):
            arm=f'{family}_{suffix}_f10'
            comparisons.append((arm,'old_'+suffix))
        comparisons += [(f'best_{suffix}_f10',f'e3_{suffix}_f10'),(f'swap42_{suffix}_f10',f'e3_{suffix}_f10'),(f'swap43_{suffix}_f10',f'e3_{suffix}_f10'),(f'rawold_{suffix}_f10',f'best_{suffix}_f10'),(f'best_{suffix}_f30',f'best_{suffix}_f10'),(f'best_{suffix}_free',f'best_{suffix}_f10')]
    for family in ('e3','best','rawold','swap42','swap43'):
        comparisons.append((f'{family}_fixed5_f10',f'{family}_plain_f10'))
    base=w.loc[w.arm.eq('old_plain')&w.fee.eq(10)]
    ff=base.loc[base.rule.eq('full')].set_index('date').reindex(s.index);ww=base.loc[base.rule.eq('wide')].set_index('date').reindex(s.index)
    olddir=s.old_plain.gt(0)
    longs=np.where(olddir,ff.gross_ret,(ww.gross_ret-.6*ff.gross_ret)/.4)
    shorts=np.where(olddir,(ww.gross_ret-.7*ff.gross_ret)/.3,ff.gross_ret)
    realgap=longs-shorts
    grossdiag=[]
    for arm in s:
        target=realgap>0;pred=s[arm].gt(0).to_numpy();regret=np.where(pred,longs,shorts)-np.maximum(longs,shorts)
        grossdiag.append(dict(arm=arm,**diagnostic(s[arm],target),gross_regret_bp=float(regret.mean()*1e4),missed_gap_bp=float(np.abs(realgap[pred!=target]).mean()*1e4)))
    pd.DataFrame(grossdiag).to_csv(OUT/'actual20_gross_diagnostics.csv',index=False)
    for arm,control in comparisons:
        diff=s[arm].gt(0)!=s[control].gt(0)
        directions.append(dict(arm=arm,control=control,changed_weeks=int(diff.sum()),changed_dates='|'.join(s.index[diff].strftime('%Y-%m-%d')),long_weeks=int(s[arm].gt(0).sum())))
        dw=s[arm].gt(0).astype(float)-s[control].gt(0).astype(float)
        for rule in ('wide','full'):
            scale=.3 if rule=='wide' else 1.
            gross=float((scale*dw*realgap).mean());exposure=float(scale*dw.mean()*realgap.mean())
            for fee in (10,30):
                a=w.loc[w.arm.eq(arm)&w.rule.eq(rule)&w.fee.eq(fee)].set_index('date').reindex(s.index)
                b=w.loc[w.arm.eq(control)&w.rule.eq(rule)&w.fee.eq(fee)].set_index('date').reindex(s.index)
                np.testing.assert_allclose(a.gross_ret-b.gross_ret,scale*dw*realgap,atol=1e-12,rtol=0)
                tc=float((a.trading_cost-b.trading_cost).mean());bc=float((a.borrow_cost-b.borrow_cost).mean());net=float((a.net-b.net).mean())
                np.testing.assert_allclose(net,gross-tc-bc,atol=1e-12,rtol=0)
                attribution.append(dict(arm=arm,control=control,rule=rule,fee=fee,net_delta_bp=net*1e4,gross_delta_bp=gross*1e4,exposure_bp=exposure*1e4,timing_covariance_bp=(gross-exposure)*1e4,extra_trading_bp=tc*1e4,extra_borrow_bp=bc*1e4))
                pairs.append(dict(arm=arm,control=control,rule=rule,fee=fee,**paired(a.net.to_numpy(),b.net.to_numpy())))
    for key,g in w.assign(year=w.date.dt.year).groupby(['arm','rule','fee','year']):
        g=g.sort_values('date');r=g.net.to_numpy();nav=np.r_[1.,np.cumprod(1+r)]
        yearly.append(dict(zip(['arm','rule','fee','year'],key))|dict(weeks=len(g),total_return=float(nav[-1]-1),max_drawdown=float(np.max(1-nav/np.maximum.accumulate(nav))),sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(52)),long_bp=float(g.long_net.mean()*1e4),short_bp=float(g.short_net.mean()*1e4)))
    for name,rows in [('directions',directions),('paired',pairs),('attribution',attribution),('yearly',yearly)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    labels={'old':'旧BCE','e3':'B0第3轮','best':'B0各自best','rawold':'best＋旧特征','swap42':'只换42为best','swap43':'只换43为best'}
    def display(arm):
        if arm.startswith('old_'):return '旧BCE / '+arm.split('_')[1]
        family,suffix,mode=arm.split('_');return labels[family]+' / '+suffix+' / '+mode
    def values(q,k,scale=1,digits=2):return ' / '.join(f'{q.at[f,k]*scale:.{digits}f}'+('%' if scale==100 else '') for f in (10,30))
    def metric_table(rule,supplement=False):
        arms=[f'{family}_{suffix}'+('' if family=='old' else '_f10') for family in ('old','e3','best') for suffix in ('plain','fixed5')]
        if supplement:arms=[f'{family}_{suffix}_{mode}' for family,mode in [('best','f30'),('best','free'),('rawold','f10'),('swap42','f10'),('swap43','f10')] for suffix in ('plain','fixed5')]
        rows=[]
        for arm in arms:
            q=m.loc[m.arm.eq(arm)&m.rule.eq(rule)].set_index('fee')
            rows.append([display(arm),values(q,'annual_calendar',100),values(q,'worst10_avg',100),values(q,'long_bp'),values(q,'short_bp'),values(q,'sharpe',digits=3),values(q,'max_drawdown',100)])
        return table(['信号 / 增强 / 冻结','净复合年化10/30bp','最差10%周10/30bp','多腿净bp/周10/30bp','空腿净bp/周10/30bp','Sharpe10/30bp','最大回撤10/30bp'],rows)
    di=pd.DataFrame(diag);gro=pd.DataFrame(grossdiag);cover=pd.read_csv(OUT/'feature_source_changes.csv');co=cover.groupby(['seed','phase']).agg(requested=('requested_stocks','sum'),changed=('changed_windows','sum'),max_vector_change=('max_vector_change','max')).reset_index()
    co.to_csv(OUT/'feature_source_summary.csv',index=False)
    diagrows=[]
    for arm in ['old_plain','e3_plain_f10','best_plain_f10','old_fixed5','e3_fixed5_f10','best_fixed5_f10']:
        q=di.loc[di.arm.eq(arm)].set_index('phase');r=gro.set_index('arm').loc[arm]
        diagrows.append([display(arm),f"{q.at['validation','bce']:.6f}",f"{q.at['test','accuracy']:.2%}",f"{q.at['test','balanced_accuracy']:.2%}",f"{q.at['test','bce']:.6f}",f"{r.accuracy:.2%}",f"{r.gross_regret_bp:.2f}",int(q.at['test','predicted_long'])])
    selected_table=table(['增强','验证选冻结方案','集成验证BCE'],[[x['suffix'],x['arm'],f"{x['ensemble_bce']:.6f}"] for x in lock['freeze_choices']])
    DOC.write_text(f'''# B0原生best检查点：固定旧40/10任务的实训与逐项诊断

2026-11-19，Asia/Shanghai，complete。[冻结协议](research_20261119_bce_b0_best_protocol.md)；父批[固定旧名单三向量](research_20261119_bce_old_roster_vectors_results.md)、[冻结步数](research_20261119_bce_freeze_steps_results.md)。本轮30个BCE头各100步，CPU完成新增编码和训练，未重训股票主干。全部已读历史复检。证据见[轻量归档](evidence/20261119_bce_b0_best/README.md)。

## 比较边界与实际操作

固定旧40/10输入股票身份/标签、原473训练/123验证/295测试、236增强启用周/708固定变体、旧20/10实际下单、训练预算/初始化/损失及执行口径。主干唯一主干预是B0三seed第3/3/3轮改为各自原生验证best1/6/3；三份best与对应epoch张量逐项完全相同，原training_metrics最大验证多头净Sharpe与metrics.best_epoch一致。股票排序多头验证最优，不等于多空择时或表示质量最优。

主比较固定第10步冻结；各自坐标只在473基础训练周拟合mu/std+1e-6，因此更换表示导致统计数值变化是原固定训练算法的一部分。每seed单独头、最后等权平均logit，未平均跨模型原始向量。只换42、只换43的信号由配对已锁成员组合，seed44因best=e3在全部训练/验证/测试输出逐位相同。

补充冻结时点只在相同best输入上改变10/30/不冻结，6组前10及前30步对应路径完全相同；原冻结头和free实际参数变化均验收。各增强按集成验证BCE锁冻结方案，所有预注册方案均报告，不按测试反选。

{selected_table}

## 主对照完整绩效

每格10bp+8.35%年券费 / 30bp+12%年券费。预算1，旧Top20/Bottom10固定排序与入场可成交规则、现金、退出代理、净值漂移一致。净复合年化按2021-01-11至2026-10-19实际日历；最差10%为最差30周净收益均值；两腿是总NAV净bp/周贡献，不可相加当复合年化。Sharpe周频sqrt(52)，回撤周估值。

### 宽档：多40%或70%

{metric_table('wide')}

### 全多或全空：多0%或100%

{metric_table('full')}

## 检查点成员、输入源和冻结时点的独立补充

以下每组只与其登记父对照配对；例如best f30/free对best f10，旧特征对best Q1 f10，部分成员替换对e3。不能把不同补充彼此直接称为单因素比较。

### 宽档

{metric_table('wide',True)}

### 全切换

{metric_table('full',True)}

## 预测准确率、目标错位与收益幅度

原监督是1[旧Top40平均simple收益+旧Bottom10平均simple收益+.08/52>0]。测试监督方向按同名单及既有缺收益记平构建，未用这些标签筛推理日期。下表40/10准确率是在该目标上，实际20/10只比较同执行账本恢复的满额多腿和空腿毛收益L>S，后者不含交易/券费；二者不能混称扣费后可实现oracle。

{table(['信号','验证BCE','测试40/10准确率','测试40/10平衡准确率','测试BCE','实际20/10毛胜负准确率','相对毛oracle遗憾bp/周','判多周数'],diagrows)}

完整三阶段训练/验证/测试BCE、Brier、混淆计数、logit均值/尺度及常数.5和仅训练先验见prediction_diagnostics.csv；各成员见member_diagnostics.csv，选步见selected_steps.csv。实际20/10错向幅度见actual20_gross_diagnostics.csv。命中率将小幅和大幅错向同等计数，收益需要另看错向幅度、仓位和换向成本；训练监督的40/10与下单20/10亦有任务迁移。

## 特征源实际改变和金额归因

旧源与Q1共享同一B0冻结预处理、同一旧事前mask、股票名单与标签，只改变冻结网络收到的原始特征源。该控制不重训股票主干，因此只识别推理/小头样本构建的输入源效应，不能证明主干训练样本或标签历史没有影响。

{table(list(co.columns),co.round(9).itertuples(index=False,name=None))}

方向差dw带来全切换毛差dw*(L−S)，宽档为.3*dw*(L−S)；净差再减交易费与券费差。逐周和均值恒等式验收；均值拆成平均暴露项mean(dw)*mean(L−S)及协方差项，不把会计分解误称经济因果。完整见attribution.csv、directions.csv（换向日期）及paired.csv（8周块5000次均值/尾部区间）。区间和大量比较均属探索，不以测试排名重新改规则。

## 验收、已有因素与未识别项

30头3000步、64指标18880周、旧/e3共4720条父周收益复现；e3六头参数及统计复现，seed44 best=e3输出完全一致。54个原比较头不在此重复宣称新训。本次完整580股forward与裁剪恒等路径36项逐seed/阶段核对；冻结为零的关系门已实际断言。新增CPU仅编码需要的基础/增强股票，未声称使用GPU或训练主干。所有输入/源码哈希、初始BCE闭式与解析梯度、原验证最优步、各选中验证BCE、三头平均logit及64指标独立复算通过。

前批分别控制过编码器数量、名单/标签、输入归一化、冻结时点与旧信号重训；本批追加检查点及冻结模型输入源。可识别的是这些条件下的效果，不是“新架构必然不如旧架构”或某个因素解释全部差距。要严格拆主干763/779训练周、主干监督或训练实现版本，需要固定共同数据、初始化和源码重新训练主干；本轮未执行这些大任务，不把已有元信息差异当因果结论。

执行PYTHONPATH=scripts;src下python -u outputs/bce_b0_best_v1/run.py及python outputs/bce_b0_best_v1/report.py。初次编码日志状态参数重名曾在任何头训练前报错，修复后整批重跑；不更改协议或样本。已锁后不覆盖原产物。本批complete、无待跑本批任务；其它会话GPU队列未改动。

继承旧473标签完整性筛选、样本内主干表征/名单、旧验证选择与小头训练交叠、固定幸存池、数据历史修订/行业回溯、缺收益记平和退出估值、真实券源未知等限制。测试已反复读，不作为新独立样本外结论。原始数据、向量、主干、30头和完整账本仅本机，未核异地备份；轻量证据不等于全量持久备份。未自动Git提交或部署。
''',encoding='utf8')
    hashes=json.loads((OUT/'inputs.json').read_text())
    for p,h in hashes.items():assert sha(p)==h,('changed input',p)
    for p in [Path(__file__),ROOT/'scripts/replay_kit_risk.py']:
        hashes[str(p.resolve())]=sha(p)
    write(OUT/'inputs.json',hashes)
    verify=json.loads((OUT/'verification.json').read_text());verify.update(status='complete',report_complete=True,independent_metric_rows=64,paired_comparisons=len(pairs),full_forward_checks=len(pd.read_csv(OUT/'full_forward_checks.csv')),input_hashes_verified=len(hashes),validation_checkpoints_verified=30,seed44_exact=True)
    write(OUT/'verification.json',verify);write(OUT/'state.json',verify)
    names=['run.py','report.py','run.log','run.err.log','model_inventory.json','dates.json','feature_source_changes.csv','feature_source_summary.csv','full_forward_checks.csv','identity.csv','loss_checks.csv','e3_parameter_reproduction.csv','prefix_checks.csv','source_training_checks.csv','parameter_snapshots.csv','training_loss.csv','validation_lock.json','validation_ensemble.csv','selected_steps.csv','signals.csv','member_logits.csv','prediction_diagnostics.csv','member_diagnostics.csv','actual20_gross_diagnostics.csv','directions.csv','paired.csv','attribution.csv','yearly.csv','metrics.csv','inputs.json','verification.json','state.json']
    EVIDENCE.mkdir(parents=True,exist_ok=True);manifest=[]
    sources=[(OUT/name,name) for name in names]+[(ROOT/'outputs/bce_freeze_steps_v1/run.py','fit_source.py'),(ROOT/'docs/research_20261119_bce_b0_best_protocol.md','protocol.md')]
    for src,name in sources:
        dest=EVIDENCE/name;shutil.copyfile(src,dest);h=sha(src);assert sha(dest)==h
        manifest.append(dict(source=str(src.relative_to(ROOT)),copy=name,bytes=src.stat().st_size,sha256=h))
    write(EVIDENCE/'manifest.json',manifest)
    (EVIDENCE/'README.md').write_text(f'''# B0原生best固定旧任务：轻量证据

2026-11-19，Asia/Shanghai。[结果](../../research_20261119_bce_b0_best_results.md)。{len(manifest)}份源码、协议、30头训练/验证/梯度/身份、64指标、诊断与配对证据逐字节复制；[manifest](manifest.json) SHA256 `{sha(EVIDENCE/'manifest.json')}`。原数据、向量、真实主干及BCE权重和完整账本仅本机、未归档至异地；不是独立全量重训包。未自动Git提交。
''',encoding='utf8')
    print(metric_table('full'));print(table(['signal','validation BCE','test accuracy40','balanced','test BCE','accuracy20gross','regretbp','longweeks'],diagrows));print('Published',DOC,'copies',len(manifest),'manifest',sha(EVIDENCE/'manifest.json'),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        write(OUT/'state.json',dict(status='failed',phase='report',error=repr(exc)))
        raise
