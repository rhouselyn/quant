"""Check the frozen three-encoder comparison and publish the complete matrix."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from attention_followup import LegPool
from replay_kit_risk import paired

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/bce_three_encoder_v1'
PARENT=ROOT/'outputs/bce_native40_transfer_v1'
DOC=ROOT/'docs/research_20261119_bce_three_encoder_results.md'
EVIDENCE=ROOT/'docs/evidence/20261119_bce_three_encoder'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def table(frame):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows.extend('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(rows)

def main():
    torch.set_num_threads(4)
    m=pd.read_csv(OUT/'metrics.csv');w=pd.read_parquet(OUT/'weekly.parquet')
    s=pd.read_csv(OUT/'signals.csv',parse_dates=['date']).set_index('date')
    members=pd.read_csv(OUT/'member_logits.csv',parse_dates=['date']).set_index('date')
    trace=pd.read_csv(OUT/'training_loss.csv');identity=pd.read_csv(OUT/'batch_identity.csv')
    lock=json.loads((OUT/'validation_lock.json').read_text(encoding='utf8'))
    inventory=json.loads((OUT/'model_inventory.json').read_text(encoding='utf8'))
    verify=json.loads((OUT/'verification.json').read_text(encoding='utf8'))
    full=pd.read_csv(OUT/'full_forward_checks.csv')
    assert len(m)==88 and len(w)==25960 and len(trace)==1200 and len(full)==12
    assert len(s)==len(members)==295 and len(lock['selections'])==12
    assert not w.duplicated(['book','arm','rule','fee','date']).any()
    assert w.groupby(['book','arm','rule','fee']).size().eq(295).all()
    assert full.max_vector_error.max()<2e-6
    assert pd.read_csv(OUT/'seed42_reproduction.csv').max_parameter_error.eq(0).all()
    np.testing.assert_allclose(w.net,w.long_net+w.short_net,atol=1e-12,rtol=0)
    years=(pd.Timestamp('2026-10-19')-pd.Timestamp('2021-01-11')).days/365.2425
    for row in m.itertuples():
        g=w.loc[w.book.eq(row.book)&w.arm.eq(row.arm)&w.rule.eq(row.rule)&w.fee.eq(row.fee)].sort_values('date')
        r=g.net.to_numpy();nav=np.r_[1.,np.cumprod(1+r)]
        actual=[nav[-1]**(1/years)-1,np.max(1-nav/np.maximum.accumulate(nav)),r.mean()/r.std(ddof=1)*np.sqrt(52),
                np.sort(r)[:30].mean(),g.long_net.mean()*1e4,g.short_net.mean()*1e4]
        np.testing.assert_allclose(actual,[row.annual_calendar,row.max_drawdown,row.sharpe,row.worst10_avg,row.long_bp,row.short_bp],atol=1e-12,rtol=0)
    comparison=m.loc[m.arm.str.contains('_three_')].merge(m.loc[m.arm.str.startswith('old_')],on=['book','rule','fee'],suffixes=('_new','_old'))
    comparison=comparison.loc[comparison.arm_new.str.split('_').str[-1].eq(comparison.arm_old.str.split('_').str[-1])]
    assert len(comparison)==32 and comparison.annual_calendar_new.lt(comparison.annual_calendar_old).all()
    diagnostics=[];correlations=[];validation=[];selections=[];directions=[]
    for family in ('B0','P6'):
        models=[r for r in inventory['models'] if r['family']==family]
        assert len(models)==3 and len({r['encoder_sha256'] for r in models})==3
        for suffix in ('plain','fixed5'):
            arm=family+'_three_'+suffix;one=family+'_one_'+suffix;old='old_'+suffix
            ident=identity.loc[identity.family.eq(family)&identity.suffix.eq(suffix)]
            assert len(ident)==3 and ident.inputs.nunique()==3
            assert ident.labels.nunique()==ident.weights.nunique()==1
            cols=[arm+'_s'+str(seed) for seed in (42,43,44)]
            np.testing.assert_allclose(s[arm],members[cols].mean(axis=1),atol=1e-7,rtol=0)
            corr=members[cols].corr()
            for i,a in enumerate(cols):
                seed=(42,43,44)[i]
                r=next(r for r in lock['selections'] if r['arm']==arm and r['encoder_seed']==seed)
                selections.append(dict(arm=arm,seed=seed,selected_step=r['step'],after_projection_freeze=r['step']>10,val_bce=r['val_bce']))
                diagnostics.append(dict(arm=arm,seed=seed,mean_logit=members[a].mean(),std_logit=members[a].std(),long_weeks=int(members[a].gt(0).sum())))
                for j in range(i+1,3):correlations.append(dict(arm=arm,seed_a=seed,seed_b=(42,43,44)[j],correlation=corr.loc[a,cols[j]]))
            three_val=[];one_val=[]
            for seed in (42,43,44):
                for kind,cp_path,data_path,values in (
                    ('three',OUT/f'{arm}_s{seed}.pt',OUT/f'{family}_s{seed}_data.npz',three_val),
                    ('one',PARENT/f'{family}_{suffix}_{seed}.pt',OUT/f'{family}_s42_data.npz',one_val)):
                    cp=torch.load(cp_path,weights_only=True);head=LegPool(False).eval();head.load_state_dict(cp['state'])
                    with np.load(data_path) as a:
                        vx=torch.from_numpy(a['vx']);vy=torch.tensor(a['vy']>0,dtype=torch.float32)
                    with torch.no_grad():values.append(head((vx-cp['mu'])/cp['sd']).flatten())
            validation.append(dict(arm=arm,three_ensemble_bce=float(F.binary_cross_entropy_with_logits(torch.stack(three_val).mean(0),vy)),
                                   one_ensemble_bce=float(F.binary_cross_entropy_with_logits(torch.stack(one_val).mean(0),vy))))
            for control in (one,old,family+'_three_plain'):
                if control==arm:continue
                flip=s[arm].gt(0).ne(s[control].gt(0))
                directions.append(dict(arm=arm,control=control,long_weeks=int(s[arm].gt(0).sum()),changed_weeks=int(flip.sum()),
                    changed_dates=';'.join(d.strftime('%Y-%m-%d') for d in s.index[flip])))
    for name,rows in [('members_summary',diagnostics),('member_correlations',correlations),('validation_ensemble',validation),
                      ('selected_steps',selections),('directions',directions)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    relationships=(('B0','old'),('B0','B0'),('P6','old'),('P6','P6'))
    pairs=[];attribution=[]
    for family,book in relationships:
        for suffix in ('plain','fixed5'):
            arm=family+'_three_'+suffix
            for rule in ('wide','full'):
                for fee in (10,30):
                    q=w.loc[w.book.eq(book)&w.rule.eq(rule)&w.fee.eq(fee)].pivot(index='date',columns='arm',values='net')
                    controls=[family+'_one_'+suffix,'old_'+suffix]
                    if suffix=='fixed5':controls.append(family+'_three_plain')
                    for control in controls:pairs.append(dict(book=book,arm=arm,control=control,rule=rule,fee=fee,**paired(q[arm].to_numpy(),q[control].to_numpy())))
        base=w.loc[w.book.eq(book)&w.fee.eq(10)&w.arm.eq('old_plain')]
        ff=base.loc[base.rule.eq('full')].set_index('date').reindex(s.index)
        ww=base.loc[base.rule.eq('wide')].set_index('date').reindex(s.index)
        direction=s.old_plain.gt(0)
        long=np.where(direction,ff.gross_ret,(ww.gross_ret-.6*ff.gross_ret)/.4)
        short=np.where(direction,(ww.gross_ret-.7*ff.gross_ret)/.3,ff.gross_ret)
        gap=long-short
        for suffix in ('plain','fixed5'):
            arm=family+'_three_'+suffix
            for control in (family+'_one_'+suffix,'old_'+suffix):
                dw=s[arm].gt(0).astype(float)-s[control].gt(0).astype(float)
                gross=float((dw*gap).mean());exposure=float(dw.mean()*gap.mean())
                for fee in (10,30):
                    a=w.loc[w.book.eq(book)&w.arm.eq(arm)&w.rule.eq('full')&w.fee.eq(fee)].set_index('date').reindex(s.index)
                    b=w.loc[w.book.eq(book)&w.arm.eq(control)&w.rule.eq('full')&w.fee.eq(fee)].set_index('date').reindex(s.index)
                    np.testing.assert_allclose(a.gross_ret-b.gross_ret,dw*gap,atol=1e-12,rtol=0)
                    tc=float((a.trading_cost-b.trading_cost).mean());bc=float((a.borrow_cost-b.borrow_cost).mean());net=float((a.net-b.net).mean())
                    np.testing.assert_allclose(net,gross-tc-bc,atol=1e-12,rtol=0)
                    attribution.append(dict(book=book,arm=arm,control=control,fee=fee,net_delta_bp=net*1e4,gross_delta_bp=gross*1e4,
                        exposure_bp=exposure*1e4,covariance_bp=(gross-exposure)*1e4,extra_trading_bp=tc*1e4,extra_borrow_bp=bc*1e4))
    pairs=pd.DataFrame(pairs);assert len(pairs)==80;pairs.to_csv(OUT/'paired.csv',index=False)
    attribution=pd.DataFrame(attribution);attribution.to_csv(OUT/'attribution.csv',index=False)
    yearly=[]
    for key,g in w.assign(year=w.date.dt.year).groupby(['book','arm','rule','fee','year']):
        g=g.sort_values('date');r=g.net.to_numpy();nav=np.r_[1.,np.cumprod(1+r)]
        yearly.append(dict(zip(['book','arm','rule','fee','year'],key))|dict(weeks=len(g),total_return=float(nav[-1]-1),
            max_drawdown=float(np.max(1-nav/np.maximum.accumulate(nav))),sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(52)),
            long_bp=float(g.long_net.mean()*1e4),short_bp=float(g.short_net.mean()*1e4)))
    pd.DataFrame(yearly).to_csv(OUT/'yearly.csv',index=False)
    def metric_table(book,rule):
        rows=[]
        arms=['old_plain','old_fixed5']
        for family in (('B0','P6') if book=='old' else (book,)):
            arms.extend(family+'_'+count+'_'+suffix for count in ('one','three') for suffix in ('plain','fixed5'))
        for arm in arms:
            q=m.loc[m.book.eq(book)&m.rule.eq(rule)&m.arm.eq(arm)].set_index('fee')
            def values(col,scale=1,digits=2):return ' / '.join(f'{q.at[f,col]*scale:.{digits}f}'+('%' if scale==100 else '') for f in (10,30))
            rows.append([arm,values('annual_calendar',100),values('max_drawdown',100),values('sharpe',digits=3),values('worst10_avg',100),values('long_bp'),values('short_bp')])
        return table(pd.DataFrame(rows,columns=['信号','净复合年化10/30bp','最大回撤10/30bp','夏普10/30bp','最差10%周均10/30bp','多腿bp/周10/30bp','空腿bp/周10/30bp']))
    results='\n\n'.join(f"## 实际{book}20/10：{'全多/全空' if rule=='full' else '40%/70%宽档'}\n\n{metric_table(book,rule)}" for book in ('old','B0','P6') for rule in ('full','wide'))
    dirs=pd.DataFrame(directions).drop(columns='changed_dates')
    ap=attribution.loc[attribution.fee.eq(10)&attribution.control.str.startswith('old_')]
    pp=pairs.loc[pairs.rule.eq('full')&pairs.control.str.startswith('old_')]
    pairrows=[[r.book,r.arm,r.fee,f'{r.mean_delta_bps:.2f}',f'[{r.mean_lo_bps:.2f}, {r.mean_hi_bps:.2f}]'] for r in pp.itertuples()]
    DOC.write_text(f'''# B0/P6三个真实编码器的40/10信号：完整控制对照

2026-11-19，Asia/Shanghai，complete。[执行协议](research_20261119_bce_three_encoder_protocol.md)；父批[单编码器三头](research_20261119_bce_native40_transfer_results.md)、[旧重训及编码器数量审计](research_20261119_bce_old_retrain_audit_results.md)。轻量证据见[目录](evidence/20261119_bce_three_encoder/README.md)。本次12头完整新训，六个股票主干复用已有真实独立训练权重。

## 结论

补齐三个真实编码器没有消除旧新差距。四个来源/实际账本关系、两增强、两规则、两费用共32个对应年化，全部低于同增强旧三编码器信号；不能据此声称每个尾部指标都由旧信号支配。B0无增强指导自身20/10全切换相对单编码器由20.71%/6.64%升为23.72%/9.34%，固定5%反而由27.80%/12.96%降为22.80%/8.75%。P6无增强由21.33%/2.16%降为4.11%/−11.68%，固定5%同为4.11%/−11.68%。旧三编码器指导自身旧20/10全切换仍为50.01%/30.91%和51.41%/31.97%。

因此上一批“旧编码器多样性有贡献”仍成立，但“只要新模型也用三个编码器就能补齐差距”不成立。编码器数量相同不保证成员信息和误差结构相同，更不保证阈值后的交易更好。旧与新家族的40/10名单和标签任务仍不同，不能把剩余差距唯一宣布为股票架构本身优劣。

## 实际做了什么、控制了什么

用户本轮明确选择固定父批名单和标签，只换编码器向量。每家族40/10基础名单、Top50/Bottom20随机池、seed61118的236启用周及708固定变体完全沿父批seed42冻结版本；实际旧/B0/P6的20/10下单名单不变。没有让新增43/44重新选股或重算各自标签。

同家族旧方案：三个MLP初始化42/43/44全读编码器42；新方案：初始化42/43/44分别读真实编码器42/43/44。向量128维，各头分别按自己的473基础周拟合mu/sd，0/5%共用该统计；编码器坐标不同，禁止直接平均原始向量。各家族三份输入指纹不同，而同增强的监督和权重指纹完全相同。种子42重训四头与父批参数逐位相同，证明未意外更换其余数值路径。

六个主干实际权重身份与路径见model_inventory.json：B0为exploratory_F0/kit_transfer_seed_v1，固定第3轮；P6为group_fox_v1/m6_history_v1，固定第8轮。m6_history目录中的模型配置是原P6，没有启用M6分支。每家族779周主干训练数据、mask、X/y指纹、预处理和8轮预算一致；实际配置仅seed和source_sha256不同，编码器自身SHA256两两不同。source_sha256并非相同，不能声称归档源码逐字一致；沿既有兼容审计，并新增43/44三阶段共12次完整forward数值检查，向量最大误差{full.max_vector_error.max():.3g}。42为已验收只读历史档案，43/44有committed状态，不宣称42支持精确续训。

本次没有重新训练股票主干，因为已经存在符合门槛的独立8轮训练。新增四编码器所需股票向量实际用GPU导出；随后12个配比头在CPU4线程完整训练100步，共1200条训练/验证记录。不是只改MLP种子，也不是只加载现成BCE预测。

## 数据、损失与锁定流程

原473训练周2001-11-23至2017-12-15；原123验证监督周，位于133周推理范围2018-04-06至2020-12-18；完整295测试决策周2021-01-08至2026-10-09。首入场2021-01-11、末退出2026-10-19。旧覆盖门槛和缺收益记平按此前用户授权统一处理，未删任何指定周。B0/P6基础训练分别55/69周含部分记平，三个编码器读相同监督。训练集名单与向量仍为主干样本内，未冒充滚动样本外。

40支多股和10支空股的128维向量加side成129维token，129→32投影；各腿取均值，LayerNorm、残差32→64→32 FFN后拼接，64→1输出logit。这是分腿均值MLP，没有新增注意力。原监督仍为y=1[mean(simple_long)+mean(simple_short)+0.08/52>0]，预测的是相对多空腿优劣，不是单只股票上涨概率。

无增强每周权重1；固定5%仅236启用周，原名单.95，三个固定随机变体合计.05，其余周原名单1。加权BCE=sum(weight*BCEWithLogits)/473。Adam lr=.001、weight_decay=.01，100全批更新，第11步开始冻结投影；按原123周最低BCE独立选步，不强制选冻结后。三个头先各锁检查点，再平均原始logit，阈值0。无新阈值、温度或种子搜索。

{table(pd.DataFrame(selections).round(6))}

B0两个增强的三头均选5/4/6步，全部在投影冻结前；P6均选37/6/3步，只有seed42经过冻结阶段。所有头实际都跑满100步，不能因所选检查点早就声称未训练。新增seed43/44可能改变验证选步，所以这是同算法/预算下的向量来源对照，不是固定相同梯度步的参数因果分解。

{table(pd.DataFrame(validation).round(6))}

上表只复算已锁检查点在原123周的集成BCE，不用它新增筛选规则。成员P6 seed43最低验证BCE约.6984，高于恒定.5的log(2)=.6931，提示该成员在此任务上的验证质量弱；不据已读测试删成员。

## 交易与指标单位

同一信号分别用于自身和旧20/10，账本沿原锁执行、成交限制、退出代理和净值漂移。wide为多头40%或70%，full为0或100%；同NAV总预算1，受阻开仓可留现金。每格顺序10bp+8.35%年券费 / 30bp+12%年券费。复合年化按实际日历跨度，周频Sharpe按sqrt(52)，最大回撤为周估值；最差10%周为最差30周净收益均值。多/空腿是对总NAV的净bp/周贡献，合计净周均，不可相加成复合年化。

old_plain/fixed5=旧三编码器；B0/P6_one=单编码器三MLP初始化；B0/P6_three=本次三个真实编码器。plain无增强，fixed5固定5%。以下六表覆盖全部88行。

{results}

## 换向、配对与差距解释

{table(dirs)}

B0三编码器判多244/246周，占82.71%/83.39%，5%只改变两周；P6判多271周，占91.86%，0/5%的logit幅度虽不同，295周判向完全相同，因此两费用两规则收益相同。成员及集成logit、相关性和全部换向日期分别存member_logits.csv、member_correlations.csv、directions.csv。P6三成员判多248/235/279周（无增强），并未带来互相抵消的有效误差；低相关性本身也不证明有效。

全切换新三编码器减同增强旧三编码器，净周均差与8周移动块5000次95%区间（bp/周）：

{table(pd.DataFrame(pairrows,columns=['实际账本','新信号','费档','净均差bp/周','95%区间']))}

共80项配对同时保存三对单、三对旧三及5%对0的两规则两费用，尾部差也在paired.csv；未作多重比较校正，跨零不代表严格等效。逐年收益、年内回撤、Sharpe、两腿见yearly.csv，2026是截至最后持仓退出的部分年度。

同实际账本恢复每周满额多腿L与空腿S，令dw为新减旧判多指示差，则毛收益差=dw*(L−S)。其均值=mean(dw)*mean(L−S)+cov(dw,L−S)，净差再扣交易费差与券费差。逐周和均值恒等式全部断言通过。以下10bp全切换新三减旧三，单位bp/周：

{table(ap.round(3))}

这是金额记账，不识别唯一经济原因。新信号更偏多、同期方向选择、成员logit尺度及验证早选都可能共同影响；不能把高判多率单独称为退步原因。与本家族单编码器相比只换向量来源；与旧三对照仍有名单、标签任务和表示家族差异，编码器数量已控制。

## 验收、复现与边界

六真实编码器身份/数据审计、12完整forward、四seed42头参数逐位复现通过；1200条完整轨迹和12个检查点全部锁定后才生成新测试预测。88指标、25960周账本唯一键、每组295周、两腿闭合、均值logit恒等式均通过；旧三及单编码器冻结对照共16520条周收益与父批逐周误差≤1e−12。所有原输入哈希再次验收，未覆盖父批。

命令（项目根目录，PYTHONPATH=scripts;src）：python outputs/bce_three_encoder_v1/audit_models.py → python -u outputs/bce_three_encoder_v1/prepare.py → python -u outputs/bce_three_encoder_v1/train_replay.py → python outputs/bce_three_encoder_v1/report.py。独占GPU锁仅用于准备向量，已正常释放；本批complete，无待跑训练，不修改其它队列，不部署，不自动Git提交。

本批仍是已读历史探索复检，继承固定473选周、主干样本内名单、固定幸存池、修订数据、缺收益记平/退出估值及真实券源未知限制。没有重新训练股票主干或另扫新的学习率、预算、冻结时点与融合权重，不能声称原算法已经对B0/P6全局最优。原始行情、六主干、12新头、输入数组和完整周账本仅本机，未核实异地备份；轻量证据不是脱离这些资产的独立全量重训包。
''',encoding='utf8')
    hashes=json.loads((OUT/'inputs.json').read_text(encoding='utf8'))
    for path,digest in hashes.items():
        if Path(path).resolve()!=Path(__file__).resolve():assert sha(path)==digest,('changed frozen input',path)
    for path in (Path(__file__),ROOT/'scripts/replay_kit_risk.py'):hashes[str(path.resolve())]=sha(path)
    (OUT/'inputs.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    verify.update(report_complete=True,training_trace_rows=len(trace),paired_comparisons=len(pairs),full_forward_checks=len(full),
                  max_full_forward_error=float(full.max_vector_error.max()),input_hashes_verified=len(hashes),logit_mean_verified=True,
                  metric_rows_independently_recomputed=len(m),matched_old_three_comparisons=len(comparison))
    for name in ('verification.json','state.json'):(OUT/name).write_text(json.dumps(verify,indent=2),encoding='utf8')
    files=['audit_models.py','prepare.py','train_replay.py','report.py','prepare.log','train.log','model_inventory.json','batch_identity.csv',
        'full_forward_checks.csv','missing_returns.csv','seed42_reproduction.csv','training_loss.csv','validation_lock.json','member_logits.csv',
        'signals.csv','metrics.csv','members_summary.csv','member_correlations.csv','validation_ensemble.csv','selected_steps.csv','directions.csv',
        'paired.csv','attribution.csv','yearly.csv','inputs.json','verification.json','state.json']
    EVIDENCE.mkdir(parents=True,exist_ok=True);manifest=[]
    for name in files:
        src=OUT/name;dest=EVIDENCE/name;shutil.copyfile(src,dest);digest=sha(src);assert sha(dest)==digest
        manifest.append(dict(source=str(src.relative_to(ROOT)),copy=name,bytes=src.stat().st_size,sha256=digest))
    (EVIDENCE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');digest=sha(EVIDENCE/'manifest.json')
    (EVIDENCE/'README.md').write_text(f'''# 三个真实编码器的B0/P6配比头证据

2026-11-19，Asia/Shanghai。[结果](../../research_20261119_bce_three_encoder_results.md)；[协议](../../research_20261119_bce_three_encoder_protocol.md)。{len(files)}份逐字节核对复制件，来源、大小与SHA256见[manifest](manifest.json)，manifest SHA256为`{digest}`。包含六编码器身份、12头完整训练和锁、88指标、80配对及年度/归因/换向记录。

全量行情、输入、真实主干、配比头权重和周账本仍仅本机，未核实异地备份；非独立全量重训包。源码应按原工作区路径复现，未自动Git提交或推送。
''',encoding='utf8')
    print(pd.DataFrame(validation).to_string(index=False))
    print('Published',DOC,'copies',len(files),'manifest',digest)

if __name__=='__main__':main()
