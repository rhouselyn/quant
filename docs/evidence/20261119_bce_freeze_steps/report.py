"""Audit freeze interventions, validation choices and all fixed-book metrics."""
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
OUT=ROOT/'outputs/bce_freeze_steps_v1'
PARENT=ROOT/'outputs/bce_old_roster_vectors_v1'
DOC=ROOT/'docs/research_20261119_bce_freeze_steps_results.md'
EVIDENCE=ROOT/'docs/evidence/20261119_bce_freeze_steps'
MODES=('f10','f30','free')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def table(frame):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows.extend('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(rows)

def main():
    sys.stdout.reconfigure(encoding='utf8');torch.set_num_threads(4)
    m=pd.read_csv(OUT/'metrics.csv');w=pd.read_parquet(OUT/'weekly.parquet');t=pd.read_csv(OUT/'training_loss.csv')
    s=pd.read_csv(OUT/'signals.csv',parse_dates=['date']).set_index('date');members=pd.read_csv(OUT/'member_logits.csv',parse_dates=['date']).set_index('date')
    sn=pd.read_csv(OUT/'parameter_snapshots.csv');prefix=pd.read_csv(OUT/'prefix_checks.csv');loss=pd.read_csv(OUT/'loss_checks.csv')
    lock=json.loads((OUT/'validation_lock.json').read_text(encoding='utf8'));choice=json.loads((OUT/'selection_lock.json').read_text(encoding='utf8'))
    assert len(m)==72 and len(w)==21240 and len(t)==5400 and len(lock['selections'])==54 and len(choice['choices'])==6
    assert len(sn)==216 and len(prefix)==18 and prefix.first10_exact.all() and prefix.f30_free_first30_exact.all() and prefix.free_projection_changed_after30.all()
    assert not w.duplicated(['arm','rule','fee','date']).any() and w.groupby(['arm','rule','fee']).size().eq(295).all()
    np.testing.assert_allclose(w.net,w.long_net+w.short_net,atol=1e-12,rtol=0)
    np.testing.assert_allclose(w.net,w.gross_ret-w.trading_cost-w.borrow_cost,atol=1e-12,rtol=0)
    assert loss.closed_form_error.max()<2e-6 and loss.gradient_error.max()<1e-8 and loss.frozen_projection_max_error.eq(0).all()
    for row in loss.itertuples():np.testing.assert_allclose(row.initial_loss,t.loc[t.arm.eq(row.arm)&t.head_seed.eq(row.seed)&t.step.eq(1),'train_bce'].item(),atol=2e-6,rtol=0)
    years=(pd.Timestamp('2026-10-19')-pd.Timestamp('2021-01-11')).days/365.2425
    for row in m.itertuples():
        a=w.loc[w.arm.eq(row.arm)&w.rule.eq(row.rule)&w.fee.eq(row.fee)].sort_values('date');r=a.net.to_numpy();nav=np.r_[1.,np.cumprod(1+r)]
        vals=[nav[-1]**(1/years)-1,np.max(1-nav/np.maximum.accumulate(nav)),r.mean()/r.std(ddof=1)*np.sqrt(52),np.sort(r)[:30].mean(),a.long_net.mean()*1e4,a.short_net.mean()*1e4]
        np.testing.assert_allclose(vals,[row.annual_calendar,row.max_drawdown,row.sharpe,row.worst10_avg,row.long_bp,row.short_bp],atol=1e-12,rtol=0)
    checkpoints=[];segments=[];vparts={}
    for row in lock['selections']:
        cp=torch.load(row['path'],weights_only=True);h=LegPool(False).eval();h.load_state_dict(cp['state'])
        with np.load(PARENT/f"{row['family']}_s{row['encoder_seed']}_data.npz") as a:
            flat=a['base'].reshape(-1,129);np.testing.assert_array_equal(cp['mu'],flat.mean(0));np.testing.assert_array_equal(cp['sd'],flat.std(0)+1e-6)
            target=torch.tensor(a['vy']>0,dtype=torch.float32)
            with torch.no_grad():p=h((torch.from_numpy(a['vx'])-cp['mu'])/cp['sd']).flatten();value=float(F.binary_cross_entropy_with_logits(p,target))
        np.testing.assert_allclose(value,row['val_bce'],atol=2e-6,rtol=0);vparts.setdefault(row['arm'],[]).append(p)
        checkpoints.append(dict(arm=row['arm'],seed=row['encoder_seed'],step=row['step'],val_error=abs(value-row['val_bce'])))
        g=t.loc[t.arm.eq(row['arm'])&t.head_seed.eq(row['encoder_seed'])]
        segments.append(dict(arm=row['arm'],seed=row['encoder_seed'],selected_step=row['step'],val_min_1_10=g.loc[g.step.le(10),'val_bce'].min(),
            val_min_11_30=g.loc[g.step.between(11,30),'val_bce'].min(),val_min_31_100=g.loc[g.step.gt(30),'val_bce'].min(),last_train_bce=g.iloc[-1].train_bce,last_val_bce=g.iloc[-1].val_bce))
    validation=pd.read_csv(OUT/'validation_ensemble.csv').set_index('arm')
    for arm,ps in vparts.items():
        value=float(F.binary_cross_entropy_with_logits(torch.stack(ps).mean(0),target));np.testing.assert_allclose(value,validation.at[arm,'ensemble_bce'],atol=2e-6,rtol=0)
        np.testing.assert_allclose(s[arm],members[[arm+'_s'+str(seed) for seed in (42,43,44)]].mean(1),atol=1e-7,rtol=0)
    for r in choice['choices']:
        candidates=validation.loc[validation.family.eq(r['family'])&validation.suffix.eq(r['suffix'])]
        best=candidates.ensemble_bce.min();expected=next(mode for mode in MODES if candidates.loc[candidates['mode'].eq(mode),'ensemble_bce'].item()<=best+1e-12)
        assert r['selected_mode']==expected
    pd.DataFrame(checkpoints).to_csv(OUT/'checkpoint_checks.csv',index=False);pd.DataFrame(segments).to_csv(OUT/'training_segments.csv',index=False)
    pairs=[];selected_pairs=[];directions=[]
    for rule in ('wide','full'):
        for fee in (10,30):
            q=w.loc[w.rule.eq(rule)&w.fee.eq(fee)].pivot(index='date',columns='arm',values='net')
            for family in ('old','B0','P6'):
                for suffix in ('plain','fixed5'):
                    control=f'{family}_{suffix}_f10'
                    for mode in ('f30','free'):
                        arm=f'{family}_{suffix}_{mode}';pairs.append(dict(kind='freeze',arm=arm,control=control,rule=rule,fee=fee,**paired(q[arm].to_numpy(),q[control].to_numpy())))
            for family in ('B0','P6'):
                for suffix in ('plain','fixed5'):
                    for mode in MODES:
                        arm=f'{family}_{suffix}_{mode}';control=f'old_{suffix}_{mode}';pairs.append(dict(kind='source',arm=arm,control=control,rule=rule,fee=fee,**paired(q[arm].to_numpy(),q[control].to_numpy())))
            for family in ('old','B0','P6'):
                for mode in MODES:
                    arm=f'{family}_fixed5_{mode}';control=f'{family}_plain_{mode}';pairs.append(dict(kind='augmentation',arm=arm,control=control,rule=rule,fee=fee,**paired(q[arm].to_numpy(),q[control].to_numpy())))
            for r in choice['choices']:
                arm=r['selected_arm'];control=f"{r['family']}_{r['suffix']}_f10";selected_pairs.append(dict(arm=arm,control=control,rule=rule,fee=fee,**paired(q[arm].to_numpy(),q[control].to_numpy())))
    assert len(pairs)==132 and len(selected_pairs)==24
    pd.DataFrame(pairs).to_csv(OUT/'paired.csv',index=False);pd.DataFrame(selected_pairs).to_csv(OUT/'selected_paired.csv',index=False)
    for family in ('old','B0','P6'):
        for suffix in ('plain','fixed5'):
            for mode in ('f30','free'):
                arm=f'{family}_{suffix}_{mode}';control=f'{family}_{suffix}_f10';changed=s[arm].gt(0).ne(s[control].gt(0))
                directions.append(dict(arm=arm,control=control,long_weeks=int(s[arm].gt(0).sum()),changed_weeks=int(changed.sum()),changed_dates=';'.join(d.strftime('%Y-%m-%d') for d in s.index[changed])))
    pd.DataFrame(directions).to_csv(OUT/'directions.csv',index=False)
    yearly=[]
    for key,g in w.assign(year=w.date.dt.year).groupby(['arm','rule','fee','year']):
        r=g.sort_values('date').net.to_numpy();nav=np.r_[1.,np.cumprod(1+r)]
        yearly.append(dict(zip(['arm','rule','fee','year'],key))|dict(weeks=len(g),total_return=float(nav[-1]-1),max_drawdown=float(np.max(1-nav/np.maximum.accumulate(nav))),sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(52)),long_bp=float(g.long_net.mean()*1e4),short_bp=float(g.short_net.mean()*1e4)))
    pd.DataFrame(yearly).to_csv(OUT/'yearly.csv',index=False)
    def values(q,col,k=1,digits=2):return ' / '.join(f'{q.at[f,col]*k:.{digits}f}'+('%' if k==100 else '') for f in (10,30))
    def metric_table(family,rule):
        rows=[]
        for suffix in ('plain','fixed5'):
            for mode in MODES:
                arm=f'{family}_{suffix}_{mode}';q=m.loc[m.arm.eq(arm)&m.rule.eq(rule)].set_index('fee')
                rows.append([suffix,mode,values(q,'annual_calendar',100),values(q,'worst10_avg',100),values(q,'long_bp'),values(q,'short_bp'),values(q,'sharpe',digits=3),values(q,'max_drawdown',100)])
        return table(pd.DataFrame(rows,columns=['增强','冻结','净复合年化10/30bp','最差10%周10/30bp','多腿净bp/周10/30bp','空腿净bp/周10/30bp','夏普10/30bp','最大回撤10/30bp']))
    choice_rows=[];step_rows=[];summary=[]
    for r in choice['choices']:
        arm=r['selected_arm'];a=m.loc[m.arm.eq(arm)&m.rule.eq('full')].set_index('fee');b=m.loc[m.arm.eq(arm)&m.rule.eq('wide')].set_index('fee')
        v=validation.at[f"{r['family']}_{r['suffix']}_f10",'ensemble_bce']
        choice_rows.append([r['family'],r['suffix'],r['selected_mode'],f'{v:.6f}',f"{r['validation_bce']:.6f}",values(a,'annual_calendar',100),values(b,'annual_calendar',100)])
    for family in ('old','B0','P6'):
        for suffix in ('plain','fixed5'):
            row=[family,suffix]
            for mode in MODES:
                items=sorted([r for r in lock['selections'] if r['arm']==f'{family}_{suffix}_{mode}'],key=lambda r:r['encoder_seed'])
                step_rows.append([family,suffix,mode,' / '.join(str(r['step']) for r in items),' / '.join(f"{r['val_bce']:.6f}" for r in items)])
                q=m.loc[m.arm.eq(f'{family}_{suffix}_{mode}')&m.rule.eq('full')].set_index('fee');row.append(values(q,'annual_calendar',100))
            summary.append(row)
    pd.DataFrame(summary,columns=['source','augmentation','f10_cagr_10_30','f30_cagr_10_30','free_cagr_10_30']).to_csv(OUT/'freeze_comparison.csv',index=False)
    results='\n\n'.join(f"## {family}向量：{'全多/全空' if rule=='full' else '40%/70%宽档'}\n\n{metric_table(family,rule)}" for family in ('old','B0','P6') for rule in ('full','wide'))
    selected_table=pd.DataFrame(choice_rows,columns=['来源','增强','验证选冻结','原f10集成BCE','所选集成BCE','全多空年化10/30bp','宽档年化10/30bp'])
    DOC.write_text(f'''# 第10/30步冻结与不冻结投影：旧任务三表征完整对照

2026-11-19，Asia/Shanghai，complete。[预注册协议](research_20261119_bce_freeze_steps_protocol.md)；父批[固定旧40/10三表征](research_20261119_bce_old_roster_vectors_results.md)。54个BCE头全部本轮新训100步，CPU4线程；主干与向量没有新增训练/导出。轻量证据见[目录](evidence/20261119_bce_freeze_steps/README.md)。

## 验证选择与结论读法

第10步冻结限制了这套模型的验证概率拟合，但不能解释全部新旧收益差距。六组来源/增强都由验证BCE选中不冻结：旧约.68595→.68460、B0约.69016→.68886、P6约.69347→.69026（无增强）。P6不冻结时seed44选到第91/88步，表明后期适配确有验证收益；另外两个头仍选10/9步。f30的P6仍选10/9/4，因而与f10判向和交易完全相同。

验证改善没有转化为稳定扣费收益。按事前BCE锁定的不冻结方案，旧向量全多空0/5%年化43.16%/24.53%、44.40%/25.64%，低于原f10；B0为32.48%/13.83%、33.33%/14.57%，总体未提升；P6无增强升至23.10%/7.93%，但最大回撤由32.30%/44.89%增至38.84%/45.64%，5%高费年化反降为5.27%。验证所选减f10的净均差块区间全部跨零，未确认稳定收益增量。

B0的f30历史表现较好：全多空0/5%年化37.54%/17.53%、38.13%/17.93%，高于f10；但验证选择是free，不能按测试年化回头改选f30。按相同冻结比较或各自验证选冻结，旧向量仍领先。结论限定为固定100步/原BCE选择算法的局部适配效果，不将原先全部差距归结为第10步冻结，也不以测试最好臂替代验证锁。

原BCE平均衡量40/10相对收益二值标签的概率误差，实际执行仍是20/10并计交易/券费；每周收益幅度、换向成本和尾部损失没有直接进入该BCE。因此更低验证BCE不保证更高测试净收益或更小回撤。本次将这些口径全部固定，未因收益排序不同而换损失或重选验证规则。

先按每头最低123周验证BCE选步，再平均三个头logit复算集成BCE，按这个BCE选择冻结方案。完全并列（1e−12）优先f10、f30、free。全部54头与六项选择在新测试预测前锁定；不能用测试年化替换下表选择。三冻结方案的完整历史结果全部列出，属于已读历史探索复检。

{table(selected_table)}

全多空三冻结净复合年化对照：

{table(pd.DataFrame(summary,columns=['来源','增强','第10步后冻结10/30bp','第30步后冻结10/30bp','不冻结10/30bp']))}

这个比较只改proj是否继续更新：f10完成10步后关闭，f30完成30步后关闭，free100步不关闭。总更新步数、学习率、Adam状态初始化、其它可训练层、数据及头初始化不变，不重建优化器。原验证规则仍允许选冻结前检查点，不能把“后来才会冻结”的配置名等同于所选检查点已适配30或100步。

## 输入、前缀恒等与训练核验

全部直接使用父批9份输入npz、公共旧40/10/708变体、标签及周权重。原473训练周2001-11-23至2017-12-15；123监督验证周位于2018-04-06至2020-12-18；295测试决策周2021-01-08至2026-10-09，持仓2021-01-11至2026-10-19。旧best3/2/6、B0第3轮三seed、P6第8轮三seed及其特征边界不动。所有来源既有mu/sd按父批473基础周拟合，直接核对父统计，不让冻结方案改变标准化。0/5%仍236启用周、61118、固定708变体、周内.95/.05及每周总权重1，不换股或删周。

LegPool(False)结构、BCE标签/权重和Adam .001/.01沿父批。54条完整100步轨迹共5400行。每来源/seed/增强共18组：三臂前10步损失轨迹和模型参数哈希完全相同，f30/free前30步相同；f10冻结后投影保持第10步，f30保持第30步，free在第30到100步投影确有改变。模型/投影0/10/30/100哈希见parameter_snapshots.csv，轨迹/冻结断言见prefix_checks.csv与loss_checks.csv。初始BCE闭式和解析梯度、54个已选检查点原验证BCE、mu/sd与平均logit恒等式均通过。

f10本轮18头参数/统计/选步复现父批，7080条父周收益逐周复现，容差1e−12。不同冻结对早期路径没有改变，故后续没有更低验证BCE时，早期检查点自然被保留；这与没有执行完整训练不同。

{table(pd.DataFrame(step_rows,columns=['来源','增强','冻结','seed42/43/44所选步','各头验证BCE']))}

每头1–10、11–30、31–100步最低验证BCE及末步训练/验证损失另存training_segments.csv，用于诊断后期是否改善；未用这些段内最优额外创建强制晚步测试组。

## 全部72指标的执行口径

所有信号统一旧20/10实际下单名单、交易面板、现金/受阻成交/退出代理和净值漂移，总名义预算1、同NAV。宽档多头40/70%，全切换多头0/100%。每格10bp+8.35%年券费 / 30bp+12%年券费。复合年化按实际日历跨度，最差10%周为最差30周净收益均值，两腿为对总NAV净bp/周贡献（相加为净周均），Sharpe按周收益sqrt(52)，回撤为周估值。换手/费用原表metrics.csv，验证所选方案24指标另在validation_selected_metrics.csv。

{results}

## 配对、换向与年度

{table(pd.DataFrame(directions).drop(columns='changed_dates'))}

完整换向日期见directions.csv；成员/集成logit见member_logits.csv/signals.csv。paired.csv共132项：晚冻结/free减同来源同增强f10、同冻结新向量减旧向量、各来源各冻结5%减0；selected_paired.csv另24项为验证所选减原f10，含并列原方案的零差。8周移动块5000次95%区间同时给净均差和尾部差；多重比较未校正，跨零不证明严格等效。逐年区间收益、年内MDD、Sharpe及两腿见yearly.csv，2026为部分年度。

## 验收、复现与限制

54头/5400轨迹、18组前缀与冻结核对、54个所选验证、六个冻结选择及平均logit通过；72指标/21240周唯一键、两腿及费用净和、独立汇总通过；18个f10头及7080周父账本复现。全部输入沿父批SHA逐项验证，没有重抽增强、改数据源/预处理、扩100步预算或改变测试日期。

项目根目录设置PYTHONPATH=scripts;src，python -u outputs/bce_freeze_steps_v1/run.py，再python outputs/bce_freeze_steps_v1/report.py。当前complete，无本批待跑任务，不占GPU、不改其它队列，不部署、不自动Git提交。

本轮只检验100步、原学习率/模型/验证选择下的冻结时点，不判定更长训练或其它优化方法的上限。继承原473选择、主干样本内向量/名单、幸存池/修订、记平与退出估值及真实券源未知限制；历史已读，未做新的独立样本外验证。原行情、向量、9主干及54头仅本机，未核实异地备份，轻量证据不是全量重训包。
''',encoding='utf8')
    hashes=json.loads((OUT/'inputs.json').read_text(encoding='utf8'))
    for p,h in hashes.items():
        if Path(p).resolve()!=Path(__file__).resolve():assert sha(p)==h,('changed input',p)
    for p in (Path(__file__),ROOT/'scripts/replay_kit_risk.py'):hashes[str(p.resolve())]=sha(p)
    (OUT/'inputs.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    verify=json.loads((OUT/'verification.json').read_text(encoding='utf8'));verify.update(report_complete=True,independent_metric_rows=72,main_paired_comparisons=132,selected_paired_comparisons=24,input_hashes_verified=len(hashes),selected_validation_verified=54,mean_logit_verified=True)
    for name in ('verification.json','state.json'):(OUT/name).write_text(json.dumps(verify,indent=2),encoding='utf8')
    names=['run.py','report.py','run.log','model_inventory.json','training_loss.csv','parameter_snapshots.csv','loss_checks.csv','prefix_checks.csv','f10_parameter_reproduction.csv',
        'validation_lock.json','selection_lock.json','validation_ensemble.csv','signals.csv','member_logits.csv','metrics.csv','validation_selected_metrics.csv','checkpoint_checks.csv',
        'training_segments.csv','paired.csv','selected_paired.csv','directions.csv','yearly.csv','freeze_comparison.csv','inputs.json','verification.json','state.json']
    EVIDENCE.mkdir(parents=True,exist_ok=True);manifest=[]
    for src,name in [(OUT/name,name) for name in names]+[(ROOT/'docs/research_20261119_bce_freeze_steps_protocol.md','protocol.md')]:
        dest=EVIDENCE/name;shutil.copyfile(src,dest);h=sha(src);assert sha(dest)==h
        manifest.append(dict(source=str(src.relative_to(ROOT)),copy=name,bytes=src.stat().st_size,sha256=h))
    (EVIDENCE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');h=sha(EVIDENCE/'manifest.json')
    (EVIDENCE/'README.md').write_text(f'''# 投影冻结时点完整对照证据

2026-11-19，Asia/Shanghai。[结果](../../research_20261119_bce_freeze_steps_results.md)。{len(manifest)}份源码、54头完整训练/锁、前缀参数核对、72指标及配对/年度轻量证据逐字节复制，见[manifest](manifest.json)，manifest SHA256为`{h}`。原数据、向量、主干、BCE权重及完整周账本仅本机，未核异地备份；非全量重训包，源码按原工作区依赖运行，未自动Git提交。
''',encoding='utf8')
    print(table(selected_table));print('Published',DOC,'with',len(manifest),'verified evidence copies; manifest',h)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        (OUT/'state.json').write_text(json.dumps(dict(status='failed',phase='report',error=repr(exc)),indent=2),encoding='utf8');raise
