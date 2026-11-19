"""Analyze the locked equivalence experiment and preserve reviewable evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from replay_kit_risk import paired

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/bce_equiv_init_v1"
DOC=ROOT/"docs/research_20261119_bce_equiv_init_results.md"
EVIDENCE=ROOT/"docs/evidence/20261119_bce_equiv_init"

def table(frame):
    rows=["| "+" | ".join(map(str,frame.columns))+" |","| "+" | ".join(["---"]*len(frame.columns))+" |"]
    rows.extend("| "+" | ".join(map(str,row))+" |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(rows)

def main():
    m=pd.read_csv(OUT/"metrics.csv");w=pd.read_parquet(OUT/"weekly.parquet")
    s=pd.read_csv(OUT/"signals.csv",parse_dates=["date"]).set_index("date")
    verify=json.loads((OUT/"verification.json").read_text())
    lock=json.loads((OUT/"validation_lock.json").read_text())
    init=pd.read_csv(OUT/"initial_equivalence.csv");div=pd.read_csv(OUT/"divergence.csv")
    assert len(m)==48 and len(w)==14160 and len(s)==295
    assert not w.duplicated(["date","arm","rule","fee","long_slots"]).any()
    np.testing.assert_allclose(w.net,w.long_net+w.short_net,atol=1e-12,rtol=0)
    pair_rows=[]
    for rule in ("wide","full"):
        for suffix in ("plain","fixed5"):
            for fee in (10,30):
                q=w.query("rule==@rule and fee==@fee and long_slots==20").pivot(index="date",columns="arm",values="net")
                for a,b in (("equiv","old"),("equiv","naive"),("old","base40")):
                    pair_rows.append(dict(rule=rule,suffix=suffix,fee=fee,candidate=a,control=b,
                                          **paired(q[a+"_"+suffix].to_numpy(),q[b+"_"+suffix].to_numpy())))
                a=w.loc[w.arm.eq("base40_"+suffix)&w.rule.eq(rule)&w.fee.eq(fee)&w.long_slots.eq(40)].sort_values("date")
                b=w.loc[w.arm.eq("base40_"+suffix)&w.rule.eq(rule)&w.fee.eq(fee)&w.long_slots.eq(20)].sort_values("date")
                pair_rows.append(dict(rule=rule,suffix=suffix,fee=fee,candidate="base40_execute40",control="base40_execute20",**paired(a.net.to_numpy(),b.net.to_numpy())))
    pairs=pd.DataFrame(pair_rows);pairs.to_csv(OUT/"paired.csv",index=False)
    events=[];directions=[]
    for suffix in ("plain","fixed5"):
        changed=s["equiv_"+suffix].gt(0)!=s["old_"+suffix].gt(0)
        directions.append(dict(suffix=suffix,changed_vs_old=int(changed.sum()),old_long=int(s["old_"+suffix].gt(0).sum()),
                               equiv_long=int(s["equiv_"+suffix].gt(0).sum()),changed_vs_naive=int((s["equiv_"+suffix].gt(0)!=s["naive_"+suffix].gt(0)).sum())))
        for day in s.index[changed]:events.append(dict(suffix=suffix,date=day,old_logit=s.at[day,"old_"+suffix],equiv_logit=s.at[day,"equiv_"+suffix]))
    pd.DataFrame(directions).to_csv(OUT/"directions.csv",index=False)
    pd.DataFrame(events).to_csv(OUT/"changed_weeks.csv",index=False)
    labels=dict(base40="原40/10输入及标签基准",old="20/10，原统计",naive="20/10，自身统计，普通初始化",equiv="20/10，自身统计，等价初始化")
    def metric_table(rule,execute40=False):
        rows=[]
        for a in (("base40",) if execute40 else ("base40","old","naive","equiv")):
            for suffix in ("plain","fixed5"):
                arm=a+"_"+suffix;nl=40 if execute40 else 20
                q=m.query("arm==@arm and rule==@rule and long_slots==@nl").set_index("fee")
                def values(col,scale=1.,digits=2):return " / ".join(f"{q.at[f,col]*scale:.{digits}f}"+("%" if scale==100 else "") for f in (10,30))
                rows.append([labels[a],"无增强" if suffix=="plain" else "固定5%",values("annual_calendar",100),values("max_drawdown",100),
                             values("sharpe",digits=3),values("worst10_avg",100),values("long_bp"),values("short_bp")])
        return table(pd.DataFrame(rows,columns=["配比头","增强","净复合年化10/30bp","最大回撤10/30bp","夏普10/30bp","最差10%周均10/30bp","多腿净bp/周10/30bp","空腿净bp/周10/30bp"]))
    step_rows=[]
    for suffix in ("plain","fixed5"):
        for a in ("old","equiv"):
            rows=[r for r in lock["selections"] if r["arm"]==a+"_"+suffix]
            assert [r["seed"] for r in rows]==[42,43,44]
            step_rows.append([a,suffix," / ".join(str(r["step"]) for r in rows)," / ".join(f"{r['val_bce']:.6f}" for r in rows)])
    q=m.query("long_slots==20 and rule=='full'").set_index(["arm","fee"])
    recovery=[]
    for suffix in ("plain","fixed5"):
        for fee in (10,30):
            old=q.at[("old_"+suffix,fee),"annual_calendar"]
            naive=q.at[("naive_"+suffix,fee),"annual_calendar"]
            equiv=q.at[("equiv_"+suffix,fee),"annual_calendar"]
            recovery.append(dict(suffix=suffix,fee=fee,old_cagr=old,naive_cagr=naive,equivalent_cagr=equiv,
                                 equivalent_minus_old_pp=(equiv-old)*100,equivalent_minus_naive_pp=(equiv-naive)*100))
    pd.DataFrame(recovery).to_csv(OUT/"recovery.csv",index=False)
    cost_rows=[]
    for a in ("old","equiv"):
        for suffix in ("plain","fixed5"):
            arm=a+"_"+suffix
            r=m.query("arm==@arm and rule=='full' and long_slots==20").set_index("fee")
            cost_rows.append([arm,f"{r.at[10,'turnover']:.4f} / {r.at[30,'turnover']:.4f}",
                              f"{r.at[10,'trading_bp']:.2f} / {r.at[30,'trading_bp']:.2f}",f"{r.at[10,'borrow_bp']:.2f} / {r.at[30,'borrow_bp']:.2f}"])
    p=pairs.query("rule=='full' and candidate=='equiv'")
    prows=[[r.suffix,r.fee,r.control,f"{r.mean_delta_bps:.2f}",f"[{r.mean_lo_bps:.2f}, {r.mean_hi_bps:.2f}]"] for r in p.itertuples()]
    rtab=pd.DataFrame(recovery).copy()
    for col in ("old_cagr","naive_cagr","equivalent_cagr"):rtab[col]=(rtab[col]*100).map(lambda x:f"{x:.2f}%")
    for col in ("equivalent_minus_old_pp","equivalent_minus_naive_pp"):rtab[col]=rtab[col].map(lambda x:f"{x:+.2f}")
    text=f"""# 20/10等价初始化及旧40/10实际下单复核

2026-11-19，Asia/Shanghai，complete。[协议](research_20261119_bce_equiv_init_protocol.md)，父批[因子拆解](research_20261118_bce_input_label_results.md)。本批实际新训12头，完整锁定后评分295周；[轻量证据](evidence/20261119_bce_equiv_init/README.md)。原40/10和普通初始化自身统计为已登记参考，不再以测试选择方法。

## 结论

等价初始化修复了本批观察到的自身归一化退步。全切换无增强由普通自身统计的7.57%/−9.36%提高到19.04%/0.69%，固定5%由9.90%/−7.31%提高到24.69%/5.79%；对应原统计20/10分别为18.13%/−0.31%及19.80%/1.13%。等价初始化略高于原统计，尚未消除20/10输入及标签相对原40/10信号的巨大差距。不能把“修复已观察退步”推成普适稳定收益优势，配对区间见后文。

原40/10信号指导实际旧40/10也已实测，本批复核全切换0/5%年化48.14%/30.25%、50.11%/31.91%，回撤均26.59%/27.35%。相同信号指导旧20/10的收益稍高、回撤也稍高。

## 问题和比较

本批只问：固定20/10输入及标签后，把自身归一化的第一层改成与原归一化同一初始函数，能否修复父批退化？原40/10训练目标及输入保持参考，实际下单主表全部旧Top20/Bottom10。另复核原40/10信号指导旧Top40/Bottom10，输入篮子、标签篮子和下单篮子分别注明。

原473训练周2001-11-23至2017-12-15、原123完整验证监督周（2018-04-06至2020-12-18的133周范围）、295测试决策周2021-01-08至2026-10-09固定。入场2021-01-11至末退出2026-10-19，未按未来标签可用性筛测试日期。三个旧abl2000_base主干seed42/43/44、best轮次3/2/6、向量和榜单不变，主干均冻结。原473周仍是主干样本内构造，未改为滚动样本外名单。验证及测试历史已读。

架构沿用129→32投影、分腿均值、LayerNorm、32→64→32残差FFN、拼接64→1；全批Adam lr=.001、weight_decay=.01，100步，第11步开始冻结投影。最低验证BCE选步后平均三头logit，0阈值选向。标签仍为旧名义20/10的1[mean(R_long)+mean(R_short)+.08/52>0]，是两腿相对收益比较。标签及输入收益口径与执行层沿用父批差异，没有随归一化改变。

5%使用父批D/E成对抽样：Top50/Bottom20候选池、61118随机流、236启用周、220有效周、634个原40/10变体，20取同一40变体排序前20，空头10相同。有效周.95基础损失加.05变体均值，无变体周权重1；每周总权重1。两统计的原始输入、标签及损失权重指纹完全相同，并与父批D逐项相同。本批不更换为原生直接抽20的G抽样器。

## 等价初始化实际做法

原统计mu_o,s_o、自身统计mu_n,s_n，令原随机初始化为W_o,b_o。

\\[
W_n=W_o\\operatorname{{diag}}(s_n/s_o),\\qquad
b_n=b_o+W_o((mu_n-mu_o)/s_o).
\\]

于是W_n((x-mu_n)/s_n)+b_n=W_o((x-mu_o)/s_o)+b_o。其它层逐位拷贝，同一seed的两头起始函数相同。没有拿训练完成的40/10权重暖启动，没有使用验证或测试拟合统计。训练前在完整训练（包括变体）和验证输入上逐项检验投影及最终logit；初始logit最大误差{init.filter(like='logit_error').to_numpy().max():.3g}，投影最大误差{init.filter(like='projection_error').to_numpy().max():.3g}，均通过1e-5浮点容差。

{table(init.round(9))}

等价初始化只保证第0步。两臂均在自己的参数坐标中继续执行原Adam及L2衰减，未变换梯度、步长或优化器状态。简单缩放w_n=d*w_o时，即使无正则，保持同一函数要求Delta w_n=d*Delta w_o；逐坐标Adam更新通常不满足该关系。L2范数也随坐标变化，||W_o diag(s_n/s_o)||²通常不等于||W_o||²。因此训练后分叉有数学原因，不能将等价初始化描述为训练全过程等价。

首次更新后验证logit最大差达到{div.query('step==1').logit_max.max():.4f}。第10步后投影已冻结，投影函数差随后保持，但后续非线性头继续优化。逐步误差见divergence.csv。

{table(div.query('step==1 or step==10 or step==100').round(6))}

## 全多/全空结果：实际旧20/10下单

以下单元按10bp/30bp顺序。10bp同时配年券费8.35%，30bp配12%；总名义预算1，同一NAV分母，开仓受阻可留现金。年化按首入场至末退出日历跨度；最大回撤为周估值；最差10%为295周中最差30周净收益均值。两腿是同一NAV下平均周净贡献，不能相加为复合年化。

{metric_table('full')}

归一化修复的全切换年化变化如下，末两列单位为百分点：

{table(rtab)}

原归一化与等价初始化各组的seed42/43/44验证选步一致，均为100/16/100。因此本次它们的测试差异不能归于检查点步数不同；预存原归一化固定阶段状态与选中状态相同。普通自身统计参考E此前为100/17/100。

{table(pd.DataFrame(step_rows,columns=['归一化','增强','选步42/43/44','最低验证BCE42/43/44']))}

方向变化相对于同增强的原统计20/10信号：

{table(pd.DataFrame(directions))}

完整日期见changed_weeks.csv。不能仅因验证BCE略低推断扣费收益提高，方向损失和金额损失目标不同。

## 同信号40%/70%宽档

{metric_table('wide')}

全切换的平均每周换手及费用（同样按10/30bp顺序；换手为权重L1，1表示100% NAV成交额）：

{table(pd.DataFrame(cost_rows,columns=['配比头','平均换手','交易bp/周','券费bp/周']))}

## 原40/10信号指导旧Top40/Bottom10：已测且本批复核

这项已有[完整295周审计](research_20261118_bce_final_audit_results.md)，本批用同一旧40/10配比信号重放，0/5%、两规则、两费用全部1180行每组逐周净收益与旧独立账本一致。下单40/10与主表下单20/10用完全相同信号，仅执行篮子不同。增加多头数量还可能通过避开多头的空头筛选规则影响空头名单，实际按同一pick_targets重算，没有强行假设空头绝对相同。

全多/全空：

{metric_table('full',True)}

40%/70%宽档：

{metric_table('wide',True)}

40/10实际执行的全切换收益略低于同信号20/10，回撤也略低，收益和风险存在取舍。本批没有证明更宽实际执行优于20/10，也没有继续扫描更多人数。

## 配对不确定性和判断边界

沿用8周移动块、5000次采样的95%配对自助区间。全切换的净周均增量如下：

{table(pd.DataFrame(prows,columns=['增强','费用bp','对照','净均差bp/周','95%区间']))}

完整32项比较及尾部差区间见paired.csv。年化差含复利和路径，不能按单因素直接加总；区间基于重复读取的历史，未作多重比较校正。初始化是否修复，须同时看它相对普通自身统计E和原统计D的表现；即使修复归一化部分，也不能把D与原40/10输入标签A的差距归于归一化。

## 验收与执行记录

12头×100步=1200条损失，6组成对训练支持指纹、6组初始等价误差、600条逐步分叉。投影完成第10步后逐位不再更新，实际验证选中检查点均超过第10步。12头选步全部先锁后读测试收益。

原统计20/10的测试logit复现父批D。原统计、自身普通初始化参考、原40/10参考在20/10执行共7080行，加原40/10执行2360行，合计{verify['archived_weekly_rows_reproduced']}行周净收益与父批/独立账本一致（1e-12容差）。48行指标、14160行周账本，long_net+short_net=net；日期无重复，295周无删减。

首次run.py已经完成12头训练、选步锁及测试推理，但在对照账本核对时发生pandas查询字符串拼接语法错误，退出1，未发布指标。修正为明确布尔索引后以--score-only读取原验证锁，校验全部原训练源码、依赖及12权重SHA256，不更新模型、优化器或选步，重新回放并退出0。原训练源码保留training_source.py，原错误保留run.log，修复回放见replay.log；此变更只涉及核对与评分恢复，不属于追加训练。

运行命令：`$env:PYTHONPATH='scripts;src'; python -u outputs/bce_equiv_init_v1/run.py`；评分恢复：`python -u outputs/bce_equiv_init_v1/run.py --score-only`；分析及归档：`python outputs/bce_equiv_init_v1/report.py`。训练CPU4线程，未重训主干、未改GPU队列。本批state/verification为complete，无本批待跑任务。

源码、原训练源码、依赖与检查点哈希见inputs.json/validation_lock.json。原向量和行情、12权重及Parquet仍仅本机，未核实异地备份；轻量证据复制代码、锁、损失、指标、配对和信号，不能独立完成全量重训。固定幸存池、历史数据修订、旧样本内训练名单、缺收益记零、受阻退出代理、真实券源及已读测试选择偏差均继承。未改优化器或正则几何，未进行后续温度/冻结预算搜索。
"""
    DOC.write_text(text,encoding="utf8")
    inputs=json.loads((OUT/"inputs.json").read_text())
    with Path(__file__).open("rb") as stream:inputs[str(Path(__file__).resolve())]=hashlib.file_digest(stream,"sha256").hexdigest()
    (OUT/"inputs.json").write_text(json.dumps(inputs,indent=2),encoding="utf8")
    verify.update(report_complete=True,paired_comparisons=len(pairs),initial_max_logit_error=float(init.filter(like="logit_error").to_numpy().max()))
    (OUT/"verification.json").write_text(json.dumps(verify,indent=2),encoding="utf8")
    (OUT/"state.json").write_text(json.dumps(verify,indent=2),encoding="utf8")
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    files=['run.py','training_source.py','report.py','run.log','replay.log','metrics.csv','training_loss.csv','validation_lock.json','identity.csv','initial_equivalence.csv','divergence.csv','signals.csv','paired.csv','directions.csv','changed_weeks.csv','recovery.csv','inputs.json','verification.json','state.json']
    manifest=[]
    for name in files:
        src=OUT/name;dst=EVIDENCE/name
        shutil.copyfile(src,dst)
        with src.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        with dst.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha
        manifest.append(dict(source=str(src.relative_to(ROOT)),copy=name,bytes=src.stat().st_size,sha256=sha))
    (EVIDENCE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    with (EVIDENCE/'manifest.json').open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    (EVIDENCE/'README.md').write_text(f'''# 归一化等价初始化的轻量证据

2026-11-19，Asia/Shanghai。[结果入口](../../research_20261119_bce_equiv_init_results.md)。本目录{len(manifest)}份逐字节复制件，来源、大小和SHA256见[manifest](manifest.json)，manifest SHA256为`{sha}`。保留原训练源码和失败日志、修复回放入口与日志、1200条训练损失、验证锁、48行指标及32组配对区间。

原始主干、12个新权重、向量/行情和完整周账本仅在原data/outputs，未核实异地备份。本目录可版本化，不代表已Git提交或推送，也不支持脱离原数据的全量独立重训。脚本按原工作区路径运行，复制件只作审阅。
''',encoding='utf8')
    print(pd.DataFrame(recovery).round(6).to_string(index=False))
    print('Published',DOC,'with',len(manifest),'verified evidence copies; manifest',sha)

if __name__=='__main__':main()
