"""Publish the completed, locked factorial diagnosis without choosing on test."""
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/bce_input_label_v1"
DOC=ROOT/"docs/research_20261118_bce_input_label_results.md"
EVIDENCE=ROOT/"docs/evidence/20261118_bce_input_label"

def table(frame):
    rows=["| "+" | ".join(map(str,frame.columns))+" |","| "+" | ".join(["---"]*len(frame.columns))+" |"]
    rows.extend("| "+" | ".join(map(str,row))+" |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(rows)

def main():
    m=pd.read_csv(OUT/"metrics.csv")
    lock=json.loads((OUT/"validation_lock.json").read_text())
    verify=json.loads((OUT/"verification.json").read_text())
    assert verify["status"]=="complete" and verify["normalization_affine_equivalence_verified"]
    labels=pd.read_csv(OUT/"label_changes.csv")
    changes=labels.groupby("split").changed.agg(["count","sum"])
    direction=pd.read_csv(OUT/"directions.csv").set_index("arm")
    titles=dict(A="原输入40/10、标签40/10、原归一化",B="只换输入20/10",C="只换标签20/10",D="输入及标签均20/10，原归一化",E="D再换自身归一化，成对抽样",F="D再换原生抽样，原归一化",G="输入标签20/10、自身归一化、原生抽样")
    tables={}
    for rule in ("full","wide"):
        rows=[]
        for name in "ABCDEFG":
            for suffix in ("plain","fixed5"):
                arm=name+"_"+suffix
                ten=m.query("arm==@arm and rule==@rule and fee==10").iloc[0]
                thirty=m.query("arm==@arm and rule==@rule and fee==30").iloc[0]
                rows.append([name,"无增强" if suffix=="plain" else "固定5%",f"{ten.annual_calendar*100:.2f}% / {thirty.annual_calendar*100:.2f}%",f"{ten.max_drawdown*100:.2f}% / {thirty.max_drawdown*100:.2f}%",f"{ten.sharpe:.3f} / {thirty.sharpe:.3f}",f"{ten.worst10_avg*100:.2f}% / {thirty.worst10_avg*100:.2f}%",f"{ten.long_bp:.2f} / {thirty.long_bp:.2f}",f"{ten.short_bp:.2f} / {thirty.short_bp:.2f}"])
        tables[rule]=table(pd.DataFrame(rows,columns=["配置","增强","净复合年化10/30bp","最大回撤10/30bp","夏普10/30bp","最差10%周均10/30bp","多腿净bp/周10/30bp","空腿净bp/周10/30bp"]))
    steps=[]
    for name in "ABCDEFG":
        vals=[]
        for suffix in ("plain","fixed5"):
            vals.append("/".join(str(x["step"]) for x in lock["selections"] if x["arm"]==name+"_"+suffix))
        steps.append([name,*vals])
    fixed=[]
    for name in "ABCDEFG":
        vals=[]
        for suffix in ("plain","fixed5"):
            arm=name+"_"+suffix+"_fixedstep"
            q=m.query("arm==@arm and rule=='full'").set_index("fee")
            vals.append(f"{q.at[10,'annual_calendar']*100:.2f}% / {q.at[30,'annual_calendar']*100:.2f}%")
        fixed.append([name,*vals])
    flows=[]
    for name in "ABCDEFG":
        a,b=direction.loc[name+"_plain"],direction.loc[name+"_fixed5"]
        flows.append([name,f"{a.changed_weeks} / {b.changed_weeks}",f"{a.long_weeks} / {b.long_weeks}"])
    pairs=pd.read_csv(OUT/"paired.csv").query("fee==10 and arm.str.endswith('plain')",engine="python")
    pair_rows=[[x.arm+"−"+x.control,f"{x.mean_delta_bps:.2f}",f"[{x.mean_lo_bps:.2f}, {x.mean_hi_bps:.2f}]"] for x in pairs.itertuples()]
    effects=pd.read_csv(OUT/"decomposition.csv").round(3)
    attr=pd.read_csv(OUT/"attribution.csv").query("fee==10 and arm.str.endswith('plain')",engine="python")
    attr_rows=[[x.arm+"−"+x.control,*[f"{v:.3f}" for v in (x.net,x.gross_ret,x.trading_cost,x.borrow_cost)]] for x in attr.itertuples()]
    event=pd.read_csv(OUT/"largest_loss_weeks.csv").query("arm=='G_plain' and control=='A_plain' and fee==10").head(5)
    text=f"""# 20/10退化：输入、标签、归一化和抽样的控制变量验收

2026-11-18，Asia/Shanghai，complete。[协议](research_20261118_bce_input_label_protocol.md)，父批为[篮子人数比较](research_20261118_bce_basket_sizes_results.md)。本批实际重新训练42个配比头，包含重复配置复现；没有重新训练股票主干。轻量源码和结果见[证据目录](evidence/20261118_bce_input_label/README.md)。验证及测试历史均已读，本批是解释既有退化的探索诊断。

## 结论

原20/10结果同时更换了输入篮子、标签篮子和归一化统计，固定5%还更换了抽样分布，不能把差距归为只改标签。本次固定473训练周、123验证选步周、295测试周和旧20/10实际执行，分开后：只换输入损失最大，只换标签也退步，两者同时改变存在交互；自身归一化进一步拉低收益。固定原训练步数后上述退化仍存在，验证选步放大了部分差距。原生抽样器在固定5%中略有修复，未解释主要退化。

原40/10信号依然领先本批所有替代方案。其训练目标是比较两腿相对收益，训练观察较宽篮子，再把方向迁移至20/10实际执行，数学上没有要求输入、标签与下单篮子人数必须一致。数据对齐本身不保证预测更好。

## 固定项、数据与训练

- 原三个abl2000_base主干seed42/43/44、原best.pt轮次3/2/6、同向量缓存、同选股排名，主干均冻结。每只股票128维向量加正负腿标记，129→32投影，分别按明确人数均值池化、LayerNorm、32→64→32残差FFN、拼接后64→1。无注意力分支，参数量相同。
- 原473周的主干向量和名单属于原样本内构造，本次为控制变量予以保留，没有改成滚动样本外名单；其泛化偏差继承原实验，不能把本次拆解当作解决该问题。
- 原473周从2001-11-23至2017-12-15；原验证123个完整监督周处于2018-04-06至2020-12-18的133周范围，未换验证集或删训练周。日期逐项保存在validation_lock.json。
- 全批Adam，lr=.001，weight_decay=.01，100步；第11步开始投影requires_grad=False，其它层继续更新。按原规则各seed取最低验证BCE，平均三头logit后以0选向。选中第3/4等步发生在冻结前，不声称这些选中权重经历了冻结后训练。
- 测试固定2021-01-08至2026-10-09的295个决策周，2021-01-11入场至2026-10-19退出。未按未来标签是否可用筛测试日期。全部策略实际下单均旧Top20/Bottom10，成交、现金留存、受阻退出代理等沿用最终审计。
- 主2×2的A/B/C/D均沿用原40/10各seed的mu/sd，不重新拟合。E/G的自身统计只在原473周基础训练输入拟合，验证和测试不参与。
- 费用分别为单边10bp+年券费8.35%、单边30bp+年券费12%，因此两档包含券费压力变化。全切换为多头比例0或100%；宽档为40%或70%；预算分母为同一NAV。实际成交受阻时可留现金，名义全仓不代表每周实际满仓。

训练标签令旧名义多头篮子未来算术均值为R_L，空头股票本身未来算术均值为R_S：

\\[
y_t=\\mathbf1\\{{R_{{L,t}}+R_{{S,t}}+0.08/52>0\\}}.
\\]

这是R_L与−R_S−0.08/52谁更好的比较，标签1不表示所有股票上涨，也不保证所选腿绝对赚钱。标签是原旧收益口径，实际回放另用现代执行；本次保持这个差异，而没有把执行标签一并改掉。

\\[
\\mathcal L=\\frac1{{473}}\\sum_t\\sum_jw_{{tj}}\\operatorname{{BCEWithLogits}}(f_\\theta(X_{{tj}}),y_{{tj}}),\\qquad\\sum_jw_{{tj}}=1.
\\]

无增强每周仅原名单，权重1。固定5%使用原seed61118和236个启用周、Top50/Bottom20池；可构造变体的周原样本权重.95，变体总权重.05，至多3个变体，抽样后不再随步数变化。主成对方案有220个有效周、634个变体；没有有效变体的周仍是原样本权重1。

## 配置与抽样控制

{table(pd.DataFrame([[k,v] for k,v in titles.items()],columns=["配置","定义"]))}

A/B/C/D共用同一有效抽样支持及样本权重。20输入或标签取同一个原40变体中按原排序的前20，空头10相同；因此B与A标签完全一致，C与A输入完全一致。这种成对20抽样不等于直接从50均匀抽20。F/G再换回上批原生20抽样器，桥接回之前12.39%的固定5%结果。普通组F=D、G=E，本批也重训以核对重复配置；实际42头而非协议原拟重合模型复用，训练预算和评价规则未变。

## 验证选步后的全切换

各单元均按10bp / 30bp顺序。尾部为295周中最差30周净收益的算术均值。多空两腿为对同一组合NAV的平均周净贡献，二者相加为组合平均周净收益，不是各腿独立复合年化。

{tables['full']}

## 相同信号的40%/70%宽档

{tables['wide']}

## 固定训练阶段，排除选步的混杂

原步骤按seed42/43/44，无增强100/14/67，固定5%100/14/81。以下用各自训练轨迹中这些预先固定步数的状态，不按测试重新选步，仅全切换。

{table(pd.DataFrame(fixed,columns=["配置","无增强年化10/30bp","固定5%年化10/30bp"]))}

只换输入在固定阶段仍由50.01%降至29.13%，只换标签降至39.49%；同时换输入标签为23.38%，自身归一化再降至15.19%。所以早停不是全部原因。固定阶段结果比各自验证所选步更好，并不授权用测试改选轮。

实际最低验证BCE所选步数：

{table(pd.DataFrame(steps,columns=["配置","无增强42/43/44","固定5%42/43/44"]))}

## 为什么变化能这么大

**输入改变**：投影是仿射层，均值池化满足mean(Wx+b)=W mean(x)+b。40改20直接改变多头平均向量，并去掉排名21–40的共同信息；不会只是更换token数量而保持同一输入。无增强测试相对原信号换57周方向。较宽篮子更适合这批择时头的结论由对照支持，但“减少噪声”或“更多市场信息”尚未进一步逐特征识别，不能当作已证明的经济机制。

**标签改变**：473训练周只有{int(changes.at['train','sum'])}周翻标签，123验证周有{int(changes.at['validation','sum'])}周翻标签。BCE的logit梯度为p−y，单个0/1翻转在相同参数点会使该样本梯度改变1；多次更新和非线性可使最终多个周越过0。验证BCE也改变了，C组选中步由100/14/67变100/3/4，这解释为什么极少标签改变仍能使训练和选择显著分叉。固定步诊断仍退化，说明不只选步。

**归一化改变**：即使股票原向量没变，40/10 token的正负标记比例为4:1，20/10是2:1。均值约.6→.333、标准差约.8→.943；归一化后多头标记约.5→.707，空头−2→−1.414。128个股票维度的mu/sd也改变。因此同一随机初始化不等于同一初始函数，前10步投影及其冻结状态会不同。自身归一化是合法的训练期拟合，不是未来泄漏。

归一化本身没有删除信息。对原(W,b,mu,sd)，换(mu',sd')时存在精确等价投影：

\\[
W'=W\\operatorname{{diag}}(sd'/sd),\\quad b'=b+W((mu'-mu)/sd),
\\]

\\[
W'((x-mu')/sd')+b'=W((x-mu)/sd)+b.
\\]

三seed数值自检通过。这证明该差异属于本训练初始化、Adam/正则和冻结/选步流程对坐标的敏感性，不是自身归一化在理论上一定较差。没有分别隔离标记维和128股票维，也没有用等价重参数化再训练来确定二者份额。

**抽样改变**：固定5%在旧统计下成对D为19.80%，原生F为21.64%；自身统计下E为9.90%，原生G为12.39%。这项改动略改善，幅度不足以抵消输入和归一化退步；不能把原生G的12.39与成对E的9.90混写。

方向变化（相对同增强A；295周）：

{table(pd.DataFrame(flows,columns=["配置","换向周数：无增强/5%","做多周数：无增强/5%"]))}

## 可相加的收益分解和费用归因

令a/b/c/d为A/B/C/D逐周净收益，输入效应b−a，标签效应c−a，交互d−b−c+a；三者相加恰为d−a。年化有复利与波动影响，不能直接相加归因。

{table(effects)}

上表除fee外均为平均bp/周，suffix为plain无增强或fixed5。无增强10bp：−33.889−21.737+9.574=−46.052；再加自身归一化−17.665，总退化−63.717bp/周。5%有交互，不能拿普通组比例直接分摊。

净差=毛差−交易费用差−券费差，单位bp/周，正费用差表示费用增加：

{table(pd.DataFrame(attr_rows,columns=["比较","净差","毛差","交易费用差","券费差"]))}

原生无增强G相对A毛收益少57.556bp/周，额外交易及券费约6.162bp/周；主要退化发生在方向对应的毛收益，费用只是部分放大。换向次数和多头周数会改变费用，不能固定一笔费用手工从毛收益表扣。

G无增强相对A，10bp亏损增量最大的5周如下（单周bp，对总组合NAV）：

{table(event[["date","delta_bp"]].round(2))}

## 配对不确定性

对295周共同日期做8周移动块、5000次自助采样，95%区间，算法沿用replay_kit_risk.paired。无增强10bp的净周均差：

{table(pd.DataFrame(pair_rows,columns=["比较","净均差bp/周","95%块自助区间"]))}

只换输入及完整原生20/10退化区间不跨零；只换标签和归一化单因素部分区间跨零。这些是重复读取历史上的描述性区间，不是多重比较校正后的确认性检验。其它增强/费用的完整区间见paired.csv。

## 验收、偏差和复现

- 42头×100步=4200条训练损失；42份输入/监督/权重/统计指纹；全部投影完成第10步后不再更新。
- A/B/C/D同seed统计及样本权重哈希相同；A/B监督相同、A/C输入相同等全部断言通过。
- A复现原40/10信号；G复现上批原生20/10信号。A两增强、两规则、两费用共2360行周净收益与原独立账本逐条相同，atol=1e−12。
- 84行指标、24780行逐周账本；long_net+short_net=net、日期不重不漏、交互及毛费归因恒等式通过。归一化等价投影自检通过。
- 实际本批CPU训练；主干向量缓存复用，未占GPU重训主干。42头实际包含协议原拟复用的重复组，已披露。验证123完整标签周与原133推理范围不同，不暗中把133个周全用于监督。
- 命令：`$env:PYTHONPATH='scripts;src'; python -u outputs/bce_input_label_v1/run.py`；分析：`python -u outputs/bce_input_label_v1/analyze.py`。运行退出0，analysis.log与verification.json均complete，本批无待跑训练任务。
- 原协议保留其planned登记文本，本页及verification为完成状态；本页指向当前结果，不覆盖父批数字。源码、锁和引用SHA256见inputs.json，原始权重、向量、行情及完整Parquet仅本机，未核实异地备份。

下一步不自动扩预算或按测试换检查点。若要检验更窄篮子能否重新适应，可另登记等价初始化、冻结步数与验证选择的配对实验；本轮没有证明20/10天然不可预测，也没有证明40/10或固定5%是普适最优。保留固定幸存池、数据修订、缺失收益记零、受阻退出代理及真实券源未知等继承边界；最大回撤为周度估值，不是日内回撤。
"""
    DOC.write_text(text,encoding="utf8")
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    files=["run.py","analyze.py","report.py","run.log","analysis.log","metrics.csv","training_loss.csv","identity.csv","validation_lock.json","signals.csv","decomposition.csv","paired.csv","attribution.csv","directions.csv","normalization.csv","label_changes.csv","largest_loss_weeks.csv","verification.json","inputs.json"]
    entries=[]
    for name in files:
        src=OUT/name;dst=EVIDENCE/name
        shutil.copyfile(src,dst)
        digest=hashlib.file_digest(src.open("rb"),"sha256").hexdigest()
        assert hashlib.file_digest(dst.open("rb"),"sha256").hexdigest()==digest
        entries.append(dict(source=str(src.relative_to(ROOT)),copy=name,bytes=src.stat().st_size,sha256=digest))
    (EVIDENCE/"manifest.json").write_text(json.dumps(entries,indent=2),encoding="utf8")
    digest=hashlib.file_digest((EVIDENCE/"manifest.json").open("rb"),"sha256").hexdigest()
    (EVIDENCE/"README.md").write_text(f"""# 20/10输入与标签拆解轻量证据

2026-11-18，Asia/Shanghai。主入口：[结果](../../research_20261118_bce_input_label_results.md)。

共{len(entries)}份逐字节复制件，包含源码、42头损失和验证锁、84行指标、配对区间、因子指纹、分解及验收。详见[manifest](manifest.json)，SHA256：`{digest}`。原始42权重和完整逐周Parquet、原股票主干/向量/行情仍在本机原data/outputs；未核实异地备份，轻量目录不能独立完成全量重训。脚本按原项目相对路径运行，复制件供审阅，不直接在证据目录运行。未自动Git提交或推送。
""",encoding="utf8")
    print("Published",DOC,"evidence",len(entries),"manifest",digest)
    print("Label flips:",changes.to_dict())

if __name__=="__main__":main()
