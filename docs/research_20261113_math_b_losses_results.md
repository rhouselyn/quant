# A股数学建模研究 B：损失函数与交易目标

日期：2026-11-13，Asia/Shanghai。状态：complete（研究、源码审计与合成数值检验；没有训练和收益实验）。
父研究：[共同事实底稿](research_20261113_math_facts.md)。协议边界：[三年验证](research_20261113_standard_execution.md)、[研究经验](research_lessons.md)。

## 结论

**保留当前四项损失作为基线，优先验证一个单因素候选：只把 RankIC 的标签侧软秩替换成精确平均秩。暂不增加 Sharpe、风险、换手或梯度手术损失。** 当前损失是“尾部分类 + 平滑排序代理 + 时序表征正则 + 分数行业均值正则”的组合，不能解释为条件预期收益、净效用或完整概率模型的极大似然。

最明确的理论差异有三个：第一，所谓 RankIC 对预测分数和收益均使用绝对温度 sigmoid 软秩，因而不是精确 Spearman，对正比例尺度变换也不严格不变；第二，尾部分类按股票样本数归约，另三项主要按日期归约，日期的相对影响不同；第三，分数行业 η² 小不保证 Top20 持仓行业中性。这些差异可由代码和反例证明，但其收益影响仍需配对实验。

历史证据不支持“辅助项越少越好”：旧 canonical 批次删除 InfoNCE、去掉行业惩罚、修改温度等已有负面或不确定结果。它们不证明永远无效，但本轮没有足以重复扩大这些队列的新证据。

## 1. 真实对象、实现和版本

主对象：`outputs/dataset_v2/standard_v1/catalog.json` 引用的 `research_B0_F0_s42_a1`、CS、CB、CB_SELF、ADN、ADN_SELF。逐一读六臂 `outputs/dataset_v2/exploratory_F0/runs/<run>/config.yaml`，其训练目标配置一致：

| 项目 | 实际值及入口 |
| --- | --- |
| 尾部分类 | `score_weight=1`、`extreme_frac=.2`、`score_temperature=1`；`hierarchy_engine.py:595–649` |
| 软秩相关 | `rankic_weight=.5`、`rankic_temperature=.05`；`model.py:814–915` |
| 未来表征 | `temporal_weight=.1`、`temporal_horizon=5`、`temperature=.1`、`ema_decay=.99`；`hierarchy_engine.py:654–665,756–759` |
| 行业暴露 | `industry_exposure_weight=1`；`hierarchy_engine.py:685–695`、`hierarchy.py:529–546` |
| 其他目标 | 六臂未启用 cutoff pairwise、连续尾权、平滑标签、换手控制、行业辅助头或第二周头 |

`raw_return_ranking`、`contrastive_negatives`、`train_tradable_mask` 在这六份配置均未指定，因此分别采用“截断收益排序”“全横截面负例”“监督不额外与 tradable 相交”的默认路径。后者仍要求输入可用且标签有效，并不意味着所有缺失或不可用股票都参与监督。

监督标签是入场到退出的 **log 收益**，见 `scripts/build_dataset_snapshot.py:91–97`、`hierarchy_data.py:181–198`。训练中默认截断到 ±0.25；收益回放保留原标签并转换为简单收益。旧 run 中 F0 的 2020 选模字段不代表当前标准选择；当前 lock 是 2018–2020 的153成熟验证周，2021+ 为已读历史复检。本报告没有根据历史测试重新选择候选。

HEAD：`f2e198e8769ab960b7c8d6a8aa2ba50eb1bb8fe3`，工作树已有修改。读取源码 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `src/quant_mvp/model.py` | `630e0ecb67f0db1ae926f82eb8465df578ffcfb508189f5d734988fe3223276a` |
| `src/quant_mvp/hierarchy.py` | `98b5dae2f295b57cec5568ba13b05cfb9798fa0a5791bac6501df09e2e955850` |
| `src/quant_mvp/hierarchy_engine.py` | `b92e39aa3028a3bbce5c64b13f0fb77e3efb428613189d08ac6d843ea515764b` |

以 Python AST 忽略位置元数据，核对当前 `extreme_score_loss`、`_soft_rank`、`rank_ic_loss`、`masked_infonce`、`industry_exposure_loss`，与 standard_v1 两个冻结源 `source/0c62ea40178ac71a`、`source/ea0a4fbcf9c75c65` 的对应函数全部相同。这只证明五个函数的结构一致，不代替完整训练源码或所有 checkpoint 复用审计。

## 2. 四项损失的准确数学含义

令 t 为周决策锚，i 为股票，s_it 为输出分数，r_it 为原始未来 log 收益，y_it=clip(r_it,−.25,.25)。A_t 为输入可用股票集合，V_t 为 A_t 与有效标签的交集。B 为一个随机日期 minibatch。不同集合只用于对应损失，不修改推理候选。

### 2.1 尾部分类：分类概率比，不是收益预测

在 V_t 内按 y 排序，k_t=max(1,round(.2|V_t|))；头尾 k_t 分别赋 a_it=+1、−1，中间为0。无效标签为0。令 T_t={i:a_it≠0}，实际默认目标是：

\[
L_{tail}=\frac{\sum_{t\in B}\sum_{i\in T_t}\log(1+\exp(-a_{it}s_{it}/\tau_s))}{\sum_{t\in B}|T_t|},\qquad \tau_s=1.
\]

这是逐股票二元 logistic，和 [PyTorch BCEWithLogitsLoss 的定义](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BCEWithLogitsLoss.html)等价重编码；不是用分数差的 RankNet。对一个有效尾样本，导数是 `−a σ(−a s/τ_s)/(τ_s × 尾样本总数)`。在只有该项、充分容量及合适采样的理想条件下，其点态最优分数是“被选尾部条件下，上尾与下尾概率之比”的对数乘温度；联合其他损失后不能声称分数概率已校准。

头尾各20%对应的是相对强弱，弱市的上尾也可能亏损；它不预测正收益概率。580股全可用时约116个头部样本，而交易只取Top20，尾部内部细排序主要依赖 RankIC。clip 会把极端收益压成并列值；`extreme_tail_labels` 的 argsort 未指定稳定排序或平均处理，处于尾部截点的并列样本可能被任意分配标签。是否经常发生应在训练集计数，不能仅凭理论可能性宣布当前收益失真。

按全 minibatch 尾样本平均，相当于日期 t 权重正比 |T_t|。它可以理解为“均匀抽股票周”的经验风险，而非“均匀抽交易周”。这未必错误，但应与后续每周收益平均区分。

### 2.2 RankIC：可微平滑代理，对单位和尺度敏感

对有效股票定义

\[
R_i(x;\tau)=\sum_{j\in V_t}\sigma((x_i-x_j)/\tau),\quad
L_{rank}=1-\frac1{|B_*|}\sum_{t\in B_*}\operatorname{Corr}(R(s_t;.05),R(y_t;.05)).
\]

B_* 只包含至少3个有效股票、两侧软秩方差超过阈值的日期；标签侧停止梯度。有效性不足的日期不进入平均；全 batch 无合格日期时返回1，梯度为零，不能误读成“每个无效日期贡献一个有效观测”。实现在相关分母使用 `clamp_min(1e-8)`，极小方差下是数值正则后的相关。

方向正确：大 x_i 对应更大软秩；自身比较贡献0.5常数，不影响相关。对 j≠i，`∂R_i/∂x_i=Σ_j σ_ij(1−σ_ij)/τ`，`∂R_i/∂x_j=−σ_ij(1−σ_ij)/τ`。分数间距远大于温度会使 pairwise 梯度饱和，间距远小时近线性。这是优化尺度问题，不是排序方向反了。

精确 Spearman 对严格单调变换不变，有限温度 R 没有这个性质。同一 `.05` 同时作用于模型任意单位分数和 log 收益，因此包含收益差距大小的信息。当 |x_i−x_j|≪τ 时，R_i 近似 N/2+N(x_i−均值)/(4τ)，目标更接近线性相关；温度趋小且无并列时才逼近普通秩。可微排序的必要性和代理选择见 [Blondel 等原论文](https://proceedings.mlr.press/v119/blondel20a.html)；项目使用的是二次复杂度 sigmoid 两两比较，不是该文的投影算法。这里没有规模证据要求替换库。

### 2.3 InfoNCE：识别同股未来表示，不等于预测超额收益

令 C_t 为当前可用且未来连续5个观测周均可用的股票；q_i 为在线历史表示经过独立 projector 后的单位向量，k_i 为 EMA 教师编码未来5周输入后的单位向量。教师停止梯度。设 K=|C_t|、M_ij=q_iᵀk_j/.1：

\[
L_{NCE,t}=-\frac1{2K}\sum_i\left[\log\frac{e^{M_{ii}}}{\sum_j e^{M_{ij}}}+\log\frac{e^{M_{ii}}}{\sum_j e^{M_{ji}}}\right].
\]

取 K≥2 的日期平均，均无可用日期时为图连接的零。每次优化后教师参数更新 `θ_bar←.99 θ_bar+.01 θ`。梯度经 q 回在线 encoder/projector；教师参数、未来输入无反传；该项不经过评分关系分支或 score 头，但共享时序 encoder。

这是 [CPC 原论文](https://arxiv.org/html/1807.03748v2)启发的预测性对比目标。原文互信息论证依赖负样本分布等条件；本项目同日股票具有行业和市场相关、并非直接验证的独立边际采样，不能把 `log K−loss` 宣称为已认证的收益信息量。损失可能奖励持久的个股身份、波动特征，也可能惩罚经济上相近的同业表示；是否有害需要消融。未来可用性 mask 只改变辅助监督样本，不进入推理候选，是监督设计；它可能偏向未来持续交易的样本，须报告覆盖而非称为未来输入泄漏。

### 2.4 行业 η²：分数均值正则，不是组合风险约束

在 A_t（不要求未来标签）内，按行业 g 分组，n_g 为股票数，均值记为 μ_g，全体均值 μ，标准差 σ。忽略实现的 ε，有：

\[
L_{ind,t}=\eta_t^2=\frac{\sum_g n_g(\mu_g-\mu)^2}{\sum_i(s_i-\mu)^2}.
\]

实现等价于先 z 标准化分数再累计各组均值平方；至少3个有效值且分数标准差足够才纳入日期平均。它对统一平移和非零缩放近似不变（ε使近零情况不严格）；惩罚行业间均值差，但不约束行业内方差、尾部厚度、Top20入选数、市场 beta 或组合协方差。

在固定分数随机分配行业且非退化时，有限样本零假设期望为 `(G−1)/(N−1)`，与代码 `industry_exposure_baseline` 的表达代数等价；G 为非空行业数，N为有效股票数。把η²压到零不等于消除了有害风险，也可能抹去有效行业信号。

## 3. 已完成的轻量反例

使用本机 Python/PyTorch CPU，仅导入损失函数并传入合成张量；无 checkpoint、市场数据、优化器或训练调用，断言通过、进程退出码0。

1. 取 s=(0,.1,.2,.3)，r=(0,.01,.02,.2)，温度.05。`1−rank_ic_loss(s,r)` 为 **0.8457753102**；标签改成0.1r，秩序不变且仍在clip范围内，值变为 **0.8215910775**。两者真实 Spearman 均为1。这证明标签侧平滑不是尺度不变的秩目标；不是市场有效性检验。
2. 行业A分数(−3,−2,5)，行业B分数(−1,0,1)，两组均值均为0，实际 `industry_exposure_loss` 返回 **0**，Top1却100%属于A。按相同思路扩大尾部可构造Top20反例。这证明η²不保证持仓中性。

复现第一项：调用 `rank_ic_loss(torch.tensor([[0.,.1,.2,.3]],dtype=torch.float64), torch.tensor([[0.,.01,.02,.2]],dtype=torch.float64)*a, temperature=.05)`，分别设置a=1和.1。第二项调用 `industry_exposure_loss(torch.tensor([[-3.,-2.,5.,-1.,0.,1.]]), torch.tensor([[0,0,0,1,1,1]]), torch.ones((1,6),dtype=torch.bool))`。这些内存结果作为本页数值记录，没有单独机器产物或备份。

## 4. 与净交易目标的距离

实际净收益由 `R_net,t=w_tᵀR_t−C(Δw_t)−borrow_t` 决定；R_t为简单收益，w_t来自分数排序和执行约束，Δw_t应相对漂移后的旧持仓计算。四项训练损失均不直接估计该函数；选轮净Sharpe只是后续选择算法，不会使训练梯度自动变成净收益梯度。

因此先在组合/回放层处理已知的真实费用、仓位与成交约束。若把换手惩罚放入训练，必须模拟同一持仓递推、无券、缺失和受阻成交；仅比较相邻周 softmax 分数与真实Top20账本并不等价。固定容量的截断单纯形投影也一般不是离散TopK等权：靠近阈值时允许更多持仓带部分权重。相关 `weekly_stickiness_loss` 未在六臂启用，本轮不据其注释承认“完全等于交易账本”。

对联合目标，应看共享 encoder 的 `||λ_k∇L_k||` 和两项梯度余弦，而非比较 loss 标量。它们还能与全局梯度裁剪阈值1互动（`hierarchy_engine.py:752–759`）。[PCGrad 原论文](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)研究梯度冲突，但并不证明本项目需要梯度手术；负余弦也可能是合理的任务权衡。先用冻结训练期样本测量，只有持续有害冲突再提出新候选。

## 5. 最小候选与反证

### 候选 R1：只替换标签侧为精确平均秩

保持 score soft-rank 温度.05、标签clip、尾部定义、损失系数、mask、日期归约、架构、训练预算及选轮规则不变。只把标签侧 R(y;.05) 替为：

\[
H_i(y)=\tfrac12+\sum_{j\in V_t}[\mathbf1(y_j<y_i)+\tfrac12\mathbf1(y_j=y_i)],\quad
L_{rank}^{R1}=1-\operatorname{mean}_t\operatorname{Corr}(R(s_t;.05),H(y_t)).
\]

H 是从1起算、并列取平均的升序秩；标签无需反传，硬排序不阻碍分数梯度。此次仍保留clip造成的并列，避免同时改变收益截断。预测侧有限温度仍依赖分数尺度，因此不得宣称R1已经具备完整 Spearman 不变性。标准化score或去掉clip是另外的候选，本轮不捆绑。

**假设**：当前标签侧平滑无意加入了收益尺度依赖，精确目标秩能提高排序目标的含义清晰度，并在同一策略中保留或改善净收益。反面可能是平滑差距提供了有用经济强度，精确秩放大噪声，损失梯度更差，收益下降。

**技术通过**：单调变换不改变标签平均秩（变换须保留clip后并列结构）；并列、无效样本、全常数、少于3股、日期隔离、有限梯度都有定向检查；旧默认分支数值不变；同样本下用真Spearman与软指标分开记录。

**经济通过/否决**：只与同cohort B0比较，配对42/43/44、每臂8轮，冻结主L20净收益、10bp及30bp压力、选轮与多重比较规则，按综合协议统一的最小经济增益与时间块区间判定；净指标不能只靠IC提升代替。区间不支持实质提升则维持B0，历史测试仅描述。这里不单独发明另一套经济门槛，也不自动启动这6个训练run。

### 保持现状与暂缓的第二方向

当前优先保留B0；R1为唯一损失训练候选。逐周等权的尾部BCE是合理的第二方向，但先只做训练集尾样本数分布与梯度贡献诊断。若各周 |T_t| 差异足够小，它近似现状；若差异显著且主目标是等周风险，再登记 `mean_t mean_i softplus(...)` 的独立候选。这与 [数据实现方案的日期归一约定](research_20261113_dataset_implementation.md)衔接，不新增一套时间加权体系。

历史边界：[台账](experiment_ledger.md)§26记录旧canonical删除score、行业惩罚和把温度改.20等未得到稳健优势；§26.12记录旧s42固定[2,3,4]对照删除InfoNCE，配对−4.5bp/周、t=−.59、梯度余弦约−.016。此处是**转述既有文档，未重算**；不能外推到standard_v1，也不能称已证明删除永远失败。它们使“仅凭理论怀疑就重跑删项”优先级低于一个新且明确的目标检验。旧四合一 stratified cutoff 失败同样不能单独否决其中每项，见台账223–273行的组合臂记录。

## 6. 未来实现影响与验收范围

本轮仅新增本报告。若R1获后续执行授权，复用 `model.py:856` 的公共函数并保留默认行为，显式传入标签秩模式，避免静默改变所有调用方。

已检索影响面：`hierarchy_engine.py` 主RankIC及第二周/行业辅助头；`engine.py` 原PairRanker训练；`scripts/a2_dose_precheck.py`、`followup_day8.py`、`followup_dynamic_day8.py`；`tests/test_stock_weights.py`；`__init__.py`已有函数导出可保持。目标模式只在登记的主stock ranking开启，辅助头和历史默认不变。配置、metadata、训练身份和完整恢复校验应绑定该模式，旧checkpoint可以照旧推理，但不应以不同损失冒充同一训练续跑。

既有测试证据是源码阅读：`tests/test_hierarchy.py`已有InfoNCE mask及η²范围/梯度/置换零假设，`tests/test_stock_weights.py`已有加权归约检查。本轮未运行测试套件，不把这些测试的存在当成当前通过。

未做：全训练期clip比例/并列截点/尾样本日期权重计数，真实checkpoint的分项梯度诊断，候选训练与配对回放，全部历史原始产物重算。风险：未来可用监督偏倚、当前行业回溯分类和执行代理仍可能主导结果；更符合排序数学定义不能保证未来收益提升。
