# 周频选股模型的架构与辅助监督研究

2026-11-11。研究对照采用 `abl2000_base`。每股输入 40 周 × 30 特征，`Linear(30,128) + LayerNorm` 后进入四层四头因果时间注意力。每块已有 RoPE、Q/K RMSNorm、注意力输出 sigmoid 门、两条残差和 `128→384→128` GELU FFN。时间塔在各层更新历史 token，末层只取最新周，经过 embedding 投影与最终归一化后由 `Linear(128,1)` 打分。基线的跨股票关系更新被零门关闭。 [冻结配置](../outputs/abl2000_base/config.yaml) · [时间块](../src/quant_mvp/model.py#L108) · [周投影与读出](../src/quant_mvp/model.py#L226) · [score](../src/quant_mvp/hierarchy.py#L245) · [关系层核对](../outputs/exp580/STATUS.md)

## 1. Block 和周 embedding

保留四层。想改变 block 内部时，先单测 **等参数 SwiGLU**，把 `128→384→128` GELU 换成两个 `128→256` 输入分支及 `256→128` 输出，线性权重均为 98,304 个，偏置略有差异。它检验特征乘法门在当前宽度下是否有用。历史两层对照在固定第 2–4 轮上约 −9.0 bp/周，零层约 −20.8 bp/周。四层模型采用跨层注意力残差的理由较弱，相关论文瞄准深层堆叠下早期层贡献稀释。 [本地减层对照](research_exec_20261110_report.md#3-训练臂m1-删-infoncem2-浅层对照) · [GLU Variants](https://arxiv.org/abs/2002.05202) · [Attention Residuals，2026](https://arxiv.org/abs/2603.15031)

周 embedding 若要改，先单测 `LN(Wx + W₂ GELU(W₁x))`，周内支路用 `30→64→128`，最后一层零初始化。它检验同一周价量特征的非线性交互。当前 `30→128` 线性层已经在扩维，周内 MLP 的价值需要通过样本外排序来证明。Chronos-2 对带时间索引与缺失掩码的 patch 做残差编码，随后交替做时间注意力与同一时间位置的组注意力。这里可先借鉴残差编码，维持一周一个 token。 [Chronos-2 §3](https://arxiv.org/html/2510.15821v1#S3)

周间卷积放在第二优先级。已有 5 周卷积加逐周线性残差，保留 40 个 token，三种子历史纯多年化净收益/夏普为 23.04%/0.896，基线为 27.31%/1.011。40→8 的均值块与可学习块也都未通过效率门。若重开局部混合，单测第一个 block 的 `k=3` 因果 depthwise 残差支路，保留每个周点，旧周与近端仍能全局交互。MambaVision 的可迁移之处是早层提取局部形态、晚层整合全局；CVPR 2026 的 UCAN 也以局部纹理和长程依赖分工，其收益来自图像超分任务。 [卷积融合结果](research_20261111_execution_plan.md#34-时间块注意力与近端保真) · [时间块验收](../outputs/research_20261111/acceptance.md#6-34-时间块注意力--两臂-stop净不可分辨无提速) · [MambaVision，CVPR 2025](https://arxiv.org/abs/2407.08083) · [UCAN，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_UCAN_Unified_Convolutional_Attention_Network_for_Expansive_Receptive_Fields_in_CVPR_2026_paper.html)

## 2. score 的 Linear 和 MLP

默认留 Linear。末端的一维线性头在非线性时间塔上读出分数，当前全模型已有足够多的非线性。值得单跑一个全截面、端到端的 `Linear(z) + MLP(z)`，支路 `128→32→1`、GELU、最后一层零初始化，继续用原排序损失。它允许同一个 embedding 在不同股票状态下产生非线性排序，同时从基线函数出发。旧 MLP 对已选成员做腿内权重，纯多两臂比等权约 −1.21 和 −0.66 bp/周；这个结果只约束腿内加权的优先级。候选要看分数头部排序、成分变动、换手和扣费净收益。 [旧 MLP 结果](score_mlp_results.md) · [当前打分实现](../src/quant_mvp/hierarchy.py#L339)

## 3. 下一周 embedding 辅助损失

先做点预测。学生用当前 40 周的关系层前 embedding，接一个小预测头；teacher 复用既有 EMA 模型，用其周投影对真实 `x[t+1]` 编码并停止梯度。目标仅在训练中使用，真实可观测、且在训练截止前成熟的下一周才计入。首臂固定既有 5 周 InfoNCE、训练锚点和末尾留窗。现有 InfoNCE 的目标是区分同股未来五周表示与其他股票，E2 的目标是下一周九维路径统计变化量。E2 在有意义的维度上没有明显超过零训练对照，并明显落后于持久性基准。 [现有教师与留窗](../src/quant_mvp/hierarchy_engine.py#L389) · [InfoNCE](../src/quant_mvp/hierarchy_engine.py#L524) · [E2 结果](../outputs/exp580/STATUS.md)

首版对预测和目标做 L2 归一化，直接最小化 `||normalize(pred) − normalize(target)||²`，按有效股周取均值。单位向量的平方距离等于 `2−2cos`。如果想保留变化幅度，另试逐维按训练期尺度标准化后的 Huber。用相同 teacher 的当前周 embedding 作为持久性基准，记录与目标的误差差、目标各维方差、加权辅助梯度相对主梯度的大小及余弦。即使 MSE 明显改善，也要单独验证头部排序与扣费净收益。

训练时序首选第 1 轮先按 baseline 跑，**第 2 轮内**从零逐步拉起辅助权重。可在固定训练 batch 上以共享时间塔的加权辅助梯度约为主梯度的 5%–10% 选首个剂量。这是首筛起点，现有证据没有确定的最优权重或开关时点。基线三个种子的最优轮已有 2、3、6 的差异，等到训练中后段会让部分 seed 的常用 checkpoint 缺少有效干预。只有首筛出现经济增益，才补同剂量从第 1 轮启用的时序对照。 [基线选轮与模型口径](research_20261111_execution_plan.md#1-术语与对照) · [梯度探针先例](research_exec_20261110_report.md#4-e0-计时m1-梯度探针m4-层诊断)

LeWorldModel 用下一步 embedding MSE 加 SIGReg 防止塌缩，HEPA 用未来窗口 latent 的 L1 加 SIGReg，V-JEPA 2.1 在视觉任务中扩展到可见/遮蔽位置与多层监督。ICML 2026 的美股 JEPA 论文在市场级别发现学到的表示与波动、相关性结构关联较强，与方向关联较弱。这些工作支持做可控的辅助预测实验，也给出了金融迁移中可能主要学到风险的原因。 [LeWorldModel](https://arxiv.org/html/2603.19312v3) · [HEPA](https://arxiv.org/html/2605.11130v1) · [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) · [美股 JEPA](https://proceedings.mlr.press/v306/mahns26a.html)

2026 年 10 月的视觉预印本 VANE 用消融显示，目标编码器和预测粒度都能改变下游表现。若本地首臂有收益，下一步固定同一个 5 周 teacher，把 InfoNCE 和 MSE 的对照补齐，再比较 1 周和 5 周。这样能追查收益来自点预测目标还是时间跨度。 [VANE §4.4，表 5](https://arxiv.org/html/2610.30724v1#S4.SS4)

Diffusion 和 flow matching 留给需要采样多种下一周情景的任务。直接 embedding MSE 给点预测。Flow matching 的 MSE 拟合带时间条件的速度场，生成时再积分。2026 年 10 月的 MercerFlow 给出了在通用时间序列上使用轻量 latent flow 的例子。若只需几个风险分位数，可以先用分位数头做成本更低的基准。 [Flow Matching](https://arxiv.org/abs/2210.02747) · [MercerFlow](https://arxiv.org/abs/2610.06039) · [Chronos-2 分位数头](https://arxiv.org/html/2510.15821v1#S3.SS2)

## 4. 回看长度与注意力读出

优先跑 `20 / 40 / 80` 周的单因素对照。短窗可能减少陈旧状态干扰，长窗可能帮助识别慢周期。`ret60/mom60/vol60` 已提供长视野，改变的是模型能读取多少个周点。当前 `min_history_weeks=32` 会直接排除所有 20 周候选。先在共同锚点、共同可用股票、共同监督窗口上比较，再报告自然覆盖口径下的新股和缺失历史覆盖。对比每轮耗时和峰值显存，已有压缩实验说明 token 数与端到端速度未必同步。 [覆盖条件](../src/quant_mvp/hierarchy_data.py#L117) · [冻结配置](../outputs/abl2000_base/config.yaml) · [时间块耗时](../outputs/research_20261111/acceptance.md#6-34-时间块注意力--两臂-stop净不可分辨无提速)

值得跑 **只有最新周更新** 的时间塔。固定 40 周，四层中让当前股票的最新 token 做唯一 Q，每层从同一份最初的逐周投影取 K/V（包含最新周原始投影），保留时间位置和最新 token 的残差。历史投影可以继续端到端训练，前向中历史 token 不随层更新。这个实验直接检验旧周之间的上下文混合是否有价值。若纯读原始历史有损失，再试前两层更新全序列、后两层只更新最新 token。视觉里的 TAR-ViTPose 在聚合视频历史后增强当前帧表示，更接近后一种折中。TQNet 从原始输入构造 K/V，用学习得到的 query 做跨变量交互，提供了另一个非对称读出的先例。LLM 的因果性与历史 K/V 是否经过各层变换是两个选择，现有 B2 双向消融只切换因果掩码。 [TAR-ViTPose，CVPR 2026](https://arxiv.org/abs/2603.05929) · [TQNet，ICML 2025](https://proceedings.mlr.press/v267/lin25e.html) · [B2 三种子](experiment_ledger.md)

实现时可先做一个等价性检查。只把最后一个 block 改成单 Q，让 K/V 来自原本倒数第二层的完整序列并保留最新 token 自身，eval 下输出应和基线一致。Q 的 RoPE 使用真实锚点位置，单 Q 的允许范围应显式覆盖全部已知输入。每层都读初始历史是另一项实验。初筛让 5 周 teacher 沿用全序列路径，避免同时改变 InfoNCE 的目标形状。 [原注意力实现](../src/quant_mvp/model.py#L147) · [教师分支](../src/quant_mvp/hierarchy_engine.py#L524)

跨股是另一个变量。暂缓让每只股票的当前 Q 同时读全市场 580×40 个历史 token，单个决策周的注意力矩阵已有约 1,346 万个 QK 配对/头。已跑的同周跨股注意力在门实际打开时，三种子仍为约 −4.7 bp/周（t=−1.10）。若重新验证横截面信息，可借鉴 Chronos-2 的分组，把同周行业或市场汇总 token 接在独立时间塔之后，并使用当时可知的分组和成分。 [本地跨股对照](experiment_ledger.md#205-m2b-打开门kill--收束) · [Chronos-2 组注意力](https://arxiv.org/html/2510.15821v1#S3.SS2)

## 5. 提前冻结输入 Linear

现有三种子实验在 **第 1 轮结束后**冻结输入 `Linear`，LayerNorm 仍训练。固定 Top-40/Bottom-10，80% 多头的口径下，验证配对 −3.62 bp/周（8 周区块 95% CI [−8.46, −0.47]），已读历史 +1.98（[−1.29, +5.76]）。50% 多头下分别为 −9.38 和 +0.72。该回放每个 seed 用自己选出的单个 `best.pt`，与固定轮次的架构对照口径不同。组合配置头的投影冻结曾给出偏正结果，那个投影属于另一个模块。 [股票输入冻结明细](../outputs/research_stock_freeze_comparison/report.json) · [回放脚本](../scripts/research_stock_freeze_compare.py) · [组合头冻结结果](attention_full_validation_results.md#projection-freeze-timing)

若按同 seed 的 baseline 选轮 `E*` 定时，建议把“提前一轮”定义为完成 `E*−1` 后冻结，从 `E*` 起冻结。例子是 best=3，前两轮照常跑，第 3 轮起冻结。三个基线 `E*` 是 3/2/6，这一臂对应冻结点 2/1/5；现有固定 E1 的实验仅在 seed 43 与该规则一致。如果“提前一轮”指让 `E*−1` 整轮也冻结，best=3 对应从 E2 起，seed 42 已覆盖该时点。固定 E1 的历史结果不足以支持作为默认。 [seed 42](../outputs/abl2000_base/metrics.json) · [seed 43](../outputs/abl2000_base_s43/metrics.json) · [seed 44](../outputs/abl2000_base_s44/metrics.json) · [现有冻结条件](../src/quant_mvp/hierarchy_engine.py#L455)

若继续，冻结规则由基线验证窗确定并在候选开跑前锁定。主对照取同 seed 的 `E*` 及后续固定轮，因为统一 `[2,3,4]` 会让 `E*=6` 的候选在全部对比轮都尚未冻结。检查冻结前路径相同，冻结后输入权重和偏置不再更新。输入 Linear 只有 3,968 个参数，冻结后的收益变化也可能来自优化路径与后层适应，单凭全局验证曲线无法定位浅层过拟合。 [冻结配置与训练](../scripts/research_stock_freeze_runs.py) · [输入投影](../src/quant_mvp/model.py#L241)

LayerLock 在视频掩码重建中观察到浅层比深层更早收敛，并据此设计渐进冻结。本项目可借鉴其逐层诊断。若实现按轮冻结，续训入口也要根据当前轮恢复冻结状态。EMA teacher 仍按既有规则更新，冻结后的教师权重会继续趋近固定的学生权重。 [LayerLock，ICCV 2025](https://arxiv.org/abs/2509.10156) · [EMA 更新](../src/quant_mvp/hierarchy_engine.py#L582)

## 首筛顺序

先比较回看长度，再做最新周单 Q 和残差 score。三个问题分别改历史长度、历史更新规则和最终读出。下一周 MSE 放在后一批；周内残差 MLP 与等参数 SwiGLU 各自单跑。有候选改善同周配对扣费净收益，且头部梯度、换手和回撤可接受时，再扩种子 43/44。一般架构对照沿用固定第 2–4 轮并附原选轮法的敏感性，冻结时序用上文的干预后同轮对照。既有历史回放已多次被查阅，后续确认要用未参与选择的时段或前瞻记录。 [现有执行与晋级规则](research_20261111_execution_plan.md#4-排期预算和晋级)
