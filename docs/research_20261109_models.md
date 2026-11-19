# 模型与计算预算研究：先识别有效机制，再迁移新架构

研究日期：2026-11-09。范围是当前工作区代码阅读、历史实验核对与原始论文检索；没有训练、回测、改动模型、配置或 baseline。文献检索截至当日，并非穷尽全部最新工作。下文的收益机制与时间估算均是待验证假设，论文在语言模型或通用时序数据上的结果不构成 A 股收益证据。

**建议优先次序：删除 InfoNCE 的干净对照 → 零层锚点浅模型 → 5 周 patch → 有前置证据才做 AttnRes；空头目标池对齐另受 PIT 数据质量约束。** 五项是候选研究路线，不应在一夜全部铺开。过去的 KILL 是特定数据、种子、训练和选模规则下的结果；不能把它升级为架构无效定理，也不能忽略它已提供的负向证据。

## 1. 当前代码到底在做什么

| 事实 | 对研究设计的影响 | 代码证据 |
|---|---|---|
| 30 个价量/日径特征，40 周输入；d=128、4 层、4 头、8 epochs | 这是短窗口、小模型问题；许多长上下文优化针对的瓶颈不存在 | [baseline 配置](C:/Users/14815/Projects/quant/configs/hierarchical_580_monopen_baseline.yaml:17) |
| 每个股票独立做时间注意力，最后一周读出 | 不能把横截面 580 股误算成 temporal attention 的序列长度 580 | [encode](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy.py:248)、[最后位置读出](C:/Users/14815/Projects/quant/src/quant_mvp/model.py:248) |
| 已有 Q/K RMSNorm、RoPE、SDPA 后 sigmoid 门 | 2025 gated attention 的核心机制已经在 baseline 内，不是可再次添加的新方向 | [注意力实现](C:/Users/14815/Projects/quant/src/quant_mvp/model.py:159) |
| L-G-L-G-L 门被冻结为 0，仍运行 attention/FFN | 有效股的关系更新为恒等，市场 token 不影响其打分；“模型已经利用行业关系”是不准确的。最终 LayerNorm 和归一化仍生效 | [冻结](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy_engine.py:359)、[门控更新](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy.py:21)、[关系层与输出](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy.py:315) |
| InfoNCE 用同一股票未来 5 周窗口的 EMA 编码，负样本为同日其他股票 | 学会可预测的股票身份/风险风格，不必然学会下周可交易收益 | [教师与训练期保留](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy_engine.py:389)、[InfoNCE 分支](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy_engine.py:491) |
| 损失同时含头尾二分类、全截面软 RankIC、InfoNCE、行业暴露惩罚 | “训练 loss 下降”不能定位哪条监督产生了经济价值 | [总目标](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy_engine.py:485) |
| `score_top_weight` / `score_bottom_weight` 在 loss 函数有参数，但当前 hierarchical 调用没有传它们 | 若将多空权重消融只写成 YAML 改键，可能得到静默无效实验；当前 1/1 baseline 不受此问题影响 | [函数参数](C:/Users/14815/Projects/quant/src/quant_mvp/model.py:596)、[实际调用](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy_engine.py:485) |

2025 原始 gated-attention 论文支持“在 SDPA 后增加输入相关 sigmoid 输出门”这一机制的研究价值；它的语言模型规模、任务及数据与本项目不同。本仓库是对机制的借鉴，并非 Qwen3-Next 完整架构复现。[Gated Attention 原论文](https://arxiv.org/abs/2505.06708)

### 1.1 计算成本的数量级

只计单层 temporal block 的主要矩阵乘，忽略 bias、norm、激活、softmax 与硬件实现：

\[
\text{MAC}_{\rm block}\approx 5Ld^2+2mLd^2+2L^2d,
\quad L=40,d=128,m=3.
\]

前两项分别为 Q/K/V/gate/output 五个投影与两层 FFN。代入得到投影/FFN 约 721 万 MAC、注意力两次乘法约 41 万 MAC；后者仅占这部分约 **5.4%**。这是依据实际代码推导的算术量，不是实测耗时。batch=16、580 股意味着每步最多处理 9,280 个股票窗口；此外还有未来教师前向、`[B,N,N]` 软排序/InfoNCE、冻结关系层的无效计算和 checkpoint 重算。[投影/FFN](C:/Users/14815/Projects/quant/src/quant_mvp/model.py:133)、[软排序](C:/Users/14815/Projects/quant/src/quant_mvp/model.py:623)、[整塔 checkpoint](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy.py:248)

因此，Kimi Linear 的百万 token 解码吞吐、KV cache 节省不能直接转化为这里的速度收益。其核心 KDA 是有限状态的门控 delta 更新，与现有 softmax attention 有不同归纳偏置，但移植自定义 kernel/混合架构的工程成本高于本项目先做简单对照的成本。[Kimi Linear 原论文](https://arxiv.org/abs/2510.26692)

冻结关系层可成为**单独的工程等价优化**候选：先证明 eval 输出与梯度等价，再测时间。不能把跳过 dropout 后训练随机数序列变化造成的结果差异称为 alpha；也不能同时将它混入架构收益实验。本次不实施。代码内 encoder block 另有 checkpoint 开关，但在已读调用路径中未见其被设为 true，故不能断言当前存在双重 checkpoint。

## 2. 历史证据怎样约束，而不封死研究

| 已有研究 | 能支持的结论 | 不能支持的结论 |
|---|---|---|
| 特征轴 B1：`Linear(40→d)` token + CLS，排序 loss 几乎不动；教师 5 周被池化到 40 周 | 当前 tokenizer、读出和辅助任务组合失败 | 所有 iTransformer/特征轴方法均无效；“线性投影必然丢失全部路径信息”也过强——其参数与维度允许保留信息，真正原因仍待拆分 |
| 非因果 B2：三 seed IC 变差，纯多账本差异很小 | 此数据与配方下无支持替换的证据 | 所有窗口内非因果模型无效；它本身不构成未来信息泄漏 |
| 市场 FiLM、行业相对特征、末端跨股 attention | 已有直接负向或近零证据；重复同构实验优先级低 | 跨时间、跨股票的联合交互已经被完整验证 |
| 尾部四合一臂 | 捆绑方案失败，主要损失来自空头 | raw rank、tradable mask、分层尾部、cutoff pairwise 每项单独无效 |
| 11 月 9 日训练臂均为单 seed VAL-ONLY | 目前没有可靠收益改进；某些表面提升由选轮窗口解释 | 因一个特定 t 门未过便证明效应为零；不能外推到所有年代/多 seed |

对应仓库证据：[B1/B2](C:/Users/14815/Projects/quant/docs/experiment_ledger.md:347)、[关系层消融](C:/Users/14815/Projects/quant/docs/experiment_ledger.md:642)、[四合一的可归因边界](C:/Users/14815/Projects/quant/docs/experiment_ledger.md:221)、[近期训练及窗口解释](C:/Users/14815/Projects/quant/docs/overnight_20261109_report.md:125)。

## 3. R1：删除未来身份任务，测量它的净贡献

**唯一变化：`temporal_weight: 0.1 → 0`。** 保持 `common_training_horizon: 5`、训练日期、评分目标、行业惩罚、编码器、8 epochs、交易/融合规则全部一致。代码已经支持教师关闭，通常不需修改模型代码。[教师构造](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy_engine.py:389)、[统一 reserve](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy_engine.py:403)

**假设与数学理由。** 当前目标可写成 `L = L_return + 0.1 L_identity`。如果共享编码器上的梯度满足 `g_return · g_identity < 0`，辅助任务会一阶干扰收益监督；即使内积为正，它仍可能主要强化股票稳定身份、流动性或波动风格，而非条件超额收益。InfoNCE 的原始动机是预测未来潜变量，不保证该潜变量对特定交易目标充分。[CPC/InfoNCE 原论文](https://arxiv.org/abs/1807.03748)

**先看什么。** 若后续获准执行，选固定训练期 batch，测 `cos(g_return,g_identity)`、两个加权梯度范数、InfoNCE 正负样本相似度、流动性/波动分桶表征可分性；无需引入 PCGrad 等新算法。不能单凭“两个 loss 数值量级差大”判定梯度失衡。

**关键对照。** 主比较为完整原始 baseline 与关闭 InfoNCE；只关闭教师时仍保留最后 5 周 reserve，避免偷偷多用训练样本。固定 epoch 的分数差与既定选轮规则分别报告；若研究额外的 λ 剂量，只预定一个中间值 0.03，不能扫描后挑最优。

**替代解释与失败条件。** 关闭后改善可能来自训练噪声/正则强度变化，而非负梯度冲突；必须同时报跨 seed 方差、训练与验证拟合差距。若出现更快收敛但尾部净收益/风险在独立年代恶化，收益假设失败；若收益在预先约定的经济容忍带内、耗时显著下降，仍可作为工程性简化候选，不应包装为 alpha。

**预算与防泄漏。** 1 个完整 8-epoch 臂启动，不承诺速度倍率；省去教师前向与对比矩阵，但不会省掉主编码器反向。输入保持到 Friday；未来数据只存在于被删除的训练辅助目标。重要范围包括 `hierarchy_engine.train` 的教师生命周期、配置快照、训练日期断言、checkpoint 元数据与 `tests/test_hierarchy.py` 的无教师分支。当前检索未发现该干净单因素在新 canonical 上已有定稿结论，实施前仍应查 run 元数据避免重复。

## 4. R2：零层锚点浅模型——先问四层时间注意力是否值得

**唯一变化：`n_layers: 4 → 0`。** 现有 `AttentionEncoder` 的空 `ModuleList` 会自然跳过 attention；保留输入线性投影/LayerNorm、embedding、最终归一化和 score head。输出仍是最后位置。这是带归一化的锚点浅模型，**不是纯线性回归**；`ret60/mom60/vol60` 等已给最后一行编码了较长历史，因此也不是“只看一周”。[编码器定义](C:/Users/14815/Projects/quant/src/quant_mvp/model.py:225)、[逐层循环和读出](C:/Users/14815/Projects/quant/src/quant_mvp/model.py:238)

**假设。** 在有效独立时间样本远小于 `580×周数`、输入已经包含多尺度人工统计量的环境，四层网络可能主要增加估计方差。若浅模型保留了大部分可交易收益，下一步应投向数据或监督而不是堆结构。通用时序研究也有简单线性/MLP 胜过更复杂网络的实证，但那只能支持设置此对照，不能预测本仓库的胜负。[Google TSMixer 原始研究](https://research.google/pubs/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/)

**消融。** 首轮只有 0 层与 4 层；不同时换优化器、损失、lookback、宽度。若 0 层接近 4 层，预定的下一点为 1 层，以区分“完全无需学习时间交互”和“一层足够”。若 R1 也有线索，后续 `depth × InfoNCE` 2×2 能识别交互；不可直接把零层+去 InfoNCE 捆绑后归因为其中一项。零层教师只读取未来窗口末行，这是架构因子改变辅助任务方式的自然结果，需明确列为替代解释。

**验证与失效。** 固定同一交易日期，报告 top-20、可执行短腿、分数自相关、净成本、子时代和低流动性桶；若只有全截面 IC 保持而真实尾部受损，不能称为替代。若所有浅层 seed 在独立年代持续丢失主要毛收益，说明时间交互有价值；此时再研究 patch 或跨层路由才更有依据。简单模型训练更快也不能抵消不可接受的尾部损失。

**预算/接入。** 无新依赖、首臂仅配置变化；预计主时间塔运算显著减少，但冻结关系层与损失开销仍在，实际占比须 profile。最多以 1 个 baseline 训练时长作为初筛上限。未来实现最小检查为 `n_layers=0` 的 forward/backward、输出 shape、同一输入末行而历史不同的输出应一致；这证明实验确实禁用了可学习时间交互。推理 `live_score` 无需新增输出格式，部署仍须同样完成融合/执行链检查。

## 5. R3：5 周 patch，改变局部表示与计算分配

**唯一结构因子：将 40 个逐周 token 改成 8 个连续的 5 周 patch token。** 每个 patch 展平为 150 维后投影到 128 维，后续仍为 4 层、4 头 gated attention，最后一个 patch 结束于当前 Friday。每个 patch 内保留先后顺序；不改市场/行业路径、不增加日频支路。借鉴 PatchTST 的 patch 原理，但保留 30 特征联合投影，**不声称复现其 channel-independent 完整模型**。[PatchTST 原论文](https://arxiv.org/abs/2211.14730)

\[
u_j=W_p\operatorname{vec}(X_{5j:5j+5})+b_p,\qquad j=0,\ldots,7.
\]

**动机。** 周粒度 attention 必须自行组合局部路径；patch 先把五周形态压入一个表示，再做跨 patch 交互。它还把主要 FFN/投影 token 数降至 1/5，attention 图降至 1/25；这些是指定张量形状下的理论比例，不是端到端提速承诺。增加的输入投影权重约为 `(150−30)×128=15,360`，与完整模型相比较小；信息压缩可能损失刚发生的反转。

**为什么不是重跑 B1。** B1 把每个特征的完整 40 周一次投影为一个 token，并颠倒注意力轴；此臂维持时间轴，仅做局部分块，允许不同局部形态再交互。它也不同于已实现的 `ConvAttentionEncoder`：后者 stride=1 保持 40 个 token，既不减少 token 数，也不能单独检验压缩收益。[现有卷积 tokenizer](C:/Users/14815/Projects/quant/src/quant_mvp/model.py:258)

**必要控制。** 第一轮 baseline vs patch 即可筛查。若出现收益或明显效率线索，必须增补“每五周均值后用原投影”的无学习压缩对照，区分降噪/减少计算与局部路径表示。还要用 R1 为背景做 patch 的无 InfoNCE 对照：当前 5 周教师在 patch 下只剩 1 token，任务变简单可能解释结果。这个 2×2 是后续归因阶段，不在首轮一口气铺开。不能同时改变 position scaling、层数与注意力因果性。

**接入/检查。** 扩展 [TemporalTower 编码器选择](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy.py:130) 和 [AttentionEncoder 输入阶段](C:/Users/14815/Projects/quant/src/quant_mvp/model.py:238)，保持 `normalize`/`return_sequence` 接口、EMA 同结构和 metadata 的 backend 名称。40 周主窗与 5 周教师均可整除；其他长度若不支持应显式拒绝，不做隐式未来 padding。测试应覆盖最后 patch 对齐、patch 内排列改变会改变输出、禁止使用 t 之后输入、teacher 5 周输出有限且可反传到 online 分支。

**失败条件/预算。** 若优势只有教师 loss 下降，或仅在少数训练种子/epoch 窗口出现，无法支持局部表示假设；若 mean-pool 与 patch 等效，可选择更简单的 mean-pool。实现规模小于全新 SSM/kernel，但包含新编码器，初筛训练预算上限 1B（B 为同机 baseline 8 epoch 耗时）。显存与真实训练/推理耗时必须测；更快但尾部净收益持续恶化，只能记录效率—效果曲线，不能替换 baseline。

## 6. R4：2026 Attention Residuals，小规模且有条件地迁移

2026-03 的 AttnRes 将按深度固定累加的 residual 改成对既有表示的内容相关加权；Block AttnRes 在块内求和、块间路由，缓解存储开销。论文与官方伪代码均已核验。其证据来自语言模型，尤其远深于本项目四层的设定。[原论文](https://arxiv.org/abs/2603.15031)、[作者官方仓库](https://github.com/MoonshotAI/Attention-Residuals)

**先决问题。** 当前四层是否真存在深度稀释？先在已有 checkpoint 的训练期窗口上记录每层 hidden RMS、更新/主干范数比、输入和层输出的相关程度，以及浅层/深层梯度；若没有逐层幅度增长或有效特征被稀释的迹象，降低此路线优先级。注意旧深度实验并不直接验证 AttnRes，而四层也不值得仅为追新而改深。

**最小可识别方案。** 固定原来的四层 attention 和 FFN 计算，只改变 residual 聚合方式；使用 2 个 Transformer 层一块，按官方定义 block_size=4 个子层。对历史块表示与当前块部分和 `v_i`，用每个子层一个伪查询：

\[
\alpha_i=\operatorname{softmax}_i(w_l^T\operatorname{RMSNorm}(v_i)),
\quad h_l=\sum_i\alpha_i v_i.
\]

这是具体待实现的 Block AttnRes 配置，不需要增加 heads、宽度或跨股票 attention。当前 block 将 attention/FFN residual 封装在一个 `forward` 内，接入需要显式拆出子层输出后在 encoder 管理块状态；不能简单在最终 embedding 上加一个层平均就声称复现论文。[当前两个 residual](C:/Users/14815/Projects/quant/src/quant_mvp/model.py:159)、[encoder 循环](C:/Users/14815/Projects/quant/src/quant_mvp/model.py:238)

**替代解释及消融。** 收益可能只来自归一化尺度改变或浅层特征集成，而非动态路由。若首轮有信号，必须补 `α` 固定均匀的块混合对照，与相同保存张量/归一化路径相比；只有动态权重超过固定权重，才支持内容相关选择。报告各块权重熵、跨股票差异和年代变化；“权重很漂亮”不是收益证据。若权重近常数，保留简单固定混合或放弃机制，而非增加门/头继续搜索。

**成本与失效。** 新增计算相对 FFN 较小，但保存历史块会增加激活内存，checkpoint 的闭包/状态重算也更复杂；8GB 机器上不能承诺论文里的低开销。初筛设 1.25B 的硬件预算上限；达到预算或显存不够应停止并记录，不能减 batch 又改变其他优化口径而不声明。若无前置稀释证据、不能超过均匀混合、或收益只存在于同一个调参年代，则该机制在本项目缺乏支持。防泄漏与 baseline 一样：深度路由只能访问同一已知窗口的层表示。

## 7. R5：只对空头负标签的候选池做决策对齐

**为什么不是又缩一次 extreme_frac。** 当前训练负尾部来自全体可标注股票，但真实短腿先取 Friday 成交额 q4/q5，再过滤资格并顺延补位；这会使“把不可做空的极差股票排得更差”占用监督容量。11 月 8 日 extreme_frac=.10/.05 是全市场尾宽变化，11 月 9 日 G2 是另一种 decision-aligned loss；它们没有单独回答此处“训练负类目标池与可执行短腿是否错配”。[训练尾部标签](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy_engine.py:474)、[空头候选规则](C:/Users/14815/Projects/quant/scripts/skipfill_short.py:72)、[近期训练清单](C:/Users/14815/Projects/quant/docs/overnight_20261109_report.md:127)

**必须先满足的数据条件。** 定义 `E_t` 只使用 Friday 已知的成交额分位、已生效且可追溯的融券标的资格；有券商历史券源则进一步区分“有资格”和“实际可借”，没有则如实称为资格池。现有季度文件匹配方式不能单独证明 PIT，必须完成发布日期/生效日审计。不允许用周一开盘才知道的 `tradable[t]` 作为 Friday 目标池：它在当前代码中由 `open[t+1]` 和入场跳空构造。未来成交状态用于研究训练目标不是自动等于输入泄漏，但会改变条件分布，不能冒充时点可知 eligibility。[当前资格匹配](C:/Users/14815/Projects/quant/scripts/skipfill_short.py:58)、[事后 tradable](C:/Users/14815/Projects/quant/src/quant_mvp/hierarchy_data.py:158)

**唯一目标变化。** 上尾正类 `P_t` 仍用原全市场 top20%；负类由原全市场 bottom20% 改为 `E_t \ P_t` 内收益 bottom20%，以避免同一股票被同时标正负。保持 RankIC、InfoNCE、行业惩罚、单 score head 与所有执行规则不变。将 extreme score 写为类平衡均值：

\[
L_s=\tfrac12\operatorname{mean}_{i\in P_t}\operatorname{softplus}(-s_i)
 +\tfrac12\operatorname{mean}_{j\in B(E_t\setminus P_t)}\operatorname{softplus}(s_j).
\]

上式展示单个日期；实际应按原 batch 的样本归约方式实现，不能无意把原“按有效样本加权”改成“每个日期等权”。将各日期正负集合分别合并后，在 baseline 正负数相等时，两项类均值的平均与现有损失等价；实现前需逐 batch 证明等价，再替换负类集合。若可用负类太少，事前固定回退原负类的规则，记录触发次数；若多数训练周需要回退，目标池训练缺乏数据，应停止而非解释为“模型没学会”。

**替代解释与消融。** 收益可能仅来自减少负类样本、移除低流动性股票、或隐式放松了原负尾部任务。后续同样数量的原负类随机子集是样本量安慰剂，匹配流动性但不加资格是 pool 因素拆分。首轮只做主对照，不同时加第二个 head、pairwise cutoff 或成本惩罚。当前历史市场不一定有可靠资格序列；缺失阶段不能把最近季度表回填，最多在存在 PIT 记录的训练/评估子窗研究并同样重训 matched baseline。

**验证、失效和成本。** 主结果是同一执行器下短腿净收益、填满率、借券与成交成本敏感性，同时设多头损害容忍带；只改善全市场 bottom-tail IC 而实际短腿没有改善则假设失败。即使资格池内改善，真实券源缺失仍阻止宣称可实盘获利。损失计算开销接近 baseline，数据整理可能比训练更贵；首臂 1B，上线前另需独立纸面记录。具体影响面是 tail label helper、loss 类归一化、PIT eligibility loader、metadata 与资格/极少样本单测；`live_score`/书本输出格式应保持不变。

## 8. 验证协议与有限预算

### 8.1 不用固定 t 门机械裁决，也不把点估计当事实

每个候选预先写清：期望改进哪条经济通道、可接受的收益损害、成本/显存目标，以及何种结果推翻机制。报告配对周收益差及区块自助法区间、胜率、跨 seed 方向、独立年代、尾部风险与真实成本敏感性；t 值作为不确定性摘要，而非唯一生死线。区间宽且跨零可以保留低成本、机制清晰的候选继续纸面验证，但不得宣称已经证明提升。

同一 seed 的 matched baseline、完全一致的交易日期/可交易掩码、相同标签成熟边界是前置条件。训练期未来监督和标签只允许落在训练截止之前；模型选择窗口的标签须在研究决策时已经成熟。当前 2021–2026 历史 test 已多次参与研究，应称为重复使用的历史评估集，不能再次包装成全新 holdout。另设时间上向前的纸面观察，或预先保留的独立年代。

每臂同时报告各自选轮结果与预定公共 epoch 窗口的机制诊断；公共窗口是解释工具，不是看到结果后改主规则。边界 epoch 只有两成员和三成员融合的差异要统一。最近报告已展示这种差异足以解释看似不错的训练改进。[窗口混杂](C:/Users/14815/Projects/quant/docs/overnight_20261109_report.md:140)

### 8.2 8 小时预算示例：少数路线，留下复现余量

近期同机 8-epoch 臂约 2485–2512 秒，且受 CPU 分析竞争影响。[近期实测记录](C:/Users/14815/Projects/quant/docs/overnight_20261109_report.md:354) 本次只读汇总历史 `seconds` 列，baseline s42/s43/s44 的 8 epochs 分别为 837.12/874.99/836.80 秒，约 14 分钟：[s42](C:/Users/14815/Projects/quant/outputs/abl2000_base/training_metrics.csv:1)、[s43](C:/Users/14815/Projects/quant/outputs/abl2000_base_s43/training_metrics.csv:1)、[s44](C:/Users/14815/Projects/quant/outputs/abl2000_base_s44/training_metrics.csv:1)。因此排程按 **15–45 分钟/臂的历史观测区间**预估，开始前用短 profile 重估 B；下表用接近上沿的 B≈42 分钟安排余量，既不是下界也不是未来运行承诺。GPU 必须单训练进程串行；此前 8GB 4060 并发出现过非法访问。[并发记录](C:/Users/14815/Projects/quant/docs/experiment_ledger.md:394)

| 阶段 | 预算建议 | 产出/停止点 |
|---|---:|---|
| 数据时点、融合、执行会计前置核对；计时预热与机制诊断 | 45–75 分钟 | 有污染/对照不等价先修研究口径，不启动收益比较 |
| matched baseline s42（若已有产物严格等价可复用）+ 最多 3 个 s42 候选 | 约 4B，预留到 3.5 小时 | 首选 R1、R2，再从 R3/R4/R5 选一个；不是五项全做 |
| 最多一个有价值方向的 s43/s44，与相同 seed baseline | 最多 4B，约 2.8 小时 | baseline 可严格复用时节省时间；不能拿 1 seed 新臂比较 3 seed 旧臂 |
| 融合、配对、分腿、成本与失败分析 | 30–45 分钟 | 没有机制/经济价值线索则停止，不为用完 GPU 临时发明新臂 |

上表最坏情况接近预算边界，必须串行按剩余时间删减第三候选或复用已验证对照；不承诺同时完成多年代×三 seed×多个架构的大网格。正式重新训练前将“每臂耗时上限、检验清单、筛选与复现阶段”固定下来。

## 9. 本轮交付边界

已完成代码接入点核对、历史证据边界梳理和论文原始来源核验；只新增本研究文档。没有启动训练、性能 profile、梯度探针、回测或实际券源验证。

主要风险仍是研究计量链与 PIT 数据不完整、历史评估集重复使用、低流动性收益和现实融券成本；更复杂的 Transformer 本身不能消除这些风险。
