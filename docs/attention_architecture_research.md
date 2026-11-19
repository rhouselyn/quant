# 注意力架构与训练方法研究

检索截至 2026-11-11。范围是原论文、作者发布和本仓库实验记录。本文提出一个待验证的下游消融，未运行训练或回测。

## 首选消融：多空两腿之间加一个交互项

周频骨干对每只股票的 40 周 × 30 特征序列独立编码，使用 128 维、四层四头、RoPE、Q/K RMSNorm 和 SDPA 后 sigmoid 门，已采用 2025 年 Gated Attention 论文所研究的输出门。下游冻结 2018 截点的 42/43/44 三个骨干，用共同的 Top-40 多头和合资格 Bottom-10 空头。283 个已读历史周里，在同一 70%/40% 多头分配规则下，分腿均值池化加 BCE 为年化净收益 22.28%、夏普 1.54。加每股 GELU、加腿内标准差、用 MSE 或 Huber 替换 BCE 的配对净收益点估计均较低。原始全篮单查询和分腿双查询的早期对照也未显示对均值池化的优势。比较不同分配规则的绝对收益时，需回到同一账本。 [骨干配置](../configs/hierarchical_580_monopen_baseline.yaml) · [时间注意力实现](../src/quant_mvp/model.py#L108) · [Gated Attention 论文](https://arxiv.org/abs/2505.06708) · [三种子协议](directional_three_seed_prereg.md) · [注意力复查](attention_followup_results.md) · [最新分腿消融](leg_pool_ablation_results.md)

`LegPool(False)` 把两腿分别均值池化，再用同一个 FFN 和 LayerNorm 各自得到 32 维向量 `z_L`、`z_S`。最后的线性层只给出 `b + w_L·z_L + w_S·z_S`。因此一腿某项表征的作用无法随另一腿的状态改变。借鉴多变量模型的同一时点信息交换，在原 logit 上只加 `v·(z_L ⊙ z_S)`。新增 32 个零初始化权重，初始化时两臂 logits 应一致。这个对角双线性交互只检验两个已聚合的组合状态能否共同解释方向。其依据来自本地计算图与通用时序研究，属于待验证的迁移推断。 [分腿池化代码](../scripts/attention_followup.py#L29) · [Chronos-2 分组注意力消融](https://arxiv.org/html/2610.41956v1)

1. 用种子 42 配对首筛，复跑均值 BCE 以核对缓存、输入标准化和初始 logits。保持相同完整标签周、BCE、优化器、训练步预算和验证选步规则。沿用零阈值和正分 70%、非正分 40% 的多头比例，空头补足其余预算。新增参数初始对照通过后，检查交互权重确实收到梯度。
2. 在事先固定的开发周上比较配对净收益、BCE、交易与券费、回撤及正分周数。报告改变方向的周数和其实际收益差，用开发期事先锁定的固定比例对照检查平均敞口是否解释结果。若候选只改善验证损失，或配对收益优势集中在少数周，则结束首筛。按现有执行协议用约 5 基点/周作为扩种子的收益量级，正向线索补 43/44 后再冻结复查。 [既有分腿协议](leg_pool_ablation_protocol.md) · [项目执行与验收协议](research_20261111_execution_plan.md)

## 近期方法及适用范围

| 方法和日期 | 原始证据与本项目取舍 |
| --- | --- |
| **LLM · Attention Residuals / Kimi K3** · 2026-03-16 / 07-27 | AttnRes 用 softmax 从较浅层选取残差来源，Block 版对块摘要做选择。首篇大模型实验为 48B 总参数、3B 激活参数、1.4T tokens。K3 又将其与 Kimi Delta Attention 组合。仓库已探测四层骨干的深度稀释并关闭 M4，保留探针结果作为当前深度下的判断依据。 [AttnRes 论文](https://arxiv.org/abs/2603.15031) · [Kimi K3 报告](https://arxiv.org/abs/2607.24653) · [仓库 M4 结果](research_exec_20261110_report.md#10-判定矩阵与本轮检验登记) |
| **LLM · DeepSeek-V4 / Zyphra ZHA** · 2026-04 / 11-02 | V4 交替使用压缩后稀疏检索和重压缩注意力。ZHA 对 KV 摘要建树，逐层路由并用冻结密集模型蒸馏索引器，超长序列优势来自作者的性能模型。这里每股只有 40 个时间 token，且每个决策只需一次编码；已有算量估计中，QK 和注意力乘 V 约占单层主要矩阵乘的 5.4%。仓库已把 40 周压成 8 块做过首种子对照，端到端训练约 950–958 秒，对照约 837 秒，两臂均停止。当前算力证据降低了移植长上下文内核的优先级。 [DeepSeek-V4 报告](https://arxiv.org/abs/2606.19348) · [ZHA 论文](https://arxiv.org/abs/2610.42112) · [本地算量推导](research_20261109_models.md) · [时间块验收 §6](../outputs/research_20261111/acceptance.md) |
| **时序 · TimesFM-3 / Chronos-2** · 2026-08-31 / 10-30 | TimesFM-3 是 330M 参数、超过一万亿时点预训练的解码器，按 32 点成块，交替做因果时间注意力与同一时点的跨变量注意力。Chronos-2 的 10 月报告描述 120M 参数编码器、同一块索引的分组注意力、分位数损失，以及用随机即时与滞后映射合成多变量训练样本。可借鉴有边界的同刻交互和可得性掩码。仓库的 40→8 两个时间块臂在开发窗分别落后 5.37 和 0.45 基点/周，区间均跨零。两个末端跨股注意力臂约落后 4–5 基点/周。若将来增加协变量，只用决策时已知的日历或已发布数据，并保留原历史对照。 [TimesFM-3 作者说明](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) · [Chronos-2 报告](https://arxiv.org/html/2610.41956v1) · [时间块方案](research_20261111_execution_plan.md#34-时间块注意力与近端保真) · [跨股实验](experiment_ledger.md) |
| **金融 · Kronos / KiT** · AAAI 2026 / 2026-09-28 | Kronos 在逾 120 亿根 K 线上做离散化后的自回归训练。KiT 用条件流匹配联合生成未来 OHLCV 路径，论文主模型为 284M 参数。在 2026-01-01 至 04-10、分钟到日频的七种分辨率上，作者报告平均收益 RankIC 0.0571，Kronos 微调基线为 0.0521。这个比较没有回答周频策略的净收益问题。本地 Kronos 零样本和低秩微调纯多分别落后约 41 和 38 基点/周，且历史与预训练语料年代有重叠；重跑外部权重优先级低。 [Kronos 论文](https://ojs.aaai.org/index.php/AAAI/article/view/39730) · [KiT 论文](https://arxiv.org/html/2609.34507v1) · [本地 Kronos 记录](research_20261111_execution_plan.md) |

## 可以借鉴的训练与验证方法

- **辅助目标要验证其净贡献。** DeepSeek-V4 沿用了多 token 预测。10 月的 Soft-MTP 预印本在语言模型中把额外预测梯度拆为自蒸馏与低信噪比残差。可借鉴的做法是分别观察主任务和辅助任务的梯度与样本覆盖。仓库已做去 InfoNCE 的同轮对照，首种子略负。计划中的隔夜/日内辅助头已有等规模对照。下游有直接监督证据：收益差加权 BCE 走弱，MSE 和 Huber 对实现收益差的相关分别只有 0.021 和 0.034。当前给方向头增加收益差辅助监督的优先级低于交互消融。 [DeepSeek-V4 报告 §2.1](https://arxiv.org/html/2606.19348v1) · [Soft-MTP 预印本](https://arxiv.org/abs/2610.38331) · [InfoNCE 对照](research_exec_20261110_report.md) · [隔夜/日内对照计划](research_20261111_execution_plan.md#33-持有期内的隔夜与日内监督) · [本地分腿结果](leg_pool_ablation_results.md)
- **方向、幅度和经济结果分别测量。** Finance1K 的 1,000 只美股小时收益实验以同一骨干对比 MSE 与 MSE 加同刻 Pearson 相关损失，报告 IC 从 0.046 到 0.126，同时预测幅度仍是实际幅度的约 1.84 倍。这个预印本支持直接核对目标尺度与交易所需排序，不能把截面 Pearson IC 等同于周度两腿胜率或成本后收益。仓库股票评分训练已有截面软 RankIC，分配头则保留 BCE 和共同账本。11 月 9 日的预注册期货预印本中，五个通用基础模型在盲测 160 个配置里无一胜过价格不变预测，也说明朴素对照的必要性。 [Finance1K 论文表 4](https://arxiv.org/pdf/2608.14106#page=17) · [期货盲测预印本](https://arxiv.org/abs/2611.12141) · [本地评分训练](../src/quant_mvp/hierarchy_engine.py)
- **按时点锁住信息。** 记录语料范围、最后训练日期、权重版本、验证和选模时点。TimesFM-3 的前视只适用于事先已知的协变量，作者对下载的 3.0 权重另有非商业、非生产限制。对 A 股头，券源资格、复权输入和完整标签各按实际可得时间对齐，周一成交结果留在执行账本。2021–2026 已复查的 283 周和代理券费只支持历史比较。前瞻期采用同一股票池、成本和成熟标签口径。 [TimesFM-3 作者说明](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) · [官方权重许可说明](https://github.com/google-research/timesfm#license-notice-for-pretrained-weights) · [项目时间隔离协议](research_20261111_execution_plan.md#2-基准重建与信息隔离)
