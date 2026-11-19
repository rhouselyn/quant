# KiT 日线对照实施与入队

2026-11-16，Asia/Shanghai。状态 running（02:48 核查）。协议 [KiT 迁移实验登记](research_20261116_kit_transfer_protocol.md)。风险阶段的正式结果见 [风险结果](research_20261116_kit_risk_results.md)。

日线从标准历史快照同一原价、后复权、股票池与日历生成，逐日检查 `跳空 + 实体 = 收盘到收盘 log 收益`。标准化训练前缀至 2017-12-29，日线对照与蜡烛对照的逐位置掩码完全一致。每周 5 个交易日位置，每位置 5 个量和 1 个观测掩码，短周、停牌及无连续前收盘位置零填。日线特征和快照覆盖通过文件哈希绑定，评分期只读截至当期快照的输入。所有臂继续使用 B0 的 779 个训练锚、标签和候选掩码。

新增 KITWEEK、KITDAILY、KITCANDLE 各 1,483,905 个总参数。复用现有时间支路接口，每周一块、最近 8 周、1 层宽 32 的 PyTorch Mamba2。共享主路径用同一初始随机状态，日分支门从零开始。实际 CUDA 批次 `16×580×8×30` 前后向通过，三个臂的初始分数与 B0 逐元素相同，门梯度有限且非零，检查峰值分配约 1,156 MiB。原有回归测试 54 项通过。首个对照的已提交 checkpoint 在验证快照上评分 1,160 行，有限分数掩码与 B0 逐项相同。原训练锚哈希、监督计数和 InfoNCE 掩码亦与已存 B0 证据相同。

GPU 队列在完成第 2 轮后转为独立后台工作进程，使用完整训练状态恢复，第 3 轮日志已推进。初次提交的队列请求和两轮权重被保留，重启时工作器重新打开当前日志，前两轮摘要仍可在训练状态和训练指标中查证。首两轮实测各 277–283 秒，三臂训练约需 110–115 分钟，另外有验证评分、锁轮、历史评分和回放。这个估计替代原登记的一小时估计，实验预算不变。

源码 `cae99164efd062cb986e0009abc6dcffaa3a9080cd74c3ec96826372e3e4f8f6`。训练、选轮、历史评分和账本依次由 `outputs/kit_transfer_v1/queue/inbox` 执行。`report` 写机器表，`archive` 核对报告身份、计算 RankIC 和边界换股，并生成 `docs/research_20261116_kit_transfer_results.md`。归档流程已用独立工作目录和已有完整账本的模拟对照跑通。实际状态由 [训练日志](../outputs/kit_transfer_v1/logs/research_KITWEEK_F0_s42_a1.log)、[后台工作器日志](../outputs/kit_transfer_v1/worker_detached.log)、[冻结清单](../outputs/kit_transfer_v1/registry/pipeline_inputs.json) 和 `queue/state` 核对。原始权重与产物只保存在本机。
