# KiT 迁移、风险回放与 M6 实验会话归档

日期 2026-11-17，Asia/Shanghai。状态 complete。队列状态为当日 12:38 的快照。

## 研究链和结论

1. [KiT 项目分析](research_20261116_kit_results.md)核对 Sunnycave 仓库、作者上游、论文与模型卡。KiT 是条件化未来 OHLCV 路径生成器，以未来段流速度 MSE 训练。作者报告的波动排序证据强于收益排序，公开回测及训练数据存在可复现和时点限制。具体架构、数据、损失和来源差异见原报告。
2. [风险诊断](research_20261116_kit_risk_results.md)中，验证期 EWMA 波动 RankIC 为 0.4720，60 日滚动为 0.4590。历史路径名义 90% 终点区间覆盖 83.40%。风险预测改进本身不证明组合净收益改善。
3. [日线迁移实验](research_20261116_kit_transfer_results.md)比较 B0、周特征控制、普通日线与蜡烛分解日线。三个扩展臂参数量相同；seed42 的蜡烛臂相对 B0 验证净差 +3.30 bp/周，区间跨零。[三种子配对复核](research_20261116_kit_seed_results.md)已完成，蜡烛相对普通日线、周特征、B0 均未达到预登记方向复现标准。蜡烛相对 B0 的验证融合净差为 −8.93 bp/周，95% 区间 [−16.18, −3.76]。
4. 用户要求保持 B0 评分、候选、预算和费用，先做零训练风险回放。[等预算回放](research_20261116_kit_risk_replay_results.md)中，验证期 EWMA 与连续历史路径相对 B0 的净差分别为 +0.18 与 −0.36 bp/周，最差 10% 周收益差为 +0.84 与 +5.38 bp，区间均跨零。30bp 压力和缺失持仓惩罚进一步限制采用，保留 B0 等权。用户明确路径生成头暂不加。
5. 用户要求先验证市场前八周综合表示，再使用同一表示对照 AdaLN。[登记](research_20261116_m6_history_protocol.md)、[验收与逐年](research_20261117_m6_acceptance_results.md)。第一阶段摘要与 GRU 相对 P6，以及第二阶段 AdaLN 相对冻结末端接入，均未通过预登记效果门槛。继续使用原 P6；年度结果及逐种子区间见验收文档与两阶段报告。

## 共同口径与限制

研究使用 2018–2020 的 153 个验证周和 2021 起 297 个已读历史周，模型及多空腿按验证净 Sharpe 选最佳单轮并融合相邻三轮。主档单边交易费 10bp，压力档 30bp；L20、资金和成交规则按各批登记，借券成本只对空头相关账本适用。各模型独立选轮，三种子共享市场历史，不能视作三段独立市场样本。

固定幸存股票池、回溯行业、复权与成交代理、缺失收益处理等限制继承原实验。历史数据已经反复用于研究，结果为探索性比较，不是新独立样本。第二阶段 AdaLN 与末端接入的投影容量不同，比较包含容量差。

## 执行与验收状态

2026-11-17 12:38 核对，KiT 迁移队列 9/9、配对种子队列 18/18、M6 队列 32/32 均 done。两批后台日志显示正常退出并释放 GPU 锁，未发现 Python 进程和该共享锁。实时状态随时间变化，接手时重新查询进程、日志、队列及产物。

M6 两阶段 CUDA 预检通过，15 个模型的评分及选中检查点哈希复核通过。第二阶段三个种子的两个臂均逐张量匹配其第一阶段第八轮 GRU 权重。M6 相关测试 34 项通过。风险回放测试 5 项通过，并以 1e−12 误差对齐原 B0 周账本。

下一步如需继续，应先明确新增研究假设和评价范围，重新核对实时状态，避免依据已读历史回调方案。当前会话没有待恢复任务，路径生成头未进入实现。对其他会话的独立队列，按其各自协议和状态接手。

## 证据与复现

冻结源码、配置、快照引用、日志、锁和机器输出位于对应 outputs 目录；核心结果保存在上方版本化研究文档。以下为本次归档时 SHA-256：

| 证据 | 文件 | SHA-256 |
| --- | --- | --- |
| KiT 迁移冻结清单 | [pipeline_inputs.json](../outputs/kit_transfer_v1/registry/pipeline_inputs.json) | `d01711bb6f7c964c952272e4a6fe7368fa79fd4e7943c50e8d4070caea386834` |
| KiT 配对种子冻结清单 | [pipeline_inputs.json](../outputs/kit_transfer_seed_v1/registry/pipeline_inputs.json) | `3bdec623d4158bdb94ce3b0341f7b1de279a0cf70fddeba0214406ce7c3a1d58` |
| KiT 配对结果 | [seed_pairs.csv](../outputs/kit_transfer_seed_v1/seed_pairs.csv) | `fe82ca2242a14051185ef6e27ab06031c521f36bf0e6273fc4e1d99e250c8e76` |
| 风险回放输入 | [inputs.json](../outputs/kit_risk_replay_v1/inputs.json) | `1a54c2116341e65c049ed39804634d4c84ab852c53a5ff682f13ccb23e6901e6` |
| 风险回放指标 | [metrics.csv](../outputs/kit_risk_replay_v1/metrics.csv) | `e2643ace7319653a0c1bf317c1ea76af27261505569073587a6ab2d5bf9903cb` |
| M6 冻结清单 | [pipeline_inputs.json](../outputs/m6_history_v1/registry/pipeline_inputs.json) | `a9cf633d5efad1524a9bfe8628641943d673c0fdd0d15ef9b04daf73ab278097` |
| M6 第一阶段摘要 | [summary.json](../outputs/m6_history_v1/report_1/summary.json) | `72232b8cd2931911ff9749f34d0a27c9e053d1d9222e68e2c6b292860ff6fd0b` |
| M6 第二阶段摘要 | [summary.json](../outputs/m6_history_v1/report_2/summary.json) | `081c7640651f6350072b4139aa813f1f5c468a22a8bf33292bf682e743c96e04` |

输入及源码路径由各清单记录；选轮锁位于各批 selections 目录。风险回放入口为 `python -u scripts/replay_kit_risk.py`，M6 注册入口为 `python scripts/queue_m6_history.py register`。两个入口会使用已存在的输出目录，复现前应先检查脚本的输出保护及目录约束，保留原结果，不能直接覆盖。原始 outputs/data 证据仅在本机，未核实异地备份；上述哈希供后续迁移核验，不代表已备份。文档和代码尚未因本归档自动提交。
