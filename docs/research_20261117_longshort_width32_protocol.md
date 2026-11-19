# 多空双头连续 MLP32 对照 MLP16

2026-11-17，Asia/Shanghai。批次 longshort_width32_v1。复用[双头训练](research_20261117_longshort_two_heads_protocol.md)与[计费规则](research_20261117_longshort_cost_protocol.md)。仅比较连续 257→16→1 与 257→32→1 的独立两头，保持标准化、两腿收益 RMS、MSE、AdamW、seed42、每轮相同的样本顺序、12 轮及每 13 周扩窗重拟合；最少 40 个成熟周，历史首周冻结。先对账 MLP16 原预测与原计费账本。

同一锁定 B0 名单和信息截止，固定分母，长腿预算 45%–55%，同一计费优化、单边 10/30 bp 与年券费 8.35%/12%。153 周验证和 297 周已读历史。主要比较宽度 32 减 16 的周均净差与最差 10% 周，8 周移动块配对区间；另列逐腿、纯空头持原仓位、预测 RMSE 与换手。无新股票选择或费用调参。单种子与已读样本只支持探索。

CPU 根目录 `PYTHONPATH=src`，执行 `python -u scripts/replay_longshort_width32.py outputs/longshort_width32_v1/report`。产物独立保存，完成两种宽度全段对照即停止。
