# 原冻结 BCE 的低温加密搜索

2026-11-17，Asia/Shanghai。沿用[前轮温度选择](research_20261117_old_longshort_softmax_temperature_protocol.md)的同信号、固定 Top40/Bottom10、40%–70% 多头映射、133 周验证和 283 周已读历史。将 0.005 以下的搜索扩为从 0.000001 至 0.005 等对数排列的 121 个正温度。另纳入 T=0 的原宽档跳仓。同样用验证期 10/30 bp 扣费周均的平均值选温度，距最优不超过 1e−8 bp 时取较高温度。旧扫描已覆盖 0.005 至 1。

锁定后只回放选中值与原宽档，报告净周均、最差 10% 周和逐腿净贡献，检查基准复现及逐周净收益恒等式。10/30 bp 分别配 8%/12% 年券费。原采样窗口已读，搜索为网格近似，低温可能在浮点精度下饱和。

CPU 设置 PYTHONPATH=src，执行 `python -u scripts/replay_old_longshort_softmax_temperature.py outputs/old_longshort_softmax_lower_v1/report --lower-grid`。
