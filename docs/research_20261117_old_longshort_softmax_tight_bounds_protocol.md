# 固定温度 Softmax 的三档回撤约束

2026-11-17，Asia/Shanghai。复用[固定温度仓位范围](research_20261117_old_longshort_softmax_bounds_protocol.md)的 2,601 组验证网格、T=.005、原冻结 BCE 和 Top40/Bottom10。三组分别要求验证 10/30 bp 相对全多全空最大回撤都至少下降 2、3、5 个百分点。在各约束下最大化两费档扣费净周均平均值，同分容差与选组规则沿用上轮。先写锁，再在相同 283 周已读历史复检。对照为原全多全空、原 Softmax 40%–70% 及上轮 Softmax 0%–97%。

原 133 周验证结果、10/30 bp 搭配年券费 8%/12% 和慢速费用账本均沿用。对赢家验算验证收益和回撤，对三个基准逐周对账。报告净周均、自身最差10%周均、两腿净贡献、52 周年化夏普、最大回撤、以及对全多全空的 8 周移动块配对区间。CPU 设置 PYTHONPATH=src 后执行 `python -u scripts/replay_old_longshort_softmax_tight_bounds.py outputs/old_longshort_softmax_tight_bounds_v1/report`。
