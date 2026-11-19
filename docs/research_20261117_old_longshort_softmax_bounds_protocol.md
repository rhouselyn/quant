# 固定温度 Softmax 的多头仓位上下限扫描

2026-11-17，Asia/Shanghai。固定旧冻结 BCE 三种子平均 logit、Top40/Bottom10 名单、T=.005。在[上一版温度实验](research_20261117_old_longshort_softmax_temperature_results.md)的映射中，零分对应 50% 多头，正负方向分别向上限和下限移动。枚举下限 0%–50%、上限 50%–100%，步长 1 个百分点，共 2,601 组，两腿内等权、名义总仓 100%。

仅用原 133 周验证。要求两档最大回撤相对全多全空均下降至少 1 个百分点，再取两档扣费净周均等权平均最高者。同分容差 1e−8 bp，取与 40/70 的曼哈顿距离最小者，再按低档和高档由小到大。先锁，再只用锁定组合与原 40/70、T=.005 及原全多全空复检 283 周已读历史。交易费 10/30 bp 配 8%/12% 年券费。报告净周均、最差 10% 周均、两腿净贡献、年化夏普和最大回撤及 8 周块配对。

CPU 设置 PYTHONPATH=src 后执行 `python -u scripts/replay_old_longshort_softmax_bounds.py outputs/old_longshort_softmax_bounds_v1/report`。复用旧费用账本，对照和中选臂逐周核对。已读历史和模拟成交、券源，无训练。
