# 原冻结 BCE 的 softmax 温度选择

2026-11-17，Asia/Shanghai。复用[原宽档回放](research_20261117_old_longshort_bce_rules_results.md)的冻结 BCE 三种子平均 logit、固定 Top40/Bottom10 与旧周账本。沿用仓位映射，以 50% 多头为中点，sigmoid(z/T) 两边分别线性伸缩到 40% 和 70%。原宽档 z>0 用 70%，否则 40%。不重训信号。

预定温度 0.005、0.01、0.02、0.05、0.1、0.2、0.5、1。仅在原验证 133 周内，以 10/30 bp 两档各自扣费周均的算术平均选一个共同温度；相差不超过 1e−8 bp 取较高温度。先写锁，再只用入选温度及原宽档在 283 个已读历史周复检。10/30 bp 搭配年券费 8%/12%。报告组合周均、最差 10% 周、多空净贡献、换手、以及移动 8 周块配对区间。验证结果包含调参，历史已读。

CPU 设置 PYTHONPATH=src 后运行 `python -u scripts/replay_old_longshort_softmax_temperature.py outputs/old_longshort_softmax_temperature_v1/report`。检查宽档原账本和验证 T=1 的既有账本、仓位边界、逐腿与组合净收益恒等式。
