# 原冻结 BCE 的全范围 softmax

2026-11-17，Asia/Shanghai。固定[原冻结 BCE](research_20261117_old_longshort_softmax_lower_results.md)三种子平均 logit 和每周 Top40/Bottom10 名单，沿用 133 周验证、283 周已读历史。多头比例直接取 sigmoid(z/T)，空头取余，名义总仓位为 100%。基准为 z>0 全多，其余全空。候选为从 0.00001 到 2 的 121 个对数等距温度，加 0.005 和上轮验证锁定的精确低温。

仅验证选择。要求 10 bp 和 30 bp 档最大回撤都至少比原全切换小 1.00 个百分点，再最大化两档净周均的平均值。同分容差 1e−8 bp，取较高温度。如无温度满足，才放宽为两档回撤均严格较低并记录；若仍无可行值，则锁全切换。先写锁，再只回放选中的温度和原全切换到历史。原有两个温度的全范围结果仅作验证诊断。交易费 10/30 bp 配年券费 8%/12%，按各自仓位路径重新扣费，逐腿净贡献复核。报告净周均、自身最差10%周、逐腿净贡献、52 周口径年化夏普、最大回撤及移动 8 周块配对。回撤约束只在选择期适用。

CPU 设置 PYTHONPATH=src 后执行 `python -u scripts/replay_old_longshort_softmax_full.py outputs/old_longshort_softmax_full_v1/report`。无重训或验证后改温度。
