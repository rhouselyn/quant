# 原冻结 BCE 的双费率尾部约束

2026-11-17，Asia/Shanghai。沿用[原 5,151 组扫描](research_20261117_old_longshort_band_grid_protocol.md)的旧 BCE、Top40/Bottom10 名单、1 个百分点网格和原 133/283 周。验证中每个费档分别取最差 ceil(10%×周数) 周的平均净收益，只保留 10 bp/8% 券费与 30 bp/12% 券费下的尾部都不比旧 40%/70% 档差的组。以验证两档扣费净周均的算术平均选胜者，同分规则沿用无约束批次。先保存锁并给出平均配比，再读取已读历史回放。历史只使用锁定档、原宽档、上轮无约束赢家，以及将上轮验证期平均 84.962406% 多头冻结为每周固定比例的对照。

报告净周均、最差 10% 周均、多空扣费净贡献、年化夏普（周净收益均值除样本标准差乘根号 52，无风险利率零）和最大回撤（周期从净值 1 起算）、平均配比和原宽档配对区间。旧历史与执行、券源代理不变。验证约束不作历史硬约束。

CPU，设置 PYTHONPATH=src，先执行 `python -u scripts/replay_old_longshort_band_grid.py outputs/old_longshort_band_tail_v1 validation --tail-floor`，锁定后再执行同脚本同输出目录并用 history 替换末项。网格和验证锁独立保存。
