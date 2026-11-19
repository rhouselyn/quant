# 原冻结 BCE 的 softmax 温度选择结果

2026-11-17，Asia/Shanghai。状态 complete。[协议](research_20261117_old_longshort_softmax_temperature_protocol.md) · [父批](research_20261117_old_longshort_bce_rules_results.md)。固定原 BCE 三种子聚合信号和每周 Top40/Bottom10。温度在 133 个验证周锁定为 0.005，只把该温度带到 283 个已读历史周。验证目标为 10/30 bp 两档扣费周均的平均值。按温度 0.005、0.01、0.02、0.05、0.1、0.2、0.5、1 顺序，目标为 38.32、37.83、36.83、32.22、26.66、22.71、20.08、19.15 bp。原跳档验证两档均值为 39.72 bp。

| 样本 | 单边 bp | 规则 | 组合净周均 bp | 最差 10% 周均 bp | 多头净贡献 bp | 空头净贡献 bp | 平均目标多头 | 目标多头标准差，百分点 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 验证 | 10 | 原宽档 | 52.30 | -240.04 | 43.07 | 9.23 | 65.49% | 10.72 |
| 验证 | 10 | Softmax .005 | 50.91 | -240.00 | 42.49 | 8.42 | 65.45% | 10.60 |
| 验证 | 30 | 原宽档 | 27.14 | -264.04 | 31.47 | -4.33 | 65.49% | 10.72 |
| 验证 | 30 | Softmax .005 | 25.73 | -263.91 | 30.88 | -5.16 | 65.45% | 10.60 |
| 已读历史 | 10 | 原宽档 | 44.69 | -271.18 | 34.04 | 10.65 | 62.90% | 12.75 |
| 已读历史 | 10 | Softmax .005 | 43.60 | -268.64 | 33.59 | 10.01 | 62.61% | 12.44 |
| 已读历史 | 30 | 原宽档 | 18.27 | -297.74 | 22.57 | -4.31 | 62.90% | 12.75 |
| 已读历史 | 30 | Softmax .005 | 17.07 | -295.17 | 22.14 | -5.07 | 62.61% | 12.44 |

Softmax 相对跳档的周均差在验证为 −1.39/−1.41 bp，历史 −1.09/−1.20 bp；四个 8 周块配对 95% 区间均跨零。历史尾部差为 +2.54/+2.57 bp，区间分别 [−4.03, +11.21]、[−4.18, +10.72]。最低网格温度胜出，持仓波动与原跳档相近，原跳档的周均点估计四档均更高。没有继续扫网格下沿，低温极限在非零 logit 上会收敛到原跳档。

单边 10/30 bp 分别配 8%/12% 年券费。锁在历史回放前写入。CPU 退出 0，3,526 条唯一记录；原宽档四档账本、验证 T=1 两档账本逐值复现，各臂净收益等于多空净贡献之和。旧历史已经阅读过，继续采用模拟成交和代理券费。

[温度锁](../outputs/old_longshort_softmax_temperature_v1/report/selection.json) · [完整指标](../outputs/old_longshort_softmax_temperature_v1/report/metrics.csv) · [配对区间](../outputs/old_longshort_softmax_temperature_v1/report/paired.csv) · [周账本](../outputs/old_longshort_softmax_temperature_v1/report/weekly.parquet) · [日志](../outputs/old_longshort_softmax_temperature_v1/run.log) · [输入哈希](../outputs/old_longshort_softmax_temperature_v1/inputs.json)。本机产物未异地归档。
