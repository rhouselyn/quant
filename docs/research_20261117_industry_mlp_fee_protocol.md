# MLP16/64 计费优化对照

2026-11-17，Asia/Shanghai。批次 `outputs/industry_mlp_fee_v1`。复用[MLP 宽度对照](research_20261117_industry_mlp_width_results.md)冻结的算术相对收益预测，宽度 16、64 均来自同一 128 维入选股行业平均表示。保持旧 softmax 已用的预测量纲和数值，不重训、重校准或扫参数。

沿用[显式计费实验](research_20261117_industry_full_protocol.md)的 B0 Top20、历史收缩篮子风险、决策时旧仓估值、凸优化参数及上下限。每个宽度各回放计费与优化费置零两臂。两臂实际执行均按同一费率扣除成本。验证最初无预测周都用仅风险且计费规则。旧 softmax、仅风险计费、Attn 固定风险计费和 B0 作对照。MLP 输入未做 Attn 的预测收缩，跨架构对比包含信号量纲与校准差异。

报告验证 153 周与已读历史 297 周，同时单列有两种 MLP 信号的 104/297 周，费用路径不重新起算。主看 10/30 bp 的周均净差、最差 10% 周差及同周 8 周块区间。区间跨零不视为确认的增益。核对仅风险版复现旧账本。CPU 运行 `python -u scripts/replay_industry_mlp_fee.py outputs/industry_mlp_fee_v1/report`，批次目录保存输入摘要、日志、指标和逐周账本。
