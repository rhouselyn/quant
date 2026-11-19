# 行业 MLP 宽度对照

2026-11-17，Asia/Shanghai。批次 `outputs/industry_mlp_width64_v1`。沿用[首轮 MLP 协议](research_20261117_industry_mlp_protocol.md)的评估日期、已成熟标签、B0 编码器、回放和拟合规则。唯一候选改动是隐藏宽度从 16 到 64（2081 到 8321 个头部参数）。同样初始化规则、seed42、12 轮和训练预算，不按现有验证或已读历史增选宽度。跑完后与冻结的宽度 16 产物按周配对，先逐值核对 B0、风险和 Ridge 三个对照。主看验证的算术 RMSE、10/30 bp 扣费均值与最差 10% 周，使用原 8 周自助区间。历史复检是诊断。扣费均值区间跨零不升格。CPU 四线程。入口 `python -u scripts/replay_industry_mlp.py outputs/industry_mlp_width64_v1/report --width 64`。
