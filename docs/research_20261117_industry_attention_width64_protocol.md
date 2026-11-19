# 行业 Attn64 宽度与计费配权对照

2026-11-17，Asia/Shanghai。批次 `outputs/industry_attention_width64_v1`。在[Attn16](research_20261117_industry_attention_protocol.md)上把模型宽度 16 改为 64，FFN 宽度按原 2 倍比例从 32 到 128，两个轴均保留 2 头。训练截止、输入、标签、固定协方差损失、12 轮、AdamW、seed42、两折训练期增量校准都保持。保存训练轨迹和注意力权重。验证 153 周，2021 年起 297 周为已读历史。

固定 B0 Top20。用冻结的 Attn64 算术均值增量比较原 softmax 与[固定历史风险求解器](research_20261117_industry_full_protocol.md)。固定风险两臂只改变优化器内部费用系数，两者回放均扣相同费用。对照 Attn16 计费版、MLP64 连续收益 softmax、风险计费和原 B0。10/30 bp，比较周均净差、最差 10% 周差与原 8 周配对区间。区间跨零不判定改善。

CPU 运行 `python -u scripts/replay_industry_attention.py outputs/industry_attention_width64_v1/prediction --width 64`，再运行 `python -u scripts/replay_industry_joint_risk.py outputs/industry_attention_width64_v1/report --forecast-dir outputs/industry_attention_width64_v1/prediction`。输入和源码摘要、日志存批次目录。新宽度不拟合动态协方差。探索结果受既有历史暴露、Q1 股池及代理成交限制。
