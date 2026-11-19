# 行业宽度、损失与配权补全对照

2026-11-17，Asia/Shanghai。延续[联合损失与动态风险](research_20261117_industry_full_protocol.md)、[Attn64](research_20261117_industry_attention_width64_protocol.md)和[MLP16/64 计费](research_20261117_industry_mlp_fee_protocol.md)。训练和回放口径不改，补齐已登记宽度与对照的缺项。

- Attn64 补训秩 3 动态协方差头。复现已冻结 Attn64 算术均值，以两个早期前缀生成的 200 周样本外 log 残差拟合风险头 12 轮。比较固定和时变协方差，在计费求解器中保持均值一致。与 Attn16 动态风险直接按周配对。
- 扩展原 MLP16 两种匹配损失至 MLP64，仍保持相同训练周、周 batch、初始化规则和 12 轮。两个宽度、两种损失均回放 softmax、固定风险计费和同优化不计费。原 MLP16/64 行更新 MSE 旧结果仍作为参考，不混入周 batch 的损失对照。
- 固定 B0 Top20、腿内比例、预算、成本，主报告 153 周验证、297 周已读历史，以及 MLP 共同有信号周。10/30 bp、净收益、尾部和块配对区间并列。选择结论与点估计名次分开。

CPU 运行。Attn64 动态风险使用 `replay_industry_joint_risk.py --fit-dynamic --forecast-dir outputs/industry_attention_width64_v1/prediction`。MLP64 用 `replay_industry_mlp_joint.py --width 64`，然后分别用 `replay_industry_mlp_fee.py --loss mse` 和 `--loss joint`。源和产物摘要、训练日志及回放存各新批次目录。未增加 seed 或调参搜索。
