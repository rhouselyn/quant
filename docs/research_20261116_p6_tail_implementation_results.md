# P6 残差排序头实现与队列状态

2026-11-16，Asia/Shanghai。协议：[登记](research_20261116_p6_tail_protocol.md)。状态：已实现、测试通过、真实训练及评价完成；四个队列任务均done，最终结论见[结果页](research_20261116_p6_tail_results.md)。实时进程以队列、日志与进程核查为准。

新增默认关闭的线性排序残差头：`score=stopgrad(mu_log)+s_ref*h(z)`。分类标签沿B0的截尾、头尾20%规则，损失每日期等权；0.1权重、温度1。原P6的MSE、风险NLL、组内排序、分解和风险梯度隔离保留。旧模型不生成新头、不改变旧checkpoint；新增头初始化不改变原随机状态。启用时评分分别保存决策score及原单位mu/scale，epoch/seed融合分别求均值，概率评估按mu与尺度构建密度，旧score等于mu的约束仍对旧模型生效。

CPU检查：`python -m pytest tests/test_financial_decision_head.py tests/test_financial_return.py tests/test_financial_faithful.py tests/test_financial_group_pairs.py tests/test_standard_scoring.py tests/test_dataset_three_year.py -q`，39 passed；随后新增真实评分入口变体，`python -m pytest tests/test_financial_decision_head.py tests/test_financial_return.py::test_array_and_scoring_cli_use_frozen_statistics_without_label_access -q`，7 passed。包括无效候选mask、旧参数同初始化、零权重两轮训练模型参数和评分逐项相同、尾损失不直接传到mu和scale但可经共享编码器传递、概率及决策分数分别融合、训练中断恢复、标准评分和旧模型概率回归。

冻结源码SHA256：`1653e3cb1acfe6a8b7286edce4d887f8afc7fcae3e2bf8da93a20cb80cc8e5b9`；清单 `outputs/p6_tail_v1/registry/pipeline_inputs.json`。队列顺序：真实尺寸CUDA前后向零优化步→P6TAIL八轮及三年验证逐腿锁→锁后历史评分→对照评估与文档归档。沿用P6和B0原产物；启动器通过共享 `outputs/dataset_v2/gpu.lock` 串行，失败阻断下游。最终[结果页](research_20261116_p6_tail_results.md)由报告任务生成，生成前不代表结果已完成。

排序头增加128参数。B0概率头不存在，因此只与P6比较概率；历史已暴露，单seed，固定幸存股池、回溯行业、Q1成交和券源代理，不是独立前瞻或实盘收益。原始产物和冻结源码仅本机，未异地备份。

补充：用临时测试副本将同一合成逐腿选轮和评价流程改为`score != mu`的新语义，`test_locked_leg_fusion_and_evaluation_artifacts` 通过（1 passed）；临时副本已移除。真实CUDA门通过：RTX 4060、16日期×580股票、峰值分配约1643 MiB、梯度有限、0优化步；2026-11-16 02:29:54报告完成且锁已释放。
