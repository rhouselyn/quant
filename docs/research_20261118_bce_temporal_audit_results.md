# 多空BCE未来信息与历史偏差审计

2026-11-18后续更正见[最终日历审计](research_20261118_bce_final_audit_results.md)。原日期来源后续确认使用未来标签可用性筛选，完整同首末历史已由283周补为295周；本页保留原局部核验范围。
2026-11-18。状态 complete（局部源码及产物核验）。范围为标准B0配比训练和旧BCE控制审计，未逐原始日行情重建所有30特征。
核验标准新236训练周为2002-07-19至2017-11-17。这些日期的93338个新数据有效标签行全部在2018-01-01之前成熟，最晚2017-11-27 18:00 Asia/Shanghai。主干样本边界及五周辅助预留见training_protocol。主干及头标准化仅训练范围或冻结训练统计，周内横截面标准化用当周输入，未用全测试均值。头只以训练索引反向传播，验证只选步，测试锁后推理。shift(-1)代码位于target构造，不在登记30个输入特征列表。输入窗口及覆盖滚动门槛均截至当周。
不能认定整套历史完全无未来信息偏差。样本内主干已经用同一历史标签训练，再向此前训练周生成表示和名单，属于两阶段样本内拟合的泛化偏差。旧best.pt通过2018–2020选轮，新B0的窗口也经2018–2020选轮，下游头继续使用同段验证，属于复用验证，验证收益具有模型选择偏差，不能解释成独立前推结果。数据manifest明确latest_history、fixed survivor pool、static industry、vendor historical adjustments without vintages，因此非严格PIT，历史幸存者选择及行业回溯、复权修订风险存在。2021后测试反复用于研究决策，新一轮锁定不恢复独立盲测。
训练期按完整未来标签筛可训练样本仅用于监督，抽样标签未知的变体丢弃；这是监督缺失选择，输入候选不读未来标签，但可能造成训练分布选择偏差。原283周为旧研究固定日期交集，本轮没有证明原日期生成全过程严格PIT，完整153/297周标准回放和旧283交集不可混称为相同取样。
结论：本次查的模型输入和头优化路径未发现直接将下周收益输入当周预测；标准新训练标签成熟核验通过。整套旧新研究无法担保无未来函数/偏差，因为包含已确认的回溯历史数据及两层复用验证，且未完成全源PIT审计。
证据：src/quant_mvp/hierarchy_data.py、scripts/replay_old_longshort_bce_loss.py、scripts/directional_2018_experiment.py、docs/directional_three_seed_prereg.md、outputs/standard_bce_pool_v1/validation_lock.json、outputs/dataset_v2/standard_v1/snapshots/validation/manifest.json。
