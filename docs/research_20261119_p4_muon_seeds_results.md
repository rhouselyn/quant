# P4与Muon＋Squared ReLU五seed复核：运行交接

2026-11-19 15:30 Asia/Shanghai。状态 running，尚无新增五seed绩效结论。[冻结协议](research_20261119_p4_muon_seeds_protocol.md)。

当前RankNet批次已完整完成，其根级complete.json绑定的产物逐项核验后，本批隐藏后台监督PID50620、训练子进程14612开始P4 seed43。此为带时间的快照；查询现在进度须检查 outputs/p4_muon_seeds_v1/state.json、supervisor.stdout.log、supervisor.stderr.log、实际进程及共享GPU锁。当前锁由本批持有。

两臂各seed42–46、每训八轮；两发现seed42评分复用，新增四seed各两臂，共八训64epoch。两组各自等权完整排名集成，LS50空侧按最终集成长20重新计算共同排除。预算、数据划分和两费档继承，原P4三轮原单位均值融合，未改用B0的zscore融合。全部十锁后才生成新seed历史评分，CPU报告随后自动发布本页。

入队前检查通过：完整冻结source与输入哈希；队列排他锁/不删除他人锁/依赖失败拒绝及依赖产物变动拒绝；排名共同池、symbol平局、成员顺序不变性、空侧排除、空池/单股票；两个seed42的5400行周账本回放及24组全期四指标与原报告一致。核验记录 seed42_check.json、queue_selfcheck.json，未以预检查扩训练预算。

预检查修订保留 input_manifest.preflight1.json、input_manifest.preflight2.json、preflight_amendment.json、dependency_preflight_correction.json；只修正原汇总中重复但相同的P4参考行去重及当前RankNet实际完成标记路径，未修改口径或容差。实际输入冻结在 input_manifest.json，规则在rule_lock.json。

候选seed42权重及训练state于另一会话15:22存储清理删除。原评分、配置、预处理、训练记录及带SHA删除清单保留，本批逐项核归档评分链；该成员不能重新forward/精确续训，不能把评分回放核验写成权重重新推理。原P4 seed42完整权重仍在。新增训练权重完整保存。

自动报告入口 `python scripts/report_p4_muon_seeds.py`；完整完成以 outputs/p4_muon_seeds_v1/report/complete.json 为准。最终报告将列出逐年收益、MDD、Sharpe、周均双边换手、全期指标、all5/new4稳定性、同seed改善数量、集成对各成员和对照集成、费用压力及配对区间。目前不能判断低回撤是否复现。

本批是已读历史探索性复检，不是新独立市场验证；原券源和成交代理等限制继承。权重和原始产物仅本机，未部署。
