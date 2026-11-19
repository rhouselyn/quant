# P6完整主干增加全池随机配对

2026-11-18 Asia/Shanghai。planned，用户明确要求P6可行则加入队列。依赖backbone_pair_v1/report/complete.json，排在B0两臂之后，单GPU串行。独立outputs/p6_global_pair_v1/queue/state.json记录计划及依赖，不修改正在运行B0的冻结源码或manifest。

父P6为group_fox_v1/runs/research_P6_F0_s42_a1，faithful_mean_nll、decomposed_state、normal_log，已有within_group_pair_weight=.1保留。新臂P6_GLOBAL_PAIR增加.1*global random pair，原主干架构、8轮seed42、数据BQ_F0_r2、2018截止/5周预留、优化器/batch和成熟边界均继承，完整主干从同seed重新训练。helper沿B0每有效股票4个非自身同周对手、有放回，收益相等贡献0保留分母，独立日期epoch采样RNG。此采样与旧P6组内不等标签最多4对手不同，不能称为纯范围单变量。

分数采用mu_log/s_ref，标签原成熟log收益不新增截断；无效mu先按forward available mask归零避免无监督空和NaN。全池pair梯度进入mu均值头/主干，保持scale风险输入z.detach和NLL mu.detach原路径，pair不直接更新scale头。同日市场公共项在差分中消失，跨行业pair可监督行业/个股相对项，不能声称直接改善市场均值。

数学分解和正尺度分布合法性可保持，但MSE+排序的复合目标一般改变纯条件均值最优解，概率校准可能恶化；父P6本来已含组内排序。除净收益/回撤/Sharpe/换手外报告NLL、RMSE、RankIC、90%覆盖，不能用排序改善替代概率校准。训练日志MSE归因须扣除新增global pair损失项。

完整2018–2020验证153周逐腿10bp净Sharpe最佳单轮±1三轮融合，先锁后2021-01-08起297周已读历史复检。费用10/30bp对应年借券8.35%/12%，原L20/S10/LS50资本、资格/流动性、同股/现金/缺失估值与执行规则不变。与父P6配对，B0同列参考。逐年按决策年累计净收益、含初始1年内MDD、Sharpe及周均双边换手，2026非全年；均值差给8周块区间。固定幸存池/回溯行业/供应商复权和执行/券源代理风险继承。

新入口scripts/run_p6_global_pair.py准备独立frozen_source/src及scripts快照，只在快照允许faithful金融global pair；不改当前B0源。实施后先测试真实小P6梯度到均值/主干、不直接到scale，默认关闭兼容、金融mask/全无标签，然后获取原共享GPU锁串行训练，保存真实源码/配置/数据/检查点/锁hash。原training_state恢复规则沿用，不自动扩轮/种子或按历史调系数。

预计一臂约10–20分钟加评分，以实测修正。仅本机未异地备份。队列planned不代表已启动；完成需核对进程、日志与complete产物。执行结果随后另页归档。
