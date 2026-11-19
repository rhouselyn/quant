# P4主干增加全池pair的独立对照

2026-11-18 Asia/Shanghai，planned。用户指出P6已有组内pair可能重叠，故新增P4GLOBAL：以原finance_faithful_v1/runs/research_P4_F0_s42_a1为父，无组内pair，新增global pair .1；保留已授权P6GLOBAL，不按历史挑父模型。P4更清晰识别纯均值监督上增加全池排序的影响，P6检验已有组内排序上的增量，两者不互相替代。

方法、成熟/有效mask、mu_log/s_ref、原始log标签、四随机非self对手、独立RNG、8轮seed42、153验证best±1三轮锁和297历史、双费用及L20/S10/LS50资本预算、概率诊断和研究限制均沿[金融全局pair协议](research_20261118_p6_global_pair_protocol.md)。唯一父模型差异为within_group_pair_weight=0而非.1。对原P4配对并列B0，不把跨父模型差异全归于新loss。

脚本scripts/run_p4_global_pair.py，独立outputs/p4_global_pair_v1/frozen_source，依赖p6_global_pair_v1/report/complete.json后串行GPU。计划写queue/state.json及源码hash，正式前自检必须显式从本批frozen/src导入并断言模块路径，防根目录shim绕过。未启动不称运行。预计一臂10–20分钟加评分；不增加种子或按历史调权，未部署。结果需包括逐年收益/含初始1MDD/Sharpe/周均换手、费用、概率NLL/RMSE/覆盖与锁核对，记录所有负结果。
