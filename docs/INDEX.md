# 项目文档导航

用途：新会话的单一阅读入口。维护导航，不在此复制所有结论。更新日期：2026-11-15。

## 接手顺序

1. 读本页，明确任务属于研究、数据、训练恢复还是实盘。研究基线与已部署策略不是同一个对象。
2. 读 [研究经验](research_lessons.md)，写文件时再读 [记录与归档规则](research_recording.md)。
3. 从下表选择对应协议和实现，只加载相关章节；用 `rg -n "关键词" docs scripts src tests` 查旧实验，避免无差别读完大台账。
4. 操作前核对目标批次的配置、数据 manifest、选轮 lock、queue/state 和实际进程。完成标准：能说明范围、对照、数据边界、下一动作，以及哪些结论仍不确定。

## 按任务查找

[可点击的实验改动树](experiment_tree.html)可按改动展开，也可从 B0、P4、P6 和旧 BCE 沿对照、复用和组合关系查看各批。[供 AI 读取的关系数据](experiment_tree.json) · [新增实验与更新说明](experiment_tree.md)。

主干训练周×标签口径2×2（2026-11-19已入队）：[冻结协议](research_20261119_bce_backbone_factor_protocol.md) · [入队交接/后续结果](research_20261119_bce_backbone_factor_results.md)。固定旧输入/覆盖/初始化和训练参数，4臂×3seed各8轮，后接48头及原473/123/295、旧20/10、0/5%与两费档；CPU及队列锁自检通过。等待P4/Muon、空侧RankNet、锚点精排三批结束后执行，实际状态看outputs/bce_backbone_factor_v1/state.json与supervisor日志。

截至 2026-11-19 的跨批对照见[实验对照报告](research_20261119_experiment_comparison_results.md)。首表给出按原口径匹配的效应大小。附表索引各批状态、原值、逐年分腿、费用、同 seed 差值、锁及缺失证据。活动队列状态为报告截点快照。

B0后层128维线性精排与原特征组合（2026-11-19 16:08排队）：[冻结协议](research_20261119_anchor_linear_seeds_protocol.md) · [输入盘点/运行交接/后续结果](research_20261119_anchor_linear_seeds_results.md)。两组线性各五seed及排名集成，对照旧线性/直接；旧MLP及候选裁判按原seed42迁移，full/Swiss6/淘汰共头。已经用128维的全池/滚动锚点裁判不重复改造。后台等待P4五seed完整批次，实际进度见 outputs/anchor_linear_seeds_v1/state.json 和日志，尚无新收益结论。

空侧跨界 RankNet 主干（2026-11-19）：[配对协议](research_20261119_short_ranknet_protocol.md)。seed43–46 固定 B0 多腿，对照原 B0 与旧混搭。前序 P4 Muon 持锁时等待，运行以 `outputs/backbone_short_ranknet_v1/state.json` 和队列日志为准。

B0各seed原生best固定旧40/10任务（2026-11-19完成）：[协议](research_20261119_bce_b0_best_protocol.md) · [完整结果与方向/费用归因](research_20261119_bce_b0_best_results.md) · [最终证据](evidence/20261119_bce_b0_best/final_v2/README.md)。best1/6/3对原e3/e3/e3，固定473/123/295、旧20/10与0/5%；30头实训、64指标/18880周，旧/e3共4720周复现。best全切换36.36/19.03%、36.89/21.11%高于e3但低于旧BCE；毛收益下降，净改善来自交易/券费节省。旧特征源0换向，验证选f30与free并列，未消除差距。

跨界 RankNet 多空腿替换（2026-11-19）：[配对协议](research_20261119_ranknet_legs_protocol.md) · [双亏、互补、冻结 BCE 与去掉 2026](research_20261119_ranknet_legs_results.md)。seed43–46 原权重与原锁定分数，18,880 条回放完成。主费 50/50 周收益增量来自换空，多腿单换走弱。去掉 2026 后换空净收益点估计仍正，冻结 BCE 最大回撤高于 B0。

P4与Muon＋Squared ReLU五seed配对及排名集成（2026-11-19 15:30已启动）：[冻结协议](research_20261119_p4_muon_seeds_protocol.md) · [运行交接/后续结果](research_20261119_p4_muon_seeds_results.md)。两臂seed42–46，补八个八轮训练，RankNet完整完成后已开始P4 seed43；153验证/297历史、两费档和原P4原单位三轮融合不变。原候选seed42权重被其它会话清理，保留的评分/锁/清理SHA链单列审计例外。进度核对 outputs/p4_muon_seeds_v1/state.json、supervisor日志及report/complete.json，不把启动当成低回撤已复现。

本地存储清理（2026-11-19完成）见[盘点与执行结果](storage_audit_20261119.md)，包括实盘、baseline、各方向保留权重和实际释放空间。

固定旧任务的投影冻结10/30/不冻结（2026-11-19完成）：[协议](research_20261119_bce_freeze_steps_protocol.md) · [完整结果](research_20261119_bce_freeze_steps_results.md) · [证据](evidence/20261119_bce_freeze_steps/README.md)。54头新训、5400步记录，72指标/21240周；前10/30步前缀恒等和原f10 7080周复现。六组验证BCE均选不冻结，但测试收益未稳定改善；B0 f30有局部收益改善，P6 free后期头91/88步有验证拟合收益，未消除新旧差距。

固定旧40/10仅换向量来源（2026-11-19完成）：[协议](research_20261119_bce_old_roster_vectors_protocol.md) · [完整结果](research_20261119_bce_old_roster_vectors_results.md) · [证据](evidence/20261119_bce_old_roster_vectors/README.md)。旧/B0/P6各三编码器，公共旧名单/标签/708变体，18头全部重训，原473/123/295及统一旧20/10。全多空无增强年化旧50.01%/30.91%、B0 34.11%/14.03%、P6 19.54%/6.67%；换回旧任务均恢复部分收益，仍低于旧向量。24指标/7080周与旧2360周复现验收通过。

跨界 RankNet 四个补充种子（2026-11-19）：[配对复核协议](research_20261119_ranknet_seeds_protocol.md) · [互补、回撤和 2024 年集中度](research_20261119_ranknet_seeds_results.md)。8 份验证锁和 11,888 条周账本验收通过。50/50 空腿共同弱周改善，双亏均值持平。冻结 BCE 收益优势与 seed42 的 2024 增益均未稳定复现。

B0直接/线性集成排名混合（2026-11-19完成）：[冻结规则](research_20261119_rank_blend_protocol.md) · [逐年与两端对比](research_20261119_rank_blend_results.md)。仅验证Sharpe选出线性75%，历史18.71%/MDD17.38%/Sharpe1.341；回撤高于直接12.75%及线性16.70%，未达到折衷目标。50%历史19.42%/16.17%较好但未据此改选。13500账本及选权/完整排名独立验收通过，无新增训练。

IO 逐合约报价 CUDA 注意力（2026-11-19）：[协议](research_20261119_io_quote_attention_protocol.md) · [结果](research_20261119_io_quote_attention_results.md)。RTX 4060 两主权重加十六滚动拟合。固定合约 IV 误差优于 GRU 但高于持平，开放选择未稳定改善方向、买跨式和保护。

旧线性精排五seed与排名集成（2026-11-19完成）：[协议](research_20261119_linear_seed_ensemble_protocol.md) · [逐年四指标/稳定性](research_20261119_linear_seed_ensemble_results.md)。旧方法与seed42精确复现，十头原预算完成。线性相对各自直接年化/Sharpe为4/5较高，年化均值18.20%对15.63%、std1.33对2.60个百分点；线性排名集成18.79%/MDD16.70%/Sharpe1.317，相对直接集成17.02%/12.75%/1.313提高收益但扩大回撤，非降回撤升级。

三主干融合公共40/10名单BCE（2026-11-19完成）：[协议](research_20261119_bce_fused_three_protocol.md) · [完整结果](research_20261119_bce_fused_three_results.md) · [证据](evidence/20261119_bce_fused_three/README.md)。12头实训，原473/123/295、原锁定及融合20/10两类执行，136指标/40120周完成。原锁定交易名单的32个年化配对中，本次融合信号高于只换向量10个，高于旧BCE0个。

B0/P6三个真实编码器的40/10信号（2026-11-19完成）：[固定名单/标签协议](research_20261119_bce_three_encoder_protocol.md) · [完整结果](research_20261119_bce_three_encoder_results.md) · [证据](evidence/20261119_bce_three_encoder/README.md)。六独立主干身份审计、12头实训、88指标完成；B0无增强全切换自身20/10为23.72%/9.34%，固定5%22.80%/8.75%，P6两组4.11%/−11.68%；各关系年化均未超同增强旧三编码器。补齐编码器数量没有补齐旧新差距，不能把旧集成收益推广到新家族。

IO 原生期权曲面与选合约（2026-11-19）：[协议](research_20261119_io_native_protocol.md) · [结果](research_20261119_io_native_results.md)。20 个期限和价值度节点、两套原生权重、固定/扩训练限制/开放回放完成。IV 误差高于持平基线，开放合约未稳定改善方向、买波动和股票保护。

双端选股与多空互补主干（2026-11-19）：[头尾 RankIC 复算与提案](research_20261119_backbone_complementarity_proposal.md) · [端到端四组协议](research_20261119_backbone_complementarity_protocol.md) · [互补、逐年净收益与回撤](research_20261119_backbone_complementarity_results.md)。四组完成。新主干已改变硬名单，JOINT 的互补增量未通过验证，RANK 同冻结 BCE 的历史收益有改善线索。

旧40/10独立重训与接线审计（2026-11-19）：[协议及单编码器增量](research_20261119_bce_old_retrain_audit_protocol.md) · [结果](research_20261119_bce_old_retrain_audit_results.md) · [证据](evidence/20261119_bce_old_retrain_audit/README.md)。16头新训；原数据与当前Q1两臂0换向复现旧信号；3912篮子价格标签、12新路线输入/首步/选中验证核对通过。单旧编码器后B0固定5%差距显著缩小，不能将三编码器优势全部归为新架构差。

境内期权模型与股票保护（2026-11-19）：[整体研究方案](research_20261119_option_model_proposal.md) · [一手证据](research_20261119_option_model_evidence.md) · [IO 协议](research_20261119_io_weekly_protocol.md) · [IO 首批结果](research_20261119_io_weekly_results.md)。重建沪深300股指期权合约及曲面，完成固定策略、三组小头及股票保护回放。P6 方向复检领先，纯期权买跨式验证为正但复检为负，B0 动态保护尾部最小，增量仍需样本外确认。

五seed等权排名集成（2026-11-19完成）：[协议](research_20261119_rank_ensemble_protocol.md) · [逐年四指标及单seed对照](research_20261119_rank_ensemble_results.md)。六路径全部完成并通过独立验收。B0直接集成年化17.02%/回撤12.75%，回撤低于5/5单seed；B0全比较回撤20.40%亦低于5/5。替换pair全比较/Swiss集成回撤35.37%/30.83%，反高于各自全部单seed，未达到降回撤目标。

B0/P6自选40/10信号迁移（2026-11-19）：[协议及缺失处理授权](research_20261119_bce_native40_transfer_protocol.md) · [四组全指标与解释](research_20261119_bce_native40_transfer_results.md) · [轻量证据](evidence/20261119_bce_native40_transfer/README.md)。18头已完成，固定473/123/295，旧门槛和记平统一三来源及0/5%，四组未超过同增强旧信号；原旧5%另列参考，未混作新基准。

20/10归一化等价初始化（2026-11-19）：[协议](research_20261119_bce_equiv_init_protocol.md) · [结果](research_20261119_bce_equiv_init_results.md) · [轻量证据](evidence/20261119_bce_equiv_init/README.md)。12头新训，固定473/123/295及成对抽样，等价初始化修复自身统计退步，但20/10信号仍落后原40/10；旧40/10实际执行295周复核通过。

20/10退化输入/标签拆解（2026-11-18）：[协议](research_20261118_bce_input_label_protocol.md) · [结果](research_20261118_bce_input_label_results.md) · [轻量证据](evidence/20261118_bce_input_label/README.md)。固定473/123/295及旧40/10归一化，42头含基准复现；只换输入、只换标签均退化，自身归一化进一步降低收益，固定原步数后仍存在，原生抽样略有修复。

旧BCE信号篮子40/10、40/20、20/10（2026-11-18）：[协议](research_20261118_bce_basket_sizes_protocol.md) · [结果](research_20261118_bce_basket_sizes_results.md)。新增30头，固定旧20/10下单295周；原473周与共同426周分别配对，新增信号未改善原40/10。

多空篮子双向尾部互补与下一周择时（2026-11-19）：[执行协议](research_20261118_complementary_books_protocol.md) · [冻结结果](research_20261118_complementary_books_results.md) · [互补、判向、逐年收益与回撤复核](research_20261119_complementary_books_review.md)。B0/P6 各五组完成，三种互补精排均未改变实际名单。

行业至旧冻结BCE会话归档（2026-11-18）：[统一结论、计算流程、实验链和交接](research_20261118_bce_session_archive_results.md) · [轻量证据及manifest](evidence/20261118_bce_session/README.md)。当前组合以295周更正表为准，原始权重和数据仍仅本机。

旧冻结BCE高收益审计及日历更正（2026-11-18）：[协议](research_20261118_bce_final_audit_protocol.md) · [结果与最终组合](research_20261118_bce_final_audit_results.md)。发现旧测试日期依赖未来标签可用性，同首末补12周为295周；替代旧283周的完整历史收益解读。

P6原生Q1特征源单因素（2026-11-18）：[协议](research_20261118_p6_q1_features_protocol.md) · [结果](research_20261118_p6_q1_features_results.md)。原473周及全部监督保留，Q1特征改变测试logit但换向0周；另有六单头同名单回放。

境内期权日线（2026-11-18）见[归档与查询](option_archive.md)。与股票日线共用归档读取接口，覆盖范围和缺口以期权 release 的 manifest 为准。

P6冻结预处理单因素（2026-11-18）：[协议](research_20261118_p6_preprocessing_protocol.md) · [结果](research_20261118_p6_preprocessing_results.md)。原473周及旧原始面板固定，P6冻结统计与旧计算统计相同，实训及回放无变化。

固定旧BCE输入仅替换P6编码器（2026-11-18）：[协议](research_20261118_p6_original_controls_protocol.md) · [结果](research_20261118_p6_original_controls_results.md)。旧473周与标签完全沿用，六组配对，P6扰动方向不变。

完整2018滚动锚点精排五种子（2026-11-19完成）：[配对协议](research_20261118_rolling_anchor_seeds_protocol.md) · [逐年四指标与稳定性](research_20261118_rolling_anchor_seeds_results.md)。B0/替换softIC各seed42–46全部完成并通过独立验收。替换pair瑞士相对B0瑞士CAGR为5/5较高，但回撤5/5更大；新增四seed年化17.18%对14.86%，Sharpe仅2/4较高。五seed不等于五份独立市场证据。

P6 专训 BCE 统一输入（2026-11-18）：[协议](research_20261118_p6_matched_bce_protocol.md) · [结果](research_20261118_p6_matched_bce_results.md)。两臂共同 343 训练周，锁后比较两套名单，复现旧 47.34%。

完整2018主干接滚动样本外锚点精排（2026-11-18）：[协议](research_20261118_rolling_anchor_rerank_protocol.md) · [逐年四指标与验收](research_20261118_rolling_anchor_rerank_results.md)。B0/加法pair/替换softIC三父完成直接、全比较、Swiss6；替换pair瑞士年化26.02%但回撤28.20%，对B0同赛制增量区间跨零。128历史锚点处理跨折基底，仍存在相似结构漂移。

旧三种子冻结 BCE 同主干控制（2026-11-18）：[协议](research_20261118_bce_old_selector_protocol.md) · [结果](research_20261118_bce_old_selector_results.md)。同旧选股器及相同更新步数时，固定 5% 仍只多换两周。

pair主干接样本外向量精排（2026-11-18）：[协议](research_20261118_pair_backbone_rerank_protocol.md) · [逐年结果](research_20261118_pair_backbone_rerank_results.md)。两个2010截止pair前缀、同坐标样本外裁判、全比较/六轮瑞士及同编码器直接排序均已完成；新pair精排均低于原B0精排。

P6专训多空BCE（2026-11-18）：[协议](research_20261118_p6_own_bce_protocol.md) · [结果](research_20261118_p6_own_bce_results.md)。335训练周，榜单统一Top20/Bottom10，与原47.34%配置对照。

P6冻结BCE集成迁移（2026-11-18）：[协议](research_20261118_bce_fixed5_ensemble_p6_protocol.md) · [结果](research_20261118_bce_fixed5_ensemble_p6_results.md)。共同283周的B0/P6逐年对照。

固定名单5%五种子集成（2026-11-18）：[协议](research_20261118_bce_fixed5_ensemble_protocol.md) · [结果](research_20261118_bce_fixed5_ensemble_results.md)。验证并列锁均值，测试投票与均值方向和收益相同。

固定名单5%抽样种子（2026-11-18）：[协议](research_20261118_bce_fixed5_seeds_protocol.md) · [结果](research_20261118_bce_fixed5_seeds_results.md)。四个新随机种子仅一组复现原优势，另三组低于无增强。

冻结BCE小权重全部历史回放（2026-11-18）：[协议](research_20261118_bce_low_grid_full_protocol.md) · [结果](research_20261118_bce_low_grid_full_results.md)。11组仅三种方向，固定5%相对基准换两周。

旧冻结BCE 1%–5%固定与动态重抽（2026-11-18）：[协议](research_20261118_bce_resample_low_grid_protocol.md) · [结果](research_20261118_bce_resample_low_grid_results.md)。验证并列锁1%，固定与无增强测试一致。

旧冻结BCE每步重抽强度（2026-11-18）：[协议](research_20261118_bce_resample_strength_protocol.md) · [结果](research_20261118_bce_resample_strength_results.md)。5%验证胜出，与无增强并列，测试优于30%动态。

夜间训练汇总（2026-11-18）：[逐年对照与恢复报告](research_20261118_overnight_comparison_results.md) · [论文证据](research_20261118_optimizer_activation_evidence.md) · [优化器与激活协议](research_20261118_optimizer_activation_protocol.md) · [原队列失败记录](research_20261118_optimizer_activation_results.md)。P4/P6 加全池 pair 已完成。B0/P4 优化器与激活十臂均训练八轮，原登记失败后的评分状态见 outputs/optimizer_activation_v1/recovery_v1/state.json。

原冻结BCE每步重抽（2026-11-18）：[协议](research_20261118_bce_resample_steps_protocol.md) · [结果](research_20261118_bce_resample_steps_results.md)。回到473周，动态重抽不及固定抽样和无增强，统一B0 Top40/Bottom10共同283周。

旧新BCE控制变量审计（2026-11-18）：[协议](research_20261118_bce_factor_audit_protocol.md) · [结果](research_20261118_bce_factor_audit_results.md)。固定B0 Top40/Bottom10共同283周，核对周覆盖、向量、标签、名单及验证流程。

完整主干随机配对（2026-11-18）：[B0协议](research_20261118_backbone_pair_protocol.md) · [B0逐年结果](research_20261118_backbone_pair_results.md) · [P6协议](research_20261118_p6_global_pair_protocol.md) · [P4协议](research_20261118_p4_global_pair_protocol.md)。B0两臂已完成且历史低于B0；金融父模型对照依次串行，实际进度核对outputs各批次队列、日志与产物。

同裁判赛制与全股票向量排序（2026-11-18）：[赛制协议](research_20261118_pair_tournament_protocol.md) · [赛制逐年结果](research_20261118_pair_tournament_results.md) · [全池向量协议](research_20261118_vector_pair_full_protocol.md) · [全池逐年结果](research_20261118_vector_pair_full_results.md)。两阶段完成；三赛制未优于旧线性，全池向量年化提高但尾部更弱、均值增量区间跨零。

标准B0训练期池内增强（2026-11-18）：[协议](research_20261118_standard_bce_pool_protocol.md) · [结果](research_20261118_standard_bce_pool_results.md)。236有效训练周，153验证/297测试，all增强宽档20.82%/6.95%。

Top100／Bottom50 随机增强（2026-11-18）：[协议](research_20261118_longshort_oof_bce_pool100_protocol.md) · [结果](research_20261118_longshort_oof_bce_pool100_results.md)。验证选半数周增强，测试低于Top50／Bottom20。

TopK／BottomL 池内随机增强（2026-11-18）：[协议](research_20261118_longshort_oof_bce_pool_aug_protocol.md) · [结果](research_20261118_longshort_oof_bce_pool_aug_results.md)。固定374周，半数训练周 Top50/Bottom20 随机增强在测试宽档年化 22.76%/5.39%。

冻结 baseline 的 Top/Bottom 精排（2026-11-18）：[提案](research_20261118_tail_rerank_proposal.md) · [证据](research_20261118_tail_rerank_evidence.md) · [协议](research_20261118_tail_rerank_protocol.md) · [逐年结果](research_20261118_tail_rerank_results.md)。两折OOF、408训练周完成；MLP历史退化，线性有改善但净差区间跨零。

多空配比头滚动训练扩样（2026-11-18）：[协议](research_20261118_longshort_oof_bce_extended_protocol.md) · [结果](research_20261118_longshort_oof_bce_extended_results.md)。五折新主干、374 个共同训练周、283 周历史测试。

多空配比头滚动样本外训练与扰动（2026-11-17）：[协议](research_20261117_longshort_oof_bce_protocol.md) · [结果](research_20261117_longshort_oof_bce_results.md)。共同 45 训练周，宽档与全切换在 283 周历史配对。

等权多空选股只数核对（2026-11-17）：[协议](research_20261117_equal_weight_model_books_protocol.md) · [结果](research_20261117_equal_weight_model_books_results.md)。旧三种子、新版 B0、P6 的同日 Top40/Bottom10 对照和部署历史口径。

旧冻结 BCE 迁移新版 B0/P6（2026-11-17）：[协议](research_20261117_b0_p6_frozen_bce_transfer_protocol.md) · [结果](research_20261117_b0_p6_frozen_bce_transfer_results.md)。共同 283 周，固定各自 Top20/Bottom10 比较三档配比。

本会话归档（2026-11-17）：[KiT 项目、配对种子、风险回放及 M6/AdaLN 结论与接手](research_20261117_kit_m6_session_archive_results.md)。

周频行业预测与持仓倾斜（2026-11-16）：[研究提案](research_20261116_industry_forecast_proposal.md)、[一手文献证据](research_20261116_industry_forecast_evidence.md)。固定 B0 Top20，分离线性均值、双轴注意力和联合风险的增量；仅设计，未执行。

B0 行业调权批次（2026-11-16）：[预注册](research_20261116_industry_allocation_protocol.md)。固定名单，比较现成 P6 聚合、真实收益回归和涨跌分类；执行状态见 `outputs/industry_allocation_v1/queue/state`。

行业双轴注意力（2026-11-17）：[首轮协议](research_20261117_industry_attention_protocol.md) · [首轮结果](research_20261117_industry_attention_results.md)。共用固定协方差，比较行业 Ridge 和双轴注意力的均值预测及同一 B0 名单回放。

行业 Attn64（2026-11-17）：[协议](research_20261117_industry_attention_width64_protocol.md) · [结果](research_20261117_industry_attention_width64_results.md)。实训宽度 64 后比较 softmax、固定风险计费及忽略费用。

行业宽度、损失与配权补全（2026-11-17）：[协议](research_20261117_industry_matched_completion_protocol.md) · [结果](research_20261117_industry_matched_completion_results.md)。已补训 Attn64 动态风险与 MLP64 匹配损失，两个 MLP 宽度做完两种损失的三种配权。

行业联合损失、动态风险与计费配权（2026-11-17）：[协议](research_20261117_industry_full_protocol.md) · [验收](research_20261117_industry_full_results.md)。对比同结构 MLP 损失、固定与时变协方差及费用进入优化目标的增量。

行业 MLP16/64 计费配权（2026-11-17）：[协议](research_20261117_industry_mlp_fee_protocol.md) · [验收](research_20261117_industry_mlp_fee_results.md)。两种隐藏宽度共享固定风险求解器，比较优化费、旧 softmax、Attn 和 B0。

行业 MLP（2026-11-17）：[协议](research_20261117_industry_mlp_protocol.md) · [结果](research_20261117_industry_mlp_results.md) · [宽度 16/64 对照](research_20261117_industry_mlp_width_results.md)。冻结 B0 编码器，按行业平均入选股向量，同输入对比 Ridge 与 MLP。

MLP64 二值标签与温度（2026-11-17）：[协议](research_20261117_mlp64_binary_protocol.md) · [结果](research_20261117_mlp64_binary_results.md)。绝对涨跌、相对强弱和连续收益各比较 T=.02/.05。

多空比例与行业叠加（2026-11-17）：[协议](research_20261117_longshort_prediction_protocol.md) · [结果](research_20261117_longshort_prediction_results.md)。34臂组合回放完成，逐腿净贡献待补。

多空独立双头（2026-11-17）：[协议](research_20261117_longshort_two_heads_protocol.md) · [结果](research_20261117_longshort_two_heads_results.md)。分别预测两腿收益或方向，并对照直接预测差值。

多空比例固定档对照（2026-11-17）：[协议](research_20261117_longshort_ratio_rules_protocol.md) · [结果](research_20261117_longshort_ratio_rules_results.md)。对比固定 45/55、50/50、55/45，限幅 softmax、三档切换和市场方向新标签。

多空比例显式计费（2026-11-17）：[协议](research_20261117_longshort_cost_protocol.md) · [结果](research_20261117_longshort_cost_results.md)。两条双头信号同组比较计费、不计费、三档、softmax 与固定比例。

多空比例 MLP16/32（2026-11-17）：[协议](research_20261117_longshort_width32_protocol.md) · [结果](research_20261117_longshort_width32_results.md)。连续双头共享样本和计费规则，比较隐藏宽度。

旧多空配比冻结 BCE 损失（2026-11-17）：[协议](research_20261117_old_longshort_bce_loss_protocol.md) · [结果](research_20261117_old_longshort_bce_loss_results.md)。复现旧冻结 BCE 后实训幅度加权和方向加幅度。

旧冻结 BCE 配权规则（2026-11-17）：[协议](research_20261117_old_longshort_bce_rules_protocol.md) · [结果](research_20261117_old_longshort_bce_rules_results.md)。同一个旧信号比较跳档、softmax、显式计费及同优化忽略费用。

旧冻结 BCE 温度选择（2026-11-17）：[协议](research_20261117_old_longshort_softmax_temperature_protocol.md) · [结果](research_20261117_old_longshort_softmax_temperature_results.md)。验证选中最低测试温度 .005，历史收益接近宽档跳仓。

旧冻结 BCE 低温加密搜索（2026-11-17）：[协议](research_20261117_old_longshort_softmax_lower_protocol.md) · [结果](research_20261117_old_longshort_softmax_lower_results.md)。验证目标在近零温饱和，历史与原宽档近似持平。

旧冻结 BCE 全范围 softmax（2026-11-17）：[协议](research_20261117_old_longshort_softmax_full_protocol.md) · [结果](research_20261117_old_longshort_softmax_full_results.md)。验证回撤约束选 T≈.0419，历史尾部较好、周均较低。

旧冻结 BCE 固定温度仓位上下限（2026-11-17）：[协议](research_20261117_old_longshort_softmax_bounds_protocol.md) · [结果](research_20261117_old_longshort_softmax_bounds_results.md)。T=.005，锁定 0%–97% 多头，历史回撤低于全切换。

旧冻结 BCE 固定温度三档回撤约束（2026-11-17）：[协议](research_20261117_old_longshort_softmax_tight_bounds_protocol.md) · [结果](research_20261117_old_longshort_softmax_tight_bounds_results.md)。锁定 0%–95%、0%–93%、0%–88%。

旧冻结 BCE 分腿下行风险（2026-11-17）：[协议](research_20261117_old_longshort_leg_risk_protocol.md) · [结果](research_20261117_old_longshort_leg_risk_results.md)。验证选 135 bp 预计正损失预算，历史没有降低最大回撤。

旧冻结 BCE 高低档扫描（2026-11-17）：[协议](research_20261117_old_longshort_band_grid_protocol.md) · [结果](research_20261117_old_longshort_band_grid_results.md)。按验证净周均锁定单腿端点，历史净均值更高但尾部更弱。

旧冻结 BCE 双费率尾部约束（2026-11-17）：[协议](research_20261117_old_longshort_band_tail_protocol.md) · [结果](research_20261117_old_longshort_band_tail_results.md)。验证选中 31/69，已读历史的尾部落后原 40/70。

旧冻结 BCE 全多全空计费（2026-11-17）：[协议](research_20261117_old_longshort_full_cost_protocol.md) · [结果](research_20261117_old_longshort_full_cost_results.md)。端点计费、连续配权及弱信号回退。

候选池行业筛选与投影冻结（2026-11-17）：[协议](research_20261117_industry_freeze10_protocol.md) · [结果](research_20261117_industry_freeze10_results.md)。Top100/Bottom50 内实训行业 BCE，比较 30 种行业数及股数匹配对照。

纯多头 Top10 行业事后排名（2026-11-17）：[协议](research_20261117_industry_long_oracle_protocol.md) · [结果](research_20261117_industry_long_oracle_results.md)。B0 Top100 内行业按实际未来收益排序，随后以 B0 分数选至多 40 股。

全截面行业涨跌分类（2026-11-17）：[协议](research_20261117_industry_full_sign_protocol.md) · [结果](research_20261117_industry_full_sign_results.md)。全截面汇总行业股票，比较方向全对的事后筛选和真实训练的 BCE。

全截面行业 BCE 幅度损失（2026-11-17）：[协议](research_20261117_industry_sign_loss_protocol.md) · [结果](research_20261117_industry_sign_loss_results.md)。实训幅度加权 BCE 和方向加幅度似然，对照原冻结 BCE。

行业排名两端多空（2026-11-17）：[协议](research_20261117_industry_tails_protocol.md) · [结果](research_20261117_industry_tails_results.md)。15 种信号，比较前三/后三、前五/后五及行业内限只数或全取。

行业完美排序反事实（2026-11-17）：[协议](research_20261117_industry_oracle_protocol.md) · [结果](research_20261117_industry_oracle_results.md)。用未来实际行业收益排前五/后五，固定 B0 个股选择和执行规则。

M6 八周市场表示与 AdaLN（2026-11-16）：[登记协议](research_20261116_m6_history_protocol.md)、[验收与逐年对比](research_20261117_m6_acceptance_results.md)。先比较 P6、八周摘要和八周 GRU，再固定同一 GRU 比较末端接入与 AdaLN。状态见 outputs/m6_history_v1/queue/state，每阶段完成后单独归档。

KiT 等预算风险回放（2026-11-16）：[登记协议](research_20261116_kit_risk_replay_protocol.md)、[零训练回放结果](research_20261116_kit_risk_replay_results.md)。固定 B0 Top20，比较等权、EWMA、历史路径权重。

KiT 配对种子复核（2026-11-16）：[登记协议](research_20261116_kit_seed_protocol.md)、[配对种子结果](research_20261116_kit_seed_results.md)。复用 seed42，补四臂 seed43/44。状态见 `outputs/kit_transfer_seed_v1/queue/state`。

P6TAIL 零训练归因（2026-11-16）：[排序头反事实、持仓、行业与市场状态](research_20261116_p6_tail_attribution_results.md)。原验证锁和历史口径不变；同模型回退均值排序未修复历史表现，后续方案仅为建议。

KiT 周频迁移（2026-11-16）：[外部模型审计](research_20261116_kit_results.md)、[本地实验登记](research_20261116_kit_transfer_protocol.md)、[零训练风险结果](research_20261116_kit_risk_results.md)、[实现与入队](research_20261116_kit_transfer_implementation_results.md)。[日线对照结果](research_20261116_kit_transfer_results.md)。执行证据由 `outputs/kit_transfer_v1/queue/state`、日志及产物交叉核对。

P6尾部分类与残差排序头（2026-11-16）：[执行协议](research_20261116_p6_tail_protocol.md)、[实现与入队](research_20261116_p6_tail_implementation_results.md)、[最终结果与分析](research_20261116_p6_tail_results.md)。本批训练及评价完成，实时状态仍以进程、日志、队列及产物核对。

模型研究会话归档（2026-11-15）：[P6/B0损失互借与完整研究链结论](research_20261115_loss_transfer_session_results.md)。覆盖六方向研究至P6GAP、FOXCONST、P4GROUP、BDIFF2最终结果；区分分解恒等式、均值校准和排序目标。新双头损失方案仅为提案，未实现、未入队。

跨会话接手点（2026-11-15）：[项目会话结论与证据导航](research_20261115_session_archive_results.md)，覆盖当前可识别的项目会话、历史对照、数据修补和未决限制。实际状态仍需重新核查。

数据研究最新结果（2026-11-15）：[缺口补证、修补快照及最终比较](research_20261114_gap_followup_results.md)；其修补结果替代[2026-11-14二源缺口进度快照](research_20261114_gap_resolution_results.md)中未解释的67条和189条计数，旧12轮结果仍按原边界有效。[D3/D4协议](research_20261113_data_trials_protocol.md)和[修补快照协议](research_20261114_repaired_bridge_protocol.md)分别适用于各自批次；全局严格数据门槛仍未通过。

继续本轮未完工作：先读 [管道补齐结果与剩余队列](research_20261113_pipeline_completion_results.md)，再按需读 [执行协议](research_20261113_pipeline_completion_protocol.md)、[P64三种子复核](research_20261113_p64_recheck_results.md)、[归档补录与后台恢复](research_20261113_archive_completion_results.md)。选模历史与更严谨验证方法见 [方法审计](research_20261113_model_selection_audit.md)。下面standard_v1仍是已完成八模型结果，不代表总数据管道已验收。

| 任务 | 先读文档 | 接着看实现或证据 |
| --- | --- | --- |
| 当前数据研究、三年验证协议 | [标准执行与恢复](research_20261113_standard_execution.md)、[切分纠正来源](research_20261113_split_correction.md) | [本轮评估结论](research_20261113_checkpoint_review.md)、[本地批次状态](../outputs/dataset_v2/standard_v1/status.json) |
| 总体基线与历史实验 | [模型/策略基线](baseline_monopen.md)、[实验台账](experiment_ledger.md) | 台账按假设和日期查；旧判决只适用于当时设置 |
| 数据归档、清洗与因果性 | [数据执行方案](research_20261113_dataset_execution_runbook.md)、[实现](research_20261113_dataset_implementation.md)、[审计](research_20261113_dataset_code_audit.md) | `scripts/build_dataset_snapshot.py`、`src/quant_mvp/hierarchy_data.py`、`tests/test_dataset_contract.py` |
| 数据来源与论文启发 | [来源](research_20261113_dataset_sources.md)、[方法](research_20261113_dataset_methods.md)、[原管道](akshare_pipeline.md) | 区分论文主张、候选假设和本项目实证 |
| CS、CB、ADN及SELF机制 | [条件交互登记](research_20261113_conditioning_trials.md)、[adaLN比较](research_20261113_adaln_comparison.md)、[模型说明](hierarchy_implementation.md) | `src/quant_mvp/hierarchy.py`；配置决定实际开启分支 |
| 训练、断点恢复、队列 | [标准执行与恢复](research_20261113_standard_execution.md) | `scripts/run_weekly_architecture.py` → `src/quant_mvp/hierarchy_engine.py` → `training_state.py`；队列 `scripts/weekly_followup_queue.py` |
| 验证选轮与收益回放 | [标准执行](research_20261113_standard_execution.md)、[本轮结论](research_20261113_checkpoint_review.md) | `scripts/score_dataset_standard.py` → `scripts/evaluate_dataset_three_year.py`；对应 `tests/test_standard_scoring.py`、`test_dataset_three_year.py` |
| 实盘与部署 | [部署说明](live_deployment.md)、[策略基线](baseline_monopen.md) | `scripts/live_pipeline.py`、`src/quant_mvp/live_score.py`；先核实线上现状 |
| 金融建模驱动的预测改进（当前研究路线） | [实现与合成验收](research_20261113_finance_implementation_results.md)、[数学设计与四agent报告](research_20261113_finance_model_results.md) | [研究协议](research_20261113_finance_model_protocol.md)、[原研究证据](research_20261113_finance_evidence.md)；已接入训练/评分与概率评价，[真实尾标签审计与执行](research_20261113_finance_real_preflight.md)、[真实结果](research_20261113_finance_real_results.md)；旧模型保留 |
| A股数学建模、损失与解释性研究（历史审计） | [六方向综合建议](research_20261113_math_synthesis_results.md)、[共同事实与源码指纹](research_20261113_math_facts.md) | [已替代优先级的旧协议](research_20261113_math_protocol.md)；六份独立报告见综合页，旧协议未执行，当前方向见上行 |
| 新实验或交接归档 | [记录与归档规则](research_recording.md) | 协议→机器产物→分析→台账→本页指针 |
| 行业投影与差分注意力追加研究 | [实现与入队](research_20261115_model_extensions_implementation_results.md)、[证据](research_20261115_model_extensions_evidence.md)、[行业投影与差分注意力协议](research_20261115_model_extensions_protocol.md)、[行业投影与差分注意力结果](research_20261115_model_extensions_results.md) | `outputs/model_extensions_v1`，后排于降噪排序批次 |
| 降噪组内排序与 FOX 常数衰减 | [协议与校准审计](research_20261114_p6gap_foxconst_protocol.md)、[最终结果](research_20261114_p6gap_foxconst_results.md) | `outputs/p6gap_foxconst_v1`，2026-11-15结果归档完成；实时状态以队列和日志为准 |
| P6缓冲归因与FOX遗忘门零训练诊断 | [诊断结果与下一步方案](research_20261114_p6_fox_diagnostics_results.md)、[协议及抽样范围](research_20261114_p6_fox_diagnostics_protocol.md) | `outputs/p6_fox_diagnostics_v1`；P6全153/297周；FOX35周机制抽样，非全周干预回测；未改模型、未训练 |
| 组内排序、FoX与Top20/30回放 | [组内排序与FoX协议](research_20261114_group_fox_protocol.md)、[组内排序与FoX结果](research_20261114_group_fox_results.md)、[缓冲回放结果](research_20261114_buffer_2030_results.md)、[实现进度](research_20261114_group_fox_progress.md) | `outputs/group_fox_v1`；队列及结果以state、日志、产物为准 |
| 金融模型零训练归因、排序与换手改造、纯架构论文 | [持仓/成本/行业/状态归因](research_20261114_finance_attribution_results.md)、[纯网络架构论文](research_20261114_architecture_papers_results.md) | `outputs/finance_attribution_v1`；分析完成，改造仅提案，未改模型代码、未开训 |
| 金融概率模型的均值训练改进（P4/P5追加） | [均值优先目标协议](research_20261114_finance_faithful_protocol.md)、[真实结果](research_20261114_finance_faithful_results.md)、[实现与入队检查](research_20261114_finance_faithful_implementation_results.md) | `outputs/finance_faithful_v1`；追加在父P0–P3完成后，保留原NLL对照；已登记不代表已开训 |

## 文档状态与冲突

- 当前数据研究协议入口是 standard_execution；dataset_execution_runbook、parallel_exploration 等含先前两年F0方案，阅读时同时查看切分纠正。历史文件名里的F0不证明新评分仍使用旧切分。
- 协议说明“应该做什么”；运行时配置、数据哈希和锁说明“实际做了什么”；进程与输出说明“完成到哪里”。三者不一致就是需要报告的偏差。
- README里的旧服务器、训练状态和缓存周数是历史记录，不能当实时状态。具体任务日期和最新用户要求优先核实。
- `data/`、`outputs/` 被Git忽略，下面的本地证据链接在新克隆可能缺失。此时读版本化结论，明确证据不可用；按记录寻找备份或复现，不能声称已经核验。

## 本轮历史快照（不是实时状态）

2026-11-13：8个已有seed42、8轮模型完成三年验证和297周历史测试；17个新训练暂缓。评估工作进程退出。旧权重缺完整训练状态，后续代码已支持状态恢复。后来的会话须重新检查状态，完整证据路径和边界见 [验收摘要](research_20261113_checkpoint_review.md)。

2026-11-13晚：旧B0/CS/P64/CS×P64三种子全部复用评估完成，GPU释放；归档补回5,872条旧码行情，103个权益候选核实并发布。沪深3,486只二源采集及自动后处理仍在后台，严格gate保持false。后续先核实时进程和finish_status，再读[最新补齐记录](research_20261113_pipeline_completion_results.md)，不要把本文快照当实时完成状态。

- [2026-11-14 缺口补证、修补快照与训练进度](research_20261114_gap_followup_results.md)
