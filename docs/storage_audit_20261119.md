# 项目存储盘点与实验权重保留清单

2026-11-19，北京时间。本机项目范围内盘点 76,265 个文件，逻辑大小 68.01 GiB。其中 `outputs/` 58.08 GiB、`data/` 8.36 GiB、`.git/` 1.10 GiB。共找到 3,240 个权重或训练张量文件，合计 14.80 GiB。项目内已经共享的硬链接合计少占约 0.45 GiB。以下数字按文件字节计算，实际释放量以卷统计为准。

[逐路径清单](storage_audit_20261119.tsv)覆盖 `outputs/` 和 `data/` 的一级条目、所有权重类文件、已确认的重复文件和本轮列出的中间产物。可按「建议」筛选。目录和文件是不同层级，行大小有重叠。`待核对` 表示已记录体积，尚未完成该条目的下游和选轮检查。

清理之后的容量与下一批候选见[滚动种子和 dataset_v2 复查](storage_audit_20261119_followup.md)。

## 2026-11-19 清理执行结果

15:22 北京时间完成。按文件原磁盘占用，本次释放 **10.71 GiB**（11,497,582,592 字节）。C 盘可用空间由 **9.39 GiB** 增至 **19.68 GiB**，净增 **10.28 GiB**。清理期间同卷还有运行任务写入。 [实测汇总](../outputs/storage_cleanup_20261119/summary.json) [逐文件操作记录](../outputs/storage_cleanup_20261119/actions.jsonl)

| 操作 | 文件数 | 释放 |
| --- | ---: | ---: |
| 合并重复排名表，保留两条原路径及原散列 | 42 对 | 3.50 GiB |
| 清理已验证可重打包的 r2021 临时包 | 1 | 0.42 GiB |
| 清理股票冻结消融向量，以及 M6 和金融概率逐样本诊断 | 14 | 5.14 GiB |
| 清理已确认只留结果的权重和训练状态 | 359 | 1.65 GiB |

原清单中的 **18 份融合 BCE 向量（1.48 GiB）已保留**。固定旧 40/10 名单的复盘脚本按原路径读取它们，两批输入清单和文件散列均一致。M6 被清理的实验保留周账本、概率摘要、验证和历史评分。原完整验收命令会校验已清理权重。股票消融的源 `best.pt`、配置和快照已复核，未执行全量重新导出。 [固定旧名单复盘入口](../outputs/bce_old_roster_vectors_v1/run.py)

清理前后对 476 个保护文件做了存在性和文件身份检查，并检查本地 15 个实盘相关 run 的实际选轮。RankNet 补充种子在清理期间进入历史评分，多空实盘名单进程在等待收盘数据。远端实际部署未核验。

## 第一批可释放约 3.92 GiB

| 对象 | 预计释放 | 处理理由 |
| --- | ---: | --- |
| `rolling_anchor_seeds_v1/seed43–46/report/orders/` 中 32 对 `*_L20.parquet` 与 `*_LS50.parquet` | 3.11 GiB | 同一验证或历史阶段、同 arm、同赛制，两个文件 SHA-256 一致。保留两个文件名，通过 NTFS 硬链接共用内容。该批已完成，写入时用临时文件原子替换。 [写入入口](../scripts/run_rolling_anchor_seeds.py) |
| `rank_blend_v1/orders/` 中 10 对 `q*_L20.parquet` 与 `q*_LS50.parquet` | 0.39 GiB | 同批同权重的两套完整排名逐字节一致，完成标记也保存了相同散列。可共用内容并保留两路径。 [写入入口](../scripts/replay_rank_blend.py) |
| `tmp_r2021_pkg/abl2000_base_r2021_5seed.tar` | 0.42 GiB | 包内 105 个常规文件逐个通过本机 `outputs/abl2000_base_r2021*` 内容校验。原 run 可重新打包，保留其配置、权重和选轮文件。 |

对 ≥512 KiB 的非权重文件按大小筛选，再用完整 SHA-256 确认，共得到 400 组有独立副本的同内容文件，理论重复空间 6.23 GiB。上表只把已核对批次完成和写入方式的 42 对列为合并候选。其余包括跨批 BCE 向量、旧行情缓存和 `linear_seed_ensemble_v1` 的订单表。线性集成脚本会直接写目标文件，暂缓对它跨目录建硬链接。`p6_matched_bce*_v1/training_scores.npz` 的两路径已经共享同一文件，删除单个路径没有约 224 MiB 的释放收益。 [线性集成写入入口](../scripts/replay_linear_seed_ensemble.py)

## 可按只留结果处理的约 8.27 GiB

下面的前提是允许相关消融以后从保留结果查阅，必要时重跑生成诊断。保留原配置、预处理、冻结源码、选轮锁、输入清单、验证和测试汇总、周度账本，以及仍被使用的预测与分数。清理有路径或哈希绑定的输入前，应确认对应批次已停止运行，并验证保留的重建入口。逐样本核验和精确续训所需的文件以明细表为准。

| 对象 | 预计释放 | 为什么可以只留结果 |
| --- | ---: | --- |
| `m6_history_v1/report_1`、`report_2` 下的 4 份 `probability_observations/members.parquet` | 2.76 GiB | 从分轮评分汇编的逐样本概率表。逐年、逐周和配对结果有独立产物。保留原评分和锁，逐样本概率审计按需重算。M6 两阶段各自主验证均未过门槛。 [验收](research_20261117_m6_acceptance_results.md) [生成位置](../scripts/evaluate_dataset_three_year.py) |
| `bce_fused_three_v1` 的 18 份 `*_vectors.npy` | 1.48 GiB | 六编码器的 train/validation/test 全截面向量是下游头训练和复核的中间输入。已留 136 条绩效和 40,120 条周账本，原执行名单的 32 个比较中融合头没有胜过旧 BCE。父编码器、原选轮、输入哈希、选中头及统计都要留。重新做逐样本 forward 需要导出向量。 [融合结果](research_20261119_bce_fused_three_results.md) |
| `research_stock_freeze_comparison/` 的 6 份消融 `.npz` | 1.39 GiB | 两个训练改法各三 seed 的向量缓存。原模型在验证中领先，消融的历史增量区间跨零。`weekly.csv` 和 `report.json` 已落盘，回放入口在缓存缺失时从各 run 的 `best.pt` 提取。先保留这 6 份源 `best.pt` 及配置。 [结果](experiment_ledger.md#27-2026-11-11-下午股票权重多空择时与分层训练) [重建入口](../scripts/research_stock_freeze_compare.py) |
| `finance_faithful_v1/report` 和 `finance_return_v1/report` 中 4 份概率明细 | 1.00 GiB | 容量来自逐样本诊断，绩效和校准摘要独立保存。P4、P6 及其基线/选轮评分有下游价值，保留权重和原评分。 [金融建模研究链](research_20261115_loss_transfer_session_results.md) |
| KiT 未通过复核的非保留 run 权重 | 0.65 GiB | 66 个补充种子和 22 个初筛权重文件。三项预登记方向复现均未通过。保留 s42 KITCANDLE 作为此方向代表，以及 s43/s44 B0，后两者还被 RankNet 清单绑定。 [KiT 结果](research_20261116_kit_seed_results.md) [当前依赖](../outputs/backbone_ranknet_seeds_v1/input_manifest.json) |
| M6 非保留 run 权重 | 0.36 GiB | 99 个文件，属于未过门槛的摘要和其余种子。留 P6 基线及 s42 GRU/FROZEN/AdaLN 代表链，GRU 的 e8 是第二阶段冻结初始化。 [验收](research_20261117_m6_acceptance_results.md) [接线](../scripts/queue_m6_history.py) |
| 旧 580 与 1893 已裁决权重 | 0.16 GiB | 36 个文件。包括 λ0.3、head20/50、gatelr20、无冻结 λ1、stickiness、连续尾部、行业尾部和 cellbal 的显式名单。已裁决结论与逐周/预测证据保留。无冻结 λ1 的 s42 和历史部署用的冻结门 s43/s44 容易混淆，已按目录单列。 [台账](experiment_ledger.md) |
| 优化器与激活非保留 run 权重 | 0.27 GiB | 55 个文件。全部十臂结果可读，保留双基线 Squared ReLU、单独 Muon，以及 B0 Muon+SiLU 代表臂。B0/P4 的 Squared ReLU 两档费下历史年化都有局部改善，作为通用候选保留。 [逐年结果](research_20261118_overnight_comparison_results.md) |
| 冒烟、测试及放弃训练的权重 | 0.12 GiB | 59 个文件。用于检验管线或留下失败记录，保留日志和结果即可。 [旧盘点](../tmp_weights_report.md) |
| `p4_global_pair_v1`、`p6_global_pair_v1` 的实验权重 | 0.08 GiB | 22 个文件。两臂全池 pair 历史收益均降且回撤扩大，保留逐年与配对结果及原 P4/P6。 [逐年结果](research_20261118_overnight_comparison_results.md) |

## 应保持可推理的各方向模型

- **实盘及历史 baseline。** 留 `abl2000_base_r2021*` 与 `abl2000_base_r2023*` 各五 seed，及 `live_pipeline.py` 默认的五个 `...d128h4_livev*` run。保留实际读取的轮次和预处理。`abl2000_base` 三 seed、早期三 seed MonOpen 和冻结门部署基线具有历史对照价值。 [周任务](../scripts/weekly_qq.sh) [基线](baseline_monopen.md)
- **通用主干和通用改进。** 留探索标准 B0、标准三年验证与 CS/P64 对照，P4/P6 金融主干，以及优化器/激活的代表性局部改进。P6 的三个真实编码器在 `group_fox_v1` 和 `m6_history_v1`，是后续 BCE 的可重复输入。 [三年复核](research_20261113_checkpoint_review.md) [编码器清单](../outputs/bce_fused_three_v1/model_inventory.json)
- **低回撤选股、线性精排和替换 pair 专项。** 留 B0 五 seed 直接排名、线性精排五个 `heads.pt` 及配套 OOF 输入，保留替换 pair/Swiss 的原冻结候选。五 seed B0 直接集成对应较低回撤，线性对应更高收益和更大回撤，替换 pair 属有收益线索的专项。`rank_blend_v1` 只是冻结两端的排名混合，保留选权锁、汇总和逐周结果即可解释结论。 [排名集成](research_20261119_rank_ensemble_results.md) [线性精排](research_20261119_linear_seed_ensemble_results.md) [五种子 pair](research_20261118_rolling_anchor_seeds_results.md)
- **多空择时和互补。** 留旧 40/10 三编码器、原投影冻结 BCE 的无增强和固定 5% 入选头、B0/P6 实盘名单的上游轮次和选择器。固定冻结 10/30/free 的 54 个头合计仅约 2 MiB，可保留校准对照。RANK seed42 有冻结 BCE 下的增量线索，保留整个已登记复核链。 [冻结对照](research_20261119_bce_freeze_steps_results.md) [RankNet 协议](research_20261119_ranknet_seeds_protocol.md) [名单推理入口](../scripts/complementary_books_live.py)
- **金融分布、尾部、行业和期权。** P4/P6 及当前代表性研究头保留。P6TAIL 验证较强但历史退化，仍有专项对照价值。行业和 IO 单头节省空间有限，保留已入选和部署依赖权重。IO 主权重连同平值持平/固定规则对照用于继续比较；滚动拟合如需精简先确认对应预测与锁完整。 [P6TAIL](research_20261116_p6_tail_results.md) [IO 注意力](research_20261119_io_quote_attention_results.md)

`best.pt` 可能对应训练器的单一指标，正式结果可能用多头和空头各自的 win3 或要求固定 e8。保留模型时按冻结选轮取各窗口的并集，考虑下游要求的额外轮次。带 `config.initial.yaml` 的新 run 在部分评分入口需要 `training_state.pt` 验证已提交轮次，即使只做预测也会读。标记为「只留结果候选」的训练，在删除完整状态后无法按原状态继续，也可能无法通过原验收脚本的 checkpoint 哈希检查。 [训练状态检查](../src/quant_mvp/training_state.py)

## 暂缓的其他大目录

| 对象 | 当前大小 | 依赖或价值 |
| --- | ---: | --- |
| `rolling_anchor_seeds_v1` | 13.57 GiB | 全量订单表约 9.33 GiB，其中第一批已含 3.11 GiB 重复空间。余下为按赛制重跑和独立验收的输入，向量与主干还有线性精排、集成和 RankNet 的下游复用。 [五种子结果](research_20261118_rolling_anchor_seeds_results.md) |
| `dataset_v2/` | 8.05 GiB | 涉及当前 Q1 快照、锁、标准基准和归档证据。其中 `archive_completion_v1` 3.80 GiB，主要是 BaoStock 分页和状态证据。`repaired_bridge_v1` 1.26 GiB 具有修补前后同口径对照和冻结快照，拆权重需要逐 run 查锁。 [归档补录](research_20261113_archive_completion_results.md) [修补配对](research_20261114_gap_followup_results.md) |
| `weekly_architecture_v1/`、`weekly_followup_v1/` | 1.89 / 0.98 GiB | 旧结构筛选、三种子历史对照和部分父模型，盘点约 1.53 / 0.53 GiB 权重。大规模清理前应按已登记对照和下游实际读取的 epoch 复核。 [架构研究入口](INDEX.md) |
| `linear_seed_ensemble_v1/`、`rolling_anchor_rerank_v1/`、`pair_backbone_rerank_v1/` | 0.78 / 1.05 / 0.75 GiB | 有 OOF 或验证排序输入和后续期权/精排引用，先保留；线性排序表的重复文件已列明。 [滚动精排结果](research_20261118_rolling_anchor_rerank_results.md) |
| `directional_2018/` | 0.71 GiB | 主体是旧三种子向量，仍可为原 40/10 基准提供重算输入。先保留原头和父模型，避免把同名 20/10 衍生数据当替代。 [旧训练审计](research_20261119_bce_old_retrain_audit_results.md) |
| `data/` | 8.36 GiB | 包括股票与期权发布归档、旧数据口径、历史复权、PIT 证据和实盘输入。`ashare_v2` 约 2.96 GiB，含已到期期权和供应商原响应。 [期权归档](option_archive.md) |
| `.git/` | 1.10 GiB | `git count-objects` 显示 17 项临时/异常对象约 194 MiB。版本库有本地修改，历史对象和 GC 单独核查。 |

盘点期间，RankNet 补充种子的状态、GPU 锁、PID 和更新中的日志一致，seed44 RANK 在训练。本地 `complementary_books_live.py 2026-11-20` 进程也还在，相关父模型、资格和状态文件列为保留。第一批和第二批候选共 434 个文件，清单已逐路径核对存在且大小符合快照。本轮只生成清单。未实测逐批重建、未验证远端部署实际版本和异地备份，`待核对` 的权重不能据此执行删除。
