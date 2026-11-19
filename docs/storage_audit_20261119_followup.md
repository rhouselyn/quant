# 清理后存储复查与下一批候选

2026-11-19，北京时间。本轮为只读盘点，项目内新增清单和报告。上轮已释放 10.71 GiB。复扫约 76,045 个文件，按文件 ID 扣除硬链接后约 57.11 GiB。以下释放量为估计，未执行合并、归档或删除。

[逐路径清单](storage_audit_20261119_followup.tsv)列出 932 条记录，包括原路径、同内容路径或归档所需输入、预估字节及可核对的 SHA-256。

## 滚动种子与 dataset_v2

滚动种子 `outputs/rolling_anchor_seeds_v1` 当前约 **10.46 GiB**，`outputs/dataset_v2` 约 **7.96 GiB**。这两个目录的下一轮候选合计约 **8.58 GiB**。其中约 **3.69 GiB** 来自重复文件、小账本及透明压缩，约 **0.24 GiB** 关联当前运行输入，约 **4.65 GiB** 要先建立和验证可恢复的排名归档。

| 对象 | 预计可省 | 核对结果与处理条件 |
| --- | ---: | --- |
| 滚动种子 `seed43–46/report/orders/` | **4.65 GiB** | 尚占 6.22 GiB，96 条路径对应 64 份独立文件。按种子、阶段、arm 划成 16 组，每组保留 `full_L20.parquet` 和其余三份文件在共同前缀之后的原始字节，保留文件名映射。已在内存对 48 份独立文件及 64 条对应路径逐项核对，拼回后的 SHA-256 都一致。预计归档后占 1.57 GiB。需要先做小范围归档及按原路径恢复验证，日常查账继续读既有周账本。 [缓存生成及读取](../scripts/run_rolling_anchor_seeds.py) |
| `dataset_v2/archive_completion_v1/secondary_status_ohlcv/raw/` | **约 2.61 GiB** | 3,504 份原始 JSON，当前约 3.64 GiB，二源分页验收涉及原始回执及散列。建议原路径 NTFS 透明压缩。100 份按文件大小分层取样，模拟约 72% 释放，另取 10 份临时副本实测系统压缩与散列不变。保留原始响应和归档清单。与 `data/ashare_v2` 的重复也有发现，避免将两项收益相加。 [回执证据](research_20261113_archive_completion_results.md) [散列校验](../outputs/dataset_v2/archive_completion_v1/secondary_status_ohlcv_worker.py) |
| `dataset_v2/{data_trials_v1,data_trials_extension_v1,repaired_bridge_v1}/runs/` | **0.64 GiB** | 58 组独立 checkpoint 副本，含同一个 D3/D3_ZERO 配方在多个批次的相同轮次。可在确认相关 run 停写后合并存储，所有 checkpoint 名、选轮和原哈希保留。`training_state.pt`、独有权重和各批结果另留。 [修补对照](research_20261114_gap_followup_results.md) |
| 滚动种子同个 run 的 `best.pt`、`newest.pt` 和对应 `epoch_*.pt` | **0.24 GiB** | 42 组，44 份可合并的独立副本。全部核对了原 SHA-256，现行写入走原子替换。原文件名、所有选择轮窗、父主干和训练状态继续保留。 [权重写入](../src/quant_mvp/hierarchy_engine.py) |
| `dataset_v2` 关联运行的重复快照与回放带 | **0.24 GiB** | `BQ_F0`、`BQ_F0_dated`、`BQ_F0_r2` 的 `weekly.parquet` 三份相同，仅该组约 207 MiB。`replay_inputs` 与 `replay_inputs_full` 的 `raw_open.parquet` 也同内容，另有与 `standard_v1` 配对的重复标签。当前队列读取其中的 Q1 快照、标准快照及回放带，列入等队列结束后复核的范围。每个快照目录及 manifest 都保留。 [当前队列](../outputs/p4_muon_seeds_v1/state.json) |
| `dataset_v2` 其余冻结快照和评分 | **0.19 GiB** | 包括 `BQ_F0_outer` / `BQ_F0_outer_r2` 的 111 MiB 行情表和若干同内容的 D3 配对分数。停写确认后可按文件合并，保留不同快照各自的元数据。 [快照生成](../scripts/build_dataset_snapshot.py) |
| 滚动种子 `seed43–46/report/ledgers/` 小账本 | **0.013 GiB** | 384 份小表逐项核对，可从同 seed 的 `weekly.parquet` 按阶段、账本、arm、费率切回。`long_w`、`short_w` 需按原 dtype 还原，重写文件本身的散列未验证。 [账本发布](../scripts/report_rolling_anchor_seeds.py) |

直接删掉全部 6.22 GiB 排名缓存需要原 CUDA 路径完成全量复算验收。初查中，seed43 B0 验证抽三周重建完全一致，seed43 PAIR_REPLACE 历史抽三周有一张表的 `score_long` 约 0.115% 行出现名次差异。因此推荐用能恢复原字节的归档方案，6.22 GiB 全删方案未计入合计。原 `comparisons.csv`、周账本、候选锁、裁判头、全部轮窗、评分、向量和上游输入保留。 [滚动种子结果](research_20261118_rolling_anchor_seeds_results.md)

## 其他较大批次

| 目录 | 当前大小 | 另可省 | 核对结果 |
| --- | ---: | ---: | --- |
| `data/` | 8.13 GiB | 暂缓 | `ashare_v2` 约 2.72 GiB，含发布归档及股票和期权原始输入。发现部分 BaoStock 响应和 `dataset_v2` 内容一致，跨库去重暂缓，避免与透明压缩重复计量。 [归档与查询](option_archive.md) |
| `rank_blend_v1`、`linear_seed_ensemble_v1` | 0.83 / 0.78 GiB | **0.47 GiB** | 混合 0%/100% 两端与直接/线性集成的订单表存在同内容副本。线性集成目前直接写订单路径，待改为原子替换或冻结重跑入口，再考虑保留路径去重。此处已扣除上一轮合并的副本。 [线性集成](../scripts/replay_linear_seed_ensemble.py) [混合回放](../scripts/replay_rank_blend.py) |
| `bce_fused_three_v1` | 1.88 GiB | **0.42 GiB** | seed42 的 B0/P6 训练及测试向量和 `bce_native40_transfer_v1` 的 4 份数据逐字节相同。两批写入方式需要复核，旧 40/10 复盘还会校验原路径和散列。其余向量、头和证据保留。 [融合写入](../outputs/bce_fused_three_v1/prepare.py) [固定旧名单复盘](../outputs/bce_old_roster_vectors_v1/run.py) |
| `weekly_architecture_v1`、`weekly_followup_v1` | 1.89 / 0.98 GiB | **0.055 GiB** | 30 组同内容轮次和评分缓存。核对写入方式后可合并，选轮/父模型仍按老路径保留。跨模型的大范围权重清理不在本次候选内。 |
| `rolling_anchor_rerank_v1` | 1.05 GiB | 暂缓 | 向量和裁判头仍是五种子发现臂及下游输入。小规模权重重复有发现，本轮未纳入清单。 [滚动精排结果](research_20261118_rolling_anchor_rerank_results.md) |

上表另有约 **0.95 GiB** 的已验证同内容空间，执行还需各自通过写入方式核对。8.58 GiB 核心候选与这 0.95 GiB 不重叠。估算不把已有硬链接、原始回执跨库重复或 6.22 GiB 直接删缓存方案重复相加。

## 保留范围

15:44 核对时，P4 与 Muon＋Squared ReLU 五种子还在训练 seed43 的实验臂。监督进程、最新日志、状态及 GPU 锁一致。实盘名单仍在等待 11 月 20 日收盘数据。保留整个新批次、`BQ_F0_r2`、标准验证与历史快照、回放带/融券资格和所有已登记依赖。保留实盘两版与默认入口、baseline、各方向实际入选权重、窗口、预处理和完整训练状态。此次权重候选只合并同内容物理副本。 [P4 运行交接](research_20261119_p4_muon_seeds_results.md) [实盘候选说明](storage_audit_20261119.md)

上轮清理移除了 `P4_MUON_SQRELU` seed42 的权重和完整训练状态，随后该臂被选作新低回撤复核的发现成员。检查了 145 个大小匹配的本地 `.pt`，没有找到同 SHA-256 的副本。现队列经审计以保留的 seed42 评分、锁和清理散列链回放，该种子无法直接重新 forward 或续训。新增 seed 权重和完整状态纳入保护。 [P4 运行交接](research_20261119_p4_muon_seeds_results.md)
