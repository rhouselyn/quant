# 基准操作文件：monopen 30 特征标准训练验证流程（2026-11-04 修订）

本文件是仓库的**当前默认臂**：任何新想法都以此为对照、相对它只改一处、用同一套流程验证。
配套：配置 `configs/hierarchical_580_monopen_baseline.yaml`，一键入口 `scripts/run_monopen_baseline.py`。
历史全账见 [experiment_ledger.md](experiment_ledger.md)；本文件只固化"标准流程"。

## 基线切换记录（2026-11-04，第四次）

数据集保持 **akshare hfq ak2000**（2000-01-07..2026-10-23，1399 周），模型回到
**原始 monopen 30 特征配方**：架构、30 特征、损失、8 epochs、切分日历
（train 0..938 / val 939..1095 / test 1096..）与 2004-yahoo 旧基线一致，只换了
数据集与起点。冻结参考账本 = `merge_abl2000_base_3seed`（3 seed × best±1 ens3
的跨 seed z 融合，公共窗 283 周 2021-01-08..2026-10-09，monday-open）。

依据是 abl2000 2×2（{2000,2004 起点}×{30,32 特征}，每格 3 seed，台账 §19）：

* **fin2（32 特征）被淘汰**：同起点 C−D RankIC +0.0069 (t=3.40)；四格中纯多书
  最差（+26.0bps / Sharpe 0.850）。
* **base30（30 特征）在 2000 与 2004 起点纯多打平**（35.2 vs 34.9, t=0.06）；
  RankIC 2000 起点 0.0835、2004 起点 0.0855。
* 本轮冻结 **2000 起点**（当前数据集 + 原始模型）；2004 起点 `merge_abl2004_ak_base_3seed`
  作为并列实验臂保留。
* 上线候选（recency 切分，非本 baseline）= `abl2000_base_r2023` 5seed，
  训练→2023-10、验证 2023-11..2026-08。

**指针变化**：canonical 配置 = `configs/hierarchical_580_monopen_baseline.yaml`
（现内容与 `configs/abl2000_base.yaml` 逐值一致，30 特征）；`REFERENCE_RUN =
merge_abl2000_base_3seed`。fin2 配方归档
`configs/hierarchical_580_monopen_baseline_fin2_20261104.yaml`；2004-yahoo 配方
`hierarchical_580_monopen_baseline_ak2004_20261103.yaml`；mamba 配方
`hierarchical_580_monopen_baseline_mamba2_20261024.yaml`。

## 历史：第三次切换（2026-11-03，已被第四次推翻）

当时把数据切到 ak2000 同时把特征 30 → 32（+`fin_npgr` +`fin_ocfy`），冻结参考
`abl2000_fin2_ens3`。该判决只基于单 seed 的 epoch 融合；3seed 2×2 显示 32 特征
纯多书最差、RankIC 显著低于同起点 30 特征（C−D t=3.40）。fin2 不再作为 baseline。
数据维护仍走 `scripts/update_akshare_data.py`（见 [akshare_pipeline.md](akshare_pipeline.md)）。

## 0. 一键流程

```powershell
python scripts/run_monopen_baseline.py
```

依次执行：**训练（8 epochs）→ ens3 融合（best±1）→ 测试窗汇总 + 参考读数复现检查 →
确保 dashboard 在 8501 → 浏览器打开本臂 ens3 页面**。幂等：已完成的步骤自动跳过。
`--dry-run` 只打印计划；`--skip-train` 跳到融合；`--seed 43` 跑复现臂（见 §6）。

## 1. 数据与执行口径

| 项 | 值 |
| --- | --- |
| 数据源 | **akshare 后复权（hfq）**：新浪日线（`scripts/download_akshare_580_daily.py`）+ 东财季度财务（`download_akshare_580_fin_em.py`）；增量维护 `scripts/update_akshare_data.py` |
| 缓存 | `data/sw_580_observed_features_2000_fri_path30/weekly.parquet`（1399 周 × 580 股，2000-01-07 ~ 2026-10-23；价格 hfq、量额 raw） |
| 宇宙 | 申万一级 29 行业 × 20 只随机股（当前快照回溯），等权基准 = 全体 580 |
| 锚点 | 周五收盘打分（W-FRI bar，含 6 个日径特征 d_on/d_in/dfront/d_onn/d_back/d_mon） |
| 执行 | **周一开盘**：`label_mode: open_open`，下周一开盘买入、再下周一开盘卖出；
| | 跌停/缺失开盘回落到当周收盘；入场要求开盘可成交（缺口低于涨停板幅度） |
| 成本 | 单边 10bps；Top-20 等权；缺失收益记平并计数 |

## 2. 特征（基线 30 个价量；2 个财务特征已退役）

价量 30（全部 OHLCV 派生）：`ret1/2/5/10/20/60、oc、hl、close_pos、vol5/20/60、
mom5/20/60、vlog、vchg、volume_z5/20/60、range5/20/60、amount_z20` + 6 个日径
`d_on/d_in/dfront/d_onn/d_back/d_mon`。管线内逐周横截面 z-score（clip ±6），
停牌 last-known-value carry。

财务 2（**已从 baseline 退役**（2026-11-04，fin2 被 2×2 淘汰），保留为归档参考。
`scripts/add_fin_features.py`，东财 yjbb+zcfz，**PIT 法定披露窗**对齐：
Q1+30d / 半年报+62d / Q3+31d / 年报+120d——EM 的 ann_date 是刷新时间戳不可用，
中位偏移 ~390 天；缺失用训练窗中位数填充；分母 = akshare **未复权**周收盘）：

| 特征 | 定义 | 入选依据（增量 IC，Frisch-Waugh 于 30 特征基底 + 行业中性化） |
| --- | --- | --- |
| `fin_npgr` | 净利润同比增长率（np_yoy） | 条件增量 t=6.82，H2(2020后) t=1.93，核心稳固 |
| `fin_ocfy` | 每股经营现金流 / 未复权价（= OCF 产出率） | 中性化 H2 t=2.25（全场唯一>2）；与 npgr/bp 秩相关仅 0.09/0.17；H1 弱（t≈0.57），属近年风格 |

**已裁决的输入侧变体**：`fin_bp`（bps/未复权价）在 fin3 消融中比 fin2 差 7.4bps/周
(t=−1.67)，杀（2026-11-03，台账 §18）；fin2 整体（+npgr +ocfy）随后被 3seed 2×2 淘汰
（同起点 RankIC C−D +0.0069 t=3.40），杀（2026-11-04，台账 §19）；
36 特征（+6 Alpha360 探针）paired −14.4bps t=−1.44，杀（2026-10-24）；
60 日原始日频分支（daily12）输给 30 特征，杀（2026-09）。

## 3. 切分与模型

| 项 | 值 |
| --- | --- |
| 训练 | 索引 0..938（2000-01-07 ~ 2017-12-29），有效训练从第 40 周（lookback）起 |
| 验证 | 索引 939..1095（2018-01-05 ~ 2021-01-01），157 周 |
| 测试 | 索引 1096..1398（2021-01-08 ~ 2026-10-23），303 周 |
| 编码器 | 门控注意力 attn（Qwen3-Next 风格 RoPE + QK-RMSNorm + sigmoid 输出门），d_model 128、4 头、4 层、embedding 128、输入 40 周（2026-11-01 起；旧 Mamba2 d96 已归档） |
| 关系层 | L-G-L-G-L 行业注意力，**门冻结为恒等**（init 0.0），1 个市场 token × 6 市场特征 |
| 其它 | lookback 40、batch 16、dropout 0.1、min_history_weeks 32、eval_batch_dates 4 |

## 4. 损失与选模（总损失 = 各项加权和）

| 项 | 权重 / 参数 |
| --- | --- |
| extreme score（头尾各 20% 等权） | 1.0（score_top/bottom_weight 均 1.0） |
| 全量 RankIC | 0.5（温度 0.05） |
| masked InfoNCE（h=5，EMA 0.99，时序塔上） | 0.1 |
| 行业暴露 η² 惩罚 | 1.0（零假设 ≈ 0.0484） |
| 关闭项 | group 0、block_rank 0、stickiness 0 |
| 优化 | AdamW lr 8e-4、weight decay 1e-5、**8 epochs**、seed 42 |
| 选模 | 验证集 Top-20 纯多头净 Sharpe argmax（噪声大：详见 §7 读数规则）。**替换该指标的三次反事实全部证伪**（val_rank_ic、无验证窗口、缓冲账本 Sharpe）；缓冲账本 Sharpe 选轮 = 测试 IC 显著更低（−0.0055，t=−3.07）+ 账本不可分辨，见台账 §15 |

## 5. ens3 融合（默认规则：best ± 1）

`python scripts/ensemble_best_window.py <run> --rule window`（训练须 `save_epoch_scores: true`，
脚本 `--rule top3` 是另一个规则——全 epoch 验证 Sharpe 前三，部署臂曾用；**本基准默认 ±1**）。
边界窗口：`window` 规则 best=1 → [1,2]；`window3` 规则（E1 guard，与实盘
`live_score.epoch_window` 一致）best=1 → [1,2,3]。冻结参考的 3 个 seed 都没有 best=1，
两种规则等价；实盘打分用 `window3`。融合 = 各轮分数逐周横截面 z 后等权平均，只用验证集选轮。

## 6. 多种子扩展（默认单 seed 42，复现是额外步骤）

```powershell
python scripts/run_monopen_baseline.py --seed 43   # -> real_hierarchical_580_monopen_baseline_s43
python scripts/run_monopen_baseline.py --seed 44
```

读法纪律（台账 §7-B3/B4）：**跨 seed 复现检查 = 逐臂对基准的周度 paired t**。
冻结参考书本身是 **3 seed × ens3 的跨 seed z 融合**（匹配旧 3seed 参考的口径）；
单 seed 复现是额外步骤，不改变基准定义。执行层规则、单点增益主张，一律跨 seed 复检后才算数。

## 7. 读数规则（判读任何一次训练结果）

1. **先报周度 paired t 和胜率**，再谈 Sharpe/IC 排序；两臂差异 <33bps/周在本样本不可分辨（SE≈16.6）。
2. **RankIC 与账本解耦是常态**（x6 臂：IC 微升、账本 −14bps；stick3_cap_zone：最高 IC 最差书；反向也成立：台账 §15 显著更低的 IC 换出不可分辨的书）。
3. **绝对超额必须连同流动性匹配基准读**（台账 §4：多头账面超额大半是流动性/行业倾斜）。
4. 验证 Sharpe 选模有赢家诅咒：逐轮摆动可达 ±0.7；报单臂时附全轮 val 曲线。
5. 派生臂纪律：复制基准配置、**只改一处**、跑前预注册 keep/kill、对照跑同一 ens3 流程。
6. **编码器方向已关闭（2026-11-02）**：特征轴倒置（iTransformer 式 B1）与窗口内非因果注意力（TSMamba 式 B2）均已实测 KILL；B2 的单 seed 正号在三 seed 反转。详见台账 §14。不要重开，除非先换卷积 tokenizer 并加独立控制臂。

## 8. 参考读数（2026-11-04 冻结，canonical 切换后）

| 臂 | 窗口 | RankIC | 净年化 | 净 Sharpe | maxdd | 换手 | 超额 bps/周 (t) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **base30 3seed × ens3**（`merge_abl2000_base_3seed`） | 2021-01-08..2026-10-09，283 周 | **0.0835** | **27.31%** | **1.011** | −29.0% | 1.038 | **+35.2 (3.62)** |

出处：`outputs/merge_abl2000_base_3seed`（seed 42/43/44，ak2000 数据集 + **30 特征**，
monday-open、Top-20、10bps）。这是**原始 monopen 模型在当前数据集上的等价基线**：
架构/特征/损失/8 epochs 与 2004-yahoo 旧基线一致，只换了数据集与起点。

**已归档/不采纳**：

* fin2（32 特征）s42 ens3：0.0811 / 25.59% / 0.972 / +32.3 (3.32) —— 单 seed 口径下曾
  是参考；但 3seed 2×2 显示 fin2/32 纯多书最差（+26.0 / 0.850），已被 2026-11-04 切换淘汰。
* fin2 3seed `merge_abl2000_fin2_3seed`：0.0809 / 21.5% / 0.850 / +26.0 (2.80)。
* 旧 2004-yahoo 参考 `merge_monopen_attnenc_d128h4_3seed`：0.0817 / 21.87% / 0.878 /
  −31.2% / 0.688 / +26.2 (2.86)，保留作历史归档。
* **上线候选（recency 切分）**：`merge_abl2000_base_r2023_5seedw3_val`
  （验证窗 2023-11-03..2026-08-28，142 周：0.0824 / 25.5% / 0.850 / −30.0% / 0.90 / +33.2）。
  注意 epoch 选择与验证窗同窗，轻度内生性；不对应本 baseline 的 283 周测试窗。
* 空头缓冲：3seed 参考书 K=30 最稳（验证 89.6 / 测试 62.3 @C）；5seed recency 书 K=50 最优。
  见 `outputs/three_seed_buffer_valtest.csv`、`outputs/k_sweep_books.csv`。
* 该臂受 §7-3 流动性告诫约束，绝对数按"相对全体等权基准"口径交付。

## 9. 文件清单

| 文件 | 作用 |
| --- | --- |
| `configs/hierarchical_580_monopen_baseline.yaml` | 基准配置（ak2000 + **30 特征**；派生臂的唯一起点） |
| `configs/hierarchical_580_monopen_baseline_fin2_20261104.yaml` | 已淘汰的 32 特征配方归档 |
| `configs/abl2000_base.yaml` / `abl2004_ak_base.yaml` | 30 特征基准（2000 起点 / 2004 起点并列臂） |
| `configs/abl2000_fin2.yaml` / `abl2000_fin3.yaml` | 已淘汰消融臂归档（32 / 33 特征） |
| `configs/abl2000_base_r2023.yaml` | 上线候选配方（recency 5seed，训练→2023-10） |
| `scripts/run_monopen_baseline.py` | 一键标准流程（训练→ens3→对比→开网页） |
| `scripts/ensemble_best_window.py --rule window` | ens3 ±1 融合（被一键脚本调用） |
| `scripts/update_akshare_data.py` | 数据增量维护（行情+季度财务+缓存重建+fin 特征，见 [akshare_pipeline.md](akshare_pipeline.md)） |
| `scripts/download_akshare_580_daily.py` / `download_akshare_580_fin_em.py` | akshare 全量下载（新浪日线 hfq/raw + 东财季度财务） |
| `scripts/build_raw_ak2000.py` / `add_fin_features.py` | 缓存构建与 PIT 财务特征注入 |
| `scripts/run_abl2000.py` / `queue_abl2000_recency.py` | 消融三臂 / recency 5seed 队列（幂等） |
| `scripts/build_abl2000_compare.py` | 对比网页（`outputs/abl2000_compare/compare.html`） |
| `outputs/merge_abl2000_base_3seed` | 新基线账本（冻结参考，3seed × ens3） |
| `outputs/abl2000_base_r2023{,_s43..s46}` | 上线候选/部署权重（5 seed，recency） |

---

# 组合策略基线：双年双版分数融合 + 缓冲多头 + 顺延空头（2026-11-05 冻结）

本文件前半部分是**模型层**基线（单模型配方与对照纪律）。本节是**策略层**基线：
上线交易的就是这套，任何组合/执行层的对照都相对它只改一处。依据与全账见
[experiment_ledger.md](experiment_ledger.md) §22（22.1 分腿 / 22.2 融合 / 22.3 组合矩阵 /
22.4 选轮窗）。

## 1. 规则（冻结，勿逐周改动）

| 项 | 值 |
| --- | --- |
| 打分 | **两个版本等权融合**：最新版 + 上一版，各自 5 seed × best±1 轮，先各自横截面 z 再平均 |
| 版本节奏 | **双年**（每 2 年训一版）；当前对 = `abl2000_base_r2021` + `abl2000_base_r2023` |
| 多头 | Top-20 等权，缓冲 K=100（跌出全市场前 100 才卖），每周只补缺口 |
| 空头 | 活跃池（成交额 q4+q5）最差 10 只，**当季两融名单过滤 + 顺延补位**，出场圈 = 活跃池 bottom-30 |
| 权重 | 两腿各占账户 50%，腿内等权（多头每只 5%、空头每只 10%）；手数取整、每票 margin_ratio |
| 成本口径 | 单边 10bps，按换手计费；空头借券费 ~8%/年**未扣**（研究口径） |
| 多空重叠 | 两腿共用**同一份融合分数** → 排名带天然不相交；代码仍双向护栏（多头持仓不进空头候选，反之亦然） |

## 2. 为什么是"融合"而不是"每年重训一套新权重"

- 年度重训（3 年验证）对 2018 截止固定臂：多头 +3.2bps/周 t=0.60（UNRESOLVED），
  单 seed 口径全负 → 没有 alpha，只有"老模型塌方时救场"的保险价值（§22）。
- 分数层融合两个相邻版本：2020-26 多空年化 48.3% vs 每年最新单模型 43.2% vs
  固定 2015 版 36.3%；**逐 seed 最稳**（39.6/46.0/40.6），训练量减半（§22.2）。
- 增益几乎全在**空头腿**（融合 64.0% vs 固定 36.2%）；多头各模式统计打平
  （|t|<0.15），所以多头不必单独优化（§22.3 七种多头 × 六种空头全扫一遍，无胜者）。
- 选轮：实盘走 `reselect_epochs.py` 全标注窗重选（旧版本每年可重选，不需要重训）；
  4 年窗 vs 3 年窗 +1.3bps t=0.22，无害但不显著（§22.4）。

## 3. 冻结读数（3 seed 融合书，净 10bps，2020-01..2026-10，334 周）

| 腿 | 年化 | Sharpe | MDD |
| --- | ---: | ---: | ---: |
| 多头缓冲100 | 24.7% | 1.0 | −28.1% |
| 顺延空头10只 | 64.0% | — | — |
| **多空组合** | **48.3%** | **3.0** | **−12.3%** |

对照：固定 2015 版组合 36.3%（−11.9%）；每年重训单模型 43.2%。逐 seed 组合年化
39.6 / 46.0 / 40.6。**必须同时声明**：空头腿未扣借券费（≈ 组合 −4pp/年）、
券商实际券源未验证、2020-22 落后固定臂、增益集中在 2023 之后。

## 4. 上线流程（服务器 152.136.96.212）

```bash
# 本地：训练上一版（无 GPU 的服务器只打不训）
python tmp_base_r2021_queue.py            # 5 seed，~2.5h

# 上传两个版本的 run 目录（含 epoch_*.pt），保留目录结构
# 服务器 weekly_qq.sh 里给 live_pipeline.py 追加：
#   --prev-runs outputs/abl2000_base_r2021 outputs/abl2000_base_r2021_s43 \
#                outputs/abl2000_base_r2021_s44 outputs/abl2000_base_r2021_s45 \
#                outputs/abl2000_base_r2021_s46
# 建议先平行记录 4-8 周（同时出单版本与融合版的书），再彻底切换
```

## 5. 变更纪律

- 版本节奏、融合权重数、缓冲 K、空头规则**四者任一动，都算新策略**，必须整轮重测
  （含逐 seed 与 ≥3 年样本），并在台账新开一节。
- 每训完一个新版本，上线前必须 `python scripts/reselect_epochs.py --base <run>`。
- 单版本信号层实验（特征/标签/架构）继续按本文件前半部分的模型层 baseline 做对照，
  **不要**拿策略层读数去判定模型层结论。

## 6. 研究基线 v2（2026-11-10）：PIT 名单 + 选轮窗裁尾

两项**正确性修正**已进入默认口径（不重新训练、不改变已冻结产物）：

1. **空头名单改为 PIT**。`scripts/skipfill_short.py:load_predictions_with_liquidity`
   默认 `pit=True`：决策落在 Q 季度只用 Q-1 季末快照。旧实现把季末名单套整季
   （1—3 月用 3 月 31 日才公布的名单），属前视。`pit=False` 仅用于复现 2026-11-05
   冻结数字。季末当天决策取上一季（比"当天快照"更保守），与手写 PIT 实现仅 5 行差异。
2. **选轮窗裁末 2 个 val 锚点**。`selection_label_lag_weeks: 2`（新配置键，引擎默认 2）：
   t+2 开盘标签在首个 test 决策之后才成熟的锚点不参与选轮。报告列
   `val_long_net_sharpe` 仍为全窗（与历史可比），选轮用新列
   `val_long_net_sharpe_selection`；`ensemble_best_window.pick_epochs` 优先读该列。

**实测影响**：三个 seed 的 best epoch 全部不变（42:3/[2,3,4]、43:2/[1,2,3]、
44:6/[5,6,7]），长腿与冻结基线逐位一致（配对差 0.0bps，283 周）。

研究基线 v2（test 283 周，10bps）：

| 口径 | 纯多 | 纯空 | 50/50 combo |
|---|---:|---:|---:|
| 旧（同季名单） | 27.31% / 1.011 | 14.78% / 0.581 | 25.57% / 1.766 / −19.68% |
| **v2（PIT）** | 27.31% / 1.011 | **16.30% / 0.621** | **26.41% / 1.826 / −19.68%** |
| v2 + 8%/yr 券费 | 同上 | 7.35% / 0.381 | 21.45% / 1.527 / −21.81% |
| **v2 + Top-40 长腿** | 27.72% / 1.055 | 16.30% / 0.621 | **26.51% / 1.905 / −17.85%** |
| 上 + 券费 | 同上 | 7.35% / 0.381 | 21.55% / 1.593 / −19.77% |

产物：`outputs/baseline_v2_research/`（定义、曲线、逐年、配对）。
旧 `outputs/baseline_stack/` 保留为 2026-11-05 冻结版本，不再覆盖。
部署（服务器 live_pipeline）仍按 §5 的版本融合与缓冲 K=100，本文件不改变线上。

## 7. 多空研究基线 v3（2026-11-13，归档对照）

多空实验的主对照升级为 **B0 多头 + 完整 CS 空头**。B0 选 Top-20，CS 同时使用行业与市场横截面摘要，在成交额前 40% 且有时点名单的股票中选 Bottom-10。两腿各按 50% 名义预算、各腿等权，排除多空同股。使用已冻结的三种子分数与轮窗 `[2,3,4]`、`[1,2,3]`、`[5,6,7]`，采用共同覆盖的 matched B0。该配置用于多空研究回放，生产参数见本文件「组合策略基线」。

```powershell
python scripts/followup_replay_baselines.py --baseline v3 --out outputs/baseline_v3_research
```

`metrics.json` 和 `weekly.parquet` 的 `LS50/vs-CS` 行给出联训或 P64 空腿对新基线的直接配对。`LS50/cash` 使用同一 B0 多腿加 50% 现金。`LS50` 下直接接费用数字的键保留旧 B0 空腿对照，供追溯。每项比较共用决策日、股池、长腿、首个市场会话和费用，覆盖 10/30 bp 单边交易费和 8%/12% 年券费的现金与新基线比较。输出路径已存在时须指定新目录。

完整 CS 在相对旧空腿的历史比较中有明显增益。相对同多头加现金，历史净差在 10 bp/8% 为 +14.13 bp/周，在 30 bp/8% 为 −2.22，在 30 bp/12% 为 −6.12。CS×P64 联训空腿在 10 bp/8% 下比完整 CS 多 +3.48 bp/周，8 周块区间为 [−2.62, +6.59]，保留为挑战者。沪深同覆盖验证为 38 周，2021–2026 历史复检为 295 周。固定股票池、静态行业、代理券源与受阻退出代理仍适用。

[运行记录](research_20261113_followup_execution_results.md) · [新基线回放入口](../scripts/followup_replay_baselines.py)

## 8. 多空研究基线 v4

默认策略为 **B0 多头 + CS 第 4–6 轮空头**。B0 沿用 seed 42/43/44 的 `[2,3,4]`、`[1,2,3]`、`[5,6,7]`；CS 三个 seed 均使用 `[4,5,6]`。各腿独立标准化并融合自己的 checkpoint 分数，分别选 Top-20 和符合流动性、时点券源门的 Bottom-10，再按 50/50 名义预算组合。CS 的轮窗由各 seed 的纯空验证净 Sharpe 直接从六个连续三轮窗中选出。`[4,5,6]` 对应本次冻结的三个 run，后续新 vintage 按相同规则重新验收。

```powershell
python scripts/followup_replay_baselines.py
```

默认写入 `outputs/baseline_v4_research_835/`。主档年化融券费率为 **8.35%**，单边交易成本用 10/30 bp，另报 12% 年券费压力档。`B0+CS_WIN3/LS50/vs-v3` 是相对旧研究基线的直接配对，候选和对照用同一费率；`B0+CS_WIN3/LS50/cash` 是相对同一半仓 B0 多头加现金的配对。后续空腿实验看 `LS50/vs-CS_WIN3`。旧 v3 仍可用 `--baseline v3` 复现。

按年券费 8.35% 重放。单边 10 bp 的 295 周历史复检相对同费率 v3 多 2.93 bp/周，Sharpe 从 1.728 升至 1.871，最大回撤从 12.27% 降至 10.12%，相对同多头加现金多 16.72 bp/周。单边 30 bp 时，相对 v3 多 3.10 bp/周、相对现金多 0.54 bp/周。30 bp/12% 时相对现金为 −3.01 bp/周。空腿轮窗在 38 个沪深同覆盖验证周内选择，历史主档相对 v3 的 8 周块区间为 [−7.66,+7.84] bp/周。最初年券费 8% 的读数归档在 `outputs/baseline_v4_research/`。生产策略沿用既有配置。

[轮窗选择](../outputs/weekly_followup_v1/eval/direct_win3_by_leg_v1/selection.json) · [v4 回放](../outputs/baseline_v4_research_835/metrics.json)
