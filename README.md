# quant MVP

这是一个可运行的周频量化原型：合成/真实 A 股数据 → 技术特征 → Mamba2 风格时序编码器 → 时间 InfoNCE + 极端分位 score + 全量 RankIC loss → 样本外多空和 Top-20 多头回测与图表。

## 默认配置与快速运行

当前默认启动配置是 `configs/hierarchical_580_h2.yaml`：**580股、29行业各20股、一个市场token、L→G→L→G→L分层注意力、单score头、10epochs**。时间编码器保持40周输入和128维embedding。Score头尾20%等权、全量RankIC权重0.5；未来两周InfoNCE权重0.1，仅作用于关系层前的时序表示与独立投影头。独立EMA包含时序编码器和投影头。固定对照同时包含关闭InfoNCE及未来五周权重0.1；全部完成后按验证Top20净Sharpe更新默认配置。

使用 `data/sw_580_observed_features/`，从原始日线及2016年前历史修复滚动特征；滚动窗口以实际观测周计数，日历空周仅向前携带过去特征。原始OHLCV及下一日历周收益标签不填造，旧580/600/500缓存保留。市场token另输入6项当时可见市场状态，按训练期统计标准化。架构说明见 [实现与实验记录](docs/hierarchy_implementation.md)。

| 参数 | 默认值 |
| --- | --- |
| 周线观测 | 2016-09-30 ～ 2026-09-11，520个缓存周观测 |
| Query / temporal key | 最近40周 / 初始未来2周单key；五周为对照 |
| 模型 | torch_mamba2 + LGLGL行业注意力 + 一个市场token，128维，单score头 |
| Score loss | 收益前后各 20%，头部/尾部等权（1.0 / 1.0），中间忽略 |
| RankIC loss | 全量横截面 RankIC |
| 总损失 | 初始Score + 0.1×两周InfoNCE + 0.5×全量RankIC；关闭/五周对照 |
| 选模 | 验证集 Top-20 纯多头净 Sharpe，平局保留较早轮次 |
| 权重文件 | 仅 best.pt / newest.pt |
| 关闭 | 随机窗口、多正样本、块间排序、去相关 |

```powershell
python scripts/run_hierarchy_comparison.py
```

历史800股配置和数据仍存于 `configs/real_csi800_weekly.yaml` 与 `data/csi800/`，不再是默认。官网最新成分回溯有存续偏差；上市前/缺失周保持缺失，收益标签屏蔽。短历史股票缺失输入按现有流程置零，时间 InfoNCE 尚未加入历史可用性掩码。

**中信一级30行业分类尚未下载，当前标记为待补。** 已确认数立方提供 `a_index_memberscitics`（中信一级指数成分）和 `ashare_ind_class_citics`（股票中信分类）API，调用需要 token 和相应权限；详见 [数立方接口核对](docs/datacube_industries.md)。`a_cni_industries` 是国证行业，不能替代中信。Tushare `ci_index_member` 也可作为授权来源。行业暂不参与训练损失；获得完整映射后可运行：

```powershell
python scripts/import_citic_industries.py --mapping path/to/citic.csv --source "数据提供方" --asof YYYY-MM-DD
```

输入 `code,citic_l1` 或 Tushare `ts_code,l1_name`，要求800股全部唯一覆盖并符合中信30行业名称。分类是元数据，不执行行业中性化。

旧数据和小范围测试入口全部保留：

```powershell
python scripts/run_mvp.py --config configs/default_500_top20.yaml --output outputs/test_500
python scripts/run_mvp.py --config configs/real_contrastive_weekly_100_yahoo.yaml --output outputs/test_100
python scripts/run_mvp.py --smoke-test --synthetic --output outputs/smoke
```

`default_500_top20.yaml` 保存切换前的500股票8轮默认参数；100/500历史缓存均不覆盖。每次运行请用独立输出目录。

## 数据

单头风格分析：运行 `python scripts/analyze_single_score_style.py`，结果在 `outputs/single_score_500_style_analysis/`，网页入口为 `/style-analysis`。复用基线epoch8预测，比较7个历史行情因子的50%/100%中性化，以及行业快照回溯对照。原始Top20测试净收益36.51%、Sharpe1.167；行情完全中性化后35.67%、1.122，回撤从20.44%扩大至25.74%。验证期也支持保留原始分数，因此默认不启用中性化。该检验不覆盖市值、估值等历史基本面，行业快照有后验信息；不是完整风险中性策略。

默认分层训练使用 `data/sw_580_observed_features/weekly.parquet` 与同目录市场特征；构建脚本 `python scripts/prepare_hierarchy_580.py`（已完成，不重复覆盖）。原580股数据保留在 `data/sw_580_29x20/`。行业仍是2026年快照回溯；注意力候选mask来自当前行情与历史覆盖，未来标签mask仅用于训练监督。`--synthetic` 可切换可复现合成数据并使用独立缓存；其他配置仍支持 Baostock（免费前复权日线）。不能把合成回测结果当作真实交易结论。

真实数据（Baostock，免费前复权日线）运行：

```powershell
python -m pip install baostock
python scripts/run_mvp.py --config configs/real_smoke.yaml --real --output outputs/real_smoke
python scripts/run_mvp.py --config configs/real_medium.yaml --real --output outputs/real_medium
python scripts/run_mvp.py --config configs/real_long20.yaml --real --output outputs/real_long20
python scripts/run_mvp.py --config configs/real_long.yaml --real --output outputs/real_long
```

Baostock 下载可能受网络和服务端限流影响；下载结果会缓存到 `data/`，重复运行不会重复请求。

## 前端面板

```powershell
python dashboard/app.py --output-root outputs --port 8501
```

打开 <http://127.0.0.1:8501>，可切换实验并查看训练损失、RankIC、净值、回撤、最新交易日散点和多空排名。面板会对合成数据实验显示提示。

## 设计

窗口只读取当前及历史特征；周频标签为下一周 close-to-close 对数收益。模型输出 L2 归一化表示 `z` 和独立的标量 score。训练目标包括：

* **极端 score 损失**：每周将下一周收益横截面最高 20% 标成 `+1`、最低 20% 标成 `-1`，对每只股票的 score 使用 signed logistic loss，中间 60% 不参与该损失；
* **时间 InfoNCE**：默认 H=5 实验使用同一股票未来 5 周的因果窗口作为正样本；代码仍支持未来 1/3 周及多正样本接口。H=5 表示更平滑的中期状态，适合与 H=1 的短周期结果比较。
* **RankIC loss 与评估**：默认训练使用全量横截面 RankIC，总权重 `rankic_weight=0.5`；头尾子集和混合损失保留在历史实验配置中。
* **四分位块间排序（可选接口）**：支持只比较不同收益组的排序损失；当前默认关闭，组内不产生 pairwise 梯度。

回测阶段使用 score head 输出（多 head 实验时取平均），按每周 score 做多 top 10%、做空 bottom 10%，并额外计算 score 最高的 5 支股票等权做多曲线。多空组合扣除双边成本，Top-5 组合扣除单边成本。历史单 score 基准头尾等权；当前默认为头尾等权（1.0 / 1.0）。Group 去相关、output 去相关和块间排序保留为可选接口，当前默认关闭。

历史500股周频实验使用 `configs/default_500_top20.yaml`：Yahoo 公共接口真实日线 OHLCV 聚合为周五收盘周线，再计算周级特征、下一周收益和回测。固定最近 40 周 query，同一股票未来 5 周窗口编码为单个 key，单一 score head，128 维 embedding。总损失为 `score_loss + temporal_infonce + 0.5 * rankic_loss`。Score loss 对每周收益前 20% / 后 20% 头尾等权（均为 1.0），中间 60% 忽略；RankIC loss 使用全横截面。500 支股票、520 周、8 epochs，后端 `torch_mamba2`。当前缓存日期范围为 2016-07-29 ～ 2026-09-11。

随机窗口、多正样本、块间排序、Group 去相关和 output 去相关接口保留，但本配置全部关闭（`context_lengths: [40]`、`temporal_horizons: [5]`、`block_rank_weight: 0`、`group_weight: 0`）。

```powershell
python scripts/run_mvp.py --config configs/default_500_top20.yaml --output outputs/real_weekly_500_top20_8ep
```

## 因子信号质量评估

### 500 股票 Top-20 做多选模

`configs/real_weekly_500_top20.yaml` 使用 500 股票、520 周、8 epochs 和相同训练损失，仅将选模改为验证集 Top-20 等权纯多头净 Sharpe（相等时选较早 epoch）。每轮覆盖 newest、按验证指标更新 best，并记录验证收益、回撤、超额收益、换手；测试集只评估验证选出的模型。这也是当前默认选模方式，逐轮保存权重接口保留但默认关闭。

```powershell
python scripts/run_mvp.py --config configs/real_weekly_500_top20.yaml --output outputs/real_weekly_500_top20_8ep
```

新做多评估先将标签的对数收益转换为简单收益；按漂移后的持仓计算买卖总换手，单边综合成本为 10 bps，含首次建仓、不含期末清仓。比较同股票池等权基准；相对收益定义为策略净值 / 基准净值 - 1。选股不依赖下一期收益是否缺失，缺失持仓收益按 0 标记并统计。仍为周收盘到下周收盘的研究估算，未实现下一开盘成交、停牌/涨跌停限制与逐期税率。页面上的旧多空/Top-5 曲线保留历史计算口径，不能与新做多收益直接作严格同口径比较。

500 股票扩展实验使用 `configs/real_weekly_500_yahoo.yaml`，保留历史 5 epochs 和其他训练参数，使用独立缓存与产物。为适应 8 GB 显存，启用 `activation_checkpointing: true`（反向传播时重算激活），评估按股票数量分块，不改变损失定义或横截面大小：

```powershell
python scripts/prepare_500_weekly.py
python scripts/run_mvp.py --config configs/real_weekly_500_yahoo.yaml --output outputs/real_weekly_500_520_h5_rankic
```

该股票池按 Yahoo 当前名称排除 ST/退市标记，只选股票类型、有覆盖起始日期的历史、近期有成交且至少有 494 个观测周的沪深股票；排除科创板和北交所。筛选日志在 `data/real_500_universe_audit.json`，入选名单在 `data/real_500_universe.csv`。这是当前存续股票池，不能代表历史逐期非 ST 股票池。日历与 100 股票基准对齐；缺失交易周不填造价格，其收益标签屏蔽。历史500股票实验使用此筛选股票池；缓存不存在或规模不符时直接报错，须先运行准备脚本，避免退回未筛选下载。

除 RankIC 外，回测报告和前端还展示 Normal IC（Pearson）、ICIR、IC t-stat/p 值、按分数分组的平均收益与单调性，以及 Top-Bottom Spread（含周均值和年化值）。这些指标只在样本外预测上计算，用于区分线性相关、稳定性、显著性、非线性分组效果和多空空间。

详细方案见 [BUILD.md](BUILD.md)。

## 本地可视化面板

训练产物可以用轻量 Flask 面板查看训练曲线、样本外多空/Top-5 净值、每周 RankIC、最近 10 个周末的分数与多空候选：

```powershell
python dashboard/app.py --output-root outputs --port 8501
```

浏览器访问 <http://127.0.0.1:8501>。面板会自动发现 `outputs/` 下的实验目录；完整说明见 [dashboard/README.md](dashboard/README.md)。

### 极端子集 RankIC 对照

`configs/real_weekly_500_tail_rankic.yaml` 使用历史 500 股票、8 epochs、Top-20 净 Sharpe 选模，仅新增 `rankic_tail_fraction: 0.1`：每周按有效真实下一周收益选择最高/最低各 10%，在合并子集上计算原 RankIC loss（权重 0.5）。中间 80% 对该项梯度为零，头尾内部仍存在排序约束。Score loss 仍为前后各 20%。该对照配置保留；当前默认为580股全量 RankIC。输出为 `outputs/real_weekly_500_tail_rankic_8ep`，仅 best/newest 权重。

### CSI800 Score 等权对照

历史等权对照 `configs/real_csi800_equal_score.yaml` 保留0.8头尾/0.2全量RankIC。当前默认 Score 为1.0/1.0（归一化后各50%），RankIC为全量横截面。此前0.7/0.3实验及其配置快照保留在 `outputs/real_csi800_mixed_rankic_8ep/`。

### CSI800 纯头尾 RankIC（历史对照）

`configs/real_csi800_tail_rankic.yaml` 保留历史纯头尾配置：`rankic_tail_fraction: 0.1`、`rankic_tail_mix: 1.0`、`rankic_weight: 0.5`。按每周有效真实下一周收益选取最高/最低各10%，在合并子集上计算RankIC，保留头尾内部排序约束；中间80%对RankIC项的梯度为零。中间股票仍可能通过Score的20%标签范围或InfoNCE参与其他损失。全量RankIC不再计算。

800股、520周、8轮、Score等权与Top20净Sharpe选模保持不变。历史混合RankIC等权结果保留在 `outputs/real_csi800_equal_score_8ep/`，本轮独立输出 `outputs/real_csi800_tail_rankic_8ep/`。

### 中信30行业 × 每行业20股（待准备数据）

RankIC 默认恢复0.8头尾/0.2全量，总权重0.5；Score等权。新实验配置 `configs/real_citic30_20_weekly.yaml`：600股、520周、8 epochs，固定seed42，全沪深A股范围按中信一级各抽20支、不放回。仍按验证Top20净Sharpe选模，仅best/newest。当前800股默认缓存保留，新缓存尚未生成，不能视为已训练。

需要有来源、当前有效的全市场 `code,citic_l1` CSV（每股一行），不是单行业接口示例，也不是国证分类。数立方接口和权限见 `docs/datacube_industries.md`。程序验证30个行业各至少20股，不足时报错，不重复股票凑数。上市前/缺失行情不造价，不按下载结果随机换股。独立周线缓存复用可用原始行情，旧缓存不改写。沿用与前次相同的2016-09-30～2026-09-11日历以控制变量。

```powershell
python scripts/prepare_citic30_weekly.py --mapping path/to/current_citic_l1.csv --source "数据提供方" --asof YYYY-MM-DD
python scripts/run_mvp.py --config configs/real_citic30_20_weekly.yaml --output outputs/real_citic30_20_mixed_rankic_8ep
```

### 原500股缓存的混合RankIC对照

`configs/real_weekly_500_mixed_rankic.yaml` 使用原 `data/real_500_520_weekly_yahoo.parquet`，500股/520周/8轮，Score等权，RankIC总权重0.5，其中0.8为头尾各10%合并子集、0.2为全量。沿用旧500股编码器与优化配置，以验证Top20净Sharpe选模，仅best/newest。独立输出，保留原有数据和实验。

```powershell
python scripts/run_mvp.py --config configs/real_weekly_500_mixed_rankic.yaml --output outputs/real_weekly_500_mixed_rankic_8ep
```

### 用户XLS申万行业分层实验（独立缓存）

`StockClassifyUse_stock.xls` 为申万行业分类变动历史，非中信分类。每股按计入日期、更新日期排序取最后生效记录，按行业代码前两位取一级行业。当前沪深上市名单来自AKShare交易所列表（2026-09-16），剔除ST/退市名称。一级行业31类，不足20的行业按用户确认取全部，其余20，seed42随机排序后依次选合格股。

正常股票筛选：历史开始不晚于2016-09-30，520日历周中至少494个观测周，非活跃周不超过5%，2026-09-11行情截止前7日内有成交、收盘价有效且大于零。当前上市/非ST筛选不能保证历史上一直非ST；长历史筛选存在存续偏差，采用当前行业回溯历史。

结果600股：29类各20、综合14（5支因非活跃周过多剔除）、美容护理6（21支历史不足）。日历2016-09-30～2026-09-11。`data/sw_file_20/` 独立保存原XLS、文件hash、最新分类筛选记录、行情筛选记录、行业数量、股票清单和周线缓存，旧缓存不修改。

```powershell
python scripts/prepare_sw_file_weekly.py
python scripts/run_mvp.py --config configs/real_sw_file_20_weekly.yaml --output outputs/real_sw_file_20_mixed_rankic_8ep
```

已完成的缓存或实验不要覆盖；准备脚本发现已有完整缓存会停止。配置为600股/520周/8epochs、40周query/5周key、单score头、Score头尾各20%等权；RankIC为0.5×(0.8头尾各10%子集+0.2全量)，验证Top20净Sharpe选模，仅best/newest。前端展示真实申万行业分布。与500股旧实验的比较同时改变了股票池和日历，不能视为仅行业抽样的因果检验。

### 原500股全量RankIC十轮实验

`configs/real_weekly_500_full_rankic_10ep.yaml` 使用原500股缓存，10epochs，RankIC仅全量横截面（权重0.5，移除tail_fraction/tail_mix），Score仍为收益头尾各20%等权。其余沿用原500股设置，验证Top20净Sharpe选模，仅best/newest。输出 `outputs/real_weekly_500_full_rankic_10ep`，不覆盖旧实验或默认配置。

### 原500股第8轮筛选Top30%后二阶段训练

`configs/real_teacher500_top30_5ep.yaml`：第一阶段 `real_weekly_500_full_rankic_10ep/best.pt`（epoch8），在训练末周2023-08-25以原500股标准化输入预测，按score降序取150股（相同分数按symbol）。单周高分不代表长期因子适配。缓存 `data/teacher500_top30_trainend/` 保留500股评分、150股名单、权重hash和筛选日期。第二阶段从头训练5轮，重新计算150股横截面标准化和标签，保持全量RankIC0.5、Score等权、query40/key5及Top20验证净Sharpe选模。测试区间保持原切分；未用测试标签筛股，但第一阶段训练内筛选且两阶段重复使用验证集，仅是探索实验。

### 按涨跌预测可靠性筛选150股（探索结果）

运行 `python scripts/run_reliability_experiment.py`。源模型为原500股全量RankIC第8轮，固定权重，只对分数作方向校准，不把score正负直接当涨跌。验证前26周校准一维logistic概率，后52周每周用之前的验证记录更新校准，概率>=0.5预测上涨。股票筛选按两半时期各自上/下召回率的Beta(5,5)收缩值，取较低半段排名，选150股；每半段至少5次上涨和5次下跌。涨跌幅绝对值<=1e-8与缺失标签不参与方向评估，收益回测口径不变。

独立缓存 `data/reliable500_top150/`、独立配置 `configs/real_reliable500_top150_5ep.yaml`，从头训练5轮并保留原损失（Score头尾等权、全量RankIC0.5、InfoNCE）。对照原模型500股、原模型仅交易筛选150股、150股重训，Top20与成本规则一致。原模型在缩小股票池时保留500股输入预处理，重训模型按150股重新预处理。每个模型测试期方向概率由其验证预测拟合校准后固定，不用测试标签拟合。

输出 `outputs/real_reliable500_top150_5ep/` 包含完整可靠性排名、逐周校准预测、三组测试预测和曲线、CSV/JSON报告。已存在的目录不会覆盖。页面展示方向平衡准确率、涨跌召回率、Brier、净收益和回撤。第4轮best结果：三组测试平衡准确率52.06%/51.79%/51.10%，净收益36.51%/25.82%/23.65%。这次筛选和重训没有改善测试表现。

注意：只有校准按时间滚动，编码器没有重新滚动训练；原epoch8已经用完整验证集选模，筛股和学生选模也复用验证集，测试期已在历史实验被查看。因此这是一轮探索，不是完整端到端OOF或独立验证。强制取前150是相对排名，不代表150股都显著超过50%。当前模型仍以收益排序训练，不能据此否定专门训练方向预测模型的潜力。
## 保留500股的可靠性加权实验

当前按用户指定**只跑2倍、8 epochs**：固定使用 `data/reliable500_top150/universe.csv` 中的150股名单，保留原500股缓存和全部500股交易候选，其余350股权重为1。seed=42从头训练，保存 best.pt / newest.pt。配置为 `configs/real_reliable500_weight2_8ep.yaml`，权重表位于 `data/reliable500_weight2_8ep/stock_weights.csv`。

```powershell
python scripts/run_mvp.py --config configs/real_reliable500_weight2_8ep.yaml --output outputs/real_reliable500_weight2_8ep
```

此前五倍数扫描已按用户要求停止，`data/reliable500_weight_sweep/CANCELLED.txt` 记录取消状态；不会继续运行其余倍数。

`stock_weight_path` 指向包含全部股票各一行的 CSV（symbol、weight）；weight 必须为正且有限。Score 的每项损失乘股票权重后除以有效标签权重之和，收益头尾类别仍等权，中间60%仍忽略。全量 RankIC 保留全部有效股票的软排名，仅对排名的均值、协方差和方差做加权。时间 InfoNCE 不变，避免同时改变表示学习任务；不是增加重复样本或只对150股计算相关性。

2倍为用户指定，不再搜索倍数。最佳轮次按**验证集 Top20 做多净 Sharpe**选择，同分优先较早轮次。与已有原500股实验比较时直接复用历史结果，不重新训练1倍基线；历史实验训练10轮、最佳为第8轮，本次训练8轮，应同时注明训练预算差异。

150股来自之前的验证期可靠性筛选，本轮仍复用验证期调参；测试期此前也已查看。因此此实验只能用于探索，不能当作新的独立样本外确认。原数据缓存、历史实验和默认训练配置保留。
## 500股等权：4头平均与输出去相关实验

配置 `configs/real_weekly_500_heads4_decorr_8ep.yaml` 使用原500股缓存、520周、固定40周query与未来5周key、8epochs；恢复500股等权训练和回测，没有150股筛选或样本加权。仅新增4个线性score头取平均与 `output_decorr_weight: 0.001`。Score头尾等权、全量RankIC总权重0.5、InfoNCE和验证Top20净Sharpe选模规则均保留。旧Group去相关关闭；输出去相关不warm-up。

输出去相关按每周有效股票单独计算4头相关矩阵，再对非对角元素平方和取周平均，即 `mean_week(||offdiag(Corr(head1,...,head4))||_F²)`。不是在一个batch中混合不同周计算相关，也不是对模型权重矩阵做正交约束。常数头使用数值稳定处理；该损失本身不保证避免塌缩，前端同时展示各头分数标准差。

```powershell
python scripts/run_mvp.py --config configs/real_weekly_500_heads4_decorr_8ep.yaml --output outputs/real_weekly_500_heads4_decorr_8ep
python scripts/evaluate_four_heads.py
```

仅训练这一组新模型；单头对照直接使用已有 `real_weekly_500_full_rankic_10ep` 的epoch8 best，不重训。页面包括同500股Top20净值对照、验证/测试相关热图、各头RankIC和训练去相关loss。四头结构与正则同时改变，只能比较整体方案，不能单独归因于正则；测试期此前已查看，结果仍为探索性比较。
