# 数据与管道代码审计：新快照迁移执行说明

审计日期：北京时间 2026-11-13。范围为当前工作区文件和本地数据；工作区已有大量未提交修改，以下行号指本次读取版本。只读检查源码、Parquet 元数据及少量列统计；没有下载、训练、外部查询或修改业务代码。本文件是唯一写入物。

## 1. 结论与执行顺序

先建立可复现、明确时点的新数据快照，再扩股票池或重训。现有模型已能处理行业内股票数不等，不需要先重写模型。真正优先的问题是：真实成交 bar 与零量占位 bar 混入滚动特征；标签与评价器使用不同日历；数据血缘误记；训练/推理预处理没有绑定到同一份快照。

建议顺序：

1. **P0：冻结旧结果，修复新版本的数据契约。** 新路径保存原始价、复权因子、成交量额、交易日历、股票状态、成员有效区间与标签；修复来源记录和执行前内容校验。旧 B0/v4 不覆盖。
2. **P0：统一真实 bar、显式标签和成熟时点。** 由同一会话日历生成 entry/exit/label_available_at，`arrays()` 读取 v2 标签；训练、验证选轮、H2、回放使用同一契约。
3. **P0/P1：固定 fit/transform 边界。** 周线和日线统一训练截止日，推理读训练保存的参数。先用很小的本地夹具验收，不重训验证数据加工是否正确。
4. **P1：扩池、财务和逐日持仓。** universe 时间有效性、财务版本可用性、券源及企业行动缺一项就保留对应风险标记；不把静态快照结果改称历史可交易全市场结果。
5. **P2：最后再跑同数据、同日期、同券池、同成本的模型比较。** v4 参照为 B0 长腿 + CS 短腿三轮 `[4,5,6]`、三 seed，借券年费 `.0835`；入口见 `scripts/followup_replay_baselines.py:20-25,69-76`。旧 v3 不是当前默认基线。

## 2. 实际调用链与保留的正确部分

| 层 | 当前调用链 / 最小修改位置 | 应保留的行为 |
|---|---|---|
| Yahoo 下载 | `scripts/download_full_universe.py:42` → `src/quant_mvp/data.py:73` `_yahoo_one` | 按 symbol 独立文件、重试；新版本要补身份/范围/复权验证和原子发布 |
| AkShare 原始日线 | `scripts/download_akshare_580_daily.py`；更新 `scripts/update_akshare_data.py:92` → `build_raw_ak2000.py:19` | raw 与 hfq 分开保存，量额优先取 raw；不能因扩池丢掉此区别 |
| 周缓存 | `scripts/prepare_hierarchy_580_mon.py:97` → `aggregate_weekly`、`align_observed_features`、`path_features`、`market_features` | 已拒绝覆盖已有 cache；已按 symbol/date 对齐；保留 warm-up 缺失 |
| 模型数组 | `src/quant_mvp/hierarchy_data.py:108` `arrays` | 基础特征全局 fit 仅 `raw[:train_end]`；当周横截面标准化；`available` 只由当前与历史输入决定 |
| 训练/导分 | `hierarchy_engine.py:328` → `arrays` → `make_batch` → `HierarchicalRanker` → `predict` | 训练 label mask 与 score availability 分离；`predict` 仍为无标签股票输出分数；真实收益不再统一 clip |
| 新版回放 | `followup_replay_baselines.py:94` → `eval_weekly_architecture.py:64,143,175` | 已剔除整周休市；可校验市场首会话；SH/SZ 券池映射和严格早于决策日的 snapshot 已实现 |
| 股数账本 | `weekly_followup_ledger.py:5` → `followup_turnover_rules.py:70,118`、`followup_one_step_turnover.py:131` | 已保存 shares/cash/marks/targets、未成交订单、现金与借券约束；可在此扩展，不另造账本 |

`tests/test_hierarchy.py:59` 已检验闭市周不破坏历史特征且未来修改不影响过去；`:164` 已检验原始评价收益不受训练 clip 影响。`tests/test_weekly_followup.py:19,38,53,129,190,208,219` 已覆盖券池 venue/PIT、H2 首会话、休市周、阻塞卖出、受限卖空现金和实际借券天数。应扩充这些测试，而非另写平行测试框架。

## 3. 本机数据证据：不要把缺失都归为同一种问题

| 缓存 | 股票 × 周 | 日期范围 | observed 行 | 整周 observed 全 False | 来源审计唯一 raw_path 数 |
|---|---:|---|---:|---:|---:|
| `data/sw_580_observed_features_2000_fri_path30` | 580 × 1399 | 2000-01-07—2026-10-23 | 581496 | 41 | 1 |
| `data/sw_1893_full_2004_fri_path30` | 1893 × 1191 | 2004-01-02—2026-10-23 | 1319935 | 30 | 1 |
| `data/sw_1887_full_2004_fri_path30` | 1887 × 1191 | 同上 | 1125586 | 30 | 1 |
| `outputs/weekly_architecture_v1/dev_data` | 580 × 1096 | 2000-01-07—2021-01-01 | 409331 | 35 | 无该审计文件 |

这四个缓存所查 `(date,symbol)` 无重复，observed 无空值，close 无非正值。不能因此认定所有日线、金额单位、复权或标签均已通过验收。

- 580 缓存 2021 年后的 172745 个 observed 行，在抽查的 `ret1/ret60/vol60/amount_z20/d_on/d_in/dfront/d_onn/d_back/d_mon` 中缺失为零。
- 1893 同期 532687 个 observed 行，`ret60/vol60` 各缺 25120，`amount_z20` 缺 7190，6 个 path 列各缺 89。19 只股票在这一段没有任何有限 ret60。`prepare_audit` 的 416 个 warm-window failure 多为晚上市和预热问题；不能直接删除这 416 只，也不能全部补零再宣称同覆盖。
- 1893 的 35357 行 `observed=False` 但 close 非空，量为零；其中 34743 行 target 非空。5 个整周所有 observed 为 False，但至少一只股票 close 非空，例 2004-01-23、2004-05-07、2005-02-11。580 只有 2 行此类（2025-03-21 的 600095.SS、601519.SS），没有整周这种冲突。
- 580 每年出现过真实 bar 的股票数从 2000 年 190 只增至 2016 年 580 只；1893 从 2004 年 556 只增至 2026 年 1892 只。这不是历史全 A 股成员数，而是当前静态池历史上可观察的子集。
- 本机 AkShare raw/hfq 各 2752962 行、580 只、2000-01-04—2026-11-02；键无重复。周 cache 只到 10-23。不能把本机日线最新日等同周缓存最新日或运行端最新日。
- `data/sw_full_1893/download_audit.csv` 记录 1874 ok、19 failed，ok 中 2 只只到 10-20；但 19 个 failed 对应文件目前全部存在，抽查 002115.SZ、688671.SS 已到 10-27。这证明审计文件与当前目录并非完整的一次不可变快照；不能据旧 audit 直接认定现有文件仍下载失败。

## 4. 已证实问题、最小修复点和验收断言

### A. 零成交占位 bar 仍进入特征和市场日历（P0）

证据：`prepare_hierarchy_580_mon.py:155` 用 close 非空且 volume>0 定义 observed；但 `hierarchy_data.py:12` 仅 dropna(close)，`market_features:21` 也只看 close，`path_features` 在 `prepare_hierarchy_580_mon.py:59-72` 未滤零量。1893 上节统计证明不是纯理论风险。

最小修复：先在原始层区分 `price_observed`、`traded`、`session_open`、`suspended`，在 builder 传入真正观察到的有效 bar；基本特征和 path 用同一规则。保留上一已知值用于输入 carry/持仓估值，但它不能变成真实成交样本或标签成交价。`market_features` 使用权威日历和当时有效股票，而非 close 是否存在推断开市。不要在 `available` 最末端才修这个问题。

同步：`prepare_hierarchy_580.py:34`、`prepare_hierarchy_580_2014.py:55`、`prepare_hierarchy_580_mon.py:151` 三个共享调用；日线 branch 的 observed 定义也要一致。新版本默认严格，旧 cache 走原契约复现。

验收：加入一个 close 合法但 volume=0 的停牌/节假日占位 bar 后，真实 observed 序列计算的 ret60/vol60 与不含占位行时相等；当周不能被评分/成交；市场开市表不变；carry 输入允许存在但 `observed=False`。未来 bar 改动不得影响过去任何特征和 universe eligibility。

### B. 周标签、H2 和经济回放日历不一致（P0）

证据：`hierarchy_data.py:175-183` 用固定 `open_[t+1],open_[t+2]`，且缺失/跌停退出直接改用未来一周 close；`eval_weekly_architecture.py:64-67` 先删整周休市，再取下一开市周；`prepare_week2_open_labels.py:16-36` 使用真实市场首会话，H2 还导出成熟时间。`aggregate_weekly` 的每股 first open 可能是该股周三复牌，不等于全市场周一开盘。新版 replay 传 `open_sessions` 已防该入口，但训练标签没有同样检查。

最小修复：由 snapshot 的一个标签构造步骤导出 `(anchor,symbol,label_name,entry_at,exit_at,label_available_at,ret,valid,invalid_reason)`；`arrays()` 对 v2 读取并验证它，不再自己 shift 猜交易时点。有效价格/复权收益与可成交性分开：主排序标签不可被未发生的 entry_ok 作为输入 mask；执行器再按实际 session 状态处理。无可用 exit 的训练标签可以 invalid，账本的旧持仓不能消失。

同步：`hierarchy_engine.predict/train`、`run_weekly_architecture.export_dev_scores/export_history_scores`、H2 builder、`evaluation_dates/build_proxy_panel` 以及 `followup_replay_baselines`。`eval_weekly_architecture.py:288,483,572` 与多个 `weekly_arch_*` 旧调用未传 `session_data`；应让 v2 调用必需，或显式标 legacy，避免新训练仍被旧评价入口裁决。

验收：春节整周休市后 entry/exit 跳到下一真实市场 session；周三复牌股不能假作周一开盘可买；H2 与 H1 为独立相邻持有区间，且各有成熟时点；短样本尾端只标 invalid，不循环、不 forward-fill 标签、不把 NaN 变可交易零收益；`label_available_at < 训练截止时点` 是每个监督标签的必要条件。选轮标签则严格早于独立的 selection_at，统一采用主执行手册的排他边界。

### C. 可选日线分支 fit 吃入第一验证周；其他期限仍靠常数预留（P0，启用分支前必须修）

证据：周线 `hierarchy_data.py:147-149` fit `raw[:train_end]`；日线 `:72` 却包含 `weekly_dates[train_end]` 当日及此前的 daily，第一验证周也参与均值方差。当前 B0 无 daily_path，不能据此宣称所有 B0 已泄漏。

`hierarchy_engine.py:453-460` 训练预留为 max(temporal_horizon,common_training_horizon)，没有纳入 target_horizon 或主标签真实成熟时点。现有常用 temporal_horizon=5 对普通 t+2-open 足够，**本次未证实当前 B0 训练跨界**；但 target_horizon>5、较长停牌退出、未来新标签将使该保证失效。H2 已在 `:439-440` 按成熟日过滤，可以复用这一思路。验证选轮 `:709-717` 用固定 2 周裁尾，短验证窗甚至跳过裁尾；也应改按真实成熟日。

最小修复：配置保存明确的 train/validation/history 时间界限，索引由 calendar.searchsorted 派生；基础训练特征只 fit 到训练最后 anchor，日线最多到该 anchor 的 decision_time；监督标签按成熟时间过滤；InfoNCE 再独立保证未来 encoder window 留在训练区间。

验收：把第一验证周和其后的 daily feature 改为极端值，保存的 daily_mean/std 与训练 X 不变；改变 target_horizon/跨节标签，不得有一条训练标签成熟于训练截止后；验证选择不得读取首历史决策时尚未成熟的标签。同步 `tests/test_hierarchy.py`，增加 daily split 和主标签/H2成熟边界夹具。

### D. 数据血缘记录错误、下载状态不足以断言完成（P0）

证据：builder 第一循环 `prepare_hierarchy_580_mon.py:134` 找 raw_path，第二循环 `:172` 仍用最后一次 raw_path，所以三套本地 audit 都只有一个路径；manifest `:229` source 固定 `SOURCE_DIR`，不反映 `--raw-dir`。1893 的 `configs/hierarchical_1893_full_attn128_2021split.yaml:85-87` prepare_command 还写 1887 输出/股票池。

`download_full_universe.py:46-48` 缓存只看 max(date)，不核查起点/缺口/字段/请求 universe/复权版本；`:57-60` 任何非空响应都 status=ok，即使末日没有达到 complete-through。默认日期仍是 2026-10-27/29，不能在 11-13 无参数运行后称已更新至今。`_yahoo_one` 会吞掉异常返回 None，审计中无错误类型（`data.py:119`）。

最小修复：每 symbol 保存 resolved path、文件内容 hash、source、fetch time、request start/end、实际首末日、行数、schema/adjustment/unit；将 path 随 daily/weekly 一起存入 per_symbol，而非使用外层变量。manifest 引用完整源清单及加工代码 hash。下载区分 complete/partial/no_data/failed，非交易日终点按固定日历解释；若上市前无记录，必须由 listing date 证明，而非一律视作缺历史。新目录 `.partial` 验收后原子发布，失败不覆盖旧文件。

验收：两个 symbol 来自两个 raw 目录时 audit 各自匹配路径/hash；末日符合但起点变晚/缺一列的缓存不能直接 cached；空响应/损坏响应/失败重跑不破坏原文件；每份 snapshot 的文件数、键、日期、内容 hash 与 manifest 完全一致。

### E. AkShare 更新链存在覆盖不全和成对写入风险（P1）

证据：`update_akshare_data.py:102-105` 只在 last < Friday-3天 才刷新，周三/周四尾日会被视作足够新；只枚举已有 raw symbol，完全缺失的请求股票永不补齐。首个 warm-up symbol 在 `:125-130` 绕过后续 overlap 校验。`:141-156` 先 append hfq 再处理 raw，raw 全量复拉失败时已 append 的 hfq 不会撤回。`:231-233` 财务仅判断某季度是否出现，某季度有部分公司就不会补晚披露或修订。

`step_cache` 固定写现存主 CACHE（`:46,268-280`），而 builder 明确拒绝现有 OUT（`prepare_hierarchy_580_mon.py:124-125`），所以当前 `--build-cache` 和 builder 的保护语义冲突。应新建快照目录，不建议通过删除旧 cache 解决。

最小修复：以请求 universe 和最后完成市场 session 对账；所有股票共用同一校验函数，raw/hfq 对同一 symbol 都成功验证后才进入发布集合；原始两表也原子提交成同版本。晚披露按报告/公司键增量对账，不仅季度存在性。

验收：周四旧尾日会刷新周五；缺失股票会进入任务；warm-up symbol 的复权修订触发全量重取；任一腿失败时旧 raw/hfq 成对保留；构建新快照成功且旧 cache hash 不变。不要在此次审计中执行网络重取。

### F. 复权/量额口径与旧 backfill 实现不一致（P1；不要误称当前 B0 已受损）

`_yahoo_one` 在 `data.py:107-113` 保存调整后 OHLC，amount=调整后 close×原 volume；amount 是代理值，不是原始成交金额。`build_raw_ak2000.py:25-34` 则使用 hfq OHLC 和 raw volume/amount。二者若混入全池会改变 amount_z20、市场 activity、液性分桶和 short 流动性前40%筛选，不能靠相同列名视作同语义。

旧 `build_raw_2000_backfill.py:42` 的 early 来自不复权 a；`:60` 估计 r4/hfq 比例；`:73-75` 却把该比例乘到 raw early OHLC，而非 hfq early，和头部声称不符。只检查拼接缝附近15%阈值不能验证整个早期企业行动收益。当前主580配置 prepare_command 明确用 raw_ak2000 全量 hfq，故这里只认定旧脚本错误，未证实当前主 cache 使用这段错误数据。

最小修复：v2 原始层保存未复权 OHLC、原成交 volume/amount、独立调整因子及其版本；研究 returns 使用明确 adjusted series，执行价格/手数/价格限制使用 raw。跨源 fallback 必须有单位和调整规则对账，无法证明则不要静默混源。旧 backfill 如继续使用，早期 OHLC 应取对应 hfq 再缩放，量额仍取 raw，并核查完整 overlap 收益，不只两天比例。

验收：任意拆股/分红夹具中原始价跳变可见、调整收益无伪跳变、成交金额不随以后复权因子改写；raw→hfq→因子关系逐日可验证；每个 symbol 的字段单位在整个历史区间一致。

### G. 财务数据并非已证明的历史 PIT（P1）

`add_fin_features.py:49-62` 只用 report_date + 固定 lag 作 avail_date，舍弃来源 ann_date；去重每 report 只留最后值，没有修订版本。现有财务44968行中42565行来源 ann_date晚于合成 avail_date，10行没有 ann_date。这不能证明42565条必然泄漏（源 ann_date 可能是更新/修订日），但足以说明现有表无法证明原始披露可用性。不要把代码头部的“No lookahead”当作验收结果。

`:76-84` 用指定训练截止的中位数填充后直接覆盖 weekly；参数、缺失标记和训练区间未在该步骤独立保存。更换 train_end/universe 后复用这个缓存，填充值可能来自另一研究窗口。`add_fin_features_2005.py:21,34-45` 把 Yahoo raw_2004 中已调整 close 当未复权 close，而且只 exact-Friday join，节假日周分母也缺失。

最小修复：保存 `(symbol,report_period,announcement_time,version_time,ingested_at,value)`，由决策时可得版本 asof join；已知公告日但无时刻，采用明确的下一可交易决策时点规则。缺可靠版本的财务先单独消融/标为非PIT，不为了“完整”伪造时间。v2 cache 保留财务 NaN 和 missing flag；中位数在训练 fit 保存、推理 transform 重用；以 raw close 作估值分母。

验收：后续修订不改变先前决策特征；晚披露不得使用法规期限伪造更早可得性；未来财务极端值不改变训练 imputer；换 universe/切分时预处理身份不同；缺失与真实数值0可区分。同步 `scripts/live_pipeline.py:127`、`update_akshare_data.py:281` 的调用，禁止在 live 更新时重写研究快照。

### H. 行业相对特征分子未按有效样本屏蔽（P1，仅启用该分支受影响）

`hierarchy_data.py:218` counts 用 available，`:219` sums 却累加所有 vals，包括不可用股票 carry 的旧值；分母分子集合不同。最小改动是在 sums 之前对 vals 施加同一个 available mask。现有 B0/CS 不应因此一概判失效；F1/industry_relative_names 分支需修后复核。

验收：把 unavailable 同行业股票的输入从0改成极端值，可用股票的行业相对特征完全不变；全行业 unavailable 不产生 NaN/Inf；单个可用股票的行业相对值遵守明确约定。

## 5. 多 universe 的最小迁移，而非改写模型

580 的选择明确带存活/长历史偏差：`prepare_sw_file_weekly.py:20-27` 取2026当前分类、当前上市/非ST；`:45-48` 用未来区间覆盖与尾日活跃筛选；`prepare_sw_580_weekly.py:28-35` 再删两个不足20只行业。manifest 也明确写 current snapshot retrospective。1893 的 CSV 只有 symbol/industry/code 五列，没有 membership 起止、上市退市、行业有效期，不能称 point-in-time 全市场。

建议在新 snapshot 中保存一个 master security 表和按 anchor 的 universe eligibility，至少区分：旧580复现池、1893静态扩展对照、可得历史全市场池、执行候选池/短池。**不要为了不同池创建不同数据层。** 同一原始快照生成命名 universe；snapshot 内容身份必须包含其 membership/hash。

现模型的实际限制：

- `hierarchy.py:210-239` 已按 industry_id 最大值构造 token、按最大组 padding；25—157只的不等行业不是待开发功能。
- 仍需连续无空洞 industry_id；`hierarchy_data.py:236-237`、`hierarchy.py:213-215` 会拒绝空行业。最小版本用固定 master symbol 轴 + 当期 membership/available mask，行业词表固定；不在每周删除列重新编码 id。
- 行业归属现在是静态 `ids[N]`，来自 `hierarchy_data.py:201-202`，训练 `make_batch` 不传 per-date ids。不要声称原模型已支持动态行业归属。首版若保留静态行业对照，应显式标记；若要求历史行业，需扩数据 batch→relate 的接口并测试每日期映射。
- checkpoint 中 `industry_ids` 是 persistent buffer，而 slots 等由构造函数生成且 nonpersistent。相同N但换了股票/行业后直接 load_state_dict 可能加载旧 ids、保留新 slots；必须先校验 symbols/id词表/feature顺序/hash。跨 universe 初始化优先复用已有 temporal-only 支路 `hierarchy_engine.py:353-367`，不把全模型当可任意迁移。
- `arrays()` 当前只检查 feature维度，不检查 cfg的n_stocks/n_days与实数；新快照 loader 应做严格形状/唯一键/日期递增/manifest一致性断言。`observed` reindex 后应显式 fillna(False)，而不是 `astype(bool)` 把缺行NaN转换True（`:118`；当前所查完整网格尚无该问题）。

需要同步的硬编码：`run_weekly_architecture.py:128` 的2021、`:171-174` 固定580历史路径、`:179,183-199` 的1096/1094；`weekly_architecture_snapshot.py:36` 的939；历史/回放脚本的DEV_CUT与2021开始日。由 snapshot 的 split 时间界限和 label maturity 生成索引，保留现有 runner 和输出隔离。

验收：N=3/7、不等行业、缺行业当前可用股票、上市后逐步解锁、退市后退出、同N错误symbol顺序/同维度错误特征顺序均有小夹具；当前不可用股票任意值不影响可用输出；已有 permutation/empty-industry 测试继续有效。`tests/test_followup_entry.py:77` 当前构造1096固定日历，需要改成两个不同起始日/长度的快照，仍证明 history export 不覆盖原 dev_scores。

## 6. 评价器现实性与短池：当前能说什么

新版 replay 比旧版有真实改进：首会话检查、整周休市跳过、SH/SZ券池匹配、借券按 elapsed calendar days、缺失退出压力情景，都在现实现中。仍是研究 proxy，原因有具体代码：

- `eval_weekly_architecture.py:94-99` 不能开盘退出时直接使用该周close，没有证明该close时能成交。
- `:175-223` 每周从 intended 重建 held，缺失收益主情景记0，previous只算权重漂移/换手，不维护被阻塞退出的真实股数。某股未能退出且新一期不再入选，经济暴露仍会从持仓中消失。
- `pick_targets:166-171` 用当期 amount 前40%且名单short_ok作为短池；不足10只留空预算，没有真实每股可借数量/定位记录/召回/费率时序。当前名单证明交易所资格，不证明券商当时有券。应分别报告覆盖日期、目标数、实际借到数、失败原因和现金。
- `board_limit` (`hierarchy_data.py:96-105`) 仅处理300/301改革，ST5%明确未建模，688/689没有分支。1893有180只688/689、346只300/301；新池必须使用按日股票状态/交易所规则表或明确blocked，不能继续套一个0.098阈值。IPO特殊期间、除权参考价、停复牌、涨跌停是否可成交都需状态数据；本审计未外查其现行法规。
- 跨universe比较时 market synthetic index、标准化截面、行业权重和short液性前40%同时变化。既报告各自自然覆盖结果，也在共同日期/共同股票/相同资金和短池下配对；不可只对比两份年化收益。

## 7. 复用现有 Ledger 的最小扩展与验收

`weekly_followup_ledger.py` 只有104行，已能做逐日持仓账本，适合扩展。需要完成下面的真实约束再把它当股数主验收器；proxy保留快速研究和旧基线复现。

1. **raw价格输入。** `followup_turnover_rules.py:54` 目前读取 raw_ak2000，实际为hfq；`:99` lot=1，因此当前的shares是调整价计量的研究份额。改成 snapshot 原始OHLC和交易单位表，信号仍可用调整特征。验收同资本下手数/费用按原始价格计算，且不因以后复权改变历史交易。
2. **企业行动事件。** 在 Ledger.step 开盘交易前，最小增加已确定生效日的 corporate-action 应用入口；事件id去重。拆并股同步调整 shares、目标股数、上一mark；现金分红把多头权益和空头应付股息计入NAV，到账/扣款日期改变cash而不再改变NAV。无事件表不能仅切到raw价格就上线。若首版不支持配股/复杂重组，明确标不可估值而不是吞掉事件。验收2拆1时价格减半股数加倍、NAV不变；除息日多头应收/空头应付抵消价格跳变；重复应用同事件拒绝/幂等；调整收益与raw账本在无成本简单情景对账。
3. **持仓跨周连续。** 未成交卖单/回补单保留股数和待执行量；停牌时沿用上次mark并标 stale，不补未来价格；退市终值只用真实结算/保守未决状态。已有 `test_ledger_carries_blocked_exit_and_accounts_fees` 可扩为跨多个决策周。验收连续跌停3天，期间风险暴露/NAV/借券费不消失，第四天可成交才释放资金。
4. **费用与券源。** 现有fee_rate对称统一bps，short_margin是常数，borrow_available只限制新开仓；尚无券源召回与持仓持续费率。现 `step:98` 在当日订单成交后按剩余空仓收费；若把周末3天传进来，会对新周一空仓收过去3天费、对已回补旧空仓漏收。应先对上一计息区间持仓计提，再处理当日订单；按真实calendar gap，不照搬当前长期long-only调用的`calendar_days=1`（`followup_turnover_rules.py:118`）。验收周五空仓周一回补仍付周末费用，新周一开仓不付过去周末；locate不足留现金；召回无法回补进入pending；股息补偿与借券费分账。
5. **T+1和成交单位。** 现Ledger只有一个全局lot，且无可卖股数按日状态。以股票/市场的交易约束表作为执行边界；不用改排序模型。验收当日新买不能同日卖，合法旧持仓可卖，主板/其他板块及零股卖出分别处理，负现金不能因费用/整手截断被绕过。
6. **与proxy可解释对账。** 不要求真实账本和含fallback的proxy总收益完全相等。先在无停牌、无企业行动、无交易限制、可分股、同成交价/同目标数量和同费用的简单夹具严格相等；再逐项开停牌、费用、整手、企业行动、借券，差异逐笔归因。真实实验同时落 daily_nav、positions、orders/fills、corporate_actions、borrow/fees、unresolved_marks。

同步调用限于 `followup_turnover_rules.py`、`followup_one_step_turnover.py`、`tests/test_weekly_followup.py` 的直接Ledger调用；其他proxy使用者暂保留明确proxy身份，避免一次把几十份研究脚本改成不同经济含义。

## 8. 可复现 fit、checkpoint 与队列发布门

`hierarchy_engine.py:446` 写 preprocessing.json，但 `arrays()` 每次重新fit；`live_score.py:94-98` 只改cache_path，仍沿用原market_path/universe_path并重新fit；`run_weekly_architecture.py:177-190` 同样重新fit再对比prefix。新数据刷新、修订或换pool后，模型权重不变不等于输入转换不变。

最小方案：保留 `arrays()` 公共入口，明确fit和已保存preprocessing的transform两种调用；训练产物保存训练边界、feature顺序、symbols/id词表、财务imputer、市场/特征统计、snapshot ID和内容hash。推理必须绑定一致的weekly/market/universe/labels/calendar，复用冻结fit；若需新版数据修订，则新run明确记录而不是悄悄更新老checkpoint的输入。

队列目前 `weekly_followup_queue.py:64` 仅抄入 cfg/data hash；`:85` 仅检查输出存在；`:95-98` 在共享REPO运行当前代码。`run_weekly_architecture.log_trial:111` 的 source_sha256 实际等于 run name，不是源码hash。队列的“done”现在只说明命令退出0且路径存在，不能证明运行使用了提交时的数据/源码。

最小方案：提交manifest列出源码、配置、数据、标签、日历及checkpoint输入的真实SHA256；worker启动前重新算并逐项比对；固定只读源码快照/checkout作为cwd及PYTHONPATH。已有工作区未提交修改不能只记git HEAD。输出由唯一attempt目录原子发布，完成标记还需验证产物schema/非空/预期metadata/hash。锁不匹配时BLOCKED，不自动重训。

验收扩展 `tests/test_followup_queue.py`：提交后修改config/data/source任一字节不得执行训练命令；陈旧空产物不能使job done；相同job ID成功后仍不重跑；上游failed会BLOCKED；GPU锁保持互斥。`tests/test_followup_entry.py` 同步证明传入不同snapshot路径/日历后导分正确，且旧run目录字节不变。

## 9. 待查风险和本轮停止条件

以下不是已证实的当前模型收益错误：真实历史退市/行业变更覆盖；各数据源完整除权因子与历史修订；AkShare金额/股数的接口单位随版本变化；来源公告时间语义；真实broker locate/召回/费率；成交量参与率与订单滑点；当前运行端是否拥有更新后的本机文件。本轮无网络与训练，不能用猜测补齐它们。

新数据版本的代码验收停止条件是：小夹具全部通过、不可变snapshot完成离线审计、所有输入/输出内容hash绑定、旧基线文件保持原样、每个未满足的真实数据条件都有明确missing reason。到此才允许下一阶段训练；不以“成功读入Parquet”或“训练能跑”替代数据验收。

未运行训练、下载、外部数据核验或测试套件；建议中的验收断言尚待实现执行。旧存活池、非PIT财务和proxy成交结果仍有已列明的解释边界。
