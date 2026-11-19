# 周频选股第二轮实验执行手册

2026 年 11 月 12 日初版 · 11 月 13 日修订评估协议 · 北京时间

## 0. 先读这一页

**适用范围：现有 580 股和申万行业是 2026 当前快照回溯；本轮所有回测、种子比较都是该快照条件下的研究代理，不能视为严格时点历史验证或可部署证明。逐时点行业、股票池和退市数据重建参见 [执行结果](research_20261113_followup_execution_results.md)。**

用现有 580 股数据和上一轮权重验证三个问题。CS 空头的增量来自哪里，在实际成本下还有多少。W20 的 Top-40 或候选重排能否改善纯多。独立的第 2 周预测能否改善下周选股或减少不必要换仓。每次只改一个因素。日线、对比学习、RL 与生成式预测按下文条件启动。最后按纯多、纯空、多空逐项验收，并把通过的改动逐个加到最强父方案上。

**均值略高可以继续验证。** 上轮的 +3 基点/周是组合追加预算的预注册筛选线。约 +2 基点/周在 52 周算术口径约是 1 个百分点，若成本估计可靠、风险和运维负担合适，值得保留。均值 +2 但区间仍宽、收益集中或优势依赖偏低的费用假设时，保留为影子候选或待证。新一轮采用第 2 节的新判据，旧一轮的判定和产物保持原样。

| 方向 | 当前判断 | 本轮动作 |
| --- | --- | --- |
| CS 空头、B0 多 + CS 空 | 修正券源映射、休市日历、开盘会话和券费后，沪深验证和测试 S10 相对 B0 约 +31.45/+25.50 基点/周。组合相对现金在测试压力档余量小 | 按 §3 预检券源映射、费用和成交。GPU 首训拆分 CS 行业与市场分支。组合同时对旧空腿与现金验收 |
| P64、CS+P64 联训 | 固定 B0 多腿，联训相对 CS 在沪深验证 +2.87、沪深历史测试 +3.48 基点/周。早期深市诊断方向相反 | 联训用已有分数进入复核。按市场、流动性和费用归因，纯 P64 保留为容量对照。现阶段两臂均无需重训 |
| W20 Top-40、W80 Top-20 | 修正后的标准验证窗融合分别 +6.26、+8.15 基点/周，逐年均正。历史测试分别 −3.51、+0.84 | 保留已有模型。W80 继续低成本持仓与风险诊断，W20 暂缓自然覆盖重训 |
| CS 纯多、F、Q0、Q2 | CS/联训长腿在初版历史复检明显落后。F 冻结后同窗 L40 验证 +1.75 bp/周且 30 bp +1.70，已排固定冻点 seed43；Q2 三种子 L20 验证 +2.04、已读历史 −3.04。Q0/Q2 纯多存在轮窗或种子差异 | CS/联训长腿暂缓。F 用同窗补种子；Q0 和 Q2 只在有明确延迟预算时用存量权重测端到端效率 |
| 候选重排、独立第 2 周头 | 值得做新试验。旧 MLP 固定成员只调权，旧多周实验用累计或平滑目标 | 先做冻结候选的线性/MLP 重排。双期限首训保留第 1 周主任务 |
| 固定日线分块、末端一致性、换仓策略 | 按条件验证。已有同股掩码预训练缺乏经济增益 | 先验证日线增量和固定换仓。通过后再开动态切块或多步策略 |
| 其它架构及生成模型 | 独立研究储备。G256、C3、全池 H32、旧 A2 已有负结果或仅表示预测线索 | 只在下面列明的新信号或资源条件出现时登记 |

[11 月 13 日复验](research_20261113_weekly_reacceptance.md) · [上轮验收](../outputs/weekly_architecture_v1/acceptance.md) · [旧历史配对](../outputs/weekly_architecture_v1/eval/history_comparisons.json) · [三种子阶段结果](../outputs/weekly_architecture_v1/stage3c_summary.csv) · [上轮执行手册](research_20261111_weekly_architecture_runbook.md)

## 1. 启动顺序和耗时

机器上的 8GB 4060 只运行一个 CUDA 作业。训练、GPU 评分、GPU embedding 导出都占同一个队列。CPU 做数据对账、清洗、账本、打分融合、小头训练、配对分析。CPU 线程和进程优先级限制到训练的 GPU 利用率不明显下降，建议先用每个 Python 分析进程 2 个线程。以下为排班估算，以实际日志为准。上一轮 seed42 的 8 轮实测 B0 38.1 分钟，W20 23.5 分钟，W80 78.4 分钟。每个新 GPU 臂首轮记下 10 个完整 batch 的中位时间与峰值，校正剩余预算。

| 时段 | GPU 串行队列 | GPU 工作期间的 CPU 任务 | 完成标志 |
| --- | --- | --- | --- |
| 0–30 分钟 | 核对设备和占用。冻结开发快照和 3 个 B0 run 后，给 CS 行业拆分做 CPU smoke | 核对全验证逐周配对和旧测试已读范围。登记新版协议，跑队列首条命令的路径与配置预检 | manifest、新版 prereg、smoke 第一版。第一条可跑任务已入队 |
| 约第 0.5–2 小时 | 先训 CS 行业 seed42，出分后接 CS 市场 seed42。第二臂只改配置，可在第一臂训练期间立即挂到队列 | 写事件账本和数据样例。重放 B0/CS/P64/JOIN 与现金。补齐有存量分数的纯多全验证、逐年和压力档。构造两期限标签和辅助头 smoke | 旧结果对账，CS 两种覆盖读数待 CPU 自动生成，H2 配置准备好 |
| 约第 2–3 小时 | 先出候选重排需要的固定轮 embedding，再训 H2 seed42 | CPU 按时间折拟合线性和 MLP，验收零更新的 Top-50 逐周复现。CS 同覆盖结果齐后判断追加种子。准备换仓模拟和费用压力 | H2 首头日期一致，R50 折外拼接验收。后续任务随结果原子入队 |
| 约第 3–5 小时 | CS/H2 过对应验证门的家族各至多一个赢家追加 43/44。固定日线分块与同容量控制在输入和读出 smoke 完成后接入 | 完成组合、漂移、整手、极端周分析。按全验证或折外比较换仓和重排的父项。准备自然覆盖和历史冻结清单 | 晋级或停止理由齐全，复检前冻结 arm、权重、日期、费用和账本 |
| 余量 | 条件通过的日线、自然覆盖、末端一致性或增量臂。GPU 有空才补非关键诊断 | CPU 读完成产物，跑 4/8/13 周配对块区间和分年，生成同起点三账本及部署代理与现金对照。校验输出完整性 | 队列无未处理任务或每条均有明确阻塞原因，汇总一次验收 |

预算按首批三条 seed42 训练约 2–3 GPU 小时估算，另加评分和少量导出。补种子每个赢家通常加两条约 40 分钟。日线或联合臂预检后再报时长。资源紧先做 **CS 口径核对及拆分、R50、H2、窗口长度只读复检**。发现关键代码口径问题时暂停受影响的依赖链，继续跑其它已验收任务。

### 1.1 路径和入口

- 后续用 `outputs/weekly_followup_v1/` 的现有子目录。原预注册和失败尝试作为已发生记录。按本手册另存 `registry/prereg_validation_v2.md`，写明启用日期、旧产物已读范围和冻结的新判据；结果登记版本。新评估另用带版本的路径。旧 `outputs/weekly_architecture_v1/` 及所有模型、基线和现行交易流程只读。
- 读取上轮 `base_dev.yaml`。用新根目录引用一份截至 2021-01-01 的只读开发快照，可以直接引用上轮已截断的 `dev_data`。冻结其哈希。保留 `train_end=939`、`valid_end=1094` 排他、`matched_lookbacks=[20,40,80]`、`min_train_anchor=80`、`epochs=8`、`batch_size=16`、`learning_rate=8e-4`、`save_epoch_checkpoints=true`、`save_epoch_scores=false`。同因素两臂同步修改任何预算或掩码。
- [run_weekly_architecture.py](../scripts/run_weekly_architecture.py) 已支持 `--root`、`--dev-data`、`--device`，历史导出已有单独入口。训练与开发评分都显式传同一 `--dev-data`。昨晚 CS 两臂四次尝试均因省略它而停在路径检查，且 `CS_market_s42_a2` 的命令误指向行业配置。重试前逐项核对任务 ID、YAML 的 `cross_summary_parts`、run 目录、三个输入路径及数据哈希，使用新 attempt。保持历史配置、输入及每轮 checkpoint 哈希校验。复检前缀检查 symbol 与行业顺序、market、周特征、有效掩码及新增输入和标签的已成熟区间。
- 继续用 [eval_weekly_architecture.py](../scripts/eval_weekly_architecture.py) 的融合、日历、券源和代理重放函数。旧 `compare` 的 `comparisons.json` 仍以 `DEV_YEAR=2020` 判门，只用于兼容重放。新 [followup_compare.py](../scripts/followup_compare.py) 从原始分数按 §2.1 重算逐周曲线，L40 等缺少压力档的书从原始分数按相同日期重放。涉及续持、限卖与部分换仓时用第 3 节事件账本。

已落地的执行入口（历史输出已存在，重跑须指定新输出目录，不覆盖）：

- [prepare_weekly_open_sessions.py](../scripts/prepare_weekly_open_sessions.py) 和 [prepare_week2_open_labels.py](../scripts/prepare_week2_open_labels.py)：构建首个市场会话和 H2 v3 标签；源为 `data/sw_file_20/raw_ak2000`。v3 跨整周休市按下一个有交易的周顺延。
- [followup_replay_baselines.py](../scripts/followup_replay_baselines.py) 与 [followup_compare.py](../scripts/followup_compare.py)：前者重放多种已存基线，后者对配对分数执行全验证、历史、成本及券源口径，历史模式必须指定 `--phase history --b0`。
- [weekly_followup_ledger.py](../scripts/weekly_followup_ledger.py)、[followup_turnover_rules.py](../scripts/followup_turnover_rules.py) 与 [followup_one_step_turnover.py](../scripts/followup_one_step_turnover.py)：日账本、固定换仓和时间折外一步策略；已实现历史账本及拟合截止，不是实际券商成交记录。
- [export_weekly_candidates.py](../scripts/export_weekly_candidates.py)、[followup_r50.py](../scripts/followup_r50.py)、[followup_r50_history.py](../scripts/followup_r50_history.py)：冻结表征、按时间折重排和冻结后已读历史复检；[followup_r50_capped.py](../scripts/followup_r50_capped.py) 是事后探索。
- [prepare_day8_cache.py](../scripts/prepare_day8_cache.py) 与 [followup_day8.py](../scripts/followup_day8.py)：固定 8 周日块及同容量周线对照；[followup_h2_diagnostics.py](../scripts/followup_h2_diagnostics.py)、[followup_h2_economic.py](../scripts/followup_h2_economic.py)、[followup_h2_cs_route.py](../scripts/followup_h2_cs_route.py) 分别验 H2 机制、第二周经济结果与多空组合。
- 实际运行状态、数据缺陷、复现命令和结果请先查 [后续执行记录](research_20261113_followup_execution_results.md) 及 `outputs/weekly_followup_v1/registry/`，不要将其中已读历史视为新的独立测试。

从仓库根目录用 PowerShell 执行。示例把 GPU 训练和评分放进同一任务。CPU 先跑兼容重放，随后从其周度输出生成新版验证判门。

~~~powershell
$r = 'outputs/weekly_followup_v1'
$d = 'outputs/weekly_architecture_v1/dev_data'
$cfg = "$r/configs/CS_industry_s42.yaml"
$run = "$r/runs/CS_industry_s42"
python -u scripts/run_weekly_architecture.py train --root $r --dev-data $d --device cuda --cfg $cfg --run $run
python -u scripts/run_weekly_architecture.py scores --root $r --dev-data $d --device cuda --cfg $cfg --run $run
python scripts/eval_weekly_architecture.py compare --candidate $run --control outputs/weekly_architecture_v1/runs/B0_s42 --epochs 2 3 4 --data $d --out "$r/eval/CS_industry_s42_legacy"
~~~

### 1.2 增量队列

用一个小 Python worker 扫描任务目录。可在 GPU 空闲时复用 [weekly_arch_orchestrator.py](../scripts/weekly_arch_orchestrator.py) 的子进程调用。队列按任务 ID 和优先级排序，每个任务先写临时文件再原子重命名发布。CPU 进程只写任务请求，不直接启动 CUDA。日志记录 PID、命令、提交时间、开始/结束、退出码、配置和数据哈希。验收脚本可在 score 文件完成后立即产出同名 gate JSON，执行 agent 读取后提交下一任务。训练失败保留当次目录，修复后用新 attempt ID 重跑。

任务 JSON 示例。

~~~json
{
  "id": "CS_industry_s42_a1",
  "priority": 10,
  "needs": ["smoke_cs.ok"],
  "commands": [
    ["python", "-u", "scripts/run_weekly_architecture.py", "train",
     "--root", "outputs/weekly_followup_v1",
     "--dev-data", "outputs/weekly_architecture_v1/dev_data",
     "--device", "cuda",
     "--cfg", "outputs/weekly_followup_v1/configs/CS_industry_s42.yaml",
     "--run", "outputs/weekly_followup_v1/runs/CS_industry_s42_a1"],
    ["python", "-u", "scripts/run_weekly_architecture.py", "scores",
     "--root", "outputs/weekly_followup_v1",
     "--dev-data", "outputs/weekly_architecture_v1/dev_data",
     "--device", "cuda",
     "--cfg", "outputs/weekly_followup_v1/configs/CS_industry_s42.yaml",
     "--run", "outputs/weekly_followup_v1/runs/CS_industry_s42_a1"]
  ],
  "outputs": ["runs/CS_industry_s42_a1/dev_scores.parquet"]
}
~~~

worker 保存为 `scripts/weekly_followup_queue.py`，从仓库根目录执行 `python -u scripts/weekly_followup_queue.py --root outputs/weekly_followup_v1`。依赖、预检 `.ok` 文件和状态均相对于批次根目录。用 `Path.open("x")` 排他创建 GPU 锁，退出时释放。若 worker 异常退出，先查记录的 worker、子进程 PID 和 CUDA 占用再清理旧锁。运行第一条任务前确认旧训练或推理进程已结束。另开 CPU 终端继续实现和跑评估，由 CPU 端写 `inbox/<id>.json`。用 `subprocess.Popen(..., shell=False)` 逐命令执行并把 stdout、stderr 导入 `logs/<id>.log`，CUDA 子进程退出并校验产物可读后再取下一条。任务完成用实际退出码和产物验收，心跳只表示 worker 存活。

~~~python
while True:
    pending = read_new_jobs_sorted_by_priority()
    pending = mark_failed_dependencies_blocked(pending)
    ready = [job for job in pending if dependencies_ready(job)]
    if ready:
        job = ready[0]
        write_state_atomically(job, "running")
        exit_codes = run_serial_and_stop_on_failure(job["commands"], log_path(job))
        ok = len(exit_codes) == len(job["commands"]) and all(code == 0 for code in exit_codes)
        ok = ok and validate_outputs(job["outputs"])
        write_state_atomically(job, "done" if ok else "failed")
        continue
    if queue_closed():
        mark_unmet_dependencies_blocked(pending)
        break
    write_idle_heartbeat()
    time.sleep(5)
~~~

失败依赖标 `BLOCKED` 后从 pending 移除。worker 完成空队列时先待命，等所有首筛读数和增量决定登记完才写 `queue/closed`。收到关闭标记后，将仍无法满足依赖的任务标 `BLOCKED` 并退出。至少验证一次追加两个假任务、一个失败任务后仍能运行不相干任务、已经完成任务不会重跑、关闭时缺依赖可收尾。所有评估脚本逐臂可重复读，输出先写临时文件，完成后再改名。禁止对半写 parquet 开始分析。

## 2. 窗口、判据和增量规则

### 2.1 时间协议

1. **标准切分**按决策锚点用 2000–2017 训练（0–938）、2018–2020 验证（939–1095，含缓冲锚点 2021-01-01）、2021–2026 测试（1096–1398，2021-01-08 至 2026-10-23 的快照）。开发只读截至 2021-01-01 的物理快照。验证原始 157 个锚点；共同一周账本 152 个可评估决策周，按年 51/51/50。排除三周无市场观测和两周标签未成熟，逐条落盘。H2 等期限独立记录成熟覆盖，主 H1 账本保持全验证共同周。用 `evaluation_dates` 生成日期清单，明确控制与候选双向日期差，同时保存 `decision_year` 与 `exit_year`。
2. **纯多主干与架构**在 2000–2017 拟合，晋级主读数用 2018–2020 全验证、逐年与压力档。2020 单年用于快速诊断并单列读数。已有分数先做全窗配对再决定追加 GPU。单种子 42 先跑，主轮窗 [2,3,4]；43/44 固定 [1,2,3]/[5,6,7]。同一主干逐轮每周 z、均值、再每周 z，之后跨 seed 等权。全窗主比较用固定轮窗。额外用 2018–2019 选轮的读数标明这些周参与选模，保留未参与该次选轮的分段成绩。历史上参与候选或轮窗设计的读数也要标注。
3. **学习小头与换仓**用 §2.2 的按时间推进验证。所有特征、标签、名单和价格在其实际可用时点进入管线。每张标签表保存 `decision_at, entry_at, exit_at, label_available_at, valid`。按 `label_available_at < fit_cutoff` 选择训练及选轮行；未来信息变动不应改变更早的订单。
4. **空头和多空**分别生成同覆盖深市空腿诊断（2018–2020 有深市快照的 139 周）和具备沪深两侧名单的主验证周（2020-04-03 起 38 周）。按 §3 验证 `.SS` 映射，逐周用严格早于决策的快照并列出各市场的合格池、入选数、填单数。候选、旧空腿和同一多头加现金对照共用日期、长腿预算和费用。早期深市诊断支持归因、跨年稳定性和补种子判断。两个覆盖口径分歧时登记为待证，用存量分数做覆盖与流动性归因后再分配预算。
5. **标准测试与前瞻**在候选、父项、权重、轮窗、日期掩码、费用及交易规则冻结后，整段跑 2021–2026，报全窗和逐年。当前快照从 2021-01-08 起至 2026-10-23，除六个无决策休市周和两个未成熟尾周后有 295 个决策周，末个为 2026-10-09。已有历史结果是已暴露复检，试验台账标记已读臂和日期。验证晋级和增量配方在本次测试前决定。历史复检可揭示过拟合、风险和成本失效；根据历史改配方后将新方案记为探索，冻结后用新数据确认。前瞻从冻结后首个可评分锚点开始，26 个成熟周中报、52 个成熟周复审。

### 2.2 小头和换仓的时间折

1. 固定 §5 的 2018 前主干 vintage、特征、训练轮数和参数。默认 2018 上半年积累初始化标签；从 2018 下半年开始，每半年一折，共五折，起点为 2018-07-01、2019-01-01、2019-07-01、2020-01-01、2020-07-01。每折取起点以前所有已成熟标签拟合缩放、校准和头，模型仅预测该折的后续决策周。每折至少有 20 个有效拟合周。若达不到，登记最早可拟合折与缺周原因，基线跟随同日期。如果已有同口径且冻结的 2018 前时点外主干分数，可预登记更早的起折。
2. 默认每折使用固定轮数。需要选轮或超参时，只在该折拟合段内部做更早拟合和末端选择，按同一成熟规则清除跨界标签。把选择区间、拟合截止时点、最晚成熟标签、配置哈希和 `fold_id` 写入折表。外层折成绩的拼接值用于比较候选和对照，另报逐折与逐年；全期模型选择已经使用过拼接读数。
3. 将折外意图按决策时间拼接，每周唯一。换仓账本在相邻折之间传递实际持仓、净值、现金和残单，新头仅接管动作。用相同启动持仓与只从当时可得信息产生的预热路径初始化所有比较臂。测试默认冻结拟合截止和最终模型。用标准验证快照内全部已成熟样本重拟合一次后锁权重，测试全窗使用该版本。若登记滚动策略，锁重训规则并单列，按日期逐折消费当时可见的新标签。

### 2.3 同一指标决定追加预算

这些判据是本轮建议的默认值，开跑前写入 prereg。**+3 基点可继续作为高优先级线，小增益另有一条低成本复核路径。**

同一账本、日期和敞口下，全验证窗的扣费后平均收益或整体 Sharpe 提高都可作为追加复核的理由。若 Sharpe 改善来自波动降低，同时报告平均收益、回撤、费用和压力档。平均收益下降时将收益与风险的交换量写清，由相同名义和风险预算的父项作对照。两项均以逐周实际净收益计算，年度 Sharpe 不直接取平均充当全窗 Sharpe。小幅变化及已读测试的判断保留不确定性。

| 状态 | 判断与动作 |
| --- | --- |
| `CONTINUE` | 同账本、同名义、同成本、同周比较最强父项。主档净均值为正，前后半或独立时间块至少多数同向，剔除最大绝对差周后仍为正，且压力档与风险没有抵消收益。改善接近或超过 +3 基点优先；低于 +3 可复用现有分数、补一个种子、跑更强对照或进入影子。分别记录统计区间和风险变化 |
| `RISK_OR_EFFICIENCY` | 预先选择目标。若目的是省算力，记录同机 ≥25% 端到端提速，净均值损失预算预登记为 ≤2 基点/周。若目的是降换手，预登记相对下降 ≥20%、净均值损失预算 ≤2 基点/周。风险型比较同敞口与 Top-40，并同时看尾部损失和回撤。发生风险交换时说明交易账户的实际预算 |
| `INCONCLUSIVE` | 净均值虽正，配对区间仍覆盖显著亏损，或贡献集中于少数周、分数融合才赢、费率变化后转负。先安排最便宜且能缩小不确定性的动作，禁止用增加参数网格弥补样本不足 |
| `STOP` | 既无扣费后改善，又无事前指定风险/效率增益。机制目标过门但硬账本输的臂标 `MECHANISM_ONLY`。数据无法支撑可成交或估值时标 `BLOCKED_DATA`，可继续可识别的代理或纯多分支 |
| `SHADOW` | 规则和版本已冻结，有充分的历史方向证据或成本后风险改善。明确账户、现金对照、券源与成交记录。继续积累可得时点的前瞻订单和真实成本 |

晋级预算按 §2.1 对应主窗口决定。净差为正，同时两个半段均正或连续四分段中至少三段为正。剪去绝对差最大的五周仍为正。30 bp/边的压力档相对最强父项不得低于 −2 bp/周，最大回撤恶化不超过 2 个百分点。满足条件且净差 ≥3 时，家族胜者可同时排 43/44。净差在 0 到 +3 时先做存量重放或只补 43，同向才补 44。2020 单年纯多的正负均不能决定追加或淘汰。未过收益门时按降风险或省资源门判断，也可停在机制验证。进入影子前给出正收益在该账户成本区间内的成立范围和失败费率。分年允许有负年，报告收益集中度和压力失效。主干候选完成三个种子后，按预注册规则冻结家族胜者。CPU 小头用重复小头种子和冻结后的历史检验核对随机性。

开发同周报告净差、NW5、配对 8 周块 95% 区间，附 4/13 周。时间不连续处显式断块，按同一日期索引联合抽样。报告去掉绝对差最大 1/5/10 周、各年度及去掉贡献最大的年。临时查看的候选、费用和权重记入 trials，不当作独立确认。压力用 10/30 bp/边和 8%/12% 年券费的预设格子，取得真实报价后单列。单纯空腿相对旧空腿的增益和相对现金的增益分别验收。

## 3. 先对齐重放，再实现换仓账本

1. 用冻结分数和原 `replay` 复现旧结果。对齐旧 2020 的 50 周读数和旧历史的 301 个日历锚点，使用相同起点、费用、缺价处理和打分路由。逐周检查净值差绝对值 ≤1e-10，发现日期缺漏先用外连接报告。新版报告另按 §2.1 取共同实际决策周，整市休市不生成新意图。真实账本在休市时延续持仓和估值。沿用旧交易代理的读数标 `legacy_proxy`，旧函数每个决策周按原始目标重新生成持仓，受阻退出使用收盘代理。
2. 在 `scripts/weekly_followup_ledger.py` 实现换仓账本。输入是决策表、上一账户状态、按时间排序的可成交事件、各会话估值和费用。决策时先生成目标和有序候补。计划量按决策时 NAV、当时可得参考价和最小交易单位计算。首个开盘以后才能知道开盘成交。若拒单，用仍可成交的下一事件处理预排候补，或保留现金。使用开盘价格且无逐笔事件的回放保留未成交仓位和空槽。冻结被拒绝的卖单或回补，持仓带到下个可成交时点。无可成交价格时保留最后合法 mark，单列陈旧估值、持仓天数和不利情景。
3. 用签名股数和账户现金核算。先处理公司行动、回补和减仓，按可用现金、券量、冲突和总敞口约束处理增仓。买入/回补减少现金，卖出/融券卖出增加总现金，融券抵押计入受限现金。可用现金要扣受限资金和预约保证金。每笔扣实际换仓费，持空按日历日计券费并按约定处理召回。总 NAV = 总现金 + 签名股数 × 合法估值。严格记录所有敞口变化。等权是目标金额，成交后受漂移、整手和拒单影响。持仓缓冲和部分换仓只更改目标成员，用同一执行器决定成交及现金。HFQ 用于复权收益代理。真实整手、价格上限和交易费用的重放需要未复权成交价、公司行动及逐日交易状态。缺字段时单列连续份额代理，真实成交验收标 `BLOCKED_DATA`。
4. 建议同一个 driver 同时输出三个账户。旧研究账户 L20/S10/LS50，等名义的多头加现金对照，以及 [当前部署代理](live_deployment.md) 对照。部署对照用已发布的两套 vintage 及实际缓冲规则复现。历史时段取不到当时的发布分数或有数据发布时间缺口时明确标注，已有 B0 研究分数仅用于 matched 研究比较。无券源的 A 股历史空头以时点前名单和交易量代理、不同可借覆盖与费率做情景。现金对照用相同账户净值、固定 50% 长仓预算与每周实际成交，现金收益假定 0 或采用预注册的同一利率，禁止把空腿单账本加总冒充多空组合。
5. **券源映射预检。** canonical 里 352 只 `.SZ`、228 只 `.SS`，名单目录的沪市键为 `SH`。旧 `attach_short_eligibility` 对 `.SH` 后缀做直匹配，旧历史事件的 11,800 条空头事件全部为 `.SZ`。把交易所映射加到新版评估路径，保留原 symbol 供打分、行情和事件表连接。旧模式只为精确复现显式调用。新模式对三个合成标的验收，`600000.SS` 用 `SH`、`000001.SZ` 用 `SZ`、当日或未来的沪市快照不能让标的提前合格。缺后缀、未知市场、跨市场重复代码及时间边界都要有明确校验。实数据还要核对沪深均有合格候选及逐周事件，任何一侧整体空白先查映射。

券源函数保留 `frame.symbol` 原值，在按市场分组之前映射后缀。保留现有严格向前查找和按快照匹配代码的逻辑。

~~~python
symbol_venue = frame["symbol"].str.rsplit(".", n=1).str[-1].replace({"SS": "SH"})
for venue, snapshots in by_venue.items():
    rows = frame.loc[symbol_venue.eq(venue)]
    # Group rows by decision date and reuse the strict snapshot lookup.
~~~

输入输出至少固定以下字段。

| 表 | 主键与字段 |
| --- | --- |
| `intents.parquet` | `account, decision_at, symbol, side, rank, candidate, target_weight, target_qty, signal_vintage, known_at, policy` |
| `trade_events.parquet` | `account, event_at, order_id, symbol, qty_requested, qty_filled, raw_price, mark_price, fee, available_to_borrow, reject_reason, status`，状态含 `pending, partial, filled, rejected, expired` |
| `positions.parquet` | `account, session, symbol, qty, mark, mark_age, borrow_rate, pending_qty` |
| `weekly.parquet` | `account, decision_week, valuation_end, nav, gross_return, net_return, turnover, fee, borrow, long_gross, short_gross, cash, restricted_cash, blocked_n, missing_mark_n` |

最小验收样例。A 买入涨停，目标 A 未进入持仓。B 持仓后跌停，卖单未成交，持仓和价差进入后续 NAV。C 无历史券量，空头只出代理结果。半量只产生已执行数量对应的费用。双腿冲突先保留预注册的长腿。开盘事件晚于下单且晚到价格不会改变意图。价格恒定时，净损益等于费用及券费之和的负数。用一个 3 周路径逐步手算长、空及现金。现有 [weekly_risk_timing.py](../scripts/weekly_risk_timing.py) 的 `book_replay` 在非 legacy 路径不能作为成交门，已有不可买股票仍可进入持仓的反例。

## 4. CS 拆分、存量路由、窗口长度

### 4.1 首训 CS 行业和市场

从相同的 B0 配置复制两臂，分别用下列键。两臂共享参数形状及初始值。分支切换发生在行业与全市场 leave-one-out 摘要拼接前，主干仍使用原横截面和训练目标。

~~~yaml
cross_summary: true
cross_summary_parts: [industry]
~~~

市场臂为 `cross_summary_parts: [market]`。在 [hierarchy.py](../src/quant_mvp/hierarchy.py) 的构造函数里设 `self.cross_summary_parts = tuple(cfg.get("cross_summary_parts", ("industry", "market")))`。校验该项仅在 `cross_summary` 启用时允许显式设置，成员非空、无重复，限 `industry`、`market`。保留当前 `nn.Linear(2 * dim, dim, bias=False)` 和零初始化。在 `_cross_summary_context` 最后一行前加入以下代码。

~~~python
if "industry" not in self.cross_summary_parts:
    sector = torch.zeros_like(sector)
if "market" not in self.cross_summary_parts:
    market_mean = torch.zeros_like(market_mean)
~~~

1. 用三股、两行业、一个无效位置构造 CPU case，验证单股行业的摘要为零、屏蔽股不进入其它股票摘要、改变自身 embedding 不改变自己的 LOO 摘要。两个分支都开时 strict-load CS 并复现原分数。零初始化输出等于原输入。
2. 按队列先跑 seed42。与同 seed 的 B0 和完整 CS 配对，先看 `S10`、`LS50_B0long` 两个券源口径，附全验证 `L20/L40`、每周持空数量、空头换手、成交和空槽。先验证空头候选选择，再做行业、交易所、流动性分桶，报告持股重叠和条件毛收益；桶重配的诊断须报告可匹配比例。
3. 拆分臂若保持完整 CS 的净均值与压力表现，并能稳定定位来源、改善集中或减少费用，可追加 43/44。按 §2.3 排预算，优先追加在沪深主验证和同覆盖诊断都有支持的臂。相对完整 CS 有经济或风险改善才作为新策略候选。若两臂都输，冻结完整 CS 做影子，登记可能存在交互。

### 4.2 固定 B0 多头的存量重放

后续多空实验使用 [多空研究基线 v4](baseline_monopen.md#8-多空研究基线-v4) 的 B0 原多腿 + CS 第 4–6 轮空腿作为主对照，主档年券费 8.35%，同时报告同一 B0 多头加现金。`followup_replay_baselines.py` 默认回放 v4，保留 v3 原轮窗用于同费率配对。

读 B0、CS、P64、JOIN_CS_P64 已有三个 run，用对应固定轮窗融合。按 `date,symbol` 一对一连接 B0 的 `score_long` 和三者各自的 `score_short`。短候选共用当时快照、流动性门和相同长腿冲突规则。按 §2.1 输出两种覆盖的逐年配对和现金对照，保存 `route_existing/weekly_paired.parquet`、`trade_events.parquet`、`summary.json`。兼容上轮时从 2018 连续预热再切 2020，保持旧深市匹配。旧账本、10/30 bp 每边、8% 券费下 B0+CS 相对 B0/B0 为 +22.24/+21.45 bp/周，B0+P64 相对 B0+CS 为 −13.58/−12.89，B0+JOIN 相对 B0+CS 为 −1.95/−1.95。新版联训比较见 [11 月 13 日复验](research_20261113_weekly_reacceptance.md#纯空和多空)。优先追踪两种覆盖下差异的来源，追加联训要有独立的纯多或空头改善证据。

### 4.3 窗口长度纯多验证

1. 用存量逐轮分数重放全验证，主比较为 `W20 L40 vs B0 L40` 和 `W80 L20 vs B0 L20`。记录这两个主书已按历史研究路径被读取。主 10 bp，压力 30 bp。保持三个种子各自固定轮窗，报告逐年、逐种子、前后半、头部每十名收益、持股交集、换手和剪尾；另一持股宽度作为副读数。已读开发窗的 50/50 标准分融合如保留，标为探索性。
2. 现有 `histall_W20_s*`、`histall_W80_s*` 已有 2021–2026 分数，核对来源、轮次、日期、共同可用池后 CPU 复检。旧历史导出是探索性，版本已读。未来新方案需要历史导出时，用旧 `recheck_lock.json` 为模板，写入各 run 的 config 与对应 `epoch_N.pt` SHA-256，登记全量 weekly、market、universe 和券源的哈希。下面是固定轮窗导出示例，需时才执行。B0 历史分数的范围、来源与轮窗匹配时复用。

~~~powershell
$r = 'outputs/weekly_followup_v1'
$oldRun = 'outputs/weekly_architecture_v1/runs/W20_s42'
python -u scripts/run_weekly_architecture.py history-scores --root $r --device cuda --cfg "$oldRun/config.yaml" --run $oldRun --lock "$r/registry/w20_history_lock.json" --out "$r/eval/history_W20_s42"
~~~

3. 自然覆盖是独立假设，依据标准验证判门、覆盖归因和预算安排。11 月 13 日复验后暂缓排入首批，有新的覆盖归因才启动。已读历史表现用于提示迁移风险，此后新增方案按探索性登记。由 B0 和候选的 matched 配置派生，**两臂同时**删除 `matched_lookbacks`，保留共同 `min_train_anchor=80`。W20 保留 `lookback=20, context_lengths=[20]`、`min_history_weeks=16`；W80 对应 80、[80]、64；B0 对应 40、[40]、32。先各一个 seed。对各自自然池比较净结果，再在可用池交集重跑，拆开覆盖贡献与模型排序贡献，报告新增持仓数和上市年龄。通过该臂的全验证后再追加种子或做匹配宽度的增量臂。

## 5. R50 冻结候选重排

目标是测试小头能否改善 **Top-100 中最终 50 只的等权成员**。主对照为同一 100 只按原分数取 50。与 B0 的 L20/L40 横比只给背景。保持主干冻结，用两阶段监督。另一个头学习何时换、换几个见 §8。

1. 固定 B0 seed42 epoch3 的 checkpoint 哈希。它仅用 2018 前数据训练，本轮直接锁 epoch3。导出 2018–2020 特征时不使用旧 best 选轮或验证指标。若上轮轮次选择已使用这些日期，整个区间标为已读开发，后续另走冻结验证。通过 `HierarchicalRanker`、`arrays`、`make_batch` strict-load 和 `eval()`，保存 `z`、`score`、`available`。批次取 `z` [B,N,128] 与同次输出分数，按日期和 symbol 唯一写入。先与 `dev_scores` 的 epoch3 对账。后续 seed 按自身事前锁定的单轮编码各自拟合，最终融合分数。
2. 每个决策日按全截面可用原分数标准化，按分数降序、symbol 升序固定前 100 个。只有决策时可用的行业、停牌状态和仓位可作为输入。未来是否可开仓留给执行器，标签表只提供训练掩码。计划开盘无有效价格且无一致估值时保留候选，标签无效，另报缺失率与按分数分桶。未来价格修改不得影响候选和原特征。
3. 执行 §2.2 五折。每折只用拟合段缩放 embedding，保存拟合均值、标准差与缺失计数。线性与 MLP 用同一成对损失，CPU 跑 `AdamW(lr=1e-3,weight_decay=1e-3)`，固定 20 epoch，每次 8 周，训练 seed 42。用折外拼接的硬 Top-50 扣费结果比较两种头，逐折和逐年附报。晋级后用全部成熟验证样本重拟合一次并锁权重。每 13 周更新一次、最多取过去 104 周的滚动臂需单独登记并在验证段按时点重放。

下面是小头和边界损失。每周先按 symbol 排定输入，`base_z` 在进候选前由同周全池计算。用有效标签比例调整局部真实边界，候选全有效时边界为第 50 名。残差初始化与原排序完全相同。

~~~python
import torch
from torch import nn
from torch.nn import functional as F

class CandidateRanker(nn.Module):
    def __init__(self, use_mlp):
        super().__init__()
        self.delta = (nn.Sequential(nn.Linear(129, 32), nn.GELU(), nn.Linear(32, 1))
                      if use_mlp else nn.Sequential(nn.Linear(129, 1)))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, z, base_z):
        features = torch.cat((z, base_z.unsqueeze(-1)), dim=-1)
        return base_z + self.delta(features).squeeze(-1)

def cutoff_week_loss(score, target, valid, n=50, k=100):
    eligible = torch.where(valid & torch.isfinite(target))[0]
    if eligible.numel() < max(10, int(0.8 * k)):
        return None
    ordered = eligible[torch.argsort(target[eligible], descending=True, stable=True)]
    boundary = round(n * len(ordered) / k)
    if not 0 < boundary < len(ordered):
        return None
    upper = ordered[max(0, boundary - 10):boundary]
    lower = ordered[boundary:boundary + 10]
    spread = target[upper, None] - target[lower][None, :]
    weight = (spread / 0.005).clamp(0, 1)
    if weight.sum() == 0:
        return None
    pair = F.softplus(-(score[upper, None] - score[lower][None, :]))
    return (pair * weight).sum() / weight.sum()
~~~

每个 batch 只平均有有效成对梯度的周。记录 80% 门跳过几周，训练时 50 bp 以上收益差权重封顶。经济验收用候选全部当前可用股票，按分数、symbol 打破平手，无效标签不得剔除可交易意图。对同一 50 槽记录平均真实换人数、毛收益、成本、风险、行业/流动性偏移、现金及成交比例。零残差时要求名单、账本、周收益一致。再随机打乱同周训练标签一次，应不能稳定超过原头。只有实际成员调整带来增益，才考虑加候选内行业摘要或 1 层 4 头窄注意力。先做一个上下文臂，验证同容量单股头能否解释差异。

若任务切到纯多 L40，预先登记 Top-80→40，改函数 `n=40,k=80`，重新拟合，沿用同协议。空头可以用独立 Bottom-30→10。候选在时点券源、流动性、多空冲突过滤后建立，排序目标取负资产收益，`n=10,k=30`，按可借量及费用逐笔验收。历史券量不足时只跑同一名单代理诊断。两边 50 等权另开敞口/容量实验，特别记录空槽和成本。

## 6. H2 独立第 2 周辅助头

首批只训 2 个独立区间。沿用原 1 周主头与交易评估，新辅助预测同一个决策时刻出发的第 2 个持有周，权重 0.2。当前 `open_open` 忽略 `target_horizon`，改该键无效。旧多周均值把缺失标签填零的问题在本实验通过独立掩码解决。旧 H2 累计收盘信号相对同 K100 缓冲约 +0.44 bp/周。本地旧主头在 Top-100 内对独立第 2 周 RankIC 约 0.0051。先用这两个量级判断可观测信号。

1. 在 [hierarchy_data.py](../src/quant_mvp/hierarchy_data.py) 或独立预处理脚本生成 `week2_open_labels.parquet`。从统一市场会话日历取决策之后第 1/2/3 个**有交易的市场周**内的首个开盘会话 `O1/O2/O3`。整周休市跳过，不能按日历周偏移，否则会把第一交易周误记为第二交易周；决策周无市场交易时不生成有效标签。个股在计划开盘无合法打印时同样标缺失。对价格源做复权一致性检查，收益用 `log(O3/O2)`。排序训练截断 [-0.25,+0.25]，账本使用原始回报。字段包含 `date,symbol,entry_at,exit_at,label_available_at,ret,valid`。标签未成熟、非正价格、非有限值皆无效。开发版文件直接由截断日线/日历产出。历史冻结后用相同生成器处理长前缀，只有成熟标签参与诊断。
2. 对齐 `data[6:8]` 的日期与 symbol 排列。显式校验唯一键、未知股票、时区和 maturity。H2 特征和辅助标签各自物理隔离，训练加载器只消费已知数据。原 B0 训练 reserve 已是 5，理论上覆盖第 3 周开盘；仍逐行校验训练及验证成熟时间。主头训练周、逐周人数、有效掩码与 B0 的 `train_verification.json` 对齐。若必须改变主头日期，另生成同日期 B0 作对照。对保留 B0 配方的实验，`O1/O2` 统一首开盘标签另作机制诊断。若想替换主标签，之后单独登记 `unified O1/O2` 的共同基线和 H2 臂。

~~~yaml
week2_aux_weight: 0.2
week2_label_path: outputs/weekly_followup_v1/cache/week2_open_labels_v3.parquet
~~~

3. 在 [hierarchy.py](../src/quant_mvp/hierarchy.py) 的 `HierarchicalRanker.__init__` 增加以下训练头。保持 `forward()` 返回的四元组和 `predict()` 主分数格式。历史导出默认只需要输入特征和模型配置，导出 H2 评分时另外记录 `score_h2`。

~~~python
self.week2_head = None
if float(cfg.get("week2_aux_weight", 0)) > 0:
    with torch.random.fork_rng(devices=[]):
        self.week2_head = nn.Linear(dim, 1)
~~~

在 [hierarchy_engine.py](../src/quant_mvp/hierarchy_engine.py) 的 train 读入二维 `y2, valid2, mature2`。下例中 `mature2` 是对该训练折实际拟合截止时间预先构造的布尔数组，`mask` 是当前可用股。加在既有损失之后、backward 之前。分别记录 h2 损失、周数和有效股数。

~~~python
if model.week2_head is not None:
    y2_batch = torch.as_tensor(y2[ts], dtype=z.dtype, device=device)
    keep2 = torch.as_tensor(valid2[ts] & mature2[ts], device=device) & mask
    keep2 = keep2 & torch.isfinite(y2_batch)
    use_week = keep2.sum(dim=1) >= 10
    if use_week.any():
        pred2 = model.week2_head(z[use_week]).squeeze(-1)
        loss2 = rank_ic_loss(pred2, y2_batch[use_week].clamp(-0.25, 0.25),
                             mask=keep2[use_week],
                             temperature=cfg["rankic_temperature"])
        loss = loss + cfg["week2_aux_weight"] * loss2
~~~

4. smoke 必测价格 100→110→121 时两个独立标签都是 `log(1.1)`，第一段上涨、第二段下跌时方向不同，缺一个开盘会话时只屏蔽受影响的区间，训练边界跨界整段屏蔽，晚到价格仅改变标签，配置关闭时旧分数和主损失一致。模型初始化校验主干权重与 B0 相同，并验证 H2 梯度到共享 `z` 的上游。单种子在全验证验收 `L20`，附 L40 和同覆盖短腿，保持一周执行。若首头未改善但 H2 显著改善候选内未来排序，允许单开便宜的续持回放。H2 排序比较同候选上的旧 H1 对第 2 周收益的相关、增量 RankIC 和已持仓 vs 替代股票的差值。只有首头或第 2 周换仓价值过门才补种子。
5. 组合用途优先做「第一周入选、下一决策对已持有和替代席位评估留仓」。使用 §2.2 时间折，各拟合段用分箱或岭回归校准成预期净基点，随后的评分段冻结映射。只使用当时已产生的 h2 特征。若试两周持有，分别建每周各 50% 资金的两批持仓，逐日统一估值、冲突和净交易费，比较相同持有期且只用 H1 的两批控制。多周收益重叠不能按独立周收益重复记账。第 4 周留到确有跨期留仓价值才试，先登记为独立第 4 周，8 周暂缓。

## 7. 固定日线分块、末端增强与动态路由

### 7.1 固定日块先过门

保留 40 周主干和 30 周特征。最近 8 个日历周内，用每日隔夜、日内、高低振幅、过去 20 个有效交易日成交额中位数归一后的 log 成交额、距上一合法报价的日历间隔、块内会话位置共 6 项。使用同源 `raw_ak2000` 与 [周线生成器](../scripts/prepare_hierarchy_580_mon.py) 重聚合抽样。日期和单位保持一致。最近 20 日统计只取当前之前。特征缩放只 fit 到 2018 之前，保存缩放和已观测标记。交易所会话不是个股观测。停牌和假期无观测位置存零加独立 bool mask，拒绝在 token 编码前填成合法日；跨周成交日间距来自历史已知日期。建 dev 和 full 各一套只读缓存，记录 prefix 哈希。

实现 `daily_patch_path` 专用分支。预处理存 `[calendar_day,stock,6]` 与 `observed_day`，建 `[anchor,8,5]` 的会话索引，缺槽用 -1，batch 内安全 gather 后清零。若源日历发现超过 5 个会话的周，预检应失败并显式处理维度。通过 `make_batch` 原 daily 槽传 `(x, valid)`，`x[B,N,8,5,6]`，`valid[B,N,8,5]`。新增数据存储放在已有索引之后或命名属性，保持旧位置和无分支调用。旧 `daily_path` 与 `daily_patch_path` 互斥。先冻结主干和关系层，只用 2018 前已成熟的最近 104 个共同训练锚点训练小分支，使用同一 B0 checkpoint、固定轮数，按 2018–2020 全验证验收。冻结权重，关系层 eval，仍让分数梯度通过关系层进入小分支。能缓存周主干的 `temporal` 和 market，纯 CPU 拟合或短 CUDA 作业都可。通过后才做从 B0 初始种子联合训练与同配方的对照。续用原训练对比目标时，干净周表示作为教师输入。

同时实现 `WRES8` 容量对照，复用同一个模块。给每周第 5 槽填入既有六个周路径特征 `d_on,d_in,dfront,d_onn,d_back,d_mon`，valid 只标实际观测的周；其它四槽留空。它也读最近八周并拥有同样的参数量。报告两个臂的实际运行时间、有效槽数和结果，同时对比原 B0。

~~~yaml
daily_patch_weeks: 8
daily_patch_dim: 32
daily_patch_heads: 2
daily_patch_readout: last_valid
~~~

核心模块如下。初始化放入与 CS 相同的 RNG fork，输出为 [B,N,128]。在调用 `relate` 前加入 `temporal = temporal + self.daily_patch(*daily_windows)`。这里只把近期路径分支用于评分。若联合训练时沿用周教师，让 `forward()` 的第一返回值保持未加日分支的周 `temporal`，避免连原 InfoNCE 查询也改变。先按原批次的小尺寸验收，冻结阶段学习率 1e-3、weight decay 1e-3、固定 12 轮、固定主损失的 RankIC 及头尾项，验证硬 L20，WRES8 同预算。

~~~python
class DailyPatchResidual(nn.Module):
    def __init__(self, d_out=128, width=32, n_weeks=8):
        super().__init__()
        self.cell = nn.GRUCell(6, width)
        self.week_pos = nn.Embedding(n_weeks, width)
        self.attn = nn.MultiheadAttention(width, 2, batch_first=True)
        self.proj = nn.Linear(width, d_out)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, valid):
        batch, n_stocks, n_weeks, n_days, n_feat = x.shape
        days = x.masked_fill(~valid[..., None], 0).reshape(-1, n_days, n_feat)
        mask = valid.reshape(-1, n_days)
        h = x.new_zeros(days.shape[0], self.proj.in_features)
        for day in range(n_days):
            proposed = self.cell(days[:, day], h)
            h = torch.where(mask[:, day, None], proposed, h)
        tokens = h.reshape(-1, n_weeks, h.shape[-1])
        week_ok = valid.any(dim=-1).reshape(-1, n_weeks)
        pos = torch.arange(n_weeks, device=x.device)
        last = pos.expand_as(week_ok).masked_fill(~week_ok, -1).amax(dim=1)
        safe_keys = week_ok | ((last < 0)[:, None] & (pos == 0)[None, :])
        keys = tokens + self.week_pos(pos)[None, :]
        rows = torch.arange(keys.shape[0], device=x.device)
        query = keys[rows, last.clamp_min(0)].unsqueeze(1)
        pooled, _ = self.attn(query, keys, keys, key_padding_mask=~safe_keys,
                              need_weights=False)
        result = self.proj(pooled.squeeze(1)).masked_fill((last < 0)[:, None], 0)
        return result.reshape(batch, n_stocks, -1)
~~~

验收无日数据时残差为零，padding 中的极端值和 NaN 不影响评分，块尾有效日的改动在非零头时可传到输出，修改决策后的日线没有路径进入过去评分。单周、短周、复牌、新股、全缺失逐一验收。初始零残差比对 B0 分数。开发看 DAY8 相对 WRES8 与 B0 的 L20，同时报 L40、S10、覆盖、峰值和端到端速度。日线效果大致同向才追加种子。首轮预算无法同时容纳两臂时顺延这组，保留对照完整。

### 7.2 H-Net 风格动态分块及最后 token

**已准备的确定性对照。** [prepare_dynamic_day8_cache.py](../scripts/prepare_dynamic_day8_cache.py) 在每个截断决策窗口内压缩可观察日，按 `abs(隔夜对数收益+日内对数收益)+0.01*abs(成交额相对值)` 的固定惊讶度，动态规划出最多 8 块、每块 1–8 个有效会话；并列取较早边界。缓存 `outputs/weekly_followup_v1/cache/dynamic_day8_v1.npz` 只包含 104 个成熟训练周和标准验证周，不把后续会话输入前一决策。单元测试和 100 个真实股票窗口逐日对账验证了缺失掩码、有效会话保序及末块读取。

[followup_dynamic_day8.py](../scripts/followup_dynamic_day8.py) 复用 DAY8 的 GRU32、两头注意力、零投影及同一冻结 B0 epoch3，训练同样 12 轮；[followup_day8_monitor.py](../scripts/followup_day8_monitor.py) 仅在 seed43 日线对 BASE/WRES 的两种子方向与费用条件成立时把 DYN8 seed42 入队，[followup_dynamic_day8_monitor.py](../scripts/followup_dynamic_day8_monitor.py) 按登记结果决定是否追加 seed43。结果应与 DAY8、WRES8、BASE 同种子/同成本比较。这是固定惊讶度的动态切块，不是可学习 H-Net；若未触发，保留准备与停止理由，不运行额外 GPU。该对照在观察 seed42 验证后提出，标为事后探索。


只有固定日块对同容量周残差有可重复的扣费增益才排这项。用同一批近 8 周日输入、同一个 32 维因果日编码、同样的最终 attention 与 ranker。固定周块、同 token 数的等长块、确定性事件边界和可学习边界的比较应共享块内汇聚。连续值直接映射为向量。首个预算每股最多 8 个有效块，极少交易的窗口用更少块，单块长度 1–8 个有效会话，结束日与上块间距作为位置输入。最后一块强制闭合至最后有效日。

可学习边界作为二阶段独立消融。将过去的因果日状态映射为 16 维相邻投影，余弦差给每个合法 cut 的分数。cut 表示**在该日之前**关闭上一块。动态规划在 1–8 个有效会话的限制下挑 `K=min(8, n_valid)` 块，最大化 `K-1` 个选中 cut 的分数和，平分时取较早边界。先把分数换成仅由过去可见的绝对收益与量能惊喜做 CPU 路由对照。若学习切分，直通分配的硬路由和梯度路径可按以下单股核心实现，批量 padding 放在模块外。它的路由使用整个已截断的决策窗口。要为每天产生可用前缀预测，需逐决策截断重算边界。

~~~python
# states [D,32] contains only observed sessions, D >= 2.
# cuts returns K-1 sorted integers from 1 to D-1 and checks lengths in [1,8].
def routed_chunks(states, left_proj, right_proj, cuts):
    count = states.shape[0]
    k = min(8, count)
    logits = -F.cosine_similarity(left_proj(states[:-1]), right_proj(states[1:]), dim=-1)
    starts = cuts(logits.detach(), count, k)
    starts = torch.as_tensor(starts, dtype=torch.long, device=states.device)
    day = torch.arange(count, device=states.device)
    hard_idx = torch.bucketize(day, starts, right=True)
    hard = F.one_hot(hard_idx, num_classes=k).to(states.dtype)
    p = (logits.float() * 4).sigmoid()
    soft_idx = torch.cat((p.new_zeros(1), p.cumsum(0))) * (k - 1) / p.sum().clamp_min(1e-6)
    bins = torch.arange(k, device=states.device, dtype=p.dtype)
    soft = (-(soft_idx[:, None] - bins[None, :]).square() / 0.25).softmax(dim=-1)
    soft = soft.to(hard.dtype)
    assign = hard + (soft - soft.detach())
    return assign.T @ states / assign.sum(dim=0)[:, None].clamp_min(1), hard_idx
~~~

零和一天的情况直接为空或单块。这个代码是最小可消融路由，首个 learnable 臂固定预算、只让主排序训练边界，无需另加压缩率目标。三臂统一用块内掩码平均及相同的因果日编码器。用每个块的末端日期加位置，取**最后一个有效块的 attention 输出**，或用相同周末 query 读全部合法块。原 H-Net 压缩取块起点，若直接读该向量会漏掉最后边界后的交易日；若接原版 H-Net 的解块解码器，则取最后有效日的解码表示。固定边界后改最后两天，验证能改变末端表示。另测 trailing padding、假期、仅有一日、截断后末块、边界长度、梯度可达路由、相同日输入与时间预算。报告 padding 后最长批、块数、显存、耗时和相对固定分块的净差。动态臂过门才讨论整段日线替换。

### 7.3 对比学习先做末端抗扰动

当前末端 InfoNCE 用同股未来表示，valid mask 排除无效样本；旧另一个预训练确实做过两视图随机连续 8/40 周掩码，一视图另遮末周，识别 loss 下去但硬 Top-40 未改善。若固定输入主线稳定且有缺失/噪声目标，追加一个单种子 `CONS_END`。在最近两周之前的 38 周中用训练独立 RNG 选一段长度 4，标准化输入填 0，保留原 availability，最后两周及市场特征不动，正式出分始终用干净输入。保留干净主损失和原 InfoNCE。只对时序末端归一化向量做 stop-gradient 一致性，权重 0.05。匹配种子、窗、优化步，增强多出的时间单列。

~~~python
aug = windows.clone()
# starts is one integer per batch date, sampled from [0, 34] with a dedicated RNG.
time_index = torch.arange(windows.shape[2], device=windows.device)
erase = (time_index[None, :] >= starts[:, None]) & (time_index[None, :] < starts[:, None] + 4)
aug = aug.masked_fill(erase[:, None, :, None], 0)
z_masked = model.encode(aug)
cos = F.cosine_similarity(z_masked, temporal.detach(), dim=-1)
nonempty = mask.any(dim=1)
if nonempty.any():
    date_loss = ((1 - cos).masked_fill(~mask, 0).sum(dim=1)
                 / mask.sum(dim=1).clamp_min(1))
    loss = loss + 0.05 * date_loss[nonempty].mean()
~~~

显存超限时优先把一致性支路缩为每步固定抽样的若干完整周，保持每个抽样周的全横截面和每周等权，记录实际监督数，和同预算控制对照。先看真实缺失和自然噪声分桶的下周排序及扣费表现。另取同股错位窗口做诊断，若只强化身份可分性且排序无改善，停止。逐 token 抗扰动只有末端过门才比较；仅选 3 个过去锚点，各用因果前缀、按唯一日期降重复权重、保持总监督强度，和末端同预算跑。不要把同一个原 token 直接当配对，也不把未来回报或未来状态送进推理。

## 8. 换仓策略、决策损失和其它低成本消融

### 8.1 两个小头各做一件事

R50 决定候选之间谁更值得持有。换仓头读取 R50 或 B0 的冻结分数、仓位和交易状态，决定今天换几个席位。首轮让它们分开，用 B0 固定分数先测换仓；组合时只换评分输入。同样的候选秩和风险预算下测试以下次序。旧门槛规则曾把周换手从 103.85% 降到 28.61%，但每周少赚约 6.96 bp，先前毛收益牺牲约 14.54 bp，因而要看毛利保留和真实费用。

1. 先用 §3 账本 CPU 重放。纯多首组固定 L20 等权。将候选原始目标 Top-20 和当前持仓的差集配成边际进出，先强制不可留或超预算退出，其它按候选分差从大到小换，`a∈{0,2,4}` 是最多的自主席位数，未满仓的新建和强制退出单列。另存完全换到 Top-20、K100 出场缓冲、L40 无缓冲为控制。初始空仓用相同 Top-N 建仓。连续持仓允许漂移；实际换手同时包括必要再平衡。可另比较半量 `w_prev + .5*(w_target-w_prev)`，独立报告持股数、有效持股数和偏离等权。以同等换手、同敞口或同有效持股数的控制判断增量。
2. 一步预测。每个开发周用同一当期状态分别重放动作 0/2/4，使用各自成熟后的 `log(NAV_end/NAV_start)` 作目标。持仓和状态来自事前的固定策略 rollouts，实际验证时模型从自己的历史持仓递推。状态只含决策前可得的持仓重叠率、边际换入/换出分差、过去 4 周分数持久性、漂移、可用现金、预估费用、量能、集中度和未成交比例。首个模型用带截距的岭回归，输入标准化后 `alpha=10`，预测各动作相对同起点动作 0 的基点差。每周等权，多状态同周拆分权重。预测最大且扣费后为正才换，自主阈值 0 bp，平手取换仓更少的动作。使用 §2.2 五折，在同一折外路径配对固定动作和缓冲基准。
3. 一步有信号再做两周滚动优化。动作空间仍为 0/2/4，枚举 9 条序列，第二周用拟合段估计的分数衰减与真实联合残差的滞后块采样生成 32 个已知模型情景。无 H2 使用历史估计，验证 H2 时只替换这一预期输入。每条路径走相同持仓转移和费用模型，目标为两周净对数收益均值，先执行第一步，下周重求。回放预测时无法读取待交易周的真实分数、成交或收益。两步失败时不升三步。单列预测信号失败、费用预估偏差与成交约束影响。
4. 只有两步价值在独立周块和费用档下可复现，且足以支付系统复杂度，才训外接有限动作 fitted Q。用同一动作、同一账本、冻结评分。历史反事实只在可核验的成交状态和低冲击范围使用。记录各策略生成状态、动作覆盖、拒单和后继持仓；无覆盖动作禁止优化。可先用岭回归版本，真需要非线性时用 `Linear(d,32) → GELU → Linear(32,3)`。沿用 §2.2，用 Huber 监督 `r_t + 0.95 * max_allowed Q_target(s_next)`，每次迭代固定上一轮估计，首个目标为一步回报，预先固定 4 次迭代，AdamW lr 1e-3、weight decay 1e-3、每迭代 10 epoch。拟合含后继状态的目标只用跨期路径已成熟的行。净增长已扣费。另与无持仓状态模型、固定规则、一步和滚动优化配对。若策略生成的状态落入训练支持外，回退到预注册固定动作。用完整路径验收收益、实际换手、日回撤和最差 10% 周。
5. 控回撤单独登记。等敞口固定规则、L40 分散化和固定低敞口是控制。预设风险门为最差 10% 周损失均值相对改善 ≥10%、净均值损失 ≤2 bp/周、最大日回撤幅度无恶化。只在有收益后效信息或风险校准确实改善时试择时，使用决策时已知的 NAV 和 HWM。8% 券费等历史代理不提供可回测的券源覆盖，空腿先走影子订单与报价对账。策略对第 3 节账本或数据有依赖时，记录阻塞并让其他线继续。

### 8.2 加入模型损失的顺序

冻结 R50 排序先过门。把实际持仓可保留的 A 和候选 B 组成样本，标签为同区间 `(r_B-r_A)-c_swap`，B/A 收益用简单收益，`c_swap` 是按同一单槽名义折算的卖出、买入、冲击和券费差。训练前由可用仓位生成可选对，最终可成交由账本验证。用 §5 同一线性或 MLP 结构和固定候选，监督损失改成这组边际偏好，或与原候选损失按 0.8/0.2 组合，只选一个预注册权重。单槽名义下 10 bp/边约需 20 bp 毛差覆盖，每周整个 L50 组合换一槽约付 0.4 bp。验收用硬入选/等权和同账本，代理与硬账本反向则停。

若想把决策损失传回主干，只在冻结小头稳定之后加三个同预算臂。冻结小头，解冻末层仅用原监督，解冻同层并加决策项。保留原排序，使用连续周段、有上期漂移的持仓和扣费净对数增长作可微代理，`lambda_decision=0.1`，在训练段先缩放使两个损失量级可比。至少 4 个连续周每段，用更早决策预热仓位和现金，固定目标股池并记录软目标与硬执行的偏差。四层整塔解冻等这个对照通过才评估。禁止对随机打乱的周 batch 算组合回撤，或用未来已实现的成交决定训练期的当周意图。

### 8.3 把论文思想变成小而可证伪的实验

| 想法 | 触发及具体动作 | 主要验收 |
| --- | --- | --- |
| 多尺度记忆，KDA/FRAC/Mamba-3 的廉价对照 | 检查主头残差是否随旧可见波动与时序尺度系统性变化。对每股观测收益、平方收益、对数成交量计算半衰期 2/8/32 **周**的 EMA。更新系数 `alpha=1-2**(-elapsed_weeks/half_life)`，缺失跳过更新、保留 age。优先把九维加给冻结线性/MLP 头，基线输入同样只在开发拟合 | 同一 R50/空腿候选扣费净差及解释稳定性。只有更长上下文确有净增量且受速度限制才建线性状态空间塔对照 |
| 同轨迹 checkpoint soup | 有推理预算目标时，按同 seed 的固定三轮，在 CPU 逐个检查 key、shape、dtype。平均浮点参数，非浮点 buffer 逐项要求相等，EMA 或统计 buffer 按实际加载语义校验。先比同 seed 的单轮、原分数融合与单次 soup。相互独立初始化的 seed 保留预测层融合 | strict-load、固定轮窗、主力书净差、同机端到端速度。历史表现仍满足 §2 的效率门才扩种子 |
| Q0、Q2 与 Variable-Width | 有端到端延迟要求时，用存量权重先测 Q0 的 L20 和 Q2 的 L40；Q2 的 L20 历史 −4.32。实际 profiler 证实时间塔宽度是瓶颈且存量方案不满足需求，再登记 `128/96/96/128`。在 128 残差空间的中间层，仅前 96 维做四头 attention 和 FFN，其余 32 原样旁路；用同参数的均匀子层对照，bias、归一和 dropout 同口径 | GPU 峰值、完整数据到分数耗时、精度。当前已有 SDPA 后按头门控，先读该实现和已有臂，避免重做近似方案 |
| 横截面注意力与图 | 先用 CS 归因或候选头证明上下文收益。仅在固定 Top-K 上测窄的稠密上下文。跨全池的动态 top-k 索引要单列全对评分与搬运耗时。时点行业固定邻接是必要对照 | 候选同集合、同容量、替换和费用。供应链或共持仓等时点图有数据后另开，不挪用当前分类回溯当历史图 |
| 潜预测、风险头、Flow Matching/diffusion/MeanFlow | 旧 A2 曾胜持久性而硬组合未赢。多周点预测先做 §6。只有路径尾部和相关性真的改变留仓或仓位，才做条件分位数或 2–3 周块残差基线。联合场景的首个单元是低维市场、行业及组合残差。生成模型需先胜该简单基线并改变账本决策。一步蒸馏等采样已证实成为瓶颈后再做 | 分位数覆盖、联合路径尾部、同敞口的净费后回撤和收益。仅同时输出两个周分数不触发生成式训练 |
| 滚动适应与其它预训练 | 每 13 周更新头是首个滚动对照。只有 head 更新胜固定且过前瞻，再尝试解冻末层、低秩或市场条件化。MoE、强化学习式全市场搜索、合成预训练及文本分支需独立新数据或机制证据后另开预算 | 时间外成熟标签、同交易、稳健性和登记过的尝试数。LLM 新模块或更复杂目标没有预设增益 |

## 9. 增量实验、交接和一次验收

1. 独立臂各验收后，先用已存评分、同一账本合成。每次至少回放 `B0,A,B,AB`，记录 `AB-A`、`AB-B`、`AB-A-B+B0` 的同周配对。最强父项由登记的开发指标决定。交互项用于解释，资源追加看组合相对最强父项，额外省成本或降风险的效益按 §2 决定。
2. 推荐组合顺序。首先 B0 长 + 过门的 CS 空，与旧双腿、完整 CS 空、现金三项同时比。其次同一 L20 的 H2 与固定换仓做 2×2；如只 R50 通过，就在 **同 L50** 将重排与换仓合成。若要把通过的 L40 与 CS 空放在同一组合，先开 L40/S10 新账户规格，四个父项都用同 40/10、同敞口、同短候选冲突过滤。W20 已过门时可做冻结 W20 的 L40 和新拟合 Top-80→40 重排的 2×2；对应基准要有 B0 的同重排。固定日线加入其余胜者后先做存量分数融合或冻结小头。两个模块确定互补才开一次联合训练，先 seed42 后按 §2 追加。每轮增量再套一个就停，直到有新证据。
3. 把每个阶段的代码提交或工作区补丁哈希、输入与 label hash、config、启动命令、seed/epoch、开始结束、峰值显存和失败原因归入 registry。对开发/历史、旧代理/事件账本、纯多/纯空/多空、实际券量/代理名单分别标清。曲线图注明起点与现金。对同周平均净差解释 `gross - fees - borrow` 是否能对账，若涉及不同敞口再给等敞口对照。
4. 在首个训练期间写好 `followup_eval.py` 及以下落盘形状。每个作业出分即跑同一 CPU 报告生成器，整个队列收口只合并登记过的结果。复检可由 `recheck_lock.json` 单次批量导出，然后一次在统一账本验收。只出影子目标和差分文件，真实下单沿用现有授权流程。

| 必备产物 | 验收项 |
| --- | --- |
| `registry/{manifest.json,prereg_validation_v2.md,recheck_lock.json}` | 冻结哈希、版本、成熟日期、最强父项、候选和对照明确。逐臂说明主假设与进阶条件。旧协议按原日期存档 |
| `registry/{trials.csv,smoke.json}`、`queue/state/*.json`、`logs/` | 成功、停止、依赖失败、重试、数据阻塞都有状态和证据。一个时刻最多一个 CUDA 作业 |
| `eval/<arm>/{weekly_paired.parquet,trade_events.parquet,yearly.csv,metrics.json,gate.json}` | 全验证或拼接折外与标准测试分列。写可用周数、日期、选择用途、覆盖、规模、费用网格、配对区间、回撤、尾损、换手、行业与成交。无隐式删周 |
| `eval/<arm>/folds.csv`（学习小头） | 包含每折起点、训练和选轮区间、最晚成熟标签、拟合与评分周数、参数哈希。跨折现金、残单和持仓连续 |
| `eval/incremental.csv` | 每臂与两个父项及交互项，对齐书、同周、同资金规模和费用 |
| `acceptance.md` | 分纯多、纯空、多空列出事实、假设与下一步；每个候选有 `CONTINUE / RISK_OR_EFFICIENCY / INCONCLUSIVE / STOP / SHADOW` 之一及依据。补录 `MECHANISM_ONLY / BLOCKED_DATA` 时说明缺哪个条件 |
| `shadow_protocol.md` | 晋级才写，冻结信号与交易规则，记录日期、可借量、费率、逐笔成交与拒单。明确 26/52 周复查日和停止条件 |

最后对照 prereg 核对队列状态、日期覆盖、旧产物哈希和所有新对照。报告中单列正均值但不足以判定方向的结果。历史反复查看、券源代理和当前 580 股回溯池的限制要写进验收。真正确认以冻结后的前瞻成交及相应账户对照为准。小正收益值得推进到哪一步，取决于复核成本、净增的区间与压力费用，而非统一要求每年或每臂都过 +3。

[基线](baseline_monopen.md) · [部署规则](live_deployment.md) · [重排和多期限证据](research_20261112_ranker_followups.md) · [日线和分块证据](research_20261112_daily_tokenization.md) · [组合与 RL 证据](research_20261112_rl_portfolio.md)
