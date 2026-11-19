# 项目会话结论归档与证据导航

日期/时区：2026-11-15，Asia/Shanghai。状态：本次可访问项目会话的结论索引完成；不是聊天逐字备份，也不替代各批次冻结协议、机器证据或实时状态。返回[项目导航](INDEX.md)、[记录规则](research_recording.md)。

## 归档范围

检索本机 Codex 会话列表，以工作目录 `C:\Users\14815\Projects\quant` 识别出下列 10 个项目会话；读取各会话内容及相关已有报告。当前会话自身截至本次归档；其他会话后续新增内容须另行归档。未把其他项目或无法确认属于本项目的 ChatGPT 会话归入本记录，也未对聊天执行归档、删除等界面操作。会话标题有省略号时为列表返回的截断标题；ID 用于辨认，事实以链接的结果和产物为准。

| 会话（ID 前缀） | 归档内容与入口 |
| --- | --- |
| 制定股票数据改进执行方案（01a1d3bf） | 数据管道、切分纠正、D3/D4、缺口修补及新旧快照对照；[数据最终结果](research_20261114_gap_followup_results.md)、[修补协议](research_20261114_repaired_bridge_protocol.md) |
| 研究A股模型并编写多角度提示词（01a1d502） | 数学与金融模型、P0–P6、FOX、P6GAP、P4GROUP、BDIFF2 和损失互借；[完整研究链](research_20261115_loss_transfer_session_results.md)；新双头方案仅为提案 |
| 研究强化学习提升模型表现（01a1ce95） | RL 与组合决策研究及当时的回放与基线；[RL方案](research_20261112_rl_portfolio.md)、[后续结果](research_20261113_followup_execution_results.md)、[周频复验](research_20261113_weekly_reacceptance.md)、[排序器后续](research_20261112_ranker_followups.md) |
| 优化策略研究执行提示词（01a1d4ce） | 形成研究执行提示与约束，是研究指令而非实验完成证明；相应可追溯的研究结论见[六方向综合](research_20261113_math_synthesis_results.md)及[金融研究链](research_20261115_loss_transfer_session_results.md) |
| 研究并优化模型架构（01a1cb33） | 周频架构研究及执行设想；[架构研究](research_20261111_weekly_architecture.md)、[执行说明](research_20261111_weekly_architecture_runbook.md)；实际是否完成须看[台账](experiment_ledger.md)及具体产物 |
| 评估 MLP 选股与注意力持仓（01a1c9b3） | 重排、注意力和多空分组实验；[MLP](score_mlp_results.md)、[三种子](directional_three_seed_results.md)、[注意力](attention_full_validation_results.md)、[双阶段](two_stage_embedding_results.md)、[腿与股票池](leg_pool_ablation_results.md) |
| 既然580股票是目前最好的数据集（可以反驳），那是不是一直用580随机选取股票（限制重复率）直到几乎全覆盖，然后全局z…（01a1c3ca） | 580 股、股票池归一化、风险表示和时机研究；[风险表示交接](experiment_handoff_580_risk_embedding.md)、[周频风险与择时交接](experiment_handoff_weekly_risk_and_timing.md)。固定 580 是既有实验边界，不是证明其最优 |
| 给你充足的时间，你可以上网查资料，看开源仓库等进行资料检索 然后基于知识和判断选择：对当前策略做改进（不管是针对多头，…（01a1c669） | 多方向研究与历史执行规划；[方案](research_20261111_execution_plan.md)。候选与已完成结果分开，旧时间划分不覆盖后续批次协议 |
| 研究并设计策略改进方案（01a1c0e0） | 较早策略研究、基线与划分审计；[历史方案](research_20261109_plan.md)、[台账](experiment_ledger.md)。早期测试暴露须在后续研究中保留 |
| 讲解下这个项目和截图中这些项目的区别（01a1ac30） | 项目定位、研究与部署差异；[策略基线](baseline_monopen.md)、[部署说明](live_deployment.md)。实际线上状态须另查 |

## 跨会话保留的判断

1. **选择和测试边界**：当前数据研究各模型、种子、腿分别在 2018–2020 三年成熟验证集按扣费 Sharpe 选轮，按登记的相邻 win3 融合；2021 起全部成熟测试周已多次暴露，只能称历史复检。旧方案的固定轮次、不同切分与样本量保留为历史设置，不是后续协议的默认。模型优先依验证及预先登记的比较，不能用已读测试筛赢家。[切分纠正](research_20261113_split_correction.md)、[方法审计](research_20261113_model_selection_audit.md)。
2. **模型不是只看单次收益**：CS、ADN、P64、D3 和容量对照需比较相同口径、种子、费用及多空腿；单种子与三种子融合不构成同预算稳定性对照。[P64三种子复核](research_20261113_p64_recheck_results.md)的配对区间包含零，不支持宣称稳定优势；跨股信息与新增容量须有匹配的 SELF/零通道对照。历史固定 4–6 轮结果不能直接替代逐腿验证选轮结果。
3. **改进有负结果也要保留**：已有 MLP 重排、全股票注意力、部分缓冲与概率模型未稳定胜过各自基线，约束的是对应数据、实现及预算，不证明整个方向无效。FOX 验证改善、P6GAP/P4GROUP 未提升其对照多头，BDIFF2 的验证空头明显退化；不能用历史测试中漂亮的单项晋级。[模型研究链及数值](research_20261115_loss_transfer_session_results.md)。
4. **数学定义与效益分别验收**：结构性分解可成立而估计未校准；排序或分类损失可能改变均值目标。行业暴露和市场 beta 不等于 alpha，额外成本与可交易性会改变组合收益。P6/B0 双头迁移尚未实现、未入队，详见[损失互借](research_20261115_loss_transfer_session_results.md)。RL 和更复杂策略也应与简单、同成本约束对照，而不是因结构复杂就认定更好。
5. **数据和执行代理有限**：股票身份、退市与停复牌、跨源报价、复权版本和信息可用时间必须分别核对。现有 580 固定幸存股票池、回溯来源和缺少券源历史，使严格 PIT、无幸存偏差及真实空头成交都未获证明；不读取财务字段的价量试验可在明确限制下独立开展。[数据验收](research_20261114_gap_resolution_results.md)、[修补跟进](research_20261114_gap_followup_results.md)。

## 最新缺口与修补对照（2026-11-15）

此前记录的沪深 67 条无状态和北交所 189 条缺口已全部分类：前者 42 条有发行人或交易所证据、8 条有独立行情商历史日历证据、9 条在已核实退市之后；后者 184 条有发行人披露、5 条有行情商转述的交易所日历证据。不同来源证据等级不同；分类解释不等于填补行情或证明当时可知。[验收及查询方法](research_20261114_gap_followup_results.md)记录可查询覆盖层与身份修订。

修补进入本批 580 股快照的只有两条 2000-01-26 价格，后复权为近似。9 组八轮训练（B0、CS、ADN 各 seed42；D3、D3_ZERO 各 seed42/43/44）及统一选轮回放完成；44 组旧新配对的年化差与逐周收益差均为零，只能说明这两条数据在此批设置下没有改变收益。D3 相对零通道三种子融合：验证多/空扣费年化 18.69%/−0.62% 对 14.21%/−0.36%；已读历史 29.25%/4.37% 对 27.88%/−7.26%。验证多头周净差区间跨零，增益集中于一个种子，空头验证不稳，不构成已证实的稳定 alpha。单 seed42 基线数据及具体成本、测试边界、beta 和逐种子结果见[最终报告](research_20261114_gap_followup_results.md)。

注意区分旧数据的 12 轮配对仅有 seed42/43，和修补快照的三种子 8 轮协议；不能拼成 12 轮三种子结果。曾触及预算边界不证明已收敛，也不证明续训必然改善。

## 状态、证据与后续

2026-11-15 约 09:07（北京时间）核查：修补对照、P6GAP/FOXCONST、P4GROUP/BDIFF2 的相关队列和最终报告完成，相关训练/汇总进程未见；这是带时点的局部检查，不代表其他任务没有运行。后续接手先查各自 `queue/state`、日志、进程及产物，再决定是否登记新实验；已有历史测试不再作为独立确认集。

关键本机证据 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `outputs/dataset_v2/archive_completion_v1/gap_resolution_acceptance_20261114.json` | `b26b6123023451bfd8efe1cecb345654dcb1c31f8274a1e24d37d0ff6caad99aa` |
| `outputs/dataset_v2/repaired_bridge_v1/comparison_report_all/old_vs_repaired.csv` | `5c4e6184185309f5a04ce610462a5a4a02e4fd33aae49e75f6d7924ff7d50bb2` |
| `outputs/dataset_v2/repaired_bridge_v1/report_all/metrics.csv` | `494134c1a66be89cf5c243c43ca1b8e2df732d08723c73bc29eeeaaedb9eb9a0` |

原始数据、权重、队列和机器输出在被 Git 忽略的 `data/`、`outputs/`，目前仅本机、未确认异地备份；哈希不能代替备份。文档进入工作区不等于已提交或推送。本次仅汇总已完成研究及已登记提案，未重跑训练、回测或重新认证所有历史权重。
