# 2026-11-14 缺口补证与修补快照执行记录

状态：数据状态补证与验收完成；修补快照模型比较运行中。研究登记见 [修补快照实验协议](research_20261114_repaired_bridge_protocol.md)，后者在批次执行期间冻结。

## 缺口口径和可查询版本

- 本次起点沪深 67 条：42 条停牌有发行人或交易所转述证据、8 条有独立行情商历史日历停牌证据、9 条发生于已核实退市日期之后；全部分类，后 8 条的证据等级低于发行人披露。上港集箱旧身份 `SH:600018_PREMERGER` 与换股合并后同代码证券区分。
- 本次起点北交所 189 条：184 条有发行人历史停复牌披露，5 条来自行情商转述的北交所停复牌日历。此前已解释的 103 条不重复计算；根目录的验收脚本对 139 个后续日期逐项做集合相减、交叉检查和查询断言。各公告有原始文件哈希及各自分区 release。
- 一站式状态层 `gap_status_overlay_7d43f26bebd820c1` 包含本批和此前披露的停复牌区间，保留来源 release、证据级别及原始哈希。它仅包含 `security_lifecycle` 与 `security_alias`，不承诺包含行情或全部证券主数据。纠正后的证券主数据在 `mixed_lifecycle_42ce0882a90376dc`；前身报价保留在独立 `qq_600018_premerger_v2_d41241b5c57b86ad`。不能把停牌日填成价格、也不能将不同 release 当成自动完成了全市场 union。

```powershell
python scripts/ashare_dataset.py query --release gap_status_overlay_7d43f26bebd820c1 --dataset security_lifecycle --codes 600018.SH
python scripts/ashare_dataset.py query --release gap_status_overlay_7d43f26bebd820c1 --dataset security_lifecycle --codes 920680.BJ
python outputs/dataset_v2/archive_completion_v1/accept_remaining_gaps.py
```

验收明细 `outputs/dataset_v2/archive_completion_v1/gap_resolution_acceptance_20261114.json`；各期 `bj_issuer_halts_v*.json` 和 `sh600018_*halts.json` 记录公告、区间和哈希；统一覆盖层元数据 `gap_status_overlay.json`。状态被解释不代表缺失报价被补齐，亦不代表公告在被回测时点已经可知。所有补证及旧供应商复权源均没有完整历史版本，不能声称严格 point-in-time；明确禁止以这些回溯发现的状态作为历史推理筛股依据。

## 修补训练快照和实验

580 股票池与 4,238 行补录交集 30 行，其中 28 行早于快照起点，仅 000729.SZ、000962.SZ 的 2000-01-26 两行进入本快照。价格采用 BaoStock 原价；缺失后复权价以 2000-01-25 既有后复权/原始收盘比折算，只作为 Q1 近似。供应商历史版本与当天可能的公司行为无法独立证明，底层后复权系列也无历史版本，因此实验不满足严格 PIT。旧新快照标签一致；保存各自快照、审计和哈希，不修改原数据。

批次 `outputs/dataset_v2/repaired_bridge_v1` 包含 validation/history 双快照、训练队列、锁、日志、协议和结果。153 周验证 2018–2020；每模型每种子多空分别按本腿扣费 Sharpe 选最佳轮，取对应 win3，完整测试 2021 起 297 个成熟周。B0、CS、ADN 种子 42；D3 和零通道种子 42/43/44，固定 8 轮以与旧 8 轮同配置配对；是否续训由独立预注册分析确定，不能因测试收益挑轮或扩队列。费用为单边 10bp、借券年化 8.35%，借券可用性、开盘成交、受阻卖出仅为代理；固定幸存股池及已暴露测试集限制推断。

B0 seed42 单独预报告 `report_b0_early` 和 `comparison_b0_early/old_vs_repaired.csv`：新旧结果逐周完全相同；验证多/空 CAGR 15.47% / 2.42%，完整测试 26.39% / 1.61%，均为扣费代理，无法代表 D3 或其它种子。自动工作进程依次生成 `report_seed42`、`report_all`；独立配对监视器在两者完成后生成 `comparison_report_seed42`、`comparison_report_all`，输出同模型同种子的旧新净 CAGR 差及周收益配对区间，登记旧输入哈希。自身比较脚本通过 44 条零差断言。B0 空头选中边界第 8 轮；本批严格遵守 8 轮登记，不能据此假定其收敛。

检查活跃状态时看进程、`queue/state/*.json`、`logs` 和最终报告。Windows 防睡眠辅助进程仅在工作进程运行期间有效。剩余模型尚未完成时不得把既有测试数称为本批修补效果；若任务或监视器失败，应保留错误并单独记录修复与重试。

CS seed42 独立预报告 `report_cs_early` 和 `comparison_cs_early/old_vs_repaired.csv`：验证及测试逐周收益新旧完全一致；验证多/空 CAGR 13.43% / 11.38%，完整测试 17.77% / 12.51%。预测值本身存在最大约 8.2e-6 的数值差，不能据此断言所有模型不受修补影响。B0 与 CS 各腿独立选轮，均碰到 8 轮预算边界；不将测试结果作为选择或扩训依据。

查询指定时段内仍然有效的停牌区间，请用 `python scripts/query_gap_status.py --codes 430090.BJ --start 2023-08-04 --end 2023-08-08`；区间左闭右开。通用 `ashare_dataset.py query --dataset security_lifecycle --start ...` 按**事件开始日期**过滤，不会返回在窗口开始前已开始的持续停牌，因此不能用空结果推断该窗口可交易。状态覆盖层不等于完整停牌表。

ADN seed42 独立预报告 `report_adn_early` 和 `comparison_adn_early/old_vs_repaired.csv`：新旧逐周收益一致；验证多/空 CAGR 16.57% / 6.67%，完整测试 24.61% / 14.62%。上述三个基线都仅为 seed42；这些历史测试已经暴露，不能据此选赢家。

D3_ZERO seed42 独立预报告 `report_zero_early` 和 `comparison_zero_early/old_vs_repaired.csv`：新旧逐周收益一致；验证多/空 CAGR 17.57% / −4.80%，完整测试 26.62% / −15.12%。这是本批固定 8 轮对照；不可和另行登记的 12 轮结果混用。

执行调度调整：`report_seed42` 优先级由 15 改为 90，排到剩余 GPU 种子后，避免 CPU 汇总使 GPU 等待；独立 `watch_pair42.py` 在 D3 与 D3_ZERO 的 seed42 都完成后，校验冻结输入并并行执行两模型同口径回放，产物为 `report_pair42_parallel` 与 `comparison_pair42_parallel`。不改变模型、种子、选轮或成本协议。最终 `report_all` 和新旧比较继续照常执行。

额外验证：`python -m pytest tests/test_dataset_contract.py tests/test_dataset_evaluation.py tests/test_dataset_exploration_report.py tests/test_ashare_dataset_store.py -q`，27 passed；这些测试证明相关代码边界，不证明供应商历史版本存在。

D3 vs D3_ZERO seed42 配对已完成，产物 `report_pair42_parallel` 和 `comparison_pair42_parallel`。均为 8 轮、各腿验证选轮加 win3：验证多头 18.08% vs 17.57%（+0.52 个百分点），空头 −8.71% vs −4.80%（−3.91 个百分点）；测试多头 29.29% vs 26.62%（+2.67 个百分点），空头 −3.21% vs −15.12%（+11.91 个百分点）。验证周均净差 L8 区间多头 [−13.23,10.87] bp、空头 [−33.75,15.24] bp，均跨零；一个种子不足以确认增益。两臂修补前后扣费收益逐周完全相同，仍需种子 43、44。

进一步调度：全量 `report_all` 优先于重复的 `report_seed42`，配对监视器也先处理全量报告，再处理 seed42 汇总；首组并行配对与三组基线的单模型报告均已完成。无实验设置改变。

种子 43 完成后将由 `watch_pair43.py` 额外并行产出 `report_pair43_parallel` 和对应旧新比较，不阻塞种子 44；它不改变全量三种子汇总的选轮或评估。

全量比较产物完成后，查看 `outputs/dataset_v2/repaired_bridge_v1/final_analysis.md`；机器可读数据分别在 `report_all/metrics.csv`、`report_all/paired.csv` 和 `comparison_report_all/old_vs_repaired.csv`。若文件不存在，先查队列状态以及 `logs/final_analysis2.stderr.log`、`logs/compare_watch2.stderr.log`，不能将预期产物当作已完成。

为区分市场暴露，已对 seed42 的 D3 与零通道做事后单因子描述性诊断 `attribution_pair42/market_attribution.csv`，使用已冻结的沪深 300 价格指数、持仓相同开盘区间及零现金利率；验证多头市场 beta 为 0.914 vs 0.947，空头 −1.004 vs −1.008。仅控制一个市场指数，不足以证明可交易 alpha 或因果效果，指数历史版本亦非 PIT。全模型完成后由 `watch_attribution.py` 生成 `attribution_all`，不参与选轮或选模型。

2026-11-15 约 01:00 出现一次数分钟暂停；原因未能确认。训练恢复后新增 `keep_awake.py` 在单个线程周期性请求系统保持唤醒，主工作进程退出后自动解除；显式睡眠、休眠或关机仍可能中断。

D3 vs D3_ZERO seed43 配对 `report_pair43_parallel` 已完成：验证多头 17.81% vs 9.11%（+8.69 个百分点）、空头 6.07% vs 2.50%（+3.56）；测试多头 27.81% vs 26.35%（+1.45）、空头 4.84% vs 0.65%（+4.19）。验证周净差 L8 区间多头 [4.21,25.37] bp、空头 [−20.13,32.20] bp。两臂新旧数据结果逐周完全一致。`attribution_pair43` 是沪深 300 单因子事后描述，并非完整风险归因。注意多次实验和同一市场历史使这些区间不能视为独立复现。

D3 vs D3_ZERO seed44 配对 `report_pair44_parallel`：验证多头 14.60% vs 14.27%（+0.33 个百分点）、空头 −0.24% vs −0.66%（+0.42）；测试多头 28.28% vs 32.06%（−3.78）、空头 −4.92% vs −7.29%（+2.37）。验证周净差 L8 区间多头 [−6.92,9.40] bp、空头 [−10.96,10.62] bp，均跨零。两臂新旧数据结果逐周完全一致；三个种子同样如此。跨种子融合仍等待 `report_all`。

## 最终结果（2026-11-15）

全量 9 组训练、三年验证多空独立选轮和完整测试回放完成：`report_all` 含 11 个组合（包括两个三种子融合），153 个验证周、297 个测试周。`comparison_report_all/old_vs_repaired.csv` 共 44 项同模型、同种子/融合、多空、阶段配对；年化差和逐周净收益差全部为 0。结论仅限本批 580 股池中的两条补价及当前设置，不能说明全市场补证无用。

三种子融合 D3 对 D3_ZERO：验证多头 18.69% vs 14.21%，空头 −0.62% vs −0.36%；完整测试多头 29.25% vs 27.88%，空头 4.37% vs −7.26%。验证多头平均周净差 7.27 bp，L8 区间 [−0.59,15.05] bp；验证空头差不稳，L8 区间 [−21.45,19.31] bp。测试期仅作已暴露历史复检，不可用来筛选赢家。三个种子验证多头分别高 0.52、8.69、0.33 个百分点，效应集中在 43；测试多头分别 +2.67、+1.45、−3.78。验证空头分别 −3.91、+3.56、+0.42。多头有探索性正向迹象，尚不足证明稳定增益；空头验证没有稳定改善。

`attribution_all/market_attribution.csv` 为事后沪深 300 单因子诊断。融合验证多头 D3 beta 0.932、零通道 0.949；空头 −0.967 vs −0.957。仅控制市场价格指数，无法排除行业、风格、风险暴露或执行代理影响，也不等于可交易 alpha。

基线 seed42：验证 B0 多/空 15.47% / 2.42%，CS 13.43% / 11.38%，ADN 16.57% / 6.67%；完整测试分别 26.39% / 1.61%、17.77% / 12.51%、24.61% / 14.62%。基线只有一个种子，与三种子融合不构成同预算稳定性比较。全部为扣费代理，不能视为实盘收益。来源和审计约束见本记录上文。
