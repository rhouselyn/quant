# 二源断点恢复与数据实验就绪检查

2026-11-13晚，Asia/Shanghai。状态partial。父记录：[管道补齐](research_20261113_pipeline_completion_results.md)。本轮没有更换三年验证/完整历史测试规则，没有激活暂缓的17项训练。

## 实际动作

22:24二源采集发生网络接收错误10002007及连接被关闭，自动暂停；`refusal=false`，不是已确认限流。独立分页审计发现170个OK中1个缺协议页，因此只有169个通过。保存暂停、错误、分页与覆盖报告的哈希快照，按现有校验器将该条重新入队；另一条网络失败也保留待重试。没有改冻结worker、删除原响应或重下已核数据。5分钟冷却期结束后，22:30重新启动采集和独立收尾，运行信息见[恢复记录](../outputs/dataset_v2/data_readiness_v1/resume.json)。

22:34修订：恢复中再次观察到SDK空分页仍返回成功码。实际修复是在冻结worker之外新增带哈希登记的启动器，成功响应先调用现有完整分页校验；不完整响应改为`socket_error`进入已有重连/一次重试流程。拒绝状态保持原样，不绕过源限制。旧worker源码与队列未改；已核数据仍复用。真实多页成功样本、追加空页失败、拒绝保留三项断言通过。22:35已切到新启动器，并将收尾重试也指向它；切换时178只完整分页通过，另1只重取。证据：[当前启动记录](../outputs/dataset_v2/data_readiness_v1/resume_guarded.json)、[校验](../outputs/dataset_v2/data_readiness_v1/guard_checks.json)。

22:37实际复验：先前反复缺页的000711已取得6,885行完整协议证据，另核对600381、000020、002569新响应也均通过；证明保护入口已实际继续取数，不只是启动进程。22:36进度为182/3,486只，之后仍有新产物。[恢复取数证明](../outputs/dataset_v2/data_readiness_v1/recovery_verified.json)、[新的总验收快照](../outputs/dataset_v2/data_readiness_v1/acceptance_guarded/acceptance.json)。

确认没有活跃worker并满足源冷却后，后续断点应运行以下入口；不要重新用旧的无前置校验入口。当前采集和收尾已启动，无需重复运行。

```powershell
python -u outputs/dataset_v2/data_readiness_v1/guarded_secondary_worker.py
# 若重挂自动收尾，WorkerPid必须来自新启动的真实进程。
pwsh -NoProfile -File outputs/dataset_v2/archive_completion_v1/finalize_secondary_status.ps1 -WorkerPid <实际PID> -Ohlcv -RetryWorker outputs/dataset_v2/data_readiness_v1/guarded_secondary_worker.py
```

刷新了实际覆盖核对：当前二源支持131,495个额外停牌会话，官方区间另有2,718个；已补旧码5,872行。14条通过价格结构及正成交量检查的原价候选仍分来源保存；958条交易状态为1却不满足报价验收，不能填为有效成交。尚未下载和已经查无数据分开，不能把待查459,406个会话说成源不存在。沪深范围之外北交所尚有292个待证日期。这是阶段结果，终态须看最新progress与finish_status。

## 已执行D4训练分布诊断

使用归档B0实际779个训练日期（2002-07-19至2017-11-24）及2018-01-01之前成熟的285,490条标签。只读取训练标签的日期、有效性和成熟时间，未读取收益值或验证/测试成绩。对照全期均匀、5年半衰期、最近10年硬窗口，保持周权重平均为1。

| 配方 | 非零训练周 | 周权重有效样本数 | 最近5年权重占比 |
| --- | ---: | ---: | ---: |
| 全期均匀 | 779 | 779.0 | 32.35% |
| 5年半衰期 | 779 | 577.9 | 56.36% |
| 最近10年 | 507 | 507.0 | 49.70% |

断言检查非负有限、归一化、时间衰减比例、训练日期唯一和标签成熟边界均通过。有效样本数是权重集中度，不是独立市场样本量；标签计数尚未应用模型历史输入可用性mask。没有声称D4提高收益。[可复算脚本](../outputs/dataset_v2/data_readiness_v1/d4_probe.py)、[机器结果](../outputs/dataset_v2/data_readiness_v1/d4_probe.json)、[逐周权重](../outputs/dataset_v2/data_readiness_v1/d4_week_weights.csv)。

## 就绪判断与真正缺少的资料

CPU分布诊断现在可以开展且已经执行。严格全池D1、带PIT财务的D3以及原协议下正式数据收益比较、AL1/scaling仍未就绪：G_ARCHIVE/G_CAUSAL/G_LEDGER为false。不能把CPU检查或103个权益候选字段核实提升为完整全市场/真实成交验收。[逐项就绪清单](../outputs/dataset_v2/data_readiness_v1/readiness.json)。

公开名单不能提供账户特定的历史券源库存、已成交借券数量/费率/召回，也不能证明账户实际配股认购、到账与税款；这部分需要真实账户或券商档案。目前未取得完整行业/财务历史修订链、全部旧码身份及竞价参考资料，但未证明这些资料不存在，不称永久不可获得。二源网络恢复后继续取得可访问的数据。

预计完成时间须恢复稳定吞吐后重新估计；此前9–12小时未计入本次中断，不能继续当作承诺。所有证据仅本机保存，未做跨机器备份。未启动正式训练、未认证可实现收益。
