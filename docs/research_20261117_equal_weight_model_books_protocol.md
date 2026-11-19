# 旧三种子、新版 B0 与 P6 的等权仓位核对

2026-11-17，Asia/Shanghai。核对用户所问的沿用账本与可比 Top40/Bottom10。使用 B0/P6 的 [group_fox_v1 评分锁](../outputs/group_fox_v1/catalog.json) 和原 297 个已读历史周。仅回放，两腿名义预算各 50%，腿内等权。原生新版 B0/P6 为 Top20/Bottom10，另在同 297 周以 Top40/Bottom10 回放。每种用原锁定分数，按新版的开盘、流动性与借券代理。用旧冻结三种子回放的历史日期与新版取交集，再让三种分数都通过同一新版执行 Top40/Bottom10。旧打分用两个方向共用一列，新模型各腿分数以原锁定的轮次得出。

单边 10/30 bp 配年券费 8.35%/12%。回放日历复合年化以首入场至末退出天数，缺周包含在年化跨度。首要比较新版共用费用和执行对照，不将旧收益或部署回放当作同频同费用。对新版 20/10 的常规费用逐周复核原账本。旧账本与部署规则各自的年化及仓位数从既有结果核对。沿用已读历史、固定股池、代理券源。无训练或重新选轮。

CPU 设置 PYTHONPATH=src 后执行 `python -u scripts/replay_equal_weight_model_books.py outputs/equal_weight_model_books_v1/report`。
