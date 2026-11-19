# 原冻结 BCE 迁移到新版 B0/P6 的 Top20/Bottom10

2026-11-17，Asia/Shanghai。沿用[新版等权对照](research_20261117_equal_weight_model_books_results.md)的 B0、P6 评分锁和同一开盘执行、流动性及借券代理。历史严格取旧原冻结 BCE 与新版账本交集 283 周。按日期读取同一份三种子平均 logit，不重新拟合，不改变两模型各自 Top20/Bottom10 名单或腿内等权规则。固定 50/50 多空、旧宽档 logit>0 为 70/30 否则 40/60、全多全空 logit>0 为 100/0 否则 0/100。

10/30 bp 单边交易费分别配年券费 8.35%/12%。按各自持仓重新扣费，缺股或模拟未成交份额留现金。报告净日历复合年化（首入场至末退出，跳过的周计入期间）、52 周年化夏普、周度净值最大回撤及净周均的 8 周块配对。验证各臂使用相同名单、固定 50/50 按原入口复现。测试历史已读，主要检验旧方向对新篮子的迁移。

CPU 设置 PYTHONPATH=src 后运行 `python -u scripts/replay_b0_p6_frozen_bce_transfer.py outputs/b0_p6_frozen_bce_transfer_v1/report`。
