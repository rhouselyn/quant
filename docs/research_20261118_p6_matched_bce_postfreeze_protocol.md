# 统一 P6 BCE 冻结后选步复核

2026-11-18，Asia/Shanghai。[主协议](research_20261118_p6_matched_bce_protocol.md)的追加控制。原验证最低 BCE 在两臂都选择 7/9/6 步，均早于第 10 步后冻结。为验收实际完成冻结的检查点，保留主批，另在同一训练和验证数据上重训两臂，统一将候选限定在第 11–100 步，仍按相同标签验证 BCE 最小值选步。只根据已锁验证步发现问题，截至登记尚未查看主批历史回放结果。其余所有规则及指标沿用主协议。

从根目录设置 `$env:PYTHONPATH='scripts;src'`，运行 `python -u outputs/p6_matched_bce_postfreeze_v1/run_postfreeze.py`。复用经输入哈希验证的主批 GPU 缓存，CPU 执行。验收除主协议检查之外，两组共六头的新验证步均大于 10，100 步完整训练曲线须复现主批。预计约 2–4 分钟。
