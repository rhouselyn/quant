# 实盘部署手册（服务器 / cron / 上线流程）

本文件记录实盘链路的**操作性**细节。策略规则本身在
[baseline_monopen.md](baseline_monopen.md) 的"组合策略基线"一节。

## 1. 服务器

- 主机 **152.136.96.212**（腾讯云 OpenCloudOS，TZ=Asia/Beijing，python3=/usr/bin/python3，无 GPU）。
- 代码目录 `/root/quant`；历史备份 `/root/quant.bak-20261104`。
- 只做打分与推送，**不做训练**（无 GPU）；训练都在本地 4060 上串行完成。
- 同步要点：scp 传代码/权重；**上传与启动必须分两条 ssh**（同一条里 `&` 会把会话后台化，
  断开后写空文件），传完对 md5。

## 2. 周任务

cron `0 8 * * 0` → `/root/quant/scripts/weekly_qq.sh`：

```
live_pipeline.py --runs <5 个部署 run> [--prev-runs <上一版 5 个 run>]
                 --cache-dir data/live_580_fri_ak2000
                 --raw-dir data/sw_file_20/raw_ak2000
                 --refresh-cmd "python3 scripts/update_akshare_data.py --now {anchor} --skip-fin"
                 --top-n 20 --short-n 10 --pool-k 100 --short-exit-k 30
                 --state outputs/live_state_base30.json --out-dir outputs/live_base30
  ↓
qq_send_book.py <最新 live_book_*.json> --target qqbot:c2c:35C2C6F27D21E5DAD77BA679E9C2EA23
```

失败会经 openclaw 发告警；日志 `/root/quant/weekly_cron.log`。

## 3. 周任务内容（2026-11-05 起 = 组合策略基线）

5seed 融合打分（双年双版时再融合上一版）→ 多头缓冲 Top-20/Top-100 → 空头 = 顺延规则
（成交额 q4+q5 池、当季两融名单过滤、出场圈 bottom-30、顺延补足 10）→ 等权整手 sizing →
QQ 推送（每票带手数与资金摘要）。

## 4. 资金与融券

- 账户金额 / 融券保证金比例 / 现金预留存 `outputs/live_sizing.json`（默认 50 万、按个股表、
  预留 3%）；对话里对老仙女说"金额改成60万"即可（skill `quant-book` → `scripts/set_sizing.py`）。
- 融券资格名单：`scripts/fetch_margin_lists.py` 维护（SZSE 权威 + SSE 明细代理，2026-11-05
  起双市合并；此前只有深市，见台账 §21.9）。
- 融券保证金比例取 akshare 逐股表（平安证券口径、仅深市），监管下限 100%（80% 是融资比例），
  见台账 §21.10；券商自己的比例表用 `scripts/import_broker_ratios.py` 导入（优先于代理表），
  见 §21.11。
- 离线重算某周书：`scripts/size_live_book.py`（`--live-prices` 可用实时价）。

## 5. 部署一个 run 目录的前置条件

1. 含 `epoch_*.pt`（训练配置 `save_epoch_checkpoints: true`）。
2. `config.yaml` 的 `prepare_command` **不能带括号注释**（shlex 会传坏参数）。
3. `universe_path` / `market_path` 指向 `data/live_580_fri_ak2000/`（否则服务器 FileNotFoundError）。
4. 新版本上线前跑 `reselect_epochs.py --base <run>`，生成 `live_selection.json`。

## 6. 上线双版本融合（2026-11-05 待执行）

1. 本地训完 `abl2000_base_r2021` × 5 seed（`tmp_base_r2021_queue.py`），各跑 reselect。
2. 上传 10 个 run 目录（当前版 r2023 已在服务器；只需补上传 r2021 的 5 个）。
3. `weekly_qq.sh` 给 `live_pipeline.py` 追加 `--prev-runs` 5 个路径。
4. 先平行记录 4-8 周（同时保存单版本与融合版的书），确认行为正常后再只保留融合版。

## 7. 已知缺口

借券费（~8%/年）未计入回测；券商当日实际券源未验证；SSE 资格是代理口径。


腾讯云 152.136.96.212 每周日 08:00 cron 跑 `scripts/weekly_qq.sh` →
`scripts/live_pipeline.py`（5seed 融合打分 → 多头缓冲 Top-20/Top-100（§21.7）→
空头 = §21.8 顺延规则（成交额 q4+q5 池、当季两融名单过滤、出场圈 bottom-30、
顺延补足 10）→ 等权整手 sizing）→ `scripts/qq_send_book.py` 经 openclaw 推送
QQ（老仙女），每票带手数与资金摘要。账户金额/融券保证金比例/现金预留存
`outputs/live_sizing.json`（默认 50 万、按个股表、预留 3%），对话里对老仙女说"金额改成60万"
即可（skill `quant-book` 调 `scripts/set_sizing.py`）。融券资格名单由
`scripts/fetch_margin_lists.py` 维护（SZSE 权威 + SSE 明细代理，2026-11-05
起双市合并；此前只有深市，见台账 §21.9）。离线重算：`scripts/size_live_book.py`。
组合/执行层读数见 `outputs/baseline_stack/` 与台账 §21；借券费与券商实际券源
仍未建模。融券保证金比例取 akshare 逐股表（平安证券口径、仅深市），
监管下限 100%（80% 是融资比例），见台账 §21.10；券商自己的比例表用
`scripts/import_broker_ratios.py` 导入（优先于代理表），见 §21.11。
