# Akshare 数据管道（替代 Yahoo）

2026-11-03 起部署。整个链路单一数据源（akshare/新浪行情 + 东财数据中心基本面），
价格统一**后复权（hfq，因子锚定上市日、只增不改）**，量额为未复权原始值。

## 为什么是后复权 + 增量追加是安全的

- hfq 因子锚定在上市日，历史数据**永不重写**（分红除权只追加新因子），因此
  周频增量拉取后直接拼接不会破坏复权正确性。
- 每次增量拉取保留 10 天重叠窗口，重叠日收盘价与库内值**逐日精确比对**，
  不一致（上游因子修订）的股票自动全量重拉——复权正确性有守门员。
- 2026-11-01 与旧 yahoo 前复权缓存对照：周收益 corr 0.9944、target corr 0.9981。

## 周频更新（周五收盘后 / 周末跑）

```bash
# 1. 数据刷新：增量行情 + 缺失季度财务 + 导出 raw + 重建周频缓存 + fin 特征
python scripts/update_akshare_data.py --build-cache

# 2. 5 seed 融合书：下周持仓（周一开盘执行）
python scripts/live_pipeline.py \
    --runs outputs/abl2000_fin2_r2023 outputs/abl2000_fin2_r2023_s43 \
           outputs/abl2000_fin2_r2023_s44 outputs/abl2000_fin2_r2023_s45 \
           outputs/abl2000_fin2_r2023_s46 \
    --refresh-cmd "python scripts/update_akshare_data.py --now {anchor}" \
    --raw-dir data/sw_file_20/raw_ak2000 \
    --cache-dir data/live_580_fri_ak2000 \
    --out-dir outputs/live_ak2000
```

live_pipeline 自动完成：完整性门控（≥95% 个股有 anchor 周五 bar）→ 按
prepare_command 重建缓存（含 fin 特征后处理钩子）→ 5 seed × best±1 epoch
z-score 融合打分 → 缓冲规则（多头 Top-20/池 150，空头 Bottom-10/池 100）→
产出 `live_book_<anchor>.csv/json` + `outputs/live_state.json`。

只要下周榜单：`python scripts/score_next_week.py --run outputs/abl2000_fin2_r2023 --run ...`（重复 5 个 run）。

## 文件地图

| 脚本 | 作用 |
|------|------|
| `scripts/update_akshare_data.py` | 增量行情 + 季度财务 + raw 导出（+ 可选重建缓存） |
| `scripts/build_raw_ak2000.py` | hfq 日线 → 逐股 parquet（`--cutoff` 可参数化） |
| `scripts/prepare_hierarchy_580_mon.py` | 周频缓存构建（path6，30 特征） |
| `scripts/add_fin_features.py` | PIT 财务特征（法定披露窗；训练窗中位数填充） |
| `scripts/live_pipeline.py` | 周频打分 + 缓冲持仓簿（`--refresh-cmd` 可插拔数据源） |
| `scripts/score_next_week.py` | 只出下周多空榜单（N run 融合） |

数据落盘：

- `data/akshare_580_daily/daily_{hfq,raw}/daily.parquet` — 全量日线（增量追加）
- `data/akshare_580_daily/fin_indicators/yjbb_raw.parquet` — 季度财务（东财）
- `data/sw_file_20/raw_ak2000/` — 逐股 parquet（prepare 脚本输入）
- `data/sw_580_observed_features_2000_fri_path30/` — 周频特征缓存（2000-01-07 起）

## 注意事项

- py_mini_racer（akshare 的 js 引擎）并发初始化会 C 级崩溃，`update_akshare_data.py`
  已在所有 akshare 调用上加全局锁，勿移除。
- 东财 `最新公告日期` 字段不可信（数据刷新时间戳），PIT 一律用法定披露窗口
  （Q1+30d / 半年报+62d / Q3+31d / 年报+120d）。
- 新浪源偶发限流：脚本内建 3 次退避重试 + 末尾串行补拉；失败的股票保留旧数据并在日志中列出。
