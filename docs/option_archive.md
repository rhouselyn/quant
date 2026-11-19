# 境内期权日线归档

截止日为 2026-11-18（北京时间，含当日）。目标覆盖境内八个交易场所自首个期权交易日起的合约日线，包括已到期合约。当前公开源回补范围为中金所、上期所、郑商所、上期能源和广期所。

## 下载与存储

```powershell
python scripts/option_dataset.py --cutoff 2026-11-18
```

- 原始交易所响应和请求回执位于 `data/ashare_v2/raw/exchange/options/`，回执包含来源 URL、抓取时间及原始内容 SHA-256。
- 每个 release 发布到 `data/ashare_v2/silver/<release>/`。`option_daily` 按交易所和年份分区，`option_contracts` 为从已下载行情提取的合约目录。
- `data/ashare_v2/manifests/<release>.json` 记录数据文件哈希、预期与已取得交易日、缺口及请求失败。重跑未发布的范围会复用经校验的原始响应。已有 release 可用新后缀发布修订，例如 `--revision r3`。
- `outputs/option_archive/backfill_20261118.log` 记录本轮回补进度。数据仅保存在本机。

## 查询

```powershell
python scripts/ashare_dataset.py query --release options_exchange_20150209_20261118_cffex_czce_gfex_ine_shfe_r2 --dataset option_daily --venues CFFEX --codes HO2611-C-2450.CFFEX --start 2026-11-16 --end 2026-11-19 --columns trade_date symbol close settlement volume_contracts
```

```python
from quant_mvp.dataset_store import load_dataset

bars, manifest = load_dataset(
    "data/ashare_v2", "option_daily",
    release="options_exchange_20150209_20261118_cffex_czce_gfex_ine_shfe_r2",
    venues=["CFFEX"], start="2026-11-16", end="2026-11-19",
)
```

查询日期左闭右开。`codes` 使用带交易所后缀的标准合约代码，行情保留原交易所代码 `contract_code`。郑商所旧三位月份码按交易日展开年份，月度系列标记 `MS` 单独保留。成交量、持仓量和行权量按合约计，成交额折算为人民币元，价格、零值和空值沿用各源口径。IV 与 Delta 为各交易所原值。

合约目录的 `first_observed`、`last_observed` 是归档中的首末观测日，挂牌日、到期日和合约乘数需要额外的合约主档。归档按 `latest_history` 提供。交易日齐备表示收入了每个预期交易日的报表，逐合约完整性和历史可用时点需另行验证。

## 覆盖边界

当前发布版本为 `options_exchange_20150209_20261118_cffex_czce_gfex_ine_shfe_r2`，状态为 `partial`。共 11,009,090 条合约日线、139,108 个归档合约。

| 市场 | 首个观测日 | 已取得 / 预期交易日 | 日线行数 |
| --- | --- | ---: | ---: |
| 中金所 | 2019-12-23 | 1,673 / 1,673 | 870,884 |
| 郑商所 | 2017-04-19 | 2,328 / 2,328 | 4,972,094 |
| 上期所 | 2018-09-21 | 1,975 / 1,975 | 4,841,892 |
| 上期能源 | 2021-06-21 | 1,313 / 1,313 | 286,560 |
| 广期所 | 2022-12-23 | 70 / 944 | 37,660 |
| 上交所 | — | 0 / 2,860 | 0 |
| 深交所 | — | 0 / 1,673 | 0 |
| 大商所 | — | 0 / 2,339 | 0 |

已逐文件校验 306 个 Parquet 的哈希，并核对合约日期键唯一性、分区日期、原始报表哈希引用及 manifest 的覆盖计数。修订版补入了 2023-01-17 至 2023-01-19 的三份广期所报表。Manifest SHA-256 为 `aabcac7ae9735ed445b7440fc24a4e5e040c0731cf2d57e9c647a6e9c4f49c80`。

查询会返回已归档的记录，使用前可通过返回的 `manifest["coverage"]` 核对所需市场和日期。

沪深 ETF 期权和大商所期权尚缺可验证的全历史行情源。上交所公开挂牌接口可以查历史合约，新浪单合约日线样本存在收盘价和量字段口径问题。深交所样本未能取得历史合约日线，大商所公开导出接口本机探针返回 HTTP 412。广期所历史请求出现 HTTP 567 访问拦截。未采集分钟与逐笔。
