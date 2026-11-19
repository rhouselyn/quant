# A 股归档补录与查询验收

> 22:35恢复入口修订：SDK存在空分页误报成功，已新增前置分页校验启动器并实际恢复采集。后续恢复优先按[网络恢复与就绪记录](research_20261113_data_readiness_results.md)执行；下文旧命令保留历史，不能绕过新校验。全量仍未完成。

2026-11-13，Asia/Shanghai。状态 `INCOMPLETE`。父协议见 [数据执行方案](research_20261113_dataset_execution_runbook.md) 与 [管道补齐协议](research_20261113_pipeline_completion_protocol.md)。行情截至 2026-11-12。此轮未读取模型表现或改变训练和测试协议。

## 已发布数据

`sina_lifecycle_58e963c0cd0187a5` 补录 `security_master`、`security_alias`、`security_lifecycle`。日线引用父版本 `sina_6ca61729c2fcbc21` 并校验父 manifest SHA-256，保留 18,316,366 行原价。后续版本 `sina_szalias_45d7e086a044229a` 新增同一数据源的旧码行情，原价增至 18,322,238 行。两个版本均为 `Q0 / latest_history`，对严格 PIT 请求返回错误。 [生命周期发布清单](../data/ashare_v2/manifests/sina_lifecycle_58e963c0cd0187a5.json) · [历史旧码发布清单](../data/ashare_v2/manifests/sina_szalias_45d7e086a044229a.json) · [机器验收](../outputs/dataset_v2/archive_completion_v1/summary.json)

| 证据 | 核对结果 |
| --- | --- |
| 沪市终止日期 | 148 条。交易所页面脚本将同一 `COMPANY_STATUS=3` 查询的 `DELIST_DATE` 用作终止上市日期。6 个近期摘牌公告逐项匹配。 |
| 沪市代码身份 | 600190、600555、600614、600625、600680、600695 的源表同时有 A 股和 B 股记录。采用 `STOCK_TYPE=1` 的 A 股日期，修正其中三只的 A 股上市日。原 A/B 行随父档保留。 |
| 北交所代码区间 | 248 个新旧码成对验证，244 对为本轮新增覆盖。旧码和新码各有 248 条有效期记录，6 个试点于 2025-05-06 换码，其余于 2025-10-09 换码。官方表中早于北交所成立的精选层日期裁到北交所开市日。 |
| 深市板块 | 002001–004999 的已归档深市 A 股在 2021-04-06 前按 `sme` 查询，合并后按 `main` 查询。板块和日期一起过滤。 |

沪市原始响应保留 `STOCK_TYPE` 和 `STATE_CODE_STOCK`。北交所官网直连此前返回 HTTP 403，对照表通过官方页面的索引片段取得。提取后核对序号连续、两列唯一性、六只试点代码及全部 248 个现行代码的名录和上市日期。其他代码的区间仍是现有名录推定。 [上交所退市列表](https://www.sse.com.cn/assortment/stock/list/delisting/) · [页面脚本](https://www.sse.com.cn/xhtml/home/2021public/querySearch/search_stocksDepositoryReceipts_2021.js) · [北交所新旧代码表](https://www.bse.cn/service/code_mapping.html) · [试点名单](https://www.bse.cn/important_news/200025487.html) · [试点上线](https://www.bse.cn/important_news/200025603.html) · [存量切换](https://www.bse.cn/important_news/200026735.html) · [深交所板块合并](https://www.szse.cn/disclosure/notice/general/t20210331_585343.html)

| 北交所代码 | 申报转板停牌起点 | 北交所终止日 | 后续上市 | 北交所确认无交易会话 |
| --- | --- | --- | --- | ---: |
| 832317.BJ | 2021-10-21 | 2022-04-26 | 688287.SH，2022-05-25 | 108 |
| 833874.BJ | 2021-11-08 | 2022-07-18 | 301192.SZ，2022-08-11 | 163 |
| 833994.BJ | 2021-11-04 | 2022-07-25 | 301321.SZ，2022-08-18 | 168 |

三只从北交所开市至终止前已停牌。原始供应商序列的末日与申报前最后交易日一致，状态重分类为 `VERIFIED_NO_TRADING`。新生命周期表提供独立的后续证券 ID；跨市场转板的成交记录按交易所分开。 [观典转板报告](https://static.sse.com.cn/stock/disclosure/announcement/c/202203/832317_20220331_FXHL.pdf) · [观典北交所终止](https://www.bse.cn/disclosure/2022/2022-04-25/1650875588_282390.pdf) · [观典科创板上市](https://www.sse.com.cn/disclosure/announcement/listing/c/c_20220523_5702466.shtml) · [泰祥转板报告](https://static.cninfo.com.cn/finalpage/2022-08-10/1214261921.PDF) · [翰博转板报告](https://disc.static.szse.cn/download/disc/disk03/finalpage/2022-08-17/51fce967-293a-490b-916b-60c40783b28f.PDF)

## 交易会话覆盖

按修订后的上市和终止日期与既有新浪日历重新逐证券计数。半开区间从上市日到终止日或冻结日后一天。名录仍为 5,921 只，其中 5,918 只含结构通过的行情，三只转板已解释。旧报告的 898,916 个粗略缺口受终止日期空缺影响。

| 市场 | 原价行 | 名录内未见行情的预期会话 | 有公告证据的北交所无交易会话 |
| --- | ---: | ---: | ---: |
| 沪市 | 8,205,236 | 225,555 | 0 |
| 深市 | 9,842,251 | 380,092 | 0 |
| 北交所 | 268,879 | 292 | 439 |
| 合计 | 18,316,366 | 605,939 | 439 |

605,939 个预期会话中，首条行情之前 11,870 个，内部 579,601 个，末条之后 14,468 个。生命周期之外为 0 行。000007.SZ 的 1993-01-03 有一条行情处在代理日历之外，保留该行并等待交易所日期证据。 [逐证券覆盖](../outputs/dataset_v2/archive_completion_v1/coverage.parquet)

四只沪深股票的交易所或发行人公告确认 2,718 个会话无交易。BaoStock 对四个固定短窗的 41 个缺日返回 `tradestatus=0`，其中 40 个已被公告区间覆盖，另一个是 000156.SZ 的 2001-06-01。腾讯返回行情窗口，但没有状态列。公告和 BaoStock 作为两类证据分别留档，未用供应商停牌平盘记录造价。此小样本仍余 603,220 个预期会话待解释。 [逐区间公告及计数](../outputs/dataset_v2/archive_completion_v1/status_evidence.json) · [二源探针请求与响应](../outputs/dataset_v2/archive_completion_v1/secondary_probe.json)

### 深赤湾 A 换码前行情补录

深赤湾 A 于 2018-12-26 从 `000022.SZ` 切换为 `001872.SZ`，上市主体与 A 股权利延续。新浪现行代码只返回换码后的行情，原代码接口可取 1993-05-05 至 2018-12-20 的 5,872 条记录。新版本保存原始代码并使用稳定证券 ID `SZ:001872`，5,872 条均通过 OHLCV 检查，未发现同证券同日重复。 [换码与停牌公告](https://static.cninfo.com.cn/finalpage/2018-12-21/1205681867.PDF) · [发行报告](https://static.cninfo.com.cn/finalpage/2018-12-24/1205685040.PDF) · [补录验收](../outputs/dataset_v2/archive_completion_v1/sz_alias_recovery.json)

换码窗口的旧码 9 个和新码 6 个重叠日，两源四价和成交量逐项相同，成交额有差异。逐日状态交叉核对表明 5,871 条补录对应 BaoStock `tradestatus=1`，1 条无状态，另外 14 个 BaoStock 标交易会话仍无新浪记录。363 个供应商标停牌会话与本次补录无冲突。另一固定窗口 `000042.SZ` 的 1995-02-06 至 02-10 在 BaoStock 有原价而新浪无记录，后两个共同交易日两源四价、量额相同。供应商独有报价留作独立证据。补录后原价仍缺 600,067 个预期会话，扣除官方无交易区间后待解释 597,349 个。

## 查询与复现

21 项真实查询通过，覆盖普通和试点换码、同一证券跨换码历史、有效码窗口、合并前后板块、沪市退市、北交所无交易区间、证券类别、580 桥接原价和后复权、财务公告和报告期、行业供应商更新日，以及 PIT 拒绝。`--code-mode effective` 接受具备官方映射证据的 496 条北交所代码区间，旧码补录版另含 000022 和 001872 两个区间。其余已知码可按 `identity` 查询。结果中的 `symbol` 保留行情归档代码，换码时点查询代码见别名表。深市 `sme` 是本批次日线输出口径。 [查询验收](../outputs/dataset_v2/archive_completion_v1/query_acceptance.json)

```powershell
python scripts/ashare_dataset.py query --release sina_lifecycle_58e963c0cd0187a5 --codes 002001.SZ --start 2021-04-02 --end 2021-04-07 --boards sme --security-types A_share
python scripts/ashare_dataset.py query --release sina_lifecycle_58e963c0cd0187a5 --codes 920819.BJ --code-mode effective --start 2025-04-29 --end 2025-05-08
python scripts/ashare_dataset.py query --release sina_lifecycle_58e963c0cd0187a5 --dataset security_lifecycle --venues BJ --start 2021-11-15 --end 2021-11-16
python scripts/ashare_dataset.py query --release sina_szalias_45d7e086a044229a --codes 000022.SZ --start 2018-12-20 --end 2018-12-28
python -m pytest tests/test_ashare_dataset_store.py -q
python outputs/dataset_v2/archive_completion_v1/verify.py
```

6 项归档单测通过。构建脚本 `outputs/dataset_v2/archive_completion_v1/build.py` 和本地源文件的哈希见机器验收。正式发布 manifest SHA-256 为 `1ecbc1ae9eaf5927a909a2818bb9ab880d8db1df1fbfb02aa9aacf0bb805a024`。覆盖 Parquet SHA-256 为 `39255a7b0ffc99725f8868f02b8d79a0bc6d732da8b4974ab083f57b35992c18`。父版与更早阶段版本保留。`data/` 和 `outputs/` 证据仅本机保存，未做跨机器备份。

## 后续状态核验与尚缺证据

已启动单连接 BaoStock 状态队列，冻结 3,486 只沪深证券和 605,647 个待证日期。每证券只请求 `date,code,tradestatus`，覆盖其最早至最晚待证日，按实际日键匹配。请求、分页和登录共用一秒间隔，单次套接字超时 12 秒。超时或连接失败重连重试一次，持续失败或拒绝暂停。原始协议响应、解析状态、请求参数、时间和哈希按请求写入，完成时生成按日证据和逐证券状态报告。 [官方访问规则](https://www.baostock.com/blacklist) · [冻结队列清单](../outputs/dataset_v2/archive_completion_v1/secondary_status/queue_manifest.json) · [运行进度](../outputs/dataset_v2/archive_completion_v1/secondary_status/progress.json)

21:20:57 的分层首 20 只证券全部返回，耗时 48.36 秒，对应 8,940 个待证会话。BaoStock 标记 3,019 个停牌、5,920 个交易，另有 1 个缺失。5,920 个中有 5,885 个属于换码的 `001872.SZ`，已按上节复核。状态采集版最后保留 110 只证券的完整分页响应，21:28:05 安全切换至扩展版。扩展版沿用原队列与已经验证的结果，在余下请求额外保留不复权 OHLCV、昨收与成交额原值。仍使用供应商各自来源和时间，供后续独立报价核验。 [扩展队列清单](../outputs/dataset_v2/archive_completion_v1/secondary_status_ohlcv/queue_manifest.json)

扩展版每 20 只证券或每约 60 秒更新进度。21:31:58 的早期实测完成 124 只，近 14 个扩展请求每分钟 3.77 只。按剩余区间约 6,491 页和每页耗时四分位数估计尚需约 8.5–10.8 小时。长历史、实际可返回页数、网络和重试会影响预计时间。 [当前进度](../outputs/dataset_v2/archive_completion_v1/secondary_status_ohlcv/progress.json) · [页数与吞吐估计](../outputs/dataset_v2/archive_completion_v1/secondary_status_ohlcv/eta_model.json)

采集进程退出后，后台收尾会校验页号、原始响应、回显参数及终页完整性。出现分页异常时，仅重开对应证券并恢复一次。随后重算扣除新浪旧码补录和公告区间后的覆盖，单独保存 BaoStock 疑似缺价。21:35 的部分审计在 130 条具有二源原价的 `tradestatus=1` 候选中发现 121 条平盘且成交量为零，只有 9 条满足 OHLCV 结构并有正成交量。尚未校验价格的继承状态单列，候选原价保留供应商来源。 [部分覆盖与报价审计](../outputs/dataset_v2/archive_completion_v1/secondary_status_ohlcv/post_recovery_audit.json)

终态由 `secondary_status_ohlcv/finish_status.json` 和 `result.json` 共同记录，供应商无记录、失败和北交所未核对天数仍单列。确认采集和收尾进程均已退出后，可用下列命令恢复。明确拒绝只可在核实官方解除限制后向采集脚本传 `--resume-refusal`。

```powershell
python -u outputs/dataset_v2/archive_completion_v1/verify_secondary_status.py --ohlcv --reopen-incomplete
python -u outputs/dataset_v2/archive_completion_v1/secondary_status_ohlcv_worker.py
python -u outputs/dataset_v2/archive_completion_v1/verify_secondary_status.py --ohlcv
python -u outputs/dataset_v2/archive_completion_v1/report_secondary_prices.py
```

严格 `G_ARCHIVE` 仍为 false。待补材料包括完整的跨交易所历史名录及改码链、覆盖每个市场的交易状态与日历证据、可独立复核的价格数据和历史发布版本。供应商状态可以解释停牌缺日，正式策略数据还需通过因果快照、执行账本和历史版本的独立验收。
