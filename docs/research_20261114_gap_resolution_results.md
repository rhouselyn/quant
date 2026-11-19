# 二源缺口处理和模型结论

日期：2026-11-14，Asia/Shanghai。状态：部分完成，仍有未解释缺口。协议背景见 [数据研究执行记录](research_20261113_data_trials_results.md)。

## 数据发布与核验

原二源3,486个请求均通过分页核验。发布 `mixed_baostock_c8a25e9903a6505b`，在 `sina_szalias_45d7e086a044229a` 的逻辑行情基础上增加4,238条BaoStock未复权报价，按证券和日期验证与父数据无重叠，保留每行源响应哈希，原深赤湾旧码5,872条仍可查询。补充修正000003.SZ上市日为1991-07-03、000004.SZ为1991-01-14；此前错误日期造成136个上市前伪缺口。来源分别是[金田实业年报](https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10179889&stockid=000003)和[国农科技年报](https://static.cninfo.com.cn/finalpage/2014-04-22/63889870.PDF)。原始源文件和SHA-256已保存。

上海医药1994-03-24至2010-02-03的3,634条BaoStock报价可以用生效代码600849.SH查询。2010-03-05后代码为601607.SH，证券主体延续，但2009–2010年发生换股吸收合并，不能将重组简单看作无经济变化的改名。 [上海医药2010年年报](https://static.sse.com.cn/cs/zhs/scfw/gg/ssgs/2011-03-09/601607_2010_n.pdf)

Tencent和BaoStock对3,634个共同日开高低收全部一致，成交量3,619日一致、15日不一致。15条在发布的 `quality_status` 中标为 `cross_vendor_volume_conflict`，两源逐行结果在 `outputs/dataset_v2/archive_completion_v1/qq_independent_v1/sh601607_comparison.parquet`。腾讯旧代码600849接口返回空，腾讯现行代码带有旧史。有效代码查询不应把供应商回填代码当成当日有效代码。

原1,655个无状态日中，上港集箱600018的1,510日里，Tencent补获1,448条有效价量，发布独立 `qq_600018_premerger_v2_d41241b5c57b86ad`；金额字段源中未可靠确认，保持null。其为被吸收合并前独立主体 `SH:600018_PREMERGER`，与合并后主体不混作一个证券；自2006-10-20起旧主体终止上市，此后至10-25的4个会话是旧生命周期外，不记缺口。 [上证所公告报道](https://epaper.cs.com.cn/zgzqb/images/2006-10/19/A02/ZBSCa19C.pdf)

原1,655日因此得到1,448条独立供应商报价、136条上市前修正及4条终止后修正，剩67日未解释，其中600018为58日，000015.SZ为2日，000556.SZ为2日，600625.SH、600190.SH各1日，600786.SH为3日。以上是初始无状态集合的划分，不等于所有证券全历史无缺口。

北交所292个缺口重新生成并按有效旧码、现行码检查。腾讯现行代码28个证券年请求覆盖所有292个目标日，目标日均无记录；供应商不返回记录不能单独证明停牌。公司停复牌披露确认920305的46个会话、920680的57个会话确属停牌，合计103个；已发布 `official_bj_halts_96cb044aa2e1cf41` 作为可查询生命周期事件，剩189个未解释。 [云创停牌](https://www.bse.cn/disclosure/2026/2026-04-29/431a0bca807849be8df63d636cb12f8f.pdf) · [云创复牌](https://www.bse.cn/disclosure/2026/2026-07-01/7d206912396248988bfd4c68f768eaef.pdf) · [广道停牌](https://notice.10jqka.com.cn/api/pdf/72bb0de769fb5884.pdf) · [广道复牌](https://www.bse.cn/disclosure/2025/2025-12-03/042baf7419ce41de97c4eea63206174b.pdf)

验收包括每行有效OHLCV与正成交量、无父档重复、发行人代码区间、深赤湾旧码查询、15个冲突标记、公告停牌区间无父档成交、PIT模式拒绝、腾讯目标日与基准核对、源响应内容哈希存档。`tests/test_ashare_dataset_store.py` 七项通过。标准化查询仅为latest-history；不可用作严格PIT财务或真实券源。两次早期发布尝试 `mixed_baostock_f7f482b88eb59e8b`、`mixed_baostock_ef048b206a883bcb`、`mixed_baostock_a374096d41c48d60` 及早期 `qq_600018_premerger_d41241b5c57b86ad` 未通过完整验收，应使用上面列出的当前版本。

```powershell
python scripts/ashare_dataset.py query --release mixed_baostock_c8a25e9903a6505b --codes 600849.SH --code-mode effective --start 1994-03-24 --end 2010-03-05
python scripts/ashare_dataset.py query --release qq_600018_premerger_v2_d41241b5c57b86ad --codes 600018.SH --code-mode effective --start 2000-07-19 --end 2006-10-20
python scripts/ashare_dataset.py query --release official_bj_halts_96cb044aa2e1cf41 --dataset security_lifecycle --codes 920305.BJ
```

上述数值与验收汇总为 `outputs/dataset_v2/archive_completion_v1/gap_resolution_20261114.json`；代码、原始响应和产物仅本机，尚无异地归档。已发布状态仍为incomplete：约58万条其他BaoStock停牌状态尚未经逐日交易所公告核验，券源、财务PIT、历史行业和全量退市生命周期也未证明完备。新增数据尚未重建模型快照，模型收益不反映本次修补。

## 模型效果

8轮三种子D3多头相对同容量零通道验证净CAGR差为+0.52、+8.69、+0.33个百分点，三种子融合验证18.69%对14.21%；周差8周块区间跨零。三种子测试融合多头29.25%对27.88%、空头4.37%对−7.26%，测试已多次暴露，作为描述。12轮仅续训8轮时多头选轮触边的42和43及对应对照，验证多头分别18.43%/17.57%、18.70%/9.11%；完整测试28.04%/26.62%、27.00%/26.35%。该两种子属于有选择的续训，不能叫三种子12轮验证。价量缺失信息有初步正向信号，但多头增量主要依赖seed43、区间及数据偏差较大；不宜据此确认稳定实盘收益，也没有分解市场/行业暴露而不能认定是纯alpha。详见[模型续训记录](research_20261113_data_trials_results.md)。
