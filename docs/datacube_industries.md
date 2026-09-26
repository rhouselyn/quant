# 数立方行业分类 API 核对

核对日期：2026-09-14。官方网站：https://datacube.foundersc.com/ 。PyPI 的 dcube 项目主页也指向该站。

| 接口 | 用途 | 官方文档 |
| --- | --- | --- |
| a_cni_industries | 国证行业，非中信 | https://datacube.foundersc.com/document/2?doc_id=10319 |
| a_index_memberscitics | 中信一级指数成分，最适合本项目的一级分类 | https://datacube.foundersc.com/document/2?doc_id=10717 |
| ashare_ind_class_citics | 股票中信行业代码及分类变更日期 | https://datacube.foundersc.com/document/2?doc_id=10799 |
| a_share_Industriescode | 行业代码、名称、级数和有效标志；需核实返回数据中的体系及层级 | https://datacube.foundersc.com/document/2?doc_id=10714 |

官方文档说明单次最多10000条、日更新，SDK通过 `dc.pro_api(token)` 初始化。API文档公开不代表行情/分类数据可匿名下载；尚未用有效凭据实测权限和800股覆盖率。

官方中信一级示例：

```python
import os
import dcube as dc
pro = dc.pro_api(os.environ['DCUBE_TOKEN'])
df = pro.a_index_memberscitics(code='CI005001.WI')
# columns: code, s_con_windcode, con_indate, con_outdate, cur_sign, ...
```

此示例仅查询一个行业指数，不能当作已取得全部30行业。完整接入需核验当前30个一级行业指数代码及名称，拉取成分并根据最新标志和纳入/剔除日期确定当前归属，校验800股每股恰好一个分类。证券代码 `.SH` 与本项目Yahoo的 `.SS` 需要统一；不得将二三级或国证分类静默替代一级中信。

若使用股票分类接口，返回 `citics_ind_code`，还需核对行业代码表/父级关系后得到一级名称，不能凭未经验证的字符串截断推断层级。最终归一化为 `code,citic_l1` CSV，使用 `scripts/import_citic_industries.py` 校验并导入。保留提供方、分类日期和原始结果作为来源。行业目前仅为元数据，不影响本轮训练。


## 2026-09-15 补充来源核查

尚未取得最新全市场分类文件，也未生成600股缓存或开始训练。检查了AKShare安装包的股票/指数接口、中信证券公开网站、公开搜索和代码仓库；没有找到可核验的最新全市场中信一级映射。部分网页返回403或验证页面，不将其视为公开可用数据源，也不绕过限制。

另一条明确有官方文档的路径：米筐RQData。
https://www.ricequant.com/doc/rqdata/python/stock-mod

- `get_industry_mapping(source='citics_2019', date=...)` 返回一级/二级/三级行业代码及名称。
- `get_industry(industry=一级行业代码或名称, source='citics_2019', date=...)` 返回股票列表。
- 应选 `citics_2019`；文档中的 `citics` 是2019新版之前的旧分类。
- 文档公开不代表接口免鉴权；本机尚无可用权限，未实测全市场返回值或试用资格。

建议申请数立方或RQData的行业分类权限/试用，只需要分类数据，行情继续复用Yahoo缓存。可直接提出以下数据请求（尚未向任何第三方发送）：

> 用于本地量化研究，需要指定最新交易日的沪深A股中信2019版一级行业映射。请提供证券代码、证券简称、一级行业代码、一级行业名称、分类纳入/剔除日期、数据日期，以及合法使用范围。仅需当前全市场分类快照，不需要行情。请确认是否可提供试用或CSV导出。

取得数据后先统计每类数量，再执行30×20不放回抽样；不能预先保证每个行业都有20支股票。缺口必须报告，不能重复抽股或跨行业补齐。第三方过期快照、研究报告局部表和接口示例不能冒充当前完整名单。
