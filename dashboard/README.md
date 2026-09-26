# Quant MVP 可视化面板

面板使用 Flask 提供本地页面，Plotly 从 CDN 加载交互图表。它会自动发现 `outputs/` 下包含 `metrics.json` 的实验目录，因此重新训练后只要刷新页面就能查看新结果。

默认训练配置 `configs/default.yaml` 与 `configs/real_sw_580_full_rankic_10ep.yaml` 一致：原600股剔除综合14股和美容护理6股，保留29个申万行业各20股，共580股；Yahoo周线、520周、10 epochs、40周query、未来5周单key、128维embedding、单score head。Score收益头尾各20%等权；RankIC使用全横截面，权重0.5。分层注意力目前为架构建议，尚未实现或训练。

页面仍优先展示已完成的历史 `real_weekly_500_full_rankic_10ep`，不能把旧结果当作580股新数据的回测。其他实验和原始产物保留。风格分析位于 `/style-analysis`，比较历史500股原分数与中性化结果，不改变原模型或回测。

当前会话服务在 <http://127.0.0.1:8502/>；以下命令默认端口8501。

在仓库根目录运行：

```powershell
python dashboard/app.py --output-root outputs --port 8501
```

然后打开 <http://127.0.0.1:8501>。也可以指定其他产物目录：

```powershell
python dashboard/app.py --output-root outputs/real_akshare --port 8501
```

页面包含：

- 对包含 long_only 指标的实验，优先展示 Top-20 做多净收益、净 Sharpe、回撤、等权基准、相对收益和换手，以及逐轮验证集选模表；选股排名展示 Top-20。页面披露周收盘收益代理、成本和缺失持仓收益处理。

- RankIC 均值、Normal IC、ICIR、IC t-stat/p 值、Top-Bottom Spread、净最终净值、Sharpe、最大回撤、股票/交易日数和训练后端；
- 样本外多空 Gross/Net 与 Top-5 long-only Gross/Net 净值曲线；
- 总损失、Score tail、Temporal InfoNCE、RankIC loss、块间排序、Group 正则与验证 RankIC；
- 每日/每周 RankIC；
- 因子信号质量分组收益图、单调性和年化 Top-Bottom Spread；
- 最近 10 个交易日/周的模型分数-实际收益散点图，以及对应日期的做多/做空候选排名。

选择周频实验时，页面会自动将标题切换为“每周 RankIC”“最新交易周”，并按 52 周年化 Sharpe 展示结果。

面板只读取本地产物，不会触发数据下载或重新训练。Plotly 使用公共 CDN；如果离线访问，指标和表格仍会加载，但图表需要将 Plotly JS 下载到本地后才能显示。

默认按验证集 Top-20 净 Sharpe 选模，只保存 best.pt 和 newest.pt（逐轮覆盖）。修改默认配置不改变历史产物。

纯头尾RankIC结果页显示与上一轮等权Score、混合RankIC的同区间测试对照。历史实验保留各自的参数与结果。

中信30×20实验配置已准备，等待真实分类映射生成600股缓存和训练结果。页面当前历史实验参数不随默认配置更改。

用户XLS申万分层实验 `real_sw_file_20_mixed_rankic_8ep` 在完成后优先展示，600股、31个行业，页面显示实际行业分布（综合14、美容护理6，其余20）。来源为用户申万分类文件，不标记成中信。

可靠性实验 `real_reliable500_top150_5ep` 展示原模型500股、筛选150股、150股重训三组曲线，以及平衡准确率、方向召回率和Brier。原始排名与测试标签分离保存，页面注明验证复用与滚动校准限制。

580股行业未来收益相关性报告：`/industry-analysis`。包含29×29行业矩阵、不同未来累计期限、未来第1至5周单周收益相关、同业/跨业股票相关、特征持续性及缺失诊断。仅为数据分析，未训练注意力模型或进行InfoNCE消融。

分层模型已实现：`/hierarchy-comparison` 显示关闭InfoNCE、两周0.1、五周0.1三组各10轮的实时状态，完成后显示验证选模与测试对照。市场token参与两次行业间注意力，最后一次行业内注意力将全局信息传回股票；没有股票cross30。
