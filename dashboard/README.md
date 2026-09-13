# Quant MVP 可视化面板

面板使用 Flask 提供本地页面，Plotly 从 CDN 加载交互图表。它会自动发现 `outputs/` 下包含 `metrics.json` 的实验目录，因此重新训练后只要刷新页面就能查看新结果。

在仓库根目录运行：

```powershell
python dashboard/app.py --output-root outputs --port 8501
```

然后打开 <http://127.0.0.1:8501>。也可以指定其他产物目录：

```powershell
python dashboard/app.py --output-root outputs/real_akshare --port 8501
```

页面包含：

- RankIC 均值、净最终净值、Sharpe、最大回撤、股票/交易日数和训练后端；
- 样本外多空 Gross/Net 与 Top-5 long-only Gross/Net 净值曲线；
- 总损失、Score tail、Temporal InfoNCE、块间排序、Group 正则与验证 RankIC；
- 每日/每周 RankIC；
- 最近 10 个交易日/周的模型分数-实际收益散点图，以及对应日期的做多/做空候选排名。

选择周频实验时，页面会自动将标题切换为“每周 RankIC”“最新交易周”，并按 52 周年化 Sharpe 展示结果。

面板只读取本地产物，不会触发数据下载或重新训练。Plotly 使用公共 CDN；如果离线访问，指标和表格仍会加载，但图表需要将 Plotly JS 下载到本地后才能显示。
