# quant MVP

这是一个可运行的周频量化原型：合成/真实 A 股数据 → 技术特征 → Mamba2 风格时序编码器 → 时间 InfoNCE + 极端分位 score + 可选四分位块间排序 → 样本外多空和 Top-5 多头回测与图表。

## 快速运行

```powershell
python scripts/run_mvp.py --smoke-test --synthetic --output outputs/smoke
```

默认 smoke test 使用 20 支股票和 180 个交易日，CPU/GPU 均可运行。完整合成实验：

```powershell
python scripts/run_mvp.py --synthetic --output outputs/full
```

运行后会生成 `metrics.json`、`training_metrics.csv`、`predictions.parquet`、`equity_curve.csv`、`training_loss.png`、`rankic.png` 和 `equity_curve.png`。可直接打开 PNG 和 JSON 检查训练及回测结果。

## 数据

`src/quant_mvp/data.py` 默认生成可复现的合成面板数据，并缓存到 Parquet。真实数据入口支持 Baostock（免费前复权日线）和 Yahoo 备用接口。不能把合成回测结果当作真实交易结论。

真实数据（Baostock，免费前复权日线）运行：

```powershell
python -m pip install baostock
python scripts/run_mvp.py --config configs/real_smoke.yaml --real --output outputs/real_smoke
python scripts/run_mvp.py --config configs/real_medium.yaml --real --output outputs/real_medium
python scripts/run_mvp.py --config configs/real_long20.yaml --real --output outputs/real_long20
python scripts/run_mvp.py --config configs/real_long.yaml --real --output outputs/real_long
```

Baostock 下载可能受网络和服务端限流影响；下载结果会缓存到 `data/`，重复运行不会重复请求。

## 前端面板

```powershell
python dashboard/app.py --output-root outputs --port 8501
```

打开 <http://127.0.0.1:8501>，可切换实验并查看训练损失、RankIC、净值、回撤、最新交易日散点和多空排名。面板会对合成数据实验显示提示。

## 设计

窗口只读取当前及历史特征；周频标签为下一周 close-to-close 对数收益。模型输出 L2 归一化表示 `z` 和独立的标量 score。训练目标包括：

* **极端 score 损失**：每天将下一日收益横截面最高 20% 标成 `+1`、最低 20% 标成 `-1`，对每只股票的 score 使用 signed logistic loss，中间 60% 不参与该损失；
* **时间 InfoNCE**：同一股票未来 1、3、5 周的因果 prefix 都作为正样本；每个 horizon 单独计算后平均，同一 anchor 周的其他股票作为负样本，key 使用 EMA encoder，只在训练时存在。
* **四分位块间排序**：把下一周收益横截面分成 4 块，只比较不同块，块内不产生 pairwise 梯度；同时保留较小权重的经典 RankIC 以维持块内排序能力。

回测阶段只使用 query 分支的 score，按每周 score 做多 top 10%、做空 bottom 10%，并额外计算 score 最高的 5 支股票等权做多曲线。多空组合扣除双边成本，Top-5 组合扣除单边成本。128 维表示仍可选用每 8 维一组的去相关正则；group 正则默认先 warm-up 后启用。

周频实验使用 `configs/real_contrastive_weekly_100_yahoo.yaml`：先将日线 OHLCV 聚合为周五收盘周线，再计算周级特征、下一周收益、RankIC 和回测。当前配置使用每个 batch 随机选择 20/40/60 周 query、同一股票未来 1/3/5 周的多正样本、四分位块间排序、经典 RankIC 和 0.001 权重的 Group 去相关正则（前两个 epoch warm-up）。配置包含 100 支股票、520 周、5 个 epoch，并使用 `torch_mamba2`（纯 PyTorch Mamba2 selective state-space 实现）。

详细方案见 [BUILD.md](BUILD.md)。

## 本地可视化面板

训练产物可以用轻量 Flask 面板查看训练曲线、样本外多空/Top-5 净值、每周 RankIC、最近 10 个周末的分数与多空候选：

```powershell
python dashboard/app.py --output-root outputs --port 8501
```

浏览器访问 <http://127.0.0.1:8501>。面板会自动发现 `outputs/` 下的实验目录；完整说明见 [dashboard/README.md](dashboard/README.md)。
