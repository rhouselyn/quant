# B0旧线性精排五seed与等权排名集成：结果

登记日期：2026-11-19 Asia/Shanghai。状态complete，五seed直接/线性及两个排名集成账本齐全。协议：[固定预算与对应OOF训练](research_20261119_linear_seed_ensemble_protocol.md)。父批：[旧线性首轮](research_20261118_tail_rerank_results.md)、[五seed主干](research_20261118_rolling_anchor_seeds_results.md)、[直接排名集成](research_20261119_rank_ensemble_results.md)。

## 结论

all5：线性相对对应seed直接排序，CAGR较高4/5、Sharpe较高4/5、回撤较低3/5；平均CAGR差+2.57个百分点，平均MDD改善+0.17个百分点。

new4：线性相对对应seed直接排序，CAGR较高3/4、Sharpe较高3/4、回撤较低2/4；平均CAGR差+2.79个百分点，平均MDD改善-0.19个百分点。

B0_direct排名集成：CAGR 17.02%、MDD 12.75%、Sharpe 1.313、周均换手122.06%；CAGR高于2/5个自身单seed，MDD低于5/5。

B0_linear排名集成：CAGR 18.79%、MDD 16.70%、Sharpe 1.317、周均换手121.30%；CAGR高于3/5个自身单seed，MDD低于3/5。

线性排名集成减直接排名集成：CAGR +1.77个百分点，MDD改善-3.95个百分点，Sharpe差+0.004；这些是同一历史路径的描述性差异，不等于未读市场期的可靠优势。

all5为42–46，new4为43–46；seed42参与旧线性发现阶段，新增四seed须单独观察。等权排名集成是重新选股执行后的净值，不能把单seed指标均值或周收益均值当成这一可执行组合。

## 方法与评价边界

五个B0各用自己的2010/2014截止模型产生滚动样本外分数，2010–2013/2014–2017两折，410输入周、408成熟监督周，保留SOURCE完整股票日期键左合并。缓存encoder_score先转float64，再沿原每周zscore，seed42与旧原分数及归一化一致；不读取跨折隐藏向量。原30特征加标准化分数、候选秩/N及与K位边界分差，共33维。

旧线性33→1、零初始化、.25*tanh残差；长Top100/短Bottom50预筛，候选外保留原主干顺序。原pair收益幅度/边界权重、.01修正平方罚、AdamW .001/.01、12轮、日期等权完全保留。每seed多空头各自训练统计，仅用成熟OOF候选拟合；对应seed控制训练日期排列。训练短头排除OOF基线Top20，组合推理排除最终精排长20，S10无多排除；旧分布差异未改。

各seed完整2018主干仍按原验证锁做多空独立三轮融合。线性每腿按153周完整验证10bp净Sharpe选12轮内单头，平局早轮；十头全锁后才生成本批历史精排结果，不按历史挑seed、头或修正强度。等权集成对同一完整可用池的完整秩取平均，先集成长20，再各seed在共同排除名单下重新取得Bottom50并用自己的统计/线性头精排，之后集成空侧。不混不同seed截断候选的局部秩。

2018–2020验证153周，2021-01-08起历史297周，最后决策2026-10-23、退出11月2日。L20/S10各全资本，LS50为Top20多/Bottom10空各50%；交易10bp配8.35%年借券，压力30bp配12%。候选资格、流动性、执行、现金及缺失估值规则沿原管线。CAGR按持有日历跨度，年度收益为累计净收益，MDD含初始1，Sharpe按周sqrt52，换手是连续权重绝对变动周均；2026非全年。

## 历史LS50：10bp十二策略

| arm       | seed         | ensemble   | cagr   | max_drawdown   |   sharpe | mean_turnover   |
|:----------|:-------------|:-----------|:-------|:---------------|---------:|:----------------|
| B0_direct | 42           | False      | 17.81% | 14.42%         |    1.405 | 109.20%         |
| B0_linear | 42           | False      | 19.51% | 12.82%         |    1.476 | 107.56%         |
| B0_direct | 43           | False      | 17.35% | 16.19%         |    1.35  | 114.41%         |
| B0_linear | 43           | False      | 19.53% | 15.57%         |    1.454 | 113.85%         |
| B0_direct | 44           | False      | 13.33% | 19.73%         |    1.057 | 125.55%         |
| B0_linear | 44           | False      | 16.83% | 18.97%         |    1.227 | 125.15%         |
| B0_direct | 45           | False      | 12.30% | 22.18%         |    0.998 | 120.96%         |
| B0_linear | 45           | False      | 18.22% | 22.49%         |    1.308 | 121.20%         |
| B0_direct | 46           | False      | 17.35% | 16.73%         |    1.351 | 122.97%         |
| B0_linear | 46           | False      | 16.91% | 18.57%         |    1.253 | 123.04%         |
| B0_direct | 等权排名集成 | True       | 17.02% | 12.75%         |    1.313 | 122.06%         |
| B0_linear | 等权排名集成 | True       | 18.79% | 16.70%         |    1.317 | 121.30%         |

## 跨seed稳定性（不含集成）

| cohort   | arm       | CAGR均值±样本std   | CAGR范围         | MDD均值±样本std   | MDD范围          | Sharpe均值±样本std   | Sharpe范围     | 换手均值±样本std   | 换手范围           |
|:---------|:----------|:-------------------|:-----------------|:------------------|:-----------------|:---------------------|:---------------|:-------------------|:-------------------|
| all5     | B0_direct | 15.630 ± 2.600     | [12.300, 17.812] | 17.850 ± 3.086    | [14.417, 22.179] | 1.232 ± 0.189        | [0.998, 1.405] | 118.620 ± 6.685    | [109.204, 125.548] |
| all5     | B0_linear | 18.199 ± 1.326     | [16.830, 19.528] | 17.683 ± 3.663    | [12.819, 22.487] | 1.344 ± 0.115        | [1.227, 1.476] | 118.161 ± 7.296    | [107.560, 125.152] |
| new4     | B0_direct | 15.084 ± 2.651     | [12.300, 17.351] | 18.709 ± 2.790    | [16.187, 22.179] | 1.189 ± 0.188        | [0.998, 1.351] | 120.973 ± 4.759    | [114.414, 125.548] |
| new4     | B0_linear | 17.872 ± 1.276     | [16.830, 19.528] | 18.899 ± 2.833    | [15.568, 22.487] | 1.311 ± 0.101        | [1.227, 1.454] | 120.811 ± 4.914    | [113.849, 125.152] |

均值和样本std（ddof=1）描述训练随机性；不同seed共享市场路径，不能当五份独立市场样本。

## 同seed及集成比较

| comparison                         | candidate_arm   | control_arm   | cohort   |   seed_count |   cagr_delta_mean |   sharpe_delta_mean |   mdd_improvement_mean |   turnover_reduction_mean |   cagr_delta_positive_count |   sharpe_delta_positive_count |   mdd_improvement_positive_count |   turnover_reduction_positive_count |
|:-----------------------------------|:----------------|:--------------|:---------|-------------:|------------------:|--------------------:|-----------------------:|--------------------------:|----------------------------:|------------------------------:|---------------------------------:|------------------------------------:|
| ensemble_linear_vs_ensemble_direct | B0_linear       | B0_direct     | ensemble |            1 |             1.771 |               0.004 |                 -3.948 |                     0.762 |                           1 |                             1 |                                0 |                                   1 |
| ensemble_vs_members                | B0_direct       | B0_direct     | all5     |            5 |             1.392 |               0.081 |                  5.099 |                    -3.445 |                           2 |                             2 |                                5 |                                   2 |
| ensemble_vs_members                | B0_direct       | B0_direct     | new4     |            4 |             1.938 |               0.124 |                  5.957 |                    -1.091 |                           2 |                             2 |                                4 |                                   2 |
| ensemble_vs_members                | B0_linear       | B0_linear     | all5     |            5 |             0.594 |              -0.026 |                  0.984 |                    -3.141 |                           3 |                             3 |                                3 |                                   2 |
| ensemble_vs_members                | B0_linear       | B0_linear     | new4     |            4 |             0.922 |               0.007 |                  2.200 |                    -0.491 |                           3 |                             3 |                                3 |                                   2 |
| linear_vs_direct                   | B0_linear       | B0_direct     | all5     |            5 |             2.570 |               0.111 |                  0.167 |                     0.458 |                           4 |                             4 |                                3 |                                   3 |
| linear_vs_direct                   | B0_linear       | B0_direct     | new4     |            4 |             2.788 |               0.121 |                 -0.191 |                     0.162 |                           3 |                             3 |                                2 |                                   2 |

CAGR/MDD改善/换手降低的平均差单位为百分点，Sharpe差无量纲；改善及胜出计数只计严格正值。ensemble_vs_members表示每条集成分别比较自身五个成员，ensemble_linear_vs_ensemble_direct为两个实际集成比较。

## 历史LS50逐年四指标：10bp

|   year | arm       | seed         | ensemble   | cumulative   | max_drawdown   |   sharpe | mean_turnover   |
|-------:|:----------|:-------------|:-----------|:-------------|:---------------|---------:|:----------------|
|   2021 | B0_direct | 42           | False      | -3.50%       | 14.42%         |   -0.223 | 106.13%         |
|   2021 | B0_linear | 42           | False      | 3.90%        | 12.82%         |    0.35  | 104.19%         |
|   2021 | B0_direct | 43           | False      | 2.07%        | 14.20%         |    0.226 | 113.65%         |
|   2021 | B0_linear | 43           | False      | 6.06%        | 15.57%         |    0.537 | 112.48%         |
|   2021 | B0_direct | 44           | False      | 2.25%        | 16.19%         |    0.243 | 120.99%         |
|   2021 | B0_linear | 44           | False      | -1.02%       | 18.97%         |   -0     | 120.24%         |
|   2021 | B0_direct | 45           | False      | -8.46%       | 17.39%         |   -0.613 | 116.43%         |
|   2021 | B0_linear | 45           | False      | 1.00%        | 17.46%         |    0.14  | 117.55%         |
|   2021 | B0_direct | 46           | False      | 11.75%       | 4.71%          |    1.022 | 118.64%         |
|   2021 | B0_linear | 46           | False      | 10.27%       | 10.29%         |    0.838 | 117.63%         |
|   2021 | B0_direct | 等权排名集成 | True       | 2.95%        | 12.20%         |    0.294 | 117.47%         |
|   2021 | B0_linear | 等权排名集成 | True       | -0.88%       | 16.70%         |    0.01  | 116.69%         |
|   2022 | B0_direct | 42           | False      | 36.94%       | 6.93%          |    2.858 | 103.59%         |
|   2022 | B0_linear | 42           | False      | 33.35%       | 7.95%          |    2.491 | 103.22%         |
|   2022 | B0_direct | 43           | False      | 34.67%       | 7.79%          |    2.723 | 110.76%         |
|   2022 | B0_linear | 43           | False      | 33.90%       | 6.10%          |    2.544 | 110.41%         |
|   2022 | B0_direct | 44           | False      | 24.57%       | 5.38%          |    1.968 | 121.05%         |
|   2022 | B0_linear | 44           | False      | 25.61%       | 5.14%          |    2.034 | 121.17%         |
|   2022 | B0_direct | 45           | False      | 28.83%       | 4.68%          |    2.479 | 120.14%         |
|   2022 | B0_linear | 45           | False      | 37.77%       | 4.80%          |    2.979 | 121.49%         |
|   2022 | B0_direct | 46           | False      | 41.35%       | 5.23%          |    3.349 | 116.23%         |
|   2022 | B0_linear | 46           | False      | 47.67%       | 5.06%          |    3.515 | 116.12%         |
|   2022 | B0_direct | 等权排名集成 | True       | 35.20%       | 6.07%          |    2.728 | 117.63%         |
|   2022 | B0_linear | 等权排名集成 | True       | 43.79%       | 4.19%          |    2.995 | 115.71%         |
|   2023 | B0_direct | 42           | False      | 7.77%        | 7.22%          |    0.778 | 114.07%         |
|   2023 | B0_linear | 42           | False      | 13.51%       | 7.02%          |    1.291 | 111.06%         |
|   2023 | B0_direct | 43           | False      | 12.94%       | 3.95%          |    1.383 | 119.03%         |
|   2023 | B0_linear | 43           | False      | 16.13%       | 4.28%          |    1.68  | 118.00%         |
|   2023 | B0_direct | 44           | False      | 4.33%        | 11.70%         |    0.415 | 130.77%         |
|   2023 | B0_linear | 44           | False      | 12.83%       | 9.91%          |    1.017 | 130.42%         |
|   2023 | B0_direct | 45           | False      | 7.41%        | 8.16%          |    0.681 | 126.94%         |
|   2023 | B0_linear | 45           | False      | 7.78%        | 10.05%         |    0.694 | 125.96%         |
|   2023 | B0_direct | 46           | False      | 4.71%        | 12.55%         |    0.439 | 126.97%         |
|   2023 | B0_linear | 46           | False      | 1.92%        | 15.31%         |    0.211 | 128.08%         |
|   2023 | B0_direct | 等权排名集成 | True       | 6.32%        | 8.67%          |    0.588 | 127.80%         |
|   2023 | B0_linear | 等权排名集成 | True       | 2.39%        | 10.27%         |    0.246 | 127.42%         |
|   2024 | B0_direct | 42           | False      | 31.71%       | 4.21%          |    2.401 | 112.85%         |
|   2024 | B0_linear | 42           | False      | 32.46%       | 4.63%          |    2.468 | 112.27%         |
|   2024 | B0_direct | 43           | False      | 37.97%       | 10.19%         |    2.406 | 113.79%         |
|   2024 | B0_linear | 43           | False      | 26.68%       | 10.56%         |    1.754 | 113.06%         |
|   2024 | B0_direct | 44           | False      | 43.74%       | 6.73%          |    2.687 | 131.80%         |
|   2024 | B0_linear | 44           | False      | 47.34%       | 7.73%          |    2.991 | 130.87%         |
|   2024 | B0_direct | 45           | False      | 38.92%       | 7.54%          |    2.589 | 121.70%         |
|   2024 | B0_linear | 45           | False      | 49.51%       | 7.04%          |    3.177 | 120.74%         |
|   2024 | B0_direct | 46           | False      | 36.30%       | 11.10%         |    2.229 | 126.34%         |
|   2024 | B0_linear | 46           | False      | 36.03%       | 11.44%         |    2.216 | 126.45%         |
|   2024 | B0_direct | 等权排名集成 | True       | 39.51%       | 8.39%          |    2.64  | 125.64%         |
|   2024 | B0_linear | 等权排名集成 | True       | 46.94%       | 9.45%          |    2.942 | 124.48%         |
|   2025 | B0_direct | 42           | False      | 15.07%       | 6.24%          |    1.264 | 113.52%         |
|   2025 | B0_linear | 42           | False      | 16.58%       | 6.50%          |    1.318 | 110.93%         |
|   2025 | B0_direct | 43           | False      | 13.83%       | 6.91%          |    1.257 | 114.70%         |
|   2025 | B0_linear | 43           | False      | 13.16%       | 9.95%          |    1.185 | 115.26%         |
|   2025 | B0_direct | 44           | False      | 6.27%        | 12.14%         |    0.56  | 125.67%         |
|   2025 | B0_linear | 44           | False      | 8.32%        | 11.91%         |    0.652 | 124.33%         |
|   2025 | B0_direct | 45           | False      | 7.20%        | 12.56%         |    0.684 | 120.46%         |
|   2025 | B0_linear | 45           | False      | 9.70%        | 15.18%         |    0.746 | 120.31%         |
|   2025 | B0_direct | 46           | False      | 14.35%       | 7.32%          |    1.273 | 125.24%         |
|   2025 | B0_linear | 46           | False      | 11.83%       | 7.69%          |    1.066 | 125.25%         |
|   2025 | B0_direct | 等权排名集成 | True       | 11.70%       | 6.59%          |    1.037 | 122.65%         |
|   2025 | B0_linear | 等权排名集成 | True       | 16.41%       | 10.19%         |    1.316 | 122.59%         |
|   2026 | B0_direct | 42           | False      | 20.02%       | 5.68%          |    1.507 | 104.13%         |
|   2026 | B0_linear | 42           | False      | 15.91%       | 7.24%          |    1.182 | 102.84%         |
|   2026 | B0_direct | 43           | False      | 3.86%        | 10.22%         |    0.359 | 114.62%         |
|   2026 | B0_linear | 43           | False      | 19.18%       | 11.08%         |    1.259 | 113.90%         |
|   2026 | B0_direct | 44           | False      | 1.91%        | 8.85%          |    0.237 | 122.59%         |
|   2026 | B0_linear | 44           | False      | 10.22%       | 7.49%          |    0.874 | 123.78%         |
|   2026 | B0_direct | 45           | False      | 3.98%        | 15.93%         |    0.393 | 120.16%         |
|   2026 | B0_linear | 45           | False      | 7.48%        | 14.58%         |    0.606 | 121.41%         |
|   2026 | B0_direct | 46           | False      | -1.76%       | 13.24%         |   -0.079 | 124.69%         |
|   2026 | B0_linear | 46           | False      | -1.89%       | 17.85%         |   -0.068 | 125.12%         |
|   2026 | B0_direct | 等权排名集成 | True       | 8.03%        | 9.27%          |    0.676 | 121.15%         |
|   2026 | B0_linear | 等权排名集成 | True       | 8.90%        | 8.75%          |    0.701 | 120.93%         |

逐年每seed精确差值及all5/new4改善计数见yearly_comparisons.csv、yearly_comparison_summary.csv；逐年稳定性见yearly_stability.csv。年度累计收益以上表cumulative为准，年度CAGR仅在CSV作为补充。

## 压力费用：30bp交易、12%年借券

| arm       | seed         | ensemble   | cagr   | max_drawdown   |   sharpe | mean_turnover   |
|:----------|:-------------|:-----------|:-------|:---------------|---------:|:----------------|
| B0_direct | 42           | False      | 3.46%  | 24.48%         |    0.34  | 109.33%         |
| B0_linear | 42           | False      | 5.14%  | 20.13%         |    0.46  | 107.69%         |
| B0_direct | 43           | False      | 2.51%  | 23.89%         |    0.262 | 114.57%         |
| B0_linear | 43           | False      | 4.48%  | 23.03%         |    0.405 | 114.00%         |
| B0_direct | 44           | False      | -2.13% | 27.68%         |   -0.107 | 125.73%         |
| B0_linear | 44           | False      | 0.93%  | 23.15%         |    0.137 | 125.34%         |
| B0_direct | 45           | False      | -2.57% | 29.96%         |   -0.147 | 121.13%         |
| B0_linear | 45           | False      | 2.56%  | 29.98%         |    0.256 | 121.38%         |
| B0_direct | 46           | False      | 1.61%  | 24.63%         |    0.191 | 123.15%         |
| B0_linear | 46           | False      | 1.22%  | 27.47%         |    0.159 | 123.22%         |
| B0_direct | 等权排名集成 | True       | 1.42%  | 21.67%         |    0.176 | 122.24%         |
| B0_linear | 等权排名集成 | True       | 3.04%  | 24.85%         |    0.287 | 121.47%         |

### 压力费用稳定性

| cohort   | arm       | CAGR均值±样本std   | CAGR范围        | MDD均值±样本std   | MDD范围          | Sharpe均值±样本std   | Sharpe范围      | 换手均值±样本std   | 换手范围           |
|:---------|:----------|:-------------------|:----------------|:------------------|:-----------------|:---------------------|:----------------|:-------------------|:-------------------|
| all5     | B0_direct | 0.575 ± 2.755      | [-2.570, 3.461] | 26.128 ± 2.600    | [23.890, 29.957] | 0.108 ± 0.221        | [-0.147, 0.340] | 118.782 ± 6.706    | [109.335, 125.731] |
| all5     | B0_linear | 2.864 ± 1.890      | [0.934, 5.135]  | 24.752 ± 3.924    | [20.134, 29.977] | 0.283 ± 0.145        | [0.137, 0.460]  | 118.323 ± 7.318    | [107.688, 125.335] |
| new4     | B0_direct | -0.146 ± 2.579     | [-2.570, 2.506] | 26.540 ± 2.807    | [23.890, 29.957] | 0.050 ± 0.207        | [-0.147, 0.262] | 121.144 ± 4.772    | [114.565, 125.731] |
| new4     | B0_linear | 2.296 ± 1.616      | [0.934, 4.475]  | 25.907 ± 3.412    | [23.031, 29.977] | 0.239 ± 0.122        | [0.137, 0.405]  | 120.982 ± 4.928    | [114.000, 125.335] |

## 配对不确定性

| comparison                         | candidate_arm   |   candidate_seed | control_arm   |   control_seed |   mean_delta_bps |   mean_lo_bps |   mean_hi_bps |   tail10_delta_bps |   tail10_lo_bps |   tail10_hi_bps |
|:-----------------------------------|:----------------|-----------------:|:--------------|---------------:|-----------------:|--------------:|--------------:|-------------------:|----------------:|----------------:|
| linear_vs_direct                   | B0_linear       |               42 | B0_direct     |             42 |            2.912 |        -2.212 |         8.539 |            -12.298 |         -33.463 |          12.443 |
| linear_vs_direct                   | B0_linear       |               43 | B0_direct     |             43 |            3.712 |        -4.952 |        12.172 |             -1.250 |         -18.943 |          18.868 |
| linear_vs_direct                   | B0_linear       |               44 | B0_direct     |             44 |            6.168 |        -1.555 |        13.887 |            -14.146 |         -50.285 |          17.129 |
| linear_vs_direct                   | B0_linear       |               45 | B0_direct     |             45 |           10.363 |         3.276 |        16.434 |            -31.291 |         -60.493 |          -5.226 |
| linear_vs_direct                   | B0_linear       |               46 | B0_direct     |             46 |           -0.551 |        -7.543 |         5.759 |            -22.856 |         -50.304 |          -1.964 |
| ensemble_linear_vs_ensemble_direct | B0_linear       |               -1 | B0_direct     |             -1 |            3.267 |        -3.794 |         9.434 |            -35.456 |         -60.571 |         -11.790 |
| ensemble_vs_members                | B0_direct       |               -1 | B0_direct     |             42 |           -1.222 |       -12.372 |        11.625 |            -15.782 |         -51.372 |          24.987 |
| ensemble_vs_members                | B0_direct       |               -1 | B0_direct     |             43 |           -0.511 |       -11.073 |         9.605 |             -5.963 |         -39.300 |          28.986 |
| ensemble_vs_members                | B0_direct       |               -1 | B0_direct     |             44 |            6.275 |        -4.781 |        16.849 |             13.152 |         -11.460 |          40.573 |
| ensemble_vs_members                | B0_direct       |               -1 | B0_direct     |             45 |            8.117 |        -3.302 |        17.868 |             16.407 |         -18.189 |          49.350 |
| ensemble_vs_members                | B0_direct       |               -1 | B0_direct     |             46 |           -0.510 |       -12.424 |        12.691 |             -4.856 |         -45.120 |          38.318 |
| ensemble_vs_members                | B0_linear       |               -1 | B0_linear     |             42 |           -0.866 |       -14.839 |        13.536 |            -38.939 |         -81.700 |           3.595 |
| ensemble_vs_members                | B0_linear       |               -1 | B0_linear     |             43 |           -0.956 |       -15.067 |        12.345 |            -40.169 |         -84.472 |           2.770 |
| ensemble_vs_members                | B0_linear       |               -1 | B0_linear     |             44 |            3.375 |        -7.555 |        12.529 |             -8.158 |         -45.076 |          37.014 |
| ensemble_vs_members                | B0_linear       |               -1 | B0_linear     |             45 |            1.021 |       -10.649 |        11.234 |             12.242 |         -31.835 |          52.277 |
| ensemble_vs_members                | B0_linear       |               -1 | B0_linear     |             46 |            3.308 |        -8.302 |        16.626 |            -17.455 |         -63.296 |          30.334 |
| mean_linear_vs_mean_direct         | B0_linear       |               -2 | B0_direct     |             -2 |            4.521 |         0.462 |         8.458 |            -20.467 |         -34.824 |          -5.237 |
| mean_linear_vs_mean_direct         | B0_linear       |               -2 | B0_direct     |             -2 |            4.923 |        -0.074 |         9.581 |            -22.645 |         -39.335 |          -7.214 |

沿原8周共同市场日期移动块重抽样，单位bp/周。mean_linear_vs_mean_direct是单seed周收益均值对照，不等于真实排名集成。历史已读、多方案比较且未校正多重比较，不能从区间或成员胜出个数推出普适优越性。

## 验收与复现

32400周账本、144全期和648年度指标均由原周账本重新计算，并以独立公式交叉检查收益、CAGR、MDD、Sharpe和换手；153/297日历逐案一致、无重复、费用分解通过。训练来源/归一化、锁、原直接账本及实际排名/排除规则需结合主脚本输入清单和独立执行审计，指标重算不能替代接线验证。

复现：`python scripts/report_linear_seed_ensemble.py --out outputs/linear_seed_ensemble_v1`；`--self-check`仅跑合成指标、种子匹配和比较计数。完整三组合/双费用/验证及历史见report目录。源码与输入冻结于本批manifest；结果与权重仅本机，未异地归档。

### 核心SHA256

- `C:\Users\14815\Projects\quant\scripts\report_linear_seed_ensemble.py`：`95f1ac8646da931882363a0cd476f2a0114215d1a118aa7e7ce800bb980bb31f`

- `C:\Users\14815\Projects\quant\scripts\report_rolling_anchor_seeds.py`：`900cb4832c1429d5b6810efa237680b830c9005f04e9d223bd21ead5031f47da`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\report\weekly.parquet`：`9fe4ce7b2d6ccb5e1eb38ebd1f8e997135ff3976fe72d9d7dde043b18392b628`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\input_manifest.json`：`0106db6b2d0ea1b2bdef0708da44ca8fec87f19f49d0b19e39c8190d857aadf1`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\rule_lock.json`：`7070713a580b0022638c0fd218ae51725e000fabc36e68d0dbcfd98dff0240b7`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\all_validation_locked.json`：`0b5073b02c359c9d70c49f4a842ff78790ff83e1df696415dee3997567fcb027`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\report\metrics.csv`：`2d59ef7cc00ee28e8ea437a436532882ae0702ce8173e2f6256bd2dc53f20819`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\report\yearly.csv`：`69834d53ca4663728527ac4d9ba0715411bcad8af997f6eee14219277e123282`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\report\stability.csv`：`805787a11f0b5f5c6eebade784fab76093e9eeec139c1057eaeea47b1209b8fa`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\report\yearly_stability.csv`：`3abad63417d7905df370c5081c6d0fbccb5d5a50b1c467b988059c61b4b21426`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\report\comparisons.csv`：`5617a3f9df38a3cc1e957f04471da59c2985f0dae77d3edf17e67d5a1e678b0a`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\report\comparison_summary.csv`：`ab64b0e091bcb0508b9f87241a221b4c1166051d3296d9d3f08e37ba38040b0d`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\report\yearly_comparisons.csv`：`6dd0e385a30b0518759241102f3c8ff74e82e4024ec27956ef04c4f7cded1194`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\report\yearly_comparison_summary.csv`：`0116b1aa216c2594a168b3e53aa74ab86b0296b3d6c9ddcf4b642f8d36cece36`

- `C:\Users\14815\Projects\quant\outputs\linear_seed_ensemble_v1\report\paired.csv`：`6861ee3906297ff8b4522ece99778c7a42b0a3450e79b4f2e34bc029c240a22f`

## 限制

本批为用户指定的已读历史复检；未更换数据切分、费用、训练预算、修正幅度或集成权重。固定幸存池、回溯行业/复权、代理执行和估值、历史资格不等于真实券源等原限制继承。未验证新未读市场期或真实成交；未部署、未扩seed或超参队列。
