# 行业收益分布与个股组合覆盖层的文献证据

2026-11-16（Asia/Shanghai）。已完成一手文献核对。本页供方案设计使用，未进行本地建模或回放。

## 行业历史路径是否有信号

- **最贴近本项目的实证。** Ellington、Stamatogiannis 和 Zheng（2022）用 1997 年 7 月至 2017 年 12 月沪深 A 股构造 6 个或 26 个 SIC 市值加权行业组合，使用行业滞后收益做 OLS post-LASSO。作者发现石油、电信和金融行业在该样本中有领先作用。主要样本外预测与多空策略表采用月频递推，附注 22 报告周频和日频点预测结论方向相近，未给出相应全表。可据此设跨行业收缩线性基线。行业定义、收益窗口、持仓约束、成交和成本均需按本项目重新检验。[论文 §2、§4.2–4.3](https://livrepository.liverpool.ac.uk/3156983/1/IRFA_R%26R_2022.pdf) · [出版记录](https://doi.org/10.1016/j.irfa.2022.102249)
- **动量只提供基线。** Moskowitz 和 Grinblatt（1999）发现美国过去赢家行业与输家行业在其历史样本有表现差异。Boubaker、Du 和 Liu（2022）对 2007 年 6 月至 2020 年 12 月沪深 A 股的月度行业动量报告了按历史相关性合并行业后的改善。两篇论文足以把动量列为预设基线，周频信号的方向、窗口和可兑现幅度仍待检验。[Moskowitz 和 Grinblatt](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00146) · [Boubaker、Du 和 Liu](https://link.springer.com/article/10.1057/s41260-021-00248-8)
- **波动率和均值是两个研究问题。** Luo 等（2025）用美国总市场及行业高频数据，将实现协方差拆为波动率与相关系数，检验多变量 HAR 和行业分散化。可据此设仅改变风险、均值置零的对照，超额收益预测仍需要独立证据。[Luo 等](https://doi.org/10.1016/j.jempfin.2025.101595)

## 架构与分布头

| 方法 | 一手结果与可迁移部件 |
| --- | --- |
| iTransformer，ICLR 2024 | 把每条变量历史编码为一个 token，注意力跨变量。论文以 MSE 和 MAE 评价多步点预测，Exchange 数据为 8 国日度汇率水平。可迁移跨行业注意力，与同预算的无跨行业共享编码器配对比较。[论文 §3–4、附录 A](https://proceedings.iclr.cc/paper_files/paper/2024/file/2ea18fdc667e0ef2ad82b2b4d65147ad-Paper-Conference.pdf) |
| Chronos-2，2025 技术报告及 2026 预印本 | 交替执行时间注意力与分组注意力，通过分位数头输出各变量及时间点的边际分位数。边际分位数留有多种依赖结构，因此现成头无法确定组合收益的联合尾部。可借鉴小型时间编码器和行业间注意力。使用外部预训练权重做历史实验前需核对训练覆盖。[2025 报告 §3](https://arxiv.org/html/2510.15821v1) · [2026 预印本 §2、附录 A](https://arxiv.org/html/2610.41956v1) |
| GPVar，NeurIPS 2019 | 共享 RNN 加低秩高斯 copula，为边际分布和时变依赖建立参数化模型。论文基准包括汇率水平、电力、交通等。可借鉴共享参数和低秩协方差，首版可简化为高斯头与训练期估计的收缩相关矩阵。[Salinas 等，§3–5](https://proceedings.neurips.cc/paper_files/paper/2019/file/0b105cf1504c4e241fcc6d519ea962fb-Paper.pdf) |
| TACTiS-2，ICLR 2024 | Transformer 生成条件联合分布，分阶段学习边际和 copula。论文用合成样本及电力、宏观、交通等公开数据评估。只有低秩结构在本地联合校准上出现可复现缺口时，才有理由检验更灵活的依赖模型。[Ashok 等，§4–5](https://openreview.net/pdf?id=xtOydkE1Ku) |
| TimePrism，ICLR 2026 | 三个线性层直接输出带概率权重的多变量场景。论文在汇率、电力等公开数据上主要报告基于 L1 距离的 weighted CRPS 与最小场景误差，分别衡量边际分布质量与最接近实际值的场景误差。若采用这个场景表示，仍需另外验证联合相关、组合投影和尾部校准。[Dai 等，§3–4](https://openreview.net/pdf/b2c90588f26d19fab20408636a7cfd8ac2686188.pdf) |

## 评分如何对齐决策

- 行业均值用日期等权平方误差核对，边际分布用 CRPS 和 PIT，联合分布用样本外联合 log score 或欧氏距离 energy score。Energy score 理论上严格适当，对相关性错设的有限样本灵敏度仍需检查。Variogram score 可补足行业间依赖诊断，对共同位移不敏感，可搭配组合投影 CRPS。[Gneiting 和 Raftery](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) · [Scheuerer 和 Hamill](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf)
- 对每周决策前可知的基线行业权重、预定行业价差及覆盖增量，在同一预测上计算 $a_t^\top Y_{t+1}$ 的分布，再检验该投影的 CRPS 或分位数覆盖。共享且预先确定的投影允许跨模型比较。各模型自己的交易权重则用于各自的决策回放。投影分数只识别所选投影的分布。[Pic 等，§3](https://ascmo.copernicus.org/articles/11/23/2025/index.html)
- 协方差先用常数收缩和 EWMA 收缩验证，给相同的均值预测更换风险头，检验时变相关能否带来额外信息。若由同一行业收益向量构造市场 $b_t^\top Y_{t+1}$，且 $b_t$ 权重和为一，则相对收益满足 $b_t^\top(Y_{t+1}-\mathbf 1b_t^\top Y_{t+1})=0$。按本项目推导，联合似然应作用于 $G-1$ 维对比坐标，或在原行业坐标建模后对场景做同一变换。[Ledoit 和 Wolf](https://www.ledoit.net/Well-conditioned2004.pdf)
- 组合采用原排序选出的股票，以带约束的行业权重调节检验增量。预期行业收益、协方差、交易成本与偏离原组合的惩罚都要基于当时信息，按同一成交和费用重新回放。跨模型比较至少包括原权重、仅风险、仅均值与均值加风险。约束下的效用优化有成熟文献，估计误差足以吞掉理论收益，原组合应保留为配对基线。[Boyd 等](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html) · [DeMiguel、Garlappi 和 Uppal](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901)

本项目的 P6TAIL 行业贡献来自事后静态归因。已有结果能说明历史损益发生在哪些行业，周频事前可预测性仍需用冻结的选股、时间可用行业标签、真实持有窗口和扣费回放检验。[P6TAIL 归因](research_20261116_p6_tail_attribution_results.md) · [研究经验](research_lessons.md)
