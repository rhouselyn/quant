# 端到端主干的双端排序与多空互补：一手证据

日期与时区 2026-11-19，Asia/Shanghai。文献核验完成。本页记录可检验的设计推论，训练、回放和收益验收待独立登记。此前证据已覆盖 RankNet、平滑 TopK 及对侧弱周的基本定义。 [冻结基线两端精排](research_20261118_tail_rerank_evidence.md) · [多空篮子互补](research_20261118_complementary_books_evidence.md)

## 双端排序需要贴近实际截点

- **Vigna 加权 Kendall τ** 在成对一致度中引入排名权重，支持分数并列。加法双曲权重为 $w_{ij}=1/(1+\rho_i)+1/(1+\rho_j)$，$\rho=0$ 为最优名次。原文结尾讨论让 TopK 之外的权重归零，跨界对仍有权重。项目可在各腿可交易池按较大单股经济收益为好分别报告 τ，并明确权重来自标签秩还是预测秩。等权篮子内部换序会改变 τ，硬名单及同周篮子收益保持一致，因此同时报告截点的名单命中与实际收益。 [Vigna，§4、§7](https://arxiv.org/pdf/1404.3325)
- **LambdaLoss** 把当前预测榜单交换一对的指标变动作为 pairwise logistic 权重，文中式 (11) 用 $\Delta\mathrm{NDCG}$。项目可将该思想应用于等权硬篮子的单腿毛收益。令 $a_i=\mathbf 1[\rho_i^{\mathrm{pred}}\le K]$，预测秩从 1 起，同腿换位的收益改变量绝对值恰为 $|a_i-a_j|\,|u_i-u_j|/K$。多腿 $u_i=r_i$，空腿 $u_i=-r_i$，各腿按自身资格与股数计算。如果是带约束联合选择，只能把它解释为可行单腿交换的收益改变量。每次按当前排序算权重后对分数差反传，这是本项目的训练代理。其局部权重随硬名单变化，零权重的同侧对没有直接监督，极端收益和 pair 抽样也会改变训练重心。两腿成本或联合风险未进入这条权重。 [Wang 等，§4.3 式 (11)](https://marc.najork.org/papers/cikm2018.pdf)
- **可微秩** 有两条路线。Cuturi 等对运输计划作熵正则，用 Sinkhorn 秩构造 TopK 分类代理。Blondel 等直接在置换多面体上作投影，前向 $O(n\log n)$、空间 $O(n)$，命题 2 的无并列低温极限恢复硬秩。软秩适合给排序目标供梯度。把软秩逐股送进 sigmoid 并不能保证权重和恰等于 $K$，两腿分别这样做也不保证互斥。需要交易约束时应把约束加在选择层。 [Cuturi 等，§3–4](https://proceedings.neurips.cc/paper/2019/file/d8c24ca8f23c562a5600876ca2a550ce-Paper.pdf) · [Blondel 等，命题 2、§5](https://proceedings.mlr.press/v119/blondel20a/blondel20a.pdf)

## 硬名单扰动与组合尾损

**扰动优化器。** Berthet 等在预测目标系数上加 $\varepsilon Z$，重复求解原线性目标的离散问题，用期望平滑决策。设 $\mathcal Y_t$ 是决策日可行二值名单集合，令 $y(s_t)=\arg\max_{y\in\mathcal Y_t}\langle s_t,y\rangle$。两腿配额、各自资格和互斥若放入 $\mathcal Y_t$，每次求解都满足这些条件。平均入选向量保持线性约束，通常呈分数仓位。无扰动可复用同一硬求解器。原文的唯一最优解低温极限需满足其正则条件，固定股数导致的仿射约束可在仿射包内处理。温度、有限次求解噪声和近边界名单敏感性需要测量。 [Berthet 等，定义 2.1、命题 2.2–2.3、§3](https://proceedings.neurips.cc/paper_files/paper/2020/file/6bb56208f672af0dd65451f869fedfd9-Paper.pdf)

**先算硬名单损失。** 设 $\ell_t$ 是固定成熟标签下的名单损失。项目推论中，训练 $\mathbb E_Z[\ell_t(y(s_t+\varepsilon Z))]$ 与训练 $\ell_t(\mathbb E_Z[y(s_t+\varepsilon Z)])$ 是两个目标。以两个等概率硬名单的多腿收益 −2% 和 +2%、目标 $h=1\%$ 为例，先算缺口 $(h-L)_+$ 再平均得 1.5%，按平均权重算得 1%。凸缺口的 Jensen 差因此可能让软训练低估实际名单的亏损。用标准高斯平滑时，前一目标可从原文 §3 的积分分部推导出梯度 $\varepsilon^{-1}\mathbb E_Z[\ell_t(y(s_t+\varepsilon Z))Z]$，前提是损失对 $s_t$ 的依赖经过选择且满足可积条件。若损失含换手，应在连续回放中从实际前仓、当周硬名单和成交规则算费用。 [Berthet 等，§3](https://proceedings.neurips.cc/paper_files/paper/2020/file/6bb56208f672af0dd65451f869fedfd9-Paper.pdf)

**决策损失与尾部风险。** SPO 衡量预测所诱导的决策相对事后最优的目标差，SPO+ 是线性目标下的凸代理。每周固定可行域、实现收益线性计入目标时，可用硬名单毛收益遗憾衡量预测是否改善了名单。含路径状态的费用或非线性互补目标需重新核对 SPO+ 的假设。组合 CVaR 则对损失分布计量尾部，$\operatorname{CVaR}_\alpha(X)=\min_b[b+(1-\alpha)^{-1}\mathbb E[(X-b)_+]]$。经验分布在阈值有并列时，这一表示处理了应进入尾部的阈值概率质量，直接平均所有超过或达到阈值的周可能改变口径。 [Elmachtoub 与 Grigas，§3–4、§6.2](https://arxiv.org/pdf/1710.08005) · [Rockafellar 与 Uryasev，定理 10](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf)

## 组合互补的独立验收

设 $L_t,S_t$ 是同周已执行名单的单腿收益，空头获利记正。每周股票收益整体加一个正数后，横截面秩保持一致，多腿毛收益升、空腿毛收益降。若两腿预算各半，腿收益为 −2%／+1% 或 −0.5%／−0.5% 时，组合收益同为 −0.5%，双亏与正收益补偿却不同。逐周组合收益相同就会得到同一个组合 CVaR。项目上可同时列出 $\mathbb E[(h-L_t)_+\mid S_t^0<a]$ 与 $\mathbb E[(h-S_t)_+\mid L_t^0<a]$，其中 $h>0$ 为正收益目标，$a$ 为弱势阈值，$L_t^0,S_t^0$ 为共同父模型腿收益。另报事件样本数、单腿有符号收益、达目标比例、候选自身双亏率和全日历净收益。弱周标签只用于成熟训练监督及评价。固定事件能保持父模型与候选的比较样本相同，候选自身风险仍需另验。 [既有互补定义与事件筛选依据](research_20261118_complementary_books_evidence.md)

方法比较应沿用同一可交易池、实际硬名单股数、费用、切分与锁定规则；分别核对排序变化、硬名单换股和腿收益同步变化。现有冻结精排三种目标的实际名单一致，端到端获得有效换股及经济增量仍待验证。 [硬名单复核](research_20261119_complementary_books_review.md) · [研究经验](research_lessons.md)
