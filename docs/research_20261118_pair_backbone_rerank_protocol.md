# 随机pair主干接全池向量裁判的全比较和瑞士轮

执行状态更新为complete。两前缀、四新裁判头与所有回放已完成，见[逐年结果与验收](research_20261118_pair_backbone_rerank_results.md)。以下保留事前登记。

2026-11-18 Asia/Shanghai，planned，用户授权执行并给逐年净收益、MDD、Sharpe、换手。父研究为backbone_pair_v1（ADD .1保留softIC、REPLACE .5关闭softIC）与vector_pair_full_v1。范围为这两个B0派生主干，原B0是共同对照。冻结主干之后训练额外裁判，不反传改变主干。

## 样本外和比较对象

为避免2018截止主干对自身训练期提取表示造成二阶段样本内偏差，也避免不同折坐标混接，沿原全池向量研究固定2010编码器做法。分别从ADD/REPLACE父cfg重新训练fit_cutoff=2010-01-01的8轮seed42前缀，原BQ_F0_r2、5周辅助预留、train_end由真实日历确定，固定e8不看未来选轮。各父的训练/验证/历史全部使用各自同一个2010主干、冻结前缀预处理与真实forward的z128，排除标签影响forward输入。原B0前缀/vector缓存和原vector_full头直接复用并验hash。两新前缀需要额外实训，不能把旧2017表示称OOF。

精排训练日期复用tail_rerank_v1的410个2010–2017事前决策周，2018前成熟标签408周；完整2018–2020验证153周，2021-01-08起297历史周。固定2010表示的原头direct2010是同编码器直接排序必要对照；上一批2018截止完整ADD/REPLACE direct2017另列系统参考，不能把跨年代差全归精排效果。原B0 full/swiss使用同一旧vector裁判，原B0 full应复现其vector_pair_full_v1账本；此前raw特征候选池裁判不是向量裁判，不能混用。

## 同一裁判与赛制

裁判沿原vector_pair_full的Judge，每股128→32 GELU，concat64→16 GELU→1，ell(i,j)=(Fij-Fji)/2，末层零，无baseline分数/秩/边界输入或回退。两腿独立，所有有效标签pair按收益差min(abs(dRi-dRj)/.03,3)加权逻辑损失，同收益排除，日期等权；chunk8192累积同周梯度后一次AdamW更新。lr.001、wd.01、seed42、12轮。原主干训练的随机pair使用单股score差且不做幅度权重，故本裁判是额外比较函数，不能称训练推理已完全同一。

新两父各自按全比较路径完整验证的L20/S10 10bp净Sharpe选最佳单轮，平局早优；同一个选中裁判用于full和swiss6，不按瑞士结果另选。两新父的头全部锁定后才导出新历史向量/排序。原B0旧头沿已锁5/9轮，不重选。

full对所有事前可用股票两两比较，用有向logit行均值聚合。swiss6第一轮symbol稳定排序后seed42随机相邻配，后五轮按当前综合分接近配且优先未交手，奇数轮空优先避重复；重复边不新增证据。只查询本轮对阵，不能先算全图。已比较边综合分为argmin w*sum[(u_i-u_j)-ell_ij]^2+.01*sum[u_i^2]，w=n/(2|E|)，采用稀疏解。全图与rowmean仅差正比例，排名一致。空/单股/空图明确处理，tie按symbol。

## 投资与评价

全可用股票池，多侧不作Top100预筛；空侧按原全截面成交额前40%及历史资格，在LS50排除最终多头，纯S10无多头排除。训练空资格沿原OOF明确流动性代理，不按未来label筛候选。原成交/现金/缺失估值代理沿用。L20/S10各自全仓，LS50各半名义预算，10/30bp单边成本对应8.35%/12%年借券。CAGR按真实持有日历，年度按决策年累计净收益，含初始1年内回撤，Sharpe按周，换手为连续持仓双边变动周均。2026最后决策10月23日退出11月2日，非全年。

主要对照full/swiss减同父direct2010、新完整direct2017，以及相同赛制原B0向量精排；保留原B0 direct2017。各对照同153/297周、资本/费用，不按历史选择赛制或混接多空。给8周块配对区间、多空腿、全期与逐年四指标、比较次数。单seed和已读历史探索限制继承。

## 运行与验收

scripts/prepare_pair_backbone_vectors.py训练两前缀并导出train/validation，锁后export-test；scripts/replay_pair_backbone_rerank.py训练头与评分回放。输出outputs/pair_backbone_rerank_v1。源码、cfg、snapshot、preprocessing、vector/checkpoint/锁hash冻结，base_dev.yaml在训练前提供。获取共享GPU锁串行两前缀及裁判，不打断其他任务。每epoch保留可核验权重/优化器/随机状态，已锁头不能因回放故障重选。

验收包括实际2010 cutoff/辅助任务成熟、固定坐标及z重构score、缓存身份、mask/缺失标签、antisym/梯度、chunk梯度等价、full与稀疏图排名一致、Swiss置换一致/6轮计数/不使用未查询边、各对照账本对齐、净收益费用分解和逐年重算。固定幸存池/回溯行业/复权和执行券源代理风险继承。预计新增前缀约12分钟、向量与头/回放另约15–40分钟，以吞吐更新。无自动扩实验或部署，仅本机未异地备份。
