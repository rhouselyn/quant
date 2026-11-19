# 金融建模研究 C：收益概率头与损失组织

最终集成规格与分歧裁决见[金融模型设计](research_20261113_finance_model_results.md)，实施范围见[待执行协议](research_20261113_finance_model_protocol.md)；本报告的独立备选不自动叠加到第一版。

日期/时区：2026-11-13，Asia/Shanghai。状态：complete（研究设计、源码核对、内存合成公式检查）；模型实现、训练、概率校准及交易收益验证均未执行。
父研究：[前轮损失研究](research_20261113_math_b_losses_results.md)、[共同事实](research_20261113_math_facts.md)。边界：[研究经验](research_lessons.md)、[记录规则](research_recording.md)、[标准执行协议](research_20261113_standard_execution.md)。本文是候选设计，不修改既有协议、配置、快照、队列或已发布证据。

## 1. 裁决：有限指数矩的 log 收益两尺度混合，排序只使用 log 均值

第一版采用**同中心、固定混合权重、固定尺度比的两个 Gaussian 的 log 收益混合分布**。保留共同因素与个股残差分解

\[
\mu_{it}=m_t+\delta_{g(i),t}+\alpha_{it},\qquad
Y_{it}=\log(1+R_{it}),\quad
Y_{it}\mid\mathcal F_t\sim .9\,N(\mu_{it},s_{it}^2)+.1\,N(\mu_{it},9s_{it}^2).
\]

这里的收益是既有入场开盘到约定退出开盘的未扣交易费资产收益。模型族内 \(\mu=E[Y\mid\mathcal F_t]\)，不是 \(E[R\mid\mathcal F_t]\)，更不是等权组合的预期 log 效用。分解在预测时可用集合 \(A_t\) 上按 A 方向的约束中心化；不能因某股票未来标签缺失而重新中心化 forward。标签缺失只影响监督哪个总 \(\mu_{it}\)，不改变各因子身份。

**单模型排序列 `score=mu_log`；尺度只输出为概率附加列，不进入默认排序。** 相同 \(\mu\) 的股票不因 \(s\) 较大得到更高分。模型分布可算出的 simple 均值只作诊断，不能无登记替换排序、选轮或持仓目标。可选择一个最终横截面 z 变换供仅要求排序的旧接口使用，但该列必须叫排序分数，原始 \(\mu\) 另存。

固定 \(\pi=(.9,.1),a=(1,3)\) 是**拟议工程默认值**，并非由本项目数据估计或已证最优。有限 Gaussian 混合有较高峰度和较宽尾部，渐近仍是 Gaussian 尾，不能称为幂律尾模型。固定两项只需一个尺度头，避免逐股学习混合概率、尺度比、自由度和多个位置导致识别困难。市场状态先进入现有均值结构及尺度线性项，不把这两个混合成分命名为已识别的“牛熊状态”。

收益提升假说是：在统一收益单位下估计共同信号和个股偏离，对高噪声样本降低其推动均值的强度，保留真实极端幅度，并为预测失准提供分布诊断。**数学闭合不证明均值已校准，也不证明排序或净收益提升。** 在模型错设下，混合族极大似然的位置参数是 KL 投影的参数，未必等于真实条件均值。

## 2. 实际实现事实与需改变的位置

源码 HEAD 为 `f2e198e8769ab960b7c8d6a8aa2ba50eb1bb8fe3`，工作树已有修改；下表为本轮读取的行号，不代表部署状态。读取的 B0 是 `outputs/dataset_v2/exploratory_F0/runs/research_B0_F0_s42_a1/config.yaml`，不是一份新训练配置。

| 事实 | 实际入口 | 对本设计的约束 |
| --- | --- | --- |
| 快照标签是 `log(exit/entry)` | `scripts/build_dataset_snapshot.py:91` | 不能把同一标签直接放进 simple-return NLL |
| v2 标签需 valid、成熟时间及训练/选择截止 | `src/quant_mvp/hierarchy_data.py:184–198` | 不扩大样本期，不用未来可观测性筛推理股票 |
| 原始 y 保留，缺失填零另有 valid_y | `hierarchy_data.py:228–231` | NLL 必须使用原 mask，填充零不是零收益观测 |
| 输出是一个自由线性 score | `src/quant_mvp/hierarchy.py:281,452–463,505–526` | 需成为有单位的均值 decoder；增加尺度输出 |
| 预测保存 score/realized_return 等 | `src/quant_mvp/hierarchy_engine.py:75–98` | 必须保存 mu、scale、分布版本，而非只保 score |
| 默认把 y clamp 到 ±.25 | `hierarchy_engine.py:598` | 概率臂要用经过质量审计的 raw_targets |
| 尾 BCE + soft RankIC + InfoNCE | `hierarchy_engine.py:648–665` | 新概率臂另开目标身份，不能静默续训旧 run |
| 行业分数 η²、总梯度裁剪 | `hierarchy_engine.py:685–695,752–759` | η²不属于收益似然；已有全局梯度阈值1要监控 |
| soft rank两侧均使用温度 | `src/quant_mvp/model.py:814–915` | 任意单位旧score与收益单位mu不能直接共用温度 |
| 按epoch z再均值、再跨seed z融合 | `scripts/eval_weekly_architecture.py:32–57` | 概率参数不得走这一融合函数 |
| 标准评估导入上述融合，simple回放使用expm1 | `scripts/evaluate_dataset_three_year.py:13,115` | 排序兼容与概率融合须显式区分 |
| 标准评分目前丢弃额外预测列 | `scripts/score_dataset_standard.py:81` | 只改predict不够，评分导出也要保留分布 |

B0 实读：`score_weight=1,rankic_weight=.5,temporal_weight=.1,industry_exposure_weight=1`；两种温度分别 `rankic_temperature=.05,score_temperature=1`；8轮、seed42、batch16、lookback40、embedding128，关系门零初始化且冻结。未指定 `raw_return_ranking`，默认使用上述clip。配置中的旧F0日期不代替已纠正的标准三年选择规则。本文仅阅读既有历史结果；2018–2020验证和2021+历史测试已暴露，不可据本报告把它们重新称为盲测。

本轮源码SHA256：`hierarchy.py=98b5dae2f295b57cec5568ba13b05cfb9798fa0a5791bac6501df09e2e955850`；`hierarchy_engine.py=b92e39aa3028a3bbce5c64b13f0fb77e3efb428613189d08ac6d843ea515764b`；`model.py=630e0ecb67f0db1ae926f82eb8465df578ffcfb508189f5d734988fe3223276a`。未重新验证全部旧checkpoint训练身份。

## 3. 分布定义、完整NLL与输出契约

令 B 为日期batch、N为当日对齐股票槽位、D=128。时序表示为 `[B,N,D]`，市场状态 M 为 `[B,6]`，行业路由为 `[B,N]` 或静态 `[N]`。均值、尺度、输入可用mask A、监督mask V 均为 `[B,N]`；两个成分参数广播到 `[B,N,2]`。本方案输出的是逐股边际分布，**不是已建模全部股票联合协方差的多元分布**。

设 \(e=y-\mu\)，\(s>0\)，\(a_1=1,a_2=3\)，则

\[
\ell(y;\mu,s)=-\operatorname{logsumexp}_{k=1,2}
\left[\log\pi_k-\log s-\log a_k-\tfrac12\log(2\pi)
-\frac{e^2}{2a_k^2s^2}\right].
\]

必须保留 `log(s)`、`log(a_k)`、Gaussian归一常数及混合权重。直接把各成分NLL加权平均不是混合NLL；前者不允许每个观测由不同尺度承担。PyTorch已有 `Categorical + Normal + MixtureSameFamily`，成分维位于最右batch维，可作为实现参考和数值对照；无需引入新依赖。[官方分布文档](https://docs.pytorch.org/docs/2.15/distributions.html#mixturesamefamily)

若内部用 \(x=y/s_{ref}\)、\(\bar\mu=\mu/s_{ref}\)、\(\bar s=s/s_{ref}\)，则原log收益单位密度的NLL是 \(\ell_Y=\ell_X+\log s_{ref}\)。这个常数虽不影响同一训练的梯度，但跨模型NLL比较和保存单位时不能丢掉。不能对每个日期重新 z 标签，那会删掉共同市场收益及真实风险尺度。

\[
E[Y]=\mu,\quad Var(Y)=1.8s^2,\quad
E[R]=\exp(\mu)\left[.9e^{s^2/2}+.1e^{9s^2/2}\right]-1.
\]

\[
E[(1+R)^q]=\exp(q\mu)\sum_k\pi_k\exp(q^2a_k^2s^2/2),
\quad Var(R)=E[(1+R)^2]-E[1+R]^2.
\]

有限参数下以上正指数矩有限。`expm1(mu)` 是此对称log混合下的simple中位数，不能标成simple期望。simple密度为 \(p_R(r)=p_Y(\log(1+r))/(1+r),r>-1\)；若按simple标签报告NLL，需加 Jacobian：\(\ell_R(r)=\ell_Y(\log(1+r))+\log(1+r)\)。同一比较必须固定密度单位。

CDF为 \(F_Y(y)=\sum_k\pi_k\Phi((y-\mu)/(a_ks))\)，分位数可对单调CDF二分，simple分位数为 `expm1(q_log)`。最小附加输出：`mu_log,scale_log,distribution_family,distribution_version`，常数π/a/s_ref随模型metadata保存；评估时派生NLL、CDF、分位数、simple均值即可，不必扩成几十个训练头。

有 \(R>-1\) 支持不代表满足A股涨跌停、停牌、可成交或退市清算机制。真正总损失 \(R=-1\) 是边界原子，对应 \(Y=-\infty\)，现有有限正价格log标签不能表达；若数据合同纳入该事件，须另登记边界质量/混合原子或删失机制，不能用任意大负数伪装，更不能静默删去。

## 4. 为何不选Student-t；其合法备选范围

以log收益拟合Student-t，完整NLL为

\[
\ell_t=\log s+\tfrac12\log(\nu\pi)+\log\Gamma(\nu/2)
-\log\Gamma((\nu+1)/2)
+\frac{\nu+1}{2}\log\left(1+\frac{(y-\mu)^2}{\nu s^2}\right).
\]

这是位置-尺度参数化，\(s\)不是标准差；\(\nu>2\)时方差为 \(\nu s^2/(\nu-2)\)。官方Student-t文档给出其规范化密度及参数定义。[PyMC StudentT](https://www.pymc.io/projects/docs/en/v5.6.0/api/distributions/generated/pymc.StudentT.html)

关键反证是其右尾 \(p_Y(y)\asymp y^{-\nu-1}\)，所以任何有限ν都有 \(\int e^y p_Y(y)dy=\infty\)。即使ν=100、log方差有限，也没有有限simple期望。`exp(mu)-1`只能是simple中位数。直接对simple收益使用未截断t则给 \(R<-1\) 正概率，不具备资产收益支持。该结论由密度尾部直接推导，并非“厚尾分布一般都不能转simple”。

如果后续只需稳健的log位置和分位数，固定ν=5的log-t是诚实可行的独立候选；不学习逐股ν，避免ν、s共同吸收异常且梯度不稳。它不输出有限simple均值，不接算术均值组合优化。规范化截断t也可行，但需先有事前支持区间，NLL加 \(\log[F(b)-F(a)]\)，均值和分位数重新按截断分布计算；不能把数据清洗的±.25阈值假装经济支持。因此第一版选择有限Gaussian混合，而非新增截断参数或自由ν。

## 5. 尺度、初始化与梯度约束

只用已成熟训练样本、按日期等权估计固定参考尺度：每日期先算log收益中位数及MAD，取 `s_ref=max(1e-4,median_t(1.4826*MAD_t))`；少于3个有效股票的日期不用于该参考估计。训练期数值、有效周数和样本哈希写入metadata。这里1.4826只是常用Gaussian参照转换，**不宣称它无偏估计混合的标准差**；s_ref只决定数值单位。

\[
u_{it}=b_\sigma+w_\sigma^Tz_{it}+v_\sigma^TM_t,
\quad s_{it}=s_{ref}\,[.01+\operatorname{softplus}(u_{it})].
\]

市场项先用独立6维线性权重，允许尺度读取同一M，但不强迫复用均值调制系数。初始化 `w_sigma=v_sigma=0`，`b_sigma=softplus_inverse(.99)`，使初始s=s_ref、总体log方差=1.8s_ref²；不是从 `softplus(0)=.693` 随意得到周收益尺度。尺度floor为 `.01*s_ref`。均值decoder在标准化单位输出，最终乘s_ref还原；市场截距可用训练期等周均值初始化，行业/alpha残差零初始化，避免自由score旧初始化造成几个单位的“周log收益”。旧排序checkpoint的score不能直接当mu使用或当新概率训练的完整恢复点。

设成分后验责任 \(\gamma_k=\pi_k\phi_k(y)/\sum_j\pi_j\phi_j(y)\)，有

\[
\frac{\partial\ell}{\partial\mu}=-(y-\mu)\sum_k\frac{\gamma_k}{a_k^2s^2},
\qquad
\frac{\partial\ell}{\partial\log s}=1-\frac{(y-\mu)^2}{s^2}\sum_k\frac{\gamma_k}{a_k^2}.
\]

宽尺度成分减少中等极端样本的均值梯度，但有限混合最终仍有线性增大的远尾位置梯度，不具备Student-t的红降影响函数。s增大又会减弱mu梯度，网络可能通过“难预测股票给大尺度”掩盖均值欠拟合；`log(s)`惩罚防止无代价无限膨胀，却不能保证优化落到好解。异方差NLL的这一优化风险已有原论文合成证据；其β-NLL是候选修正而非本项目必要结论。[Seitzer等，ICLR 2022](https://arxiv.org/abs/2203.09168)

第一版保留真实NLL，不同时加入β-NLL、额外均值MSE、尺度惩罚、学习ν和梯度手术。记录每轮尺度分位数/floor占比、不同s分组的均值误差、均值和尺度头及共享encoder的梯度范数、全局裁剪频率、NLL和CDF校准。NLL计算至少float32，合成核验用float64，采用logsumexp；非有限mu/s/NLL或梯度必须报错，不静默clamp损失。经审计合法极端样本引起数值失稳时中止候选并修改新协议，不能训练时自动删样本。是否需要均值预热或固定尺度对照由这些诊断触发，不能先堆多个补丁。

## 6. 原始标签、日期归约和去clip边界

定义 \(V_t=A_t\cap\{\text{标签有效且在训练截止前成熟}\}\)，则

\[
L_{NLL}=\frac1{|B_*|}\sum_{t\in B_*}\frac1{|V_t|}\sum_{i\in V_t}\ell(y_{it};\mu_{it},s_{it}),
\qquad B_*=\{t:|V_t|>0\}.
\]

每个交易日期权重相同，避免股票覆盖高的日期主导风险。NLL独立观测写法是**边际复合似然/预测评分**，不是声称同日股票独立；经济和统计区间仍按日期或时间块计算。全batch无有效监督跳过optimizer更新，不能让纯weight decay改参数；先按mask安全索引再算NLL，不用“NaN乘0”遮盖无效值。未来标签不可用不影响A和forward，预测保留全部当时可评分股票。

训练日历仍冻结为统一研究登记的779个训练锚；关闭未来5周InfoNCE后，也不据此释放原先保留的尾部日期或扩大成熟队列。s_ref和初始化统计仅来自这一固定训练日历及同一监督mask。proper log score不消除分布族错设或缺失标签的选择偏倚：本目标直接识别的是可观察标签子总体，未证明其与未来缺失股票同分布。

去clip是概率任务必要的数据语义变更，不等于“所有原始数字可信”。上线训练前先在训练段留审计表：原始正价格/证券身份、复权与权益口径、同一约定入退场、跨源复核、非有限/零负价格、断档以及超过旧±.25的数量与原因。合法的复牌跳跃、连续多日涨跌或企业事件须保留其幅度；纯因收益极大而删样本会删掉待预测的尾部。确证源错误更正来源并升快照版本；未决异常带原因隔离监督并报告覆盖，不能偷偷改变推理股票池。未完成这些审计，不可声称原始尾部已可训练。

对clip后的y仍套连续NLL会把边界累积质量错当成真实精确观测。如果只保有winsorized观测，左端贡献 `-log F(-.25)`、右端 `-log(1-F(.25))`、内部为密度NLL，这是删失观测模型；如果样本是按区间被筛掉，则是截断模型、需要除以区间概率，二者不同。既有数据已保raw y，首版不必引入它们。特征预处理clip与收益标签clip是两回事，本文不要求移除特征clip。

## 7. 损失只保留一个概率主目标；排序辅助另设单一候选

| 项目 | 概率首版决定 | 理由与限制 |
| --- | --- | --- |
| NLL | 系数1，主目标 | 给位置/尺度一致的概率定义；日期等权 |
| 尾BCE | 置0 | 旧条件尾部分类赔率不是收益均值，不能强制同一mu兼任自由logit |
| 旧soft RankIC | 置0；保留评估指标 | 原始温度作用于任意score，不直接适配收益单位 |
| InfoNCE | 置0，概率首版不带入 | 未来同股身份识别不直接监督收益；新NLL基点优先明确概率目标 |
| 行业η² | 置0 | 总mu应容纳有效行业收益，压掉它违背共同因素任务；alpha中心化不是总mu中性 |
| Sharpe/换手/尾部BCE/风险惩罚 | 不新增 | 首先验证概率与均值信号，组合约束留在执行层 |

旧B0四损失保持原样作历史基线。新概率臂同时改变了表示任务、损失与输出含义，**不能把它与B0的差值单独归因于去InfoNCE或厚尾分布**。前轮删除InfoNCE的负面/不确定记录限制了“删掉一定好”的论断，却不要求新概率目标永远背上全部旧损失。

若主目标NLL已稳定但排序弱，唯一登记的辅助候选使用同一个mu，而不增加自由score。可取每日期不同真实收益的无序有效股票对 \(P_t\)，令 \(d_{ij}=\operatorname{sign}(y_i-y_j)\)：

\[
L_{pair}=\operatorname{mean}_{t:|P_t|>0}
\operatorname{mean}_{(i,j)\in P_t}
\operatorname{softplus}\left(-d_{ij}\frac{\mu_i-\mu_j}{s_{ref}}\right),
\quad L=L_{NLL}+\lambda L_{pair}.
\]

等值对不参与，至少两个有效且收益不同的股票才有该日期；所有对不按收益幅度或尾部加权。预测差以训练期固定s_ref变成无量纲，标签用精确顺序，不把收益温度和任意score温度混合。不得用预测s当温度，否则尺度会成为排序损失逃生通道；不能对y按截面z后再拿其幅度当概率目标。pair的梯度只沿mu及其共享特征回传，尺度头没有直接pair梯度。

首版λ=0，完全冻结。辅助臂的系数在预先固定的8个训练batch上，用初始化模型的均值decoder参数梯度测量 \(G_N=median\|\nabla_{\theta_\mu}L_{NLL}\|\)、\(G_P=median\|\nabla_{\theta_\mu}L_{pair}\|\)，一次设置 \(\lambda=.1G_N/G_P\)，随后不动态调整、不看验证Sharpe搜系数。使用两项各自归约后的梯度，参数集合、batch键、初始化seed和数值写锁；若任一范数非有限或不超过1e−12（参数与输出已按s_ref规范化），该辅助预检失败，不能凭经验临时补系数。0.1是允许初始辅助梯度约占主梯度10%的工程预算，不是经济最优定理；实际训练中比例仍须记录。不同seed使用同一个预检锁系数，不各自重调。

该辅助损失一般会移动纯NLL的总体最优点，不能继续声称严格proper scoring的最优校准保证。它依靠同一mu使排序改善必须同时接受NLL检验；若概率校准明显变差，即使IC升高也不能声称概率预测整体改善。现有 `rank_ic_loss` 可保留旧路径；不需要为此重写所有旧调用方或引入排序库。

## 8. 跨epoch/seed融合：混合分布而非平均z后的参数

令已锁定成员j各输出 \(p_j(y)\)，沿用既有成员集合及冻结等权 \(w_j\)：

\[
p_{ens}(y)=\sum_jw_jp_j(y),\quad
\bar\mu=\sum_jw_j\mu_j,
\quad Var_{ens}(Y)=\sum_jw_j(1.8s_j^2+\mu_j^2)-\bar\mu^2.
\]

它是2J成分混合，NLL必须为 `-logsumexp(log w_j + log p_j(y))`，不是成员NLL的均值；CDF按权重相加，分位数解混合CDF，不能平均成员分位数。simple均值按成员simple均值加权，不可对平均mu/scale代公式；若一成员某股票缺参数，按当前严格mask契约报错，不在未知处随意重归一成员权重。

**概率主路径排序 `score=mu_log_ensemble`，先平均有单位的mu，再至多做一次最终日期z用于旧纯排序接口。** 旧B0维持其现有逐epoch z融合；若需隔离融合算法影响，预登记一份新模型的旧z-score排序副结果，名称明确为legacy排序融合，不能把它写成概率混合的均值。不得悄悄把标准融合函数的全局行为改掉。选择规则及近邻三轮成员不因概率输出自动改成按NLL选轮；需要换选模准则时另注册，不能从历史结果挑一个有利规则。

模型内尺度主要表达条件结果分散度，也包含错设、缺失特征和源噪声；不是模型认知不确定性的证明。成员均值差异可作为模型分歧诊断，`平均成员内方差 + 成员均值方差`是混合方差恒等式；同一seed邻轮强相关，少数seed并非严格后验抽样，分歧小不保证分布外可信。Deep Ensembles原论文支持其作为实用预测分布组合方法，但不提供本项目金融校准保证。[Lakshminarayanan等，NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)

## 9. 轻量检验、验收与失败边界

已完成：仅Python/NumPy内存构造混合密度，未导入模型、未读取行情、未训练。float64梯形积分网格为 `[mu−2,mu+2]`、200001点，参数mu=.01、s=.04；断言密度积分、均值、方差和simple矩公式误差均小于1e−9，命令退出码0。

| 检查 | 实得结果 |
| --- | --- |
| 密度积分 | 1.0000000000000002 |
| log均值 | .010000000000000002 |
| log方差 | .00288 = 1.8×.04² |
| simple期望数值积分与解析差 | 3.64e−17 |
| 峰度 | 8.33333333333333（Gaussian为3） |
| 股票A：mu=.01,s=.02 | simple均值 .010413967052781636 |
| 股票B：mu=.005,s=.1 | simple均值 .014172260013353588 |

末两行展示：B的log均值更低，却因尺度大而有更高模型simple均值。它并非数学错误，而是更换决策目标会改变排序；首版固定mu排序，避免把这种纯参数转换当作新增alpha。上述内存结果无单独机器产物，不是市场实证。

未来实现必须核验：混合NLL对官方分布log_prob一致；尺度floor/单位换算/梯度有限；空日期、全mask、单股、等值pair、NaN填充隔离；改变未来mask不改变forward因子；概率成分和成员融合积分归一及混合矩；保存加载和续训身份包含分布版本；旧B0推理不变；评分脚本不再丢弃参数。新测试只针对这些新契约，不镜像所有旧实现。

科学验收分两层：先报告训练外逐周NLL、CDF/PIT分布、50/80/95%区间覆盖与宽度、按市场状态/预测尺度/行业的校准、mu误差与RankIC；再按原账本费用、资本预算、L20/S10及压力条件考察净收益。区间覆盖提升但宽度无限膨胀不是成功；NLL改善、mu变差是均值被方差掩盖；全体校准良好但尾部/状态失准仍不能供风险决策。NLL不含行业共同冲击协方差，不能从逐股尺度相加得到可靠组合VaR或风险预算。严格尾部幂律、偏度和跳跃原子并非本首版能力；若失准显著，记录失败后重新立项，不自动增加更多混合成分。

更改面：`hierarchy.py`均值/尺度head与forward契约，`hierarchy_engine.py`目标准备/归约/预测/metadata，标准评分参数导出，概率融合新分支，配置与训练身份、存储与验证测试。标签质量修订若发生属于独立快照版本变更；本报告不授权改数据。无需新框架或通用概率抽象。

下一步是把上述选择合入统一预注册，先做训练段标签审计和无训练公式/契约测试，再决定唯一概率臂与必要对照；不在此文另排训练队列。本文没有新启动训练、回放或监控任务，也没有检查已有后台任务实时状态。

未做：真实训练期尾部质量/比例统计、真实梯度测量、训练、checkpoint概率推理、校准与市场回测。风险：有限混合错设、均值被尺度掩盖、历史测试已暴露及数据/成交代理偏差，均可能压过数学建模收益。
