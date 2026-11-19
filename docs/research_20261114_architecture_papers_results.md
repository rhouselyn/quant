# B0纯网络架构：近期论文筛选与接入边界

日期：2026-11-14，Asia/Shanghai。状态：complete（文献与源码分析完成；候选均未实现、未训练、未证明收益提升）。父研究：原B0的预测改善；金融分解、概率分布、标签和损失属于另一条研究线。协议边界沿用[标准执行](research_20261113_standard_execution.md)，本页是候选分析，不是开训登记。遵循[研究经验](research_lessons.md)与[归档规则](research_recording.md)。

## 结论

优先研究 **FoX式动态遗忘偏置**，其次是 **DIFF V2式差分注意力**；两者改变B0的时间信息混合方式，能够保留40×30输入、排序头、关系门和既有目标。**ModernTCN**适合做结构不同的骨干对照，改动面和归因成本明显更大。**FiX**是已核实的2026论文，但应作为FoX有证据后的条件候选，暂不抢占首轮预算。

四者在本仓库没有检索到同名实现或实验登记。这里的“未发现”来自本轮对`src/`、`configs/`与`docs/`的相关符号检索，不是遍历全部历史权重后的不存在证明。没有任何候选已经证明能提高A股周频排序或扣费收益。论文的新颖性、语言模型loss下降与本项目预测改善是三件不同的事。

| 顺序 | 候选与可核实版本 | 本项目要检验的架构假设 | 证据与主要限制 |
| --- | --- | --- | --- |
| 1 | FoX，ICLR 2025 | 让每个注意力头按输入状态决定历史记忆衰减，而非仅靠QK内容匹配 | 正式论文；语言任务证据；40周短序列未必受益 |
| 2 | Differential Transformer，2024预印本/ICLR 2025；DIFF V2，2026-01-20作者发布 | 两张共享K/V的注意力图相减，改善混合表示的选择性 | V2是作者技术发布，页面仍说明正式报告待完成；不能称已有完整同行评审V2结果 |
| 3 | ModernTCN，ICLR 2024 | 时间、变量内特征、变量间信息分别混合，提供强时序归纳偏置 | 正式时序论文；全骨干替换，必须与已有C3区分 |
| 4 | FiX，ICML 2026 | 将FoX每头标量遗忘扩为逐值通道遗忘 | 正式论文与作者实现；数值及内核适配成本更高，不能只加sigmoid就称FiX |

来源：[FoX论文](https://arxiv.org/abs/2503.02130)、[DIFF论文](https://arxiv.org/abs/2410.05258)、[微软DIFF V2发布](https://huggingface.co/blog/microsoft/diff-attn-v2)、[ModernTCN论文](https://openreview.net/pdf?id=vpJMJerXHU)、[FiX论文登记](https://proceedings.mlr.press/v306/li26dl.html)。全部链接在本轮联网读取；“近期”覆盖2024—2026，不声称穷尽截至今日的所有新论文。

## 1. 实际B0：已经具备什么

核对对象是[standard_v1 catalog](../outputs/dataset_v2/standard_v1/catalog.json)登记的`research_B0_F0_s42_a1`，实际[运行配置](../outputs/dataset_v2/exploratory_F0/runs/research_B0_F0_s42_a1/config.yaml)为40周、30特征、d=128、4层4头、GELU FFN宽384、dropout=0.1。`configs/abl2000_base.yaml`可帮助理解谱系，但运行配置和当前锁才是本批身份。旧P64/CS×P64属于不同训练cohort，不能混成此B0的单因素排行榜，见[共同事实](research_20261113_math_facts.md)与[P64复核](research_20261113_p64_recheck_results.md)。

`AttentionEncoder`在[model.py](../src/quant_mvp/model.py)执行：

```text
40×30 → Linear(30,128)+LayerNorm
      → 4×[pre-LN → Q/K RMSNorm+RoPE → causal SDPA
            → sigmoid输出门 → 输出投影+残差
            → pre-LN → 128→384→128 GELU FFN+残差]
      → LayerNorm → 最后一周 → embedding Linear(128,128)
```

`HierarchicalRanker.encode()`在[hierarchy.py](../src/quant_mvp/hierarchy.py)以`normalize=False`调用时序编码器；评分支路随后经final LayerNorm、L2 normalize、线性score。B0的行业/市场关系模块虽然被构造，`relation_gate_init=0`且`freeze_relation_gates=true`使其在有限数值输入下不传递新增关系信息。已有CS/CB/ADN是另外的配置分支，不能把类名“Hierarchical”当成B0已经启用跨股交互。

已有RoPE、Q/K RMSNorm、attention输出门，不再作为新增建议；输出门抑制的是已聚合表示，FoX抑制的是各历史位置进入聚合的权重，两者不等价。时间塔与EMA teacher共用编码接口，teacher有5周输入与`return_sequence`等调用要求；任何换骨干方案必须同时保证这些路径有效，不能只改40周评分分支而悄悄改变辅助目标。

### 已做方向与本轮去重

| 方向 | 本地证据 | 本轮处理 |
| --- | --- | --- |
| SwiGLU | G256已有实现与实验，`128→256`双支路；[runbook §4.2](research_20261111_weekly_architecture_runbook.md)、[完整验证重扫](research_20261113_followup_execution_results.md) | 不把换GELU为SwiGLU当新点子；不据局部历史数字宣布稳定优胜 |
| 输入非线性 | P64保留线性捷径并加30→64→128残差；三种子旧cohort统一复检没有稳定整体优势 | 不重新命名为新tokenizer；精确结果见[P64复核](research_20261113_p64_recheck_results.md) |
| 局部卷积 | C3在首层前加零初始化k=3 depthwise残差；还有ConvAttentionEncoder | ModernTCN若只变成首层加卷积，就是重复C3；完整候选必须检验逐层解耦结构 |
| 倒置特征token | B1 FeatureTokenEncoder已有明显负面首种子结果，teacher长度适配亦改变任务难度 | 不重推iTransformer默认替换；旧结果不证明所有变量轴网络无效，见[台账§14](experiment_ledger.md) |
| 非因果、Mamba/SSM、时间压缩 | 已有相关实现和历史比较 | 不把LLM/时序榜单当理由重开；原任务只有40token，长上下文复杂度优势很弱 |
| Q0/Q2单锚点查询 | 已有存量权重；Q2三种子收益证据不足 | 有明确延迟目标时测效率；不是本轮新收益候选 |
| Attention Residuals | M4已做4层前置探针，未见预设深度稀释，因而关闭，见[探针报告](research_exec_20261110_report.md) | 这是前置条件不成立，不能写成AttnRes已完整训练失败；当前不优先 |
| 跨股末层attention | M2/M2b、CS及SELF等有登记 | 本页不重新讨论金融/横截面条件结构，以免混入另一研究线 |

## 2. 候选一：FoX式输入依赖遗忘偏置

论文的核心是每头每时点的标量遗忘门，给因果attention logit加上沿时间累加的负偏置；固定门退化为固定相对距离衰减。原文也研究了附带输出门、QK norm等组件的Pro版本。主证据来自语言模型，不能直接迁移其收益结论。[论文§3](https://arxiv.org/html/2503.02130v1)

本项目建议是**FoX机制迁移消融**，而非复现整套FoX-Pro：保留B0的RoPE、Q/K RMSNorm、输出门、FFN和score head，只改`_GatedSelfAttentionBlock.forward()`中SDPA之前的logit偏置。令第h头

```text
f[t,h] = sigmoid(W_f[h] · LN(x[t]) + b_f[h])
c[t,h] = sum_{k≤t} log(f[k,h])
D[t,s,h] = c[t,h] - c[s,h]，s≤t；s>t为 -∞
A[t,s,h] = softmax_s(Q[t,h]K[s,h]^T / sqrt(32) + D[t,s,h])
```

以上公式是原机制在B0维度上的代入。本地待检验假设：不同输入状态下，某些头应保留较长历史、某些头应更快衰减。它只改变序列内记忆，不要求增加市场特征、改变金融目标或改变训练样本时间权重；后者D4与本候选是不同轴。

**参数与公平性（解析计算，未运行模型）：** 每层`Linear(128,4)`含516参数，4层增加2,064。若将该臂FFN宽384减到382，每层减514，净增2、全塔净增8；可形成近乎等参的主对照。现有配置只给整数`ffn_mult`，382不是当前可用配置开关；这里是将来最小实现所需的显式宽度，不是假装已有支持。若不缩FFN，应同时报告原宽候选和略增容量基线，不能隐去容量差。

**必要对照：** B0；预先固定或每头可学习常数衰减；输入依赖衰减。常数衰减用于区分“偏爱近端”与“根据状态调整”，不是等参容量对照。建议后续正式协议只把动态臂和B0设为主要比较，静态臂作为机制对照，避免扩大筛选面。

**边界和否决条件：** 不能用`sigmoid(0)=0.5`作为看似中性的初始化，连续40步会极强遗忘；可按预注册的初始记忆尺度设置bias，不能看历史测试反复选半衰期。`D`必须把未来位置设为不可见；若使用显式浮点mask，合并因果mask后调用SDPA，不能凭旧调用假设后端仍走原快速路径。小参数量不代表没有显存或内核开销。门长期全饱和、动态门不胜静态衰减、或优势只来自单一年份时，都降低继续优先级。

## 3. 候选二：差分注意力，优先参考DIFF V2结构

DIFF V1正式论文提出两张softmax图相减。微软2026-01发布的V2改为共享K/V、增加Q分支，按token和head产生sigmoid系数，去掉差分输出后的逐头RMSNorm；该页给出LLM训练观察，但仍明确正式报告待后续完成。本轮没有找到并核实独立正式V2论文，因此证据等级低于已发表V1。[V1论文](https://arxiv.org/abs/2410.05258)、[V2作者说明与伪代码](https://huggingface.co/blog/microsoft/diff-attn-v2)

本项目迁移草案保留4个输出头、每头32维以及B0输出门：

```text
y1 = SDPA(RoPE(Q1), RoPE(K), V)
y2 = SDPA(RoPE(Q2), RoPE(K), V)
y  = y1 - sigmoid(Linear(128,4)(LN(x))) * y2
output = O(y * sigmoid(existing_gate(LN(x))))
```

这检验的是“显式差分路由是否比现有单图输出门更好”，不是声称普通attention绝对表达不了相减；原有输出投影已可组合不同头。差分图含负权重，不再是概率分布，不能把它直接称为可解释的历史贡献或收益分解。

**接入点：** `_GatedSelfAttentionBlock`的Q投影和SDPA输出之间；保持`AttentionEncoder`的末时点读出与`HierarchicalRanker`全部评分路径。Q1/Q2对应同一K/V头，不能将不对应的头随意切两半相减。保留B0的Q/K norm与输出门属于本地混合设计，需命名为`B0+DIFF-V2 mechanism`，不能把它当作者原版成绩的复现。

**解析参数账：** 新Q2 `128→128`增加16,512/层；lambda `128→4`增加516/层，总计17,028/层、68,112/四层。这里假设Q1/Q2共享现有Q-normalization参数；若另建Q2 norm必须另记参数。FFN由384缩至318减少66×257=16,962/层，留下66/层、264/四层净增。这个选择固定总容量但重新分配FFN容量，不能声称单独识别了差分算子的贡献。额外同预算普通双查询对照才可进一步识别差分约束；它应在首筛出现信号后再登记，而非立即扩成架构网格。

**计算限制：** 两个attention乘法链会增加算量；共享K/V不等于免费。作者关于LLM解码吞吐的观察不能用来预测本机短序列、整批股票训练。必须计入teacher、checkpoint重算和每周评分的实际时间。初始lambda、两分支dropout及小输出范数可能影响优化；最小验收应验证有限梯度、分支头对应关系、5/40周形状和全序列返回，不需要引入作者自定义CUDA内核作为首个实现步骤。

## 4. 候选三：ModernTCN的逐层解耦骨干

ModernTCN以大核depthwise时间卷积、变量内通道混合ConvFFN1、跨变量ConvFFN2分开处理三个轴；作者源码给出分组pointwise卷积实现。本页只借鉴结构，不迁移论文benchmark排名。[原论文](https://openreview.net/pdf?id=vpJMJerXHU)、[作者代码](https://github.com/luodhhh/ModernTCN/blob/main/ModernTCN-short-term/models/ModernTCN.py)

**与本地已有方案的差异：** C3是30→128投影后仅做一次k=3卷积，之后保留Transformer；ModernTCN则持续保留“输入变量×潜通道×时间”三维表示并逐层分别混合。本地B1把整条40周路径先线性压为一个变量token，ModernTCN仍在每层保存时间轴，因此不是B1改名。但它同时改动token化、混合器和归纳偏置，对照解释更困难。

**可讨论的最小完整接入：** 在`TemporalTower`增加独立encoder选择，输入仍为`[batch,40,30]`；每个原特征映射为D个隐通道，四层保持周长度，最后一周的30D维表示压回128，再复用B0的embedding和打分链。为控制变量，先不使用重采样、不增输入窗口、不加预训练，也不加入RevIN等新归一化。作者通常的patch化、非因果padding和归一化若改为本地形式，必须记录为“ModernTCN启发的因果变体”，不能写作完整原论文复现。

**容量估算：** 若F=30、潜通道D=16、FFN扩张r=4、时间核k=7，则每块主要权重约为

```text
F D k + 2 r F D² + 2 r D F²
= 3,360 + 61,440 + 115,200 = 180,000
```

与B0每块主要权重180,224接近；但其480维内部激活明显大于B0的128，末端480→128还有61,440权重，所以这不是已经等参的完整模型。正式落地必须把stem、readout、bias、norm、重参数化分支一起计入，再固定D/r或头宽；不能只比较block。此解析示例用于证明存在合理容量量级，不是新配置或调参授权。4个k=7因果卷积的局部感受野仅25周，尚未覆盖40周；若改k=11使理论覆盖41周，则每块DW权重为5,280，需要同步重算和注册。理论感受野不等于有效使用历史。

**为什么排第三：** 可能提供不同归纳偏置，但完整实现和teacher适配的工作量最大；一旦只取“卷积”而丢掉变量/通道解耦，便会退回已做的C3。输入变量中已有ret/mom/vol多尺度统计，也可能降低新增时间卷积的边际价值。这是本地推断，不是论文结论。

## 5. 候选四：FiX的逐值通道遗忘，保留为条件方向

FiX的官方ICML 2026/PMLR条目已核实：作者利用输出RMSNorm之后的尺度性质，使逐元素遗忘可通过值侧重写实现，并设计其与QK侧RoPE互补。作者实现说明遗忘张量需float32，支持16/32/64/128维head，并提供Triton/FLA实现。[正式论文条目](https://proceedings.mlr.press/v306/li26dl.html)、[作者仓库](https://github.com/AlayaDB-AI/FiX)

**本地可能价值：** FoX同一头内所有值维度共享遗忘速度；FiX允许每个通道不同。若FoX先显示动态遗忘确有信息价值、而每头标量是明显限制，才有理由扩展。不能因为年份更新就先采用。

**接入与公平对照：** 接在SDPA值聚合处，输出接回原o_proj和残差。需要原文公式、输出norm和数值稳定实现整体核对，不能把现有post-SDPA sigmoid输出门当成FiX。对照至少包括B0、仅新增相同输出norm的B0、FoX和FiX；否则无法区分输出归一化与细粒度遗忘。一个直接`128→128`遗忘投影已是16,512参数/层，与FoX516/层不同，必须削减别处或给同容量对照。完整参数仍取决于采用的门参数化，本页不冒充已定稿。

**优先级限制：** 现有B0仅Q/K有RMSNorm，没有attention输出后的逐头RMSNorm；FiX依赖的代数条件并非当前网络天然满足。epsilon、有限精度和dropout下，尺度抵消不能当作无条件数值恒等。作者代码依赖也未在本机安装或benchmark，32维head形状可支持不等于Windows训练链路可直接运行。先确认FoX机制价值，再决定是否付出这一实现成本。

## 6. 统一的参数、算力与评价纪律

以下是本轮解析估算，不是性能测量。B0单块的主要矩阵权重为`5d²+2dm`：Q/K/V/O/输出门共5个d×d，FFN两层d×m/m×d；d=128、m=384时为180,224。40周单股票单层主要MAC约：

```text
投影与FFN：40 × 180,224 = 7,208,960
QK及AV：2 × 40² × 128 = 409,600
合计：7,618,560 MAC
attention二次项占约5.38%
```

忽略bias、norm、softmax、门、反向与融合内核，因此不能直接换算墙钟时间。它说明本任务不是典型长上下文瓶颈：将attention二次项降为线性，也不会凭复杂度式子获得数量级训练加速。等参数、等MAC、等更新次数、等墙钟是不同比较口径，不能同时宣称“完全公平”。

后续若用户批准实现/训练，建议正式协议冻结：

1. **主比较固定输入与目标。** 同一Q1快照、相同available/valid_y、预处理、训练日期、初始化共享权重、采样次序、优化器和8轮更新预算；保持B0四项loss。不同骨干都改变EMA目标的表示，须记录这一耦合，不得私下把某臂InfoNCE关闭。
2. **同时报总参数和实际参与预测/训练的参数。** B0存在冻结零门后的关系模块，只用全模型总参数易掩盖有效计算图差异；共享模块保持原样，重点对齐时间塔与评分支路。先报完整参数清单，再报解析MAC、显存和实测全流程时间。
3. **首筛只开FoX与配对B0，机制对照预先登记。** 本页没有排队或发起训练。若进入下一步，先做形状/梯度/因果性/teacher/状态恢复验收，再按现有三年验证规则选轮；不因方法更深就自动延长预算。
4. **验证成功再补42/43/44配对。** B0 seed42只有在数据和源码等价性审计通过时才复用；旧cohort种子不能补进当前配对。单seed有利不是确认。
5. **预测指标与交易指标并列。** 精确RankIC、分腿排序稳定性、尾部分类/换手、扣费多头/空头/等资本组合都报告；损失下降不能替代收益结论。固定同轮机制比较与独立验证选轮的整个系统比较分开。
6. **历史测试已暴露。** 沿用2018–2020完整验证、各腿独立选轮及win3；2021–2026已读历史只能描述复检。测试不参与决定门半衰期、核宽、FFN宽或候选晋级。不能把增加seed重新命名为新独立测试。

当前建议的停止条件是：FoX不能在配对验证中给出稳定、非单年驱动的预测/扣费方向，且动态门不优于固定衰减，则不自动扩FiX或与DIFF联训；保留负结果即可。具体数值门槛、预算和多重比较处理须在执行协议登记，本页不临时伪造一套新验收标准。

## 7. 证据身份、完成项与未完成项

HEAD：`f2e198e8769ab960b7c8d6a8aa2ba50eb1bb8fe3`。工作树已有修改，HEAD不代表本轮所读源码；本轮读取时SHA256如下：

| 文件 | SHA256 |
| --- | --- |
| `src/quant_mvp/model.py` | `2432b3e00eef6ae4428756c8edef6b537470e601d2f463f6c906edcb679f8aa5` |
| `src/quant_mvp/hierarchy.py` | `bf4b4cde9091d7bf8749dbc6d586041528fd5f6c0218257176466bf798e0757a` |
| B0运行`config.yaml` | `bd57708c8ae8d950f290be47ff1129483fba910434ae6153d4aac588f81c71a0` |
| `standard_v1/catalog.json` | `a02dd8b8956dbad9812f39e6053c29893067c81903498454fd7066d1d2237cd7` |

本轮完成源码/配置/catalog和历史实验相关段落阅读、主来源网页核实、解析参数与MAC推导、研究报告。未改实现、未安装依赖、未训练、未执行真实forward、未生成预测或回测，未更改任何队列。未核验其他会话实时进程，不对其是否在跑作判断。旧结果只用于去重和限定适用范围，未读取逐行历史收益据此调参。

DIFF V2仓库子目录网页抓取遇robots限制，使用作者发布页的完整机制说明及伪代码，未宣称已审计其完整实现。FiX PDF原始下载链接工具未能解析，正式PMLR摘要、会议元信息与作者README已核实，复杂实现细节保留为待读完整论文的门槛。ModernTCN作者代码可读，但未安装或运行。`outputs/`证据仅本机、未在本轮外部备份；本文是版本化研究记录，不是外部原始产物归档。
