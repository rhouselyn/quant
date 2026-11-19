# 优化器与激活候选：一手证据及本地迁移边界

日期/时区：2026-11-18 Asia/Shanghai。状态：evidence complete；本文只做文献核验和实验建议，不是已冻结执行协议，也不声称已训练。

父对象：当前 B0、P4、P6 完整主干；已读 [研究经验](research_lessons.md)、[记录规则](research_recording.md)、[P4全池pair协议](research_20261118_p4_global_pair_protocol.md)、[P6全池pair协议](research_20261118_p6_global_pair_protocol.md)。本次建议优化器/激活独立于 GLOBAL_PAIR，不能把两个变化混成一臂。报告由受委托研究 agent 编写，仅此文件有写入权限。

## 建议

Muon 有足够的一手实现和语言/视觉实验支持，适合成为候选；没有证据证明能提升本项目 128 宽、8 epoch、金融周频收益。固定训练步数下比较质量，同时记录实际耗时；不要把 LLM 的计算效率结论直接写成金融收益预期。

激活建议用零参数 SquaredReLU 作为较新候选，SiLU 作为保守可选项。StarReLU 也合理，但要如实登记每个激活模块新增两个标量。若资源只够一个激活候选，预登记 SquaredReLU，避免把名字相似的实现混用；若更重视论文中直接胜过 GELU 的视觉证据，可预先换成 StarReLU，但不能看历史收益后改。

## Muon：采用哪个算法

原始作者将 Muon 用于隐藏层矩阵，并建议输入、输出、标量、向量交给 AdamW。其 Newton–Schulz 五次迭代用 bfloat16，系数为 `(3.4445, -4.7750, 2.0315)`，先按 Frobenius 范数与 `1e-7` 归一化，长矩阵转置后计算以降低成本。[作者说明](https://kellerjordan.github.io/posts/muon/)

作者当前源码的默认 momentum 为 .95，使用 EMA momentum 与 Nesterov 混合后再做 NS5；矩阵更新使用原始形状缩放 `sqrt(max(1, rows/cols))`。源码明确有限步的结果不是精确极分解 `UVᵀ`。这些是版本相关事实，正式实现应冻结源码或记录引用版本。[作者源码](https://raw.githubusercontent.com/KellerJordan/Muon/master/muon.py)

Moonshot 的 2025 论文采用 `0.2*sqrt(max(rows,cols))` 匹配更新 RMS，并加入解耦 weight decay；其 LLM 实验据此复用调好的 AdamW 学习率和衰减，momentum=.95、NS=5。因此本项目推荐明确叫 **Muon RMS匹配变体 + AdamW**，不混用原作者 `.02` 学习率和 Kimi 缩放。[论文 §2.2](https://arxiv.org/html/2502.16982v1#S2.SS2)

这不是“所有矩阵都换 Muon”：嵌入即使二维也应保留 AdamW，输出预测/概率头建议保留 AdamW；显式列出输入投影和所有最终读出头。隐藏 `Linear.weight`、注意力投影矩阵归 Muon；bias、LayerNorm/缩放标量、向量与其它不合规则张量归 AdamW。冻结前保存每个参数的名称、shape、元素数、路由、lr 和 decay，并断言所有可训练参数恰好覆盖一次。此处是本地迁移设计，不是论文保证。

### 本地实现和恢复约束

本仓库 `src/quant_mvp/hierarchy_engine.py` 当前在单一 `torch.optim.AdamW(groups,...)` 下训练。`low_names` 组、冻结后第二组学习率乘 .1、`grad=None` 时跳过 decay、优化器 state 保存/恢复都必须继续生效。不能只替换构造器而忽略后续参数组索引语义。

- 本地若确为 torch 2.5.1 且无 `torch.optim.Muon`，不为此升级运行环境；使用小型 `Optimizer` 实现或现有可兼容的本地类，参数组携带路由标志。当前 PyTorch main 文档已经提供 Muon，不能据此假定已安装版本有该 API。[官方文档](https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html)
- 必须保留本项目 `grad is None: continue`。作者当前 SingleDeviceMuon 为同步而补零，不宜原样搬入冻结层流程；否则旧 momentum 和 decay 可继续改动本应不更新的参数。
- 采用非原地构造 Nesterov 更新，避免作者 `grad.lerp_` 修改 `.grad` 影响后续诊断。固定 momentum=.95、Nesterov=true、NS=5、eps=1e-7；本轮不再扩这些超参数。
- CUDA NS 运算按作者 bfloat16 路径；CPU 自检建议 float32，并显式记录设备/dtype 差异，不能要求 CPU/CUDA 逐位一致。本地 CPU bfloat16 性能与算子可用性须实测，不能从 GPU 代码推断。
- 解耦 decay 使用基础 lr：`p *= 1-lr*wd`，形状系数乘正交更新项；若把系数直接乘组 lr，务必避免同时放大 decay。保留 AdamW fallback 原 betas、eps 与原组 lr 规则。
- 状态至少包含 Muon momentum、AdamW 一二阶矩和各自 step、参数组路由及所有已有恢复字段。中断恢复不能回退成只加载权重。

这些约束须在实施时检查：零梯度/零矩阵更新有限；方形、宽、长矩阵输出 shape/dtype 正确；`grad=None` 参数及状态不变；路由无重叠遗漏；含两个路由组的连续训练与保存恢复一致。NS5 测试不应断言精确正交；另用一小步参考公式检查 shape factor、Nesterov、decay。

## 激活：同形状替换和门控网络分开

| 候选 | 公式及参数变化 | 一手依据和局限 |
| --- | --- | --- |
| SiLU | `x*sigmoid(x)`；零新增参数 | Swish 的 beta=1 形式；早期视觉实验支持可替换性，不是 2025/2026 新发明。[作者论文](https://arxiv.org/abs/1710.05941) |
| SquaredReLU | `relu(x)^2`；零新增参数 | MetaFormer 作者提供直接实现；较 GELU 简单，但平方放大大正输入，本地须检查梯度/损失是否有限。[作者代码](https://raw.githubusercontent.com/sail-sg/metaformer/main/metaformer_baselines.py) |
| StarReLU | `s*relu(x)^2+b`；每模块两个可学习标量，源码默认 s=1、b=0 | MetaFormer 是 2022 初版、2023 修订的视觉研究；其消融中 StarReLU 优于 GELU，SquaredReLU 在部分模型低于 GELU。因此不能把 SquaredReLU 等同于 StarReLU 的收益证据。[作者论文 §II-D/III-D](https://arxiv.org/html/2210.13452v3) |
| SwiGLU | `SiLU(xW)*xV` 后再输出投影 | 增加门控投影；若不缩小隐藏宽度会增加参数/FLOPs，不能叫“同形状 elementwise 激活替换”。需另开匹配容量实验，本轮不必加入。[作者论文](https://arxiv.org/abs/2002.05202) |

`hierarchy.py` 多处显式 `nn.GELU()`，包括 FFN、市场投影、projector、残差头等。首轮建议事先固定仅替换共享编码器 FFN 的非线性，保持概率正尺度/分布变换、GRU 内部门、最终头结构和 attention softmax 不变。若选择“所有显式 GELU”，则应列清单并对 B0/P4/P6 统一，不可不同父模型替换不同部位后宣称纯激活差异。新增激活无参数不代表优化几何不变。

## 最小矩阵与公平预算

主 agent 2026-11-18 已采用的首轮范围：**B0、P4两个父模型，每父 AdamW/Muon × GELU/SiLU/SquaredReLU 六格**；各复用通过一致性核查的 AdamW+GELU 父对照，因此每父五个新臂，共十个新臂。Muon 为 `.2*sqrt(max)` RMS匹配，lr继承父 `.0008`，仅称固定配方初试；8轮seed42，不加GLOBAL_PAIR，P6首轮不执行。Muon仅路由 temporal 内隐藏 `nn.Linear.weight` 白名单，排除input/output/financialheads/embeddings，`grad=None` skip。此方案有充分的变量隔离，无严重文献或算法异议；不要据结果再扩首轮队列。

下表为父模型覆盖更全时的研究设计参考，**不是上述首轮实际队列**；训练、数据、验证、历史边界应复制选定父配置、manifest 与选轮协议，不从文档日期推断真实样本。

| 范围 | 最小固定矩阵 | 用途 |
| --- | --- | --- |
| B0 | AdamW/Muon × GELU/SquaredReLU，共4臂 | 分别识别优化器、激活及二者交互 |
| P4 | 同样4臂；不额外加 GLOBAL_PAIR | 检验概率纯均值父目标上的迁移 |
| P6 | 同样4臂；保留其原组内pair | 检验已有组内排序父目标上的迁移 |

完整最小矩阵为12臂，包含各父的匹配基线；已存在基线仅在数据、源码、随机序列、预算与评分完全可核对时复用。可选 SiLU 扩为三种激活，总18臂，但必须在历史评分前决定是否执行。不同父模型不能用 B0 的改善替代 P4/P6 各自的对照。

lr 有两种诚实口径：

1. **固定配方迁移**：RMS匹配 Muon 以父 AdamW lr 为预登记起点，12臂均不调 lr；如实称“该超参数配方比较”，不宣称每个优化器已调到最优。数值稳定烟测只判断有限/可恢复，不凭短训收益选 lr。
2. **同验证搜索预算**：每个父、每个优化器、每种激活都给相同 `0.5×/1×/2×` 父 lr 候选及8轮预算，然后按同一完整153验证周规则选择。候选数增为36；若加SiLU则54。先登记候选选择和每臂选轮规则，再看测试。不能 Muon 搜三档而 AdamW 只算现成一档，最后称纯优化器胜负。

若今夜预算有限，优先完整12臂固定配方，明确探索性；不为某个末轮表现继续加 epoch。论文建议的 lr 迁移只能提供起点，金融小模型不同 loss、batch 和8轮步数仍需本地证据。原始形状缩放版本若另做，则其 lr 不可直接和 AdamW 同值解释。

seed42、8轮、相同训练步数/有效样本/增强序列、同153周验证 best±1 三轮锁、同297周已读历史复检和10/30bp费用等应沿父批次冻结；先核实际配置与快照。每个系统独立验证选轮，比较的是“训练+选择”系统。若使用验证选择最优组合，须保存全部臂结果并披露多重选择；历史已读不能重新成为独立确认测试。

结果并列报逐腿及同资本组合的净收益、Sharpe、含初始1的MDD、换手、费用、逐年读数与8周块配对区间；P4/P6另报NLL、RMSE、覆盖与RankIC。记录训练loss/梯度、每步/总耗时、峰值显存和更新RMS，验证优化稳定性及真实资源开销。固定8轮优于基线不代表相同墙钟预算或收敛后也优。

## 未执行与限制

本研究 agent 未改代码、未提交队列、未占用GPU，也未核验当前进程或声称任一任务正在运行。没有升级依赖，未做SwiGLU、Shampoo或更多新优化器；增加复杂依赖和容量变化不服务于首轮因果区分。环境版本、参数路由、实际日期和源码哈希由执行主 agent 冻结后补入协议。所引GitHub main/master为核验时内容，正式执行仍应固定引用版本。本文件仅本机、未异地备份。
