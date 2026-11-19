# 股票表示与横截面条件：低秩双线性、adaLN 与 Set Transformer

2026-11-13。范围：原文与代码核查；未修改模型、实验队列，未启动训练。

**直接用 adaLN/FiLM 是合理的优先候选。低秩双线性有文献与数学依据，但没有依据断言它比 adaLN 更适合当前选股任务。两者都能表达“同一股票特征在不同市场/行业状态下有不同作用”。先固定上下文来源，只比较注入方式；Set Transformer 留作上下文提取方式的独立问题。**

## 1. 文献确实支持什么

- FiLM 定义的是由条件产生逐特征仿射变换；不要求一定带 LayerNorm。adaLN 可理解为在 LayerNorm 后施加这种条件仿射。FiLM 的原始实证是视觉推理。[FiLM 原文](https://arxiv.org/abs/1709.07871)
- DiT 的 adaLN-Zero 在其 ImageNet 扩散模型对比中优于普通 adaLN、cross-attention 与 in-context conditioning；这支持其作为成熟条件注入结构，不是金融排序收益证据。[DiT §3.2、§5](https://arxiv.org/html/2212.09748v2)
- 投影后 Hadamard 乘积实现低秩双线性池化已有 ICLR 2017 原文，实证是视觉问答。借到股票与市场/行业摘要上属于结构迁移。[Hadamard Product for Low-rank Bilinear Pooling](https://arxiv.org/abs/1610.04325)
- Set Transformer 用注意力建模集合元素交互；固定 inducing-point 数量时，ISAB 对集合大小的复杂度为线性。它解决“如何从集合提取/交互信息”，不直接取代条件调制。[Set Transformer 原文](https://proceedings.mlr.press/v97/lee19d.html)

## 2. 二者的关系：以下是公式推导，不是论文的金融结论

令股票表示为 `h`，可用市场/行业摘要拼接为 `c`，`u = LN(h)`。

逐维条件调制：

```text
y = (1 + gamma(c)) ⊙ u + beta(c)
```

若 `gamma(c)=Ac`，乘法项第 j 维就是 `u_j * Σ_k A_jk c_k`，已经包含股票×条件的双线性交互。使用 MLP 产生 gamma 则成为非线性条件调制。不能说 adaLN 只有加法、无法表达交互。

低秩双线性残差：

```text
y = h + W[(U h) ⊙ (V c)]
Δy_j = Σ_r W_jr (Σ_a U_ra h_a)(Σ_b V_rb c_b)
```

对应三阶交互张量 `T_jab = Σ_r W_jr U_ra V_rb`，CP 秩至多为隐维 `r`。它在学习到的投影方向上调制，并能混合股票通道；FiLM 在当前表示坐标上逐通道调制。低秩限制与对角结构是不同归纳偏置，有限宽度下不能直接宣布前者严格包含后者。若 U、V 前后再加非线性，应称“低秩乘法融合”，不再是对原始 h、c 严格双线性的函数。

端到端训练还允许时间塔改变表示基底，因此“FiLM 对角”并不等于整网无法学跨特征关系。是否需要额外 U/W，必须由本地受控比较回答。

更具体地，若暂时没有末端 L2，scale-only 调制接线性评分为 `wᵀ[(1+Γc)⊙u] = wᵀu + uᵀdiag(w)Γc`。当 w 各维非零、Γ 不受限时，后项可表达一般标量双线性形式；若 `Γ=AB`，其交互矩阵秩至多 r。因此不能把“表示层逐维调制”误读成“评分函数表达力必然弱于双线性”。本项目保留 L2 后还存在随 u、c 变化的分母，不能再宣布它与加法双线性评分精确等价。

## 3. DiT 的零初始化不能照名字猜

官方块为 `h + gate(c) * F((1+scale(c))*LN(h)+shift(c))`，注意力与 MLP 各一支；LN 不带固定 affine，条件输出最后线性层置零，F 的线性层正常初始化。最终解码层另外置零。[官方 models.py：块与初始化](https://github.com/facebookresearch/DiT/blob/main/models.py#L87-L191)

**梯度推导：** 对 `y=h+g*f_theta`，若 `g=0` 且 `f_theta=0` 对所有输入恒成立，则 `∂y/∂g=f_theta=0`，`∂y/∂theta=g*∂f/∂theta=0`。在无其他训练路径或扰动时，该分支不能靠任务梯度启动。因此不要同时把同一乘积的外门和分支输出层置零。

可行的恒等初始化只需一个零点：

- 零 gate + 正常初始化 F；gate 首先能收到梯度。
- 正常初始化 U/V + 零 W，无额外零 gate；W 首先能收到梯度。
- 直接 FiLM：gamma/beta 的输出层置零，主路径乘子是 `1+gamma`，无需再套零 gate；输出层可先学习。

DiT 的零最终解码器不是上述死分支：最终线性层可从非零隐藏状态得到梯度，打开后梯度再传向块。这与在同一残差乘积中同时清零 gate 和 F 的输出不同。这里说明的是一般非退化输入/损失下的梯度路径，并不保证每个具体 batch 的梯度非零。

## 4. 当前仓库中的位置比名称重要

当前 `cross_summary` 为同行业与全市场 leave-one-out 均值拼接，经零初始化线性层后加到股票表示；`available`、单成员、动态行业与未知行业已有处理。当前 `market_film` 则用外部 market 特征生成 scale/shift，作用于原始 temporal，随后才走最终归一化；它不是“使用横截面摘要的 adaLN”。[摘要实现](../src/quant_mvp/hierarchy.py#L433) · [现有 FiLM](../src/quant_mvp/hierarchy.py#L365)

最终评分路径是 `z = L2(final_norm(h)); score = Linear(z)`；动态行业路径单独走这条链，且明确拒绝 `market_film`。所以不能只给 `relate()` 接一个新模块就假定覆盖全市场动态行业版本。[常规评分](../src/quant_mvp/hierarchy.py#L424) · [动态行业入口](../src/quant_mvp/hierarchy.py#L467) · [配置限制](../src/quant_mvp/hierarchy.py#L211)

**归一化的数学影响：** LN 消去所有通道相同的平移；正的统一缩放在忽略 epsilon 时也被消去。L2 消去非退化向量的正径向缩放。逐维 scale、非均匀 shift 一般会改变方向，不能说它们全部被归一化抹掉；也不能忽略放置位置造成的变化。

若只测试最小末端条件化，一个明确候选是：

```text
u = existing_final_norm(h)
z = normalize((1 + gamma(c)) ⊙ u + beta(c))
score = existing_score(z)
```

保留既有 final_norm 参数与 L2，gamma/beta 零初始化时可与原评分路径对齐。这是末端 adaLN 风格 FiLM，不是完整 DiT adaLN-Zero 块。它与“先调制 raw h，再 LN”是不同实验。若未来采用残差 adaLN-Zero，需要明确 F 是哪一个分支；不必为借一个条件化方法新增一整个 Transformer。

具体最小候选可先只做 scale：`gamma(c)=A(Bc)`，无 bias，`A:128×8`、`B:8×256`，共 3072 参数；A 零初始化、B 正常随机初始化。它接在现有 final_norm 后、现有 L2 前，不新增 LN；第一步 A 有学习路径，B 待 A 非零后开始接收梯度。可与约 3080 参数的低秩双线性评分候选做预算相近的比较。对所有时序块加 DiT 风格条件化是另一个更大的假设，不与这项末端比较混为一谈。

## 5. 建议的比较边界

优先保持相同 leave-one-out 摘要、mask、时间截止与训练目标，仅比较已有加法摘要、逐维调制、低秩乘法。优先检验逐维调制是工程判断，不是性能预测。对零初始化至少应核对初始输出一致，以及首步/后续步骤存在预期梯度；不要用“初始输出没变”代替可训练性验证。

Set Transformer 可稍后用于学习集合摘要，或用等变编码器输出逐股表示。若替换 leave-one-out 均值，PMA/共享摘要默认可能含目标股票，必须明确是否仍要求排除自身；股票逐行打分需要随股票置换而同步置换，单个不变集合输出本身不是逐股打分器。这些是适配任务时的结构要求，论文没有证明它会改善本仓库收益。

未做：训练、回测、性能测量及代码修改。风险：当前原始论文实证均不直接支持金融收益迁移，初始化与归一化放置错误会让架构对比失真。
