# A股数学建模研究：共同事实底稿

日期：2026-11-13，Asia/Shanghai。状态：complete（代码与现有登记的只读核对，不代表新实验完成）。
用户范围：六角度研究、报告与待执行方案；不改代码、不训练、不改队列。主协议：[三年验证](research_20261113_standard_execution.md)；记录规则：[归档](research_recording.md)。

## 审计对象与证据边界

主对象为 `outputs/dataset_v2/standard_v1/catalog.json` 中已复用的 Q1 B0/CS/CB/CB_SELF/ADN/ADN_SELF；A1、D2保留为其他研究臂，不把它们视作相同输入的纯架构对照。B0原运行是 `outputs/dataset_v2/exploratory_F0/runs/research_B0_F0_s42_a1`。旧 `weekly_architecture_v1` 的 B0/P64/CS/JOIN 与本对象训练指纹不一致，见 p64_recheck_v1/audit.json；二者不可合并作架构因果排名。部署策略另属 live 链路，本轮不认证线上现状。

HEAD为 `f2e198e8769ab960b7c8d6a8aa2ba50eb1bb8fe3`；工作树已有大量已修改和未跟踪文件，HEAD不能唯一代表实际实现。下表固定当前读取源码身份；冻结源码兼容性由各方向另核对。本轮不重新执行reuse_audit，不将已存审计结果写成新算出的证明。

## 一个实际模型的计算链

记 t 为周五决策锚、i 为股票，原价成交/复权特征、历史分类及可用性来自所选快照。B0配置为580只固定样本、30特征、40周输入；有效股票数随日期变化。`hierarchy_data.py:110` 开始构造数组；当前候选基于已观测行情和有限特征，且该配置启用20/40/80周共同可用性限制（分别至少16/32/64个有效观测），并非只要求40周内32周。监督有效性另与标签有效性交集，推理候选不按未来收益筛选。

训练前缀拟合全局缩放，随后逐日期横截面缩放、截断和缺失历史携带；因此B0虽没有显式跨股关系网络，原始输入的标准化仍依赖当时横截面。当前行业分类配置明确为回溯快照，不能解释为完整PIT数据。

单股票输入 X_it∈R^(40×30) → Linear(30,128)+LayerNorm → 四层、四头带RoPE的因果时序注意力 → 末时点表示、LayerNorm和embedding Linear → h_it∈R^128。实现：model.py:285–445，hierarchy.py:130–170、354–368。注意：评分直接使用encoder(normalize=False)的输出，独立projector用于对比学习，不混为同一输出头。

B0关系模块仍被构造，但 relation_gate_init=0、freeze_relation_gates=true，训练明确置零并冻结关系门（hierarchy_engine.py:413–414）。在有限数值输入下关系层是恒等映射；市场token、行业token的注意力路径不改变有效股票评分。时序注意力本身保留。于是 B0 实际评分可简写为：

    h_it = E_theta(X_it)
    z_it = normalize_L2(LayerNorm(h_it))
    s_it = w^T z_it + b

CS在评分支路加行业/市场leave-one-out摘要；CB加低秩条件双线性交互；ADN在L2归一化前做条件逐维缩放；SELF用自身表示形成等形状条件。具体公式与参数由架构报告核对。结构名称不等于分支启用。

B0实际训练目标（hierarchy_engine.py:595–695）：

    L = L_tail + 0.5 L_soft_rank_IC + 0.1 L_InfoNCE + 1.0 L_industry_eta2

- tail：按有效横截面未来收益头尾各20%给±1标签，逐股票二元logistic，忽略中间项，按batch全部被选股票归约；不是逐对RankNet。
- soft-rank IC：对预测和训练标签分别用pairwise sigmoid求软秩，逐日期求相关后平均；共同温度0.05，不等于精确Spearman。
- InfoNCE：当前40周表示对未来5周同股票EMA编码，当前/未来可用mask限定训练样本，其他可用股票作负例，温度0.1、EMA0.99；未来窗口不进入评分输入。
- 行业eta²：评分跨行业均值解释的方差比例，在当日输入可用mask上计算，不要求未来标签有效。它不直接约束Top20组合的行业权重。

训练标签默认截断到±0.25；验证/测试保存原始标签，回放将log收益expm1转简单收益。详见损失报告。未启用的附加损失不算进B0。

## 日期、选择和交易边界

B0 training_protocol.json记录779个实际训练锚，首日2002-07-19、末日2017-11-24，预留未来5周；reuse_audit.json记录最后时序上下文2017-12-29。训练截止边界为2018-01-01。旧run配置仍带selection_at/outer_start=2020-01-01等F0字段，这是旧配置身份，不是standard当前验证范围。

当前B0选轮lock记录153成熟验证周，2018-01-05至2020-12-25；每腿净Sharpe独立选最佳轮及邻近三轮。已存B0多头窗口1/2/3，空头6/7/8；窗口是选择结果，不是未来候选固定参数。当前标准协议历史复检297周，自2021-01-08至2026-10-23，最后退出2026-11-02。本轮未重算收益或读取逐行历史测试来调候选。

评分→每轮截面标准化→所选窗均值再标准化→跨seed融合→L20/S10组合→受可交易/资格代理限制的回放。主费每边10bp、空头年借券费8.35%；压力30bp/12%。50:50组合和同多头预算加现金分开比较。未来实际券源、停牌退出、权益与缺失收益仍是执行证据缺口；公开融券资格不等于可借券。

## 本轮完成与限制

完成：导航/协议/配置/锁/关键实现交叉阅读，记录架构名称与实际启用路径、旧配置与当前选轮边界的差异。未运行训练、模型推理、回测、单元测试或全量数据审计。另以标准库assert验证一个解析反例：行业A分数[-3,-2,2,3]、行业B分数[-0.3,-0.2,0.2,0.3]，两组均值均为0，因此行业eta²=0，但Top2全属行业A；这是构造反例，不是市场实验。此底稿不宣称其他会话的任务当前正在运行或停止；status.json是历史快照。原始data/outputs仅本机可见，未在本轮外部备份；本文件保留核心事实与哈希，仍需原产物才能完整复现。

## 读取时证据指纹

| 文件 | SHA256（本轮读取时） |
| --- | --- |
| [src/quant_mvp/model.py](../src/quant_mvp/model.py) | `630e0ecb67f0db1ae926f82eb8465df578ffcfb508189f5d734988fe3223276a` |
| [src/quant_mvp/hierarchy.py](../src/quant_mvp/hierarchy.py) | `98b5dae2f295b57cec5568ba13b05cfb9798fa0a5791bac6501df09e2e955850` |
| [src/quant_mvp/hierarchy_engine.py](../src/quant_mvp/hierarchy_engine.py) | `b92e39aa3028a3bbce5c64b13f0fb77e3efb428613189d08ac6d843ea515764b` |
| [src/quant_mvp/hierarchy_data.py](../src/quant_mvp/hierarchy_data.py) | `2faf89efe5c55af9b746d8123e46204f932dea1e2a8916d094d054caff046796` |
| [scripts/run_weekly_architecture.py](../scripts/run_weekly_architecture.py) | `b6716233f8175bb044981610806c919bf08366674a3818ee06a87b4d66c7b540` |
| [scripts/score_dataset_standard.py](../scripts/score_dataset_standard.py) | `387e57e7d3f1ed55501fd75a676eb40ab598a65c712810c8b431fb1c97f30c23` |
| [scripts/evaluate_dataset_three_year.py](../scripts/evaluate_dataset_three_year.py) | `484599afd854ed4e70ea3395c21ed0530ce672aa9bbc35c25258afc4cce268d2` |
| [src/quant_mvp/dataset_ledger.py](../src/quant_mvp/dataset_ledger.py) | `953a45b96a3211624819994d006835a04ef3abad05cf6ba3e3f580caeb5ef27e` |
| [outputs/dataset_v2/standard_v1/reuse_audit.json](../outputs/dataset_v2/standard_v1/reuse_audit.json) | `a974f6c5afd634278e174db67d70e75f652d8e6e86548a85fbd85685cb5821b9` |
| [outputs/dataset_v2/standard_v1/catalog.json](../outputs/dataset_v2/standard_v1/catalog.json) | `a02dd8b8956dbad9812f39e6053c29893067c81903498454fd7066d1d2237cd7` |
| [outputs/dataset_v2/standard_v1/selections/research_B0_F0_s42_a1/lock.json](../outputs/dataset_v2/standard_v1/selections/research_B0_F0_s42_a1/lock.json) | `0c02bbbd91c727fc204600863498d2820cdbf2d116ae57525b15477953029662` |
| [outputs/dataset_v2/exploratory_F0/runs/research_B0_F0_s42_a1/config.yaml](../outputs/dataset_v2/exploratory_F0/runs/research_B0_F0_s42_a1/config.yaml) | `bd57708c8ae8d950f290be47ff1129483fba910434ae6153d4aac588f81c71a0` |
| [outputs/dataset_v2/exploratory_F0/runs/research_B0_F0_s42_a1/training_protocol.json](../outputs/dataset_v2/exploratory_F0/runs/research_B0_F0_s42_a1/training_protocol.json) | `816ccf043d14429ae624673352cb98850d0717ace8ac175e2ca16245cf739c33` |
| [outputs/dataset_v2/p64_recheck_v1/audit.json](../outputs/dataset_v2/p64_recheck_v1/audit.json) | `edf80c43b1c4880eb1d7a76b30efbf991cc98716704edb999e2fa3095ef53348` |
