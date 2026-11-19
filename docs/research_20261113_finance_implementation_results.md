# 金融收益模型 v1：实现与合成验收

日期/时区：2026-11-13，Asia/Shanghai。状态：complete（代码实现与合成验收）；金融新模型真实训练状态：未启动、未入队。
父方案：[数学与架构定义](research_20261113_finance_model_results.md)；研究边界：[协议](research_20261113_finance_model_protocol.md)。

用户最新授权是实现收益分解、市场状态和厚尾概率头，并完成合成与兼容检查；不运行真实训练。测试中的优化器更新、CPU/CUDA前后向和交易回放全部使用生成数据及临时目录，不构成金融实证。

## 已实现的行为

新增独立 `model_type=financial_return_v1`。未指定类型仍构造原 `HierarchicalRanker`；B0、CS/CB/ADN的类定义、原损失公式、旧配置和发布锁未被替换。新类直接复用AttentionEncoder，不构造旧关系token或InfoNCE projector。

- 均值：候选集合内汇总行业和市场表示，输出共同均值、行业偏离、组内残差。小于5股的行业和未知组并入OTHER；空候选不输出预测。分解先在无量纲坐标工作，再乘训练参考尺度 `s_ref`，贡献之和等于 `mu_log`。
- 市场状态：6→8→4状态映射与股票128→4投影做低秩交互，616参数；增益零初始化、映射非零。P1关闭且冻结均值状态分支，所有P的135参数风险头仍读取M6。
- 概率：P3为同中心固定双尺度混合；P0/P1/P2为匹配总方差的Gaussian。使用原始log标签、日期等权NLL，不截断合法收益、不叠加旧loss。全batch无监督跳过optimizer，包括weight decay；整轮无更新则明确失败。
- 单位：`score=mu_log`，`scale_log`始终是基准s；单Gaussian的标准差为√1.8s。P0不伪造三分贡献。128维主配置的P2/P3金融decoder为7088参数，P0为6928；主干及final_norm不计入此数。
- 数据：保持旧候选、训练锚与共同五周预留。只在训练锚估计 `s_ref`/初始均值，保存日期、标签mask及标签哈希。scores_only必须读取冻结统计且不访问标签文件。缺失/非有限M不填零，在实际训练/评分使用时拒绝；冻结缩放和静态行业身份校验。
- 恢复：复用现有完整状态、优化器、随机数和原子写入，teacher=None；概率字段随最佳验证预测修复，NLL指标随提交历史回滚。新类加载权重还验证 `s_ref` 与行业buffer。
- 评分与评价：标准验证/锁定测试导出保留概率字段；新run依据自身提交状态核对配置和预处理，不借用B0 reuse-audit。选轮锁绑定分布与融合语义，概率文件manifest绑定checkpoint、配置、预处理和分数哈希。按每腿窗口分别混合密度和原单位均值，跨seed继续混合完整成员；旧模型保持原z融合。

## 文件与接口

| 文件 | 职责 |
| --- | --- |
| [financial_return.py](../src/quant_mvp/financial_return.py) | 新模型、工厂、训练锚、冻结统计、分布定义和NLL |
| [financial_scoring.py](../src/quant_mvp/financial_scoring.py) | 概率schema、跨轮/seed混合、CDF/区间/矩及概率诊断 |
| [hierarchy_data.py](../src/quant_mvp/hierarchy_data.py) | 新类型的冻结统计与严格M/行业身份；旧路径保留 |
| [hierarchy_engine.py](../src/quant_mvp/hierarchy_engine.py) | 新旧目标分派、概率预测、训练指标和共同保存恢复 |
| [run_weekly_architecture.py](../scripts/run_weekly_architecture.py) | 原训练/开发评分入口接入新工厂；CPU资源记录修复 |
| [score_dataset_standard.py](../scripts/score_dataset_standard.py) | 完整验证/锁定测试概率导出；新增显式CPU评分选项 |
| [evaluate_dataset_three_year.py](../scripts/evaluate_dataset_three_year.py) | 逐腿锁定、原单位融合、概率附表及P2−P0/P1、P3−P2比较 |
| [live_score.py](../src/quant_mvp/live_score.py) | 明确拒绝未接入实盘的新类型，防止套用旧z融合 |
| [新模型测试](../tests/test_financial_return.py) | 数学、梯度、mask、恢复、真实CLI的合成调用、锁与评价产物 |
| [入口兼容测试](../tests/test_followup_entry.py) | 工厂接入后调整mock位置，保留原导出隔离断言 |

配置已保存为 [P0](../configs/financial_return_p0.yaml)、[P1](../configs/financial_return_p1.yaml)、[P2](../configs/financial_return_p2.yaml)、[P3](../configs/financial_return_p3.yaml)。它们沿用BQ_F0_r2训练输入、seed42/8轮预算、40/30/128主干，默认不自动执行。训练入口中的2018–2019验证仅是原开发诊断；最终仍使用标准评分入口的完整2018–2020验证选轮，没有将主协议改回两年。

配置的 `source_sha256` 按既有 `freeze_source` 文件清单和算法计算，已更新为本轮代码摘要，不沿用旧B0源码摘要；本轮未为真实执行冻结副本。执行前若源码再变，必须重新冻结与登记。工作区另有并行D3/D4改动，本任务保留其实现及文档；金融v1明确拒绝同时设置时间衰减或质量通道，以免把日期等权NLL与另批干预混用。这里的“未训练”仅指本金融模型任务，不描述其他会话的实时任务。

训练新增 `distribution.json`、`preprocessing.json.financial_return`、`nll_loss/val_nll`及尺度分位数/floor占比；验证预测包含概率字段。概率评分文件带 `.json` 哈希与语义旁表。评价新增：

- `probability_members.parquet`：逐phase/portfolio/leg的完整Gaussian成分、权重及成员标识。
- `probability_observations.parquet`：成熟有效标签上的NLL、PIT、50%/90%区间及simple收益期望。
- `probability_daily.csv` / `probability_summary.csv`：先股票平均，再日期等权的概率指标与硬RankIC。
- `probability_risk_buckets.csv`：尺度风险分层的均值误差和NLL；只作评分后的诊断。
- `legacy_z_diagnostic.csv`：P3相同成员旧z融合的附读策略结果，不切换主选择规则。

## 验收范围与已发现的问题

合成断言覆盖参数预算、候选中心化、股票置换、singleton/空候选、无效输入不污染、零状态增益与后续梯度、Gaussian/混合NLL对torch参考分布、有限差分梯度、收益单位Jacobian、原始极端标签、日期等权、冻结统计无标签评分、混合矩/区间及Parquet往返。

四个新模型变体分别在CPU小数据上比较连续两轮与一轮后断电恢复，逐位核对模型/优化器/随机数/最佳预测。另有CUDA小张量前后向检查；没有运行128维全规模CUDA训练或性能基准。

真实CLI合成检查从临时v2快照调用 `run_weekly_architecture.py train --device cpu` 和标准评分入口。评价检查使用另一组生成的市场、成交与借券记录，执行选轮→成员锁→概率融合→评价产物发布，并检查不同腿的成员窗口。该存储/评价测试的checkpoint是身份fixture，不冒称真实神经模型权重；模型权重和恢复由前一组真实合成训练检查覆盖。

调试中修正：测试参考密度的布尔索引写法、Windows锁文件UTF-8读取，以及合成成交额全相同导致原流动性规则排空。也修复原CLI CPU训练成功后把空字符串当显存数值格式化的错误。最初并行检查曾因开发过程中源码变化触发恢复指纹拒绝；最终检查冻结源码后运行，不放宽恢复身份规则。

## 使用边界与下一步

新模型支持现有研究训练、开发评分、标准完整验证评分、逐腿选轮、锁定历史评分和评价。旧 `history-scores`/旧evaluate及live路径明确拒绝新类型，引导到标准概率流程；不偷偷把新分布按旧score处理。

真实训练前仍需完成协议要求的原始尾标签来源审计，以及执行批次的环境/源码/输入身份和资源预算登记。实际收益增益、真实市场概率校准、全规模显存/吞吐均未验证。Q1幸存股池与回溯行业边界仍在。

未来若用户授权运行：在独立批次根目录准备 `base_dev.yaml`（可复制所选P配置）后，使用现有命令形状：

```powershell
python scripts/run_weekly_architecture.py train --root <新批次目录> --dev-data outputs/dataset_v2/snapshots/BQ_F0_r2 --cfg configs/financial_return_p3.yaml --run <新批次目录>/runs/research_P3_F0_s42_a1 --device cuda
python scripts/score_dataset_standard.py validation --run <run> --snapshot outputs/dataset_v2/standard_v1/snapshots/validation --out <新评分目录>/validation.parquet --device cuda
python scripts/evaluate_dataset_three_year.py select --run <run> --scores <validation.parquet> --snapshot outputs/dataset_v2/standard_v1/snapshots/validation --out <新选择目录>
python scripts/score_dataset_standard.py test --run <run> --snapshot outputs/dataset_v2/standard_v1/snapshots/history --lock <新选择目录>/lock.json --out <新评分目录>/test.parquet --device cuda
python scripts/evaluate_dataset_three_year.py evaluate --validation-snapshot outputs/dataset_v2/standard_v1/snapshots/validation --snapshot outputs/dataset_v2/standard_v1/snapshots/history --member <run> <validation.parquet> <lock.json> <test.parquet> --out <新评价目录>
```

上述带占位符的命令仅说明接口，本轮未执行真实批次，也未创建 `outputs/finance_return_v1`、修改原队列、旧模型选择锁或训练产物。旧完整状态按原严格源码身份规则，不支持在源码变更后冒充精确续训；旧权重的评分/对照用途仍保留。未提交或推送Git。

## 最终验收记录

主检查命令：

```powershell
python -m pytest -q tests/test_financial_return.py tests/test_hierarchy.py tests/test_conditional_scoring.py tests/test_training_resume.py tests/test_standard_scoring.py tests/test_dataset_three_year.py tests/test_dataset_contract.py tests/test_followup_entry.py
```

结果：103 passed，130.32秒；1项已存在的常量输入Spearman警告。包含新模型17项、原模型/接口/恢复等86项。随后针对并行D3/D4配置互斥增加断言并检查相关原有测试；结果另列于下方。

补充检查：`python -m pytest -q tests/test_financial_return.py::test_statistics_only_train_anchors_and_frozen_scores tests/test_data_trials.py tests/test_stock_weights.py`，8 passed，15.28秒（其中1项与主检查重叠）。该检查证明新v1拒绝混入D3/D4，同时保留它们原有默认损失与数据行为。

文档/身份验收：8份金融文档96处本地链接全部可解析；下方14项实现/测试/配置文件已登记SHA256；4份P配置的source摘要与当前既有冻结算法一致。原B0配置、发布lock及BQ_F0_r2 manifest与设计阶段指纹相同；`outputs/finance_return_v1`不存在。未检查其他会话实时任务，也未干预它们。

这不是全仓库测试、真实市场回测或全尺寸性能测量。所有合成产物位于pytest临时目录，未保存为候选收益证据。最终源码与配置SHA256随本记录附表登记；原设计证据底稿的旧指纹保留，不把历史指纹改成新实现。

### 最终文件指纹

| 文件 | SHA256 |
| --- | --- |
| [src/quant_mvp/financial_return.py](../src/quant_mvp/financial_return.py) | `7a795be5a85c7cb83e3fa0822100a57566b5badaf7e08c67a71adca2660045a2` |
| [src/quant_mvp/financial_scoring.py](../src/quant_mvp/financial_scoring.py) | `177ad8c6b54707e019b3653dc3728161f5d2d41e99cc1a9e1dd239e29c0cad73` |
| [src/quant_mvp/hierarchy_data.py](../src/quant_mvp/hierarchy_data.py) | `ec5ae147647384c2ed6dcdf37f66ce8e1c3a16965b118774a864abdb6c504b7a` |
| [src/quant_mvp/hierarchy_engine.py](../src/quant_mvp/hierarchy_engine.py) | `20be6d9a177073a2a7828de5f05dbf8434cc9bd5073fcfd19d18e768a93ddad3` |
| [src/quant_mvp/live_score.py](../src/quant_mvp/live_score.py) | `df3f334f14a831644710e102d2381b74a7b6cff0f7a9574e0ba1a9c8fb3c02d2` |
| [scripts/run_weekly_architecture.py](../scripts/run_weekly_architecture.py) | `196c4126df305711301c359ff378f38ec649f5709389e6b0942ae4040bbed69d` |
| [scripts/score_dataset_standard.py](../scripts/score_dataset_standard.py) | `e4c8b22c9282cdc554100db7741b050767f638d30e5f831b175f75f4241a0cf5` |
| [scripts/evaluate_dataset_three_year.py](../scripts/evaluate_dataset_three_year.py) | `e7d50a0fc49fc438b2fa833c8542a6edcf54603b6bc3222d716180b82a66edd7` |
| [tests/test_financial_return.py](../tests/test_financial_return.py) | `2e60e8b86bed905efab5860f50c0b62c814691b1e796c14099e93cce18bab4c1` |
| [tests/test_followup_entry.py](../tests/test_followup_entry.py) | `613bbcda94399460747ebed8a4faca5fcdc4854d1c5782eaea5620fa54ebd07a` |
| [configs/financial_return_p0.yaml](../configs/financial_return_p0.yaml) | `9595cc2dbd7ccee997d4734187b36408b16651b67ae02b4a0d090d6b562eed0a` |
| [configs/financial_return_p1.yaml](../configs/financial_return_p1.yaml) | `6ead7f0083dda974002164bec8ab998395965b31470f11dc5b0e82393257c79f` |
| [configs/financial_return_p2.yaml](../configs/financial_return_p2.yaml) | `94ce26e156c418ceae53e5b13b72818a3161d2cb7c0d59f064f8a4102a510fe2` |
| [configs/financial_return_p3.yaml](../configs/financial_return_p3.yaml) | `895cd38b62470bbe4b0146a48782d1b6ac5cf11a85c01fc350c9d255496df2f3` |
