# P4 / P5 均值优先训练：实现、检查与入队

日期/时区：2026-11-14，Asia/Shanghai。状态：complete（实现与CPU检查、队列登记）；真实训练/效果：pending。
协议：[P4 / P5 金融目标与梯度定义](research_20261114_finance_faithful_protocol.md)。

## 已完成

新增 `loss=faithful_mean_nll`，复用收益分解、M6状态交互及原Gaussian/双尺度混合头。仅当新目标开启时：风险头输入detach、风险NLL内均值detach，均值学习训练参考尺度下的日期等权half-MSE；均值与风险参数分别裁剪。模型参数量、推理输出、评分排序和两腿选轮语义保持原定义。P0–P3仍用原NLL，旧B0路径保持。

训练日志区分组合目标和NLL，验证仍记录真实NLL。报告增加P4−P2、P5−P3、P5−P4以及各臂对B0的配对；归档器另核对P4/P5同轮验证mu差，并按预登记的验证RMSE/L20净Sharpe方向给出判读。未把该计算图称为DML或有一阶去偏保证的金融估计器。

修改涉及 [模型/损失](../src/quant_mvp/financial_return.py)、[训练循环](../src/quant_mvp/hierarchy_engine.py)、[评价配对](../scripts/evaluate_dataset_three_year.py)、[P4配置](../configs/financial_return_p4.yaml)、[P5配置](../configs/financial_return_p5.yaml)、[梯度测试](../tests/test_financial_faithful.py)、[训练/评分/恢复测试](../tests/test_financial_return.py)。不接入实盘，不覆盖旧实验产物。

## 验证证据

命令：`python -m pytest tests/test_financial_faithful.py tests/test_financial_return.py tests/test_dataset_three_year.py tests/test_standard_scoring.py tests/test_followup_entry.py tests/test_followup_queue.py -q -k "not cuda"`。

结果：45 passed、1 deselected（CUDA），29.15秒。验证包括：

- 不同日期样本数下均值梯度严格匹配日期等权MSE，尺度梯度匹配原密度NLL。
- 对高斯及混合分别构造大风险梯度和非零尺度权重，检查风险不能更新共享/均值参数；连续3步AdamW、分别裁剪后，均值参数和优化器动量逐项完全匹配纯MSE子网络。
- 新目标两种概率族均通过训练中断恢复；CLI合成训练、冻结预处理、无标签评分路径通过。候选/监督mask、尾标签、概率融合、选轮锁和旧入口兼容性检查通过。
- 对比原冻结源码，四个旧金融臂的初始化、前向、NLL和裁剪后梯度逐项相等；新目标的真实训练数组/预处理与父实验一致，779锚、215690监督行。原尾标签审计通过原manifest源哈希继承。
- 六个队列任务的依赖图、实际脚本路径、编译和新旧配置差异检查通过；新臂相对对应父配置只改变loss和source_sha256。

CPU检查与数组证据：[preflight.json](../outputs/finance_faithful_v1/registry/preflight.json)。完整冻结：[pipeline_inputs.json](../outputs/finance_faithful_v1/registry/pipeline_inputs.json)。源码身份：`3d9c51f6410e11a1739ced124ba919eeac694b416371a1e1985d8df04dbba450`；目录 `outputs/finance_faithful_v1/source/3d9c51f6410e11a1`。环境Python3.11.7、Torch2.5.1+cu121，其余版本见manifest。真实CUDA全尺寸前后向尚待GPU锁释放后执行，门未通过前不会训练新臂。

派发前登记修订2：修正归档脚本插入INDEX结果链接时匹配的标题，旧manifest保存在 `registry/pipeline_inputs.registration_v1.json`；源码、配置、任务和数学协议均未改变，全部输入哈希重新核验通过。

## 队列状态快照与恢复入口

2026-11-14 00:38（Asia/Shanghai）：新增隐藏启动器PID18460，等待父批P0–P3所有任务及报告完成。共享GPU锁仍由既有D3队列worker22556占用；D3_s43_a1训练进程46164存在。原金融启动器37184等待该锁。这里的PID仅是本次快照，后续须重新核对进程、日志、state和产物。

本批6任务：CUDA/父产物门、P4训练+验证锁、P5训练+验证锁、P4历史评分、P5历史评分、报告。新增真实训练预算16轮，各seed42/8轮；复用B0/P2/P3。153周三年验证、逐腿独立选轮、297周已读历史复检与成本/压力口径不变。两新臂均锁定后才评分历史。父批失败时启动器退出并记录原因，不抢GPU或绕过门。

后台日志：[launcher.log](../outputs/finance_faithful_v1/launcher.log)与 `launcher.err`；执行入口：[launch.py](../outputs/finance_faithful_v1/launch.py)，具体命令在 `queue/inbox/*.json`。训练前再次验证父批已完成且产物哈希一致，锁定复用父产物到 `registry/parent_outputs.json`；每个后续任务再次核对。恢复遵循既有完整状态规范，失败新建attempt，不直接重跑有已发布评分的多命令任务。

完成后自动输出机器报告到 `outputs/finance_faithful_v1/report`，并将核心结果与提升分析写入 `docs/research_20261114_finance_faithful_results.md`、INDEX及实验台账。本页没有真实提升结论，不能用CPU梯度等价证明金融泛化收益。

## 限制

2026-11-14 03:04恢复：首次CUDA门在父锁检查时因Windows默认GBK读取UTF-8文件失败，未开始训练。保留原failed/BLOCKED状态与日志；新增UTF-8读取脚本、独立哈希清单和`_retry2`任务，原源码、配置及输出路径不变。父产物校验及P4/P5真实尺寸CUDA前后向均通过，P4进入训练。见 `registry/pipeline_inputs_retry2.json`、`logs/cuda_preflight_retry2.log`。

2026-11-14 04:49–04:51：P4/P5训练、验证选轮和历史评分均完成；标准机器评估成功，但`report_retry2`归档脚本仍引用不在本批catalog内的P0/P1，触发StopIteration，原失败记录保留。仅修复文字报告的比较列表为预登记的P4−P2、P5−P3、P5−P4及P5−B0，以 `registry/archive_results_retry3.py` 对已有机器产物执行归档，不重训或重算评估。结果文档、INDEX和台账已生成；报告job状态仍保留failed，不能误称原任务成功。修复后结果见[真实对照结果](research_20261114_finance_faithful_results.md)。

未实现目标尚未明确的影响函数/DML、训练期OOF残差修正或算术收益期望选股。新方法可能因平方误差对尾样本敏感而逊于原NLL，也可能损失风险表示学习；这些是P4/P2和P5/P3对照要回答的问题。原Q1幸存股池、非PIT及执行代理限制仍在。证据仅本机、未异机备份；本轮未提交或推送Git。
