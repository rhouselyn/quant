# 组内排序与 FoX：实现进度

日期/时区：2026-11-14，Asia/Shanghai。状态：运行中，后续续接见本文末；本页此前记录了partial快照。父方案：[归因和排序/缓冲](research_20261114_finance_attribution_results.md)、[架构研究](research_20261114_architecture_papers_results.md)。

已加入可选组内两两排序辅助项，按日期等权、同行业等有效组采样、每股最多4个有效不等标签对；仅启用于faithful分解模型，默认关闭。预测均值差以训练参考尺度归一，权重候选0.1。加入分项训练日志，保留风险梯度隔离。另加入可选FoX机制：每头输入依赖遗忘门、因果logit偏置，初始遗忘系数约0.99；默认关闭，不允许非因果、压缩及单锚点路径。两者是独立开关。配置草案不能作为冻结实验配置使用。

测试：`tests/test_fox_attention.py tests/test_financial_group_pairs.py tests/test_financial_faithful.py tests/test_financial_return.py -q -k 'not cuda'`：29 passed、1 deselected；`tests/test_training_resume.py tests/test_followup_entry.py tests/test_dataset_three_year.py -q -k 'not cuda'`：23 passed、2 deselected。包含合成训练中断恢复、组内掩码/梯度、FoX因果及5周编码；未完成真实全尺寸CUDA检查和FoX的B0训练/评分全链合成测试。

未完成：冻结源码与配置、独立对照协议及公平预算登记、Top20/30保持原回放执行约束的零训练回放、GPU训练队列与自动报告。任何本地配置中的旧source_sha256须在冻结时重算，不得据此入队。原B0/P4、已完成结果及锁均未覆盖；真实训练没有启动。工作区源代码已改动，恢复工作前须检查现有diff及其他用户改动，不可重置工作区。

## 2026-11-14 16:03 后续执行快照

以上“未完成”是上一次会话的历史快照，现已补齐。已登记[独立实验协议](research_20261114_group_fox_protocol.md)和[零训练缓冲结果](research_20261114_buffer_2030_results.md)。P6只改变P4目标增加0.1组内排序，FOX只给B0添加FoX动态遗忘；两个配置相对各自父运行配置仅改变对应开关及冻结源码SHA。验证、锁定历史前后关系不变。

综合合成/兼容测试：`python -m pytest tests/test_fox_attention.py tests/test_financial_group_pairs.py tests/test_financial_faithful.py tests/test_financial_return.py tests/test_training_resume.py tests/test_followup_entry.py tests/test_dataset_three_year.py tests/test_buffer_2030.py -q -k 'not cuda'`，54 passed、3 deselected。包含FoX真实训练入口小模型、EMA teacher、标准验证评分和概率臂完整状态恢复；FoX与B0共有参数seed42初始化逐项完全相等，新增2064参数。缓冲原策略逐周净收益和换手对账。完整实际批次的CUDA前后向通过，RTX4060、batch16、580股票；FoX含EMA与原损失，峰值1229.7MiB，排序臂1656.8MiB，优化器更新0；证据`outputs/group_fox_v1/registry/cuda_preflight.json`。

注册时旧父manifest含过去版本的工作区测试文件，其哈希与新增测试修改不同；本批manifest改用本次测试的实际哈希，原旧manifest与旧证据保留。重新核验全部冻结输入通过。源码SHA `19947f8ea318750a5cd1a8446259ad28875aa298b6d7dcbccc2258fbad1dc593`，冻结源码和新配置、数据及父对照见`outputs/group_fox_v1/registry/pipeline_inputs.json`。草案YAML仍非登记配置；实际配置在批次configs下。

16:03快照：worker PID37584持有共享GPU锁，P6第1/8轮进行中，FOX排在P6之后；随后两臂均锁定后分别评分历史，最后生成[真实对照结果](research_20261114_group_fox_results.md)并补INDEX/台账。原文档链接在报告成功前尚不存在，不应将队列安排误称完成。失败/恢复按[标准执行](research_20261113_standard_execution.md)和任务state处理，不删除旧锁或重跑已发布评分。结束前须重新核查进程、日志、state及最终产物。

16:49完成快照：五项任务全部done，report两个命令均exit0、无缺失产物，worker退出并释放共享GPU锁。P6和FOX各8轮训练、153周逐腿验证锁、297周历史评分及配对报告完成；两臂实际各779训练日期/215690监督样本。已发布[真实对照和限制](research_20261114_group_fox_results.md)，不自动扩种子或部署。以上PID仅为快照，未来查询当前状态仍需重新检查。
