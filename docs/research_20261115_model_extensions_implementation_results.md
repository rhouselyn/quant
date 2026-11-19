# 行业投影与差分注意力实现、研究与入队结果

2026-11-15，Asia/Shanghai。研究、实现、合成验收和入队完成。真实训练等待共享GPU。候选推导见[证据报告](research_20261115_model_extensions_evidence.md)，执行边界见[协议](research_20261115_model_extensions_protocol.md)。

新增两项独立实验。P4GROUP对P4用权重1的中心化行业投影损失强调相对行业收益误差，新增参数0。BDIFF2对B0引入双Q共享K/V差分注意力，实测新增68112参数；原FFN维度和金融目标沿用各自父模型。行业投影是本项目推导，DIFF V2来自作者技术发布，所有金融收益均待实证。

完成39项独立合成/兼容检查。首轮全选测试38 passed，追加差分精确恢复后新模块5 passed。覆盖有效标签屏蔽、共同平移不变性、单组零项、因果性、有限梯度、共享初始化、两臂连续训练与中断恢复、EMA teacher、标准评分、概率恢复与选轮评价。实测参数计数、冻结比较对、原队列manifest及隔离临时产物上的报告生成检查通过。模型新开关默认关闭，旧checkpoint通过原配置加载。

冻结源码SHA256为 22f09833747e2b7b4987cd07c964e7d92609ae2f5c3cd637b12e5838cf4a645e。批次[清单](../outputs/model_extensions_v1/registry/pipeline_inputs.json)绑定源码、配置、父模型和数据，6个任务依次包含CUDA检查、两次训练与验证选轮、两次锁后历史评分、最终对照报告。两次历史评分共同依赖两个新臂的验证锁。仍为seed42、各8轮、153成熟周三年验证和297成熟周已读历史。

后台启动器等待 p6gap_foxconst_v1 整批进入终态后竞争共享GPU锁。现有实验使用其冻结源码，核验仍通过。原始输入、源码、任务和输出均可由本机目录复查，结果页由队列最终生成，当前收益尚不可比较。真实尺寸CUDA检查为训练前依赖，检查失败会阻断训练。固定幸存股池、回溯行业、代理成交、额外参数与历史样本暴露仍限制结论。原始产物仅本机，未异地备份。

复现入口为 outputs/model_extensions_v1/registry/register.py 和 outputs/model_extensions_v1/launch.py。命令与依赖保存在 queue/inbox，运行状态在 queue/state，逐任务日志在 logs。新脚本 post_registration_checks.json 记录归档及原队列校验，synthetic_gate.json记录测试。实际最终对照将保存为 research_20261115_model_extensions_results.md。
