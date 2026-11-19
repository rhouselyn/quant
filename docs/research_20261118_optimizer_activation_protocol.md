# Muon与零参数激活组合：夜间GPU队列

2026-11-18 Asia/Shanghai。planned。用户明确授权查论文、规划、试优化器/激活及组合，直接排GPU队列，希望醒来可读结果。[一手研究证据](research_20261118_optimizer_activation_evidence.md)。本页在新臂验证/历史读取前冻结设计，不因夜间无人确认而改变数据或扩队。

## 实验矩阵

父模型为原B0（group_fox_v1登记research_B0_F0_s42_a1）和原P4（finance_faithful_v1/runs/research_P4_F0_s42_a1）。各父AdamW/Muon与GELU/SiLU/SquaredReLU的完整2×3因子设计：原AdamW+GELU复用经一致性检查的父产物；新训MUON_GELU、ADAM_SILU、ADAM_SQUARED、MUON_SILU、MUON_SQUARED，共10新臂。P4原组内pair为0，B0原softIC等保留；不叠加random/global pair。P6/P4正在排队的global pair任务先完成，新批不得打断或占用原GPU锁。首轮不增P6优化器矩阵，不测StarReLU/SwiGLU/其他容量变化或更多优化器。

每臂8轮seed42，从同父初始化种子完整训练；30特征、580固定股池、40周窗口、128宽4层、batch16、学习率.0008、weight_decay1e-5、所有父数据/标签成熟/五周辅助预留和优化步数不变。实际配置从父run复制，不依赖文档猜测。全套新臂不做超参搜索，Muon采用RMS匹配后同父lr，仅称固定配方迁移，不声称每个优化器最优或梯度强度严格匹配。

## Muon及激活定义

本机torch2.5.1无内置Muon，不升级依赖。混合Muon+AdamW：只有时间编码器attention block的q/k/v/gate/o投影和FFN两层的隐藏nn.Linear.weight用Muon。输入投影/tokenizer、embedding、最终输出投影、独立主干之外各均值/尺度/预测头、所有bias/normalization/向量/标量保持原AdamW。二维不等于自动Muon；保存完整名称/形状/route和覆盖断言。

Muon beta=.95、Nesterov、NS五步，系数(3.4445,-4.775,2.0315)，Frobenius norm+1e-7，按短维Gram选择是否转置；CUDA bfloat16 NS、CPU自检float32。更新乘Kimi RMS形状因子.2*sqrt(max(rows,cols))，decay仍基础lr。保持grad=None跳过衰减与动量，不原地污染grad。原参数组顺序、slow lr规则、金融faithful均值/尺度独立clip不变。state_dict完整保存Muon/Adam状态并与模型/随机数恢复。有限NS不是精确正交化，不作错误正交断言。

SiLU=x*sigmoid(x)，SquaredReLU=relu(x)^2；均只替换时间encoder.blocks的FFN中间GELU，每个父均4个模块，保持Linear形状/初始化和参数量。其他GELU、attention softmax、QK RMSNorm、门控、概率正尺度变换均不变。model factory cfg启用，训练与各期评分必须同结构，保存替换计数及模块路径。默认cfg不启用时直接原AdamW/GELU，须逐位兼容验收。

## 数据、选择与报告

原BQ_F0_r2、训练截止2018-01-01，完整2018–2020验证153成熟周。每臂每腿按10bp净Sharpe选最佳单轮±1三轮融合，平局早优，先锁config/preprocessing/checkpoint/分数hash再评分2021-01-08起297成熟历史周。历史已反复暴露，探索性，不能按历史选激活/lr/追加轮数。历史最后决策2026-10-23、退出2026-11-02，2026非全年。不同臂独立验证选轮衡量整个训练选择系统。

费用单边10/30bp，年借券8.35%/12%，原资格/流动性/同股/现金/执行缺失代理不变。L20/S10各自全仓，LS50各半名义预算。报告每臂全期净CAGR/初始1MDD/Sharpe/周均双边换手、费用/耗时/显存，逐年累计净收益/年内MDD/Sharpe/换手，以及8周块相对父对照区间；P4另报NLL/RMSE/90覆盖/RankIC。固定幸存股池、回溯行业、供应商复权和执行/券源代理限制继承。全部臂及失败保留，不只公布历史赢家。单seed不当独立市场样本，不能以多个比较中最好一项当确认结论。

## 实现、持久队列及恢复

scripts/prepare_optimizer_activation.py准备outputs/optimizer_activation_v1/frozen_source，不改当前src或P6/P4已冻结来源。scripts/run_optimizer_activation_queue.py负责准备验证、持久supervise与串行run。必须等待p4_global_pair_v1/report/complete.json且共享GPU锁消失；依赖失败则记录dependency_failed并停止。隐藏后台Python监督独立于对话/agent存活，启动记录PID、stdout/stderr与queue/state.json；每臂仍exclusive open原outputs/dataset_v2/gpu.lock，绝不删除他人锁。可以CPU研究/轻验收，不并发重GPU。

每个训练/评分/选轮命令与hash、状态、失败原因落盘；子任务失败可继续其它预注册臂但保留failed，不调参救收益。共享数据/源码身份或安全门失败应停止整批，不能继续误比。原training_state恢复模型/优化器/teacher/随机状态，不以只加载权重续训。已锁回放允许修复派生文件但不能重选；失败任务不标complete。source快照发布后禁止修改；新修订另存并记录与旧manifest关系。

自动发布机器CSV/周账本/逐年表/概率诊断、结果页docs/research_20261118_optimizer_activation_results.md和验收摘要，INDEX指向协议与结果。全部完成才write complete.json；partial/failed报告也必须清晰。仅本机、未异地备份，不自动提交/推送/部署。

## 训练前门槛与资源

必须验证原AdamW fallback逐位一致，Muon参数路由唯一/无embedding误入，gradNone参数与状态不变，矩阵各种shape/zero有限，NS/Nesterov/缩放/decay单步参考一致，保存恢复连续对照（两优化器状态），默认GELU参数/RNG/输出兼容，实际B0/P4激活替换计数/前后向及P4梯度隔离。CPU与CUDA数值路径差异明示，不要求跨设备逐位。先CUDA有限/恢复烟测，不能用短训收益选择配方。

当前P6首轮约279秒、P4未实测，预计本批10臂另需约3–6小时，串行依赖还会增加等待；实际每轮耗时修正ETA。不保证固定起床时间完成；夜间报告始终展示已完成/运行/失败和剩余队列。遇到不可恢复故障保留证据，不能伪造结果或降预算。停止条件为上述十臂均尝试并完成/明确失败归档，不扩大队列。

准备验收更新：CPU数值参考、AdamW回退、gradNone、真实B0/P4完整训练状态恢复及激活检查通过，证据cpu_selfcheck.json。根代理另核49输入hash、336冻结文件及10路由清单；每个Muon臂路由28个隐藏矩阵/720896元素，其余保持AdamW。实际cfg名为optimizer与temporal_ffn_activation。持久隐藏监督已启动，PID记录supervisor.pid，先等待依赖；CUDA gate尚未运行，必须待P4完成且获取GPU锁后通过才训练。supervisor stdout/stderr、queue/state.json与逐臂logs持续落盘，不能将waiting称为已训练。运行入口python -u scripts/run_optimizer_activation_queue.py --supervise。队列依赖失败或全局门禁失败明确停止并写结果页；不删除其他任务锁。
