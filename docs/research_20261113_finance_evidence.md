# 金融建模改进研究：证据与边界

日期：2026-11-13，Asia/Shanghai。状态：complete（读取身份登记，不代表候选实验完成）。
用户目标：以金融收益生成假设组织架构和损失，提高预测；本轮多agent研究与文档，不改代码、不训练、不推理、不回测、不修改队列。

## 主对象与历史

主对象为Q1 standard_v1复用的B0及CS/CB/ADN。旧weekly_architecture_v1的P64和CS×P64另属训练cohort；新版[管道结果](research_20261113_pipeline_completion_results.md)已经记录其评估完成，不能沿用前轮“仍待完成”的快照。本文没有核验其他会话实时进程，不报告其当前是否运行。

HEAD为`f2e198e8769ab960b7c8d6a8aa2ba50eb1bb8fe3`；工作树原有大量修改，以下实际哈希比HEAD更准确。旧研究[数学事实底稿](research_20261113_math_facts.md)提供B0形状和标签边界，本轮仍按当前源码核对。

B0: 40×30→128、四层四头Attention，末位置读出、LN/L2线性score；行业市场关系门冻结零。M6已存在且按训练前缀固定缩放，其绝对状态不被逐期横截面z消去；它在当前B0评分路径被零关系门隔离。已有market_film不等于新金融模型，旧M1/M1F1结果须保留边界。原损失tail BCE1+softIC.5+InfoNCE.1+行业η²1。

所选输入为BQ_F0_r2，779训练锚2002-07-19—2017-11-24，末未来5周上下文2017-12-29，训练边界2018-01-01。验证2018-01-05—2020-12-25共153成熟周；2021-01-08—2026-10-23共297周为已暴露历史复检，最后退出2026-11-02。未来新模型即使移除h5辅助，也先保留共同779训练锚，不能把额外训练周混入模型贡献。

## 本轮关键实现证据

- `hierarchy_data.py:110` arrays返回X/y/valid_y/M/available等位置元组；M6训练缩放见232–241，原始标签与训练loss截断分离。
- `hierarchy_engine.py:41` make_batch已经传入M[t]和当前available；`:76` predict消费四元组并仅导出score；`:396`固定构造HierarchicalRanker；`:595`以后汇总旧目标；metadata仍含固定旧架构文本。
- `score_dataset_standard.py:73–82`固定构造旧模型且只保留date/symbol/score/epoch列，新增概率字段不能只在forward添加后期待自动保存。
- `eval_weekly_architecture.py:32–58`逐epoch、窗口、seed做z融合；这保留排序惯例，不能视作收益单位均值或概率分布混合。
- `training_state.py:32–81`把数据、源码、环境和完整cfg绑定恢复身份；新模型/新目标不能从旧weight-only权重冒充精确续训。
- `live_score.py:105`及run_weekly_architecture多个导出入口固定旧模型构造；新研究model_type须明确分派或明确拒绝未支持入口，不应自动部署。

## 研究与来源纪律

A收益分解、B市场状态、C概率目标分别写独立报告；D综合审查核对公式与工程接口。原始论文主张、当前代码事实、拟议默认参数、本项目提升机制假设分别标记。所有候选均未实施。合成算术若使用，只证明公式性质，不是收益证据。

原始data/outputs仅本机、未在本轮外部归档。主协议、现有报告与队列均保留；新增方案不自动切换数据集或扩训练。最终集成定义以新的金融模型设计/协议为准，早期单方向建议差异在综合裁决中说明。

## 读取时哈希

| 读取文件 | SHA256 |
| --- | --- |
| [src/quant_mvp/model.py](../src/quant_mvp/model.py) | `630e0ecb67f0db1ae926f82eb8465df578ffcfb508189f5d734988fe3223276a` |
| [src/quant_mvp/hierarchy.py](../src/quant_mvp/hierarchy.py) | `98b5dae2f295b57cec5568ba13b05cfb9798fa0a5791bac6501df09e2e955850` |
| [src/quant_mvp/hierarchy_data.py](../src/quant_mvp/hierarchy_data.py) | `2faf89efe5c55af9b746d8123e46204f932dea1e2a8916d094d054caff046796` |
| [src/quant_mvp/hierarchy_engine.py](../src/quant_mvp/hierarchy_engine.py) | `b92e39aa3028a3bbce5c64b13f0fb77e3efb428613189d08ac6d843ea515764b` |
| [src/quant_mvp/training_state.py](../src/quant_mvp/training_state.py) | `6db9941c2a44b7f2019407b149aab44d2622fd6c01cb23ef457ebd45650e685f` |
| [src/quant_mvp/live_score.py](../src/quant_mvp/live_score.py) | `07d288b1a0f232953ad03f97f11ef0afa05df8a0c3ea2b94fdb4ded5923288c7` |
| [scripts/run_weekly_architecture.py](../scripts/run_weekly_architecture.py) | `b6716233f8175bb044981610806c919bf08366674a3818ee06a87b4d66c7b540` |
| [scripts/score_dataset_standard.py](../scripts/score_dataset_standard.py) | `387e57e7d3f1ed55501fd75a676eb40ab598a65c712810c8b431fb1c97f30c23` |
| [scripts/evaluate_dataset_three_year.py](../scripts/evaluate_dataset_three_year.py) | `484599afd854ed4e70ea3395c21ed0530ce672aa9bbc35c25258afc4cce268d2` |
| [scripts/eval_weekly_architecture.py](../scripts/eval_weekly_architecture.py) | `2b1696fafe99dd5ab17b0a38219e9a7906b06bce564150b879e8437822f596d7` |
| [outputs/dataset_v2/exploratory_F0/runs/research_B0_F0_s42_a1/config.yaml](../outputs/dataset_v2/exploratory_F0/runs/research_B0_F0_s42_a1/config.yaml) | `bd57708c8ae8d950f290be47ff1129483fba910434ae6153d4aac588f81c71a0` |
| [outputs/dataset_v2/standard_v1/selections/research_B0_F0_s42_a1/lock.json](../outputs/dataset_v2/standard_v1/selections/research_B0_F0_s42_a1/lock.json) | `0c02bbbd91c727fc204600863498d2820cdbf2d116ae57525b15477953029662` |
| [outputs/dataset_v2/snapshots/BQ_F0_r2/manifest.json](../outputs/dataset_v2/snapshots/BQ_F0_r2/manifest.json) | `46852cd7bf4fa23dcf5007d417caf9825442ca51d495a01a4b10df08e3faefb1` |
