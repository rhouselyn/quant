# 五seed等权排名集成

执行完成：2026-11-19 Asia/Shanghai，状态complete。六个集成系统、三组合、两费用档及153验证/297历史周全部完成，共16200行周账本、72全期/324年度指标，主程序与独立验收退出码均0。独立核对五成员完整秩、共同池/集成长20排除、费用、单seed对照及CPU与原排序抽样全部通过，见outputs/rank_ensemble_v1/report/audit_completed.json。本批无剩余运行任务，未训练或调参、未扩队列、未部署。结果见[完整逐年与单seed对比](research_20261119_rank_ensemble_results.md)。下文保留执行前登记。

登记：2026-11-19 Asia/Shanghai，planned。父批[五seed结果](research_20261118_rolling_anchor_seeds_results.md)。用户授权B0和替换softIC pair各保留直接排序、全比较、六轮瑞士轮，等权集成42–46五seed排名，检验回撤变化，报告逐年收益/MDD/Sharpe/换手及各自单seed对照。

不增加或重训主干/裁判，不重选轮、不筛seed、不按历史调权。父批各seed对应的2010/2014滚动样本外训练、固定历史锚点坐标及完整2018最终表示全部复用；保持408成熟元训练周、153验证周/297历史周。父批全历史已读，本批是用户指定的探索性历史复检，不声称未读确认性测试。

## 固定集成定义

对每父、赛制、决策周、腿，在相同事前可用股票池P内，将每seed排序转为最佳为1的唯一秩r_s(i)，并计算q_s(i)=(r_s(i)-1)/(N-1)，N=1时为0。集成按q_bar(i)=sum_s q_s(i)/5升序，最终并列按symbol升序。原seed的分数并列也沿其原symbol规则。所有seed池必须逐股票完全一致，缺排名拒绝而不取交集或重归一化权重。因为各seed同池，归一化秩平均与原始秩平均等价；使用秩而不是logit/收益/持仓权重平均。

直接排序各seed先沿原完整验证锁的多空三轮融合，长按score降序、短按score升序，然后集成。精排各seed仍分别使用自己的向量与已锁定多空裁判；全比较与Swiss6仍共享同一已选head，Swiss配对seed42规则不变，不平均或混合不同seed向量/权重。

先对完整多池集成排序选Top20。S10在原完整空池分别排名再集成；LS50先将集成后Top20从原空池排除，再让每seed裁判在这一共同剩余空池重新执行对应赛制、集成选Bottom10。不能平均旧各seed LS50缓存，因为其空池排除的是各自不同多头名单。流动性前40%在原全截面计算后再排除多头，原资格和留现金规则不变。L20只使用集成长排名。

## 回放、对照与验收

六个主要集成系统：B0/PAIR_REPLACE × direct/full/swiss6。每系统均回放L20/S10/LS50，10bp交易＋8.35%年借券，以及30bp＋12%压力档。沿原run_book执行与连续持仓账本，不平均单seed净收益。年度收益为累计净收益，年度MDD初始1，Sharpe按周sqrt52，换手为真实连续权重绝对变动周均。2026为部分年，最后决策10月23日、退出11月2日。

对照复用父批五个单seed的同父/同赛制/同组合/同费率账本；同时列均值、median、min/max和每个seed，统计集成的收益/Sharpe较高、MDD/换手较低的个数，并做原8周块配对区间。不把单seed指标均值当成种子集成净值。主要问题是实际集成MDD是否下降、下降是否牺牲收益，保留所有年份与费用，不按测试挑集成规则。

执行使用CPU推理，无GPU训练或占用其他批次锁；原裁判无dropout/batchnorm，评估模式、分块与排序规则相同。验收包含输入/模型/锁/向量hash、同池和五成员完整覆盖、秩方向和单成员复现、交换seed顺序不变、排除最终多头后重算空侧、周数/成本/年度指标独立重算、单seed基准原产物一致。

实现scripts/replay_rank_ensemble.py，输出outputs/rank_ensemble_v1：输入清单、规则锁、各seed池内秩与集成排序、ledger、metrics/yearly/comparisons/paired及complete。旧产物不覆盖；按输入身份和阶段缓存恢复，源码冻结后不改。命令`python scripts/replay_rank_ensemble.py --self-check`及`python -u scripts/replay_rank_ensemble.py > outputs/rank_ensemble_v1/queue.log 2>&1`。预计CPU推理/回放数十分钟，按吞吐更新。不额外试集成权重、子集、风险配比或扩训练队列；仅本机、未异地归档、未部署。
