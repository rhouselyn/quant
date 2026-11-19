# 完整2018主干滚动锚点精排：五种子配对复现

执行完成：2026-11-19 Asia/Shanghai，状态complete。30个所需主干中复用8个、新训22个，各新训8轮；新增16个多空裁判各12轮。所有新增选择锁先于新历史导出，五seed共108000周账本，480全期/2160年度指标及跨seed统计独立验收通过，主队列和验收退出码均0。本批无运行或剩余队列任务，未扩seed/超参/日期、未部署。结果见[逐年四指标与稳定性](research_20261118_rolling_anchor_seeds_results.md)。下文保留执行前登记及运行修订。

日期：2026-11-18 Asia/Shanghai。状态planned，在新增seed历史结果前登记。用户授权B0与替换softIC的pair主干各五seed，保留直接排序、全比较、六轮瑞士，逐年四指标和跨seed稳定性；父批[单seed结果](research_20261118_rolling_anchor_rerank_results.md)。

## 固定矩阵与预算

seed固定42、43、44、45、46，B0与PAIR_REPLACE逐seed配对。42为发现阶段，复用已验收三截止主干/裁判/账本；主汇总同时列五seed和新增43–46四seed，不挑seed、不加集成。B0原损失rankIC=.5/random_pair=0；REPLACE为rankIC=0/random_pair=.5，其它配置沿各自父批。每主干8轮、batch16、AdamW和原调度、580股票/30特征/40周/128宽/4层。主干、两个OOF前缀、随机pair采样和裁判初始化/训练排列均使用对应seed。裁判每腿12轮、AdamW lr.001/wd.01、同架构及全pair逻辑损失，不加预算。

每seed每父需要2010、2014、2018三个截止模型，共30个。42六模型复用；43/44的B0完整2018模型已做committed epoch1..8、配置、冻结预处理、评分/锁/hash和源代码默认路径审计，配置与父42除seed/source_sha外相同。源变更只增加默认关闭分支，见outputs/rolling_anchor_seeds_v1/reuse_audit.json，仅读test sidecar/hash不读收益。合计新增22个主干及16个多空裁判。42原B0是legacy档案，无完整恢复状态，复用既有已验收账本而非声称可精确续训。

## 划分与表示

原BQ_F0_r2训练快照、前缀冻结预处理及辅助监督成熟预留不变。每seed自己的2010截止模型预测2010–2013的204周，自己的2014截止模型预测2014–2017的206周；410输入周、408在2018前成熟监督周。按原监督mask训练，缺未来收益不删forward候选。2018最终模型只生成153周2018–2020验证和297周2021-01-08起历史，最后决策2026-10-23退出2026-11-02。验证不参与主干梯度，元训练无2018模型参数；不换验证范围。数据快照与原批一致。

128个2008–2009历史输入锚点完全沿父批anchor_keys.csv（抽样seed42），所有训练seed/折/父使用相同keys；每模型分别用自身预处理/参数编码股票及锚点，精排输入为128维余弦相似度。各坐标具同一历史语义，正交旋转/置换不变；相似结构漂移仍保留诊断，不按测试调映射/锚点。主干向量不直接跨折拼接，也不通过最终模型重新生成OOF训练表示。

## 选轮、排名与费用

每seed每父原头各腿按153周完整验证净Sharpe选单轮、取邻近三轮融合；新增直接评分用既有score_dataset_standard/evaluate_dataset_three_year接口。精排最终向量固定2018第8轮，并额外报告e8_direct同编码器对照。裁判多空按全比较完整验证10bp净Sharpe选12轮内单轮，平局早轮。各父full/Swiss严格同head；不同seed裁判各自训练。

所有新增seed的主干直接选轮及裁判选择锁全部完成后，才开始新历史向量/评分/回放。42旧历史已知；新增43/44旧B0直接test评分只在该门通过后复制使用。全比较穷尽全可用池股票对，logit行均值；Swiss6固定symbol和首轮配对seed42，后续近分/未交手优先，重复边不当新增证据，稀疏图ridge .01聚合。固定瑞士随机数使本轮seed差异主要来自训练。

保留L20/S10/LS50，20多/10空、LS50两侧各50%资本；空池历史资格和全截面成交额前40%、组合排除最终多头。交易10bp配年借券8.35%，压力30bp配12%。周账本真实日期和原现金/缺失估值/受阻成交代理不变。Sharpe按周sqrt52，MDD含初始1，周均换手为连续权重绝对变动和。年度收益是累计，年度MDD重置1但换手沿真实连续持仓；2026标非全年。

## 稳定性与判决

逐seed及逐年列收益、MDD、Sharpe、换手；全期CAGR另列。各路径在all5与new4上汇总均值、样本std、中位数、min/max；同seed配对REPLACE减B0同赛制、精排减同父direct/e8、Swiss减full，列CAGR与Sharpe正增量个数、MDD/换手改善个数、差值均值和离散度。对各seed及seed平均周收益序列做原8周块配对95%区间，不能把seed当独立市场样本，不能将跨seedstd当市场置信区间。测试已读且用户因42结果选择REPLACE，本批是探索性稳定性复检，不恢复确认性独立测试。

不以单个最高seed判胜。主要看新增四seed的配对方向、收益/风险一致性、费用压力和不确定性；保留失败和全部结果，不因历史改变矩阵。无自动扩seed、超参、年份、队列或部署。

## 实现与恢复

新增prepare_rolling_anchor_seed_vectors.py负责逐seed三前缀、向量和直接选轮；rolling_anchor_seed_judge.py复用旧Judge并参数化seed、完整原子恢复；run_rolling_anchor_seeds.py持GPU锁串行执行、先全锁后历史、按case缓存账本；report_rolling_anchor_seeds.py聚合与稳定性。旧冻结单seed源不修改。输出outputs/rolling_anchor_seeds_v1，保存源码/src树/输入快照/父模型及锚点hash、state、每模型manifest、向量sidecar、选择锁、账本、comparison数与最终complete。出现输入hash变化拒绝静默恢复，不覆盖原已发布证据。

命令：`python scripts/run_rolling_anchor_seeds.py --self-check`；`python -u scripts/run_rolling_anchor_seeds.py > outputs/rolling_anchor_seeds_v1/queue.log 2>&1`。已有模型和head的完整状态恢复；账本缓存绑定输入/heads锁hash。主脚本state.json记录当前seed/stage，最终complete与退出码/日志交叉验收。预算约2–4小时，按实际吞吐更新。数据、权重和日志仅本机、未异地归档。

## 运行期源差异登记

2026-11-18：运行中外部会话修改src/quant_mvp/dataset_store.py，原全局输入清单保留，精确旧/新hash及审查保存在source_amendment.json。独立复核其仅可能通过hierarchy_data的daily_path AND daily_weekly_blocks分支到达；本批六模板、30预期配置及实际配置均不启用。run_weekly_architecture的verified_snapshot独立校验输入，不依赖此模块。因此训练、评分和样本边界无变化，继续原队列。最终验收仅允许这一不可达文件的精确登记变更，其它源差异仍拒绝。旧逐run清单未覆盖；若进程重启因原不可变清单拒绝，须按此修订显式核验后恢复，不能静默刷新hash。
