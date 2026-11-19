# 实验改动树的维护

[打开实验树](experiment_tree.html) · [读取结构化数据](experiment_tree.json) · [逐臂完整报告](research_20261119_experiment_comparison_results.md)

浏览器直接打开实验树即可。展开「共同基线」中的 B0、P4、P6 或旧 BCE，可从「作为对照」「被复用」等反向关系跳到各批。展开「进行中与待验证」可跳到同一个实验节点。

## AI 读取顺序

1. 读 [experiment_tree.json](experiment_tree.json)，按 `nodes[].id` 定位实验。按 `group` 浏览主题，按 `relations[].target` 找下游依赖的对象。反查 B0 时搜索所有 `target = b0-q1` 的关系，并查看类型和 `scope`。
2. 根据 `status` 区分已完成、部分、归档、失败和待验证。队列以 `status_as_of` 为盘点时点，回答当前进度前按原批核实状态、日志、进程与最终产物。
3. 先看 `notes` 与 `effects` 的匹配条件，再按 `sources` 读冻结协议、结果和可用原件。共用基线可以串联研究线索。计算跨批差值前仍需核对快照、日期、seed、选轮锁、实际名单、资本分母与费档。

## 关系类型

关系从当前节点指向 `target`，一项只登记一次。页面自动生成反向索引。树上所在分组只表示主题。

| `type` | 何时登记 |
| --- | --- |
| `compares_to` | 源实验使用目标作为对照。说明它是同口径配对、当期单臂对照，还是仅作历史参照。 |
| `derived_from` | 继承目标的配方、管道或实验设计。 |
| `reuses` | 复用目标的权重、分数、向量、名单、选择锁或账本。写明复用对象。 |
| `combines` | 目标是集成、信号组合或执行组合的成分。 |
| `part_of` | 源是同一批次中的实验臂或因子切片，计数时可追溯到批次。 |
| `follow_up_to` | 后续实验处理目标的一项发现或证据缺口。 |
| `waits_for` | 协议或队列设定的前序任务依赖。 |
| `waits_for_gpu_lock` | 某次状态盘点中的资源锁等待，`scope` 要写清持锁对象和时点。 |
| `cites` | 原协议或分析引用目标，证据只到引用关系。 |

## 新增一项实验

1. 按[研究记录与归档规则](research_recording.md)写好协议或结果。选已有 `group`，在 JSON 的 `nodes` 添加一项。新 `id` 使用稳定的小写字母、数字、连字符，名称可独立调整。已登记实验从等待转为完成时更新原 ID。
2. 新条目写 `id`、`name`、`group`、`kind`、`status`、`notes`、`sources`。`kind` 可用 `baseline`、`experiment`、`audit`、`archive`。可加 `batch_id` 指向同批登记 ID。状态可用 `planned`、`waiting`、`running`、`partial`、`complete`、`failed`、`deferred`、`superseded`、`documented`、`unknown`。运行、等待和部分完成须填带时区的 `status_as_of`。
3. 在 `relations` 写已证实的关系，可关联多条基线。每项填 `type`、`target` 和简要 `scope`，包括关键控制与例外。只有计划的项目写待验证问题、缺臂及已有协议。已知影响写到 `notes` 下的结果和复现说明。主要量化对照可在 `effects` 追加 `metric`、`condition`、`unit`、`control`、`candidate`、`delta`、`interval_95` 和 `source`，具体维度以原协议为准。缺失数值留空。
4. `sources` 每项填 `kind` 和仓库相对 `path`。类别可用 `protocol`、`report`、`archive`、`metrics`、`lock`、`state`、`log`、`manifest`。引用的效果表也要登记为来源。被 Git 忽略的本机产物应同时保留可随仓库分发的协议或报告入口。
5. 在仓库根运行以下两条命令。手工维护 JSON，页面由脚本生成。更新期间实际核对了哪批状态，就更新该节点的 `status_as_of`；`inventory_as_of` 和 `queue_as_of` 分别记录本次覆盖的研究/队列截点。

```powershell
python scripts/render_experiment_tree.py
python scripts/render_experiment_tree.py --check
```

提交 JSON、新生成的 HTML 及相应的研究记录。校验会检查重复 ID、关系目标、来源路径和 HTML 是否和 JSON 同步。本机原件的存在与内容以各批验收为准。一个已完成的多关系例子可查看 `rank-blend`，待验证例子可查看 `p4-factor-replication`。
