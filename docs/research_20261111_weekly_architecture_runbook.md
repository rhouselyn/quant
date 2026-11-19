# 周频选股架构实验执行手册

本手册供接手 agent 从仓库根目录独立完成实现、首筛、复检和验收。对照取 `outputs/abl2000_base/config.yaml`。运行产物统一放 `outputs/weekly_architecture_v1/`，每次训练一个新目录。保留现有运行、配置和线上打分入口。本文所列的新配置键、runner 和代码片段由执行 agent 实现。研究动机及旧实验见 [架构研究](research_20261111_weekly_architecture.md)。

## 1. 执行顺序与完成条件

| 顺序 | 执行 | 进入下一步的条件 |
| --- | --- | --- |
| 0 | 建开发快照、登记数据和代码、实现共享回放、核对关开关时的旧模型 | 快照截止、旧权重 strict load、周键及回放手算均通过 |
| 1 | 训匹配对照 B0，随后串行跑 W20、W80、Q0、H32 的种子 42 | 每臂 8 个 checkpoint，开发三账本及成对统计齐全 |
| 2 | 跑 A2、P64、G256、F 的种子 42 | 辅助目标、激活分支和冻结的专项断言通过，逐臂出开发结果 |
| 3 | 仅对通过筛选的收益臂补 43、44。按触发条件跑 Q2、A1、C3 或 CS | 每个晋级声明绑定账户、成本档和比较窗，有三个种子及融合结果 |
| 4 | 对晋级方案运行两两组合，继续用开发窗选择 | 每个组合同时对比 B0 和两个单臂父项。至少一项通过后进入更高阶组合 |
| 5 | 冻结名单与规则后读取历史复检，输出验收及前瞻登记 | 所有已跑、失败、阻塞和未晋级项均有状态和证据 |

### 1.1 冻结的试验矩阵

| ID | 在 B0 上的改动 | 预定参数 | 触发条件 |
| --- | --- | --- | --- |
| B0 | 匹配样本的四层基线 | `lookback=40`，四层四头，宽 128，线性周投影和 score | 必跑 |
| W20 / W80 | 序列长度 | 20 / 80 周。自然覆盖门槛 16 / 64 | 必跑 |
| Q0 | 全部层只更新最新周 | `anchor_query_from=0`，K/V 取初始周 token | 必跑 |
| H32 | score 加残差 MLP | `score_residual_width=32`，`128→32→1` | 必跑 |
| A2 | 加下一周 latent 点预测 | `next_latent_start=2`，`next_latent_width=64`，权重按 §4.5 | 必跑 |
| P64 | 周投影加残差 MLP | `weekly_residual_width=64`，`30→64→128` | 必跑 |
| G256 | 每个 block 的 FFN 换 SwiGLU | `attn_swiglu_width=256` | 必跑 |
| F | 按 B0 选轮冻结学生输入 Linear | `freeze_input_after_epoch=max(1,E*−1)` | 必跑 |
| Q2 | 先更新全序列再只更新锚点 | `anchor_query_from=2`，K/V 取前两层输出 | Q0 的实际中位用时降 ≥25%，或 L20 开发净差 ≤−5 bp/周，用于区分效率与历史上下文化 |
| A1 | 提前加辅助监督 | 与 A2 同目标、同最大权重，从 E1 拉起 | A2 任一主账户 2020 净差 ≥+5 bp/周，并通过持久性检查 |
| C3 | 首层之前加时间卷积残差 | `weekly_dw_kernel=3`，`128` 组，左侧补 2，步长 1 | 完成核心臂后可作为局部路径对照。若预算有限先跑 Q2 或 A1 |
| CS | 锚点跨股票摘要 | 每股读取同周行业和市场的留一均值，见 §4.7 | 单股实验完成后，单独登记横截面假设 |

首筛 1 个对照加 8 个候选训练。八轮旧训练约 14 分钟，仅供估算。训练、显存测量和 CUDA smoke 在同一设备串行。候选强制从随机初始化训练。同种子的随机种子、日期洗牌、batch、优化器、数据统计、训练轮一致。新模块初始化保留共有模块的随机数流。

## 2. 建开发快照与共同样本

1. 在根目录运行 `python -c "import torch, pandas, pyarrow, yaml; print(torch.__version__, torch.cuda.is_available())"`，保存 Python、Torch/CUDA、GPU 名称、当前提交和相关文件的 SHA-256。盘上已有 `2.5.1+cu121` 可用环境的核对结果。若换机器，锁定该机器的环境并同机重训 B0。开跑前 `git status --short`，登记已有改动，按相关文件保存差异或哈希。
2. 读取 `outputs/abl2000_base/config.yaml`，以日期定位边界。当前周历共 1,399 个锚点。训练分界是 `2018-01-05`（索引 939），历史复检首个决策锚点是 `2021-01-08`（1096）。开发文件保存索引 `0..1095`，末两条锚点只供前面标签成熟。预测输出最多到 `2020-12-18`（1093），使 `t+2` 键小于 1096。先验核对与文件不符时，按实际键定位并在 `prereg.md` 冻结，切点后的标签隔离要求照用。
3. 对开发阶段的融券文件，只加载快照日早于 `2021-01-08` 的文件。开跑前确认周数据 `(date,symbol)` 唯一、日期升序、580 个股票、特征列顺序与配置一致。保留每个周历键，包括休市周。另以决策日前信息生成统一 `evaluation_dates.csv`，包含 `anchor,entry_week,exit_week,phase,has_market_bar`。只排除没有决策的全市场休市锚点和标签尚未成熟的尾锚，记录排除原因。

将下列代码存为 `scripts/weekly_architecture_snapshot.py`，从根目录执行 `python scripts/weekly_architecture_snapshot.py`。原缓存路径只用于读取。新的候选 `config.yaml` 从生成的模板派生。

```python
from pathlib import Path
import shutil

import pandas as pd
import yaml

root = Path("outputs/weekly_architecture_v1")
dev = root / "dev_data"
dev.mkdir(parents=True, exist_ok=False)
cfg = yaml.safe_load(Path("outputs/abl2000_base/config.yaml").read_text(encoding="utf-8"))
cut = pd.Timestamp("2021-01-08")
panel = pd.read_parquet(cfg["cache_path"], filters=[("date", "<", cut)])
panel["date"] = pd.to_datetime(panel["date"])
dates = pd.DatetimeIndex(sorted(panel.date.unique()))
assert not panel.duplicated(["date", "symbol"]).any()
assert panel.symbol.nunique() == cfg["n_stocks"] == 580
assert panel.observed.notna().all() and dates[-1] < cut
assert dates[939] == pd.Timestamp("2018-01-05") and len(dates) == 1096
assert set(cfg["feature_names"]) <= set(panel.columns)
panel.to_parquet(dev / "weekly.parquet", index=False)
market = pd.read_csv(cfg["market_path"], parse_dates=["date"])
market = market.loc[market.date.isin(dates)].sort_values("date")
assert pd.DatetimeIndex(market.date).equals(dates)
market.to_csv(dev / "market_features.csv", index=False)
shutil.copyfile(cfg["universe_path"], dev / "universe.csv")
cfg.update(cache_path=(dev / "weekly.parquet").as_posix(),
           market_path=(dev / "market_features.csv").as_posix(),
           universe_path=(dev / "universe.csv").as_posix(),
           n_days=len(dates), train_end=939, valid_end=len(dates) - 2,
           save_epoch_checkpoints=True, save_epoch_scores=False,
           selection_label_lag_weeks=0, common_training_horizon=5,
           matched_lookbacks=[20, 40, 80], min_train_anchor=80)
(root / "base_dev.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
```

`valid_end` 是预测区间右端的排他索引，设为 1094 后引擎已经不看最后两个未成熟的信号锚点，所以模板把 `selection_label_lag_weeks` 置零。引擎产生的 `best.pt` 留作运行记录，首筛和 F 的定时用 §5 的规则。兼容旧回放的日期检查以实际索引为准，校验 §4 的训练代码没有另一次重复裁尾。

### 2.1 共同覆盖与训练边界

`src/quant_mvp/hierarchy_data.py::arrays` 在 `available` 原始布尔矩阵生成后、第一次 `rolling` 前保留 `raw_available`。用下面逻辑替换现有 coverage 两行。`X` 的特征拟合、周内横截面标准化和缺测前填照旧，`available` 只决定打分、关系 mask 和有效样本。保留 tuple 返回和索引，更新 `preprocessing['availability']` 以记录实际规则。

```python
raw_available = observed & np.isfinite(raw).all(axis=2)
history = pd.DataFrame(raw_available)
lookback = int(cfg["lookback"])
coverage = history.rolling(lookback, min_periods=lookback).sum().to_numpy()
available = raw_available & (coverage >= int(cfg.get("min_history_weeks", 32)))
matched = cfg.get("matched_lookbacks")
if matched:
    assert lookback in matched and all(int(width) > 0 for width in matched)
    available = raw_available.copy()
    for width in matched:
        width = int(width)
        count = history.rolling(width, min_periods=width).sum().to_numpy()
        available &= count >= (4 * width + 4) // 5
```

在 `hierarchy_engine.train` 创建 `train_dates` 时，用 `max(cfg['lookback'], cfg.get('min_train_anchor', 0))` 做循环起点，结束仍为 `cfg['train_end'] - reserve`。把 `train_dates`、`available` 的 SHA-256、每周监督有效股数写入核验。B0 和所有候选须有相同训练锚点、股票 mask、主标签及 InfoNCE 五周有效对。W20/W80 只改 `lookback`、`context_lengths: [W]`、`min_history_weeks: 16/64`，匹配筛选模式下 `available` 仍用三窗交集。序列窗口最后一行是 `X[t]`。将 `x[t+1:]` 改掉应保持当前 `make_batch` 与 logits 不变。

这轮匹配基线从第 80 个索引起训练，并使用更严格的股票覆盖，旧权重只用于功能与历史口径核对。W 胜出时，另用原 40 周、候选各自自然门槛并重训对应对照，报告覆盖、首次进入股票数、少历史股票和原训练窗恢复后的敏感性。该复检登记为一个单独试验。

### 2.2 运行登记和产物

开跑前在根产物目录写 `manifest.json`、`prereg.md`、`trials.csv`。每次运行记录 `trial_id, arm, parents, seed, phase, changes, eval_epochs, command, data_sha256, source_sha256, state, reason, wall_s, peak_cuda_mib`。`prereg.md` 固定 §5–7 的窗、费用、账户、阈值及因子取值。失败任务用新的 run ID 重试并保留旧日志，配置和对照两边同时更新的更改单独编号。8 轮任务从相同 seed 重跑。现有 `resume=` 会重建 AdamW 和 teacher，不能用于本轮的等价续训。

## 3. 训练和验证入口

创建 `scripts/run_weekly_architecture.py`，复用 `train`、`arrays`、`predict`。下列入口能分别训练与从 checkpoint 导出开发分数。用 `configs` 目录管理候选 YAML，每份文件从 `base_dev.yaml` 拷贝后只叠加种子与矩阵差异。给 F 写冻结轮之前先完成 B0。权重和分数路径均按 run 隔离。

```python
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from quant_mvp.hierarchy import HierarchicalRanker
from quant_mvp.hierarchy_data import arrays
from quant_mvp.hierarchy_engine import predict, train

parser = argparse.ArgumentParser()
parser.add_argument("stage", choices=["train", "scores"])
parser.add_argument("--cfg", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
args = parser.parse_args()
cfg = yaml.safe_load(args.cfg.read_text(encoding="utf-8"))
assert Path(cfg["cache_path"]).resolve() == (root / "outputs/weekly_architecture_v1/dev_data/weekly.parquet").resolve()
assert cfg["save_epoch_checkpoints"] and not cfg["save_epoch_scores"]
if args.stage == "train":
    assert torch.cuda.is_available(), "Use a matched GPU for full training"
    torch.cuda.reset_peak_memory_stats()
    result = train(cfg, args.run, defer_test=True)
    assert result["status"] == "validation_complete"
    peak = torch.cuda.max_memory_allocated() / 1024**2
    (args.run / "resources.json").write_text(json.dumps({"peak_cuda_mib": peak}), encoding="utf-8")
else:
    data = arrays(cfg)
    dates = pd.DatetimeIndex(data[6])
    assert dates.max() < pd.Timestamp("2021-01-08")
    model = HierarchicalRanker(cfg, data[8]).to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    frames = []
    for epoch in range(1, cfg["epochs"] + 1):
        state = torch.load(args.run / f"epoch_{epoch}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        frame = predict(model, data, cfg, cfg["train_end"], cfg["valid_end"], device)
        frames.append(frame[["date", "symbol", "score"]].assign(epoch=epoch))
    scores = pd.concat(frames, ignore_index=True)
    assert not scores.duplicated(["epoch", "date", "symbol"]).any()
    assert pd.to_datetime(scores.date).max() <= dates[-3]
    scores.to_parquet(args.run / "dev_scores.parquet", index=False)
```

```powershell
python scripts/weekly_architecture_snapshot.py
$r = 'outputs/weekly_architecture_v1'
New-Item -ItemType Directory -Path "$r/configs", "$r/runs", "$r/logs"
Copy-Item -LiteralPath "$r/base_dev.yaml" -Destination "$r/configs/B0_s42.yaml"
python -u scripts/run_weekly_architecture.py train --cfg "$r/configs/B0_s42.yaml" --run "$r/runs/B0_s42" *> "$r/logs/B0_s42.log"
if ($LASTEXITCODE -ne 0) { throw 'B0 training failed. Read its log before retrying.' }
python scripts/run_weekly_architecture.py scores --cfg "$r/configs/B0_s42.yaml" --run "$r/runs/B0_s42"
if ($LASTEXITCODE -ne 0) { throw 'B0 score export failed.' }
```

生成 W20、W80、Q0、H32、P64、G256 的 YAML 时逐项叠加下面字典。A2 与 F 在其预检结束后写入数值，Q2、A1、C3、CS 按触发条件生成。补 43、44 时同样生成 B0 与晋级臂的 YAML，每个 seed 只训一次相同配置的 B0。

```python
changes = {
    "B0": {},
    "W20": {"lookback": 20, "context_lengths": [20], "min_history_weeks": 16},
    "W80": {"lookback": 80, "context_lengths": [80], "min_history_weeks": 64},
    "Q0": {"anchor_query_from": 0},
    "H32": {"score_residual_width": 32},
    "P64": {"weekly_residual_width": 64},
    "G256": {"attn_swiglu_width": 256},
    "Q2": {"anchor_query_from": 2},
    "C3": {"weekly_dw_kernel": 3},
    "CS": {"cross_summary": True},
}
```

对每臂用同样命令换配置和路径。smoke 跑 §4 断言与一个真实 train batch 的前后向，先用 `B=1`、原 580 股和原 40 周验证正确性，再用实际 batch 和 W80 检查峰值。若 W80 OOM，先调 `encoder_chunk_size` 或 activation checkpoint，并让比较族使用同一设置，核对输出与训练随机路径受哪些影响。仍 OOM 时重登记降低 batch 的新对照族。记录时间包括数据准备、每轮训练和验证打分。GPU 计时区间前后同步，速度声明用同机器至少三次固定批量测量的中位数。完成门为 `metrics.json.status=validation_complete`、八轮文件齐全、没有 `epoch_scores.parquet` 或测试期预测。

### 3.1 开关与防泄漏 smoke

- 用 `outputs/abl2000_base/best.pt` 在所有新键缺省时 strict load。改动前后在同份开发缓存、相同 40 周及 stock mask 上 eval，要求相同设备 logits 数值一致，记录最大差。若参考旧分数涉及不同价格或 mask，先按相同输入和权重离线复现，再切新账本。
- 每个新键在 `HierarchicalRanker` 初始化时验证。对 `temporal_encoder != attn` 却请求实验键、`query_from` 越界、周长度大于 RoPE 表、负宽度、无 EMA teacher 的辅助臂主动报错。禁用时默认返回原张量和原 state dict 键。
- 将标签、成交标记和晚于锚点的数据改值，校验 `make_batch`、当周分数和订单意向不变。训练样本满足 `t + 5 < train_end`，开发成熟样本满足 `t + 2 < 1096`。候选 YAML 逐键比对，任何未登记差异在开跑前纠正。

## 4. 模型改动与专项验收

各节的代码可分别插入指明的位置。已有 `import torch`、`from torch import nn`、`from torch.nn import functional as F` 时复用。配置 `F` 为冻结臂 ID，代码中的 `F` 为 `torch.nn.functional`。

### 4.1 H32 和 P64 的残差支路

`src/quant_mvp/hierarchy.py::HierarchicalRanker.__init__` 在所有原有模块初始化后建 `self.score_residual`。在 `relate` 的现有 return 前计算 `score`，再相加。保留 `z` 和 `industry` 返回。若有早返回打分路径，搜索所有 `self.score(`，使其走同一打分方法。

```python
width = int(cfg.get("score_residual_width", 0))
self.score_residual = None
if width > 0:
    with torch.random.fork_rng(devices=[]):
        self.score_residual = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, 1))
    nn.init.zeros_(self.score_residual[-1].weight)
    nn.init.zeros_(self.score_residual[-1].bias)
```

```python
score = self.score(z)
if self.score_residual is not None:
    score = score + self.score_residual(z)
return z, score.squeeze(-1), industry
```

P64 在 `TemporalTower` 的 `AttentionEncoder(...)` 调用传入 `weekly_residual_width=int(cfg.get('weekly_residual_width', 0))`。在 `AttentionEncoder.__init__` 末尾以同样方法创建 `nn.Sequential(nn.Linear(input_dim,width), nn.GELU(), nn.Linear(width,d_model))`，末层权重偏置置零。保留 `input_proj[0]` 是原 Linear 以及原参数键。新增方法替代 `_forward_hidden` 的原周投影调用。已有 `block_size` 压缩实验使用原 `_compress`，本手册的臂均用 `block_size=0`，开启压缩和新周支路的配置报错。

```python
def project_week(self, x):
    if self.weekly_residual is None:
        return self.input_proj(x)
    return self.input_proj[1](self.input_proj[0](x) + self.weekly_residual(x))
```

验收 H32 新增参数为 4,161，P64 新增 10,304。初始化时共享参数和 eval 输出与 B0 对齐。第一步确认支路末层有梯度，第二步确认首层梯度非零。P64 的原始投影也用于 5 周 teacher，teacher 为 EMA 拷贝。A2/P64 组合的下一周标签使用 `project_week`，见 §4.5。

### 4.2 G256 等主权重 SwiGLU

在 `model.py` 定义以下 `SwiGLU`，`TemporalTower → AttentionEncoder → _GatedSelfAttentionBlock` 显式传 `attn_swiglu_width`。先按原顺序构造原 GELU FFN，再在 `fork_rng` 里覆盖 `self.ffn`。关闭时 state dict 仍走 `ffn.0`、`ffn.2`。原 dropout、pre-LN、残差、门和 RoPE 保留。`ConvAttentionEncoder` 属于旧实验入口，默认参数继续可用。

```python
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden)
        self.value = nn.Linear(dim, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.out(F.silu(self.gate(x)) * self.value(x))
```

`3×128×256 == 2×128×384`，每层矩阵权重 98,304，偏置增加 128，总共多 512 参数。共有部分初始化逐键一致，FFN 本身按新形状初始化。跑单层和整塔梯度检查，记录四层的输出均方根、训练与推理时间、峰值显存。

### 4.3 Q0 和 Q2 的锚点查询

为 `_GatedSelfAttentionBlock` 增加 `forward_anchor`。复用现有的每层 Q、K、V、门和 FFN。Q0 从第一个 block 起固定 `memory` 为初始逐周投影。Q2 从第三个 block 起固定为第二个 block 的完整输出。`query` 初值为 `memory[:, -1:]`，每次只更新 `[B,1,128]`。K/V 来自包含当前周的 memory，每层仍用本层的 K/V 权重，梯度保留到投影。固定 memory 指同次前向的张量，学生输入投影仍可训练。

```python
def forward_anchor(self, query, memory, positions):
    hq, hm = self.norm1(query), self.norm1(memory)
    b, t, _ = hm.shape
    q = self.q_proj(hq).view(b, 1, self.n_heads, self.head_dim)
    k = self.k_proj(hm).view(b, t, self.n_heads, self.head_dim)
    v = self.v_proj(hm).view(b, t, self.n_heads, self.head_dim)
    q = self._rope(self.q_norm(q).transpose(1, 2), positions[-1:])
    k = self._rope(self.k_norm(k).transpose(1, 2), positions)
    y = F.scaled_dot_product_attention(
        q, k, v.transpose(1, 2),
        dropout_p=self.dropout.p if self.training else 0.0, is_causal=False,
    )
    y = y.transpose(1, 2).reshape(b, 1, -1)
    h = query + self.dropout(self.o_proj(y * torch.sigmoid(self.gate_proj(hq))))
    return h + self.dropout(self.ffn(self.norm2(h)))
```

在 `AttentionEncoder.forward` 和 `_forward_hidden` 追加 `anchor_query_from=None`，与旧参数兼容。对于启用路径，完成 stem 后走下面的层循环，末尾走原 `self.norm` 与 embedding 头。`return_sequence=True` 时只走完整序列或显式报错。`HierarchicalRanker.encode` 仅为学生打分调用透传 `anchor_query_from`，直接编码及外层 `checkpoint(..., use_reentrant=False, **kwargs)` 同时透传。`TemporalTower.forward` 的 EMA teacher 调用保持默认参数，因而仍跑完整 5 周。注意 query 的位置用真实末周，`is_causal=False` 因 K/V 已限定为决策周和过去，矩形 SDPA 采用默认因果掩码会把单 Q 对齐到首周。

```python
positions = torch.arange(h.shape[1], device=h.device)
for index, block in enumerate(self.blocks):
    if index < anchor_query_from:
        h = block(h, positions)
    else:
        if index == anchor_query_from:
            memory, h = h, h[:, -1:]
        h = block.forward_anchor(h, memory, positions)
```

验收全层首筛的 Q 长度为 1、K/V 长度为 W、teacher Q 长度为 5。用 `anchor_query_from=3` 做功能测试，eval 下与 B0 的同权重最后位置输出 `torch.testing.assert_close(atol=2e-5,rtol=2e-5)`，CPU float32 先验收。再扰动第一周输入，确认单 Q 路径输出能变化，反传到早周 K/V 和周投影。检查 `return_sequence` 和关闭选项。Q0/Q2 主比较同样 40 周。

### 4.4 W20 / W80 与 C3

W20/W80 的数据变更见 §2.1，模型只改变输入时间长度。对 `make_batch` 用同一个 `t` 断言 20 周等于 40 周的尾 20 行、40 周等于 80 周的尾 40 行；至少记录各臂相同的训练 batch 数、股票周对数和 5 周 EMA 目标对数。保留 RoPE 的周粒度位置。

C3 在 `AttentionEncoder` 增加 `self.weekly_dw=nn.Conv1d(d_model,d_model,3,groups=d_model,bias=False)`，权重零初始化，用 `fork_rng` 保持共有权重。它作用在逐周投影后、首个 block 之前，学生和 teacher 同一调用路径。只在历史端补零。单改 `k=3`，保持步长 1。

```python
if self.weekly_dw is not None:
    signal = F.pad(h.transpose(1, 2), (2, 0))
    h = h + self.weekly_dw(signal).transpose(1, 2)
```

测试初始化等价、首层梯度、不同长度下长度不变、把 `t+1:` 改值后 stem 的 `:t+1` 完全相同。组合 Q0+C3 的 Q/K/V 读取已经经过因果卷积的 stem，结果名称要体现这一点。

### 4.5 A2 / A1 的下一周 latent

在 `HierarchicalRanker.__init__` 原模块完成后，按 `next_latent_width>0` 初始化独立 `self.next_latent = nn.Sequential(nn.LayerNorm(128), nn.Linear(128,64), nn.GELU(), nn.Linear(64,128))`。初始化用 `fork_rng`。此头读取 `train` 已得到的 `temporal`（关系层前）和同股 `X[t+1]`。既有 `nextweek_head_weight` 九维 E2 分支及五周 InfoNCE 保持原逻辑。`torch.no_grad()` 中用同一个 teacher 的 `encoder.project_week` 编码 `t+1` 单个周点，避免一个 5 周目标被替换。为 P64 的缺省调用先加 §4.1 的 `project_week` 方法，即使本臂不开 P64。

将辅助项加在所有基线损失已合并之后、`optimizer.zero_grad` 之前。以下片段假定一轮 `offset` 从 0 到 `len(indices)`、`mask` 是当前可用股、`train_end` 是排他索引。`available` 来自共同样本，`observed` 来自真实可见周。按日期均衡平均，空目标周跳过。`next_latent_lambda` 在预检后填入配置。

```python
latent_start = int(cfg.get("next_latent_start", 2))
latent_weight = float(cfg.get("next_latent_lambda", 0.0))
if model.next_latent is not None and epoch + 1 >= latent_start:
    following = np.asarray(ts) + 1
    assert teacher is not None and np.all(following < cfg["train_end"])
    valid_next = torch.as_tensor(available[following] & observed[following], device=device)
    keep = mask & valid_next
    if keep.any():
        x_next = torch.from_numpy(X[following]).to(device)
        with torch.no_grad():
            target = F.normalize(teacher.encoder.project_week(x_next).float(), dim=-1, eps=1e-6)
        prediction = F.normalize(model.next_latent(temporal).float(), dim=-1, eps=1e-6)
        error = (prediction - target).square().sum(-1)
        per_date = (error * keep).sum(1) / keep.sum(1).clamp_min(1)
        latent_loss = per_date[keep.any(1)].mean()
        ramp = min(1.0, (offset + len(ts)) / len(indices)) if epoch + 1 == latent_start else 1.0
        loss = loss + latent_weight * ramp * latent_loss
```

剂量预检只用训练数据。取 B0 的 E1 权重，复制 `model.temporal` 为 eval/no-grad teacher 作为近似，固定 `seed=42` 和 E2 洗牌的前三个训练 batch，在同个 `temporal.encoder.embedding.weight` 上对 `loss_main`（含原 InfoNCE）和 `latent_loss` 分别 `torch.autograd.grad(..., retain_graph=True)`，记录未加权梯度范数和余弦。用 `clip(0.075 * median(norm_main / max(norm_aux,1e-12)), .005, .2)` 取一个权重，记录原值、限幅与实际比例。预检时先建带预测头的模型，将 B0 checkpoint 加载到共有键，逐项断言只有 `next_latent.*` 缺失。预检不更新参数。A1、其余种子及组合复用这个权重。全程日志逐轮另写 `latent_weight, latent_loss, valid_pairs, valid_weeks`，无有效样本时损失记为缺值，计数为零。预检梯度一直为零时检查路径及目标 mask，修好后再跑。

同一 checkpoint 的辅助诊断在开发期用明确的 teacher。训练过程中在每轮写一个 `ema_epoch_N.pt`（只对 A 臂）保存 teacher `state_dict`，评估时 strict load，禁止用验证标签更新。给 B0 比较辅助 target 时统一用 A 臂同轮 teacher；辅助的持久性比较在同一股票周对上以 `teacher.encoder.project_week(X[t])` 预测，预测和目标 L2 归一化，记录 MSE、有效维度方差与覆盖率。`A2` 判为潜在机制成立需开发平均 MSE 优于持久性、前后半段同向、有效覆盖至少 70%，还须走账户筛选。测试 `target.requires_grad=False`、预测头和时间塔收到梯度、E1 在 A2 下不调用辅助目标、5 周 InfoNCE 样本与 B0 一致、全掩码 batch 有限。

若 A2 过门，A1 用完全相同的损失和最大剂量，只把 ramp 移至 E1。若要区分跨度和目标类型，在固定五周 teacher 的独立试验中对照现有 InfoNCE、同目标的 MSE 和同掩码零辅助。另登记固定目标与 EMA 的比较。只有新任务明确需要条件分布，才从 `q10/q50/q90` 分位数基准启动新协议，先定义 pinball、覆盖和配对经济指标。单周点预测阶段到直接 MSE 为止。

### 4.6 F 按轮冻结输入 Linear

每个 seed 的 `E*` 从匹配 B0 的 2018–2019 内层窗口得到，见 §5。2019 末跨年的持有标签在计算 `E*` 前裁掉。`freeze_after=max(1,E*−1)`，从 `freeze_after+1` 起作用。历史 B0 的 `3/2/6` 仅用于旧结果对照。只冻结 `model.temporal.encoder.input_proj[0]` 的 `weight/bias`，LayerNorm 与其他模块照常。新键独立于现有 `research_stage` 固定早冻分支。

```python
freeze_after = cfg.get("freeze_input_after_epoch")
if freeze_after is not None and epoch + 1 > int(freeze_after):
    layer = model.temporal.encoder.input_proj[0]
    assert isinstance(layer, torch.nn.Linear)
    for param in layer.parameters():
        param.requires_grad_(False)
        param.grad = None
```

将代码放每轮首个 batch 前，参数保持在 AdamW 的原组，`grad=None` 让该参数跳过更新和 decay。输入层保留优化器与前几轮完全一样，teacher 继续 EMA。逐轮对比 checkpoint，冻结前 B0/F 轨迹一致，冻结点及后续 E8 的 Linear `weight/bias` 位级一致，其他层更新。另记录 Linear 更新范数/权重范数和各轮三账本表现。F 的主比较用冻结后固定轮，见 §5。若 `E*=1`，首轮保持学习，因此从 E2 冻结并明示偏离。与 P64 组合时只冻结线性捷径，P64 的残差支路继续学习。若要冻结整个 stem 另建臂。

### 4.7 可选 CS 行业和市场摘要

只在打分支路 `relate()` 收到 `temporal` 后增加一支。`cross_summary: true` 时初始化 `self.cross_summary = nn.Linear(256,128,bias=False)`，使用 `fork_rng` 后把权重置零，关闭时设 `None`。用 `available` 在同周按行业和全市场算 sum/count，减去该股自身得到留一均值。仅在同组有另一可用股票时填行业均值，否则填零。市场同理。若缺历史行业标签，行业结果标为回溯代理，市场臂可独立运行。保持旧关系门为零，`HierarchicalRanker.forward` 返回给辅助目标与 InfoNCE 的 pre-relation `temporal` 仍是原向量。

```python
safe = temporal.masked_fill(~available[..., None], 0)
by_group = safe.new_zeros(safe.size(0), self.n_industries, safe.size(-1))
by_group.index_add_(1, ids, safe)
counts = safe.new_zeros(safe.size(0), self.n_industries)
counts.index_add_(1, ids, available.to(safe.dtype))
count_self = available.to(safe.dtype)
sector = (by_group[:, ids] - safe) / (counts[:, ids] - count_self).clamp_min(1)[..., None]
market_mean = (safe.sum(1, keepdim=True) - safe) / (count_self.sum(1, keepdim=True) - count_self).clamp_min(1)[..., None]
temporal = temporal + self.cross_summary(torch.cat((sector, market_mean), dim=-1))
```

`market_mean` 独立于 `relate` 既有 `market` 输入。测试遮住一个股票后它不贡献任何均值，置换同周股票与对应 ID 得到同样的置换输出，完全遮住某组时有限。记录行业来源、周覆盖及有上下文股票数。

## 5. 选轮、分数融合与分窗

下文所有 epoch 是从 1 开始。基线与候选固定同一 seed 的主窗口 `42:[2,3,4]`、`43:[1,2,3]`、`44:[5,6,7]`，在任何候选读数前确定。逐轮全量分数先按共同有效股票、同周做 z-score（`ddof=0, eps=1e-12`），等权平均轮次后再做一次逐周 z-score。跨种子等权平均标准分，候选和对照使用相同种子及相同规则。每个被比较的日期股票键都应齐全，缺预测值的可用股票作为数据错误处理；正常不可用行保持 NaN。

```python
def zscore(s):
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


def fuse_epochs(scores, epochs):
    frame = scores.loc[scores.epoch.isin(epochs)].copy()
    assert set(frame.epoch) == set(epochs)
    assert not frame.duplicated(["epoch", "date", "symbol"]).any()
    grouped = frame.groupby(["date", "symbol"])["score"]
    assert grouped.size().eq(len(epochs)).all()
    assert (grouped.count().eq(0) | grouped.count().eq(len(epochs))).all()
    frame["score"] = frame.groupby(["epoch", "date"])["score"].transform(zscore)
    out = frame.groupby(["date", "symbol"], as_index=False)["score"].mean()
    out["score"] = out.groupby("date")["score"].transform(zscore)
    return out
```

- **内层选轮。** 2018–2019 按决策周取样，`exit_week < 2020-01-01`。对每轮同一口径重放纯多 Top-20，取扣费 Sharpe 最大的轮，平手取早轮。这个 `E*` 为 B0 各 seed 的 F 定时。给所有臂附各自内层 `best±1` 的敏感性，轮次裁在 1–8。这个附读不能替代固定主窗口晋级。需要逐周对齐的所有候选以相同账户和日历计算。
- **F 窗口例外。** B0、F 及任何含 F 的组合在同 seed 统一使用 `[freeze_after+1, …, min(freeze_after+3,8)]` 的冻结后窗口，附报冻结后一轮和原固定窗口。若另一个父臂和 B0 之前用过常规窗口，直接按冻结后窗口重融合已有逐轮分数。一个比较的四个角全部用相同窗口。
- **开发和复检。** 2020 是唯一晋级开发窗，按决策锚点划分，含有成熟标签的最末锚点为 2020-12-18。账本从 2018 开始连续跑，用 2020 周子集做开发统计，前后半按有决策周的顺序均分。2018–2019 只做内层选择及敏感性。历史 `2021-01-08` 起等规则冻结后一次性读取，最后一个可评分锚点为全数据末锚前两周，检查成熟日。历史新账本以现金起算、控制和候选同步。逐年表用 `exit_week.year`，周度代理注明该键。

## 6. 三账本的构建和分析

### 6.1 输入、借券与决策时间

统一输出纯多 `L20`、纯空 `S10`、多空 `LS50` 三条独立账户曲线。首筛在同一 580 股票源、同一周度开盘代理和同一现金起点计算。以固定名义区分信号与配仓改动。`L20` 目标名义 100%、最多 20 股，`S10` 目标名义 100%、最多 10 股，`LS50` 目标 50% 多和 50% 空并共享净资产和现金。仓位不足或模拟无法入场时剩余预算留现金。主场景每成交名义 10 bp，持有空头名义券费 `8%/52` 每周，现金息 0。压力场景将 10 bp 改为 30 bp，其他参数相同。另出纯多 Top-40 敏感性，固定同一分数和执行。

`evaluation_dates.csv` 从快照周历构建。`entry_week=calendar[t+1]`、`exit_week=calendar[t+2]` 是周键，个股周内首开盘为价格代理。`has_market_bar` 用 `panel.groupby('date')['observed'].any()`。节假日周没有决策时不下单。共享持有期价格表由候选分数以外的数据建成。用 [board_limit](../src/quant_mvp/hierarchy_data.py#L96) 构造多、空各自的限价方向。`tradable` 字段只适合旧纯多回放，无法代替空头可开仓。`predict()` 导出的 `realized_return` 为对数并继承多头退出规则。新三账本用下述简单收益表，避免把空头价格错误地按多头限价处理。

```python
import numpy as np
import pandas as pd

from quant_mvp.hierarchy_data import board_limit


def build_proxy_panel(panel):
    calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
    symbols = sorted(panel.symbol.unique())
    index = pd.MultiIndex.from_product([calendar, symbols], names=["date", "symbol"])
    aligned = panel.set_index(["date", "symbol"]).reindex(index)
    shape = (len(calendar), len(symbols))
    opens = aligned["open"].to_numpy(float).reshape(shape)
    closes = aligned["close"].to_numpy(float).reshape(shape)
    limits = board_limit(symbols, calendar)
    entry, exit_open = opens[1:-1], opens[2:]
    exit_close = closes[2:]
    with np.errstate(divide="ignore", invalid="ignore"):
        entry_gap = entry / closes[:-2] - 1
        exit_gap = exit_open / closes[1:-1] - 1
        entry_base_ok = np.isfinite(entry_gap) & np.isfinite(entry) & (entry > 0) & (closes[:-2] > 0)
        exit_base_ok = np.isfinite(exit_gap) & np.isfinite(exit_open) & (exit_open > 0) & (closes[1:-1] > 0)
        long_entry = entry_base_ok & (entry_gap < limits[1:-1])
        short_entry = entry_base_ok & (entry_gap > -limits[1:-1])
        long_close = ~(exit_base_ok & (exit_gap > -limits[2:]))
        short_close = ~(exit_base_ok & (exit_gap < limits[2:]))
        long_exit = np.where(long_close, exit_close, exit_open)
        short_exit = np.where(short_close, exit_close, exit_open)
        long_r = np.where(long_entry & (long_exit > 0), long_exit / entry - 1, np.nan)
        short_r = np.where(short_entry & (short_exit > 0), short_exit / entry - 1, np.nan)
    frame = aligned.iloc[: (len(calendar) - 2) * len(symbols)][["amount"]].reset_index()
    frame["entry_week"] = np.repeat(calendar[1:-1], len(symbols))
    frame["exit_week"] = np.repeat(calendar[2:], len(symbols))
    for key, value in dict(long_entry=long_entry, short_entry=short_entry,
                           long_close=long_close, short_close=short_close,
                           long_r=long_r, short_r=short_r).items():
        frame[key] = value.ravel()
    return frame
```

以每个交易场所分别匹配决策前名单。深圳从 `data/margin_eligibility/YYYYQn.json` 取 `meta.szse_date` 和 `rq`，只匹配 `.SZ`。上海从 `data/margin_eligibility_sse_short/YYYYQn.json` 取 `snap` 和 `codes`，只匹配 `.SH`。读最新一个 `snapshot_date < anchor` 的对应交易场所快照，若有实际发布时间用较晚的时刻。只有名义快照日时明确记为名单代理。缺少该交易场所历史快照时该交易场所空头禁入，输出缺失场所和周数。上海数据是融券活动代理，深市是名单，两者都不保证库存。取沪深数据当前存量的首次与最后快照并落盘，延迟一期名单做压力附读，避免同季事后名单。下面代码在 `build_proxy_panel` 之后为开发数据建立 `short_ok`、源日期与流动性读数。训练进程只需开发周线，不访问融券目录。

```python
import json
from pathlib import Path


def read_short_snapshots(end_at):
    sources = {"SZ": Path("data/margin_eligibility"),
               "SH": Path("data/margin_eligibility_sse_short")}
    by_venue = {}
    for venue, folder in sources.items():
        snapshots = {}
        for path in sorted(folder.glob("*Q*.json")):
            # A quarter ending after the phase boundary cannot supply an earlier list.
            if pd.Period(path.stem, freq="Q").end_time.normalize() >= end_at:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            date = record.get("meta", {}).get("szse_date") if venue == "SZ" else record.get("snap")
            if not date:
                continue
            snap = pd.Timestamp(str(date))
            if snap >= end_at:
                continue
            codes = record["rq"] if venue == "SZ" else record["codes"]
            assert snap not in snapshots
            snapshots[snap] = {str(code).zfill(6) for code in codes}
        by_venue[venue] = snapshots
    return by_venue


def attach_short_eligibility(frame, by_venue):
    frame = frame.copy()
    frame["short_ok"] = False
    frame["short_snapshot"] = pd.NaT
    for venue, snapshots in by_venue.items():
        keys = pd.DatetimeIndex(sorted(snapshots))
        for day, group in frame.loc[frame.symbol.str.endswith("." + venue)].groupby("date"):
            i = keys.searchsorted(day, side="left") - 1
            if i >= 0:
                frame.loc[group.index, "short_snapshot"] = keys[i]
                codes = group.symbol.str.split(".").str[0]
                frame.loc[group.index, "short_ok"] = codes.isin(snapshots[keys[i]]).to_numpy()
    return frame
```

`end_at` 开发取 `2021-01-08`，历史取全缓存末锚加一天。使用一个检查点复核 `rq` 列、`codes` 与源文件样本。缺名单的现金周保留在账本。空腿作为晋级依据时，2020 至少 75% 有决策周拥有至少 5 个已选且有时点名单依据的空头目标，前后半都至少 15 个有此覆盖的周。历史期分年度列同样覆盖。名单达不到门槛时，仍出纯空/多空代理曲线并标注 `coverage_limited`，可继续纯多晋级。缺真实券源、停复牌或成交状态时，该腿的可执行验证留空并记录原因。

### 6.2 订单与账户算法

对每个决策周从共同股票 mask 取有限分数，分数相同时按股票代码。纯多取高分前 20。空头在正且有限 `amount` 的可用股中按周计算 `rank(pct=True)`，取 `>0.6` 的两高流动性分位，再按空头排序选低分且 `short_ok` 的前 10。多空账本的空头候选先排除本周计划多头，保持无冲突。同一策略、种子和组合都用此规则。计划名单只依赖这些列。之后才用下一周多/空开盘标记判断成交，单个目标未成交后原预算留现金，所有臂保持相同规则。

下面是 `scripts/eval_weekly_architecture.py` 的核心。先 `fuse_epochs()` 得到 `[date,symbol,score]`，正常两腿共用此 `score`。把多、空分数各命名为 `score_long, score_short` 后与共享代理表 `merge(..., validate='one_to_one')`。由 `evaluation_dates.csv` 冻结实际运行周，将所有臂 reindex 对账，检查每周两份分数的有限掩码一致。对于定向组合可用两个不同模型的分数，仍走同一订单函数。

```python
def pick_targets(frame, long_n, short_n, long_share, short_share):
    known = frame.loc[np.isfinite(frame.score_long) & np.isfinite(frame.score_short)]
    longs = known.sort_values(["score_long", "symbol"], ascending=[False, True]).head(long_n)
    liquid = known.loc[np.isfinite(known.amount) & known.amount.gt(0)].copy()
    liquid = liquid.loc[liquid.amount.rank(pct=True).gt(.6)]
    pool = liquid.loc[liquid.short_ok & ~liquid.symbol.isin(longs.symbol)]
    shorts = pool.sort_values(["score_short", "symbol"]).head(short_n)
    intended = {(1, s): long_share / long_n for s in longs.symbol}
    intended.update({(-1, s): short_share / short_n for s in shorts.symbol})
    return intended


def replay(frame, dates, spec, cost_bps=10.0, borrow_ann=.08):
    previous, weekly, events = {}, [], []
    groups = {day: group for day, group in frame.groupby("date")}
    for day in dates:
        g = groups[day]
        rows = g.set_index("symbol")
        assert rows.index.is_unique
        intended = pick_targets(g, *spec)
        held, returns = {}, {}
        missing, close_fallback = 0, 0
        for key, weight in intended.items():
            side, symbol = key
            row = rows.loc[symbol]
            prefix = "long" if side > 0 else "short"
            filled = bool(row[prefix + "_entry"])
            observed_r = float(row[prefix + "_r"]) if filled else np.nan
            if filled:
                missing += int(not np.isfinite(observed_r))
                close_fallback += int(row[prefix + "_close"])
                held[key] = weight
                returns[key] = observed_r if np.isfinite(observed_r) else 0.0
            events.append(dict(date=day, exit_week=row.exit_week, side=side, symbol=symbol,
                               desired=weight, filled=weight if filled else 0.0,
                               asset_r=observed_r, exit_close_proxy=bool(filled and row[prefix + "_close"])))
        keys = sorted(held.keys() | previous.keys())
        turn = sum(abs(held.get(k, 0.0) - previous.get(k, 0.0)) for k in keys)
        gross_ret = sum(k[0] * held[k] * returns[k] for k in sorted(held))
        long_w = sum(w for k, w in held.items() if k[0] == 1)
        short_w = sum(w for k, w in held.items() if k[0] == -1)
        trading = turn * cost_bps / 10000
        borrow = short_w * borrow_ann / 52
        net = gross_ret - trading - borrow
        assert np.isfinite(net) and 1 + net > 0
        assert long_w + short_w <= 1 + 1e-12
        previous = {k: w * (1 + returns[k]) / (1 + net) for k, w in held.items()}
        weekly.append(dict(date=day, exit_week=g.exit_week.iloc[0], net=net, gross_ret=gross_ret,
                           turnover=turn, trading_cost=trading, borrow_cost=borrow,
                           long_w=long_w, short_w=short_w, net_exposure=long_w-short_w,
                           missing=missing, close_fallback=close_fallback,
                           intended_n=len(intended), filled_n=len(held),
                           intended_short_n=sum(k[0] == -1 for k in intended)))
    return pd.DataFrame(weekly), pd.DataFrame(events)


BOOKS = {"L20": (20, 0, 1.0, 0.0), "S10": (0, 10, 0.0, 1.0),
         "LS50": (20, 10, .5, .5), "L40": (40, 0, 1.0, 0.0)}
```

这是定期重置目标的周度代理账本。对缺少持仓退出估值的股周显式记录 `missing`，主账本平价标记，附带保守情景按 20% 不利收益复算，禁用读未来标签后补位或筛掉周。周度退出收盘回退不能重现连续停牌、召回、真实跨周重叠或保证金账户，结论注明「代理」。记录上一周漂移名义及本周目标以支持费用归因。如果两侧任一正目标的已成交缺失收益超过其成交股周的 2%，相应账户可继续作机制诊断，晋级需保守情景也过门。重复比较应复用同一 `asset_r`、填单标记和口径。四个手算样例必过：

1. 一只股票开盘 10、退出 11，等名义多头毛收益 `+10%`、空头 `−10%`，都按交易名义扣费，空腿再扣券费。
2. 一只计划买入股开盘不可买，后面一只可买。本周期该槽成交为零，计划名单不随开盘状态改动。
3. 两周留同股，上一期非零收益，次期换手等于目标与漂移权重差。空头名义随标的上涨而增加。
4. 改 `long_r`、`short_r`、开盘标记、下一季度名单，当前周计划名单不变。删掉晚期行情，对截断前已成熟周的计划、盈亏和净值相同。缺名单时 `S10` 当周持仓为零，`LS50` 留出相应空头预算。

先用 `long_only_backtest()` 在旧预测和旧条件上复现已存的纯多回放，核对周键、净收益和换手。新代理账本有明确的成交次序、现金和费用公式，记录旧到新差异。所有晋级配对使用新账本的同一公式和 matched B0。 [旧纯多实现](../src/quant_mvp/long_only.py) · [旧顺延空头](../scripts/skipfill_short.py) · [会计与时点验收](research_20261111_execution_plan.md)

### 6.3 评估命令、输出与门槛

为 `scripts/eval_weekly_architecture.py` 增加三个子命令。`epochs` 读取一个 run 的八轮分数，计算内层 2018–2019 每轮纯多净 Sharpe，写 `epoch_selection.json`（逐轮 `sharpe,n_weeks,best`）。`compare` 接收 `--candidate` 和 `--control` 的 run 目录，接收 `--epochs` 或各 seed 的 F 窗口，从 scores 融合后回放。`compare` 对各 seed 独立运行，融合运行则用 `--candidate/--control` 各传三个同序目录，按 §5 执行先轮后种子。`combine` 接收 `--parents`（两个或更多已有 run 或组）及 `--method blend|route|joint`，按 §7 输出候选及所有父项对照。三者从 `--data` 指定的同一周度快照读取代理表和日历，由 `--out` 指定独立结果目录，覆盖前检查旧结果。把排序与回放函数的最小 smoke 单独执行后再启动真实 `compare`。例子中的两个命令都只访问开发快照。

```powershell
$r = 'outputs/weekly_architecture_v1'
python scripts/eval_weekly_architecture.py epochs --run "$r/runs/B0_s42" --data "$r/dev_data" --out "$r/eval/B0_s42_epochs"
python scripts/eval_weekly_architecture.py compare --candidate "$r/runs/W20_s42" --control "$r/runs/B0_s42" --epochs 2 3 4 --data "$r/dev_data" --out "$r/eval/W20_s42"
```

`compare` 运行三个账本、两个成本档、`L40` 附读，调用 `scripts/r1111_common.py::paired_stats`、`perf_summary`、`yearly_table`。先对双方以 `(phase, leg, cost, date)` 一对一断言，再比较。样本缺口处移动块按连续区段处理，预定休市周不形成观测。用 NW lag 5 和 5,000 次块 bootstrap（8 周主，4/13 周附）；在多臂或父项对照复用同一批按日期抽样的索引。除年化、Sharpe、最大回撤和换手，至少报告逐周净差 bp、周胜率、开发前后半、排除最大贡献周、逐年净复利、费用、现金、填单率、平均净敞口、空头可借覆盖和尾部最差周。另出全截面 RankIC、Top-20 与 21–40 的已实现简单收益差、Bottom-10 与 11–30 的空头收益差、Top-K 重合率和分数/仓位换手。对股票级缺标签数单列，禁止用有标签与否决定下单。逐年无波动账户的 Sharpe 记 `NA`，不能把没有券源等同于零风险空头。

**单种子收益晋级。** 在同一账户上，2020 主费率 `candidate−B0 ≥ +5 bp/周`，前后半均正，最大回撤恶化不超过 2 个百分点，移除绝对贡献最大周后均值仍正。30 bp 压力净差非负，空头依赖账户满足覆盖门，保守缺失情景按 §6.2 执行。任一主账户过线即可补种子，记录过线账户。点估计为正但未过线记 `INCONCLUSIVE`，明列最接近阈值的项。只有 RankIC 或 MSE 提升标 `MECHANISM_ONLY`。加速类 Q/W 在净差不低于 −5 bp/周且同机计时中位数缩短至少 25% 时，单独记录效率候选，可以补种子验证，但不进入收益组合池。

**三种子收益确认。** 同一账户至少 2/3 个种子的配对均值为正，三种子融合通过上面经济、回撤和覆盖门。记录三个种子都用的固定窗口和内层选轮敏感性。区间宽且包含显著经济损失时保留在报告，不把阈值通过解释成统计显著。候选晋级只覆盖过线账户。其它账户仍报告，若与 B0 相比退化，组合或应用方案须明确采用哪个账户及为什么。

## 7. 有提升的臂怎么做组合

### 7.1 列出组合与共同对照

1. 在开发数据内收集通过三种子收益确认的臂。给每臂记录 `qualifying_accounts`，如 `{L20}`、`{S10}` 或 `{LS50}`。先枚举不同因子间所有可解释的两两组合，记录完整清单及先后顺序。只有两种子确认尚缺的正向结果不能进主组合池，可留待复跑。
2. 同账户通过的父项，先跑等权分数融合 `z((z_A+z_B)/2)`。独立因素兼容时加跑**联合架构训练**，合并 YAML 因子、从初始化训练，不叠加两个训练权重。每个种子独立按 §5 选轮再融合。W20 与 W80、Q0 与 Q2、A1 与 A2 互斥，允许分数融合，联合训练记录为不适用。含 F 的配对统一比较窗。含 P64+C3 时先周内 MLP 后因果卷积。含 P64+A2 时 teacher 目标使用新的周投影，标明目标空间变动。联合臂超显存按 §3 对比较族一起处理。
3. 不同账户改善也能组合。例如 A 只提升纯多，B 只提升纯空，则为 LS50 固定「多腿 A、空腿 B」的路由，对照同时包括 `B0/B0`、`A/B0`、`B0/B`，另报 `A/A`、`B/B`。两个路由分数必须来自同样的 seed、日期和共同 mask。冲突按 §6.2 的多头优先规则，重算空头候选和全账户成交、漂移、费用，记录冲突数。不得把两条已跑纯腿净收益曲线直接相加作为结果。任何同股交集或敞口改变由独立分解列解释。
4. 普通联合训练与分数融合都输出 `B0, A, B, AB` 的同账户、同周、同成本、同窗口结果；路由用两个单腿父项。比较 `AB−B0`、`AB−A`、`AB−B` 和交互项 `AB−A−B+B0`。按开发全窗均值预定哪个是较强父项，比较同一组父项贯穿整窗，禁止逐周切换父项。交互项的正值表示超过线性增益之和，仍需用实际账本和费用看 `AB` 是否更好。

```python
def blend_scores(members):
    indexed = [f.set_index(["date", "symbol"]).score.sort_index() for f in members]
    assert len(indexed) >= 2 and all(s.index.is_unique for s in indexed)
    for s in indexed[1:]:
        assert s.index.equals(indexed[0].index) and s.notna().equals(indexed[0].notna())
    frame = pd.concat(indexed, axis=1)
    combined = frame.mean(axis=1).rename("score").reset_index()
    combined["score"] = combined.groupby("date")["score"].transform(zscore)
    return combined


def incremental_curves(net, keys=("B0", "A", "B", "AB")):
    # net is a pivot indexed by a common, ordered decision calendar for one book and cost.
    base, a, b, ab = (net[key] for key in keys)
    assert net.index.is_unique and net.index.is_monotonic_increasing
    assert net[list(keys)].notna().all().all()
    return pd.DataFrame({"vs_B0": ab-base, "vs_A": ab-a, "vs_B": ab-b,
                         "interaction": ab-a-b+base})
```

路由用 `long_frame.rename(columns={'score':'score_long'})` 与空腿相同方式改名后，按 `(date,symbol)` 一对一合并，再交给 `replay`。联合训练配置合并前断言父臂的实验因子没有冲突，B0 的公共字段一致。若从已经融合的两两模型进入三臂平均，重取原三个 arm 的分数，用 `blend_scores([a,b,c])`，避免先平均两臂、再与第三臂 50/50 导致权重改变。

### 7.2 晋级和继续叠加

- 每个组合在主账户首先通过 §6.3 对 B0 的三种子经济门，再要求三种子融合相对较强父项 **≥ +3 bp/周**，开发前后半均非负、剔除最大贡献周后仍正，回撤相对较强父项恶化不超过 2 个百分点，30 bp 下相对该父项仍非负。空腿覆盖继续按 §6.1。按种子输出 `AB−父项`，至少 2/3 正才晋级。首个种子先出 `AB` 结果，任何主账户有进一步收益才追加剩余两个种子。
- 两两组合通过后，枚举「已通过组合 + 一个兼容的通过单臂」。用同一因子集合去重，等权融合三个原始 arm 的标准分，联合训练重新初始化，路由仍按每腿事前固定的配置。三个或更多因子时，对照包含 B0、所有可实现的直接父组合和各单臂，至少先算出最强的直接父组合再晋级。每增加一项，都以同一个 +3 bp/周 的增量门和同账户风险条件验收。所有新增组合仍在开发窗完成，不用历史复检决定下一层。
- 当没有兼容父项、没有新的开发胜出组合或批次算力用完时停止扩展，列出未跑组合及原因。达到预算上限只冻结已完成的结果，保留候选列表方便续跑。组合的比较次数单列，择优结果的区间按条件性历史结果解读。对同一因子集合同时报告融合与联训，选择口径在历史读数前写明。

## 8. 历史复检、验收和交接

1. 将入围单臂和组合的配置、父项、三种子 checkpoint SHA-256、逐 seed epoch 窗、融合方式、目标账户、费用和覆盖规则写入 `recheck_lock.json`。同账户若有多个合格方案，先在开发窗按主要经济指标排序并冻结全部需要复检的 ID。历史表现只用于验证冻结声明。
2. 历史导出从锁定 checkpoint 只读推理。扩展 §3 runner 加 `history-scores` 子命令，要求锁文件和原始配置哈希匹配。用训练配置复制出只读推理 cfg，把 `cache_path,market_path,universe_path,n_days` 换回冻结原输入，`train_end` 与 matched mask 参数保持不变。验证 `arrays` 的训练/开发前缀特征、mask 和持有期标签与开发快照在已成熟周一致。评分只调 `predict(model,data,cfg,1096,len(dates)-2,device)`，模型从冻结配置构造，`strict=True`。每个选定 epoch 只落 `date,symbol,score,epoch`，加锁文件哈希和时间窗。历史账本重用 §6，2026 只报已成熟退出。旧历史已多次读过，结果定位为历史复检。
3. 历史验收按冻结的账户逐年核对，报告对 B0 和父项的净增量、全窗和分年回撤、成本、覆盖和压力场景。如果显著劣化或增益由极少数周驱动，归为待证。通过开发与历史且相对父项有增益时，可登记影子方案。从冻结后的首个可评分锚点开始，先存分数、订单意图、名单时点，再接收实际收益和可得成交记录。26/52 个成熟周做预定检查。实盘替换仍需真实账户和券源、手续费、成交能力的检验。
4. 写 `acceptance.md`，按以下清单逐项勾选。任何失败项给具体原因、影响的试验和可继续的独立分支。

| 必交 | 内容和验收 |
| --- | --- |
| `manifest.json`, `prereg.md`, `trials.csv`, `recheck_lock.json` | 代码、数据、配置和 checkpoint 可追溯。所有计划、失败和未跑原因有记录。未触发复检时写原因 |
| `smoke.json`, `coverage.csv`, `evaluation_dates.csv` | 关闭等价、旧权重可读、前缀隔离、专项断言、股票/借券/成交覆盖均可复查 |
| `dev_scores.parquet`, `epoch_selection.json`, `training_metrics.csv`, `resources.json` | 每臂八轮，固定窗与选轮附读可重建，含耗时、峰值显存、参数量、掩码与辅助诊断 |
| `weekly_paired.parquet`, `trade_events.parquet` | 逐周两臂同键。交易事件含计划、成交代理、退出价格类型、缺失及费用归因 |
| `yearly.csv`, `comparisons.json` | 每个 seed 和融合、阶段和成本档都有纯多、纯空、多空三行及缺失原因。报告累计净收益、最大回撤、Sharpe、区间、换手、费用、敞口和尾部 |
| `combinations.csv` | 父项、方式、主账户、`AB−B0`、`AB−A`、`AB−B`、交互项、各成本、晋级及未跑原因 |
| `acceptance.md` | 对每个臂说明改动、是否生效、经济结论、失败周/年份及下一步。列出不同账户各自最好的已验证候选以及最强父项。附三条账户对比净值和回撤图 |

`acceptance.md` 的结论用 `PASS_DEV`、`PASS_SEEDS`、`PASS_EFFICIENCY`、`HISTORICAL_SUPPORT`、`INCONCLUSIVE`、`STOP_DEV`、`MECHANISM_ONLY`、`BLOCKED_DATA` 或 `FAILED_CHECK` 标记。主账户都未见正向收益且无效率线索时记 `STOP_DEV`。相同市况的三个种子代表优化重复，不能当作三个独立市场样本。缺少实际交易资料的纯空结果保持名单代理标签。
