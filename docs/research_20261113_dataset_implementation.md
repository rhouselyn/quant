# 数据集 v2：实施接口、代码骨架与验收用例

> 2026-11-13 切分更正：本文 F0 示例是旧内部诊断设置，主比较须采用 2018–2020 验证和 2021 至冻结末日测试/已读历史复检，不能直接照搬 2020/2021 外层边界。先读 [更正说明](research_20261113_split_correction.md)。已运行配置与哈希保留原样。

2026-11-13。配套入口：[执行手册](research_20261113_dataset_execution_runbook.md)。**本文全部新增接口/代码都是待实施规格，本轮未创建这些脚本或运行其中的命令。** 工程实施时按小补丁逐项落地；示例只实现标出的局部逻辑，不能将它们当完整生产程序。

## 1. 变更面与兼容原则

先读 [代码审计](research_20261113_dataset_code_audit.md)，其中有读取时的行号和实测范围。实施时再次 rg 所有调用者，不按文档行号盲改。

| 文件 / 函数 | 最小改动 | 必须同步的调用 / 验收 |
| --- | --- | --- |
| scripts/download_full_universe.py | 保留旧 CLI；新归档器不沿用末日存在即完成的判断 | 下载状态、分页、缺口、退市与重试 fixture |
| 新 src/quant_mvp/dataset_store.py | 仅处理 ID、manifest、分区查询和局部 PIT 连接；不建通用 ETL 框架 | 新 scripts/ashare_dataset.py 的 discover/fetch/audit/query/snapshot 子命令 |
| scripts/prepare_hierarchy_580_mon.py | 每证券保存自身 raw_path；新增显式 raw-only 输入模式，消除隐式回退混源；保留旧复现路径 | prepare_audit、源清单、path_features、周聚合调用者 |
| src/quant_mvp/data.py::aggregate_weekly / add_features | 区分真实价格观测、成交、会话开放；不让零量占位行伪造会话 | hierarchy_data.align_observed_features、market_features、各 prepare 脚本 |
| src/quant_mvp/hierarchy_data.py::arrays / daily_arrays | v2 显式标签、时点 mask、动态行业；读取冻结预处理；修日线拟合右边界 | engine/predict、daily 分支、history-scores、live_score 兼容测试 |
| src/quant_mvp/hierarchy.py::forward / relate / _cross_summary_context | v2 可选按日期行业 ID；先复用 relate 的单日动态分组，不改注意力主体 | 曝露损失、InfoNCE 同业mask、行业辅助目标、batch/predict |
| src/quant_mvp/hierarchy_engine.py::train / make_batch / predict | 训练目标成熟 mask、训练期增强、权重、有效日期累积；打分不自动评估外层标签 | epoch 分数 schema、训练协议、teacher、所有自定义 followup trainer |
| scripts/run_weekly_architecture.py | 新快照 manifest 取代 2021/1096/580 路径常数；v2 log_trial 直接读manifest，不依赖旧base_dev.yaml/data_sha256.json；保留legacy默认 | tests/test_followup_entry.py；所有队列训练/评分 argv、日志与实际内容hash |
| scripts/weekly_followup_ledger.py | raw 价格、证券事件、真实交易单位/规则、T+1、应收分红和借券事件；保持拒单/资金校验 | 旧 Ledger 单测、合成日历、待成交订单 |
| scripts/eval_weekly_architecture.py | 旧 proxy 显式命名 legacy；新分支调用日账本，订单不读取未来结果 | replay_baselines、single_epoch、combo、r1111_common 汇总 |
| scripts/weekly_followup_queue.py | 新批次执行 cwd、哈希预检、gate 内容、严格产物校验；保留串行 GPU | tests/test_followup_queue.py，新 attempt、失败依赖、崩溃恢复 |
| 新 scripts/eval_dataset_experiments.py | 统一配对/分腿/因子/成本/规模报告与晋级 gate；调用已有统计工具 | 结果 schema 与最终 acceptance |
| scripts/attention_followup.py::LegPool、attention_context_ablation.py::fit_portfolio；新 scripts/dataset_allocation.py | AL1复用mean-pool BCE和freeze10；显式20/10腿长、缺失mask、按腿构造新标签、独立时间锁；旧CLI默认保持40/10 | followup_combo_controls仅作旧结果参考；新配比导出、CPU训练、真实账本、静态对照与泄漏fixture |
| configs / tests | v2 配置从 B0/CS 派生；新增 tests/test_dataset_contract.py / test_dataset_ledger.py | 不覆写 canonical 或部署配置，不修改 dashboard 展示口径 |

现有 arrays 返回 tuple 含可选 cell_ids，外部脚本有位置索引。首轮不要全仓改 dataclass：保持原前 13 项及 cell_ids 位置不变，v2 的附加元数据放在 data[9] 的专用键，或传独立 batch metadata；明确记录 shape，不把大数组转成 JSON。v2 的 data[7] 明确使用永久 security_id；legacy仍用symbol，二者由schema分流。v2的市场/板块/限制价全部由元数据连接，禁止把永久ID传给旧代码后缀解析或board_limit函数。输出 preprocessing.json 时剥离运行期数组，保存对应文件哈希。以后确有维护痛点再单独迁移类型。

已证实的边界问题先修，收益实验后做：日线 fit 边界、逐证券 provenance、占位 bar、v2 标签、日期硬编码。旧 build_raw_2000_backfill 的复权拼接疑点只影响走该路径的数据；当前 B0 使用 raw_ak2000，先追踪来源再判断影响。

## 2. 数据字段与版本

### 2.1 内部身份与表

security_id 是永久字符串 ID，不把纯六位代码当主键。别名表：(vendor, code, effective_from, effective_to, security_id, available_at)，区间左闭右开。今日后缀 SH/SS 映射到同一上海身份；供应商 code 原样保留。北交所换码、再上市/换股事件必须按来源判定身份是否连续，不能单靠名称相似合并。

规范化表的最小字段：

~~~text
security_master:
  security_id, security_type, venue, list_at, delist_at,
  effective_from, effective_to, available_at, recorded_at, evidence_level
daily:
  security_id, trade_date, open_raw, high_raw, low_raw, close_raw,
  prev_close_raw, reference_price, volume_shares, amount_cny,
  price_observed, traded, session_open, suspended, quality_status
corporate_action:
  security_id, action_id, action_type, announcement_at, ex_at, pay_at,
  shares_multiplier, cash_per_old_share, subscription_price, revision_id
membership / industry / status:
  security_id, value, effective_from, effective_to,
  available_at, recorded_at, revision_id, evidence_level
fundamental:
  security_id, metric, period_end, report_type, value, unit,
  available_at, recorded_at, revision_id, evidence_level
borrow:
  security_id, quote_at, available_at, eligible, inventory_shares,
  fee_annual, expiry_at, recall_at, source_kind
~~~

所有事实表共有 provider, source_request_id, content_hash, schema_version。financial 的主键包含报告类型和版本；不能仅按股票+报告期 drop_duplicates(keep='last')。daily 的 release 内键为 security_id,trade_date，修订放新 release。数值统一到“人民币元、股”，raw 保留原单位，禁止自动把万元/手猜成元/股。

catalog.sqlite 的最小任务表：

~~~sql
CREATE TABLE ingest_job (
  request_id TEXT PRIMARY KEY,
  dataset TEXT NOT NULL,
  provider TEXT NOT NULL,
  security_id TEXT,
  start_at TEXT NOT NULL,
  end_at TEXT NOT NULL,
  params_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (
    status IN ('PENDING','RUNNING','VERIFIED','EMPTY_VERIFIED',
               'FAILED','UNAVAILABLE')
  ),
  attempts INTEGER NOT NULL DEFAULT 0,
  cursor_json TEXT,
  response_sha256 TEXT,
  rows_count INTEGER,
  first_event_at TEXT,
  last_event_at TEXT,
  checked_at TEXT,
  error_class TEXT
);
~~~

VERIFIED 需要完成分页、键唯一、边界/日历核对和 hash；EMPTY_VERIFIED 需要明确如未上市/权限内真实无记录的证据，权限拒绝是 UNAVAILABLE。不得把空文件视作下载成功。

### 2.2 snapshot manifest

~~~json
{
  "schema_version": 2,
  "snapshot_id": "content-derived-id",
  "created_at": "timezone-aware-ISO8601",
  "timezone": "Asia/Shanghai",
  "evidence_mode": "reconstructed_pit",
  "quality_level": "Q2",
  "source_release": "immutable-release-id",
  "recipe_sha256": "actual-content-sha256",
  "security_order_sha256": "actual-content-sha256",
  "calendar_sha256": "actual-content-sha256",
  "feature_names": ["ret1", "...explicit-full-order..."],
  "split_file": "splits.json",
  "files": {"weekly.parquet": "sha256", "labels.parquet": "sha256"},
  "coverage_report": "coverage.json",
  "limitations": []
}
~~~

示例中的占位词不得进入正式 manifest。先验证所有文件后发布 manifest 和成功 gate。配置来源文字如 provider=yahoo 与实际 akshare 缓存不一致时，以真实文件来源修正新配置，旧配置只读保留。

## 3. 查询与 PIT 连接代码

分区读取优先 pyarrow.dataset，日期字段用 date32，端点传 Python date，不依靠字符串隐式转换。目录路径来自 catalog 的已验收 release；不把用户 dataset 字符串直接拼成任意文件路径。

~~~python
from datetime import date
import pyarrow.dataset as ds

def read_daily_partitioned(path, start, end, security_ids, columns):
    if start >= end or not security_ids:
        raise ValueError("non-empty ids and a non-empty [start,end) are required")
    table = ds.dataset(path, format="parquet", partitioning="hive")
    filt = (
        (ds.field("trade_date") >= date.fromisoformat(start))
        & (ds.field("trade_date") < date.fromisoformat(end))
        & ds.field("security_id").isin(list(security_ids))
    )
    return table.to_table(filter=filt, columns=columns).to_pandas()
~~~

上层 load_dataset 先做 alias→security_id、历史 membership 筛选、as-of 和权限/质量等级检查。全池查询允许不传 ID，不应通过传空列表暗示全池。查询回传 data 与 manifest/coverage，不因过滤后没行默默说“成功获取完整历史”。

PIT 单指标/单报告期版本连接的可执行局部骨架：

~~~python
import pandas as pd

def join_visible_revisions(decisions, versions, keys):
    # keys 必须包含 security_id, metric, period_end, report_type。
    # 调用者已固定 snapshot，并验证 revisions 唯一、时间带时区且非空。
    left = decisions.copy()
    left["_row"] = range(len(left))
    right = versions.copy()
    required = [*keys, "available_at", "revision_id"]
    if right[required].isna().any().any():
        raise ValueError("unknown availability or revision identity")
    if right.duplicated([*keys, "available_at"]).any():
        raise ValueError("ambiguous simultaneous revisions; resolve from source")
    left = left.sort_values(["decision_at", *keys])
    right = right.sort_values(["available_at", *keys])
    out = pd.merge_asof(
        left, right, left_on="decision_at", right_on="available_at",
        by=keys, direction="backward", allow_exact_matches=True
    ).sort_values("_row")
    known = out.available_at.notna()
    if not (out.loc[known, "available_at"] <= out.loc[known, "decision_at"]).all():
        raise AssertionError("future revision joined")
    return out.drop(columns="_row")
~~~

它只解决“同一报告版本选择”，不负责跨报告期选择最新已披露期。上层先针对候选报告期完成版本连接，再选当时最新合格 period_end；累计财务转单季时也只用各期当时可见值。日期精度只有天时，来源适配器先把 available_at 移到保守的下一交易时段，不能在 join 中偷偷补时刻。

成员/行业有生效区间：先选当时可知版本，再验证 effective_from ≤ decision_at < effective_to。未来公告可能提前公布行业调整；available_at 早不代表已生效。effective_to 的后续变更也要版本化，不能用事后结束日期在公告前排除证券。

## 4. 日历、标签与预处理

### 4.1 日历与标签生成

同时保留 anchor_week（兼容旧 W-FRI 桶）、last_observed_session、decision_at、entry_scheduled_at、exit_scheduled_at。周标签不直接来自某只股票 resample 后的 first open；入场是市场统一首个会话，个股那天不开盘即缺报价，不顺移到个股复牌日当成原计划。

每个 label_kind 用统一会话表算计划区间，再通过公司行动计算总收益。保留 market return label 与 raw asset label；是否在实际订单中成交在 Ledger 决定。收益用 log1p 存训练目标时，完整亏损 -100% 无有限 log：保留原 simple_return=-1 的真实账本，训练目标需要预登记的有界变换/删监督标记及损失权重，而不是修改投资结果。

~~~python
def supervision_mask(meta, fit_cutoff):
    # 所有时间已带同一时区；缺失成熟时间自然不参加监督。
    valid = (
        meta["supervised_valid"]
        & meta["label_available_at"].notna()
        & (meta["label_available_at"] < fit_cutoff)
    )
    return valid

def assert_causal_features(meta):
    known = meta["feature_available_at"].notna()
    assert meta.loc[known, "feature_available_at"].le(
        meta.loc[known, "decision_at"]
    ).all()
~~~

训练损失：main_valid、infonce_valid、h2_valid 分开。只看某个固定 reserve=5 无法证明跨长假/停牌的 maturity；v2 用时间 mask，reserve 可作为保守预过滤但不能替代断言。所有 loss 在无有效样本时返回连接计算图的 0，不能产生 NaN 或偷偷改变可交易名单。

未来变动测试要分对象：改 t 之后行情不应改变 t 的输入、可用池和冻结模型 score，但可以改变 t 的未来标签；改标签本身只应影响训练监督。冻结 model/preprocessor 后检查得分不变；不能要求重训用到未来新标签的模型也完全不变。

### 4.2 预处理拟合边界

周线 cut 为第一内部验证决策索引，训练 fit 使用 dates[:cut]。日线 fit 用显式 feature_fit_end_at 排他边界，不再取 weekly_dates[train_end] 的结束日连同首个验证周：

~~~python
# 新快照预先给出第一内层决策的时间边界；
# fit_features 只含拟合段用于训练输入的、当时已公开数据。
fit_rows = (
    daily["available_at"].lt(feature_fit_end_at)
    & daily["belongs_to_fit_inputs"]
    & daily["price_observed"]
)
~~~

保存特征列顺序、均值/标准差或稳健统计、缺失填充值、市场统计及拟合键哈希。验证、外层和线上推理从训练 run 读取 preprocessing.json，不能在 extend cache 时重算。截面变换只用同一 decision_at 已知的参考池；训练股票子采样与截面参考池分开配置。

D2 稳健截面标准化局部函数：

~~~python
import numpy as np

def robust_cross_section(values, eligible):
    # values: [N,F]；eligible 只基于决策时点事实。
    x = np.asarray(values, dtype=np.float64)
    valid = eligible[:, None] & np.isfinite(x)
    v = np.where(valid, x, np.nan)
    count = valid.sum(axis=0)
    # 无观测特征必须标缺失；避免 nanmedian 警告掩盖问题。
    center = np.zeros(x.shape[1])
    scale = np.ones(x.shape[1])
    has = count > 0
    center[has] = np.nanmedian(v[:, has], axis=0)
    mad = np.nanmedian(np.abs(v[:, has] - center[has]), axis=0)
    scale[has] = np.maximum(1.4826 * mad, 1e-6)
    out = np.clip((x - center) / scale, -6, 6)
    out[~valid] = 0.0
    return out.astype(np.float32), valid
~~~

本函数只处理一个截面，不前填；staleness 与时间前填由上层按已冻结规则完成。MAD=0 的离散特征/全相同值单列报告，必要时采用事先规定的 IQR/常数分支，不能看收益后调整。D2 与旧 z 使用同 input mask，避免把多纳入股票的收益当作标准化收益。

## 5. 动态行业与全市场的最小模型适配

主要改变数据与 mask；不换主干。当前 relate 已可接收一个 1D industry_ids，但 forward、损失、分组布局及数组仍使用静态组。先给 v2 增加可选参数，legacy None 路径保持数值一致：

~~~python
# 在完成下文固定行业词典/空组修正后，替换 forward 的局部实现。
def forward(self, windows, daily_windows, market, available, industry_ids=None):
    temporal = self.encode(windows)
    if daily_windows is not None and hasattr(self, "daily_encoder"):
        temporal = temporal + torch.tanh(self.daily_gate) * self.encode_daily(
            daily_windows
        )
    if industry_ids is None or industry_ids.ndim == 1:
        z, scores, industry = self.relate(
            temporal, market, available, industry_ids
        )
    else:
        if industry_ids.shape != available.shape:
            raise ValueError("industry_ids must have shape [B,N]")
        rows = [
            self.relate(temporal[b:b+1], market[b:b+1],
                        available[b:b+1], industry_ids[b])
            for b in range(temporal.size(0))
        ]
        z = torch.cat([row[0] for row in rows], dim=0)
        scores = torch.cat([row[1] for row in rows], dim=0)
        industry = (
            None if rows[0][2] is None
            else torch.cat([row[2] for row in rows], dim=0)
        )
    return temporal, z, scores, industry
~~~

本方法仍依赖下述 relate、分组布局和全部损失的同步修改，不能单独粘贴后宣称完成动态行业。make_batch在v2时追加当批次行业IDs，legacy保持四参；trainer中有直接调用relate的分支也必须同步。v2 首版只开启既有 B0/CS 所需分支，拒绝未适配的配置，不能用 fallback 忽略新 industry_ids。

实现细节：

1. 历史行业字典在训练配置固定，UNKNOWN 为独立值；不根据验证期收益或未来新分类扩充语义。
2. model 初始化允许词典中的组在某日为空；bincount(minlength=n_industries)，padding 使用 max(1,max_group_size)，空组 mask=false，绝不把空组 token 当有效市场贡献。
3. 逐日调用 relate 时行业总数保持词典规模，而不是当天 ids.max()+1。动态布局缓存可后置；先测 [B,N] 的逐日循环开销。
4. CS 的市场摘要用所有当时合格股票；UNKNOWN 用合法非负ID，可以参与市场摘要，但 UNKNOWN 之间不视作真实同业，行业摘要设零。首版v2只支持当前B0/CS的全部关系门冻结为0，拒绝非零或可训练关系门，避免UNKNOWN局部注意力制造伪同业。用测试确认该路径无同行token影响；以后开启关系层需另定义UNKNOWN独立路由。
5. industry_exposure_loss、industry_excess、within-industry InfoNCE、cell 样本权重和行业相对特征全部用当日 IDs。未知行业不参与行业惩罚，仍参与价格主任务；有效组过少时惩罚为 0 并记录。
6. 对老 checkpoint 的推理，不允许静默丢失/错位 industry_ids buffer。相同快照必须 strict=True；跨新证券集合的共享模型打分应明确分离无学习的身份buffer与学习参数，单独验收后才放行。新数据实验默认重训。
7. 若拿不到历史行业：B0/CS 的严格行业研究分支标 INCOMPLETE；可做 UNKNOWN/市场摘要的受限研究对照，但不得沿用“完整 CS 同架构”标签，旧静态行业仅留复现。

内存先估计再跑：X 的最小 float32 大小约 T×N×F×4，arrays 内 raw/scaled/重排副本远大于这个数。初版分年度构建 float32 缓存；超过 RAM 预算再将中间数组写 memmap，日期 batch 构造保持同样语义。先降低 eval_batch_dates 与 encoder_chunk_size；训练日期 microbatch 小于 16 时通过梯度累积维持有效日期数，InfoNCE 的同日横截面不能拆开当独立对比池。

冻结关系门为恒等不代表整个关系网络可以随意删掉；只有逐层证明对 score、梯度和 dropout 语义等价后才能优化。这个性能重构不与数据收益臂混做。

## 6. 增强、样本权重与受控规模

### 6.1 A1 遮蔽

只对训练输入的已观测历史做遮蔽；保持股票、日期、标签与合法价格账本。默认每个 [date,stock] 样本以 p=0.2 被选中，在 30 个特征中的一个预定义组上遮连续 1–4 周，保护最后 2 周，报告实际被改动比例，不另调p凑结果。

新增 feature_observed[T,N,F] 由填补/前填之前的真实字段可用性生成，并保存到快照旁表；make_batch按同样时间切片生成[B,N,L,F]。原data[5]的[T,N]只是bar观测，available为[B,N]，均不能直接传入下例。PIT财务在仍有效的前值延续和“本周新披露”应分开，feature_observed的语义以实际字段可用性为准。

特征组固定为收益/动量、量额、波动/范围、日内路径，清单来自配置列名校验。小代码骨架（输入 [B,N,L,F]）：

~~~python
import torch

def mask_past_groups(x, observed, groups, generator, p=0.2, protect=2):
    if not 0 <= p <= 1 or protect < 1 or x.size(2) <= protect:
        raise ValueError("invalid masking configuration")
    out = x.clone()
    altered = torch.zeros_like(x, dtype=torch.bool)
    bsz, names, length, _ = x.shape
    # 首版只供小规模正确性；全池版本向量化时保留相同语义。
    for b in range(bsz):
        for n in range(names):
            if torch.rand((), generator=generator).item() >= p:
                continue
            width = int(torch.randint(1, min(4, length-protect)+1, (),
                                      generator=generator).item())
            start = int(torch.randint(0, length-protect-width+1, (),
                                      generator=generator).item())
            group = groups[int(torch.randint(len(groups), (),
                                             generator=generator).item())]
            for f in group:
                altered[b, n, start:start+width, f] = observed[
                    b, n, start:start+width, f]
    out[altered] = 0
    return out, altered
~~~

观测 mask 与输入须同设备；示例 generator 为独立 CPU RNG，训练实际向量化采用固定 augmentation seed、epoch、样本键派生随机数，避免依赖多进程调度。原始 observed 不改变、标签不改变、归一化参数不重新拟合；teacher 的未来目标不因该输入遮蔽而替换。先保留模型原 dropout，并以相同梯度步数比较。

已有 InfoNCE 权重 0.1，另登记 A0：仅 temporal_weight=0 的低成本机制消融。A0 不是无风险改进，去掉辅助目标可能改变可训练日期，故两个臂保持同训练日期与主监督 mask；无需为了新方法再加一套对比框架。只有 A1 在未增强验证集上通过后才考虑重建头或跨域预训练。

### 6.2 时间权重

D4规定按日期聚合再赋权；当前各loss的reduction不一定等价，先构造uniform-date对照，不能把reduction改变混入时效增益：

~~~text
w_t = exp(-log(2) * age_years_t / 5)
loss = sum_t w_t * loss_t / sum_t w_t
~~~

age 从拟合截止日向过去计算；不能以整个数据的最后日期计算。先检验w=1能否复现原score/rankic/InfoNCE/行业loss；若不等价，保留原BU，增加BU_uniform_date作为目标归一方式对照，D4只与它比较，并另报对BU差异。可采样或可加权二选一，不能既按w抽样又乘w造成平方权重。先复用已有date minibatch和cell_sample_weights；InfoNCE权重是否跟随时间权重必须在recipe固定，默认全部日期级损失一致。

### 6.3 规模实验要分两种 N

N_label：全部股票仍进入同一截面标准化和 CS 上下文，只有稳定 ID 哈希子集参与监督。InfoNCE 原样保持；主监督、行业监督等使用各自声明的 mask。N_context：真正改变训练/推理参考池，会改变归一化和 CS 摘要，单独命名为端到端扩池。两者不能混成一条“更多标签”曲线。

~~~python
import hashlib

def label_subset(security_id, fraction, salt="dataset-v2-n-label-1"):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    digest = hashlib.sha256(
        (salt + "|" + security_id).encode("utf-8")
    ).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    return u < fraction
~~~

不同 fraction 使用相同 salt 才嵌套。按当日行业分层的版本需再考虑行业变动；首版按永久 ID 哈希并报告行业不平衡，不能从全历史“每行业最终20只”反选子集。N_label 少于某日期最小监督数时两臂用共同日期，报告减少的周数。固定 epoch 下日期 batch 数相同不等于有效股票曝光数/FLOPs 相同。

scaling loss 采用固定评估股票/日期上的可比较指标：主列每日期目标横截面 percentile 与分数 percentile 的平方误差（只作评价，和训练 loss 分开），另列 RankIC、原训练 loss、净收益。不同 N 下 InfoNCE 分母或训练尾部比例改变的 loss 不可直接当作同一幂律响应。

### 6.4 T1 残差辅助头（仅触发后）

保持B0/CS的原分数用于主账本，在构造器增加一个零初始化线性头。新增配置 residual_target_weight=0.2、residual_beta_lookback=104、residual_beta_min_weeks=52；标签由数据快照计算，模型不在forward读取未来市场收益。权重0的同容量对照保持新头但不给辅助loss。

~~~python
# __init__：只有显式配置本实验时创建，checkpoint结构因此另成版本。
self.residual_head = nn.Linear(cfg["embedding_dim"], 1)
nn.init.zeros_(self.residual_head.weight)
nn.init.zeros_(self.residual_head.bias)

# train：z是现有forward返回的表示；residual_valid含独立成熟时间过滤。
residual_score = model.residual_head(z).squeeze(-1)
loss_residual = rank_ic_loss(
    residual_score, residual_target, mask=residual_valid,
    temperature=cfg["rankic_temperature"]
)
loss = loss + cfg["residual_target_weight"] * loss_residual
~~~

先检查rank_ic_loss在零方差初始分数时梯度有限且可学习；如该损失在此初始化无有效梯度，改用预登记的小标准差初始化并同时改对照，不通过任意随机重启挑结果。该头只用于辅助监督；若要融合它的分数交易，那是新策略，需要内层选择与独立父项比较。同样长度的打乱残差标签为机制负对照。H2沿用已存在week2_head/target实现，但换成新标签表及成熟mask，不重复造头。

### 6.5 AL1 下一周多空配比头

执行协议以[主手册§8.3](research_20261113_dataset_execution_runbook.md#83-al1预测下一周多空相对收益调整两腿预算)为准。旧 scripts/followup_combo_controls.py 只是把旧BCE信号套到v3篮子重放，没有用当前B0/CS篮子重新拟合。它硬编码8%券费；新主档统一8.35%。scripts/attention_context_ablation.py 还硬编码40多+10空、旧train/val日期、完整标签筛选和旧缓存，不能直接启动当成AL1。

**可复用和必须适配：**

- LegPool(False)：共享129→32投影、分腿均值、LayerNorm、32→64→32 FFN、64→1输出。原pool固定h[:,:40]，新入口显式long_n=20、short_n=10并传mask，旧入口默认不改。
- fit_portfolio：复用Adam(lr=.001,weight_decay=.01)、100个full-batch更新、按内部BCE选step和freeze_projection机制；仅注册joint与freeze10，旧的交互头/MSE/分位数/冻结步数全扫描不进入AL1。
- 新 dataset_allocation.py 只协调 build / fit / evaluate 三步，复用v2数据读取和日账本；不建立另一套报价、费率、收益和并发实现。

输入缓存用 [周,30,129] float32：前20位多头、后10位空头，最后一维为+1/−1方向，其余128维来自**同一个冻结B0表示模型**。首版取父策略seed42已锁定三轮窗的中间checkpoint，为两腿股票提供同一坐标系的表示；CS只决定空腿排名，不将两套不对齐的embedding混均值。原B0/CS三seed分数仍用于确定父策略篮子，配比小头的42/43/44仅指其自身初始化。

每周保存决策时已知的intended篮子、股票ID、feature_available_at、parent_fit_end、parent_selection_at、checkpoint/preprocessing/basket哈希和feature_mask。较少候选只做零padding且mask=false；没有任何合格一条腿或关键输入缺失时，配比信号标缺失回50/50，真实订单仍按父策略的可用候选与现金规则处理。不能按未来实际成交/收益是否可见重新挑篮子。

只在配比拟合段的有效股票条目上拟合128维mean/std；方向位保持±1，padding不参与统计。按以下局部实现适配现有LegPool；源码修改尚未执行：

~~~python
import torch
from attention_followup import LegPool

class AllocationPool(LegPool):
    def __init__(self, long_n=20, short_n=10):
        super().__init__(two_queries=False)
        self.long_n, self.short_n = long_n, short_n

    def forward(self, x, observed):
        if x.shape[1:] != (self.long_n + self.short_n, 129):
            raise ValueError("unexpected allocation input shape")
        if observed.shape != x.shape[:2] or observed.dtype != torch.bool:
            raise ValueError("observed must be a [B,stocks] boolean mask")
        if not torch.isfinite(x[observed]).all():
            raise ValueError("non-finite observed allocation inputs")
        h = self.proj(x.masked_fill(~observed[..., None], 0.0))
        pooled = []
        for sl in (slice(0, self.long_n), slice(self.long_n, None)):
            mask = observed[:, sl]
            count = mask.sum(dim=1, keepdim=True)
            if (count == 0).any():
                raise ValueError("empty leg: caller must use the 50/50 fallback")
            safe = h[:, sl].masked_fill(~mask[..., None], 0.0)
            pooled.append(safe.sum(dim=1) / count)
        legs = self.norm(torch.stack(pooled, dim=1))
        return self.output(
            self.final_norm(legs + self.ffn(legs)).flatten(1)
        )

def target_long_share(logit):
    # 已冻结logit；此函数不接收价格、未来成交或标签。
    mid = torch.full_like(logit, 0.5)
    valid = torch.isfinite(logit)
    return torch.where(
        valid & (logit > 0), 0.55,
        torch.where(valid & (logit < 0), 0.45, mid)
    )
~~~

标签统一从**固定50/50父策略的日账本**构建：将每周多/空腿的有符号损益、归属的交易费/券费和现金项各除以其0.5名义预算，得到单位预算代理R_long/R_short，再令g=R_long−R_short、target=1[g>0]。这比旧脚本的两篮子未来涨幅相加再补固定8%/52更明确。现金、共同费用分摊规则在recipe冻结，防重复扣券费；固定50/50标签仍只是相对强弱监督，不保证其符号等于含路径费用的最优配比。

每个标签保存entry_at、scheduled_exit_at、label_end_at、label_available_at、value_status。受阻仓位依日账本保留；计划周末若能可信估值，可用该周mark-to-market且标注估值依据，不必等平仓才确认周收益；缺可靠估值时label_valid=false，不能伪造零收益。配比训练屏蔽该标签，外层交易和持仓仍保留，报告不可估值风险。

执行顺序及新参数规格：

~~~yaml
allocation_id: AL1
parent_recipe: F0_B0_CS_frozen_before_2020
representation_source: B0_seed42_middle_of_locked_window
long_n: 20
short_n: 10
head_arms: [joint, freeze10]
head_seeds: [42, 43, 44]
head_fit_start: "2020-01-01T00:00:00+08:00"
head_fit_cutoff: "2023-01-01T00:00:00+08:00"
head_selection_start: "2023-01-01T00:00:00+08:00"
head_selection_at: "2024-01-01T00:00:00+08:00"
head_evaluation_start: "2024-01-01T00:00:00+08:00"
min_fit_weeks: 104
min_selection_weeks: 40
min_train_class_count: 10
optimizer: Adam
learning_rate: 0.001
weight_decay: 0.01
updates: 100
freeze_projection_after: 10
checkpoint_metric: selection_bce
long_share_low: 0.45
long_share_high: 0.55
missing_signal_long_share: 0.50
static_long_shares: [0.45, 0.50, 0.55, 0.60]
borrow_annual_main: 0.0835
~~~

head_fit_start是最早允许日，实际从两市覆盖和父模型锁均可用之日开始。训练标签按head_fit_cutoff、选step标签按head_selection_at成熟过滤；父模型拟合和轮窗选择必须早于各头训练输入。已有父模型若选轮看过2020，则不能用于这张2020起的表：回到v2 F0按2018–2019锁窗，或晚移头训练并相应晚移全部后续窗口、重新登记，不能只改元数据。

每个head seed独立训练并选内部BCE最小step，相等取较早step；冻结模型参数后先平均三个logit，再使用固定0阈值。freeze10在第11步更新前冻结proj且断言此后逐位不变；若最佳step≤10，标明实际选择未进入冻结阶段，不虚称冻结机制有效。主候选事前指定freeze10，joint为机制对照；外层不在两者中再挑赢家，不按外层Sharpe重选step/配比/费率。

静态主对照只在2023内层按净Sharpe从预定权重选出并锁定（并列取换手低者，再取离0.5近者）。额外将内层动态预测平均权重冻结成常量作暴露对照；外层平均权重匹配仅为事后诊断。延迟logit对照使用上一个已产生决策的信号，首周或缺信号回0.5。各配置使用完全相同的intended股票集合，数量与残单由各自连续账本处理，不强行复制50/50成交路径。

新增产物放outputs/dataset_v2/allocation/AL1/：parent_lock.json、features.npz（含mask）、labels.parquet、coverage.json、head_selection.json、head_checkpoint、weekly_predictions.parquet、各策略daily_nav/trade_events、comparison.json。预测表保存logit/p/target_long/target_short、生成时间、父模型与小头锁、回退原因；未来标签单独存表，不让下单函数读取。

至少验收：旧40/10默认保持原输出；20/10分腿切片正确；padding值改变不影响logit；freeze10前10步与joint同seed一致；未来价格/标签/券源扰动不改变早期权重；外层结果改变不影响head选择；缺信号/空腿回0.5且不删周；恒定logit的账本分别严格复现静态0.45/0.55；变配比额外换手扣费、总目标敞口为1、无券部分留现金；checkpoint任一步不向B0/CS主干传播梯度。统计只对逐周收益配对，三个小头seed不是三份独立市场历史。

AL1需要G_CAUSAL、G_LEDGER、AL1_parent_lock和AL1_coverage四个通过gate；在GPU跑数据臂时进行CPU拟合/分析。缺表示才追加一次GPU导出，已有表示hash匹配就复用。不得直接执行旧compare_allocations或旧fit_portfolio的main后声称完成新切分。新增脚本及上述参数均是后续实施任务，本次只编辑文档。

## 7. 配置、入口和选轮

### 7.1 v2 配置增量规格

下面是对 canonical B0 的 overlay，省略的架构/损失保留原值。新键须在配置解析层验证；拼错/未实现键 fail fast。

~~~yaml
dataset_schema_version: 2
snapshot_manifest: data/ashare_v2/snapshots/<id>/manifest.json
labels_path: data/ashare_v2/snapshots/<id>/labels.parquet
membership_path: data/ashare_v2/snapshots/<id>/membership.parquet
industry_ids_path: data/ashare_v2/snapshots/<id>/industry_ids.parquet
preprocessing_path: null  # train拟合并写run；scores必须读取run中版本
split_manifest: outputs/dataset_v2/registry/splits_F0.json
fit_cutoff: "2018-01-01T00:00:00+08:00"
inner_start: "2018-01-01T00:00:00+08:00"
inner_end: "2020-01-01T00:00:00+08:00"
selection_at: "2020-01-01T00:00:00+08:00"
outer_start: "2020-01-01T00:00:00+08:00"
outer_end: "2021-01-01T00:00:00+08:00"
lookback: 40
min_history_weeks: 32
matched_lookbacks: null
common_training_horizon: 5
save_epoch_checkpoints: true
save_epoch_scores: false
epochs: 8
batch_size: 16
eval_batch_dates: 1
encoder_chunk_size: 256
torch_num_threads: 4
seed: 42
cross_summary: false
normalization: legacy_train_clip_then_cs_z
augmentation: none
label_weights: uniform
evidence_mode: reconstructed_pit
~~~

CS 只增加 cross_summary=true、完整 industry+market；D2/A1 等每臂只改对应键。n_stocks、n_days、train_end、valid_end 从 manifest 的实际日期生成并验证，不手写 580/1399/939/1096。common_training_horizon 为旧兼容预留，v2 真正有效性仍由 label_available_at 决定。

开发训练物理快照只包含拟合与内层数据及其已成熟标签。2020 首筛快照单独只读评分，包含计算输入所需的过去前缀，读取训练预处理。外层评分禁止调用带历史目录常数的 legacy history-scores 分支；新增 manifest 路由并保留旧复现语义。具体新增可选参数 --score-snapshot <manifest>：仅 history-scores 接受；v2 必须同时传 --lock 和 --out，按 split_manifest 的 outer_start/outer_end 导出，只加载 lock 中成员，写独立目录。无此参数的历史路径继续只供旧复现。v2 拒绝 --epochs 解锁导出；探索任务需独立标记协议和输出。

### 7.2 未来执行命令

以下仅在对应脚本和参数实现并通过测试后使用：

~~~powershell
python scripts/ashare_dataset.py discover --root data/ashare_v2 --config outputs/dataset_v2/configs/source.yaml
python scripts/ashare_dataset.py fetch --root data/ashare_v2 --jobs outputs/dataset_v2/registry/download_jobs.json
python scripts/ashare_dataset.py audit --root data/ashare_v2 --release <release>
python scripts/ashare_dataset.py snapshot --recipe outputs/dataset_v2/configs/BU_F0.yaml
python scripts/run_weekly_architecture.py train --root outputs/dataset_v2 --dev-data <dev-snapshot> --device cuda --cfg <config> --run <fresh-run>
python scripts/run_weekly_architecture.py scores --root outputs/dataset_v2 --dev-data <dev-snapshot> --device cuda --cfg <config> --run <fresh-run>
python scripts/eval_dataset_experiments.py select --scores <inner-scores> --split <split> --out <selection>
python scripts/run_weekly_architecture.py history-scores --root outputs/dataset_v2 --cfg <config> --run <run> --lock <selection-lock> --score-snapshot <outer-manifest> --out <fresh-score-output>
python scripts/eval_dataset_experiments.py compare --candidate <candidate> --control <control> --protocol <frozen-protocol> --out <fresh-eval>
~~~

不要执行 <...> 占位命令。run_weekly_architecture 的 train/scores 现有 CLI 可复用，但 manifest 路由、内外层拆分、动态行业和冻结预处理均是待实现部分。

### 7.3 连续三轮窗选择

从六个窗分别生成净账本（不能把三个 epoch 的 Sharpe 平均），传入下面的确定性排序：

~~~python
def choose_window(candidates):
    # 各candidate含 epochs, net_sharpe, turnover；同日期/费用/投资池。
    import math
    rows = [x for x in candidates if math.isfinite(x["net_sharpe"])]
    if len(rows) != 6:
        raise ValueError("all six three-epoch windows must be evaluable")
    best = max(x["net_sharpe"] for x in rows)
    tied = [x for x in rows if best - x["net_sharpe"] <= 1e-8]
    return min(tied, key=lambda x: (x["turnover"], tuple(x["epochs"])))
~~~

足够的内部成熟周、两市券源覆盖门由调用者检查；选轮有效条件是 label_available_at < selection_at，训练监督才用 fit_cutoff。示例 selection_at 在 2020 首个外层决策前；内层尾部未成熟标签明确排除。不够直接固定窗，不调用选择器。按主手册同时比较共用多头选窗和分腿选窗；各 checkpoint/seed 的标准化 mask 一致。selection.json 保存每个候选窗的结果、日期 hash、成本、成员和选择原因，outer score 导出锁定 config/checkpoint/selection hash。

## 8. 真实账本与正确性测试

复用 weekly_followup_ledger.Ledger 的现金、持仓、拒单和净值结构，新增事件处理必须与报价撮合分离。建议单会话顺序：

1. 从上一账本时刻到本事件时刻，先对该区间实际已持有的短仓计提券费及现金利息；使用已知费率、当时约定计费基数和实际经过时间。周一新开的空仓不能承担上周末费用，周一平掉的旧空仓仍应承担周末费用。
2. 处理当时生效的股份迁移、拆合股、现金分红应收、证券终止和借券召回。
3. 用决策时已知的目标和已有残单，检查当日 raw 报价、交易状态、买卖单位、T+1、借券资格及库存，处理平仓优先，再增仓。
4. 未成交部分保留；撤单/目标改变也生成事件。未借到股票不产生空头仓位。
5. 按可验证收盘价或有标记的旧估值更新资产；日内若需再次计提则按事件区间拆分，避免首尾重复。休市日仍计经过时间。
6. 到账日将应收转现金；应收与现金不能重复计净值。空头分红补偿也入负债/现金流。
7. 保存逐日 NAV、现金、受限现金、持股数、short liability、应收、交易/券费、残单；按周汇总。

NAV = cash + receivables + sum(signed_shares * raw_marks) - other_liabilities。借券卖出所得受限制，不能当作额外无成本杠杆；50/50 定义沿用总名义敞口 1 的研究预算并检查保证金。现金基准使用同多腿实际路径、同资本约束；保证金导致多腿路径不同须单列“组合可行性效应”。

测试矩阵要求每条有能失败的 fixture 或断言：

| 测试 | 构造与期望 |
| --- | --- |
| 身份 | 600000.SS / 600000.SH 同身份；BSE换码前后连续；未知别名拒绝，不截断前导零 |
| 分页/断点 | 模拟接口恰好返回上限且后面仍有数据；断网后重跑无遗漏/重复；损坏尾文件不跳过 |
| 生命周期 | 新上市前不参与截面统计；退市证券历史保留；未来加入/删除 masked 证券不改变早期 score |
| 时间 | 公告前1秒不可见、发布且处理完后可见；同日无时刻走下一会话；更正不改旧快照 |
| 数据末端 | 增量加一年后旧冻结预处理、既有成熟标签和旧snapshot hash保持不变 |
| 标签成熟 | 跨春节的一周/H2/五周目标边界分别检查；fit_cutoff相等时不纳入监督 |
| 占位行情 | volume=0但close存在不自动等于成交；交易所休市日没有真实bar；市场会话不由close非空推定 |
| 财务 | 累计转单季、首次披露与修订、同报告期不同report_type，实际可得值正确 |
| 单位 | 手→股、万元→元、decimal报价、复权因子口径，混源单位错必须失败 |
| 公司行动 | 1拆2且价格减半净值守恒；分红除息净值/应收/到账正确；空头分红补偿；换码不重复仓位 |
| 涨跌停 | 按历史有效规则和raw参考价；ST/板块/新股特殊期独立fixture；开盘受阻不能用周收盘假成交 |
| 停牌/退市 | 卖不掉时持仓和资本延续；强退市没有“收益0”；未知退出值显式未决与压力情景 |
| 下单时序 | 周五目标不读取周一涨停结果；替补单按下一真实可执行时刻成交 |
| 资金 | 两天手算收益/现金/借券费与程序一致；周五旧空仓周一回补收周末费、周一新开仓不收历史费；不重复扣分红/费用；短卖所得不放大预算 |
| 动态行业 | 股票转行业仅生效后变；空行业/UNKNOWN不NaN；静态兼容输出相等；所有行业loss与CS同日IDs |
| 增强 | 不改原输入、标签、真实observed；只改过去有效条目；最近2周不变；eval无增强；RNG可重复 |
| batch/累积 | dropout关闭的小合成例，microbatch累积与整日期batch梯度一致；尾批权重正确 |
| 前缀因果 | 冻结model和preprocessor，扰动未来价格/财务/成分/标签，过去输入/score/order不变 |
| 队列 | gate=false不启动、哈希错不启动、源码不同产生新attempt；失败依赖BLOCKED、不相关任务继续 |
| 选择隔离 | 修改外层收益不改变内层epoch选择；修改券费不能悄悄换同一冻结轮窗；固定成员缺分数失败 |

价格制度 fixture 要依据来源研究中按时期生效的规则，不能写“ST永远5%”或“688永远20%”等全历史假设。历史日线不能证明成交队列时，测试只保证保守代理内部一致，报告执行证据等级。

先跑这些必要测试，再跑受影响现有 tests/test_hierarchy.py、test_followup_entry.py、test_followup_queue.py、test_weekly_followup.py 和相关 stock_weights；不要求每个候选全仓重复所有无关测试。新非平凡逻辑有最小自检，文档本轮本身不执行测试。

## 9. 增量队列与执行隔离

现有 worker 只检查 gate 文件存在、outputs 存在；cfg/data 哈希只记录，没有强制验真；source_sha256 某路径目前写 run name。新批次必须升级为内容校验后才允许排正式任务。

任务结构沿用现有字段，增加明确 cwd 和内容清单：

~~~json
{
  "id": "D2_F0_s42_a1",
  "priority": 20,
  "needs": ["gates/G_BRIDGE.ok"],
  "needs_states": ["BU_F0_s42_a1"],
  "cwd": "absolute-path-to-frozen-source-checkout",
  "commands": [
    ["python", "-u", "scripts/run_weekly_architecture.py", "train",
     "--root", "absolute-batch-root", "--dev-data", "absolute-dev-snapshot",
     "--device", "cuda", "--cfg", "absolute-config", "--run", "absolute-new-run"],
    ["python", "-u", "scripts/run_weekly_architecture.py", "scores",
     "--root", "absolute-batch-root", "--dev-data", "absolute-dev-snapshot",
     "--device", "cuda", "--cfg", "absolute-config", "--run", "absolute-new-run"]
  ],
  "outputs": ["runs/D2_F0_s42_a1/dev_scores.parquet"],
  "hash_manifest": "registry/D2_F0_s42_a1_inputs.json"
}
~~~

所有占位路径发布前解析为绝对路径；输出相对 batch_root。冻结源码复制不能遗漏未追踪但被 import 的脚本，只复制 git HEAD 不足以冻结当前工作区。记录实际依赖文件清单、环境包版本、Python/torch/CUDA/GPU型号。数据大文件通过只读绝对引用并校验 hash，不重复复制整个数据仓库。

哈希校验骨架复用 scripts/r1111_common.py::sha256，不新增另一套算法：

~~~python
def validate_job_inputs(root, job, sha256):
    import json
    from pathlib import Path
    manifest = json.loads((Path(root) / job["hash_manifest"]).read_text("utf-8"))
    for item in manifest["files"]:
        path = Path(item["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("input must be an existing absolute file")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"input hash mismatch: {path}")
    for gate in manifest["gates"]:
        record = json.loads(Path(gate).read_text("utf-8"))
        if record.get("pass") is not True:
            raise ValueError(f"gate not passed: {gate}")
~~~

gate还必须绑定本次 snapshot/recipe/code hash，避免上次通过的文件误放行新配置。成功 .ok 只在 JSON 验收通过后原子生成；失败 JSON存在不能被当通过。

全局 GPU 排他锁必须跨 batch_root/进程生效：当前每root一个锁不能阻止另一个root开worker。新批次复用项目级锁或核对已有 CUDA PID 后单worker统管；不杀用户或其它任务进程。worker中断时先核对其子进程是否还活着，再决定恢复，不能看到心跳旧就删锁重启两份训练。

训练命令与评分都在同一GPU任务内串行；评估尽可能CPU。CPU分析读已原子发布的完整分数，生成 gate 后立即发布下一任务。保持 queue/closed 缺省不存在，所有晋级决策完成才关闭。失败用新attempt和新run目录；不能沿用现有resume声称等价续训，因为它重建optimizer/teacher。真正恢复需要保存并恢复optimizer/EMA/RNG/调度器；首版默认整任务重跑。

## 10. 结果文件与最终判定

每个 eval目录：

~~~text
manifest.json            # 输入/对照/协议hash
weekly.parquet           # 全日期候选/父项/现金/差值
daily_nav.parquet        # 完整真实时间路径
trade_events.parquet     # 订单、成交、拒单、企业行动、残单
yearly.csv               # 决策年、实际收益年口径分别标注
metrics.json             # gross/net、CAGR、风险、配对区间、费用/容量
factor_attribution.json  # 因子来源、回归/在线beta窗口、缺失限制
coverage.json            # 股票/日期/标签/成交/券源差集
decision.json            # 规则是否满足及原因
~~~

metrics 必须含 statistic_scope（inner/development/historical_recheck/forward_shadow）、evidence_level、cost_schema、capital、borrow_mode、epoch_windows、n_effective_weeks。未知数据用 null 加 reason，不用 0。跨实验比较只读 manifest一致的共同口径；不自动 inner join丢失亏损或缺价周。

最终验收脚本检查所有登记任务均有done/failed/BLOCKED终态，成功产物hash符合记录，晋级任务没有越过失败门，失败实验保留。报告按当前最可靠结果生成，不把旧初版读数与修正版混合，也不将未运行的模板表填成0收益。
