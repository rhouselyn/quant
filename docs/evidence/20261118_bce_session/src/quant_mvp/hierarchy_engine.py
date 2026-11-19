"""Chronological training for the hierarchical market model; optional deferred test."""
import copy
import hashlib
import json
import random
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml
from .hierarchy_data import arrays, industry_excess
from .data_trials import training_date_weights
from .hierarchy import HierarchicalRanker, industry_exposure_baseline, industry_exposure_loss, masked_infonce, weekly_stickiness_loss
from .model import cutoff_pairwise_loss, extreme_score_loss, rank_ic_loss
from .financial_return import (build_model, is_financial, date_mean_nll, within_group_pair_loss, industry_projection_loss, training_anchors,
                               distribution_metadata, VERSION)
from .long_only import long_only_backtest
from .engine import backtest, plot_all
from .training_state import (atomic_torch_save, atomic_parquet_save, training_identity, load_training_state,
                             save_training_state, restore_training_state)


def write_json(path, obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def is_gate(name):
    return name.endswith('attn_gate') or name.endswith('ffn_gate')


def random_pair_loss(scores, targets, valid, *, seed, date_indices, epoch):
    """Four uniform non-self opponents per valid stock, with a private date RNG."""
    losses = []
    for row, date_index in enumerate(date_indices):
        indices = torch.where(valid[row] & torch.isfinite(targets[row]))[0]
        n = len(indices)
        if n < 2:
            continue
        generator = torch.Generator(device='cpu').manual_seed(
            int(seed) + 1000003 * int(epoch) + int(date_index))
        own = torch.arange(n)[:, None].expand(n, 4)
        other = torch.randint(n - 1, (n, 4), generator=generator)
        other = other + (other >= own)
        i, j = indices[own.to(scores.device)], indices[other.to(scores.device)]
        direction = (targets[row, i] - targets[row, j]).sign()
        # Equal returns contribute zero but keep their sampling mass.
        loss = F.softplus(-direction * (scores[row, i] - scores[row, j]))
        losses.append((loss * direction.ne(0)).mean())
    return torch.stack(losses).mean() if losses else scores.sum() * 0.


def gate_summary(model):
    """Mean and spread of the opened residual gates, i.e. tanh(theta)."""
    opened=[torch.tanh(param) for name,param in model.named_parameters() if is_gate(name)]
    if not opened:return dict(gate_mean=None,gate_min=None,gate_max=None)
    flat=torch.cat(opened)
    return dict(gate_mean=float(flat.mean()),gate_min=float(flat.min()),gate_max=float(flat.max()))


def make_batch(data, dates, cfg, device):
    X, _, _, M, available = data[:5]
    windows=np.stack([X[t-cfg['lookback']+1:t+1].transpose(1,0,2) for t in dates])
    market = M[dates]
    if cfg.get('market_history'):
        if min(dates) < 7:
            raise ValueError('market history requires eight causal rows')
        market = M[np.asarray(dates)[:, None] - np.arange(7, -1, -1)]
    batch=(torch.from_numpy(windows).to(device),torch.from_numpy(market).to(device),
           torch.from_numpy(available[dates]).to(device))
    if len(data)>15 and data[15] is not None:
        batch += (torch.from_numpy(data[15][dates]).to(device),)
    D,daily_ends=data[10],data[11]
    if D is None:
        return batch[:1]+(None,)+batch[1:]
    n=int(cfg.get('daily_n_days',60))
    assert int(np.min(daily_ends[dates]))>=n-1,'daily window underflow (Friday before the first training week?)'
    daily=np.stack([D[daily_ends[t]-n+1:daily_ends[t]+1].transpose(1,0,2) for t in dates])
    return (batch[0],torch.from_numpy(daily).to(device))+batch[1:]


def mask_weekly_inputs(windows, observed, groups, generator):
    """Mask one observed feature group for 1–4 weeks per selected history."""
    shape = windows.shape[:2]
    device = windows.device
    history = windows.size(2) - 2
    if history < 1:
        raise ValueError('masking requires at least three history weeks')
    selected = torch.rand(shape, device=device, generator=generator) < .2
    width = torch.randint(1, min(4, history) + 1, shape, device=device, generator=generator)
    start = (torch.rand(shape, device=device, generator=generator) * (history - width + 1)).long()
    group = torch.randint(len(groups), shape, device=device, generator=generator)
    week = torch.arange(windows.size(2), device=device)
    span = (week >= start[..., None]) & (week < (start + width)[..., None])
    changed = (selected[..., None, None] & span[..., None]
               & groups[group][..., None, :] & observed[..., None])
    return windows.masked_fill(changed, 0), changed


@torch.no_grad()
def predict(model, data, cfg, start, end, device):
    model.eval()
    X,y,valid_y,M,available,observed,dates,symbols=data[:8]
    tradable=data[12] if len(data)>12 else None
    rows=[]
    indices=list(range(max(start,cfg['lookback']),end))
    size=cfg.get('eval_batch_dates',4)
    for off in range(0,len(indices),size):
        ts=indices[off:off+size]
        output=model(*make_batch(data,ts,cfg,device))
        probability={}
        if is_financial(cfg):
            scores=output.get('decision_score',output['mu_log'])
            z=None
            probability={key:value.cpu().numpy() for key,value in output.items() if key != 'decision_score'}
        else:
            _,z,scores,_=output
        values=scores.cpu().numpy()
        head = getattr(model, 'week2_head', None)
        week2_values = head(z).squeeze(-1).cpu().numpy() if head is not None else None
        for j,t in enumerate(ts):
            values[j,~available[t]]=np.nan
            valid=valid_y[t]&available[t]
            ic=pd.Series(values[j,valid]).corr(pd.Series(y[t,valid]),method='spearman') if valid.sum()>2 else np.nan
            rows.extend(dict(date=str(pd.Timestamp(dates[t]).date()),symbol=s,score=float(values[j,k]),
                realized_return=float(y[t,k]) if valid_y[t,k] else np.nan,rank_ic=float(ic),
                tradable=bool(tradable[t,k]) if tradable is not None else True,
                **({key:(bool(value[j,k]) if key=='available' else int(value[j,k]) if key=='effective_group'
                         else float(value[j,k])) for key,value in probability.items()} if probability else {}),
                **(dict(distribution_family=model.family,distribution_version=VERSION,
                        **({'score_semantics':'residual_decision_log_scale'} if model.decision_head is not None else {})) if probability else {}),
                **({'score_h2':float(week2_values[j,k]) if available[t,k] else np.nan}
                   if week2_values is not None else {})) for k,s in enumerate(symbols))
    return pd.DataFrame(rows)


@torch.no_grad()
def industry_weekly_ic(model, data, cfg, labels, mask, start, end, device):
    """Per-date Spearman between predicted and realized industry excess ranks."""
    if model.industry_head is None:return []
    model.eval()
    values=[]
    indices=list(range(max(start,cfg['lookback']),end))
    size=cfg.get('eval_batch_dates',4)
    for off in range(0,len(indices),size):
        ts=indices[off:off+size]
        _,_,_,predicted=model(*make_batch(data,ts,cfg,device))
        for row,t in enumerate(ts):
            valid=mask[t]
            if int(valid.sum())<5:continue
            a=pd.Series(predicted[row].cpu().numpy()[valid])
            b=pd.Series(labels[t][valid])
            if a.nunique()<2 or b.nunique()<2:continue
            values.append(float(a.corr(b,method='spearman')))
    return values


def score_eta2_by_industry(dense, available, ids):
    """Mean weekly share of score variance explained by industry means, with the
    neutral-score expectation for the same weeks (numpy twin of
    :func:`quant_mvp.hierarchy.industry_exposure_loss`, for logged predictions)."""
    values,baselines=[],[]
    group_rows=np.broadcast_to(ids,dense.shape) if np.ndim(ids)==1 else ids
    for row,keep,groups in zip(dense,available,group_rows):
        score,group=np.asarray(row,dtype=float)[keep],groups[keep]
        if len(score)<3:continue
        sd=score.std()
        if sd<=1e-9:continue
        z=(score-score.mean())/sd
        counts=np.bincount(group)
        means=np.bincount(group,weights=z)/np.maximum(counts,1)
        values.append(float((counts*means**2).sum()/len(z)))
        baseline=industry_exposure_baseline(counts)
        if baseline is not None:baselines.append(baseline)
    if not values:return None,None
    return float(np.mean(values)),(float(np.mean(baselines)) if baselines else None)


def build_stratified_tail_targets(realized_return, valid, *, fraction, count=0,
                                  long_core=20, short_core=10,
                                  long_candidate=150, short_candidate=100,
                                  core_weight=3., candidate_weight=1., coarse_weight=.5,
                                  tradable=None):
    """Build coarse/fine tail labels and core-vs-candidate pairs per date."""
    usable=valid.bool()&torch.isfinite(realized_return)
    if tradable is not None:usable=usable&tradable.bool()
    labels=torch.zeros_like(realized_return)
    weights=torch.zeros_like(realized_return)
    long_pairs=[];short_pairs=[]
    for row in range(len(realized_return)):
        indices=usable[row].nonzero().flatten()
        n=len(indices)
        if n<2:
            long_pairs.append((indices,indices));short_pairs.append((indices,indices))
            continue
        half=n//2
        coarse=min(int(count) if count else max(1,int(round(n*fraction))),half)
        order=indices[torch.argsort(realized_return[row,indices].detach(),stable=True)]
        descending=order.flip(0)
        bottom_core=order[:min(int(short_core),half)]
        top_core=descending[:min(int(long_core),half)]
        bottom_candidates=order[len(bottom_core):min(int(short_candidate),half)]
        top_candidates=descending[len(top_core):min(int(long_candidate),half)]
        bottom_coarse=order[:coarse];top_coarse=descending[:coarse]
        for group,label,weight in (
            (bottom_coarse,-1.,coarse_weight),(top_coarse,1.,coarse_weight),
            (bottom_candidates,-1.,candidate_weight),(top_candidates,1.,candidate_weight),
            (bottom_core,-1.,core_weight),(top_core,1.,core_weight)):
            labels[row,group]=label;weights[row,group]=weight
        long_pairs.append((top_core,top_candidates))
        short_pairs.append((bottom_core,bottom_candidates))
    return labels,weights,long_pairs,short_pairs


def continuous_tail_weights(realized_return, mask=None, *, fraction=.2,
                            strength=2., temperature=.03, return_quantiles=False):
    """Smooth symmetric weights from each date's raw-return percentiles."""
    usable=torch.isfinite(realized_return)
    if mask is not None:usable=usable&mask.bool()
    weights=torch.zeros_like(realized_return)
    quantile_values=torch.zeros_like(realized_return)
    for row in range(len(realized_return)):
        indices=usable[row].nonzero().flatten()
        n=len(indices)
        if not n:continue
        if n==1:
            weights[row,indices]=1.
            quantile_values[row,indices]=.5
            continue
        order=indices[torch.argsort(realized_return[row,indices].detach(),stable=True)]
        sorted_returns=realized_return[row,order].detach()
        _,inverse,counts=torch.unique_consecutive(
            sorted_returns,return_inverse=True,return_counts=True)
        positions=torch.arange(n,dtype=realized_return.dtype,device=realized_return.device)
        position_sums=torch.zeros(len(counts),dtype=realized_return.dtype,
                                  device=realized_return.device).index_add_(0,inverse,positions)
        mean_positions=position_sums/counts.to(realized_return.dtype)
        quantiles=(mean_positions/(n-1))[inverse]
        quantile_values[row,order]=quantiles
        weights[row,order]=1.+float(strength)*(
            torch.sigmoid((quantiles-(1.-float(fraction)))/float(temperature))
            +torch.sigmoid((float(fraction)-quantiles)/float(temperature)))
    if return_quantiles:return weights,quantile_values
    return weights


def cell_sample_weights(cells, valid, *, alpha=.5, cap=3., floor=.2):
    """Per-name weights equalising loss mass across observable cells.

    ``cells`` is ``[B, N]`` integer cell ids (-1 for an unusable name), and
    ``valid`` is the current week's training mask.  Counts are taken over valid
    names in each week, so the weights are recomputed from the names actually
    present.  The raw weight is ``count^{-alpha}``; after normalising the week
    to mean one, ``floor`` and ``cap`` keep small cells from exploding and
    large cells from vanishing.  Invalid names always carry zero weight.
    """
    if cells is None:
        return None
    cells = cells.to(valid.device).long()
    valid = valid.bool()
    weights = torch.zeros(valid.shape, dtype=torch.float32, device=valid.device)
    n_cells = int(cells.max()) + 1
    for row in range(valid.shape[0]):
        keep = valid[row] & (cells[row] >= 0)
        if not bool(keep.any()):
            continue
        group = cells[row][keep]
        counts = torch.bincount(group.detach().cpu(), minlength=n_cells).to(valid.device)
        raw = counts.clamp_min(1).to(torch.float32).pow(-float(alpha))
        weight = raw[group]
        weight = weight * (weight.numel() / weight.sum().clamp_min(1e-12))
        weight = weight.clamp(min=float(floor), max=float(cap))
        weights[row, keep] = weight
    return weights


def extreme_tail_labels(targets, valid, frac, count_limit=None, industry_ids=None):
    """Per-date extreme-tail labels: global cross-section by default, or
    independent per-industry tails when ``industry_ids`` is given.

    Within-industry tails keep the total label count (each industry contributes
    its own top/bottom ``frac``) but balance the label composition across
    industries, so a high-volatility industry can no longer monopolise both
    tails of the global cross-section.  Industries with a single valid name are
    skipped; ``count_limit`` (extreme_count) applies per group.
    """
    labels = torch.zeros_like(targets)
    for row in range(len(targets)):
        ids_valid = torch.where(valid[row])[0]
        if industry_ids is None:
            groups = [ids_valid]
        else:
            row_groups = industry_ids[ids_valid]
            groups = [ids_valid[row_groups == g] for g in torch.unique(row_groups)]
        for member in groups:
            if len(member) < 2:
                continue
            count = max(1, int(round(len(member) * frac)))
            if count_limit:
                count = min(count_limit, len(member) // 2)
            order = member[torch.argsort(targets[row, member])]
            labels[row, order[:count]] = -1
            labels[row, order[len(order) - count:]] = 1
    return labels


def smoothed_score_targets(realized_return, mask=None, *, fraction=.2,
                           strength=2., temperature=.03, middle_fraction=.25,
                           middle_weight=.05):
    """生成全截面中位数方向标签，并仅降低 Score 中间样本权重。"""
    rankic_weights,quantiles=continuous_tail_weights(
        realized_return,mask,fraction=fraction,strength=strength,
        temperature=temperature,return_quantiles=True)
    usable=torch.isfinite(realized_return)
    if mask is not None:usable=usable&mask.bool()
    labels=torch.where(quantiles>.5,torch.ones_like(quantiles),-torch.ones_like(quantiles))
    labels=torch.where(usable,labels,torch.zeros_like(labels))
    middle=usable&(quantiles>float(middle_fraction))&(quantiles<1.-float(middle_fraction))
    score_weights=rankic_weights.clone()
    score_weights[middle]*=float(middle_weight)
    return labels,score_weights,rankic_weights


def metadata(cfg, model, device):
    if is_financial(cfg):
        return dict(architecture='Attention + market/industry/stock means + low-rank M6 + log density',
            model_type=cfg['model_type'], backend=model.encoder.backend, device=str(device),
            objective=cfg.get('loss','date_mean_nll'), score_semantics='conditional_mean_log_return',
            mean_variant=model.variant, distribution_family=model.family, distribution_version=VERSION,
            n_stocks=cfg['n_stocks'], lookback=cfg['lookback'], epochs=cfg['epochs'],
            parameter_count=sum(p.numel() for p in model.parameters()),
            decoder_parameter_count=sum(p.numel() for name,p in model.named_parameters()
                                        if not name.startswith(('encoder.', 'final_norm.'))))
    return dict(architecture='L-G-L-G-L; 29 industry tokens + one market token; no stock cross30',
        model_type='hierarchical_market', market_tokens=1, global_layers=cfg.get('global_layers',2),
        attention_heads=cfg.get('attention_heads',4),
        data_source=cfg.get('provider') if cfg.get('dataset_schema_version') == 2 else 'yahoo',
        frequency='weekly',
        n_stocks=cfg['n_stocks'],n_days=cfg['n_days'],n_score_heads=1,lookback=cfg['lookback'],
        embedding_dim=cfg['embedding_dim'],d_model=cfg.get('d_model',64),n_layers=cfg.get('n_layers',4),
        epochs=cfg['epochs'],backend=model.temporal.encoder.backend,
        device=str(device),temporal_horizon=cfg['temporal_horizon'],temporal_weight=cfg['temporal_weight'],
        rankic_weight=cfg['rankic_weight'],score_weight=cfg['score_weight'],transaction_cost_bps=cfg['transaction_cost_bps'],
        industry_classification=cfg['industry_classification'],universe_name=cfg['universe_name'],
        objective='extreme_score_plus_full_rankic'+('_plus_temporal_infonce' if cfg['temporal_weight'] else '')
            +('_random_pair' if cfg.get('random_pair_weight', 0.) else '')
            +('_cell_balanced' if cfg.get('cell_balanced_weight') else '')
            +('_raw_return_ranking' if cfg.get('raw_return_ranking') else '')
            +('_tradable_training_mask' if cfg.get('train_tradable_mask') else '')
            +('_stratified_tail' if cfg.get('tail_objective')=='stratified_cutoff' else '')
            +('_continuous_tail_weighted_rankic_score' if cfg.get('continuous_tail_weighting') else '')
            +('_uniform_rankic' if cfg.get('continuous_tail_weighting') and not cfg.get('rankic_tail_weighting',True) else '')
            +('_median_score_smoothed' if cfg.get('score_smoothing') else '')
            +('_cutoff_pairwise' if float(cfg.get('cutoff_pairwise_weight',0.)) else '')
            +('_within_industry_tails' if cfg.get('tail_within_industry') else '')
            +('_plus_industry_exposure_penalty' if float(cfg.get('industry_exposure_weight',0.)) else '')
            +('_plus_industry_excess_head' if float(cfg.get('industry_head_weight',0.)) else '')
            +('_plus_daily_gated_branch' if cfg.get('daily_path') else '')
            +('_plus_weekly_stickiness_penalty' if float(cfg.get('turnover_control_weight',0.)) else '')
            +('_plus_nextweek_delta_head' if float(cfg.get('nextweek_head_weight',0.)) else '')
            +(f'_head{int(cfg["extreme_count"])}' if cfg.get('extreme_count') else '')
            +('_with_no_trade_region' if float(cfg.get('control_dead_zone',0.)) else ''),
        selection_metric='long_net_sharpe',periods_per_year=52,
        industry_head_weight=float(cfg.get('industry_head_weight',0.)),
        industry_exposure_weight=float(cfg.get('industry_exposure_weight',0.)),
        turnover_control_weight=float(cfg.get('turnover_control_weight',0.)),
        control_temperature_mult=float(cfg.get('control_temperature_mult',0.5)),
        control_measure=cfg.get('control_measure','softmax'),
        control_dead_zone=float(cfg.get('control_dead_zone',0.)),
        extreme_frac=float(cfg['extreme_frac']),
        extreme_count=int(cfg.get('extreme_count') or 0),
        raw_return_ranking=bool(cfg.get('raw_return_ranking')),
        industry_balanced_weight=bool(cfg.get('industry_balanced_weight')),
        cell_balanced_weight=bool(cfg.get('cell_balanced_weight')),
        cell_balance_alpha=float(cfg.get('cell_balance_alpha', .5)),
        cell_balance_cap=float(cfg.get('cell_balance_cap', 3.)),
        cell_balance_floor=float(cfg.get('cell_balance_floor', .2)),
        cell_balance_liquidity_buckets=int(cfg.get('cell_balance_liquidity_buckets', 3)),
        cell_balance_volatility_buckets=int(cfg.get('cell_balance_volatility_buckets', 3)),
        train_tradable_mask=bool(cfg.get('train_tradable_mask')),
        continuous_tail_weighting=bool(cfg.get('continuous_tail_weighting')),
        continuous_tail_fraction=float(cfg.get('continuous_tail_fraction',cfg['extreme_frac'])),
        continuous_tail_strength=float(cfg.get('continuous_tail_strength',2.)),
        continuous_tail_temperature=float(cfg.get('continuous_tail_temperature',.03)),
        score_smoothing=bool(cfg.get('score_smoothing')),
        score_middle_fraction=float(cfg.get('score_middle_fraction',.25)),
        score_middle_weight=float(cfg.get('score_middle_weight',.05)),
        rankic_tail_weighting=bool(cfg.get('rankic_tail_weighting',True)),
        tail_objective=cfg.get('tail_objective','coarse'),
        tail_within_industry=bool(cfg.get('tail_within_industry')),
        cutoff_pairwise_weight=float(cfg.get('cutoff_pairwise_weight',0.)),
        cutoff_pairwise_temperature=float(cfg.get('cutoff_pairwise_temperature',1.)),
        relation_gate_init=cfg.get('relation_gate_init'),
        relation_gate_lr_scale=cfg.get('relation_gate_lr_scale',1.0),
        relation_gates_frozen=bool(cfg.get('freeze_relation_gates')),
        contrastive_location='temporal representation, independent projection; EMA temporal encoder and projection',
        contrastive_negatives=cfg.get('contrastive_negatives','market'))


def train(cfg, out, defer_test=False, resume=None, device=None):
    random_pair_weight = float(cfg.get('random_pair_weight', 0.))
    if not np.isfinite(random_pair_weight) or random_pair_weight < 0:
        raise ValueError('random_pair_weight must be finite and nonnegative')
    if random_pair_weight and (is_financial(cfg) or cfg.get('time_half_life_years') is not None):
        raise ValueError('random pairs require nonfinancial, date-equal training')
    out=Path(out)
    resume_dir=Path(resume) if resume is not None else None
    if resume_dir is not None and out.resolve()!=resume_dir.resolve():
        raise ValueError('Resume must use the existing run directory')
    if out.exists() and (resume_dir is None or out.resolve()!=resume_dir.resolve()):
        raise FileExistsError(f'Existing run preserved: {out}')
    torch.set_num_threads(cfg.get('torch_num_threads',4))
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    v2=cfg.get('dataset_schema_version') == 2
    if v2 and (not defer_test or cfg.get('save_epoch_scores')):
        raise ValueError('v2 requires isolated scoring, including when resuming')
    if resume_dir is not None and not (resume_dir/'training_state.pt').is_file():
        raise ValueError('Exact resume requires training_state.pt; old weight-only checkpoints cannot resume')
    data=arrays(cfg)
    X,y,valid_y,M,available,observed,dates,symbols,ids,preprocessing,D,daily_ends,tradable=data[:13]
    cells=data[13] if len(data)>13 else None
    if is_financial(cfg) and not np.isfinite(M[training_anchors(cfg,valid_y,available)]).all():
        raise ValueError('nonfinite market state on registered training anchors')
    label_valid=valid_y if tradable is None or not cfg.get('train_tradable_mask') else valid_y&tradable
    stratified=cfg.get('tail_objective','coarse')=='stratified_cutoff'
    continuous=bool(cfg.get('continuous_tail_weighting'))
    score_smoothing=bool(cfg.get('score_smoothing'))
    rankic_tail_weighting=bool(cfg.get('rankic_tail_weighting',True))
    if score_smoothing and not continuous:raise ValueError('score smoothing requires continuous tail weighting')
    if continuous and stratified:raise ValueError('continuous tail weighting is a canonical-only arm; do not combine it with stratified_cutoff')
    if cfg.get('industry_balanced_weight') and cfg.get('cell_balanced_weight'):
        raise ValueError('Use either industry_balanced_weight or cell_balanced_weight, not both')
    pair_weight=float(cfg.get('cutoff_pairwise_weight',0.))
    if pair_weight and not stratified:raise ValueError('cutoff pairwise requires tail_objective=stratified_cutoff')
    # Followup runbook §1.1: the caller may pin the device; None keeps the
    # legacy auto behaviour (cuda when available).
    device=torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    identity=training_identity(cfg,data,device)
    resume_state=load_training_state(resume_dir/'training_state.pt',cfg,identity) if resume_dir is not None else None
    aug_generator=None
    if v2 and cfg.get('augmentation') == 'mask_past_groups':
        groups=cfg['augmentation_groups']
        names=preprocessing['feature_names']
        if sorted(n for group in groups for n in group) != sorted(names):
            raise ValueError('augmentation groups must partition all configured features')
        aug_groups=torch.tensor([[n in group for n in names] for group in groups],
                                dtype=torch.bool,device=device)
        aug_generator=torch.Generator(device=device).manual_seed(cfg['seed']+10000)
    model=build_model(cfg,ids,preprocessing).to(device)
    pretrained=cfg.get('pretrained_temporal')
    if pretrained and resume_dir is None:
        # Optional cross-universe init: load only the temporal tower (same
        # 30-feature, attn d128 architecture) and leave the relation layer /
        # score head fresh.  Use a checkpoint trained only on the pre-validation
        # window (``newest.pt``) to avoid validation-label leakage.
        state=torch.load(pretrained, map_location=device, weights_only=True)
        temporal_state={k[len('temporal.'):]:v for k,v in state.items()
                        if k.startswith('temporal.')}
        if not temporal_state:
            raise ValueError(f'no temporal.* tensors in {pretrained}')
        missing,unexpected=model.temporal.load_state_dict(temporal_state, strict=False)
        if missing or unexpected:
            raise ValueError(f'pretrained temporal mismatch: missing={missing} unexpected={unexpected}')
        print(f'loaded pretrained temporal: {pretrained} ({len(temporal_state)} tensors)', flush=True)
    named=dict(model.named_parameters())
    frozen_gates=[name for name in named if is_gate(name) and cfg.get('freeze_relation_gates')]
    for name in frozen_gates:named[name].data.zero_();named[name].requires_grad_(False)
    trainable=[(name,param) for name,param in named.items() if name not in frozen_gates]
    gate_ids={id(param) for name,param in trainable if is_gate(name)}
    shared=[param for name,param in trainable if id(param) not in gate_ids]
    gated=[param for name,param in trainable if id(param) in gate_ids]
    scale=float(cfg.get('relation_gate_lr_scale',1.))
    stage=cfg.get('research_stage')
    low_names=('temporal.encoder.input_proj.', 'temporal.encoder.blocks.0.')
    low=[param for name,param in trainable if name.startswith(low_names)]
    if stage=='slow_first_block':
        low_ids={id(param) for param in low}
        groups=[dict(params=[p for p in shared if id(p) not in low_ids]),dict(params=low)]
    else:
        groups=[dict(params=shared)]
    if gated:groups.append(dict(params=gated,lr=cfg['learning_rate']*scale))
    negatives=cfg.get('contrastive_negatives','market')
    if negatives not in ('market','industry'):raise ValueError(f"Unknown contrastive_negatives: {negatives}")
    within_industry=negatives=='industry'
    industry_group_ids=torch.from_numpy(np.ascontiguousarray(ids)).to(device)
    ind_weight=float(cfg.get('industry_head_weight',0.))
    if ind_weight:
        industry_labels,industry_mask=industry_excess(y,label_valid,available,ids,cfg.get('industry_min_stocks',10))
        if not industry_mask.any():raise ValueError('Industry head has no valid label weeks')
    exposure_weight=float(cfg.get('industry_exposure_weight',0.))
    ctrl_weight=float(cfg.get('turnover_control_weight',0.))
    ctrl_temp=float(cfg.get('control_temperature_mult',0.5))
    ctrl_measure=cfg.get('control_measure','softmax')
    if ctrl_measure not in ('softmax','capped'):raise ValueError(f'Unknown control_measure: {ctrl_measure}')
    ctrl_capacity=1./cfg['long_top_n'] if ctrl_measure=='capped' else None
    ctrl_dead_zone=float(cfg.get('control_dead_zone',0.)) or None
    teacher=copy.deepcopy(model.temporal).eval() if cfg['temporal_weight'] else None
    if teacher is not None:
        for param in teacher.parameters():param.requires_grad_(False)
    # Optional next-week representation supervision (experiment 580 / E2).  The
    # head lives on HierarchicalRanker when the weight is positive; the delta
    # labels are prebuilt from daily data and carry their own validity mask, so
    # the engine only aligns and applies the masked Huber loss.  With the key
    # absent everything below is inert and the run reproduces the baseline.
    aux_weight=float(cfg.get('nextweek_head_weight',0.))
    aux_delta=aux_valid=None
    if aux_weight:
        aux_frame=pd.read_parquet(cfg['nextweek_label_path'])
        dim=int(cfg.get('nextweek_label_dim',9))
        cols=[f'd{i}' for i in range(dim)]
        aux_frame['date']=pd.to_datetime(aux_frame['date'])
        index=pd.MultiIndex.from_product([pd.DatetimeIndex(dates),symbols],
                                         names=['date','symbol'])
        aligned=aux_frame.set_index(['date','symbol']).reindex(index)
        delta=aligned[cols].to_numpy().reshape(len(dates),len(symbols),dim)
        valid=aligned['valid'].fillna(False).to_numpy().reshape(len(dates),len(symbols))
        aux_delta=delta.astype(np.float32)
        aux_valid=(valid.astype(bool))&np.isfinite(delta).all(axis=2)
        del aux_frame,aligned,delta,valid
    week2_weight=float(cfg.get('week2_aux_weight',0.))
    if week2_weight:
        week2=pd.read_parquet(cfg['week2_label_path'])
        week2['date']=pd.to_datetime(week2['date'])
        week2['label_available_at']=pd.to_datetime(week2['label_available_at'])
        if week2.duplicated(['date','symbol']).any():
            raise ValueError('duplicate week2 label keys')
        week2_index=pd.MultiIndex.from_product([pd.DatetimeIndex(dates),symbols],names=['date','symbol'])
        week2=week2.set_index(['date','symbol']).reindex(week2_index)
        week2_y=week2['ret'].to_numpy().reshape(len(dates),len(symbols))
        week2_valid=week2['valid'].fillna(False).to_numpy(dtype=bool).reshape(len(dates),len(symbols))
        week2_mature=week2['label_available_at'].lt(pd.Timestamp(dates[cfg['train_end']])).to_numpy().reshape(len(dates),len(symbols))
        week2_valid &= week2_mature & np.isfinite(week2_y)
        week2_y=np.nan_to_num(week2_y).astype(np.float32)
    optimizer=torch.optim.AdamW(groups,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    out.mkdir(parents=True,exist_ok=True)
    if resume_dir is None:
        (out/'config.initial.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    config_temp=out/'config.yaml.tmp'
    config_temp.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    config_temp.replace(out/'config.yaml')
    write_json(out/'preprocessing.json',preprocessing)
    if is_financial(cfg):
        write_json(out/'distribution.json',distribution_metadata(cfg,preprocessing))
    for filename in ['universe.csv','manifest.json','feature_repair_audit.csv']:
        source=Path(cfg['universe_path']).with_name(filename)
        if source.exists():shutil.copy2(source,out/filename)
    meta=metadata(cfg,model,device)
    if resume_dir is not None:meta['resumed']=True
    write_json(out/'metrics.json',dict(meta,status='training'))
    horizon=cfg['temporal_horizon']
    reserve=max(horizon,cfg.get('common_training_horizon',horizon))
    # min_train_anchor (weekly-architecture runbook §2.1): the matched baseline
    # starts training at the first anchor every matched window can cover, so B0
    # and its candidates share identical training anchors.
    train_start=max(cfg['lookback'],cfg.get('min_train_anchor',0))
    train_dates=training_anchors(cfg,label_valid,available)
    if not len(train_dates):
        raise ValueError('no mature training anchors with sufficient labels')
    date_weights = None
    if cfg.get('time_half_life_years') is not None:
        if any((pair_weight, week2_weight, ind_weight, ctrl_weight, aux_weight,
                cfg.get('next_latent_lambda', 0), cfg.get('cell_balanced_weight'))):
            raise ValueError('D4 currently supports only the four baseline objectives')
        date_weights = training_date_weights(dates, train_dates, cfg['fit_cutoff'], cfg['time_half_life_years'])
        write_json(out/'date_weights.json', dict(half_life_years=cfg['time_half_life_years'],
            normalization='mean one over fixed training anchors; loss denominators unchanged',
            dates=[str(pd.Timestamp(dates[t]).date()) for t in train_dates],
            weights=date_weights[train_dates].tolist(),
            effective_dates=float(date_weights[train_dates].sum()**2 / (date_weights[train_dates]**2).sum())))
    write_json(out/'training_protocol.json',dict(training_dates=[str(pd.Timestamp(dates[t]).date()) for t in train_dates],
        reserve_horizon=reserve, mask=('current input candidates for decomposition; mature labels only for date NLL; common reserve retained'
            if is_financial(cfg) else 'current observed history for ranking; current+future observable masks for training-only InfoNCE'),
        no_test_during_training=True,seed=cfg['seed'],
        **(dict(epoch_score_dump='per-epoch TEST scores stored for the post-hoc equal-weight window ensemble; '
               'epoch selection itself remains validation-only') if cfg.get('save_epoch_scores') else {})))
    # Shared-sample proof (runbook §2.1): B0 and every candidate must agree on
    # train anchors, stock mask, main labels and the 5-week InfoNCE pairs;
    # hashes and per-week supervised counts make any drift checkable.
    def _sha(arr):
        return hashlib.sha256(np.ascontiguousarray(arr)).hexdigest()
    infonce_mask=np.stack([(available[t]&available[t+1:t+horizon+1].all(0)) for t in train_dates]) \
        if len(train_dates) else np.zeros((0,len(symbols)),dtype=bool)
    write_json(out/'train_verification.json',dict(
        train_anchor_start=int(train_dates.min()) if len(train_dates) else None,
        train_anchor_end=int(train_dates.max()) if len(train_dates) else None,
        n_train_dates=int(len(train_dates)),
        train_dates=[int(t) for t in train_dates],
        train_dates_sha256=_sha(train_dates),
        available_sha256=_sha(available), observed_sha256=_sha(observed),
        valid_y_sha256=_sha(valid_y), y_sha256=_sha(y), X_sha256=_sha(X),
        infonce_pairs_sha256=_sha(infonce_mask&label_valid[train_dates]) if len(train_dates) else None,
        infonce_five_week_pairs=int((infonce_mask&label_valid[train_dates]).sum()) if len(train_dates) else 0,
        **(dict(infonce_enabled=False, infonce_mask_scope='shared-sample reference only, not an active objective')
           if is_financial(cfg) else {}),
        supervised_per_week={str(int(t)):int((label_valid[t]&available[t]).sum()) for t in train_dates},
        availability=preprocessing['availability']))
    best=-float('inf');best_epoch=None;history=[];score_frames=[];val_frames=[];final_frames=[]
    start_epoch=0
    if resume_state is not None:
        start_epoch=resume_state['completed_epochs']
        history=json.loads(resume_state['history_json'])
        best=resume_state['best'];best_epoch=resume_state['best_epoch']
        # Repair outputs a crash may have advanced beyond the last committed epoch.
        pd.DataFrame(history).to_csv(out/'training_metrics.csv',index=False)
        atomic_torch_save(resume_state['model'],out/'newest.pt')
        if best_epoch is not None:
            model.load_state_dict(resume_state['best_model'])
            atomic_torch_save(resume_state['best_model'],out/'best.pt')
            atomic_parquet_save(predict(model,data,cfg,cfg['train_end'],cfg['valid_end'],device),out/'validation_predictions.parquet')
        for name,bucket in (('epoch_scores.parquet','score_frames'),
                            ('val_scores.parquet','val_frames'),
                            ('final_bar_scores.parquet','final_frames')):
            path=resume_dir/name
            if path.exists():
                frame=pd.read_parquet(path)
                frame=frame[frame.epoch<=start_epoch]
                atomic_parquet_save(frame,path)
                {'score_frames':score_frames,'val_frames':val_frames,
                 'final_frames':final_frames}[bucket].append(frame)
        path=out/'aux_val_predictions.parquet'
        if path.exists():
            frame=pd.read_parquet(path)
            atomic_parquet_save(frame[frame.epoch<=start_epoch],path)
        restore_training_state(resume_state,model,optimizer,teacher,aug_generator)
        del resume_state
    for epoch in range(start_epoch, cfg['epochs']):
        if epoch==1 and stage=='freeze_input_linear':
            for param in model.temporal.encoder.input_proj[0].parameters():param.requires_grad_(False)
        if epoch==1 and stage=='slow_first_block':
            optimizer.param_groups[1]['lr']=cfg['learning_rate']*.1
        # freeze_input_after_epoch (weekly-architecture runbook §4.6): an
        # independent, research_stage-free key.  From epoch freeze_after+1 the
        # input Linear stops learning; it stays in its AdamW group and grad=None
        # makes AdamW skip both the update and the decay for it.
        freeze_after=cfg.get('freeze_input_after_epoch')
        if freeze_after is not None and epoch+1>int(freeze_after):
            layer=model.temporal.encoder.input_proj[0]
            assert isinstance(layer,torch.nn.Linear)
            for param in layer.parameters():
                param.requires_grad_(False)
                param.grad=None
        started=time.perf_counter();model.train();sums=np.zeros(9);steps=0;aux_seen=0;aux_hits=0
        financial_nll_sum=0.
        financial_pair_sum=0.
        financial_projection_sum=0.
        financial_tail_sum=0.
        random_pair_sum=0.
        latent_epoch_loss=0.;latent_epoch_weeks=0;latent_epoch_pairs=0
        week2_epoch_loss=0.;week2_epoch_weeks=0;week2_epoch_pairs=0
        indices=np.random.default_rng(cfg['seed']+epoch).permutation(train_dates)
        if v2 and cfg.get('smoke_batches'):
            indices=indices[:int(cfg['smoke_batches'])*cfg['batch_size']]
        augmented=0
        seen_inputs=0
        for offset in range(0,len(indices),cfg['batch_size']):
            ts=indices[offset:offset+cfg['batch_size']].tolist()
            date_weight = None if date_weights is None else torch.from_numpy(date_weights[ts]).to(device)
            windows,daily,market,mask,*dated=make_batch(data,ts,cfg,device)
            if aug_generator is not None:
                # This recipe only scores rows with all 30 features observed;
                # warm-up or carried feature rows must never be augmented.
                real_rows=np.stack([
                    data[14][t-cfg['lookback']+1:t+1].T
                    for t in ts
                ])
                windows,changed=mask_weekly_inputs(
                    windows,torch.from_numpy(real_rows).to(device),aug_groups,aug_generator)
                augmented+=int(changed.sum())
                seen_inputs+=windows.numel()
            if is_financial(cfg):
                output=model(windows,daily,market,mask)
                raw_targets=torch.from_numpy(y[ts]).to(device)
                valid=torch.from_numpy(label_valid[ts]).to(device)&mask
                loss=date_mean_nll(output,raw_targets,valid,model.family,
                                   **({'mean_s_ref':model.s_ref,
                                       'industry_projection_weight':cfg.get('industry_projection_weight', 0.)} if model.faithful else {}))
                if loss is None:
                    continue
                if model.faithful:
                    with torch.no_grad():
                        financial_nll_sum+=float(date_mean_nll(output,raw_targets,valid,model.family))
                if cfg.get('industry_projection_weight', 0.):
                    financial_projection_sum += float(industry_projection_loss(output, raw_targets, valid, model.s_ref).detach())
                pair_weight = cfg.get('within_group_pair_weight', 0.)
                if pair_weight:
                    pair = within_group_pair_loss(
                        output, raw_targets, valid, model.s_ref,
                        gap_weighted=cfg.get('within_group_pair_gap_weighted', False),
                    )
                    if pair is not None:
                        financial_pair_sum += float(pair.detach())
                        loss = loss + pair_weight * pair
                tail_weight = cfg.get('decision_tail_weight', 0.)
                if tail_weight:
                    labels = extreme_tail_labels(raw_targets.clamp(-.25,.25), valid, .2)
                    tail_losses = [torch.nn.functional.softplus(-labels[t, keep] * output['decision_score'][t, keep] / model.s_ref).mean()
                                   for t in range(len(ts)) if (keep := labels[t].ne(0)).any()]
                    if tail_losses:
                        tail = torch.stack(tail_losses).mean()
                        financial_tail_sum += float(tail.detach())
                        loss = loss + tail_weight * tail
                ls=lr=li=lin=lexp=lctrl=lp=laux=loss.detach()*0
            else:
                temporal,z,scores,industry_pred=model(windows,daily,market,mask,*dated)
                raw_targets=torch.from_numpy(y[ts]).to(device)
                targets=raw_targets if cfg.get('raw_return_ranking') else raw_targets.clamp(-.25,.25)
                valid=torch.from_numpy(label_valid[ts]).to(device)&mask
                score_sample_weight=None;rankic_sample_weight=None;long_pairs=[];short_pairs=[]
                if cfg.get('industry_balanced_weight'):
                    # Per-name weight 1/|valid names in its industry|, renormalised to
                    # mean 1 over the valid cross-section, so every industry carries
                    # the same loss mass regardless of how many names it fields.  The
                    # weights apply to the two ranking losses; the InfoNCE term is a
                    # representation objective and is deliberately left unweighted.
                    vf=valid.to(device=industry_group_ids.device,dtype=torch.float32)
                    gidx=industry_group_ids.view(1,-1).expand(len(scores),-1)
                    cnt=torch.zeros(len(scores),int(industry_group_ids.max())+1,
                                    device=industry_group_ids.device).scatter_add_(1,gidx,vf)
                    w=(1.0/cnt.clamp_min(1.0)).gather(1,gidx)*vf
                    w=w*(vf.sum()/w.sum().clamp_min(1e-12))
                    score_sample_weight=w
                    rankic_sample_weight=w.clone()
                if score_smoothing:
                    labels,score_sample_weight,rankic_sample_weight=smoothed_score_targets(
                        raw_targets,valid,fraction=cfg.get('continuous_tail_fraction',cfg['extreme_frac']),
                        strength=cfg.get('continuous_tail_strength',2.),
                        temperature=cfg.get('continuous_tail_temperature',.03),
                        middle_fraction=cfg.get('score_middle_fraction',.25),
                        middle_weight=cfg.get('score_middle_weight',.05))
                    if not rankic_tail_weighting:rankic_sample_weight=None
                elif continuous:
                    score_sample_weight=continuous_tail_weights(
                        raw_targets,valid,fraction=cfg.get('continuous_tail_fraction',cfg['extreme_frac']),
                        strength=cfg.get('continuous_tail_strength',2.),
                        temperature=cfg.get('continuous_tail_temperature',.03))
                    rankic_sample_weight=score_sample_weight if rankic_tail_weighting else None
                if stratified:
                    labels,score_sample_weight,long_pairs,short_pairs=build_stratified_tail_targets(
                        targets,valid,fraction=cfg['extreme_frac'],count=cfg.get('extreme_count') or 0,
                        long_core=cfg.get('tail_long_core',20),short_core=cfg.get('tail_short_core',10),
                        long_candidate=cfg.get('tail_long_candidate',150),short_candidate=cfg.get('tail_short_candidate',100),
                        core_weight=cfg.get('tail_core_weight',3.),candidate_weight=cfg.get('tail_candidate_weight',1.),
                        coarse_weight=cfg.get('tail_coarse_weight',.5))
                elif not score_smoothing:
                    labels=extreme_tail_labels(targets,valid,float(cfg['extreme_frac']),
                        int(cfg['extreme_count']) if cfg.get('extreme_count') else None,
                        industry_group_ids if cfg.get('tail_within_industry') else None)
                if cfg.get('cell_balanced_weight'):
                    if cells is None:
                        raise ValueError('cell_balanced_weight requires arrays() to return cell ids')
                    cw=cell_sample_weights(torch.from_numpy(cells[ts]).to(device),valid,
                        alpha=cfg.get('cell_balance_alpha',.5),cap=cfg.get('cell_balance_cap',3.),
                        floor=cfg.get('cell_balance_floor',.2))
                    score_sample_weight=cw if score_sample_weight is None else score_sample_weight*cw
                    rankic_sample_weight=cw if rankic_sample_weight is None else rankic_sample_weight*cw
                ls=extreme_score_loss(scores,labels,temperature=cfg.get('score_temperature',1.),sample_weight=score_sample_weight,date_weight=date_weight)
                lr=rank_ic_loss(scores,targets,mask=valid,temperature=cfg['rankic_temperature'],sample_weight=rankic_sample_weight,date_weight=date_weight)
                lp=scores.sum()*0
                if pair_weight:
                    lp=cutoff_pairwise_loss(scores,long_pairs,short_pairs,cfg.get('cutoff_pairwise_temperature',1.))
                li=scores.sum()*0
                if teacher is not None:
                    future_mask=np.stack([available[t+1:t+horizon+1].all(0) for t in ts])
                    contrastive_mask=mask&torch.from_numpy(future_mask).to(device)
                    if (contrastive_mask.sum(1)>=2).any():
                        future=np.stack([X[t+1:t+horizon+1].transpose(1,0,2) for t in ts])
                        flat=torch.from_numpy(future.reshape(-1,horizon,X.shape[-1])).to(device)
                        with torch.no_grad():
                            key=torch.cat([teacher(chunk) for chunk in flat.split(cfg['encoder_chunk_size'])]).reshape(len(ts),len(symbols),-1)
                        query=F.normalize(model.temporal.projector(temporal),dim=-1)
                        groups=industry_group_ids[None].expand(len(ts),-1) if within_industry else None
                        li=masked_infonce(query,key,contrastive_mask,cfg['temperature'],groups=groups,date_weight=date_weight)
                loss=cfg['score_weight']*ls+cfg['rankic_weight']*lr+cfg['temporal_weight']*li+pair_weight*lp
                if random_pair_weight:
                    random_pairs = random_pair_loss(scores, targets, valid, seed=cfg['seed'],
                                                    date_indices=ts, epoch=epoch)
                    loss = loss + random_pair_weight * random_pairs
                    random_pair_sum += float(random_pairs.detach())
                if week2_weight:
                    valid2=torch.from_numpy(week2_valid[ts]).to(device)&mask
                    use_week=valid2.sum(dim=1)>=10
                    if use_week.any():
                        target2=torch.from_numpy(week2_y[ts]).to(device)
                        pred2=model.week2_head(z[use_week]).squeeze(-1)
                        loss2=rank_ic_loss(pred2,target2[use_week].clamp(-.25,.25),
                            mask=valid2[use_week],temperature=cfg['rankic_temperature'])
                        loss=loss+week2_weight*loss2
                        count2=int(use_week.sum().item())
                        week2_epoch_loss+=float(loss2.detach())*count2
                        week2_epoch_weeks+=count2
                        week2_epoch_pairs+=int(valid2[use_week].sum().item())
                lin=scores.sum()*0
                if ind_weight:
                    lin=rank_ic_loss(industry_pred,torch.from_numpy(industry_labels[ts]).to(device),
                        mask=torch.from_numpy(industry_mask[ts]).to(device),temperature=cfg['rankic_temperature'])
                    loss=loss+ind_weight*lin
                lexp=scores.sum()*0
                if exposure_weight:
                    if dated:
                        groups=dated[0]
                        keep=mask & groups.ne(cfg['unknown_industry_id'])
                    else:
                        groups=industry_group_ids[None].expand(len(ts),-1)
                        keep=mask
                    lexp=industry_exposure_loss(scores,groups,keep,date_weight=date_weight)
                    loss=loss+exposure_weight*lexp
                lctrl=scores.sum()*0
                if ctrl_weight:
                    # Last week's scores are a fixed reference: the control acts on w_t given
                    # w_{t-1}, which is the same asymmetry the deployed no-trade band has.
                    with torch.no_grad():
                        past_windows,past_daily,past_market,past_valid=make_batch(data,[t-1 for t in ts],cfg,device)
                        _,_,past_scores,_=model(past_windows,past_daily,past_market,past_valid)
                    lctrl=weekly_stickiness_loss(scores,past_scores,mask,past_valid,ctrl_temp,ctrl_capacity,ctrl_dead_zone)
                    loss=loss+ctrl_weight*lctrl
                laux=scores.sum()*0
                if aux_weight:
                    # Next-week delta prediction on the temporal representation.  The
                    # mask is per stock-week: a name with an invalid auxiliary label
                    # drops out of the auxiliary term only and never touches the
                    # main losses.  Huber(delta=1) in the standardized space, mean
                    # over valid dimensions first, then over dates with any valid
                    # name; a date with none contributes a graph-connected zero.
                    target=torch.from_numpy(aux_delta[ts]).to(device)
                    keep=torch.from_numpy(aux_valid[ts]).to(device)&torch.isfinite(target).all(-1)
                    delta_hat=model.nextweek_head(temporal)
                    # Invalid slots carry NaN targets; sanitise before the residual
                    # or huber(nan)*0 poisons the sum (nan*0 is nan, not zero).
                    safe=torch.nan_to_num(target,nan=0.,posinf=0.,neginf=0.)
                    error=delta_hat-safe
                    absolute=error.abs()
                    huber=torch.where(absolute<=1.,.5*error.square(),absolute-.5)
                    dim_valid=keep.unsqueeze(-1)&torch.isfinite(target)
                    per_date=(huber*dim_valid).sum((1,2))/dim_valid.sum((1,2)).clamp_min(1)
                    date_valid=dim_valid.sum((1,2))>0
                    if bool(date_valid.any()):
                        laux=(per_date*date_valid).sum()/date_valid.sum()
                    aux_seen+=int(keep.numel());aux_hits+=int(keep.sum())
                    loss=loss+aux_weight*laux
                # A2/A1 next-week latent point prediction (weekly-architecture
                # runbook §4.5).  Added after every baseline loss is merged and
                # before zero_grad.  Targets come from the shared EMA teacher's
                # weekly projection of X[t+1]; weeks without any valid target are
                # skipped and only reduce the logged coverage.  Per-date equal
                # weighting, ramped over the first aux epoch's batches.
                latent_start=int(cfg.get('next_latent_start',2))
                latent_weight=float(cfg.get('next_latent_lambda',0.))
                if getattr(model,'next_latent',None) is not None and epoch+1>=latent_start:
                    following=np.asarray(ts)+1
                    assert teacher is not None and np.all(following<cfg['train_end'])
                    valid_next=torch.as_tensor(available[following]&observed[following],device=device)
                    keep=mask&valid_next
                    if bool(keep.any()):
                        x_next=torch.from_numpy(X[following]).to(device)
                        with torch.no_grad():
                            target=F.normalize(teacher.encoder.project_week(x_next).float(),dim=-1,eps=1e-6)
                        prediction=F.normalize(model.next_latent(temporal).float(),dim=-1,eps=1e-6)
                        error=(prediction-target).square().sum(-1)
                        per_date=(error*keep).sum(1)/keep.sum(1).clamp_min(1)
                        latent_loss=per_date[keep.any(1)].mean()
                        ramp=min(1.,(offset+len(ts))/len(indices)) if epoch+1==latent_start else 1.
                        loss=loss+latent_weight*ramp*latent_loss
                        latent_epoch_loss+=float(latent_loss.item());latent_epoch_weeks+=1
                        latent_epoch_pairs+=int(keep.sum())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=(model.clip_grad_norm_(1.) if is_financial(cfg)
                  else torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            if teacher is not None:
                with torch.no_grad():
                    for target,online in zip(teacher.parameters(),model.temporal.parameters()):
                        target.lerp_(online,1-cfg['ema_decay'])
            sums += [loss.item(),ls.item(),lr.item(),li.item(),lin.item(),lexp.item(),lctrl.item(),lp.item(),laux.item()];steps+=1
            if steps==1 or steps%5==0:
                print(f'{out.name} epoch {epoch+1}/{cfg["epochs"]} batch {steps}/{int(np.ceil(len(indices)/cfg["batch_size"]))} loss={loss.item():.4f}',flush=True)
        if not steps:
            raise ValueError('epoch has no supervised optimizer steps')
        val=predict(model,data,cfg,cfg['train_end'],cfg['valid_end'],device)
        if cfg.get('save_epoch_scores'):
            # Snapshot ensemble in score space: test-section scores only (a few
            # MB per epoch, no extra model weights are ever written).
            test_scores=predict(model,data,cfg,cfg['valid_end'],len(data[6])-1,device)
            test_scores.insert(0,'epoch',epoch+1)
            score_frames.append(test_scores)
            atomic_parquet_save(pd.concat(score_frames,ignore_index=True),out/'epoch_scores.parquet')
            # Same per-epoch snapshot for the final (label-less) bar so the
            # deployed ens3 fusion can also score the live week; its
            # realized_return/rank_ic stay NaN and it never enters backtests.
            final_frames.append(predict(model,data,cfg,len(data[6])-1,len(data[6]),device).assign(epoch=epoch+1))
            atomic_parquet_save(pd.concat(final_frames,ignore_index=True),out/'final_bar_scores.parquet')
            # Per-epoch VALIDATION predictions enable sub-window selection
            # studies (e.g. 1-year vs 2-year val folds) without retraining.
            val_frames.append(val.assign(epoch=epoch+1))
            atomic_parquet_save(pd.concat(val_frames,ignore_index=True),out/'val_scores.parquet')
        _,vm=long_only_backtest(val,cfg['long_top_n'],cfg['transaction_cost_bps'],52)
        # Epoch selection is scored on labels that have already matured before
        # the first test decision (A5, ledger §26.12.1).  The last N validation
        # anchors' t+2-open labels resolve at/after the first test entry, so
        # they must not inform the choice.  Reporting keeps the FULL window
        # (val_long_net_sharpe) for comparability with historical runs; the
        # selection metric is stored separately.
        sel_lag=0 if v2 else int(cfg.get('selection_label_lag_weeks',2))
        vm_sel=vm
        if sel_lag>0:
            _d=pd.DatetimeIndex(sorted(pd.to_datetime(val.date).unique()))
            if len(_d)>sel_lag+40:
                _cut=_d[-1-sel_lag]
                _mask=pd.to_datetime(val.date)<=_cut
                vm_sel=long_only_backtest(val[_mask.to_numpy()],
                                          cfg['long_top_n'],cfg['transaction_cost_bps'],52)[1]
        val_ic_dates=val.groupby('date').rank_ic.first().dropna()
        aux_val_rows=[]
        val_aux_rate=None
        if aux_weight:
            # Per-epoch validation auxiliary diagnostics: predictions and targets
            # for every valid stock-week (wide d0..d8 columns), plus the valid
            # rate.  Used by the E2 post-training reconstruction comparison.
            model.eval()
            symbol_arr=np.asarray(symbols)
            _idx=list(range(max(cfg['train_end'],cfg['lookback']),cfg['valid_end']))
            _val_dates=set()
            with torch.no_grad():
                for _off in range(0,len(_idx),cfg.get('eval_batch_dates',4)):
                    _ts=_idx[_off:_off+cfg.get('eval_batch_dates',4)]
                    _w,_d2,_m,_k=make_batch(data,_ts,cfg,device)
                    _temporal,_,_,_=model(_w,_d2,_m,_k)
                    _pred=model.nextweek_head(_temporal).cpu().numpy()
                    for _j,_t in enumerate(_ts):
                        _keep=aux_valid[_t]
                        _val_dates.add(_t)
                        _frame=pd.DataFrame(dict(
                            date=str(pd.Timestamp(dates[_t]).date()),
                            symbol=symbol_arr[_keep],epoch=epoch+1))
                        for _i in range(aux_delta.shape[2]):
                            _frame[f'pred{_i}']=_pred[_j][_keep,_i]
                            _frame[f'tgt{_i}']=aux_delta[_t][_keep,_i]
                        aux_val_rows.append(_frame)
            val_aux_rate=(sum(len(r) for r in aux_val_rows)
                          / (len(_val_dates)*len(symbols))) if _val_dates else None
        ind_ic=industry_weekly_ic(model,data,cfg,industry_labels,industry_mask,cfg['train_end'],cfg['valid_end'],device) if ind_weight else []
        ind_mean=float(np.mean(ind_ic)) if len(ind_ic)>1 else None
        ind_se=float(np.std(ind_ic,ddof=1)/np.sqrt(len(ind_ic))) if len(ind_ic)>1 else None
        means=sums/steps
        wide=val.pivot(index='date',columns='symbol',values='score').reindex(columns=symbols)
        dense=wide.to_numpy(dtype=float)
        val_groups=ids
        val_mask=~np.isnan(dense)
        if is_financial(cfg):
            val_groups=val.pivot(index='date',columns='symbol',values='effective_group').reindex(columns=symbols).to_numpy()
        if len(data)>15 and data[15] is not None:
            val_positions=pd.DatetimeIndex(dates).get_indexer(pd.to_datetime(wide.index))
            val_groups=data[15][val_positions]
            val_mask &= val_groups != cfg['unknown_industry_id']
        val_eta2,val_eta2_baseline=score_eta2_by_industry(dense,val_mask,val_groups)
        eta2_text='n/a' if val_eta2 is None else f'{val_eta2:.4f}'
        rec=dict(epoch=epoch+1,loss=means[0],score_loss=means[1],rankic_loss=means[2],temporal_loss=means[3],industry_loss=means[4],
            exposure_loss=means[5],control_loss=means[6],cutoff_pairwise_loss=means[7],
            aux_loss=means[8],aux_valid_rate=(aux_hits/aux_seen if aux_seen else None),
            latent_weight=(float(cfg.get('next_latent_lambda',0.)) if getattr(model,'next_latent',None) is not None else None),
            latent_loss=(latent_epoch_loss/latent_epoch_weeks if latent_epoch_weeks else None),
            latent_valid_pairs=latent_epoch_pairs,latent_valid_weeks=latent_epoch_weeks,
            week2_loss=(week2_epoch_loss/week2_epoch_weeks if week2_epoch_weeks else None),
            week2_valid_weeks=week2_epoch_weeks,week2_valid_pairs=week2_epoch_pairs,
            val_aux_valid_rate=val_aux_rate,
            val_score_eta2_industry=val_eta2,val_score_eta2_neutral=val_eta2_baseline,
            val_rank_ic=float(val_ic_dates.mean()),val_long_net_sharpe=vm['net']['sharpe'],
            val_long_net_sharpe_selection=vm_sel['net']['sharpe'],
            val_industry_ic=ind_mean,val_industry_tstat=(None if ind_se in (None,0.) else ind_mean/ind_se),
            val_long_net_return=vm['net']['total_return'],val_long_max_drawdown=vm['net']['max_drawdown'],
            val_long_relative_return=vm['relative_return'],val_long_turnover=vm['mean_turnover'],
            val_ic_std=float(val_ic_dates.std(ddof=1)),val_icir=float(val_ic_dates.mean()/val_ic_dates.std(ddof=1)),
            val_ic_tstat=float(val_ic_dates.mean()/(val_ic_dates.std(ddof=1)/np.sqrt(len(val_ic_dates)))),
            **gate_summary(model),seconds=time.perf_counter()-started)
        if random_pair_weight:
            rec.update(random_pair_loss=random_pair_sum/steps,
                       random_pair_weight=random_pair_weight,
                       random_pair_opponents=4, random_pair_temperature=1.)
        if is_financial(cfg):
            rec['nll_loss']=financial_nll_sum/steps if model.faithful else means[0]
            if model.faithful:
                rec['training_objective_loss']=means[0]
                if not cfg.get('within_group_pair_weight', 0.):
                    rec['faithful_objective_loss']=means[0]
                rec['within_group_pair_loss']=financial_pair_sum/steps
                rec['industry_projection_loss']=financial_projection_sum/steps
                rec['decision_tail_loss']=financial_tail_sum/steps
                rec['mean_half_mse_standardized']=means[0]-rec['nll_loss']-cfg.get('within_group_pair_weight',0.)*rec['within_group_pair_loss']-cfg.get('industry_projection_weight',0.)*rec['industry_projection_loss']-cfg.get('decision_tail_weight',0.)*rec['decision_tail_loss']
            shape=(val.date.nunique(),len(symbols))
            target=torch.tensor(val.realized_return.to_numpy().reshape(shape),dtype=torch.float64)
            diagnostic=dict(mu_log=torch.tensor(val.mu_log.to_numpy().reshape(shape)),
                            scale_log=torch.tensor(val.scale_log.to_numpy().reshape(shape)),
                            available=torch.tensor(val.available.to_numpy().reshape(shape)))
            nll=date_mean_nll(diagnostic,target,torch.isfinite(target),model.family)
            rec['val_nll']=float(nll) if nll is not None else None
            scale_values=val.loc[val.available,'scale_log']
            rec['val_scale_p50']=float(scale_values.quantile(.5))
            rec['val_scale_p95']=float(scale_values.quantile(.95))
            rec['val_scale_near_floor_fraction']=float((scale_values<=.011*float(model.s_ref)).mean())
        if D is not None:
            rec['daily_gate_mean']=float(torch.tanh(model.daily_gate).mean())
            rec['daily_gate_open']=float(torch.tanh(model.daily_gate).abs().mean())
        if aug_generator is not None:
            rec['augmentation_fraction']=augmented/max(seen_inputs,1)
        history.append(rec);pd.DataFrame(history).to_csv(out/'training_metrics.csv',index=False)
        if aux_val_rows:
            # Append-and-write keeps the file current if a long run is interrupted.
            path=out/'aux_val_predictions.parquet'
            frame=pd.concat(aux_val_rows,ignore_index=True)
            if path.exists():
                old=pd.read_parquet(path)
                frame=pd.concat([old[old.epoch!=epoch+1],frame],ignore_index=True)
            atomic_parquet_save(frame,path)
        atomic_torch_save(model.state_dict(),out/'newest.pt')
        if cfg.get('save_epoch_checkpoints'):
            # Per-epoch weights so the deployed ens3 (best +/-1) can be scored bar
            # by bar outside training; without this only best/newest exist and the
            # live signal cannot reproduce the backtested 3-epoch fusion.
            atomic_torch_save(model.state_dict(),out/f'epoch_{epoch+1}.pt')
        if getattr(model,'next_latent',None) is not None and teacher is not None:
            # A-arm only (runbook §4.5): the frozen EMA teacher used for the aux
            # target, per epoch, so diagnostics and persistence baselines can
            # reload it with strict_load into model.temporal.
            atomic_torch_save(teacher.state_dict(),out/f'ema_epoch_{epoch+1}.pt')
        if vm_sel['net']['sharpe']>best+1e-12:
            best=float(vm_sel['net']['sharpe']);best_epoch=epoch+1
            atomic_torch_save(model.state_dict(),out/'best.pt')
            atomic_parquet_save(val,out/'validation_predictions.parquet')
        save_training_state(out/'training_state.pt',cfg,identity,model,optimizer,teacher,aug_generator,
                            epoch+1,best,best_epoch,history,
                            torch.load(out/'best.pt',map_location='cpu',weights_only=True) if best_epoch is not None else None)
        write_json(out/'metrics.json',dict(meta,status='training',completed_epochs=epoch+1,best_epoch=best_epoch,selection_value=best))
        print(f'{out.name} EPOCH {epoch+1} val_sharpe={vm["net"]["sharpe"]:.4f} sel_sharpe={vm_sel["net"]["sharpe"]:.4f} val_ic={rec["val_rank_ic"]:.4f} val_icir={rec["val_icir"]:.4f} val_eta2={eta2_text} exp_loss={means[5]:.4f} ctrl_loss={means[6]:.4f} pair_loss={means[7]:.4f} gate={rec["gate_mean"]} seconds={rec["seconds"]:.1f}',flush=True)
    result=dict(meta,status='validation_complete',completed_epochs=cfg['epochs'],best_epoch=best_epoch,selection_value=best)
    if v2:
        result.update(selection_scope='scheduled_open_proxy_diagnostic_only',
                      smoke_batches=cfg.get('smoke_batches'))
    write_json(out/'metrics.json',result)
    del teacher,optimizer,model
    if device.type=='cuda':torch.cuda.empty_cache()
    return result if defer_test else evaluate(out,data=data)


def evaluate(out,data=None):
    out=Path(out);cfg=yaml.safe_load((out/'config.yaml').read_text())
    if is_financial(cfg):
        raise ValueError('financial_return_v1 requires standard probability scoring/evaluation')
    if data is None:data=arrays(cfg)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=HierarchicalRanker(cfg,data[8]).to(device)
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True))
    pred=predict(model,data,cfg,cfg['valid_end'],len(data[6])-1,device)
    pred.to_parquet(out/'predictions.parquet',index=False)
    # Legacy auxiliary plots retain their old accounting; primary evaluation is Top20 simple-return net backtest.
    metrics=backtest(pred,cfg,out)
    curve,long=long_only_backtest(pred,cfg['long_top_n'],cfg['transaction_cost_bps'],52,
                                  execution=cfg.get('execution','close_to_close_proxy'))
    curve.to_csv(out/'long_only_curve.csv',index=False)
    old=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    metrics.update(old);metrics.update(long_only=long,status='complete',auxiliary_accounting='legacy log-return long-short; use long_only for primary results')
    write_json(out/'metrics.json',metrics)
    plot_all(out,pd.read_csv(out/'training_metrics.csv'),pred)
    return metrics
