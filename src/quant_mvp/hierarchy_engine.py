"""Chronological training for the hierarchical market model; optional deferred test."""
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml
from .hierarchy_data import arrays, industry_excess
from .hierarchy import HierarchicalRanker, industry_exposure_baseline, industry_exposure_loss, masked_infonce, weekly_stickiness_loss
from .model import extreme_score_loss, rank_ic_loss
from .long_only import long_only_backtest
from .engine import backtest, plot_all


def write_json(path, obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def is_gate(name):
    return name.endswith('attn_gate') or name.endswith('ffn_gate')


def gate_summary(model):
    """Mean and spread of the opened residual gates, i.e. tanh(theta)."""
    opened=[torch.tanh(param) for name,param in model.named_parameters() if is_gate(name)]
    if not opened:return dict(gate_mean=None,gate_min=None,gate_max=None)
    flat=torch.cat(opened)
    return dict(gate_mean=float(flat.mean()),gate_min=float(flat.min()),gate_max=float(flat.max()))


def make_batch(data, dates, cfg, device):
    X, _, _, M, available = data[:5]
    windows=np.stack([X[t-cfg['lookback']+1:t+1].transpose(1,0,2) for t in dates])
    return (torch.from_numpy(windows).to(device),torch.from_numpy(M[dates]).to(device),
            torch.from_numpy(available[dates]).to(device))


@torch.no_grad()
def predict(model, data, cfg, start, end, device):
    model.eval()
    X,y,valid_y,M,available,observed,dates,symbols=data[:8]
    rows=[]
    indices=list(range(max(start,cfg['lookback']),end))
    size=cfg.get('eval_batch_dates',4)
    for off in range(0,len(indices),size):
        ts=indices[off:off+size]
        _,_,scores,_=model(*make_batch(data,ts,cfg,device))
        values=scores.cpu().numpy()
        for j,t in enumerate(ts):
            values[j,~available[t]]=np.nan
            valid=valid_y[t]&available[t]
            ic=pd.Series(values[j,valid]).corr(pd.Series(y[t,valid]),method='spearman') if valid.sum()>2 else np.nan
            rows.extend(dict(date=str(pd.Timestamp(dates[t]).date()),symbol=s,score=float(values[j,k]),
                realized_return=float(y[t,k]) if valid_y[t,k] else np.nan,rank_ic=float(ic)) for k,s in enumerate(symbols))
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
    for row,keep in zip(dense,available):
        score,group=np.asarray(row,dtype=float)[keep],np.asarray(ids)[keep]
        sd=score.std()
        if len(score)<3 or sd<=1e-9:continue
        z=(score-score.mean())/sd
        counts=np.bincount(group)
        means=np.bincount(group,weights=z)/np.maximum(counts,1)
        values.append(float((counts*means**2).sum()/len(z)))
        baseline=industry_exposure_baseline(counts)
        if baseline is not None:baselines.append(baseline)
    if not values:return None,None
    return float(np.mean(values)),(float(np.mean(baselines)) if baselines else None)


def metadata(cfg, model, device):
    return dict(architecture='L-G-L-G-L; 29 industry tokens + one market token; no stock cross30',
        model_type='hierarchical_market', market_tokens=1, global_layers=cfg.get('global_layers',2),
        attention_heads=cfg.get('attention_heads',4), data_source='yahoo', frequency='weekly',
        n_stocks=cfg['n_stocks'],n_days=cfg['n_days'],n_score_heads=1,lookback=cfg['lookback'],
        embedding_dim=cfg['embedding_dim'],epochs=cfg['epochs'],backend=model.temporal.encoder.backend,
        device=str(device),temporal_horizon=cfg['temporal_horizon'],temporal_weight=cfg['temporal_weight'],
        rankic_weight=cfg['rankic_weight'],score_weight=cfg['score_weight'],transaction_cost_bps=cfg['transaction_cost_bps'],
        industry_classification=cfg['industry_classification'],universe_name=cfg['universe_name'],
        objective='extreme_score_plus_full_rankic'+('_plus_temporal_infonce' if cfg['temporal_weight'] else '')
            +('_plus_industry_exposure_penalty' if float(cfg.get('industry_exposure_weight',0.)) else '')
            +('_plus_industry_excess_head' if float(cfg.get('industry_head_weight',0.)) else '')
            +('_plus_weekly_stickiness_penalty' if float(cfg.get('turnover_control_weight',0.)) else '')
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
        relation_gate_init=cfg.get('relation_gate_init'),
        relation_gate_lr_scale=cfg.get('relation_gate_lr_scale',1.0),
        relation_gates_frozen=bool(cfg.get('freeze_relation_gates')),
        contrastive_location='temporal representation, independent projection; EMA temporal encoder and projection',
        contrastive_negatives=cfg.get('contrastive_negatives','market'))


def train(cfg, out, defer_test=False):
    out=Path(out)
    if out.exists():raise FileExistsError(f'Existing run preserved: {out}')
    torch.set_num_threads(cfg.get('torch_num_threads',4))
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    data=arrays(cfg)
    X,y,valid_y,M,available,observed,dates,symbols,ids,preprocessing=data
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=HierarchicalRanker(cfg,ids).to(device)
    named=dict(model.named_parameters())
    frozen_gates=[name for name in named if is_gate(name) and cfg.get('freeze_relation_gates')]
    for name in frozen_gates:named[name].data.zero_();named[name].requires_grad_(False)
    trainable=[(name,param) for name,param in named.items() if name not in frozen_gates]
    gate_ids={id(param) for name,param in trainable if is_gate(name)}
    shared=[param for name,param in trainable if id(param) not in gate_ids]
    gated=[param for name,param in trainable if id(param) in gate_ids]
    scale=float(cfg.get('relation_gate_lr_scale',1.))
    groups=[dict(params=shared)]
    if gated:groups.append(dict(params=gated,lr=cfg['learning_rate']*scale))
    negatives=cfg.get('contrastive_negatives','market')
    if negatives not in ('market','industry'):raise ValueError(f"Unknown contrastive_negatives: {negatives}")
    within_industry=negatives=='industry'
    industry_group_ids=torch.from_numpy(np.ascontiguousarray(ids)).to(device)
    ind_weight=float(cfg.get('industry_head_weight',0.))
    if ind_weight:
        industry_labels,industry_mask=industry_excess(y,valid_y,available,ids,cfg.get('industry_min_stocks',10))
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
    optimizer=torch.optim.AdamW(groups,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    out.mkdir(parents=True)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    write_json(out/'preprocessing.json',preprocessing)
    for filename in ['universe.csv','manifest.json','feature_repair_audit.csv']:
        source=Path(cfg['universe_path']).with_name(filename)
        if source.exists():shutil.copy2(source,out/filename)
    meta=metadata(cfg,model,device)
    write_json(out/'metrics.json',dict(meta,status='training'))
    horizon=cfg['temporal_horizon']
    reserve=max(horizon,cfg.get('common_training_horizon',horizon))
    train_dates=np.array([t for t in range(cfg['lookback'],cfg['train_end']-reserve)
                         if (valid_y[t]&available[t]).sum()>=10])
    write_json(out/'training_protocol.json',dict(training_dates=[str(pd.Timestamp(dates[t]).date()) for t in train_dates],
        reserve_horizon=reserve, mask='current observed history for ranking; current+future observable masks for training-only InfoNCE',
        no_test_during_training=True,seed=cfg['seed']))
    best=-float('inf');best_epoch=None;history=[]
    for epoch in range(cfg['epochs']):
        started=time.perf_counter();model.train();sums=np.zeros(7);steps=0
        indices=np.random.default_rng(cfg['seed']+epoch).permutation(train_dates)
        for offset in range(0,len(indices),cfg['batch_size']):
            ts=indices[offset:offset+cfg['batch_size']].tolist()
            windows,market,mask=make_batch(data,ts,cfg,device)
            temporal,z,scores,industry_pred=model(windows,market,mask)
            targets=torch.from_numpy(y[ts]).to(device).clamp(-.25,.25)
            valid=torch.from_numpy(valid_y[ts]).to(device)&mask
            labels=torch.zeros_like(targets)
            for row in range(len(ts)):
                ids_valid=torch.where(valid[row])[0]
                count=max(1,int(round(len(ids_valid)*cfg['extreme_frac'])))
                if cfg.get('extreme_count'):count=min(int(cfg['extreme_count']),len(ids_valid)//2)
                order=ids_valid[torch.argsort(targets[row,ids_valid])]
                labels[row,order[:count]]=-1;labels[row,order[-count:]]=1
            ls=extreme_score_loss(scores,labels,temperature=cfg.get('score_temperature',1.))
            lr=rank_ic_loss(scores,targets,mask=valid,temperature=cfg['rankic_temperature'])
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
                    li=masked_infonce(query,key,contrastive_mask,cfg['temperature'],groups=groups)
            loss=cfg['score_weight']*ls+cfg['rankic_weight']*lr+cfg['temporal_weight']*li
            lin=scores.sum()*0
            if ind_weight:
                lin=rank_ic_loss(industry_pred,torch.from_numpy(industry_labels[ts]).to(device),
                    mask=torch.from_numpy(industry_mask[ts]).to(device),temperature=cfg['rankic_temperature'])
                loss=loss+ind_weight*lin
            lexp=scores.sum()*0
            if exposure_weight:
                lexp=industry_exposure_loss(scores,industry_group_ids[None].expand(len(ts),-1),mask)
                loss=loss+exposure_weight*lexp
            lctrl=scores.sum()*0
            if ctrl_weight:
                # Last week's scores are a fixed reference: the control acts on w_t given
                # w_{t-1}, which is the same asymmetry the deployed no-trade band has.
                with torch.no_grad():
                    past_windows,past_market,past_valid=make_batch(data,[t-1 for t in ts],cfg,device)
                    _,_,past_scores,_=model(past_windows,past_market,past_valid)
                lctrl=weekly_stickiness_loss(scores,past_scores,mask,past_valid,ctrl_temp,ctrl_capacity,ctrl_dead_zone)
                loss=loss+ctrl_weight*lctrl
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            if teacher is not None:
                with torch.no_grad():
                    for target,online in zip(teacher.parameters(),model.temporal.parameters()):
                        target.lerp_(online,1-cfg['ema_decay'])
            sums += [loss.item(),ls.item(),lr.item(),li.item(),lin.item(),lexp.item(),lctrl.item()];steps+=1
            if steps==1 or steps%5==0:
                print(f'{out.name} epoch {epoch+1}/{cfg["epochs"]} batch {steps}/{int(np.ceil(len(indices)/cfg["batch_size"]))} loss={loss.item():.4f}',flush=True)
        val=predict(model,data,cfg,cfg['train_end'],cfg['valid_end'],device)
        _,vm=long_only_backtest(val,cfg['long_top_n'],cfg['transaction_cost_bps'],52)
        val_ic_dates=val.groupby('date').rank_ic.first().dropna()
        ind_ic=industry_weekly_ic(model,data,cfg,industry_labels,industry_mask,cfg['train_end'],cfg['valid_end'],device) if ind_weight else []
        ind_mean=float(np.mean(ind_ic)) if len(ind_ic)>1 else None
        ind_se=float(np.std(ind_ic,ddof=1)/np.sqrt(len(ind_ic))) if len(ind_ic)>1 else None
        means=sums/steps
        wide=val.pivot(index='date',columns='symbol',values='score').reindex(columns=symbols)
        dense=wide.to_numpy(dtype=float)
        val_eta2,val_eta2_baseline=score_eta2_by_industry(dense,~np.isnan(dense),ids)
        eta2_text='n/a' if val_eta2 is None else f'{val_eta2:.4f}'
        rec=dict(epoch=epoch+1,loss=means[0],score_loss=means[1],rankic_loss=means[2],temporal_loss=means[3],industry_loss=means[4],
            exposure_loss=means[5],control_loss=means[6],val_score_eta2_industry=val_eta2,val_score_eta2_neutral=val_eta2_baseline,
            val_rank_ic=float(val_ic_dates.mean()),val_long_net_sharpe=vm['net']['sharpe'],
            val_industry_ic=ind_mean,val_industry_tstat=(None if ind_se in (None,0.) else ind_mean/ind_se),
            val_long_net_return=vm['net']['total_return'],val_long_max_drawdown=vm['net']['max_drawdown'],
            val_long_relative_return=vm['relative_return'],val_long_turnover=vm['mean_turnover'],
            val_ic_std=float(val_ic_dates.std(ddof=1)),val_icir=float(val_ic_dates.mean()/val_ic_dates.std(ddof=1)),
            val_ic_tstat=float(val_ic_dates.mean()/(val_ic_dates.std(ddof=1)/np.sqrt(len(val_ic_dates)))),
            **gate_summary(model),seconds=time.perf_counter()-started)
        history.append(rec);pd.DataFrame(history).to_csv(out/'training_metrics.csv',index=False)
        torch.save(model.state_dict(),out/'newest.pt')
        if vm['net']['sharpe']>best+1e-12:
            best=vm['net']['sharpe'];best_epoch=epoch+1
            torch.save(model.state_dict(),out/'best.pt')
            val.to_parquet(out/'validation_predictions.parquet',index=False)
        write_json(out/'metrics.json',dict(meta,status='training',completed_epochs=epoch+1,best_epoch=best_epoch,selection_value=best))
        print(f'{out.name} EPOCH {epoch+1} val_sharpe={vm["net"]["sharpe"]:.4f} val_ic={rec["val_rank_ic"]:.4f} val_icir={rec["val_icir"]:.4f} val_eta2={eta2_text} exp_loss={means[5]:.4f} ctrl_loss={means[6]:.4f} gate={rec["gate_mean"]:.4f} seconds={rec["seconds"]:.1f}',flush=True)
    result=dict(meta,status='validation_complete',completed_epochs=cfg['epochs'],best_epoch=best_epoch,selection_value=best)
    write_json(out/'metrics.json',result)
    del teacher,optimizer,model
    if device.type=='cuda':torch.cuda.empty_cache()
    return result if defer_test else evaluate(out,data=data)


def evaluate(out,data=None):
    out=Path(out);cfg=yaml.safe_load((out/'config.yaml').read_text())
    if data is None:data=arrays(cfg)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=HierarchicalRanker(cfg,data[8]).to(device)
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True))
    pred=predict(model,data,cfg,cfg['valid_end'],len(data[6])-1,device)
    pred.to_parquet(out/'predictions.parquet',index=False)
    # Legacy auxiliary plots retain their old accounting; primary evaluation is Top20 simple-return net backtest.
    metrics=backtest(pred,cfg,out)
    curve,long=long_only_backtest(pred,cfg['long_top_n'],cfg['transaction_cost_bps'],52)
    curve.to_csv(out/'long_only_curve.csv',index=False)
    old=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    metrics.update(old);metrics.update(long_only=long,status='complete',auxiliary_accounting='legacy log-return long-short; use long_only for primary results')
    write_json(out/'metrics.json',metrics)
    plot_all(out,pd.read_csv(out/'training_metrics.csv'),pred)
    return metrics
