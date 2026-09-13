"""Training, evaluation, backtest and static visualisation for the MVP."""
from __future__ import annotations
from pathlib import Path
import json, math, copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .data import load_data, dataset_arrays, FEATURES, TemporalCrossSectionDataset
from .model import (PairRanker, extreme_score_loss, group_decorrelation_loss,
                    temporal_infonce, rank_ic_loss, extreme_label_stats,
                    contrastive_diagnostics, block_rank_ic_loss)

def _rank(v):
    order=np.argsort(np.argsort(v)); return order.astype(float)/(len(v)-1 if len(v)>1 else 1)
def rank_ic(a,b):
    if len(a)<3 or np.std(a)<1e-9 or np.std(b)<1e-9:return np.nan
    return float(np.corrcoef(_rank(a),_rank(b))[0,1])

def train(cfg, out: Path):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); out.mkdir(parents=True, exist_ok=True)
    df=load_data(cfg.get('provider','synthetic'), cfg['cache_path'], cfg['n_stocks'],
                 cfg['n_days'], cfg['seed'], frequency=cfg.get('frequency', 'daily'))
    context_lengths = sorted({int(v) for v in cfg.get('context_lengths', [cfg.get('lookback', 40)]) if int(v) > 0})
    if not context_lengths:
        context_lengths = [int(cfg.get('lookback', 40))]
    max_lookback = max(context_lengths)
    horizons = sorted({int(v) for v in cfg.get('temporal_horizons', [cfg.get('temporal_horizon', 5)]) if int(v) > 0})
    if not horizons:
        horizons = [int(cfg.get('temporal_horizon', 5))]
    max_horizon = max(horizons)
    X,y,dates,syms,valid_y=dataset_arrays(
        df,max_lookback,cfg['n_features'],cfg['train_end'],return_valid=True)
    tr_end=min(cfg['train_end'],len(dates)-2); va_end=min(cfg['valid_end'],len(dates)-1)
    # One dataset row per date.  This keeps InfoNCE negatives inside the same
    # cross-section (other stocks on that date), while score labels still
    # select only the daily top/bottom tails.
    ds=TemporalCrossSectionDataset(
        X, y, dates, max_lookback, max_lookback, tr_end,
        horizon=max_horizon,
        extreme_frac=cfg.get('extreme_frac', 0.2), valid_y=valid_y)
    loader=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,drop_last=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PairRanker(cfg['n_features'],d_model=cfg.get('d_model',96),embedding_dim=cfg['embedding_dim'],n_layers=cfg.get('n_layers',4),dropout=cfg.get('dropout',0.1)).to(device)
    # Momentum target encoder for the future key.  It is never used by
    # inference and receives no gradient, which stabilises temporal InfoNCE
    # compared with feeding the same rapidly changing online encoder twice.
    target_encoder = copy.deepcopy(model.encoder).to(device)
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    hist=[]; best=-1e9
    for ep in range(cfg['epochs']):
        model.train(); sums=np.zeros(7); count=0
        tail_stats={'positive_count':0.0,'negative_count':0.0,'middle_count':0.0}
        cstats={'positive_cosine':0.0,'negative_cosine':0.0,'hardest_negative_cosine':0.0,
                'negative_pairs':0.0,'positive_pairs':0.0}
        block_pair_count=0.0; block_pair_accuracy=0.0
        for xq,xk,label,t_index,*_ in loader:
            xq,xk,label=xq.to(device),xk.to(device),label.to(device)
            bsz,nstocks=xq.shape[:2]
            ctx = int(context_lengths[0]) if len(context_lengths) == 1 else int(np.random.choice(context_lengths))
            xq = xq[..., -ctx:, :]
            zq_flat,sq_flat=model(xq.reshape(bsz*nstocks,*xq.shape[2:]))
            zq=zq_flat.reshape(bsz,nstocks,-1); sq=sq_flat.reshape(bsz,nstocks)
            with torch.no_grad():
                if len(horizons) == 1:
                    # Original single-positive path: one future window [t+1,
                    # ..., t+H] produces one key embedding per stock.
                    zk_flat=target_encoder(xk.reshape(bsz*nstocks,*xk.shape[2:]))
                    zk=zk_flat.reshape(bsz,nstocks,-1)
                else:
                    zk_seq=target_encoder(xk.reshape(bsz*nstocks,*xk.shape[2:]), return_sequence=True)
                    zk_flat=torch.stack([zk_seq[:, min(h, zk_seq.shape[1])-1] for h in horizons], dim=1)
                    zk=zk_flat.reshape(bsz,nstocks,len(horizons),-1)
            ls=extreme_score_loss(sq.reshape(-1),label.reshape(-1),cfg.get('score_temperature', 1.0))
            lg=group_decorrelation_loss(zq_flat, group_size=cfg.get('group_size',8))
            lt=temporal_infonce(zq,zk,cfg.get('temperature',0.1))
            # Optional differentiable cross-sectional RankIC objective.  The
            # target is looked up by date index; no realized return is fed to
            # the model input.  A zero default preserves old configurations.
            rw=float(cfg.get('rankic_weight',0.0))
            bw=float(cfg.get('block_rank_weight',0.0))
            # Both ranking objectives use the same detached next-period
            # cross-section.  Block ranking only constrains pairs whose
            # realised-return quartile differs; names within one quartile do
            # not produce a noisy pairwise gradient.
            target = None; vm = None
            if rw or bw:
                ti=t_index.detach().cpu().numpy().astype(int)
                target=torch.as_tensor(y[ti],dtype=sq.dtype,device=device)
                vm=torch.as_tensor(valid_y[ti],dtype=torch.bool,device=device) if valid_y is not None else None
            if rw:
                lric,batch_ic=rank_ic_loss(sq,target,mask=vm,
                    temperature=cfg.get('rankic_temperature',0.1),return_ic=True)
            else:
                lric=sq.sum()*0.0; batch_ic=sq.sum()*0.0
            if bw:
                lblock, block_stats = block_rank_ic_loss(
                    sq, target, mask=vm,
                    n_blocks=cfg.get('block_rank_blocks', 4),
                    temperature=cfg.get('block_rank_temperature', 0.1),
                    return_stats=True)
            else:
                lblock=sq.sum()*0.0
                block_stats={'pair_count': 0.0, 'pair_accuracy': 0.0,
                             'block_count': float(cfg.get('block_rank_blocks',4))}
            sw=float(cfg.get('score_weight',1.0)); tw=float(cfg.get('temporal_weight',1.0))
            group_w=float(cfg.get('group_weight',0.0))
            warmup=int(cfg.get('group_warmup_epochs',0))
            if ep < warmup:
                group_w=0.0
            loss=sw*ls+tw*lt+rw*lric+bw*lblock+group_w*lg
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            # EMA update after each optimizer step (MoCo-style target).
            decay=float(cfg.get('ema_decay',0.99))
            with torch.no_grad():
                for p_t,p in zip(target_encoder.parameters(), model.encoder.parameters()):
                    p_t.mul_(decay).add_(p, alpha=1.0-decay)
            sums += [float(loss.item()),float(ls.item()),float(lt.item()),float(lg.item()),float(lric.item()),float(batch_ic.item()),float(lblock.item())]; count += 1
            block_pair_count += float(block_stats.get('pair_count', 0.0))
            block_pair_accuracy += float(block_stats.get('pair_accuracy', 0.0))
            st=extreme_label_stats(label); tail_stats['positive_count'] += st['positive_count']; tail_stats['negative_count'] += st['negative_count']; tail_stats['middle_count'] += st['middle_count']
            cd=contrastive_diagnostics(zq.detach(),zk.detach());
            for k in cstats: cstats[k] += cd.get(k,0.0)
        # Select checkpoints only on the chronological validation interval.
        val=score_panel(model,X,y,dates,tr_end,va_end,max_lookback,device,valid_y=valid_y)
        vic=np.nanmean(val['rank_ic']) if len(val['rank_ic']) else np.nan
        rec={'epoch':ep+1,'loss':sums[0]/max(count,1),'score_loss':sums[1]/max(count,1),
             'temporal_loss':sums[2]/max(count,1),'group_loss':sums[3]/max(count,1),
             'rankic_loss':sums[4]/max(count,1),'train_rank_ic':sums[5]/max(count,1),
             'block_rank_loss':sums[6]/max(count,1),
             'block_rank_pair_count':block_pair_count/max(count,1),
             'block_rank_pair_accuracy':block_pair_accuracy/max(count,1),
             'val_rank_ic':float(vic)}
        # Epoch-level checks make class imbalance and excessive contrastive
        # negatives explicit in the CSV consumed by the dashboard.
        for k,v in tail_stats.items(): rec[k]=v/max(count,1)
        for k,v in cstats.items(): rec['contrastive_'+k]=v/max(count,1)
        rec['contrastive_negative_to_positive'] = rec['contrastive_negative_pairs']/max(rec['contrastive_positive_pairs'],1e-9)
        rec['negative_to_positive'] = rec['negative_count']/max(rec['positive_count'],1e-9)
        hist.append(rec)
        if np.isfinite(vic) and vic>best: best=vic; torch.save(model.state_dict(),out/'best.pt')
    torch.save(model.state_dict(),out/'last.pt')
    pd.DataFrame(hist).to_csv(out/'training_metrics.csv',index=False)
    if (out/'best.pt').exists(): model.load_state_dict(torch.load(out/'best.pt',map_location=device))
    pred=score_panel(model,X,y,dates,va_end,len(dates)-1,max_lookback,device,syms,valid_y=valid_y)
    pred.to_parquet(out/'predictions.parquet',index=False)
    metrics=backtest(pred,cfg,out)
    metrics.update({'backend':getattr(model.encoder, 'backend', 'mamba_ssm' if model.encoder.using_mamba else 'torch_fallback'),'device':str(device),
                   'n_stocks':len(syms),'n_days':len(dates),'data_source':cfg.get('provider','synthetic'),
                   'objective':('temporal_infonce_plus_extreme_score_plus_block_rank'
                                if float(cfg.get('block_rank_weight', 0.0)) else
                                'temporal_infonce_plus_extreme_score'),
                   'temporal_horizon':int(cfg.get('temporal_horizon',5)),
                   'temporal_horizons':horizons,
                   'context_lengths':context_lengths,
                   'group_weight':float(cfg.get('group_weight',0.0)),
                   'extreme_frac':float(cfg.get('extreme_frac',0.2)),
                   'temporal_weight':float(cfg.get('temporal_weight',1.0)),
                   'score_weight':float(cfg.get('score_weight',1.0)),
                   'rankic_weight':float(cfg.get('rankic_weight',0.0)),
                   'block_rank_weight':float(cfg.get('block_rank_weight',0.0)),
                   'transaction_cost_bps':float(cfg.get('transaction_cost_bps',10.0)),
                   'block_rank_blocks':int(cfg.get('block_rank_blocks',4)),
                   'frequency':cfg.get('frequency','daily'),
                   'periods_per_year':52 if cfg.get('frequency')=='weekly' else 252})
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    plot_all(out, pd.DataFrame(hist), pred)
    return metrics

@torch.no_grad()
def score_panel(model,X,y,dates,start,end,lookback,device,syms=None,valid_y=None):
    model.eval(); rows=[]
    ts=list(range(max(start,lookback),end)); nstocks=X.shape[1]
    # Batch dates during evaluation.  The old one-date-at-a-time loop made a
    # 1000-session run need hundreds of small CUDA launches, obscuring the
    # actual model cost on Windows GPUs.
    for off in range(0,len(ts),256):
        chunk=ts[off:off+256]
        windows=np.stack([X[t-lookback+1:t+1].transpose(1,0,2) for t in chunk],axis=0)
        flat=torch.from_numpy(windows.reshape(-1,lookback,X.shape[-1])).to(device)
        _,scores=model(flat); scores=scores.detach().cpu().numpy().reshape(len(chunk),nstocks)
        for row,t in enumerate(chunk):
            s=scores[row]; ret=y[t]
            valid=np.ones(nstocks,dtype=bool) if valid_y is None else np.asarray(valid_y[t],dtype=bool)
            valid &= np.isfinite(ret); ic=rank_ic(s[valid],ret[valid])
            for i,v in enumerate(s):
                rows.append({'date':str(pd.Timestamp(dates[t]).date()),'symbol':syms[i] if syms else str(i),
                             'score':float(v),'realized_return':float(ret[i]) if valid[i] else np.nan,'rank_ic':ic})
    return pd.DataFrame(rows)

def backtest(pred,cfg,out):
    daily=[]
    for d,g in pred.groupby('date'):
        g=g.dropna(subset=['score','realized_return']).sort_values('score'); n=max(1,int(len(g)*.1)); long=g.tail(n); short=g.head(n)
        gross=float(long.realized_return.mean()-short.realized_return.mean())
        top5=g.tail(min(5,len(g)))
        long_only=float(top5.realized_return.mean()) if len(top5) else np.nan
        daily.append((d,gross,long_only))
    r=pd.DataFrame(daily,columns=['date','gross_return','top5_gross_return'])
    # The long-short portfolio pays one cost on each leg.  The top-5
    # long-only portfolio pays one side only; both are intentionally explicit
    # so the dashboard does not imply identical turnover.
    bps=float(cfg.get('transaction_cost_bps',10))/10000
    r['net_return']=r.gross_return-2*bps
    r['top5_net_return']=r.top5_gross_return-bps
    r['gross_equity']=(1+r.gross_return).cumprod(); r['net_equity']=(1+r.net_return).cumprod()
    r['top5_gross_equity']=(1+r.top5_gross_return).cumprod(); r['top5_net_equity']=(1+r.top5_net_return).cumprod()
    r.to_csv(out/'equity_curve.csv',index=False)
    def stats(x):
        eq=(1+x).cumprod(); dd=eq/eq.cummax()-1
        ppy=52 if cfg.get('frequency')=='weekly' else 252
        return {'mean_period':float(x.mean()),'sharpe':float(x.mean()/(x.std()+1e-9)*np.sqrt(ppy)),'max_drawdown':float(dd.min()),'final_equity':float(eq.iloc[-1])}
    return {'periods':len(r),'rank_ic_mean':float(pred.groupby('date').rank_ic.first().mean()),
            'gross':stats(r.gross_return),'net':stats(r.net_return),
            'top5_long_only':{'gross':stats(r.top5_gross_return),'net':stats(r.top5_net_return)}}

def plot_all(out,hist,pred):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); ax.plot(hist.epoch,hist.loss,label='total');
    for c in ('score_loss','temporal_loss','rankic_loss','block_rank_loss','group_loss'):
        if c in hist: ax.plot(hist.epoch,hist[c],label=c)
    ax.set(xlabel='epoch',ylabel='loss',title='Training losses'); ax.legend(); fig.tight_layout(); fig.savefig(out/'training_loss.png',dpi=140); plt.close(fig)
    eq=pd.read_csv(out/'equity_curve.csv'); fig,ax=plt.subplots(); ax.plot(eq.date,eq.gross_equity,label='long-short gross'); ax.plot(eq.date,eq.net_equity,label='long-short net');
    if 'top5_net_equity' in eq: ax.plot(eq.date,eq.top5_net_equity,label='top-5 long-only net',lw=1.8,ls='--')
    ax.set(title='Out-of-sample equity curves',ylabel='equity'); ax.tick_params(axis='x',rotation=45); ax.legend(); fig.tight_layout(); fig.savefig(out/'equity_curve.png',dpi=140); plt.close(fig)
    daily=pred.groupby('date').rank_ic.first(); fig,ax=plt.subplots(); ax.plot(daily.to_numpy()); ax.axhline(0,color='k',lw=.7); ax.set(title='Daily cross-sectional RankIC',xlabel='test day',ylabel='RankIC'); fig.tight_layout(); fig.savefig(out/'rankic.png',dpi=140); plt.close(fig)
