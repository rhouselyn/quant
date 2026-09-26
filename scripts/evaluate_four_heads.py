"""Diagnostics for one four-head experiment, reusing the single-head baseline."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quant_mvp.data import dataset_arrays
from quant_mvp.model import PairRanker
from quant_mvp.engine import rank_ic


@torch.no_grad()
def predictions(model, X, y, dates, symbols, valid, start, end, lookback, device):
    rows=[]
    steps=list(range(start,end))
    for offset in range(0,len(steps),16):
        ts=steps[offset:offset+16]
        windows=np.stack([X[t-lookback+1:t+1].transpose(1,0,2) for t in ts])
        inputs=torch.from_numpy(windows.reshape(-1,lookback,X.shape[-1])).to(device)
        z,_=model(inputs)
        heads=model.score(z).cpu().numpy().reshape(len(ts),len(symbols),4)
        for i,t in enumerate(ts):
            for j,symbol in enumerate(symbols):
                rows.append(dict(date=str(pd.Timestamp(dates[t]).date()),symbol=symbol,
                    realized_return=float(y[t,j]) if valid[t,j] else np.nan,
                    **{f'head{k+1}':float(heads[i,j,k]) for k in range(4)}))
    return pd.DataFrame(rows)


def describe(pred):
    cols=[f'head{i+1}' for i in range(4)]
    correlations=[]; deviations=[]; ics=[]; mean_deviations=[]
    for _,g in pred.groupby('date'):
        g=g.dropna(subset=['realized_return']+cols)
        if len(g)<3: continue
        x=g[cols].to_numpy()
        deviations.append(x.std(axis=0,ddof=1))
        mean_deviations.append(x.mean(axis=1).std(ddof=1))
        if np.all(x.std(axis=0,ddof=1)>1e-12):
            corr=np.corrcoef(x,rowvar=False)
            if np.isfinite(corr).all(): correlations.append(corr)
        ics.append([rank_ic(x[:,k],g.realized_return.to_numpy()) for k in range(4)])
    matrices=np.asarray(correlations)
    off=~np.eye(4,dtype=bool)
    return dict(correlation_mean=matrices.mean(axis=0).tolist() if len(matrices) else [[None]*4 for _ in range(4)],
        mean_abs_offdiag=float(np.abs(matrices[:,off]).mean()) if len(matrices) else None,
        mean_squared_frobenius=float(np.square(matrices[:,off]).sum(axis=1).mean()) if len(matrices) else None,
        head_std_mean=np.mean(deviations,axis=0).tolist(),
        mean_score_std=float(np.mean(mean_deviations)),
        head_rank_ic_mean=np.nanmean(ics,axis=0).tolist(),
        correlation_weeks=len(correlations),total_weeks=len(deviations))


def main():
    out=Path('outputs/real_weekly_500_heads4_decorr_8ep')
    base=Path('outputs/real_weekly_500_full_rankic_10ep')
    cfg=yaml.safe_load((out/'config.yaml').read_text())
    assert cfg['n_score_heads']==4 and cfg['n_stocks']==500 and not cfg.get('stock_weight_path')
    X,y,dates,symbols,valid=dataset_arrays(pd.read_parquet(cfg['cache_path']),cfg['lookback'],cfg['n_features'],cfg['train_end'],return_valid=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PairRanker(cfg['n_features'],n_score_heads=4,d_model=cfg['d_model'],embedding_dim=cfg['embedding_dim'],n_layers=cfg['n_layers'],dropout=cfg.get('dropout',.1)).to(device)
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True));model.eval()
    diagnostics={}
    for name,start,end in [('validation',cfg['train_end'],cfg['valid_end']),('test',cfg['valid_end'],len(dates)-1)]:
        pred=predictions(model,X,y,dates,symbols,valid,start,end,cfg['lookback'],device)
        pred.to_parquet(out/f'{name}_head_predictions.parquet',index=False)
        diagnostics[name]=describe(pred)
        if name=='test':
            mean=pred[[f'head{i+1}' for i in range(4)]].mean(axis=1)
            saved=pd.read_parquet(out/'predictions.parquet')
            assert pred[['date','symbol']].equals(saved[['date','symbol']])
            np.testing.assert_allclose(mean,saved.score,atol=1e-6)
    rows=[]; curves=None
    for key,label,path in [('single','已有单头／500股等权',base),('four','4头均值＋输出去相关／500股等权',out)]:
        m=json.loads((path/'metrics.json').read_text(encoding='utf-8'))
        rows.append(dict(key=key,label=label,epochs=m['epochs'],best_epoch=m['best_epoch'],
            val_sharpe=m['selection_value'],rank_ic=m['rank_ic_mean'],
            normal_ic=m['signal_quality']['normal_ic_mean'],long_only=m['long_only']))
        c=pd.read_csv(path/'long_only_curve.csv')[['date','long_net_equity']].rename(columns={'long_net_equity':key})
        curves=c if curves is None else curves.merge(c,on='date',validate='one_to_one')
    curves.to_csv(out/'head_comparison_curves.csv',index=False)
    report=dict(rows=rows,diagnostics=diagnostics,output_decorr_weight=cfg['output_decorr_weight'],
        note='只训练一组4头＋去相关，单头直接复用已有模型。基线训练10轮最佳为第8轮，新实验8轮；均按500股验证Top20净Sharpe选模。四头和正则同时变化，不能将差异单独归因于去相关。此前已多次查看测试期，属于探索比较。')
    (out/'head_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(base/'universe.csv',out/'universe.csv')
    table=pd.DataFrame([dict(method=r['label'],best_epoch=r['best_epoch'],val_sharpe=r['val_sharpe'],rank_ic=r['rank_ic'],normal_ic=r['normal_ic'],**r['long_only']['net']) for r in rows])
    table.to_csv(out/'head_comparison.csv',index=False)
    print(table.to_string(index=False))
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':
    main()
