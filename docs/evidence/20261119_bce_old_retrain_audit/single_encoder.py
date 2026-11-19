"""Change only encoder diversity while preserving old fused baskets and supervision."""
import json
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import torch
from attention_followup import LegPool
from replay_equal_weight_model_books import legacy_inputs,locked_history

OUT=Path(__file__).resolve().parent
ctx=runpy.run_path(str(OUT/'run.py'))

def main():
    torch.set_num_threads(4)
    lock=json.loads((OUT/'validation_lock.json').read_text(encoding='utf8'))
    trace=[];checks=[];selections=[]
    with np.load(OUT/'current_42_data.npz') as a:
        for suffix in ('plain','fixed5'):
            arm='single_'+suffix
            original=next(r for r in lock['selections'] if r['arm']=='current_'+suffix and r['seed']==42)
            selections.append(dict(original,arm=arm,reused=True))
            for seed in (43,44):
                x,y,w=(a['base'],a['y'],np.ones(473)) if suffix=='plain' else (a['aug_x'],a['aug_y'],a['aug_w'])
                entry,*_=ctx['train'](x,y,w,a['vx'],a['vy'],seed,arm,trace,checks)
                selections.append(entry)
        hx=torch.from_numpy(a['hx'])
    pd.DataFrame(trace).to_csv(OUT/'single_training_loss.csv',index=False)
    ctx['save']('single_validation_lock.json',dict(selections=selections,train_dates=lock['train_dates'],validation_dates=lock['validation_dates']))
    predictions={}
    for row in selections:
        cp=torch.load(row['path'],weights_only=True);head=LegPool(False).eval();head.load_state_dict(cp['state'])
        with torch.no_grad():p=head((hx-cp['mu'])/cp['sd']).flatten().numpy()
        predictions.setdefault(row['arm'],[]).append(p)
    days=pd.read_csv(OUT/'signals.csv',parse_dates=['date']).date
    signals=pd.DataFrame({k:np.mean(v,axis=0) for k,v in predictions.items()},index=pd.DatetimeIndex(days)).rename_axis('date')
    signals.to_csv(OUT/'single_signals.csv')
    frame,schedule,members=locked_history();schedule=schedule.loc[schedule.date.isin(signals.index)]
    d,dates,*_=legacy_inputs();key=['date','symbol'];f=frame.loc[frame.date.isin(signals.index)]
    score=pd.concat([pd.DataFrame(dict(date=day,symbol=d['symbols'],score=d['score'][dates.get_loc(day)])) for day in signals.index],ignore_index=True)
    sources={'old':f.merge(score.rename(columns={'score':'score_long'}),on=key,validate='one_to_one')}
    sources['old']['score_short']=sources['old'].score_long
    for book in ('B0','P6'):
        s=next(s for lock,s in members.values() if lock['arm']==book)
        z=f.merge(s['test']['long'][key+['score']].rename(columns={'score':'score_long'}),on=key,how='left',validate='one_to_one')
        sources[book]=z.merge(s['test']['short'][key+['score']].rename(columns={'score':'score_short'}),on=key,how='left',validate='one_to_one')
    rows,books=[],[]
    for book,source in sources.items():
        m,w=ctx['engine'](source,schedule,signals);rows.append(m.assign(book=book));books.append(w.assign(book=book))
    m=pd.concat(rows,ignore_index=True);w=pd.concat(books,ignore_index=True)
    assert len(m)==24 and len(w)==7080
    m.to_csv(OUT/'single_metrics.csv',index=False);w.to_parquet(OUT/'single_weekly.parquet',index=False)
    print(m.query("rule=='full'")[['book','arm','fee','annual_calendar','max_drawdown']].to_string(index=False),flush=True)

if __name__=='__main__':main()
