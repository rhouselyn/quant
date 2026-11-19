"""Train twelve fused-roster heads and replay fixed and fused 20/10 books."""
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import torch
from attention_followup import LegPool
from eval_weekly_architecture import pick_targets
from replay_equal_weight_model_books import legacy_inputs,locked_history
from replay_longshort_prediction import CATALOG,TAPE

OUT=Path(__file__).resolve().parent
PARENT=Path('outputs/bce_three_encoder_v1')
TRAINER=runpy.run_path(str(PARENT/'train_replay.py'))
engine=runpy.run_path('outputs/bce_equiv_init_v1/run.py')['replay']
HASHES={}

def save(name,value):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2),encoding='utf8');t.replace(p)

def record(path):
    p=Path(path).resolve()
    with p.open('rb') as f:HASHES[str(p)]=hashlib.file_digest(f,'sha256').hexdigest()

def main():
    torch.set_num_threads(4);assert json.loads((OUT/'state.json').read_text(encoding='utf8'))['status']=='prepared'
    HASHES.update(json.loads((OUT/'inputs.json').read_text(encoding='utf8')))
    for p in (__file__,PARENT/'train_replay.py',PARENT/'validation_lock.json',PARENT/'weekly.parquet',PARENT/'signals.csv',
              'outputs/bce_equiv_init_v1/run.py','outputs/bce_final_audit_v1/audit.py','scripts/attention_followup.py',
              'scripts/eval_weekly_architecture.py','scripts/replay_equal_weight_model_books.py',CATALOG,TAPE,TAPE.with_name('eligibility.parquet')):record(p)
    fit=TRAINER['fit'];fit.__globals__['OUT']=OUT;fit.__globals__['HASHES']=HASHES
    inventory=json.loads((OUT/'model_inventory.json').read_text(encoding='utf8'))
    trace=[];selections=[]
    save('state.json',dict(status='training',heads=12))
    for family in ('B0','P6'):
        for seed in (42,43,44):
            info=next(r for r in inventory['models'] if r['family']==family and r['encoder_seed']==seed)
            with np.load(OUT/f'{family}_s{seed}_data.npz') as a:
                flat=a['base'].reshape(-1,129);mu,sd=flat.mean(0),flat.std(0)+1e-6
                for suffix in ('plain','fixed5'):
                    x,y,w=(a['base'],a['y'],np.ones(473)) if suffix=='plain' else (a['aug_x'],a['aug_y'],a['aug_w'])
                    selections.append(fit(x,y,w,a['vx'],a['vy'],mu,sd,seed,family+'_fused_'+suffix,info['encoder_sha256'],trace))
    pd.DataFrame(trace).to_csv(OUT/'training_loss.csv',index=False)
    oldlock=json.loads((PARENT/'validation_lock.json').read_text(encoding='utf8'))
    save('validation_lock.json',dict(selections=selections,train_dates=oldlock['train_dates'],validation_dates=oldlock['validation_dates'],hashes=HASHES.copy()))
    save('state.json',dict(status='locked_scoring',heads=12));print('All twelve fused-roster heads locked before test inference.',flush=True)
    previous=pd.read_csv(PARENT/'signals.csv',parse_dates=['date']).set_index('date');parts={};member={}
    for row in selections:
        family=row['arm'].split('_')[0];seed=row['encoder_seed']
        with np.load(OUT/f'{family}_s{seed}_data.npz') as a:hx=torch.from_numpy(a['hx'])
        cp=torch.load(row['path'],weights_only=True);head=LegPool(False).eval();head.load_state_dict(cp['state'])
        assert cp['encoder_sha256']==row['encoder_sha256']
        with torch.no_grad():p=head((hx-cp['mu'])/cp['sd']).flatten().numpy()
        member[row['arm']+'_s'+str(seed)]=p;parts.setdefault(row['arm'],[]).append(p)
    signals=pd.DataFrame({k:np.mean(v,axis=0) for k,v in parts.items()},index=previous.index)
    for suffix in ('plain','fixed5'):
        signals['old_'+suffix]=previous['old_'+suffix]
        for family in ('B0','P6'):signals[family+'_vector_'+suffix]=previous[family+'_three_'+suffix]
    signals.to_csv(OUT/'signals.csv');pd.DataFrame(member,index=previous.index).to_csv(OUT/'member_logits.csv')
    frame,schedule,members=locked_history();schedule=schedule.loc[schedule.date.isin(signals.index)]
    assert len(schedule)==295
    d,dates,*_=legacy_inputs();f=frame.loc[frame.date.isin(signals.index)];key=['date','symbol']
    sc=pd.concat([pd.DataFrame(dict(date=day,symbol=d['symbols'],score=d['score'][dates.get_loc(day)])) for day in signals.index],ignore_index=True)
    sources={'old':f.merge(sc.rename(columns={'score':'score_long'}),on=key,validate='one_to_one')};sources['old']['score_short']=sources['old'].score_long
    for family in ('B0','P6'):
        native=next(s for lock,s in members.values() if lock['arm']==family)
        z=f.merge(native['test']['long'][key+['score']].rename(columns={'score':'score_long'}),on=key,how='left',validate='one_to_one')
        sources[family]=z.merge(native['test']['short'][key+['score']].rename(columns={'score':'score_short'}),on=key,how='left',validate='one_to_one')
        with np.load(OUT/f'{family}_actual_test_scores.npz') as a:
            scores=pd.DataFrame(dict(date=np.repeat(signals.index,len(d['symbols'])),symbol=np.tile(d['symbols'],295),score_long=a['long'].ravel(),score_short=a['short'].ravel()))
        sources[family+'_fused']=f.merge(scores,on=key,how='left',validate='one_to_one')
    for row in json.loads(Path(CATALOG).read_text(encoding='utf8'))['members']:
        if row['arm'] in ('B0','P6'):
            for k in ('test_scores','validation_scores','lock'):record(row[k])
    metrics=[];weekly=[];rosters=[]
    for book,source in sources.items():
        family=book.split('_')[0]
        cols=list(signals) if book=='old' else [c for c in signals if c.startswith(('old_',family+'_'))]
        for day,g in source.groupby('date'):
            targets=pick_targets(g,20,10,1.,1.)
            rosters.append(dict(book=book,date=day,longs='|'.join(s for side,s in targets if side==1),shorts='|'.join(s for side,s in targets if side==-1)))
        a,b=engine(source,schedule,signals[cols]);metrics.append(a.assign(book=book));weekly.append(b.assign(book=book))
    metrics=pd.concat(metrics,ignore_index=True);weekly=pd.concat(weekly,ignore_index=True)
    assert len(metrics)==136 and len(weekly)==40120
    assert not weekly.duplicated(['book','arm','rule','fee','date']).any()
    np.testing.assert_allclose(weekly.net,weekly.long_net+weekly.short_net,atol=1e-12,rtol=0)
    reference=pd.read_parquet(PARENT/'weekly.parquet');checked=0
    for arm in signals:
        if '_fused_' in arm:continue
        control=arm.replace('_vector_','_three_')
        a=weekly.loc[weekly.book.isin(['old','B0','P6'])&weekly.arm.eq(arm)]
        b=reference.loc[reference.arm.eq(control)]
        merged=a.merge(b,on=['book','date','rule','fee'],validate='one_to_one');assert len(merged)==len(a)
        np.testing.assert_allclose(merged.net_x,merged.net_y,atol=1e-12,rtol=0);checked+=len(merged)
    assert checked==16520
    metrics.to_csv(OUT/'metrics.csv',index=False);weekly.to_parquet(OUT/'weekly.parquet',index=False);pd.DataFrame(rosters).to_csv(OUT/'actual_baskets.csv',index=False)
    save('inputs.json',HASHES)
    verify=dict(status='complete',heads_trained=12,training_rows=1200,train_weeks=473,validation_weeks=123,test_weeks=295,
                metric_rows=136,weekly_rows=40120,reference_weekly_rows_reproduced=checked,actual_book_variants=5)
    save('verification.json',verify);save('state.json',verify)
    print(metrics.query("rule=='full'")[['book','arm','fee','annual_calendar','max_drawdown','sharpe']].to_string(index=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save('state.json',dict(status='failed',phase='train_replay',error=repr(exc)))
        raise
