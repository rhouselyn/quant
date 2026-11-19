"""Fit one allocation head per real encoder seed and average the three logits."""
import copy
import hashlib
import json
from pathlib import Path
import runpy
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from attention_followup import LegPool
from replay_equal_weight_model_books import legacy_inputs,locked_history
from replay_longshort_prediction import CATALOG,TAPE

OUT=Path(__file__).resolve().parent
PARENT=Path('outputs/bce_native40_transfer_v1')
engine=runpy.run_path('outputs/bce_equiv_init_v1/run.py')['replay']
HASHES={}

def save(name,value):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2),encoding='utf8')
    for attempt in range(20):
        try:t.replace(p);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def record(path):
    p=Path(path).resolve()
    with p.open('rb') as f:HASHES[str(p)]=hashlib.file_digest(f,'sha256').hexdigest()

def fit(x,y,w,vx,vy,mu,sd,seed,arm,encoder_sha,trace):
    xx=torch.from_numpy((x-mu)/sd);vv=torch.from_numpy((vx-mu)/sd)
    yy=torch.tensor(y>0,dtype=torch.float32);vt=torch.tensor(vy>0,dtype=torch.float32);wt=torch.tensor(w,dtype=torch.float32)
    assert len(vy)==123 and np.isclose(w.sum(),473)
    torch.manual_seed(seed);head=LegPool(False)
    opt=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=.01);best=np.inf
    for step in range(1,101):
        if step==11:head.proj.requires_grad_(False);frozen=[p.detach().clone() for p in head.proj.parameters()]
        opt.zero_grad();loss=(F.binary_cross_entropy_with_logits(head(xx).flatten(),yy,reduction='none')*wt).sum()/473
        loss.backward();opt.step()
        with torch.no_grad():value=F.binary_cross_entropy_with_logits(head(vv).flatten(),vt).item()
        trace.append(dict(arm=arm,head_seed=seed,encoder_seed=seed,step=step,train_bce=float(loss),val_bce=value))
        if value<best:best=value;selected=step;state=copy.deepcopy(head.state_dict())
    assert all(torch.equal(a,b) for a,b in zip(frozen,head.proj.parameters()))
    path=OUT/f'{arm}_s{seed}.pt'
    torch.save(dict(state=state,mu=torch.from_numpy(mu),sd=torch.from_numpy(sd),head_seed=seed,encoder_seed=seed,encoder_sha256=encoder_sha),path)
    record(path)
    print(arm,'real encoder/head seed',seed,'selected',selected,'val',best,flush=True)
    return dict(arm=arm,head_seed=seed,encoder_seed=seed,encoder_sha256=encoder_sha,step=selected,val_bce=best,path=str(path))

def main():
    torch.set_num_threads(4)
    assert json.loads((OUT/'state.json').read_text(encoding='utf8'))['status']=='prepared'
    HASHES.update(json.loads((OUT/'inputs.json').read_text(encoding='utf8')))
    for path in (__file__,'outputs/bce_equiv_init_v1/run.py','outputs/bce_native40_transfer_v1/train_replay.py','scripts/attention_followup.py',
                 'scripts/replay_equal_weight_model_books.py','scripts/eval_weekly_architecture.py','outputs/bce_final_audit_v1/audit.py',
                 PARENT/'signals.csv',PARENT/'weekly.parquet',PARENT/'validation_lock.json',CATALOG,TAPE,TAPE.with_name('eligibility.parquet')):record(path)
    inventory=json.loads((OUT/'model_inventory.json').read_text(encoding='utf8'))
    save('state.json',dict(status='training',heads=12))
    trace=[];selections=[];reproduction=[]
    for family in ('B0','P6'):
        for seed in (42,43,44):
            path=OUT/f'{family}_s{seed}_data.npz';record(path)
            with np.load(path) as a:
                flat=a['base'].reshape(-1,129);mu,sd=flat.mean(0),flat.std(0)+1e-6
                model=next(m for m in inventory['models'] if m['family']==family and m['encoder_seed']==seed)
                for suffix in ('plain','fixed5'):
                    x,y,w=(a['base'],a['y'],np.ones(473)) if suffix=='plain' else (a['aug_x'],a['aug_y'],a['aug_w'])
                    arm=family+'_three_'+suffix
                    row=fit(x,y,w,a['vx'],a['vy'],mu,sd,seed,arm,model['encoder_sha256'],trace);selections.append(row)
                    if seed==42:
                        refpath=PARENT/f'{family}_{suffix}_42.pt';record(refpath)
                        ref=torch.load(refpath,weights_only=True);now=torch.load(row['path'],weights_only=True)
                        error=max(float((v-ref['state'][k]).abs().max()) for k,v in now['state'].items())
                        np.testing.assert_array_equal(now['mu'],ref['mu']);np.testing.assert_array_equal(now['sd'],ref['sd'])
                        assert error<2e-6
                        reproduction.append(dict(family=family,suffix=suffix,max_parameter_error=error))
    pd.DataFrame(trace).to_csv(OUT/'training_loss.csv',index=False)
    pd.DataFrame(reproduction).to_csv(OUT/'seed42_reproduction.csv',index=False)
    parent_lock=json.loads((PARENT/'validation_lock.json').read_text(encoding='utf8'))
    save('validation_lock.json',dict(selections=selections,train_dates=parent_lock['train_dates'],validation_dates=parent_lock['validation_dates'],hashes=HASHES.copy()))
    save('state.json',dict(status='locked_scoring',heads=12))
    print('All 12 real-encoder heads locked before test inference',flush=True)
    parent=pd.read_csv(PARENT/'signals.csv',parse_dates=['date']).set_index('date')
    parts={};member={}
    for row in selections:
        family=row['arm'].split('_')[0];seed=row['encoder_seed']
        with np.load(OUT/f'{family}_s{seed}_data.npz') as a:hx=torch.from_numpy(a['hx'])
        cp=torch.load(row['path'],weights_only=True);assert cp['encoder_seed']==seed and cp['encoder_sha256']==row['encoder_sha256']
        head=LegPool(False).eval();head.load_state_dict(cp['state'])
        with torch.no_grad():p=head((hx-cp['mu'])/cp['sd']).flatten().numpy()
        member[row['arm']+'_s'+str(seed)]=p;parts.setdefault(row['arm'],[]).append(p)
    signals=pd.DataFrame({k:np.mean(v,axis=0) for k,v in parts.items()},index=parent.index)
    for suffix in ('plain','fixed5'):
        signals['old_'+suffix]=parent['old_'+suffix]
        for family in ('B0','P6'):signals[family+'_one_'+suffix]=parent[family+'_'+suffix]
    signals.to_csv(OUT/'signals.csv');pd.DataFrame(member,index=parent.index).to_csv(OUT/'member_logits.csv')
    frame,schedule,members=locked_history();schedule=schedule.loc[schedule.date.isin(signals.index)]
    assert len(schedule)==295
    d,dates,*_=legacy_inputs();f=frame.loc[frame.date.isin(signals.index)];key=['date','symbol']
    sc=pd.concat([pd.DataFrame(dict(date=day,symbol=d['symbols'],score=d['score'][dates.get_loc(day)])) for day in schedule.date],ignore_index=True)
    sources={'old':f.merge(sc.rename(columns={'score':'score_long'}),on=key,validate='one_to_one')}
    sources['old']['score_short']=sources['old'].score_long
    for book in ('B0','P6'):
        book_scores=next(s for lock,s in members.values() if lock['arm']==book)
        z=f.merge(book_scores['test']['long'][key+['score']].rename(columns={'score':'score_long'}),on=key,how='left',validate='one_to_one')
        sources[book]=z.merge(book_scores['test']['short'][key+['score']].rename(columns={'score':'score_short'}),on=key,how='left',validate='one_to_one')
    for member_info in json.loads(Path(CATALOG).read_text(encoding='utf8'))['members']:
        if member_info['arm'] in ('B0','P6'):
            for k in ('test_scores','validation_scores','lock'):record(member_info[k])
    metrics=[];books=[]
    for book,source in sources.items():
        columns=[k for k in signals if book=='old' or k.startswith(('old_',book+'_'))]
        m,w=engine(source,schedule,signals[columns]);metrics.append(m.assign(book=book));books.append(w.assign(book=book))
    metrics=pd.concat(metrics,ignore_index=True);weekly=pd.concat(books,ignore_index=True)
    assert len(metrics)==88 and len(weekly)==25960
    assert not weekly.duplicated(['book','arm','rule','fee','date']).any()
    np.testing.assert_allclose(weekly.net,weekly.long_net+weekly.short_net,atol=1e-12,rtol=0)
    reference=pd.read_parquet(PARENT/'weekly.parquet');checked=0
    for arm in signals:
        if '_three_' in arm:continue
        control=arm.replace('_one_','_')
        a=weekly.loc[weekly.arm.eq(arm)];b=reference.loc[reference.arm.eq(control)]
        merged=a.merge(b,on=['book','date','rule','fee'],validate='one_to_one')
        assert len(merged)==len(a)
        np.testing.assert_allclose(merged.net_x,merged.net_y,atol=1e-12,rtol=0);checked+=len(merged)
    metrics.to_csv(OUT/'metrics.csv',index=False);weekly.to_parquet(OUT/'weekly.parquet',index=False)
    save('inputs.json',HASHES)
    verification=dict(status='complete',actual_distinct_encoders=6,heads_trained=12,train_weeks=473,validation_loss_weeks=123,test_weeks=295,
                      main_metric_rows=56,single_control_metric_rows=32,total_metric_rows=88,weekly_rows=25960,reference_weekly_rows_reproduced=checked,
                      fixed_baskets_labels_weights=True,seed42_heads_reproduced=True)
    save('verification.json',verification);save('state.json',verification)
    print(metrics.query("rule=='full'")[['book','arm','fee','annual_calendar','max_drawdown','sharpe']].to_string(index=False),flush=True)

if __name__=='__main__':main()
