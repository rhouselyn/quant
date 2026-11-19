"""Paired projection-freeze schedules on identical legacy-roster inputs."""
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
PARENT=Path('outputs/bce_old_roster_vectors_v1')
MODES={'f10':10,'f30':30,'free':None}
engine=runpy.run_path('outputs/bce_equiv_init_v1/run.py')['replay']
HASHES={}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def tensor_sha(state):
    h=hashlib.sha256()
    for k in sorted(state):h.update(k.encode());h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def record(p,expected=None):
    p=Path(p).resolve();h=sha(p)
    if expected is not None:assert h==expected,('input changed',p)
    HASHES[str(p)]=h

def save(name,value):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2),encoding='utf8')
    for attempt in range(20):
        try:t.replace(p);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def fit(x,y,w,vx,vy,mu,sd,seed,family,suffix,mode,encoder_sha,trace,snapshots,checks):
    arm=f'{family}_{suffix}_{mode}';after=MODES[mode]
    xx=torch.from_numpy((x-mu)/sd);vv=torch.from_numpy((vx-mu)/sd)
    yy=torch.tensor(y>0,dtype=torch.float32);vt=torch.tensor(vy>0,dtype=torch.float32);wt=torch.tensor(w,dtype=torch.float32)
    assert len(vy)==123 and np.isclose(w.sum(),473)
    torch.manual_seed(seed);head=LegPool(False)
    opt=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=.01);best=np.inf
    z=head(xx).flatten();canonical=(F.binary_cross_entropy_with_logits(z,yy,reduction='none')*wt).sum()/473
    closed=((F.softplus(z)-yy*z)*wt).sum()/473;grad=torch.autograd.grad(canonical,z)[0];analytic=(torch.sigmoid(z)-yy)*wt/473
    torch.testing.assert_close(canonical,closed,atol=2e-6,rtol=2e-6);torch.testing.assert_close(grad,analytic,atol=1e-8,rtol=1e-6)
    check=dict(arm=arm,seed=seed,initial_loss=float(canonical),closed_form_error=float((canonical-closed).abs()),gradient_error=float((grad-analytic).abs().max()))
    del z,canonical,closed,grad,analytic
    def snapshot(step):snapshots.append(dict(arm=arm,family=family,suffix=suffix,mode=mode,seed=seed,step=step,model_sha256=tensor_sha(head.state_dict()),projection_sha256=tensor_sha(head.proj.state_dict())))
    snapshot(0);frozen=None
    for step in range(1,101):
        if after is not None and step==after+1:head.proj.requires_grad_(False);frozen=[p.detach().clone() for p in head.proj.parameters()]
        opt.zero_grad();loss=(F.binary_cross_entropy_with_logits(head(xx).flatten(),yy,reduction='none')*wt).sum()/473
        loss.backward();opt.step()
        with torch.no_grad():value=F.binary_cross_entropy_with_logits(head(vv).flatten(),vt).item()
        trace.append(dict(arm=arm,family=family,suffix=suffix,mode=mode,head_seed=seed,step=step,train_bce=float(loss),val_bce=value))
        if value<best:best=value;chosen=step;state=copy.deepcopy(head.state_dict())
        if step in (10,30,100):snapshot(step)
        if frozen is not None:assert all(torch.equal(a,b) for a,b in zip(frozen,head.proj.parameters()))
    assert all(p.requires_grad==(after is None) for p in head.proj.parameters())
    check['frozen_projection_max_error']=0. if frozen is None else max(float((a-b).abs().max()) for a,b in zip(frozen,head.proj.parameters()))
    checks.append(check)
    path=OUT/f'{arm}_s{seed}.pt';torch.save(dict(state=state,mu=torch.from_numpy(mu),sd=torch.from_numpy(sd),head_seed=seed,encoder_seed=seed,encoder_sha256=encoder_sha,freeze_after=after),path);record(path)
    print(arm,seed,'selected',chosen,'val',best,flush=True)
    return dict(arm=arm,family=family,suffix=suffix,mode=mode,encoder_seed=seed,encoder_sha256=encoder_sha,freeze_after=after,step=chosen,val_bce=best,path=str(path))

def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'validation_lock.json').exists(),'do not overwrite a locked batch'
    save('state.json',dict(status='checking_inputs',heads_planned=54))
    previous_hashes=json.loads((PARENT/'inputs.json').read_text(encoding='utf8'))
    for p,h in previous_hashes.items():record(p,h)
    for p in (__file__,'docs/research_20261119_bce_freeze_steps_protocol.md',PARENT/'inputs.json',PARENT/'model_inventory.json',PARENT/'identity.csv',
              PARENT/'validation_lock.json',PARENT/'signals.csv',PARENT/'weekly.parquet','outputs/bce_equiv_init_v1/run.py',
              'outputs/bce_final_audit_v1/audit.py','scripts/attention_followup.py','scripts/replay_equal_weight_model_books.py','scripts/eval_weekly_architecture.py',CATALOG,TAPE,TAPE.with_name('eligibility.parquet')):record(p)
    models=json.loads((PARENT/'model_inventory.json').read_text(encoding='utf8'))['models'];assert len(models)==9 and len({x['encoder_sha256'] for x in models})==9
    save('model_inventory.json',dict(models=models));parentlock=json.loads((PARENT/'validation_lock.json').read_text(encoding='utf8'))
    trace=[];snapshots=[];checks=[];selections=[];reproduction=[]
    for source in models:
        family=source['family'];seed=source['encoder_seed']
        with np.load(PARENT/f'{family}_s{seed}_data.npz') as a:
            cp=torch.load(PARENT/f'{family}_plain_s{seed}.pt',weights_only=True);mu,sd=cp['mu'].numpy(),cp['sd'].numpy()
            flat=a['base'].reshape(-1,129);np.testing.assert_array_equal(mu,flat.mean(0));np.testing.assert_array_equal(sd,flat.std(0)+1e-6)
            for suffix in ('plain','fixed5'):
                x,y,w=(a['base'],a['y'],np.ones(473)) if suffix=='plain' else (a['aug_x'],a['aug_y'],a['aug_w'])
                for mode in MODES:
                    save('state.json',dict(status='training',family=family,seed=seed,suffix=suffix,mode=mode,completed_heads=len(selections),heads_planned=54))
                    row=fit(x,y,w,a['vx'],a['vy'],mu,sd,seed,family,suffix,mode,source['encoder_sha256'],trace,snapshots,checks);selections.append(row)
                    if mode=='f10':
                        old=torch.load(PARENT/f'{family}_{suffix}_s{seed}.pt',weights_only=True);now=torch.load(row['path'],weights_only=True)
                        err=max(float((v-old['state'][k]).abs().max()) for k,v in now['state'].items());assert err<2e-6
                        np.testing.assert_array_equal(now['mu'],old['mu']);np.testing.assert_array_equal(now['sd'],old['sd'])
                        ref=next(r for r in parentlock['selections'] if r['arm']==f'{family}_{suffix}' and r['encoder_seed']==seed)
                        assert row['step']==ref['step'];np.testing.assert_allclose(row['val_bce'],ref['val_bce'],atol=2e-6,rtol=0)
                        reproduction.append(dict(family=family,suffix=suffix,seed=seed,max_parameter_error=err))
    trace=pd.DataFrame(trace);sn=pd.DataFrame(snapshots);assert len(trace)==5400 and len(sn)==216 and len(reproduction)==18
    prefix=[]
    for (family,suffix,seed),g in trace.groupby(['family','suffix','head_seed']):
        indexed={k:v.set_index('step') for k,v in g.groupby('mode')};ss=sn.loc[sn.family.eq(family)&sn.suffix.eq(suffix)&sn.seed.eq(seed)].set_index(['mode','step'])
        for k in ('f30','free'):np.testing.assert_array_equal(indexed['f10'].loc[:10,['train_bce','val_bce']],indexed[k].loc[:10,['train_bce','val_bce']])
        np.testing.assert_array_equal(indexed['f30'].loc[:30,['train_bce','val_bce']],indexed['free'].loc[:30,['train_bce','val_bce']])
        for step in (0,10):assert len({ss.at[(mode,step),'model_sha256'] for mode in MODES})==1
        assert ss.at[('f30',30),'model_sha256']==ss.at[('free',30),'model_sha256']
        assert ss.at[('f10',10),'projection_sha256']==ss.at[('f10',100),'projection_sha256']
        assert ss.at[('f30',30),'projection_sha256']==ss.at[('f30',100),'projection_sha256']
        prefix.append(dict(family=family,suffix=suffix,seed=seed,first10_exact=True,f30_free_first30_exact=True,
            free_projection_changed_after30=ss.at[('free',30),'projection_sha256']!=ss.at[('free',100),'projection_sha256']))
    trace.to_csv(OUT/'training_loss.csv',index=False);sn.to_csv(OUT/'parameter_snapshots.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'loss_checks.csv',index=False);pd.DataFrame(reproduction).to_csv(OUT/'f10_parameter_reproduction.csv',index=False);pd.DataFrame(prefix).to_csv(OUT/'prefix_checks.csv',index=False)
    valparts={};target=None
    for row in selections:
        with np.load(PARENT/f"{row['family']}_s{row['encoder_seed']}_data.npz") as a:vx=torch.from_numpy(a['vx']);vy=torch.tensor(a['vy']>0,dtype=torch.float32)
        if target is None:target=vy
        else:torch.testing.assert_close(target,vy,atol=0,rtol=0)
        cp=torch.load(row['path'],weights_only=True);h=LegPool(False).eval();h.load_state_dict(cp['state'])
        with torch.no_grad():p=h((vx-cp['mu'])/cp['sd']).flatten();v=float(F.binary_cross_entropy_with_logits(p,vy))
        np.testing.assert_allclose(v,row['val_bce'],atol=2e-6,rtol=0);valparts.setdefault(row['arm'],[]).append(p)
    ensemble=[]
    for arm,ps in valparts.items():
        family,suffix,mode=arm.split('_');ensemble.append(dict(arm=arm,family=family,suffix=suffix,mode=mode,ensemble_bce=float(F.binary_cross_entropy_with_logits(torch.stack(ps).mean(0),target))))
    choices=[]
    for family in ('old','B0','P6'):
        for suffix in ('plain','fixed5'):
            rows=[r for r in ensemble if r['family']==family and r['suffix']==suffix];best=min(r['ensemble_bce'] for r in rows)
            choice=next(r for mode in MODES for r in rows if r['mode']==mode and r['ensemble_bce']<=best+1e-12)
            choices.append(dict(family=family,suffix=suffix,selected_mode=choice['mode'],selected_arm=choice['arm'],validation_bce=choice['ensemble_bce']))
    pd.DataFrame(ensemble).to_csv(OUT/'validation_ensemble.csv',index=False);save('selection_lock.json',dict(rule='lowest ensemble validation BCE; ties f10,f30,free within 1e-12',choices=choices,validation=ensemble))
    save('validation_lock.json',dict(selections=selections,choices=choices,train_dates=parentlock['train_dates'],validation_dates=parentlock['validation_dates'],hashes=HASHES.copy()))
    save('state.json',dict(status='locked_scoring',heads=54));print('All 54 checkpoints and 6 freeze-schedule choices locked before test inference.',flush=True)
    testdays=pd.DatetimeIndex(pd.read_csv(PARENT/'signals.csv',parse_dates=['date']).date);parts={};members={}
    for row in selections:
        with np.load(PARENT/f"{row['family']}_s{row['encoder_seed']}_data.npz") as a:hx=torch.from_numpy(a['hx'])
        cp=torch.load(row['path'],weights_only=True);h=LegPool(False).eval();h.load_state_dict(cp['state'])
        with torch.no_grad():v=h((hx-cp['mu'])/cp['sd']).flatten().numpy()
        parts.setdefault(row['arm'],[]).append(v);members[row['arm']+'_s'+str(row['encoder_seed'])]=v
    signals=pd.DataFrame({k:np.mean(v,axis=0) for k,v in parts.items()},index=testdays).rename_axis('date');signals.to_csv(OUT/'signals.csv');pd.DataFrame(members,index=testdays).to_csv(OUT/'member_logits.csv')
    parent=pd.read_csv(PARENT/'signals.csv',parse_dates=['date']).set_index('date')
    for arm in signals:
        if arm.endswith('_f10'):
            ref=parent[arm[:-4]];np.testing.assert_allclose(signals[arm],ref,atol=1e-6,rtol=0);assert signals[arm].gt(0).eq(ref.gt(0)).all()
    save('state.json',dict(status='replaying',heads=54));frame,schedule,_=locked_history();schedule=schedule.loc[schedule.date.isin(testdays)];assert len(schedule)==295
    d,dates,*_=legacy_inputs();f=frame.loc[frame.date.isin(testdays)]
    scores=pd.concat([pd.DataFrame(dict(date=day,symbol=d['symbols'],score_long=d['score'][dates.get_loc(day)])) for day in testdays],ignore_index=True)
    source=f.merge(scores,on=['date','symbol'],validate='one_to_one');source['score_short']=source.score_long
    metrics,weekly=engine(source,schedule,signals);assert len(metrics)==72 and len(weekly)==21240
    assert not weekly.duplicated(['arm','rule','fee','date']).any();np.testing.assert_allclose(weekly.net,weekly.long_net+weekly.short_net,atol=1e-12,rtol=0)
    reference=pd.read_parquet(PARENT/'weekly.parquet');control=weekly.loc[weekly.arm.str.endswith('_f10')].copy();control['arm']=control.arm.str[:-4]
    merged=control.merge(reference,on=['arm','rule','fee','date'],validate='one_to_one');assert len(merged)==7080
    np.testing.assert_allclose(merged.net_x,merged.net_y,atol=1e-12,rtol=0)
    metrics.to_csv(OUT/'metrics.csv',index=False);weekly.to_parquet(OUT/'weekly.parquet',index=False);save('inputs.json',HASHES)
    chosen={r['selected_arm'] for r in choices};metrics.loc[metrics.arm.isin(chosen)].to_csv(OUT/'validation_selected_metrics.csv',index=False)
    verify=dict(status='complete',heads_trained=54,training_rows=5400,real_encoders=9,train_weeks=473,validation_weeks=123,test_weeks=295,
        metric_rows=72,weekly_rows=21240,validation_freeze_choices=6,prefix_groups_exact=18,f10_heads_reproduced=18,f10_weekly_rows_reproduced=7080,new_gpu_export=False)
    save('verification.json',verify);save('state.json',verify)
    print(pd.DataFrame(choices).to_string(index=False),flush=True)
    print(metrics.query("rule=='full'")[['arm','fee','annual_calendar','max_drawdown','sharpe']].to_string(index=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True);save('state.json',dict(status='failed',error=repr(exc)));raise
