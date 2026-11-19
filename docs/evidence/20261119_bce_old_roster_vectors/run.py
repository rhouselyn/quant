"""Change only representation source on the same legacy 40/10 task."""
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
from directional_2018_experiment import groups_for
from replay_equal_weight_model_books import legacy_inputs,locked_history
from replay_longshort_prediction import CATALOG,TAPE

OUT=Path(__file__).resolve().parent
AUDIT=Path('outputs/bce_old_retrain_audit_v1')
NATIVE=Path('outputs/bce_native40_transfer_v1')
VECTORS=Path('outputs/bce_fused_three_v1')
TRAINER=runpy.run_path('outputs/bce_three_encoder_v1/train_replay.py')
engine=runpy.run_path('outputs/bce_equiv_init_v1/run.py')['replay']
HASHES={}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def record(p,expected=None):
    p=Path(p).resolve();h=sha(p)
    if expected is not None:assert h==expected,('cache/input changed',p)
    HASHES[str(p)]=h

def save(name,value):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2),encoding='utf8')
    for attempt in range(20):
        try:t.replace(p);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def checked_initial_loss(x,y,w,vx,mu,sd,seed,arm):
    torch.manual_seed(seed);head=LegPool(False)
    z=head(torch.from_numpy((x-mu)/sd)).flatten();target=torch.tensor(y>0,dtype=torch.float32);wt=torch.tensor(w,dtype=torch.float32)
    canonical=(F.binary_cross_entropy_with_logits(z,target,reduction='none')*wt).sum()/473
    closed=((F.softplus(z)-target*z)*wt).sum()/473
    grad=torch.autograd.grad(canonical,z)[0];expected=(torch.sigmoid(z)-target)*wt/473
    torch.testing.assert_close(canonical,closed,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(grad,expected,atol=1e-8,rtol=1e-6)
    return dict(arm=arm,seed=seed,initial_loss=float(canonical),closed_form_error=float((closed-canonical).abs()),gradient_error=float((grad-expected).abs().max()))

def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'validation_lock.json').exists(),'do not overwrite a locked run'
    save('state.json',dict(status='preparing',pid=__import__('os').getpid()))
    for p in (__file__,'docs/research_20261119_bce_old_roster_vectors_protocol.md','outputs/bce_three_encoder_v1/train_replay.py',
              'outputs/bce_equiv_init_v1/run.py','outputs/bce_final_audit_v1/audit.py','scripts/attention_followup.py',
              'scripts/directional_2018_experiment.py','scripts/directional_three_seed.py','scripts/replay_equal_weight_model_books.py',
              'scripts/eval_weekly_architecture.py',CATALOG,TAPE,TAPE.with_name('eligibility.parquet'),
              AUDIT/'identity.csv',AUDIT/'validation_lock.json',AUDIT/'inputs.json',AUDIT/'full_forward_checks.csv',
              VECTORS/'inputs.json',VECTORS/'model_inventory.json',VECTORS/'full_forward_checks.csv',NATIVE/'validation_lock.json',NATIVE/'identity.csv',NATIVE/'signals.csv',NATIVE/'weekly.parquet'):record(p)
    oldhash=json.loads((AUDIT/'inputs.json').read_text(encoding='utf8'))
    vectorhash=json.loads((VECTORS/'inputs.json').read_text(encoding='utf8'))
    for p,h in json.loads((VECTORS/'frozen_auxiliary.json').read_text(encoding='utf8')).items():record(p,h)
    d,dates,baskets,gaps,complete,tr=legacy_inputs();groups=groups_for(d,dates)
    lock=json.loads((NATIVE/'validation_lock.json').read_text(encoding='utf8'))
    phases=dict(train=pd.DatetimeIndex(lock['train_dates']),validation=pd.DatetimeIndex(lock['validation_dates']),
                test=pd.DatetimeIndex(pd.read_csv(NATIVE/'signals.csv',parse_dates=['date']).date))
    assert dates[tr].astype(str).tolist()==lock['train_dates']
    indices={k:dates.get_indexer(days) for k,days in phases.items()}
    assert [len(x) for x in indices.values()]==[473,123,295] and all((x>=0).all() for x in indices.values())
    save('dates.json',{k:v.astype(str).tolist() for k,v in phases.items()})
    active=set(np.random.default_rng(61118).choice(473,236,replace=False));draws=[];rosters=[];miss=[]
    def gap(t,b,phase,variant):
        ids=np.r_[*b];valid=d['valid'][t,ids]&np.isfinite(d['y'][t,ids]);r=np.expm1(np.where(valid,d['y'][t,ids],0.))
        if not valid.all():miss.append(dict(date=dates[t],phase=phase,variant=variant,missing_long=int((~valid[:40]).sum()),missing_short=int((~valid[40:]).sum()),symbols='|'.join(d['symbols'][ids[~valid]])))
        return np.float32(r[:40].mean()+r[40:].mean()+.08/52)
    roster_ref=pd.read_csv(NATIVE/'signal_baskets.csv',parse_dates=['date']).query("family=='old'").set_index(['phase','date'])
    for phase,ix in indices.items():
        for t in ix:
            b=baskets[t];ls='|'.join(d['symbols'][b[0]]);ss='|'.join(d['symbols'][b[1]])
            assert ls==roster_ref.at[(phase,dates[t]),'longs'] and ss==roster_ref.at[(phase,dates[t]),'shorts']
            rosters.append(dict(phase=phase,date=dates[t],variant=-1,longs=ls,shorts=ss))
    for row,t in enumerate(indices['train']):
        variants=[]
        if row in active:
            rng=np.random.default_rng(np.random.SeedSequence([61118,int(t),1]));seen={tuple(map(tuple,baskets[t]))}
            for _ in range(3):
                b=tuple(pool[np.sort(rng.choice(len(pool),n,replace=False))] for pool,n in ((groups[t][0][:50],40),(groups[t][1][:20],10)))
                key=tuple(map(tuple,b))
                if key not in seen:seen.add(key);variants.append(b)
        for j,b in enumerate(variants):rosters.append(dict(phase='augmentation',date=dates[t],variant=j,longs='|'.join(d['symbols'][b[0]]),shorts='|'.join(d['symbols'][b[1]])))
        draws.append(variants)
    assert len(rosters)==1599 and sum(map(len,draws))==708
    pd.DataFrame(rosters).to_csv(OUT/'common_baskets.csv',index=False)
    refpath=AUDIT/'cached_42_data.npz';record(refpath,oldhash[str(refpath.resolve())])
    with np.load(refpath) as a:reference={k:a[k].copy() for k in ('y','vy','aug_y','aug_w')}
    y=np.array([gap(t,baskets[t],'train',-1) for t in indices['train']])
    vy=np.array([gap(t,baskets[t],'validation',-1) for t in indices['validation']])
    ay=np.array([gap(t,b,'augmentation',j) for t,bs in zip(indices['train'],draws) for j,b in enumerate(bs)])
    np.testing.assert_array_equal(y>0,reference['y']>0);np.testing.assert_array_equal(vy>0,reference['vy']>0)
    np.testing.assert_array_equal(np.r_[y,ay]>0,reference['aug_y']>0)
    pd.DataFrame(miss,columns=['date','phase','variant','missing_long','missing_short','symbols']).to_csv(OUT/'missing_returns.csv',index=False)
    inventory=json.loads((VECTORS/'model_inventory.json').read_text(encoding='utf8'))['models'];models=[];identities=[]
    for family in ('old','B0','P6'):
        for seed in (42,43,44):
            if family=='old':
                folder=Path('outputs')/('abl2000_base' if seed==42 else f'abl2000_base_s{seed}')
                for name in ('best.pt','config.yaml','preprocessing.json'):record(folder/name,oldhash[str((folder/name).resolve())])
                state=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
                keys=sorted(k for k in state if k.startswith(('temporal.encoder.','final_norm.')));assert len(keys)>20
                h=hashlib.sha256()
                for k in keys:h.update(k.encode());h.update(state[k].contiguous().numpy().tobytes())
                encoder_sha=h.hexdigest();source=dict(family=family,encoder_seed=seed,run=str(folder.resolve()),vector_epoch={42:3,43:2,44:6}[seed],encoder_sha256=encoder_sha)
                p=AUDIT/f'cached_{seed}_data.npz';record(p,oldhash[str(p.resolve())])
                with np.load(p) as a:features={k:a[k].copy() for k in ('base','vx','hx','aug_x')}
            else:
                source=next(x for x in inventory if x['family']==family and x['encoder_seed']==seed)
                folder=Path(source['run']);epoch=source['vector_epoch'];encoder_sha=source['encoder_sha256']
                for name in ('config.yaml','preprocessing.json',f'epoch_{epoch}.pt'):record(folder/name,vectorhash[str((folder/name).resolve())])
                features={};extra=[]
                for phase,ix in indices.items():
                    p=VECTORS/f'{family}_s{seed}_{phase}_vectors.npy';record(p,vectorhash[str(p.resolve())]);z=np.load(p,mmap_mode='r')
                    assert len(z)==len(ix) and z.shape[1:]==(580,128)
                    def xrow(row,b):return np.column_stack((z[row,np.r_[*b]],np.repeat([1.,-1.],[40,10]))).astype(np.float32)
                    features[dict(train='base',validation='vx',test='hx')[phase]]=np.stack([xrow(row,baskets[t]) for row,t in enumerate(ix)])
                    if phase=='train':extra=[xrow(row,b) for row,bs in enumerate(draws) for b in bs]
                features['aug_x']=np.concatenate([features['base'],np.stack(extra)])
            for k,x in features.items():assert np.isfinite(x).all() and (np.linalg.norm(x[:,:,:128],axis=2)>.9).all(),(family,seed,k)
            np.testing.assert_array_equal(features['base'],features['aug_x'][:473])
            for k in ('base','vx','hx','aug_x'):np.testing.assert_array_equal(features[k][:,:,128],np.tile(np.repeat([1.,-1.],[40,10]),(len(features[k]),1)))
            p=OUT/f'{family}_s{seed}_data.npz';np.savez(p,**features,**reference);record(p);models.append(source)
            flat=features['base'].reshape(-1,129);mu,sd=flat.mean(0),flat.std(0)+1e-6
            for suffix,x,yy,ww in [('plain',features['base'],reference['y'],np.ones(473)),('fixed5',features['aug_x'],reference['aug_y'],reference['aug_w'])]:
                identities.append(dict(family=family,seed=seed,suffix=suffix,inputs=digest(x,features['vx']),labels=digest(yy,reference['vy']),binary_labels=digest(yy>0,reference['vy']>0),weights=digest(ww),normalization=digest(mu,sd),encoder_sha256=encoder_sha))
            print('Prepared fixed legacy task',family,seed,flush=True)
    assert len({x['encoder_sha256'] for x in models})==9
    identity=pd.DataFrame(identities)
    for suffix,g in identity.groupby('suffix'):assert len(g)==9 and g.labels.nunique()==g.binary_labels.nunique()==g.weights.nunique()==1 and g.inputs.nunique()==9
    oldident=pd.read_csv(AUDIT/'identity.csv').set_index(['arm','seed'])
    for row in identity.loc[identity.family.eq('old')].itertuples():
        for key in ('inputs','normalization','weights'):assert getattr(row,key)==oldident.at[('cached_'+row.suffix,row.seed),key]
    identity.to_csv(OUT/'identity.csv',index=False);save('model_inventory.json',dict(models=models))
    fit=TRAINER['fit'];fit.__globals__['OUT']=OUT;fit.__globals__['HASHES']=HASHES
    save('state.json',dict(status='training',heads=18));trace=[];selections=[];losschecks=[];reproduction=[]
    for source in models:
        family=source['family'];seed=source['encoder_seed']
        with np.load(OUT/f'{family}_s{seed}_data.npz') as a:
            flat=a['base'].reshape(-1,129);mu,sd=flat.mean(0),flat.std(0)+1e-6
            for suffix in ('plain','fixed5'):
                x,yy,ww=(a['base'],a['y'],np.ones(473)) if suffix=='plain' else (a['aug_x'],a['aug_y'],a['aug_w'])
                arm=family+'_'+suffix;losschecks.append(checked_initial_loss(x,yy,ww,a['vx'],mu,sd,seed,arm))
                row=fit(x,yy,ww,a['vx'],a['vy'],mu,sd,seed,arm,source['encoder_sha256'],trace);selections.append(row)
                if family=='old':
                    p=NATIVE/f'old_{suffix}_{seed}.pt';record(p);ref=torch.load(p,weights_only=True);now=torch.load(row['path'],weights_only=True)
                    error=max(float((v-ref['state'][k]).abs().max()) for k,v in now['state'].items())
                    np.testing.assert_array_equal(now['mu'],ref['mu']);np.testing.assert_array_equal(now['sd'],ref['sd']);assert error<2e-6
                    reproduction.append(dict(arm=arm,seed=seed,max_parameter_error=error))
    assert len(trace)==1800
    pd.DataFrame(trace).to_csv(OUT/'training_loss.csv',index=False);pd.DataFrame(losschecks).to_csv(OUT/'loss_checks.csv',index=False);pd.DataFrame(reproduction).to_csv(OUT/'old_parameter_reproduction.csv',index=False)
    save('validation_lock.json',dict(selections=selections,train_dates=lock['train_dates'],validation_dates=lock['validation_dates'],hashes=HASHES.copy()))
    save('state.json',dict(status='locked_scoring',heads=18));print('Locked all eighteen freshly trained heads before test inference.',flush=True)
    parts={};member={}
    for row in selections:
        family=row['arm'].split('_')[0];seed=row['encoder_seed']
        with np.load(OUT/f'{family}_s{seed}_data.npz') as a:hx=torch.from_numpy(a['hx'])
        cp=torch.load(row['path'],weights_only=True);head=LegPool(False).eval();head.load_state_dict(cp['state'])
        assert cp['encoder_sha256']==row['encoder_sha256']
        with torch.no_grad():v=head((hx-cp['mu'])/cp['sd']).flatten().numpy()
        parts.setdefault(row['arm'],[]).append(v);member[row['arm']+'_s'+str(seed)]=v
    signals=pd.DataFrame({k:np.mean(v,axis=0) for k,v in parts.items()},index=phases['test']).rename_axis('date')
    signals.to_csv(OUT/'signals.csv');pd.DataFrame(member,index=signals.index).to_csv(OUT/'member_logits.csv')
    parent=pd.read_csv(NATIVE/'signals.csv',parse_dates=['date']).set_index('date')
    for suffix in ('plain','fixed5'):
        np.testing.assert_allclose(signals['old_'+suffix],parent['old_'+suffix],atol=1e-6,rtol=0)
        assert signals['old_'+suffix].gt(0).eq(parent['old_'+suffix].gt(0)).all()
    save('state.json',dict(status='replaying',heads=18))
    frame,schedule,_=locked_history();schedule=schedule.loc[schedule.date.isin(signals.index)];assert len(schedule)==295
    f=frame.loc[frame.date.isin(signals.index)]
    scores=pd.concat([pd.DataFrame(dict(date=day,symbol=d['symbols'],score_long=d['score'][dates.get_loc(day)])) for day in signals.index],ignore_index=True)
    source=f.merge(scores,on=['date','symbol'],validate='one_to_one');source['score_short']=source.score_long
    metrics,weekly=engine(source,schedule,signals);assert len(metrics)==24 and len(weekly)==7080
    assert not weekly.duplicated(['arm','rule','fee','date']).any()
    np.testing.assert_allclose(weekly.net,weekly.long_net+weekly.short_net,atol=1e-12,rtol=0)
    reference_weekly=pd.read_parquet(NATIVE/'weekly.parquet');ref=reference_weekly.loc[reference_weekly.book.eq('old')&reference_weekly.arm.isin(['old_plain','old_fixed5'])]
    check=weekly.loc[weekly.arm.str.startswith('old_')].merge(ref,on=['arm','rule','fee','date'],validate='one_to_one');assert len(check)==2360
    np.testing.assert_allclose(check.net_x,check.net_y,atol=1e-12,rtol=0)
    metrics.to_csv(OUT/'metrics.csv',index=False);weekly.to_parquet(OUT/'weekly.parquet',index=False);save('inputs.json',HASHES)
    verify=dict(status='complete',new_heads_trained=18,real_encoders=9,training_rows=1800,train_weeks=473,validation_weeks=123,test_weeks=295,
        fixed_baskets_labels_weights=True,augmentation_variants=708,metric_rows=24,weekly_rows=7080,old_reference_weekly_rows_reproduced=2360,new_gpu_export=False)
    save('verification.json',verify);save('state.json',verify)
    print(metrics[['arm','rule','fee','annual_calendar','max_drawdown','sharpe']].to_string(index=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True);save('state.json',dict(status='failed',error=repr(exc)))
        raise
