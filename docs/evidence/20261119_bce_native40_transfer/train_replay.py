"""Train all three signal families on every locked week, then apply each signal twice."""
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
from directional_three_seed import cache_path,stock_inputs
from directional_2018_experiment import groups_for
from score_mlp_experiment import load_eligibility
from replay_equal_weight_model_books import legacy_inputs,locked_history

OUT=Path(__file__).resolve().parent
PARENT=Path('outputs/bce_input_label_v1')
replay=runpy.run_path('outputs/bce_equiv_init_v1/run.py')['replay']
HASHES={}

def save(name,obj):
    path=OUT/name;temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2),encoding='utf8')
    for attempt in range(20):
        try:temp.replace(path);break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def record(path):
    path=Path(path).resolve()
    with path.open('rb') as f:HASHES[str(path)]=hashlib.file_digest(f,'sha256').hexdigest()

def digest(*arr):
    h=hashlib.sha256()
    for a in arr:h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def native_groups(long,short,available,dates,symbols):
    eligibility=load_eligibility();out=[]
    for sl,ss,known,day in zip(long,short,available,dates):
        known=known&np.isfinite(sl)&np.isfinite(ss)
        order=np.argsort(-sl,kind='stable');l=order[known[order]][:50]
        order=np.argsort(-ss,kind='stable')[::-1]
        quarter=str((day-pd.offsets.QuarterEnd(1)).to_period('Q'))
        allowed=eligibility.get(quarter,set())
        sh=np.array([i for i in order if known[i] and i not in l and (day.year<2018 or symbols[i].split('.')[0] in allowed)],dtype=int)[:50]
        assert len(l)>=40 and len(sh)>=10,(day,len(l),len(sh))
        out.append((l,sh))
    return out

def fit(x,y,w,vx,vy,mu,sd,seed,arm,trace):
    tx=torch.from_numpy((x-mu)/sd);vv=torch.from_numpy((vx-mu)/sd)
    target=torch.tensor(y>0,dtype=torch.float32);vt=torch.tensor(vy>0,dtype=torch.float32)
    weight=torch.tensor(w,dtype=torch.float32)
    assert len(vy)==123 and np.isclose(w.sum(),473)
    torch.manual_seed(seed);head=LegPool(False)
    opt=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=.01);best=np.inf
    for step in range(1,101):
        if step==11:
            head.proj.requires_grad_(False);frozen=[p.detach().clone() for p in head.proj.parameters()]
        opt.zero_grad()
        loss=(F.binary_cross_entropy_with_logits(head(tx)[:,0],target,reduction='none')*weight).sum()/473
        loss.backward();opt.step()
        with torch.no_grad():value=F.binary_cross_entropy_with_logits(head(vv)[:,0],vt).item()
        trace.append(dict(arm=arm,seed=seed,step=step,train_bce=float(loss),val_bce=value))
        if value<best:best,chosen,state=value,step,copy.deepcopy(head.state_dict())
    assert all(torch.equal(a,b) for a,b in zip(frozen,head.proj.parameters()))
    path=OUT/f'{arm}_{seed}.pt'
    torch.save(dict(state=state,mu=torch.from_numpy(mu),sd=torch.from_numpy(sd)),path);record(path)
    print(arm,seed,'selected',chosen,'val_bce',best,flush=True)
    return dict(arm=arm,seed=seed,step=chosen,val_bce=best,path=str(path))

def main():
    torch.set_num_threads(4)
    prep=json.loads((OUT/'state.json').read_text(encoding='utf8'));assert prep['status']=='prepared'
    HASHES.update(json.loads((OUT/'inputs.json').read_text(encoding='utf8')))
    for path in (__file__,'outputs/bce_equiv_init_v1/run.py','scripts/attention_followup.py','scripts/eval_weekly_architecture.py',
                 'scripts/replay_equal_weight_model_books.py','outputs/bce_final_audit_v1/audit.py',PARENT/'validation_lock.json',PARENT/'signals.csv'):
        record(path)
    from skipfill_short import ELIG_DIR
    for path in Path(ELIG_DIR).glob('*Q*.json'):record(path)
    save('state.json',dict(status='training',heads=18))
    d,dates,oldb,oldg,complete,tr=legacy_inputs();oldgroups=groups_for(d,dates)
    datelists=json.loads((OUT/'dates.json').read_text(encoding='utf8'))
    phases={k:pd.DatetimeIndex(v) for k,v in datelists.items()}
    priorlock=json.loads((PARENT/'validation_lock.json').read_text(encoding='utf8'))
    assert phases['train'].astype(str).tolist()==priorlock['train_dates'] and len(tr)==473
    vi=dates.get_indexer(phases['validation']);va=dates.get_indexer(priorlock['validation_dates'])
    vpick=phases['validation'].get_indexer(dates[va]);assert (vpick>=0).all() and len(vpick)==123
    active=set(np.random.default_rng(61118).choice(473,236,replace=False))
    groups={};draws={};labels={};missing=[];basket_rows=[];counts=[]
    for family in ('old','B0','P6'):
        groups[family]={}
        for phase,days in phases.items():
            oi=dates.get_indexer(days)
            if family=='old':gg=[oldgroups[t] for t in oi]
            else:
                p=OUT/f'{family}_{phase}_scores.npz';record(p)
                record(OUT/f'{family}_{phase}_vectors.npy')
                with np.load(p) as a:gg=native_groups(a['long'],a['short'],d['available'][oi],days,d['symbols'])
            groups[family][phase]=gg
            for day,b in zip(days,gg):
                basket_rows.append(dict(family=family,phase=phase,date=day,longs='|'.join(d['symbols'][b[0][:40]]),shorts='|'.join(d['symbols'][b[1][:10]])))
        def label(t,b,phase,kind,variant=-1):
            ids=np.r_[*b];valid=d['valid'][t,ids]&np.isfinite(d['y'][t,ids])
            r=np.expm1(np.where(valid,d['y'][t,ids],0.))
            y=np.float32(r[:40].mean()+r[40:].mean()+.08/52)
            if not valid.all():
                missing.append(dict(family=family,phase=phase,date=dates[t],kind=kind,variant=variant,
                                    missing_long=int((~valid[:40]).sum()),missing_short=int((~valid[40:]).sum()),symbols='|'.join(d['symbols'][ids[~valid]])))
            return y
        y=np.array([label(t,(b[0][:40],b[1][:10]),'train','base') for t,b in zip(tr,groups[family]['train'])])
        vy=np.array([label(t,(b[0][:40],b[1][:10]),'validation','base') for t,b in zip(vi,groups[family]['validation'])])
        labels[family]=(y,vy)
        if family=='old':
            np.testing.assert_array_equal(y,oldg[tr]);np.testing.assert_array_equal(vy[vpick],oldg[va])
        dd=[]
        for row,(t,b) in enumerate(zip(tr,groups[family]['train'])):
            variants=[]
            if row in active:
                seen={tuple(map(tuple,(b[0][:40],b[1][:10])))}
                rng=np.random.default_rng(np.random.SeedSequence([61118,int(t),1]))
                for attempt in range(3):
                    pair=tuple(pool[np.sort(rng.choice(len(pool),n,replace=False))] for pool,n in ((b[0][:50],40),(b[1][:20],10)))
                    key=tuple(map(tuple,pair))
                    assert len(np.unique(np.r_[*pair]))==50
                    if key not in seen:
                        seen.add(key);variants.append((pair,label(t,pair,'train','augmentation',attempt)))
            dd.append(variants)
        draws[family]=dd
        counts.append(dict(family=family,train_weeks=473,val_loss_weeks=123,active_weeks=len(active),effective_aug_weeks=sum(bool(v) for v in dd),variants=sum(map(len,dd))))
    pd.DataFrame(missing).to_csv(OUT/'missing_returns.csv',index=False)
    pd.DataFrame(basket_rows).to_csv(OUT/'signal_baskets.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'augmentation.csv',index=False)
    summaries=[]
    for family,(y,vy) in labels.items():
        for split,values,reference in (('train',y,labels['old'][0]),('validation',vy[vpick],labels['old'][1][vpick])):
            summaries.append(dict(family=family,split=split,weeks=len(values),positive=int((values>0).sum()),
                                  changed_vs_old=int(((values>0)!=(reference>0)).sum())))
    pd.DataFrame(summaries).to_csv(OUT/'label_summary.csv',index=False)
    trace,selections,identity=[],[],[]
    for family in ('old','B0','P6'):
        for seed in (42,43,44):
            if family=='old':
                record(cache_path(seed))
                with np.load(cache_path(seed)) as a:z=a['z']
                trainz=z[tr];valz=z[vi]
            else:
                for phase in ('train','validation'):record(OUT/f'{family}_{phase}_vectors.npy')
                trainz=np.load(OUT/f'{family}_train_vectors.npy',mmap_mode='r')
                valz=np.load(OUT/f'{family}_validation_vectors.npy',mmap_mode='r')
            def xrow(z,b):
                ids=np.r_[*b];assert len(ids)==50 and len(np.unique(ids))==50
                x=np.column_stack((z[ids],np.repeat([1.,-1.],[40,10]))).astype(np.float32)
                assert np.isfinite(x).all() and (np.linalg.norm(x[:,:128],axis=1)>.9).all()
                return x
            base=np.stack([xrow(zz,(b[0][:40],b[1][:10])) for zz,b in zip(trainz,groups[family]['train'])])
            vx=np.stack([xrow(zz,(b[0][:40],b[1][:10])) for zz,b in zip(valz,groups[family]['validation'])])[vpick]
            y,vy=labels[family];vy=vy[vpick]
            flat=base.reshape(-1,129);mu,sd=flat.mean(0),flat.std(0)+1e-6
            ax,ay,aw=[],[],[];bw=np.ones(473)
            for row,variants in enumerate(draws[family]):
                if variants:bw[row]=.95
                for b,gap in variants:ax.append(xrow(trainz[row],b));ay.append(gap);aw.append(.05/len(variants))
            for suffix,x,yy,w in (('plain',base,y,np.ones(473)),('fixed5',np.concatenate([base,np.stack(ax)]),np.r_[y,ay],np.r_[bw,aw])):
                arm=family+'_'+suffix
                identity.append(dict(arm=arm,seed=seed,inputs=digest(x,vx),labels=digest(yy,vy),weights=digest(w),normalization=digest(mu,sd),train_weeks=473))
                selections.append(fit(x,yy,w,vx,vy,mu,sd,seed,arm,trace))
            del trainz,valz
            if family=='old':del z
    pd.DataFrame(trace).to_csv(OUT/'training_loss.csv',index=False)
    pd.DataFrame(identity).to_csv(OUT/'identity.csv',index=False)
    save('validation_lock.json',dict(selections=selections,train_dates=priorlock['train_dates'],validation_dates=priorlock['validation_dates'],hashes=HASHES.copy()))
    print('Locked all 18 heads on original validation dates',flush=True)
    save('state.json',dict(status='locked_scoring',heads=18))
    predictions={};ti=dates.get_indexer(phases['test'])
    for family in ('old','B0','P6'):
        for seed in (42,43,44):
            if family=='old':
                with np.load(cache_path(seed)) as a:z=a['z'][ti]
            else:z=np.load(OUT/f'{family}_test_vectors.npy',mmap_mode='r')
            xx=torch.from_numpy(np.stack([np.column_stack((zz[np.r_[b[0][:40],b[1][:10]]],np.repeat([1.,-1.],[40,10]))).astype(np.float32) for zz,b in zip(z,groups[family]['test'])]))
            for suffix in ('plain','fixed5'):
                arm=family+'_'+suffix;cp=torch.load(OUT/f'{arm}_{seed}.pt',weights_only=True)
                head=LegPool(False).eval();head.load_state_dict(cp['state'])
                with torch.no_grad():p=head((xx-cp['mu'])/cp['sd'])[:,0].numpy()
                predictions.setdefault(arm,[]).append(p)
            del z
    signals=pd.DataFrame({k:np.mean(v,axis=0) for k,v in predictions.items()},index=phases['test']).rename_axis('date')
    parent=pd.read_csv(PARENT/'signals.csv',parse_dates=['date']).set_index('date')
    np.testing.assert_allclose(signals.old_plain,parent.A_plain,atol=1e-6,rtol=1e-6)
    signals.to_csv(OUT/'signals.csv')
    frame,schedule,members=locked_history();schedule=schedule.loc[schedule.date.isin(signals.index)]
    from replay_longshort_prediction import TAPE,CATALOG
    record(TAPE);record(TAPE.with_name('eligibility.parquet'));record(CATALOG)
    for member in json.loads(Path(CATALOG).read_text(encoding='utf8'))['members']:
        if member['arm'] in ('B0','P6'):
            for key in ('test_scores','validation_scores','lock'):record(member[key])
    assert len(schedule)==295
    key=['date','symbol'];f=frame.loc[frame.date.isin(signals.index)]
    score=pd.concat([pd.DataFrame(dict(date=day,symbol=d['symbols'],score=d['score'][dates.get_loc(day)])) for day in schedule.date],ignore_index=True)
    sources={'old':f.merge(score.rename(columns={'score':'score_long'}),on=key,validate='one_to_one')}
    sources['old']['score_short']=sources['old'].score_long
    for book in ('B0','P6'):
        scores=next(s for lock,s in members.values() if lock['arm']==book)
        src=f.merge(scores['test']['long'][key+['score']].rename(columns={'score':'score_long'}),on=key,how='left',validate='one_to_one')
        sources[book]=src.merge(scores['test']['short'][key+['score']].rename(columns={'score':'score_short'}),on=key,how='left',validate='one_to_one')
    metrics,weekly,refs,refweekly=[],[],[],[]
    for book,source in sources.items():
        names=[k for k in signals if book=='old' or k.startswith(('old_',book+'_'))]
        m,w=replay(source,schedule,signals[names]);metrics.append(m.assign(book=book));weekly.append(w.assign(book=book))
        r=parent[['A_fixed5']].rename(columns={'A_fixed5':'original_fixed5'})
        m,w=replay(source,schedule,r);refs.append(m.assign(book=book));refweekly.append(w.assign(book=book))
        print('Replayed native40 signals on',book,'20/10',flush=True)
    metrics=pd.concat(metrics,ignore_index=True);weekly=pd.concat(weekly,ignore_index=True)
    refs=pd.concat(refs,ignore_index=True);refweekly=pd.concat(refweekly,ignore_index=True)
    assert len(metrics)==56 and len(weekly)==16520 and len(refs)==12
    archived=pd.read_parquet('outputs/bce_final_audit_v1/calendar_weekly.parquet')
    checked=0
    for book,model in (('old','old_scores'),('P6','P6')):
        x=weekly.loc[weekly.book.eq(book)&weekly.arm.eq('old_plain')]
        y=archived.loc[archived.model.eq(model)&archived.long_slots.eq(20)&archived.arm.eq('unaugmented')]
        merged=x.merge(y,on=['date','rule','fee'],validate='one_to_one');assert len(merged)==1180
        np.testing.assert_allclose(merged.net_x,merged.net_y,atol=1e-12,rtol=0);checked+=len(merged)
    np.testing.assert_allclose(weekly.net,weekly.long_net+weekly.short_net,atol=1e-12,rtol=0)
    assert not weekly.duplicated(['book','arm','rule','fee','date']).any()
    metrics.to_csv(OUT/'metrics.csv',index=False);weekly.to_parquet(OUT/'weekly.parquet',index=False)
    refs.to_csv(OUT/'reference_metrics.csv',index=False);refweekly.to_parquet(OUT/'reference_weekly.parquet',index=False)
    save('inputs.json',HASHES)
    verification=dict(status='complete',heads_trained=18,train_weeks=473,validation_loss_weeks=123,test_weeks=295,
                      missing_returns_marked_flat_by_explicit_user_instruction=True,no_training_weeks_dropped=True,
                      original_plain_signal_reproduced=True,archived_weekly_rows_reproduced=checked,metric_rows=len(metrics),weekly_rows=len(weekly),reference_metric_rows=len(refs))
    save('verification.json',verification);save('state.json',verification)
    print(metrics.query("rule=='full'")[['book','arm','fee','annual_calendar','max_drawdown','sharpe']].to_string(index=False),flush=True)

if __name__=='__main__':main()
