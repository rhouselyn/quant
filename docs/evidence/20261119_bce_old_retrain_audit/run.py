"""Independently rebuild legacy supervision, re-encode inputs, and retrain allocation heads."""
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
import yaml
from attention_followup import LegPool
from directional_three_seed import cache_path
from directional_2018_experiment import groups_for
from replay_equal_weight_model_books import legacy_inputs,locked_history
from quant_mvp.hierarchy import HierarchicalRanker
from quant_mvp.hierarchy_data import arrays,board_limit
from quant_mvp.hierarchy_engine import make_batch

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

def digest(*values):
    h=hashlib.sha256()
    for v in values:h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()

def independent_returns(cfg,old_dates,symbols):
    frame=pd.read_parquet(cfg['cache_path'],columns=['date','symbol','open','close'])
    full_dates=pd.DatetimeIndex(sorted(frame.date.unique()))
    index=pd.MultiIndex.from_product([full_dates,symbols],names=['date','symbol'])
    frame=frame.set_index(['date','symbol']).reindex(index)
    op=frame.open.to_numpy().reshape(len(full_dates),len(symbols))
    cl=frame.close.to_numpy().reshape(op.shape)
    limit=board_limit(symbols,full_dates)
    result=np.full(op.shape,np.nan)
    # Arithmetic prices independently reconstruct the cache's log/exp label path.
    for t in range(len(full_dates)-2):
        entry=op[t+1];exit_=op[t+2].copy()
        with np.errstate(divide='ignore',invalid='ignore'):
            blocked=~np.isfinite(exit_)|(exit_/cl[t+1]-1<=-limit[t+2])
            exit_[blocked]=cl[t+2,blocked]
            result[t]=exit_/entry-1
            result[t,~(np.isfinite(entry)&(entry>0)&np.isfinite(exit_)&(exit_>0))]=np.nan
    ix=full_dates.get_indexer(old_dates);assert (ix>=0).all()
    return result[ix]

def native_data(cfg,pre,directory):
    pre=dict(pre,normalization=pre.get('normalization','legacy_z'))
    opts=dict(cfg,dataset_schema_version=2,cache_path=str(directory/'weekly.parquet'),
              market_path=str(directory/'market_features.csv'),universe_path=str(directory/'universe.csv'),
              frozen_preprocessing=pre,scores_only=True,selection_at='2021-01-08T00:00:00+08:00')
    return opts,arrays(opts)

def train(x,y,w,vx,vy,seed,arm,trace,checks):
    mu=x[:473].reshape(-1,129).mean(0);sd=x[:473].reshape(-1,129).std(0)+1e-6
    xx=torch.tensor((x-mu)/sd);v=torch.tensor((vx-mu)/sd)
    yy=torch.tensor(y>0,dtype=torch.float32);vv=torch.tensor(vy>0,dtype=torch.float32);weights=torch.tensor(w,dtype=torch.float32)
    torch.manual_seed(seed);head=LegPool(False)
    opt=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=.01)
    initial=head(xx)[:,0]
    closed=((F.softplus(initial)-yy*initial)*weights).sum()/473
    canonical=(F.binary_cross_entropy_with_logits(initial,yy,reduction='none')*weights).sum()/473
    torch.testing.assert_close(closed,canonical,atol=2e-6,rtol=2e-6)
    actual=torch.autograd.grad(canonical,initial,retain_graph=True)[0]
    expected=(torch.sigmoid(initial)-yy)*weights/473
    torch.testing.assert_close(actual,expected,atol=1e-8,rtol=1e-6)
    checks.append(dict(arm=arm,seed=seed,initial_bce=float(canonical),closed_form_error=float((closed-canonical).abs()),gradient_error=float((actual-expected).abs().max())))
    del initial,closed,canonical,actual,expected
    best=np.inf
    for iteration in range(100):
        step=iteration+1
        if iteration==10:
            head.proj.requires_grad_(False);frozen={k:v.detach().clone() for k,v in head.proj.named_parameters()}
        opt.zero_grad(set_to_none=True)
        z=head(xx).flatten()
        loss=F.binary_cross_entropy_with_logits(z,yy,weight=weights,reduction='sum')/473
        loss.backward();opt.step()
        with torch.no_grad():val=F.binary_cross_entropy_with_logits(head(v).flatten(),vv).item()
        trace.append(dict(arm=arm,seed=seed,step=step,train_bce=float(loss),val_bce=val))
        if val<best:best=val;selected=step;best_state=copy.deepcopy(head.state_dict())
    assert all(torch.equal(p,frozen[k]) for k,p in head.proj.named_parameters())
    path=OUT/f'{arm}_{seed}.pt'
    torch.save(dict(state=best_state,mu=torch.from_numpy(mu),sd=torch.from_numpy(sd)),path);record(path)
    print(arm,seed,'selected',selected,'val',best,flush=True)
    return dict(arm=arm,seed=seed,step=selected,val_bce=best,path=str(path)),digest(x,vx),digest(y,vy),digest(w),digest(mu,sd)

def main():
    torch.set_num_threads(4)
    save('state.json',dict(status='building_independent_inputs'))
    for p in (__file__,'docs/research_20261119_bce_old_retrain_audit_protocol.md','scripts/attention_followup.py',
              'src/quant_mvp/hierarchy_data.py','src/quant_mvp/hierarchy.py','outputs/bce_equiv_init_v1/run.py',PARENT/'validation_lock.json',PARENT/'identity.csv'):
        record(p)
    d,dates,baskets,gaps,complete,tr=legacy_inputs();groups=groups_for(d,dates)
    parent_lock=json.loads((PARENT/'validation_lock.json').read_text(encoding='utf8'))
    valdays=pd.DatetimeIndex(parent_lock['validation_dates']);va=dates.get_indexer(valdays)
    testdays=pd.DatetimeIndex(pd.read_csv(PARENT/'signals.csv',parse_dates=['date']).date);ti=dates.get_indexer(testdays)
    assert len(tr)==473 and len(va)==123 and len(ti)==295
    assert dates[tr].astype(str).tolist()==parent_lock['train_dates']
    allids=np.r_[tr,va,ti];assert len(set(allids))==len(allids)
    cfg=yaml.safe_load(Path('outputs/abl2000_base/config.yaml').read_text(encoding='utf8'))
    for p in (cfg['cache_path'],cfg['market_path'],cfg['universe_path']):record(p)
    arithmetic=independent_returns(cfg,dates,d['symbols'])
    cached=np.where(d['valid'],np.expm1(d['y']),0.)
    observed=np.isfinite(arithmetic)
    np.testing.assert_array_equal(observed[allids],d['valid'][allids])
    error=np.abs(np.where(observed,arithmetic,0.)[allids]-cached[allids])
    assert error.max()<2e-5,error.max()
    raw_label_checks=[]
    def label(t,b,kind):
        ids=np.r_[*b];rr=np.where(observed[t,ids],arithmetic[t,ids],0.)
        v=float(rr[:40].mean()+rr[40:].mean()+.08/52)
        cc=np.float32(cached[t,ids[:40]].mean()+cached[t,ids[40:]].mean()+.08/52)
        assert (v>0)==(cc>0),(dates[t],v,cc)
        raw_label_checks.append(dict(date=dates[t],kind=kind,arithmetic_gap=v,cache_gap=float(cc),missing=int((~observed[t,ids]).sum())))
        return np.float32(v)
    y=np.array([label(t,baskets[t],'base_train') for t in tr]);vy=np.array([label(t,baskets[t],'validation') for t in va])
    active=set(np.random.default_rng(61118).choice(473,236,replace=False));draws=[];requested={int(t):set(np.r_[*baskets[t]]) for t in allids}
    for row,t in enumerate(tr):
        variants=[]
        if row in active:
            rng=np.random.default_rng(np.random.SeedSequence([61118,int(t),1]));seen={tuple(map(tuple,baskets[t]))}
            for _ in range(3):
                b=tuple(pool[np.sort(rng.choice(len(pool),n,replace=False))] for pool,n in ((groups[t][0][:50],40),(groups[t][1][:20],10)))
                key=tuple(map(tuple,b))
                if key not in seen:
                    seen.add(key);variants.append((b,label(t,b,'augmentation')));requested[int(t)].update(np.r_[*b])
        draws.append(variants)
    assert sum(map(len,draws))==708
    pd.DataFrame(raw_label_checks).to_csv(OUT/'raw_label_checks.csv',index=False)
    original=arrays(cfg);od=pd.DatetimeIndex(original[6]);assert list(original[7])==list(d['symbols'])
    dirs=dict(train=Path('outputs/dataset_v2/snapshots/BQ_F0_r2'),validation=Path('outputs/dataset_v2/standard_v1/snapshots/validation'),
              test=Path('outputs/dataset_v2/standard_v1/snapshots/history'))
    current={}
    pre=json.loads(Path('outputs/abl2000_base/preprocessing.json').read_text(encoding='utf8'))
    for phase,directory in dirs.items():
        for name in ('weekly.parquet','market_features.csv','universe.csv','manifest.json'):record(directory/name)
        opt,data=native_data(cfg,pre,directory)
        assert list(data[7])==list(original[7])
        current[phase]=(opt,data,pd.DatetimeIndex(data[6]))
    index_phase={int(t):phase for phase,ix in (('train',tr),('validation',va),('test',ti)) for t in ix}
    encode_checks=[];full_checks=[];prepared={};start=time.perf_counter()
    for seed,stem in ((42,'abl2000_base'),(43,'abl2000_base_s43'),(44,'abl2000_base_s44')):
        run=Path('outputs')/stem
        for p in (run/'best.pt',run/'config.yaml',run/'preprocessing.json',cache_path(seed)):record(p)
        sp=json.loads((run/'preprocessing.json').read_text(encoding='utf8'))
        for key in ('feature_mean','feature_std','market_mean','market_std'):np.testing.assert_array_equal(sp[key],pre[key])
        model=HierarchicalRanker(cfg,original[8]).eval();model.load_state_dict(torch.load(run/'best.pt',weights_only=True,map_location='cpu'))
        assert model.cross_summary is None and model.market_film is None and model.cross_stock is None
        assert all(torch.count_nonzero(p)==0 for name,p in model.named_parameters() if name.endswith(('attn_gate','ffn_gate')))
        with np.load(cache_path(seed)) as cache:zz=cache['z'];rawscore=cache['score']
        oldrows={};newrows={}
        for row,t in enumerate(allids):
            names=np.array(sorted(requested[int(t)]));ot=od.get_loc(dates[t]);phase=index_phase[int(t)]
            opt,now,nd=current[phase];nt=nd.get_loc(dates[t])
            np.testing.assert_array_equal(od[ot-39:ot+1],nd[nt-39:nt+1])
            a=original[0][ot-39:ot+1,names].transpose(1,0,2)
            b=now[0][nt-39:nt+1,names].transpose(1,0,2)
            changed=np.any(a!=b,axis=(1,2))
            with torch.inference_mode():
                h=model.encode(torch.from_numpy(a[None]))[0]
                fresh=F.normalize(model.final_norm(h),dim=-1).numpy()
                sc=model.score(torch.from_numpy(fresh)).flatten().numpy()
                np.testing.assert_allclose(fresh,zz[t,names],atol=2e-6,rtol=2e-5)
                np.testing.assert_allclose(sc,rawscore[t,names],atol=2e-6,rtol=2e-5)
                new=fresh.copy()
                if changed.any():
                    nh=model.encode(torch.from_numpy(b[changed][None]))[0]
                    new[changed]=F.normalize(model.final_norm(nh),dim=-1).numpy()
            oldrows[int(t)]=(names,fresh);newrows[int(t)]=(names,new)
            encode_checks.append(dict(seed=seed,phase=phase,date=dates[t],stocks=len(names),changed_windows=int(changed.sum()),
                max_cache_vector_error=float(np.abs(fresh-zz[t,names]).max()),max_current_vector_change=float(np.abs(new-fresh).max())))
            if row in (0,len(allids)-1):
                for source,data,timeidx,expected in (('original',original,ot,fresh),('current',now,nt,new)):
                    aa=list(data);aa[4]=data[4].copy();aa[4][timeidx]=d['available'][t]
                    with torch.inference_mode():full=model(*make_batch(tuple(aa),[int(timeidx)],cfg,'cpu'))[1][0,names].numpy()
                    np.testing.assert_allclose(full,expected,atol=2e-6,rtol=2e-5)
                    full_checks.append(dict(seed=seed,source=source,date=dates[t],max_error=float(np.abs(full-expected).max())))
            if row%100==0:
                print('Re-encoded',seed,row,len(allids),'seconds',round(time.perf_counter()-start,1),flush=True)
                save('state.json',dict(status='reencoding',seed=seed,row=row,total=len(allids)))
        def xrow(t,b,which):
            if which=='cached':vec=zz[t,np.r_[*b]]
            else:
                names,z=newrows[int(t)]
                vec=z[np.searchsorted(names,np.r_[*b])]
            return np.column_stack((vec,np.repeat([1.,-1.],[40,10]))).astype(np.float32)
        for source in ('cached','current'):
            base=np.stack([xrow(t,baskets[t],source) for t in tr]);vx=np.stack([xrow(t,baskets[t],source) for t in va])
            hx=np.stack([xrow(t,baskets[t],source) for t in ti])
            ax,ay,aw=[],[],[];bw=np.ones(473)
            for t,variants,row in zip(tr,draws,range(473)):
                if variants:bw[row]=.95
                for b,g in variants:ax.append(xrow(t,b,source));ay.append(g);aw.append(.05/len(variants))
            path=OUT/f'{source}_{seed}_data.npz'
            np.savez(path,base=base,vx=vx,hx=hx,y=y,vy=vy,aug_x=np.concatenate([base,np.stack(ax)]),aug_y=np.r_[y,ay],aug_w=np.r_[bw,aw])
            record(path);prepared[source,seed]=path
        del zz,rawscore,oldrows,newrows,model
    pd.DataFrame(encode_checks).to_csv(OUT/'encoding_checks.csv',index=False)
    pd.DataFrame(full_checks).to_csv(OUT/'full_forward_checks.csv',index=False)
    trace,loss_checks,selections,identities=[],[],[],[]
    save('state.json',dict(status='training',heads=12))
    for source in ('cached','current'):
        for seed in (42,43,44):
            with np.load(prepared[source,seed]) as a:
                for suffix in ('plain','fixed5'):
                    x,y,w=(a['base'],a['y'],np.ones(473)) if suffix=='plain' else (a['aug_x'],a['aug_y'],a['aug_w'])
                    result,xd,yd,wd,nd=train(x,y,w,a['vx'],a['vy'],seed,source+'_'+suffix,trace,loss_checks)
                    selections.append(result);identities.append(dict(arm=result['arm'],seed=seed,inputs=xd,labels=yd,weights=wd,normalization=nd))
    pd.DataFrame(trace).to_csv(OUT/'training_loss.csv',index=False)
    pd.DataFrame(identities).to_csv(OUT/'identity.csv',index=False)
    pd.DataFrame(loss_checks).to_csv(OUT/'loss_checks.csv',index=False)
    save('validation_lock.json',dict(selections=selections,train_dates=parent_lock['train_dates'],validation_dates=parent_lock['validation_dates'],hashes=HASHES.copy()))
    print('Locked all 12 independently trained heads',flush=True)
    predictions={}
    for result in selections:
        source=result['arm'].split('_')[0];seed=result['seed']
        with np.load(prepared[source,seed]) as a:hx=torch.from_numpy(a['hx'])
        cp=torch.load(result['path'],weights_only=True);head=LegPool(False).eval();head.load_state_dict(cp['state'])
        with torch.no_grad():v=head((hx-cp['mu'])/cp['sd']).flatten().numpy()
        predictions.setdefault(result['arm'],[]).append(v)
    signals=pd.DataFrame({k:np.mean(v,axis=0) for k,v in predictions.items()},index=testdays).rename_axis('date')
    signals.to_csv(OUT/'signals.csv')
    save('state.json',dict(status='replaying',heads=12))
    frame,schedule,members=locked_history();schedule=schedule.loc[schedule.date.isin(testdays)]
    f=frame.loc[frame.date.isin(testdays)];key=['date','symbol']
    ss=pd.concat([pd.DataFrame(dict(date=day,symbol=d['symbols'],score=d['score'][dates.get_loc(day)])) for day in testdays],ignore_index=True)
    sources={'old':f.merge(ss.rename(columns={'score':'score_long'}),on=key,validate='one_to_one')}
    sources['old']['score_short']=sources['old'].score_long
    for book in ('B0','P6'):
        s=next(s for lock,s in members.values() if lock['arm']==book)
        z=f.merge(s['test']['long'][key+['score']].rename(columns={'score':'score_long'}),on=key,how='left',validate='one_to_one')
        sources[book]=z.merge(s['test']['short'][key+['score']].rename(columns={'score':'score_short'}),on=key,how='left',validate='one_to_one')
    rows,books=[],[]
    for book,source in sources.items():
        m,w=engine(source,schedule,signals);rows.append(m.assign(book=book));books.append(w.assign(book=book))
    metrics=pd.concat(rows,ignore_index=True);weekly=pd.concat(books,ignore_index=True)
    assert len(metrics)==48 and len(weekly)==14160
    np.testing.assert_allclose(weekly.net,weekly.long_net+weekly.short_net,atol=1e-12,rtol=0)
    metrics.to_csv(OUT/'metrics.csv',index=False);weekly.to_parquet(OUT/'weekly.parquet',index=False)
    save('inputs.json',HASHES);save('state.json',dict(status='trained_and_replayed',heads=12,metric_rows=48,weekly_rows=14160))
    print(metrics.query("book=='old' and rule=='full'")[['arm','fee','annual_calendar','max_drawdown','sharpe']].to_string(index=False),flush=True)

if __name__=='__main__':main()
