"""Fuse three real backbones on common masks, then build common 40/10 examples."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml
from quant_mvp.hierarchy import HierarchicalRanker
from quant_mvp.financial_return import build_model
from quant_mvp.hierarchy_engine import make_batch
from replay_equal_weight_model_books import legacy_inputs
from directional_three_seed import standard_scores

OUT=Path(__file__).resolve().parent
PARENT=Path('outputs/bce_three_encoder_v1')
NATIVE=Path('outputs/bce_native40_transfer_v1')
CTX=runpy.run_path(str(NATIVE/'train_replay.py'))
STD=runpy.run_path('outputs/standard_bce_pool_v1/run_standard.py')
PROTOCOL=Path('docs/research_20261119_bce_fused_three_protocol.md')
LOCK=Path('outputs/dataset_v2/gpu.lock')
TOKEN=dict(pid=os.getpid(),task=OUT.name)
HASHES={}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def record(p):
    p=Path(p).resolve()
    if str(p) not in HASHES:HASHES[str(p)]=sha(p)

def save(name,obj):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2),encoding='utf8')
    for attempt in range(20):
        try:t.replace(p);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def state(status,**kw):save('state.json',dict(status=status,pid=os.getpid(),updated_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),**kw))

def fused_epochs(raw,family,epochs,side):
    values=[]
    for e in epochs:
        s=raw[e][side]
        if family=='B0':s=(s-np.nanmean(s,axis=1,keepdims=True))/(np.nanstd(s,axis=1,keepdims=True)+1e-12)
        values.append(s)
    assert all(np.array_equal(np.isfinite(values[0]),np.isfinite(x)) for x in values[1:])
    return np.mean(values,axis=0)

def main():
    torch.set_num_threads(4);os.environ['CUDA_VISIBLE_DEVICES']='0';assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inventory=json.loads((PARENT/'model_inventory.json').read_text(encoding='utf8'))
    for p,h in inventory['input_hashes'].items():assert sha(p)==h;HASHES[p]=h
    save('model_inventory.json',inventory)
    for p in (__file__,PROTOCOL,NATIVE/'train_replay.py',PARENT/'prepare.py',PARENT/'batch_identity.csv',PARENT/'validation_lock.json',
              'src/quant_mvp/hierarchy.py','src/quant_mvp/financial_return.py','src/quant_mvp/hierarchy_data.py','scripts/directional_three_seed.py'):record(p)
    d,dates,*_=legacy_inputs();lock=json.loads((PARENT/'validation_lock.json').read_text(encoding='utf8'))
    phases=dict(train=pd.DatetimeIndex(lock['train_dates']),validation=pd.DatetimeIndex(lock['validation_dates']),
                test=pd.DatetimeIndex(pd.read_csv(PARENT/'signals.csv',parse_dates=['date']).date))
    assert [len(v) for v in phases.values()]==[473,123,295]
    save('dates.json',{k:v.astype(str).tolist() for k,v in phases.items()})
    rawval=pd.DatetimeIndex(json.loads((NATIVE/'dates.json').read_text(encoding='utf8'))['validation'])
    oldroster=pd.read_csv(NATIVE/'signal_baskets.csv',parse_dates=['date']).set_index(['family','phase','date'])
    checks=[];resources=[];basketrows=[];identities=[];missing=[];changes=[];augrows=[];coverage=[]
    for family in ('B0','P6'):
        long_epochs=[1,2,3] if family=='B0' else [6,7,8]
        short_epochs=[6,7,8] if family=='B0' else [3,4,5]
        epochs=sorted(set(long_epochs+short_epochs));allz={};fused={};actual_parts=[]
        for phase,days in phases.items():
            old=d['available'][dates.get_indexer(days)];seedparts=[];allz[phase]={}
            for seed in (42,43,44):
                info=next(x for x in inventory['models'] if x['family']==family and x['encoder_seed']==seed)
                folder=Path(info['run']);cfg=yaml.safe_load((folder/'config.yaml').read_text(encoding='utf8'))
                directory=Path(cfg['cache_path']).parent if phase=='train' else STD['STANDARD']/'snapshots'/('validation' if phase=='validation' else 'history')
                for name in ('manifest.json','weekly.parquet','market_features.csv','universe.csv'):record(directory/name)
                opts,data=STD['dataset'](directory,folder,cfg)
                assert list(data[7])==list(d['symbols'])
                idx=pd.DatetimeIndex(data[6]).get_indexer(days);assert (idx>=39).all()
                native=data[4][idx].copy();different=np.any(native!=old,axis=1)
                coverage.append(dict(family=family,seed=seed,phase=phase,changed_mask_weeks=int(different.sum())))
                a=list(data);a[4]=data[4].copy();a[4][idx]=old;a=tuple(a)
                model=None;raw={};zpath=OUT/f'{family}_s{seed}_{phase}_vectors.npy'
                selected=rawval.get_indexer(days) if phase=='validation' else np.arange(len(days))
                for epoch in epochs:
                    checkpoint=folder/f'epoch_{epoch}.pt';record(checkpoint)
                    cp=OUT/f'{family}_s{seed}_{phase}_e{epoch}.npz';meta=cp.with_suffix('.json')
                    signature=dict(prepare_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),checkpoint_sha256=sha(checkpoint),
                        preprocessing_sha256=sha(folder/'preprocessing.json'),snapshot_sha256=sha(directory/'manifest.json'),
                        dates=days.astype(str).tolist(),mask_sha256=CTX['digest'](old,native))
                    if meta.exists():
                        saved=json.loads(meta.read_text(encoding='utf8'));assert saved['signature']==signature
                        assert sha(cp)==saved['score_sha256']
                        if epoch==info['vector_epoch']:assert sha(zpath)==saved['vector_sha256']
                        with np.load(cp) as q:raw[epoch]={k:q[k] for k in q.files}
                        record(cp);print('Reusing',cp.name,flush=True);continue
                    started=time.perf_counter();ref=None
                    if seed==42:
                        refpath=NATIVE/f'{family}_{phase}_e{epoch}.npz';record(refpath)
                        with np.load(refpath) as q:score=q['score'][selected].copy()
                        if epoch==info['vector_epoch']:
                            p=NATIVE/f'{family}_{phase}_vectors.npy';record(p);np.save(zpath,np.load(p,mmap_mode='r')[selected])
                    else:score=np.full(old.shape,np.nan,dtype=np.float32)
                    native_score=np.full(old.shape,np.nan,dtype=np.float32) if phase=='test' else None
                    z=np.zeros((*old.shape,128),dtype=np.float32) if seed!=42 and epoch==info['vector_epoch'] else None
                    needed=range(len(days)) if seed!=42 else (np.flatnonzero(different) if phase=='test' else [])
                    if len(needed):
                        if model is None:model=(HierarchicalRanker(opts,data[8]) if family=='B0' else build_model(opts,data[8],data[9])).cuda().eval();model.chunk_size=128
                        model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
                        captured=[];hook=model.final_norm.register_forward_hook(lambda m,args,value:captured.append(value)) if family=='P6' and z is not None else None
                        torch.cuda.reset_peak_memory_stats()
                        for row in needed:
                            captured.clear()
                            if seed!=42:
                                with torch.inference_mode():
                                    result=model(*make_batch(a,[int(idx[row])],opts,'cuda'))
                                    sc=result[2][0].cpu().numpy() if family=='B0' else result.get('decision_score',result['mu_log'])[0].cpu().numpy()
                                    score[row]=np.where(old[row],sc,np.nan)
                                    if z is not None:z[row]=result[1][0].cpu().numpy() if family=='B0' else F.normalize(captured[0],dim=-1).reshape(580,128).cpu().numpy()
                            if phase=='test' and different[row]:
                                with torch.inference_mode():
                                    nr=model(*make_batch(data,[int(idx[row])],opts,'cuda'))
                                    ns=nr[2][0].cpu().numpy() if family=='B0' else nr.get('decision_score',nr['mu_log'])[0].cpu().numpy()
                                    native_score[row]=np.where(native[row],ns,np.nan)
                            if row%100==0:
                                state('encoding',family=family,seed=seed,phase=phase,epoch=epoch,row=int(row),total=len(days))
                                print(family,seed,phase,epoch,row,len(days),'seconds',round(time.perf_counter()-started,1),flush=True)
                        if hook:hook.remove()
                        if z is not None:np.save(zpath,z);del z
                        resources.append(dict(family=family,seed=seed,phase=phase,epoch=epoch,seconds=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
                    assert np.isfinite(score[old]).all()
                    raw[epoch]=dict(score=score)
                    if phase=='test':
                        native_score[~different]=np.where(native[~different],score[~different],np.nan)
                        assert np.isfinite(native_score[native]).all();raw[epoch]['native']=native_score
                    np.savez(cp,**raw[epoch]);record(cp)
                    if epoch==info['vector_epoch']:record(zpath)
                    content=dict(signature=signature,score_sha256=sha(cp))
                    if epoch==info['vector_epoch']:content['vector_sha256']=sha(zpath)
                    save(meta.name,content)
                if model is not None:model.cpu();del model;torch.cuda.empty_cache()
                # Complete CPU forward is independent of the export loop and cached raw arrays.
                row=len(days)//2;t=int(idx[row]);epoch=info['vector_epoch']
                model=(HierarchicalRanker(opts,data[8]) if family=='B0' else build_model(opts,data[8],data[9])).eval();model.chunk_size=128
                model.load_state_dict(torch.load(folder/f'epoch_{epoch}.pt',weights_only=True,map_location='cpu'))
                capture=[];hook=model.final_norm.register_forward_hook(lambda m,args,v:capture.append(v)) if family=='P6' else None
                with torch.inference_mode():
                    result=model(*make_batch(a,[t],opts,'cpu'))
                    sc=result[2][0].numpy() if family=='B0' else result.get('decision_score',result['mu_log'])[0].numpy()
                    zz=result[1][0].numpy() if family=='B0' else F.normalize(capture[0],dim=-1).reshape(580,128).numpy()
                if hook:hook.remove()
                zref=np.load(zpath,mmap_mode='r')[row];keep=old[row]
                np.testing.assert_allclose(sc[keep],raw[epoch]['score'][row,keep],atol=2e-5,rtol=2e-5)
                np.testing.assert_allclose(zz[keep],zref[keep],atol=2e-5,rtol=2e-5)
                checks.append(dict(family=family,seed=seed,phase=phase,max_score_error=float(np.abs(sc[keep]-raw[epoch]['score'][row,keep]).max()),max_vector_error=float(np.abs(zz[keep]-zref[keep]).max())))
                allz[phase][seed]=zpath
                seedparts.append({side:standard_scores(fused_epochs(raw,family,win,'score'),old) for side,win in [('long',long_epochs),('short',short_epochs)]})
                if phase=='test':actual_parts.append({side:standard_scores(fused_epochs(raw,family,win,'native'),native) for side,win in [('long',long_epochs),('short',short_epochs)]})
                del data,a,raw,model
            fused[phase]={side:np.mean([p[side] for p in seedparts],axis=0) for side in ('long','short')}
            np.savez(OUT/f'{family}_{phase}_scores.npz',**fused[phase]);record(OUT/f'{family}_{phase}_scores.npz')
        actual={side:np.mean([p[side] for p in actual_parts],axis=0) for side in ('long','short')}
        assert len(actual_parts)==3
        for side in ('long','short'):assert all(np.array_equal(np.isfinite(p[side]),np.isfinite(actual_parts[0][side])) for p in actual_parts)
        np.savez(OUT/f'{family}_actual_test_scores.npz',**actual);record(OUT/f'{family}_actual_test_scores.npz')
        groups={phase:CTX['native_groups'](fused[phase]['long'],fused[phase]['short'],d['available'][dates.get_indexer(days)],days,d['symbols']) for phase,days in phases.items()}
        for phase,days in phases.items():
            for day,g in zip(days,groups[phase]):
                longs='|'.join(d['symbols'][g[0][:40]]);shorts='|'.join(d['symbols'][g[1][:10]])
                basketrows.append(dict(family=family,phase=phase,date=day,longs=longs,shorts=shorts))
                changes.append(dict(family=family,phase=phase,date=day,long_overlap=len(set(longs.split('|'))&set(oldroster.at[(family,phase,day),'longs'].split('|'))),short_overlap=len(set(shorts.split('|'))&set(oldroster.at[(family,phase,day),'shorts'].split('|')))))
        def gap(day,b,kind):
            t=dates.get_loc(day);ids=np.r_[*b];valid=d['valid'][t,ids]&np.isfinite(d['y'][t,ids]);r=np.expm1(np.where(valid,d['y'][t,ids],0.))
            if not valid.all():missing.append(dict(family=family,date=day,kind=kind,missing_long=int((~valid[:40]).sum()),missing_short=int((~valid[40:]).sum()),symbols='|'.join(d['symbols'][ids[~valid]])))
            return np.float32(r[:40].mean()+r[40:].mean()+.08/52)
        y=np.array([gap(day,(g[0][:40],g[1][:10]),'base') for day,g in zip(phases['train'],groups['train'])])
        vy=np.array([gap(day,(g[0][:40],g[1][:10]),'validation') for day,g in zip(phases['validation'],groups['validation'])])
        active=set(np.random.default_rng(61118).choice(473,236,replace=False));draws=[];ay=[];aw=[];bw=np.ones(473)
        for row,(day,g) in enumerate(zip(phases['train'],groups['train'])):
            variants=[]
            if row in active:
                rng=np.random.default_rng(np.random.SeedSequence([61118,int(dates.get_loc(day)),1]));seen={tuple(map(tuple,(g[0][:40],g[1][:10])))}
                for _ in range(3):
                    b=tuple(pool[np.sort(rng.choice(len(pool),n,replace=False))] for pool,n in ((g[0][:50],40),(g[1][:20],10)))
                    key=tuple(map(tuple,b))
                    if key not in seen:seen.add(key);variants.append(b)
            if variants:bw[row]=.95
            for b in variants:ay.append(gap(day,b,'augmentation'));aw.append(.05/len(variants))
            draws.append(variants)
        assert sum(map(len,draws))==708 and np.isclose(bw.sum()+sum(aw),473)
        augrows.append(dict(family=family,active_weeks=len(active),variants=sum(map(len,draws))))
        for seed in (42,43,44):
            def xrow(z,row,b):
                x=np.column_stack((z[row,np.r_[*b]],np.repeat([1.,-1.],[40,10]))).astype(np.float32)
                assert np.isfinite(x).all() and (np.linalg.norm(x[:,:128],axis=1)>.9).all();return x
            features={};ax=[]
            for phase in phases:
                z=np.load(allz[phase][seed],mmap_mode='r');features[phase]=np.stack([xrow(z,row,(g[0][:40],g[1][:10])) for row,g in enumerate(groups[phase])])
                if phase=='train':ax=[xrow(z,row,b) for row,variants in enumerate(draws) for b in variants]
            xx=np.concatenate([features['train'],np.stack(ax)]);yy=np.r_[y,ay];ww=np.r_[bw,aw]
            p=OUT/f'{family}_s{seed}_data.npz';np.savez(p,base=features['train'],vx=features['validation'],hx=features['test'],y=y,vy=vy,aug_x=xx,aug_y=yy,aug_w=ww);record(p)
            flat=features['train'].reshape(-1,129);mu,sd=flat.mean(0),flat.std(0)+1e-6
            for suffix,x,labels,weights in [('plain',features['train'],y,np.ones(473)),('fixed5',xx,yy,ww)]:identities.append(dict(family=family,seed=seed,suffix=suffix,inputs=CTX['digest'](x,features['validation']),labels=CTX['digest'](labels,vy),weights=CTX['digest'](weights),normalization=CTX['digest'](mu,sd)))
    for name,rows in [('full_forward_checks',checks),('resource',resources),('signal_baskets',basketrows),('roster_changes',changes),('missing_returns',missing),('identity',identities),('augmentation',augrows),('coverage',coverage)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    save('inputs.json',HASHES);state('prepared',heads_planned=12)
    print('Prepared fused common baskets, both execution-score variants, all 473/123/295 weeks.',flush=True)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);state('waiting_gpu')
    while True:
        try:
            with LOCK.open('x') as f:json.dump(TOKEN,f)
            break
        except FileExistsError:time.sleep(10)
    try:main()
    except BaseException as exc:state('failed',error=repr(exc));raise
    finally:
        if LOCK.exists() and json.loads(LOCK.read_text(encoding='utf8'))==TOKEN:LOCK.unlink()
