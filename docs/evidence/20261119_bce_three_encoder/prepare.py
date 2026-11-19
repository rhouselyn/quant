"""Keep each family's frozen baskets/labels and change only the actual encoder seed."""
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

OUT=Path(__file__).resolve().parent
PARENT=Path('outputs/bce_native40_transfer_v1')
ctx=runpy.run_path(str(PARENT/'train_replay.py'))
std=runpy.run_path('outputs/standard_bce_pool_v1/run_standard.py')
LOCK=Path('outputs/dataset_v2/gpu.lock')
TOKEN=dict(pid=os.getpid(),task=OUT.name)
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

def main():
    torch.set_num_threads(4)
    os.environ['CUDA_VISIBLE_DEVICES']='0';assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inventory=json.loads((OUT/'model_inventory.json').read_text(encoding='utf8'))
    HASHES.update(inventory['input_hashes'])
    for p in (__file__,OUT/'audit_models.py',OUT/'model_inventory.json',PARENT/'identity.csv',PARENT/'signal_baskets.csv',PARENT/'validation_lock.json',
              'src/quant_mvp/hierarchy.py','src/quant_mvp/financial_return.py','src/quant_mvp/hierarchy_data.py'):record(p)
    d,dates,*_=legacy_inputs()
    lock=json.loads((PARENT/'validation_lock.json').read_text(encoding='utf8'))
    phases=dict(train=pd.DatetimeIndex(lock['train_dates']),validation=pd.DatetimeIndex(lock['validation_dates']),
                test=pd.DatetimeIndex(pd.read_csv(PARENT/'signals.csv',parse_dates=['date']).date))
    rawval=pd.DatetimeIndex(json.loads((PARENT/'dates.json').read_text(encoding='utf8'))['validation'])
    vsel=rawval.get_indexer(phases['validation'])
    assert [len(v) for v in phases.values()]==[473,123,295]
    roster=pd.read_csv(PARENT/'signal_baskets.csv',parse_dates=['date']).set_index(['family','phase','date'])
    identities=pd.read_csv(PARENT/'identity.csv').set_index(['arm','seed'])
    active=set(np.random.default_rng(61118).choice(473,236,replace=False))
    batches=[];checks=[];missing=[];grids=[];start=time.perf_counter()
    for family in ('B0','P6'):
        groups={};required={};draws=[]
        for phase,days in phases.items():
            idx=dates.get_indexer(days);pick=vsel if phase=='validation' else np.arange(len(days))
            score=PARENT/f'{family}_{phase}_scores.npz';record(score)
            with np.load(score) as a:gg=ctx['native_groups'](a['long'][pick],a['short'][pick],d['available'][idx],days,d['symbols'])
            groups[phase]=gg;required[phase]=[]
            for day,g in zip(days,gg):
                lr='|'.join(d['symbols'][g[0][:40]]);sr='|'.join(d['symbols'][g[1][:10]])
                assert lr==roster.at[(family,phase,day),'longs'] and sr==roster.at[(family,phase,day),'shorts']
                required[phase].append(set(np.r_[g[0][:40],g[1][:10]]))
        def gap(t,b,kind):
            ids=np.r_[*b];valid=d['valid'][t,ids]&np.isfinite(d['y'][t,ids])
            r=np.expm1(np.where(valid,d['y'][t,ids],0.))
            if not valid.all():missing.append(dict(family=family,date=dates[t],kind=kind,stocks=int((~valid).sum())))
            return np.float32(r[:40].mean()+r[40:].mean()+.08/52)
        ti=dates.get_indexer(phases['train']);vi=dates.get_indexer(phases['validation'])
        y=np.array([gap(t,(g[0][:40],g[1][:10]),'base') for t,g in zip(ti,groups['train'])])
        vy=np.array([gap(t,(g[0][:40],g[1][:10]),'validation') for t,g in zip(vi,groups['validation'])])
        ay,aw=[],[];bw=np.ones(473)
        for row,(t,g) in enumerate(zip(ti,groups['train'])):
            variants=[]
            if row in active:
                rng=np.random.default_rng(np.random.SeedSequence([61118,int(t),1]));seen={tuple(map(tuple,(g[0][:40],g[1][:10])))}
                for _ in range(3):
                    b=tuple(pool[np.sort(rng.choice(len(pool),n,replace=False))] for pool,n in ((g[0][:50],40),(g[1][:20],10)))
                    key=tuple(map(tuple,b))
                    if key not in seen:seen.add(key);variants.append(b)
            if variants:bw[row]=.95
            for b in variants:
                required['train'][row].update(np.r_[*b]);ay.append(gap(t,b,'augmentation'));aw.append(.05/len(variants))
            draws.append(variants)
        assert sum(map(len,draws))==708 and np.isclose(bw.sum()+sum(aw),473)
        for seed in (42,43,44):
            info=next(r for r in inventory['models'] if r['family']==family and r['encoder_seed']==seed)
            folder=Path(info['run']);cfg=yaml.safe_load((folder/'config.yaml').read_text(encoding='utf8'))
            model=None
            if seed!=42:
                model=(HierarchicalRanker(cfg,np.array(json.loads((folder/'preprocessing.json').read_text(encoding='utf8'))['industry_ids'])) if family=='B0' else None)
            features={};aug_features=[]
            for phase,days in phases.items():
                if seed==42:
                    path=PARENT/f'{family}_{phase}_vectors.npy';record(path);z=np.load(path,mmap_mode='r')
                    indices=vsel if phase=='validation' else np.arange(len(days))
                    vecs=[(np.array(sorted(names)),z[int(k),np.array(sorted(names))]) for names,k in zip(required[phase],indices)]
                else:
                    directory=Path(cfg['cache_path']).parent if phase=='train' else std['STANDARD']/'snapshots'/('validation' if phase=='validation' else 'history')
                    for name in ('manifest.json','weekly.parquet','market_features.csv','universe.csv'):record(directory/name)
                    opts,data=std['dataset'](directory,folder,cfg)
                    assert list(data[7])==list(d['symbols'])
                    if model is None:
                        model=build_model(opts,data[8],data[9]) if family=='P6' else HierarchicalRanker(opts,data[8])
                    model.load_state_dict(torch.load(folder/f"epoch_{info['vector_epoch']}.pt",weights_only=True,map_location='cpu'))
                    model=model.cuda().eval();model.chunk_size=128
                    if family=='B0':
                        assert model.cross_summary is None and model.market_film is None and model.cross_stock is None
                        assert all(torch.count_nonzero(p)==0 for n,p in model.named_parameters() if n.endswith(('attn_gate','ffn_gate')))
                    else:assert not model.market_history
                    nd=pd.DatetimeIndex(data[6]);vecs=[]
                    for row,(day,names) in enumerate(zip(days,required[phase])):
                        names=np.array(sorted(names));t=nd.get_loc(day);oi=dates.get_loc(day)
                        assert d['available'][oi,names].all()
                        window=data[0][t-39:t+1,names].transpose(1,0,2);assert np.isfinite(window).all()
                        with torch.inference_mode():
                            x=torch.from_numpy(window).cuda()
                            h=model.encode(x[None])[0] if family=='B0' else model.encoder(x,normalize=False)
                            zz=F.normalize(model.final_norm(h),dim=-1).cpu().numpy()
                        vecs.append((names,zz))
                        if row==len(days)//2:
                            aa=list(data);aa[4]=data[4].copy();aa[4][t]=d['available'][oi]
                            capture=[];hook=model.final_norm.register_forward_hook(lambda m,args,value:capture.append(value)) if family=='P6' else None
                            with torch.inference_mode():
                                out=model(*make_batch(tuple(aa),[int(t)],opts,'cuda'))
                                full=out[1][0,names].cpu().numpy() if family=='B0' else F.normalize(capture[0],dim=-1).reshape(580,128)[names].cpu().numpy()
                            if hook:hook.remove()
                            np.testing.assert_allclose(zz,full,atol=2e-6,rtol=2e-5)
                            checks.append(dict(family=family,encoder_seed=seed,phase=phase,date=day,max_vector_error=float(np.abs(zz-full).max())))
                            del out,full,aa
                        if row%100==0:
                            save('state.json',dict(status='encoding',family=family,encoder_seed=seed,phase=phase,row=row,total=len(days)))
                            print(family,seed,phase,row,len(days),'seconds',round(time.perf_counter()-start,1),flush=True)
                    model.cpu();torch.cuda.empty_cache();del data
                def xrow(row,b):
                    names,zz=vecs[row];idx=np.searchsorted(names,np.r_[*b])
                    assert np.array_equal(names[idx],np.r_[*b])
                    x=np.column_stack((zz[idx],np.repeat([1.,-1.],[40,10]))).astype(np.float32)
                    assert np.isfinite(x).all() and (np.linalg.norm(x[:,:128],axis=1)>.9).all()
                    return x
                features[phase]=np.stack([xrow(row,(g[0][:40],g[1][:10])) for row,g in enumerate(groups[phase])])
                if phase=='train':aug_features=np.stack([xrow(row,b) for row,variants in enumerate(draws) for b in variants])
                del vecs
            path=OUT/f'{family}_s{seed}_data.npz'
            xx=np.concatenate([features['train'],aug_features]);yy=np.r_[y,ay];ww=np.r_[bw,aw]
            np.savez(path,base=features['train'],vx=features['validation'],hx=features['test'],y=y,vy=vy,aug_x=xx,aug_y=yy,aug_w=ww)
            record(path)
            flat=features['train'].reshape(-1,129);mu,sd=flat.mean(0),flat.std(0)+1e-6
            for suffix,x,labels,weights in (('plain',features['train'],y,np.ones(473)),('fixed5',xx,yy,ww)):
                row=dict(family=family,encoder_seed=seed,suffix=suffix,inputs=ctx['digest'](x,features['validation']),
                         labels=ctx['digest'](labels,vy),weights=ctx['digest'](weights),normalization=ctx['digest'](mu,sd),encoder_sha256=info['encoder_sha256'])
                for key in ('labels','weights'):assert row[key]==identities.at[(family+'_'+suffix,42),key]
                if seed==42:
                    for key in ('inputs','normalization'):assert row[key]==identities.at[(family+'_'+suffix,42),key]
                batches.append(row)
            del model;model=None
    pd.DataFrame(batches).to_csv(OUT/'batch_identity.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'full_forward_checks.csv',index=False)
    pd.DataFrame(missing).to_csv(OUT/'missing_returns.csv',index=False)
    save('inputs.json',HASHES);save('state.json',dict(status='prepared',encoders=6,heads_planned=12))
    print('Prepared six genuinely different encoder inputs with frozen labels/weights.',flush=True)

if __name__=='__main__':
    waiting=False
    while True:
        try:
            with LOCK.open('x') as f:json.dump(TOKEN,f)
            break
        except FileExistsError:
            if not waiting:print('Waiting for shared GPU lock',flush=True);waiting=True
            time.sleep(3)
    try:main()
    finally:
        if LOCK.exists() and json.loads(LOCK.read_text(encoding='utf8'))==TOKEN:LOCK.unlink()
