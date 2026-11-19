"""Independently verify native batches, price labels and the first optimizer update."""
import json
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml
from attention_followup import LegPool
from replay_equal_weight_model_books import legacy_inputs

OUT=Path(__file__).resolve().parent
PARENT=Path('outputs/bce_native40_transfer_v1')
ctx=runpy.run_path(str(OUT/'run.py'))
source=runpy.run_path(str(PARENT/'train_replay.py'))

def main():
    torch.set_num_threads(4)
    d,dates,*_=legacy_inputs()
    lock=json.loads((PARENT/'validation_lock.json').read_text(encoding='utf8'))
    td=pd.DatetimeIndex(lock['train_dates']);vd=pd.DatetimeIndex(lock['validation_dates'])
    ti=dates.get_indexer(td);vi=dates.get_indexer(vd)
    phase_dates=json.loads((PARENT/'dates.json').read_text(encoding='utf8'))
    fullval=pd.DatetimeIndex(phase_dates['validation']);vsel=fullval.get_indexer(vd)
    cfg=yaml.safe_load(Path('outputs/abl2000_base/config.yaml').read_text(encoding='utf8'))
    arithmetic=ctx['independent_returns'](cfg,dates,d['symbols'])
    traces=pd.read_csv(PARENT/'training_loss.csv');ids=pd.read_csv(PARENT/'identity.csv').set_index(['arm','seed'])
    active=set(np.random.default_rng(61118).choice(473,236,replace=False))
    results=[];labels=[]
    for family in ('B0','P6'):
        with np.load(PARENT/f'{family}_train_scores.npz') as a:
            groups=source['native_groups'](a['long'],a['short'],d['available'][ti],td,d['symbols'])
        with np.load(PARENT/f'{family}_validation_scores.npz') as a:
            vg=source['native_groups'](a['long'][vsel],a['short'][vsel],d['available'][vi],vd,d['symbols'])
        z=np.load(PARENT/f'{family}_train_vectors.npy',mmap_mode='r')
        vz=np.load(PARENT/f'{family}_validation_vectors.npy',mmap_mode='r')
        def example(t,row,b,vec,kind):
            names=np.r_[*b]
            x=np.column_stack((vec[row,names],np.repeat([1.,-1.],[40,10]))).astype(np.float32)
            r=np.expm1(np.where(d['valid'][t,names]&np.isfinite(d['y'][t,names]),d['y'][t,names],0.))
            gap=np.float32(r[:40].mean()+r[40:].mean()+.08/52)
            rr=np.where(np.isfinite(arithmetic[t,names]),arithmetic[t,names],0.)
            independently=float(rr[:40].mean()+rr[40:].mean()+.08/52)
            assert (gap>0)==(independently>0)
            labels.append(dict(family=family,date=dates[t],kind=kind,cache_gap=float(gap),price_gap=independently))
            return x,gap
        base,y=zip(*(example(t,row,(g[0][:40],g[1][:10]),z,'base') for row,(t,g) in enumerate(zip(ti,groups))))
        base=np.stack(base);y=np.asarray(y)
        vx,vy=zip(*(example(t,int(row),(g[0][:40],g[1][:10]),vz,'validation') for t,row,g in zip(vi,vsel,vg)))
        vx=np.stack(vx);vy=np.asarray(vy)
        ax,ay,aw=[],[],[];bw=np.ones(473)
        for row,(t,g) in enumerate(zip(ti,groups)):
            variants=[]
            if row in active:
                seen={tuple(map(tuple,(g[0][:40],g[1][:10])))}
                rng=np.random.default_rng(np.random.SeedSequence([61118,int(t),1]))
                for _ in range(3):
                    b=tuple(pool[np.sort(rng.choice(len(pool),n,replace=False))] for pool,n in ((g[0][:50],40),(g[1][:20],10)))
                    k=tuple(map(tuple,b))
                    if k not in seen:seen.add(k);variants.append(example(t,row,b,z,'augmentation'))
            if variants:bw[row]=.95
            for x,gap in variants:ax.append(x);ay.append(gap);aw.append(.05/len(variants))
        assert len(ax)==708
        mu=base.reshape(-1,129).mean(0);sd=base.reshape(-1,129).std(0)+1e-6
        for suffix,x,yy,w in (('plain',base,y,np.ones(473)),('fixed5',np.concatenate([base,np.stack(ax)]),np.r_[y,ay],np.r_[bw,aw])):
            arm=family+'_'+suffix
            for seed in (42,43,44):
                for key,value in dict(inputs=ctx['digest'](x,vx),labels=ctx['digest'](yy,vy),weights=ctx['digest'](w),normalization=ctx['digest'](mu,sd)).items():
                    assert value==ids.at[(arm,seed),key],(arm,seed,key)
                cp=torch.load(PARENT/f'{arm}_{seed}.pt',weights_only=True)
                np.testing.assert_array_equal(mu,cp['mu'].numpy());np.testing.assert_array_equal(sd,cp['sd'].numpy())
                xx=torch.from_numpy((x-mu)/sd);vv=torch.from_numpy((vx-mu)/sd)
                target=torch.tensor(yy>0,dtype=torch.float32);vt=torch.tensor(vy>0,dtype=torch.float32);wt=torch.tensor(w,dtype=torch.float32)
                torch.manual_seed(seed);head=LegPool(False);opt=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=.01)
                loss=F.binary_cross_entropy_with_logits(head(xx).flatten(),target,weight=wt,reduction='sum')/473
                opt.zero_grad();loss.backward();opt.step()
                with torch.no_grad():first_val=F.binary_cross_entropy_with_logits(head(vv).flatten(),vt).item()
                reference=traces.loc[traces.arm.eq(arm)&traces.seed.eq(seed)&traces.step.eq(1)].iloc[0]
                np.testing.assert_allclose([float(loss),first_val],[reference.train_bce,reference.val_bce],atol=2e-6,rtol=0)
                head.load_state_dict(cp['state'])
                with torch.no_grad():selected_val=F.binary_cross_entropy_with_logits(head(vv).flatten(),vt).item()
                entry=next(r for r in lock['selections'] if r['arm']==arm and r['seed']==seed)
                np.testing.assert_allclose(selected_val,entry['val_bce'],atol=2e-6,rtol=0)
                results.append(dict(arm=arm,seed=seed,initial_loss_error=float(loss)-reference.train_bce,first_val_error=first_val-reference.val_bce,
                                    selected_val_error=selected_val-entry['val_bce'],batch_fingerprints_match=True))
        print(family,'batch, price labels, first update and selected checkpoint verified',flush=True)
    pd.DataFrame(results).to_csv(OUT/'native_training_checks.csv',index=False)
    pd.DataFrame(labels).to_csv(OUT/'native_price_label_checks.csv',index=False)

if __name__=='__main__':main()
