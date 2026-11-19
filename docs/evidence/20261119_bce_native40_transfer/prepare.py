"""Export native scores under the explicitly authorized legacy information mask."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
import yaml
from quant_mvp.hierarchy import HierarchicalRanker
from quant_mvp.financial_return import build_model
from quant_mvp.hierarchy_engine import make_batch
from replay_old_longshort_bce_loss import legacy_inputs
from directional_three_seed import cache_path

OUT=Path(__file__).resolve().parent
std=runpy.run_path('outputs/standard_bce_pool_v1/run_standard.py')
LOCK=Path('outputs/dataset_v2/gpu.lock')
STATE=Path('outputs/rolling_anchor_seeds_v1/state.json')
TOKEN=dict(pid=os.getpid(),task=OUT.name)
HASHES={}

def record(path):
    path=Path(path).resolve()
    if str(path) not in HASHES:
        with path.open('rb') as f:HASHES[str(path)]=hashlib.file_digest(f,'sha256').hexdigest()

def save(name,obj):
    path=OUT/name;temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2),encoding='utf8')
    for attempt in range(20):
        try:temp.replace(path);break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def state(status,**kw):
    save('state.json',dict(status=status,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw))

def gpu_window():
    if not LOCK.exists():
        try:
            with LOCK.open('x') as f:json.dump(TOKEN,f)
            return True
        except FileExistsError:return False
    try:
        owner=json.loads(LOCK.read_text(encoding='utf8'))
        if owner==TOKEN:return True
        other=json.loads(STATE.read_text(encoding='utf8'))
        return owner.get('task')=='rolling_anchor_seeds_v1' and owner.get('pid')==other.get('pid') and other.get('status')=='replaying'
    except (OSError,ValueError):return False

def release():
    if LOCK.exists():
        try:
            if json.loads(LOCK.read_text(encoding='utf8'))==TOKEN:LOCK.unlink()
        except (OSError,ValueError):pass

def main():
    torch.set_num_threads(4)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.12)
    for p in (__file__,'docs/research_20261119_bce_native40_transfer_protocol.md',
              'outputs/group_fox_v1/catalog.json','outputs/bce_input_label_v1/validation_lock.json',
              'src/quant_mvp/hierarchy_data.py','src/quant_mvp/hierarchy.py','src/quant_mvp/financial_return.py',
              'scripts/directional_2018_experiment.py',cache_path(42)):record(p)
    d,dates,_,_,_,train=legacy_inputs()
    parent=json.loads(Path('outputs/bce_input_label_v1/validation_lock.json').read_text(encoding='utf8'))
    assert dates[train].astype(str).tolist()==parent['train_dates']
    sig=pd.read_parquet('outputs/old_longshort_bce_loss_v1/report/signals.parquet')
    vi=dates.get_indexer(sig.query("split=='val'").date)
    test=pd.read_csv('outputs/bce_input_label_v1/signals.csv',parse_dates=['date']).date
    phases=dict(train=dates[train],validation=dates[vi],test=pd.DatetimeIndex(test))
    save('dates.json',{k:v.astype(str).tolist() for k,v in phases.items()})
    catalog=json.loads(Path('outputs/group_fox_v1/catalog.json').read_text(encoding='utf8'))['members']
    coverage=[]
    execution=pd.read_csv(OUT/'resource.csv').to_dict(orient='records') if (OUT/'resource.csv').exists() else []
    prior_source=OUT/'prepare_before_io_retry.py'
    if prior_source.exists():record(prior_source)
    for arm in ('B0','P6'):
        member=next(m for m in catalog if m['arm']==arm);run=Path(member['run'])
        cfg=yaml.safe_load((run/'config.yaml').read_text(encoding='utf8'));lock=json.loads(Path(member['lock']).read_text(encoding='utf8'))
        for p in (run/'config.yaml',run/'preprocessing.json',run/'training_protocol.json',member['lock']):record(p)
        epochs=sorted(set(lock['long_epochs']+lock['short_epochs']))
        vector_epoch=3 if arm=='B0' else 8
        assert epochs==([1,2,3,6,7,8] if arm=='B0' else [3,4,5,6,7,8])
        for epoch in epochs:
            record(run/f'epoch_{epoch}.pt')
            assert HASHES[str((run/f'epoch_{epoch}.pt').resolve())]==lock['checkpoint_sha256'][str(epoch)]
        for phase,requested in phases.items():
            directory=Path(cfg['cache_path']).parent if phase=='train' else std['STANDARD']/'snapshots'/('validation' if phase=='validation' else 'history')
            for name in ('manifest.json','weekly.parquet','market_features.csv','universe.csv'):record(directory/name)
            opts,a=std['dataset'](directory,run,cfg)
            assert list(a[7])==list(d['symbols'])
            nd=pd.DatetimeIndex(a[6]);indices=nd.get_indexer(requested);oi=dates.get_indexer(requested)
            assert (indices>=39).all() and (oi>=0).all()
            native=a[4][indices].copy();old=d['available'][oi]
            assert (old.sum(1)>=50).all()
            coverage.extend(dict(arm=arm,phase=phase,date=str(day.date()),native_available=int(n.sum()),legacy_available=int(o.sum()),newly_allowed=int((o&~n).sum())) for day,n,o in zip(requested,native,old))
            aa=list(a);aa[4]=a[4].copy();aa[4][indices]=old;a=tuple(aa)
            signature=hashlib.sha256(json.dumps(dict(hashes=HASHES,arm=arm,phase=phase,dates=requested.astype(str).tolist()),sort_keys=True).encode()).hexdigest()
            # A stable phase signature excludes later phases' newly recorded files.
            manifest_path=OUT/f'{arm}_{phase}_signature.json'
            current=dict(arm=arm,phase=phase,dates=requested.astype(str).tolist(),
                         hashes={k:v for k,v in HASHES.items() if k in {str(Path(p).resolve()) for p in
                         [__file__,run/'config.yaml',run/'preprocessing.json',member['lock'],directory/'weekly.parquet',directory/'market_features.csv',directory/'universe.csv',cache_path(42),*[run/f'epoch_{e}.pt' for e in epochs]]}})
            if manifest_path.exists():
                previous=json.loads(manifest_path.read_text(encoding='utf8'))
                if previous!=current:
                    source_key=str(Path(__file__).resolve())
                    assert prior_source.exists() and previous['hashes'][source_key]==HASHES[str(prior_source.resolve())]
                    previous['hashes'][source_key]=current['hashes'][source_key]
                    assert previous==current,'non-IO cache dependency changed'
                    save(manifest_path.name,current)
            else:save(manifest_path.name,current)
            scores={}
            for epoch in epochs:
                cp=OUT/f'{arm}_{phase}_e{epoch}.npz'
                zp=OUT/f'{arm}_{phase}_vectors.npy'
                if cp.exists() and (epoch!=vector_epoch or zp.exists()):
                    with np.load(cp) as c:scores[epoch]=c['score']
                    if not any(r['arm']==arm and r['phase']==phase and r['epoch']==epoch for r in execution):
                        execution.append(dict(arm=arm,phase=phase,epoch=epoch,seconds=None,peak_gpu_bytes=None,reused=True))
                    print('Reusing',arm,phase,epoch,flush=True);continue
                model=(HierarchicalRanker(opts,a[8]) if arm=='B0' else build_model(opts,a[8],a[9])).eval()
                model.load_state_dict(torch.load(run/f'epoch_{epoch}.pt',weights_only=True,map_location='cpu'))
                model.chunk_size=128
                allscore=np.full(old.shape,np.nan,dtype=np.float32)
                z=np.zeros((*old.shape,128),dtype=np.float32) if epoch==vector_epoch else None
                captured=[]
                hook=model.final_norm.register_forward_hook(lambda module,args,out:captured.append(out)) if arm=='P6' and z is not None else None
                on_gpu=False;paused=False;started=time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                for row,t in enumerate(indices):
                    while not gpu_window():
                        if on_gpu:model.cpu();torch.cuda.empty_cache();on_gpu=False
                        if not paused:
                            state('waiting_gpu',arm=arm,phase=phase,epoch=epoch,row=row)
                            print('Pausing for GPU owner',arm,phase,epoch,row,flush=True);paused=True
                        time.sleep(3)
                    if not on_gpu:model.cuda();on_gpu=True
                    paused=False;captured.clear()
                    with torch.inference_mode():
                        output=model(*make_batch(a,[int(t)],opts,'cuda'))
                        if arm=='B0':
                            sc=output[2][0].cpu().numpy()
                            if z is not None:z[row]=output[1][0].cpu().numpy()
                        else:
                            sc=output.get('decision_score',output['mu_log'])[0].cpu().numpy()
                            if z is not None:
                                assert len(captured)==1
                                z[row]=F_normalize(captured[0]).reshape(580,128).cpu().numpy()
                    sc[~old[row]]=np.nan;allscore[row]=sc
                    assert np.isfinite(sc[old[row]]).all()
                    if row%50==0 or row==len(indices)-1:
                        state('extracting',arm=arm,phase=phase,epoch=epoch,row=row,total=len(indices))
                        print(arm,phase,epoch,row+1,len(indices),'seconds',round(time.perf_counter()-started,1),flush=True)
                if hook is not None:hook.remove()
                model.cpu();del model,output;captured.clear();torch.cuda.empty_cache()
                execution.append(dict(arm=arm,phase=phase,epoch=epoch,seconds=time.perf_counter()-started,
                                      peak_gpu_bytes=torch.cuda.max_memory_allocated()))
                if z is not None:np.save(zp,z);del z
                np.savez(cp,score=allscore);scores[epoch]=allscore
            if arm=='B0':
                scores={e:(v-np.nanmean(v,axis=1,keepdims=True))/(np.nanstd(v,axis=1,keepdims=True)+1e-12) for e,v in scores.items()}
            long=np.mean([scores[e] for e in lock['long_epochs']],axis=0)
            short=np.mean([scores[e] for e in lock['short_epochs']],axis=0)
            np.savez(OUT/f'{arm}_{phase}_scores.npz',long=long,short=short)
            save('inputs.json',HASHES)
            pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
            pd.DataFrame(execution).to_csv(OUT/'resource.csv',index=False)
            del a,aa,scores,long,short
    state('prepared',date_counts={k:len(v) for k,v in phases.items()})
    print('Prepared all 473/133/295 native inputs without dropping dates',flush=True)

def F_normalize(x):return torch.nn.functional.normalize(x,dim=-1)

if __name__=='__main__':
    try:main()
    finally:release()
