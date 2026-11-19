"""Independent full CPU forward checks for exported GPU scores and embeddings."""
import json
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import torch
import yaml
from quant_mvp.hierarchy import HierarchicalRanker
from quant_mvp.financial_return import build_model
from quant_mvp.hierarchy_engine import make_batch
from replay_old_longshort_bce_loss import legacy_inputs

def main():
    out=Path(__file__).resolve().parent
    std=runpy.run_path('outputs/standard_bce_pool_v1/run_standard.py')
    torch.set_num_threads(4)
    d,dates,*_=legacy_inputs()
    phases=json.loads((out/'dates.json').read_text(encoding='utf8'))
    catalog=json.loads(Path('outputs/group_fox_v1/catalog.json').read_text(encoding='utf8'))['members']
    checks=[]
    for arm in ('B0','P6'):
        member=next(x for x in catalog if x['arm']==arm);run=Path(member['run'])
        cfg=yaml.safe_load((run/'config.yaml').read_text(encoding='utf8'));epoch=3 if arm=='B0' else 8
        for phase in ('train','validation','test'):
            directory=Path(cfg['cache_path']).parent if phase=='train' else std['STANDARD']/'snapshots'/('validation' if phase=='validation' else 'history')
            options,data=std['dataset'](directory,run,cfg)
            dd=pd.DatetimeIndex(phases[phase]);row=len(dd)//2;t=pd.DatetimeIndex(data[6]).get_loc(dd[row])
            available=d['available'][dates.get_loc(dd[row])]
            aa=list(data);aa[4]=data[4].copy();aa[4][t]=available;data=tuple(aa)
            model=(HierarchicalRanker(options,data[8]) if arm=='B0' else build_model(options,data[8],data[9])).eval()
            model.load_state_dict(torch.load(run/f'epoch_{epoch}.pt',weights_only=True,map_location='cpu'));model.chunk_size=128
            capture=[]
            hook=model.final_norm.register_forward_hook(lambda module,args,value:capture.append(value)) if arm=='P6' else None
            with torch.inference_mode():
                result=model(*make_batch(data,[t],options,'cpu'))
                score=result[2][0].numpy() if arm=='B0' else result.get('decision_score',result['mu_log'])[0].numpy()
                z=result[1][0].numpy() if arm=='B0' else torch.nn.functional.normalize(capture[0],dim=-1).reshape(580,128).numpy()
            if hook:hook.remove()
            with np.load(out/f'{arm}_{phase}_e{epoch}.npz') as cache:reference=cache['score'][row]
            vectors=np.load(out/f'{arm}_{phase}_vectors.npy',mmap_mode='r')[row]
            np.testing.assert_allclose(score[available],reference[available],atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(z[available],vectors[available],atol=2e-5,rtol=2e-5)
            checks.append(dict(arm=arm,phase=phase,date=str(dd[row].date()),max_score_error=float(np.abs(score[available]-reference[available]).max()),
                               max_vector_error=float(np.abs(z[available]-vectors[available]).max())))
            print(arm,phase,'full CPU forward verified',flush=True)
    pd.DataFrame(checks).to_csv(out/'forward_checks.csv',index=False)

if __name__=='__main__':main()
