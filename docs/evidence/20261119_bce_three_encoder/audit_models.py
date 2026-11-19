"""Verify independent backbone seeds before reusing their representations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from quant_mvp.training_state import require_committed_epochs

OUT=Path(__file__).resolve().parent
RUNS={
    'B0':{42:'outputs/dataset_v2/exploratory_F0/runs/research_B0_F0_s42_a1',43:'outputs/kit_transfer_seed_v1/runs/research_B0_F0_s43_a1',44:'outputs/kit_transfer_seed_v1/runs/research_B0_F0_s44_a1'},
    'P6':{42:'outputs/group_fox_v1/runs/research_P6_F0_s42_a1',43:'outputs/m6_history_v1/runs/research_P6_F0_s43_a1',44:'outputs/m6_history_v1/runs/research_P6_F0_s44_a1'}}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    rows=[];hashes={}
    for family,runs in RUNS.items():
        base=Path(runs[42]);cfg=yaml.safe_load((base/'config.yaml').read_text(encoding='utf8'))
        pre=json.loads((base/'preprocessing.json').read_text(encoding='utf8'))
        protocol=json.loads((base/'training_protocol.json').read_text(encoding='utf8'))
        verification=json.loads((base/'train_verification.json').read_text(encoding='utf8'))
        encoder_shas=[]
        for seed,folder in runs.items():
            folder=Path(folder);other=yaml.safe_load((folder/'config.yaml').read_text(encoding='utf8'))
            differences={k for k in set(cfg)|set(other) if cfg.get(k)!=other.get(k)}
            assert differences<={'seed','source_sha256'} and other['seed']==seed,(family,seed,differences)
            assert json.loads((folder/'preprocessing.json').read_text(encoding='utf8'))==pre
            pp=json.loads((folder/'training_protocol.json').read_text(encoding='utf8'))
            assert pp['training_dates']==protocol['training_dates'] and len(pp['training_dates'])==779
            vv=json.loads((folder/'train_verification.json').read_text(encoding='utf8'))
            for k in ('train_dates_sha256','available_sha256','observed_sha256','valid_y_sha256','y_sha256','X_sha256','infonce_pairs_sha256'):
                assert vv[k]==verification[k],(family,seed,k)
            require_committed_epochs(folder,range(1,9))
            epoch=3 if family=='B0' else 8
            state=torch.load(folder/f'epoch_{epoch}.pt',weights_only=True,map_location='cpu')
            prefix='temporal.encoder.' if family=='B0' else 'encoder.'
            keys=sorted(k for k in state if k.startswith(prefix) or k.startswith('final_norm.'))
            assert len(keys)>20
            h=hashlib.sha256()
            for k in keys:h.update(k.encode());h.update(state[k].contiguous().numpy().tobytes())
            encoder_sha=h.hexdigest();encoder_shas.append(encoder_sha)
            for name in ('config.yaml','preprocessing.json','training_protocol.json','train_verification.json',f'epoch_{epoch}.pt'):
                p=folder/name;hashes[str(p.resolve())]=sha(p)
            if (folder/'training_state.pt').exists():hashes[str((folder/'training_state.pt').resolve())]=sha(folder/'training_state.pt')
            rows.append(dict(family=family,encoder_seed=seed,run=str(folder.resolve()),vector_epoch=epoch,
                             encoder_sha256=encoder_sha,encoder_tensors=len(keys),training_weeks=779,
                             config_differences=sorted(differences),config_sha256=sha(folder/'config.yaml'),
                             checkpoint_sha256=sha(folder/f'epoch_{epoch}.pt'),committed_state=(folder/'training_state.pt').exists()))
        assert len(set(encoder_shas))==3,family
    for p in (Path(__file__),Path('docs/research_20261119_bce_three_encoder_protocol.md')):hashes[str(p.resolve())]=sha(p)
    (OUT/'model_inventory.json').write_text(json.dumps(dict(models=rows,input_hashes=hashes),indent=2),encoding='utf8')
    print('Verified six real backbones: per-family configs/data/preprocessing matched; encoder weights pairwise distinct.',flush=True)

if __name__=='__main__':main()
