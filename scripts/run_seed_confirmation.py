"""Replicate the lambda=1 head across seeds.

Seven arms trained off the lambda=1 base span -14.9 to +25.9 bps/week of test net
at band=0, so the single surviving number sits at the top of that distribution
rather than at its centre. Seeds only move the initialisation and the per-epoch
batch permutation - data, universe and labels are identical - so the resulting
heads are directly comparable, and the no-trade band can be re-read on each one
for free. The question is no longer which loss term wins but whether lambda=1 is
a lucky draw and whether band>0 pays on every head.
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.hierarchy_engine import train, write_json

ARMS = [
    ('lambda1_s43', 'configs/hierarchical_580_h5_15ep_gatefrozen0_lambda1_s43.yaml'),
    ('lambda1_s44', 'configs/hierarchical_580_h5_15ep_gatefrozen0_lambda1_s44.yaml'),
]


def main():
    root = Path('outputs/seed_confirmation')
    root.mkdir(exist_ok=True)
    rows = []
    for key, config in ARMS:
        out = Path(f'outputs/real_hierarchical_580_h5_15ep_{key}')
        write_json(root / 'progress.json', dict(status='training', current=key, completed=rows))
        if out.exists():
            metrics = out / 'metrics.json'
            if metrics.exists():
                result = json.loads(metrics.read_text(encoding='utf-8'))
                if result.get('status') == 'complete':
                    rows.append(dict(arm=key, run=out.name, skipped=True,
                                     best_epoch=result['best_epoch'], selection_value=result['selection_value']))
                    continue
        cfg = yaml.safe_load(Path(config).read_text())
        result = train(cfg, out)
        rows.append(dict(arm=key, run=out.name, skipped=False, seed=cfg['seed'],
                         best_epoch=result['best_epoch'], selection_value=result['selection_value']))
        write_json(root / 'progress.json', dict(status='trained', current=key, completed=rows))
    write_json(root / 'progress.json', dict(status='complete', rows=rows))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
