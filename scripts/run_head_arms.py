"""Does shrinking the score head from 20% of the cross-section to the traded book pay?

`extreme_score_loss` labels the top and bottom 20% of ~580 names, i.e. ranks
1-116, while the account only ever holds 20.  The rank-bucket profile showed the
surviving arm's edge is monotone over the first 80 names and the hinge arm's sat
entirely below rank 20, so the head definition is the one line in the loss the
diagnostics point at.  Both arms are the lambda=1-frozen arm with `extreme_count`
added and nothing else changed.
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.hierarchy_engine import train, write_json

ARMS = [
    ('head20', 'configs/hierarchical_580_h5_15ep_gatefrozen0_lambda1_head20.yaml'),
    ('head50', 'configs/hierarchical_580_h5_15ep_gatefrozen0_lambda1_head50.yaml'),
]


def main():
    root = Path('outputs/head_sweep')
    root.mkdir(exist_ok=True)
    rows = []
    for key, config in ARMS:
        out = Path(f'outputs/real_hierarchical_580_h5_15ep_gatefrozen0_lambda1_{key}')
        write_json(root / 'progress.json', dict(status='training', current=key, completed=rows))
        if out.exists():
            result = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
            if result.get('status') == 'complete':
                rows.append(dict(arm=key, skipped=True, best_epoch=result['best_epoch'],
                                 selection_value=result['selection_value']))
                continue
        cfg = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
        result = train(cfg, out)
        rows.append(dict(arm=key, run=out.name, best_epoch=result['best_epoch'],
                        selection_value=result['selection_value']))
        write_json(root / 'progress.json', dict(status='training', current=key, completed=rows))
    write_json(root / 'progress.json', dict(status='complete', completed=rows))


if __name__ == '__main__':
    main()
