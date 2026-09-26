"""Do the two turnover-control doses internalise the no-trade band?

The band sweep priced hysteresis at +10.6bps/week (paired t=2.24) on frozen
predictions, so the question for training is whether a score that is *itself*
sticky beats a score that is only made sticky at execution time.  Both arms keep
lambda=1 and frozen gates and differ only in the control weight.
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.hierarchy_engine import train, write_json

ARMS = [
    ('stick0p3', 'configs/hierarchical_580_h5_15ep_lambda1_stick0p3.yaml'),
    ('stick1', 'configs/hierarchical_580_h5_15ep_lambda1_stick1.yaml'),
    ('stick1_cap', 'configs/hierarchical_580_h5_15ep_lambda1_stick1_cap.yaml'),
    ('stick3_cap_zone', 'configs/hierarchical_580_h5_15ep_lambda1_stick3_cap_zone.yaml'),
]


def main():
    root = Path('outputs/stickiness')
    root.mkdir(exist_ok=True)
    rows = []
    for key, config in ARMS:
        out = Path(f'outputs/real_hierarchical_580_h5_15ep_lambda1_{key}')
        write_json(root / 'progress.json', dict(status='training', current=key, completed=rows))
        if out.exists():
            result = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
            if result.get('status') == 'complete':
                rows.append(dict(arm=key, skipped=True, best_epoch=result['best_epoch'],
                                 selection_value=result['selection_value']))
                continue
        cfg = yaml.safe_load(Path(config).read_text())
        result = train(cfg, out)
        rows.append(dict(arm=key, run=out.name, best_epoch=result['best_epoch'],
                        selection_value=result['selection_value']))
        write_json(root / 'progress.json', dict(status='training', current=key, completed=rows))
    write_json(root / 'progress.json', dict(status='complete', completed=rows))


if __name__ == '__main__':
    main()
