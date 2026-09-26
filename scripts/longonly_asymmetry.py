"""How much of the score's information sits on the side a long-only book can trade?

For each week, rank stocks by score and report the equal-weight forward excess (bps)
of the top band vs the universe mean, and the bottom band vs the universe mean.
A long-only book is paid only for the former.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = {
    'lambda1_s42': 'real_hierarchical_580_h5_15ep_gatefrozen0_lambda1',
    'lambda1_s43': 'real_hierarchical_580_h5_15ep_lambda1_s43',
    'lambda1_s44': 'real_hierarchical_580_h5_15ep_lambda1_s44',
    'frozen_lambda0': 'real_hierarchical_580_h5_15ep_gatefrozen0',
    'baseline': 'real_hierarchical_580_h5_15ep',
}


def load(run, split):
    f = Path('outputs') / run / ('validation_predictions.parquet' if split == 'validation' else 'predictions.parquet')
    if not f.exists():
        return None
    p = pd.read_parquet(f)
    p = p.dropna(subset=['score', 'realized_return'])
    p['rank'] = p.groupby('date').score.rank(ascending=False, method='first')
    p['rankb'] = p.groupby('date').score.rank(ascending=True, method='first')
    p['cw'] = p.groupby('date').realized_return.transform('mean')
    p['ex'] = (p.realized_return - p.cw) * 1e4
    return p


def band(p, lo, hi, col='rank'):
    sel = p[(p[col] >= lo) & (p[col] <= hi)]
    # per-week band mean excess, then average over weeks
    w = sel.groupby('date').ex.mean()
    return w


def main():
    for split in ['validation', 'test']:
        print(f'=== {split} ===', flush=True)
        rows = []
        for key, run in RUNS.items():
            p = load(run, split)
            if p is None:
                print(f'{key}: missing')
                continue
            uni = p.groupby('date').ex.mean()  # ~0 by construction
            d = dict(head=key, n_weeks=int(p.date.nunique()),
                     universe=float(uni.mean()))
            for lo, hi, lbl in [(1, 20, 'top20'), (1, 5, 'top5'), (6, 10, 'r6_10'),
                                (11, 15, 'r11_15'), (16, 20, 'r16_20'), (21, 40, 'r21_40'),
                                (41, 100, 'r41_100'), (1, 20, 'bot20'), (1, 58, 'bot10pct'),
                                (1, 290, 'top_half'), (1, 290, 'bot_half')]:
                col = 'rank' if lbl not in ('bot20', 'bot10pct', 'bot_half') else 'rankb'
                w = band(p, lo, hi, col)
                d[lbl] = float(w.mean()) if len(w) else np.nan
            # asymmetry: |bottom tail excess| vs top20
            d['ratio'] = d['top20'] / abs(d['bot20']) if d['bot20'] else np.nan
            # paired: top20 vs bot20 weekly
            t20, b20 = band(p, 1, 20), band(p, 1, 20, 'rankb')
            idx = t20.index.intersection(b20.index)
            spread = (t20.loc[idx] - b20.loc[idx])
            se = spread.std(ddof=1) / np.sqrt(len(spread))
            d['longshort_bps'] = float(spread.mean())
            d['longshort_t'] = float(spread.mean() / se) if se > 0 else np.nan
            se20 = t20.std(ddof=1) / np.sqrt(len(t20))
            d['top20_t'] = float(t20.mean() / se20)
            se_b = b20.std(ddof=1) / np.sqrt(len(b20))
            d['bot20_t'] = float(b20.mean() / se_b)
            rows.append(d)
        df = pd.DataFrame(rows)
        pd.set_option('display.width', 250)
        print(df.round(1).to_string(index=False), flush=True)
        print()


if __name__ == '__main__':
    main()
