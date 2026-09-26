"""Controls for the two long-only findings.

Timing test:  vol targeting cuts variance a lot, so a higher Sharpe is meaningless until
it beats a STATIC book of the same average exposure.  Compare same-mean-beta, paired weekly.
Capacity test: dose-response of a liquidity screen, and re-benchmarking excess against the
LIQUID universe instead of all 580 (the old benchmark contains the thin names themselves).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from longonly_capacity_targeting import COST, TOP_N, load, series

RUNS = {
    'lambda1_s42': 'real_hierarchical_580_h5_15ep_gatefrozen0_lambda1',
    'lambda1_s43': 'real_hierarchical_580_h5_15ep_lambda1_s43',
    'lambda1_s44': 'real_hierarchical_580_h5_15ep_lambda1_s44',
}


def paired(a, b, label):
    d = (a - b).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(rule=label, diff_bps=d.mean() * 1e4, t=d.mean() / se, weeks=len(d))


def main():
    timing, capacity = [], []
    for key, run in RUNS.items():
        m = load(run)
        s = series(m)
        # ---- timing: same average exposure, static vs vol-timed
        for tgt in [0.06, 0.08, 0.10]:
            tv = tgt / np.sqrt(52)
            for est, col in [('market', 'mkt'), ('book', 'net_abs')]:
                rv = s[col].rolling(26).std().shift(1)
                beta = (tv / rv).clip(upper=1.0).fillna(1.0)
                st = beta.mean()
                d = paired(beta * s.net_abs, st * s.net_abs, f'{key} {est} tgt{int(tgt*100)}')
                d.update(mean_beta=round(st, 3), timed_sharpe=round((beta * s.net_abs).mean() / (beta * s.net_abs).std(ddof=1) * np.sqrt(52), 2),
                         static_sharpe=round((st * s.net_abs).mean() / (st * s.net_abs).std(ddof=1) * np.sqrt(52), 2))
                timing.append(d)
        # ---- capacity: screen dose-response, and liquid-universe benchmark
        lb = m[m.pct >= .5].groupby('date').realized_return.mean()
        m['lbench'] = m.date.map(lb)
        for thr in [0.0, 0.05, 0.10, 0.20, 0.30]:
            rows, prev = [], set()
            for d, g in m.groupby('date'):
                ok = g[g.pct >= thr].sort_values('rank').head(TOP_N)
                names = set(ok.symbol)
                to = (len(names - prev) + len(prev - names)) / TOP_N if prev else 1.0
                rows.append(dict(date=d, ex=ok.realized_return.mean() - g.realized_return.mean(),
                                 lex=ok.realized_return.mean() - lb.loc[d], to=to))
                prev = names
            r = pd.DataFrame(rows).sort_values('date').set_index('date')
            ncut = float(((m['rank'] <= TOP_N) & (m.pct < thr)).groupby(m.date).sum().mean())
            capacity.append(dict(seed=key, screen=f'>{int(thr*100)}pct', head_names_cut=ncut,
                                 gross_ex=(r.ex * 1e4).mean(), net_ex=(r.ex * 1e4).mean() - r.to.mean() * COST,
                                 gross_vs_liquid=(r.lex * 1e4).mean(), net_vs_liquid=(r.lex * 1e4).mean() - r.to.mean() * COST,
                                 turnover=r.to.mean(),
                                 sharpe=r.ex.mean() / r.ex.std(ddof=1) * np.sqrt(52)))
    pd.set_option('display.width', 260)
    print('=== exposure TIMING vs static, identical average exposure ===')
    print(pd.DataFrame(timing).round(2).to_string(index=False))
    print('\n=== liquidity screen dose-response (excess bps/week) ===')
    print(pd.DataFrame(capacity).round(1).to_string(index=False))


if __name__ == '__main__':
    main()
