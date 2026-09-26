"""Long-only capacity read + aggregate exposure targeting.

Part 1: where do the traded names sit in that week's liquidity distribution?
Part 2: long-only cannot short, so cash is the only hedge.  Scale the whole book by a
     trailing-vol target (lagged, no lookahead) and measure the ABSOLUTE return series,
     because that is what a long-only investor actually earns.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path('data/sw_580_observed_features')
RUNS = {
    'lambda1_s42': 'real_hierarchical_580_h5_15ep_gatefrozen0_lambda1',
    'lambda1_s43': 'real_hierarchical_580_h5_15ep_lambda1_s43',
    'lambda1_s44': 'real_hierarchical_580_h5_15ep_lambda1_s44',
}
TOP_N, COST = 20, 10.0


def load(run):
    p = pd.read_parquet(Path('outputs') / run / 'predictions.parquet')
    p = p.dropna(subset=['score', 'realized_return']).copy()
    p['date'] = pd.to_datetime(p['date'])
    w = pd.read_parquet(DATA / 'weekly.parquet')
    w['date'] = pd.to_datetime(w['date'])
    w = w[['date', 'symbol', 'amount', 'ret1']]
    m = p.merge(w, on=['date', 'symbol'], how='left')
    m['rank'] = m.groupby('date').score.rank(ascending=False, method='first')
    m['cw'] = m.groupby('date').realized_return.transform('mean')
    m['pct'] = m.groupby('date').amount.rank(pct=True)
    return m


def series(m):
    """Weekly book gross/abs return and turnover, on the real top-20."""
    rows, prev = [], set()
    for d, g in m[m['rank'] <= TOP_N].groupby('date'):
        names = set(g.symbol)
        to = (len(names - prev) + len(prev - names)) / TOP_N if prev else 1.0
        rows.append(dict(date=d, abs_ret=g.realized_return.mean(),
                         excess=(g.realized_return.mean() - g.cw.iloc[0]),
                         to=to))
        prev = names
    s = pd.DataFrame(rows).sort_values('date').set_index('date')
    s['net_abs'] = s.abs_ret - s.to * COST / 1e4
    s['net_ex'] = s.excess - s.to * COST / 1e4
    s['mkt'] = m.groupby('date').realized_return.mean()
    return s


def stats(r, label):
    sd = r.std(ddof=1)
    return dict(rule=label, mean_bps=r.mean() * 1e4, ann_vol=sd * np.sqrt(52),
                sharpe=r.mean() / sd * np.sqrt(52), mdd=(1 + r).cumprod().min() / (1 + r).cumprod().max() - 1,
                weeks=len(r))


def main():
    for key, run in RUNS.items():
        m = load(run)
        head = m[m['rank'] <= TOP_N]
        print(f'--- {key} ---')
        print('amount percentile inside the head: mean %.3f  median %.3f  <10%%: %.1f%%  <25%%: %.1f%%  | universe mean %.3f' % (
            head.pct.mean(), head.pct.median(), 100 * (head.pct < .10).mean(),
            100 * (head.pct < .25).mean(), m.pct.mean()))
        s = series(m)
        out = [stats(s.net_abs, 'buy & hold top20 (abs)')]
        for tgt in [0.06, 0.08, 0.10, 0.12]:
            tv = tgt / np.sqrt(52)
            # trailing realised vol of the MARKET, shifted so only past data is used
            rv = s.mkt.rolling(26).std().shift(1)
            beta = (tv / rv).clip(upper=1.0).fillna(1.0)
            r = beta * s.net_abs + (1 - beta) * 0.0
            out.append(stats(r, f'vol-target {int(tgt*100)}% ann, market-vol'))
        for tgt in [0.06, 0.10]:
            tv = tgt / np.sqrt(52)
            rv = s.net_abs.rolling(26).std().shift(1)
            beta = (tv / rv).clip(upper=1.0).fillna(1.0)
            out.append(stats(beta * s.net_abs, f'vol-target {int(tgt*100)}% ann, book-vol'))
        print(pd.DataFrame(out).round(3).to_string(index=False))
        print('mean cash weight (vol-target 8%%): %.1f%%' % (100 * (1 - ((0.08 / np.sqrt(52)) / s.mkt.rolling(26).std().shift(1)).clip(upper=1.0).fillna(1.0)).mean()))
        print()


if __name__ == '__main__':
    main()
