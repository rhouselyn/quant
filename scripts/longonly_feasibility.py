"""Two long-only questions that need no retraining.

1. FEASIBILITY.  The book is entered at the signal week's close.  If that weekly bar
   closed locked at the up-limit (or never traded) the position is unbuyable, so the
   measured 28bps/week is partly a return you could not have collected.
2. EXPOSURE.  Under long-only the only short is cash, so the remaining dial is the
   book's aggregate beta.  Scale weights by a lagged realised-vol target.
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
TOP_N = 20
COST = 10.0  # bps per unit of buys+sells


def limit_threshold(symbol):
    return 0.195 if symbol[:3] in ('300', '688') or symbol[:2] == '68' else 0.095


def load(run):
    p = pd.read_parquet(Path('outputs') / run / 'predictions.parquet')
    p = p.dropna(subset=['score', 'realized_return']).copy()
    p['date'] = pd.to_datetime(p['date'])
    w = pd.read_parquet(DATA / 'weekly.parquet')
    w['date'] = pd.to_datetime(w['date'])
    w = w[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'ret1', 'close_pos', 'hl', 'observed']]
    m = p.merge(w, on=['date', 'symbol'], how='left', suffixes=('', '_w'))
    m['limit'] = [limit_threshold(s) for s in m.symbol]
    m['at_high'] = m.close >= m.high - 1e-9
    m['blocked'] = (m.ret1 >= 0.97 * m.limit) & m.at_high
    m['oneprice'] = m.hl <= 1e-9
    m['untradable'] = m.blocked.fillna(False) | m.oneprice.fillna(False) | (m.observed == 0)
    m['rank'] = m.groupby('date').score.rank(ascending=False, method='first')
    m['cw'] = m.groupby('date').realized_return.transform('mean')
    m['ex'] = (m.realized_return - m.cw) * 1e4
    return m


def book_metrics(m, label, sel):
    """sel: date -> DataFrame of chosen rows (already the intended weights)."""
    rows, prev = [], set()
    for d in sorted(m.date.unique()):
        g = sel.get(d)
        if g is None or len(g) == 0:
            continue
        names = set(g.symbol)
        buys = len(names - prev)
        sells = len(prev - names) if prev else 0
        to = (buys + sells) / TOP_N
        rows.append(dict(date=d, n=len(names), gross=g.ex.mean(),
                         cost=to * COST, net=g.ex.mean() - to * COST, to=to))
        prev = names
    r = pd.DataFrame(rows)
    se = r.net.std(ddof=1) / np.sqrt(len(r))
    sh = r.net.mean() / r.net.std(ddof=1) * np.sqrt(52)
    return dict(label=label, weeks=len(r), n_mean=r.n.mean(), gross=r.gross.mean(),
                cost=r.cost.mean(), net=r.net.mean(), t=r.net.mean() / se,
                sharpe=sh, turnover=r.to.mean())


def main():
    for key, run in RUNS.items():
        m = load(run)
        full = m[m['rank'] <= TOP_N]
        print(f'--- {key} ---')
        print('untradable share inside the head: %.3f%%  (blocked %.3f%%, one-price %.3f%%)' % (
            100 * full.untradable.mean(), 100 * full.blocked.fillna(False).mean(),
            100 * full.oneprice.fillna(False).mean()))
        print('universe untradable share: %.3f%%' % (100 * m.untradable.mean()))
        print('head weekly amount median %.0f vs universe %.0f' % (
            full.amount.median(), m.amount.median()))
        results = []
        results.append(book_metrics(m, 'as measured', {d: g for d, g in full.groupby('date')}))

        # (a) forced sell: untradable head names become cash, weight not redistributed
        def cash_out(mm):
            out = {}
            for d, g in mm[mm['rank'] <= TOP_N].groupby('date'):
                out[d] = g
            return out

        # (b) skip and promote to keep 20 names
        promoted = {}
        for d, g in m.groupby('date'):
            g = g.sort_values('rank')
            ok = g[~g.untradable.fillna(False)]
            promoted[d] = ok.head(TOP_N)
        results.append(book_metrics(m, 'skip unbuyable, promote next', promoted))

        # (c) liquidity screen: drop the thinnest decile of the week first
        liq = {}
        for d, g in m.groupby('date'):
            thr = g.amount.quantile(0.10)
            ok = g[g.amount >= thr].sort_values('rank')
            liq[d] = ok.head(TOP_N)
        results.append(book_metrics(m, 'top20 after 10% amount screen', liq))

        pd.set_option('display.width', 250)
        print(pd.DataFrame(results).round(2).to_string(index=False))
        print()


if __name__ == '__main__':
    main()
