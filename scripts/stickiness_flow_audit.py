"""Does the flow the stickiness loss pays for live where the account charges?

The trained loss charges ‖p_t - p_{t-1}‖_1 with p = softmax(s / mult x weekly sd)
over every covered name, while the backtest only charges for the Top-20 atoms.
This audit puts both on the same footing on frozen predictions: how much of the
penalised flow sits outside the traded neighbourhood, and how well each candidate
measure tracks realised churn compared with the capped-simplex weight vector that
*is* the book once the temperature is removed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = ['real_hierarchical_580_h5_15ep_gatefrozen0_lambda1',
        'real_hierarchical_580_h5_15ep',
        'real_hierarchical_580_h5_15ep_gatefrozen0']
TOP_N, HEAD, MULTS = 20, 40, (.25, .5, 1., 2.)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def softmax(score, mult):
    """Gibbs measure of the score at temperature `mult` x its own cross-sectional sd."""
    z = (score - score.max()) / max(score.std(ddof=1) * mult, 1e-9)
    e = np.exp(np.clip(z, -700., 0.))
    return e / e.sum()


def capped_simplex(score, cap):
    """Projection onto {0 <= w <= cap, sum w = 1} by water-filling a threshold.

    With cap = 1/20 the saturated set is the Top-20 equal-weight book, so half the
    L1 movement of this vector is the realised turnover of that book and names
    outside it carry no mass to churn.
    """
    lo, hi = score.min() - 1., score.max()
    for _ in range(60):
        mid = .5 * (lo + hi)
        if np.clip(score - mid, 0., cap).sum() > 1.:
            lo = mid
        else:
            hi = mid
    w = np.clip(score - lo, 0., cap)
    return w / w.sum()


def main():
    rows = []
    for run in RUNS:
        for split, file in [('validation', 'validation_predictions.parquet'), ('test', 'predictions.parquet')]:
            frame = pd.read_parquet(Path('outputs') / run / file)
            weeks = []
            for _, group in frame.groupby('date', sort=True):
                group = group[np.isfinite(group.score)]
                if len(group) < 50 or not np.isfinite(group.realized_return).any():
                    continue
                weeks.append(group.set_index('symbol').score)
            columns = {f'soft{mult:g}': [] for mult in MULTS}
            columns.update(dict(head=[], tail=[], cap=[], swap=[]))
            for previous, current in zip(weeks, weeks[1:]):
                common = current.index.intersection(previous.index)
                a = current[common].to_numpy(float)
                b = previous[common].to_numpy(float)
                top_prev = set(previous.nlargest(TOP_N, keep='first').index)
                top_now = set(current.nlargest(TOP_N, keep='first').index)
                near = (current[common].rank(ascending=False) <= HEAD) | (previous[common].rank(ascending=False) <= HEAD)
                for mult in MULTS:
                    delta = np.abs(softmax(a, mult) - softmax(b, mult))
                    columns[f'soft{mult:g}'].append(float(delta.sum()))
                    if mult == .5:
                        columns['head'].append(float(delta[near.to_numpy()].sum()))
                        columns['tail'].append(float(delta[~near.to_numpy()].sum()))
                columns['cap'].append(float(np.abs(capped_simplex(a, 1. / TOP_N)
                                                   - capped_simplex(b, 1. / TOP_N)).sum()))
                columns['swap'].append(1 - len(top_prev & top_now) / TOP_N)
            data = pd.DataFrame(columns)
            total = data['head'].sum() + data['tail'].sum()
            print(f'\n=== {run} / {split} ({len(data)} week pairs, {len(common)} names in the last week) ===')
            print(f'  {data["tail"].sum() / total:.1%} of the softmax(0.5) flow is outside the top-{HEAD} union')
            for key in columns:
                if key == 'swap':
                    continue
                print(f'  corr({key:9s}, realised churn) = {np.corrcoef(data[key], data["swap"])[0, 1]:+.3f}'
                      f'   mean {data[key].mean():.4f}   vs mean churn {data["swap"].mean():.3f}')
            row = dict(run=run, split=split, pairs=len(data), tail_share=float(data['tail'].sum() / total))
            for key in columns:
                if key != 'swap':
                    row[f'corr_{key}'] = float(np.corrcoef(data[key], data['swap'])[0, 1])
            rows.append(row)
    Path('outputs/flow_diagnostics').mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv('outputs/flow_diagnostics/stickiness_flow_audit.csv', index=False)


if __name__ == '__main__':
    main()
