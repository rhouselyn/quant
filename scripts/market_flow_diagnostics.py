"""Zero-training diagnostics for the Fokker-Planck reading of the cross-section.

State variable: rho, the weekly Shenwan-industry share of traded value, which sums
to one by construction, so mass conservation is an identity rather than an
assumption. Nothing here is fitted into the ranker; the point is to find out which
term of  dp/dt = -div(f p) + 1/2 div-div(D p)  is actually estimable before any
architecture is built.

  1  persistence of rho and of its increment
  2  first Kramers-Moyal moment: is the drift f predictable out of sample
  3  crowding kernel: does inflow depress the NEXT week's industry excess
  4  second moment: is the size of the next move predictable (that is D)
  5  non-equilibrium current: lead-lag antisymmetry of the transition matrix
  6  cash-out: inverse-volatility sizing and a no-trade band on the best arm
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

DATA = Path('data/sw_580_observed_features')
TRAIN_END, VALID_END = 364, 442
COST_BPS, TOP_N = 10.0, 20
CUTS = {'train': (0, TRAIN_END), 'valid': (TRAIN_END, VALID_END), 'test': (VALID_END, None)}
rng = np.random.default_rng(42)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def panel():
    frame = pd.read_parquet(DATA / 'weekly.parquet').copy()
    universe = pd.read_csv(DATA / 'universe.csv')
    frame = frame.merge(universe[['symbol', 'industry_id']], on='symbol', how='left')
    frame = frame[frame.observed == 1]
    grouped = frame.groupby(['date', 'industry_id'])
    amount = grouped.amount.sum().unstack()
    ret = grouped.target.mean().unstack()
    excess = ret.sub(frame.groupby('date').target.mean(), axis=0)
    amount = amount.reindex(columns=range(29)).fillna(0.0)
    rho = amount.div(amount.sum(axis=1), axis=0)
    return rho, excess.reindex(columns=range(29)).fillna(0.0)


def slices(frame):
    return {name: frame.iloc[a:b] for name, (a, b) in CUTS.items()}


def cross_ic(predictor, target):
    """Per-week cross-industry Spearman IC between aligned panels."""
    p, t = predictor.stack(), target.stack()
    both = pd.concat([p.rename('p'), t.rename('t')], axis=1, join='inner').dropna()
    rows = []
    for _, group in both.groupby(level=0):
        x, y = group.p.to_numpy(), group.t.to_numpy()
        if len(x) < 8 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        rows.append(stats.spearmanr(x, y).statistic)
    return pd.Series(rows, dtype=float)


def pooled_ols(predictor, target):
    """Week-demeaned OLS of aligned panels with a week-clustered standard error."""
    p, t = predictor.stack(), target.stack()
    both = pd.concat([p.rename('p'), t.rename('y')], axis=1, join='inner').dropna()
    both = both.groupby(level=0).transform(lambda s: s - s.mean())
    index = both.index.get_level_values(0)
    x, y = both.p.to_numpy(), both.y.to_numpy()
    beta = float(x @ y / (x @ x))
    score = pd.DataFrame({'s': (x * (y - beta * x)) ** 2}, index=index).groupby(level=0).sum().to_numpy().ravel()
    se = float(np.sqrt(score.sum()) / (x @ x))
    return beta, (beta / se if se else np.nan)


def report(label, predictor, target, frames):
    row = dict(check=label)
    for name, (p, t) in frames.items():
        series = cross_ic(p, t)
        row[f'ic_{name}'] = float(series.mean()) if len(series) else np.nan
        row[f't_{name}'] = float(series.mean() / (series.std(ddof=1) / np.sqrt(len(series)))) if len(series) > 2 else np.nan
    beta, t = pooled_ols(predictor, target)
    row['beta_full'], row['t_full'] = beta, t
    return row


def main():
    rho, excess = panel()
    delta = rho.diff()
    dev = rho.sub(rho.mean(axis=1), axis=0)
    adelta, aexcess = delta.abs(), excess.abs()
    vol = delta.rolling(20).std()
    print(f'weeks={len(rho)}  industries={rho.shape[1]}  mean rho={rho.mean().mean():.4f} '
          f'(1/29={1/29:.4f})  sd(rho)={rho.std().mean():.4f}')

    def frame(pair):
        p, t = pair
        return {name: (p.iloc[a:b], t.iloc[a:b]) for name, (a, b) in CUTS.items()}

    checks = [
        ('rho(t) -> rho(t+1)            persistence', rho, rho.shift(-1)),
        ('rho dev(t) -> Delta rho(t+1)  mean reversion', dev, delta.shift(-1)),
        ('Delta rho(t) -> Delta rho(t+1) current inertia', delta, delta.shift(-1)),
        ('excess(t) -> Delta rho(t+1)   capital chases return', excess, delta.shift(-1)),
        ('Delta rho(t) -> excess(t+1)   CROWDING kernel', delta, excess.shift(-1)),
        ('rho dev(t) -> excess(t+1)     level crowding', dev, excess.shift(-1)),
        ('|Delta rho|(t) -> |Delta rho|(t+1) D term', adelta, adelta.shift(-1)),
        ('vol20(t) -> |Delta rho|(t+1)  D term', vol, adelta.shift(-1)),
        ('vol20(t) -> |excess|(t+1)     D term', vol, aexcess.shift(-1)),
        ('vol20(t) -> excess(t+1)       risk-returned', vol, excess.shift(-1)),
    ]
    print('\n=== Kramers-Moyal: cross-sectional rank IC by window (beta = week-demeaned OLS) ===')
    table = pd.DataFrame([report(label, p, t, frame((p, t))) for label, p, t in checks])
    print(table.round(4).to_string(index=False))

    # ---- non-equilibrium current ---------------------------------------------
    matrix = rho.diff().to_numpy(dtype=float)
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    matrix = (matrix - matrix.mean(0)) / matrix.std(0)
    cov = np.cov(matrix.T)
    values, vectors = np.linalg.eigh(cov)
    whiten = vectors @ np.diag(1 / np.sqrt(np.maximum(values, 1e-9))) @ vectors.T
    white = matrix @ whiten

    def current(series, lag=1):
        c = series[:-lag].T @ series[lag:] / (len(series) - lag)
        return 0.5 * (c - c.T)

    observed = float(np.linalg.norm(current(white)))
    # Null: a reversible process has a symmetric whitened lagged covariance, so the
    # test has to break the time pairing while keeping every week's cross section.
    null = np.array([np.linalg.norm(current(white, lag=int(k)))
                     for k in rng.choice(np.arange(5, 105), size=2000)])
    names = pd.read_csv(DATA / 'universe.csv').drop_duplicates('industry_id') \
        .set_index('industry_id').industry_l1.to_dict()
    skew = current(white)
    ranked = sorted(((abs(skew[i, j]), i, j) for i in range(29) for j in range(i + 1, 29)), reverse=True)
    print(f'\n=== lead-lag current on whitened Delta rho, {len(matrix)} weeks ===')
    print(f'||skew(C)|| = {observed:.4f}   random-lag null mean {null.mean():.4f} '
          f'p95 {np.percentile(null,95):.4f}   p = {(null >= observed).mean():.4f}   '
          f'||sym(C)|| = {np.linalg.norm(0.5*(white[:-1].T@white[1:]/(len(white)-1)+ (white[:-1].T@white[1:]/(len(white)-1)).T)):.4f}')
    symmetric = 0.5 * (white[:-1].T @ white[1:] + (white[:-1].T @ white[1:]).T) / (len(white) - 1)
    print(f'whitened lag-1 sym eigenvalues: min {np.linalg.eigvalsh(symmetric).min():.3f} '
          f'max {np.linalg.eigvalsh(symmetric).max():.3f}')
    for value, i, j in ranked[:6]:
        print(f'  {names.get(i,i)} -> {names.get(j,j)}: skew {skew[i,j]:+.4f}')

    # ---- cash-out ------------------------------------------------------------
    run = Path('outputs/real_hierarchical_580_h5_15ep_gatefrozen0_lambda1')
    weekly = pd.read_parquet(DATA / 'weekly.parquet')
    weekly = weekly[weekly.observed == 1].sort_values(['symbol', 'date'])
    weekly['vol20'] = weekly.groupby('symbol').target.transform(lambda s: s.rolling(20).std())
    lookup = weekly.set_index(['date', 'symbol']).vol20
    rows, series = [], {}
    for split, file in [('validation', 'validation_predictions.parquet'), ('test', 'predictions.parquet')]:
        pred = pd.read_parquet(run / file)
        pred['vol20'] = lookup.reindex(pd.MultiIndex.from_arrays([pred.date, pred.symbol])).to_numpy()
        for band in (0.0, 0.125, 0.25, 0.5, 1.0):
            for mode in ('equal', 'invvol'):
                result = trade(pred, band, mode)
                series[(split, band, mode)] = result.pop('excess')
                rows.append(dict(split=split, **result))
    print(f'\n=== sizing and no-trade band on {run.name} ===')
    result = pd.DataFrame(rows)
    print(result.round(4).to_string(index=False))
    print('\n=== paired against the traded book (band=0, equal weight) ===')
    for split in ('validation', 'test'):
        for band, mode in [(0.125, 'equal'), (0.25, 'equal'), (0.5, 'equal'), (0.25, 'invvol')]:
            a, b = series[(split, band, mode)], series[(split, 0.0, 'equal')]
            both = pd.concat([a.rename('a'), b.rename('b')], axis=1, join='inner').dropna()
            diff = both.a - both.b
            n = len(diff)
            t = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
            print(f'  {split:10s} band={band:<5} {mode:6s} +{diff.mean()*1e4:6.2f} bps/wk  paired t={t:+.2f}')
    Path('outputs/flow_diagnostics').mkdir(exist_ok=True)
    table.to_csv('outputs/flow_diagnostics/kramers_moyal.csv', index=False)
    result.to_csv('outputs/flow_diagnostics/sizing_band.csv', index=False)


def trade(pred, band, mode):
    """Keep the book unless a challenger beats the weakest holding by band x weekly score sd."""
    held, previous = [], {}
    gross, net, turns, bases, kept = [], [], [], [], []
    for date, group in pred.groupby('date', sort=True):
        group = group[np.isfinite(group.score)]
        if not np.isfinite(group.realized_return).any():
            continue
        inside = group.set_index('symbol').score.to_dict()
        vol = group.set_index('symbol').vol20.to_dict()
        returns = dict(zip(group.symbol, np.expm1(group.realized_return)))
        held = [s for s in held if s in inside]
        sd = float(group.score.std())
        spare = sorted((s for s in inside if s not in held), key=lambda s: -inside[s])
        while len(held) < TOP_N and spare:
            held.append(spare.pop(0))
        while True:
            worst = min(held, key=lambda s: inside[s])
            out = [s for s in inside if s not in held]
            if not out:
                break
            best = max(out, key=lambda s: inside[s])
            if inside[best] - inside[worst] <= band * sd:
                break
            held.remove(worst)
            held.append(best)
        names = held[:TOP_N]
        if mode == 'invvol':
            raw = np.array([1.0 / vol[s] if np.isfinite(vol.get(s, np.nan)) and vol[s] > 1e-6 else 0.0 for s in names])
            raw = raw if raw.sum() > 0 else np.ones(len(names))
            weights = raw / raw.sum()
        else:
            weights = np.repeat(1 / len(names), len(names))
        target = dict(zip(names, weights))
        turnover = (sum(max(target.get(s, 0) - previous.get(s, 0), 0) for s in set(target) | set(previous))
                    + sum(max(previous.get(s, 0) - target.get(s, 0), 0) for s in set(target) | set(previous)))
        g = sum(w * returns.get(s, 0) for s, w in target.items())
        n = (1 - turnover * COST_BPS / 10000) * (1 + g) - 1
        gross.append(g)
        net.append(n)
        turns.append(turnover)
        bases.append(float(np.nanmean(np.expm1(group.realized_return))))
        kept.append(date)
        previous = {s: w * (1 + returns.get(s, 0)) / (1 + g) for s, w in target.items()}
    base = np.array(bases)
    excess = np.array(net) - base
    spread = np.array(gross) - base
    return dict(band=band, sizing=mode, weeks=len(net), gross_bps=float(spread.mean() * 1e4),
                net_bps=float(excess.mean() * 1e4),
                t=float(excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess)))),
                cost_bps=float((np.array(gross) - np.array(net)).mean() * 1e4),
                turnover=float(np.mean(turns)),
                excess=pd.Series(excess, index=[str(k) for k in kept]),
                sharpe=float(np.mean(net) / np.std(net, ddof=1) * np.sqrt(52)))


if __name__ == '__main__':
    main()
