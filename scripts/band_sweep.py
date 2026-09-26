"""Does the no-trade band survive when its width is set by the measured diffusion term?

Three eviction rules, replayed on existing predictions (no training):
  global  swap if a challenger beats the weakest holding by band x weekly score sd
  vol     same, scaled by (sigma_i / median sigma)^p, p = the diffusion exponent
  rank    keep a holding until it falls out of the weekly top K (buffer index)

Selection is always made on validation, then read on test.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path('data/sw_580_observed_features')
COST_BPS, TOP_N = 10.0, 20
RUNS = ['real_hierarchical_580_h5_15ep_gatefrozen0_lambda1',
        'real_hierarchical_580_h5_15ep_gatefrozen0',
        'real_hierarchical_580_h5_15ep',
        'real_hierarchical_580_h5_15ep_lambda1_stick0p3',
        'real_hierarchical_580_h5_15ep_lambda1_stick1',
        'real_hierarchical_580_h5_15ep_lambda1_stick1_cap',
        'real_hierarchical_580_h5_15ep_lambda1_stick3_cap_zone',
        'real_hierarchical_580_h5_15ep_gatefrozen0_lambda1_head20',
        'real_hierarchical_580_h5_15ep_gatefrozen0_lambda1_head50',
        'real_hierarchical_580_h5_15ep_lambda1_s43',
        'real_hierarchical_580_h5_15ep_lambda1_s44']
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def rules():
    out = [('global', dict(band=b, p=0.0)) for b in (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0)]
    out += [('vol', dict(band=b, p=p)) for b in (0.125, 0.25, 0.375, 0.5) for p in (0.5, 2 / 3, 1.0)]
    out += [('rank', dict(buffer=k)) for k in (24, 28, 34, 42, 60)]
    return out


def label(kind, rule):
    if kind == 'rank':
        return f'rank(K={rule["buffer"]})'
    return f'{kind}({rule["band"]:g}' + (f',p={rule["p"]:.2f}' if rule['p'] else '') + ')'


def weeks_of(pred):
    """Slice the prediction frame once so each rule only pays for the portfolio logic."""
    out = []
    for date, group in pred.groupby('date', sort=True):
        group = group[np.isfinite(group.score)]
        if not np.isfinite(group.realized_return).any():
            continue
        simple = np.expm1(group.realized_return)
        score = dict(zip(group.symbol, group.score))
        vol = dict(zip(group.symbol, group.vol20))
        # vol20 is NaN for each symbol's first 19 weeks; a NaN width would make
        # every gap look tradable and the swap loop would ping-pong forever.
        observed_vol = [v for v in group.vol20 if np.isfinite(v)]
        out.append(dict(date=str(date),
                        score=score,
                        vol=vol,
                        returns=dict(zip(group.symbol, simple)),
                        order=sorted(score, key=lambda s: -score[s]),
                        sd=float(group.score.std()),
                        floor=float(np.median(observed_vol)) if observed_vol else np.nan,
                        base=float(np.nanmean(simple))))
    return out


def replay(weeks, kind, rule):
    """Weekly Top20 equal-weight replay with the given eviction rule."""
    held, previous = [], {}
    gross, net, turns, bases, days = [], [], [], [], []
    for week in weeks:
        inside, vol, returns = week['score'], week['vol'], week['returns']
        order, sd, floor = week['order'], week['sd'], week['floor']
        held = [s for s in held if s in inside]
        if kind == 'rank':
            keep = [s for s in held if s in order[:rule['buffer']]]
            for s in order:
                if len(keep) >= TOP_N:
                    break
                if s not in keep:
                    keep.append(s)
            held = keep
        else:
            spare = [s for s in order if s not in held]
            while len(held) < TOP_N and spare:
                held.append(spare.pop(0))
            while True:
                out = [s for s in order if s not in held]
                if not out:
                    break
                best, worst = out[0], min(held, key=lambda s: inside[s])
                width = rule['band'] * sd
                if kind == 'vol':
                    sigma = vol.get(worst, np.nan)
                    if np.isfinite(sigma) and np.isfinite(floor) and floor > 0:
                        width *= (max(sigma, 1e-9) / floor) ** rule['p']
                if inside[best] - inside[worst] <= width:
                    break
                held.remove(worst)
                held.append(best)
        names = [s for s in held if s in inside][:TOP_N]
        weights = dict(zip(names, np.repeat(1 / len(names), len(names))))
        turnover = (sum(max(weights.get(s, 0) - previous.get(s, 0), 0) for s in set(weights) | set(previous))
                    + sum(max(previous.get(s, 0) - weights.get(s, 0), 0) for s in set(weights) | set(previous)))
        g = sum(w * returns.get(s, 0) for s, w in weights.items())
        net.append((1 - turnover * COST_BPS / 10000) * (1 + g) - 1)
        gross.append(g)
        turns.append(turnover)
        if not np.isfinite(week['base']):
            continue
        bases.append(week['base'])
        days.append(week['date'])
        previous = {s: w * (1 + returns.get(s, 0)) / (1 + g) for s, w in weights.items()}
    base = np.array(bases)
    excess = pd.Series(np.array(net) - base, index=days)
    spread = np.array(gross) - base
    row = dict(gross_bps=float(spread.mean() * 1e4), net_bps=float(excess.mean() * 1e4),
               t=float(excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess)))),
               cost_bps=float((np.array(gross) - np.array(net)).mean() * 1e4),
               turnover=float(np.mean(turns)), weeks=len(net),
               sharpe=float(np.mean(net) / np.std(net, ddof=1) * np.sqrt(52)))
    return excess, row


def main():
    weekly = pd.read_parquet(DATA / 'weekly.parquet')
    weekly = weekly[weekly.observed == 1].sort_values(['symbol', 'date'])
    weekly['vol20'] = weekly.groupby('symbol').target.transform(lambda s: s.rolling(20).std())
    lookup = weekly.set_index(['date', 'symbol']).vol20
    series, summary = {}, []
    for run in RUNS:
        short = run.replace('real_hierarchical_580_h5_15ep', 'arm').strip('_') or 'baseline'
        if not (Path('outputs') / run / 'predictions.parquet').exists():
            print(f'skipping {short}: no test predictions yet', flush=True)
            continue
        for split, file in [('validation', 'validation_predictions.parquet'), ('test', 'predictions.parquet')]:
            pred = pd.read_parquet(Path('outputs') / run / file)
            pred['vol20'] = lookup.reindex(pd.MultiIndex.from_arrays([pred.date, pred.symbol])).to_numpy()
            weeks = weeks_of(pred)
            for kind, rule in rules():
                excess, row = replay(weeks, kind, rule)
                key = label(kind, rule)
                series[(short, split, key)] = excess
                row.update(run=short, split=split, rule=key)
                summary.append(row)
                print(f'{short}/{split}: {key} net={row["net_bps"]:6.2f}bps t={row["t"]:+.2f} '
                      f'to={row["turnover"]:.3f}', flush=True)
    table = pd.DataFrame(summary)
    Path('outputs/flow_diagnostics').mkdir(exist_ok=True)
    table.to_csv('outputs/flow_diagnostics/band_sweep.csv', index=False)

    for run in table.run.unique():
        sub = table[table.run == run]
        wide = pd.concat({s: g.set_index('rule')[['net_bps', 't', 'turnover', 'gross_bps']]
                          for s, g in sub.groupby('split')}, axis=1).sort_index(axis=1, level=1)
        print(f'\n=== {run} ===')
        print(wide.round(2).to_string())
        pick = sub[sub.split == 'validation'].sort_values('net_bps').rule.iloc[-1]
        plain = sub[(sub.split == 'validation') & (sub.rule == 'global(0)')]
        print(f'  best on validation: {pick}   (no band: {plain.net_bps.iloc[0]:.1f}bps, t={plain.t.iloc[0]:.2f})')
        for rival in ('global(0)', 'global(0.25)'):
            a = series[(run, 'test', pick)]
            b = series[(run, 'test', rival)]
            both = pd.concat([a.rename('a'), b.rename('b')], axis=1, join='inner').dropna()
            diff = both.a - both.b
            print(f'  test {pick} - {rival}: {diff.mean()*1e4:+6.2f} bps/wk  paired t='
                  f'{diff.mean()/(diff.std(ddof=1)/np.sqrt(len(diff))):+.2f}')


if __name__ == '__main__':
    main()
