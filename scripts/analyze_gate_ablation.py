"""Diagnose whether hierarchical gates help: low-noise validation metrics plus
industry-concentration and capped-Top20 portfolios for each run."""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.long_only import long_only_backtest

CAPS = [2, 3]
EPOCH_METRICS = ['val_rank_ic', 'val_icir', 'val_industry_ic', 'gate_mean']


def epoch_matched(paths, metric):
    """Paired t-test on the shared epoch grid, i.e. every run compared round by round."""
    from scipy import stats
    curves = {}
    for path in paths:
        frame = pd.read_csv(Path(path) / 'training_metrics.csv')
        if metric not in frame.columns or not np.isfinite(frame[metric]).any():
            continue
        curves[Path(path).name] = frame.set_index('epoch')[metric]
    if len(curves) < 2:
        return pd.DataFrame()
    shared = pd.DataFrame(curves).dropna()
    reference = shared.columns[0]
    rows = []
    for name in shared.columns[1:]:
        diff = shared[name] - shared[reference]
        t, p = stats.ttest_rel(shared[name], shared[reference])
        rows.append(dict(metric=metric, run=name, reference=reference, epochs=len(diff),
                         mean_reference=float(shared[reference].mean()), mean_run=float(shared[name].mean()),
                         mean_diff=float(diff.mean()), t=float(t), p=float(p)))
    return pd.DataFrame(rows)


def industries(root):
    frame = pd.read_csv(Path(root) / 'universe.csv')
    return dict(zip(frame.symbol, frame.industry_l1))


def concentration(pred, names, top_n=20):
    rows = []
    for date, group in pred.groupby('date', sort=True):
        group = group[np.isfinite(group.score)]
        if group.empty:
            continue
        top = group.sort_values(['score', 'symbol'], ascending=[False, True]).head(top_n)
        counts = top.symbol.map(names).value_counts()
        rows.append(dict(date=str(date), industries=len(counts), largest=int(counts.iloc[0]),
                         hhi=float((counts / len(top)).pow(2).sum())))
    return pd.DataFrame(rows)


def capped(pred, names, cap, top_n=20):
    """Keep the industry order of the score ranking but allow at most `cap` names per industry."""
    out = []
    for _, group in pred.groupby('date', sort=True):
        group = group.copy()
        seen, keep = {}, []
        ordered = group[np.isfinite(group.score)].sort_values(['score', 'symbol'], ascending=[False, True])
        for symbol in ordered.symbol:
            industry = names[symbol]
            if seen.get(industry, 0) >= cap or len(keep) >= top_n:
                continue
            seen[industry] = seen.get(industry, 0) + 1
            keep.append(symbol)
        group.loc[~group.symbol.isin(keep), 'score'] = np.nan
        out.append(group)
    return pd.concat(out, ignore_index=True)


def summarise(label, curve, metrics, conc, benchmark):
    """Excess is measured against the full cross-section (gross), because
    ``long_only_backtest`` benchmarks against whatever rows it was given."""
    joined = curve.assign(date=curve.date.astype(str)).merge(benchmark, on='date', how='left')
    gross = joined.long_gross_return - joined.universe_return
    net = joined.long_net_return - joined.universe_return
    def tstat(series):
        return float(series.mean() / (series.std(ddof=1) / np.sqrt(len(series))))
    return dict(run=label, net_sharpe=metrics['net']['sharpe'], max_drawdown=metrics['net']['max_drawdown'],
                gross_excess=float(gross.mean()), gross_tstat=tstat(gross),
                net_excess=float(net.mean()), net_tstat=tstat(net),
                cost_drag=float((joined.long_gross_return - joined.long_net_return).mean()),
                turnover=metrics['mean_turnover'], industries_top20=float(conc.industries.mean()),
                largest_industry=float(conc.largest.mean()), worst_industry=int(conc.largest.max()),
                hhi=float(conc.hhi.mean()))


def universe_returns(frame):
    rows = []
    for date, group in frame.groupby('date', sort=True):
        observed = group[np.isfinite(group.realized_return)]
        if observed.empty:
            continue
        rows.append(dict(date=str(date), universe_return=float(np.expm1(observed.realized_return).mean())))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--split', default='validation', choices=['validation', 'test'])
    parser.add_argument('--out', default='outputs/gate_ablation')
    args = parser.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    rows, epoch_rows = [], []
    for run in args.runs:
        path = Path(run)
        names = industries(path)
        frame = pd.read_parquet(path / ('validation_predictions.parquet' if args.split == 'validation' else 'predictions.parquet'))
        frame['date'] = frame['date'].astype(str)
        frame['score'] = frame['score'].astype(float)
        conc = concentration(frame, names)
        benchmark = universe_returns(frame)
        rows.append(summarise(f'{path.name}/top20', *long_only_backtest(frame, 20, 10., 52), conc, benchmark))
        for cap in CAPS:
            subset = capped(frame, names, cap)
            curve, metrics = long_only_backtest(subset, 20, 10., 52)
            rows.append(summarise(f'{path.name}/cap{cap}', curve, metrics, concentration(subset, names), benchmark))
        metrics = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
        training = pd.read_csv(path / 'training_metrics.csv')
        if {'val_icir', 'gate_mean'} <= set(training.columns):
            best = int(metrics['best_epoch'])
            epoch_rows.append(dict(run=path.name, best_epoch=best,
                                   val_ic_at_best=float(training.val_rank_ic[best - 1]),
                                   val_icir_at_best=float(training.val_icir[best - 1]),
                                   val_ic_best_of_ic=float(training.val_icir.max()),
                                   epoch_of_best_icir=int(training.val_icir.idxmax()) + 1,
                                   gate_first=float(training.gate_mean[0]), gate_best=float(training.gate_mean[best - 1]),
                                   gate_last=float(training.gate_mean.iloc[-1])))
    table = pd.DataFrame(rows)
    table.to_csv(root / f'portfolios_{args.split}.csv', index=False)
    print(table.round(4).to_string(index=False))
    matched = pd.concat([frame for frame in (epoch_matched(args.runs, metric) for metric in EPOCH_METRICS) if not frame.empty], ignore_index=True) if len(args.runs) > 1 else pd.DataFrame()
    if not matched.empty:
        matched.to_csv(root / f'epoch_paired_{args.split}.csv', index=False)
        print('\n=== epoch-matched paired tests (first run is the reference) ===')
        print(matched.round(4).to_string(index=False))
    if epoch_rows:
        epochs = pd.DataFrame(epoch_rows)
        epochs.to_csv(root / 'epoch_metrics.csv', index=False)
        print('\n' + epochs.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
