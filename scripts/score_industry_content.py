"""How much of each run's weekly score is an industry bet?

Reports eta squared (share of cross-sectional score variance carried by industry
means) against the neutral-score expectation for the same weeks, plus the
largest-industry count in the Top20.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.hierarchy import industry_exposure_baseline


def weekly_content(frame, ids, top_n=20):
    values, nulls, largest, industries = [], [], [], []
    for date, group in frame.groupby('date', sort=True):
        group = group[np.isfinite(group.score)]
        if len(group) < 3:
            continue
        score = group.score.to_numpy(dtype=float)
        std = score.std()
        if std <= 1e-9:
            continue
        group_ids = group.symbol.map(ids).to_numpy()
        if pd.isna(group_ids).any():
            continue
        z = (score - score.mean()) / std
        counts = np.bincount(group_ids)
        means = np.bincount(group_ids, weights=z) / np.maximum(counts, 1)
        values.append(float((counts * means ** 2).sum() / len(z)))
        null = industry_exposure_baseline(counts)
        if null is not None:
            nulls.append(null)
        top = group.sort_values(['score', 'symbol'], ascending=[False, True]).head(top_n)
        counts_top = top.symbol.map(ids).value_counts()
        largest.append(int(counts_top.iloc[0]))
        industries.append(len(counts_top))
    if not values:
        return {}
    return dict(weeks=len(values), eta2=float(np.mean(values)), eta2_p90=float(np.percentile(values, 90)),
                eta2_max=float(max(values)), neutral=float(np.mean(nulls)),
                excess_ratio=float(np.mean(values) / np.mean(nulls)),
                worst_industry_top20=int(max(largest)), industries_top20=float(np.mean(industries)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--split', default='validation', choices=['validation', 'test'])
    parser.add_argument('--out', default='outputs/industry_content')
    args = parser.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in args.runs:
        path = Path(run)
        universe = pd.read_csv(path / 'universe.csv')
        ids = dict(zip(universe.symbol, universe.industry_id))
        file = 'validation_predictions.parquet' if args.split == 'validation' else 'predictions.parquet'
        frame = pd.read_parquet(path / file)
        frame['score'] = frame['score'].astype(float)
        rows.append(dict(run=path.name, **weekly_content(frame, ids)))
    table = pd.DataFrame(rows)
    table.to_csv(root / f'score_industry_content_{args.split}.csv', index=False)
    print(table.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
