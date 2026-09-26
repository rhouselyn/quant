"""Prepare a separate, reproducible CITIC 30 x 20 stock weekly cache."""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.data import _yahoo_one, aggregate_weekly, add_features
from import_citic_industries import CITIC_L1


def select_universe(mapping, seed=42):
    """Input must describe current members, one row per stock, not history."""
    m = mapping.rename(columns={'ts_code': 'code', 'l1_name': 'citic_l1'}).copy()
    if not {'code', 'citic_l1'}.issubset(m.columns):
        raise ValueError('Required columns: code,citic_l1 (current members only)')
    m['code'] = m.code.str.split('.').str[0].str.zfill(6)
    m = m[m.code.str.match(r'^(000|001|002|003|300|301|600|601|603|605|688|689)\d{3}$')].copy()
    if m.code.duplicated().any():
        raise ValueError('Duplicate stock codes: supply one current industry per stock')
    if not m.citic_l1.isin(CITIC_L1).all():
        raise ValueError('Unknown CITIC level-one labels; verify taxonomy')
    counts = m.citic_l1.value_counts().reindex(CITIC_L1, fill_value=0)
    if (counts < 20).any():
        raise ValueError('Cannot sample 20 unique stocks per industry: ' +
                         counts[counts < 20].to_json(force_ascii=False))
    chosen = pd.concat([m[m.citic_l1.eq(ind)].sort_values('code').sample(n=20, random_state=seed)
                        for ind in CITIC_L1], ignore_index=True)
    chosen['symbol'] = chosen.code + chosen.code.map(lambda c: '.SS' if c.startswith('6') else '.SZ')
    chosen['citic_status'] = 'sourced'
    return chosen, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mapping', required=True, help='Current full-market code,citic_l1 CSV')
    ap.add_argument('--source', required=True)
    ap.add_argument('--asof', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='data/citic30_20')
    args = ap.parse_args()
    pd.Timestamp(args.asof)
    source = Path(args.mapping)
    selected, counts = select_universe(pd.read_csv(source, dtype=str), args.seed)
    root = Path(args.output)
    cache = root / 'weekly_600_520.parquet'
    if cache.exists():
        raise FileExistsError('Completed cache exists; use a new --output to preserve it')
    root.mkdir(parents=True, exist_ok=True)
    selected['citic_source'] = args.source
    selected['citic_asof'] = args.asof
    selected.to_csv(root / 'universe.csv', index=False, encoding='utf-8-sig')
    counts.rename('available_stocks').to_csv(root / 'industry_counts.csv')
    selected.groupby('citic_l1').size().rename('sampled_stocks').to_csv(root / 'sample_counts.csv')
    (root / 'manifest.json').write_text(json.dumps(dict(
        source=args.source, asof=[args.asof], classification='CITIC level 1',
        seed=args.seed, sampling='20 without replacement per industry; Shanghai/Shenzhen A shares',
        mapping_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        calendar_start='2016-09-30', calendar_end='2026-09-11',
        historical_membership=False), ensure_ascii=False, indent=2), encoding='utf-8')
    raw = root / 'raw'
    raw.mkdir(exist_ok=True)
    calendar = pd.date_range(end='2026-09-11', periods=520, freq='W-FRI')

    def fetch(symbol):
        for folder in [raw, Path('data/csi800/raw'), Path('data/yahoo_500_raw')]:
            path = folder / f'{symbol}.parquet'
            if path.exists():
                d = pd.read_parquet(path)
                if not d.empty and pd.to_datetime(d.date).max() >= calendar[-1]:
                    return d
        for _ in range(3):
            d = _yahoo_one(symbol, '2014-01-01', '2026-09-12')
            if d is not None and not d.empty:
                d.to_parquet(raw / f'{symbol}.parquet', index=False)
                return d
        return None

    frames, coverage, missing = [], [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (symbol, d) in enumerate(zip(selected.symbol, pool.map(fetch, selected.symbol)), 1):
            if d is None or d.empty:
                missing.append(symbol)
            else:
                w = aggregate_weekly(d)
                w = w[w.date.isin(calendar)]
                if w.empty:
                    missing.append(symbol)
                else:
                    frames.append(w)
                    coverage.append(dict(symbol=symbol, observed_weeks=len(w)))
            if i % 50 == 0:
                print(f'History {i}/600; missing={len(missing)}', flush=True)
    pd.DataFrame(coverage).to_csv(root / 'coverage.csv', index=False)
    if missing:
        raise RuntimeError(f'No replacement or fabricated prices; missing: {missing}')
    grid = pd.MultiIndex.from_product([sorted(selected.symbol), calendar], names=['symbol', 'date'])
    panel = pd.concat(frames).set_index(['symbol', 'date']).reindex(grid).reset_index()
    df = add_features(panel)
    df['frequency'] = 'weekly'
    df.to_parquet(cache, index=False)
    print(f'Saved {cache}: 30 industries x 20 unique stocks / 520 weeks')


if __name__ == '__main__':
    main()
