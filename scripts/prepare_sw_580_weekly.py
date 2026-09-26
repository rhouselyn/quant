"""Keep the original 20-stock samples in all complete Shenwan industries.

Only removes the two incomplete groups; never resamples or changes old caches.
Feature scaling is recalculated by dataset_arrays on the resulting 580 stocks.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = Path('data/sw_file_20')
    output = Path('data/sw_580_29x20')
    if output.exists():
        raise FileExistsError('Prepared subset already exists; preserve the existing cache')
    universe = pd.read_csv(source/'universe.csv', dtype=str)
    assert universe.symbol.nunique() == len(universe) == 600
    assert universe.industry_l1_code.notna().all()
    counts = universe.groupby(['industry_l1_code', 'industry_l1']).size().rename('n_stocks').reset_index()
    removed = counts[counts.n_stocks.ne(20)]
    assert dict(zip(removed.industry_l1_code, removed.n_stocks)) == {'51': 14, '77': 6}
    retained = counts[counts.n_stocks.eq(20)].sort_values('industry_l1_code').copy()
    retained['industry_id'] = range(len(retained))
    selected = universe[universe.industry_l1_code.isin(retained.industry_l1_code)].copy()
    selected = selected.merge(retained[['industry_l1_code', 'industry_id']], on='industry_l1_code', validate='many_to_one')
    selected = selected.sort_values(['industry_id', 'symbol']).reset_index(drop=True)
    assert len(selected) == 580 and len(retained) == 29
    assert selected.groupby('industry_id').size().eq(20).all()
    original = pd.read_parquet(source/'weekly.parquet')
    panel = original[original.symbol.isin(selected.symbol)].copy()
    assert set(original.symbol) == set(universe.symbol)
    assert not panel.duplicated(['date', 'symbol']).any()
    assert panel.date.nunique() == 520 and len(panel) == 580*520
    assert panel.groupby('date').symbol.nunique().eq(580).all()
    assert set(panel.date) == set(original.date)
    cfg = yaml.safe_load(Path('configs/real_weekly_500_full_rankic_10ep.yaml').read_text())
    cfg.update(n_stocks=580, cache_path=(output/'weekly.parquet').as_posix(),
        universe_path=(output/'universe.csv').as_posix(),
        universe_name='Shenwan 29 industries x original 20 random eligible stocks',
        industry_classification='Shenwan level 1 (current snapshot, retrospective)',
        prepare_command='python scripts/prepare_sw_580_weekly.py', encoder_chunk_size=1000)
    source_manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    manifest = dict(source_dataset=source.as_posix(),
        source_weekly_sha256=sha(source/'weekly.parquet'),
        source_universe_sha256=sha(source/'universe.csv'),
        source_manifest_sha256=sha(source/'manifest.json'),
        source_manifest=source_manifest,
        removed_industries=removed.to_dict('records'),
        n_stocks=580, n_industries=29, stocks_per_industry=20,
        sampling='Keep original seed42 samples; drop entire groups with counts other than 20; no redraw',
        calendar_start=str(panel.date.min().date()), calendar_end=str(panel.date.max().date()),
        n_weeks=520, feature_policy='Preserve raw weekly bars, per-stock features and targets; normalize on 580 stocks at training',
        industry_id_order='Ascending industry_l1_code; join by symbol, never assume cache order',
        history_bias=source_manifest['history_bias'], asof=source_manifest['asof'],
        model_status='Dataset and single-head baseline config only; hierarchical attention not implemented or trained')
    output.mkdir(parents=True)
    panel.to_parquet(output/'weekly.parquet', index=False)
    selected.to_csv(output/'universe.csv', index=False, encoding='utf-8-sig')
    retained.to_csv(output/'industry_counts.csv', index=False, encoding='utf-8-sig')
    universe[~universe.symbol.isin(selected.symbol)].to_csv(output/'excluded_universe.csv', index=False, encoding='utf-8-sig')
    manifest['weekly_sha256'] = sha(output/'weekly.parquet')
    manifest['universe_sha256'] = sha(output/'universe.csv')
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    Path('configs/real_sw_580_full_rankic_10ep.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ['n_stocks', 'n_industries', 'removed_industries', 'calendar_start', 'calendar_end']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
