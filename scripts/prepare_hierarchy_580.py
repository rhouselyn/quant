"""Rebuild 580-stock features from original history without overwriting older data."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from quant_mvp.data import FEATURES, aggregate_weekly
from quant_mvp.hierarchy_data import align_observed_features, market_features


def main():
    source = Path('data/sw_580_29x20')
    out = Path('data/sw_580_observed_features')
    if out.exists():
        raise FileExistsError('Existing prepared data must be preserved')
    original = pd.read_parquet(source/'weekly.parquet')
    calendar = pd.DatetimeIndex(sorted(original.date.unique()))
    parts, audits = [], []
    for symbol, old in original.groupby('symbol', sort=True):
        raw_path = next(Path(folder)/(symbol+'.parquet') for folder in
            ['data/sw_file_20/raw', 'data/csi800/raw', 'data/yahoo_500_raw'] if (Path(folder)/(symbol+'.parquet')).exists())
        daily = pd.read_parquet(raw_path)
        daily = daily[daily.date <= calendar[-1]]
        weekly = aggregate_weekly(daily)
        aligned = weekly.set_index('date').reindex(calendar)
        old = old.set_index('date').reindex(calendar)
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            np.testing.assert_allclose(old[col], aligned[col], equal_nan=True, rtol=1e-6, atol=1e-6)
        fixed = old.copy()
        fixed[FEATURES] = align_observed_features(weekly, calendar)
        fixed['observed'] = old.close.notna() & old.volume.gt(0)
        fixed['symbol'] = symbol
        assert fixed.loc[fixed.observed & (fixed.index >= calendar[364]), FEATURES].notna().all().all(), symbol
        np.testing.assert_allclose(fixed.target, old.target, equal_nan=True)
        parts.append(fixed.reset_index().rename(columns={'index':'date'}))
        audits.append(dict(symbol=symbol, raw_path=str(raw_path), history_start=str(weekly.date.min().date()),
            previous_validation_missing_fraction=float(old.iloc[364:442][FEATURES].isna().to_numpy().mean()),
            new_validation_missing_fraction=float(fixed.iloc[364:442][FEATURES].isna().to_numpy().mean())))
    panel = pd.concat(parts, ignore_index=True)
    assert panel.shape[0] == 580*520 and panel.symbol.nunique() == 580
    out.mkdir(parents=True)
    panel.to_parquet(out/'weekly.parquet', index=False)
    market_features(panel).rename_axis('date').reset_index().to_csv(out/'market_features.csv', index=False)
    shutil.copy2(source/'universe.csv', out/'universe.csv')
    pd.DataFrame(audits).to_csv(out/'feature_repair_audit.csv', index=False)
    manifest = dict(source=str(source), source_sha256=hashlib.sha256((source/'weekly.parquet').read_bytes()).hexdigest(),
        n_stocks=580, n_industries=29, n_weeks=520,
        feature_policy='Compute on observed weekly bars with pre-2016 history; count observed weeks; forward-carry past features onto calendar gaps. Never fill raw bars or labels.',
        targets='Unchanged next-calendar-week close-to-close log returns; closed-week labels remain missing.',
        market_policy='Six contemporaneous aggregate market features; training-prefix scaling only.',
        history_bias='Current industry classification and survivor sample; historical sources unchanged')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    cfg=yaml.safe_load(Path('configs/real_sw_580_full_rankic_10ep.yaml').read_text())
    cfg.update(model_type='hierarchical_market', cache_path=(out/'weekly.parquet').as_posix(),
        universe_path=(out/'universe.csv').as_posix(), market_path=(out/'market_features.csv').as_posix(),
        prepare_command='python scripts/prepare_hierarchy_580.py', temporal_weight=.1,
        temporal_horizon=2, temporal_horizons=[2], attention_heads=4, global_layers=2,
        market_tokens=1, market_features=6, relation_gate_init=.1, dropout=.1,
        encoder_chunk_size=580, eval_batch_dates=4, min_history_weeks=32,
        common_training_horizon=5, torch_num_threads=4)
    Path('configs/hierarchical_580_h2.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    for name,weight,horizon in [('off',0.,2),('h5',.1,5)]:
        variant=dict(cfg,temporal_weight=weight,temporal_horizon=horizon,temporal_horizons=[horizon])
        Path(f'configs/hierarchical_580_{name}.yaml').write_text(yaml.safe_dump(variant,sort_keys=False),encoding='utf-8')
    print('580 stocks repaired. Validation feature missing fraction:',panel[panel.date.isin(calendar[364:442])][FEATURES].isna().to_numpy().mean())


if __name__=='__main__':main()
