"""Build an audited Yahoo universe for the 500-stock weekly experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import re
import json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from quant_mvp.data import _default_cn_symbols, _yahoo_one, aggregate_weekly, add_features

def main():
    root = Path('data/yahoo_500_raw')
    root.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_parquet('data/real_100_520_weekly_yahoo.parquet')['date'].drop_duplicates().sort_values()
    first, last = calendar.iloc[0], calendar.iloc[-1]
    frames, audit = [], []
    def fetch(symbol):
        path = root / (symbol + '.parquet')
        if path.exists():
            return pd.read_parquet(path)
        frame = _yahoo_one(symbol, '2014-01-01', '2026-09-12')
        if frame is not None:
            frame['stock_name'] = frame.attrs.get('name', '')
            frame['instrument_type'] = frame.attrs.get('instrument_type', '')
            frame.to_parquet(path, index=False)
        return frame
    candidates = _default_cn_symbols(2000, 42)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for offset in range(0, len(candidates), 128):
            batch = candidates[offset:offset+128]
            for symbol, daily in zip(batch, pool.map(fetch, batch)):
                if daily is None or daily.empty:
                    continue
                name = str(daily.stock_name.iloc[0])
                reason = ''
                if not name or re.search(r'(^|\W)\*?ST|退|DELIST', name, re.I):
                    reason = 'missing name or ST/delisting name'
                elif daily.instrument_type.iloc[0] != 'EQUITY':
                    reason = 'not equity'
                elif daily.date.min() > first or daily.loc[daily.volume > 0, 'date'].max() < last-pd.Timedelta(days=7):
                    reason = 'insufficient history or stale trading'
                weekly = aggregate_weekly(daily)
                weekly['stock_name'] = name
                weekly = weekly[weekly.date.isin(calendar)]
                if len(weekly) < 494 or (weekly.volume <= 0).mean() > .05:
                    reason = reason or 'incomplete weekly history or excessive inactive weeks'
                audit.append(dict(symbol=symbol,name=name,accepted=not reason,reason=reason))
                if not reason and len(frames) < 500:
                    frames.append(weekly)
            print(f'Checked {min(offset+128,len(candidates))}; eligible selected {len(frames)}/500', flush=True)
            if len(frames) == 500:
                break
    Path('data/real_500_universe_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    if len(frames) != 500:
        raise RuntimeError(f'Only {len(frames)} eligible stocks; no synthetic fallback')
    panel = pd.concat(frames, ignore_index=True)
    df = add_features(panel)
    # Retain the baseline calendar, including exchange-wide holiday weeks.
    # Missing bars stay missing; do not invent OHLC or zero-return labels.
    grid = pd.MultiIndex.from_product([sorted(panel.symbol.unique()), calendar], names=['symbol','date'])
    df = df.set_index(['symbol','date']).reindex(grid).reset_index()
    next_close = df.groupby('symbol')['close'].shift(-1)
    import numpy as np
    df['target'] = np.log(next_close / df['close'])
    df['frequency'] = 'weekly'
    df.to_parquet('data/real_500_520_weekly_yahoo.parquet',index=False)
    panel[['symbol','stock_name']].drop_duplicates().to_csv('data/real_500_universe.csv',index=False)
    print('Saved 500 stocks / 520 weeks',flush=True)

if __name__ == '__main__':
    main()
