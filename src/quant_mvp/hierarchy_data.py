"""Causal inputs for a date-grouped industry model. No label-derived input masks."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import FEATURES, add_features

MARKET_FEATURES = ['market_ret1', 'market_ret4', 'market_ret12', 'market_vol20', 'breadth', 'activity']


def align_observed_features(observed_weekly, calendar):
    """Rolling windows count observed weekly bars; closed weeks carry past features only."""
    observed = observed_weekly.dropna(subset=['close']).sort_values('date')
    features = add_features(observed).set_index('date')[FEATURES]
    index = features.index.union(pd.DatetimeIndex(calendar)).sort_values()
    return features.reindex(index).ffill().reindex(calendar)


def market_features(panel):
    close = panel.pivot(index='date', columns='symbol', values='close').sort_index()
    amount = panel.pivot(index='date', columns='symbol', values='amount').reindex_like(close)
    open_week = close.notna().any(axis=1)
    c = close.loc[open_week]
    r = c.pct_change(fill_method=None)
    market = r.mean(axis=1)
    # This synthetic equal-weight index is a contemporaneous observable, not a label.
    log_index = np.log1p(market).fillna(0).cumsum()
    log_amount = np.log1p(amount.loc[open_week].sum(axis=1, min_count=1))
    frame = pd.DataFrame(dict(market_ret1=market, market_ret4=log_index.diff(4),
        market_ret12=log_index.diff(12), market_vol20=market.rolling(20, min_periods=16).std(),
        breadth=(r > 0).sum(axis=1)/r.notna().sum(axis=1).replace(0, np.nan),
        activity=log_amount-log_amount.rolling(20, min_periods=16).mean()))
    return frame.reindex(close.index).ffill()


def industry_excess(y, valid_y, available, ids, min_stocks=10):
    """Realized equal-weight next-period return per industry minus the universe mean.

    Only current-bar availability and label validity enter; never future features.
    """
    count_ = valid_y & available
    weights = count_.astype('float32')
    onehot = np.zeros((y.shape[1], int(ids.max()) + 1), dtype='float32')
    onehot[np.arange(y.shape[1]), np.asarray(ids)] = 1.
    counts = weights @ onehot
    totals = (np.nan_to_num(y) * weights) @ onehot
    mean = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    safe = np.where(np.isfinite(mean), mean, 0.)
    grand = (safe * counts).sum(1) / np.maximum(counts.sum(1), 1)
    excess = mean - grand[:, None]
    return excess.astype('float32'), (counts >= min_stocks) & np.isfinite(excess)


def arrays(cfg):
    panel = pd.read_parquet(cfg['cache_path'])
    symbols = sorted(panel.symbol.unique())
    dates = np.array(sorted(panel.date.unique()))
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    aligned = panel.set_index(['date', 'symbol']).reindex(index)
    raw = aligned[FEATURES].to_numpy().reshape(len(dates), len(symbols), len(FEATURES))
    observed = aligned.observed.to_numpy().reshape(len(dates), len(symbols)).astype(bool)
    available = observed & np.isfinite(raw).all(axis=2)
    # Require adequate past observation coverage, never next-period label availability.
    coverage = pd.DataFrame(available).rolling(cfg['lookback'], min_periods=cfg['lookback']).sum().to_numpy()
    available &= coverage >= cfg.get('min_history_weeks', 32)
    cut = cfg['train_end']
    fit = raw[:cut][observed[:cut] & np.isfinite(raw[:cut]).all(axis=2)]
    mu, sd = fit.mean(0), fit.std(0)+1e-6
    scaled = np.clip((np.nan_to_num(raw)-mu)/sd, -8, 8)
    X = np.zeros_like(scaled)
    for t in range(len(dates)):
        mask = observed[t] & np.isfinite(raw[t]).all(axis=1)
        if mask.any():
            mean, std = scaled[t, mask].mean(0), scaled[t, mask].std(0)+1e-6
            X[t] = np.clip((scaled[t]-mean)/std, -6, 6)
            if t:
                X[t, ~mask] = X[t-1, ~mask]
        elif t:
            X[t] = X[t-1]
    y = aligned.target.to_numpy().reshape(len(dates), len(symbols))
    valid_y = np.isfinite(y)
    # Preserve observed returns for validation and trading accounting. Any
    # training-only target clipping belongs in the loss preparation below.
    y = np.nan_to_num(y)
    market = pd.read_csv(cfg['market_path'], parse_dates=['date']).set_index('date').reindex(dates)[MARKET_FEATURES]
    market_fit = market.iloc[:cut].dropna().to_numpy()
    mm, ms = market_fit.mean(0), market_fit.std(0)+1e-6
    M = np.clip((market.to_numpy()-mm)/ms, -6, 6)
    M = np.nan_to_num(M).astype('float32')
    universe = pd.read_csv(cfg['universe_path']).set_index('symbol').loc[symbols]
    ids = universe.industry_id.to_numpy(dtype=np.int64)
    assert np.bincount(ids).tolist() == [20]*29
    preprocessing = dict(feature_mean=mu.tolist(), feature_std=sd.tolist(),
        market_mean=mm.tolist(), market_std=ms.tolist(), feature_names=FEATURES,
        market_names=MARKET_FEATURES, symbols=symbols, industry_ids=ids.tolist(),
        calendar=[str(pd.Timestamp(d).date()) for d in dates], train_end=cut,
        availability='Observed positive-volume current bar and at least 32 of preceding 40 weeks observed; no future labels')
    return X.astype('float32'), y.astype('float32'), valid_y, M, available, observed, dates, symbols, ids, preprocessing
