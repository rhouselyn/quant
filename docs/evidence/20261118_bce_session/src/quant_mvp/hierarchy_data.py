"""Causal inputs for a date-grouped industry model. No label-derived input masks."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import FEATURES, add_features
from .data_trials import quality_channels

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


def daily_arrays(cfg, symbols, weekly_dates, train_end):
    """Daily-panel arrays aligned to the weekly Friday calendar for the gated
    daily branch.

    Same scaling policy as ``arrays``: global fit on the train window over
    observed values, per-day cross-sectional z-score clipped at +/-6, and
    last-known-value carry for suspended names.  ``D`` is [n_days, symbols,
    features] over the union trading calendar; ``daily_ends[t]`` is the union-
    calendar index of the last trading day at or before Friday ``t``, so the
    model window is the ``daily_n_days`` trading days ending at the anchor.
    """
    panel = pd.read_parquet(cfg['daily_path'])
    panel['date'] = pd.to_datetime(panel['date'])
    feats = list(cfg['daily_feature_names'])
    days = np.array(sorted(panel.date.unique()))
    index = pd.MultiIndex.from_product([days, symbols], names=['date', 'symbol'])
    aligned = panel.set_index(['date', 'symbol']).reindex(index)
    raw = aligned[feats].to_numpy().reshape(len(days), len(symbols), len(feats))
    observed = aligned['d_ret'].notna().to_numpy().reshape(len(days), len(symbols))
    if not 0 < train_end <= len(weekly_dates):
        raise ValueError('daily feature fit requires a non-empty training prefix')
    cut = np.searchsorted(days, weekly_dates[train_end - 1], side='right')
    fit = raw[:cut][observed[:cut] & np.isfinite(raw[:cut]).all(axis=2)]
    mu, sd = fit.mean(0), fit.std(0) + 1e-6
    scaled = np.clip((np.nan_to_num(raw) - mu) / sd, -8, 8)
    X2 = np.zeros_like(scaled)
    for t in range(len(days)):
        mask = observed[t] & np.isfinite(raw[t]).all(axis=1)
        if mask.any():
            mean, std = scaled[t, mask].mean(0), scaled[t, mask].std(0) + 1e-6
            X2[t] = np.clip((scaled[t] - mean) / std, -6, 6)
            if t:
                X2[t, ~mask] = X2[t - 1, ~mask]
        elif t:
            X2[t] = X2[t - 1]
    n = int(cfg.get('daily_n_days', 60))
    ends = np.searchsorted(days, weekly_dates, side='right') - 1
    # Only Fridays from the first training index onward are ever encoded; the
    # first lookback weeks have no model window at all (weekly lookback gate).
    if ends[cfg['lookback']] < n - 1:
        raise ValueError('Daily history shorter than daily_n_days before the first training Friday')
    return X2.astype('float32'), ends.astype(np.int64), dict(
        daily_mean=mu.tolist(), daily_std=sd.tolist(), daily_names=feats, n_days=len(days))


def board_limit(symbols, dates):
    """Per-name daily price-limit gap ratio. ChiNext (300/301) moved to +/-20%
    on 2020-08-24; SH/SZ main boards stay at +/-10%. The tighter ST 5% band is
    not flagged in the cache and therefore not modelled."""
    limit = np.full((len(dates), len(symbols)), 0.098)
    chinext = np.array([str(s).split('.')[0].startswith(('300', '301')) for s in symbols])
    reform = pd.DatetimeIndex(dates) >= pd.Timestamp('2020-08-24')
    if chinext.any() and reform.any():
        limit[np.ix_(reform, chinext)] = 0.198
    return limit


def arrays(cfg):
    from .financial_return import is_financial, validate_config, financial_statistics
    financial = is_financial(cfg)
    if financial:
        validate_config(cfg)
        if cfg.get('dataset_schema_version') != 2:
            raise ValueError('financial_return_v1 requires scheduled v2 labels')
    v2 = cfg.get('dataset_schema_version') == 2
    panel = pd.read_parquet(cfg['cache_path'])
    symbols = sorted(panel.symbol.unique())
    dates = np.array(sorted(panel.date.unique()))
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    aligned = panel.set_index(['date', 'symbol']).reindex(index)
    # Feature list is config-driven so extended caches (e.g. Monday-anchored
    # caches with daily-path features) coexist with the original 24-feature ones.
    feats = list(cfg.get('feature_names', FEATURES))
    raw = aligned[feats].to_numpy().reshape(len(dates), len(symbols), len(feats))
    observed = aligned.observed.fillna(False).to_numpy(dtype=bool).reshape(len(dates), len(symbols))
    # raw_available: a name is scoreable this week only with a real bar and every
    # feature finite.  ``available`` adds history-coverage gates and decides
    # scoring, relation masks and valid samples only; the feature fit, the weekly
    # cross-sectional standardisation and the carry-forward fill below still run
    # on raw_available rows (weekly-architecture runbook §2.1).
    raw_available = observed & np.isfinite(raw).all(axis=2)
    history = pd.DataFrame(raw_available)
    lookback = int(cfg['lookback'])
    coverage = history.rolling(lookback, min_periods=lookback).sum().to_numpy()
    available = raw_available & (coverage >= int(cfg.get('min_history_weeks', 32)))
    availability_rule = (f'raw observed bar with finite features, plus >= {cfg.get("min_history_weeks", 32)} '
                         f'of preceding {lookback} weeks observed (inclusive of current); no future labels')
    matched = cfg.get('matched_lookbacks')
    if matched:
        # Matched-coverage mode (weekly-architecture runbook §2.1): the usable
        # mask is the intersection of every listed window's own >=80%-observed
        # gate, so B0 and its lookback/query-width candidates train and score on
        # exactly the same stock-weeks regardless of each arm's lookback.  A
        # natural-coverage recheck for a W-winner drops this key.
        assert lookback in matched and all(int(width) > 0 for width in matched)
        available = raw_available.copy()
        for width in matched:
            width = int(width)
            count = history.rolling(width, min_periods=width).sum().to_numpy()
            available &= count >= (4 * width + 4) // 5
        availability_rule = ('raw observed bar with finite features, plus the intersection of '
                             + ', '.join(f'>= {(4 * w + 4) // 5}/{w} weeks observed' for w in map(int, matched))
                             + ' windows (inclusive of current); no future labels')
    cut = cfg['train_end']
    frozen = cfg.get('frozen_preprocessing') if v2 else None
    if frozen is not None:
        if frozen['feature_names'] != feats or frozen['symbols'] != symbols:
            raise ValueError('frozen preprocessing identity or feature order changed')
        if frozen['normalization'] != cfg.get('normalization', 'legacy_z'):
            raise ValueError('frozen preprocessing normalization changed')
        mu, sd = np.array(frozen['feature_mean']), np.array(frozen['feature_std'])
    else:
        fit = raw[:cut][raw_available[:cut]]
        if v2 and not len(fit):
            raise ValueError('no complete training feature rows')
        mu, sd = fit.mean(0), fit.std(0)+1e-6
    robust = v2 and cfg.get('normalization') == 'robust_mad'
    if financial and not (np.isfinite(mu).all() and np.isfinite(sd).all() and (sd > 0).all()):
        raise ValueError('invalid frozen financial feature scaling')
    scaled = np.clip((np.nan_to_num(raw)-mu)/sd, -8, 8)
    X = np.zeros_like(scaled)
    for t in range(len(dates)):
        mask = observed[t] & np.isfinite(raw[t]).all(axis=1)
        if mask.any():
            if robust:
                sample = raw[t, mask]
                center = np.median(sample, axis=0)
                scale = np.maximum(1.4826 * np.median(np.abs(sample - center), axis=0), 1e-6)
                X[t, mask] = np.clip((sample - center) / scale, -6, 6)
            else:
                mean, std = scaled[t, mask].mean(0), scaled[t, mask].std(0)+1e-6
                X[t] = np.clip((scaled[t]-mean)/std, -6, 6)
            if t:
                X[t, ~mask] = X[t-1, ~mask]
        elif t:
            X[t] = X[t-1]
    horizon = int(cfg.get('target_horizon', 1))
    label_mode = cfg.get('label_mode', 'close_close')
    tradable = None
    if v2 and cfg.get('scores_only'):
        y = np.full((len(dates), len(symbols)), np.nan)
        label_mode = 'scheduled_open_open'
    elif v2:
        labels = pd.read_parquet(cfg['labels_path'])
        labels['date'] = pd.to_datetime(labels.date)
        if labels.duplicated(['date', 'symbol']).any():
            raise ValueError('duplicate v2 label keys')
        labels = labels.set_index(['date', 'symbol']).reindex(index)
        known = pd.to_datetime(labels.label_available_at, utc=True)
        boundary = pd.Series(pd.Timestamp(cfg['selection_at']).tz_convert('UTC'), index=index)
        training = index.get_level_values('date').isin(dates[:cut])
        boundary.iloc[np.flatnonzero(training)] = pd.Timestamp(cfg['fit_cutoff']).tz_convert('UTC')
        valid = labels.valid.fillna(False) & known.lt(boundary)
        y = labels.ret.where(valid).to_numpy(dtype=float).reshape(len(dates), len(symbols))
        label_mode = 'scheduled_open_open'
    elif label_mode == 'open_open':
        # Monday-open execution: features sit at Friday close t, the position is
        # bought at next week's open (first trading day, usually Monday) and sold
        # at the following week's open. An exit whose open is missing or opens
        # limit-down (sell unfillable at the auction) falls back to that week's
        # close. Entry tradability (next open exists and gaps up by less than the
        # board limit, i.e. the buy is fillable) is exported for the backtest.
        open_ = aligned.open.to_numpy(dtype=float).reshape(len(dates), len(symbols))
        close = aligned.close.to_numpy(dtype=float).reshape(len(dates), len(symbols))
        limit = board_limit(symbols, dates)
        y = np.full_like(close, np.nan)
        entry, exit_ = open_[1:-1], open_[2:].copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            gap_exit = exit_ / close[1:-1] - 1
        exit_ = np.where(~np.isfinite(exit_) | (gap_exit <= -limit[2:]), close[2:], exit_)
        with np.errstate(divide='ignore', invalid='ignore'):
            y[:-2] = np.log(exit_ / entry)
            gap_entry = open_[1:] / close[:-1] - 1
        tradable = np.zeros_like(close, dtype=bool)
        tradable[:-1] = np.isfinite(open_[1:]) & (open_[1:] > 0) & (gap_entry < limit[1:])
    elif horizon > 1:
        # Forward h-week log return from the same Friday close; features stay at t.
        close = aligned.close.to_numpy(dtype=float).reshape(len(dates), len(symbols))
        y = np.full_like(close, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            y[:-horizon] = np.log(close[horizon:] / close[:-horizon])
    else:
        y = aligned.target.to_numpy().reshape(len(dates), len(symbols))
    valid_y = np.isfinite(y)
    # Preserve observed returns for validation and trading accounting. Any
    # training-only target clipping belongs in the loss preparation below.
    y = np.nan_to_num(y)
    market = pd.read_csv(cfg['market_path'], parse_dates=['date']).set_index('date').reindex(dates)[MARKET_FEATURES]
    if frozen is not None:
        if frozen['market_names'] != MARKET_FEATURES:
            raise ValueError('frozen market feature order changed')
        mm, ms = np.array(frozen['market_mean']), np.array(frozen['market_std'])
    else:
        market_fit = market.iloc[:cut].dropna().to_numpy()
        mm, ms = market_fit.mean(0), market_fit.std(0)+1e-6
    M = np.clip((market.to_numpy()-mm)/ms, -6, 6)
    if financial:
        if not (np.isfinite(mm).all() and np.isfinite(ms).all() and (ms > 0).all()):
            raise ValueError('invalid frozen financial market scaling')
        M[~np.isfinite(market.to_numpy())] = np.nan
    # Missing absolute market state must remain visible to the financial path.
    M = (M if financial else np.nan_to_num(M)).astype('float32')
    universe = pd.read_csv(cfg['universe_path']).set_index('symbol').loc[symbols]
    ids = universe.industry_id.to_numpy(dtype=np.int64)
    if financial and frozen is not None and frozen.get('industry_ids') != ids.tolist():
        raise ValueError('frozen static industry identity changed')
    dated_ids = None
    if v2 and cfg.get('industry_ids_path'):
        incompatible = ('industry_relative_names', 'cell_balanced_weight', 'industry_head_weight',
                        'industry_balanced_weight', 'tail_within_industry', 'turnover_control_weight',
                        'nextweek_head_weight')
        if any(cfg.get(k) for k in incompatible) or cfg.get('contrastive_negatives') == 'industry':
            raise ValueError('dated industry routing supports the B0/CS objectives')
        table = pd.read_parquet(cfg['industry_ids_path'])
        table['date'] = pd.to_datetime(table.date)
        if table.duplicated(['date', 'symbol']).any() or table.industry_id.isna().any():
            raise ValueError('invalid dated industry rows')
        unknown = int(cfg['unknown_industry_id'])
        if (unknown < 0 or not table.industry_id.between(0, unknown).all()
                or not table.industry_id.mod(1).eq(0).all()):
            raise ValueError('invalid dated industry vocabulary')
        table['available_at'] = pd.to_datetime(table.available_at, utc=True)
        table['decision_at'] = pd.to_datetime(table.decision_at, utc=True)
        visible = table.available_at.le(table.decision_at)
        if not (visible | table.industry_id.eq(unknown)).all():
            raise ValueError('dated industry assignment lacks timely evidence')
        dated_ids = (table.set_index(['date','symbol']).industry_id.reindex(index)
                     .fillna(unknown).to_numpy(dtype=np.int64).reshape(len(dates), len(symbols)))
    # Optional industry-relative inputs.  The base features are already
    # cross-sectionally z-scored across the whole universe, so subtracting the
    # industry mean gives the model a within-industry view that the frozen
    # relation layer never supplies.  Computed on the z-scored columns, then
    # rescaled cross-sectionally to unit variance to match the base scale.
    rel_names = list(cfg.get('industry_relative_names') or [])
    if rel_names:
        missing = [n for n in rel_names if n not in feats]
        if missing:
            raise ValueError(f'industry_relative_names not in feature_names: {missing}')
        base_idx = [feats.index(n) for n in rel_names]
        vals = X[:, :, base_idx].astype('float64')       # [T, N, K]
        n_ind = int(ids.max()) + 1
        onehot = np.zeros((n_ind, len(symbols)), dtype='float64')
        onehot[ids, np.arange(len(symbols))] = 1.
        counts = onehot @ available.T.astype('float64')   # [G, T]
        sums = np.einsum('gn,tnk->gtk', onehot, vals)     # [G, T, K]
        means = np.divide(sums, counts[..., None],
                          out=np.zeros_like(sums), where=counts[..., None] > 0)
        rel = vals - means[ids].transpose(1, 0, 2)        # [T, N, K]
        rel = np.nan_to_num(rel, nan=0., posinf=0., neginf=0.)
        mask3 = available[:, :, None]
        rel_n = np.maximum(mask3.sum(1, keepdims=True), 1)
        rel_mean = np.where(mask3, rel, 0.).sum(1, keepdims=True) / rel_n
        rel_var = np.where(mask3, (rel - rel_mean) ** 2, 0.).sum(1, keepdims=True) / rel_n
        rel = np.clip((rel - rel_mean) / (np.sqrt(rel_var) + 1e-6), -6, 6)
        X = np.concatenate([X, rel.astype('float32')], axis=-1)
        feats = feats + [f'{n}_ind' for n in rel_names]
    quality_mode = cfg.get('price_volume_quality', 'off')
    if quality_mode not in ('off', 'observed', 'zero'):
        raise ValueError('invalid price_volume_quality mode')
    if frozen is not None and frozen.get('price_volume_quality', 'off') != quality_mode:
        raise ValueError('frozen quality channel mode changed')
    if quality_mode != 'off':
        quality = quality_channels(raw, observed)
        if quality_mode == 'zero':
            quality.fill(0)
        X = np.concatenate((X, quality), axis=-1)
    if X.shape[-1] != int(cfg.get('n_features', X.shape[-1])):
        raise ValueError(f'n_features={cfg.get("n_features")} but X has {X.shape[-1]} columns')
    # Groups may be unequal (a full-market universe holds 25..157 names per SW
    # L1 industry); the model pads to the largest group and masks the padding.
    # Only an empty group is invalid, since its industry token would be orphaned.
    industry_sizes = np.bincount(ids[ids >= 0] if financial else ids, minlength=int(ids.max())+1).tolist()
    if not financial:
        assert min(industry_sizes) >= 1, f'empty industry group: {industry_sizes}'
    cell_ids = None
    cell_meta = None
    if cfg.get('cell_balanced_weight'):
        # Observable cells: industry x 20-week liquidity tercile x 20-week
        # volatility tercile.  Ranks use only the current cross-section and
        # trailing bars, so the cell assignment is causal at the decision date.
        close = aligned['close'].to_numpy(dtype=float).reshape(len(dates), len(symbols))
        amount = aligned['amount'].to_numpy(dtype=float).reshape(len(dates), len(symbols))
        close_df = pd.DataFrame(close, index=pd.DatetimeIndex(dates), columns=symbols)
        amount_df = pd.DataFrame(amount, index=close_df.index, columns=symbols)
        returns = close_df.pct_change(fill_method=None)
        vol20 = returns.rolling(20, min_periods=16).std()
        amount20 = amount_df.rolling(20, min_periods=16).mean()
        current = pd.DataFrame(available, index=close_df.index, columns=symbols)
        liq_pct = amount20.where(current).rank(axis=1, pct=True)
        vol_pct = vol20.where(current).rank(axis=1, pct=True)
        n_liq = int(cfg.get('cell_balance_liquidity_buckets', 3))
        n_vol = int(cfg.get('cell_balance_volatility_buckets', 3))
        liq = np.ceil(liq_pct.to_numpy() * n_liq)
        vol = np.ceil(vol_pct.to_numpy() * n_vol)
        usable = np.isfinite(liq) & np.isfinite(vol)
        liq = np.where(usable, np.clip(liq, 1, n_liq), 0).astype(np.int64)
        vol = np.where(usable, np.clip(vol, 1, n_vol), 0).astype(np.int64)
        cell_ids = (np.asarray(ids, dtype=np.int64)[None, :] * (n_liq * n_vol)
                    + (liq - 1) * n_vol + (vol - 1))
        cell_ids[~usable] = -1
        cell_ids = cell_ids.astype(np.int32)
        cell_meta = dict(dims=['industry', 'liquidity_20w', 'volatility_20w'],
                         liquidity_buckets=n_liq, volatility_buckets=n_vol,
                         alpha=float(cfg.get('cell_balance_alpha', .5)),
                         cap=float(cfg.get('cell_balance_cap', 3.)),
                         floor=float(cfg.get('cell_balance_floor', .2)),
                         n_cells=int(cell_ids.max()) + 1)
    if rel_names:
        # Placeholder entries keep preprocessing arrays aligned with
        # feature_names; the relative columns are already z-score units.
        mu = np.concatenate([mu, np.zeros(len(rel_names))])
        sd = np.concatenate([sd, np.ones(len(rel_names))])
    preprocessing = dict(feature_mean=mu.tolist(), feature_std=sd.tolist(),
        market_mean=mm.tolist(), market_std=ms.tolist(), feature_names=feats,
        market_names=MARKET_FEATURES, symbols=symbols,
        industry_ids=ids.tolist(), industry_sizes=industry_sizes,
        calendar=[str(pd.Timestamp(d).date()) for d in dates], train_end=cut,
        availability=availability_rule,
        label_mode=label_mode,
        **(dict(normalization=cfg.get('normalization', 'legacy_z')) if v2 else {}),
        **(dict(cell_balance=cell_meta) if cell_meta else {}),
        **(dict(execution='Buy at next week open, sell at the following week open; '
                'limit-down or missing exit opens fall back to that week close; '
                'entries require a fillable open (gap below board limit-up)')
           if label_mode == 'open_open' else {}))
    D, daily_ends = (None, None)
    if quality_mode != 'off':
        preprocessing.update(price_volume_quality=quality_mode,
            quality_feature_names=['observed_bar', 'nonfinite_feature_fraction', 'bar_age_weeks_cap40'])
    if cfg.get('daily_path'):
        if cfg.get('daily_weekly_blocks'):
            from .dataset_store import file_sha256

            path = Path(cfg['daily_path'])
            names = list(cfg['daily_feature_names'])
            digest = file_sha256(path)
            if frozen is not None and frozen.get('daily_blocks_sha256') != digest:
                raise ValueError('daily blocks differ from the training input')
            blocks = pd.read_parquet(path, columns=['date', 'symbol', *names], filters=[('date', '<=', pd.Timestamp(dates[-1]))])
            index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
            values = blocks.set_index(['date', 'symbol']).reindex(index)[names].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError('daily blocks must cover all weekly date/symbol keys with finite values')
            D = values.reshape(len(dates), len(symbols), len(names))
            daily_ends = np.arange(len(dates), dtype=np.int64)
            preprocessing.update(daily_blocks_sha256=digest, daily_names=names)
        else:
            D, daily_ends, daily_prep = daily_arrays(cfg, symbols, dates, cut)
            preprocessing.update(daily_prep)
    result = (X.astype('float32'), y.astype('float32'), valid_y, M, available, observed, dates,
              symbols, ids, preprocessing, D, daily_ends, tradable)
    if financial:
        preprocessing['financial_return'] = financial_statistics(cfg, result[1], valid_y, available, dates)
    if dated_ids is not None or (v2 and cfg.get('augmentation') == 'mask_past_groups'):
        result += (cell_ids, raw_available, dated_ids)
    elif cell_ids is not None:
        result += (cell_ids,)
    return result
