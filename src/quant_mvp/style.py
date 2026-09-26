"""Observable weekly style proxies and cross-sectional score residualization.

No forward returns enter exposures, regressions, or the candidate universe.
These OHLCV proxies are not a complete fundamental risk model.
"""
import numpy as np
import pandas as pd


STYLE_COLUMNS = ['return_1w', 'momentum_12_1w', 'momentum_52_4w',
                 'volatility_20w', 'beta_52w', 'log_amount_20w', 'illiquidity_20w']


def style_panel(panel):
    close = panel.pivot(index='date', columns='symbol', values='close').sort_index()
    amount = panel.pivot(index='date', columns='symbol', values='amount').reindex_like(close)
    log_price = np.log(close.where(close > 0))
    returns = close.pct_change(fill_method=None)
    market = returns.mean(axis=1)
    factors = {
        'return_1w': log_price.diff(),
        'momentum_12_1w': log_price.shift(1) - log_price.shift(12),
        'momentum_52_4w': log_price.shift(4) - log_price.shift(52),
        'volatility_20w': returns.rolling(20, min_periods=16).std(),
        'beta_52w': returns.rolling(52, min_periods=40).cov(market).div(
            market.rolling(52, min_periods=40).var().replace(0, np.nan), axis=0),
        'log_amount_20w': np.log(amount.where(amount > 0).rolling(20, min_periods=16).mean()),
        'illiquidity_20w': (returns.abs() / amount.where(amount > 0)).rolling(20, min_periods=16).mean(),
    }
    return pd.concat({k: v.stack(dropna=False) for k, v in factors.items()}, axis=1).reset_index()


def design_matrix(group, columns=STYLE_COLUMNS, industry=False):
    """Rank-standardized styles; same-date median fill. Industry is optional metadata."""
    continuous = group[list(columns)].replace([np.inf, -np.inf], np.nan)
    continuous = continuous.fillna(continuous.median()).fillna(0.)
    ranks = continuous.rank(pct=True)
    z = (ranks - ranks.mean()) / ranks.std(ddof=0).replace(0., 1.)
    parts = [np.ones((len(group), 1)), z.to_numpy()]
    if industry:
        parts.append(pd.get_dummies(group.industry.fillna('Unknown'), dtype=float).to_numpy())
    return np.column_stack(parts), z


def neutralize(scores, design, strength=1.):
    """Remove fitted cross-sectional component; preserve mean and score units."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or not np.isfinite(design).all():
        raise ValueError('Neutralization requires finite scores and design')
    if not 0 <= strength <= 1:
        raise ValueError('strength must be in [0,1]')
    fitted = design @ np.linalg.lstsq(design, scores, rcond=None)[0]
    variance = np.sum((scores - scores.mean()) ** 2)
    r2 = 1 - np.sum((scores - fitted) ** 2) / variance if variance > 1e-20 else 0.
    return scores - strength * (fitted - scores.mean()), float(r2)
