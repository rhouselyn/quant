"""Data generation, optional download adapters, feature engineering and windows."""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Multi-scale, point-in-time features.  The first 12 names are retained for
# compatibility with older cached experiments; new runs can use all 24.
FEATURES = [
    "ret1", "ret2", "ret5", "ret10", "ret20", "ret60",
    "oc", "hl", "close_pos",
    "vol5", "vol20", "vol60",
    "mom5", "mom20", "mom60",
    "vlog", "vchg", "volume_z5", "volume_z20", "volume_z60",
    "range5", "range20", "range60", "amount_z20",
]

def synthetic_panel(n_stocks=200, n_days=520, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    n_sector = max(8, n_stocks // 20)
    sector = np.arange(n_stocks) % n_sector
    market = rng.normal(0, .008, n_days)
    sec = rng.normal(0, .006, (n_days, n_sector))
    # persistent but weak momentum signal, useful for an end-to-end demo
    latent = np.zeros(n_stocks)
    rets = np.zeros((n_days, n_stocks))
    for t in range(n_days):
        latent = .94 * latent + rng.normal(0, .012, n_stocks)
        rets[t] = .35 * market[t] + sec[t, sector] + .15 * latent + rng.normal(0, .012, n_stocks)
    base = rng.uniform(8, 80, n_stocks)
    close = base[None, :] * np.exp(np.cumsum(rets, axis=0))
    overnight = rng.normal(0, .002, (n_days, n_stocks))
    open_ = close * np.exp(-rets + overnight)
    high = np.maximum(open_, close) * (1 + rng.uniform(0, .002, (n_days, n_stocks)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, .002, (n_days, n_stocks)))
    volume = rng.lognormal(13, .45, (n_days, n_stocks)) * (1 + 8*np.abs(rets))
    rows=[]
    for j in range(n_stocks):
        rows.append(pd.DataFrame({"date":dates,"symbol":f"S{j:04d}","open":open_[:,j],"high":high[:,j],"low":low[:,j],"close":close[:,j],"volume":volume[:,j],"amount":volume[:,j]*close[:,j]}))
    return pd.concat(rows, ignore_index=True)


def _default_cn_symbols(limit: int = 200, seed: int = 42) -> list[str]:
    """Deterministic non-STAR A-share universe for the Yahoo adapter.

    Yahoo uses ``600000.SS``/``000001.SZ`` style tickers.  We generate a
    broad list instead of hard-coding today's constituents; unavailable or
    delisted codes are skipped by the downloader.  STAR (688xxx) and Beijing
    (8xxxxx) codes are deliberately excluded.
    """
    codes = [f"{i:06d}" for i in range(600000, 601000)]
    codes += [f"{i:06d}" for i in range(1, 1000)]
    codes += [f"{i:06d}" for i in range(2001, 3000)]
    codes += [f"{i:06d}" for i in range(300001, 301000)]
    # Deterministic shuffle gives a less index-heavy sample while preserving
    # reproducibility across runs.
    rng = np.random.default_rng(seed)
    rng.shuffle(codes)
    out = []
    for c in codes:
        suffix = ".SS" if c.startswith("6") else ".SZ"
        out.append(c + suffix)
        if len(out) >= max(limit * 2, 240):
            break
    return out


def _yahoo_one(symbol: str, start: str, end: str, timeout: int = 20) -> pd.DataFrame | None:
    """Download one ticker through Yahoo's public chart endpoint.

    This avoids the crumb/rate-limit path in yfinance and needs no API key.
    Prices are adjusted using Yahoo's ``adjclose / close`` factor so that
    returns are split/dividend adjusted while OHLC remains internally
    consistent.
    """
    import requests
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(pd.Timestamp(end, tz="UTC").timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d&events=history"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        result = (payload.get("chart") or {}).get("result")
        if not result:
            return None
        result = result[0]
        ts = result.get("timestamp") or []
        q = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adj = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
        if not ts or not q.get("close"):
            return None
        d = pd.to_datetime(ts, unit="s").tz_localize(None).normalize()
        close = np.asarray(q.get("close"), dtype=float)
        factor = np.ones_like(close)
        if adj is not None:
            av = np.asarray(adj, dtype=float)
            factor = np.divide(av, close, out=np.ones_like(av), where=np.isfinite(close) & (np.abs(close) > 1e-12))
        frame = pd.DataFrame({
            "date": d,
            "symbol": symbol,
            "open": np.asarray(q.get("open"), dtype=float) * factor,
            "high": np.asarray(q.get("high"), dtype=float) * factor,
            "low": np.asarray(q.get("low"), dtype=float) * factor,
            "close": close * factor,
            "volume": np.asarray(q.get("volume"), dtype=float),
        })
        frame["amount"] = frame["close"] * frame["volume"]
        return frame.dropna(subset=["close"])
    except Exception:
        return None


def yahoo_panel(n_stocks: int = 200, n_days: int = 1260, seed: int = 42,
                symbols: list[str] | None = None, workers: int = 8) -> pd.DataFrame:
    """Fetch a reproducible panel of Chinese A-share daily bars from Yahoo.

    ``n_days`` is the number of most recent trading sessions retained.  The
    endpoint is public and does not require credentials; network failures are
    reported and should be handled by ``load_data``'s explicit fallback.
    """
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=max(int(n_days * 2.2), 365))
    candidates = symbols or _default_cn_symbols(max(n_stocks, 40), seed)
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(_yahoo_one, s, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")): s for s in candidates}
        for fut in as_completed(futs):
            f = fut.result()
            if f is not None and len(f) >= min(120, n_days // 2):
                frames.append(f)
            if len(frames) >= n_stocks:
                # Remaining requests are allowed to finish cleanly; this
                # keeps the executor context deterministic and avoids leaks.
                pass
    if len(frames) < n_stocks:
        raise RuntimeError(f"Yahoo returned only {len(frames)} valid tickers; need {n_stocks}. Check network/rate limits.")
    # Stable symbol order and a common recent calendar make panel creation
    # deterministic even though requests complete out of order.
    panel = pd.concat(sorted(frames[:n_stocks], key=lambda x: x.symbol.iloc[0]), ignore_index=True)
    dates = np.array(sorted(panel.date.unique()))[-int(n_days):]
    panel = panel[panel.date.isin(dates)].sort_values(["symbol", "date"]).reset_index(drop=True)
    return panel


def _baostock_candidates(seed: int = 42, limit: int = 500) -> list[str]:
    """Return exchange-qualified A-share codes (STAR/Beijing excluded)."""
    codes = []
    # Main board Shanghai and Shenzhen; include a broad range because many
    # numeric codes are unused or delisted and will simply be skipped.
    for i in range(600000, 605000):
        codes.append(f"sh.{i:06d}")
    for i in range(1, 5000):
        codes.append(f"sz.{i:06d}")
    for i in range(300001, 305000):
        codes.append(f"sz.{i:06d}")
    rng = np.random.default_rng(seed); rng.shuffle(codes)
    return codes[:limit]


def baostock_panel(n_stocks: int = 200, n_days: int = 1260, seed: int = 42,
                   symbols: list[str] | None = None, pause: float = 0.03) -> pd.DataFrame:
    """Download前复权 A-share daily bars from the free Baostock service.

    Baostock's API is stateful and not thread-safe, so requests are performed
    sequentially.  The function skips unavailable/short-history symbols and
    raises a clear error if the requested panel cannot be assembled.
    """
    import baostock as bs
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=max(int(n_days * 2.2), 365))
    candidates = symbols or _baostock_candidates(seed, max(500, n_stocks * 4))
    login = bs.login()
    if getattr(login, "error_code", "1") != "0":
        raise RuntimeError(f"Baostock login failed: {getattr(login, 'error_msg', login)}")
    frames: list[pd.DataFrame] = []
    try:
        for ix, code in enumerate(candidates):
            try:
                rs = bs.query_history_k_data_plus(
                    code,
                    "date,open,high,low,close,volume,amount",
                    start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                    frequency="d", adjustflag="2")  # 前复权
                if rs.error_code != "0":
                    continue
                raw = rs.get_data()
                if raw is None or raw.empty:
                    continue
                raw = raw.rename(columns={"date": "date"})
                for c in ["open", "high", "low", "close", "volume", "amount"]:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")
                raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
                raw["symbol"] = code
                raw = raw.dropna(subset=["date", "close"])
                # Require enough observations for the requested lookback and
                # avoid newly listed/illiquid fragments.
                if len(raw) >= max(120, int(n_days * 0.55)):
                    frames.append(raw[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]])
                if len(frames) >= n_stocks:
                    break
            except Exception:
                continue
            if pause:
                time.sleep(pause)
    finally:
        try:
            bs.logout()
        except Exception:
            pass
    if len(frames) < n_stocks:
        raise RuntimeError(f"Baostock returned only {len(frames)} valid tickers; need {n_stocks}.")
    panel = pd.concat(sorted(frames[:n_stocks], key=lambda x: x.symbol.iloc[0]), ignore_index=True)
    dates = np.array(sorted(panel.date.unique()))[-int(n_days):]
    return panel[panel.date.isin(dates)].sort_values(["symbol", "date"]).reset_index(drop=True)


def aggregate_weekly(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLCV bars to Friday-ending weekly bars.

    ``date`` is labelled with the end of the trading week (Friday).  A week
    containing a holiday therefore still receives the next Friday label, while
    OHLC values are taken from the observations that actually traded.  This
    helper intentionally aggregates only raw OHLCV columns so it can also be
    used on a cached daily panel before feature engineering.
    """
    if panel is None or panel.empty:
        return panel.copy() if panel is not None else panel
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"weekly aggregation requires columns: {sorted(missing)}")
    raw = panel.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.tz_localize(None)
    raw = raw.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"])
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }
    if "amount" in raw.columns:
        agg["amount"] = "sum"
    out = (raw.groupby(["symbol", pd.Grouper(key="date", freq="W-FRI", label="right", closed="right")],
                       sort=True, observed=True)
              .agg(agg).reset_index())
    out = out.dropna(subset=["close"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def weekly_panel(n_stocks: int = 200, n_weeks: int = 520, provider: str = "baostock",
                 seed: int = 42, symbols: list[str] | None = None,
                 workers: int = 8) -> pd.DataFrame:
    """Fetch and return a weekly OHLCV panel.

    This is a convenience API for callers that need bars without engineered
    features.  Providers still download daily observations (the free APIs do
    not expose a reliable weekly endpoint), then :func:`aggregate_weekly`
    performs the same Friday-ending aggregation used by :func:`load_data`.
    """
    provider = str(provider or "baostock").lower()
    fetch_days = max(int(n_weeks) * 5 + 365, 365)
    if provider in {"yahoo", "real", "auto"}:
        panel = yahoo_panel(n_stocks=n_stocks, n_days=fetch_days, seed=seed,
                            symbols=symbols, workers=workers)
    elif provider == "baostock":
        panel = baostock_panel(n_stocks=n_stocks, n_days=fetch_days, seed=seed,
                               symbols=symbols)
    elif provider == "synthetic":
        panel = synthetic_panel(n_stocks=n_stocks, n_days=fetch_days, seed=seed)
    else:
        raise ValueError(f"unsupported weekly provider: {provider}")
    panel = aggregate_weekly(panel)
    dates = np.array(sorted(panel["date"].dropna().unique()))[-int(n_weeks):]
    return panel[panel["date"].isin(dates)].sort_values(["symbol", "date"]).reset_index(drop=True)

def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["symbol","date"]).copy()
    g = df.groupby("symbol", group_keys=False)
    logc = np.log(df["close"].clip(lower=1e-6))
    df["ret1"] = g["close"].transform(lambda s: np.log(s).diff())
    df["ret2"] = g["close"].transform(lambda s: np.log(s).diff(2))
    df["ret5"] = g["close"].transform(lambda s: np.log(s).diff(5))
    df["ret10"] = g["close"].transform(lambda s: np.log(s).diff(10))
    df["ret20"] = g["close"].transform(lambda s: np.log(s).diff(20))
    df["ret60"] = g["close"].transform(lambda s: np.log(s).diff(60))
    df["oc"] = np.log(df["close"] / df["open"])
    df["hl"] = np.log(df["high"] / df["low"])
    df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).clip(lower=1e-6)
    df["vol5"] = g["ret1"].transform(lambda s: s.rolling(5).std())
    df["vol20"] = g["ret1"].transform(lambda s: s.rolling(20).std())
    df["vol60"] = g["ret1"].transform(lambda s: s.rolling(60).std())
    lv = np.log1p(df["volume"])
    df["vlog"] = lv
    df["vchg"] = g["volume"].transform(lambda s: np.log1p(s).diff())
    for w in (5, 20, 60):
        df[f"volume_z{w}"] = g["volume"].transform(
            lambda s, w=w: (np.log1p(s) - np.log1p(s).rolling(w).mean()) /
            (np.log1p(s).rolling(w).std() + 1e-6))
    df["mom5"] = g["close"].transform(lambda s: s / s.rolling(5).mean() - 1)
    df["mom20"] = g["close"].transform(lambda s: s / s.rolling(20).mean() - 1)
    df["mom60"] = g["close"].transform(lambda s: s / s.rolling(60).mean() - 1)
    df["range5"] = g["hl"].transform(lambda s: s.rolling(5).mean())
    df["range20"] = g["hl"].transform(lambda s: s.rolling(20).mean())
    df["range60"] = g["hl"].transform(lambda s: s.rolling(60).mean())
    if "amount" in df:
        la = np.log1p(df["amount"].clip(lower=0))
        df["amount_z20"] = g["amount"].transform(
            lambda s: (np.log1p(s.clip(lower=0)) - np.log1p(s.clip(lower=0)).rolling(20).mean()) /
            (np.log1p(s.clip(lower=0)).rolling(20).std() + 1e-6))
    else:
        df["amount_z20"] = 0.0
    # Next-bar close-to-close return (next day for daily data, next week for
    # weekly data); signal uses only current and prior rows.
    df["target"] = g["close"].transform(lambda s: np.log(s.shift(-1) / s))
    return df.replace([np.inf,-np.inf], np.nan)

def load_data(provider="synthetic", cache_path="data/daily_features.parquet", n_stocks=200,
              n_days=520, seed=42, frequency="daily"):
    """Load a panel and engineer features at the requested sampling frequency.

    ``frequency='daily'`` preserves the original behaviour.  With
    ``frequency='weekly'``, providers fetch enough daily history and bars are
    aggregated to Friday-ending weeks *before* rolling features and targets
    are computed; consequently ``n_days`` means the number of weekly rows.
    """
    frequency = str(frequency or "daily").lower()
    if frequency not in {"daily", "weekly"}:
        raise ValueError("frequency must be 'daily' or 'weekly'")
    path = Path(cache_path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        cached = pd.read_parquet(path)
        # Do not silently reuse a smoke-test cache for a full run.
        is_synth = cached["symbol"].astype(str).str.startswith("S").all()
        wants_synth = provider == "synthetic"
        # New caches carry an explicit frequency marker.  For old files infer
        # daily versus weekly from the median spacing between observations.
        if "frequency" in cached.columns:
            cached_frequency = str(cached["frequency"].dropna().iloc[0]) if cached["frequency"].notna().any() else "daily"
        else:
            med_gap = (cached.sort_values(["symbol", "date"]).groupby("symbol")["date"].diff()
                       .dt.days.median())
            cached_frequency = "weekly" if pd.notna(med_gap) and med_gap >= 4 else "daily"
        if (cached["symbol"].nunique() == n_stocks and cached["date"].nunique() == n_days
                and is_synth == wants_synth and cached_frequency == frequency):
            # Recompute derived features in place when the feature schema has
            # been upgraded. This avoids downloading the same real bars again.
            if set(FEATURES + ["target"]).issubset(cached.columns):
                if "frequency" not in cached.columns:
                    cached["frequency"] = frequency
                return cached
            try:
                out = add_features(cached)
                out["frequency"] = frequency
                return out
            except Exception:
                pass
    panel = None
    # A weekly request is assembled from daily bars.  Keep additional history
    # for 60-week rolling features and then trim to the latest n_days weeks.
    fetch_days = int(n_days)
    if frequency == "weekly":
        fetch_days = max(int(n_days * 5 + 365), 365)
    if provider in {"yahoo", "real", "auto"}:
        try:
            panel = yahoo_panel(n_stocks=n_stocks, n_days=fetch_days, seed=seed)
        except Exception as exc:
            if provider in {"yahoo", "real"}:
                raise RuntimeError(f"real data download failed: {exc}") from exc
    if panel is None and provider in {"akshare","auto"}:
        try:
            import akshare as ak  # optional dependency
            # A full universe download is intentionally left to user credentials/rate limits.
            raise RuntimeError("AKShare adapter requires a configured symbol universe; using synthetic fallback")
        except Exception:
            panel = None
    if panel is None and provider == "baostock":
        try:
            panel = baostock_panel(n_stocks=n_stocks, n_days=fetch_days, seed=seed)
        except Exception as exc:
            raise RuntimeError(f"real data download failed: {exc}") from exc
    if panel is None:
        # Generate a sufficiently long daily history, then aggregate below so
        # synthetic and real providers have identical weekly semantics.
        panel = synthetic_panel(n_stocks, fetch_days if frequency == "weekly" else n_days, seed)
    if frequency == "weekly":
        panel = aggregate_weekly(panel)
        # Keep exactly the requested number of complete weekly observations.
        keep_dates = np.array(sorted(panel["date"].dropna().unique()))[-int(n_days):]
        panel = panel[panel["date"].isin(keep_dates)].copy()
    df = add_features(panel)
    # Persist the frequency for unambiguous cache validation on subsequent
    # runs; dataset_arrays ignores this metadata column.
    df["frequency"] = frequency
    df.to_parquet(path, index=False)
    return df

def dataset_arrays(df: pd.DataFrame, lookback=40, n_features=12, train_end=None,
                   cross_sectional=True, return_valid=False):
    """Return tensors as numpy arrays: X [T,N,L,F], y [T,N], dates and symbols."""
    feats = FEATURES[:n_features]
    syms = sorted(df.symbol.unique()); dates = np.array(sorted(df.date.unique()))
    piv = {s: df[df.symbol==s].set_index("date").reindex(dates) for s in syms}
    X = np.stack([p[feats].to_numpy(float) for p in piv.values()], axis=1)  # T,N,F
    y = np.stack([p["target"].to_numpy(float) for p in piv.values()], axis=1)
    valid_y = np.isfinite(y)
    # Fill only within each split later; synthetic has no holes. Robust fill for real data.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Winsorise labels only; this limits corrupted/limit-event rows from
    # dominating pairwise gradients while preserving their sign.
    y = np.where(np.isfinite(y), np.clip(y, -0.25, 0.25), np.nan)
    # rolling standardization using the training prefix
    cut = train_end if train_end is not None else max(1, int(len(dates)*.7))
    mu = X[:cut].reshape(-1, X.shape[-1]).mean(0); sd = X[:cut].reshape(-1, X.shape[-1]).std(0)+1e-6
    X = np.clip((X-mu)/sd, -8, 8)
    if cross_sectional:
        # Same-date cross-sectional scaling is point-in-time: it uses only
        # today's observable panel and prevents price/volume scale leakage.
        cs_mu = X.mean(axis=1, keepdims=True)
        cs_sd = X.std(axis=1, keepdims=True) + 1e-6
        X = np.clip((X-cs_mu)/cs_sd, -6, 6)
    # Keep the historical four-value API by default, but let training retain
    # a point-in-time validity mask.  Invalid/last-row targets must not become
    # artificial zero-return names in the daily top/bottom tail labels.
    y = np.nan_to_num(y, nan=0.0)
    result = (X.astype("float32"), y.astype("float32"), dates, syms)
    return (*result, valid_y.astype(bool)) if return_valid else result

def cache_hash(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()[:16]

# Compatibility API used by the command-line MVP.
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

@dataclass
class DataConfig:
    source: str = "synthetic"
    cache_path: str = "data/daily_features.parquet"
    n_stocks: int = 200
    n_days: int = 520
    seed: int = 42
    frequency: str = "daily"

class DailyDataStore:
    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()
    def load(self) -> pd.DataFrame:
        return load_data(self.config.source, self.config.cache_path, self.config.n_stocks,
                         self.config.n_days, self.config.seed, self.config.frequency)

class WindowPairDataset(Dataset):
    """Random same-date pairs. Inputs are already standardized arrays."""
    def __init__(self, X, y, dates, lookback=40, start=0, end=None, pairs_per_day=128, seed=42):
        self.X, self.y, self.dates = X, y, dates
        self.lookback, self.start = lookback, max(start, lookback)
        self.end = min(len(dates)-1, end if end is not None else len(dates)-1)
        self.pairs_per_day, self.rng = pairs_per_day, np.random.default_rng(seed)
        self.indices = [(t,k) for t in range(self.start, self.end) for k in range(self.pairs_per_day)]
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        t, _ = self.indices[idx]; n = self.X.shape[1]
        i,j = self.rng.choice(n, 2, replace=False)
        x1 = self.X[t-self.lookback+1:t+1, i]; x2 = self.X[t-self.lookback+1:t+1, j]
        d = float(self.y[t, i] - self.y[t, j]); label = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
        return torch.from_numpy(x1), torch.from_numpy(x2), torch.tensor(label), torch.tensor(t), torch.tensor(i), torch.tensor(j)


class TemporalScoreDataset(Dataset):
    """Single-stock windows for temporal contrastive + extreme-score training.

    Each sample contains a query window ending at ``t`` and a positive key
    window for the same stock ending at ``t + horizon``.  ``score_label`` is
    computed from the *whole cross-section* at ``t``: +1 for the top fraction
    of next-day returns, -1 for the bottom fraction, and 0 for all middle
    names (which are ignored by the score loss).  The key is only used during
    training and is never required by inference/backtesting.
    """
    def __init__(self, X, y, dates, lookback=40, start=0, end=None,
                 horizon=5, extreme_frac=0.2, seed=42):
        self.X, self.y, self.dates = X, y, dates
        self.lookback = int(lookback)
        self.horizon = max(1, int(horizon))
        self.start = max(int(start), self.lookback - 1)
        # ``end`` is exclusive and must leave enough rows for the future key.
        # The key is the *non-overlapping* forward segment [t+1, t+H].
        last = len(dates) - self.horizon
        # ``end`` denotes the first row after the training split.  Reserve
        # the H future rows inside that split as well, otherwise a query near
        # the boundary would read validation observations during training.
        requested_end = int(end) if end is not None else last
        self.end = min(last, requested_end - self.horizon if end is not None else last)
        self.extreme_frac = float(np.clip(extreme_frac, 0.01, 0.49))
        self.rng = np.random.default_rng(seed)
        n = self.X.shape[1]
        # Pre-compute labels from each date's cross-section. This avoids the
        # common bug where a shuffled mini-batch is incorrectly treated as a
        # cross-section.
        self.labels = np.zeros_like(self.y, dtype=np.float32)
        k = max(1, int(round(n * self.extreme_frac)))
        for t in range(self.start, max(self.start, self.end)):
            vals = self.y[t].astype(float)
            # ``dataset_arrays`` fills missing targets with zero for tensor
            # compatibility, so the finite check alone is insufficient here.
            # Keep the original point-in-time validity mask out of tail label
            # construction; otherwise a missing return could become an
            # artificial middle/quantile observation.
            valid = np.isfinite(vals) & self.valid_y[t]
            ids = np.flatnonzero(valid)
            if len(ids) < 2 * k:
                continue
            order = ids[np.argsort(vals[ids])]
            self.labels[t, order[:k]] = -1.0
            self.labels[t, order[-k:]] = 1.0
        self.indices = [(t, i) for t in range(self.start, self.end) for i in range(n)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, i = self.indices[idx]
        l = self.lookback
        xq = self.X[t - l + 1:t + 1, i]
        # Non-overlapping future key: exactly the next H observations.  This
        # follows the intended ``0..t`` versus ``t..t+5`` setup and prevents
        # a shifted long window from sharing L-H rows with the query.
        xk = self.X[t + 1:t + self.horizon + 1, i]
        return (torch.from_numpy(xq), torch.from_numpy(xk),
                torch.tensor(float(self.labels[t, i])), torch.tensor(t),
                torch.tensor(i))


class TemporalCrossSectionDataset(Dataset):
    """Date-grouped temporal contrastive samples.

    One dataset row corresponds to one trading date.  It returns all stocks
    in the panel for that date, so the InfoNCE denominator is the same-date
    cross-section (different stocks) instead of a mixture of unrelated dates.
    ``xq`` has shape ``[N, lookback, F]`` and ``xk`` has shape
    ``[N, horizon, F]``; labels are +1/-1 for the date's top/bottom tail and
    zero for the middle names.  Future keys are training-only.
    """
    def __init__(self, X, y, dates, lookback=40, start=0, end=None,
                 horizon=5, extreme_frac=0.2, valid_y=None):
        self.X, self.y, self.dates = X, y, dates
        self.lookback = int(lookback)
        self.horizon = max(1, int(horizon))
        self.start = max(int(start), self.lookback - 1)
        last = len(dates) - self.horizon
        requested_end = int(end) if end is not None else last
        # end is exclusive and leaves horizon rows for a non-overlapping key.
        self.end = min(last, requested_end - self.horizon if end is not None else last)
        self.extreme_frac = float(np.clip(extreme_frac, 0.01, 0.49))
        n = self.X.shape[1]
        self.indices = list(range(self.start, max(self.start, self.end)))
        self.labels = np.zeros((len(dates), n), dtype=np.float32)
        self.valid_y = np.ones_like(self.y, dtype=bool) if valid_y is None else np.asarray(valid_y, dtype=bool)
        k = max(1, int(round(n * self.extreme_frac)))
        for t in self.indices:
            vals = np.asarray(self.y[t], dtype=float)
            valid = np.flatnonzero(self.valid_y[t] & np.isfinite(vals))
            if len(valid) < 2 * k:
                continue
            order = valid[np.argsort(vals[valid])]
            self.labels[t, order[:k]] = -1.0
            self.labels[t, order[-k:]] = 1.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        l, h = self.lookback, self.horizon
        # [N, L, F], [N, H, F]; transpose from [time, stock, feature].
        xq = self.X[t - l + 1:t + 1].transpose(1, 0, 2)
        xk = self.X[t + 1:t + h + 1].transpose(1, 0, 2)
        labels = self.labels[t]
        stock_ids = np.arange(self.X.shape[1], dtype=np.int64)
        return (torch.from_numpy(xq), torch.from_numpy(xk),
                torch.from_numpy(labels), torch.tensor(t, dtype=torch.long),
                torch.from_numpy(stock_ids))
