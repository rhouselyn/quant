"""Three-book evaluation for the weekly-architecture batch (runbook §5-§7).

Subcommands
  dates    build evaluation_dates.csv (decision anchors, phases, exclusions)
  epochs   per-epoch inner-window (2018-2019) long-only net Sharpe -> epoch_selection.json
  compare  fused candidate vs control over L20/S10/LS50(+L40) at 10/30 bps
  combine  pairwise blend/route of promoted parents vs B0 and both parents

All books share one proxy panel built from the same weekly snapshot, one
short-eligibility reader (point-in-time lists, venue separated) and one order
function; candidate/control pairs are replayed on identical asset returns.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts"))
from quant_mvp.hierarchy_data import board_limit
from r1111_common import paired_stats, perf_summary, yearly_table, write_json

DEV_CUT = pd.Timestamp("2021-01-08")
INNER_START = pd.Timestamp("2018-01-05")
DEV_YEAR = 2020


def zscore(s):
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


def fuse_epochs(scores, epochs):
    frame = scores.loc[scores.epoch.isin(epochs)].copy()
    assert set(frame.epoch) == set(epochs)
    assert not frame.duplicated(["epoch", "date", "symbol"]).any()
    grouped = frame.groupby(["date", "symbol"])["score"]
    assert grouped.size().eq(len(epochs)).all()
    assert (grouped.count().eq(0) | grouped.count().eq(len(epochs))).all()
    frame["score"] = frame.groupby(["epoch", "date"])["score"].transform(zscore)
    out = frame.groupby(["date", "symbol"], as_index=False)["score"].mean()
    out["score"] = out.groupby("date")["score"].transform(zscore)
    return out


def fuse_seeds(frames):
    """Equal-weight average of per-seed standard scores, re-z-scored per date."""
    indexed = [f.set_index(["date", "symbol"]).score.sort_index() for f in frames]
    assert len(indexed) >= 2 and all(s.index.is_unique for s in indexed)
    for s in indexed[1:]:
        assert s.index.equals(indexed[0].index) and s.notna().equals(indexed[0].notna())
    combined = pd.concat(indexed, axis=1).mean(axis=1).rename("score").reset_index()
    combined["score"] = combined.groupby("date")["score"].transform(zscore)
    return combined


def blend_scores(members):
    return fuse_seeds(members)


def build_proxy_panel(panel, legacy_calendar=False, session_data=None):
    if not legacy_calendar:
        market_open = panel.groupby("date")["observed"].any()
        panel = panel.loc[panel.date.isin(market_open.index[market_open])]
    calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
    symbols = sorted(panel.symbol.unique())
    index = pd.MultiIndex.from_product([calendar, symbols], names=["date", "symbol"])
    aligned = panel.set_index(["date", "symbol"]).reindex(index)
    shape = (len(calendar), len(symbols))
    opens = aligned["open"].to_numpy(float).reshape(shape)
    closes = aligned["close"].to_numpy(float).reshape(shape)
    limits = board_limit(symbols, calendar)
    entry, exit_open = opens[1:-1], opens[2:]
    exit_close = closes[2:]
    with np.errstate(divide="ignore", invalid="ignore"):
        entry_gap = entry / closes[:-2] - 1
        exit_gap = exit_open / closes[1:-1] - 1
        entry_base_ok = np.isfinite(entry_gap) & np.isfinite(entry) & (entry > 0) & (closes[:-2] > 0)
        exit_base_ok = np.isfinite(exit_gap) & np.isfinite(exit_open) & (exit_open > 0) & (closes[1:-1] > 0)
        if session_data is not None:
            sessions = session_data.copy()
            sessions["week"] = pd.to_datetime(sessions["week"])
            if sessions.duplicated(["week", "symbol"]).any():
                raise ValueError("duplicate weekly opening-session records")
            actual = sessions.set_index(["week", "symbol"])["on_first_session"].reindex(index)
            actual = actual.fillna(False).to_numpy(dtype=bool).reshape(shape)
            entry_base_ok &= actual[1:-1]
            exit_base_ok &= actual[2:]
        long_entry = entry_base_ok & (entry_gap < limits[1:-1])
        short_entry = entry_base_ok & (entry_gap > -limits[1:-1])
        long_close = ~(exit_base_ok & (exit_gap > -limits[2:]))
        short_close = ~(exit_base_ok & (exit_gap < limits[2:]))
        long_exit = np.where(long_close, exit_close, exit_open)
        short_exit = np.where(short_close, exit_close, exit_open)
        long_r = np.where(long_entry & (long_exit > 0), long_exit / entry - 1, np.nan)
        short_r = np.where(short_entry & (short_exit > 0), short_exit / entry - 1, np.nan)
    frame = aligned.iloc[: (len(calendar) - 2) * len(symbols)][["amount"]].reset_index()
    frame["entry_week"] = np.repeat(calendar[1:-1], len(symbols))
    frame["exit_week"] = np.repeat(calendar[2:], len(symbols))
    if session_data is not None:
        opens_by_week = sessions.groupby("week")["market_date"].first().reindex(calendar)
        elapsed = (opens_by_week.iloc[2:].to_numpy() - opens_by_week.iloc[1:-1].to_numpy())
        elapsed = elapsed / np.timedelta64(1, "D")
        if not np.isfinite(elapsed).all() or (elapsed <= 0).any():
            raise ValueError("missing or invalid market opening dates")
    else:
        elapsed = (calendar[2:] - calendar[1:-1]).days.to_numpy(dtype=float)
    frame["borrow_days"] = np.repeat(elapsed, len(symbols))
    for key, value in dict(long_entry=long_entry, short_entry=short_entry,
                           long_close=long_close, short_close=short_close,
                           long_r=long_r, short_r=short_r).items():
        frame[key] = value.ravel()
    return frame


def read_short_snapshots(end_at):
    sources = {"SZ": Path("data/margin_eligibility"),
               "SH": Path("data/margin_eligibility_sse_short")}
    by_venue = {}
    for venue, folder in sources.items():
        snapshots = {}
        for path in sorted(folder.glob("*Q*.json")):
            # A quarter ending after the phase boundary cannot supply an earlier list.
            if pd.Period(path.stem, freq="Q").end_time.normalize() >= end_at:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            date = record.get("meta", {}).get("szse_date") if venue == "SZ" else record.get("snap")
            if not date:
                continue
            snap = pd.Timestamp(str(date))
            if snap >= end_at:
                continue
            codes = record["rq"] if venue == "SZ" else record["codes"]
            assert snap not in snapshots
            snapshots[snap] = {str(code).zfill(6) for code in codes}
        by_venue[venue] = snapshots
    return by_venue


def attach_short_eligibility(frame, by_venue, lag_one=False, legacy_venue=False):
    """Point-in-time venue eligibility; ``lag_one`` delays every list one more
    snapshot (stress attach read, runbook §6.1)."""
    frame = frame.copy()
    frame["short_ok"] = False
    frame["short_snapshot"] = pd.NaT
    symbol_venue = frame.symbol.str.rsplit(".", n=1).str[-1]
    if not legacy_venue:
        symbol_venue = symbol_venue.replace({"SS": "SH"})
    for venue, snapshots in by_venue.items():
        keys = pd.DatetimeIndex(sorted(snapshots))
        for day, group in frame.loc[symbol_venue.eq(venue)].groupby("date"):
            i = keys.searchsorted(day, side="left") - 1 - (1 if lag_one else 0)
            if i >= 0:
                frame.loc[group.index, "short_snapshot"] = keys[i]
                codes = group.symbol.str.split(".").str[0]
                frame.loc[group.index, "short_ok"] = codes.isin(snapshots[keys[i]]).to_numpy()
    return frame


def pick_targets(frame, long_n, short_n, long_share, short_share):
    known = frame.loc[np.isfinite(frame.score_long) & np.isfinite(frame.score_short)]
    longs = known.sort_values(["score_long", "symbol"], ascending=[False, True]).head(long_n)
    liquid = known.loc[np.isfinite(known.amount) & known.amount.gt(0)].copy()
    liquid = liquid.loc[liquid.amount.rank(pct=True).gt(.6)]
    pool = liquid.loc[liquid.short_ok & ~liquid.symbol.isin(longs.symbol)]
    shorts = pool.sort_values(["score_short", "symbol"]).head(short_n)
    intended = {(1, s): long_share / long_n for s in longs.symbol}
    intended.update({(-1, s): short_share / short_n for s in shorts.symbol})
    return intended


def replay(frame, dates, spec, cost_bps=10.0, borrow_ann=.08, missing_penalty=0.0,
           long_share_by_date=None, reweight=None, target_picker=pick_targets):
    previous, weekly, events = {}, [], []
    groups = {day: group for day, group in frame.groupby("date")}
    for day in dates:
        g = groups[day]
        rows = g.set_index("symbol")
        assert rows.index.is_unique
        if long_share_by_date is None:
            intended = target_picker(g, *spec)
        else:
            share = float(long_share_by_date[day])
            if not 0 <= share <= 1:
                raise ValueError(f'invalid long share for {day}: {share}')
            intended = target_picker(g, *spec[:2], share, 1.0 - share)
        if reweight is not None:
            adjusted = reweight(day, intended.copy())
            if (adjusted.keys() != intended.keys()
                    or any(not np.isfinite(w) or w <= 0 for w in adjusted.values())
                    or any(not np.isclose(sum(w for k,w in adjusted.items() if k[0] == side),
                                          sum(w for k,w in intended.items() if k[0] == side), atol=1e-12, rtol=0)
                           for side in (1,-1))):
                raise ValueError(f'reweighting changed the basket or side budget for {day}')
            intended = adjusted
        held, returns = {}, {}
        missing, close_fallback = 0, 0
        for key, weight in intended.items():
            side, symbol = key
            row = rows.loc[symbol]
            prefix = "long" if side > 0 else "short"
            filled = bool(row[prefix + "_entry"])
            observed_r = float(row[prefix + "_r"]) if filled else np.nan
            if filled:
                missing += int(not np.isfinite(observed_r))
                close_fallback += int(row[prefix + "_close"])
                held[key] = weight
                if np.isfinite(observed_r):
                    returns[key] = observed_r
                else:
                    # Conservative scenario: a held position without an exit
                    # valuation takes the adverse side of a 20% asset move
                    # (main ledger marks it flat; runbook §6.2).
                    returns[key] = -missing_penalty if side > 0 else missing_penalty
            events.append(dict(date=day, exit_week=row.exit_week, side=side, symbol=symbol,
                               desired=weight, filled=weight if filled else 0.0,
                               asset_r=observed_r, exit_close_proxy=bool(filled and row[prefix + "_close"])))
        keys = sorted(held.keys() | previous.keys())
        turn = sum(abs(held.get(k, 0.0) - previous.get(k, 0.0)) for k in keys)
        gross_ret = sum(k[0] * held[k] * returns[k] for k in sorted(held))
        long_w = sum(w for k, w in held.items() if k[0] == 1)
        short_w = sum(w for k, w in held.items() if k[0] == -1)
        trading = turn * cost_bps / 10000
        borrow_days = float(g.borrow_days.iloc[0]) if 'borrow_days' in g else 7.
        borrow = short_w * borrow_ann * borrow_days / 365
        net = gross_ret - trading - borrow
        assert np.isfinite(net) and 1 + net > 0
        assert long_w + short_w <= 1 + 1e-12
        previous = {k: w * (1 + returns[k]) / (1 + net) for k, w in held.items()}
        weekly.append(dict(date=day, exit_week=g.exit_week.iloc[0], net=net, gross_ret=gross_ret,
                           turnover=turn, trading_cost=trading, borrow_cost=borrow,
                           long_w=long_w, short_w=short_w, net_exposure=long_w-short_w,
                           missing=missing, close_fallback=close_fallback,
                           intended_n=len(intended), filled_n=len(held),
                           intended_short_n=sum(k[0] == -1 for k in intended)))
    return pd.DataFrame(weekly), pd.DataFrame(events)


BOOKS = {"L20": (20, 0, 1.0, 0.0), "S10": (0, 10, 0.0, 1.0),
         "LS50": (20, 10, .5, .5), "L40": (40, 0, 1.0, 0.0)}


def load_scores(run_dirs, windows, data_dir):
    """Fuse per-run epoch windows (§5), then equal-weight across runs (§5 seeds)."""
    fused = []
    for run, window in zip(run_dirs, windows):
        path = Path(run) / "dev_scores.parquet"
        assert path.exists(), f"missing {path}; run the scores stage first"
        scores = pd.read_parquet(path)
        scores["date"] = pd.to_datetime(scores["date"])
        fused.append(fuse_epochs(scores, window))
    return fused[0] if len(fused) == 1 else fuse_seeds(fused)


def evaluation_dates(data_dir, out_dir=None, legacy_calendar=False):
    panel = pd.read_parquet(Path(data_dir) / "weekly.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
    has_market_bar = panel.groupby("date")["observed"].any()
    active_calendar = calendar if legacy_calendar else calendar[has_market_bar.reindex(calendar).to_numpy()]
    rows = []
    for t, day in enumerate(calendar):
        next_index = active_calendar.searchsorted(day, side="right")
        entry = active_calendar[next_index] if next_index < len(active_calendar) else pd.NaT
        exit_wk = active_calendar[next_index + 1] if next_index + 1 < len(active_calendar) else pd.NaT
        reason = ""
        if pd.isna(exit_wk):
            reason = "tail_label_not_mature"
        elif not bool(has_market_bar.get(day, False)):
            reason = "all_market_holiday_no_decision"
        phase = ("warmup" if day < INNER_START else
                 "inner" if day < pd.Timestamp("2020-01-01") else "dev")
        rows.append(dict(anchor=str(day.date()), anchor_index=t,
                         entry_week=str(pd.Timestamp(entry).date()) if not pd.isna(entry) else "",
                         exit_week=str(pd.Timestamp(exit_wk).date()) if not pd.isna(exit_wk) else "",
                         phase=phase, has_market_bar=bool(has_market_bar.get(day, False)),
                         included=not reason, exclude_reason=reason))
    frame = pd.DataFrame(rows)
    if out_dir is not None:
        frame.to_csv(Path(out_dir) / "evaluation_dates.csv", index=False)
    return frame


def replay_dates(eval_dates):
    """Frozen decision weeks the books actually run: 2018 onward, included."""
    frame = eval_dates[eval_dates.included].copy()
    frame["anchor"] = pd.to_datetime(frame["anchor"])
    return pd.DatetimeIndex(frame.loc[frame.anchor >= INNER_START, "anchor"])


def build_frames(data_dir, scores, by_venue, margin_end):
    """Merge one score frame onto the shared proxy panel (one_to_one)."""
    panel = pd.read_parquet(Path(data_dir) / "weekly.parquet")
    proxy = build_proxy_panel(panel)
    proxy = attach_short_eligibility(proxy, by_venue)
    proxy["date"] = pd.to_datetime(proxy["date"])
    scores = scores.rename(columns={"score": "score_long"}).copy()
    scores["score_short"] = scores["score_long"]
    merged = proxy.merge(scores[["date", "symbol", "score_long", "score_short"]],
                         on=["date", "symbol"], how="left", validate="one_to_one")
    return merged


def rank_diagnostics(merged, eval_window):
    """§6.3 cross-section diagnostics on the holding-week simple returns."""
    rows = []
    sub = merged.loc[merged.date.isin(eval_window)]
    for day, group in sub.groupby("date"):
        known = group.loc[np.isfinite(group.score_long) & np.isfinite(group.long_r)]
        if len(known) < 30:
            continue
        ic = known.score_long.corr(known.long_r, method="spearman")
        ordered = known.sort_values(["score_long", "symbol"], ascending=[False, True])
        top20 = ordered.head(20).long_r
        mid = ordered.iloc[20:40].long_r
        ascending = known.sort_values(["score_long", "symbol"], ascending=[True, True])
        short_profit = -ascending.head(30).short_r
        rows.append(dict(date=day, rank_ic=ic, n_scored=int(len(known)),
                         n_missing_labels=int(len(group) - len(known)),
                         top20_ret=float(top20.mean()), rank21_40_ret=float(mid.mean()),
                         long_spread_bps=float((top20.mean() - mid.mean()) * 1e4),
                         short_bottom10=float(short_profit.head(10).mean()),
                         short_11_30=float(short_profit.iloc[10:30].mean()),
                         short_spread_bps=float((short_profit.head(10).mean()
                                                 - short_profit.iloc[10:30].mean()) * 1e4)))
    return pd.DataFrame(rows)


def short_coverage(merged, replay_days):
    sub = merged.loc[merged.date.isin(replay_days) & merged.short_ok]
    eligible = sub.groupby("date").symbol.nunique()
    return eligible


def gate_evaluate(stats, halves, mdd_cand, mdd_ctrl, stress_mean, coverage_ok,
                  conservative_ok=None, threshold=5.0):
    # mdd values are positive drawdown magnitudes: "worsened by <=2pp" means
    # the candidate's magnitude may exceed the control's by at most 0.02.
    checks = {
        "mean_bps_ge_threshold": bool(stats["mean_bps"] >= threshold),
        "both_halves_positive": bool(halves[0] > 0 and halves[1] > 0),
        "mdd_within_2pp": bool(mdd_cand <= mdd_ctrl + 0.02),
        "drop_max_week_positive": bool(stats.get("drop_max_week_bps", np.nan) > 0),
        "stress30_nonneg": bool(stress_mean >= 0),
        "short_coverage_ok": bool(coverage_ok),
    }
    if conservative_ok is not None:
        checks["conservative_missing_nonneg"] = bool(conservative_ok)
    checks["passed"] = all(checks.values())
    return checks


def run_pair(candidate_scores, control_scores, data_dir, by_venue, margin_end,
             replay_days, out_dir, tag, books=("L20", "S10", "LS50"), costs=(10.0, 30.0),
             extra_books=("L40",), seed_tag="s42_fused"):
    cand = build_frames(data_dir, candidate_scores, by_venue, margin_end)
    ctrl = build_frames(data_dir, control_scores, by_venue, margin_end)
    # Per-week finite-score masks must agree across the pair (runbook §6.3).
    for day in replay_days:
        a = cand.loc[cand.date == day, "score_long"].to_numpy()
        b = ctrl.loc[ctrl.date == day, "score_long"].to_numpy()
        assert np.isfinite(a).sum() == np.isfinite(b).sum(), f"finite mask mismatch at {day.date()}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    dev_days = pd.DatetimeIndex([d for d in replay_days if d.year == DEV_YEAR])
    dev_days = pd.DatetimeIndex(sorted(dev_days))
    replay_days = pd.DatetimeIndex(sorted(replay_days))
    weekly_rows, yearly_rows, comparisons, events = [], [], {}, []
    eligible_weeks = short_coverage(cand, dev_days)
    for book in list(books) + list(extra_books):
        spec = BOOKS[book]
        main_only = book in extra_books
        for cost in ([costs[0]] if main_only else costs):
            cand_w, cand_e = replay(cand, replay_days, spec, cost_bps=cost)
            ctrl_w, _ = replay(ctrl, replay_days, spec, cost_bps=cost)
            pair = pd.DataFrame(dict(date=cand_w.date, exit_week=cand_w.exit_week,
                                     cand_net=cand_w.net, ctrl_net=ctrl_w.net,
                                     diff=cand_w.net - ctrl_w.net,
                                     cand_missing=cand_w.missing, ctrl_missing=ctrl_w.missing,
                                     cand_turnover=cand_w.turnover, ctrl_turnover=ctrl_w.turnover,
                                     cand_filled_n=cand_w.filled_n, cand_intended_n=cand_w.intended_n,
                                     cand_short_n=cand_w.intended_short_n))
            pair.insert(0, "cost_bps", cost)
            pair.insert(0, "book", book)
            weekly_rows.append(pair)
            events.append(cand_e.assign(book=book, cost_bps=cost))
            dev = pair[pair.date.isin(dev_days)]
            cand_dev, ctrl_dev = dev.cand_net.to_numpy(), dev.ctrl_net.to_numpy()
            stats = paired_stats(cand_dev, ctrl_dev, dates=list(dev.date),
                                 label=f"{tag}|{book}|{cost:.0f}bps")
            halves = np.array_split(dev["diff"].to_numpy(), 2)
            half_means = [float(h.mean() * 1e4) for h in halves if len(h)]
            cand_mdd = perf_summary(cand_dev)["max_drawdown"]
            ctrl_mdd = perf_summary(ctrl_dev)["max_drawdown"]
            conservative = None
            filled = int(dev.cand_filled_n.sum())
            miss_frac = float(dev.cand_missing.sum() / filled) if filled else 0.0
            if miss_frac > 0.02:
                cand_c, _ = replay(cand, replay_days, spec, cost_bps=cost, missing_penalty=0.20)
                ctrl_c, _ = replay(ctrl, replay_days, spec, cost_bps=cost, missing_penalty=0.20)
                dev_c = cand_c[cand_c.date.isin(dev_days)].net.to_numpy() - \
                    ctrl_c[ctrl_c.date.isin(dev_days)].net.to_numpy()
                conservative = float(dev_c.mean() * 1e4)
            coverage_ok = True
            if spec[1] > 0:
                # Short coverage gate (§6.1): >=75% of dev decision weeks with
                # >=5 chosen, snapshot-backed short targets; >=15 such weeks per half.
                weeks5 = int((cand_w[cand_w.date.isin(dev_days)].intended_short_n >= 5).sum())
                half_counts = np.array_split(cand_w[cand_w.date.isin(dev_days)].intended_short_n.to_numpy(), 2)
                half15 = all(int((h >= 5).sum()) >= 15 for h in half_counts)
                coverage_ok = bool(len(dev_days) and weeks5 >= 0.75 * len(dev_days) and half15)
            # Stress at 30bps is a separate replay when cost == 10.
            if cost == 10.0:
                cand_s, _ = replay(cand, replay_days, spec, cost_bps=30.0)
                ctrl_s, _ = replay(ctrl, replay_days, spec, cost_bps=30.0)
                dev_s = cand_s[cand_s.date.isin(dev_days)].net.to_numpy() - \
                    ctrl_s[ctrl_s.date.isin(dev_days)].net.to_numpy()
                stress_mean = float(dev_s.mean() * 1e4)
            else:
                stress_mean = stats["mean_bps"]
            gate = gate_evaluate(stats, half_means, cand_mdd, ctrl_mdd, stress_mean,
                                 coverage_ok,
                                 conservative_ok=None if conservative is None else conservative >= 0)
            comparisons[f"{book}@{cost:.0f}"] = dict(
                paired=stats, half_means_bps=half_means,
                cand_mdd_2020=cand_mdd, ctrl_mdd_2020=ctrl_mdd,
                missing_fill_fraction=miss_frac, conservative_mean_bps=conservative,
                stress30_mean_bps=stress_mean,
                short_intended_weeks_ge5=int((cand_w[cand_w.date.isin(dev_days)].intended_short_n >= 5).sum()),
                short_coverage_gate=coverage_ok if spec[1] > 0 else None,
                gate=gate,
                perf_cand_2020=perf_summary(cand_dev), perf_ctrl_2020=perf_summary(ctrl_dev))
            for arm_name, net_series, wk in ((f"{tag}|cand", cand_w.net, cand_w),
                                             (f"{tag}|ctrl", ctrl_w.net, ctrl_w)):
                yearly_rows.extend(yearly_table(
                    wk[wk.date.isin(replay_days)].net,
                    wk[wk.date.isin(replay_days)].exit_week,
                    book, f"{arm_name}@{cost:.0f}", control_id="B0", phase="dev",
                    cost_scenario=f"{cost:.0f}bps", seed_or_ensemble=seed_tag))
    pd.concat(weekly_rows, ignore_index=True).to_parquet(out / "weekly_paired.parquet", index=False)
    pd.concat(events, ignore_index=True).to_parquet(out / "trade_events.parquet", index=False)
    pd.DataFrame(yearly_rows).to_csv(out / "yearly.csv", index=False)
    # Cross-section diagnostics for the candidate (dev window).
    diag = rank_diagnostics(cand, dev_days)
    diag.to_csv(out / "rank_diagnostics.csv", index=False)
    comparisons["rank_diagnostics"] = dict(
        mean_rank_ic=float(diag.rank_ic.mean()),
        mean_long_spread_bps=float(diag.long_spread_bps.mean()),
        mean_short_spread_bps=float(diag.short_spread_bps.mean()),
        n_weeks=int(len(diag)))
    # Top-20 overlap and score turnover, candidate vs control.
    overlap, sturn = [], []
    prev_c = prev_t = None
    for day in dev_days:
        a = cand.loc[cand.date == day]
        b = ctrl.loc[ctrl.date == day]
        ta = set(a.loc[np.isfinite(a.score_long)].nlargest(20, "score_long").symbol)
        tb = set(b.loc[np.isfinite(b.score_long)].nlargest(20, "score_long").symbol)
        overlap.append(len(ta & tb) / 20)
        sa = a.dropna(subset=["score_long"]).set_index("symbol").score_long.sort_index()
        if prev_c is not None and sa.index.equals(prev_c.index):
            sturn.append(float((sa.rank() - prev_c.rank()).abs().mean()))
        prev_c = sa
    comparisons["book_overlap"] = dict(mean_top20_overlap=float(np.mean(overlap)),
                                       mean_score_rank_turnover=float(np.mean(sturn)))
    # Short-side coverage table.
    cov = pd.DataFrame(dict(date=list(eligible_weeks.index), short_ok_names=eligible_weeks.to_numpy()))
    cov.to_csv(out / "coverage.csv", index=False)
    write_json(out / "comparisons.json", dict(tag=tag, seed_tag=seed_tag,
                                              n_replay_weeks=int(len(replay_days)),
                                              n_dev_weeks=int(len(dev_days)),
                                              comparisons=comparisons))
    return comparisons


def cmd_epochs(args):
    Path(args.out).mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(Path(args.run) / "dev_scores.parquet")
    scores["date"] = pd.to_datetime(scores["date"])
    by_venue = read_short_snapshots(DEV_CUT)
    eval_dates = evaluation_dates(args.data, args.out)
    days = replay_dates(eval_dates)
    # Inner selection window (§5): 2018-2019 decision weeks whose exit_week
    # is still inside 2019.
    exits = dict(zip(pd.to_datetime(eval_dates.anchor), pd.to_datetime(eval_dates.exit_week)))
    inner = [d for d in days if d.year in (2018, 2019)
             and pd.notna(exits.get(d)) and exits[d] < pd.Timestamp("2020-01-01")]
    panel = pd.read_parquet(Path(args.data) / "weekly.parquet")
    proxy = build_proxy_panel(panel)
    proxy = attach_short_eligibility(proxy, by_venue)
    proxy["date"] = pd.to_datetime(proxy["date"])
    rows = []
    for epoch in sorted(scores.epoch.unique()):
        s = scores[scores.epoch == epoch][["date", "symbol", "score"]]
        merged = proxy.merge(s.rename(columns={"score": "score_long"}),
                             on=["date", "symbol"], how="left", validate="one_to_one")
        merged["score_short"] = merged["score_long"]
        week, _ = replay(merged, pd.DatetimeIndex(sorted(inner)), BOOKS["L20"], cost_bps=10.0)
        r = week.net.to_numpy()
        sharpe = perf_summary(r)["sharpe"]
        rows.append(dict(epoch=int(epoch), sharpe=float(sharpe), n_weeks=int(len(r)),
                         mean_net_bps=float(r.mean() * 1e4)))
    best = max(rows, key=lambda x: (x["sharpe"], -x["epoch"]))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.out) / "epoch_selection.json",
               dict(rule="inner 2018-2019 exit<2020-01-01, L20 replay @10bps, max net Sharpe, ties->earlier",
                    epochs=rows, best_epoch=int(best["epoch"]), best_sharpe=float(best["sharpe"])))
    print(f"epoch_selection: best={best['epoch']} sharpe={best['sharpe']:.3f}", flush=True)


def cmd_compare(args):
    cand_runs, ctrl_runs = args.candidate, args.control
    assert args.epochs, "compare needs --epochs"
    if any("," in e for e in args.epochs):
        # comma form: either one window per run (candidates first, then
        # controls), one window for every run, or -- for equal-sized seed
        # groups -- one window per SEED applied pair-wise to candidate and
        # control of that seed (the §5 fixed per-seed windows).
        windows = [[int(e) for e in w.split(",")] for w in args.epochs]
        if len(windows) == 1:
            windows = windows * (len(cand_runs) + len(ctrl_runs))
        if len(windows) == len(cand_runs) == len(ctrl_runs):
            cand_windows = ctrl_windows = windows
        else:
            assert len(windows) == len(cand_runs) + len(ctrl_runs), \
                f"need {len(cand_runs) + len(ctrl_runs)} comma-joined windows, got {len(windows)}"
            cand_windows = windows[:len(cand_runs)]
            ctrl_windows = windows[len(cand_runs):]
    else:
        window = [int(e) for e in args.epochs]   # e.g. --epochs 2 3 4
        cand_windows = [window] * len(cand_runs)
        ctrl_windows = [window] * len(ctrl_runs)
    cand = load_scores(cand_runs, cand_windows, args.data)
    ctrl = load_scores(ctrl_runs, ctrl_windows, args.data)
    by_venue = read_short_snapshots(DEV_CUT)
    eval_dates = evaluation_dates(args.data)
    days = replay_dates(eval_dates)
    run_pair(cand, ctrl, args.data, by_venue, DEV_CUT, days, args.out,
             tag=f"{Path(cand_runs[0]).name}+{'/'.join(Path(r).name for r in cand_runs[1:]) or 'solo'}",
             books=("L20", "S10", "LS50"), costs=(10.0, 30.0), extra_books=("L40",),
             seed_tag="+".join(Path(r).name for r in cand_runs))
    print(f"compare written -> {args.out}", flush=True)


def incremental_curves(net, keys=("B0", "A", "B", "AB")):
    base, a, b, ab = (net[key] for key in keys)
    assert net.index.is_unique and net.index.is_monotonic_increasing
    assert net[list(keys)].notna().all().all()
    return pd.DataFrame({"vs_B0": ab-base, "vs_A": ab-a, "vs_B": ab-b,
                         "interaction": ab-a-b+base})


def cmd_combine(args):
    """§7.1 blend/route combinations of already-trained parents.

    blend: AB = equal-weight standard-score blend of the parent fusions;
    every book reports B0/A/B/AB and the interaction term.
    route: LS50 with the long leg from parent A and the short leg from parent
    B, against B0/B0, A/B0, B0/B, A/A, B/B (never two summed leg curves).
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    assert args.control, "combine needs --control (the B0 run)"
    by_venue = read_short_snapshots(DEV_CUT)
    eval_dates = evaluation_dates(args.data)
    days = pd.DatetimeIndex(sorted(replay_dates(eval_dates)))
    dev_days = pd.DatetimeIndex([d for d in days if d.year == DEV_YEAR])
    if args.epochs and any("," in e for e in args.epochs):
        windows = [[int(e) for e in w.split(",")] for w in args.epochs]
    else:
        windows = [[int(e) for e in args.epochs]] * (len(args.parents) + 1)
    assert len(windows) == len(args.parents) + 1, \
        "provide one epoch window per parent plus the control (comma-joined for multi-seed)"
    parent_scores = [load_scores([run], [window], args.data)
                     for run, window in zip(args.parents, windows)]
    b0_scores = load_scores(list(args.control), [windows[-1]], args.data)
    panel = pd.read_parquet(Path(args.data) / "weekly.parquet")
    proxy = build_proxy_panel(panel)
    proxy = attach_short_eligibility(proxy, by_venue)
    proxy["date"] = pd.to_datetime(proxy["date"])

    def frame_for(long_scores, short_scores):
        lg = long_scores.rename(columns={"score": "score_long"})[["date", "symbol", "score_long"]]
        sh = short_scores.rename(columns={"score": "score_short"})[["date", "symbol", "score_short"]]
        merged = lg.merge(sh, on=["date", "symbol"], validate="one_to_one")
        return proxy.merge(merged, on=["date", "symbol"], how="left", validate="one_to_one")

    rows, comparisons = [], {}
    if args.method == "blend":
        assert len(parent_scores) == 2, "pairwise blend needs exactly two parents"
        ab = blend_scores(parent_scores)
        for book in ("L20", "S10", "LS50"):
            spec = BOOKS[book]
            nets = {}
            for name, scores in (("B0", b0_scores), ("A", parent_scores[0]),
                                 ("B", parent_scores[1]), ("AB", ab)):
                merged = frame_for(scores, scores)
                week, _ = replay(merged, days, spec, cost_bps=10.0)
                nets[name] = week.set_index("date").net.reindex(days)
            net = pd.DataFrame(nets)
            curves = incremental_curves(net)
            dev = curves[curves.index.isin(dev_days)]
            rows.append(dict(method="blend", parents="+".join(Path(p).name for p in args.parents),
                             book=book, vs_B0_bps=float(dev.vs_B0.mean() * 1e4),
                             vs_A_bps=float(dev.vs_A.mean() * 1e4),
                             vs_B_bps=float(dev.vs_B.mean() * 1e4),
                             interaction_bps=float(dev.interaction.mean() * 1e4)))
            comparisons[book] = {k: perf_summary(dev[k].to_numpy()) for k in curves.columns}
    elif args.method == "route":
        assert len(parent_scores) == 2, "route needs exactly two parents: long, short"
        pairs = [("AB", parent_scores[0], parent_scores[1]),
                 ("A/B0", parent_scores[0], b0_scores),
                 ("B0/B", b0_scores, parent_scores[1]),
                 ("A/A", parent_scores[0], parent_scores[0]),
                 ("B/B", parent_scores[1], parent_scores[1]),
                 ("B0/B0", b0_scores, b0_scores)]
        conflicts = []
        for name, lg_scores, sh_scores in pairs:
            merged = frame_for(lg_scores, sh_scores)
            # Planned longs are excluded from the short pool inside
            # pick_targets; count the would-be collisions before exclusion.
            for day in days:
                g = merged.loc[merged.date == day]
                known = g.loc[np.isfinite(g.score_long) & np.isfinite(g.score_short)]
                longs = set(known.nlargest(20, "score_long").symbol)
                liquid = known.loc[np.isfinite(known.amount) & known.amount.gt(0)]
                pool = liquid.loc[liquid.amount.rank(pct=True).gt(.6) & liquid.short_ok]
                shorts = set(pool.nsmallest(10, "score_short").symbol)
                if name == "AB" and longs & shorts:
                    conflicts.append(dict(date=day, n_conflicts=len(longs & shorts)))
            week, _ = replay(merged, days, BOOKS["LS50"], cost_bps=10.0)
            net_col = week.set_index("date").net.reindex(days)
            dev = net_col[net_col.index.isin(dev_days)]
            rows.append(dict(method="route", parents=name,
                             book="LS50", vs_B0_bps=float("nan"), vs_A_bps=float("nan"),
                             vs_B_bps=float("nan"), interaction_bps=float("nan"),
                             perf=perf_summary(dev.to_numpy())))
            comparisons[name] = dict(perf_2020=perf_summary(dev.to_numpy()),
                                     n_weeks=int(len(dev)))
        if conflicts:
            pd.DataFrame(conflicts).to_csv(out / "route_conflicts.csv", index=False)
            comparisons["n_conflict_weeks"] = int(pd.DataFrame(conflicts).date.nunique())
    else:
        raise ValueError("joint training is a new training run, not an eval combination")
    pd.DataFrame(rows).to_csv(out / "combinations.csv", index=False)
    write_json(out / "comparisons.json", dict(method=args.method,
                                              parents=[str(p) for p in args.parents],
                                              control=str(args.control[0]),
                                              comparisons=comparisons))
    print(f"combine written -> {out}", flush=True)


def cmd_dates(args):
    Path(args.out).mkdir(parents=True, exist_ok=True)
    frame = evaluation_dates(args.data, args.out)
    print(f"evaluation_dates: {int(frame.included.sum())} included of {len(frame)} anchors", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["dates", "epochs", "compare", "combine"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--candidate", type=Path, nargs="+")
    parser.add_argument("--control", type=Path, nargs="+")
    parser.add_argument("--parents", type=Path, nargs="+")
    parser.add_argument("--epochs", type=str, nargs="+")
    parser.add_argument("--method", choices=["blend", "route", "joint"], default="blend")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "dates":
        cmd_dates(args)
    elif args.stage == "epochs":
        cmd_epochs(args)
    elif args.stage == "compare":
        cmd_compare(args)
    else:
        cmd_combine(args)


if __name__ == "__main__":
    main()
