"""Select single epochs by leg on 2018–2020, then replay locked win3 scores."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ensemble_best_window import window_for3
from eval_dataset_experiments import compare_books, dated, json_values, local_time, metrics, publish
from eval_weekly_architecture import BOOKS, fuse_epochs, fuse_seeds, replay
from r1111_common import ROOT, sha256, write_json
from quant_mvp.financial_return import is_financial, distribution_metadata, MIXTURE
from quant_mvp.financial_scoring import (validate_predictions, fuse_probability_epochs,
                                        fuse_probability_seeds, probability_diagnostics)
from quant_mvp.training_state import require_committed_epochs
from score_dataset_standard import financial_run_identity, verify_financial_scores

START = pd.Timestamp("2018-01-01")
VAL_END = pd.Timestamp("2021-01-01")
TEST_START = pd.Timestamp("2021-01-08")
TAPE = ROOT / "outputs/dataset_v2/replay_inputs/raw_open.parquet"
ELIGIBILITY = ROOT / "outputs/dataset_v2/replay_inputs_full/eligibility.parquet"
KEY = ["date", "symbol"]
COSTS = ((10, .0835), (30, .0835), (10, .12), (30, .12))
SCOPES = {"L20": BOOKS["L20"], "S10": BOOKS["S10"],
          "LS50": BOOKS["LS50"], "LONG_CASH": (20, 0, .5, 0.)}
LIMITATIONS = [
    "2018–2020 validation and 2021+ test are previously exposed historical research.",
    "Q1 latest-history input, fixed 580-name survivor universe and retrospective industry.",
    "Adjusted returns use actual scheduled opening sessions from the snapshot labels.",
    "Raw price entry flags approximate historical limits. An exit at a blocked or absent open is not a verified trade.",
    "Opening marks where available remain in the proxy. Missing filled-position outcomes are marked flat in the main "
    "proxy and at an adverse 20% asset move in sensitivity. This weekly replay does not carry blocked exit orders.",
    "SH and SZ lists use strict prior dated proxies; the Shanghai list is derived from historical short activity. "
    "Missing venue list means ineligible and unused budget stays cash.",
    "Stock loan inventory, margin, transaction taxes and actual auction fills were not verified.",
]


def valid_scores(path, start, end, epochs, cfg=None):
    scores = dated(pd.read_parquet(path))
    if (cfg is not None and is_financial(cfg)) or 'mu_log' in scores:
        validate_predictions(scores, cfg)
    if (scores.empty or scores.symbol.isna().any() or scores.duplicated([*KEY, "epoch"]).any()
            or set(scores.epoch) != set(epochs) or np.isinf(scores.score).any()):
        raise ValueError(f"invalid score keys, checkpoint coverage or infinities: {path}")
    if not (scores.date.ge(start) & scores.date.lt(end)).all():
        raise ValueError(f"score date outside {start.date()}–{end.date()}: {path}")
    counts = scores.groupby(KEY).score.agg(["size", "count"])
    if not counts["size"].eq(len(epochs)).all() or not counts["count"].isin([0, len(epochs)]).all():
        raise ValueError(f"checkpoint score masks differ: {path}")
    return scores


def snapshot(path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("quality_level") != "Q1" or manifest.get("evidence_mode") != "latest_history":
        raise ValueError("expected a Q1 latest-history snapshot")
    for name in ("labels.parquet", "sessions.parquet", "weekly.parquet"):
        if sha256(path / name) != manifest["files"][name]:
            raise ValueError(f"snapshot asset changed: {name}")
    return manifest


def opening_matches(frame, tape, when, prefix):
    opening = frame[["symbol", when]].copy()
    opening["trade_date"] = local_time(opening.pop(when)).tz_localize(None).normalize()
    values = opening.merge(tape, on=["trade_date", "symbol"], how="left", validate="many_to_one", sort=False)
    for side in ("buy", "sell"):
        frame[f"{prefix}_{side}"] = values[f"can_{side}"].eq(True).to_numpy()


def phase_proxy(directory, manifest, tape_path, eligibility_path, *, phase):
    cutoff = local_time(manifest["end_exclusive"])
    start = START if phase == "validation" else TEST_START
    if phase == "validation" and cutoff != local_time(TEST_START):
        raise ValueError("selection needs a physically frozen snapshot ending at 2021-01-08")
    end = VAL_END if phase == "validation" else cutoff.tz_localize(None).normalize()
    if end <= start:
        raise ValueError("empty phase")
    filters = [("date", ">=", start), ("date", "<", end)]
    labels = dated(pd.read_parquet(directory / "labels.parquet", filters=filters))
    if labels.duplicated(KEY).any():
        raise ValueError("duplicate snapshot label keys")
    schedule = dated(pd.read_parquet(directory / "sessions.parquet", filters=filters))
    times = ["decision_at", "entry_at", "exit_at"]
    schedule = schedule[["date", *times]].sort_values("date")
    if schedule.date.duplicated().any():
        raise ValueError("duplicate session anchors")
    decision, entry, exit_ = (local_time(schedule[k]) for k in times)
    # This is the snapshot's conservative vendor availability bound.
    mature_at = exit_.normalize() + pd.Timedelta(hours=18)
    mature = (decision < entry) & (entry < exit_) & (mature_at < cutoff)
    schedule = schedule.loc[mature].reset_index(drop=True)
    dates = pd.DatetimeIndex(schedule.date)
    if dates.empty:
        raise ValueError("no mature exchange sessions for phase")
    labels = labels.loc[labels.date.isin(dates)].merge(
        schedule, on="date", validate="many_to_one", suffixes=("", "_calendar"))
    for name in times:
        if not local_time(labels[name]).equals(local_time(labels.pop(f"{name}_calendar"))):
            raise ValueError(f"label/calendar mismatch: {name}")
    avail = local_time(labels.label_available_at)
    observed = (labels.valid & (avail >= local_time(labels.exit_at)) & (avail < cutoff)
                & np.isfinite(labels.ret))
    if not np.array_equal(observed, labels.valid):
        raise ValueError("snapshot marked a label valid without a known pre-cutoff outcome")
    frame = labels[[*KEY, *times]].copy()
    frame["label_available_at"] = labels.label_available_at.where(observed)
    frame["long_r"] = frame["short_r"] = np.expm1(labels.ret.where(observed))
    amounts = dated(pd.read_parquet(directory / "weekly.parquet", columns=[*KEY, "amount"], filters=filters))
    frame = frame.merge(amounts, on=KEY, how="left", validate="one_to_one")
    if frame.empty or frame.date.nunique() != len(schedule):
        raise ValueError("missing label panel dates")

    raw = pd.read_parquet(tape_path, columns=["date", "symbol", "can_buy", "can_sell"],
                          filters=[("date", ">=", schedule.entry_at.min().tz_localize(None).normalize()),
                                   ("date", "<=", schedule.exit_at.max().tz_localize(None).normalize())])
    tape = dated(raw).rename(columns={"date": "trade_date"})
    if tape.empty or tape.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("opening tape is empty or ambiguous")
    if not all(pd.api.types.is_bool_dtype(tape[k]) and tape[k].notna().all() for k in ("can_buy", "can_sell")):
        raise ValueError("opening flags must be explicit booleans")
    opening_matches(frame, tape, "entry_at", "entry")
    opening_matches(frame, tape, "exit_at", "exit")
    frame["long_entry"], frame["short_entry"] = frame.entry_buy, frame.entry_sell
    frame["long_close"], frame["short_close"] = ~frame.exit_sell, ~frame.exit_buy
    frame["exit_week"] = (local_time(frame.exit_at).tz_localize(None)
                          .to_period("W-FRI").end_time.normalize())
    frame["borrow_days"] = (frame.exit_at - frame.entry_at).dt.total_seconds() / 86400
    columns = [*KEY, "short_ok", "short_snapshot", "venue", "eligibility_known"]
    eligibility = dated(pd.read_parquet(eligibility_path, columns=columns, filters=filters))
    eligibility = eligibility.loc[eligibility.date.isin(dates)].sort_values(KEY).reset_index(drop=True)
    flags = ("short_ok", "eligibility_known")
    if (eligibility.duplicated(KEY).any() or
            not all(pd.api.types.is_bool_dtype(eligibility[k]) and eligibility[k].notna().all() for k in flags)):
        raise ValueError("invalid eligibility tape")
    if not np.array_equal(eligibility.eligibility_known, eligibility.short_snapshot.notna()):
        raise ValueError("known eligibility disagrees with archived snapshots")
    if (eligibility.short_ok & ~eligibility.eligibility_known).any():
        raise ValueError("a security is borrow eligible without a known list")
    before = eligibility.short_snapshot.isna() | (local_time(eligibility.short_snapshot) < local_time(eligibility.date))
    if not before.all():
        raise ValueError("same-day or future borrowing list used")
    frame = frame.merge(eligibility, on=KEY, how="left", validate="one_to_one")
    if frame.short_ok.isna().any():
        raise ValueError("eligibility tape is incomplete on the common phase keys")
    coverage = frame.loc[frame.eligibility_known].groupby("date").venue.nunique()
    schedule["short_venue_count"] = schedule.date.map(coverage).fillna(0).astype(int)
    schedule["label_available_at"] = local_time(schedule.exit_at).normalize() + pd.Timedelta(hours=18)
    digest = payload_digest(eligibility.to_json(orient="split", date_format="iso"))
    return frame, schedule, digest


def score_pair(frame, long_score, short_score, schedule):
    long = long_score[[*KEY, 'score']].rename(columns={"score": "score_long"})
    short = short_score[[*KEY, 'score']].rename(columns={"score": "score_short"})
    common = long.merge(short, on=KEY, how="outer", validate="one_to_one")
    if not common.score_long.notna().equals(common.score_short.notna()):
        raise ValueError("long and short score masks differ")
    source = frame.merge(common, on=KEY, how="left", validate="one_to_one")
    known = source.loc[source.score_long.notna(), "date"]
    if set(known) != set(schedule.date):
        raise ValueError("locked scores do not cover all common mature decision dates")
    return source


def run_book(frame, schedule, spec, cost, borrow, penalty=0.):
    week, events = replay(frame, pd.DatetimeIndex(schedule.date), spec, cost_bps=cost,
                          borrow_ann=borrow, missing_penalty=penalty)
    week = week.rename(columns={"close_fallback": "exit_blocked_or_absent"})
    week = week.merge(schedule, on="date", validate="one_to_one")
    if "exit_close_proxy" in events:
        events = events.rename(columns={"exit_close_proxy": "exit_blocked_or_absent"})
    return week, events


def select_epoch(tables):
    # The first occurrence wins exactly equal validation Sharpes.
    return max(tables, key=lambda epoch: (tables[epoch]["sharpe"], -epoch))


def select(args):
    manifest = snapshot(args.snapshot)
    config_path = args.run / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    epochs = list(range(1, int(config["epochs"]) + 1))
    if len(epochs) < 3:
        raise ValueError("centered windows require at least three checkpoints")
    require_committed_epochs(args.run, epochs)
    if is_financial(config):
        financial_run_identity(args.run,config,json.loads((args.run/'preprocessing.json').read_text(encoding='utf-8')))
        verify_financial_scores(args.scores,args.run,config,epochs)
    scores = valid_scores(args.scores, START, TEST_START, epochs, config)
    frame, schedule, eligibility_digest = phase_proxy(args.snapshot, manifest, args.tape, args.eligibility,
                                                       phase="validation")
    objective, details = {"L20": {}, "S10": {}}, []
    cost, borrow = COSTS[0]
    both = schedule.loc[schedule.short_venue_count.eq(2)]
    for e, rows in scores.groupby("epoch"):
        score = rows[[*KEY, "score"]]
        source = score_pair(frame, score, score, schedule)
        for book in objective:
            weekly, _ = run_book(source, schedule, SCOPES[book], cost, borrow)
            result = metrics(weekly)
            result.update(missing_filled_positions=int(weekly.missing.sum()),
                          exit_blocked_or_absent=int(weekly.exit_blocked_or_absent.sum()),
                          filled_weeks=int(weekly.filled_n.gt(0).sum()))
            objective[book][int(e)] = result
            details.append({"book": book, "epoch": int(e), "scope": "full_calendar", **result})
            if book == "S10" and 0 < len(both) < len(schedule):
                week_both, _ = run_book(source, both, SCOPES[book], cost, borrow)
                details.append({"book": book, "epoch": int(e), "scope": "two_venue_subset",
                                **metrics(week_both), "missing_filled_positions": int(week_both.missing.sum()),
                                "exit_blocked_or_absent": int(week_both.exit_blocked_or_absent.sum()),
                                "filled_weeks": int(week_both.filled_n.gt(0).sum())})
    best_long, best_short = (select_epoch(objective[book]) for book in ("L20", "S10"))
    if not any(x["filled_weeks"] for x in objective["S10"].values()):
        raise ValueError("no validation short-eligibility proxy coverage for any checkpoint")
    long, short = (window_for3(e, len(epochs)) for e in (best_long, best_short))
    locked_epochs = sorted(set(long + short))
    lock = {"run": args.run.name, "seed": int(config["seed"]),
            "budget": len(epochs),
            "arm": run_arm(args.run.name), "best_long_epoch": best_long, "best_short_epoch": best_short,
            "long_epochs": long, "short_epochs": short, "epochs": locked_epochs,
            "config_sha256": sha256(config_path),
            "preprocessing_sha256": sha256(args.run / "preprocessing.json"),
            "checkpoint_sha256": {str(e): sha256(args.run / f"epoch_{e}.pt") for e in locked_epochs},
            "validation_snapshot_id": manifest["snapshot_id"],
            "validation_scores_sha256": sha256(args.scores), "tape_sha256": sha256(args.tape),
            "validation_cutoff": str(TEST_START.date()),
            "cost_bps": cost, "borrow_annual": borrow, "validation_weeks": len(schedule),
            "two_venue_proxy_weeks": int(schedule.short_venue_count.eq(2).sum()),
            "no_venue_proxy_weeks": int(schedule.short_venue_count.eq(0).sum()),
            "validation_start": str(schedule.date.iloc[0].date()),
            "validation_end": str(schedule.date.iloc[-1].date()),
            "selection": "single checkpoint per leg on full mature 2018–2020 net Sharpe; "
                         "ties choose earlier; best±1 clipped to three; absent venue is cash"}
    lock["short_information_sets_sha256"] = eligibility_digest
    if is_financial(config):
        lock.update(distribution_metadata(config, json.loads(
            (args.run/'preprocessing.json').read_text(encoding='utf-8'))))

    def save(stage):
        write_json(stage / "lock.json", lock)
        pd.DataFrame(details).to_csv(stage / "validation_single_epochs.csv", index=False)
        schedule[["date", "short_venue_count"]].to_parquet(stage / "short_coverage.parquet", index=False)

    publish(args.out, save)
    print(f"{args.run.name}: L20 {long} S10 {short}; {len(schedule)} validation weeks", flush=True)


def run_arm(name):
    match = re.fullmatch(r"(?:(?:research|budget\d+)_)?(.+?)(?:_F0)?_s\d+_a\d+", name)
    return match[1] if match else name


def payload_digest(value):
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def score_members(specs, val, test, tape, validation_eligibility_digest):
    result, hashes = {}, {}
    tape_digest = sha256(tape)
    for run, val_path, lock_path, test_path, declared in specs:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        config = run / "config.yaml"
        epochs = sorted(set(lock["long_epochs"] + lock["short_epochs"]))
        expected = {"run": run.name, "config_sha256": sha256(config),
                    "preprocessing_sha256": sha256(run / "preprocessing.json"),
                    "validation_snapshot_id": val["snapshot_id"], "validation_scores_sha256": sha256(val_path),
                    "tape_sha256": tape_digest, "epochs": epochs,
                    "short_information_sets_sha256": validation_eligibility_digest}
        arm = run_arm(run.name)
        if any(lock.get(k) != v for k, v in expected.items()) or lock["arm"] not in (arm, run.name):
            raise ValueError(f"selection lock mismatch: {run}")
        lock = dict(lock, arm=arm)
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        n = int(cfg['epochs'])
        require_committed_epochs(run, epochs)
        if is_financial(cfg):
            financial_run_identity(run,cfg,json.loads((run/'preprocessing.json').read_text(encoding='utf-8')))
            verify_financial_scores(val_path,run,cfg,list(range(1,n+1)))
            verify_financial_scores(test_path,run,cfg,epochs)
            expected_probability = distribution_metadata(cfg, json.loads(
                (run/'preprocessing.json').read_text(encoding='utf-8')))
            if any(lock.get(k) != v for k,v in expected_probability.items()):
                raise ValueError('locked probability semantics changed')
        if lock["budget"] != n or (declared and any(lock[k] != declared[k] for k in ("arm", "seed", "budget"))):
            raise ValueError(f"catalog and frozen run metadata disagree: {run}")
        if any(not 1 <= lock[f"best_{k}_epoch"] <= n for k in ("long", "short")):
            raise ValueError(f"best epoch outside run budget: {run}")
        if any(lock[k + "_epochs"] != window_for3(lock[f"best_{k}_epoch"], n) for k in ("long", "short")):
            raise ValueError(f"selection window is not the locked centered window: {run}")
        for e in epochs:
            if sha256(run / f"epoch_{e}.pt") != lock["checkpoint_sha256"][str(e)]:
                raise ValueError(f"checkpoint changed: {run} epoch {e}")
        val_scores = valid_scores(val_path, START, TEST_START, list(range(1, n + 1)), cfg)
        test_scores = valid_scores(test_path, TEST_START,
                                   local_time(test["end_exclusive"]).tz_localize(None).normalize(), epochs, cfg)
        if run.name in result:
            raise ValueError("duplicate run")
        result[run.name] = (lock, {
            phase: {leg: (fuse_probability_epochs(frame, lock[f"{leg}_epochs"], run.name) if is_financial(cfg)
                          else fuse_epochs(frame, lock[f"{leg}_epochs"])) for leg in ("long", "short")}
            for phase, frame in (("validation", val_scores), ("test", test_scores))})
        if is_financial(cfg) and cfg.get('distribution_family', MIXTURE) == MIXTURE:
            for phase, frame in (("validation", val_scores), ("test", test_scores)):
                for leg in ('long', 'short'):
                    legacy = fuse_epochs(frame, lock[f'{leg}_epochs']).rename(columns={'score':'legacy_z_score'})
                    result[run.name][1][phase][leg] = result[run.name][1][phase][leg].merge(
                        legacy, on=KEY, validate='one_to_one')
        hashes[run.name] = {str(path): sha256(path) for path in (config, val_path, lock_path, test_path)}
    return result, hashes


def matched_portfolios(members):
    portfolios = {name: (data, {"arm": lock["arm"], "seed": lock["seed"], "ensemble": False,
                               "budget": lock["budget"]})
                  for name, (lock, data) in members.items()}
    arms = {}
    for lock, data in members.values():
        grouped = arms.setdefault((lock["arm"], lock["budget"]), {})
        if lock["seed"] in grouped:
            raise ValueError("duplicate seed in one ensemble")
        grouped[lock["seed"]] = data
    for (arm, budget), seeds in arms.items():
        if set(seeds) != {42, 43, 44}:
            continue
        financial = 'component_mu' in next(iter(seeds.values()))['validation']['long']
        fuse = fuse_probability_seeds if financial else fuse_seeds
        data = {phase: {leg: fuse([member[phase][leg] for member in seeds.values()])
                        for leg in ("long", "short")} for phase in ("validation", "test")}
        portfolios[f"{arm}_e{budget}_ensemble"] = (data, {"arm": arm, "seed": None, "ensemble": True,
                                                        "budget": budget, "members": sorted(seeds)})
    return portfolios


def compare_portfolios(portfolios):
    pairs = []
    for name, (_, info) in portfolios.items():
        budget = info["budget"]
        if info["ensemble"]:
            eligible = {k: v for k, (_, v) in portfolios.items() if v["budget"] == budget
                        and v["ensemble"] and v["members"] == info["members"]}
        else:
            eligible = {k: v for k, (_, v) in portfolios.items() if v["budget"] == budget
                        and v["seed"] == info["seed"] and not v["ensemble"]}
        by_arm = {v["arm"]: k for k, v in eligible.items()}
        controls = ["B0"] + (["CS"] if info["arm"] not in ("B0", "CS") else [])
        controls += [f"{info['arm']}_SELF"] if info["arm"] in ("CB", "ADN") else []
        controls += {'P2':['P0','P1'], 'P3':['P2'], 'P4':['P2'], 'P5':['P3','P4'],
                     'P6':['P4'], 'P6GAP':['P6'], 'P6TAIL':['P6'], 'FOXCONST':['FOX'], 'P4GROUP':['P4']}.get(info['arm'], [])
        controls += ['D3_ZERO'] if info['arm'] == 'D3' else []
        controls += {'KITDAILY': ['KITWEEK'], 'KITCANDLE': ['KITDAILY', 'KITWEEK']}.get(info['arm'], [])
        pairs.extend((name, by_arm[a]) for a in controls if a in by_arm and by_arm[a] != name)
    return pairs


def evaluate(args):
    val = snapshot(args.validation_snapshot)
    if local_time(val["end_exclusive"]) != local_time(TEST_START):
        raise ValueError("validation snapshot is not physically frozen before test")
    # Validate the locked test epochs before any portfolio is formed.
    test = snapshot(args.snapshot)
    validation_proxy = phase_proxy(args.validation_snapshot, val, args.tape, args.eligibility, phase="validation")
    specs = ([(Path(item["run"]), Path(item["validation_scores"]), Path(item["lock"]),
               Path(item["test_scores"]), item)
              for item in json.loads(args.catalog.read_text(encoding="utf-8"))["members"]]
             if args.catalog else [(*values, None) for values in args.member])
    if not specs:
        raise ValueError("empty portfolio catalog")
    members, hashes = score_members(specs, val, test, args.tape, validation_proxy[2])
    portfolios = matched_portfolios(members)
    pairs = compare_portfolios(portfolios)
    report, tables, differences, trades, coverage = [], [], [], [], {}
    probability_rows, probability_daily, density_members, legacy_diagnostics = [], [], [], []
    for phase, directory, manifest in (("validation", args.validation_snapshot, val), ("test", args.snapshot, test)):
        proxy, schedule, eligibility_digest = (
            validation_proxy if phase == "validation" else
            phase_proxy(directory, manifest, args.tape, args.eligibility, phase=phase))
        coverage[phase] = {"weeks": len(schedule), "two_venue_proxy_weeks": int(schedule.short_venue_count.eq(2).sum()),
                           "no_venue_proxy_weeks": int(schedule.short_venue_count.eq(0).sum()),
                           "first_decision": str(schedule.date.iloc[0].date()),
                           "last_decision": str(schedule.date.iloc[-1].date()),
                           "short_information_sets_sha256": eligibility_digest}
        for name, (data, _) in portfolios.items():
            if 'component_mu' not in data[phase]['long']:
                continue
            labels = proxy[KEY].assign(realized_return=np.log1p(proxy.long_r))
            for leg in ('long','short'):
                fused = data[phase][leg]
                density_members.append(fused.assign(phase=phase,portfolio=name,leg=leg))
                obs, daily = probability_diagnostics(fused, labels)
                if len(daily):
                    probability_rows.append(obs.assign(phase=phase,portfolio=name,leg=leg))
                    probability_daily.append(daily.assign(phase=phase,portfolio=name,leg=leg))
            if all('legacy_z_score' in data[phase][leg] for leg in ('long','short')):
                legacy = {leg:data[phase][leg][KEY+['legacy_z_score']].rename(columns={'legacy_z_score':'score'})
                          for leg in ('long','short')}
                source = score_pair(proxy,legacy['long'],legacy['short'],schedule)
                for scope in ('L20','S10','LS50'):
                    week,_ = run_book(source,schedule,SCOPES[scope],*COSTS[0])
                    legacy_diagnostics.append(dict(phase=phase,portfolio=name,book=scope,**metrics(week)))
        scores = {name: score_pair(proxy, data[phase]["long"], data[phase]["short"], schedule)
                  for name, (data, _) in portfolios.items()}
        masks = [source.score_long.notna().to_numpy() for source in scores.values()]
        if any(not np.array_equal(masks[0], current) for current in masks[1:]):
            raise ValueError(f"{phase}: portfolio score masks differ; pairwise sample is not matched")
        both = schedule.loc[schedule.short_venue_count.eq(2)]
        for cost, borrow in COSTS:
            books = {}
            for name, source in scores.items():
                _, info = portfolios[name]
                for scope, spec in SCOPES.items():
                    for penalty in ((0., .20) if (cost, borrow) == COSTS[0] else (0.,)):
                        book, events = run_book(source, schedule, spec, cost, borrow, penalty)
                        key = dict(phase=phase, portfolio=name, book=scope, trade_bps=cost,
                                   borrow_annual=borrow, missing_penalty=penalty, coverage_scope="full_calendar")
                        books[(name, scope, penalty)] = book
                        report.append(dict(**key, arm=info["arm"], seed=info["seed"], ensemble=info["ensemble"],
                                           budget=info["budget"],
                                           **metrics(book), n_two_venue_proxy_weeks=coverage[phase]["two_venue_proxy_weeks"],
                                           missing_filled_positions=int(book.missing.sum()),
                                           exit_blocked_or_absent=int(book.exit_blocked_or_absent.sum()),
                                           filled_weeks=int(book.filled_n.gt(0).sum())))
                        tables.append(book.assign(**key))
                        if (cost, borrow) == COSTS[0] and penalty == 0.:
                            trades.append(events.assign(**key))
                # Independently reinitialize the 2020 two-venue comparison at its first eligible date.
                if (cost, borrow) == COSTS[0] and 0 < len(both) < len(schedule):
                    for scope in ("S10", "LS50", "LONG_CASH"):
                        subset, _ = run_book(source, both, SCOPES[scope], cost, borrow)
                        report.append(dict(phase=phase, portfolio=name, book=scope, trade_bps=cost,
                                           borrow_annual=borrow, missing_penalty=0.,
                                           coverage_scope="two_venue_subset", arm=info["arm"], seed=info["seed"],
                                           ensemble=info["ensemble"], budget=info["budget"], **metrics(subset),
                                           n_two_venue_proxy_weeks=len(both), filled_weeks=int(subset.filled_n.gt(0).sum()),
                                           missing_filled_positions=int(subset.missing.sum()),
                                           exit_blocked_or_absent=int(subset.exit_blocked_or_absent.sum())))
            comparisons = [(a, b, scope, scope) for a, b in pairs for scope in ("L20", "S10", "LS50")]
            comparisons.extend((name, name, "LS50", "LONG_CASH") for name in scores)
            for a, b, scope, control_scope in comparisons:
                for penalty in ((0., .20) if (cost, borrow) == COSTS[0] else (0.,)):
                    result, weekly = compare_books(books[(a, scope, penalty)], books[(b, control_scope, penalty)])
                    lo, hi = (result["paired"]["ci_L8"][key] * 1e4 for key in ("lo", "hi"))
                    differences.append({"phase": phase, "portfolio": a, "control": b,
                                        "budget": portfolios[a][1]["budget"],
                                        "book": scope, "control_book": control_scope, "trade_bps": cost,
                                        "borrow_annual": borrow, "missing_penalty": penalty,
                                        "weeks": len(weekly), "mean_delta_bps": result["paired"]["mean_bps"],
                                        "delta_L8_lo_bps": lo, "delta_L8_hi_bps": hi,
                                        "candidate_cagr": result["candidate"]["cagr"],
                                        "control_cagr": result["control"]["cagr"],
                                        "cagr_delta_pp": 100 * (result["candidate"]["cagr"] - result["control"]["cagr"])})
        print(f"{phase}: {len(schedule)} matched weeks, {len(scores)} portfolios", flush=True)

    summary = {"classification": "historical_research_proxy", "coverage": coverage, "limitations": LIMITATIONS,
               "snapshot_ids": {"validation": val["snapshot_id"], "test": test["snapshot_id"]},
               "tape_sha256": sha256(args.tape), "eligibility_path": str(args.eligibility),
               "member_inputs": hashes, "comparisons": pairs,
               "members": {name: lock for name, (lock, _) in members.items()}}
    if args.catalog:
        summary["catalog_sha256"] = sha256(args.catalog)
    table, paired = pd.DataFrame(report), pd.DataFrame(differences)
    all_weeks = pd.concat(tables, ignore_index=True)
    keys = ["phase", "portfolio", "book", "trade_bps", "borrow_annual", "missing_penalty", "coverage_scope"]
    yearly = []
    for basis, year in (("decision_year", all_weeks.date.dt.year), ("exit_year", all_weeks.exit_at.dt.year)):
        groups = all_weeks.assign(year=year).groupby([*keys, "year"], sort=True)
        yearly.append(groups.net.agg(weeks="size", net_return=lambda x: np.expm1(np.log1p(x).sum()))
                      .reset_index().assign(basis=basis))
    lines = ["# 三年验证选轮与历史测试", "",
             "2018–2020 已读验证，多空各按本腿净 Sharpe 选单轮并固定连续三轮。测试使用 2021 年以来成熟周。",
             "结果为 Q1 固定幸存股池上的周级执行代理。券源、历史涨跌停和受阻退出成交缺少完整实证。", "",
             "| 阶段 | 组合 | 腿 | 净年化 | 净 Sharpe | 周数 | 缺失持仓结果 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    main = table.loc[table.trade_bps.eq(10) & table.borrow_annual.eq(.0835) & table.missing_penalty.eq(0)
                     & table.coverage_scope.eq("full_calendar")]
    for row in main.itertuples():
        lines.append(f"| {row.phase} | {row.portfolio} | {row.book} | {row.cagr:.2%} | "
                     f"{row.sharpe:.3f} | {row.weeks} | {row.missing_filled_positions} |")
    lines += ["", "配对表包括对同种子 B0、CS 和各自 self 的净年化差，另列多空对同多头加现金。"
              "10/30 bp、8.35%/12% 费用敏感性与缺失持仓结果的 −20% 压力情景详见 CSV。"]
    main_pairs = paired.loc[paired.trade_bps.eq(10) & paired.borrow_annual.eq(.0835)
                            & paired.missing_penalty.eq(0)]
    lines += ["", "| 阶段 | 组合 vs 对照 | 腿 vs 对照腿 | 净年化差（百分点） | 每周净差（bp） | 周数 |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in main_pairs.itertuples():
        lines.append(f"| {row.phase} | {row.portfolio} vs {row.control} | {row.book} vs {row.control_book} | "
                     f"{row.cagr_delta_pp:+.2f} | {row.mean_delta_bps:+.2f} | {row.weeks} |")

    def save(stage):
        if density_members:
            pd.concat(density_members,ignore_index=True).to_parquet(stage/'probability_members.parquet',index=False)
        if probability_rows:
            observations = pd.concat(probability_rows,ignore_index=True)
            observations.to_parquet(stage/'probability_observations.parquet',index=False)
            daily = pd.concat(probability_daily,ignore_index=True)
            daily.to_csv(stage/'probability_daily.csv',index=False)
            daily.groupby(['phase','portfolio','leg']).mean(numeric_only=True).to_csv(stage/'probability_summary.csv')
            observations.groupby(['phase','portfolio','leg','date','risk_bucket'])[
                ['mean_error','squared_error','nll']].mean().groupby(
                ['phase','portfolio','leg','risk_bucket']).mean().to_csv(stage/'probability_risk_buckets.csv')
        if legacy_diagnostics:
            pd.DataFrame(legacy_diagnostics).to_csv(stage/'legacy_z_diagnostic.csv',index=False)
        table.to_csv(stage / "metrics.csv", index=False)
        paired.to_csv(stage / "paired.csv", index=False)
        all_weeks.to_parquet(stage / "weekly.parquet", index=False)
        pd.concat(yearly, ignore_index=True).to_csv(stage / "yearly.csv", index=False)
        pd.concat(trades, ignore_index=True).to_parquet(stage / "trade_events.parquet", index=False)
        write_json(stage / "summary.json", json_values(summary))
        (stage / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    publish(args.out, save)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("select", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--snapshot", required=True, type=Path)
        sub.add_argument("--out", required=True, type=Path)
        sub.add_argument("--tape", default=TAPE, type=Path)
        sub.add_argument("--eligibility", default=ELIGIBILITY, type=Path)
        if command == "select":
            sub.add_argument("--run", required=True, type=Path)
            sub.add_argument("--scores", required=True, type=Path)
        else:
            sub.add_argument("--validation-snapshot", required=True, type=Path)
            members = sub.add_mutually_exclusive_group(required=True)
            members.add_argument("--catalog", type=Path)
            members.add_argument("--member", action="append", nargs=4, type=Path,
                                 metavar=("RUN", "VALIDATION", "LOCK", "TEST"))
    args = parser.parse_args()
    (select if args.command == "select" else evaluate)(args)


if __name__ == "__main__":
    main()
