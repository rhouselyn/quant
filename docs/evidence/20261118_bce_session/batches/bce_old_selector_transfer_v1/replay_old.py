"""Replay locked allocation signals on the old and P6 stock selectors."""
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from directional_three_seed import SEEDS, cache_path
from replay_equal_weight_model_books import legacy_inputs, locked_history


OUT = Path(__file__).resolve().parent
PARENT = OUT.parent / "bce_fixed5_ensemble_v1"
ctx = runpy.run_path(str(PARENT / "replay.py"))
std = ctx["std"]
std.OUT = OUT


def main():
    lockfile = PARENT / "validation_lock.json"
    seedbook = OUT.parent / "bce_fixed5_seeds_v1/weekly.parquet"
    oldbook = OUT.parent / "old_longshort_bce_loss_v1/report/signals.parquet"
    p6book = OUT.parent / "p6_own_bce_v1/weekly.parquet"
    refbook = OUT.parent / "b0_p6_frozen_bce_transfer_v1/report/weekly.parquet"
    for path in (__file__, lockfile, PARENT / "signals.csv", seedbook, oldbook, p6book, refbook,
                 PARENT / "replay.py", "scripts/replay_equal_weight_model_books.py",
                 "scripts/eval_weekly_architecture.py", "scripts/evaluate_dataset_three_year.py",
                 "outputs/p6_own_bce_v1/validation_lock.json",
                 "docs/research_20261118_bce_old_selector_protocol.md",
                 std.__file__, std.CATALOG, std.TAPE, std.TAPE.with_name("eligibility.parquet")):
        std.record(path)
    lock = json.loads(lockfile.read_text(encoding="utf-8"))
    assert lock["winners"] == {"wide": "mean_logit", "full": "mean_logit"}
    for seed in SEEDS:
        path = cache_path(seed)
        assert std.record(path) == lock["hashes"][str(path.resolve())]
    for item in json.loads(std.CATALOG.read_text(encoding="utf-8"))["members"]:
        if item["arm"] == "P6":
            for key in ("lock", "test_scores"):
                std.record(item[key])

    sig = pd.read_csv(PARENT / "signals.csv", parse_dates=["date"],
                      float_precision="round_trip").set_index("date", verify_integrity=True)
    legacy = pd.read_parquet(oldbook, filters=[("split", "=", "test")]).set_index("date")
    pd.testing.assert_index_equal(sig.index, legacy.index)
    np.testing.assert_allclose(sig.plain, legacy.freeze_projection, rtol=1e-6, atol=1e-6)
    for name, path, arm in (("fixed_5", seedbook, "fixed_5"), ("p6_bce", p6book, "p6_bce")):
        signal = pd.read_parquet(path, filters=[("arm", "=", arm), ("rule", "=", "full"),
                                                ("fee", "=", 10)]).set_index("date")
        pd.testing.assert_index_equal(sig.index, signal.index)
        sig[name] = signal.logit
    assert sig.fixed_5.gt(0).equals(sig.mean_logit.gt(0))
    assert sig.majority_vote.gt(0).equals(sig.mean_logit.gt(0))
    assert len(sig) == 283 and np.isfinite(sig.to_numpy()).all()
    sig = sig[["plain", "fixed_5", "mean_logit", "p6_bce"]].rename(columns={"plain": "unaugmented"})
    print("Verified frozen signals and matched 5% single-seed directions", flush=True)

    frame, schedule, members = locked_history()
    common = schedule.loc[schedule.date.isin(sig.index)]
    pd.testing.assert_index_equal(pd.DatetimeIndex(common.date), sig.index)
    data, dates, _, _, complete, train = legacy_inputs()
    assert len(train) == 473
    val = pd.read_parquet(oldbook, filters=[("split", "=", "val")])
    val_idx = dates.get_indexer(val.date)
    assert (val_idx >= 0).all() and complete[val_idx].sum() == 123
    print("Old training:", dates[train[0]], dates[train[-1]],
          "validation:", val.date.min(), val.date.max(), flush=True)
    old_scores = pd.concat([pd.DataFrame(dict(date=day, symbol=data["symbols"],
                     score=data["score"][dates.get_loc(day)])) for day in common.date], ignore_index=True)
    old = std.score_pair(frame.loc[frame.date.isin(sig.index)], old_scores, old_scores, common)
    _, p6 = next((lock, scores) for lock, scores in members.values() if lock["arm"] == "P6")
    p6 = std.score_pair(frame, p6["test"]["long"], p6["test"]["short"], schedule)

    rows, books = [], []
    original_replay = std.replay
    for model, slots, source in (("old_scores", 40, old), ("old_scores", 20, old), ("P6", 20, p6)):
        def replay_with_slots(frame, dates, spec, **kwargs):
            return original_replay(frame, dates, (slots, *spec[1:]), **kwargs)
        try:
            std.replay = replay_with_slots
            metrics, weekly = ctx["replay"](source, common, sig, "test")
        finally:
            std.replay = original_replay
        rows.append(metrics.assign(model=model, long_slots=slots))
        books.append(weekly.assign(model=model, long_slots=slots))
        print(f"{model} {slots}/10 finished", flush=True)

    stats, weekly = pd.concat(rows, ignore_index=True), pd.concat(books, ignore_index=True)
    keys = ["model", "long_slots", "date", "rule", "fee"]
    assert len(stats) == 48 and len(weekly) == 283 * len(stats)
    assert not weekly.duplicated([*keys, "arm"]).any()
    np.testing.assert_allclose(weekly.long_net + weekly.short_net, weekly.net, rtol=0, atol=1e-12)
    p6ref = pd.read_parquet(p6book).query("rule in ['wide', 'full']").replace({"plain": "unaugmented"})
    ref = weekly.query("model=='P6' and arm in ['unaugmented','p6_bce']").merge(
        p6ref, on=["arm", "rule", "fee", "date"], validate="one_to_one")
    assert len(ref) == 2264
    np.testing.assert_allclose(ref.net_x, ref.net_y, rtol=0, atol=1e-12)
    p6ref = pd.read_parquet(refbook).query("model=='P6' and arm in ['wide','full_switch']")
    p6ref = p6ref.rename(columns={"arm": "rule", "trade_bps": "fee"}).replace({"full_switch": "full"})
    ref = weekly.query("model=='P6' and arm=='unaugmented'").merge(
        p6ref, on=["rule", "fee", "date"], validate="one_to_one")
    assert len(ref) == 1132
    np.testing.assert_allclose(ref.net_x, ref.net_y, rtol=0, atol=1e-12)
    net = weekly.pivot(index=keys, columns="arm", values="net")
    np.testing.assert_array_equal(net.fixed_5, net.mean_logit)
    paired = ctx["ctx"]["paired"]
    pairs = [dict(model=model, long_slots=slots, rule=rule, fee=fee, arm=arm,
                 **paired(g[arm].to_numpy(), g.unaugmented.to_numpy()))
             for (model, slots, rule, fee), g in net.groupby(["model", "long_slots", "rule", "fee"])
             for arm in ("fixed_5", "p6_bce")]
    years = []
    year_keys = ["model", "long_slots", "arm", "rule", "fee", "year"]
    for year_case, g in weekly.assign(year=weekly.date.dt.year).groupby(year_keys):
        summary = std.perf_summary(g.sort_values("date").net)
        years.append(dict(zip(year_keys, year_case), weeks=len(g), net_return=summary["total"],
                          max_drawdown=summary["max_drawdown"]))
    stats.to_csv(OUT / "metrics.csv", index=False)
    weekly.to_parquet(OUT / "weekly.parquet", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "paired.csv", index=False)
    pd.DataFrame(years).to_csv(OUT / "yearly.csv", index=False)
    sig.to_csv(OUT / "signals.csv")
    std.save("inputs.json", std.HASHES)
    print("Verified 48 metric rows, 13584 weekly rows, and all P6 reference weeks", flush=True)


if __name__ == "__main__":
    main()
