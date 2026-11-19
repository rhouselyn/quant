"""Check whether the original training-step lock changes the 5% perturbation test."""
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


OUT = Path(__file__).resolve().parent
PARENT = OUT.parent / "bce_fixed5_ensemble_v1"
ctx = runpy.run_path(str(OUT.parent / "bce_fixed5_seeds_v1/run_seeds.py"))
std = ctx["std"]
std.OUT = OUT


def main():
    torch.set_num_threads(4)
    std.record(__file__)
    std.record("docs/research_20261118_bce_old_selector_fixed_steps_protocol.md")
    old_lock = OUT.parent / "bce_resample_steps_v1/validation_lock.json"
    aug_lock = ctx["BASE"] / "validation_lock.json"
    for path in (old_lock, aug_lock):
        std.record(path)
    old = json.loads(old_lock.read_text(encoding="utf-8"))
    aug = json.loads(aug_lock.read_text(encoding="utf-8"))
    old_steps = {r["seed"]:r["step"] for r in old["selections"] if r["arm"] == "plain"}
    aug_steps = {r["seed"]:r["step"] for r in aug["selections"] if r["arm"] == "fixed_5"}
    assert old_steps == {42:100,43:14,44:67} and aug_steps == {42:100,43:14,44:81}
    data, dates, baskets, gaps, _, train = ctx["legacy_inputs"]()
    assert dates[train].strftime("%Y-%m-%d").tolist() == aug["train_dates"]
    groups = ctx["groups_for"](data, dates)
    path = ctx["cache_path"](44)
    assert std.record(path) == aug["hashes"][str(path.resolve())]
    with np.load(path) as cache:
        encoded = cache["z"]
    active = set(aug["active_train_indices"])
    assert active == set(np.random.default_rng(61118).choice(len(train), len(train)//2, replace=False))
    x, y, weights = [], [], []
    for row, t in enumerate(train):
        primary = baskets[t]
        variants = []
        if row in active:
            long, short = groups[t]
            rng = np.random.default_rng(np.random.SeedSequence([61118, int(t), 1]))
            seen = {tuple(map(tuple, primary))}
            for _ in range(3):
                pair = tuple(pool[np.sort(rng.choice(len(pool), n, replace=False))]
                             for pool, n in ((long[:50], 40), (short[:20], 10)))
                ids = np.r_[*pair]
                key = tuple(map(tuple, pair))
                if key not in seen and data["valid"][t, ids].all():
                    seen.add(key)
                    returns = np.expm1(data["y"][t, ids])
                    variants.append((pair, returns[:40].mean() + returns[40:].mean() + .08/52))
        x.append(ctx["stock_inputs"](encoded, t, primary)[2])
        y.append(gaps[t] > 0)
        weights.append(.95 if variants else 1.)
        for pair, gap in variants:
            x.append(ctx["stock_inputs"](encoded, t, pair)[2])
            y.append(gap > 0)
            weights.append(.05 / len(variants))
    assert np.isclose(sum(weights), len(train))
    cp = ctx["prior_checkpoint"]("fixed_5", 44)
    assert std.record(cp) == aug["hashes"][str(cp.resolve())]
    ref = torch.load(cp, weights_only=True)
    x = (torch.from_numpy(np.stack(x)) - ref["mu"]) / ref["sd"]
    y = torch.tensor(y, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)
    torch.manual_seed(44)
    model = ctx["LegPool"](False)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.01)
    matched = OUT / "fixed5_44_step67.pt"
    for step in range(1, aug_steps[44]+1):
        if step == 11:
            model.proj.requires_grad_(False)
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(model(x)[:,0], y, reduction="none") * weights).sum() / len(train)
        loss.backward()
        opt.step()
        if step == old_steps[44]:
            torch.save(dict(state=model.state_dict(), mu=ref["mu"], sd=ref["sd"]), matched)
    for key, param in model.state_dict().items():
        torch.testing.assert_close(param, ref["state"][key], atol=1e-6, rtol=1e-6)
    print("Reproduced seed 44 step 81 training parameters", flush=True)
    std.save("matched_steps_lock.json", dict(steps=old_steps, hashes=std.HASHES.copy(),
                                           checkpoint_sha256=std.record(matched)))

    reference = OUT.parent / "bce_fixed5_seeds_v1/weekly.parquet"
    ledger = pd.read_parquet(reference).query("fee==10 and rule=='full'")
    sig = ledger.pivot(index="date", columns="arm", values="logit")[["plain","fixed_5"]]
    idx = dates.get_indexer(sig.index)
    assert (idx >= 0).all() and len(idx) == 283
    forecasts = []
    for seed in (42,43,44):
        path = ctx["cache_path"](seed)
        assert std.record(path) == aug["hashes"][str(path.resolve())]
        if seed == 44:
            vectors = encoded
        else:
            with np.load(path) as cache:
                vectors = cache["z"]
        xs = torch.from_numpy(np.stack([ctx["stock_inputs"](vectors, t, baskets[t])[2] for t in idx]))
        cp = ctx["prior_checkpoint"]("fixed_5", seed)
        digest = aug["hashes"][str(cp.resolve())]
        forecasts.append(ctx["forecast"](cp, xs, digest))
        if seed == 44:
            new44 = ctx["forecast"](matched, xs, std.record(matched))
    np.testing.assert_allclose(np.mean(forecasts, axis=0), sig.fixed_5, atol=1e-6, rtol=1e-6)
    sig["fixed_5_steps_matched"] = np.mean([*forecasts[:2], new44], axis=0)
    sig.to_csv(OUT / "matched_steps_signals.csv")
    for name in ("plain", "fixed_5"):
        flips = sig.index[sig[name].gt(0).ne(sig.fixed_5_steps_matched.gt(0))]
        print(f"Matched steps versus {name}: {len(flips)} flips {flips.strftime('%Y-%m-%d').tolist()}")


if __name__ == "__main__":
    main()
