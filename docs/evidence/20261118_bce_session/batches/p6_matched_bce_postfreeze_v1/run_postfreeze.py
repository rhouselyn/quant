"""Repeat the locked P6 data pipeline with validation selection after freezing."""
import hashlib
import json
import os
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


OUT = Path(__file__).resolve().parent
PARENT = OUT.parent / "p6_matched_bce_v1"
ctx = runpy.run_path(str(PARENT / "run.py"))
runner = ctx["main"].__globals__
std = ctx["std"]


def cached_scores(run, cfg, data, ids, lock):
    digest = hashlib.sha256(json.dumps(std["HASHES"], sort_keys=True).encode()).hexdigest()
    artifact = PARENT / "training_scores.npz"
    original = json.loads((PARENT/"validation_lock.json").read_text(encoding="utf-8"))
    assert ctx["record"](artifact) == original["hashes"][str(artifact)]
    with np.load(artifact) as cache:
        assert cache["digest"].item() == digest
        result = tuple(cache[k] for k in ("long","short","vectors"))
    linked = OUT / artifact.name
    if not linked.exists():
        os.link(artifact, linked)
    assert os.path.samefile(artifact, linked)
    ctx["record"](__file__)
    ctx["record"]("docs/research_20261118_p6_matched_bce_postfreeze_protocol.md")
    ctx["record"](PARENT/"validation_lock.json")
    ctx["record"](PARENT/"training_loss.csv")
    return result


def fit(arm, seed, batch, normalizer, vx, vy, nweeks, trace):
    x, y, w = batch
    mu, sd = normalizer
    tx = torch.from_numpy((x-mu)/sd)
    target = torch.tensor(y>0, dtype=torch.float32)
    weight = torch.tensor(w, dtype=torch.float32)
    use = np.isfinite(vy)
    vv = torch.from_numpy((vx[use]-mu)/sd)
    vtarget = torch.tensor(vy[use]>0, dtype=torch.float32)
    torch.manual_seed(seed)
    model = ctx["base"]["LegPool"](False)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.01)
    best = np.inf
    history = []
    for step in range(1,101):
        if step == 11:
            model.proj.requires_grad_(False)
            frozen = [p.detach().clone() for p in model.proj.parameters()]
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(model(tx)[:,0], target, reduction="none") * weight).sum()/nweeks
        loss.backward()
        opt.step()
        with torch.no_grad():
            val = F.binary_cross_entropy_with_logits(model(vv)[:,0], vtarget).item()
        history.append(dict(arm=arm,seed=seed,step=step,train_bce=float(loss),val_bce=val))
        if step >= 11 and val < best:
            best, chosen = val, step
            state = {k:p.detach().clone() for k,p in model.state_dict().items()}
    assert all(torch.equal(a,b) for a,b in zip(frozen,model.proj.parameters()))
    reference = pd.read_csv(PARENT/"training_loss.csv").query("arm==@arm and seed==@seed").sort_values("step")
    pd.testing.assert_frame_equal(pd.DataFrame(history).reset_index(drop=True), reference.reset_index(drop=True),
                                 check_exact=False, rtol=0, atol=1e-7)
    trace.extend(history)
    cp = OUT / f"{arm}_{seed}.pt"
    torch.save(dict(state=state,mu=torch.from_numpy(mu),sd=torch.from_numpy(sd)), cp)
    print(arm, seed, "post-freeze step", chosen, "validation BCE", best, flush=True)
    return dict(arm=arm,seed=seed,step=chosen,val_bce=best,checkpoint=str(cp),sha256=ctx["record"](cp))


if __name__ == "__main__":
    runner.update(OUT=OUT,train_scores=cached_scores,fit=fit)
    std["OUT"] = OUT
    ctx["main"]()
