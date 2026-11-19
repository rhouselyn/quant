"""Select and replay two aggregation rules for five frozen sampling seeds."""
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent / "bce_fixed5_seeds_v1"
ctx = runpy.run_path(str(PARENT / "run_seeds.py"))
std = ctx["std"]
std.OUT = OUT
MEMBERS = ("fixed_5", *ctx["GRID"])
METHODS = ("mean_logit", "majority_vote")


def aggregate(logits):
    assert np.isfinite(logits.to_numpy()).all()
    members = logits[list(MEMBERS)]
    return logits[["plain"]].assign(mean_logit=members.mean(axis=1),
                                     majority_vote=2*members.gt(0).sum(axis=1)-5)


def replay(source,schedule,signals,phase):
    groups = {}
    for arm in signals:
        key = signals[arm].gt(0).to_numpy().tobytes()
        groups.setdefault(key,[]).append(arm)
    rows,books=[],[]
    std.backtests(source,schedule,{a[0]:[signals[a[0]].to_numpy()] for a in groups.values()},phase,rows,books)
    metrics=pd.DataFrame(rows).query("rule in ['wide','full']")
    weekly=pd.concat(books,ignore_index=True).query("rule in ['wide','full']")
    return (pd.concat([metrics.loc[metrics.arm.eq(a[0])].assign(arm=arm)
                       for a in groups.values() for arm in a],ignore_index=True),
            pd.concat([weekly.loc[weekly.arm.eq(a[0])].assign(arm=arm,logit=lambda x:x.date.map(signals[arm]))
                       for a in groups.values() for arm in a],ignore_index=True))


def main():
    torch.set_num_threads(4)
    std.record(__file__)
    std.record(Path("docs/research_20261118_bce_fixed5_ensemble_protocol.md"))
    lockfile=PARENT/"validation_lock.json"
    std.record(lockfile)
    locked=json.loads(lockfile.read_text(encoding="utf-8"))
    for path,digest in locked["hashes"].items():
        assert std.record(path)==digest,path
    _,dates,baskets,_,_,_=ctx["legacy_inputs"]()
    sigpath=Path("outputs/old_longshort_bce_loss_v1/report/signals.parquet")
    std.record(sigpath)
    base=pd.read_parquet(sigpath,filters=[("split","=","val")]).set_index("date")
    idx=dates.get_indexer(base.index)
    assert len(idx)==133 and (idx>=0).all()
    raw={arm:[] for arm in ("plain",*MEMBERS)}
    for seed in (42,43,44):
        with np.load(ctx["cache_path"](seed)) as cache:
            embeddings=cache["z"]
        x=torch.from_numpy(np.stack([ctx["stock_inputs"](embeddings,t,baskets[t])[2] for t in idx]))
        for arm in raw:
            path=ctx["prior_checkpoint"](arm,seed) if arm in ctx["CONTROL"] else PARENT/f"{arm}_{seed}.pt"
            raw[arm].append(ctx["forecast"](path,x,locked["hashes"][str(path.resolve())]))
    vraw=pd.DataFrame({arm:np.mean(arr,axis=0) for arm,arr in raw.items()},index=base.index)
    np.testing.assert_allclose(vraw.plain,base.freeze_projection,atol=1e-6,rtol=1e-6)
    member=next(m for m in json.loads(std.CATALOG.read_text(encoding="utf-8"))["members"] if m["arm"]=="B0")
    directory=std.STANDARD/"snapshots/validation"
    vf,vs,_=std.phase_proxy(directory,std.snapshot(directory),std.TAPE,std.TAPE.with_name("eligibility.parquet"),phase="validation")
    for k in ("lock","validation_scores"): std.record(Path(member[k]))
    lock=json.loads(Path(member["lock"]).read_text(encoding="utf-8"))
    scores=pd.read_parquet(member["validation_scores"])
    scores["date"]=pd.to_datetime(scores.date)
    source=std.score_pair(vf,std.fuse_epochs(scores,lock["long_epochs"]),std.fuse_epochs(scores,lock["short_epochs"]),vs)
    vs=vs.loc[vs.date.isin(vraw.index)]
    assert len(vs)==133
    vsig=aggregate(vraw).reindex(vs.date)
    val,vweek=replay(source,vs,vsig,"val")
    scores=val.loc[val.arm.isin(METHODS)].groupby(["rule","arm"]).annual_calendar.mean()
    winners={rule:scores.loc[rule].reindex(METHODS).idxmax() for rule in ("wide","full")}
    val.to_csv(OUT/"validation_metrics.csv",index=False)
    std.save("validation_lock.json",dict(winners=winners,validation_dates=vs.date.dt.strftime("%Y-%m-%d").tolist(),
         tie_break="mean_logit",hashes=std.HASHES.copy()))
    print("Validation locked",winners,flush=True)

    ledger=PARENT/"weekly.parquet"
    std.record(ledger)
    prior=pd.read_parquet(ledger)
    raw=prior.query("rule=='full' and fee==10").pivot(index="date",columns="arm",values="logit")
    sig=aggregate(raw)
    assert len(sig)==283
    frame,schedule,members=ctx["locked_history"]()
    b0=next(scores for lock,scores in members.values() if lock["arm"]=="B0")
    source=std.score_pair(frame,b0["test"]["long"],b0["test"]["short"],schedule)
    common=schedule.loc[schedule.date.isin(sig.index)]
    pd.testing.assert_index_equal(pd.DatetimeIndex(common.date),sig.index)
    metrics,weekly=replay(source,common,sig,"test")
    controls=weekly.query("arm=='plain'").merge(prior,on=["arm","rule","fee","date"],validate="one_to_one",suffixes=("","_prior"))
    assert len(controls)==1132 and len(weekly)==3396 and len(metrics)==12
    np.testing.assert_array_equal(controls.net,controls.net_prior)
    directions=sig.gt(0)
    flips=[]
    for arm in METHODS:
        changed=directions[arm].ne(directions.plain)
        print(arm,"direction changes",changed.sum(),flush=True)
        for date in sig.index[changed]:
            flips.append(dict(arm=arm,date=date,to_long=bool(directions.at[date,arm]),
                signal=sig.at[date,arm],base_logit=sig.at[date,"plain"]))
    matrix=weekly.pivot(index=["date","rule","fee"],columns="arm",values="net")
    pairs=[]
    for (rule,fee),g in matrix.groupby(level=["rule","fee"]):
        for arm in METHODS:
            pairs.append(dict(arm=arm,rule=rule,fee=fee,**ctx["paired"](g[arm].to_numpy(),g.plain.to_numpy())))
    metrics.to_csv(OUT/"metrics.csv",index=False)
    weekly.to_parquet(OUT/"weekly.parquet",index=False)
    pd.DataFrame(flips,columns=["arm","date","to_long","signal","base_logit"]).to_csv(OUT/"direction_changes.csv",index=False)
    pd.DataFrame(pairs).to_csv(OUT/"paired.csv",index=False)
    sig.to_csv(OUT/"signals.csv")
    vsig.to_csv(OUT/"validation_signals.csv")
    std.save("inputs.json",std.HASHES)
    print("Completed and reproduced baseline",flush=True)


if __name__=="__main__":
    main()
