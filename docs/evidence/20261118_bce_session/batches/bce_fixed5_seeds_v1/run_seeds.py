"""Compare frozen random augmentation with resampling at every BCE optimizer step."""
import copy
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from attention_followup import LegPool
from directional_2018_experiment import groups_for
from directional_three_seed import cache_path,stock_inputs
from replay_old_longshort_bce_loss import legacy_inputs
from replay_equal_weight_model_books import locked_history
from replay_kit_risk import paired

OUT=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("standard_replay",OUT.parent/"standard_bce_pool_v1/run_standard.py")
std=importlib.util.module_from_spec(sp)
sp.loader.exec_module(std)
std.OUT=OUT
BASE=OUT.parent/"bce_resample_low_grid_v1"
CONTROL=("plain","fixed_5")
GRID={f"sample_{n}":.05 for n in range(61119,61123)}
ARMS=(*CONTROL,*GRID)

def prior_checkpoint(arm,seed):
    root=OUT.parent/"bce_resample_steps_v1" if arm=="plain" else BASE
    return root/f"{arm}_{seed}.pt"

def forecast(path,x,digest):
    assert std.record(path)==digest
    cp=torch.load(path,weights_only=True)
    model=LegPool(False).eval()
    model.load_state_dict(cp["state"])
    with torch.no_grad():
        return model((x-cp["mu"])/cp["sd"]).numpy().ravel()


def main():
    torch.set_num_threads(4)
    std.record(__file__)
    std.record(Path("docs/research_20261118_bce_fixed5_seeds_protocol.md"))
    for path in ["scripts/attention_followup.py","scripts/directional_three_seed.py",
                 "scripts/directional_2018_experiment.py","scripts/replay_old_longshort_bce_loss.py"]:
        std.record(path)
    prior=json.loads((BASE/"validation_lock.json").read_text(encoding="utf-8"))
    for path,digest in prior["hashes"].items():
        assert std.record(path)==digest, path
    data,dates,baskets,gaps,complete,train=legacy_inputs()
    groups=groups_for(data,dates)
    sigpath=Path("outputs/old_longshort_bce_loss_v1/report/signals.parquet")
    std.record(sigpath)
    sig=pd.read_parquet(sigpath)
    vdates=pd.DatetimeIndex(sig.query("split=='val'").date)
    vi=dates.get_indexer(vdates)
    valkeep=complete[vi]
    assert len(train)==473 and valkeep.sum()==123 and len(vi)==133
    active=set(np.random.default_rng(61118).choice(len(train),len(train)//2,replace=False))
    # The draws and labels are shared across model seeds.
    draws={}
    diag=[]
    for arm in GRID:
        sampling_seed=int(arm.rsplit("_",1)[-1])
        entries=[]
        for row,t in enumerate(train):
            variants=[]
            if row in active:
                long,short=groups[t]
                primary=baskets[t]
                seen={(tuple(primary[0]),tuple(primary[1]))}
                rng=np.random.default_rng(np.random.SeedSequence([sampling_seed,int(t),1]))
                for attempt in range(3):
                    pair=tuple(pool[np.sort(rng.choice(len(pool),n,replace=False))]
                               for pool,n in ((long[:50],40),(short[:20],10)))
                    key=tuple(map(tuple,pair))
                    ids=np.r_[*pair]
                    assert len(np.unique(ids))==50
                    if key not in seen and data["valid"][t,ids].all():
                        r=np.expm1(data["y"][t,ids])
                        variants.append((pair,float(r[:40].mean()+r[40:].mean()+.08/52)))
                        seen.add(key)
            entries.append(variants)
        draws[arm]=entries
        diag.append(dict(arm=arm,valid_variants=sum(map(len,entries)),
                         augmented_weeks=sum(bool(v) for v in entries),
                         augmented_weight=.05*sum(bool(v) for v in entries)/len(train)))
    pd.DataFrame(diag).to_csv(OUT/"draw_diagnostic.csv",index=False)
    traces=[];selections=[];valpred={a:[] for a in ARMS}
    for seed in (42,43,44):
        std.record(cache_path(seed))
        with np.load(cache_path(seed)) as cache: encoded=cache["z"]
        base=np.stack([stock_inputs(encoded,t,baskets[t])[2] for t in train])
        vx=np.stack([stock_inputs(encoded,t,baskets[t])[2] for t in vi])
        mu=base.reshape(-1,129).mean(0)
        sd=base.reshape(-1,129).std(0)+1e-6
        tx=torch.from_numpy((base-mu)/sd)
        target=torch.tensor(gaps[train]>0,dtype=torch.float32)
        vtorch=torch.from_numpy((vx-mu)/sd)
        vtarget=torch.tensor(gaps[vi][valkeep]>0,dtype=torch.float32)
        for arm in CONTROL:
            cp=prior_checkpoint(arm,seed)
            valpred[arm].append(forecast(cp,torch.from_numpy(vx),prior["hashes"][str(cp.resolve())]))
        def training_batch(arm,alpha):
            xx=[];yy=[];ww=[]
            for row,(t,variants) in enumerate(zip(train,draws[arm])):
                xx.append(base[row]);yy.append(gaps[t]>0);ww.append(1-alpha if variants else 1.)
                for pair,gap in variants:
                    xx.append(stock_inputs(encoded,t,pair)[2])
                    yy.append(gap>0);ww.append(alpha/len(variants))
            assert np.isclose(sum(ww),len(train))
            return (torch.from_numpy((np.stack(xx)-mu)/sd),
                    torch.tensor(yy,dtype=torch.float32),torch.tensor(ww,dtype=torch.float32))
        for arm,alpha in GRID.items():
            torch.manual_seed(seed)
            model=LegPool(False)
            opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.01)
            fixed_batch=training_batch(arm,alpha)
            best=float("inf")
            for step in range(1,101):
                if step==11:
                    model.proj.requires_grad_(False)
                    fixed=[p.detach().clone() for p in model.proj.parameters()]
                x,y,w=fixed_batch
                opt.zero_grad()
                loss=(F.binary_cross_entropy_with_logits(model(x)[:,0],y,reduction="none")*w).sum()/len(train)
                loss.backward();opt.step()
                with torch.no_grad():
                    output=model(vtorch)[:,0]
                    value=F.binary_cross_entropy_with_logits(output[valkeep],vtarget).item()
                traces.append(dict(arm=arm,seed=seed,step=step,train_bce=float(loss),val_bce=value))
                if value<best:
                    best,chosen,state=value,step,copy.deepcopy(model.state_dict())
            assert all(torch.equal(a,b) for a,b in zip(fixed,model.proj.parameters()))
            model.load_state_dict(state)
            cp=OUT/f"{arm}_{seed}.pt"
            torch.save(dict(state=state,mu=torch.from_numpy(mu),sd=torch.from_numpy(sd)),cp)
            std.record(cp)
            with torch.no_grad():
                valpred[arm].append(model(vtorch).numpy().ravel())
            selections.append(dict(arm=arm,seed=seed,step=chosen,val_bce=best,after_freeze=chosen>10))
            print(selections[-1],flush=True)
    ref=sig.set_index("date").freeze_projection
    np.testing.assert_allclose(np.mean(valpred["plain"],axis=0),ref.reindex(vdates),atol=1e-6,rtol=1e-6)

    pd.DataFrame(traces).to_csv(OUT/"training_loss.csv",index=False)
    std.save("validation_lock.json",dict(selections=selections,hashes=std.HASHES.copy(),
             train_dates=dates[train].strftime("%Y-%m-%d").tolist()))
    print("Locked all four new sampling seeds",flush=True)
    signals=sig.query("split=='test'")
    test_dates = pd.DatetimeIndex(signals.date)
    idx = dates.get_indexer(test_dates)
    assert len(idx) == 283 and (idx >= 0).all()
    frame, schedule, members = locked_history()
    b0 = next(scores for lock, scores in members.values() if lock["arm"] == "B0")
    source = std.score_pair(frame, b0["test"]["long"], b0["test"]["short"], schedule)
    common = schedule.loc[schedule.date.isin(test_dates)]
    pd.testing.assert_index_equal(pd.DatetimeIndex(common.date).rename(None), test_dates.rename(None))

    forecasts = {arm: [] for arm in ARMS}
    for seed in (42, 43, 44):
        with np.load(cache_path(seed)) as cache:
            z = cache["z"]
        x = torch.from_numpy(np.stack([stock_inputs(z,t,baskets[t])[2] for t in idx]))
        for arm in forecasts:
            path=prior_checkpoint(arm,seed) if arm in CONTROL else OUT/f"{arm}_{seed}.pt"
            forecasts[arm].append(forecast(path,x,std.HASHES[str(path.resolve())]))
    logits = pd.DataFrame({arm: np.mean(out,axis=0) for arm,out in forecasts.items()},index=test_dates)
    np.testing.assert_allclose(logits.plain,signals.freeze_projection,atol=1e-6,rtol=1e-6)
    assert np.isfinite(logits).all().all()
    direction = logits.gt(0)
    groups = defaultdict(list)
    for arm in forecasts:
        groups[direction[arm].to_numpy().tobytes()].append(arm)
    actual = {}
    for arms in groups.values():
        actual[arms[0]] = forecasts[arms[0]]
    print(f"Inferred all frozen checkpoints; {len(actual)} distinct direction sequences",flush=True)
    rows,books = [],[]
    std.backtests(source,common,actual,"test",rows,books)
    weekly = pd.concat(books,ignore_index=True).query("rule in ['wide','full']")
    stats = pd.DataFrame(rows).query("rule in ['wide','full']")
    w,t = [],[]
    for arms in groups.values():
        a = arms[0]
        for arm in arms:
            w.append(weekly.loc[weekly.arm.eq(a)].assign(arm=arm,logit=lambda x: x.date.map(logits[arm])))
            t.append(stats.loc[stats.arm.eq(a)].assign(arm=arm))
    weekly = pd.concat(w,ignore_index=True)
    stats = pd.concat(t,ignore_index=True)
    std.record(OUT.parent/"bce_resample_low_grid_full_v1/weekly.parquet")

    old=pd.read_parquet(OUT.parent/"bce_resample_low_grid_full_v1/weekly.parquet")
    old=old.loc[old.arm.isin(CONTROL)]
    check=weekly.merge(old,on=["date","arm","rule","fee"],validate="one_to_one",suffixes=("","_old"))
    assert len(check)==len(old)==2*283*4
    np.testing.assert_array_equal(check.net,check.net_old)
    np.testing.assert_array_equal(check.logit,check.logit_old)
    assert len(weekly)==6*283*4 and len(stats)==24
    assert not weekly.duplicated(["arm","rule","fee","date"]).any()
    changes=[]
    for arm in forecasts:
        changed=direction[arm].ne(direction.plain)
        for date in test_dates[changed]:
            changes.append(dict(arm=arm,date=date,to_long=bool(direction.at[date,arm]),
                           logit=logits.at[date,arm],baseline_logit=logits.at[date,"plain"]))
    matrix=weekly.pivot(index=["date","rule","fee"],columns="arm",values="net")
    contrasts=[]
    for (rule,fee),group in matrix.groupby(level=["rule","fee"]):
        for arm in forecasts:
            if arm!="plain":
                contrasts.append(dict(arm=arm,rule=rule,fee=fee,
                    **paired(group[arm].to_numpy(),group.plain.to_numpy())))
    stats.to_csv(OUT/"metrics.csv",index=False)
    weekly.to_parquet(OUT/"weekly.parquet",index=False)
    pd.DataFrame(changes).to_csv(OUT/"direction_changes.csv",index=False)
    pd.DataFrame({"changes":direction.ne(direction.plain,axis=0).sum(),
                  "changes_vs_original":direction.ne(direction.fixed_5,axis=0).sum()}).to_csv(OUT/"direction_counts.csv")
    pd.DataFrame(contrasts).to_csv(OUT/"paired.csv",index=False)
    std.save("inputs.json",std.HASHES)
    print("Verified both prior controls",flush=True)

if __name__=="__main__":
    main()
