"""Compare frozen random augmentation with resampling at every BCE optimizer step."""
import copy
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
BASE=OUT.parent/"bce_resample_steps_v1"
CONTROL=("plain",)
GRID={f"{method}_{n}":n/100 for method in ("fixed","dynamic") for n in range(1,6)}
ARMS=(*CONTROL,*GRID)


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
    std.record(Path("docs/research_20261118_bce_resample_low_grid_protocol.md"))
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
    for step in range(1,101):
        entries=[]
        for row,t in enumerate(train):
            variants=[]
            if row in active:
                long,short=groups[t]
                primary=baskets[t]
                seen={(tuple(primary[0]),tuple(primary[1]))}
                rng=np.random.default_rng(np.random.SeedSequence([61118,int(t),step]))
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
        draws[step]=entries
        diag.append(dict(step=step,valid_variants=sum(map(len,entries)),
                         augmented_weeks=sum(bool(v) for v in entries),
                         augmented_weight=.3*sum(bool(v) for v in entries)/len(train)))
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
            cp=BASE/f"{arm}_{seed}.pt"
            valpred[arm].append(forecast(cp,torch.from_numpy(vx),prior["hashes"][str(cp.resolve())]))
        def training_batch(step,alpha):
            xx=[];yy=[];ww=[]
            for row,(t,variants) in enumerate(zip(train,draws[step])):
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
            fixed_batch=training_batch(1,alpha) if arm.startswith("fixed") else None
            best=float("inf")
            for step in range(1,101):
                if step==11:
                    model.proj.requires_grad_(False)
                    fixed=[p.detach().clone() for p in model.proj.parameters()]
                x,y,w=fixed_batch if arm.startswith("fixed") else training_batch(step,alpha)
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
    member=next(m for m in json.loads(std.CATALOG.read_text(encoding="utf-8"))["members"] if m["arm"]=="B0")
    directory=std.STANDARD/"snapshots/validation"
    vf,vs,_=std.phase_proxy(directory,std.snapshot(directory),std.TAPE,std.TAPE.with_name("eligibility.parquet"),phase="validation")
    lock=json.loads(Path(member["lock"]).read_text(encoding="utf-8"))
    raw=pd.read_parquet(member["validation_scores"])
    raw["date"]=pd.to_datetime(raw.date)
    vsource=std.score_pair(vf,std.fuse_epochs(raw,lock["long_epochs"]),std.fuse_epochs(raw,lock["short_epochs"]),vs)
    vs=vs.loc[vs.date.isin(vdates)]
    order=vdates.get_indexer(vs.date)
    assert (order>=0).all()
    vrows,vbooks=[],[]
    std.backtests(vsource,vs,{arm:[x[order] for x in parts] for arm,parts in valpred.items()},"val",vrows,vbooks)
    vmetrics=pd.DataFrame(vrows)
    ranks=vmetrics.groupby(["rule","arm"]).annual_calendar.mean()
    winners={rule:{method:ranks.loc[rule].reindex(
        [f"{method}_{n}" for n in range(1,6)]).idxmax() for method in ("fixed","dynamic")}
        for rule in ("wide","full")}
    selected={rule:{"plain",*(f"{method}_{winner.rsplit('_',1)[-1]}"
        for method in ("fixed","dynamic") for winner in picks.values())}
        for rule,picks in winners.items()}
    overall={rule:ranks.loc[rule].idxmax() for rule in winners}
    vmetrics.to_csv(OUT/"validation_metrics.csv",index=False)
    std.save("validation_lock.json",dict(selections=selections,winners=winners,overall=overall,selected={k:sorted(v) for k,v in selected.items()},hashes=std.HASHES.copy(),
        train_dates=dates[train].strftime("%Y-%m-%d").tolist(),active_train_indices=sorted(map(int,active))))
    print("LOCKED",winners,"overall",overall,"validation weeks",len(vs),flush=True)

    frame,schedule,members=locked_history()
    b0=next((l,s) for l,s in members.values() if l["arm"]=="B0")
    source=std.score_pair(frame,b0[1]["test"]["long"],b0[1]["test"]["short"],schedule)
    tdates=pd.DatetimeIndex(sig.query("split=='test'").date)
    common=schedule.loc[schedule.date.isin(tdates)]
    assert len(common)==283
    ti=dates.get_indexer(tdates)
    testpred={a:[] for a in sorted(set.union(*selected.values()))}
    for seed in (42,43,44):
        with np.load(cache_path(seed)) as cache: z=cache["z"]
        x=torch.from_numpy(np.stack([stock_inputs(z,t,baskets[t])[2] for t in ti]))
        for arm in testpred:
            path=(BASE if arm in CONTROL else OUT)/f"{arm}_{seed}.pt"
            testpred[arm].append(forecast(path,x,std.HASHES[str(path.resolve())]))
    np.testing.assert_allclose(np.mean(testpred["plain"],axis=0),ref.reindex(tdates),atol=1e-6,rtol=1e-6)
    rows=[];books=[]
    std.backtests(source,common,testpred,"test",rows,books)
    rows=[r for r in rows if r["arm"] in CONTROL or r["arm"] in selected[r["rule"]]]
    weeks=pd.concat([b for b in books if b.arm.iloc[0] in CONTROL or b.arm.iloc[0] in selected[b.rule.iloc[0]]],ignore_index=True)
    assert len(rows)==4+2*sum(map(len,selected.values())) and len(weeks)==len(rows)*283
    assert not weeks.duplicated(["arm","rule","fee","date"]).any()
    # The baseline produces two additional fixed-ratio policies.
    refbook=pd.read_parquet(OUT.parent/"bce_factor_audit_v1/weekly.parquet").query("arm=='old473'")
    check=weeks.query("arm=='plain' and rule in ['wide','full']").merge(
        refbook,on=["date","rule","fee"],validate="one_to_one",suffixes=("","_ref"))
    assert len(check)==1132
    np.testing.assert_allclose(check.net,check.net_ref,atol=1e-12,rtol=0)
    comparisons=[]
    for (rule,fee),group in weeks.query("rule in ['wide','full']").groupby(["rule","fee"]):
        net=group.pivot(index="date",columns="arm",values="net")
        pairs=[(arm,"plain") for arm in sorted(set(net.columns)-{"plain"})]
        pairs.extend((arm,"fixed_"+arm.rsplit("_",1)[-1]) for arm in sorted(net) if arm.startswith("dynamic"))
        for arm,base in pairs:
            comparisons.append(dict(rule=rule,fee=fee,arm=arm,reference=base,
                **paired(net[arm].to_numpy(),net[base].to_numpy())))
    pd.DataFrame(rows).to_csv(OUT/"metrics.csv",index=False)
    pd.DataFrame(traces).to_csv(OUT/"training_loss.csv",index=False)
    pd.DataFrame(comparisons).to_csv(OUT/"paired.csv",index=False)
    weeks.to_parquet(OUT/"weekly.parquet",index=False)
    std.save("inputs.json",std.HASHES)
    print(pd.DataFrame(rows)[["arm","rule","fee","annual_calendar","max_drawdown"]].to_string(index=False),flush=True)
    print("COMPLETE old signal and B0 execution verified",flush=True)

if __name__=="__main__": main()
