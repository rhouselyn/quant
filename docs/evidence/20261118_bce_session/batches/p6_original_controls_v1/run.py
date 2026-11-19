"""Control all legacy data and replace only the frozen stock encoder."""
import copy
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml

from attention_followup import LegPool
from directional_three_seed import cache_path, stock_inputs
from directional_2018_experiment import groups_for, arrays, HierarchicalRanker
from replay_old_longshort_bce_loss import legacy_inputs
from quant_mvp.financial_return import build_model
from replay_equal_weight_model_books import locked_history

OUT = Path(__file__).resolve().parent
ctx = runpy.run_path(str(OUT.parent/"bce_fixed5_ensemble_v1/replay.py"))
std = ctx["std"]
std.OUT = OUT


@torch.inference_mode()
def extract(data, dates, requested, legacy, p6run):
    cfg = yaml.safe_load(Path("outputs/abl2000_base/config.yaml").read_text(encoding="utf8"))
    original = arrays(cfg)
    original_dates = pd.DatetimeIndex(original[6])
    assert list(original[7]) == list(data["symbols"])
    pcfg = yaml.safe_load((p6run/"config.yaml").read_text(encoding="utf8"))
    pre = json.loads((p6run/"preprocessing.json").read_text(encoding="utf8"))
    p6 = build_model(pcfg, original[8], pre).eval()
    p6.load_state_dict(torch.load(p6run/"epoch_8.pt",map_location="cpu",weights_only=True))
    old = HierarchicalRanker(cfg, original[8]).eval()
    old.load_state_dict(torch.load("outputs/abl2000_base/best.pt",map_location="cpu",weights_only=True))
    out = np.zeros_like(legacy)
    missing = 0
    for row, (t, names) in enumerate(sorted(requested.items())):
        ti = original_dates.get_loc(dates[t])
        names = np.array(sorted(names))
        assert ti >= cfg["lookback"]-1
        windows = torch.from_numpy(original[0][ti-cfg["lookback"]+1:ti+1,names].transpose(1,0,2))
        z = F.normalize(p6.final_norm(p6.encoder(windows,normalize=False)),dim=-1).numpy()
        out[t,names] = z
        missing += int((~original[4][ti,names]).sum())
        if row in (0, len(requested)//2, len(requested)-1):
            oldz = F.normalize(old.final_norm(old.encode(windows[None])),dim=-1).numpy()[0]
            np.testing.assert_allclose(oldz,legacy[t,names],rtol=2e-5,atol=2e-6)
        if row%100 == 0:
            print("Same legacy windows encoded",row,len(requested),flush=True)
    assert missing == 0
    return out


def fit(x,y,w,vx,vy,seed,arm,trace):
    primary = x[:473]
    mu = primary.reshape(-1,129).mean(0)
    sd = primary.reshape(-1,129).std(0)+1e-6
    tx = torch.from_numpy((x-mu)/sd)
    vv = torch.from_numpy((vx-mu)/sd)
    target = torch.tensor(y>0,dtype=torch.float32)
    vt = torch.tensor(vy>0,dtype=torch.float32)
    weight = torch.tensor(w,dtype=torch.float32)
    torch.manual_seed(seed)
    model = LegPool(False)
    opt = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.01)
    best = np.inf
    for step in range(1,101):
        if step == 11:
            model.proj.requires_grad_(False)
            frozen = [p.detach().clone() for p in model.proj.parameters()]
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(model(tx)[:,0],target,reduction="none")*weight).sum()/473
        loss.backward()
        opt.step()
        with torch.no_grad():
            val = F.binary_cross_entropy_with_logits(model(vv)[:,0],vt).item()
        trace.append(dict(arm=arm,seed=seed,step=step,train_bce=float(loss),val_bce=val))
        if val<best:
            best,chosen,state=val,step,copy.deepcopy(model.state_dict())
    assert all(torch.equal(a,b) for a,b in zip(frozen,model.proj.parameters()))
    cp = OUT/f"{arm}_{seed}.pt"
    torch.save(dict(state=state,mu=torch.from_numpy(mu),sd=torch.from_numpy(sd)),cp)
    print(arm,seed,chosen,best,flush=True)
    return dict(arm=arm,seed=seed,step=chosen,val_bce=best,checkpoint=str(cp),sha256=std.record(cp))


def main():
    torch.set_num_threads(4)
    for p in (__file__,"docs/research_20261118_p6_original_controls_protocol.md",
              "scripts/directional_2018_experiment.py","scripts/attention_followup.py",
              "scripts/replay_old_longshort_bce_loss.py","outputs/abl2000_base/config.yaml",
              "outputs/abl2000_base/best.pt",std.CATALOG):
        std.record(p)
    member = next(m for m in json.loads(std.CATALOG.read_text(encoding="utf8"))["members"] if m["arm"]=="P6")
    p6run=Path(member["run"])
    for p in (p6run/"config.yaml",p6run/"preprocessing.json",p6run/"epoch_8.pt",Path(member["lock"])):
        std.record(p)
    d,dates,baskets,gaps,complete,train = legacy_inputs()
    sigfile=Path("outputs/old_longshort_bce_loss_v1/report/signals.parquet")
    std.record(sigfile)
    sig=pd.read_parquet(sigfile)
    vi=dates.get_indexer(sig.query("split=='val'").date)
    select=vi[complete[vi]]
    assert len(train)==473 and len(vi)==133 and len(select)==123
    groups=groups_for(d,dates)
    active=set(np.random.default_rng(61118).choice(473,236,replace=False))
    variants=[]
    requested={}
    for t in np.r_[train,vi]:
        requested[t]=set(np.r_[*baskets[t]])
    for row,t in enumerate(train):
        draws=[]
        if row in active:
            seen={tuple(map(tuple,baskets[t]))}
            rng=np.random.default_rng(np.random.SeedSequence([61118,int(t),1]))
            for _ in range(3):
                pair=tuple(pool[np.sort(rng.choice(len(pool),n,replace=False))]
                           for pool,n in ((groups[t][0][:50],40),(groups[t][1][:20],10)))
                ids=np.r_[*pair]
                key=tuple(map(tuple,pair))
                if key not in seen and d["valid"][t,ids].all():
                    seen.add(key)
                    r=np.expm1(d["y"][t,ids])
                    draws.append((pair,float(r[:40].mean()+r[40:].mean()+.08/52)))
                    requested[t].update(ids)
        variants.append(draws)
    traces,selections=[],[]
    with np.load(cache_path(42)) as cache:
        reference=cache["z"]
    p6=extract(d,dates,requested,reference,p6run)
    np.save(OUT/"p6_trainval_vectors.npy",p6)
    std.record(OUT/"p6_trainval_vectors.npy")
    for rep in ("old_triple","old_single","p6"):
        for seed in (42,43,44):
            std.record(cache_path(seed if rep=="old_triple" else 42))
            if rep=="p6":
                z=p6
            else:
                with np.load(cache_path(seed if rep=="old_triple" else 42)) as c:
                    z=c["z"]
            primary=np.stack([stock_inputs(z,t,baskets[t])[2] for t in train])
            vx=np.stack([stock_inputs(z,t,baskets[t])[2] for t in select])
            extra,labels,weights=[],[],np.ones(473)
            for row,(t,draws) in enumerate(zip(train,variants)):
                if draws:
                    weights[row]=.95
                for pair,y in draws:
                    extra.append(stock_inputs(z,t,pair)[2])
                    labels.append(y)
            aug_x=np.concatenate([primary,np.stack(extra)])
            aug_y=np.r_[gaps[train],labels]
            aug_w=np.r_[weights,[.05/len(draws) for draws in variants for _ in draws]]
            assert np.isclose(aug_w.sum(),473)
            for suffix,x,y,w in (("plain",primary,gaps[train],np.ones(473)),("fixed5",aug_x,aug_y,aug_w)):
                selections.append(fit(x,y,w,vx,gaps[select],seed,f"{rep}_{suffix}",traces))
    pd.DataFrame(traces).to_csv(OUT/"training_loss.csv",index=False)
    std.save("validation_lock.json",dict(selections=selections,train_dates=dates[train].astype(str).tolist(),
                                         validation_dates=dates[select].astype(str).tolist(),hashes=std.HASHES.copy()))
    print("Validation locked before test encoding",flush=True)
    ti=dates.get_indexer(sig.query("split=='test'").date)
    requested={t:set(np.r_[*baskets[t]]) for t in ti}
    p6test=extract(d,dates,requested,reference,p6run)
    preds={}
    for row in selections:
        rep=row["arm"].rsplit("_",1)[0]
        if rep=="p6": z=p6test
        else:
            with np.load(cache_path(row["seed"] if rep=="old_triple" else 42)) as c: z=c["z"]
        x=torch.from_numpy(np.stack([stock_inputs(z,t,baskets[t])[2] for t in ti]))
        cp=torch.load(row["checkpoint"],weights_only=True)
        model=LegPool(False).eval()
        model.load_state_dict(cp["state"])
        with torch.no_grad():
            preds.setdefault(row["arm"],[]).append(model((x-cp["mu"])/cp["sd"])[:,0].numpy())
    signals=pd.DataFrame({arm:np.mean(parts,axis=0) for arm,parts in preds.items()},index=dates[ti])
    signals.index.name="date"
    np.testing.assert_allclose(signals.old_triple_plain,sig.query("split=='test'").freeze_projection,atol=1e-6,rtol=1e-6)
    prior=pd.read_parquet("outputs/bce_fixed5_seeds_v1/weekly.parquet").query("arm=='fixed_5' and fee==10 and rule=='full'").set_index("date").logit
    np.testing.assert_allclose(signals.old_triple_fixed5,prior.reindex(signals.index),atol=1e-6,rtol=1e-6)
    signals.to_csv(OUT/"signals.csv")
    frame,schedule,members=locked_history()
    sc=next(s for l,s in members.values() if l["arm"]=="P6")
    source=std.score_pair(frame,sc["test"]["long"],sc["test"]["short"],schedule)
    common=schedule.loc[schedule.date.isin(signals.index)]
    replay=std.replay
    std.replay=lambda f,dt,spec,**kw:replay(f,dt,(20,10,*spec[2:]),**kw)
    metrics,weekly=ctx["replay"](source,common,signals,"test")
    old=pd.read_parquet("outputs/b0_p6_frozen_bce_transfer_v1/report/weekly.parquet").query("model=='P6' and arm in ['wide','full_switch']")
    old=old.rename(columns={"arm":"rule","trade_bps":"fee"}).replace({"full_switch":"full"})
    check=weekly.query("arm=='old_triple_plain'").merge(old,on=["date","rule","fee"],validate="one_to_one")
    assert len(check)==1132 and len(metrics)==24 and len(weekly)==6792
    np.testing.assert_allclose(check.net_x,check.net_y,atol=1e-12,rtol=0)
    metrics.to_csv(OUT/"metrics.csv",index=False)
    weekly.to_parquet(OUT/"weekly.parquet",index=False)
    std.save("inputs.json",std.HASHES)
    print(metrics[["arm","rule","fee","annual_calendar","max_drawdown"]].to_string(index=False),flush=True)
    print("Complete: original signals and P6 47.34% weekly ledger reproduced",flush=True)


if __name__=="__main__":
    main()
