"""Swap only native Q1 feature values, preserving every legacy supervision row."""
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml

PARENT = Path("outputs/p6_original_controls_v1")
OUT = Path(__file__).resolve().parent
ctx = runpy.run_path(str(PARENT/"run.py"))
runner = ctx["main"].__globals__
std = ctx["std"]
std.OUT = OUT
parent_fit = ctx["fit"]
lock = json.loads((PARENT/"validation_lock.json").read_text(encoding="utf8"))
trace = pd.read_csv(PARENT/"training_loss.csv")
coverage, changes, fingerprints = [], [], []
test_x = None


def digest(*values):
    hasher=hashlib.sha256()
    for value in values:
        arr=np.ascontiguousarray(value)
        hasher.update(str(arr.dtype).encode())
        hasher.update(str(arr.shape).encode())
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def fit(x,y,w,vx,vy,seed,arm,traces):
    info=dict(arm=arm,seed=seed,supervision_sha256=digest(y,w,vy),shape=list(x.shape),
              validation_shape=list(vx.shape),input_sha256=digest(x,vx))
    fingerprints.append(info)
    if arm.startswith("p6_"):
        std.save("training_identity.json",fingerprints)
        return parent_fit(x,y,w,vx,vy,seed,arm,traces)
    row=next(r for r in lock["selections"] if r["arm"]==arm and r["seed"]==seed)
    assert std.record(row["checkpoint"])==row["sha256"]
    traces.extend(trace.query("arm==@arm and seed==@seed").to_dict(orient="records"))
    return row


@torch.inference_mode()
def extract(data,dates,requested,legacy,p6run):
    global test_x
    cfg=yaml.safe_load(Path("outputs/abl2000_base/config.yaml").read_text(encoding="utf8"))
    old=ctx["arrays"](cfg)
    old_dates=pd.DatetimeIndex(old[6])
    pcfg=yaml.safe_load((p6run/"config.yaml").read_text(encoding="utf8"))
    pre=json.loads((p6run/"preprocessing.json").read_text(encoding="utf8"))
    for p in (__file__,"docs/research_20261118_p6_q1_features_protocol.md",PARENT/"run.py",
              PARENT/"validation_lock.json","src/quant_mvp/hierarchy_data.py",
              *(cfg[k] for k in ("cache_path","market_path","universe_path"))):
        std.record(p)
    training_dates=set(pd.to_datetime(lock["train_dates"]))
    features=cfg["feature_names"]
    oldraw=pd.read_parquet(cfg["cache_path"],columns=["date","symbol","observed",*features]).set_index(["date","symbol"])
    phase="trainval" if len(requested)==606 else "test"
    if phase=="trainval":
        parts=[("train",Path(pcfg["cache_path"]).parent,[(t,n) for t,n in requested.items() if dates[t] in training_dates]),
               ("val",std.STANDARD/"snapshots/validation",[(t,n) for t,n in requested.items() if dates[t] not in training_dates])]
    else:
        parts=[("test",std.STANDARD/"snapshots/history",list(requested.items()))]
    result=np.zeros_like(legacy)
    for split,directory,entries in parts:
        for name in ("manifest.json","weekly.parquet","market_features.csv","universe.csv"):
            std.record(directory/name)
        options,native=ctx["ctx"]["ctx"]["std"].dataset(directory,p6run,pcfg)
        assert native[9]["feature_mean"]==pre["feature_mean"] and native[9]["feature_std"]==pre["feature_std"]
        assert list(native[7])==list(old[7])
        nd=pd.DatetimeIndex(native[6])
        native_raw=pd.read_parquet(directory/"weekly.parquet",columns=["date","symbol","observed",*features]).set_index(["date","symbol"])
        model=ctx["build_model"](options,native[8],native[9]).eval()
        model.load_state_dict(torch.load(p6run/"epoch_8.pt",map_location="cpu",weights_only=True))
        for row,(t,names) in enumerate(sorted(entries)):
            day=dates[t]
            ti=old_dates.get_loc(day)
            ni=nd.get_loc(day)
            assert ti>=39 and ni>=39
            np.testing.assert_array_equal(old_dates[ti-39:ti+1],nd[ni-39:ni+1])
            names=np.array(sorted(names))
            symbols=np.asarray(old[7])[names]
            key=pd.MultiIndex.from_product([nd[ni-39:ni+1],symbols],names=["date","symbol"])
            a=oldraw.reindex(key); b=native_raw.reindex(key)
            aa=a[features].to_numpy(); bb=b[features].to_numpy()
            finite=np.isfinite(aa)&np.isfinite(bb)
            changed=(np.abs(aa-bb)>0)&finite
            window=native[0][ni-39:ni+1,names].transpose(1,0,2)
            baseline=old[0][ti-39:ti+1,names].transpose(1,0,2)
            assert np.isfinite(window).all()
            # Keep the legacy stock list and mask while Q1 supplies filled feature values.
            z=F.normalize(model.final_norm(model.encoder(torch.from_numpy(window),normalize=False)),dim=-1).numpy()
            result[t,names]=z
            coverage.append(dict(split=split,date=day,requested_stocks=len(names),
                                 q1_unavailable=int((~native[4][ni,names]).sum()),
                                 raw_changed_values=int(changed.sum()),raw_mask_changed=int((np.isfinite(aa)!=np.isfinite(bb)).sum()),
                                 observed_changed=int(a.observed.fillna(False).ne(b.observed.fillna(False)).sum()),
                                 q1_missing_window_rows=int((~b.observed.fillna(False)).sum()),
                                 input_changed_values=int((baseline!=window).sum()),
                                 input_abs_max=float(np.max(np.abs(baseline-window)))))
            if row in (0,len(entries)//2,len(entries)-1):
                original_z=F.normalize(model.final_norm(model.encoder(torch.from_numpy(baseline),normalize=False)),dim=-1).numpy()
                changes.append(dict(split=split,date=str(day.date()),vector_max_abs=float(np.abs(z-original_z).max()),
                                    vector_mean_abs=float(np.abs(z-original_z).mean())))
            if row%100==0: print("Native Q1 features",split,row,len(entries),flush=True)
        del model,native,native_raw
    pd.DataFrame(coverage).to_csv(OUT/"feature_coverage.csv",index=False)
    std.save("vector_diagnostic.json",changes)
    if phase=="test":
        _,_,baskets,*_=ctx["legacy_inputs"]()
        test_dates=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(PARENT/"signals.csv").date))
        test_ids=dates.get_indexer(test_dates)
        test_x=np.stack([ctx["stock_inputs"](result,t,baskets[t])[2] for t in test_ids])
        np.save(OUT/"test_inputs.npy",test_x)
    return result


def compare():
    current=pd.read_parquet(OUT/"weekly.parquet")
    parent=pd.read_parquet(PARENT/"weekly.parquet")
    both=pd.concat([current.assign(source="Q1"),
                    parent.query("arm in ['p6_plain','p6_fixed5']").assign(source="old")],ignore_index=True)
    metrics=pd.concat([pd.read_csv(OUT/"metrics.csv").assign(source="Q1"),
                       pd.read_csv(PARENT/"metrics.csv").query("arm in ['p6_plain','p6_fixed5']").assign(source="old")])
    paired=ctx["ctx"]["ctx"]["paired"]
    pairs=[]
    for (arm,rule,fee),g in both.query("arm in ['p6_plain','p6_fixed5']").groupby(["arm","rule","fee"]):
        q=g.pivot(index="date",columns="source",values="net")
        pairs.append(dict(arm=arm,rule=rule,fee=fee,**paired(q.Q1.to_numpy(),q.old.to_numpy())))
    metrics.to_csv(OUT/"comparison_metrics.csv",index=False)
    both.to_parquet(OUT/"comparison_weekly.parquet",index=False)
    pd.DataFrame(pairs).to_csv(OUT/"paired.csv",index=False)
    old_signals=pd.read_csv(PARENT/"signals.csv",parse_dates=["date"]).set_index("date")
    new_signals=pd.read_csv(OUT/"signals.csv",parse_dates=["date"]).set_index("date")
    for arm in ("p6_plain","p6_fixed5"):
        flipped=old_signals[arm].gt(0).ne(new_signals[arm].gt(0))
        print(arm,"source changed directions",flipped.sum(),new_signals.index[flipped].tolist(),flush=True)
    # Individual head evaluation uses the same P6 trading source as the ensemble.
    d,dates,baskets,*_=ctx["legacy_inputs"]()
    ids=dates.get_indexer(new_signals.index)
    preds={}
    for seed in (42,43,44):
        with np.load(ctx["cache_path"](seed)) as cache:
            old_vectors=cache["z"]
        old_x=np.stack([ctx["stock_inputs"](old_vectors,t,baskets[t])[2] for t in ids])
        for rep,x in (("old_triple",old_x),("p6",test_x)):
            row=next(r for r in (json.loads((OUT/"validation_lock.json").read_text(encoding="utf8"))["selections"])
                     if r["arm"]==f"{rep}_plain" and r["seed"]==seed)
            cp=torch.load(row["checkpoint"],weights_only=True)
            head=ctx["LegPool"](False).eval()
            head.load_state_dict(cp["state"])
            with torch.no_grad():
                preds[f"{rep}_{seed}"]=head((torch.from_numpy(x)-cp["mu"])/cp["sd"])[:,0].numpy()
    single_signals=pd.DataFrame(preds,index=new_signals.index)
    for rep in ("old_triple","p6"):
        averaged=single_signals[[f"{rep}_{seed}" for seed in (42,43,44)]].mean(axis=1)
        np.testing.assert_allclose(averaged,new_signals[f"{rep}_plain"],atol=1e-6,rtol=1e-6)
        np.testing.assert_array_equal(averaged.gt(0),new_signals[f"{rep}_plain"].gt(0))
    single_signals.to_csv(OUT/"single_head_signals.csv")
    frame,schedule,members=ctx["locked_history"]()
    sc=next(s for l,s in members.values() if l["arm"]=="P6")
    source=std.score_pair(frame,sc["test"]["long"],sc["test"]["short"],schedule)
    common=schedule.loc[schedule.date.isin(single_signals.index)]
    single_m,single_w=ctx["ctx"]["replay"](source,common,single_signals,"test")
    single_m.to_csv(OUT/"single_head_metrics.csv",index=False)
    single_w.to_parquet(OUT/"single_head_weekly.parquet",index=False)
    std.save("inputs.json",std.HASHES)
    print("Completed feature-only comparison and six single-head diagnostics",flush=True)


if __name__=="__main__":
    runner.update(OUT=OUT,fit=fit,extract=extract)
    ctx["main"]()
    compare()
