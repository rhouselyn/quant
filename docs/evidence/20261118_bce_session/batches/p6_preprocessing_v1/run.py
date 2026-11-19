"""Change only P6's input preprocessing on the fixed legacy panel."""
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
diagnostic = []


def fit(x,y,w,vx,vy,seed,arm,traces):
    if arm.startswith("p6_"):
        return parent_fit(x,y,w,vx,vy,seed,arm,traces)
    row = next(r for r in lock["selections"] if r["arm"]==arm and r["seed"]==seed)
    assert std.record(row["checkpoint"]) == row["sha256"]
    traces.extend(trace.query("arm==@arm and seed==@seed").to_dict(orient="records"))
    return row


@torch.inference_mode()
def extract(data, dates, requested, legacy, p6run):
    cfg = yaml.safe_load(Path("outputs/abl2000_base/config.yaml").read_text(encoding="utf8"))
    pre = json.loads((p6run/"preprocessing.json").read_text(encoding="utf8"))
    for p in (__file__,"docs/research_20261118_p6_preprocessing_protocol.md",
              PARENT/"run.py",PARENT/"validation_lock.json","src/quant_mvp/hierarchy_data.py",
              *(cfg[k] for k in ("cache_path","market_path","universe_path"))):
        std.record(p)
    original = ctx["arrays"](cfg)
    opts = dict(cfg,dataset_schema_version=2,scores_only=True,frozen_preprocessing=pre,
                normalization=pre["normalization"])
    updated = ctx["arrays"](opts)
    np.testing.assert_array_equal(original[6],updated[6])
    np.testing.assert_array_equal(original[7],updated[7])
    np.testing.assert_array_equal(original[4],updated[4])
    assert updated[9]["feature_mean"] == pre["feature_mean"]
    assert updated[9]["feature_std"] == pre["feature_std"]
    diff=updated[0]-original[0]
    if not diagnostic:
        diagnostic.append(dict(kind="input",mean_absolute=float(np.abs(diff).mean()),
                               maximum_absolute=float(np.abs(diff).max()),
                               fraction_changed_1e6=float((np.abs(diff)>1e-6).mean()),
                               normalization=pre["normalization"]))
    assert list(updated[7]) == list(data["symbols"])
    all_dates = pd.DatetimeIndex(updated[6])
    pcfg = yaml.safe_load((p6run/"config.yaml").read_text(encoding="utf8"))
    model = ctx["build_model"](pcfg,updated[8],pre).eval()
    model.load_state_dict(torch.load(p6run/"epoch_8.pt",map_location="cpu",weights_only=True))
    result=np.zeros_like(legacy)
    vec_abs,vec_count,vec_max=0.,0,0.
    for row,(t,names) in enumerate(sorted(requested.items())):
        ti=all_dates.get_loc(dates[t])
        names=np.array(sorted(names))
        assert ti>=cfg["lookback"]-1 and original[4][ti,names].all()
        window=updated[0][ti-cfg["lookback"]+1:ti+1,names].transpose(1,0,2)
        z=F.normalize(model.final_norm(model.encoder(torch.from_numpy(window),normalize=False)),dim=-1)
        result[t,names]=z.numpy()
        if row in (0,len(requested)//2,len(requested)-1):
            oldwindow=original[0][ti-cfg["lookback"]+1:ti+1,names].transpose(1,0,2)
            oldz=F.normalize(model.final_norm(model.encoder(torch.from_numpy(oldwindow),normalize=False)),dim=-1).numpy()
            delta=np.abs(z.numpy()-oldz)
            vec_abs+=delta.sum()
            vec_count+=delta.size
            vec_max=max(vec_max,float(delta.max()))
        if row%100==0: print("P6 native preprocessing",row,len(requested),flush=True)
    diagnostic.append(dict(kind="vectors",phase="trainval" if len(requested)==606 else "test",
                           mean_absolute=float(vec_abs/vec_count),maximum_absolute=vec_max))
    std.save("preprocessing_diagnostic.json",diagnostic)
    return result


if __name__=="__main__":
    runner.update(OUT=OUT,fit=fit,extract=extract)
    ctx["main"]()
    current=pd.read_parquet(OUT/"weekly.parquet")
    parent=pd.read_parquet(PARENT/"weekly.parquet")
    current_metrics=pd.read_csv(OUT/"metrics.csv")
    prior_metrics=pd.read_csv(PARENT/"metrics.csv")
    all_weekly=pd.concat([current.assign(preprocessing="p6_native"),
                         parent.query("arm in ['p6_plain','p6_fixed5']").assign(preprocessing="legacy")],ignore_index=True)
    all_metrics=pd.concat([current_metrics.assign(preprocessing="p6_native"),
                          prior_metrics.query("arm in ['p6_plain','p6_fixed5']").assign(preprocessing="legacy")],ignore_index=True)
    pairs=[]
    for (arm,rule,fee),g in all_weekly.query("arm in ['p6_plain','p6_fixed5']").groupby(["arm","rule","fee"]):
        q=g.pivot(index="date",columns="preprocessing",values="net")
        pairs.append(dict(arm=arm,rule=rule,fee=fee,**ctx["ctx"]["ctx"]["paired"](q.p6_native.to_numpy(),q.legacy.to_numpy())))
    pd.DataFrame(pairs).to_csv(OUT/"paired.csv",index=False)
    all_weekly.to_parquet(OUT/"comparison_weekly.parquet",index=False)
    all_metrics.to_csv(OUT/"comparison_metrics.csv",index=False)
    prior_s=pd.read_csv(PARENT/"signals.csv",parse_dates=["date"]).set_index("date")
    new_s=pd.read_csv(OUT/"signals.csv",parse_dates=["date"]).set_index("date")
    for arm in ("p6_plain","p6_fixed5"):
        print(arm,"changed directions versus old preprocessing",
              int(prior_s[arm].gt(0).ne(new_s[arm].gt(0)).sum()),flush=True)
    print("Preprocessing-only comparison complete",flush=True)
