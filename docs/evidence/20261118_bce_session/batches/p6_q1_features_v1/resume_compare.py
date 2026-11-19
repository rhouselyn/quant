"""Resume the diagnostic after eliminating repeated decompression in its reader."""
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch

OUT=Path(__file__).resolve().parent
ctx=runpy.run_path(str(OUT/"run.py"))
state=ctx["compare"].__globals__
parent=ctx["runner"]
parent["OUT"]=OUT
std=ctx["std"]
torch.set_num_threads(4)
old_inputs=OUT/"inputs.json"
for path,digest in ctx["json"].loads(old_inputs.read_text(encoding="utf8")).items():
    if path!=str((OUT/"run.py").resolve()):
        assert std.record(path)==digest,path
for path in (__file__,OUT/"run.py",OUT/"validation_lock.json"):
    std.record(path)
data,dates,baskets,*_=ctx["ctx"]["legacy_inputs"]()
days=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ctx["PARENT"]/"signals.csv").date))
ids=dates.get_indexer(days)
requested={t:set(np.r_[*baskets[t]]) for t in ids}
with np.load(ctx["ctx"]["cache_path"](42)) as cache:
    legacy=cache["z"]
member=next(m for m in ctx["json"].loads(std.CATALOG.read_text(encoding="utf8"))["members"] if m["arm"]=="P6")
prior=pd.read_csv(OUT/"feature_coverage.csv")
prior_changes=ctx["json"].loads((OUT/"vector_diagnostic.json").read_text())
if set(prior.split)!={"train","val","test"}:
    train_days=pd.DatetimeIndex(pd.to_datetime(ctx["lock"]["train_dates"]))
    val_days=pd.DatetimeIndex(pd.to_datetime(pd.read_parquet("outputs/old_longshort_bce_loss_v1/report/signals.parquet")
                                            .query("split=='val'").date))
    tv=dates.get_indexer(train_days.union(val_days))
    vectors=np.load(OUT/"p6_trainval_vectors.npy",mmap_mode="r")
    train_requested={t:set(np.flatnonzero(np.any(vectors[t]!=0,axis=1))) for t in tv}
    ctx["extract"](data,dates,train_requested,legacy,Path(member["run"]))
    state["coverage"].extend(prior.loc[prior.split.eq("test")].to_dict(orient="records"))
    state["changes"].extend(r for r in prior_changes if r["split"]=="test")
    pd.DataFrame(state["coverage"]).to_csv(OUT/"feature_coverage.csv",index=False)
    std.save("vector_diagnostic.json",state["changes"])
else:
    state["coverage"]=prior.loc[prior.split.ne("test")].to_dict(orient="records")
    state["changes"]=[r for r in prior_changes if r["split"]!="test"]
cache=OUT/"test_inputs.npy"
if cache.exists():
    state["test_x"]=np.load(cache)
    assert state["test_x"].shape==(283,50,129)
    std.record(cache)
else:
    ctx["extract"](data,dates,requested,legacy,Path(member["run"]))
original=std.replay
std.replay=lambda frame,days,spec,**kw:original(frame,days,(20,10,*spec[2:]),**kw)
ctx["compare"]()
