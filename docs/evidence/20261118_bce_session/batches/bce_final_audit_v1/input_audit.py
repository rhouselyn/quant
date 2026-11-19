"""Check legacy cache alignment and that inference excludes labels/future windows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from directional_three_seed import cache_path
from quant_mvp.hierarchy import HierarchicalRanker
from quant_mvp.hierarchy_data import arrays
from quant_mvp.hierarchy_engine import make_batch
from prepare_hierarchy_580_mon import path_features

OUT=Path(__file__).resolve().parent
torch.set_num_threads(4)
days=pd.to_datetime(["2021-01-08","2024-02-09","2024-09-20","2026-05-22","2026-10-09"])
reports=[]
for seed,name in ((42,"abl2000_base"),(43,"abl2000_base_s43"),(44,"abl2000_base_s44")):
    run=Path("outputs")/name
    cfg=yaml.safe_load((run/"config.yaml").read_text(encoding="utf8"))
    data=arrays(cfg)
    idx=pd.DatetimeIndex(data[6]).get_indexer(days)
    assert (idx>=cfg["lookback"]-1).all()
    model=HierarchicalRanker(cfg,data[8]).eval()
    model.load_state_dict(torch.load(run/"best.pt",map_location="cpu",weights_only=True))
    with np.load(cache_path(seed)) as cache:
        ci=pd.DatetimeIndex(pd.to_datetime(cache["dates"])).get_indexer(days)
        z=cache["z"][ci]
        s=cache["score"][ci]
        avail=cache["available"][ci]
    batch=make_batch(data,idx.tolist(),cfg,"cpu")
    with torch.inference_mode():
        _,actual,score,_=model(*batch)
    np.testing.assert_array_equal(avail,data[4][idx])
    np.testing.assert_allclose(actual.numpy(),z,atol=3e-6,rtol=3e-5)
    np.testing.assert_allclose(score.numpy(),s,atol=3e-6,rtol=3e-5)
    altered=list(data)
    altered[1]=np.full_like(data[1],123.)
    altered[2]=~data[2]
    for before,after in zip(batch,make_batch(altered,idx.tolist(),cfg,"cpu")):
        if before is not None:
            assert torch.equal(before,after)
    for t in idx:
        # Windows are already materialized here. Future rows cannot enter them.
        alternative=list(data)
        alternative[0]=data[0].copy()
        alternative[0][t+1:]=17.
        a,b=make_batch(data,[int(t)],cfg,"cpu"),make_batch(alternative,[int(t)],cfg,"cpu")
        assert all(x is None or torch.equal(x,y) for x,y in zip(a,b))
    reports.append(dict(seed=seed,sample_dates=days.astype(str).tolist(),
                        cache_vectors_and_scores_reproduced=True,labels_not_input=True,future_rows_not_input=True))
    print("Input audit",seed,"cache and causal windows verified",flush=True)
raw=Path("data/sw_file_20/raw_ak2000")
paths=list(raw.glob("*.parquet"))
assert paths
daily=pd.read_parquet(paths[0])
calendar=pd.to_datetime(daily.date)
test_day=pd.Timestamp("2024-09-20")
full=path_features(daily,freq="W-FRI")
prefix=path_features(daily.loc[calendar<=test_day],freq="W-FRI")
np.testing.assert_allclose(full.loc[test_day],prefix.loc[test_day],equal_nan=True,atol=0,rtol=0)
(OUT/"input_audit.json").write_text(json.dumps(dict(checks=reports,daily_path_symbol=paths[0].name,
    daily_path_prefix_equal=True,scope="sampled model reconstruction and source-path causality, not a complete PIT vendor audit"),
    indent=2),encoding="utf8")
print("Daily path prefix invariance verified",flush=True)
