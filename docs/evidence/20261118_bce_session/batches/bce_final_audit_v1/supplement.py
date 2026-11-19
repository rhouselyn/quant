"""Complete the repaired-calendar P6-signal comparison and missing-mark stress."""
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch
import yaml

from attention_followup import LegPool
from directional_three_seed import stock_inputs
from replay_equal_weight_model_books import locked_history,legacy_inputs
from quant_mvp.financial_return import build_model

OUT=Path(__file__).resolve().parent
ctx=runpy.run_path(str(OUT/"audit.py"))
std=runpy.run_path("outputs/standard_bce_pool_v1/run_standard.py")
torch.set_num_threads(4)
frame,schedule,members=locked_history()
signals=pd.read_csv(OUT/"calendar_signals.csv",parse_dates=["date"]).set_index("date")
schedule=schedule.loc[schedule.date.isin(signals.index)]
d,dates,baskets,*_=legacy_inputs()
known=pd.read_csv("outputs/p6_q1_features_v1/signals.csv",parse_dates=["date"]).set_index("date")
missing=signals.index.difference(known.index)
assert len(missing)==12
member=next(m for m in json.loads(std["CATALOG"].read_text(encoding="utf8"))["members"] if m["arm"]=="P6")
run=Path(member["run"])
cfg=yaml.safe_load((run/"config.yaml").read_text(encoding="utf8"))
opts,data=std["dataset"](std["STANDARD"]/"snapshots/history",run,cfg)
model=build_model(opts,data[8],data[9]).eval()
model.load_state_dict(torch.load(run/"epoch_8.pt",map_location="cpu",weights_only=True))
di=pd.DatetimeIndex(data[6])
assert list(data[7])==list(d["symbols"])
xs=[]
with torch.inference_mode():
    for day in missing:
        t=di.get_loc(day)
        b=baskets[dates.get_loc(day)]
        ids=np.r_[*b]
        window=data[0][t-39:t+1,ids].transpose(1,0,2)
        z=torch.nn.functional.normalize(model.final_norm(model.encoder(torch.from_numpy(window),normalize=False)),dim=-1).numpy()
        xs.append(np.column_stack((z,np.repeat([1.,-1.],[40,10]))).astype(np.float32))
    xx=torch.from_numpy(np.stack(xs))
    for arm,stored in (("p6_controlled","p6_plain"),("p6_controlled_fixed5","p6_fixed5")):
        preds=[]
        for seed in (42,43,44):
            cp=torch.load(Path("outputs/p6_q1_features_v1")/f"{stored}_{seed}.pt",weights_only=True)
            head=LegPool(False).eval()
            head.load_state_dict(cp["state"])
            preds.append(head((xx-cp["mu"])/cp["sd"])[:,0].numpy())
        signals[arm]=known[stored].reindex(signals.index)
        signals.loc[missing,arm]=np.mean(preds,axis=0)
assert signals.notna().all().all()
signals.to_csv(OUT/"complete_calendar_signals.csv")
scores=pd.concat([pd.DataFrame(dict(date=day,symbol=d["symbols"],score=d["score"][dates.get_loc(day)]))
                  for day in schedule.date],ignore_index=True)
key=["date","symbol"]
old=frame.loc[frame.date.isin(schedule.date)].merge(scores.rename(columns={"score":"score_long"}),on=key,how="left",validate="one_to_one")
old["score_short"]=old.score_long
sc=next(s for l,s in members.values() if l["arm"]=="P6")
p6=frame.loc[frame.date.isin(schedule.date)].merge(sc["test"]["long"][key+["score"]].rename(columns={"score":"score_long"}),on=key,how="left",validate="one_to_one")
p6=p6.merge(sc["test"]["short"][key+["score"]].rename(columns={"score":"score_short"}),on=key,how="left",validate="one_to_one")
rows,weeks=[],[]
for model_name,n,source in (("old_scores",40,old),("old_scores",20,old),("P6",20,p6)):
    for arm in ("p6_controlled","p6_controlled_fixed5"):
        for rule in ("wide","full"):
            share=pd.Series(np.where(signals[arm]>0,.7,.4),index=signals.index) if rule=="wide" else signals[arm].gt(0).astype(float)
            for fee,borrow in ((10,.0835),(30,.12)):
                w,_=ctx["independent"](source,schedule,n,share,fee,borrow)
                case=dict(model=model_name,long_slots=n,arm=arm,rule=rule,fee=fee,scenario="main")
                rows.append(dict(**case,**ctx["stats"](w,schedule)))
                weeks.append(w.assign(**case))
    for arm in ("unaugmented","fixed_5"):
        share=signals[arm].gt(0).astype(float)
        for fee,borrow in ((10,.0835),(30,.12)):
            w,_=ctx["independent"](source,schedule,n,share,fee,borrow,stress=True)
            rows.append(dict(model=model_name,long_slots=n,arm=arm,rule="full",fee=fee,
                             scenario="missing20",**ctx["stats"](w,schedule)))
    print("Calendar P6 head and stress",model_name,n,flush=True)
base=pd.read_csv(OUT/"calendar_metrics.csv").assign(scenario="main")
pd.concat([base,pd.DataFrame(rows)],ignore_index=True).to_csv(OUT/"final_metrics.csv",index=False)
pd.concat(weeks,ignore_index=True).to_parquet(OUT/"calendar_p6_weekly.parquet",index=False)
print(pd.DataFrame(rows).query("rule=='full'")[["model","long_slots","arm","scenario","fee","annual_calendar","max_drawdown"]].to_string(index=False),flush=True)
print("Completed full-calendar P6 comparison and missing-mark stress",flush=True)
