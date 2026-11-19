"""Repair the evaluation calendar using existing frozen heads, with no label gate."""
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch

from attention_followup import LegPool
from directional_three_seed import cache_path,stock_inputs
from replay_equal_weight_model_books import legacy_inputs,locked_history

OUT=Path(__file__).resolve().parent
audit=runpy.run_path(str(OUT/"audit.py"))
independent,stats=audit["independent"],audit["stats"]
torch.set_num_threads(4)
for path in (__file__,OUT/"audit.py","docs/research_20261118_bce_calendar_audit_protocol.md"):
    audit["record"](path)
frame,schedule,members=locked_history()
old=pd.read_csv("outputs/bce_fixed5_ensemble_v1/signals.csv",parse_dates=["date"]).set_index("date")
schedule=schedule.loc[schedule.date.between(old.index.min(),old.index.max())]
assert len(schedule)==295
# The scheduled opening-to-opening intervals cover the complete corrected span.
np.testing.assert_array_equal(schedule.exit_at.iloc[:-1],schedule.entry_at.iloc[1:])
d,dates,baskets,_,_,train=legacy_inputs()
idx=dates.get_indexer(schedule.date)
assert (idx>=0).all()
assert all(tuple(map(len,baskets[t]))==(40,10) for t in idx)
preds={arm:[] for arm in ("unaugmented","fixed_5")}
for seed in (42,43,44):
    with np.load(cache_path(seed)) as cache:
        vectors=cache["z"]
    x=torch.from_numpy(np.stack([stock_inputs(vectors,t,baskets[t])[2] for t in idx]))
    for arm,batch,stem in (("unaugmented","bce_resample_steps_v1","plain"),("fixed_5","bce_resample_low_grid_v1","fixed_5")):
        path=Path("outputs")/batch/f"{stem}_{seed}.pt"
        audit["record"](path)
        cp=torch.load(path,weights_only=True)
        head=LegPool(False).eval()
        head.load_state_dict(cp["state"])
        with torch.no_grad():
            preds[arm].append(head((x-cp["mu"])/cp["sd"])[:,0].numpy())
signals=pd.DataFrame({arm:np.mean(p,axis=0) for arm,p in preds.items()},index=pd.DatetimeIndex(schedule.date).rename("date"))
np.testing.assert_allclose(signals.reindex(old.index).unaugmented,old.plain,atol=1e-6,rtol=1e-6)
np.testing.assert_array_equal(signals.reindex(old.index).fixed_5.gt(0),old.mean_logit.gt(0))
signals.to_csv(OUT/"calendar_signals.csv")
scores=pd.concat([pd.DataFrame(dict(date=day,symbol=d["symbols"],score=d["score"][dates.get_loc(day)]))
                  for day in schedule.date],ignore_index=True)
key=["date","symbol"]
oldsource=frame.loc[frame.date.isin(schedule.date)].merge(scores.rename(columns={"score":"score_long"}),on=key,how="left",validate="one_to_one")
oldsource["score_short"]=oldsource.score_long
p6scores=next(s for l,s in members.values() if l["arm"]=="P6")
p6source=frame.loc[frame.date.isin(schedule.date)].merge(p6scores["test"]["long"][key+["score"]].rename(columns={"score":"score_long"}),on=key,how="left",validate="one_to_one")
p6source=p6source.merge(p6scores["test"]["short"][key+["score"]].rename(columns={"score":"score_short"}),on=key,how="left",validate="one_to_one")
rows,weeks=[],[]
for model,slots,source in (("old_scores",40,oldsource),("old_scores",20,oldsource),("P6",20,p6source)):
    for arm in ("unaugmented","fixed_5","fixed50"):
        for rule in (("wide","full") if arm!="fixed50" else ("fixed50",)):
            if rule=="fixed50": shares=pd.Series(.5,index=signals.index)
            elif rule=="wide": shares=pd.Series(np.where(signals[arm]>0,.7,.4),index=signals.index)
            else: shares=signals[arm].gt(0).astype(float)
            for fee,borrow in ((10,.0835),(30,.12)):
                w,e=independent(source,schedule,slots,shares,fee,borrow)
                case=dict(model=model,long_slots=slots,arm=arm,rule=rule,fee=fee)
                rows.append(dict(**case,**stats(w,schedule),borrow=borrow))
                weeks.append(w.assign(**case))
    print("Repaired calendar",model,slots,flush=True)
pd.DataFrame(rows).to_csv(OUT/"calendar_metrics.csv",index=False)
pd.concat(weeks,ignore_index=True).to_parquet(OUT/"calendar_weekly.parquet",index=False)
(OUT/"calendar_inputs.json").write_text(json.dumps(audit["HASHES"],indent=2),encoding="utf8")
print(pd.DataFrame(rows).query("rule=='full'")[["model","long_slots","arm","fee","annual_calendar","max_drawdown","sharpe"]].to_string(index=False),flush=True)
print("Completed 295-week full-calendar correction with frozen signals",flush=True)
