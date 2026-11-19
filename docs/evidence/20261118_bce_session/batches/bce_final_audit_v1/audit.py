"""Independent signed-weight accounting and timing audit of frozen BCE overlays."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from directional_three_seed import cache_path
from replay_equal_weight_model_books import locked_history, legacy_inputs
from eval_weekly_architecture import pick_targets

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
HASHES={}


def record(path):
    path=Path(path)
    with path.open("rb") as handle:
        HASHES[str(path.resolve())]=hashlib.file_digest(handle,"sha256").hexdigest()


def stats(week,schedule):
    r=week.net.to_numpy()
    nav=np.r_[1.,np.cumprod(1+r)]
    years=(schedule.exit_at.iloc[-1]-schedule.entry_at.iloc[0]).total_seconds()/86400/365.2425
    return dict(weeks=len(r),annual_calendar=float(nav[-1]**(1/years)-1),
                max_drawdown=float(np.max(1-nav/np.maximum.accumulate(nav))),
                sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(52)),
                worst10_avg=float(np.sort(r)[:int(np.ceil(.1*len(r)))].mean()),
                long_bp=float(week.long_net.mean()*1e4),short_bp=float(week.short_net.mean()*1e4))


def independent(source,schedule,long_slots,shares,fee,borrow,stress=False):
    symbols=pd.Index(sorted(source.symbol.unique()))
    prev=np.zeros(len(symbols))
    weeks,events=[],[]
    for day,g in source.groupby("date",sort=True):
        if day not in shares.index:
            continue
        budget=float(shares.at[day])
        target=pick_targets(g,long_slots,10,budget,1-budget)
        rows=g.set_index("symbol")
        held=np.zeros(len(symbols))
        asset=np.zeros(len(symbols))
        missing_weight,blocked_weight=0.,0.
        for (side,symbol),weight in target.items():
            row=rows.loc[symbol]
            prefix="long" if side>0 else "short"
            entry=bool(row[prefix+"_entry"])
            if weight==0 or not entry:
                continue
            idx=symbols.get_loc(symbol)
            held[idx]=side*weight
            value=float(row[prefix+"_r"])
            missing=not np.isfinite(value)
            blocked=bool(row[prefix+"_close"])
            asset[idx]=(-side*.2 if stress and missing else (0. if missing else value))
            missing_weight+=weight*missing
            blocked_weight+=weight*blocked
            events.append(dict(date=day,side=side,symbol=symbol,weight=weight,asset_r=value,
                               missing=missing,blocked_exit=blocked))
        gross=float(held@asset)
        transaction=float(np.abs(held-prev).sum()*fee/1e4)
        borrow_cost=float(-held[held<0].sum()*borrow*float(g.borrow_days.iloc[0])/365)
        long_cost=float(np.abs(np.maximum(held,0)-np.maximum(prev,0)).sum()*fee/1e4)
        short_cost=transaction-long_cost
        net=gross-transaction-borrow_cost
        assert np.isfinite(net) and net>-1 and np.abs(held).sum()<=1+1e-12
        weeks.append(dict(date=day,net=net,gross_ret=gross,trading_cost=transaction,borrow_cost=borrow_cost,
                          long_net=float(np.maximum(held,0)@asset-long_cost),
                          short_net=float(np.minimum(held,0)@asset-short_cost-borrow_cost),
                          missing_weight=missing_weight,blocked_weight=blocked_weight))
        prev=held*(1+asset)/(1+net)
    week=pd.DataFrame(weeks)
    np.testing.assert_array_equal(week.date,schedule.date)
    np.testing.assert_allclose(week.long_net+week.short_net,week.net,atol=1e-12,rtol=0)
    return week,pd.DataFrame(events)


def main():
    for path in (__file__,"docs/research_20261118_bce_final_audit_protocol.md",
                 "scripts/eval_weekly_architecture.py","scripts/replay_old_longshort_bce_loss.py",
                 "scripts/attention_context_ablation.py","scripts/prepare_hierarchy_580_mon.py",
                 "src/quant_mvp/hierarchy_data.py","src/quant_mvp/hierarchy_engine.py",
                 "outputs/bce_fixed5_ensemble_v1/signals.csv",
                 "outputs/bce_old_selector_transfer_v1/weekly.parquet",
                 "outputs/p6_q1_features_v1/signals.csv"):
        record(path)
    frame,schedule,members=locked_history()
    d,dates,baskets,gaps,complete,train=legacy_inputs()
    sig=pd.read_csv("outputs/bce_fixed5_ensemble_v1/signals.csv",parse_dates=["date"],
                    float_precision="round_trip").set_index("date")
    common=schedule.loc[schedule.date.isin(sig.index)]
    assert len(common)==283 and len(train)==473
    np.testing.assert_array_equal(common.date,sig.index)
    q1=pd.read_csv("outputs/p6_q1_features_v1/signals.csv",parse_dates=["date"]).set_index("date")
    signals=sig[["plain","mean_logit"]].rename(columns={"plain":"unaugmented","mean_logit":"fixed_5"})
    signals["p6_controlled"]=q1.p6_plain.reindex(sig.index)
    missing=[]
    for day in schedule.loc[~schedule.date.isin(sig.index)].date:
        t=dates.get_loc(day)
        missing.append(dict(date=day,long_count=len(baskets[t][0]),short_count=len(baskets[t][1]),
                            any_future_valid=bool(d["valid"][t].any()),feature_available=int(d["available"][t].sum())))
    pd.DataFrame(missing).to_csv(OUT/"omitted_weeks.csv",index=False)
    print("Calendar audit omitted weeks",len(missing),pd.DataFrame(missing).to_dict(orient="records"),flush=True)
    original=pd.read_parquet("outputs/old_longshort_bce_loss_v1/report/signals.parquet")
    v=original.query("split=='val'")
    vi=dates.get_indexer(v.date)
    assert len(v)==133 and complete[vi].sum()==123
    assert dates[train].max()<v.date.min() and v.date.max()<sig.index.min()
    assert not set(dates[train])&set(sig.index)
    protocols=[]
    for seed,name in ((42,"abl2000_base"),(43,"abl2000_base_s43"),(44,"abl2000_base_s44")):
        path=Path("outputs")/name
        for f in ("config.yaml","training_protocol.json","best.pt"):
            record(path/f)
        record(cache_path(seed))
        protocol=json.loads((path/"training_protocol.json").read_text(encoding="utf8"))
        tt=pd.to_datetime(protocol["training_dates"])
        assert tt.max()<pd.Timestamp("2018-01-01")
        protocols.append(dict(seed=seed,train_start=str(tt.min()),train_end=str(tt.max()),training_anchors=len(tt)))
    # Recompute all predictions from saved heads without any label as an input.
    from attention_followup import LegPool
    from directional_three_seed import stock_inputs
    import torch
    torch.set_num_threads(4)
    test_idx=dates.get_indexer(sig.index)
    predictions={k:[] for k in ("unaugmented","fixed_5")}
    checkpoint_info=[]
    for seed in (42,43,44):
        with np.load(cache_path(seed)) as cache:
            vectors=cache["z"]
        x=torch.from_numpy(np.stack([stock_inputs(vectors,t,baskets[t])[2] for t in test_idx]))
        for arm,base,stem in (("unaugmented","bce_resample_steps_v1","plain"),("fixed_5","bce_resample_low_grid_v1","fixed_5")):
            path=Path("outputs")/base/f"{stem}_{seed}.pt"
            record(path)
            cp=torch.load(path,weights_only=True)
            model=LegPool(False).eval()
            model.load_state_dict(cp["state"])
            with torch.no_grad():
                predictions[arm].append(model((x-cp["mu"])/cp["sd"])[:,0].numpy())
            cpmeta=json.loads((path.parent/"validation_lock.json").read_text(encoding="utf8"))
            chosen=next(r for r in cpmeta["selections"] if r["seed"]==seed and r["arm"]==stem)
            checkpoint_info.append(dict(arm=arm,seed=seed,step=chosen["step"]))
    # fixed_5 is the single sampling-seed version; its directions match the five-seed aggregation.
    np.testing.assert_allclose(np.mean(predictions["unaugmented"],axis=0),signals.unaugmented,rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(np.mean(predictions["fixed_5"],axis=0)>0,signals.fixed_5.gt(0))
    print("Checkpoint forecasts and split boundaries verified",flush=True)
    keys=["date","symbol"]
    old_score=pd.concat([pd.DataFrame(dict(date=day,symbol=d["symbols"],score=d["score"][dates.get_loc(day)]))
                         for day in common.date],ignore_index=True)
    old=frame.loc[frame.date.isin(common.date)].merge(
        old_score.rename(columns={"score":"score_long"}),on=keys,how="left",validate="one_to_one")
    old["score_short"]=old.score_long
    p6_scores=next(s for l,s in members.values() if l["arm"]=="P6")
    p6=frame.loc[frame.date.isin(common.date)].merge(
        p6_scores["test"]["long"][keys+["score"]].rename(columns={"score":"score_long"}),
        on=keys,how="left",validate="one_to_one").merge(
        p6_scores["test"]["short"][keys+["score"]].rename(columns={"score":"score_short"}),
        on=keys,how="left",validate="one_to_one")
    assert (frame.decision_at<frame.entry_at).all() and (frame.entry_at<frame.exit_at).all()
    known=frame.short_snapshot.notna()
    assert (pd.to_datetime(frame.loc[known,"short_snapshot"],utc=True)<pd.to_datetime(frame.loc[known,"decision_at"],utc=True)).all()
    existing=pd.read_parquet("outputs/bce_old_selector_transfer_v1/weekly.parquet")
    rows,weeks,events=[],[],[]
    for model,slots,source in (("old_scores",40,old),("old_scores",20,old),("P6",20,p6)):
        for arm in ("unaugmented","fixed_5","p6_controlled","fixed50"):
            z=signals[arm] if arm!="fixed50" else pd.Series(0.,index=signals.index)
            for rule in (("wide","full") if arm!="fixed50" else ("fixed50",)):
                shares=pd.Series(.5,index=signals.index) if rule=="fixed50" else (
                    pd.Series(np.where(z>0,.7,.4),index=z.index) if rule=="wide" else z.gt(0).astype(float))
                for fee,borrow in ((10,.0835),(30,.12)):
                    week,event=independent(source,common,slots,shares,fee,borrow)
                    case=dict(model=model,long_slots=slots,arm=arm,rule=rule,fee=fee)
                    if arm in ("unaugmented","fixed_5"):
                        ref=existing.query("model==@model and long_slots==@slots and arm==@arm and rule==@rule and fee==@fee")
                        merged=week.merge(ref,on="date",validate="one_to_one")
                        assert len(merged)==283
                        for col in ("net","gross_ret","trading_cost","borrow_cost"):
                            np.testing.assert_allclose(merged[col+"_x"],merged[col+"_y"],atol=1e-12,rtol=0)
                    rows.append(dict(**case,scenario="main",**stats(week,common),
                                     missing_weeks=int(week.missing_weight.gt(0).sum()),
                                     blocked_weeks=int(week.blocked_weight.gt(0).sum())))
                    weeks.append(week.assign(**case))
                    if arm in ("unaugmented","fixed_5") and rule=="full":
                        stress,_=independent(source,common,slots,shares,fee,borrow,stress=True)
                        rows.append(dict(**case,scenario="missing20",**stats(stress,common)))
                    if arm in ("unaugmented","fixed_5") and rule=="full" and fee==10:
                        events.append(event.assign(**case))
        print(model,slots,"independent ledgers verified",flush=True)
    stats_df=pd.DataFrame(rows)
    week_df=pd.concat(weeks,ignore_index=True)
    event_df=pd.concat(events,ignore_index=True)
    stats_df.to_csv(OUT/"metrics.csv",index=False)
    week_df.to_parquet(OUT/"weekly.parquet",index=False)
    event_df.to_parquet(OUT/"events.parquet",index=False)
    # Independently check filled-position simple returns against daily adjusted opening prices.
    hfq=Path("data/akshare_580_daily/daily_hfq/daily.parquet")
    record(hfq)
    event=event_df.merge(common[["date","entry_at","exit_at"]],on="date",validate="many_to_one")
    event["entry_date"]=event.entry_at.dt.tz_localize(None).dt.normalize()
    event["exit_date"]=event.exit_at.dt.tz_localize(None).dt.normalize()
    days=sorted(set(event.entry_date)|set(event.exit_date))
    daily=pd.read_parquet(hfq,columns=["date","symbol","open"],filters=[("date","in",days)])
    entry=daily.rename(columns={"date":"entry_date","open":"open_entry"})
    exit_=daily.rename(columns={"date":"exit_date","open":"open_exit"})
    price=event.merge(entry,on=["symbol","entry_date"],how="left",validate="many_to_one").merge(
        exit_,on=["symbol","exit_date"],how="left",validate="many_to_one")
    good=price.asset_r.notna()&price.open_entry.gt(0)&price.open_exit.gt(0)
    np.testing.assert_allclose(price.loc[good,"asset_r"],price.loc[good,"open_exit"]/price.loc[good,"open_entry"]-1,
                               rtol=1e-10,atol=1e-10)
    audit=dict(independent_ledger_matching=True,signal_recomputed=True,training_separation=protocols,
               selected_steps=checkpoint_info,matched_price_rows=int(good.sum()),held_event_rows=len(price),
               historical_exposure="repeatedly read validation and test",omitted_schedule_weeks=missing)
    (OUT/"audit.json").write_text(json.dumps(audit,default=str,indent=2),encoding="utf8")
    (OUT/"inputs.json").write_text(json.dumps(HASHES,indent=2),encoding="utf8")
    print("Independent ledger, price ratios, prediction recomputation and timing audit complete",flush=True)
    print(stats_df.query("scenario=='main' and rule=='full'")[["model","long_slots","arm","fee","annual_calendar","max_drawdown","sharpe"]].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
