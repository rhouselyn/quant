"""Train a dedicated frozen-P6 allocation head for Top20/Bottom10."""
from pathlib import Path
import json
import os
import runpy
import time

import numpy as np
import pandas as pd
import torch
import yaml

from quant_mvp.financial_return import build_model
from evaluate_dataset_three_year import fuse_probability_epochs
from replay_equal_weight_model_books import locked_history

OUT = Path(__file__).resolve().parent
DEVICE = os.getenv("P6_BCE_DEVICE","cuda")
base = runpy.run_path(str(OUT.parent/"standard_bce_pool_v1/run_standard.py"))
std = base["fit"].__globals__
std["OUT"] = OUT
replay = std["replay"]
def replay20(source,dates,spec,**kwargs):
    return replay(source,dates,(20,10,*spec[2:]),**kwargs)
std["replay"] = replay20
record, save = base["record"], base["save"]


@torch.inference_mode()
def model_data(directory,run,cfg,training=False):
    options,data=base["dataset"](directory,run,cfg,training=training)
    model=build_model(options,data[8],data[9]).to(DEVICE).eval()
    model.load_state_dict(torch.load(run/"epoch_8.pt",map_location="cpu",weights_only=True))
    return options,data,model


def xrow(z,nlong,nshort):
    return base["xrow"](z,(np.repeat(np.arange(nlong),2),np.arange(nlong,nlong+nshort)))


@torch.inference_mode()
def evaluation_input(directory,run,cfg,source,schedule,with_labels):
    opts,data,model=model_data(directory,run,cfg)
    dates=pd.DatetimeIndex(data[6])
    symbols=pd.Index(data[7])
    features,labels=[],[]
    for date,g in source.loc[source.date.isin(schedule.date)].groupby("date",sort=True):
        targets=base["pick_targets"](g,20,10,.5,.5)
        names=[[s for side,s in targets if side==sign] for sign in (1,-1)]
        assert len(names[0])==20 and len(names[1])<=10
        ids=symbols.get_indexer(names[0]+names[1])
        t=dates.get_loc(date)
        assert (ids>=0).all() and data[4][t,ids].all()
        x=torch.from_numpy(data[0][t-cfg["lookback"]+1:t+1,ids].transpose(1,0,2)).to(DEVICE)
        z=torch.nn.functional.normalize(model.final_norm(model.encoder(x,normalize=False)),dim=-1).cpu().numpy()
        features.append(xrow(z,*map(len,names)))
        if with_labels:
            frame=g.set_index("symbol")
            usage,returns=[],[]
            for side,nn,denom in zip(("long","short"),names,(20,10)):
                f=frame.loc[nn]
                r=f.loc[f[side+"_entry"].astype(bool),side+"_r"]
                returns.append(r.sum(skipna=False)/denom)
                usage.append(len(r)/denom)
            labels.append(sum(returns)+usage[1]*.0835*g.borrow_days.iloc[0]/365)
    assert len(features)==len(schedule)
    return np.stack(features),np.asarray(labels)


@torch.inference_mode()
def training_input(directory,run,cfg):
    opts,data,model=model_data(directory,run,cfg,True)
    dates=pd.DatetimeIndex(data[6])
    trained=json.loads((run/"training_protocol.json").read_text(encoding="utf-8"))["training_dates"]
    idx=dates.get_indexer(pd.to_datetime(trained))
    assert len(idx)==779 and (idx>=0).all() and (dates[idx]<pd.Timestamp("2018-01-01")).all()
    symbols=np.asarray(data[7])
    amounts=pd.read_parquet(directory/"weekly.parquet",columns=["date","symbol","amount"])
    amt=amounts.pivot(index="date",columns="symbol",values="amount").reindex(index=dates,columns=symbols).to_numpy()
    latent=[]
    handle=model.final_norm.register_forward_hook(lambda m,inp,out:latent.append(out))
    x,y,kept=[],[],[]
    for ids in np.array_split(idx,(len(idx)+3)//4):
        if ids[0] in idx[::100]: print("Training features",int(np.searchsorted(idx,ids[0])),flush=True)
        latent.clear()
        out=model(*base["make_batch"](data,ids.tolist(),opts,DEVICE))
        score=out.get("decision_score",out["mu_log"]).cpu().numpy()
        assert len(latent)==1
        z=torch.nn.functional.normalize(latent[0],dim=-1).reshape(len(ids),len(symbols),128).cpu().numpy()
        for i,t in enumerate(ids):
            good=np.isfinite(score[i])
            long=np.lexsort((symbols,-score[i]))
            long=long[good[long]][:20]
            mask=good&np.isfinite(amt[t])&(amt[t]>0)
            liquid=pd.Series(amt[t,mask]).rank(pct=True).to_numpy()>.6
            pool=np.flatnonzero(mask)[liquid]
            pool=pool[~np.isin(pool,long)]
            short=pool[np.lexsort((symbols[pool],score[i,pool]))][:10]
            basket=np.r_[long,short]
            if len(long)!=20 or len(short)!=10 or not data[2][t,basket].all():
                continue
            r=np.expm1(data[1][t,basket])
            y.append(r[:20].mean()+r[20:].mean()+.0835*7/365)
            x.append(xrow(z[i,basket],20,10))
            kept.append(str(dates[t].date()))
    handle.remove()
    assert len(kept)>40
    print("P6 head train weeks",len(kept),"of",len(idx),flush=True)
    return np.stack(x),np.asarray(y),kept


def main():
    torch.set_num_threads(4)
    record(__file__)
    record(Path("docs/research_20261118_p6_own_bce_protocol.md"))
    m=next(m for m in json.loads(base["CATALOG"].read_text(encoding="utf-8"))["members"] if m["arm"]=="P6")
    run=Path(m["run"])
    for p in (run/"config.yaml",run/"preprocessing.json",run/"training_protocol.json",Path(m["lock"]),Path(m["validation_scores"])):
        record(p)
    cfg=yaml.safe_load((run/"config.yaml").read_text(encoding="utf-8"))
    lock=json.loads(Path(m["lock"]).read_text(encoding="utf-8"))
    assert record(run/"epoch_8.pt")==lock["checkpoint_sha256"]["8"]
    directory=base["STANDARD"]/"snapshots/validation"
    tape=base["TAPE"]
    vf,vs,_=base["phase_proxy"](directory,base["snapshot"](directory),tape,tape.with_name("eligibility.parquet"),phase="validation")
    raw=pd.read_parquet(m["validation_scores"])
    raw.date=pd.to_datetime(raw.date)
    source=base["score_pair"](vf,fuse_probability_epochs(raw,lock["long_epochs"],run.name),
                              fuse_probability_epochs(raw,lock["short_epochs"],run.name),vs)
    assert len(vs)==153
    x,y,kept=training_input(directory,run,cfg)
    vx,vy=evaluation_input(directory,run,cfg,source,vs,True)
    val=[]
    for seed in (42,43,44):
        val.append(base["fit"]("p6_bce",seed,x,y,np.ones(len(y)),vx,vy,len(y)))
        record(OUT/f"p6_bce_{seed}.pt")
    save("validation_lock.json",dict(selections=base["SELECT"],train_dates=kept,hashes=base["HASHES"].copy()))
    print("Validation locked",flush=True)

    frame,schedule,members=locked_history()
    scores=next(s for l,s in members.values() if l["arm"]=="P6")
    test=base["score_pair"](frame,scores["test"]["long"],scores["test"]["short"],schedule)
    signalfile=Path("outputs/old_longshort_bce_loss_v1/report/signals.parquet")
    record(signalfile)
    old=pd.read_parquet(signalfile,filters=[("split","=","test")]).set_index("date").freeze_projection
    common=schedule.loc[schedule.date.isin(old.index)]
    assert len(common)==283
    hx,_=evaluation_input(base["STANDARD"]/"snapshots/history",run,cfg,test,common,False)
    preds=[]
    for seed in (42,43,44):
        cp=torch.load(OUT/f"p6_bce_{seed}.pt",weights_only=True)
        head=base["LegPool"](False).eval()
        head.load_state_dict(cp["state"])
        with torch.no_grad():
            preds.append(head((torch.from_numpy(hx)-cp["mu"])/cp["sd"]).numpy().ravel())
    rows,books=[],[]
    base["backtests"](test,common,{"plain":[old.reindex(common.date).to_numpy()],"p6_bce":preds},"test",rows,books)
    metrics=pd.DataFrame(rows)
    w=pd.concat(books,ignore_index=True)
    reference=pd.read_parquet("outputs/b0_p6_frozen_bce_transfer_v1/report/weekly.parquet")
    reference=reference.query("model=='P6'").replace({"full_switch":"full"})
    cmp=w.query("arm=='plain'").merge(reference,left_on=["rule","fee","date"],right_on=["arm","trade_bps","date"])
    assert len(cmp)==1698
    np.testing.assert_allclose(cmp.net_x,cmp.net_y,rtol=0,atol=1e-12)
    metrics.to_csv(OUT/"metrics.csv",index=False)
    w.to_parquet(OUT/"weekly.parquet",index=False)
    pd.DataFrame(base["TRACE"]).to_csv(OUT/"training_loss.csv",index=False)
    switches=w.query("fee==10 and rule=='full'").pivot(index="date",columns="arm",values="logit")
    print("Direction disagreements",int(switches.plain.gt(0).ne(switches.p6_bce.gt(0)).sum()),flush=True)
    print(metrics[["arm","rule","fee","annual_calendar","max_drawdown"]].to_string(index=False),flush=True)
    print("Complete old P6 controls reproduced",flush=True)

if __name__=="__main__":
    if DEVICE=="cpu":
        main()
        raise SystemExit(0)
    guard=OUT.parent/"dataset_v2/gpu.lock"
    print("Waiting for shared GPU",flush=True)
    while True:
        try:
            with guard.open("x") as f: json.dump(dict(pid=os.getpid(),task=OUT.name),f)
            break
        except FileExistsError: time.sleep(5)
    try: main()
    finally: guard.unlink()
