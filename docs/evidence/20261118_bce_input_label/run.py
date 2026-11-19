"""Factor input size, supervision, normalization and augmentation independently."""
import copy
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from directional_three_seed import cache_path,stock_inputs
from directional_2018_experiment import groups_for
from replay_equal_weight_model_books import legacy_inputs,locked_history
from eval_weekly_architecture import pick_targets

OUT=Path(__file__).resolve().parent
ctx=runpy.run_path("outputs/bce_basket_sizes_v1/run.py")
Pool=ctx["BasketPool"]
HASHES={}


def record(p):
    p=Path(p).resolve()
    with p.open("rb") as f: HASHES[str(p)]=hashlib.file_digest(f,"sha256").hexdigest()


def fit(x,y,w,vx,vy,mu,sd,seed,nl,arm,fixed_step,trace):
    tx=torch.from_numpy((x-mu)/sd);vv=torch.from_numpy((vx-mu)/sd)
    target=torch.tensor(y>0,dtype=torch.float32);vt=torch.tensor(vy>0,dtype=torch.float32)
    wt=torch.tensor(w,dtype=torch.float32)
    torch.manual_seed(seed)
    head=Pool(nl,10)
    opt=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=.01)
    best=np.inf
    for step in range(1,101):
        if step==11:
            head.proj.requires_grad_(False)
            frozen=[p.detach().clone() for p in head.proj.parameters()]
        opt.zero_grad()
        loss=(F.binary_cross_entropy_with_logits(head(tx)[:,0],target,reduction="none")*wt).sum()/473
        loss.backward();opt.step()
        with torch.no_grad():value=F.binary_cross_entropy_with_logits(head(vv)[:,0],vt).item()
        trace.append(dict(arm=arm,seed=seed,step=step,train_bce=float(loss),val_bce=value))
        if value<best: best,chosen,state=value,step,copy.deepcopy(head.state_dict())
        if step==fixed_step: fixed=copy.deepcopy(head.state_dict())
    assert all(torch.equal(a,b) for a,b in zip(frozen,head.proj.parameters()))
    path=OUT/f"{arm}_{seed}.pt"
    torch.save(dict(state=state,fixed=fixed,mu=torch.from_numpy(mu),sd=torch.from_numpy(sd),nl=nl),path)
    record(path)
    print(arm,seed,"selected",chosen,"fixed",fixed_step,flush=True)
    return dict(arm=arm,seed=seed,step=chosen,val_bce=best,path=str(path),fixed_step=fixed_step)


def main():
    torch.set_num_threads(4)
    for p in (__file__,"docs/research_20261118_bce_input_label_protocol.md","outputs/bce_basket_sizes_v1/run.py",
              "scripts/attention_followup.py","outputs/bce_final_audit_v1/audit.py"): record(p)
    d,dates,baskets,gaps,complete,tr=legacy_inputs()
    sig=pd.read_parquet("outputs/old_longshort_bce_loss_v1/report/signals.parquet")
    vi=dates.get_indexer(sig.query("split=='val'").date); va=vi[complete[vi]]
    assert len(tr)==473 and len(va)==123
    groups=groups_for(d,dates)
    active=set(np.random.default_rng(61118).choice(473,236,replace=False))
    def gap(t,b):
        assert d["valid"][t,np.r_[*b]].all()
        return np.float32(np.expm1(d["y"][t,b[0]]).mean()+np.expm1(d["y"][t,b[1]]).mean()+.08/52)
    draws={}
    for sampler,nl in (("paired",40),("native",20)):
        lists=[]
        for row,t in enumerate(tr):
            variants=[]
            if row in active:
                seen={tuple(map(tuple,(groups[t][0][:nl],groups[t][1][:10])))}
                rng=np.random.default_rng(np.random.SeedSequence([61118,int(t),1]))
                for _ in range(3):
                    pair=tuple(pool[np.sort(rng.choice(len(pool),n,replace=False))]
                               for pool,n in ((groups[t][0][:50],nl),(groups[t][1][:20],10)))
                    key=tuple(map(tuple,pair))
                    if key not in seen and d["valid"][t,np.r_[*pair]].all():
                        seen.add(key);variants.append(pair)
            lists.append(variants)
        draws[sampler]=lists
    configs=[("A",40,40,"old","paired"),("B",20,40,"old","paired"),
             ("C",40,20,"old","paired"),("D",20,20,"old","paired"),
             ("E",20,20,"own","paired"),("F",20,20,"old","native"),
             ("G",20,20,"own","native")]
    traces,selections,identity=[],[],[]
    for seed in (42,43,44):
        record(cache_path(seed))
        with np.load(cache_path(seed)) as cache:z=cache["z"]
        normcp=torch.load(Path("outputs/bce_resample_steps_v1")/f"plain_{seed}.pt",weights_only=True)
        oldmu,oldsd=normcp["mu"].numpy(),normcp["sd"].numpy()
        for name,nl,ln,norm,sampler in configs:
            def pair(t,n):return groups[t][0][:n],groups[t][1][:10]
            base=np.stack([stock_inputs(z,t,pair(t,nl))[2] for t in tr])
            vx=np.stack([stock_inputs(z,t,pair(t,nl))[2] for t in va])
            y=np.array([gap(t,pair(t,ln)) for t in tr])
            vy=np.array([gap(t,pair(t,ln)) for t in va])
            flat=base.reshape(-1,129)
            mu,sd=(oldmu,oldsd) if norm=="old" else (flat.mean(0),flat.std(0)+1e-6)
            ax,ay,aw=[],[],[]
            bw=np.ones(473)
            for row,(t,variants) in enumerate(zip(tr,draws[sampler])):
                if variants:bw[row]=.95
                for b in variants:
                    assert len(b[0])>=nl and len(b[0])>=ln
                    ax.append(stock_inputs(z,t,(b[0][:nl],b[1]))[2])
                    ay.append(gap(t,(b[0][:ln],b[1])))
                    aw.append(.05/len(variants))
            for suffix,xx,yy,ww in (("plain",base,y,np.ones(473)),
                                    ("fixed5",np.concatenate([base,np.stack(ax)]),np.r_[y,ay],np.r_[bw,aw])):
                assert np.isclose(ww.sum(),473)
                arm=name+"_"+suffix
                def digest(*values):
                    h=hashlib.sha256()
                    for a in values:h.update(np.ascontiguousarray(a).tobytes())
                    return h.hexdigest()
                identity.append(dict(arm=arm,seed=seed,input_count=nl,label_count=ln,norm=norm,sampler=sampler,
                                     inputs=digest(xx,vx),labels=digest(yy,vy),weights=digest(ww),normalization=digest(mu,sd)))
                fixed_step=dict(plain={42:100,43:14,44:67},fixed5={42:100,43:14,44:81})[suffix][seed]
                selections.append(fit(xx,yy,ww,vx,vy,mu,sd,seed,nl,arm,fixed_step,traces))
    pd.DataFrame(traces).to_csv(OUT/"training_loss.csv",index=False)
    pd.DataFrame(identity).to_csv(OUT/"identity.csv",index=False)
    (OUT/"validation_lock.json").write_text(json.dumps(dict(selections=selections,configs=configs,
        train_dates=dates[tr].astype(str).tolist(),validation_dates=dates[va].astype(str).tolist(),hashes=HASHES.copy()),indent=2),encoding="utf8")
    print("Locked all validation choices",flush=True)
    dateset=pd.read_csv("outputs/bce_final_audit_v1/calendar_signals.csv",parse_dates=["date"]).set_index("date")
    ti=dates.get_indexer(dateset.index)
    predictions={}
    for seed in (42,43,44):
        with np.load(cache_path(seed)) as cache:z=cache["z"]
        for row in selections:
            if row["seed"]!=seed:continue
            cp=torch.load(row["path"],weights_only=True);nl=cp["nl"]
            xx=torch.from_numpy(np.stack([stock_inputs(z,t,(groups[t][0][:nl],groups[t][1][:10]))[2] for t in ti]))
            head=Pool(nl,10).eval()
            for state in ("state","fixed"):
                head.load_state_dict(cp[state])
                with torch.no_grad():pred=head((xx-cp["mu"])/cp["sd"])[:,0].numpy()
                arm=row["arm"]+("_fixedstep" if state=="fixed" else "")
                predictions.setdefault(arm,[]).append(pred)
    signals=pd.DataFrame({a:np.mean(p,axis=0) for a,p in predictions.items()},index=dateset.index)
    np.testing.assert_allclose(signals.A_plain,dateset.unaugmented,atol=1e-6,rtol=1e-6)
    np.testing.assert_array_equal(signals.A_fixed5.gt(0),dateset.fixed_5.gt(0))
    prior=pd.read_csv("outputs/bce_basket_sizes_v1/signals.csv",parse_dates=["date"]).set_index("date")
    for suffix in ("plain","fixed5"):
        np.testing.assert_allclose(signals["G_"+suffix],prior["t473_20x10_"+suffix],atol=1e-6,rtol=1e-6)
    signals.to_csv(OUT/"signals.csv")
    frame,schedule,_=locked_history()
    schedule=schedule.loc[schedule.date.isin(signals.index)]
    score=pd.concat([pd.DataFrame(dict(date=day,symbol=d["symbols"],score=d["score"][dates.get_loc(day)]))
                     for day in schedule.date],ignore_index=True)
    source=frame.loc[frame.date.isin(signals.index)].merge(score,on=["date","symbol"],validate="one_to_one")
    source["score_long"]=source["score_short"]=source.score
    symbols=pd.Index(sorted(source.symbol.unique())); ns=len(symbols)
    longs=np.zeros((295,ns));shorts=longs.copy();returns=longs.copy();days=[]
    for i,(day,g) in enumerate(source.groupby("date",sort=True)):
        assert day==signals.index[i]
        targets=pick_targets(g,20,10,1.,1.)
        f=g.set_index("symbol")
        for (side,s),w in targets.items():
            prefix="long" if side>0 else "short"
            if bool(f.at[s,prefix+"_entry"]):
                (longs if side>0 else shorts)[i,symbols.get_loc(s)]=w
        returns[i]=f.long_r.reindex(symbols).fillna(0).to_numpy()
        days.append(g.borrow_days.iloc[0])
    stats=runpy.run_path("outputs/bce_final_audit_v1/audit.py")["stats"]
    rows,books=[],[]
    for arm in signals:
        for rule in (("full",) if arm.endswith("_fixedstep") else ("wide","full")):
            share=np.where(signals[arm]>0,.7,.4) if rule=="wide" else signals[arm].gt(0).astype(float).to_numpy()
            for fee,borrow in ((10,.0835),(30,.12)):
                prev=np.zeros(ns);weeks=[]
                for i,date in enumerate(signals.index):
                    held=share[i]*longs[i]-(1-share[i])*shorts[i]
                    r=returns[i]
                    gross=float(held@r)
                    lc=float(np.abs(np.maximum(held,0)-np.maximum(prev,0)).sum()*fee/1e4)
                    sc=float(np.abs(np.minimum(held,0)-np.minimum(prev,0)).sum()*fee/1e4)
                    bc=float(-held[held<0].sum()*borrow*days[i]/365)
                    net=gross-lc-sc-bc
                    weeks.append(dict(date=date,net=net,gross_ret=gross,trading_cost=lc+sc,borrow_cost=bc,
                                      long_net=float(np.maximum(held,0)@r-lc),short_net=float(np.minimum(held,0)@r-sc-bc)))
                    prev=held*(1+r)/(1+net)
                w=pd.DataFrame(weeks);case=dict(arm=arm,rule=rule,fee=fee)
                rows.append(dict(**case,**stats(w,schedule)));books.append(w.assign(**case))
    metrics=pd.DataFrame(rows);weekly=pd.concat(books,ignore_index=True)
    ref=pd.read_parquet("outputs/bce_final_audit_v1/calendar_weekly.parquet").query("model=='old_scores' and long_slots==20")
    for suffix,baseline in (("plain","unaugmented"),("fixed5","fixed_5")):
        comp=weekly.query("arm==@('A_'+suffix)") if False else weekly.loc[weekly.arm.eq("A_"+suffix)]
        merged=comp.merge(ref.query("arm==@baseline"),on=["date","rule","fee"],validate="one_to_one")
        assert len(merged)==1180
        np.testing.assert_allclose(merged.net_x,merged.net_y,atol=1e-12,rtol=0)
    metrics.to_csv(OUT/"metrics.csv",index=False);weekly.to_parquet(OUT/"weekly.parquet",index=False)
    (OUT/"inputs.json").write_text(json.dumps(HASHES,indent=2),encoding="utf8")
    print(metrics.query("rule=='full' and not arm.str.contains('fixedstep')",engine="python")[["arm","fee","annual_calendar","max_drawdown","sharpe"]].to_string(index=False),flush=True)
    print("Completed factorial and fixed-stage ledgers; both original baselines reproduced",flush=True)


if __name__=="__main__":main()
