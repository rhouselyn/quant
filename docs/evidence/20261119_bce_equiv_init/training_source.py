"""Paired normalization training with identical initial functions."""
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
PARENT=Path("outputs/bce_input_label_v1")
Pool=runpy.run_path("outputs/bce_basket_sizes_v1/run.py")["BasketPool"]
stats=runpy.run_path("outputs/bce_final_audit_v1/audit.py")["stats"]
HASHES={}

def record(path):
    path=Path(path).resolve()
    with path.open("rb") as stream:HASHES[str(path)]=hashlib.file_digest(stream,"sha256").hexdigest()

def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def fit_pair(x,y,w,vx,vy,oldmu,oldsd,ownmu,ownsd,seed,suffix,traces,divergence,checks):
    torch.manual_seed(seed)
    original=Pool(20,10)
    equivalent=copy.deepcopy(original)
    with torch.no_grad():
        weight=original.proj.weight.detach().clone()
        equivalent.proj.weight.copy_(weight*torch.from_numpy(ownsd/oldsd))
        equivalent.proj.bias.copy_(original.proj.bias+weight@torch.from_numpy((ownmu-oldmu)/oldsd))
    models={"old":original,"equiv":equivalent}
    norms={"old":(oldmu,oldsd),"equiv":(ownmu,ownsd)}
    tx={k:torch.from_numpy((x-mu)/sd) for k,(mu,sd) in norms.items()}
    vv={k:torch.from_numpy((vx-mu)/sd) for k,(mu,sd) in norms.items()}
    target=torch.tensor(y>0,dtype=torch.float32);vt=torch.tensor(vy>0,dtype=torch.float32)
    wt=torch.tensor(w,dtype=torch.float32)
    with torch.no_grad():
        errors={}
        for split,xx in (("train",tx),("validation",vv)):
            a,b=original(xx["old"]),equivalent(xx["equiv"])
            torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
            pa,pb=original.proj(xx["old"]),equivalent.proj(xx["equiv"])
            torch.testing.assert_close(pa,pb,atol=1e-5,rtol=1e-5)
            errors[split+"_max_logit_error"]=float((a-b).abs().max())
            errors[split+"_max_projection_error"]=float((pa-pb).abs().max())
        for k in original.state_dict():
            if not k.startswith("proj."):assert torch.equal(original.state_dict()[k],equivalent.state_dict()[k])
        checks.append(dict(seed=seed,suffix=suffix,**errors))
    opts={k:torch.optim.Adam(h.parameters(),lr=.001,weight_decay=.01) for k,h in models.items()}
    best={k:np.inf for k in models};selected={};fixed={};states={}
    fixed_step={42:100,43:16,44:100}[seed]
    for step in range(1,101):
        if step==11:
            frozen={}
            for k,h in models.items():
                h.proj.requires_grad_(False)
                frozen[k]=[p.detach().clone() for p in h.proj.parameters()]
        valpred={}
        for k,h in models.items():
            opts[k].zero_grad()
            loss=(F.binary_cross_entropy_with_logits(h(tx[k])[:,0],target,reduction="none")*wt).sum()/473
            loss.backward();opts[k].step()
            with torch.no_grad():
                valpred[k]=h(vv[k])[:,0]
                value=F.binary_cross_entropy_with_logits(valpred[k],vt).item()
            traces.append(dict(arm=k+"_"+suffix,seed=seed,step=step,train_bce=float(loss),val_bce=value))
            if value<best[k]:
                best[k]=value;selected[k]=step;states[k]=copy.deepcopy(h.state_dict())
            if step==fixed_step:fixed[k]=copy.deepcopy(h.state_dict())
        with torch.no_grad():
            pdiff=(original.proj(vv["old"])-equivalent.proj(vv["equiv"])).abs()
            ldiff=(valpred["old"]-valpred["equiv"]).abs()
        divergence.append(dict(seed=seed,suffix=suffix,step=step,projection_max=float(pdiff.max()),
                               logit_max=float(ldiff.max()),logit_mean=float(ldiff.mean()),
                               different_validation_directions=int((valpred["old"].gt(0)!=valpred["equiv"].gt(0)).sum())))
    rows=[]
    for k,h in models.items():
        assert all(torch.equal(a,b) for a,b in zip(frozen[k],h.proj.parameters()))
        arm=k+"_"+suffix;mu,sd=norms[k]
        path=OUT/f"{arm}_{seed}.pt"
        torch.save(dict(state=states[k],fixed=fixed[k],mu=torch.from_numpy(mu),sd=torch.from_numpy(sd),nl=20),path)
        record(path)
        rows.append(dict(arm=arm,seed=seed,step=selected[k],val_bce=best[k],path=str(path),fixed_step=fixed_step))
        print(arm,seed,"selected",selected[k],"val_bce",best[k],flush=True)
    return rows

def replay(source,schedule,signals):
    symbols=pd.Index(sorted(source.symbol.unique()));n=len(schedule)
    rows,books=[],[]
    for nl in (20,40):
        longs=np.zeros((n,len(symbols)));shorts=longs.copy();lr=longs.copy();sr=longs.copy();days=[]
        for i,(day,g) in enumerate(source.groupby("date",sort=True)):
            assert day==signals.index[i]
            targets=pick_targets(g,nl,10,1.,1.)
            f=g.set_index("symbol")
            for (side,s),weight in targets.items():
                prefix="long" if side>0 else "short"
                if bool(f.at[s,prefix+"_entry"]):(longs if side>0 else shorts)[i,symbols.get_loc(s)]=weight
            lr[i]=f.long_r.reindex(symbols).fillna(0).to_numpy()
            sr[i]=f.short_r.reindex(symbols).fillna(0).to_numpy()
            days.append(g.borrow_days.iloc[0])
        for arm in signals:
            if nl==40 and not arm.startswith("base40_"):continue
            for rule in (("full",) if arm.endswith("_fixedstep") else ("wide","full")):
                share=np.where(signals[arm]>0,.7,.4) if rule=="wide" else signals[arm].gt(0).astype(float).to_numpy()
                for fee,borrow in ((10,.0835),(30,.12)):
                    prev=np.zeros(len(symbols));weeks=[]
                    for i,date in enumerate(signals.index):
                        held=share[i]*longs[i]-(1-share[i])*shorts[i]
                        r=np.where(held<0,sr[i],lr[i])
                        gross=float(held@r)
                        lt=float(np.abs(np.maximum(held,0)-np.maximum(prev,0)).sum())
                        st=float(np.abs(np.minimum(held,0)-np.minimum(prev,0)).sum())
                        lc,sc=lt*fee/1e4,st*fee/1e4
                        bc=float(-held[held<0].sum()*borrow*days[i]/365)
                        net=gross-lc-sc-bc
                        assert np.isfinite(net) and 1+net>0 and np.abs(held).sum()<=1+1e-12
                        weeks.append(dict(date=date,net=net,gross_ret=gross,trading_cost=lc+sc,borrow_cost=bc,turnover=lt+st,
                                          long_net=float(np.maximum(held,0)@r-lc),short_net=float(np.minimum(held,0)@r-sc-bc)))
                        prev=held*(1+r)/(1+net)
                    w=pd.DataFrame(weeks);case=dict(arm=arm,rule=rule,fee=fee,long_slots=nl)
                    rows.append(dict(**case,**stats(w,schedule),turnover=float(w.turnover.mean()),
                                     trading_bp=float(w.trading_cost.mean()*1e4),borrow_bp=float(w.borrow_cost.mean()*1e4)))
                    books.append(w.assign(**case))
    return pd.DataFrame(rows),pd.concat(books,ignore_index=True)

def main():
    torch.set_num_threads(4)
    (OUT/"state.json").write_text(json.dumps(dict(status="training")),encoding="utf8")
    for p in (__file__,"docs/research_20261119_bce_equiv_init_protocol.md","outputs/bce_basket_sizes_v1/run.py",
              "scripts/attention_followup.py","scripts/replay_equal_weight_model_books.py",
              "scripts/directional_three_seed.py","scripts/directional_2018_experiment.py",
              "scripts/eval_weekly_architecture.py","outputs/bce_final_audit_v1/audit.py",
              PARENT/"identity.csv",PARENT/"validation_lock.json",PARENT/"signals.csv"):record(p)
    d,dates,_,_,complete,tr=legacy_inputs()
    sig=pd.read_parquet("outputs/old_longshort_bce_loss_v1/report/signals.parquet")
    record("outputs/old_longshort_bce_loss_v1/report/signals.parquet")
    vi=dates.get_indexer(sig.query("split=='val'").date);va=vi[complete[vi]]
    assert len(tr)==473 and len(va)==123
    priorlock=json.loads((PARENT/"validation_lock.json").read_text())
    assert dates[tr].astype(str).tolist()==priorlock["train_dates"]
    assert dates[va].astype(str).tolist()==priorlock["validation_dates"]
    groups=groups_for(d,dates)
    active=set(np.random.default_rng(61118).choice(473,236,replace=False))
    draws=[]
    for row,t in enumerate(tr):
        variants=[]
        if row in active:
            seen={tuple(map(tuple,(groups[t][0][:40],groups[t][1][:10])))}
            rng=np.random.default_rng(np.random.SeedSequence([61118,int(t),1]))
            for _ in range(3):
                pair=tuple(pool[np.sort(rng.choice(len(pool),n,replace=False))]
                           for pool,n in ((groups[t][0][:50],40),(groups[t][1][:20],10)))
                key=tuple(map(tuple,pair))
                if key not in seen and d["valid"][t,np.r_[*pair]].all():seen.add(key);variants.append(pair)
        draws.append(variants)
    assert sum(bool(v) for v in draws)==220 and sum(map(len,draws))==634
    def pair(t):return groups[t][0][:20],groups[t][1][:10]
    def gap(t,b):
        assert d["valid"][t,np.r_[*b]].all()
        return np.float32(np.expm1(d["y"][t,b[0]]).mean()+np.expm1(d["y"][t,b[1]]).mean()+.08/52)
    traces,divergence,checks,selections,identity=[],[],[],[],[]
    pi=pd.read_csv(PARENT/"identity.csv").set_index(["arm","seed"])
    for seed in (42,43,44):
        record(cache_path(seed))
        with np.load(cache_path(seed)) as cache:z=cache["z"]
        base=np.stack([stock_inputs(z,t,pair(t))[2] for t in tr])
        vx=np.stack([stock_inputs(z,t,pair(t))[2] for t in va])
        y=np.array([gap(t,pair(t)) for t in tr]);vy=np.array([gap(t,pair(t)) for t in va])
        for name in ("D_plain","E_plain"):record(PARENT/f"{name}_{seed}.pt")
        old=torch.load(PARENT/f"D_plain_{seed}.pt",weights_only=True)
        own=torch.load(PARENT/f"E_plain_{seed}.pt",weights_only=True)
        oldmu,oldsd=old["mu"].numpy(),old["sd"].numpy()
        flat=base.reshape(-1,129);ownmu,ownsd=flat.mean(0),flat.std(0)+1e-6
        np.testing.assert_array_equal(ownmu,own["mu"].numpy());np.testing.assert_array_equal(ownsd,own["sd"].numpy())
        ax,ay,aw=[],[],[];bw=np.ones(473)
        for row,(t,variants) in enumerate(zip(tr,draws)):
            if variants:bw[row]=.95
            for b in variants:
                basket=(b[0][:20],b[1])
                ax.append(stock_inputs(z,t,basket)[2]);ay.append(gap(t,basket));aw.append(.05/len(variants))
        for suffix,xx,yy,ww in (("plain",base,y,np.ones(473)),("fixed5",np.concatenate([base,np.stack(ax)]),np.r_[y,ay],np.r_[bw,aw])):
            assert np.isclose(ww.sum(),473)
            fingerprints=dict(inputs=digest(xx,vx),labels=digest(yy,vy),weights=digest(ww))
            for key,value in fingerprints.items():assert value==pi.at[("D_"+suffix,seed),key],(seed,suffix,key)
            identity.append(dict(seed=seed,suffix=suffix,**fingerprints,old_normalization=digest(oldmu,oldsd),own_normalization=digest(ownmu,ownsd)))
            selections+=fit_pair(xx,yy,ww,vx,vy,oldmu,oldsd,ownmu,ownsd,seed,suffix,traces,divergence,checks)
    pd.DataFrame(traces).to_csv(OUT/"training_loss.csv",index=False)
    pd.DataFrame(divergence).to_csv(OUT/"divergence.csv",index=False)
    pd.DataFrame(checks).to_csv(OUT/"initial_equivalence.csv",index=False)
    pd.DataFrame(identity).to_csv(OUT/"identity.csv",index=False)
    (OUT/"validation_lock.json").write_text(json.dumps(dict(selections=selections,train_dates=dates[tr].astype(str).tolist(),
        validation_dates=dates[va].astype(str).tolist(),hashes=HASHES.copy()),indent=2),encoding="utf8")
    print("Locked all 12 validation checkpoints before test inference",flush=True)
    (OUT/"state.json").write_text(json.dumps(dict(status="locked_scoring")),encoding="utf8")
    record(PARENT/"weekly.parquet");record("outputs/bce_final_audit_v1/calendar_weekly.parquet")
    previous=pd.read_csv(PARENT/"signals.csv",parse_dates=["date"]).set_index("date")
    assert len(previous)==295
    ti=dates.get_indexer(previous.index);assert (ti>=0).all()
    predictions={}
    for seed in (42,43,44):
        with np.load(cache_path(seed)) as cache:z=cache["z"]
        xx=torch.from_numpy(np.stack([stock_inputs(z,t,pair(t))[2] for t in ti]))
        for row in selections:
            if row["seed"]!=seed:continue
            cp=torch.load(row["path"],weights_only=True);head=Pool(20,10).eval()
            for state in ("state","fixed"):
                head.load_state_dict(cp[state])
                with torch.no_grad():p=head((xx-cp["mu"])/cp["sd"])[:,0].numpy()
                predictions.setdefault(row["arm"]+("_fixedstep" if state=="fixed" else ""),[]).append(p)
    signals=pd.DataFrame({k:np.mean(v,axis=0) for k,v in predictions.items()},index=previous.index)
    for suffix in ("plain","fixed5"):
        np.testing.assert_allclose(signals["old_"+suffix],previous["D_"+suffix],atol=1e-6,rtol=1e-6)
        signals["base40_"+suffix]=previous["A_"+suffix]
        signals["naive_"+suffix]=previous["E_"+suffix]
    signals.to_csv(OUT/"signals.csv")
    frame,schedule,_=locked_history()
    schedule=schedule.loc[schedule.date.isin(signals.index)]
    assert len(schedule)==295
    score=pd.concat([pd.DataFrame(dict(date=day,symbol=d["symbols"],score=d["score"][dates.get_loc(day)])) for day in schedule.date],ignore_index=True)
    source=frame.loc[frame.date.isin(signals.index)].merge(score,on=["date","symbol"],validate="one_to_one")
    source["score_long"]=source["score_short"]=source.score
    metrics,weekly=replay(source,schedule,signals)
    ref=pd.read_parquet(PARENT/"weekly.parquet")
    checked=0
    for a,b in (("old","D"),("base40","A"),("naive","E")):
        for suffix in ("plain","fixed5"):
            x=weekly.query("arm==@a+'_'+@suffix and long_slots==20")
            y=ref.query("arm==@b+'_'+@suffix")
            joined=x.merge(y,on=["date","rule","fee"],validate="one_to_one")
            assert len(joined)==1180
            np.testing.assert_allclose(joined.net_x,joined.net_y,atol=1e-12,rtol=0);checked+=len(joined)
    archived=pd.read_parquet("outputs/bce_final_audit_v1/calendar_weekly.parquet").query("model=='old_scores' and long_slots==40")
    for suffix,oldarm in (("plain","unaugmented"),("fixed5","fixed_5")):
        x=weekly.query("arm==@('base40_'+suffix) and long_slots==40") if False else weekly.loc[weekly.arm.eq("base40_"+suffix)&weekly.long_slots.eq(40)]
        y=archived.query("arm==@oldarm")
        joined=x.merge(y,on=["date","rule","fee"],validate="one_to_one")
        assert len(joined)==1180
        np.testing.assert_allclose(joined.net_x,joined.net_y,atol=1e-12,rtol=0);checked+=len(joined)
    assert len(metrics)==48 and len(weekly)==48*295
    np.testing.assert_allclose(weekly.long_net+weekly.short_net,weekly.net,atol=1e-12,rtol=0)
    metrics.to_csv(OUT/"metrics.csv",index=False);weekly.to_parquet(OUT/"weekly.parquet",index=False)
    (OUT/"inputs.json").write_text(json.dumps(HASHES,indent=2),encoding="utf8")
    verification=dict(status="complete",heads_trained=12,train_weeks=473,validation_weeks=123,test_weeks=295,
        matched_parent_training_fingerprints=True,initial_function_equivalence=True,projection_frozen_after_step10=True,
        validation_locked_before_test=True,old20_signal_reproduced=True,archived_weekly_rows_reproduced=checked,
        metric_rows=len(metrics),weekly_rows=len(weekly))
    (OUT/"verification.json").write_text(json.dumps(verification,indent=2),encoding="utf8")
    (OUT/"state.json").write_text(json.dumps(verification,indent=2),encoding="utf8")
    print(metrics.query("rule=='full'")[["arm","long_slots","fee","annual_calendar","max_drawdown","sharpe"]].to_string(index=False),flush=True)
    print("Completed paired equivalent initialization and old40 execution verification",flush=True)

if __name__=="__main__":main()
