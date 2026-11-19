"""Paired fixed-P6 BCE heads with identical research baskets in all three phases."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml

from evaluate_dataset_three_year import fuse_probability_epochs, local_time
from replay_equal_weight_model_books import locked_history
from quant_mvp.financial_return import build_model


OUT = Path(__file__).resolve().parent
PROTOCOL = Path("docs/research_20261118_p6_matched_bce_protocol.md")
base = runpy.run_path(str(OUT.parent / "standard_bce_pool_v1/run_standard.py"))
std = base["fit"].__globals__
std["OUT"] = OUT
record, save = base["record"], base["save"]
pick = base["pick_targets"]
SEEDS = (42, 43, 44)
CUTOFF = "2018-01-01T00:00:00+08:00"
SETS = ("p6_clean", "p6_fixed_5")
original_replay = std["replay"]


def replay20(source, dates, spec, **kwargs):
    return original_replay(source, dates, (20, 10, *spec[2:]), **kwargs)


std["replay"] = replay20


def checkpoint(run, lock, epoch):
    path = run / f"epoch_{epoch}.pt"
    assert record(path) == lock["checkpoint_sha256"][str(epoch)]
    return torch.load(path, map_location="cpu", weights_only=True)


@torch.inference_mode()
def train_scores(run, cfg, data, ids, lock):
    files = std["HASHES"].copy()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    cache = OUT / "training_scores.npz"
    if cache.exists():
        with np.load(cache) as artifact:
            if artifact["digest"].item() == digest:
                print("Reusing hash-matched training scores and vectors", flush=True)
                return artifact["long"], artifact["short"], artifact["vectors"]
    guard = OUT.parent / "dataset_v2/gpu.lock"
    pending = False
    while True:
        try:
            with guard.open("x") as handle:
                json.dump(dict(pid=os.getpid(), task=OUT.name), handle)
            break
        except FileExistsError:
            if not pending:
                print("Waiting for the shared GPU", flush=True)
                pending = True
            time.sleep(5)
    try:
        print("Extracting train scores with the shared GPU", flush=True)
        # Legacy imported replay modules hide CUDA for their CPU tasks.
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        assert torch.cuda.is_available()
        model = build_model(cfg, data[8], data[9]).cuda().eval()
        scores, vectors = {}, None
        for epoch in sorted(set(lock["long_epochs"] + lock["short_epochs"])):
            model.load_state_dict(checkpoint(run, lock, epoch))
            parts, vec = [], []
            captured = []
            hook = model.final_norm.register_forward_hook(lambda module, args, out: captured.append(out)) if epoch == 8 else None
            for group in np.array_split(ids, (len(ids)+3)//4):
                captured.clear()
                output = model(*base["make_batch"](data, group.tolist(), cfg, "cuda"))
                parts.append(output.get("decision_score", output["mu_log"]).cpu().numpy())
                if epoch == 8:
                    assert len(captured) == 1
                    vec.append(F.normalize(captured[0], dim=-1).reshape(len(group), len(data[7]), 128).cpu().numpy())
            if hook is not None:
                hook.remove()
                vectors = np.concatenate(vec)
            scores[epoch] = np.concatenate(parts)
            np.testing.assert_array_equal(np.isfinite(scores[epoch]), data[4][ids])
            print(f"P6 epoch {epoch}: {len(ids)} training dates scored", flush=True)
        assert vectors is not None
        long, short = (np.mean([scores[e] for e in lock[key]], axis=0)
                       for key in ("long_epochs", "short_epochs"))
        del model, output, captured
        torch.cuda.empty_cache()
    finally:
        guard.unlink()
    temp = OUT / "training_scores.tmp.npz"
    np.savez(temp, digest=digest, long=long, short=short, vectors=vectors)
    temp.replace(cache)
    return long, short, vectors


def training_source(directory, train_dates, symbols, long, short):
    key = ["date", "symbol"]
    f = [("date", ">=", train_dates.min()), ("date", "<=", train_dates.max())]
    labels = pd.read_parquet(directory / "labels.parquet", filters=f).set_index(key)
    calendar = pd.read_parquet(directory / "sessions.parquet", filters=f).set_index("date").reindex(train_dates)
    cutoff = local_time(CUTOFF)
    decision, entry, exit_ = (local_time(calendar[col]) for col in ("decision_at", "entry_at", "exit_at"))
    mature = (decision < entry) & (entry < exit_) & (exit_.normalize()+pd.Timedelta(hours=18) < cutoff)
    borrow = pd.Series((exit_ - entry).total_seconds() / 86400, index=train_dates)
    available_at = local_time(labels.label_available_at)
    observed = labels.valid & (available_at >= local_time(labels.exit_at)) & (available_at < cutoff)
    grid = pd.MultiIndex.from_product((train_dates, symbols), names=key)
    amounts = pd.read_parquet(directory / "weekly.parquet", columns=[*key, "amount"], filters=f).set_index(key)
    source = pd.DataFrame(dict(score_long=long.ravel(), score_short=short.ravel(),
                               amount=amounts.amount.reindex(grid), long_r=np.expm1(labels.ret.where(observed)).reindex(grid)),
                          index=grid).reset_index()
    source["short_r"] = source.long_r
    source["borrow_days"] = source.date.map(borrow)
    return source.loc[source.date.isin(train_dates[mature])]


def basket_names(targets):
    return [np.array([s for side, s in targets if side == sign], dtype=object) for sign in (1, -1)]


def gap(frame, basket):
    if [len(n) for n in basket] != [20, 10]:
        return np.nan
    long, short = basket
    return float(frame.loc[long, "long_r"].mean(skipna=False) + frame.loc[short, "short_r"].mean(skipna=False)
                 + .0835 * frame.borrow_days.iloc[0] / 365)


def build_rows(source, vector, *, augment=False):
    primary, gaps, dates, baskets, draws = [], [], [], [], []
    for date, g in source.groupby("date", sort=True):
        g = g.assign(short_ok=True)
        basket = basket_names(pick(g, 20, 10, .5, .5))
        frame = g.set_index("symbol")
        y = gap(frame, basket)
        if augment and not np.isfinite(y):
            continue
        dates.append(date)
        baskets.append(basket)
        primary.append(vector(date, basket))
        gaps.append(y)
        if augment:
            pools = (basket_names(pick(g, 50, 0, .5, 0.))[0],
                     basket_names(pick(g, 20, 20, .5, .5))[1])
            draws.append((frame, pools))
    x, y = np.stack(primary), np.asarray(gaps)
    if not augment:
        return x, y, pd.DatetimeIndex(dates)
    active = np.random.default_rng(61118).choice(len(x), len(x)//2, replace=False)
    ax, ay, aw = [], [], []
    bw = np.ones(len(x))
    for row in sorted(active):
        frame, (long, short) = draws[row]
        seen = {tuple(map(tuple, baskets[row]))}
        variants = []
        if len(long) >= 20 and len(short) >= 10:
            rng = np.random.default_rng(np.random.SeedSequence([61118, int(dates[row].toordinal()), 1]))
            for _ in range(3):
                alt_long = long[np.sort(rng.choice(len(long), 20, replace=False))]
                short_pool = short[~np.isin(short, alt_long)]
                if len(short_pool) < 10:
                    continue
                alt = (alt_long, short_pool[np.sort(rng.choice(len(short_pool), 10, replace=False))])
                key = tuple(map(tuple, alt))
                label = gap(frame, alt)
                if key not in seen and np.isfinite(label):
                    seen.add(key)
                    variants.append((alt,label))
        if variants:
            bw[row] = .95
            for alt, label in variants:
                ax.append(vector(dates[row], alt))
                ay.append(label)
                aw.append(.05/len(variants))
    assert ax and np.isclose(bw.sum() + sum(aw), len(x))
    print(f"Common training: {len(x)} weeks, {len(active)} drawn weeks, {(bw<1).sum()} effective, {len(ax)} variants", flush=True)
    return x, y, pd.DatetimeIndex(dates), (np.concatenate([x,np.stack(ax)]),
                                           np.r_[y,ay], np.r_[bw,aw]), (len(active), int((bw<1).sum()),len(ax))


def xrow(vectors, ids):
    return base["xrow"](vectors, (np.repeat(ids[0], 2), ids[1]))


@torch.inference_mode()
def eval_rows(directory, run, cfg, lock, source):
    opts, data = base["dataset"](directory, run, cfg)
    model = build_model(opts, data[8], data[9]).eval()
    model.load_state_dict(checkpoint(run, lock, 8))
    dates, symbols = pd.DatetimeIndex(data[6]), pd.Index(data[7])

    def vector(date, basket):
        ids = tuple(symbols.get_indexer(n) for n in basket)
        idx = np.concatenate(ids)
        t = dates.get_loc(date)
        assert (idx >= 0).all() and data[4][t, idx].all() and t >= cfg["lookback"]-1
        window = data[0][t-cfg["lookback"]+1:t+1,idx].transpose(1,0,2)
        z = F.normalize(model.final_norm(model.encoder(torch.from_numpy(window), normalize=False)), dim=-1)
        return xrow(z.numpy(), (np.arange(len(ids[0])), np.arange(len(ids[0]),len(idx))))

    return build_rows(source, vector)


def fit(arm, seed, batch, normalizer, vx, vy, nweeks, trace):
    x, y, w = batch
    mu, sd = normalizer
    tx = torch.from_numpy((x-mu)/sd)
    target = torch.tensor(y>0, dtype=torch.float32)
    weight = torch.tensor(w, dtype=torch.float32)
    use = np.isfinite(vy)
    vv = torch.from_numpy((vx[use]-mu)/sd)
    vtarget = torch.tensor(vy[use]>0, dtype=torch.float32)
    torch.manual_seed(seed)
    model = base["LegPool"](False)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.01)
    best = np.inf
    for step in range(1,101):
        if step == 11:
            model.proj.requires_grad_(False)
            frozen = [p.detach().clone() for p in model.proj.parameters()]
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(model(tx)[:,0], target, reduction="none") * weight).sum()/nweeks
        loss.backward()
        opt.step()
        with torch.no_grad():
            val = F.binary_cross_entropy_with_logits(model(vv)[:,0], vtarget).item()
        trace.append(dict(arm=arm,seed=seed,step=step,train_bce=float(loss),val_bce=val))
        if val < best:
            best, chosen = val, step
            state = {k:p.detach().clone() for k,p in model.state_dict().items()}
    assert all(torch.equal(a,b) for a,b in zip(frozen,model.proj.parameters()))
    cp = OUT / f"{arm}_{seed}.pt"
    torch.save(dict(state=state, mu=torch.from_numpy(mu), sd=torch.from_numpy(sd)), cp)
    print(arm, seed, "step", chosen, "validation BCE", best, flush=True)
    return dict(arm=arm,seed=seed,step=chosen,val_bce=best,checkpoint=str(cp),sha256=record(cp))


@torch.inference_mode()
def signals(x, selections):
    out = {arm: [] for arm in SETS}
    for row in selections:
        assert record(row["checkpoint"]) == row["sha256"]
        cp = torch.load(row["checkpoint"], weights_only=True)
        model = base["LegPool"](False).eval()
        model.load_state_dict(cp["state"])
        out[row["arm"]].append(model((torch.from_numpy(x)-cp["mu"])/cp["sd"])[:,0].numpy())
    return {arm: np.mean(parts,axis=0) for arm,parts in out.items()}


def replay_cases(test, common, predictions):
    rows, books = [], []
    for name, source in (("uniform", test.assign(short_entry=test.short_entry & test.short_ok, short_ok=True)),
                         ("legacy_book", test)):
        part_stats, part_books = [], []
        base["backtests"](source, common, {k:[v] for k,v in predictions.items()}, "test", part_stats, part_books)
        rows.append(pd.DataFrame(part_stats).assign(book=name))
        books.append(pd.concat(part_books, ignore_index=True).assign(book=name))
    return pd.concat(rows, ignore_index=True), pd.concat(books, ignore_index=True)


def main():
    torch.set_num_threads(4)
    valdir, histdir = (base["STANDARD"]/"snapshots"/p for p in ("validation","history"))
    tape = base["TAPE"]
    cat = base["CATALOG"]
    member = next(m for m in json.loads(cat.read_text(encoding="utf-8"))["members"] if m["arm"]=="P6")
    run = Path(member["run"])
    for path in (__file__, PROTOCOL, cat, member["lock"], member["validation_scores"],
                 run/"config.yaml", run/"training_protocol.json", run/"preprocessing.json",
                 tape, tape.with_name("eligibility.parquet"), std["__file__"],
                 Path("scripts/attention_followup.py"), Path("scripts/eval_weekly_architecture.py"),
                 Path("src/quant_mvp/financial_return.py"), Path("src/quant_mvp/hierarchy_data.py")):
        record(path)
    for name in ("manifest.json","weekly.parquet","market_features.csv","universe.csv","labels.parquet","sessions.parquet"):
        record(valdir/name)
    cfg = yaml.safe_load((run/"config.yaml").read_text(encoding="utf-8"))
    lock = json.loads(Path(member["lock"]).read_text(encoding="utf-8"))
    assert lock["long_epochs"] == [6,7,8] and lock["short_epochs"] == [3,4,5]
    for e in range(3,9):
        assert record(run/f"epoch_{e}.pt") == lock["checkpoint_sha256"][str(e)]
    opts, data = base["dataset"](valdir,run,cfg)
    dates, symbols = pd.DatetimeIndex(data[6]), pd.Index(data[7])
    train = pd.DatetimeIndex(pd.to_datetime(json.loads((run/"training_protocol.json").read_text(encoding="utf-8"))["training_dates"]))
    ids = dates.get_indexer(train)
    assert len(ids) == 779 and (ids >= cfg["lookback"]-1).all() and train.max() < pd.Timestamp("2018-01-01")
    long, short, z = train_scores(run,opts,data,ids,lock)
    record(OUT/"training_scores.npz")
    source = training_source(valdir,train,symbols,long,short)

    def vector(date, basket):
        i = train.get_loc(date)
        positions = [symbols.get_indexer(n) for n in basket]
        assert all((idx >= 0).all() and data[4][ids[i],idx].all() for idx in positions)
        return xrow(z[i],positions)

    tx, ty, train_dates, aug, counts = build_rows(source,vector,augment=True)
    del source, z, data
    valsrc, vs, _ = base["phase_proxy"](valdir,base["snapshot"](valdir),tape,tape.with_name("eligibility.parquet"),phase="validation")
    raw = pd.read_parquet(member["validation_scores"])
    raw.date = pd.to_datetime(raw.date)
    valsrc = base["score_pair"](valsrc, *(fuse_probability_epochs(raw,lock[k],run.name)
                                         for k in ("long_epochs", "short_epochs")), vs)
    vx, vy, vd = eval_rows(valdir,run,cfg,lock,valsrc)
    pd.testing.assert_index_equal(vd, pd.DatetimeIndex(vs.date).rename(None))
    assert len(vd) == 153 and np.isfinite(vy).sum() > 100 and len(tx) > 100
    flat = tx.reshape(-1,129)
    norm = flat.mean(0), flat.std(0) + 1e-6
    trace, selections = [], []
    for arm, batch in zip(SETS,((tx,ty,np.ones(len(ty))),aug)):
        for seed in SEEDS:
            selections.append(fit(arm,seed,batch,norm,vx,vy,len(tx),trace))
    pd.DataFrame(trace).to_csv(OUT/"training_loss.csv",index=False)
    save("validation_lock.json",dict(selections=selections,train_dates=train_dates.strftime("%Y-%m-%d").tolist(),
                                     validation_dates=vd[np.isfinite(vy)].strftime("%Y-%m-%d").tolist(),
                                     aug_counts=counts,hashes=std["HASHES"].copy()))
    print(f"Validation locked on {np.isfinite(vy).sum()}/153 label weeks",flush=True)

    oldbook = OUT.parent/"old_longshort_bce_loss_v1/report/signals.parquet"
    reference = OUT.parent/"b0_p6_frozen_bce_transfer_v1/report/weekly.parquet"
    for path in (member["test_scores"],histdir/"manifest.json",oldbook,reference):
        record(path)
    old = pd.read_parquet(oldbook,filters=[("split","=","test")]).set_index("date").freeze_projection
    frame, schedule, members = locked_history()
    scores = next(s for l,s in members.values() if l["arm"]=="P6")
    test = base["score_pair"](frame,scores["test"]["long"],scores["test"]["short"],schedule)
    common = schedule.loc[schedule.date.isin(old.index)]
    test = test.loc[test.date.isin(common.date)]
    hx, hy, hd = eval_rows(histdir,run,cfg,lock,test)
    pd.testing.assert_index_equal(hd, old.index.rename(None))
    assert len(hd) == 283
    preds = signals(hx,selections) | {"legacy_signal":old.reindex(hd).to_numpy()}
    assert np.isfinite(np.stack(list(preds.values()))).all()
    prediction = pd.DataFrame(preds,index=hd).rename_axis("date")
    prediction.assign(gap=hy).to_csv(OUT/"signals.csv")
    metrics, weekly = replay_cases(test,common,preds)
    key = ["book","date","rule","fee"]
    assert len(metrics) == 24 and len(weekly) == 6792 and not weekly.duplicated([*key,"arm"]).any()
    np.testing.assert_allclose(weekly.long_net+weekly.short_net,weekly.net,atol=1e-12,rtol=0)
    ref = pd.read_parquet(reference).query("model=='P6' and arm in ['wide','full_switch']")
    ref = ref.rename(columns={"arm":"rule","trade_bps":"fee"}).replace({"full_switch":"full"})
    comparison = weekly.query("book=='legacy_book' and arm=='legacy_signal'").merge(
        ref,on=["date","rule","fee"],validate="one_to_one")
    assert len(comparison) == 1132
    np.testing.assert_allclose(comparison.net_x,comparison.net_y,atol=1e-12,rtol=0)
    q = weekly.pivot(index=key,columns="arm",values="net")
    pairs = [dict(book=book,rule=rule,fee=fee,arm=arm,control=control,
                  **base["paired"](g[arm].to_numpy(),g[control].to_numpy()))
             for (book,rule,fee),g in q.groupby(["book","rule","fee"])
             for arm,control in (("p6_fixed_5","p6_clean"),("p6_clean","legacy_signal"),("p6_fixed_5","legacy_signal"))]
    metrics.to_csv(OUT/"metrics.csv",index=False)
    weekly.to_parquet(OUT/"weekly.parquet",index=False)
    pd.DataFrame(pairs).to_csv(OUT/"paired.csv",index=False)
    save("inputs.json",std["HASHES"])
    print(metrics[["book","arm","rule","fee","annual_calendar","max_drawdown"]].to_string(index=False),flush=True)
    print("Verified both books and reproduced 1132 archived P6 baseline returns",flush=True)


if __name__ == "__main__":
    main()
