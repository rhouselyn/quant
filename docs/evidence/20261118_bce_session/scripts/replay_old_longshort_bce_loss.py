"""Fit new losses on the original fixed-basket long/short allocation task."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits

from attention_context_ablation import OUT as CONTROL, fit_portfolio
from attention_followup import replay
from directional_2018_experiment import groups_for
from directional_three_seed import SEEDS, cache_path, standard_scores, stock_inputs
from eval_dataset_experiments import metrics, publish
from replay_kit_risk import paired, tail_mean


ARMS = ("freeze_projection", "freeze_weighted", "freeze_magnitude")


def legacy_inputs():
    stock_scores = []
    for seed in SEEDS:
        with np.load(cache_path(seed)) as cache:
            if seed==42:
                data = {key:cache[key] for key in cache.files if key!="z"}
            stock_scores.append(standard_scores(cache["score"],data["available"]))
    data["score"] = np.mean(stock_scores,axis=0)
    dates = pd.to_datetime(data["dates"])
    baskets = [(long[:40],short[:10]) for long,short in groups_for(data,dates)]
    complete = np.zeros(len(dates),dtype=bool)
    gaps = np.full(len(dates),np.nan,dtype=np.float32)
    for t,(long,short) in enumerate(baskets):
        if len(long)!=40 or len(short)!=10:
            continue
        ids = np.r_[long,short]
        assert len(np.unique(ids))==50
        valid = data["valid"][t,ids] & np.isfinite(data["y"][t,ids])
        complete[t] = valid.all()
        r = np.expm1(np.where(valid,data["y"][t,ids],0))
        gaps[t] = r[:40].mean()+r[40:].mean()+.08/52
    train = np.flatnonzero(complete & (np.arange(len(dates))<dates.searchsorted("2018-01-05")-2))
    return data,dates,baskets,gaps,complete,train


def main(out, checkpoints):
    reference = pd.read_csv(CONTROL/"projection_freeze_sweep/weekly.csv", parse_dates=["date"],
                            float_precision="round_trip").query("arm=='freeze_10'").reset_index(drop=True)
    data,dates,baskets,gaps,complete,train = legacy_inputs()
    evaluation = dates.get_indexer(reference.date)
    assert (evaluation>=0).all() and reference.date.is_unique
    val = evaluation[reference.split.eq("val").to_numpy() & complete[evaluation]]
    assert (len(train),len(val),len(evaluation))==(473,123,416)
    estimates = {arm:[] for arm in ARMS}
    diagnostics, traces = [],[]
    for seed in SEEDS:
        with np.load(cache_path(seed)) as cache:
            encoded = cache["z"]
        x = np.zeros((len(dates),50,129),dtype=np.float32)
        for t in np.r_[train,evaluation]:
            x[t] = stock_inputs(encoded,t,baskets[t])[2]
        del encoded
        pred, chosen, trace = fit_portfolio(x,gaps,train,val,evaluation,seed,arms=ARMS,out=checkpoints)
        diagnostics.extend(chosen)
        traces.extend(trace)
        for arm in ARMS:
            output = pred[arm]
            score = output[:,0]
            if arm=="freeze_magnitude":
                up,down = np.logaddexp(0.,output[:,1:]).clip(1e-5).T
                p = expit(score)
                score = (p*up-(1-p)*down)*gaps[train].std()
            assert np.isfinite(score).all()
            estimates[arm].append(score)

    signals = pd.DataFrame({arm:np.mean(ensemble,axis=0) for arm,ensemble in estimates.items()})
    np.testing.assert_allclose(signals.freeze_projection,reference.score,atol=1e-6,rtol=1e-6)
    old = pd.read_csv(CONTROL/"projection_freeze_sweep/learning_curves.csv").query("arm=='freeze_10'")
    old = old.sort_values(["seed","step"])
    trace = pd.DataFrame(traces)
    ref_trace = trace.query("arm=='freeze_projection'").sort_values(["seed","step"])
    np.testing.assert_allclose(ref_trace[["train_loss","val_loss"]],old[["train_loss","val_loss"]],atol=1e-6,rtol=1e-6)
    step_reference = json.loads((CONTROL/"projection_freeze_sweep/report.json").read_text())["checkpoints"]
    original_steps = {v["seed"]:v["step"] for v in step_reference if v["arm"]=="freeze_10"}
    assert {v["seed"]:v["step"] for v in diagnostics if v["arm"]=="freeze_projection"}==original_steps
    signals = signals.assign(date=reference.date,split=reference.split,gap=gaps[evaluation],label_valid=complete[evaluation])
    print("Original freeze10 forecasts, checkpoint steps and losses reproduced",flush=True)
    all_books, table, comparisons, quality = [],[],[],[]
    for split in ("val","test"):
        use = signals.split.eq(split)
        indices = evaluation[use]
        block = signals.loc[use]
        covers = block.label_valid
        weights = {"static50":np.full(len(block),.5), "static55":np.full(len(block),.55)}
        for arm in ARMS:
            positive = block[arm].gt(0)
            correct = positive[covers].eq(block.loc[covers,"gap"].gt(0))
            quality.append(dict(split=split,arm=arm,weeks=int(covers.sum()),positive_rate=positive.mean(),
                                accuracy=correct.mean(),weighted_accuracy=np.average(correct,weights=block.loc[covers,"gap"].abs())))
            for band,low,high in (("narrow",.45,.55),("wide",.4,.7)):
                share = np.where(positive,high,low)
                weights[f"{arm}_{band}"] = share
                weights[f"{arm}_{band}_mean"] = np.full(len(block),share.mean())
        for fee,borrow in ((10,.08),(30,.12)):
            books = {}
            for arm, shares in weights.items():
                book = replay(data,dates,baskets,indices,shares,arm,cost_bps=fee,borrow_ann=borrow,attribution=True)
                np.testing.assert_allclose(book.long_net+book.short_net,book.net,atol=1e-12,rtol=0)
                if fee==10 and arm=="freeze_projection_wide":
                    np.testing.assert_allclose(book.net,reference.loc[use,"net"],atol=1e-12,rtol=0)
                books[arm] = book
                case = dict(split=split,trade_bps=fee,arm=arm)
                all_books.append(book.assign(**case))
                table.append(dict(**case,**metrics(book),tail10_bps=tail_mean(book.net.to_numpy())*1e4,
                                  mean_long_share=shares.mean(),long_bps=book.long_net.mean()*1e4,
                                  short_bps=book.short_net.mean()*1e4))
            for arm in ARMS:
                for band in ("narrow","wide"):
                    key = f"{arm}_{band}"
                    refs = ["static50","static55",f"{key}_mean"]
                    if arm!="freeze_projection":
                        refs.append(f"freeze_projection_{band}")
                    for ref in refs:
                        comparisons.append(dict(split=split,trade_bps=fee,arm=key,control=ref,
                                                **paired(books[key].net.to_numpy(),books[ref].net.to_numpy())))
        print(f"{split}: {len(block)} weeks, 10/30 bp replay complete",flush=True)

    def save(stage):
        pd.concat(all_books,ignore_index=True).to_parquet(stage/"weekly.parquet",index=False)
        signals.to_parquet(stage/"signals.parquet",index=False)
        for name, value in (("metrics",table),("paired",comparisons),("quality",quality),
                            ("selection",diagnostics),("training_loss",traces)):
            pd.DataFrame(value).to_csv(stage/f"{name}.csv",index=False)
    publish(out,save)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out",type=Path)
    args = parser.parse_args()
    checkpoints = args.out.parent/"checkpoints"
    checkpoints.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        main(args.out.resolve(),checkpoints)
