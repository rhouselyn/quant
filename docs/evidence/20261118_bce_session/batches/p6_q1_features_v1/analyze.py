"""Verify the fixed-data comparison and attribute the historical signal gap."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

OUT=Path(__file__).resolve().parent
PARENT=OUT.parent/"p6_original_controls_v1"


def main():
    prior=json.loads((PARENT/"validation_lock.json").read_text(encoding="utf8"))
    current=json.loads((OUT/"validation_lock.json").read_text(encoding="utf8"))
    assert prior["train_dates"]==current["train_dates"] and len(current["train_dates"])==473
    assert prior["validation_dates"]==current["validation_dates"] and len(current["validation_dates"])==123
    identity=pd.read_json(OUT/"training_identity.json")
    for suffix in ("plain","fixed5"):
        group=identity.loc[identity.arm.str.endswith("_"+suffix)]
        assert group.supervision_sha256.nunique()==1
    head_equal=[]
    for arm in ("p6_plain","p6_fixed5"):
        for seed in (42,43,44):
            a,b=[torch.load(p/f"{arm}_{seed}.pt",weights_only=True) for p in (PARENT,OUT)]
            same=all(torch.equal(a["state"][k],b["state"][k]) for k in a["state"])
            same=same and all(torch.equal(a[k],b[k]) for k in ("mu","sd"))
            assert same
            head_equal.append(dict(arm=arm,seed=seed,parameters_identical=same))
    stats=pd.read_csv(OUT/"metrics.csv")
    oldstats=pd.read_csv(PARENT/"metrics.csv")
    pd.testing.assert_frame_equal(stats,oldstats,check_exact=False,rtol=0,atol=1e-12)
    weekly=pd.read_parquet(OUT/"weekly.parquet")
    oldweekly=pd.read_parquet(PARENT/"weekly.parquet")
    assert len(weekly)==6792 and not weekly.duplicated(["arm","rule","fee","date"]).any()
    np.testing.assert_array_equal(weekly.net,oldweekly.net)
    np.testing.assert_allclose(weekly.long_net+weekly.short_net,weekly.net,rtol=0,atol=1e-12)
    sig=pd.read_csv(OUT/"signals.csv",parse_dates=["date"]).set_index("date")
    olds=pd.read_csv(PARENT/"signals.csv",parse_dates=["date"]).set_index("date")
    pd.testing.assert_index_equal(sig.index,olds.index)
    assert len(sig)==283
    logit_change={arm:dict(max_abs=float((sig[arm]-olds[arm]).abs().max()),
                          flips=int(sig[arm].gt(0).ne(olds[arm].gt(0)).sum())) for arm in ("p6_plain","p6_fixed5")}
    fixed=pd.read_parquet(OUT.parent/"b0_p6_frozen_bce_transfer_v1/report/weekly.parquet")
    fixed=fixed.query("model=='P6' and arm=='fixed50' and trade_bps==10").set_index("date").reindex(sig.index)
    full=weekly.query("arm=='old_triple_plain' and rule=='full' and fee==10").set_index("date").reindex(sig.index)
    direction=sig.old_triple_plain.gt(0).to_numpy()
    long=np.where(direction,full.gross_ret,2*fixed.gross_ret-full.gross_ret)
    short=2*fixed.gross_ret.to_numpy()-long
    borrowing=2*fixed.borrow_cost.to_numpy()
    gap=long-(short-borrowing)
    use=fixed.missing.eq(0).to_numpy()
    single=pd.read_csv(OUT/"single_head_signals.csv",parse_dates=["date"]).set_index("date")
    signals=pd.concat([sig,single],axis=1)
    signal_stats=[]
    for arm in signals:
        pred=signals[arm].gt(0).to_numpy()
        errors=pred[use]!=(gap[use]>0)
        signal_stats.append(dict(arm=arm,long_weeks=int(pred.sum()),weeks=int(use.sum()),
                                 accuracy=float(1-errors.mean()),
                                 auc=float(roc_auc_score(gap[use]>0,signals[arm].to_numpy()[use])),
                                 magnitude_weighted_accuracy=float(1-np.abs(gap[use])[errors].sum()/np.abs(gap[use]).sum()),
                                 theoretical_leg_mean_bp=float(np.where(pred,long,short-borrowing).mean()*1e4)))
    pd.DataFrame(signal_stats).to_csv(OUT/"signal_quality.csv",index=False)
    attribution=[]
    worst=[]
    for fee in (10,30):
        o=weekly.query("arm=='old_triple_plain' and rule=='full' and fee==@fee").set_index("date").reindex(sig.index)
        n=weekly.query("arm=='p6_plain' and rule=='full' and fee==@fee").set_index("date").reindex(sig.index)
        delta=n.net-o.net
        attribution.append(dict(fee=fee,net_mean_difference_bp=float(delta.mean()*1e4),
                                 gross_mean_difference_bp=float((n.gross_ret-o.gross_ret).mean()*1e4),
                                 extra_trading_bp=float((n.trading_cost-o.trading_cost).mean()*1e4),
                                 extra_borrowing_bp=float((n.borrow_cost-o.borrow_cost).mean()*1e4),
                                 disagreement_weeks=int(sig.p6_plain.gt(0).ne(sig.old_triple_plain.gt(0)).sum())))
        np.testing.assert_allclose(np.where(sig.p6_plain.gt(0),long,short),n.gross_ret,atol=1e-12,rtol=0)
        for date in delta.nsmallest(5).index:
            worst.append(dict(fee=fee,date=date,old_net=o.at[date,"net"],p6_net=n.at[date,"net"],delta=delta.at[date]))
    pd.DataFrame(attribution).to_csv(OUT/"attribution.csv",index=False)
    pd.DataFrame(worst).to_csv(OUT/"worst_disagreements.csv",index=False)
    single_stats=pd.read_csv(OUT/"single_head_metrics.csv")
    single_weekly=pd.read_parquet(OUT/"single_head_weekly.parquet")
    assert len(single_stats)==24 and len(single_weekly)==6792
    assert not single_weekly.duplicated(["arm","rule","fee","date"]).any()
    audit=dict(training_weeks=473,validation_label_weeks=123,validation_inference_weeks=133,test_weeks=283,
               supervision_matched=True,head_parameters=head_equal,logit_change=logit_change,
               weekly_net_returns_identical=True,completed_single_head_replays=6)
    (OUT/"verification.json").write_text(json.dumps(audit,indent=2),encoding="utf8")
    inputs=json.loads((OUT/"inputs.json").read_text(encoding="utf8"))
    with Path(__file__).open("rb") as handle:
        inputs[str(Path(__file__).resolve())]=hashlib.file_digest(handle,"sha256").hexdigest()
    (OUT/"inputs.json").write_text(json.dumps(inputs,indent=2),encoding="utf8")
    print("Verified 473/123/133/283 dates, supervision, six equal heads and all net returns.")
    print(single_stats.query("rule=='full'")[["arm","fee","annual_calendar","max_drawdown","sharpe"]].to_string(index=False))
    print(pd.DataFrame(signal_stats).round(4).to_string(index=False))
    print(pd.DataFrame(attribution).to_string(index=False))
    print(pd.DataFrame(worst).head().to_string(index=False))


if __name__=="__main__":
    main()
