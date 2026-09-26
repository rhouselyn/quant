"""Direction calibration and stock reliability estimates without score-sign assumptions."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def direction_rows(pred):
    # Flat and missing weeks are not up/down labels; retain them in return backtests.
    return pred[np.isfinite(pred.score) & np.isfinite(pred.realized_return)
                & pred.realized_return.abs().gt(1e-8)].copy()


def calibrate(past, future):
    known = direction_rows(past)
    labels = known.realized_return.gt(0).astype(int)
    if labels.nunique() != 2:
        raise ValueError('Calibration needs both up and down observations')
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    estimator.fit(known[['score']], labels)
    result = future.copy()
    result['up_probability'] = estimator.predict_proba(result[['score']])[:, 1]
    return result


def walk_forward_calibration(val, warmup=26):
    dates = sorted(val.date.unique())
    if len(dates) <= warmup:
        raise ValueError('Too few validation weeks')
    frames = []
    for date in dates[warmup:]:
        frames.append(calibrate(val[val.date < date], val[val.date == date]))
    return pd.concat(frames, ignore_index=True)


def direction_metrics(frame):
    d = direction_rows(frame)
    y = d.realized_return.gt(0).to_numpy()
    p = d.up_probability.to_numpy()
    predicted = p >= .5
    up, down = int(y.sum()), int((~y).sum())
    tp, tn = int((y & predicted).sum()), int((~y & ~predicted).sum())
    tpr, tnr = tp/up if up else None, tn/down if down else None
    return dict(n=len(d), n_up=up, n_down=down, tp=tp, tn=tn,
                up_recall=tpr, down_recall=tnr,
                balanced_accuracy=(tpr+tnr)/2 if up and down else None,
                accuracy=float((y == predicted).mean()) if len(d) else None,
                brier=float(np.mean((p-y)**2)) if len(d) else None,
                up_rate=float(y.mean()) if len(d) else None,
                predicted_up_rate=float(predicted.mean()) if len(d) else None,
                label_coverage=len(d)/len(frame) if len(frame) else 0.)


def rank_reliability(oof, count=150, prior=5):
    """Rank by the worse half's balanced accuracy, with Beta(5,5) shrinkage."""
    dates = sorted(oof.date.unique())
    midpoint = dates[len(dates)//2]
    rows = []
    for symbol, group in oof.groupby('symbol', sort=True):
        metrics = [direction_metrics(g) for g in
                   [group, group[group.date < midpoint], group[group.date >= midpoint]]]
        shrunk = [((m['tp']+prior)/(m['n_up']+2*prior)
                   + (m['tn']+prior)/(m['n_down']+2*prior))/2 for m in metrics]
        eligible = all(m['n_up'] >= 5 and m['n_down'] >= 5 for m in metrics[1:])
        rows.append(dict(symbol=symbol, eligible=eligible, n=metrics[0]['n'],
                         n_up=metrics[0]['n_up'], n_down=metrics[0]['n_down'],
                         val_balanced_accuracy=metrics[0]['balanced_accuracy'],
                         val_up_recall=metrics[0]['up_recall'], val_down_recall=metrics[0]['down_recall'],
                         shrunk_accuracy=shrunk[0], half1_shrunk=shrunk[1], half2_shrunk=shrunk[2],
                         reliability=min(shrunk[1:]), stability_gap=abs(shrunk[1]-shrunk[2])))
    table = pd.DataFrame(rows).sort_values(
        ['eligible', 'reliability', 'shrunk_accuracy', 'symbol'], ascending=[False, False, False, True])
    if table.eligible.sum() < count:
        raise ValueError('Insufficient stocks with up/down coverage in both halves')
    table['selected'] = False
    table.loc[table.index[:count], 'selected'] = True
    table['rank'] = range(1, len(table)+1)
    return table.reset_index(drop=True)
