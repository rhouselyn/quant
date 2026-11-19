"""Independent MLP stock weighting and attention leg selection."""
import json

import numpy as np
import pandas as pd
import torch

from directional_2018_experiment import OUT as CACHE, groups_for, select_model
from directional_attention_experiment import Allocation, backtest
from r1111_common import newey_west_t, perf_summary

OUT = CACHE / 'three_seed'
SEEDS = (42, 43, 44)


def cache_path(seed):
    return CACHE / ('features.npz' if seed == 42 else f'features_s{seed}.npz')


def standard_scores(score, available):
    masked = np.where(available, score, np.nan)
    out = np.full(score.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(masked).any(axis=1)
    keep = masked[valid]
    out[valid] = (keep - np.nanmean(keep, axis=1, keepdims=True)) / (
        np.nanstd(keep, axis=1, keepdims=True) + 1e-6)
    return out


def stock_inputs(z, t, group):
    long, short = group
    ids = np.r_[long, short]
    sides = np.r_[np.ones(len(long)), -np.ones(len(short))]
    return ids, sides, np.column_stack((z[t, ids], sides)).astype(np.float32)


def fit_predict(seed, d, groups, baskets, train, val, test):
    with np.load(cache_path(seed)) as cache:
        z = cache['z']

    def examples(indices, kind):
        xs, ys = [], []
        for t in indices:
            group = groups[t] if kind == 'mlp' else baskets[t]
            ids, sides, x = stock_inputs(z, t, group)
            y = d['y'][t, ids]
            valid = d['valid'][t, ids] & np.isfinite(y)
            if kind == 'mlp':
                keep = valid & (y != 0)
                xs.append(x[keep])
                ys.append((y[keep] * sides[keep] > 0).astype(np.float32))
            elif len(group[0]) == 40 and len(group[1]) == 10 and valid.all():
                r = np.expm1(y)
                xs.append(x)
                ys.append(r[:40].mean() > -r[40:].mean() - .08 / 52)
        if kind == 'mlp':
            return np.concatenate(xs), np.concatenate(ys)
        return np.stack(xs), np.asarray(ys, dtype=np.float32)

    estimates, selections = {}, {}
    for kind in ('mlp', 'attention'):
        a, ay = examples(train, kind)
        b, by = examples(val, kind)
        flat = a.reshape(-1, a.shape[-1])
        mean, std = flat.mean(0), flat.std(0) + 1e-6
        torch.manual_seed(seed)
        if kind == 'mlp':
            model = torch.nn.Sequential(torch.nn.Linear(129, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
        else:
            model = Allocation(True)
            model.output = torch.nn.Linear(32, 1)
        model, step, bce = select_model(model, torch.from_numpy((a-mean)/std), torch.from_numpy(ay),
                                        torch.from_numpy((b-mean)/std), torch.from_numpy(by))
        selections[kind] = dict(step=step, val_bce=bce, train_n=len(ay), val_n=len(by))
        print(seed, kind, selections[kind], flush=True)
        torch.save(dict(model=model.state_dict(), mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                        step=step), OUT / f'{kind}_s{seed}.pt')
        values = {}
        with torch.inference_mode():
            for t in test:
                _, _, x = stock_inputs(z, t, groups[t] if kind == 'mlp' else baskets[t])
                features = torch.from_numpy((x-mean)/std)
                if kind == 'mlp':
                    values[t] = torch.sigmoid(model(features)).numpy().ravel()
                else:
                    values[t] = model(features[None]).item()
        estimates[kind] = values
    return estimates, selections


def book_choices(t, groups, baskets, probability, long_share):
    long, short = groups[t]
    ids = np.r_[long, short]
    sides = np.r_[np.ones(len(long)), -np.ones(len(short))]
    chosen = np.argsort(-probability, kind='stable')[:50]
    w = np.exp(probability[chosen] - probability[chosen].max())
    mlp = (t, ids[chosen], sides[chosen], w / w.sum())
    long, short = baskets[t]
    weights = np.r_[np.full(len(long), long_share / max(len(long), 1)),
                    np.full(len(short), (1 - long_share) / max(len(short), 1))]
    attention = (t, np.r_[long, short], np.r_[np.ones(len(long)), -np.ones(len(short))], weights)
    return mlp, attention


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    scores = []
    for seed in SEEDS:
        with np.load(cache_path(seed)) as cache:
            if seed == 42:
                d = {key: cache[key] for key in cache.files if key != 'z'}
            else:
                for key in ('dates', 'symbols', 'available', 'valid'):
                    assert np.array_equal(d[key], cache[key]), f'Cache mismatch: {seed} {key}'
                assert np.array_equal(d['y'], cache['y'], equal_nan=True), f'Return mismatch: {seed}'
            scores.append(standard_scores(cache['score'], d['available']))
    d['score'] = np.mean(scores, axis=0)
    dates = pd.to_datetime(d['dates'])
    groups = groups_for(d, dates)
    baskets = [(long[:40], short[:10]) for long, short in groups]
    first_val = dates.searchsorted('2018-01-05')
    first_test = dates.searchsorted('2021-01-08')
    train = np.arange(first_val - 2)
    val = np.arange(first_val, first_test - 2)
    test = [t for t in range(first_test, len(dates)) if d['valid'][t].any()
            and sum(map(len, groups[t])) >= 50]
    assert all(t < first_test for t in val) and test[0] >= first_test
    predictions, selections = {}, {}
    for seed in SEEDS:
        predictions[seed], selections[seed] = fit_predict(seed, d, groups, baskets, train, val, test)
    predictions['ensemble'] = {
        kind: {t: np.mean([predictions[s][kind][t] for s in SEEDS], axis=0) for t in test}
        for kind in ('mlp', 'attention')}
    books = {}
    rows, stock_rows = [], []
    for seed, pred in predictions.items():
        mlp_choices, att_choices = [], []
        for t in test:
            score = float(pred['attention'][t])
            mlp, att = book_choices(t, groups, baskets, pred['mlp'][t], float(score > 0))
            assert np.isclose(mlp[3].sum(), 1) and len(mlp[1]) == 50
            assert att[3][att[2] == (-1 if score > 0 else 1)].sum() == 0
            mlp_choices.append(mlp)
            att_choices.append(att)
            rows.append(dict(seed=seed, date=dates[t], score=score, long=int(score > 0),
                             mlp_long_count=int((mlp[2] == 1).sum())))
            if seed == 'ensemble':
                stock_rows.extend(dict(date=dates[t], symbol=d['symbols'][i], side=int(side), weight=float(w))
                                  for i, side, w in zip(*mlp[1:]))
        books[f'mlp_{seed}'] = mlp_choices
        books[f'attention_{seed}'] = att_choices
    matched = float(np.mean([predictions['ensemble']['attention'][t] > 0 for t in test]))
    for name, share in [('old_long', 1.), ('old_short', 0.), ('old_half', .5), ('old_matched', matched)]:
        books[name] = [book_choices(t, groups, baskets, predictions['ensemble']['mlp'][t], share)[1] for t in test]
    frames = {name: backtest(d, dates, entries, name) for name, entries in books.items()}
    control = frames['old_half'].net
    report = dict(selections=selections, ensemble_long_week_share=matched, arms={})
    for name, frame in frames.items():
        delta = frame.net - control
        stats = dict(**perf_summary(frame.net), delta_bps=float(delta.mean()*1e4),
                     nw_t=newey_west_t(delta), avg_long=float(frame.long_weight.mean()),
                     turnover=float(frame.turnover.mean()))
        report['arms'][name] = stats
        print(name, f'{stats["ann"]:.2%} ann, {stats["sharpe"]:.2f} Sharpe, '
              f'{stats["max_drawdown"]:.2%} drawdown, {stats["avg_long"]:.1%} long', flush=True)
    pd.concat(frames.values()).to_csv(OUT / 'weekly.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'scores.csv', index=False)
    pd.DataFrame(stock_rows).to_csv(OUT / 'mlp_stocks.csv', index=False)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
