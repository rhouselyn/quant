"""Controlled attention ablations and validation-selected allocation rules."""
import numpy as np
import pandas as pd
import torch
from scipy.stats import shapiro, skew
from torch import nn

from directional_2018_experiment import groups_for, select_model
from directional_attention_experiment import Allocation, backtest
from directional_three_seed import OUT as PRIOR, SEEDS, cache_path, standard_scores, stock_inputs
from quant_mvp.long_only import long_only_backtest
from r1111_common import ROOT, moving_block_bootstrap, newey_west_t, perf_summary, write_json

OUT = PRIOR / 'followup'
VARIANTS = ('gap_weighted', 'cost_aware', 'mean_pool', 'two_queries')


class CostAware(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Allocation(True)
        self.attention.output = nn.Identity()
        self.output = nn.Linear(33, 1)

    def forward(self, x):
        return self.output(torch.cat((self.attention(x[..., :129]), x[:, 0, 129:]), dim=-1))


class LegPool(nn.Module):
    def __init__(self, two_queries):
        super().__init__()
        self.proj = nn.Linear(129, 32)
        self.two_queries = two_queries
        if two_queries:
            self.query = nn.Parameter(torch.randn(1, 2, 32) * .02)
            self.cross = nn.MultiheadAttention(32, 4, batch_first=True)
        self.norm = nn.LayerNorm(32)
        self.ffn = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.final_norm = nn.LayerNorm(32)
        self.output = nn.Linear(64, 1)

    def pool(self, h):
        long, short = h[:, :40], h[:, 40:]
        if self.two_queries:
            q = self.query.expand(len(h), -1, -1)
            legs = [self.cross(q[:, i:i+1], leg, leg, need_weights=False)[0]
                    for i, leg in enumerate((long, short))]
            pooled = q + torch.cat(legs, dim=1)
        else:
            pooled = torch.stack((long.mean(1), short.mean(1)), dim=1)
        return pooled

    def forward(self, x):
        pooled = self.norm(self.pool(self.proj(x)))
        return self.output(self.final_norm(pooled + self.ffn(pooled)).flatten(1))


def trade_gap(previous, current):
    """Cost(long) minus cost(short) using target notional available at decision."""
    old, sign = previous
    if not len(old):
        return 0.
    leg = current[0 if sign else 1]
    overlap = len(np.intersect1d(old, leg)) / max(len(old), len(leg))
    return .002 * overlap * (-1 if sign else 1)


def fit_scores(d, baskets, train, val_labels, val, test, gaps, prior_basket):
    predictions = {name: [] for name in ('original', *VARIANTS)}
    selections = []
    for seed in SEEDS:
        with np.load(cache_path(seed)) as cache:
            z = cache['z']
        x = {t: stock_inputs(z, t, baskets[t])[2] for t in sorted(set(train + val_labels + val + test))}
        del z
        a = np.stack([x[t] for t in train])
        mu, sd = a.reshape(-1, 129).mean(0), a.reshape(-1, 129).std(0) + 1e-6
        original = Allocation(True)
        original.output = nn.Linear(32, 1)
        saved = torch.load(PRIOR / f'attention_s{seed}.pt', map_location='cpu', weights_only=True)
        original.load_state_dict(saved['model'])
        with torch.inference_mode():
            predictions['original'].append({t: original.eval()((torch.from_numpy(x[t])[None] - saved['mean'])
                                                               / saved['std']).item() for t in val + test})
        for name in VARIANTS:
            torch.manual_seed(seed)
            if name == 'gap_weighted':
                net = Allocation(True)
                net.output = nn.Linear(32, 1)
            elif name == 'cost_aware':
                net = CostAware()
            else:
                net = LegPool(name == 'two_queries')

            def examples(indices):
                features = np.stack([(x[t]-mu)/sd for t in indices])
                gap = gaps[indices]
                if name == 'cost_aware':
                    cost = np.array([[trade_gap((prior_basket[t][i], i == 0), baskets[t])
                                      for i in (0, 1)] for t in indices], dtype=np.float32).ravel()
                    features = np.repeat(features, 2, axis=0)
                    extra = np.broadcast_to(cost[:, None, None] / .002, (*features.shape[:2], 1))
                    features = np.concatenate((features, extra), axis=2)
                    gap = np.repeat(gap, 2) - cost
                weight = np.abs(gap)
                return (torch.from_numpy(features), torch.tensor(gap > 0, dtype=torch.float32),
                        torch.tensor(weight / weight.mean(), dtype=torch.float32) if name == 'gap_weighted' else None)

            xx, yy, wt = examples(train)
            vx, vy, vw = examples(val_labels)
            net, step, bce = select_model(net, xx, yy, vx, vy, wt, vw)
            selections.append(dict(seed=seed, arm=name, step=step, val_loss=bce))
            print(selections[-1], flush=True)
            torch.save(dict(model=net.state_dict(), mean=torch.from_numpy(mu), std=torch.from_numpy(sd)),
                       OUT / f'{name}_s{seed}.pt')
            output = {}
            with torch.inference_mode():
                for t in val + test:
                    xx = (x[t]-mu)/sd
                    if name == 'cost_aware':
                        # Precompute both previous-side states on the chronological replay path.
                        last = prior_basket[t]
                        xx = np.stack([np.column_stack((xx, np.full(len(xx), trade_gap(
                            (last[i], i == 0), baskets[t]) / .002, dtype=np.float32))) for i in (0, 1)])
                        output[t] = net(torch.from_numpy(xx)).numpy().ravel()
                    else:
                        output[t] = net(torch.from_numpy(xx[None])).item()
            predictions[name].append(output)
    averaged = {name: {t: np.mean([v[t] for v in outputs], axis=0) for t in val + test}
                for name, outputs in predictions.items()}
    return averaged, selections


def replay(d, dates, baskets, indices, shares, arm, *, gross=None, **ledger_options):
    positions = []
    gross = np.ones(len(indices)) if gross is None else np.asarray(gross,dtype=float)
    if gross.shape!=(len(indices),) or not np.isfinite(gross).all() or not ((gross>=0)&(gross<=1)).all():
        raise ValueError("gross must provide a finite 0–1 scale per week")
    for t, share, scale in zip(indices, shares, gross):
        long, short = baskets[t]
        w = scale*np.r_[np.full(len(long), share / len(long)), np.full(len(short), (1-share) / len(short))]
        positions.append((t, np.r_[long, short], np.r_[np.ones(len(long)), -np.ones(len(short))], w))
    return backtest(d, dates, positions, arm, **ledger_options)


def compare(frame, reference):
    d = frame.net.to_numpy() - reference.net.to_numpy()
    ci = moving_block_bootstrap(d, blocks=(8,), reps=2000)['L8']
    return dict(mean_bps=float(d.mean()*1e4), nw_t=newey_west_t(d),
                ci95_bps=[ci['lo']*1e4, ci['hi']*1e4])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    scores = []
    for seed in SEEDS:
        with np.load(cache_path(seed)) as c:
            if seed == 42:
                d = {key: c[key] for key in c.files if key != 'z'}
            scores.append(standard_scores(c['score'], d['available']))
    d['score'] = np.mean(scores, axis=0)
    dates = pd.to_datetime(d['dates'])
    baskets = [(l[:40], s[:10]) for l, s in groups_for(d, dates)]
    first_val = dates.searchsorted('2018-01-05')
    first_test = dates.searchsorted('2021-01-08')
    full = [t for t, (l, s) in enumerate(baskets) if len(l) == 40 and len(s) == 10 and d['valid'][t].any()]
    gaps = np.full(len(dates), np.nan)
    complete = np.zeros(len(dates), dtype=bool)
    for t in full:
        l, s = baskets[t]
        ids = np.r_[l, s]
        valid = d['valid'][t, ids] & np.isfinite(d['y'][t, ids])
        r = np.expm1(np.where(valid, d['y'][t, ids], 0))
        gaps[t] = r[:40].mean() + r[40:].mean() + .08/52
        complete[t] = valid.all()
    train = [t for t in full if t < first_val-2 and complete[t]]
    val = [t for t in full if first_val <= t < first_test-2]
    val_labels = [t for t in val if complete[t]]
    test = [t for t in full if t >= first_test]
    prior_basket = {}
    for split in ([t for t in full if t < first_val-2], val, test):
        for i, t in enumerate(split):
            prior_basket[t] = baskets[split[i-1]] if i else (np.array([], dtype=int),)*2
    models, selections = fit_scores(d, baskets, train, val_labels, val, test, gaps, prior_basket)
    previous = pd.read_csv(PRIOR / 'scores.csv', parse_dates=['date'])
    original = previous[previous.seed == 'ensemble'].set_index('date').score
    assert np.allclose(original.loc[dates[test]], [models['original'][t] for t in test], atol=1e-6)
    report = dict(selections=selections, validation_weeks=len(val), test_weeks=len(test), ablations={})
    book_frames = []
    for split, ix in [('val', val), ('test', test)]:
        book_frames.append(replay(d, dates, baskets, ix, np.full(len(ix), .5), 'same_baskets_50_50').assign(split=split))
        for name, pred in models.items():
            if name == 'cost_aware':
                direction = True
                shares = []
                for t in ix:
                    direction = bool(pred[t][0 if direction else 1] > 0)
                    shares.append(float(direction))
                shares = np.asarray(shares)
            else:
                shares = np.array([pred[t] > 0 for t in ix], dtype=float)
            frame = replay(d, dates, baskets, ix, shares, name).assign(split=split)
            book_frames.append(frame)
            report['ablations'][f'{split}_{name}'] = dict(**perf_summary(frame.net),
                accuracy=float(((shares > 0) == (gaps[ix] > 0))[complete[ix]].mean()),
                long_share=float(shares.mean()), turnover=float(frame.turnover.mean()))
    weekly = pd.concat(book_frames, ignore_index=True)
    test_weekly = weekly[weekly.split == 'test'].pivot(index='date', columns='arm', values='net').reset_index()
    frozen = pd.read_parquet(ROOT / 'outputs/baseline_v2_research/predictions_trim2.parquet')
    assert 'tradable' not in frozen.columns, 'Frozen comparator changed entry filter'
    frozen['date'] = pd.to_datetime(frozen.date)
    long40, _ = long_only_backtest(frozen, top_n=40)
    base = pd.read_csv(PRIOR.parent / 'baseline_comparison.csv', parse_dates=['date'])
    base = base.merge(long40[['date', 'long_net_return']], on='date', validate='one_to_one', suffixes=('_20','_40'))
    base['baseline40'] = (base.long_net_return_40 + base.short_net_return - .08/52) / 2
    aligned = test_weekly.merge(base[['date', 'baseline', 'baseline40']], on='date', validate='one_to_one')
    assert len(aligned) == len(test)
    aligned.to_csv(OUT / 'aligned_test.csv', index=False)
    for name in ('baseline', 'baseline40', 'same_baskets_50_50', 'original', *VARIANTS):
        report[name] = dict(**perf_summary(aligned[name]), vs_top40=compare(
            pd.DataFrame({'net':aligned[name]}), pd.DataFrame({'net':aligned.baseline40})))
        if name in VARIANTS:
            report[name]['vs_original'] = compare(pd.DataFrame({'net':aligned[name]}),
                                                  pd.DataFrame({'net':aligned.original}))

    validation_scores = np.array([models['original'][t] for t in val])
    sd = validation_scores.std(ddof=1)
    report['score_scale'] = float(sd)
    absolute_bins = np.quantile(np.abs(validation_scores), [1/3, 2/3])
    for split, ix in [('val', val), ('test', test)]:
        logits = np.array([models['original'][t] for t in ix])
        report[f'{split}_distribution'] = dict(mean=float(logits.mean()), std=float(logits.std(ddof=1)),
            skew=float(skew(logits)), shapiro_p=float(shapiro(logits).pvalue), bins=[])
        for b in range(3):
            mask = (np.digitize(np.abs(logits), absolute_bins) == b) & complete[ix]
            gap = gaps[ix][mask]
            correct = (logits[mask] > 0) == (gap > 0)
            report[f'{split}_distribution']['bins'].append(dict(n=int(mask.sum()),
                accuracy=float(correct.mean()), signed_gap_bps=float(np.mean(np.where(logits[mask]>0, gap, -gap))*1e4)))
    report['confidence_bin_edges'] = absolute_bins.tolist()
    grid, val_returns, test_returns = [], [], []
    params = [(k, hi, lo) for k in (0,.25,.5,.75,1,1.5,2,3)
              for hi in (.6,.7,.8,.9,1.) for lo in (0.,.1,.2,.3,.4)]
    params.append((np.inf, 1., 0.))
    for k, hi, lo in params:
        row = dict(k=k, threshold=k*sd, high_long=hi, low_long=lo)
        for split, ix, returns in [('val', val, val_returns), ('test', test, test_returns)]:
            logits = np.array([models['original'][t] for t in ix])
            shares = np.where(logits > k*sd, hi, np.where(logits < -k*sd, lo, .5))
            frame = replay(d, dates, baskets, ix, shares, split)
            returns.append(frame.net.to_numpy())
            stats = perf_summary(frame.net)
            row.update({f'{split}_{key}': stats[key] for key in ('ann','sharpe','max_drawdown')})
            row[f'{split}_neutral'] = float((shares == .5).mean())
        grid.append(row)
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / 'allocation_grid.csv', index=False)
    val_returns, test_returns = np.asarray(val_returns), np.asarray(test_returns)
    rng = np.random.default_rng(42)
    starts = rng.integers(0,len(val)-7,size=(1000,int(np.ceil(len(val)/8))))
    boot_ix = (starts[...,None]+np.arange(8)).reshape(1000,-1)[:, :len(val)]
    boot_sharpe = np.stack([v[boot_ix].mean(1)/v[boot_ix].std(1,ddof=1) for v in val_returns])
    orig = int(grid.index[(grid.k == 0) & (grid.high_long == 1) & (grid.low_long == 0)][0])
    for name, mask in [('threshold', (grid.high_long == 1)&(grid.low_long == 0)),
                       ('weights', grid.k == 0), ('joint', grid.k >= 0)]:
        pool = grid.index[mask].to_numpy()
        winner = int(grid.loc[pool,'val_sharpe'].idxmax())
        selected = pool[boot_sharpe[pool].argmax(0)]
        top = pd.Series(selected).value_counts().head(5)
        report[f'{name}_grid'] = dict(best_val=grid.loc[winner].to_dict(),
            best_test_retro=grid.loc[grid.loc[pool,'test_sharpe'].idxmax()].to_dict(),
            rank_corr=float(grid.loc[pool,['val_sharpe','test_sharpe']].corr(method='spearman').iloc[0,1]),
            boot_exact_frequency=float((selected == winner).mean()),
            boot_top=[dict(**grid.loc[i,['k','threshold','high_long','low_long']].to_dict(),
                           frequency=float(n/len(selected))) for i,n in top.items()],
            vs_original=compare(pd.DataFrame({'net':test_returns[winner]}),pd.DataFrame({'net':test_returns[orig]})),
            vs_top40=compare(pd.DataFrame({'net':test_returns[winner]}),pd.DataFrame({'net':aligned.baseline40})))
    weekly.to_csv(OUT / 'ablation_weekly.csv', index=False)
    write_json(OUT / 'report.json', report)
    print('completed', OUT, flush=True)


if __name__ == '__main__':
    main()
