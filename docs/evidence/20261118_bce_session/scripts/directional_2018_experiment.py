"""2018-cutoff validation-selected direction and allocation heads."""
import copy
import json
import numpy as np
import pandas as pd
import torch
import yaml

from score_mlp_experiment import ROOT, make_batch, arrays, HierarchicalRanker, load_eligibility
from directional_attention_experiment import Allocation, backtest, leg_weights
from r1111_common import perf_summary, newey_west_t

OUT = ROOT / 'outputs/directional_2018'


def extract(run=ROOT / 'outputs/abl2000_base', output=OUT / 'features.npz'):
    cfg = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf8'))
    data = arrays(cfg)
    model = HierarchicalRanker(cfg, data[8]).cuda().eval()
    model.load_state_dict(torch.load(run / 'best.pt', map_location='cuda', weights_only=True))
    zz, ss = [], []
    start = cfg['lookback']
    with torch.inference_mode():
        for i in range(start, len(data[6]), 4):
            ts = list(range(i, min(i + 4, len(data[6]))))
            _, z, s, _ = model(*make_batch(data, ts, cfg, 'cuda'))
            zz.append(z.cpu().numpy())
            ss.append(s.cpu().numpy())
            if (i - start) % 120 == 0:
                print('extract', i, '/', len(data[6]), flush=True)
    np.savez_compressed(output, z=np.concatenate(zz), score=np.concatenate(ss),
        dates=data[6][start:].astype('datetime64[D]'), symbols=np.asarray(data[7]),
        available=data[4][start:], y=data[1][start:], valid=data[2][start:])


def groups_for(d, dates):
    eligibility = load_eligibility()
    result = []
    for t, date in enumerate(dates):
        order = np.argsort(-d['score'][t], kind='stable')
        available = d['available'][t] & np.isfinite(d['score'][t])
        long = [i for i in order if available[i]][:50]
        quarter = str((date - pd.offsets.QuarterEnd(1)).to_period('Q'))
        allowed = eligibility.get(quarter, set())
        short = [i for i in order[::-1] if available[i] and i not in long and
                 (date.year < 2018 or d['symbols'][i].split('.')[0] in allowed)][:50]
        result.append((np.array(long, dtype=int), np.array(short, dtype=int)))
    return result


def select_model(model, x, y, vx, vy, train_weight=None, val_weight=None, regression_loss=None):
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.01)
    best, state, step = float('inf'), None, 0
    for n in range(1, 101):
        model.train()
        opt.zero_grad()
        pred = model(x).flatten()
        loss = (regression_loss(pred, y) if regression_loss is not None else
                torch.nn.functional.binary_cross_entropy_with_logits(pred, y, weight=train_weight))
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(vx).flatten()
            v = (regression_loss(pred, vy) if regression_loss is not None else
                 torch.nn.functional.binary_cross_entropy_with_logits(pred, vy, weight=val_weight)).item()
        if v < best:
            best, state, step = v, copy.deepcopy(model.state_dict()), n
    model.load_state_dict(state)
    return model.eval(), step, best


def main():
    torch.set_num_threads(4)
    with np.load(OUT / 'features.npz') as source:
        d = {key: source[key] for key in source.files}
    dates = pd.to_datetime(d['dates'])
    groups = groups_for(d, dates)
    first_val = np.flatnonzero(dates >= '2018-01-05')[0]
    first_test = np.flatnonzero(dates >= '2021-01-08')[0]
    train = np.arange(0, first_val - 2)
    val = np.arange(first_val, first_test - 2)
    test = np.arange(first_test, len(dates))
    def rows(indices):
        xs, ys = [], []
        for t in indices:
            for side, sign in ((0, 1), (1, -1)):
                ix = groups[t][side]
                keep = d['valid'][t, ix] & np.isfinite(d['y'][t, ix]) & (d['y'][t, ix] != 0)
                ix = ix[keep]
                xs.append(np.column_stack((d['z'][t, ix], np.full(len(ix), sign))))
                ys.append((sign * d['y'][t, ix] > 0).astype('float32'))
        return np.concatenate(xs).astype('float32'), np.concatenate(ys)
    a, ay = rows(train)
    b, by = rows(val)
    mean, std = a.mean(0), a.std(0) + 1e-6
    torch.manual_seed(42)
    mlp = torch.nn.Sequential(torch.nn.Linear(129, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
    mlp, mlp_step, mlp_loss = select_model(mlp, torch.tensor((a - mean)/std), torch.tensor(ay),
                                           torch.tensor((b - mean)/std), torch.tensor(by))
    print('direction selected', mlp_step, mlp_loss, flush=True)
    stocks, allocation_x, labels = {}, {}, {}
    def build(indices):
      for t in indices:
        long, short = groups[t]
        ix = np.r_[long, short]
        side = np.r_[np.ones(len(long)), -np.ones(len(short))]
        if len(ix) < 50 or not d['valid'][t].any():
            continue
        x = np.column_stack((d['z'][t, ix], side)).astype('float32')
        with torch.no_grad():
            prob = torch.sigmoid(mlp(torch.tensor((x-mean)/std))).numpy().ravel()
        chosen = np.argsort(-prob, kind='stable')[:50]
        ids, sides, p = ix[chosen], side[chosen], prob[chosen]
        stocks[t] = (ids, sides, p)
        allocation_x[t] = x[chosen]
        valid = d['valid'][t, ids] & np.isfinite(d['y'][t, ids])
        ret = np.where(valid, np.expm1(d['y'][t, ids]), 0)
        utility = []
        for budget in (.6, .4):
            w = leg_weights(sides, p, budget)
            utility.append(np.dot(w*sides, ret) - w[sides == -1].sum() * .08 / 52)
        labels[t] = float(utility[0] > utility[1])
    build(np.r_[train, val])
    tr = [t for t in train if t in stocks]
    va = [t for t in val if t in stocks]
    amean = np.stack([allocation_x[t] for t in tr]).reshape(-1,129).mean(0)
    astd = np.stack([allocation_x[t] for t in tr]).reshape(-1,129).std(0) + 1e-6
    def alloc_data(indices):
        return (torch.tensor((np.stack([allocation_x[t] for t in indices])-amean)/astd),
                torch.tensor([labels[t] for t in indices], dtype=torch.float32))
    tx, ty = alloc_data(tr)
    vx, vy = alloc_data(va)
    torch.manual_seed(42)
    net = Allocation(True)
    net.output = torch.nn.Linear(32, 1)
    net, att_step, att_loss = select_model(net, tx, ty, vx, vy)
    print('attention selected', att_step, att_loss, flush=True)
    torch.save(dict(mlp=mlp.state_dict(), mlp_mean=mean, mlp_std=std, mlp_step=mlp_step,
                    attention=net.state_dict(), attention_mean=amean, attention_std=astd,
                    attention_step=att_step), OUT / 'selected_heads.pt')
    build(test)
    choices = {k: [] for k in ('global', 'fixed60', 'fixed40', 'attention', 'old40_10')}
    selected, predictions = [], []
    for t in test:
        if t not in stocks:
            continue
        ids, sides, p = stocks[t]
        raw = np.exp(p-p.max())
        choices['global'].append((t, ids, sides, raw/raw.sum()))
        for name, budget in (('fixed60', .6), ('fixed40', .4)):
            choices[name].append((t, ids, sides, leg_weights(sides, p, budget)))
        with torch.no_grad():
            score = net(torch.tensor((allocation_x[t][None]-amean)/astd)).item()
        choices['attention'].append((t, ids, sides, leg_weights(sides, p, .6 if score > 0 else .4)))
        predictions.append(dict(date=dates[t], score=score, truth=labels[t]))
        for ix, side, probability, weight in zip(ids, sides, p, raw/raw.sum()):
            selected.append(dict(date=dates[t], symbol=d['symbols'][ix], side=int(side),
                                 confidence=float(probability), weight=float(weight)))
        long, short = groups[t]
        ids0 = np.r_[long[:40], short[:10]]
        sides0 = np.r_[np.ones(min(40, len(long))), -np.ones(min(10, len(short)))]
        choices['old40_10'].append((t, ids0, sides0, leg_weights(sides0, np.zeros(len(ids0)), .5)))
    frames = {k: backtest(d, dates, v, k) for k, v in choices.items()}
    pd.concat(frames.values()).to_csv(OUT / 'weekly.csv', index=False)
    pd.DataFrame(selected).to_csv(OUT / 'stocks.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'attention_scores.csv', index=False)
    base = frames['fixed60'].set_index('date').net
    report = dict(mlp_step=mlp_step, mlp_val_bce=mlp_loss, attention_step=att_step,
                  attention_val_bce=att_loss, train_weeks=len(tr), val_weeks=len(va), arms={})
    for name, frame in frames.items():
        delta = frame.set_index('date').net - base
        report['arms'][name] = dict(**perf_summary(frame.net), delta_bps=float(delta.mean()*1e4),
                                   nw_t=newey_west_t(delta), avg_long=float(frame.long_weight.mean()),
                                   missing=int(frame.missing.sum()))
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    assert np.allclose(leg_weights(np.array([1, -1]), np.zeros(2), .6), [.6, .4])
    if not (OUT / 'features.npz').exists():
        extract()
    main()
