"""Compare cross-leg interaction, auxiliary risk supervision and stock context."""
import copy

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch import nn
from torch.nn import functional as F

from attention_followup import LegPool, compare, replay
from directional_2018_experiment import groups_for
from directional_three_seed import SEEDS, cache_path, standard_scores, stock_inputs
from leg_pool_ablation import OUT as PRIOR
from quant_mvp.long_only import long_only_backtest
from r1111_common import ROOT, perf_summary, write_json

OUT = PRIOR.parent / 'context_ablation_full_val'
PORTFOLIO_ARMS = ('mean_bce', 'cross_leg', 'quantile_aux', 'gap_mse', 'freeze_projection', 'freeze_representation')
FREEZE_STEP = 10


class InteractionOutput(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.interaction = nn.Parameter(torch.zeros(32))

    def forward(self, x):
        return self.linear(x) + ((x[:, :32] * x[:, 32:]) @ self.interaction)[:, None]


def portfolio_loss(arm, output, target, gap):
    if arm == 'gap_mse':
        return F.mse_loss(output[:, 0], gap)
    if arm == 'freeze_weighted':
        weight = gap.abs()
        return F.binary_cross_entropy_with_logits(output[:, 0], target, weight=weight/weight.mean())
    loss = F.binary_cross_entropy_with_logits(output[:, 0], target)
    if arm == 'freeze_magnitude':
        means = F.softplus(output[:, 1:]).clamp_min(1e-5)
        mean = torch.where(target.bool(), means[:, 0], means[:, 1])
        loss = loss + (mean.log() + gap.abs()/mean).mean()
    return loss


def fit_portfolio(x, gaps, train, selection, evaluation, seed, arms=PORTFOLIO_ARMS, out=OUT,
                  freeze_step=FREEZE_STEP):
    train_x = x[train]
    mean = train_x.reshape(-1, 129).mean(0)
    std = train_x.reshape(-1, 129).std(0) + 1e-6
    z = torch.from_numpy((x - mean) / std)
    target = torch.tensor(gaps > 0, dtype=torch.float32)
    scale = float(gaps[train].std())
    gap = torch.from_numpy(gaps / scale)
    estimates, diagnostics, traces = {}, [], []
    for arm in arms:
        torch.manual_seed(seed)
        model = LegPool(False)
        if arm == 'cross_leg':
            with torch.no_grad():
                initial = model(z[train[:3]])
            model.output = InteractionOutput(model.output)
            with torch.no_grad():
                assert torch.equal(model(z[train[:3]]), initial)
        elif arm == 'quantile_aux':
            model.output = JointOutput(model.output, nn.Linear(64, 1))
        elif arm == 'freeze_magnitude':
            magnitude = nn.Linear(64, 2)
            nn.init.zeros_(magnitude.weight)
            nn.init.zeros_(magnitude.bias)
            model.output = JointOutput(model.output, magnitude)
        parameters = model.parameters()
        if arm == 'slow_projection':
            parameters = [{'params': model.proj.parameters()},
                          {'params': [p for name, p in model.named_parameters() if not name.startswith('proj.')]}]
        optimizer = torch.optim.Adam(parameters, lr=.001, weight_decay=.01)
        best_loss = float('inf')
        frozen = {}
        for step in range(1, 101):
            if step == freeze_step + 1 and arm == 'slow_projection':
                optimizer.param_groups[0]['lr'] = .0001
            if step == freeze_step + 1 and arm.startswith('freeze_'):
                for name, parameter in model.named_parameters():
                    freeze = not name.startswith('output.') if arm == 'freeze_representation' else name.startswith('proj.')
                    if freeze:
                        frozen[name] = parameter.detach().clone()
                        parameter.requires_grad_(False)
            model.train()
            optimizer.zero_grad()
            pred = model(z[train])
            loss = portfolio_loss(arm, pred, target[train], gap[train])
            if arm == 'quantile_aux':
                error = gap[train] - pred[:, 1]
                loss = loss + .25 * torch.maximum(.25*error, -.75*error).mean()
            loss.backward()
            if arm == 'cross_leg' and step == 1:
                assert model.output.interaction.grad.abs().sum() > 0
            optimizer.step()
            model.eval()
            with torch.no_grad():
                prediction = model(z[selection])
                val_loss = portfolio_loss(arm, prediction, target[selection], gap[selection]).item()
            traces.append(dict(seed=seed, arm=arm, step=step, train_loss=loss.item(), val_loss=val_loss))
            if val_loss < best_loss:
                best_loss, best_step = val_loss, step
                state = copy.deepcopy(model.state_dict())
        assert all(torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters()
                   if name in frozen)
        model.load_state_dict(state)
        with torch.no_grad():
            estimates[arm] = model(z[evaluation]).numpy()
            if arm == 'gap_mse':
                estimates[arm] *= scale * 1e4
        torch.save(dict(model=state, mean=torch.from_numpy(mean), std=torch.from_numpy(std), gap_scale=scale),
                   out / f'{arm}_s{seed}.pt')
        diagnostics.append(dict(seed=seed, arm=arm, step=best_step, selection_loss=best_loss,
                                after_freeze=bool(frozen and best_step > freeze_step)))
        print(diagnostics[-1], flush=True)
    return estimates, diagnostics, traces


class JointOutput(nn.Module):
    def __init__(self, direction, quantile):
        super().__init__()
        self.direction = direction
        self.quantile = quantile

    def forward(self, x):
        return torch.cat((self.direction(x), self.quantile(x)), dim=1)


def fit_stock_context(z, data, dates, seed, stock_valid, stock_train, evaluation, industries):
    train_z = torch.from_numpy(z[stock_train][data['available'][stock_train]])
    center = train_z.mean(0)
    centered = train_z - center
    _, eigenvectors = torch.linalg.eigh(centered.T @ centered / len(centered))
    projected = ((torch.from_numpy(z) - center) @ eigenvectors[:, -32:]).numpy()
    del train_z, centered
    industry = np.eye(industries.max()+1, dtype=np.float32)[industries]

    def features(t):
        own = projected[t]
        mask = data['available'][t, :, None]
        known = np.where(mask, own, 0)
        sector_sum = (industry @ (industry.T @ known)) - known
        sector_count = (industry @ (industry.T @ mask.astype(np.float32))) - mask
        market_count = mask.sum() - mask
        sector = sector_sum / np.maximum(sector_count, 1)
        market = (known.sum(0) - known) / np.maximum(market_count, 1).astype(np.float32)
        return np.concatenate((own, own*sector, own*market), axis=1)

    designs, targets = [], []
    for t in stock_train:
        keep = stock_valid[t]
        if keep.sum() < 40:
            continue
        designs.append(features(t)[keep])
        targets.append((rankdata(data['y'][t, keep]) / (keep.sum()+1) - .5).astype(np.float32))
    design = torch.from_numpy(np.concatenate(designs))
    target = torch.from_numpy(np.concatenate(targets))
    mean, std = design.mean(0), design.std(0, correction=0) + 1e-6
    design = (design - mean) / std
    evaluated = torch.from_numpy(np.stack([features(t) for t in evaluation]))
    evaluated = (evaluated - mean) / std
    results = {}
    for arm, n in (('own_stock', 32), ('week_context', 96)):
        matrix = design[:, :n]
        gram = matrix.T @ matrix + .01 * len(matrix) * torch.eye(n)
        coef = torch.linalg.solve(gram, matrix.T @ target)
        results[arm] = (evaluated[..., :n] @ coef).numpy()
    print(f'stock context seed {seed}, fit rows {len(design)}', flush=True)
    return results


def mse_diagnostic(data, dates, baskets):
    weekly = pd.read_csv(PRIOR / 'weekly.csv', parse_dates=['date'])
    report = {}
    for split in ('val', 'test'):
        curves = {arm: weekly[(weekly.split == split) & (weekly.arm == arm)].reset_index(drop=True)
                  for arm in ('mean_bce', 'gap_mse')}
        control, mse = curves.values()
        assert control.date.equals(mse.date)
        ix = dates.get_indexer(control.date)
        shares = [float(f.long_weight.mean()) for f in (control, mse)]
        static = [replay(data, dates, baskets, ix, np.full(len(ix), w), 'static') for w in shares]
        delta = mse.net - control.net
        exposure = static[1].net - static[0].net
        report[split] = dict(diff=compare(mse, control), bce_avg_long=shares[0], mse_avg_long=shares[1],
            exposure_bps=float(exposure.mean()*1e4), residual_bps=float((delta-exposure).mean()*1e4),
            static_bce_sharpe=perf_summary(static[0].net)['sharpe'],
            static_mse_sharpe=perf_summary(static[1].net)['sharpe'],
            by_year_bps=(delta.groupby(mse.date.dt.year).mean()*1e4).to_dict(),
            largest_abs_week=mse.loc[delta.abs().idxmax(),'date'],
            diff_without_largest_week_bps=float(delta.drop(delta.abs().idxmax()).mean()*1e4))
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    stock_scores = []
    for seed in SEEDS:
        with np.load(cache_path(seed)) as cache:
            if seed == 42:
                data = {key: cache[key] for key in cache.files if key != 'z'}
            else:
                assert np.array_equal(data['dates'], cache['dates']) and np.array_equal(data['symbols'], cache['symbols'])
            stock_scores.append(standard_scores(cache['score'], data['available']))
    data['score'] = np.mean(stock_scores, axis=0)
    dates = pd.to_datetime(data['dates'])
    baskets = [(l[:40], s[:10]) for l, s in groups_for(data, dates)]
    indices = np.arange(len(dates))
    val0, test0 = [dates.searchsorted(d) for d in ('2018-01-05', '2021-01-08')]
    stock_valid = data['valid'] & data['available'] & np.isfinite(data['y'])
    gaps = np.full(len(dates), np.nan, dtype=np.float32)
    complete = np.zeros(len(dates), dtype=bool)
    for t, (long, short) in enumerate(baskets):
        if len(long) != 40 or len(short) != 10 or not data['valid'][t].any():
            continue
        ids = np.r_[long, short]
        valid = data['valid'][t, ids] & np.isfinite(data['y'][t, ids])
        r = np.expm1(np.where(valid, data['y'][t, ids], 0))
        gaps[t] = r[:40].mean() + r[40:].mean() + .08/52
        complete[t] = valid.all()
    train = indices[(indices < val0-2) & complete]
    splits = {'val': indices[(indices >= val0) & (indices < test0-2) & np.isfinite(gaps)],
              'test': indices[(indices >= test0) & np.isfinite(gaps)]}
    evaluation = np.concatenate(list(splits.values()))
    validation = splits['val'][complete[splits['val']]]
    assert (len(train), len(validation), len(splits['val']), len(splits['test'])) == (473, 123, 133, 283)
    industry_frame = pd.read_csv(ROOT / 'data/sw_580_observed_features_2000_fri_path30/universe.csv')
    industries = industry_frame.set_index('symbol').loc[data['symbols'], 'industry_id'].to_numpy(dtype=int)
    portfolio_scores = {arm: [] for arm in PORTFOLIO_ARMS}
    ranked = {arm: [] for arm in ('own_stock', 'week_context')}
    selections, traces = [], []
    for i, seed in enumerate(SEEDS):
        with np.load(cache_path(seed)) as cache:
            z = cache['z']
        x = np.zeros((len(dates), 50, 129), dtype=np.float32)
        for t in np.r_[train, evaluation]:
            x[t] = stock_inputs(z, t, baskets[t])[2]
        pred, rows, trace = fit_portfolio(x, gaps, train, validation, evaluation, seed)
        selections.extend(rows)
        traces.extend(trace)
        for arm in PORTFOLIO_ARMS:
            portfolio_scores[arm].append(pred[arm])
        stock = fit_stock_context(z, data, dates, seed, stock_valid, indices[:val0-2], evaluation, industries)
        for arm, score in stock.items():
            ranked[arm].append(.75*stock_scores[i][evaluation] + .25*standard_scores(score, data['available'][evaluation]))
        del z
    ranks = {arm: np.mean(parts, axis=0) for arm, parts in ranked.items()}
    ranks['frozen'] = data['score'][evaluation]
    ensemble = {arm: np.mean(parts, axis=0) for arm, parts in portfolio_scores.items()}
    report = dict(checkpoints=selections, selection_label_weeks=len(validation), portfolio={}, stock={})
    frames, stock_frames = [], []
    start = 0
    for split, ix in splits.items():
        sl = slice(start, start+len(ix))
        start += len(ix)
        portfolio = {}
        for arm, prediction in ensemble.items():
            pred = prediction[sl]
            score = pred[:, 0]
            frame = replay(data, dates, baskets, ix, np.where(score > 0, .7, .4), arm)
            portfolio[arm] = frame
            frames.append(frame.assign(split=split, score=score))
            keep = complete[ix]
            stats = dict(**perf_summary(frame.net), accuracy=float(((score>0)==(gaps[ix]>0))[keep].mean()),
                         long_weeks=int((score>0).sum()))
            if arm == 'quantile_aux':
                quantile = pred[keep,1] * gaps[train].std()
                actual = gaps[ix][keep]
                stats['lower_quartile_coverage'] = float((actual <= quantile).mean())
                error = actual - quantile
                constant_error = actual - np.quantile(gaps[train], .25)
                stats['quartile_pinball_bps'] = float(np.maximum(.25*error, -.75*error).mean()*1e4)
                stats['constant_pinball_bps'] = float(np.maximum(.25*constant_error, -.75*constant_error).mean()*1e4)
            report['portfolio'][f'{split}_{arm}'] = stats
        for arm in PORTFOLIO_ARMS[1:]:
            report['portfolio'][f'{split}_{arm}']['vs_mean_bce'] = compare(portfolio[arm], portfolio['mean_bce'])
        if split == 'val':
            report['val_sharpe_winner'] = max(PORTFOLIO_ARMS, key=lambda arm: report['portfolio'][f'val_{arm}']['sharpe'])
        stock_curves = {}
        for arm, score in ranks.items():
            pred = pd.DataFrame(dict(date=np.repeat(dates[ix], len(data['symbols'])),
                symbol=np.tile(data['symbols'], len(ix)), score=score[sl].ravel(),
                realized_return=np.where(stock_valid[ix], data['y'][ix], np.nan).ravel()))
            curve, _ = long_only_backtest(pred, top_n=40)
            rank_ic = []
            for row, t in enumerate(ix):
                keep = stock_valid[t] & np.isfinite(score[sl][row])
                rank_ic.append(np.corrcoef(rankdata(score[sl][row, keep]), rankdata(data['y'][t, keep]))[0,1])
            curve = curve[['date','long_net_return']].rename(columns={'long_net_return':'net'})
            assert len(curve) == len(ix)
            stock_curves[arm] = curve
            stock_frames.append(curve.assign(split=split, arm=arm, rank_ic=rank_ic))
            report['stock'][f'{split}_{arm}'] = dict(**perf_summary(curve.net), rank_ic=float(np.nanmean(rank_ic)))
        report['stock'][f'{split}_week_context']['vs_own_stock'] = compare(stock_curves['week_context'], stock_curves['own_stock'])
    report['previous_mse'] = mse_diagnostic(data, dates, baskets)
    pd.concat(frames, ignore_index=True).to_csv(OUT / 'portfolio_weekly.csv', index=False)
    pd.concat(stock_frames, ignore_index=True).to_csv(OUT / 'stock_weekly.csv', index=False)
    pd.DataFrame(traces).to_csv(OUT / 'learning_curves.csv', index=False)
    write_json(OUT / 'report.json', report)
    print(OUT, flush=True)


if __name__ == '__main__':
    main()
