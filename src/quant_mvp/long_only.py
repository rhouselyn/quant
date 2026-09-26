"""Weekly long-only prototype; scores never use forward returns for selection."""
import numpy as np
import pandas as pd


def performance(returns, periods_per_year=52):
    r = np.asarray(returns, dtype=float)
    if not len(r):
        return dict(final_equity=1., total_return=0., annualized_return=0., sharpe=0., max_drawdown=0.)
    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1., equity])[1:]
    std = np.std(r, ddof=1) if len(r) > 1 else 0.
    return dict(final_equity=float(equity[-1]), total_return=float(equity[-1]-1),
                annualized_return=float(equity[-1]**(periods_per_year/len(r))-1),
                sharpe=float(np.mean(r)/std*np.sqrt(periods_per_year)) if std > 1e-12 else 0.,
                max_drawdown=float(np.min(equity/peak-1)))


def long_only_backtest(pred, top_n=20, cost_bps=10., periods_per_year=52):
    """Equal-weight Top-N vs universe. Rebalance cost uses buy+sell notional.

    Input returns are logs; convert to simple returns before aggregation.
    Missing returns are marked flat, counted and disclosed (no filtering of
    candidates by future return availability). No terminal liquidation.
    Close-to-close accounting is a research proxy, not next-open execution.
    """
    top_n = int(top_n)
    if top_n < 1 or cost_bps < 0:
        raise ValueError('top_n must be positive and costs nonnegative')
    previous, previous_benchmark = {}, {}
    rows = []
    for date, group in pred.groupby('date', sort=True):
        group = group[np.isfinite(group.score)].copy()
        if group.empty:
            continue
        group = group.sort_values(['score', 'symbol'], ascending=[False, True])
        # Entire closed/missing weeks should not trigger fabricated trading.
        if not np.isfinite(group.realized_return).any():
            continue
        names = group.head(top_n).symbol.tolist()
        weights = {s: 1/len(names) for s in names}
        bench = {s: 1/len(group) for s in group.symbol}
        observed = dict(zip(group.symbol, np.expm1(group.realized_return)))
        returns = {s: float(r) if np.isfinite(r) else 0. for s,r in observed.items()}
        def step(target, prev):
            buys = sum(max(target.get(s,0)-prev.get(s,0),0) for s in set(target)|set(prev))
            sells = sum(max(prev.get(s,0)-target.get(s,0),0) for s in set(target)|set(prev))
            turnover = buys+sells
            gross = sum(w*returns.get(s,0) for s,w in target.items())
            cost = turnover*cost_bps/10000
            net = (1-cost)*(1+gross)-1
            drift = {s: w*(1+returns.get(s,0))/(1+gross) for s,w in target.items()}
            return gross, net, turnover, drift
        gross,net,turnover,previous = step(weights,previous)
        _,benchmark,_,previous_benchmark = step(bench,previous_benchmark)
        missing = sum(not np.isfinite(observed.get(s,np.nan)) for s in names)
        rows.append(dict(date=date,long_gross_return=gross,long_net_return=net,
                         benchmark_net_return=benchmark,long_turnover=turnover,
                         long_missing_returns=missing,long_holdings=len(names)))
    curve = pd.DataFrame(rows)
    if curve.empty:
        raise ValueError('No valid periods for long-only evaluation')
    curve['long_net_equity']=(1+curve.long_net_return).cumprod()
    curve['benchmark_net_equity']=(1+curve.benchmark_net_return).cumprod()
    curve['long_relative_equity']=curve.long_net_equity/curve.benchmark_net_equity
    excess = curve.long_net_return-curve.benchmark_net_return
    metrics = dict(top_n=top_n, net=performance(curve.long_net_return,periods_per_year),
                   benchmark=performance(curve.benchmark_net_return,periods_per_year),
                   relative_return=float(curve.long_relative_equity.iloc[-1]-1),
                   excess_mean=float(excess.mean()),
                   mean_turnover=float(curve.long_turnover.mean()),
                   missing_held_returns=int(curve.long_missing_returns.sum()),
                   periods=len(curve),cost_bps=float(cost_bps),
                   execution='close_to_close_proxy',
                   missing_policy='mark_flat_and_count', terminal_liquidation=False)
    return curve,metrics
