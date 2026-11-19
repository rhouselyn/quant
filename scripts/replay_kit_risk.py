"""Replay the locked B0 long basket with two causal risk weighting rules."""
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from eval_weekly_architecture import BOOKS, pick_targets, replay
from evaluate_dataset_three_year import phase_proxy, score_members, score_pair, snapshot
from eval_dataset_experiments import metrics, publish
from kit_transfer_data import ROOT, STANDARD, load_source
from r1111_common import BOOTSTRAP_SEED, sha256


PARENT = ROOT/'outputs/kit_transfer_v1'
OUT = ROOT/'outputs/kit_risk_replay_v1'
PROTOCOL = ROOT/'docs/research_20261116_kit_risk_replay_protocol.md'
SPEC = BOOKS['L20']


def max_drawdown(levels):
    return np.max(1 - levels / np.maximum.accumulate(levels,axis=-1),axis=-1)


def path_risk(returns, horizon):
    blocks = np.lib.stride_tricks.sliding_window_view(returns,horizon,axis=0)
    risks = []
    for paths in blocks.swapaxes(0,1):
        complete = paths[np.isfinite(paths).all(axis=1)]
        levels = np.exp(np.column_stack([np.zeros(len(complete)),complete.cumsum(axis=-1)]))
        risks.append(-tail_mean(-max_drawdown(levels), .2) if len(complete)>=20 else np.nan)
    return risks


def tail_mean(values, fraction=.1):
    n = math.ceil(values.shape[-1]*fraction)
    return np.partition(values,n-1,axis=-1)[...,:n].mean(axis=-1)


def risk_weights(intended, signal):
    weights = intended.copy()
    ranks = signal.dropna().rank(ascending=False)
    total = sum(intended[(1,s)] for s in ranks.index)
    for symbol, rank in ranks.items():
        key = (1,symbol)
        weights[key] = .75*intended[key] + .25*total*rank/ranks.sum()
    return weights


def paired(candidate, control):
    n = len(candidate)
    assert n == len(control) and n >= 16
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    starts = rng.integers(0,n-7,size=(5000,math.ceil(n/8)))
    idx = (starts[...,None]+np.arange(8)).reshape(len(starts),-1)[:,:n]
    a,b = candidate[idx], control[idx]
    draws = ((a-b).mean(axis=1),tail_mean(a)-tail_mean(b))
    stats = {}
    for key,point,draw in zip(('mean','tail10'),((candidate-control).mean(),tail_mean(candidate)-tail_mean(control)),draws):
        lo,hi = np.quantile(draw,[.025,.975])*1e4
        stats.update({f'{key}_delta_bps':point*1e4,f'{key}_lo_bps':lo,f'{key}_hi_bps':hi})
    return stats


def aligned_drawdowns(ledger, events, opening, calendar):
    out = {}
    for day, g in events.loc[events.filled.gt(0)].groupby('date'):
        row = ledger.loc[day]
        start,end = calendar.get_indexer(pd.DatetimeIndex([row.entry_at,row.exit_at]).tz_localize(None).normalize())
        assert 0 <= start < end
        px = opening.iloc[start:end+1][g.symbol.to_numpy()].to_numpy()
        if not np.isfinite(px).all():
            continue
        moves = px / px[0] - 1
        np.testing.assert_allclose(moves[-1],g.asset_r,atol=1e-12,rtol=0)
        levels = 1 - row.trading_cost + moves @ g.filled.to_numpy()
        np.testing.assert_allclose(levels[-1]-1,row.net,atol=1e-12,rtol=0)
        out[day] = max_drawdown(np.r_[1.,levels])
    return pd.Series(out,dtype=float)


def main():
    eq = {(1,'a'):.05,(1,'b'):.05,(1,'c'):.05}
    tilted = risk_weights(eq,pd.Series({'a':.2,'b':.1,'c':np.nan}))
    assert tilted[(1,'a')]<.05<tilted[(1,'b')] and tilted[(1,'c')]==.05 and np.isclose(sum(tilted.values()),.15)
    assert np.isclose(max_drawdown(np.array([1.,1.25,1.])),.2)
    np.testing.assert_allclose(path_risk(np.tile([.02,-.03],30)[:,None],2),[1-np.exp(-.03)])
    daily,calendar,symbols,inputs = load_source()
    opening = daily.pivot(index='date',columns='symbol',values='open').reindex(index=calendar,columns=symbols)
    returns = np.log(opening).diff()
    masks = returns.rolling(60,min_periods=40).count().ge(40)
    direct_ewma = returns.ewm(halflife=10,min_periods=40,ignore_na=False).std().where(masks)
    val,history = (STANDARD/'snapshots'/p for p in ('validation','history'))
    tape = STANDARD.parent/'replay_inputs_full/raw_open.parquet'
    eligibility = tape.with_name('eligibility.parquet')
    catalog = PARENT/'catalog.json'
    record = next(r for r in json.loads(catalog.read_text(encoding='utf-8'))['members'] if r['arm']=='B0')
    paths = [Path(record[k]) for k in ('run','validation_scores','lock','test_scores')]
    proxies = {phase:phase_proxy(directory,snapshot(directory),tape,eligibility,phase=phase)
               for phase,directory in [('validation',val),('test',history)]}
    member, hashes = score_members([(*paths,record)],snapshot(val),snapshot(history),tape,proxies['validation'][2])
    lock,scores = member[record['id']]
    assert lock['long_epochs']==[1,2,3] and record['seed']==42
    risk_file = PARENT/'risk_b0_candidates_v2/risk_rows.parquet'
    risk = pd.read_parquet(risk_file,columns=['date','symbol','ewma','holding_sessions']).set_index(['date','symbol'])
    assert risk.index.is_unique
    archived = PARENT/'report/weekly.parquet'
    name = record['id']
    original = pd.read_parquet(archived).query('portfolio == @name and book == "L20"')
    metadata = {**inputs,**hashes[record['id']]}
    for path in [catalog,tape,eligibility,risk_file,archived,PROTOCOL,Path(__file__),Path(__file__).with_name('eval_weekly_architecture.py'),
                 val/'manifest.json',history/'manifest.json', paths[0]/'preprocessing.json',
                 paths[1].with_suffix('.json'),paths[3].with_suffix('.json')]:
        metadata[str(path)] = sha256(path)
    for path in [paths[1],paths[3]]:
        assert metadata[str(path)] == json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['scores_sha256']
    rows, comparisons, weekly, target_rows, yearly = [], [], [], [], []
    for phase,directory in [('validation',val),('test',history)]:
        frame,schedule,_ = proxies[phase]
        dates = pd.DatetimeIndex(schedule.date)
        assert len(dates)==(153 if phase=='validation' else 297)
        source = score_pair(frame,scores[phase]['long'],scores[phase]['short'],schedule)
        sessions = pd.read_parquet(directory/'sessions.parquet').set_index('date')
        by_arm = {k:{} for k in ('ewma','path')}
        for day,g in source.groupby('date'):
            intended = pick_targets(g,*SPEC)
            names = [s for _,s in intended]
            assert len(names)==20
            schedule_row = sessions.loc[day]
            end,entry,exit_ = calendar.get_indexer([schedule_row.last_session,schedule_row.entry_date,schedule_row.exit_date])
            assert 59<=end<entry<exit_
            h = exit_-entry
            old = risk.loc[day].loc[names]
            assert old.holding_sessions.eq(h).all()
            ewma = direct_ewma.iloc[end][names]
            np.testing.assert_allclose(ewma,old.ewma,atol=1e-12,rtol=0)
            path = pd.Series(path_risk(returns.iloc[end-59:end+1][names].to_numpy(),h),index=names)
            common = ewma.notna() & path.notna()
            for arm,signal in [('ewma',ewma),('path',path)]:
                by_arm[arm][day] = risk_weights(intended,signal.where(common))
                target_rows.extend(dict(phase=phase,date=day,symbol=s,arm=arm,eligible=bool(common[s]),
                                        risk=signal[s],weight=by_arm[arm][day][(1,s)]) for s in names)
        for fee,penalty in [(10,0.),(30,0.),(10,.2)]:
            books,intra = {},{}
            for arm in ('b0','ewma','path'):
                fn = None if arm=='b0' else lambda day, _: by_arm[arm][day]
                book, events = replay(source,dates,SPEC,cost_bps=fee,missing_penalty=penalty,reweight=fn)
                book = book.merge(schedule,on='date',validate='one_to_one').set_index('date')
                keys = events[['date','symbol','side']].assign(filled=events.filled.gt(0))
                if arm=='b0':
                    base_keys = keys
                    base = original.loc[original.phase.eq(phase)&original.trade_bps.eq(fee)&original.borrow_annual.eq(.0835)
                                        &original.missing_penalty.eq(penalty)].set_index('date').loc[dates]
                    assert len(base)==len(dates)
                    np.testing.assert_allclose(book[['net','turnover','trading_cost','long_w']],base[['net','turnover','trading_cost','long_w']],atol=1e-12,rtol=0)
                else:
                    pd.testing.assert_frame_equal(keys,base_keys)
                books[arm] = book
                if penalty==0:
                    intra[arm] = aligned_drawdowns(book,events,opening,calendar)
            if intra:
                drawdowns = pd.DataFrame(intra)
                assert drawdowns.notna().nunique(axis=1).eq(1).all()
            for arm,book in books.items():
                case = dict(phase=phase,trade_bps=fee,missing_penalty=penalty,arm=arm)
                path_dd = drawdowns[arm].dropna() if intra else pd.Series(dtype=float)
                rows.append(dict(**case,**metrics(book.reset_index()),tail10=float(tail_mean(book.net.to_numpy())),
                                 intrahold_weeks=len(path_dd),intrahold_dd=path_dd.mean(),
                                 intrahold_worst10=-tail_mean(-path_dd.to_numpy()) if len(path_dd) else np.nan))
                weekly.append(book.assign(**case,intrahold_dd=intra.get(arm)).reset_index())
                for year, group in book.groupby(book.index.year):
                    yearly.append(dict(**case,year=year,weeks=len(group),net_total=np.expm1(np.log1p(group.net).sum()),
                                       tail10=tail_mean(group.net.to_numpy())))
            for arm,control in [('ewma','b0'),('path','b0'),('path','ewma')]:
                assert books[arm].index.equals(books[control].index)
                comparisons.append(dict(phase=phase,trade_bps=fee,missing_penalty=penalty,arm=arm,control=control,
                                        **paired(books[arm].net.to_numpy(),books[control].net.to_numpy())))
    def save(stage):
        pd.DataFrame(rows).to_csv(stage/'metrics.csv',index=False)
        pd.DataFrame(comparisons).to_csv(stage/'paired.csv',index=False)
        pd.DataFrame(yearly).to_csv(stage/'yearly.csv',index=False)
        pd.concat(weekly,ignore_index=True).to_parquet(stage/'weekly.parquet',index=False)
        pd.DataFrame(target_rows).to_parquet(stage/'targets.parquet',index=False)
        (stage/'inputs.json').write_text(json.dumps(dict(files=metadata,python=sys.version,numpy=np.__version__,
                                                        pandas=pd.__version__,lock=lock),indent=2),encoding='utf-8')
    publish(OUT,save)
    print(OUT)


if __name__=='__main__':
    main()
