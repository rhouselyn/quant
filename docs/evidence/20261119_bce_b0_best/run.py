"""Paired native-best B0 checkpoints on the unchanged legacy allocation task."""
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

from attention_followup import LegPool
from quant_mvp.hierarchy import HierarchicalRanker
from quant_mvp.hierarchy_data import arrays
from quant_mvp.hierarchy_engine import make_batch
from replay_equal_weight_model_books import legacy_inputs, locked_history
from replay_longshort_prediction import CATALOG, TAPE

OUT = Path(__file__).resolve().parent
PARENT = Path('outputs/bce_old_roster_vectors_v1')
PROTOCOL = Path('docs/research_20261119_bce_b0_best_protocol.md')
STD = runpy.run_path('outputs/standard_bce_pool_v1/run_standard.py')
FIT = runpy.run_path('outputs/bce_freeze_steps_v1/run.py')['fit']
REPLAY = runpy.run_path('outputs/bce_equiv_init_v1/run.py')['replay']
HASHES = {}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def record(path, expected=None):
    p = Path(path).resolve()
    h = sha(p)
    if expected is not None:
        assert h == expected, ('changed input', str(p))
    HASHES[str(p)] = h
    return h


def digest(*values):
    h = hashlib.sha256()
    for v in values:
        h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def save(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf8')
    for attempt in range(20):
        try:
            tmp.replace(p)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def status(stage, **kw):
    save('state.json', dict(status=stage, pid=os.getpid(), updated_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(), **kw))


def assert_plain(model):
    assert model.cross_summary is model.market_film is model.cross_stock is None
    assert model.conditional_score is None and not hasattr(model, 'daily_encoder')
    gates = [p for k, p in model.named_parameters() if k.endswith(('attn_gate', 'ffn_gate'))]
    assert len(gates) > 0 and all(torch.count_nonzero(p).item() == 0 for p in gates)


def encode(model, window):
    with torch.inference_mode():
        return F.normalize(model.final_norm(model.encode(torch.from_numpy(window[None]))[0]), dim=-1).numpy()


def prepare(d, dates, days, roster, models):
    """Encode only requested stocks after proving the frozen zero-gate path."""
    torch.set_num_threads(2)
    symbols = pd.Index(d['symbols'])
    positions = {}
    for row in roster.itertuples():
        ids = symbols.get_indexer((row.longs + '|' + row.shorts).split('|'))
        assert len(ids) == 50 and len(set(ids)) == 50 and (ids >= 0).all()
        positions[(row.phase, pd.Timestamp(row.date), int(row.variant))] = ids
    assert len(positions) == 1599
    old_cfg = yaml.safe_load(Path('outputs/abl2000_base/config.yaml').read_text(encoding='utf8'))
    old_pre = json.loads(Path(models[0]['run'], 'preprocessing.json').read_text(encoding='utf8'))
    old_pre.setdefault('normalization', 'legacy_z')
    original = arrays(dict(old_cfg, frozen_preprocessing=old_pre))
    od = pd.DatetimeIndex(original[6])
    for name in ('cache_path', 'market_path', 'universe_path'):
        record(old_cfg[name])
    phases = {}
    for phase in days:
        directory = Path(yaml.safe_load(Path(models[0]['run'], 'config.yaml').read_text())['cache_path']).parent if phase == 'train' else STD['STANDARD'] / 'snapshots' / ('validation' if phase == 'validation' else 'history')
        cfg = yaml.safe_load(Path(models[0]['run'], 'config.yaml').read_text())
        opts, data = STD['dataset'](directory, Path(models[0]['run']), cfg)
        for name in ('weekly.parquet', 'market_features.csv', 'universe.csv', 'manifest.json'):
            record(directory / name)
        assert list(data[7]) == list(symbols) == list(original[7])
        phases[phase] = (opts, data, pd.DatetimeIndex(data[6]))
    coverage, checks, inventory = [], [], []
    for source in models:
        seed = source['encoder_seed']
        folder = Path(source['run'])
        cfg = yaml.safe_load((folder / 'config.yaml').read_text())
        metrics = json.loads((folder / 'metrics.json').read_text())
        epoch = metrics['best_epoch']
        log = pd.read_csv(folder / 'training_metrics.csv')
        column = 'val_long_net_sharpe_selection' if 'val_long_net_sharpe_selection' in log else 'val_long_net_sharpe'
        chosen = int(log.loc[log[column].idxmax(), 'epoch'])
        assert chosen == epoch and epoch == {42: 1, 43: 6, 44: 3}[seed]
        a = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        b = torch.load(folder / f'epoch_{epoch}.pt', map_location='cpu', weights_only=True)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        for name in ('config.yaml', 'metrics.json', 'training_metrics.csv', 'preprocessing.json', 'best.pt', f'epoch_{epoch}.pt'):
            record(folder / name)
        h = hashlib.sha256()
        for k in sorted(k for k in a if k.startswith(('temporal.encoder.', 'final_norm.'))):
            h.update(k.encode()); h.update(a[k].contiguous().numpy().tobytes())
        inventory.append(dict(**source, best_epoch=epoch, best_checkpoint_sha256=sha(folder / 'best.pt'), best_encoder_sha256=h.hexdigest(), original_validation_argmax_verified=True))
        model = HierarchicalRanker(cfg, original[8]).eval()
        model.chunk_size = 128
        model.load_state_dict(a)
        assert_plain(model)
        source_data = PARENT / f'B0_s{seed}_data.npz'
        record(source_data)
        with np.load(source_data) as q:
            control = {k: q[k].copy() for k in q.files}
        qrows, orows = {}, {}
        started = time.perf_counter()
        for phase, phase_days in days.items():
            opts, data, nd = phases[phase]
            for j, day in enumerate(phase_days):
                key = (phase, day, -1)
                names = set(positions[key])
                if phase == 'train':
                    for variant in range(3):
                        names.update(positions.get(('augmentation', day, variant), []))
                names = np.array(sorted(names))
                nt, ot = nd.get_loc(day), od.get_loc(day)
                np.testing.assert_array_equal(nd[nt-39:nt+1], od[ot-39:ot+1])
                new_window = data[0][nt-39:nt+1, names].transpose(1, 0, 2)
                old_window = original[0][ot-39:ot+1, names].transpose(1, 0, 2)
                changed = np.any(new_window != old_window, axis=(1, 2))
                if seed == 44:
                    p = Path('outputs/bce_fused_three_v1') / f'B0_s44_{phase}_vectors.npy'
                    if j == 0:
                        record(p)
                        cached = np.load(p, mmap_mode='r')
                    z = np.asarray(cached[j, names]).copy()
                else:
                    z = encode(model, new_window)
                oz = z.copy()
                if changed.any():
                    oz[changed] = encode(model, old_window[changed])
                qrows[key] = (names, z)
                orows[key] = (names, oz)
                coverage.append(dict(seed=seed, phase=phase, date=day, requested_stocks=len(names), changed_windows=int(changed.sum()), max_vector_change=float(np.abs(z-oz).max())))
                if j in (0, len(phase_days)//2):
                    for label, full_data, t, expected in [('Q1', data, nt, z), ('old', original, ot, oz)]:
                        aa = list(full_data); aa[4] = full_data[4].copy(); aa[4][t] = d['available'][dates.get_loc(day)]
                        assert aa[4][t, names].all()
                        with torch.inference_mode():
                            full = model(*make_batch(tuple(aa), [int(t)], cfg, 'cpu'))[1][0, names].numpy()
                        np.testing.assert_allclose(full, expected, atol=2e-6, rtol=2e-5)
                        checks.append(dict(seed=seed, phase=phase, input_source=label, date=day, max_vector_error=float(np.abs(full-expected).max())))
                if j % 100 == 0:
                    status('encoding_cpu', seed=seed, phase=phase, row=j, total=len(phase_days))
                    print('Encoded', seed, phase, j, len(phase_days), 'seconds', round(time.perf_counter()-started, 1), flush=True)
        def feature(which, phase, day, variant=-1):
            names, z = which[('train' if phase == 'augmentation' else phase, day, -1)]
            ids = positions[(phase, day, variant)]
            return np.column_stack((z[np.searchsorted(names, ids)], np.repeat([1., -1.], [40, 10]))).astype(np.float32)
        for family, which in [('best', qrows), ('rawold', orows)]:
            content = {k: control[k] for k in ('y', 'vy', 'aug_y', 'aug_w')}
            for phase, k in [('train', 'base'), ('validation', 'vx'), ('test', 'hx')]:
                content[k] = np.stack([feature(which, phase, day) for day in days[phase]])
            ax = [feature(which, 'augmentation', pd.Timestamp(row.date), int(row.variant)) for row in roster.loc[roster.phase.eq('augmentation')].itertuples()]
            content['aug_x'] = np.concatenate([content['base'], np.stack(ax)])
            if family == 'best' and seed == 44:
                for k in control:
                    np.testing.assert_array_equal(content[k], control[k])
            for k in ('base', 'vx', 'hx', 'aug_x'):
                assert np.isfinite(content[k]).all() and (np.linalg.norm(content[k][..., :128], axis=-1) > .9).all()
            p = OUT / f'{family}_s{seed}_data.npz'
            np.savez(p, **content); record(p)
        del model, qrows, orows, control
    pd.DataFrame(coverage).to_csv(OUT/'feature_source_changes.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT/'full_forward_checks.csv', index=False)
    save('model_inventory.json', dict(models=inventory))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'validation_lock.json').exists(), 'do not overwrite a locked batch'
    status('preparing')
    for p in (__file__, PROTOCOL, PARENT/'dates.json', PARENT/'model_inventory.json', PARENT/'common_baskets.csv', PARENT/'identity.csv', PARENT/'signals.csv', PARENT/'member_logits.csv', PARENT/'weekly.parquet', PARENT/'validation_lock.json',
              'outputs/bce_freeze_steps_v1/run.py', 'outputs/standard_bce_pool_v1/run_standard.py', 'outputs/bce_equiv_init_v1/run.py', 'outputs/bce_final_audit_v1/audit.py', 'scripts/attention_followup.py',
              'scripts/replay_equal_weight_model_books.py', 'scripts/eval_weekly_architecture.py', 'src/quant_mvp/hierarchy.py', 'src/quant_mvp/hierarchy_data.py', 'src/quant_mvp/hierarchy_engine.py', CATALOG, TAPE, TAPE.with_name('eligibility.parquet')):
        record(p)
    d, dates, _, _, _, tr = legacy_inputs()
    days = {k: pd.DatetimeIndex(v) for k, v in json.loads((PARENT/'dates.json').read_text()).items()}
    assert [len(v) for v in days.values()] == [473, 123, 295]
    np.testing.assert_array_equal(dates[tr], days['train'])
    save('dates.json', {k: v.astype(str).tolist() for k, v in days.items()})
    roster = pd.read_csv(PARENT/'common_baskets.csv', parse_dates=['date'])
    inventory = json.loads((PARENT/'model_inventory.json').read_text())['models']
    models = [x for x in inventory if x['family'] == 'B0']
    assert len(models) == 3
    signature = dict(files=HASHES.copy(), best={str(x['encoder_seed']): sha(Path(x['run'])/'best.pt') for x in models})
    if (OUT/'prepared.json').exists():
        cache = json.loads((OUT/'prepared.json').read_text())
        assert cache['signature'] == signature
        for p, h in cache['hashes'].items(): record(p, h)
        print('Reused verified CPU encoding cache', flush=True)
    else:
        prepare(d, dates, days, roster, models)
        save('prepared.json', dict(signature=signature, hashes=HASHES.copy()))
    torch.set_num_threads(4)
    FIT.__globals__['OUT'] = OUT
    FIT.__globals__['HASHES'] = HASHES
    FIT.__globals__['record'] = record
    model_info = json.loads((OUT/'model_inventory.json').read_text())['models']
    trace, snapshots, checks, selections, reproduction, identity = [], [], [], [], [], []
    for family, modes in [('e3', ('f10',)), ('best', ('f10', 'f30', 'free')), ('rawold', ('f10',))]:
        for seed in (42, 43, 44):
            p = PARENT/f'B0_s{seed}_data.npz' if family == 'e3' else OUT/f'{family}_s{seed}_data.npz'
            record(p)
            with np.load(p) as a:
                flat = a['base'].reshape(-1, 129); mu, sd = flat.mean(0), flat.std(0)+1e-6
                source = next(x for x in model_info if x['encoder_seed'] == seed)
                encoder_sha = source['encoder_sha256'] if family == 'e3' else source['best_encoder_sha256']
                for suffix in ('plain', 'fixed5'):
                    x, y, w = (a['base'], a['y'], np.ones(473)) if suffix == 'plain' else (a['aug_x'], a['aug_y'], a['aug_w'])
                    identity.append(dict(family=family, seed=seed, suffix=suffix, inputs=digest(x, a['vx']), labels=digest(y, a['vy']), weights=digest(w), normalization=digest(mu, sd)))
                    for mode in modes:
                        status('training_cpu', family=family, seed=seed, suffix=suffix, mode=mode, completed_heads=len(selections), planned_heads=30)
                        row = FIT(x, y, w, a['vx'], a['vy'], mu, sd, seed, family, suffix, mode, encoder_sha, trace, snapshots, checks)
                        selections.append(row)
                        if family == 'e3':
                            ref_path = PARENT/f'B0_{suffix}_s{seed}.pt'; record(ref_path)
                            ref = torch.load(ref_path, weights_only=True); now = torch.load(row['path'], weights_only=True)
                            err = max(float((v-ref['state'][k]).abs().max()) for k,v in now['state'].items())
                            assert err < 2e-6
                            np.testing.assert_array_equal(now['mu'], ref['mu']); np.testing.assert_array_equal(now['sd'], ref['sd'])
                            reproduction.append(dict(seed=seed, suffix=suffix, max_parameter_error=err))
    assert len(selections) == 30 and len(trace) == 3000
    ident = pd.DataFrame(identity)
    for suffix, g in ident.groupby('suffix'):
        assert g.labels.nunique() == g.weights.nunique() == 1
    ident.to_csv(OUT/'identity.csv', index=False)
    tt = pd.DataFrame(trace)
    prefix = []
    for (suffix, seed), g in tt.loc[tt.family.eq('best')].groupby(['suffix','head_seed']):
        q = {k: v.set_index('step') for k, v in g.groupby('mode')}
        for mode in ('f30', 'free'):
            np.testing.assert_array_equal(q['f10'].loc[:10, ['train_bce','val_bce']], q[mode].loc[:10, ['train_bce','val_bce']])
        np.testing.assert_array_equal(q['f30'].loc[:30, ['train_bce','val_bce']], q['free'].loc[:30, ['train_bce','val_bce']])
        prefix.append(dict(suffix=suffix, seed=seed, first10_exact=True, f30_free_first30_exact=True))
    for name, values in [('training_loss',trace),('parameter_snapshots',snapshots),('loss_checks',checks),('e3_parameter_reproduction',reproduction),('prefix_checks',prefix)]:
        pd.DataFrame(values).to_csv(OUT/(name+'.csv'), index=False)
    parts, member_validation = {}, {}
    def load_array(family, seed):
        return np.load(PARENT/f'B0_s{seed}_data.npz' if family == 'e3' else OUT/f'{family}_s{seed}_data.npz')
    for row in selections:
        cp = torch.load(row['path'], weights_only=True); head = LegPool(False).eval(); head.load_state_dict(cp['state'])
        with load_array(row['family'], row['encoder_seed']) as a:
            y = torch.tensor(a['vy'] > 0, dtype=torch.float32)
            with torch.no_grad(): pred = head((torch.from_numpy(a['vx'])-cp['mu'])/cp['sd']).flatten()
        value = float(F.binary_cross_entropy_with_logits(pred, y))
        np.testing.assert_allclose(value, row['val_bce'], atol=2e-6, rtol=0)
        parts.setdefault(row['arm'], []).append(pred)
        member_validation[row['arm']+'_s'+str(row['encoder_seed'])] = pred.numpy()
    validation = [dict(arm=k, ensemble_bce=float(F.binary_cross_entropy_with_logits(torch.stack(v).mean(0),y))) for k,v in parts.items()]
    choices = []
    for suffix in ('plain','fixed5'):
        candidates = [r for r in validation if r['arm'].startswith('best_'+suffix+'_')]
        chosen = min(candidates, key=lambda r:(r['ensemble_bce'], ['f10','f30','free'].index(r['arm'].split('_')[-1])))
        choices.append(dict(suffix=suffix, **chosen))
    pd.DataFrame(validation).to_csv(OUT/'validation_ensemble.csv',index=False)
    pd.DataFrame(member_validation,index=days['validation']).rename_axis('date').to_csv(OUT/'validation_member_logits.csv')
    save('validation_lock.json',dict(selections=selections,freeze_choices=choices,train_dates=days['train'].astype(str).tolist(),validation_dates=days['validation'].astype(str).tolist(),hashes=HASHES.copy()))
    status('locked_scoring',heads=30)
    print('Locked all 30 heads and validation freeze choices before test prediction', flush=True)
    test_members, signal_parts = {}, {}
    for row in selections:
        cp=torch.load(row['path'],weights_only=True);h=LegPool(False).eval();h.load_state_dict(cp['state'])
        with load_array(row['family'],row['encoder_seed']) as a:
            with torch.no_grad(): pred=h((torch.from_numpy(a['hx'])-cp['mu'])/cp['sd']).flatten().numpy()
        test_members[row['arm']+'_s'+str(row['encoder_seed'])]=pred
        signal_parts.setdefault(row['arm'],[]).append(pred)
    signals=pd.DataFrame({k:np.mean(v,axis=0) for k,v in signal_parts.items()},index=days['test']).rename_axis('date')
    for suffix in ('plain','fixed5'):
        for seed in (42,43):
            arm=f'swap{seed}_{suffix}_f10'
            signals[arm]=np.mean([test_members[f"{'best' if s==seed else 'e3'}_{suffix}_f10_s{s}"] for s in (42,43,44)],axis=0)
        np.testing.assert_array_equal(test_members[f'best_{suffix}_f10_s44'],test_members[f'e3_{suffix}_f10_s44'])
    parent=pd.read_csv(PARENT/'signals.csv',parse_dates=['date']).set_index('date')
    for suffix in ('plain','fixed5'):
        signals['old_'+suffix]=parent['old_'+suffix]
        np.testing.assert_allclose(signals['e3_'+suffix+'_f10'],parent['B0_'+suffix],atol=1e-6,rtol=0)
        assert signals['e3_'+suffix+'_f10'].gt(0).eq(parent['B0_'+suffix].gt(0)).all()
    assert len(signals.columns)==16
    signals.to_csv(OUT/'signals.csv');pd.DataFrame(test_members,index=days['test']).rename_axis('date').to_csv(OUT/'member_logits.csv')
    status('replaying',heads=30)
    frame,schedule,_=locked_history();schedule=schedule.loc[schedule.date.isin(days['test'])];assert len(schedule)==295
    f=frame.loc[frame.date.isin(days['test'])]
    scores=pd.concat([pd.DataFrame(dict(date=day,symbol=d['symbols'],score_long=d['score'][dates.get_loc(day)])) for day in days['test']],ignore_index=True)
    source=f.merge(scores,on=['date','symbol'],validate='one_to_one');source['score_short']=source.score_long
    metrics,weekly=REPLAY(source,schedule,signals)
    assert len(metrics)==64 and len(weekly)==18880
    np.testing.assert_allclose(weekly.net,weekly.long_net+weekly.short_net,atol=1e-12,rtol=0)
    ref=pd.read_parquet(PARENT/'weekly.parquet')
    controls=weekly.loc[weekly.arm.str.startswith(('old_','e3_'))].copy()
    controls['arm']=controls.arm.str.replace('e3_','B0_',regex=False).str.replace('_f10','',regex=False)
    q=controls.merge(ref,on=['arm','rule','fee','date'],validate='one_to_one');assert len(q)==4720
    np.testing.assert_allclose(q.net_x,q.net_y,atol=1e-12,rtol=0)
    metrics.to_csv(OUT/'metrics.csv',index=False);weekly.to_parquet(OUT/'weekly.parquet',index=False)
    for p,h in list(HASHES.items()): assert sha(p)==h,('changed input during run',p)
    save('inputs.json',HASHES)
    verify=dict(status='training_replay_complete',heads_trained=30,training_rows=3000,train_weeks=473,validation_weeks=123,test_weeks=295,signals=16,metric_rows=64,weekly_rows=18880,reference_weekly_rows_reproduced=4720,seed44_best_e3_exact=True,new_gpu_work=False)
    save('verification.json',verify);status('training_replay_complete',**{k:v for k,v in verify.items() if k!='status'})
    print(metrics.loc[metrics.rule.eq('full'),['arm','fee','annual_calendar','max_drawdown','sharpe']].to_string(index=False),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        status('failed',error=repr(exc))
        raise
