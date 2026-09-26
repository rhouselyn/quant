"""Fixed five-candidate exploratory sweep; choose using validation only."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.engine import train
from quant_mvp.reliability import calibrate, direction_metrics
from run_reliability_experiment import predict_validation


def main():
    root = Path('data/reliable500_weight_sweep')
    root.mkdir(exist_ok=False)
    source = Path('outputs/real_weekly_500_full_rankic_10ep')
    membership = Path('data/reliable500_top150/universe.csv')
    selected = set(pd.read_csv(membership).symbol)
    cfg = yaml.safe_load((source/'config.yaml').read_text())
    symbols = sorted(pd.read_parquet(cfg['cache_path']).symbol.unique())
    assert len(selected) == 150 and len(symbols) == 500 and selected <= set(symbols)
    multipliers = [1., 1.5, 2., 3., 5.]
    manifest = dict(multipliers=multipliers, epochs=5, seed=42,
        selected_count=150, total_stocks=500, membership=str(membership),
        membership_sha256=hashlib.sha256(membership.read_bytes()).hexdigest(),
        method='Normalized stock weighting in Score and full-cross-section soft RankIC moments; InfoNCE unchanged. All500 remain eligible for Top20 trades.',
        selection='Highest validation Top20 net Sharpe; ties prefer lower multiplier then earlier epoch.',
        caveat='Membership and teacher epoch used the same validation interval. Hyperparameter selection reuses it; test has been inspected before. Exploratory comparison, not independent confirmation.')
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    runs = []
    for factor in multipliers:
        tag = f'{factor:g}'.replace('.', 'p')
        name = f'real_reliable500_weight{tag}_5ep'
        out = Path('outputs')/name
        if out.exists():
            raise FileExistsError(out)
        weights = pd.DataFrame(dict(symbol=symbols, weight=[factor if s in selected else 1. for s in symbols], selected=[s in selected for s in symbols]))
        weight_path = root/f'weights_{tag}.csv'
        weights.to_csv(weight_path, index=False)
        run_cfg = dict(cfg, epochs=5, stock_weight_path=str(weight_path), stock_weight_multiplier=factor,
                       universe_name=f'Original500; validation-selected150 supervised weight {factor:g}x')
        Path(f'configs/{name}.yaml').write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding='utf-8')
        print(f'START multiplier={factor:g}', flush=True)
        train(run_cfg, out)
        shutil.copy2(source/'universe.csv', out/'universe.csv')
        shutil.copy2(root/'manifest.json', out/'weight_manifest.json')
        history = pd.read_csv(out/'training_metrics.csv')
        best = history.sort_values(['val_long_net_sharpe', 'epoch'], ascending=[False, True]).iloc[0]
        runs.append(dict(experiment=name, multiplier=factor, best_epoch=int(best.epoch),
                         val_sharpe=float(best.val_long_net_sharpe), val_return=float(best.val_long_net_return)))
        gc.collect()
        torch.cuda.empty_cache()
    # Freeze the recommendation before reading any test results across candidates.
    chosen = sorted(runs, key=lambda r: (-r['val_sharpe'], r['multiplier']))[0]
    (root/'selection.json').write_text(json.dumps(chosen, indent=2), encoding='utf-8')
    print('VALIDATION CHOICE '+json.dumps(chosen), flush=True)
    curves = None
    for run in runs:
        out = Path('outputs')/run['experiment']
        metrics = json.loads((out/'metrics.json').read_text(encoding='utf-8'))
        config = yaml.safe_load((out/'config.yaml').read_text())
        val = predict_validation(config, out/'best.pt')
        val.to_parquet(out/'validation_predictions.parquet', index=False)
        test = pd.read_parquet(out/'predictions.parquet')
        direction = calibrate(val, test)
        direction.to_parquet(out/'direction_predictions.parquet', index=False)
        run.update(direction=direction_metrics(direction),
                   selected_direction=direction_metrics(direction[direction.symbol.isin(selected)]),
                   remaining_direction=direction_metrics(direction[~direction.symbol.isin(selected)]),
                   long_only=metrics['long_only'], chosen=run['experiment']==chosen['experiment'])
        curve = pd.read_csv(out/'long_only_curve.csv')[['date','long_net_equity']].rename(columns={'long_net_equity':run['experiment']})
        curves = curve if curves is None else curves.merge(curve, on='date', validate='one_to_one')
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(manifest=manifest, chosen_experiment=chosen['experiment'], chosen_multiplier=chosen['multiplier'], runs=runs)
    (root/'weight_sweep_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    curves.to_csv(root/'weight_sweep_curves.csv', index=False)
    summary = pd.DataFrame([dict(multiplier=r['multiplier'],best_epoch=r['best_epoch'],val_sharpe=r['val_sharpe'],
        balanced_accuracy=r['direction']['balanced_accuracy'],selected_balanced_accuracy=r['selected_direction']['balanced_accuracy'],
        test_return=r['long_only']['net']['total_return'],test_sharpe=r['long_only']['net']['sharpe'],
        max_drawdown=r['long_only']['net']['max_drawdown'],chosen=r['chosen']) for r in runs])
    summary.to_csv(root/'comparison.csv', index=False)
    for run in runs:
        out = Path('outputs')/run['experiment']
        for file in ['weight_sweep_report.json','weight_sweep_curves.csv','comparison.csv']:
            shutil.copy2(root/file, out/file)
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
