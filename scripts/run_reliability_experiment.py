"""One exploratory reliability-screening round; freeze membership before test evaluation."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.data import dataset_arrays
from quant_mvp.model import PairRanker
from quant_mvp.engine import score_panel, train
from quant_mvp.long_only import long_only_backtest
from quant_mvp.reliability import calibrate, walk_forward_calibration, rank_reliability, direction_metrics


def predict_validation(cfg, checkpoint):
    data = pd.read_parquet(cfg['cache_path'])
    lookback = max(cfg['context_lengths'])
    X, y, dates, symbols, valid = dataset_arrays(data, lookback, cfg['n_features'], cfg['train_end'], return_valid=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PairRanker(cfg['n_features'], n_score_heads=cfg['n_score_heads'],
                      d_model=cfg['d_model'], embedding_dim=cfg['embedding_dim'],
                      n_layers=cfg['n_layers'], dropout=cfg.get('dropout', .1)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    val = score_panel(model, X, y, dates, cfg['train_end'], cfg['valid_end'], lookback, device, symbols, valid_y=valid)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return val


def main():
    source = Path('outputs/real_weekly_500_full_rankic_10ep')
    root = Path('data/reliable500_top150')
    out = Path('outputs/real_reliable500_top150_5ep')
    if root.exists() or out.exists():
        raise FileExistsError('Use a new experiment name; existing artifacts are preserved')
    cfg = yaml.safe_load((source/'config.yaml').read_text())
    old_metrics = json.loads((source/'metrics.json').read_text())
    assert old_metrics['best_epoch'] == 8
    root.mkdir(parents=True)
    val = predict_validation(cfg, source/'best.pt')
    val.to_parquet(root/'teacher_validation.parquet', index=False)
    oof = walk_forward_calibration(val, warmup=26)
    oof.to_parquet(root/'validation_direction_predictions.parquet', index=False)
    ranking = rank_reliability(oof)
    names = pd.read_csv(source/'universe.csv')
    ranking = ranking.merge(names, on='symbol', validate='one_to_one')
    ranking.to_csv(root/'reliability_ranking.csv', index=False)
    universe = ranking[ranking.selected].copy()
    universe.to_csv(root/'universe.csv', index=False)
    selected = set(universe.symbol)
    manifest = dict(method='validation_direction_reliability', teacher_epoch=8,
        teacher_checkpoint=str(source/'best.pt'),
        teacher_sha256=hashlib.sha256((source/'best.pt').read_bytes()).hexdigest(),
        validation_start=val.date.min(), validation_end=val.date.max(),
        calibration_warmup_weeks=26, scoring_start=oof.date.min(), scoring_end=oof.date.max(),
        ranking_rule='Minimum of two half-period Beta(5,5) shrunk balanced accuracies; whole-period shrinkage then symbol break ties',
        min_each_direction_per_half=5, selected_count=150, original_count=500,
        probability_rule='Pooled one-feature logistic calibration of raw score; threshold0.5; refit from preceding validation weeks only',
        flats='abs(log return)<=1e-8 and missing excluded from direction labels, retained in return backtests',
        caveat='Teacher epoch8 selected on the whole validation interval; only calibration is walk-forward. Membership is chosen on validation and student selection reuses validation. Test has been inspected in previous experiments; exploratory, not independent validation.',
        asof=[val.date.max()])
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    panel = pd.read_parquet(cfg['cache_path'])
    subset = panel[panel.symbol.isin(selected)]
    assert subset.symbol.nunique() == 150 and subset.date.nunique() == 520
    subset.to_parquet(root/'weekly.parquet', index=False)
    student_cfg = dict(cfg, n_stocks=150, epochs=5, cache_path=str(root/'weekly.parquet'),
        universe_path=str(root/'universe.csv'), universe_name='150 stocks selected by validation direction reliability',
        prepare_command='python scripts/run_reliability_experiment.py')
    Path('configs/real_reliable500_top150_5ep.yaml').write_text(yaml.safe_dump(student_cfg, sort_keys=False), encoding='utf-8')
    print('Membership frozen: 150 stocks selected by validation reliability. Starting 5-epoch training.', flush=True)
    train(student_cfg, out)
    for name in ['reliability_ranking.csv', 'validation_direction_predictions.parquet']:
        shutil.copy2(root/name, out/name)
    # Test is first read after membership is frozen and the student is trained.
    teacher_test = pd.read_parquet(source/'predictions.parquet')
    teacher_direction = calibrate(val, teacher_test)
    student_test = pd.read_parquet(out/'predictions.parquet')
    student_val = predict_validation(student_cfg, out/'best.pt')
    student_val.to_parquet(out/'validation_predictions.parquet', index=False)
    student_direction = calibrate(student_val, student_test)
    assert sorted(teacher_test.date.unique()) == sorted(student_test.date.unique())
    comparisons = []
    common_curve = None
    methods = [('teacher500', '原模型 · 全500股', teacher_direction),
               ('teacher150', '原模型 · 仅可靠150股', teacher_direction[teacher_direction.symbol.isin(selected)]),
               ('student150', '可靠150股 · 重训5轮', student_direction)]
    for key, label, predictions in methods:
        curve, result = long_only_backtest(predictions, 20, cfg['transaction_cost_bps'], 52)
        baseline = old_metrics['long_only']['benchmark']['final_equity']
        result['relative_to_original500'] = result['net']['final_equity']/baseline-1
        comparisons.append(dict(key=key, label=label, n_stocks=predictions.symbol.nunique(),
                                direction=direction_metrics(predictions), long_only=result))
        predictions.to_parquet(out/f'{key}_direction_predictions.parquet', index=False)
        curve.to_csv(out/f'{key}_curve.csv', index=False)
        keep = curve[['date', 'long_net_equity']].rename(columns={'long_net_equity': key})
        common_curve = keep if common_curve is None else common_curve.merge(keep, on='date', validate='one_to_one')
    common_curve.to_csv(out/'reliability_curves.csv', index=False)
    report = dict(method=manifest, comparisons=comparisons,
        validation_selection=[dict(label=label, **direction_metrics(predictions)) for label, predictions in
            [('全部500股', oof), ('选中150股', oof[oof.symbol.isin(selected)]), ('其余350股', oof[~oof.symbol.isin(selected)])]],
        test_unselected=direction_metrics(teacher_direction[~teacher_direction.symbol.isin(selected)]),
        original500_benchmark=old_metrics['long_only']['benchmark'],
        reliability_summary=universe[['reliability','shrunk_accuracy','stability_gap']].describe().to_dict())
    (out/'reliability_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(comparisons, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
