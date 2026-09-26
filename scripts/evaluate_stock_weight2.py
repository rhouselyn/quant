"""Report the fixed 2x experiment against existing results; never trains a baseline."""
import json
from pathlib import Path
import shutil
import sys
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from quant_mvp.reliability import calibrate, direction_metrics
from run_reliability_experiment import predict_validation


def main():
    out = Path('outputs/real_reliable500_weight2_8ep')
    source = Path('outputs/real_weekly_500_full_rankic_10ep')
    cfg = yaml.safe_load((out/'config.yaml').read_text())
    selected = set(pd.read_csv('data/reliable500_top150/universe.csv').symbol)
    assert cfg['epochs'] == 8 and cfg['stock_weight_multiplier'] == 2
    metrics = json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    old = json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    validation = predict_validation(cfg, out/'best.pt')
    validation.to_parquet(out/'validation_predictions.parquet', index=False)
    direction = calibrate(validation, pd.read_parquet(out/'predictions.parquet'))
    direction.to_parquet(out/'direction_predictions.parquet', index=False)
    original = pd.read_parquet('outputs/real_reliable500_top150_5ep/teacher500_direction_predictions.parquet')
    assert set(original.symbol) == set(direction.symbol) and set(original.date) == set(direction.date)
    rows = []
    for label, pred, m in [('原500股历史模型（未重训）', original, old), ('150股2倍权重／全部500股训练', direction, metrics)]:
        rows.append(dict(label=label, epochs=m['epochs'], best_epoch=m['best_epoch'],
            direction=direction_metrics(pred), selected=direction_metrics(pred[pred.symbol.isin(selected)]),
            remaining=direction_metrics(pred[~pred.symbol.isin(selected)]), net=m['long_only']['net']))
    report=dict(rows=rows, multiplier=2, selected_count=len(selected),
        caveat='2倍由用户指定。历史模型训练10轮选中epoch8，本轮训练8轮；未重新训练1倍。150股名单和选模复用了验证期，测试期此前也已查看，属于探索对照。')
    (out/'stock_weight_evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(source/'universe.csv', out/'universe.csv')
    summary=pd.DataFrame([dict(method=r['label'],best_epoch=r['best_epoch'],balanced_accuracy=r['direction']['balanced_accuracy'],
        selected_accuracy=r['selected']['balanced_accuracy'],remaining_accuracy=r['remaining']['balanced_accuracy'],
        total_return=r['net']['total_return'],sharpe=r['net']['sharpe'],max_drawdown=r['net']['max_drawdown']) for r in rows])
    summary.to_csv(out/'stock_weight_comparison.csv', index=False)
    print(summary.to_string(index=False))


if __name__=='__main__':
    main()
