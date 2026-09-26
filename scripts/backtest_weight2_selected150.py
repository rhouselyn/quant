"""Restrict the existing weighted model's trades; do not train or reselect it."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from quant_mvp.engine import backtest, rank_ic, plot_all
from quant_mvp.long_only import long_only_backtest


def main():
    source=Path('outputs/real_reliable500_weight2_8ep')
    out=Path('outputs/real_reliable500_weight2_selected150_8ep')
    out.mkdir(exist_ok=False)
    selected=pd.read_csv('data/reliable500_top150/universe.csv')
    symbols=set(selected.symbol)
    assert len(symbols)==150
    full=pd.read_parquet(source/'predictions.parquet')
    pred=full[full.symbol.isin(symbols)].copy()
    assert pred.symbol.nunique()==150 and set(pred.date)==set(full.date)
    # Recompute cross-sectional diagnostics inside the trading universe.
    ic={date:rank_ic(g.dropna(subset=['score','realized_return']).score.to_numpy(),
                     g.dropna(subset=['score','realized_return']).realized_return.to_numpy())
        for date,g in pred.groupby('date')}
    pred['rank_ic']=pred.date.map(ic)
    pred.to_parquet(out/'predictions.parquet',index=False)
    selected.to_csv(out/'universe.csv',index=False)
    cfg=yaml.safe_load((source/'config.yaml').read_text())
    cfg.update(evaluation_universe_path='data/reliable500_top150/universe.csv',
               evaluation_n_stocks=150,checkpoint_path=str(source/'best.pt'))
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    metrics=json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    full_long=metrics['long_only']
    metrics.update(backtest(pred,cfg,out))
    curve,long=long_only_backtest(pred,cfg['long_top_n'],cfg['transaction_cost_bps'],52)
    curve.to_csv(out/'long_only_curve.csv',index=False)
    holdings=pred[pred.date.isin(curve.date)].sort_values(['date','score','symbol'],ascending=[True,False,True]).groupby('date',sort=True).head(cfg['long_top_n'])
    holdings.to_csv(out/'holdings.csv',index=False)
    assert holdings.groupby('date').size().eq(20).all() and set(holdings.symbol)<=symbols
    metrics.update(long_only=long,n_stocks=150,training_n_stocks=500,
        evaluation_only=True,checkpoint_path=str(source/'best.pt'),
        checkpoint_sha256=hashlib.sha256((source/'best.pt').read_bytes()).hexdigest(),
        universe_name='Fixed selected150 trading universe; weighted model trained on500',
        scope_comparison=[dict(label='同模型／全部500股候选',**full_long),dict(label='同模型／仅选中150股候选',**long)])
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(source/'training_metrics.csv',out/'training_metrics.csv')
    plot_all(out,pd.read_csv(out/'training_metrics.csv'),pred)
    (out/'experiment_notes.md').write_text('使用原2倍加权模型best.pt（epoch8），仅过滤既有预测，不重新训练、标准化或选模。训练500股，回测仅固定150股内选Top20；基准亦为150股等权。RankIC与其他信号指标已按150股重新计算。验证选模记录仍为原500股口径。历史500股候选回测保留。名单来自原验证期，测试期已多次查看，属于探索结果。',encoding='utf-8')
    print(json.dumps(metrics['scope_comparison'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
