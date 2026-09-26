"""Train fixed no-InfoNCE/H2/H5 trials; freeze selection on validation before tests."""
import json
from pathlib import Path
import sys
import shutil
import pandas as pd
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quant_mvp.hierarchy_engine import train,evaluate,write_json

TRIALS=['off','h2','h5']


def main():
    root=Path('outputs/hierarchical_580_comparison');root.mkdir(exist_ok=True)
    rows=[]
    for key in TRIALS:
        cfg=yaml.safe_load(Path(f'configs/hierarchical_580_{key}.yaml').read_text())
        out=Path(f'outputs/real_hierarchical_580_{key}_10ep')
        write_json(root/'progress.json',dict(status='training',current=key,completed=rows))
        if out.exists():
            old_cfg=yaml.safe_load((out/'config.yaml').read_text())
            if cfg!=old_cfg:raise ValueError(f'Config differs from existing run {out}')
            result=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
            if result.get('status') not in ['validation_complete','complete']:
                raise RuntimeError(f'Incomplete run requires explicit recovery: {out}')
        else:
            result=train(cfg,out,defer_test=True)
        rows.append(dict(method=key,run=out.name,best_epoch=result['best_epoch'],
            validation_sharpe=result['selection_value'],temporal_weight=cfg['temporal_weight'],horizon=cfg['temporal_horizon']))
        pd.DataFrame(rows).to_csv(root/'validation_comparison.csv',index=False)
    winner=max(rows,key=lambda r:r['validation_sharpe'])['method']
    write_json(root/'selection.json',dict(selected=winner,criterion='highest validation Top20 net Sharpe; fixed trial order breaks ties',
        rows=rows,test_consulted=False,common_training_horizon=5))
    # Only after the architecture/loss choice is frozen do we read test results.
    curves=[]
    for row in rows:
        out=Path('outputs')/row['run']
        write_json(root/'progress.json',dict(status='test_evaluation',current=row['method'],selected=winner))
        result=evaluate(out)
        row.update(rank_ic=result['rank_ic_mean'],**result['long_only']['net'],
                   turnover=result['long_only']['mean_turnover'],periods=result['long_only']['periods'])
        curve=pd.read_csv(out/'long_only_curve.csv')
        curves.append(curve[['date','long_net_equity']].rename(columns={'long_net_equity':row['method']}))
    table=pd.DataFrame(rows);table.to_csv(root/'comparison.csv',index=False)
    combined=curves[0]
    for curve in curves[1:]:combined=combined.merge(curve,on='date',validate='one_to_one')
    combined.to_csv(root/'curves.csv',index=False)
    report=dict(selected=winner,rows=rows,note='Same corrected580 data, market token, LGLGL architecture, seed42, 10 epochs. No historical500 comparison. Test explored after validation choice frozen; no causal claim from one seed.')
    write_json(root/'comparison.json',report)
    shutil.copy2(f'configs/hierarchical_580_{winner}.yaml','configs/default.yaml')
    import plotly.graph_objects as go
    fig=go.Figure([go.Scatter(x=combined.date,y=combined[key],name=key) for key in TRIALS])
    fig.update_layout(title='580股：关闭 / 两周 / 五周InfoNCE · Top20净值',template='plotly_white',height=500)
    html='<html lang="zh"><meta charset="utf-8"><style>body{font-family:system-ui;max-width:1400px;margin:30px auto;padding:20px}table{border-collapse:collapse;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ddd}</style><h1>行业层次注意力＋一个市场token</h1>'
    html+=f'<p>验证集选定：{winner}。off=关闭；h2=未来两周，权重0.1；h5=未来五周，权重0.1。共同保留5周训练边界，同股票池、数据、种子、10轮及费用。</p>'
    html+=table.to_html(index=False,float_format=lambda v:f'{v:.4f}')+fig.to_html(full_html=False,include_plotlyjs=True)
    html+='<p>Score为下一周收益头尾20%等权，RankIC为全量0.5；InfoNCE仅作用于个股时序表示和独立投影头，key为EMA编码的未来窗口。日历、原始价格与标签保留；修复60周特征，加入市场状态输入，因此不能将与旧模型的差异单独归因于注意力。行业分类为当前快照回溯。单种子探索结果，不保证泛化。</p></html>'
    (root/'report.html').write_text(html,encoding='utf-8')
    write_json(root/'progress.json',dict(status='complete',selected=winner,rows=rows))
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':main()
