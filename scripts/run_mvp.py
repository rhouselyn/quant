import argparse, pathlib, sys, yaml, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.engine import train

def main():
    ap=argparse.ArgumentParser(description='Run the quant MVP end to end (daily or weekly via config)')
    ap.add_argument('--config',default='configs/default.yaml'); ap.add_argument('--output',default=None)
    ap.add_argument('--smoke-test',action='store_true'); ap.add_argument('--synthetic',action='store_true')
    ap.add_argument('--real', action='store_true', help='download real A-share data via Baostock')
    args=ap.parse_args(); cfg=yaml.safe_load(open(args.config,encoding='utf-8'))
    if args.synthetic:
        cfg['provider']='synthetic'
        cfg['cache_path']='data/synthetic_weekly.parquet' if cfg.get('frequency')=='weekly' else 'data/synthetic_daily.parquet'
    if args.real: cfg['provider']='baostock'
    if args.smoke_test:
        cfg.update({'n_stocks':20,'n_days':180,'lookback':20,'context_lengths':[20],'train_end':110,'valid_end':145,'epochs':2,'pairs_per_day':32,'batch_size':64})
    run=args.output or ('outputs/smoke' if args.smoke_test else 'outputs/mvp_run')
    if cfg.get('model_type') == 'hierarchical_market':
        if args.smoke_test or args.synthetic or args.real:
            raise ValueError('Hierarchical training requires its prepared industry dataset; use tests/test_hierarchy.py for small checks')
        from quant_mvp.hierarchy_engine import train as train_hierarchy
        metrics=train_hierarchy(cfg,pathlib.Path(run))
    else:
        metrics=train(cfg,pathlib.Path(run))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
