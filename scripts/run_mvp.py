import argparse, pathlib, sys, yaml, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from quant_mvp.engine import train

def main():
    ap=argparse.ArgumentParser(description='Run the quant MVP end to end (daily or weekly via config)')
    ap.add_argument('--config',default='configs/default.yaml'); ap.add_argument('--output',default=None)
    ap.add_argument('--smoke-test',action='store_true'); ap.add_argument('--synthetic',action='store_true')
    ap.add_argument('--real', action='store_true', help='download real A-share data via Baostock')
    args=ap.parse_args(); cfg=yaml.safe_load(open(args.config,encoding='utf-8'))
    if args.synthetic: cfg['provider']='synthetic'
    if args.real: cfg['provider']='baostock'
    if args.smoke_test:
        cfg.update({'n_stocks':20,'n_days':180,'lookback':20,'train_end':110,'valid_end':145,'epochs':2,'pairs_per_day':32,'batch_size':64})
    run=args.output or ('outputs/smoke' if args.smoke_test else 'outputs/mvp_run')
    metrics=train(cfg,pathlib.Path(run)); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
