"""Official latest CSI300 + CSI500 universe, with separate Yahoo history cache."""
import sys, json, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from quant_mvp.data import _yahoo_one, aggregate_weekly, add_features


def main():
    root=Path('data/csi800');root.mkdir(exist_ok=True)
    parts=[];sources=[]
    for index,count in [('000300',300),('000905',500)]:
        url=f'https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/{index}cons.xls'
        r=requests.get(url,timeout=40);r.raise_for_status()
        raw=root/f'{index}cons.xls';raw.write_bytes(r.content)
        d=pd.read_excel(BytesIO(r.content),dtype=str)
        assert len(d)==count
        d=d.iloc[:,:9];d.columns=['asof','index_code','index_name','index_en','code','name','name_en','exchange','exchange_en']
        d['code']=d.code.str.zfill(6)
        d['symbol']=d.code+np.where(d.code.str.startswith('6'),'.SS','.SZ')
        parts.append(d)
        sources.append(dict(url=url,sha256=hashlib.sha256(r.content).hexdigest(),rows=len(d)))
    universe=pd.concat(parts,ignore_index=True)
    assert universe.code.nunique()==800 and len(universe)==800
    # Preserve sourced labels by stock code when refreshing index membership.
    old_path=root/'universe.csv'
    if old_path.exists():
        old=pd.read_csv(old_path,dtype={'code':str})
        cols=[c for c in ['citic_l1','citic_status','citic_source','citic_asof'] if c in old]
        if cols:
            universe=universe.merge(old[['code']+cols],on='code',how='left',validate='one_to_one')
    if 'citic_l1' not in universe:universe['citic_l1']=''
    universe['citic_l1']=universe['citic_l1'].fillna('')
    if 'citic_status' not in universe:universe['citic_status']='pending_verified_source'
    universe.loc[universe.citic_l1.eq(''),'citic_status']='pending_verified_source'
    universe.to_csv(root/'universe.csv',index=False,encoding='utf-8-sig')
    (root/'manifest.json').write_text(json.dumps(dict(fetched_at=datetime.now(timezone.utc).isoformat(),sources=sources,asof=sorted(universe['asof'].unique()),classification='CITIC level 1' if universe.citic_l1.ne('').all() else 'pending CITIC level-one source',classified_stocks=int(universe.citic_l1.ne('').sum())),ensure_ascii=False,indent=2),encoding='utf-8')
    rawroot=root/'raw';rawroot.mkdir(exist_ok=True)
    calendar=pd.date_range(end='2026-09-11',periods=520,freq='W-FRI')
    # Use calendar weeks; no random replacement for new listings or missing bars.
    def fetch(symbol):
        path=rawroot/f'{symbol}.parquet'
        if path.exists():return pd.read_parquet(path)
        old=Path('data/yahoo_500_raw')/f'{symbol}.parquet'
        if old.exists():d=pd.read_parquet(old)
        else:
            d=None
            for _ in range(3):
                d=_yahoo_one(symbol,'2014-01-01','2026-09-12')
                if d is not None and not d.empty:break
        if d is not None and not d.empty:d.to_parquet(path,index=False)
        return d
    frames=[];missing=[];coverage=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(symbol,d) in enumerate(zip(universe.symbol,pool.map(fetch,universe.symbol)),1):
            if d is None or d.empty:missing.append(symbol)
            else:
                w=aggregate_weekly(d);w=w[w.date.isin(calendar)]
                frames.append(w);coverage.append(dict(symbol=symbol,first=str(d.date.min()),last=str(d.date.max()),observed_weeks=len(w)))
            if i%50==0:print(f'History {i}/800; missing={len(missing)}',flush=True)
    pd.DataFrame(coverage).to_csv(root/'coverage.csv',index=False)
    if missing:raise RuntimeError(f'No random/synthetic fallback; missing Yahoo symbols: {missing}')
    panel=pd.concat(frames,ignore_index=True)
    grid=pd.MultiIndex.from_product([sorted(universe.symbol),calendar],names=['symbol','date'])
    # Compute features on the actual calendar; never backfill before listing.
    panel=panel.set_index(['symbol','date']).reindex(grid).reset_index()
    df=add_features(panel);df['frequency']='weekly'
    df.to_parquet(root/'weekly_800_520.parquet',index=False)
    print('Saved official 800 stocks / 520 calendar weeks',flush=True)

if __name__=='__main__':main()
