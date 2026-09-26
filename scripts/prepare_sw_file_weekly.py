"""Audited stratified sample from user-provided Shenwan classification history."""
import sys,json,hashlib,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quant_mvp.data import _yahoo_one,aggregate_weekly,add_features

NAMES=dict(zip('11 22 23 24 27 28 33 34 35 36 37 41 42 43 45 46 48 49 51 61 62 63 64 65 71 72 73 74 75 76 77'.split(),
'农林牧渔 基础化工 钢铁 有色金属 电子 汽车 家用电器 食品饮料 纺织服饰 轻工制造 医药生物 公用事业 交通运输 房地产 商贸零售 社会服务 银行 非银金融 综合 建筑材料 建筑装饰 电力设备 机械设备 国防军工 计算机 传媒 通信 煤炭 石油石化 环保 美容护理'.split()))

def main():
 root=Path('data/sw_file_20');root.mkdir(exist_ok=True)
 if (root/'weekly.parquet').exists():raise FileExistsError('Completed cache exists')
 raw=root/'raw';raw.mkdir(exist_ok=True)
 src=Path('StockClassifyUse_stock.xls');shutil.copy2(src,root/src.name)
 d=pd.read_excel(src,dtype={'股票代码':str,'行业代码':str})
 d=d[d['计入日期']<=pd.Timestamp('2026-09-16')].sort_values(['股票代码','计入日期','更新日期']).drop_duplicates('股票代码',keep='last')
 d=d.rename(columns={'股票代码':'code','行业代码':'industry_code'})
 names=pd.read_csv('data/current_a_stock_names_20260916.csv',dtype=str)
 d=d.merge(names,on='code',how='left',validate='one_to_one')
 d['industry_l1_code']=d.industry_code.str[:2];d['industry_l1']=d.industry_l1_code.map(NAMES)
 d['reason']=np.where(d.name.isna(),'not currently listed',np.where(d.name.fillna('').str.contains('ST|退',case=False),'ST or delisting name',np.where(~d.code.str.match(r'^(000|001|002|003|300|301|600|601|603|605|688|689)\d{3}$'),'outside Shanghai/Shenzhen','')))
 d.to_csv(root/'classification_audit.csv',index=False,encoding='utf-8-sig')
 d=d[d.reason.eq('')].copy()
 if d.industry_l1.isna().any():raise ValueError('Unrecognized industry code in listed candidates')
 d['symbol']=d.code+np.where(d.code.str.startswith('6'),'.SS','.SZ')
 calendar=pd.date_range(end='2026-09-11',periods=520,freq='W-FRI');first,last=calendar[0],calendar[-1]
 def fetch(row):
  symbol=row['symbol'];daily=None
  for folder in [raw,Path('data/csi800/raw'),Path('data/yahoo_500_raw')]:
   f=folder/(symbol+'.parquet')
   if f.exists():daily=pd.read_parquet(f);break
  if daily is None:
   for _ in range(3):
    daily=_yahoo_one(symbol,'2014-01-01','2026-09-12')
    if daily is not None and not daily.empty:
     daily.to_parquet(raw/(symbol+'.parquet'),index=False);break
  reason='';w=None
  if daily is None or daily.empty:reason='download failed or no prices'
  else:
   daily=daily.sort_values('date');w=aggregate_weekly(daily);w=w[w.date.isin(calendar)]
   if daily.date.min()>first:reason='history starts after calendar start'
   elif daily.loc[daily.volume>0,'date'].max()<last-pd.Timedelta(days=7):reason='stale trading at history endpoint'
   elif len(w)<494:reason='fewer than 494 observed weeks'
   elif (w.volume<=0).mean()>.05:reason='excessive inactive weeks'
   elif (w.close<=0).any() or not np.isfinite(w.close).all():reason='invalid prices'
  return row,w,reason
 selected=[];frames=[];audit=[];counts=[]
 with ThreadPoolExecutor(max_workers=8) as pool:
  for code,name in NAMES.items():
   candidates=d[d.industry_l1_code.eq(code)].sort_values('code').sample(frac=1,random_state=42).to_dict('records');accepted=0
   for offset in range(0,len(candidates),24):
    for row,w,reason in pool.map(fetch,candidates[offset:offset+24]):
     chosen=not reason and accepted<20
     audit.append(dict(symbol=row['symbol'],name=row['name'],industry_l1=name,reason=reason,eligible=not reason,selected=chosen,observed_weeks=0 if w is None else len(w)))
     if chosen:selected.append(row);frames.append(w);accepted+=1
    if accepted>=20:break
   counts.append(dict(industry_l1_code=code,industry_l1=name,candidates=len(candidates),selected=accepted,shortfall=20-accepted))
   pd.DataFrame(audit).to_csv(root/'screening_audit.csv',index=False,encoding='utf-8-sig')
   pd.DataFrame(counts).to_csv(root/'industry_counts.csv',index=False,encoding='utf-8-sig')
   print(f'{code}: selected {accepted}/20 from {len(candidates)} candidates',flush=True)
 u=pd.DataFrame(selected);u.to_csv(root/'universe.csv',index=False,encoding='utf-8-sig')
 if not len(u) or any(r['selected']==0 for r in counts):raise ValueError('At least one industry has no eligible stocks')
 grid=pd.MultiIndex.from_product([sorted(u.symbol),calendar],names=['symbol','date'])
 panel=pd.concat(frames).set_index(['symbol','date']).reindex(grid).reset_index();df=add_features(panel);df['frequency']='weekly';df.to_parquet(root/'weekly.parquet',index=False)
 meta=dict(classification='Shenwan level 1 (user file)',source_file=src.name,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),classification_rule='Latest entry date then update timestamp per stock',asof=['2026-09-16'],file_latest_update=str(d['更新日期'].max()),seed=42,sampling='random permutation, first 20 eligible per industry; take all if fewer',n_stocks=len(u),n_industries=31,calendar_start=str(first.date()),calendar_end=str(last.date()),history_bias='Current listed non-ST stocks with long history; retrospective current classification',screen='Current Shanghai/Shenzhen listing, non-ST/delisted name, price history starts by calendar start, >=494 observed weeks, <=5% inactive weeks, trading within last7days of price endpoint')
 (root/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 c=yaml.safe_load(Path('configs/real_weekly_500_mixed_rankic.yaml').read_text());c.update(n_stocks=len(u),cache_path=str(root/'weekly.parquet'),universe_path=str(root/'universe.csv'),universe_name='User Shenwan industry stratified non-ST stocks',industry_classification='Shenwan level 1',encoder_chunk_size=1000,prepare_command='python scripts/prepare_sw_file_weekly.py')
 Path('configs/real_sw_file_20_weekly.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
 print(f'Saved {len(u)} stocks /31 industries /520 weeks',flush=True)
if __name__=='__main__':main()
