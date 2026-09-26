"""Validate and attach a sourced CITIC level-one mapping (never infer from names)."""
import argparse,json
from pathlib import Path
import pandas as pd

CITIC_L1 = ['石油石化','煤炭','有色金属','电力及公用事业','钢铁','基础化工',
            '建筑','建材','轻工制造','机械','电力设备及新能源','国防军工','汽车',
            '商贸零售','消费者服务','家电','纺织服装','医药','食品饮料','农林牧渔',
            '银行','非银行金融','房地产','交通运输','电子','通信','计算机','传媒','综合','综合金融']

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mapping',required=True,help='CSV: code,citic_l1 (or Tushare ts_code,l1_name)')
    ap.add_argument('--source',required=True)
    ap.add_argument('--asof',required=True)
    args=ap.parse_args()
    path=Path('data/csi800/universe.csv')
    u=pd.read_csv(path,dtype={'code':str})
    m=pd.read_csv(args.mapping,dtype=str).rename(columns={'ts_code':'code','l1_name':'citic_l1'})
    m['code']=m.code.str.split('.').str[0].str.zfill(6)
    m=m[m.code.isin(u.code)]
    if m.code.duplicated().any() or set(m.code)!=set(u.code):
        raise ValueError('Mapping must cover all 800 constituent codes exactly once')
    if not m.citic_l1.isin(CITIC_L1).all():
        raise ValueError('Unknown CITIC level-one labels; verify taxonomy rather than substituting another system')
    u=u.drop(columns=['citic_l1','citic_status'],errors='ignore').merge(m[['code','citic_l1']],on='code',validate='one_to_one')
    u['citic_status']='sourced';u['citic_source']=args.source;u['citic_asof']=args.asof
    u.to_csv(path,index=False,encoding='utf-8-sig')
    manifest_path=path.with_name('manifest.json');meta=json.loads(manifest_path.read_text(encoding='utf-8'))
    meta.update(classification='CITIC level 1',industry_source=args.source,industry_asof=args.asof)
    manifest_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(u.citic_l1.value_counts().reindex(CITIC_L1,fill_value=0))

if __name__=='__main__':main()
