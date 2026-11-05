import argparse, json
from src.pipeline import ROOT,download,run
from src.visualize import visualize
from src.qa import audit_dataset,corporate_action_checks

parser=argparse.ArgumentParser()
parser.add_argument('--use-cache',action='store_true',help='Reuse existing downloaded raw market_data.parquet')
parser.add_argument('--frequency',choices=['monthly','quarterly','semiannual','all'],default='all')
args=parser.parse_args()
cfg=json.loads((ROOT/'config/pipeline.json').read_text())
if args.use_cache:
    import pandas as pd
    raw=pd.read_parquet(ROOT/'data/raw/market_data.parquet')
else:
    raw,log=download(cfg)
event_file=ROOT/'outputs'/'corporate_action_review.csv'
event_file.parent.mkdir(exist_ok=True)
corporate_action_checks(raw).to_csv(event_file,index=False)
for freq in cfg['frequencies'] if args.frequency=='all' else [args.frequency]:
    cfg['rebalance_frequency']=freq
    panel,screened,ready,audit=run(cfg,raw)
    visualize(screened,audit,ROOT/'outputs'/'figures'/freq)
    report=audit_dataset(raw,screened,ready,audit)
    (ROOT/'outputs'/f'qa_{freq}.json').write_text(json.dumps(report,indent=2,default=str))
    if not report['pass']: raise RuntimeError(f'QA failed for {freq}: {report["issues"]}')
    print(freq,'raw',len(raw),'panel',len(panel),'screened',len(screened),'ready',len(ready))
