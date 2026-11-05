"""Unified, time-respecting model interface. Train / validation / final refit / OOS."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .pipeline import ROOT, FEATURES

GRIDS = {
    'ridge': [{'alpha': a} for a in [0.1,1,10,100,1000,10000]],
    'lasso': [{'alpha': a} for a in [0.00001,0.0001,0.001,0.01,0.1]],
    'rf': [{'max_depth': d, 'min_samples_leaf': leaf} for d in [3,6] for leaf in [10,25]],
    'xgb': [{'max_depth': d, 'learning_rate': lr, 'n_estimators': 200} for d in [2,3] for lr in [0.03,0.1]],
}

def build_model(name, params, seed=42):
    if name == 'ridge': return Pipeline([('scale',StandardScaler()),('model',Ridge(**params))])
    if name == 'lasso': return Pipeline([('scale',StandardScaler()),('model',Lasso(**params,max_iter=20000,random_state=seed))])
    if name == 'rf': return RandomForestRegressor(n_estimators=250,max_features=0.7,n_jobs=-1,random_state=seed,**params)
    if name == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(objective='reg:squarederror',subsample=0.8,colsample_bytree=0.8,reg_lambda=10,tree_method='hist',random_state=seed,n_jobs=4,**params)
    raise ValueError(name)

def metrics(y, pred):
    return {'mse':float(mean_squared_error(y,pred)),'mae':float(mean_absolute_error(y,pred)),
            'r2':float(r2_score(y,pred))}

def run_model(frequency='monthly', model='ridge', seed=42):
    data_path=ROOT/'data/processed_ready'/frequency/'model_ready.parquet'
    df=pd.read_parquet(data_path).copy()
    if df.empty: raise RuntimeError('Empty model-ready dataset')
    df['decision_date']=pd.to_datetime(df.decision_date)
    df['label_end_date']=pd.to_datetime(df.label_end_date)
    if any(df.ticker.astype(str).str.startswith('SYN')): raise RuntimeError('Synthetic input refused')
    target='target_forward_return'
    tr=df[(df.split=='train') & (df.label_end_date<=pd.Timestamp('2020-12-31'))].copy()
    va=df[(df.split=='validation') & (df.label_end_date<=pd.Timestamp('2022-12-31'))].copy()
    te=df[df.split=='test'].copy()
    incomplete=te.groupby('decision_date').ticker.nunique()
    if (incomplete!=30).any():
        raise RuntimeError('Test date has fewer than 30 labeled candidates. Inspect screened_with_label_status; do not backtest an ex-post surviving subset.')
    if min(len(tr),len(va),len(te))==0: raise RuntimeError('Empty chronological split; inspect dates and label coverage')
    history=[]
    for params in GRIDS[model]:
        estimator=build_model(model,params,seed)
        estimator.fit(tr[FEATURES],tr[target])
        a=metrics(tr[target],estimator.predict(tr[FEATURES]))
        b=metrics(va[target],estimator.predict(va[FEATURES]))
        history.append({'model':model,'params':json.dumps(params,sort_keys=True),**{'train_'+k:v for k,v in a.items()},**{'validation_'+k:v for k,v in b.items()}})
    tuning=pd.DataFrame(history).sort_values(['validation_mse','params'])
    best=json.loads(tuning.iloc[0]['params'])
    final=build_model(model,best,seed)
    # Only labels realized before the 2023 test starts; no test rows used.
    refit=df[(df.decision_date<pd.Timestamp('2023-01-01'))&(df.label_end_date<=pd.Timestamp('2022-12-31'))]
    final.fit(refit[FEATURES],refit[target])
    te['prediction']=final.predict(te[FEATURES])
    te['model']=model
    te['pred_rank']=te.groupby('decision_date')['prediction'].rank(method='first',ascending=False)
    test_metrics=metrics(te[target],te.prediction)
    out=ROOT/'outputs'/'models'/frequency/model;out.mkdir(parents=True,exist_ok=True)
    tuning.to_csv(out/'training_validation_grid.csv',index=False)
    te.to_parquet(out/'test_predictions.parquet',index=False)
    te[['decision_date','ticker','model','prediction','pred_rank','target_forward_return','execution_date','label_end_date']].to_csv(out/'test_predictions.csv',index=False)
    (out/'model_summary.json').write_text(json.dumps({'best_params':best,'test_metrics':test_metrics,'train_rows':len(tr),'validation_rows':len(va),'refit_rows':len(refit),'test_rows':len(te),'seed':seed,'primary_target':target},indent=2))
    return te,tuning,final
