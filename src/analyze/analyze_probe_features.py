#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REG=['value','abs_value','current_delta','next_value','next_delta','second_diff','local_slope','local_volatility']

def load(prefix,label):
    z=np.load(prefix); m=pd.read_csv(prefix+'.metadata.csv'); return z,m,label

def split(meta,seed,train=.7):
    ids=meta.series_id.unique(); rng=np.random.default_rng(seed); rng.shuffle(ids); cut=int(len(ids)*train)
    tr=meta.series_id.isin(set(ids[:cut])).to_numpy(); te=~tr; return tr,te

def score_one(X,y,tr,te,target,pca_dim,seed):
    valid=np.isfinite(y); tr=tr&valid; te=te&valid
    if tr.sum()<20 or te.sum()<10:return None
    steps=[StandardScaler()]
    if pca_dim and pca_dim<X.shape[1]: steps.append(PCA(n_components=min(pca_dim,tr.sum()-1),random_state=seed))
    if target=='sign': steps.append(LogisticRegression(max_iter=2000,class_weight='balanced',random_state=seed)); model=make_pipeline(*steps); model.fit(X[tr],y[tr].astype(int)); pred=model.predict(X[te]); return {'score':balanced_accuracy_score(y[te].astype(int),pred),'metric':'balanced_accuracy','mae':np.nan}
    steps.append(Ridge(alpha=10.0)); model=make_pipeline(*steps); model.fit(X[tr],y[tr]); pred=model.predict(X[te]); return {'score':r2_score(y[te],pred),'metric':'r2','mae':mean_absolute_error(y[te],pred)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mamba_prefix',required=True); p.add_argument('--llama_prefix',required=True); p.add_argument('--output_dir',required=True)
    p.add_argument('--pca_dim',type=int,default=256); p.add_argument('--seeds',default='3407,3408,3409,3410,3411'); a=p.parse_args()
    mz,mm,_=load(a.mamba_prefix,'mamba'); lz,lm,_=load(a.llama_prefix,'llama')
    keys=['series_id','kind','position'];
    if not mm[keys].equals(lm[keys]): raise ValueError('Mamba/Llama metadata rows are not aligned')
    rows=[]; seeds=[int(s) for s in a.seeds.split(',')]
    for label,z in [('mamba',mz),('llama',lz)]:
        for name in z.files:
            layer=int(name.split('_')[0][1:]); state=name.split('_',1)[1]; X=z[name]
            for seed in seeds:
                tr,te=split(mm,seed)
                for target in REG+['sign']:
                    y=mm[target].to_numpy(float); r=score_one(X,y,tr,te,target,a.pca_dim,seed)
                    if r: rows.append({'model':label,'layer':layer,'state':state,'target':target,'seed':seed,**r})
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); raw=pd.DataFrame(rows); raw.to_csv(out/'probe_scores_all_seeds.csv',index=False)
    summary=raw.groupby(['model','layer','state','target','metric'],as_index=False).agg(score_mean=('score','mean'),score_std=('score','std'),mae_mean=('mae','mean'))
    summary.to_csv(out/'probe_scores_summary.csv',index=False)
    best=summary.loc[summary.groupby(['model','state','target'])['score_mean'].idxmax()].copy()
    pivot=best.pivot_table(index=['state','target'],columns='model',values='score_mean').reset_index()
    if {'mamba','llama'}.issubset(pivot.columns): pivot['mamba_minus_llama']=pivot.mamba-pivot.llama
    pivot.to_csv(out/'best_model_difference.csv',index=False)
    print(pivot.to_string(index=False)); print('Saved to',out)
if __name__=='__main__': main()
