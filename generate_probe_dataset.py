#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', default='controlled_probe_series.csv')
    p.add_argument('--n_series', type=int, default=4000)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--seed', type=int, default=3407)
    a=p.parse_args()
    if a.length < 16: raise ValueError('length must be >= 16')
    rng=np.random.default_rng(a.seed); rows=[]
    kinds=['level','linear','quadratic','sine','step','ar1','random_walk','trend_noise']
    for sid in range(a.n_series):
        kind=kinds[sid % len(kinds)]; t=np.arange(a.length,dtype=np.float64)
        level=rng.uniform(-0.7,0.7); slope=rng.uniform(-0.025,0.025); noise=rng.uniform(0.0,0.035)
        if kind=='level': x=np.full(a.length,level)
        elif kind=='linear': x=level+slope*t
        elif kind=='quadratic': x=level+slope*t+rng.uniform(-4e-4,4e-4)*(t-a.length/2)**2
        elif kind=='sine': x=level+rng.uniform(0.1,0.45)*np.sin(2*np.pi*t/rng.integers(8,33)+rng.uniform(0,2*np.pi))
        elif kind=='step':
            x=np.full(a.length,level); cp=int(rng.integers(8,a.length-8)); x[cp:]+=rng.uniform(-0.5,0.5)
        elif kind=='ar1':
            x=np.empty(a.length); x[0]=level; phi=rng.uniform(0.35,0.95)
            for i in range(1,a.length): x[i]=level+phi*(x[i-1]-level)+rng.normal(0,0.08)
        elif kind=='random_walk': x=level+np.cumsum(rng.normal(0,0.045,a.length))
        else: x=level+slope*t+rng.normal(0,max(noise,0.01),a.length)
        if kind not in {'ar1','random_walk','trend_noise'} and noise>0: x=x+rng.normal(0,noise,a.length)
        x=np.clip(x,-0.95,0.95)
        rows.append({'series_id':sid,'kind':kind,'values':' '.join(f'{v:.8f}' for v in x)})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
    print(f'Saved {len(rows)} series to {out}')
if __name__=='__main__': main()
