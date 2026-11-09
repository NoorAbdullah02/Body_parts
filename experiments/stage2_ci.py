"""Patient-bootstrap CIs and paired differences for the Stage 2 OOF ensembles (uses the Kaggle arrays in results/kaggle_raw/stage2)."""
import numpy as np, pandas as pd, json
R="results/kaggle_raw/stage2"
folds=pd.read_csv(f"{R}/folds.csv"); lab=[c for c in folds.columns if c.startswith("label_")]
def load(d):
    z=np.load(f"{R}/{d}_ensemble_arrays.npz",allow_pickle=True); return z["oof"], z["pool_uid"]
def pred(p):
    y=(p>=0.5).astype(int); e=y.sum(1)==0; y[e,p[e].argmax(1)]=1; return y
def f1s(Y,P):
    tp=(Y*P).sum(1); den=Y.sum(1)+P.sum(1); return np.where(den>0,2*tp/np.maximum(den,1),1.0)
out={}; per={}
for d in ["s2a","s2b","s2u"]:
    oof,uid=load(d); f=folds.set_index("uid").loc[uid]; Y=f[lab].to_numpy(int); pat=f["patient_id"].astype(str).to_numpy()
    s=f1s(Y,pred(oof)); per[d]=s; out[d]={"sample_f1":float(s.mean())}
codes,inv=np.unique(pat,return_inverse=True); n=len(codes); idx=[np.where(inv==k)[0] for k in range(n)]
rng=np.random.default_rng(0); B=2000; boots={d:[] for d in per}
for _ in range(B):
    ch=rng.integers(0,n,n); ii=np.concatenate([idx[k] for k in ch])
    for d in per: boots[d].append(per[d][ii].mean())
b={d:np.array(v) for d,v in boots.items()}
for d in per: out[d]["ci"]=[float(np.quantile(b[d],.025)),float(np.quantile(b[d],.975))]
for a,c in [("s2u","s2a"),("s2b","s2a"),("s2u","s2b")]:
    diff=b[a]-b[c]; out[f"{a}-{c}"]={"diff":float(per[a].mean()-per[c].mean()),"ci":[float(np.quantile(diff,.025)),float(np.quantile(diff,.975))],"p_two_sided":float(2*min((diff<=0).mean(),(diff>=0).mean()))}
print(json.dumps(out,indent=1)); json.dump(out,open("results/stage2/stage2_ci.json","w"),indent=1)
