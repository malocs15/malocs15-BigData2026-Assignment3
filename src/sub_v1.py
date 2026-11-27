"""Reproduces submission v1 (public LB 0.4428): Ridge + LGBM 50/50 blend, raw features, thresholds."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, 'src'); from common import *
tr, te, y = load(); X, Xte = tr[RAW], te[RAW]
P = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, random_state=SEED, verbose=-1)
o_r, p_r, _ = cv_reg(lambda: make_pipeline(StandardScaler(), Ridge()), X, y, Xte)
o_l, p_l, _ = cv_reg(lambda: lgb.LGBMRegressor(**P), X, y, Xte)
oof, pte = (o_r + o_l) / 2, (p_r + p_l) / 2; s, t = score(oof, y)
print(f'CV QWK {s:.4f}')
pd.DataFrame({'id': te.id, 'quality': apply_t(pte, t)}).to_csv(sys.argv[1] if len(sys.argv) > 1 else 'submissions/v1_repro.csv', index=False)
