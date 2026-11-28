"""Extra leaderboard submissions: the v3 models retrained with other CV seed triples.
   Same model and threshold method as v3 (CBM, CBM + LGBM, fit_t); only the random splits/seeds change,
   so CV quality is the same and the files differ only for borderline wines.
   Writes submissions/v4_<model>_s<seeds>.csv.
   Speed: each (seed, fold, model) fit runs in its own process with 1 LightGBM thread; the models are small,
   so running many single-threaded fits side by side is much faster than one multi-threaded fit at a time."""
import sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, 'src'); from common import *; from models import MODELS
from concept_model import ConceptBottleneck
SEED_SETS = [(1, 2, 3), (11, 12, 13), (21, 22, 23)]
tr, te, y = load(); X, Xte = tr[RAW], te[RAW]

def make(name, seed):
    if name == 'CBM': return ConceptBottleneck(seed=seed, n_jobs=1)
    return MODELS[name](seed).set_params(n_jobs=1)

def fit_one(name, seed, a, b):
    m = make(name, seed).fit(X.iloc[a], y[a])
    return name, seed, b, m.predict(X.iloc[b]), m.predict(Xte)

jobs = [(n, s, a, b) for seeds in SEED_SETS for s in seeds for n in ('CBM', 'LGBM') for a, b in folds(y, s)]
res = Parallel(n_jobs=-1, verbose=0)(delayed(fit_one)(*j) for j in jobs)
OOF, TE = {}, {}
for n, s, b, po, pt in res:
    OOF.setdefault((n, s), np.zeros(len(y)))[b] = po
    TE[(n, s)] = TE.get((n, s), 0) + pt / 5

for seeds in SEED_SETS:
    tag = 's' + '-'.join(map(str, seeds))
    for name, members in [('cbm', ['CBM']), ('cbm_lgbm', ['CBM', 'LGBM'])]:
        oofs = [np.mean([OOF[(m, s)] for m in members], 0) for s in seeds]
        t = np.mean([fit_t(o, y) for o in oofs], 0)
        cv = np.mean([qwk(y, apply_t(o, t)) for o in oofs])
        pred = apply_t(np.mean([TE[(m, s)] for m in members for s in seeds], 0), t)
        ref = pd.read_csv(f'submissions/v3_{name}.csv').quality.values
        pd.DataFrame({'id': te.id, 'quality': pred}).to_csv(f'submissions/v4_{name}_{tag}.csv', index=False)
        print(f'v4_{name}_{tag}: CV {cv:.4f}  differs from v3 on {(pred != ref).sum()} wines', flush=True)
