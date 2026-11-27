"""Final training: 4 models x 3 CV seeds x 5 folds. Writes
   submissions/v2_cbm.csv          (intrinsic: concept-bottleneck model alone)
   submissions/v2_blend.csv        (CBM + EBM + LGBM average)
   and src/_oof.pkl (OOF/test predictions used by the notebook)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src'); from common import *; from models import MODELS
SEEDS = (42, 7, 2026)
tr, te, y = load(); X, Xte = tr[RAW], te[RAW]
OOF = {n: np.zeros((len(SEEDS), len(y))) for n in MODELS}; TE = {n: np.zeros(len(Xte)) for n in MODELS}
for i, s in enumerate(SEEDS):
    for n, mk in MODELS.items():
        for a, b in folds(y, s):
            m = mk(s).fit(X.iloc[a], y[a]); OOF[n][i, b] = m.predict(X.iloc[b]); TE[n] += m.predict(Xte) / (5 * len(SEEDS))
    print('seed', s, {n: round(score(OOF[n][i], y)[0], 4) for n in MODELS}, flush=True)
pd.to_pickle({'oof': OOF, 'te': TE, 'seeds': SEEDS}, 'src/_oof.pkl')
def write(name, members):
    oofs = [np.mean([OOF[m][i] for m in members], 0) for i in range(len(SEEDS))]
    t = np.mean([fit_t(o, y) for o in oofs], 0)               # thresholds averaged over seeds
    cv = [qwk(y, apply_t(o, t)) for o in oofs]
    pred = apply_t(np.mean([TE[m] for m in members], 0), t)
    pd.DataFrame({'id': te.id, 'quality': pred}).to_csv(f'submissions/{name}.csv', index=False)
    print(f'{name}: CV QWK {np.mean(cv):.4f} ± {np.std(cv):.4f}  thresholds {np.round(t, 3)}  dist {np.bincount(pred, minlength=9)[3:]}')
write('v2_cbm', ['CBM'])
write('v2_blend', ['CBM', 'EBM', 'LGBM'])
