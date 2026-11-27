"""Shared utilities: data loading, features, QWK thresholds, CV."""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

SEED = 42
RAW = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
       'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

def load(path='.'):
    tr = pd.read_csv(f'{path}/train.csv'); te = pd.read_csv(f'{path}/test.csv')
    return tr, te, tr['quality'].astype(int).values

def add_features(df):
    """Domain-motivated features (wine chemistry)."""
    d = df[RAW].copy()
    d['total_acidity'] = d['fixed acidity'] + d['volatile acidity'] + d['citric acid']
    d['bound_so2'] = d['total sulfur dioxide'] - d['free sulfur dioxide']
    d['free_so2_ratio'] = d['free sulfur dioxide'] / d['total sulfur dioxide']
    d['alcohol_x_sulphates'] = d['alcohol'] * d['sulphates']
    d['alcohol_per_density'] = d['alcohol'] / d['density']
    d['volatile_per_fixed'] = d['volatile acidity'] / d['fixed acidity']
    return d

# Concept groups used by the concept-bottleneck model (wine-tasting notions)
CONCEPTS = {
    'acidity':      ['fixed acidity','volatile acidity','citric acid','pH'],
    'preservation': ['free sulfur dioxide','total sulfur dioxide','sulphates'],
    'body':         ['alcohol','density','residual sugar'],
    'salinity':     ['chlorides'],
}

qwk = lambda a, b: cohen_kappa_score(a, b, weights='quadratic')
T0 = np.array([3.5, 4.5, 5.5, 6.5, 7.5])
def apply_t(p, t): return np.digitize(p, np.sort(t)) + 3
def fit_t_nm(p, y):
    """Nelder-Mead from the naive cut points (used for submission v1; often stops in a local optimum)."""
    return np.sort(minimize(lambda t: -qwk(y, apply_t(p, t)), T0, method='Nelder-Mead').x)

def fit_t(p, y, iters=3, n_grid=80):
    """Nelder-Mead, then coordinate search: each cut point is grid-searched between its neighbours."""
    t = fit_t_nm(p, y)
    for _ in range(iters):
        for k in range(len(t)):
            lo = t[k - 1] if k else p.min(); hi = t[k + 1] if k < len(t) - 1 else p.max()
            cands = np.linspace(lo, hi, n_grid)[1:-1]
            vals = [qwk(y, apply_t(p, np.r_[t[:k], v, t[k + 1:]])) for v in cands]
            if max(vals) > qwk(y, apply_t(p, t)): t[k] = cands[int(np.argmax(vals))]
    return t

def folds(y, seed=SEED, k=5):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))

def cv_reg(make, X, y, Xte, seed=SEED):
    """Return OOF preds, averaged test preds, fitted models."""
    oof = np.zeros(len(y)); pte = np.zeros(len(Xte)); ms = []
    for a, b in folds(y, seed):
        m = make(); m.fit(X.iloc[a], y[a])
        oof[b] = m.predict(X.iloc[b]); pte += m.predict(Xte) / 5; ms.append(m)
    return oof, pte, ms

def score(oof, y):
    t = fit_t(oof, y); return qwk(y, apply_t(oof, t)), t
