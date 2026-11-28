"""Sequential concept-bottleneck-style model (Lecture 6, CBM / SENN idea, unsupervised concepts):
   g_k: concept-group features -> concept score c_k (shallow LGBM, trained on quality)
   f  : linear model on [c_acidity, c_preservation, c_body, c_salinity] -> quality
   Prediction = sum_k w_k * c_k + b  -> each concept's contribution is explicit."""
import numpy as np, lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from common import CONCEPTS, SEED

G = dict(n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=80, reg_lambda=5,
         subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1)

class ConceptBottleneck:
    def __init__(self, concepts=CONCEPTS, inner=5, positive=True, seed=SEED, n_jobs=None):
        self.concepts, self.inner, self.positive, self.seed, self.n_jobs = concepts, inner, positive, seed, n_jobs
    def _g(self):
        extra = {} if self.n_jobs is None else {'n_jobs': self.n_jobs}   # None keeps LightGBM's default
        return lgb.LGBMRegressor(**{**G, 'random_state': self.seed, **extra})
    def fit(self, X, y):
        X = X.reset_index(drop=True); C = np.zeros((len(X), len(self.concepts)))
        # inner OOF concept scores so f is not trained on overfit concept scores
        for a, b in KFold(self.inner, shuffle=True, random_state=self.seed).split(X):
            for k, cols in enumerate(self.concepts.values()):
                C[b, k] = self._g().fit(X.loc[a, cols], y[a]).predict(X.loc[b, cols])
        self.g = [self._g().fit(X[cols], y) for cols in self.concepts.values()]
        self.f = LinearRegression(positive=self.positive).fit(C, y); return self
    def concept_scores(self, X):
        return np.column_stack([g.predict(X[c]) for g, c in zip(self.g, self.concepts.values())])
    def predict(self, X): return self.f.predict(self.concept_scores(X))
    def contributions(self, X):
        """Per-sample contribution w_k * c_k(x) of each concept (prediction = sum + intercept)."""
        return self.f.coef_ * self.concept_scores(X)
