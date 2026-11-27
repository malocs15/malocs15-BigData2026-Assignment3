"""Model factories used by train.py and the notebook."""
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from interpret.glassbox import ExplainableBoostingRegressor
from concept_model import ConceptBottleneck

LGB = dict(n_estimators=1500, learning_rate=0.01, num_leaves=7, min_child_samples=50, reg_lambda=5,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
MODELS = {
    'Ridge': lambda s: make_pipeline(StandardScaler(), Ridge()),
    'LGBM':  lambda s: lgb.LGBMRegressor(**LGB, random_state=s),
    'EBM':   lambda s: ExplainableBoostingRegressor(random_state=s, interactions=10),
    'CBM':   lambda s: ConceptBottleneck(positive=True, seed=s),
}
