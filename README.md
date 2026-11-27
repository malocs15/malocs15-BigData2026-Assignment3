# Assignment 3 — Explainable Wine Quality Inference

Big Data Analytics Techniques and Applications, Fall 2026.

## Setup
Python 3.12. Put `train.csv` and `test.csv` (from the Kaggle Data tab) in the repository root.
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Reproduce the submissions
```bash
.venv/bin/python src/sub_v1.py submissions/v1_ridge_lgbm_blend.csv   # v1: Ridge + LGBM blend
.venv/bin/python src/train.py                                         # v2: submissions/v2_cbm.csv, submissions/v2_blend.csv
```
All randomness is seeded (CV seeds 42, 7, 2026), so the outputs are byte-identical.

| File | Model | CV QWK | Public LB |
|---|---|---|---|
| `submissions/v1_ridge_lgbm_blend.csv` | Ridge + LightGBM, thresholds | 0.4138 | 0.4428 |
| `submissions/v2_cbm.csv` | Concept-bottleneck model (intrinsic) | 0.4160 | – |
| `submissions/v2_blend.csv` | CBM + EBM + LightGBM, thresholds | 0.4166 | – |

## Report figures and analysis
```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace assignment3.ipynb
```
Figures are written to `figures/`.

## Code
- `src/common.py`: data loading, QWK metric, threshold optimisation, CV
- `src/concept_model.py`: concept-bottleneck model (features → wine concepts → quality)
- `src/models.py`: model configurations
- `src/train.py`: final training and submission files
