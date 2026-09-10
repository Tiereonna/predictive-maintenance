# Predictive Maintenance (local demo)

Classification on **simulated** industrial sensor data: will this machine fail
within 7 days? Compares logistic regression to a boosted tree (XGBoost, or
scikit-learn HGB if XGBoost’s native library is missing) using precision,
recall, F1, and ROC-AUC, then prints feature importance.

This is a learning project. There is no real plant feed and **nothing is
deployed**. A small FastAPI page on your laptop can score one synthetic
reading after you train.

## Local demo UI

1. `python src/train.py` — writes `artifacts/models.joblib` and `artifacts/metrics.csv`
2. `uvicorn app:app --app-dir src --reload`
3. Open `http://127.0.0.1:8000`

The form POSTs to `/api/predict`. If training has not been run, scoring fails
until you do step 1. Do not describe this as a production or hosted app.

On some Macs XGBoost needs OpenMP (`brew install libomp`). If that is missing,
training falls back to `HistGradientBoostingClassifier`. Say whichever model
actually printed.

## Setup

```bash
cd ~/Documents/projects/predictive-maintenance
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
uvicorn app:app --app-dir src --reload
```

## Layout

```
src/generate_data.py   synthetic rows + planted failure rule
src/train.py           split, fit, metrics, importances, save models
src/app.py             local FastAPI demo
static/index.html      one-page form (localhost only)
LEARN.md               questions to answer before listing this on a resume
```

Generated files under `artifacts/` are gitignored.
