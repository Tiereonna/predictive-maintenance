# Predictive Maintenance (minimum)

Classification pipeline on **simulated** industrial sensor data. Estimates
whether a machine fails within 7 days. Compares logistic regression to
XGBoost using precision, recall, F1, and ROC-AUC, and prints feature
importance so a score is not a black box.

This is a learning project, not a plant deployment. There is no real SCADA
feed and no FastAPI / React UI yet.

On some Macs XGBoost needs OpenMP (`brew install libomp`). If that library
is missing, `train.py` falls back to scikit-learn's `HistGradientBoostingClassifier`
— same idea (boosted trees), different package. Say whichever one actually ran.

## What to say in an interview

- **Data:** I generated it (`src/generate_data.py`). Failures are more likely
  when hours, vibration, temperature, and current are high. I planted that
  rule so I can check whether the model found it.
- **Why not accuracy:** Failures are the minority class. Predicting “never
  fails” looks accurate and is useless.
- **Precision:** of the machines I flagged, how many actually failed.
- **Recall:** of the machines that failed, how many I caught. Missing a
  failure (false negative) is the expensive mistake here.
- **F1:** balance of those two.
- **ROC-AUC:** ranking quality across thresholds, not just the 0.5 cutoff.
- **Importance:** XGBoost gain is “how much this feature helped splits.” It
  is not proof the sensor *causes* failure.

## Setup

```bash
cd ~/Documents/projects/predictive-maintenance
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

Expect a printed metrics table and `artifacts/metrics.csv`.

## Layout

```
src/generate_data.py   synthetic rows + planted failure rule
src/train.py           split, fit, metrics, importances
```

## Resume (after you can explain the script)

> Trained classification models on simulated industrial sensor data to
> estimate 7-day failure risk; compared logistic regression and XGBoost with
> precision, recall, F1, and ROC-AUC; inspected feature importance instead
> of treating the score as a black box.

Do not add FastAPI / React to the PDF until those files exist and you can
walk through them.
