# predictive-maintenance

Will this machine fail in the next 7 days?

Two parts:

1. Fake sensor rows + a small FastAPI page that scores one reading.  
   Demo: https://predictive-maintenance-8hqf.onrender.com  
   Free Render box, so the first load after idle can take ~30s.
2. Real [Backblaze hard drive stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data). Failures are rare. Use these numbers, not the toy ~0.80 ROC-AUC.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python src/train.py
uvicorn app:app --app-dir src --reload
```

http://127.0.0.1:8000

If XGBoost is missing (needs `libomp` on some Macs) the script falls back to `HistGradientBoostingClassifier`.

## toy metrics (generated data)

```
logistic               ~0.80 roc-auc, ~71% recall
HistGradientBoosting   about the same
```

About 43% of the fake rows are labeled fail, so accuracy would look fine even if the model was lazy.

## backblaze (real data)

```bash
python src/etl_backblaze.py --path data/backblaze/days
python src/train_backblaze.py
```

Last run: 16 days from 2013, 359727 rows. Split by date so test days are after train days.

```
logistic               precision 0.012  recall 0.387  f1 0.024  roc_auc 0.731
HistGradientBoosting   precision 0.007  recall 0.419  f1 0.014  roc_auc 0.689
```

Logistic won on ROC-AUC. Precision is low because almost every drive is healthy.

`artifacts/` is gitignored.
