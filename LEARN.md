# Learn this before it goes on the resume

Run `python src/train.py`. Then close Cursor and answer these out loud.
If you cannot, do not add the project to the PDF yet.

1. Where did the data come from? (Generated. Not a factory.)
2. What does `failed_within_7_days = 1` mean?
3. Why is accuracy a bad headline metric here?
4. Precision vs recall in one sentence each, using machines that fail.
5. Which error is worse for a plant: false positive or false negative? Why?
6. What is F1 doing?
7. What is ROC-AUC measuring that F1 is not?
8. Why fit logistic regression at all if you have boosting?
9. Why `class_weight="balanced"` / `scale_pos_weight`?
10. Why can feature importance not be used as "this sensor causes failure"?

Then open `src/generate_data.py` and point to the planted `logit` line.
Then open `src/train.py` and point to `train_test_split` and `metrics_dict`.

Rewrite `metrics_dict` on paper (or a blank file) without looking. If that
works, you can defend the minimum version in an intern interview.

Optional local demo (after training): `uvicorn app:app --app-dir src --reload`
then score one row at `http://127.0.0.1:8000`. That page is localhost only,
not a deployed product. Do not put “FastAPI dashboard” on a resume until you
can explain the metrics above without looking.
