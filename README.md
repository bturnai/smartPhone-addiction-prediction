# Phone Addiction Prediction

Predicting the probability that a user is addicted to their phone, from usage and demographic data. Kaggle Playground Series S6E8.

**Result: 0.96463 public leaderboard ROC-AUC**, up from a 0.91 logistic-regression baseline.

The repo contains the full analysis notebook and a containerized Gradio app that serves the trained model.

---

## The interesting part

The single biggest gain didn't come from the model or from tuning — it came from how missing values were handled.

| Setup | Local CV ROC-AUC | Public LB ROC-AUC |
|---|---|---|
| XGBoost, rows with missing values dropped | 0.968 | 0.94899 |
| XGBoost, full dataset, native missing-value handling | 0.959 | **0.96087** |
| XGBoost, full dataset, tuned | 0.963 | **0.96463** |

The two columns rank in opposite order. Locally, dropping incomplete rows looked better — but that model was also being *evaluated* on complete rows only, which is an easier test. On the real test set those rows come back, and you can't discard users just because a field is missing. The model has to learn on the same kind of data it will actually face.

Letting XGBoost handle missing values natively instead of imputing them was worth **0.012 AUC** — more than the entire hyperparameter search produced.

---

## Approach

**The output is a probability, not a label.** The competition scores ROC-AUC, but a ranking metric alone doesn't tell you whether the numbers themselves are trustworthy — divide every prediction by two and AUC doesn't move. So log loss, Brier score and calibration curves were tracked alongside it.

**Baseline: logistic regression.** ~0.91 ROC-AUC. Its calibration curve drifted off the diagonal in the 0.5–0.7 range, which was the first evidence that the underlying relationship isn't linear.

**XGBoost** confirmed it: 0.9588 CV AUC, better Brier score (0.074 vs 0.098) *and* better calibration — unusual, since boosted trees often trade calibration for ranking.

**SHAP** explained why. The effect of daily screen time is flat below ~6 hours, rises steeply between 6 and 10, then saturates above 10. That plateau–rise–plateau shape is exactly what a linear model can't represent.

**Hyperparameter tuning** (150 configurations, 3-fold CV, ~5 hours) improved the score by 0.004 and established something more useful: the parameter space is flat. The top 10 configurations differ by 0.00014, well below the fold-to-fold standard deviation of ~0.0005. Performance here is limited by the data, not the hyperparameters.

**Three engineered ratio features were tested and rejected** — the hypothesis was that addiction is about the *structure* of usage rather than the volume. CV went from 0.96087 to 0.95976, so they were dropped. SHAP later showed the model had already learned the distinction on its own: work/study hours and gaming hours push predictions down, while the screen-time features push them up.

---

## Live demo

A Gradio interface wraps the trained pipeline so predictions can be made interactively.

Fields can be left empty — the model handles missing values natively, which is the same property that produced the 0.012 AUC gain.

### Run it

```bash
docker build -t phone-addiction-app .
docker run -p 7860:7860 phone-addiction-app
```

Then open http://localhost:7860

No training needed. The container loads the serialized pipeline and starts serving in seconds.

**Apple Silicon note:** the image builds for the host architecture by default. To deploy the Mac-built image to an AMD64/x86 cloud server, use `docker buildx build --platform linux/amd64`.

---

## Repo contents

```
.
├── xgboost_phoneAddiction.ipynb   # full analysis: EDA, modelling, tuning, SHAP
├── app.py                          # Gradio interface + prediction logic
├── model_pipeline.joblib           # trained preprocessing + XGBoost pipeline
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── .gitignore
```

Preprocessing and the model live in a single scikit-learn `Pipeline`, so the app passes raw input straight through — no separate encoding step to keep in sync between training and serving.

The dataset is not included. Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e8) and place them in the project root to re-run the notebook.

---

## Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `SHAP` · `Gradio` · `Docker`

---

## Limitations

- **Training data covers ages 18–35 only.** Nothing here should be assumed to transfer to teenagers or older adults.
- **`addicted_label` comes with the dataset** and its methodology is unknown — likely self-reported. This is a risk score correlated with usage patterns, not a diagnosis.
- **The data may be synthetic.** The regular column structure and evenly distributed missingness suggest generated data, in which case the model may be learning the generating rule rather than human behaviour.
- **If this ever produced hard yes/no decisions**, the threshold would be an ethical choice rather than a statistical one — false positives and false negatives are not symmetric costs when the label is "addicted".
