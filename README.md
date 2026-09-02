<h1 align="center">Phone Addiction Prediction</h1>

<p align="center">
  Predicting how likely someone is to be addicted to their phone, from usage and demographic data.<br>
  <em>Kaggle Playground Series S6E8</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ROC--AUC-0.96463-brightgreen?style=for-the-badge" alt="ROC-AUC">
  <img src="https://img.shields.io/badge/rows-691k-blue?style=for-the-badge" alt="Rows">
  <img src="https://img.shields.io/badge/demo-live-orange?style=for-the-badge" alt="Demo">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-2.x-EE4C2C" alt="XGBoost">
  <img src="https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/SHAP-explainability-8A2BE2" alt="SHAP">
  <img src="https://img.shields.io/badge/Gradio-UI-FF7C00" alt="Gradio">
  <img src="https://img.shields.io/badge/Docker-containerized-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

---

Final score: 0.96463 ROC-AUC on the public leaderboard. The logistic regression I started with got 0.91.

The repo has the full analysis notebook and a containerized Gradio app that serves the trained model.

---

## What surprised me most

> [!IMPORTANT]
> The biggest single improvement had nothing to do with the model or the tuning. It came from how I handled missing values.

| Setup | Local CV ROC-AUC | Public LB ROC-AUC |
|---|:---:|:---:|
| XGBoost, rows with missing values dropped | 0.968 | 0.94899 |
| XGBoost, full dataset, native missing-value handling | 0.959 | **0.96087** |
| XGBoost, full dataset, tuned | 0.963 | **0.96463** |

Look at the two columns. They rank in opposite order.

Dropping the incomplete rows looked like the better option locally. But that model was also being tested on complete rows only, which is an easier exam. On the real test set those rows come back, and I can't just throw away users because one field is empty. The model has to learn on the same kind of data it will actually get.

> [!TIP]
> Letting XGBoost deal with the missing values itself, instead of filling them with the mean, was worth 0.012 AUC. That's more than the whole 5-hour hyperparameter search gave me.

---

## How I approached it

### The output is a probability

The competition scores ROC-AUC. But AUC only measures ranking: divide every prediction by two and the score doesn't budge, even though the numbers are now nonsense. Since the deliverable here is a percentage about a person, I tracked log loss, Brier score and calibration curves as well.

### Model progression

| Model | ROC-AUC | Brier | Worst calibration gap |
|---|:---:|:---:|:---:|
| Logistic regression | 0.9292 | 0.0977 | 6.4 pp |
| XGBoost | **0.9588** | **0.0740** | **3.7 pp** |

The baseline's calibration curve drifted off the diagonal between 0.5 and 0.7. That was my first hint that the relationship isn't linear.

XGBoost beat it on every measure, which I didn't expect. Boosted trees usually give you better ranking at the cost of worse calibration, so I was ready to fix that afterwards. Didn't need to.

### What SHAP showed

The effect of daily screen time turned out to be far from a straight line:

```
0-6 hours    ▁▁▁▁▁▁         flat, barely moves the prediction
6-10 hours   ▁▂▄▆█          steep rise, crossing zero around 7-8h
10+ hours    ██████         saturates, another hour adds almost nothing
```

A plateau, then a climb, then another plateau. Logistic regression can only draw a straight line through that, which is where most of the gap between the two models comes from.

### Hyperparameter tuning

150 configurations, 3-fold CV, about 5 hours of runtime. It bought me 0.004.

> [!NOTE]
> The more useful finding: the parameter space is flat. The top 10 configurations differ from each other by 0.00014, which is well below the fold-to-fold standard deviation of 0.0005. They're the same model. The data is the limit here, not the parameters.

### An experiment that didn't work

I thought addiction might be more about how someone uses the phone than how much, so I built three ratio features:

| Feature | Idea |
|---|---|
| `weekend / daily screen time` | Is the usage concentrated on weekends? |
| `screen time / app opens` | How long is one session on average |
| `notifications / app opens` | Opening the phone unprompted vs. reacting to alerts |

CV went from 0.96087 down to 0.95976. So I dropped them.

The funny part came later: SHAP showed the model had already figured out the distinction I was trying to hand it. Work/study hours and gaming hours push the prediction down, while the screen-time features push it up. It worked that out on its own, and my ratios only added noise on top.

---

## Try it

A Gradio interface wraps the trained pipeline, so you can put in numbers and get a prediction.

> [!TIP]
> You can leave fields empty. The model handles missing values natively, which is the same property that gave me the 0.012 AUC gain.

```bash
docker build -t phone-addiction-app .
docker run -p 7860:7860 phone-addiction-app
```

Then open http://localhost:7860

No training needed. The container loads the saved pipeline and starts serving in seconds.

> [!NOTE]
> On Apple Silicon the image builds for the host architecture. To deploy a Mac-built image to an AMD64 cloud server, use `docker buildx build --platform linux/amd64`.

---

## What's in the repo

```
.
├── phoneAddiction.ipynb   # the full analysis: EDA, modelling, tuning, SHAP
├── app.py                          # Gradio interface and prediction logic
├── model_pipeline.joblib           # trained preprocessing + XGBoost pipeline
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── .gitignore
```

The preprocessing and the model sit in one scikit-learn `Pipeline`, so the app hands it raw input and doesn't need to know anything about encoding. Nothing to keep in sync between training and serving.

> [!WARNING]
> The dataset isn't in the repo. Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e8) and put them in the project root if you want to re-run the notebook.

---

## Limitations

> [!CAUTION]
> This is a risk score based on usage patterns. It is not a diagnosis.

The training data only covers ages 18-35, so I wouldn't assume any of this holds for teenagers or older adults.

`addicted_label` came with the dataset and I don't know how it was produced, probably self-reported.

The data may well be synthetic. The column structure is very regular and the missing values are spread evenly, which is what generated data tends to look like. If that's the case, the model may be learning the generating rule rather than anything about people.

And if this ever fed into an actual yes/no decision, the threshold would be an ethical call, not a statistical one. When the label is "addicted", a false positive and a false negative don't cost the same thing.
