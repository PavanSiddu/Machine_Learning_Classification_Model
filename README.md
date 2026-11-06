# Spam Email Detection with From-Scratch Classifiers

| | |
|---|---|
| **GitHub Repository** | `https://github.com/PavanSiddu/Machine_Learning_Classification_Model` |
| **Live Streamlit App** | `https://machinelearningclassificationmodel-bypavan.streamlit.app` |

---

## a. Problem Statement

Unsolicited bulk e-mail (spam) wastes time and is a common vehicle for phishing and fraud. The goal of this project is to build a **binary classifier** that decides whether an e-mail is **spam (1)** or **not spam (0)** using word/character frequency statistics extracted from the e-mail body.

Five classical classification algorithms are implemented **from scratch using only NumPy** (no scikit-learn or any other built-in classifier):

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour Classifier
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine (linear, soft-margin)

All models are trained on the same training split, evaluated on the same held-out test split with six metrics (Accuracy, AUC, Precision, Recall, F1, MCC – also implemented from scratch), saved to disk, and served through an interactive Streamlit web application that evaluates the saved models on an uploaded test CSV **without retraining**.

---

## b. Dataset Description

| Property | Value |
|---|---|
| Name | **Spambase** |
| Source | UCI Machine Learning Repository – https://archive.ics.uci.edu/dataset/94/spambase |
| Task | Binary classification (`is_spam`: 1 = spam, 0 = not spam) |
| Instances | 4,601 (1,813 spam / 2,788 not spam) |
| Features | 57 continuous, non-negative numeric features |
| Missing values | None |

**Feature groups**

| Group | Count | Meaning |
|---|---|---|
| `word_freq_<WORD>` | 48 | Percentage of words in the e-mail that match `<WORD>` (e.g. `free`, `money`, `remove`, `hp`, `george`, `meeting`) |
| `char_freq_<CHAR>` | 6 | Percentage of characters that are `;` `(` `[` `!` `$` `#` (stored as `semicolon`, `parenthesis`, `bracket`, `exclamation`, `dollar`, `hash`) |
| `capital_run_length_average` | 1 | Average length of uninterrupted runs of capital letters |
| `capital_run_length_longest` | 1 | Length of the longest run of capital letters |
| `capital_run_length_total` | 1 | Total number of capital letters in the e-mail |

**Pre-processing (implemented in [model/preprocessing.py](model/preprocessing.py))**

1. **Duplicate removal** – 391 exact duplicate rows were dropped (4,601 → 4,210 rows: 1,679 spam / 2,531 not spam) so the same e-mail cannot appear in both train and test sets.
2. **Stratified 80/20 split** (seed `2026`) – 3,368 training rows and **842 test rows** (336 spam / 506 not spam). The test split is saved as [test_data.csv](test_data.csv).
3. **Log transform + standardisation** – every feature is transformed with `log(1 + x)` and then z-scored using the training mean/std. The features are heavily right-skewed (e.g. `capital_run_length_total` reaches ~15,000), so the log step keeps distance- and gradient-based models from being dominated by a few extreme values. The fitted scaler is saved as `model/scaler.pkl`.

---

## c. GitHub Repository Link

`https://github.com/PavanSiddu/Machine_Learning_Classification_Model`

Repository structure:

```
project-folder/
│-- app.py                     # Streamlit application
│-- requirements.txt           # Python dependencies
│-- README.md                  # This file
│-- test_data.csv              # Held-out test split (57 features + is_spam)
│-- model/
    │-- __init__.py
    │-- config.py              # Paths, feature names, model file names
    │-- classifiers.py         # 5 classifiers
    │-- metrics.py             # Accuracy, AUC, Precision, Recall, F1, MCC, 
    │-- preprocessing.py       # Log + standard scaler, stratified split
    │-- train_models.py        # Training script
    │-- spambase.csv           # Full raw dataset (downloaded from UCI)
    │-- scaler.pkl             # Fitted pre-processing artefact
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- svm.pkl
    │-- metrics_summary.csv    # Test-set metrics of all models
```

---

## d. Models Used

| Model | Implementation details (all in [model/classifiers.py](model/classifiers.py)) |
|---|---|
| Logistic Regression | Sigmoid output, full-batch gradient descent on log-loss, L2 penalty `1e-3`, learning rate `0.1`, 3,000 iterations |
| Decision Tree | CART-style tree, Gini impurity, exhaustive threshold search on sorted feature values, `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=5` |
| KNN | Euclidean distance (vectorised `‖a‖² + ‖b‖² − 2ab`), `k = 7`, majority vote; score = fraction of spam neighbours |
| Naive Bayes | Gaussian NB – per-class mean/variance, log-priors, log-likelihoods combined with the log-sum-exp trick |
| Support Vector Machine | Linear soft-margin SVM, primal hinge loss + L2 (`λ = 1e-3`), mini-batch sub-gradient descent (batch 64, 200 epochs, decaying learning rate) |

### Comparison Table (on `test_data.csv`, 842 e-mails)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.9454** | **0.9843** | **0.9531** | 0.9077 | **0.9299** | **0.8859** |
| Decision Tree | 0.9074 | 0.9544 | 0.9082 | 0.8542 | 0.8804 | 0.8059 |
| KNN | 0.9347 | 0.9703 | 0.9404 | 0.8929 | 0.9160 | 0.8634 |
| Naive Bayes | 0.8397 | 0.9631 | 0.7269 | **0.9583** | 0.8267 | 0.7054 |
| Support Vector Machine | 0.9394 | 0.9834 | 0.9467 | 0.8988 | 0.9221 | 0.8734 |

*Precision, Recall and F1 are reported for the positive class (spam). AUC uses predicted probabilities (SVM uses its signed margin).*

Confusion matrices (`[[TN, FP], [FN, TP]]`):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 491 | 15 | 31 | 305 |
| Decision Tree | 477 | 29 | 49 | 287 |
| KNN | 487 | 19 | 36 | 300 |
| Naive Bayes | 385 | 121 | 14 | 322 |
| Support Vector Machine | 489 | 17 | 34 | 302 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Best model overall – highest Accuracy (0.9454), AUC (0.9843), Precision (0.9531), F1 and MCC (0.8859). After the log transform the two classes are almost linearly separable, so a simple linear model with L2 regularisation generalises very well. It wrongly flags only 15 of 506 genuine e-mails. The learned weights are also interpretable: `$`, `remove`, `!`, `capital_run_length_total` and `000` push towards spam, while `hp`, `george`, `edu`, `meeting` (words from the donor's own mailbox) push towards non-spam. |
| **Decision Tree** | Weakest of the discriminative models (Accuracy 0.9074, MCC 0.8059). A single tree makes axis-parallel, piecewise-constant decisions and has high variance – it has the most missed spam (49 FN). Its AUC (0.9544) is the lowest because the 65 leaves give only a handful of distinct probability values, which makes ranking coarse. Its advantage is interpretability: the root splits on `char_freq_dollar`, then on `word_freq_remove` / `word_freq_hp` – the same signals Logistic Regression found. |
| **KNN** | Strong performance (Accuracy 0.9347, MCC 0.8634) because the log + z-score scaling makes Euclidean distance meaningful across all 57 features. Slightly behind the linear models due to the curse of dimensionality (57 sparse, mostly-zero features) and because with `k = 7` the score can only take 8 values, lowering AUC (0.9703). It is a lazy learner: no training cost, but it must store all 3,368 training rows and compute distances to all of them at prediction time. |
| **Naive Bayes** | Lowest Accuracy (0.8397), Precision (0.7269) and MCC (0.7054) but the **highest Recall (0.9583)** – it catches almost all spam but wrongly flags 121 of 506 genuine e-mails (24%). Two assumptions are violated: the features are zero-inflated rather than Gaussian, and they are strongly correlated (e.g. `hp`/`hpl`/`george`, `415`/`650`/`857`, the three capital-run features), so evidence is double-counted and probabilities become over-confident. Its AUC is still high (0.9631), meaning the ranking is good and a tuned decision threshold would help a lot. Very fast to train. |
| **Support Vector Machine** | Very close second (Accuracy 0.9394, AUC 0.9834, MCC 0.8734). Like Logistic Regression it learns a linear boundary, but optimises the hinge loss (maximum margin) instead of log-loss. The small gap comes from stochastic sub-gradient optimisation and a linear kernel; it gives a confidence margin rather than a calibrated probability. |
| **Overall, Winner for your dataset?** | **Logistic Regression.** It wins 5 of 6 metrics (all except Recall) and has the fewest false positives, which matters most in spam filtering because losing a genuine e-mail is worse than letting one spam through. Linear SVM is a close runner-up; Naive Bayes is only preferable if catching every spam (recall) is the sole priority. |

---

## Streamlit App Features

The app ([app.py](app.py)) loads the saved models from `model/` and **never retrains**.

- **Dataset upload (CSV)** – upload test data from the sidebar (the bundled `test_data.csv` is used until a file is uploaded, and can be downloaded from the sidebar). Columns are validated before scoring.
- **Model selection dropdown** – choose any of the five classifiers.
- **Evaluation metrics** – Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
- **Confusion matrix and classification report** for the selected model, plus a predictions CSV download.
- **Compare all models** tab – metrics of all five models on the uploaded data with a bar chart and the best model highlighted.
- **Test data overview** tab – row count, class balance and a preview of the uploaded data.

---

## How to Run Locally

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-train all models - downloads Spambase if model/spambase.csv is missing
python -m model.train_models

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---
