"""
Trains the five from-scratch classifiers on the UCI Spambase dataset and saves:
  - model/*.pkl              trained models + fitted scaler
  - model/metrics_summary.csv evaluation metrics on the held-out test split
  - test_data.csv            the held-out test split (raw features + target)

Run from the project root:
    python -m model.train_models
"""
import io
import pickle
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python model/train_models.py` as well as `python -m model.train_models`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.classifiers import (  # noqa: E402
    DecisionTreeScratch,
    GaussianNaiveBayesScratch,
    KNNScratch,
    LinearSVMScratch,
    LogisticRegressionScratch,
)
from model.config import (  # noqa: E402
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_SIZE,
    UCI_SPAMBASE_URL,
)
from model.metrics import evaluate  # noqa: E402
from model.preprocessing import LogStandardScaler, stratified_train_test_split  # noqa: E402


def load_spambase():
    if not RAW_DATA_PATH.exists():
        print(f"Downloading Spambase from {UCI_SPAMBASE_URL} ...")
        with urllib.request.urlopen(UCI_SPAMBASE_URL, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        raw = pd.read_csv(archive.open("spambase.data"), header=None)
        raw.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        raw.to_csv(RAW_DATA_PATH, index=False)
    return pd.read_csv(RAW_DATA_PATH)


def build_models():
    return {
        "Logistic Regression": LogisticRegressionScratch(
            learning_rate=0.1, n_iterations=3000, l2_penalty=1e-3
        ),
        "Decision Tree": DecisionTreeScratch(
            max_depth=10, min_samples_split=20, min_samples_leaf=5
        ),
        "KNN": KNNScratch(k=7),
        "Naive Bayes": GaussianNaiveBayesScratch(var_smoothing=1e-9),
        "Support Vector Machine": LinearSVMScratch(
            lambda_param=1e-3, learning_rate=0.05, n_epochs=200,
            batch_size=64, random_state=RANDOM_STATE,
        ),
    }


def main():
    df = load_spambase()
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {n_raw} rows, {n_raw - len(df)} duplicates removed -> {len(df)} rows, "
          f"{len(FEATURE_COLUMNS)} features")
    print("Class balance:", df[TARGET_COLUMN].value_counts().to_dict())

    y_all = df[TARGET_COLUMN].to_numpy(dtype=int)
    train_idx, test_idx = stratified_train_test_split(
        y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows -> saved {TEST_DATA_PATH.name}")

    scaler = LogStandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].to_numpy(dtype=float))
    X_test = scaler.transform(test_df[FEATURE_COLUMNS].to_numpy(dtype=float))
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=int)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=int)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    results = []
    for name, clf in build_models().items():
        start = time.perf_counter()
        clf.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        metrics = evaluate(y_test, clf.predict(X_test), clf.decision_scores(X_test))
        results.append({"Model": name, **metrics})
        with open(MODEL_DIR / MODEL_FILES[name], "wb") as f:
            pickle.dump(clf, f)
        print(f"  trained {name:<24} in {elapsed:6.2f}s")

    summary = pd.DataFrame(results).set_index("Model").round(4)
    summary.to_csv(METRICS_PATH)

    print("\n| ML Model Name | " + " | ".join(summary.columns) + " |")
    print("|---" * (len(summary.columns) + 1) + "|")
    for name, row in summary.iterrows():
        print(f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |")


if __name__ == "__main__":
    main()
