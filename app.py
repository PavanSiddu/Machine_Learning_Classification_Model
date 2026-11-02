"""
Streamlit front-end for the Spambase classifiers.
Loads the pre-trained (from-scratch) models from model/ and evaluates them on an
uploaded test CSV -- no retraining happens here.
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from model.config import (
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_FILES,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
)
from model.metrics import classification_report, confusion_matrix, evaluate

CLASS_NAMES = ("Not Spam", "Spam")
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Spam Detector | Scratch ML Models", layout="wide")


@st.cache_resource
def load_artifacts():
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    models = {}
    for name, filename in MODEL_FILES.items():
        with open(MODEL_DIR / filename, "rb") as f:
            models[name] = pickle.load(f)
    return scaler, models


def validate_test_data(df):
    errors = []
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        errors.append(f"Missing {len(missing)} required column(s): {preview}")
        return errors

    numeric = df[FEATURE_COLUMNS + [TARGET_COLUMN]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_cols = numeric.columns[numeric.isna().any()].tolist()
        errors.append(f"Non-numeric or empty values found in: {', '.join(bad_cols[:8])}")
    elif not set(numeric[TARGET_COLUMN].unique()).issubset({0, 1}):
        errors.append(f"Column '{TARGET_COLUMN}' must contain only 0 (not spam) and 1 (spam).")
    return errors


@st.cache_data(show_spinner="Scoring test data with all five models...")
def run_all_models(df):
    scaler, models = load_artifacts()
    X = scaler.transform(df[FEATURE_COLUMNS].to_numpy(dtype=float))
    y_true = df[TARGET_COLUMN].to_numpy(dtype=int)

    outputs, rows = {}, []
    for name, clf in models.items():
        y_pred = clf.predict(X)
        scores = clf.decision_scores(X)
        outputs[name] = (y_pred, scores)
        rows.append({"Model": name, **evaluate(y_true, y_pred, scores)})
    return y_true, outputs, pd.DataFrame(rows).set_index("Model")


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 14}, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_model_comparison(summary):
    fig, ax = plt.subplots(figsize=(10, 4))
    summary[METRIC_NAMES].T.plot(kind="bar", ax=ax, width=0.8, colormap="Set2")
    ax.set_ylim(max(0.0, summary[METRIC_NAMES].min().min() - 0.1), 1.0)
    ax.set_ylabel("Score")
    ax.set_xticklabels(METRIC_NAMES, rotation=0)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------- Sidebar ---------------------------------------
st.sidebar.header("1. Test data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a test CSV (57 features + 'is_spam')", type=["csv"]
)
with open(TEST_DATA_PATH, "rb") as f:
    st.sidebar.download_button(
        "Download sample test_data.csv", f.read(), file_name="test_data.csv", mime="text/csv"
    )

st.sidebar.header("2. Model")
selected_model = st.sidebar.selectbox("Choose a classifier", list(MODEL_FILES.keys()))

# ----------------------------- Header ----------------------------------------
st.title("Spam Email Detector")
st.markdown(
    "Evaluate five **hand-built** classifiers on the "
    "[UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase) dataset. "
    "Upload held-out test data, pick a model and inspect its performance."
)

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
        st.error(f"Could not read the uploaded file as CSV: {exc}")
        st.stop()
    source_label = uploaded_file.name
else:
    test_df = pd.read_csv(TEST_DATA_PATH)
    source_label = "bundled test_data.csv"
    st.info("No file uploaded yet - showing results for the bundled **test_data.csv**. "
            "Upload your own CSV from the sidebar to evaluate it instead.")

problems = validate_test_data(test_df)
if problems:
    for msg in problems:
        st.error(msg)
    st.stop()

y_true, model_outputs, summary = run_all_models(test_df)
y_pred, scores = model_outputs[selected_model]

tab_eval, tab_compare, tab_data = st.tabs(
    ["Selected model", "Compare all models", "Test data overview"]
)

# ----------------------------- Selected model --------------------------------
with tab_eval:
    st.subheader(f"{selected_model} on {source_label} ({len(test_df)} emails)")

    metric_cols = st.columns(len(METRIC_NAMES))
    for col, metric in zip(metric_cols, METRIC_NAMES):
        col.metric(metric, f"{summary.loc[selected_model, metric]:.4f}")

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig = plot_confusion_matrix(cm, selected_model)
        st.pyplot(fig)
        plt.close(fig)
        (tn, fp), (fn, tp) = cm
        st.caption(f"Spam caught: {tp} | Spam missed: {fn} | "
                   f"Genuine mails wrongly flagged: {fp} | Genuine mails passed: {tn}")

    with right:
        st.markdown("**Classification report**")
        report = pd.DataFrame(classification_report(y_true, y_pred, target_names=CLASS_NAMES)).T
        st.dataframe(
            report.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                 "f1-score": "{:.4f}", "support": "{:.0f}"}, na_rep=""),
            width="stretch",
        )

        predictions = test_df[[TARGET_COLUMN]].copy()
        predictions["predicted"] = y_pred
        predictions["score"] = np.round(scores, 4)
        st.download_button(
            "Download predictions (CSV)",
            predictions.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{MODEL_FILES[selected_model].replace('.pkl', '')}.csv",
            mime="text/csv",
        )

# ----------------------------- Comparison ------------------------------------
with tab_compare:
    st.subheader("All models on the same test data")
    st.dataframe(
        summary.style.format("{:.4f}").highlight_max(axis=0, color="#616145"),
        width="stretch",
    )
    best_model = summary["MCC"].idxmax()
    st.success(f"Best model by MCC on this data: **{best_model}** "
               f"(MCC = {summary.loc[best_model, 'MCC']:.4f}, "
               f"Accuracy = {summary.loc[best_model, 'Accuracy']:.4f})")
    fig = plot_model_comparison(summary)
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------- Data overview ---------------------------------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(test_df))
    c2.metric("Spam emails", int((y_true == 1).sum()))
    c3.metric("Non-spam emails", int((y_true == 0).sum()))
    st.dataframe(test_df.head(20), width="stretch")
