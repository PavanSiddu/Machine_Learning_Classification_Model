"""Evaluation metrics for binary classification, implemented with NumPy only."""
import numpy as np


def confusion_matrix(y_true, y_pred):
    """Returns [[TN, FP], [FN, TP]]."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true, y_pred, positive=1):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == positive) & (y_true == positive))
    return _safe_divide(tp, np.sum(y_pred == positive))


def recall(y_true, y_pred, positive=1):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == positive) & (y_true == positive))
    return _safe_divide(tp, np.sum(y_true == positive))


def f1_score(y_true, y_pred, positive=1):
    p = precision(y_true, y_pred, positive)
    r = recall(y_true, y_pred, positive)
    return _safe_divide(2 * p * r, p + r)


def matthews_corrcoef(y_true, y_pred):
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred).astype(float)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return _safe_divide(tp * tn - fp * fn, denominator)


def roc_auc(y_true, scores):
    """
    AUC via the Mann-Whitney U statistic: probability that a random positive
    is scored higher than a random negative (ties count as half).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    average_ranks = np.cumsum(counts) - (counts - 1) / 2.0  # 1-based, tie-averaged
    ranks = average_ranks[inverse]
    u_statistic = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u_statistic / (n_pos * n_neg))


def evaluate(y_true, y_pred, scores):
    return {
        "Accuracy": accuracy(y_true, y_pred),
        "AUC": roc_auc(y_true, scores),
        "Precision": precision(y_true, y_pred),
        "Recall": recall(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def classification_report(y_true, y_pred, labels=(0, 1), target_names=("Not Spam", "Spam")):
    """Per-class precision/recall/F1/support plus macro and weighted averages."""
    y_true = np.asarray(y_true).astype(int)
    rows = {}
    supports = []
    for label, name in zip(labels, target_names):
        support = int(np.sum(y_true == label))
        supports.append(support)
        rows[name] = {
            "precision": precision(y_true, y_pred, label),
            "recall": recall(y_true, y_pred, label),
            "f1-score": f1_score(y_true, y_pred, label),
            "support": support,
        }

    per_class = list(rows.values())
    total = sum(supports)
    rows["accuracy"] = {
        "precision": None, "recall": None,
        "f1-score": accuracy(y_true, y_pred), "support": total,
    }

    weights = np.array(supports) / total if total else np.zeros(len(supports))
    for avg_name, w in (("macro avg", np.full(len(per_class), 1 / len(per_class))),
                        ("weighted avg", weights)):
        rows[avg_name] = {
            metric: float(np.dot(w, [r[metric] for r in per_class]))
            for metric in ("precision", "recall", "f1-score")
        }
        rows[avg_name]["support"] = total
    return rows
