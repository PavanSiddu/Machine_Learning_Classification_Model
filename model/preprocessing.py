"""Pre-processing utilities written from scratch (no scikit-learn)."""
import numpy as np


class LogStandardScaler:
    """
    Applies log(1 + x) followed by z-score standardisation.
    The Spambase features are non-negative and heavily right-skewed
    (e.g. capital_run_length_total goes up to ~15,000), so the log step
    pulls in extreme values before scaling.
    """

    def fit(self, X):
        X_log = np.log1p(np.clip(X, 0, None))
        self.mean_ = X_log.mean(axis=0)
        self.std_ = X_log.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        X_log = np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))
        return (X_log - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def stratified_train_test_split(y, test_size=0.2, random_state=None):
    """Returns (train_idx, test_idx) keeping the original class proportions in both parts."""
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []
    for label in np.unique(y):
        label_idx = rng.permutation(np.flatnonzero(y == label))
        n_test = int(round(len(label_idx) * test_size))
        test_idx.append(label_idx[:n_test])
        train_idx.append(label_idx[n_test:])

    return rng.permutation(np.concatenate(train_idx)), rng.permutation(np.concatenate(test_idx))
