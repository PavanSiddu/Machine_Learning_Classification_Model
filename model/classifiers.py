"""
Binary classifiers implemented from scratch using only NumPy.

Every classifier exposes the same interface:
    fit(X, y)            -> trains on X (n_samples, n_features) and y in {0, 1}
    decision_scores(X)   -> continuous score for the positive class (used for AUC)
    predict(X)           -> hard 0/1 labels
"""
import numpy as np


def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionScratch:
    """Logistic regression trained with full-batch gradient descent + L2 penalty."""

    def __init__(self, learning_rate=0.1, n_iterations=3000, l2_penalty=1e-3):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_penalty = l2_penalty

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        for _ in range(self.n_iterations):
            error = _sigmoid(X @ self.weights_ + self.bias_) - y
            grad_w = X.T @ error / n_samples + self.l2_penalty * self.weights_
            grad_b = error.mean()
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        return self

    def predict_proba(self, X):
        return _sigmoid(X @ self.weights_ + self.bias_)

    def decision_scores(self, X):
        return self.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class _TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # P(class = 1) at a leaf

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeScratch:
    """CART-style binary decision tree using Gini impurity."""

    def __init__(self, max_depth=10, min_samples_split=20, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    @staticmethod
    def _gini(p):
        return 2.0 * p * (1.0 - p)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        total_pos = y.sum()
        parent_impurity = self._gini(total_pos / n_samples)
        best_gain, best_feature, best_threshold = 0.0, None, None

        n_left = np.arange(1, n_samples)
        n_right = n_samples - n_left
        size_ok = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)

        for j in range(n_features):
            order = np.argsort(X[:, j], kind="mergesort")
            x_sorted = X[order, j]
            y_sorted = y[order]

            # Candidate split after position i: left = [0..i], right = [i+1..]
            pos_left = np.cumsum(y_sorted)[:-1]
            pos_right = total_pos - pos_left
            valid = size_ok & (x_sorted[:-1] < x_sorted[1:])
            if not valid.any():
                continue

            weighted = (
                n_left * self._gini(pos_left / n_left)
                + n_right * self._gini(pos_right / n_right)
            ) / n_samples
            weighted[~valid] = np.inf

            i = int(np.argmin(weighted))
            gain = parent_impurity - weighted[i]
            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0

        return best_feature, best_threshold

    def _grow(self, X, y, depth):
        n_samples = len(y)
        positive_rate = y.mean()

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or positive_rate in (0.0, 1.0)
        ):
            return _TreeNode(value=positive_rate)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return _TreeNode(value=positive_rate)

        mask = X[:, feature] <= threshold
        left = self._grow(X[mask], y[mask], depth + 1)
        right = self._grow(X[~mask], y[~mask], depth + 1)
        return _TreeNode(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.root_ = self._grow(X, y.astype(float), depth=0)
        return self

    def _predict_row(self, row):
        node = self.root_
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict_proba(self, X):
        return np.array([self._predict_row(row) for row in X])

    def decision_scores(self, X):
        return self.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class KNNScratch:
    """K-Nearest Neighbours with Euclidean distance and majority voting."""

    def __init__(self, k=7, batch_size=512):
        self.k = k
        self.batch_size = batch_size

    def fit(self, X, y):
        # KNN is a lazy learner: training only memorises the data
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=int)
        self._train_sq_norms = (self.X_train_ ** 2).sum(axis=1)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probs = np.empty(len(X))
        for start in range(0, len(X), self.batch_size):
            batch = X[start:start + self.batch_size]
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab, computed for the whole batch at once
            sq_dist = (
                (batch ** 2).sum(axis=1)[:, None]
                + self._train_sq_norms[None, :]
                - 2.0 * batch @ self.X_train_.T
            )
            neighbours = np.argpartition(sq_dist, self.k - 1, axis=1)[:, : self.k]
            probs[start:start + len(batch)] = self.y_train_[neighbours].mean(axis=1)
        return probs

    def decision_scores(self, X):
        return self.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class GaussianNaiveBayesScratch:
    """Gaussian Naive Bayes: features assumed independent and normal within each class."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        epsilon = self.var_smoothing * X.var(axis=0).max()
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.vars_ = np.array([X[y == c].var(axis=0) + epsilon for c in self.classes_])
        self.log_priors_ = np.log(np.array([np.mean(y == c) for c in self.classes_]))
        return self

    def _joint_log_likelihood(self, X):
        jll = []
        for idx in range(len(self.classes_)):
            mean, var = self.means_[idx], self.vars_[idx]
            log_pdf = -0.5 * np.sum(np.log(2.0 * np.pi * var)) - 0.5 * np.sum(
                (X - mean) ** 2 / var, axis=1
            )
            jll.append(self.log_priors_[idx] + log_pdf)
        return np.column_stack(jll)

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        # log-sum-exp trick for numerically stable normalisation
        jll -= jll.max(axis=1, keepdims=True)
        probs = np.exp(jll)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, 1]

    def decision_scores(self, X):
        return self.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class LinearSVMScratch:
    """
    Soft-margin linear SVM trained by mini-batch sub-gradient descent on the primal:
        min_w,b  (lambda / 2) * ||w||^2 + mean(max(0, 1 - y_i * (w . x_i + b)))
    """

    def __init__(self, lambda_param=1e-3, learning_rate=0.05, n_epochs=200,
                 batch_size=64, random_state=None):
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        y_signed = np.where(y == 1, 1.0, -1.0)
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        step = 0
        for _ in range(self.n_epochs):
            permutation = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                idx = permutation[start:start + self.batch_size]
                Xb, yb = X[idx], y_signed[idx]
                margins = yb * (Xb @ self.weights_ + self.bias_)
                violators = margins < 1.0

                grad_w = self.lambda_param * self.weights_ - (
                    yb[violators, None] * Xb[violators]
                ).sum(axis=0) / len(idx)
                grad_b = -yb[violators].sum() / len(idx)

                lr = self.learning_rate / (1.0 + self.learning_rate * self.lambda_param * step)
                self.weights_ -= lr * grad_w
                self.bias_ -= lr * grad_b
                step += 1
        return self

    def decision_scores(self, X):
        # Signed distance-like margin; larger means "more spam"
        return X @ self.weights_ + self.bias_

    def predict(self, X):
        return (self.decision_scores(X) >= 0.0).astype(int)
