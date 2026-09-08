"""
Ridge regression gaze-to-screen mapper.

Matches the base paper's (Cărbunescu-Stoenescu & Giurgiu, ICCP 2025)
methodology in Section III.C:
  - gaze-only mode: 2 features (gaze_yaw, gaze_pitch) -- their
    "Linear Regression" baseline, though we use Ridge with alpha
    near 0 here so both modes share one implementation and are
    directly comparable (alpha=0 Ridge == plain least squares).
  - gaze+head mode: 5 features (gaze_yaw, gaze_pitch, head_yaw,
    head_pitch, head_roll) -- their Ridge-regularized model.
  - Both expanded to 2nd-order polynomial features (with cross
    terms) before fitting, matching their "polynomial order of 2
    for feature transformation."

This is a from-scratch NumPy implementation (no sklearn dependency)
so it drops in next to your existing final_evaluation.py style.
"""

import numpy as np


def polynomial_features_degree2(X):
    """
    X: (n_samples, n_features) raw features.
    Returns (n_samples, n_poly_features) with:
      [1, x1, x2, ..., xn, x1^2, x2^2, ..., xn^2,
       x1*x2, x1*x3, ..., x_{n-1}*xn]
    This generalizes the manual 6-column design matrix used
    elsewhere in your pipeline (for 2 features) to any number of
    input features, so the same function works for both the
    2-feature (gaze-only) and 5-feature (gaze+head) cases.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    cols = [np.ones(n_samples)]

    # linear terms
    for i in range(n_features):
        cols.append(X[:, i])

    # squared terms
    for i in range(n_features):
        cols.append(X[:, i] ** 2)

    # pairwise cross terms
    for i in range(n_features):
        for j in range(i + 1, n_features):
            cols.append(X[:, i] * X[:, j])

    return np.stack(cols, axis=1)


class RidgeGazeMapper:
    """
    feature_mode: "gaze_only" or "gaze_head"
    alpha: ridge regularization strength (0.01 matches the base
        paper's tuned optimum for their gaze+head model; use a
        very small value like 1e-6 for an effectively-unregularized
        gaze-only fit if you want a closer match to their plain
        Linear Regression baseline).
    """

    def __init__(self, feature_mode="gaze_only", alpha=0.01):
        if feature_mode not in ("gaze_only", "gaze_head"):
            raise ValueError("feature_mode must be 'gaze_only' or 'gaze_head'")
        self.feature_mode = feature_mode
        self.alpha = float(alpha)
        self.coef_x = None
        self.coef_y = None
        self._n_raw_features = 2 if feature_mode == "gaze_only" else 5

    def _build_raw_features(self, gaze_yaw, gaze_pitch,
                             head_yaw=None, head_pitch=None, head_roll=None):
        gaze_yaw = np.asarray(gaze_yaw, dtype=np.float64)
        gaze_pitch = np.asarray(gaze_pitch, dtype=np.float64)

        if self.feature_mode == "gaze_only":
            return np.stack([gaze_yaw, gaze_pitch], axis=1)

        if head_yaw is None or head_pitch is None or head_roll is None:
            raise ValueError(
                "feature_mode='gaze_head' requires head_yaw, head_pitch, "
                "head_roll to be provided"
            )

        head_yaw = np.asarray(head_yaw, dtype=np.float64)
        head_pitch = np.asarray(head_pitch, dtype=np.float64)
        head_roll = np.asarray(head_roll, dtype=np.float64)

        return np.stack(
            [gaze_yaw, gaze_pitch, head_yaw, head_pitch, head_roll], axis=1
        )

    def fit(self, gaze_yaw, gaze_pitch, target_x, target_y,
            head_yaw=None, head_pitch=None, head_roll=None):

        raw = self._build_raw_features(
            gaze_yaw, gaze_pitch, head_yaw, head_pitch, head_roll
        )
        A = polynomial_features_degree2(raw)

        n_features = A.shape[1]
        reg = self.alpha * np.eye(n_features)
        reg[0, 0] = 0.0  # never regularize the bias term

        AtA = A.T @ A + reg
        self.coef_x = np.linalg.solve(AtA, A.T @ np.asarray(target_x, dtype=np.float64))
        self.coef_y = np.linalg.solve(AtA, A.T @ np.asarray(target_y, dtype=np.float64))

        return self

    def predict(self, gaze_yaw, gaze_pitch,
                head_yaw=None, head_pitch=None, head_roll=None):

        if self.coef_x is None or self.coef_y is None:
            raise RuntimeError("RidgeGazeMapper.fit() must be called before predict()")

        raw = self._build_raw_features(
            gaze_yaw, gaze_pitch, head_yaw, head_pitch, head_roll
        )
        A = polynomial_features_degree2(raw)

        pred_x = A @ self.coef_x
        pred_y = A @ self.coef_y
        return pred_x, pred_y