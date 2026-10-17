"""Extra Trees gaze-to-screen mapper using gaze and head yaw/pitch only.

Features (head roll intentionally excluded):
    gaze_yaw, gaze_pitch, head_yaw, head_pitch

The API mirrors RidgeGazeMapper: fit(...) and predict(...).
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


class ExtraTreesGazeMapper:
    def __init__(self, n_estimators=200, max_depth=None, min_samples_leaf=2,
                 random_state=42, n_jobs=-1):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.model_x = None
        self.model_y = None

    @staticmethod
    def _features(gaze_yaw, gaze_pitch, head_yaw, head_pitch):
        arrays = [
            np.asarray(gaze_yaw, dtype=np.float64).reshape(-1),
            np.asarray(gaze_pitch, dtype=np.float64).reshape(-1),
            np.asarray(head_yaw, dtype=np.float64).reshape(-1),
            np.asarray(head_pitch, dtype=np.float64).reshape(-1),
        ]
        if len({len(a) for a in arrays}) != 1:
            raise ValueError("All four input feature arrays must have equal lengths")
        X = np.column_stack(arrays)
        if not np.all(np.isfinite(X)):
            raise ValueError("Input features contain NaN or infinite values")
        return X

    def fit(self, gaze_yaw, gaze_pitch, target_x, target_y,
            head_yaw, head_pitch):
        X = self._features(gaze_yaw, gaze_pitch, head_yaw, head_pitch)
        target_x = np.asarray(target_x, dtype=np.float64).reshape(-1)
        target_y = np.asarray(target_y, dtype=np.float64).reshape(-1)
        if len(X) != len(target_x) or len(X) != len(target_y):
            raise ValueError("Features and target arrays must have equal lengths")
        if len(X) < 2:
            raise ValueError("At least two calibration samples are required")
        if not np.all(np.isfinite(target_x)) or not np.all(np.isfinite(target_y)):
            raise ValueError("Target coordinates contain NaN or infinite values")
        args = dict(n_estimators=self.n_estimators, max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state, n_jobs=self.n_jobs)
        self.model_x = ExtraTreesRegressor(**args)
        self.model_y = ExtraTreesRegressor(**args)
        self.model_x.fit(X, target_x)
        self.model_y.fit(X, target_y)
        return self

    def predict(self, gaze_yaw, gaze_pitch, head_yaw, head_pitch):
        if self.model_x is None or self.model_y is None:
            raise RuntimeError("Call fit() before predict()")
        X = self._features(gaze_yaw, gaze_pitch, head_yaw, head_pitch)
        return (np.asarray(self.model_x.predict(X), dtype=np.float64).reshape(-1),
                np.asarray(self.model_y.predict(X), dtype=np.float64).reshape(-1))
