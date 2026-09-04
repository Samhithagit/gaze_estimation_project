# module3/screen_mapper.py
# =============================================================
# Fits a regularized polynomial regression from (gaze_yaw, gaze_pitch)
# to (screen_x, screen_y) using 9-point calibration data.
# Standardizes features and uses Ridge regularization to ensure
# smooth extrapolation at all 4 screen corners (top-left, bottom-right).
# =============================================================

import json
import pickle
import numpy as np
from pathlib import Path


class ScreenMapper:
    """
    Maps (gaze_yaw, gaze_pitch) → (screen_x, screen_y)
    using standardized, Ridge-regularized polynomial regression.
    """

    def __init__(self, ridge_alpha=0.1):
        self.weights_x = None
        self.weights_y = None
        self.r2_x = None
        self.r2_y = None
        self.yaw_mean = 0.0
        self.yaw_std = 1.0
        self.pitch_mean = 0.0
        self.pitch_std = 1.0
        self.ridge_alpha = ridge_alpha
        self.is_fitted = False

    def _standardize(self, yaws, pitches):
        y_norm = (np.array(yaws, dtype=np.float64) - self.yaw_mean) / self.yaw_std
        p_norm = (np.array(pitches, dtype=np.float64) - self.pitch_mean) / self.pitch_std
        return y_norm, p_norm

    def _build_features(self, y_norm, p_norm):
        ones = np.ones_like(y_norm)
        return np.column_stack([
            ones,
            y_norm,
            p_norm,
        ])

    def fit(self, calibration_data):
        yaws    = np.array([d["gaze_yaw"]   for d in calibration_data], dtype=np.float64)
        pitches = np.array([d["gaze_pitch"] for d in calibration_data], dtype=np.float64)
        sx      = np.array([d["screen_x"]   for d in calibration_data], dtype=np.float64)
        sy      = np.array([d["screen_y"]   for d in calibration_data], dtype=np.float64)

        # Calculate mean & std for feature scaling
        self.yaw_mean   = float(np.mean(yaws))
        self.yaw_std    = float(np.std(yaws)) if np.std(yaws) > 1e-5 else 1.0
        self.pitch_mean = float(np.mean(pitches))
        self.pitch_std  = float(np.std(pitches)) if np.std(pitches) > 1e-5 else 1.0

        # Sample weights from confidence
        weights = np.array([d.get("confidence_mean", 1.0) for d in calibration_data], dtype=np.float64)
        weights = weights / weights.sum() * len(weights)

        y_norm, p_norm = self._standardize(yaws, pitches)
        Phi = self._build_features(y_norm, p_norm)
        W   = np.diag(weights)

        # Ridge Regression: (Φᵀ W Φ + α I) w = Φᵀ W y
        PhiTW = Phi.T @ W
        lhs   = PhiTW @ Phi + self.ridge_alpha * np.eye(Phi.shape[1])
        rhs_x = PhiTW @ sx
        rhs_y = PhiTW @ sy

        self.weights_x = np.linalg.solve(lhs, rhs_x)
        self.weights_y = np.linalg.solve(lhs, rhs_y)

        # Compute R²
        pred_sx = Phi @ self.weights_x
        pred_sy = Phi @ self.weights_y

        self.r2_x = self._r2(sx, pred_sx)
        self.r2_y = self._r2(sy, pred_sy)
        self.is_fitted = True

        quality = self._quality_label()
        return {
            "r2_x": self.r2_x,
            "r2_y": self.r2_y,
            "quality": quality,
        }

    def predict(self, yaw, pitch):
        if not self.is_fitted:
            raise RuntimeError("ScreenMapper is not fitted. Call fit() first.")

        y_norm, p_norm = self._standardize([yaw], [pitch])
        phi = self._build_features(y_norm, p_norm)[0]

        screen_x = float(phi @ self.weights_x)
        screen_y = float(phi @ self.weights_y)

        return screen_x, screen_y

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "weights_x": self.weights_x,
                "weights_y": self.weights_y,
                "yaw_mean": self.yaw_mean,
                "yaw_std": self.yaw_std,
                "pitch_mean": self.pitch_mean,
                "pitch_std": self.pitch_std,
                "r2_x": self.r2_x,
                "r2_y": self.r2_y,
                "is_fitted": self.is_fitted,
            }, f)
        print(f"[ScreenMapper] Saved to {path}")

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No mapper found at {path}. Run calibrate.py first.")
        with open(path, "rb") as f:
            data = pickle.load(f)

        mapper = cls()
        mapper.weights_x  = data["weights_x"]
        mapper.weights_y  = data["weights_y"]
        mapper.yaw_mean   = data.get("yaw_mean", 0.0)
        mapper.yaw_std    = data.get("yaw_std", 1.0)
        mapper.pitch_mean = data.get("pitch_mean", 0.0)
        mapper.pitch_std  = data.get("pitch_std", 1.0)
        mapper.r2_x       = data["r2_x"]
        mapper.r2_y       = data["r2_y"]
        mapper.is_fitted  = data["is_fitted"]

        print(
            f"[ScreenMapper] Loaded from {path}  "
            f"(R²_x={mapper.r2_x:.3f}, R²_y={mapper.r2_y:.3f}, "
            f"{mapper._quality_label()})"
        )
        return mapper

    # =========================
    # Helpers
    # =========================

    @staticmethod
    def _r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0
        return float(1.0 - ss_res / ss_tot)

    def _quality_label(self):
        avg = (self.r2_x + self.r2_y) / 2
        if avg >= 0.90:
            return "GOOD"
        elif avg >= 0.75:
            return "ACCEPTABLE"
        else:
            return "POOR — please recalibrate"

    def summary(self):
        if not self.is_fitted:
            return "Not fitted."
        return (
            f"R²_x = {self.r2_x:.3f} | "
            f"R²_y = {self.r2_y:.3f} | "
            f"Quality: {self._quality_label()}"
        )
