import numpy as np


# ============================================================
# FIXED KALMAN
# ============================================================

class FixedKalmanGaze:
    """
    Fixed-parameter Kalman filter for gaze yaw and pitch.

    State:
        x = [yaw, pitch]

    Measurement:
        z = [measured_yaw, measured_pitch]

    Q = fixed process noise
    R = fixed measurement noise
    """

    def __init__(
        self,
        process_noise=0.01,
        measurement_noise=1.0,
        initial_uncertainty=1.0
    ):

        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_uncertainty = float(initial_uncertainty)

        self.A = np.eye(2, dtype=np.float64)
        self.H = np.eye(2, dtype=np.float64)

        self.Q = np.eye(2, dtype=np.float64) * self.process_noise
        self.R = np.eye(2, dtype=np.float64) * self.measurement_noise
        self.P = np.eye(2, dtype=np.float64) * self.initial_uncertainty

        self.state = None

    def reset(self):
        self.P = np.eye(2, dtype=np.float64) * self.initial_uncertainty
        self.state = None

    def update(self, yaw, pitch, confidence=None):
        # confidence is accepted (and ignored) here so ALL filter
        # classes share a uniform call signature -- lets you swap
        # filters in phase2_calibration.py without branching logic.

        if yaw is None or pitch is None:
            return None

        try:
            measurement = np.array([float(yaw), float(pitch)], dtype=np.float64)
        except (TypeError, ValueError):
            return None

        if not np.all(np.isfinite(measurement)):
            return None

        # ---- INITIALIZATION ----
        if self.state is None:
            self.state = measurement.copy()
            return {"yaw": float(self.state[0]), "pitch": float(self.state[1])}

        # ---- PREDICTION ----
        predicted_state = self.A @ self.state
        predicted_covariance = self.A @ self.P @ self.A.T + self.Q

        # ---- INNOVATION ----
        innovation = measurement - self.H @ predicted_state
        innovation_covariance = self.H @ predicted_covariance @ self.H.T + self.R

        # ---- KALMAN GAIN ----
        kalman_gain = predicted_covariance @ self.H.T @ np.linalg.inv(innovation_covariance)

        # ---- STATE UPDATE ----
        self.state = predicted_state + kalman_gain @ innovation

        # ---- COVARIANCE UPDATE ----
        identity = np.eye(2, dtype=np.float64)
        self.P = (identity - kalman_gain @ self.H) @ predicted_covariance

        return {"yaw": float(self.state[0]), "pitch": float(self.state[1])}


# ============================================================
# ADAPTIVE Q KALMAN
#
# Unchanged from your working version -- this logic is CORRECT:
# large innovation (real motion, e.g. saccade to a new target)
# -> increase Q -> trust the model's own prediction less -> react
# faster to the new measurement. This is why Adaptive Q performed
# best in your last trial run.
# ============================================================

class AdaptiveQKalmanGaze(FixedKalmanGaze):

    def __init__(
        self,
        process_noise=0.01,
        measurement_noise=1.0,
        initial_uncertainty=1.0,
        q_min=0.001,
        q_max=0.20,
        adaptation_rate=0.10
    ):
        super().__init__(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_uncertainty=initial_uncertainty
        )

        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.adaptation_rate = float(adaptation_rate)

        self.current_q = self.process_noise

    def update(self, yaw, pitch, confidence=None):
        # confidence accepted for signature compatibility, not
        # used by this class (Q reacts to innovation, not confidence)

        if yaw is None or pitch is None:
            return None

        try:
            measurement = np.array([float(yaw), float(pitch)], dtype=np.float64)
        except (TypeError, ValueError):
            return None

        if not np.all(np.isfinite(measurement)):
            return None

        # ---- INITIALIZATION ----
        if self.state is None:
            self.state = measurement.copy()
            self.current_q = self.process_noise
            self.Q = np.eye(2, dtype=np.float64) * self.current_q
            return {"yaw": float(self.state[0]), "pitch": float(self.state[1])}

        # ---- INNOVATION-BASED Q ADAPTATION ----
        predicted_state = self.A @ self.state
        innovation = measurement - self.H @ predicted_state
        innovation_magnitude = np.linalg.norm(innovation)

        adaptive_q = self.process_noise + self.adaptation_rate * innovation_magnitude ** 2
        adaptive_q = np.clip(adaptive_q, self.q_min, self.q_max)

        self.current_q = float(adaptive_q)
        self.Q = np.eye(2, dtype=np.float64) * self.current_q

        # ---- PREDICTION ----
        predicted_covariance = self.A @ self.P @ self.A.T + self.Q

        # ---- KALMAN UPDATE ----
        innovation_covariance = self.H @ predicted_covariance @ self.H.T + self.R
        kalman_gain = predicted_covariance @ self.H.T @ np.linalg.inv(innovation_covariance)
        self.state = predicted_state + kalman_gain @ innovation

        identity = np.eye(2, dtype=np.float64)
        self.P = (identity - kalman_gain @ self.H) @ predicted_covariance

        return {"yaw": float(self.state[0]), "pitch": float(self.state[1])}


# ============================================================
# ADAPTIVE R KALMAN -- FIXED VERSION
#
# BUG IN THE ORIGINAL: R was scaled by the Kalman innovation
# (residual) itself -- self-referential, since a genuine target
# jump (real saccade to a new calibration point) produces the
# same large-residual signal as sensor noise. The filter ended
# up DISTRUSTING real gaze movement and lagging behind it: low
# jitter, but MAE barely better than baseline.
#
# FIX: scale R by the model's own CONFIDENCE (entropy-derived
# from L2CS-Net's softmax output), which is independent of
# whether the true gaze moved. High confidence -> trust this
# measurement -> low R. Low confidence -> distrust -> high R.
# This matches the confidence-gated R design in your own
# architecture diagram (Fig 4.1 / 4.2).
# ============================================================

class AdaptiveRKalmanGaze(FixedKalmanGaze):

    def __init__(
        self,
        process_noise=0.01,
        measurement_noise=1.0,
        initial_uncertainty=1.0,
        r_min=0.10,
        r_max=10.0,
        conf_window_size=60,   # NEW: rolling window for normalization
    ):
        super().__init__(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_uncertainty=initial_uncertainty
        )
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.current_r = self.measurement_noise

        # NEW: track recent confidence values to normalize against
        # the ACTUAL observed range, not an assumed [0,1] range.
        # Your model's confidence only ever spans ~0.55-0.65, so a
        # raw linear map onto [r_min, r_max] barely moves R at all.
        # Normalizing against a rolling min/max restores full
        # dynamic range even when raw confidence is narrow.
        self.conf_window_size = conf_window_size
        self._conf_history = []

    def _normalize_confidence(self, confidence):
        self._conf_history.append(confidence)
        if len(self._conf_history) > self.conf_window_size:
            self._conf_history.pop(0)

        if len(self._conf_history) < 5:
            # not enough history yet -- fall back to raw value
            return confidence

        c_min = min(self._conf_history)
        c_max = max(self._conf_history)

        if c_max - c_min < 1e-6:
            return 0.5  # no variation observed, stay neutral

        return (confidence - c_min) / (c_max - c_min)

    def update(self, yaw, pitch, confidence=None):
        """
        confidence: model confidence in [0, 1] for THIS frame's
            prediction (pass estimator.predict_gaze()'s third
            return value). If None, falls back to the fixed
            measurement_noise (no adaptation) -- pass it for a
            meaningful ablation.
        """
        if yaw is None or pitch is None:
            return None

        try:
            measurement = np.array([float(yaw), float(pitch)], dtype=np.float64)
        except (TypeError, ValueError):
            return None

        if not np.all(np.isfinite(measurement)):
            return None

        # ---- R FROM NORMALIZED CONFIDENCE (not raw, not innovation) ----
        if confidence is not None:
            norm_conf = self._normalize_confidence(float(confidence))
            norm_conf = float(np.clip(norm_conf, 0.0, 1.0))
            adaptive_r = self.r_max - norm_conf * (self.r_max - self.r_min)
        else:
            adaptive_r = self.measurement_noise

        self.current_r = float(adaptive_r)
        self.R = np.eye(2, dtype=np.float64) * self.current_r

        # ---- INITIALIZATION ----
        if self.state is None:
            self.state = measurement.copy()
            return {"yaw": float(self.state[0]), "pitch": float(self.state[1])}

        # ---- PREDICTION ----
        predicted_state = self.A @ self.state
        predicted_covariance = self.A @ self.P @ self.A.T + self.Q

        # ---- INNOVATION (used only for state update, NOT for R) ----
        innovation = measurement - self.H @ predicted_state
        innovation_covariance = self.H @ predicted_covariance @ self.H.T + self.R

        # ---- KALMAN GAIN / UPDATE ----
        kalman_gain = predicted_covariance @ self.H.T @ np.linalg.inv(innovation_covariance)
        self.state = predicted_state + kalman_gain @ innovation

        identity = np.eye(2, dtype=np.float64)
        self.P = (identity - kalman_gain @ self.H) @ predicted_covariance

        return {"yaw": float(self.state[0]), "pitch": float(self.state[1])}


# ============================================================
# ADAPTIVE Q+R KALMAN -- the actual "Adaptive Kalman Filter"
# your report architecture describes: Q reacts to how fast
# things are genuinely changing (innovation-driven), R reacts
# to how much you trust THIS measurement (confidence-driven).
# Use this as your headline "Proposed / AKF" method.
# ============================================================

class AdaptiveQRKalmanGaze(FixedKalmanGaze):

    def __init__(
        self,
        process_noise=0.01,
        measurement_noise=1.0,
        initial_uncertainty=1.0,
        q_min=0.001,
        q_max=0.20,
        q_adaptation_rate=0.10,
        r_min=0.10,
        r_max=10.0,
        conf_window_size=60,
    ):
        super().__init__(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_uncertainty=initial_uncertainty
        )
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.q_adaptation_rate = float(q_adaptation_rate)
        self.r_min = float(r_min)
        self.r_max = float(r_max)

        self.current_q = self.process_noise
        self.current_r = self.measurement_noise

        self.conf_window_size = conf_window_size
        self._conf_history = []

    def _normalize_confidence(self, confidence):
        self._conf_history.append(confidence)
        if len(self._conf_history) > self.conf_window_size:
            self._conf_history.pop(0)

        if len(self._conf_history) < 5:
            return confidence

        c_min = min(self._conf_history)
        c_max = max(self._conf_history)

        if c_max - c_min < 1e-6:
            return 0.5

        return (confidence - c_min) / (c_max - c_min)

    def update(self, yaw, pitch, confidence=None):
        if yaw is None or pitch is None:
            return None

        try:
            measurement = np.array([float(yaw), float(pitch)], dtype=np.float64)
        except (TypeError, ValueError):
            return None

        if not np.all(np.isfinite(measurement)):
            return None

        # ---- R from normalized confidence ----
        if confidence is not None:
            norm_conf = self._normalize_confidence(float(confidence))
            norm_conf = float(np.clip(norm_conf, 0.0, 1.0))
            adaptive_r = self.r_max - norm_conf * (self.r_max - self.r_min)
        else:
            adaptive_r = self.measurement_noise
        self.current_r = float(adaptive_r)
        self.R = np.eye(2, dtype=np.float64) * self.current_r

        # ---- INITIALIZATION ----
        if self.state is None:
            self.state = measurement.copy()
            self.current_q = self.process_noise
            self.Q = np.eye(2, dtype=np.float64) * self.current_q
            return {"yaw": float(self.state[0]), "pitch": float(self.state[1])}

        # ---- Q from innovation (real motion -> react faster) ----
        predicted_state_preview = self.A @ self.state
        innovation_preview = measurement - self.H @ predicted_state_preview
        innovation_magnitude = np.linalg.norm(innovation_preview)

        adaptive_q = self.process_noise + self.q_adaptation_rate * innovation_magnitude ** 2
        adaptive_q = np.clip(adaptive_q, self.q_min, self.q_max)
        self.current_q = float(adaptive_q)
        self.Q = np.eye(2, dtype=np.float64) * self.current_q

        # ---- PREDICTION (with updated Q) ----
        predicted_state = self.A @ self.state
        predicted_covariance = self.A @ self.P @ self.A.T + self.Q

        # ---- INNOVATION / UPDATE (with confidence-based R) ----
        innovation = measurement - self.H @ predicted_state
        innovation_covariance = self.H @ predicted_covariance @ self.H.T + self.R

        kalman_gain = predicted_covariance @ self.H.T @ np.linalg.inv(innovation_covariance)
        self.state = predicted_state + kalman_gain @ innovation

        identity = np.eye(2, dtype=np.float64)
        self.P = (identity - kalman_gain @ self.H) @ predicted_covariance

        return {"yaw": float(self.state[0]), "pitch": float(self.state[1])}