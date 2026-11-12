import time
from collections import deque

import numpy as np




class OneEuroScalar:
    """One Euro low-pass filter for one scalar coordinate."""

    def __init__(self, min_cutoff=1.5, beta=0.01, d_cutoff=1.5):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self.t = None
        self.x = None
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, value, timestamp):
        value = float(value)
        timestamp = float(timestamp)
        if self.t is None:
            self.t, self.x, self.dx = timestamp, value, 0.0
            return value
        dt = max(1e-3, timestamp - self.t)
        self.t = timestamp
        a_d = self._alpha(self.d_cutoff, dt)
        derivative = (value - self.x) / dt
        self.dx = a_d * derivative + (1.0 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        alpha = self._alpha(cutoff, dt)
        self.x = alpha * value + (1.0 - alpha) * self.x
        return float(self.x)


class OneEuroGaze:
    """Independent One Euro filters for screen-space x and y coordinates."""

    def __init__(self, min_cutoff=1.5, beta=0.01, d_cutoff=1.5):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self.fx = OneEuroScalar(self.min_cutoff, self.beta, self.d_cutoff)
        self.fy = OneEuroScalar(self.min_cutoff, self.beta, self.d_cutoff)

    def update(self, x, y, timestamp=0.0):
        try:
            return (self.fx.update(x, timestamp), self.fy.update(y, timestamp))
        except (TypeError, ValueError, OverflowError):
            return None


# ============================================================
# FIXED KALMAN
# ============================================================

class FixedKalmanGaze:
    """Fixed constant-velocity gaze Kalman filter following the base paper.

    Paper-reported parameters:
      process-noise density = 8e-4
      measurement variance = 2e-1

    State is [x, y, vx, vy]; measurements contain [screen_x, screen_y].
    The paper specifies a dynamically adjusted time step but does not report
    the initial covariance or spell out every element of Q. This class uses
    the standard continuous white-acceleration covariance discretization.

    Compatibility note: adaptive subclasses retain their previous algorithm,
    matrices, parameters, and dictionary return format. Only direct instances
    of FixedKalmanGaze use the paper-style implementation below.
    """

    def __init__(
        self,
        process_noise=8e-4,
        measurement_noise=0.2,
        initial_uncertainty=1.0,
        dt=1.0,
        process_noise_density=None,
    ):
        if process_noise_density is not None:
            process_noise = process_noise_density
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_uncertainty = float(initial_uncertainty)
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.process_noise < 0:
            raise ValueError("process_noise must be non-negative")
        if self.measurement_noise <= 0:
            raise ValueError("measurement_noise must be positive")
        if self.initial_uncertainty <= 0:
            raise ValueError("initial_uncertainty must be positive")

        self.A = self._transition(self.dt)
        self.H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]], dtype=np.float64)
        self.Q = self._build_paper_Q(self.dt)
        self.R = np.eye(2, dtype=np.float64) * self.measurement_noise
        self.P = np.eye(4, dtype=np.float64) * self.initial_uncertainty
        self.state = None
        self.last_innovation = None
        self.last_innovation_magnitude = None
        self._last_timestamp = None

    @staticmethod
    def _transition(dt):
        return np.array([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ], dtype=np.float64)

    def _build_paper_Q(self, dt):
        # Discrete white-acceleration process noise for x and y axes.
        d2, d3, d4 = dt**2, dt**3, dt**4
        q = self.process_noise
        return q * np.array([
            [d4/4, 0., d3/2, 0.],
            [0., d4/4, 0., d3/2],
            [d3/2, 0., d2, 0.],
            [0., d3/2, 0., d2],
        ], dtype=np.float64)

    def _build_Q(self, q_value):
        # Kept unchanged in meaning for adaptive subclasses' existing model.
        dt = self.dt
        G = np.array([
            [0.5 * dt**2, 0.], [0., 0.5 * dt**2],
            [dt, 0.], [0., dt],
        ], dtype=np.float64)
        return float(q_value) * (G @ G.T)

    def reset(self):
        self.P = np.eye(4, dtype=np.float64) * self.initial_uncertainty
        self.state = None
        self.last_innovation = None
        self.last_innovation_magnitude = None
        self._last_timestamp = None

    def _get_measurement(self, screen_x, screen_y):
        try:
            z = np.array([float(screen_x), float(screen_y)], dtype=np.float64)
        except (TypeError, ValueError):
            return None
        return z if np.all(np.isfinite(z)) else None

    def _format_output(self):
        x, y = float(self.state[0]), float(self.state[1])
        if type(self) is FixedKalmanGaze:
            return (x, y)
        return {
            "x": x, "y": y,
            "vx": float(self.state[2]), "vy": float(self.state[3]),
            "innovation": self.last_innovation_magnitude,
        }

    def update(self, screen_x, screen_y, confidence=None, head_motion=None,
               measurement_valid=True, timestamp=None, dt=None):
        # Preserve the adaptive filters' prior implementation/API exactly.
        if type(self) is not FixedKalmanGaze:
            if dt is not None:
                dt_value = float(dt)
                if np.isfinite(dt_value) and dt_value > 0:
                    self.dt = dt_value
                    self.A = self._transition(dt_value)
                    # Preserve the current adaptive Q level when present; rebuild
                    # its matrix for the new dt. Adaptive-R-only uses base process noise.
                    q_value = getattr(self, "current_q", self.process_noise)
                    self.Q = self._build_Q(q_value)
            measurement = self._get_measurement(screen_x, screen_y) if measurement_valid else None
            if self.state is None:
                if measurement is None:
                    return None
                self.state = np.array([measurement[0], measurement[1], 0., 0.], dtype=np.float64)
                return self._format_output()
            predicted_state = self.A @ self.state
            predicted_covariance = self.A @ self.P @ self.A.T + self.Q
            if measurement is None:
                self.state, self.P = predicted_state, predicted_covariance
                self.last_innovation = self.last_innovation_magnitude = None
                return self._format_output()
            innovation = measurement - self.H @ predicted_state
            S = self.H @ predicted_covariance @ self.H.T + self.R
            K = np.linalg.solve(S, self.H @ predicted_covariance).T
            self.state = predicted_state + K @ innovation
            I_KH = np.eye(4, dtype=np.float64) - K @ self.H
            self.P = I_KH @ predicted_covariance @ I_KH.T + K @ self.R @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            self.last_innovation = innovation.copy()
            self.last_innovation_magnitude = float(np.linalg.norm(innovation))
            return self._format_output()

        # The paper describes dynamic dt. Use a supplied timestamp when present;
        # otherwise use elapsed wall-clock time, with the configured dt on frame 1.
        import time
        now = float(timestamp) if timestamp is not None else time.monotonic()
        if self._last_timestamp is None:
            dt = self.dt
        else:
            dt = now - self._last_timestamp
            if not np.isfinite(dt) or dt <= 0:
                dt = self.dt
        self._last_timestamp = now
        self.A = self._transition(dt)
        self.Q = self._build_paper_Q(dt)

        z = self._get_measurement(screen_x, screen_y) if measurement_valid else None
        if self.state is None:
            if z is None:
                return None
            self.state = np.array([z[0], z[1], 0., 0.], dtype=np.float64)
            return self._format_output()

        x_pred = self.A @ self.state
        P_pred = self.A @ self.P @ self.A.T + self.Q
        if z is None:
            self.state, self.P = x_pred, P_pred
            self.last_innovation = self.last_innovation_magnitude = None
            return self._format_output()

        innovation = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ P_pred).T
        self.state = x_pred + K @ innovation
        I_KH = np.eye(4, dtype=np.float64) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.last_innovation = innovation.copy()
        self.last_innovation_magnitude = float(np.linalg.norm(innovation))
        return self._format_output()




# ============================================================
# FIXED PIPELINE: 5-FRAME MOVING MEDIAN -> FIXED KALMAN
# ============================================================

class MovingMedianGaze:
    """Moving median over the latest five gaze-coordinate measurements."""

    def __init__(self, window_size=5):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = int(window_size)
        self.buffer = deque(maxlen=self.window_size)

    def reset(self):
        self.buffer.clear()

    def update(self, x, y):
        try:
            measurement = np.array([float(x), float(y)], dtype=np.float64)
        except (TypeError, ValueError):
            return None

        if not np.all(np.isfinite(measurement)):
            return None

        self.buffer.append(measurement)
        samples = np.asarray(self.buffer, dtype=np.float64)
        median = np.median(samples, axis=0)
        return float(median[0]), float(median[1])


class FixedKalmanGazePipeline:
    """Apply the 5-frame moving median first, then the fixed Kalman filter.

    Use this wrapper for the paper-style fixed baseline. Adaptive Kalman
    classes remain separate and are not routed through the median filter.
    """

    def __init__(self, window_size=5, **kalman_kwargs):
        self.median = MovingMedianGaze(window_size=window_size)
        self.kalman = FixedKalmanGaze(**kalman_kwargs)

    def reset(self):
        self.median.reset()
        self.kalman.reset()

    def update(self, screen_x, screen_y, measurement_valid=True, timestamp=None):
        if not measurement_valid:
            # Let the Kalman filter advance using prediction only; do not add
            # an invalid measurement to the moving-median buffer.
            return self.kalman.update(
                screen_x, screen_y,
                measurement_valid=False,
                timestamp=timestamp,
            )

        median_measurement = self.median.update(screen_x, screen_y)
        if median_measurement is None:
            return self.kalman.update(
                screen_x, screen_y,
                measurement_valid=False,
                timestamp=timestamp,
            )

        median_x, median_y = median_measurement
        return self.kalman.update(
            median_x, median_y,
            measurement_valid=True,
            timestamp=timestamp,
        )


# ============================================================
# ADAPTIVE Q
# ============================================================

class AdaptiveQKalmanGaze(FixedKalmanGaze):
    """
    Adaptive process-noise Kalman filter.

    Q responds to frame-to-frame HEAD-POSE CHANGE.

        Q_t = Q_base * (1 + alpha * Delta_head)

    Larger head motion:
        -> larger Q
        -> filter can respond faster

    Smaller head motion:
        -> smaller Q
        -> stronger temporal smoothing
    """

    def __init__(
        self,
        process_noise=25.0,
        measurement_noise=100.0,
        initial_uncertainty=100.0,
        q_alpha=1.0,
        q_min=0.1,
        q_max=2500.0,
        dt=1.0
    ):

        super().__init__(
            process_noise,
            measurement_noise,
            initial_uncertainty,
            dt
        )

        self.q_alpha = float(
            q_alpha
        )

        self.q_min = float(
            q_min
        )

        self.q_max = float(
            q_max
        )

        self.current_q = (
            self.process_noise
        )

    def reset(self):

        super().reset()

        self.current_q = (
            self.process_noise
        )

    def update(
        self,
        screen_x,
        screen_y,
        confidence=None,
        head_motion=0.0,
        measurement_valid=True
    ):

        if head_motion is None:
            head_motion = 0.0

        head_motion = max(
            0.0,
            float(head_motion)
        )

        # ----------------------------------------------------
        # ADAPT Q FROM HEAD MOTION
        #
        # Q_t = Q_base(1 + alpha * Delta_head)
        # ----------------------------------------------------

        q_value = (

            self.process_noise
            * (
                1.0
                + self.q_alpha
                * head_motion
            )

        )

        self.current_q = float(
            np.clip(
                q_value,
                self.q_min,
                self.q_max
            )
        )

        self.Q = self._build_Q(
            self.current_q
        )

        return super().update(
            screen_x,
            screen_y,
            confidence=confidence,
            head_motion=head_motion,
            measurement_valid=measurement_valid
        )


# ============================================================
# ADAPTIVE R
# ============================================================

class AdaptiveRKalmanGaze(FixedKalmanGaze):
    """
    Adaptive measurement-noise Kalman filter.

    R responds to L2CS entropy-based confidence.

        R_t = R_base * exp(k * (1 - C_t))

    High confidence:
        -> smaller R
        -> trust measurement more

    Low confidence:
        -> larger R
        -> trust prediction more
    """

    def __init__(
        self,
        process_noise=25.0,
        measurement_noise=100.0,
        initial_uncertainty=100.0,
        r_k=2.0,
        r_min=25.0,
        r_max=1000.0,
        dt=1.0
    ):

        super().__init__(
            process_noise,
            measurement_noise,
            initial_uncertainty,
            dt
        )

        self.r_k = float(
            r_k
        )

        self.r_min = float(
            r_min
        )

        self.r_max = float(
            r_max
        )

        self.current_r = (
            self.measurement_noise
        )

    def reset(self):

        super().reset()

        self.current_r = (
            self.measurement_noise
        )

    def update(
        self,
        screen_x,
        screen_y,
        confidence=None,
        head_motion=None,
        measurement_valid=True
    ):

        if confidence is None:
            confidence = 0.5

        confidence = float(
            np.clip(
                confidence,
                0.0,
                1.0
            )
        )

        # ----------------------------------------------------
        # ADAPT R FROM CONFIDENCE
        #
        # R_t = R_base * exp(k(1-C_t))
        # ----------------------------------------------------

        r_value = (

            self.measurement_noise
            * np.exp(
                self.r_k
                * (1.0 - confidence)
            )

        )

        self.current_r = float(
            np.clip(
                r_value,
                self.r_min,
                self.r_max
            )
        )

        self.R = (

            np.eye(
                2,
                dtype=np.float64
            )
            * self.current_r

        )

        return super().update(
            screen_x,
            screen_y,
            confidence=confidence,
            head_motion=head_motion,
            measurement_valid=measurement_valid
        )


# ============================================================
# ADAPTIVE Q + R
# ============================================================

class AdaptiveQRKalmanGaze(FixedKalmanGaze):
    """
    Proposed adaptive Kalman filter.

    Q responds to head motion.

    R responds to L2CS confidence.

        Q_t = Q_base(1 + alpha * Delta_head)

        R_t = R_base * exp(k(1-C_t))
    """

    def __init__(
        self,
        process_noise=25.0,
        measurement_noise=100.0,
        initial_uncertainty=100.0,
        q_alpha=1.0,
        q_min=0.1,
        q_max=2500.0,
        r_k=2.0,
        r_min=25.0,
        r_max=1000.0,
        dt=1.0
    ):

        super().__init__(
            process_noise,
            measurement_noise,
            initial_uncertainty,
            dt
        )

        self.q_alpha = float(
            q_alpha
        )

        self.q_min = float(
            q_min
        )

        self.q_max = float(
            q_max
        )

        self.r_k = float(
            r_k
        )

        self.r_min = float(
            r_min
        )

        self.r_max = float(
            r_max
        )

        self.current_q = (
            self.process_noise
        )

        self.current_r = (
            self.measurement_noise
        )

    def reset(self):

        super().reset()

        self.current_q = (
            self.process_noise
        )

        self.current_r = (
            self.measurement_noise
        )

    def update(
        self,
        screen_x,
        screen_y,
        confidence=None,
        head_motion=0.0,
        measurement_valid=True
    ):

        # ----------------------------------------------------
        # HEAD MOTION
        # ----------------------------------------------------

        if head_motion is None:
            head_motion = 0.0

        head_motion = max(
            0.0,
            float(head_motion)
        )

        # ----------------------------------------------------
        # ADAPT Q
        # ----------------------------------------------------

        q_value = (

            self.process_noise
            * (
                1.0
                + self.q_alpha
                * head_motion
            )

        )

        self.current_q = float(
            np.clip(
                q_value,
                self.q_min,
                self.q_max
            )
        )

        self.Q = self._build_Q(
            self.current_q
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if confidence is None:
            confidence = 0.5

        confidence = float(
            np.clip(
                confidence,
                0.0,
                1.0
            )
        )

        # ----------------------------------------------------
        # ADAPT R
        # ----------------------------------------------------

        r_value = (

            self.measurement_noise
            * np.exp(
                self.r_k
                * (1.0 - confidence)
            )

        )

        self.current_r = float(
            np.clip(
                r_value,
                self.r_min,
                self.r_max
            )
        )

        self.R = (

            np.eye(
                2,
                dtype=np.float64
            )
            * self.current_r

        )

        # ----------------------------------------------------
        # KALMAN UPDATE
        # ----------------------------------------------------

        return super().update(
            screen_x,
            screen_y,
            confidence=confidence,
            head_motion=head_motion,
            measurement_valid=measurement_valid
        )


# ============================================================
# ADDITIONAL FILTERS (original classes above are unchanged)
# ============================================================

# ============================================================
# NEW METHODS (separate from the paper-style fixed KF)
# ============================================================

class DoubleExponentialSmoothingGaze:
    """Holt double exponential smoothing for 2-D gaze coordinates."""
    def __init__(self, alpha=0.28, beta=0.045, max_trend_step=35.0):
        if not 0.0 < alpha <= 1.0 or not 0.0 <= beta <= 1.0:
            raise ValueError("alpha must be in (0,1] and beta in [0,1]")
        self.alpha, self.beta = float(alpha), float(beta)
        self.max_trend_step = float(max_trend_step)
        self.reset()

    def reset(self):
        self.level = None
        self.trend = np.zeros(2, dtype=np.float64)

    def update(self, x, y):
        z = np.asarray([x, y], dtype=np.float64)
        if not np.all(np.isfinite(z)):
            return None
        if self.level is None:
            self.level = z.copy()
            self.trend[:] = 0.0
            return float(z[0]), float(z[1])
        prediction = self.level + self.trend
        residual = z - prediction
        magnitude = float(np.linalg.norm(residual))
        if self.max_trend_step > 0 and magnitude > self.max_trend_step:
            residual *= self.max_trend_step / magnitude
        old_level = self.level.copy()
        self.level = self.alpha * z + (1.0 - self.alpha) * prediction
        self.trend = self.beta * (self.level - old_level) + (1.0 - self.beta) * self.trend
        return float(self.level[0]), float(self.level[1])



class AdaptiveAlphaBetaGaze:
    """Confidence- and innovation-gated alpha-beta gaze tracker."""
    def __init__(self, alpha_min=0.08, alpha_max=0.55, beta=0.045,
                 confidence_power=1.8, outlier_scale=65.0, max_speed=45.0):
        self.alpha_min, self.alpha_max = float(alpha_min), float(alpha_max)
        self.beta = float(beta)
        self.confidence_power = float(confidence_power)
        self.outlier_scale = float(outlier_scale)
        self.max_speed = float(max_speed)
        self.reset()

    def reset(self):
        self.position = None
        self.velocity = np.zeros(2, dtype=np.float64)

    def update(self, x, y, confidence=None, head_motion=None):
        z = np.asarray([x, y], dtype=np.float64)
        if not np.all(np.isfinite(z)):
            return None
        confidence = 0.5 if confidence is None else float(np.clip(confidence, 0.0, 1.0))
        if self.position is None:
            self.position = z.copy()
            self.velocity[:] = 0.0
            return float(z[0]), float(z[1])
        prediction = self.position + self.velocity
        innovation = z - prediction
        magnitude = float(np.linalg.norm(innovation))
        conf_gain = confidence ** self.confidence_power
        outlier_gain = 1.0 / (1.0 + (magnitude / max(self.outlier_scale, 1e-6)) ** 2)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * conf_gain * outlier_gain
        self.position = prediction + alpha * innovation
        self.velocity += self.beta * alpha * innovation
        speed = float(np.linalg.norm(self.velocity))
        if speed > self.max_speed:
            self.velocity *= self.max_speed / speed
        return float(self.position[0]), float(self.position[1])


class ConfidenceAdaptiveEMAGaze:
    """EMA whose gain increases with confidence and credible movement."""
    def __init__(self, min_alpha=0.12, max_alpha=0.52, confidence_power=1.5,
                 movement_scale=35.0, max_step=45.0):
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.confidence_power = float(confidence_power)
        self.movement_scale = float(movement_scale)
        self.max_step = float(max_step)
        self.reset()

    def reset(self):
        self.value = None

    def update(self, x, y, confidence=None):
        z = np.asarray([x, y], dtype=np.float64)
        if not np.all(np.isfinite(z)):
            return None
        if self.value is None:
            self.value = z.copy()
            return float(z[0]), float(z[1])
        confidence = 0.5 if confidence is None else float(np.clip(confidence, 0.0, 1.0))
        delta = z - self.value
        distance = float(np.linalg.norm(delta))
        # A large jump only increases responsiveness gradually, reducing outlier sensitivity.
        movement = distance / (distance + max(self.movement_scale, 1e-6))
        trust = confidence ** self.confidence_power
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (0.75 * trust + 0.25 * movement)
        if self.max_step > 0 and distance > self.max_step:
            delta *= self.max_step / distance
        self.value = self.value + alpha * delta
        return float(self.value[0]), float(self.value[1])


class InnovationClippedHoltGaze(DoubleExponentialSmoothingGaze):
    """Holt smoother that clips large measurement innovations before updating level/trend."""
    def __init__(self, alpha=0.24, beta=0.035, innovation_limit=28.0, max_trend_step=22.0):
        super().__init__(alpha=alpha, beta=beta, max_trend_step=max_trend_step)
        self.innovation_limit = float(innovation_limit)

    def update(self, x, y):
        z = np.asarray([x, y], dtype=np.float64)
        if not np.all(np.isfinite(z)):
            return None
        if self.level is None:
            self.level = z.copy()
            self.trend[:] = 0.0
            return float(z[0]), float(z[1])
        prediction = self.level + self.trend
        residual = z - prediction
        magnitude = float(np.linalg.norm(residual))
        if self.innovation_limit > 0 and magnitude > self.innovation_limit:
            residual *= self.innovation_limit / magnitude
        robust_z = prediction + residual
        old_level = self.level.copy()
        self.level = self.alpha * robust_z + (1.0 - self.alpha) * prediction
        self.trend = self.beta * (self.level - old_level) + (1.0 - self.beta) * self.trend
        return float(self.level[0]), float(self.level[1])
