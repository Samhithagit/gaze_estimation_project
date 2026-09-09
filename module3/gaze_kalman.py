import numpy as np


# ============================================================
# FIXED KALMAN
# ============================================================

class FixedKalmanGaze:
    """
    Constant-velocity Kalman filter for screen-space gaze.

    State:

        [x, y, vx, vy]

    Measurement:

        [screen_x, screen_y]

    The filter assumes that gaze position evolves with
    approximately constant velocity between frames.
    """

    def __init__(
        self,
        process_noise=25.0,
        measurement_noise=100.0,
        initial_uncertainty=100.0,
        dt=1.0
    ):

        self.process_noise = float(
            process_noise
        )

        self.measurement_noise = float(
            measurement_noise
        )

        self.initial_uncertainty = float(
            initial_uncertainty
        )

        self.dt = float(dt)

        # ----------------------------------------------------
        # STATE TRANSITION MATRIX
        #
        # x'  = x + vx * dt
        # y'  = y + vy * dt
        # vx' = vx
        # vy' = vy
        # ----------------------------------------------------

        self.A = np.array([

            [1.0, 0.0, self.dt, 0.0],

            [0.0, 1.0, 0.0, self.dt],

            [0.0, 0.0, 1.0, 0.0],

            [0.0, 0.0, 0.0, 1.0]

        ], dtype=np.float64)

        # ----------------------------------------------------
        # MEASUREMENT MATRIX
        #
        # We observe x and y only.
        # ----------------------------------------------------

        self.H = np.array([

            [1.0, 0.0, 0.0, 0.0],

            [0.0, 1.0, 0.0, 0.0]

        ], dtype=np.float64)

        # ----------------------------------------------------
        # Base Q
        #
        # Standard constant-velocity process-noise structure.
        # ----------------------------------------------------

        self.Q = self._build_Q(
            self.process_noise
        )

        # ----------------------------------------------------
        # Measurement noise
        # ----------------------------------------------------

        self.R = (
            np.eye(
                2,
                dtype=np.float64
            )
            * self.measurement_noise
        )

        # ----------------------------------------------------
        # Initial covariance
        # ----------------------------------------------------

        self.P = (
            np.eye(
                4,
                dtype=np.float64
            )
            * self.initial_uncertainty
        )

        self.state = None

        self.last_innovation = None
        self.last_innovation_magnitude = None

    # ========================================================
    # BUILD PROCESS NOISE MATRIX
    # ========================================================

    def _build_Q(self, q_value):

        dt = self.dt

        G = np.array([

            [0.5 * dt ** 2, 0.0],

            [0.0, 0.5 * dt ** 2],

            [dt, 0.0],

            [0.0, dt]

        ], dtype=np.float64)

        return (
            q_value
            * (G @ G.T)
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.P = (
            np.eye(
                4,
                dtype=np.float64
            )
            * self.initial_uncertainty
        )

        self.state = None

        self.last_innovation = None
        self.last_innovation_magnitude = None

    # ========================================================
    # MEASUREMENT
    # ========================================================

    def _get_measurement(
        self,
        screen_x,
        screen_y
    ):

        try:

            measurement = np.array([

                float(screen_x),
                float(screen_y)

            ], dtype=np.float64)

        except (
            TypeError,
            ValueError
        ):

            return None

        if not np.all(
            np.isfinite(measurement)
        ):

            return None

        return measurement

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        screen_x,
        screen_y,
        confidence=None,
        head_motion=None,
        measurement_valid=True
    ):

        measurement = self._get_measurement(
            screen_x,
            screen_y
        )

        if measurement is None:

            return None

        # ----------------------------------------------------
        # INITIALIZATION
        # ----------------------------------------------------

        if self.state is None:

            self.state = np.array([

                measurement[0],
                measurement[1],
                0.0,
                0.0

            ], dtype=np.float64)

            return {

                "x": float(self.state[0]),
                "y": float(self.state[1]),
                "vx": float(self.state[2]),
                "vy": float(self.state[3]),
                "innovation": 0.0

            }

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predicted_state = (
            self.A @ self.state
        )

        predicted_covariance = (
            self.A
            @ self.P
            @ self.A.T
            + self.Q
        )

        # ----------------------------------------------------
        # INVALID MEASUREMENT
        #
        # Prediction only.
        # ----------------------------------------------------

        if not measurement_valid:

            self.state = predicted_state

            self.P = predicted_covariance

            return {

                "x": float(self.state[0]),
                "y": float(self.state[1]),
                "vx": float(self.state[2]),
                "vy": float(self.state[3]),
                "innovation": None

            }

        # ----------------------------------------------------
        # INNOVATION
        # ----------------------------------------------------

        innovation = (

            measurement
            - self.H @ predicted_state

        )

        innovation_magnitude = float(
            np.linalg.norm(
                innovation
            )
        )

        self.last_innovation = (
            innovation.copy()
        )

        self.last_innovation_magnitude = (
            innovation_magnitude
        )

        # ----------------------------------------------------
        # INNOVATION COVARIANCE
        # ----------------------------------------------------

        innovation_covariance = (

            self.H
            @ predicted_covariance
            @ self.H.T
            + self.R

        )

        # ----------------------------------------------------
        # KALMAN GAIN
        # ----------------------------------------------------

        kalman_gain = (

            predicted_covariance
            @ self.H.T
            @ np.linalg.inv(
                innovation_covariance
            )

        )

        # ----------------------------------------------------
        # STATE UPDATE
        # ----------------------------------------------------

        self.state = (

            predicted_state
            + kalman_gain @ innovation

        )

        # ----------------------------------------------------
        # COVARIANCE UPDATE
        # ----------------------------------------------------

        identity = np.eye(
            4,
            dtype=np.float64
        )

        self.P = (

            identity
            - kalman_gain @ self.H

        ) @ predicted_covariance

        return {

            "x": float(self.state[0]),
            "y": float(self.state[1]),
            "vx": float(self.state[2]),
            "vy": float(self.state[3]),
            "innovation": innovation_magnitude

        }


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