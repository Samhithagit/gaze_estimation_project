import numpy as np


# ============================================================
# FIXED KALMAN
# ============================================================

class FixedKalmanGaze:
    """
    Fixed-parameter Kalman filter for screen gaze coordinates.

    State:
        x = [screen_x, screen_y]

    Measurement:
        z = [measured_screen_x, measured_screen_y]

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

        self.Q = (
            np.eye(2, dtype=np.float64)
            * self.process_noise
        )

        self.R = (
            np.eye(2, dtype=np.float64)
            * self.measurement_noise
        )

        self.P = (
            np.eye(2, dtype=np.float64)
            * self.initial_uncertainty
        )

        self.state = None


    def reset(self):

        self.P = (
            np.eye(2, dtype=np.float64)
            * self.initial_uncertainty
        )

        self.state = None


    def update(
        self,
        screen_x,
        screen_y,
        confidence=None
    ):

        if screen_x is None or screen_y is None:
            return None

        try:

            measurement = np.array(
                [
                    float(screen_x),
                    float(screen_y)
                ],
                dtype=np.float64
            )

        except (TypeError, ValueError):

            return None


        if not np.all(np.isfinite(measurement)):

            return None


        # ----------------------------------------------------
        # INITIALIZATION
        # ----------------------------------------------------

        if self.state is None:

            self.state = measurement.copy()

            return {
                "x": float(self.state[0]),
                "y": float(self.state[1])
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
        # INNOVATION
        # ----------------------------------------------------

        innovation = (
            measurement
            - self.H @ predicted_state
        )

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
            2,
            dtype=np.float64
        )

        self.P = (
            identity
            - kalman_gain @ self.H
        ) @ predicted_covariance


        return {
            "x": float(self.state[0]),
            "y": float(self.state[1])
        }


# ============================================================
# ADAPTIVE Q KALMAN
# ============================================================

class AdaptiveQKalmanGaze(FixedKalmanGaze):

    """
    Adaptive-Q Kalman filter operating on screen coordinates.

    Q increases when the innovation becomes large.

    Large innovation:
        likely rapid gaze movement
        -> increase Q
        -> filter responds faster

    Small innovation:
        stable fixation
        -> decrease Q
        -> stronger smoothing
    """

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
            process_noise,
            measurement_noise,
            initial_uncertainty
        )

        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.adaptation_rate = float(
            adaptation_rate
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
        confidence=None
    ):

        if screen_x is None or screen_y is None:
            return None


        try:

            measurement = np.array(
                [
                    float(screen_x),
                    float(screen_y)
                ],
                dtype=np.float64
            )

        except (TypeError, ValueError):

            return None


        if not np.all(np.isfinite(measurement)):

            return None


        # ----------------------------------------------------
        # INITIALIZATION
        # ----------------------------------------------------

        if self.state is None:

            self.state = measurement.copy()

            return {
                "x": float(self.state[0]),
                "y": float(self.state[1])
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
        # INNOVATION
        # ----------------------------------------------------

        innovation = (
            measurement
            - self.H @ predicted_state
        )


        innovation_magnitude = float(
            np.linalg.norm(innovation)
        )


        # ----------------------------------------------------
        # ADAPT Q
        # ----------------------------------------------------

        q_value = (
            self.process_noise
            + self.adaptation_rate
            * innovation_magnitude ** 2
        )


        self.current_q = float(
            np.clip(
                q_value,
                self.q_min,
                self.q_max
            )
        )


        self.Q = (
            np.eye(2, dtype=np.float64)
            * self.current_q
        )


        # Recalculate prediction covariance
        predicted_covariance = (
            self.A
            @ self.P
            @ self.A.T
            + self.Q
        )


        # ----------------------------------------------------
        # KALMAN UPDATE
        # ----------------------------------------------------

        innovation_covariance = (
            self.H
            @ predicted_covariance
            @ self.H.T
            + self.R
        )


        kalman_gain = (
            predicted_covariance
            @ self.H.T
            @ np.linalg.inv(
                innovation_covariance
            )
        )


        self.state = (
            predicted_state
            + kalman_gain @ innovation
        )


        identity = np.eye(
            2,
            dtype=np.float64
        )

        self.P = (
            identity
            - kalman_gain @ self.H
        ) @ predicted_covariance


        return {
            "x": float(self.state[0]),
            "y": float(self.state[1])
        }


# ============================================================
# ADAPTIVE R KALMAN
# ============================================================

class AdaptiveRKalmanGaze(FixedKalmanGaze):

    """
    Adaptive-R Kalman filter operating on screen coordinates.

    Measurement noise R is controlled by gaze confidence.

    High confidence:
        lower R
        -> trust Ridge measurement more

    Low confidence:
        higher R
        -> trust prediction more
    """

    def __init__(
        self,
        process_noise=0.01,
        measurement_noise=1.0,
        initial_uncertainty=1.0,
        r_min=0.10,
        r_max=10.0
    ):

        super().__init__(
            process_noise,
            measurement_noise,
            initial_uncertainty
        )

        self.r_min = float(r_min)
        self.r_max = float(r_max)


    def update(
        self,
        screen_x,
        screen_y,
        confidence=None
    ):

        if screen_x is None or screen_y is None:
            return None


        try:

            measurement = np.array(
                [
                    float(screen_x),
                    float(screen_y)
                ],
                dtype=np.float64
            )

        except (TypeError, ValueError):

            return None


        if not np.all(np.isfinite(measurement)):

            return None


        if self.state is None:

            self.state = measurement.copy()

            return {
                "x": float(self.state[0]),
                "y": float(self.state[1])
            }


        # ----------------------------------------------------
        # CONFIDENCE → R
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


        # High confidence → low R
        # Low confidence  → high R

        r_value = (
            self.r_max
            - confidence
            * (
                self.r_max
                - self.r_min
            )
        )


        self.R = (
            np.eye(2, dtype=np.float64)
            * r_value
        )


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
        # INNOVATION
        # ----------------------------------------------------

        innovation = (
            measurement
            - self.H @ predicted_state
        )


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
        # UPDATE
        # ----------------------------------------------------

        self.state = (
            predicted_state
            + kalman_gain @ innovation
        )


        identity = np.eye(
            2,
            dtype=np.float64
        )

        self.P = (
            identity
            - kalman_gain @ self.H
        ) @ predicted_covariance


        return {
            "x": float(self.state[0]),
            "y": float(self.state[1])
        }


# ============================================================
# ADAPTIVE Q + R
# ============================================================

class AdaptiveQRKalmanGaze(FixedKalmanGaze):

    """
    Proposed Adaptive Q+R Kalman filter.

    Q:
        adapted from innovation magnitude.

    R:
        adapted from gaze confidence.

    Therefore:

        Head/gaze movement
              ↓
             Q

        Gaze confidence
              ↓
             R

    Both process and measurement uncertainty are adapted.
    """

    def __init__(
        self,
        process_noise=0.01,
        measurement_noise=1.0,
        initial_uncertainty=1.0,
        q_min=0.001,
        q_max=0.20,
        q_adaptation_rate=0.10,
        r_min=0.10,
        r_max=10.0
    ):

        super().__init__(
            process_noise,
            measurement_noise,
            initial_uncertainty
        )

        self.q_min = float(q_min)
        self.q_max = float(q_max)

        self.q_adaptation_rate = float(
            q_adaptation_rate
        )

        self.r_min = float(r_min)
        self.r_max = float(r_max)

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
        confidence=None
    ):

        if screen_x is None or screen_y is None:
            return None


        try:

            measurement = np.array(
                [
                    float(screen_x),
                    float(screen_y)
                ],
                dtype=np.float64
            )

        except (TypeError, ValueError):

            return None


        if not np.all(np.isfinite(measurement)):

            return None


        if self.state is None:

            self.state = measurement.copy()

            return {
                "x": float(self.state[0]),
                "y": float(self.state[1])
            }


        # ====================================================
        # PREDICTION
        # ====================================================

        predicted_state = (
            self.A @ self.state
        )


        # ====================================================
        # INNOVATION
        # ====================================================

        innovation = (
            measurement
            - self.H @ predicted_state
        )


        innovation_magnitude = float(
            np.linalg.norm(innovation)
        )


        # ====================================================
        # ADAPT Q
        # ====================================================

        q_value = (
            self.process_noise
            +
            self.q_adaptation_rate
            *
            innovation_magnitude ** 2
        )


        self.current_q = float(
            np.clip(
                q_value,
                self.q_min,
                self.q_max
            )
        )


        self.Q = (
            np.eye(2, dtype=np.float64)
            * self.current_q
        )


        # ====================================================
        # ADAPT R
        # ====================================================

        if confidence is None:

            confidence = 0.5


        confidence = float(
            np.clip(
                confidence,
                0.0,
                1.0
            )
        )


        r_value = (
            self.r_max
            -
            confidence
            *
            (
                self.r_max
                -
                self.r_min
            )
        )


        self.current_r = float(
            r_value
        )


        self.R = (
            np.eye(2, dtype=np.float64)
            * self.current_r
        )


        # ====================================================
        # PREDICT COVARIANCE
        # ====================================================

        predicted_covariance = (
            self.A
            @ self.P
            @ self.A.T
            + self.Q
        )


        # ====================================================
        # INNOVATION COVARIANCE
        # ====================================================

        innovation_covariance = (
            self.H
            @ predicted_covariance
            @ self.H.T
            + self.R
        )


        # ====================================================
        # KALMAN GAIN
        # ====================================================

        kalman_gain = (
            predicted_covariance
            @ self.H.T
            @ np.linalg.inv(
                innovation_covariance
            )
        )


        # ====================================================
        # STATE UPDATE
        # ====================================================

        self.state = (
            predicted_state
            + kalman_gain @ innovation
        )


        # ====================================================
        # COVARIANCE UPDATE
        # ====================================================

        identity = np.eye(
            2,
            dtype=np.float64
        )

        self.P = (
            identity
            - kalman_gain @ self.H
        ) @ predicted_covariance


        return {
            "x": float(self.state[0]),
            "y": float(self.state[1])
        }