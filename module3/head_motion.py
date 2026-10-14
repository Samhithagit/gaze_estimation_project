import math


class HeadMotionEstimator:
    """
    Timestamp-aware head-motion estimator.

    Uses head yaw and pitch.

    Motion is calculated as angular displacement divided by
    elapsed time, then normalized to a 30-FPS reference frame.

    Long timestamp gaps are rejected to avoid interpreting
    widely separated samples as ordinary frame-to-frame motion.
    """

    def __init__(
        self,
        reference_fps=30.0,
        max_gap_seconds=0.20,
    ):
        if reference_fps <= 0:
            raise ValueError(
                "reference_fps must be greater than 0"
            )

        if max_gap_seconds <= 0:
            raise ValueError(
                "max_gap_seconds must be greater than 0"
            )

        self.reference_fps = float(reference_fps)
        self.max_gap_seconds = float(max_gap_seconds)

        self.previous_yaw = None
        self.previous_pitch = None
        self.previous_timestamp = None

        self.last_dt = None
        self.last_angular_displacement = 0.0
        self.last_angular_velocity = 0.0
        self.last_motion = 0.0

    def reset(self):
        self.previous_yaw = None
        self.previous_pitch = None
        self.previous_timestamp = None

        self.last_dt = None
        self.last_angular_displacement = 0.0
        self.last_angular_velocity = 0.0
        self.last_motion = 0.0

    def update(self, head_pose, timestamp):
        if head_pose is None:
            self.reset()
            return 0.0

        try:
            current_yaw = float(head_pose["yaw"])
            current_pitch = float(head_pose["pitch"])
            current_timestamp = float(timestamp)
        except (KeyError, TypeError, ValueError):
            self.reset()
            return 0.0

        if not all(
            math.isfinite(value)
            for value in (
                current_yaw,
                current_pitch,
                current_timestamp,
            )
        ):
            self.reset()
            return 0.0

        if self.previous_timestamp is None:
            self._store_previous(
                current_yaw,
                current_pitch,
                current_timestamp,
            )
            self._clear_motion()
            return 0.0

        dt = current_timestamp - self.previous_timestamp

        if not math.isfinite(dt) or dt <= 0.0:
            self._store_previous(
                current_yaw,
                current_pitch,
                current_timestamp,
            )
            self._clear_motion()
            return 0.0

        if dt > self.max_gap_seconds:
            # Re-establish the reference pose after a long gap.
            self._store_previous(
                current_yaw,
                current_pitch,
                current_timestamp,
            )
            self._clear_motion()
            self.last_dt = dt
            return 0.0

        delta_yaw = current_yaw - self.previous_yaw
        delta_pitch = current_pitch - self.previous_pitch

        displacement = math.hypot(delta_yaw, delta_pitch)
        velocity = displacement / dt
        motion = velocity / self.reference_fps

        self._store_previous(
            current_yaw,
            current_pitch,
            current_timestamp,
        )

        self.last_dt = dt
        self.last_angular_displacement = displacement
        self.last_angular_velocity = velocity
        self.last_motion = motion

        return float(motion)

    def _store_previous(self, yaw, pitch, timestamp):
        self.previous_yaw = yaw
        self.previous_pitch = pitch
        self.previous_timestamp = timestamp

    def _clear_motion(self):
        self.last_angular_displacement = 0.0
        self.last_angular_velocity = 0.0
        self.last_motion = 0.0

    def get_debug_info(self):
        return {
            "dt": self.last_dt,
            "angular_displacement": self.last_angular_displacement,
            "angular_velocity": self.last_angular_velocity,
            "motion": self.last_motion,
        }
