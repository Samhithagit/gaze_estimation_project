import math


class HeadMotionEstimator:
    """
    Computes frame-to-frame head motion from
    smoothed head pose.

    Δhead_t =
        |yaw_t - yaw_(t-1)|
        +
        |pitch_t - pitch_(t-1)|

    Motion is calculated from filtered head pose.
    """

    def __init__(self, max_gap_frames=1):

        self.previous_yaw = None
        self.previous_pitch = None

        self.frames_since_previous = 0

        self.max_gap_frames = int(
            max_gap_frames
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.previous_yaw = None
        self.previous_pitch = None

        self.frames_since_previous = 0

    # ========================================================
    # UPDATE
    # ========================================================

    def update(self, head_pose):

        if head_pose is None:

            self.frames_since_previous += 1

            return None

        current_yaw = float(
            head_pose["yaw"]
        )

        current_pitch = float(
            head_pose["pitch"]
        )

        if not (
            math.isfinite(current_yaw)
            and math.isfinite(current_pitch)
        ):

            self.reset()

            return None

        # ----------------------------------------------------
        # First valid pose
        # ----------------------------------------------------

        if (
            self.previous_yaw is None
            or self.previous_pitch is None
        ):

            self.previous_yaw = current_yaw
            self.previous_pitch = current_pitch

            self.frames_since_previous = 0

            return 0.0

        # ----------------------------------------------------
        # Avoid artificial motion after long gaps
        # ----------------------------------------------------

        if (
            self.frames_since_previous
            > self.max_gap_frames
        ):

            self.previous_yaw = current_yaw
            self.previous_pitch = current_pitch

            self.frames_since_previous = 0

            return 0.0

        # ----------------------------------------------------
        # Head motion
        #
        # Δhead_t =
        # |yaw_t - yaw_(t-1)|
        # +
        # |pitch_t - pitch_(t-1)|
        # ----------------------------------------------------

        delta_yaw = abs(
            current_yaw
            - self.previous_yaw
        )

        delta_pitch = abs(
            current_pitch
            - self.previous_pitch
        )

        head_motion = (
            delta_yaw
            + delta_pitch
        )

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        self.previous_yaw = current_yaw
        self.previous_pitch = current_pitch

        self.frames_since_previous = 0

        return float(
            head_motion
        )