import math


class HeadEyeFusion:
    """
    Head-eye gaze fusion.

    The eye estimator provides gaze direction relative
    to the eye/head coordinate system.

    Head pose provides the orientation of the head
    relative to the camera.

    The fused gaze is obtained by combining the two.

    Current formulation:

        fused_yaw =
            gaze_weight * gaze_yaw
            +
            head_weight * yaw_sign * head_yaw

        fused_pitch =
            gaze_weight * gaze_pitch
            +
            head_weight * pitch_sign * head_pitch

    The signs are configurable because coordinate
    conventions can differ between the gaze estimator
    and head-pose estimator.
    """

    def __init__(
        self,
        gaze_weight=1.0,
        head_weight=1.0,
        yaw_sign=1.0,
        pitch_sign=1.0
    ):

        self.gaze_weight = float(gaze_weight)
        self.head_weight = float(head_weight)

        self.yaw_sign = float(yaw_sign)
        self.pitch_sign = float(pitch_sign)

    # ========================================================
    # ANGLE NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_angle(angle):

        return float(
            (angle + 180.0) % 360.0 - 180.0
        )

    # ========================================================
    # FUSION
    # ========================================================

    def fuse(
        self,
        gaze_yaw,
        gaze_pitch,
        head_yaw,
        head_pitch
    ):

        values = [
            gaze_yaw,
            gaze_pitch,
            head_yaw,
            head_pitch
        ]

        if not all(
            value is not None
            and math.isfinite(float(value))
            for value in values
        ):
            return None

        gaze_yaw = float(gaze_yaw)
        gaze_pitch = float(gaze_pitch)

        head_yaw = float(head_yaw)
        head_pitch = float(head_pitch)

        fused_yaw = (
            self.gaze_weight * gaze_yaw
            +
            self.head_weight
            * self.yaw_sign
            * head_yaw
        )

        fused_pitch = (
            self.gaze_weight * gaze_pitch
            +
            self.head_weight
            * self.pitch_sign
            * head_pitch
        )

        fused_yaw = self.normalize_angle(
            fused_yaw
        )

        fused_pitch = self.normalize_angle(
            fused_pitch
        )

        return {
            "yaw": fused_yaw,
            "pitch": fused_pitch
        }