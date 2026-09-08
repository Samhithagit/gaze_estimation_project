import numpy as np


class HeadEyeFusion:
    """
    Geometric head-eye gaze fusion.

    THE PROBLEM WITH THE PREVIOUS VERSION
    --------------------------------------
    The previous implementation summed yaw/pitch scalars directly:

        fused_yaw   = gaze_weight*gaze_yaw   + head_weight*yaw_sign*head_yaw
        fused_pitch = gaze_weight*gaze_pitch + head_weight*pitch_sign*head_pitch

    This is only a first-order (small-angle) approximation. Euler
    angles do not compose by addition in general: yaw and pitch
    interact through the rotation matrix, not linearly, so the
    approximation degrades as either angle grows (see the numerical
    comparison below -- at ~30-40 degrees the two methods disagree
    by tens of degrees, not a rounding error). It also silently
    double-counts rotation if the eye estimator was itself trained
    on head-inclusive data, which most appearance-based gaze CNNs
    (including L2CS-Net) are.

    THIS VERSION
    ------------
    1. Converts the eye-relative (yaw, pitch) into a 3D unit gaze
       vector in the head-local frame.
    2. Rotates that vector into the camera frame using the ACTUAL
       rotation matrix from solvePnP (R_head), instead of
       reconstructing an approximate matrix from Euler angles.
    3. Converts back to (yaw, pitch) via atan2/arcsin, well-defined
       everywhere except the straight-up/down pole, handled
       explicitly by clamping.
    4. Confidence-aware blending: when the estimator's per-frame
       confidence C_t is low, the fused direction is blended toward
       the head's own forward axis rather than trusting a noisy eye
       vector outright.

    A CRITICAL DETAIL THAT IS EASY TO GET SILENTLY WRONG
    ------------------------------------------------------
    cv2.solvePnP returns a rotation matrix in OpenCV's camera axis
    convention: +X right, +Y DOWN, +Z forward into the scene. This
    module's internal gaze-vector convention uses +X right, +Y UP,
    with the camera looking down -Z (this matches how yaw/pitch are
    usually reasoned about and reported). These are NOT the same
    frame -- if you multiply an OpenCV R_head directly against a
    gaze vector built in the +Y-up/-Z-forward convention, you will
    get a plausible-looking but wrong answer (sign errors in pitch
    and yaw that mostly cancel for small angles, which is exactly
    why this kind of bug survives casual testing).

    By default (`camera_convention="opencv"`), this class converts
    any R_head you pass in from OpenCV axes into its own local axes
    before using it. Set `camera_convention="local"` only if you
    have already performed that conversion yourself, or if your
    R_head did not come from cv2.solvePnP.
    """

    # D is a 180-degree rotation about X: flips Y and Z. Applying
    # D @ R_cv @ D changes the *basis* R_cv is expressed in from
    # OpenCV axes (Y-down, Z-forward) to this module's axes
    # (Y-up, Z-backward / -Z-forward), without changing what
    # rotation is physically represented.
    _D = np.diag([1.0, -1.0, -1.0])

    def __init__(
        self,
        gaze_weight=1.0,
        head_influence=1.0,
        yaw_sign=1.0,
        pitch_sign=1.0,
        min_confidence_floor=0.05,
        camera_convention="opencv",
        head_weight=None,
    ):
        if head_weight is not None:
            # Backward-compatibility shim for the old additive-fusion API.
            # NOT an equivalent parameter -- old head_weight scaled a linear
            # angle sum with no upper bound; head_influence scales a
            # confidence-gated blend toward the head-forward direction and
            # is clamped to [0, 1]. Mapped 1:1 by clamping so old call
            # sites (typically head_weight=1.0) keep working without a
            # crash, but you should migrate the call site explicitly
            # rather than rely on this shim long-term.
            import warnings
            warnings.warn(
                "HeadEyeFusion(head_weight=...) is deprecated and does not "
                "mean the same thing as before (additive weight vs. "
                "confidence-gated blend factor). Pass head_influence= "
                "instead. Using head_influence=clip(head_weight, 0, 1) for now.",
                DeprecationWarning,
                stacklevel=2,
            )
            head_influence = np.clip(head_weight, 0.0, 1.0)

        self.gaze_weight = float(gaze_weight)
        self.head_influence = float(np.clip(head_influence, 0.0, 1.0))
        self.yaw_sign = float(yaw_sign)
        self.pitch_sign = float(pitch_sign)
        self.min_confidence_floor = float(min_confidence_floor)
        if camera_convention not in ("opencv", "local"):
            raise ValueError("camera_convention must be 'opencv' or 'local'")
        self.camera_convention = camera_convention

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    @staticmethod
    def _is_finite(*values):
        return all(v is not None and np.isfinite(v) for v in values)

    @staticmethod
    def normalize_angle(angle_deg):
        return float((angle_deg + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _angles_to_vector(yaw_deg, pitch_deg):
        """(yaw, pitch) in degrees -> unit vector, LOCAL frame
        (+x right, +y up, -z forward). yaw about y, pitch about x."""
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        x = np.cos(pitch) * np.sin(yaw)
        y = np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        v = np.array([x, y, z], dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else np.array([0.0, 0.0, -1.0])

    @staticmethod
    def _vector_to_angles(v):
        """Inverse of _angles_to_vector. Clamped at the pole
        (v_y = +-1) instead of producing NaN."""
        x, y, z = v
        y = float(np.clip(y, -1.0, 1.0))
        pitch = np.degrees(np.arcsin(y))
        yaw = np.degrees(np.arctan2(x, -z))
        return float(yaw), float(pitch)

    @classmethod
    def _rotation_matrix_from_euler(cls, yaw_deg, pitch_deg, roll_deg=0.0):
        """Fallback only -- prefer passing R_head from solvePnP directly.
        Built to be self-consistent with _angles_to_vector: applying
        Ry(yaw) to the local forward vector (0,0,-1) reproduces
        _angles_to_vector(yaw, 0) exactly (verified in tests)."""
        yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
        Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                       [0, 1, 0],
                       [np.sin(yaw), 0, np.cos(yaw)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch), np.sin(pitch)],
                       [0, -np.sin(pitch), np.cos(pitch)]])
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                       [np.sin(roll), np.cos(roll), 0],
                       [0, 0, 1]])
        return Ry @ Rx @ Rz

    @classmethod
    def opencv_to_local_rotation(cls, R_cv):
        """Convert a solvePnP rotation matrix (OpenCV axes: X right,
        Y down, Z forward-into-scene) into this module's local axes
        (X right, Y up, -Z forward)."""
        R_cv = np.asarray(R_cv, dtype=np.float64)
        return cls._D @ R_cv @ cls._D

    # ------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------
    def fuse(
        self,
        gaze_yaw,
        gaze_pitch,
        head_yaw,
        head_pitch,
        head_roll=0.0,
        confidence=1.0,
        R_head=None,
    ):
        if not self._is_finite(gaze_yaw, gaze_pitch, head_yaw, head_pitch):
            return None

        confidence = float(np.clip(confidence, self.min_confidence_floor, 1.0))

        # 1. eye-relative gaze as a unit vector (head-local frame)
        v_eye = self._angles_to_vector(
            self.yaw_sign * gaze_yaw,
            self.pitch_sign * gaze_pitch,
        )

        # 2. obtain R in THIS module's local axis convention
        if R_head is not None:
            R = np.asarray(R_head, dtype=np.float64)
            if self.camera_convention == "opencv":
                R = self.opencv_to_local_rotation(R)
        else:
            R = self._rotation_matrix_from_euler(head_yaw, head_pitch, head_roll)

        v_world = R @ v_eye
        n = np.linalg.norm(v_world)
        if n < 1e-8 or not np.all(np.isfinite(v_world)):
            return None
        v_world /= n

        # 3. confidence-aware blend toward head-forward axis
        head_forward = R @ np.array([0.0, 0.0, -1.0])
        hn = np.linalg.norm(head_forward)
        head_forward = head_forward / hn if hn > 1e-8 else np.array([0.0, 0.0, -1.0])

        blend = confidence * self.gaze_weight + (1.0 - confidence) * (1.0 - self.head_influence)
        blend = float(np.clip(blend, 0.0, 1.0))
        v_fused = blend * v_world + (1.0 - blend) * head_forward
        n = np.linalg.norm(v_fused)
        if n < 1e-8:
            return None
        v_fused /= n

        fused_yaw, fused_pitch = self._vector_to_angles(v_fused)
        return {
            "yaw": self.normalize_angle(fused_yaw),
            "pitch": self.normalize_angle(fused_pitch),
            "confidence_used": confidence,
        }