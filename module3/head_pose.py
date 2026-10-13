import cv2
import numpy as np
import mediapipe as mp
import atexit


# ============================================================
# MEDIAPIPE FACE MESH
# ============================================================

mp_face = mp.solutions.face_mesh

_HEAD_POSE_DEBUG_COUNTS = {
    'face_landmarks_missing': 0,
    'non_finite_landmarks': 0,
    'pnp_exception': 0,
    'pnp_unsuccessful': 0,
    'reprojection_error_high': 0,
    'non_finite_angles': 0,
    'invalid_frame': 0,
}

def reset_head_pose_debug_counts():
    """Reset rejection counters for a new collection run."""
    for key in _HEAD_POSE_DEBUG_COUNTS:
        _HEAD_POSE_DEBUG_COUNTS[key] = 0

def get_head_pose_debug_counts():
    """Return a copy of the head-pose rejection counters."""
    return dict(_HEAD_POSE_DEBUG_COUNTS)

def _print_head_pose_debug_summary():
    total = sum(_HEAD_POSE_DEBUG_COUNTS.values())
    print('\\nHEAD-POSE DEBUG SUMMARY')
    print(f'Total rejected frames counted: {total}')
    for reason, count in _HEAD_POSE_DEBUG_COUNTS.items():
        print(f'  {reason}: {count}')

atexit.register(_print_head_pose_debug_summary)

face_mesh = mp_face.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# FACE LANDMARK INDICES
# ============================================================

NOSE_TIP = 1
CHIN = 152

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

LEFT_MOUTH = 61
RIGHT_MOUTH = 291


LANDMARK_IDS = [
    NOSE_TIP,
    CHIN,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    LEFT_MOUTH,
    RIGHT_MOUTH
]


# ============================================================
# 3D FACE MODEL POINTS
# ============================================================

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)


# ============================================================
# ANGLE NORMALIZATION
# ============================================================

def normalize_angle(angle):
    return float(
        (angle + 180.0) % 360.0 - 180.0
    )


def normalize_head_angle(angle):

    angle = normalize_angle(angle)

    if angle > 90.0:
        angle = 180.0 - angle

    elif angle < -90.0:
        angle = -180.0 - angle

    return float(angle)


# ============================================================
# CAMERA MATRIX
# ============================================================

def create_camera_matrix(width, height):

    focal_length = float(width)

    center_x = width / 2.0
    center_y = height / 2.0

    return np.array([
        [focal_length, 0.0, center_x],
        [0.0, focal_length, center_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


# ============================================================
# REPROJECTION ERROR
# ============================================================

def calculate_reprojection_error(
    model_points,
    image_points,
    rotation_vector,
    translation_vector,
    camera_matrix,
    distortion_coefficients
):

    projected_points, _ = cv2.projectPoints(
        model_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coefficients
    )

    projected_points = projected_points.reshape(-1, 2)

    errors = np.linalg.norm(
        image_points - projected_points,
        axis=1
    )

    return float(
        np.sqrt(np.mean(errors ** 2))
    )


# ============================================================
# ROTATION MATRIX -> YAW/PITCH/ROLL
# ============================================================

def rotation_matrix_to_angles(rotation_matrix):

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(
        rotation_matrix
    )

    pitch = normalize_head_angle(
        float(angles[0])
    )

    yaw = normalize_head_angle(
        float(angles[1])
    )

    roll = normalize_head_angle(
        float(angles[2])
    )

    return yaw, pitch, roll


# ============================================================
# HEAD POSE ESTIMATION
# ============================================================

def estimate_head_pose(frame):

    if frame is None:
        _HEAD_POSE_DEBUG_COUNTS['invalid_frame'] += 1
        return None

    height, width = frame.shape[:2]

    if width < 100 or height < 100:
        _HEAD_POSE_DEBUG_COUNTS['invalid_frame'] += 1
        return None

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(
        rgb_frame
    )

    if not results.multi_face_landmarks:
        _HEAD_POSE_DEBUG_COUNTS['face_landmarks_missing'] += 1
        return None

    face_landmarks = (
        results.multi_face_landmarks[0]
    )

    # --------------------------------------------------------
    # Extract 2D landmarks
    # --------------------------------------------------------

    image_points = []

    for landmark_id in LANDMARK_IDS:

        landmark = (
            face_landmarks.landmark[
                landmark_id
            ]
        )

        x = landmark.x * width
        y = landmark.y * height

        if not np.isfinite(x) or not np.isfinite(y):
            _HEAD_POSE_DEBUG_COUNTS['non_finite_landmarks'] += 1
            return None

        image_points.append([
            x,
            y
        ])

    image_points = np.asarray(
        image_points,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Camera parameters
    # --------------------------------------------------------

    camera_matrix = create_camera_matrix(
        width,
        height
    )

    distortion_coefficients = np.zeros(
        (4, 1),
        dtype=np.float64
    )

    # --------------------------------------------------------
    # PnP
    # --------------------------------------------------------

    
    try:
        success, rotation_vector, translation_vector = (
            cv2.solvePnP(
                MODEL_POINTS,
                image_points,
                camera_matrix,
                distortion_coefficients,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        )
    except cv2.error as e:
        print("solvePnP error:", e)
        return None

    if not success:
        return None


    # --------------------------------------------------------
    # LM refinement
    # --------------------------------------------------------

    try:

        rotation_vector, translation_vector = (
            cv2.solvePnPRefineLM(
                MODEL_POINTS,
                image_points,
                camera_matrix,
                distortion_coefficients,
                rotation_vector,
                translation_vector
            )
        )

    except cv2.error:

        pass

    # --------------------------------------------------------
    # Reprojection error
    # --------------------------------------------------------

    reprojection_error = (
        calculate_reprojection_error(
            MODEL_POINTS,
            image_points,
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion_coefficients
        )
    )

    # Reject poor estimates
    if not np.isfinite(reprojection_error) or reprojection_error > 20.0:
        _HEAD_POSE_DEBUG_COUNTS['reprojection_error_high'] += 1
        return None

    # --------------------------------------------------------
    # Rotation matrix
    # --------------------------------------------------------

    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )

    # --------------------------------------------------------
    # Angles
    # --------------------------------------------------------

    yaw, pitch, roll = (
        rotation_matrix_to_angles(
            rotation_matrix
        )
    )

    if not all(
        np.isfinite(value)
        for value in [
            yaw,
            pitch,
            roll
        ]
    ):
        _HEAD_POSE_DEBUG_COUNTS['non_finite_angles'] += 1
        return None

    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
        "reprojection_error": float(
            reprojection_error
        )
    }