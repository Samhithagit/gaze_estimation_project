import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# MEDIAPIPE FACE MESH
# ============================================================

mp_face = mp.solutions.face_mesh

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
        return None

    height, width = frame.shape[:2]

    if width < 100 or height < 100:
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

        success, rotation_vector, translation_vector, inliers = (
            cv2.solvePnPRansac(
                MODEL_POINTS,
                image_points,
                camera_matrix,
                distortion_coefficients,
                flags=cv2.SOLVEPNP_ITERATIVE,
                iterationsCount=100,
                reprojectionError=8.0,
                confidence=0.99
            )
        )

    except cv2.error:

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
    if reprojection_error > 20.0:
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
        return None

    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
        "reprojection_error": float(
            reprojection_error
        )
    }