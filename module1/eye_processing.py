import cv2
import mediapipe as mp


# ============================================================
# MediaPipe Face Mesh
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# Get FACE crop
#
# FIX: default face_size changed from 224 -> 448 to match the
# official L2CS-Net Gaze360 input resolution. GazeEstimator's
# transform will also resize to 448x448, so this crop size just
# needs to avoid unnecessary upscaling; keeping them in sync
# avoids a redundant double-resize.
# ============================================================

def crop_face(frame, landmarks, face_size=448):

    h, w, _ = frame.shape

    xs = []
    ys = []

    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)

        xs.append(x)
        ys.append(y)

    x_min = max(0, min(xs))
    x_max = min(w, max(xs))
    y_min = max(0, min(ys))
    y_max = min(h, max(ys))

    face_width = x_max - x_min
    face_height = y_max - y_min

    if face_width <= 0 or face_height <= 0:
        return None

    # --------------------------------------------------------
    # FIX: pad to a SQUARE box (not independent x/y padding),
    # so the resize to face_size x face_size below doesn't
    # stretch the face non-uniformly. Use the larger of
    # width/height as the square side, centered on the bbox.
    # --------------------------------------------------------

    box_side = int(max(face_width, face_height) * 1.20)  # 20% padding baked in

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    half = box_side // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    # --------------------------------------------------------
    # FIX: convert BGR (raw cv2/VideoCapture frame) -> RGB
    # BEFORE resizing/returning. GazeEstimator.predict_gaze()
    # does Image.fromarray(...).convert("RGB"), which does NOT
    # swap channel order -- it assumes the array is already RGB.
    # Without this conversion the model sees a channel-swapped
    # (blue/red inverted) face, which silently degrades
    # confidence even when yaw/pitch look roughly plausible.
    # --------------------------------------------------------

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # Resize to model input size (now safe since crop is square)
    # --------------------------------------------------------

    face = cv2.resize(
        face,
        (face_size, face_size),
        interpolation=cv2.INTER_AREA
    )

    return face


# ============================================================
# Main face processing function
# ============================================================

def get_face(frame, face_size=448):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return frame, None

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = face_landmarks.landmark

    # ========================================================
    # Crop entire face
    # ========================================================

    face_crop = crop_face(
        frame,
        landmarks,
        face_size
    )

    # ========================================================
    # Draw face bounding box
    # ========================================================

    h, w, _ = frame.shape

    xs = [
        int(lm.x * w)
        for lm in landmarks
    ]

    ys = [
        int(lm.y * h)
        for lm in landmarks
    ]

    x1 = max(0, min(xs))
    x2 = min(w, max(xs))
    y1 = max(0, min(ys))
    y2 = min(h, max(ys))

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    return frame, face_crop