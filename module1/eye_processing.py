import cv2
import mediapipe as mp
import numpy as np

# =========================
# MediaPipe FaceMesh Setup
# =========================

mp_face = mp.solutions.face_mesh

face_mesh = mp_face.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Eye Landmark Indices
# =========================

LEFT_EYE = [
    33, 133, 160, 159, 158, 157,
    173, 155, 154, 153, 144, 145
]

RIGHT_EYE = [
    362, 263, 387, 386, 385, 384,
    398, 382, 381, 380, 374, 373
]

# =========================
# Eye Extraction Function
# =========================

def process_frame(frame, eye_size=224):

    # Mirror view
    frame = cv2.flip(frame, 1)

    # Frame dimensions
    h, w, _ = frame.shape

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # FaceMesh
    results = face_mesh.process(rgb_frame)

    # Default outputs
    left_eye_resized = None
    right_eye_resized = None

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # =========================
            # LEFT EYE
            # =========================

            left_eye_points = []

            for idx in LEFT_EYE:

                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                left_eye_points.append((x, y))

                # Draw landmarks
                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )

            left_eye_np = np.array(left_eye_points)

            lx, ly, lw, lh = cv2.boundingRect(
                left_eye_np
            )

            padding = 10

            lx = max(lx - padding, 0)
            ly = max(ly - padding, 0)

            lw = lw + 2 * padding
            lh = lh + 2 * padding

            # Draw eye box
            cv2.rectangle(
                frame,
                (lx, ly),
                (lx + lw, ly + lh),
                (255, 0, 0),
                2
            )

            # Crop left eye
            left_eye_crop = frame[
                ly:ly + lh,
                lx:lx + lw
            ]

            # =========================
            # RIGHT EYE
            # =========================

            right_eye_points = []

            for idx in RIGHT_EYE:

                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                right_eye_points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )

            right_eye_np = np.array(right_eye_points)

            rx, ry, rw, rh = cv2.boundingRect(
                right_eye_np
            )

            rx = max(rx - padding, 0)
            ry = max(ry - padding, 0)

            rw = rw + 2 * padding
            rh = rh + 2 * padding

            # Draw eye box
            cv2.rectangle(
                frame,
                (rx, ry),
                (rx + rw, ry + rh),
                (255, 0, 0),
                2
            )

            # Crop right eye
            right_eye_crop = frame[
                ry:ry + rh,
                rx:rx + rw
            ]

            # =========================
            # Resize for CNN
            # =========================

            if left_eye_crop.size != 0:

                left_eye_resized = cv2.resize(
                    left_eye_crop,
                    (eye_size, eye_size)
                )

            if right_eye_crop.size != 0:

                right_eye_resized = cv2.resize(
                    right_eye_crop,
                    (eye_size, eye_size)
                )

    # =========================
    # Return Outputs
    # =========================

    return (
        frame,
        left_eye_resized,
        right_eye_resized
    )