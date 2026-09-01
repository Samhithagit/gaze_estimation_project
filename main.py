import cv2
import toml
import time

# =========================
# Import Module 1
# =========================

from module1.eye_processing import process_frame

# =========================
# Import Module 2
# =========================

from module2.gaze_estimation import GazeEstimator

# =========================
# Load Config
# =========================

config = toml.load("config.toml")

MODEL_PATH = config["model"]["weights"]
DEVICE = config["model"]["device"]

CAMERA_ID = config["camera"]["camera_id"]
EYE_SIZE = config["input"]["eye_size"]

SHOW_EYE_WINDOWS = config["display"]["show_eye_windows"]
SHOW_YAW_PITCH = config["display"]["show_yaw_pitch"]

# =========================
# Initialize Gaze Estimator
# =========================

gaze_estimator = GazeEstimator(
    MODEL_PATH,
    DEVICE
)

# =========================
# Start Webcam
# =========================

cap = cv2.VideoCapture(CAMERA_ID)

# =========================
# FPS Control
# =========================

TARGET_FPS = 30
FRAME_TIME_SEC = 1.0 / TARGET_FPS

cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

# =========================
# Main Loop
# =========================

while True:

    loop_start = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror View
    frame = cv2.flip(frame, 1)

    # =========================
    # MODULE 1
    # =========================

    frame, left_eye, right_eye = process_frame(
        frame,
        EYE_SIZE
    )

    # =========================
    # MODULE 2
    # =========================

    yaw_left, pitch_left = gaze_estimator.predict_gaze(
        left_eye
    )

    yaw_right, pitch_right = gaze_estimator.predict_gaze(
        right_eye
    )

    # =========================
    # Average Both Eyes
    # =========================

    if yaw_left is not None and yaw_right is not None:

        yaw = (yaw_left + yaw_right) / 2


        pitch = (pitch_left + pitch_right) / 2

        print("LEFT EYE:")
        print(f"Yaw: {yaw_left:.2f}, Pitch: {pitch_left:.2f}")

        print("RIGHT EYE:")
        print(f"Yaw: {yaw_right:.2f}, Pitch: {pitch_right:.2f}")

        print("--------------------")

        print(
            f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}"
        )

        # =========================
        # Display Yaw/Pitch
        # =========================

        if SHOW_YAW_PITCH:

            cv2.putText(
                frame,
                f"Yaw: {yaw:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Pitch: {pitch:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # =========================
    # Show Eye Windows
    # =========================

    if SHOW_EYE_WINDOWS:

        if left_eye is not None:
            cv2.imshow(
                "Left Eye",
                left_eye
            )

        if right_eye is not None:
            cv2.imshow(
                "Right Eye",
                right_eye
            )

    # =========================
    # Final Output Window
    # =========================

    cv2.imshow(
        "Integrated Gaze System",
        frame
    )

    # =========================
    # FPS Limiter
    # =========================

    elapsed = (
        time.perf_counter() - loop_start
    )

    remaining_ms = max(
        1,
        int((FRAME_TIME_SEC - elapsed) * 1000)
    )

    # ESC Key

    if cv2.waitKey(remaining_ms) & 0xFF == 27:
        break

# =========================
# Cleanup
# =========================

cap.release()

cv2.destroyAllWindows()