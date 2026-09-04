import cv2
import toml
import time
import csv
import os

from module1.eye_processing import get_face
from module2.gaze_estimation import GazeEstimator

from module3.head_pose import estimate_head_pose
from module3.head_eye_fusion import HeadEyeFusion
from module3.gaze_kalman import FixedKalmanGaze


# ============================================================
# CONFIGURATION
# ============================================================

config = toml.load("config.toml")


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = config["model"]["weights"]
DEVICE = config["model"]["device"]


# ============================================================
# CAMERA
# ============================================================

CAMERA_ID = config["camera"]["camera_id"]


# ============================================================
# INPUT
# ============================================================

FACE_SIZE = config["input"]["eye_size"]


# ============================================================
# DISPLAY
# ============================================================

SHOW_FACE_WINDOW = (
    config["display"]["show_eye_windows"]
)

SHOW_YAW_PITCH = (
    config["display"]["show_yaw_pitch"]
)


# ============================================================
# FUSION
# ============================================================

FUSION_ENABLED = (
    config["fusion"]["enabled"]
)

FUSION_GAZE_WEIGHT = (
    config["fusion"]["gaze_weight"]
)

FUSION_HEAD_WEIGHT = (
    config["fusion"]["head_weight"]
)

FUSION_YAW_SIGN = (
    config["fusion"]["yaw_sign"]
)

FUSION_PITCH_SIGN = (
    config["fusion"]["pitch_sign"]
)


# ============================================================
# KALMAN
# ============================================================

KALMAN_ENABLED = (
    config["kalman"]["enabled"]
)

KALMAN_PROCESS_NOISE = (
    config["kalman"]["process_noise"]
)

KALMAN_MEASUREMENT_NOISE = (
    config["kalman"]["measurement_noise"]
)

KALMAN_INITIAL_UNCERTAINTY = (
    config["kalman"]["initial_uncertainty"]
)


# ============================================================
# OUTPUT
# ============================================================

RESULTS_FILE = "gaze_pipeline_results.csv"


# ============================================================
# INITIALIZE GAZE ESTIMATOR
# ============================================================

gaze_estimator = GazeEstimator(
    MODEL_PATH,
    DEVICE
)


# ============================================================
# INITIALIZE HEAD-EYE FUSION
# ============================================================

head_eye_fusion = HeadEyeFusion(
    gaze_weight=FUSION_GAZE_WEIGHT,
    head_weight=FUSION_HEAD_WEIGHT,
    yaw_sign=FUSION_YAW_SIGN,
    pitch_sign=FUSION_PITCH_SIGN
)


# ============================================================
# INITIALIZE FIXED KALMAN
# ============================================================

kalman = FixedKalmanGaze(
    process_noise=KALMAN_PROCESS_NOISE,
    measurement_noise=KALMAN_MEASUREMENT_NOISE,
    initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_ID
)

if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    exit()


TARGET_FPS = 30.0

FRAME_TIME_SEC = (
    1.0 / TARGET_FPS
)


# ============================================================
# CSV
# ============================================================

csv_file = open(
    RESULTS_FILE,
    "w",
    newline=""
)

csv_writer = csv.writer(
    csv_file
)

csv_writer.writerow([
    "frame",
    "timestamp",

    # Raw gaze
    "raw_gaze_yaw",
    "raw_gaze_pitch",
    "gaze_confidence",

    # Head pose
    "head_yaw",
    "head_pitch",
    "head_roll",
    "head_reprojection_error",

    # Fusion
    "fused_yaw",
    "fused_pitch",

    # Kalman
    "kalman_yaw",
    "kalman_pitch"
])


# ============================================================
# STATUS
# ============================================================

print("=" * 70)
print("GAZE PIPELINE")
print("=" * 70)

print()

print(
    f"Head-eye fusion : "
    f"{'ON' if FUSION_ENABLED else 'OFF'}"
)

print(
    f"Fixed Kalman    : "
    f"{'ON' if KALMAN_ENABLED else 'OFF'}"
)

print()

print(
    "Press ESC to exit."
)

print()


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0

start_time = time.perf_counter()


while True:

    loop_start = time.perf_counter()

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read frame."
        )

        break


    frame_number += 1


    # ========================================================
    # MIRROR
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    timestamp = (
        time.perf_counter()
        -
        start_time
    )


    # ========================================================
    # HEAD POSE
    # ========================================================

    head_pose = estimate_head_pose(
        frame
    )


    # ========================================================
    # FACE / EYE PROCESSING
    # ========================================================

    frame, face_crop = get_face(
        frame,
        FACE_SIZE
    )


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    raw_yaw = None
    raw_pitch = None
    confidence = 0.0

    head_yaw = None
    head_pitch = None
    head_roll = None
    reprojection_error = None

    fused_yaw = None
    fused_pitch = None

    kalman_yaw = None
    kalman_pitch = None


    # ========================================================
    # HEAD VALUES
    # ========================================================

    if head_pose is not None:

        head_yaw = float(
            head_pose["yaw"]
        )

        head_pitch = float(
            head_pose["pitch"]
        )

        head_roll = float(
            head_pose["roll"]
        )

        reprojection_error = float(
            head_pose["reprojection_error"]
        )


    # ========================================================
    # GAZE ESTIMATION
    # ========================================================

    if face_crop is not None:

        raw_yaw, raw_pitch, confidence = (
            gaze_estimator.predict_gaze(
                face_crop
            )
        )


    # ========================================================
    # HEAD-EYE FUSION
    # ========================================================

    if (
        FUSION_ENABLED
        and raw_yaw is not None
        and raw_pitch is not None
        and head_yaw is not None
        and head_pitch is not None
    ):

        fused_result = (
            head_eye_fusion.fuse(
                gaze_yaw=raw_yaw,
                gaze_pitch=raw_pitch,
                head_yaw=head_yaw,
                head_pitch=head_pitch
            )
        )

        if fused_result is not None:

            fused_yaw = (
                fused_result["yaw"]
            )

            fused_pitch = (
                fused_result["pitch"]
            )

    else:

        # ----------------------------------------------------
        # Fusion OFF
        #
        # The gaze measurement itself becomes the
        # input to the next stage.
        # ----------------------------------------------------

        fused_yaw = raw_yaw
        fused_pitch = raw_pitch


    # ========================================================
    # FIXED KALMAN
    # ========================================================

    if (
        KALMAN_ENABLED
        and fused_yaw is not None
        and fused_pitch is not None
    ):

        kalman_result = (
            kalman.update(
                fused_yaw,
                fused_pitch
            )
        )

        if kalman_result is not None:

            kalman_yaw = (
                kalman_result["yaw"]
            )

            kalman_pitch = (
                kalman_result["pitch"]
            )

    else:

        # ----------------------------------------------------
        # Kalman OFF
        # ----------------------------------------------------

        kalman_yaw = fused_yaw
        kalman_pitch = fused_pitch


    # ========================================================
    # SAVE DATA
    # ========================================================

    csv_writer.writerow([
        frame_number,
        f"{timestamp:.6f}",

        "" if raw_yaw is None
        else f"{raw_yaw:.6f}",

        "" if raw_pitch is None
        else f"{raw_pitch:.6f}",

        f"{confidence:.6f}",

        "" if head_yaw is None
        else f"{head_yaw:.6f}",

        "" if head_pitch is None
        else f"{head_pitch:.6f}",

        "" if head_roll is None
        else f"{head_roll:.6f}",

        "" if reprojection_error is None
        else f"{reprojection_error:.6f}",

        "" if fused_yaw is None
        else f"{fused_yaw:.6f}",

        "" if fused_pitch is None
        else f"{fused_pitch:.6f}",

        "" if kalman_yaw is None
        else f"{kalman_yaw:.6f}",

        "" if kalman_pitch is None
        else f"{kalman_pitch:.6f}"
    ])


    # ========================================================
    # DISPLAY
    # ========================================================

    if SHOW_FACE_WINDOW and face_crop is not None:

        cv2.imshow(
            "FACE INPUT TO L2CS",
            face_crop
        )


    if SHOW_YAW_PITCH:

        y = 35

        if raw_yaw is not None:

            cv2.putText(
                frame,
                f"Raw Gaze: "
                f"Y {raw_yaw:.2f} "
                f"P {raw_pitch:.2f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            y += 35


        if head_yaw is not None:

            cv2.putText(
                frame,
                f"Head: "
                f"Y {head_yaw:.2f} "
                f"P {head_pitch:.2f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            y += 35


        if fused_yaw is not None:

            cv2.putText(
                frame,
                f"Fused: "
                f"Y {fused_yaw:.2f} "
                f"P {fused_pitch:.2f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            y += 35


        if kalman_yaw is not None:

            cv2.putText(
                frame,
                f"Kalman: "
                f"Y {kalman_yaw:.2f} "
                f"P {kalman_pitch:.2f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 255),
                2
            )

            y += 35


        cv2.putText(
            frame,
            f"Confidence: "
            f"{confidence:.2f}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Integrated Gaze System",
        frame
    )


    # ========================================================
    # EXIT / FPS
    # ========================================================

    elapsed = (
        time.perf_counter()
        -
        loop_start
    )

    remaining = max(
        1,
        int(
            (
                FRAME_TIME_SEC
                -
                elapsed
            ) * 1000
        )
    )

    key = cv2.waitKey(
        remaining
    ) & 0xFF

    if key == 27:

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

csv_file.close()

cv2.destroyAllWindows()


print()
print("=" * 70)
print("PIPELINE FINISHED")
print("=" * 70)

print(
    f"Results saved to: "
    f"{os.path.abspath(RESULTS_FILE)}"
)