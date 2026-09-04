import cv2
import numpy as np
import time
import csv
import os
import sys

from module1.eye_processing import get_face
from module2.gaze_estimation import GazeEstimator

from module3.head_pose import estimate_head_pose
from module3.head_eye_fusion import HeadEyeFusion

from module3.gaze_kalman import (
    FixedKalmanGaze,
    AdaptiveQKalmanGaze,
    AdaptiveRKalmanGaze,
    AdaptiveQRKalmanGaze,   # NEW: combined adaptive filter -- your actual "Proposed" method
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\majorproject\gaze_project\models\L2CSNet_gaze360.pkl"

CAMERA_ID = 0

FIXATION_SECONDS = 5.0

DISCARD_FRACTION = 0.40

# FIX: was 224 -- this silently overrode the 448 fix made in
# gaze_estimation.py / eye_processing.py, meaning every method
# in the last trial run (including baseline) was evaluated on
# the WRONG input resolution. Must match GazeEstimator's
# transforms.Resize((448, 448)).
FACE_SIZE = 448


# ============================================================
# OUTPUT
# ============================================================

# Run:
#
# python phase2_calibration.py trial_01
#
# or:
#
# python phase2_calibration.py trial_02
#
# etc.

if len(sys.argv) >= 2:

    TRIAL_NAME = sys.argv[1]

else:

    TRIAL_NAME = "trial_01"


TRIAL_DIR = os.path.join(
    "trials",
    TRIAL_NAME
)

os.makedirs(
    TRIAL_DIR,
    exist_ok=True
)


OUTPUT_FILE = os.path.join(
    TRIAL_DIR,
    "unified_evaluation.csv"
)


# ============================================================
# FUSION
# ============================================================

FUSION_GAZE_WEIGHT = 1.0
FUSION_HEAD_WEIGHT = 1.0

FUSION_YAW_SIGN = 1.0
FUSION_PITCH_SIGN = 1.0


# ============================================================
# FIXED KALMAN
# ============================================================

KALMAN_PROCESS_NOISE = 0.01

KALMAN_MEASUREMENT_NOISE = 1.0

KALMAN_INITIAL_UNCERTAINTY = 1.0


# ============================================================
# ADAPTIVE Q
# ============================================================

ADAPTIVE_Q_MIN = 0.001

ADAPTIVE_Q_MAX = 0.20

ADAPTIVE_Q_RATE = 0.10


# ============================================================
# ADAPTIVE R
# ============================================================

ADAPTIVE_R_MIN = 0.10

ADAPTIVE_R_MAX = 10.0

# NOTE: ADAPTIVE_R_RATE removed -- the fixed AdaptiveRKalmanGaze
# no longer takes an innovation-based adaptation_rate; R is now
# derived directly from model confidence via r_min/r_max only.


# ============================================================
# SCREEN SIZE
# ============================================================

def get_screen_size():

    try:

        import tkinter as tk

        root = tk.Tk()

        root.withdraw()

        width = root.winfo_screenwidth()

        height = root.winfo_screenheight()

        root.destroy()

        if width > 0 and height > 0:

            return width, height

    except Exception:

        pass


    try:

        from screeninfo import get_monitors

        monitors = get_monitors()

        primary = next(
            (
                m
                for m in monitors
                if getattr(
                    m,
                    "is_primary",
                    False
                )
            ),
            monitors[0]
        )

        return primary.width, primary.height

    except Exception:

        pass


    print(
        "WARNING: Could not detect screen size."
    )

    return 1366, 768


SCREEN_W, SCREEN_H = get_screen_size()


print(
    f"Detected screen resolution: "
    f"{SCREEN_W} x {SCREEN_H}"
)


# ============================================================
# GRID
# ============================================================

MARGIN_X = int(
    SCREEN_W * 0.10
)

MARGIN_Y = int(
    SCREEN_H * 0.10
)


GRID_POINTS = [

    (MARGIN_X, MARGIN_Y),

    (SCREEN_W // 2, MARGIN_Y),

    (SCREEN_W - MARGIN_X, MARGIN_Y),

    (MARGIN_X, SCREEN_H // 2),

    (SCREEN_W // 2, SCREEN_H // 2),

    (SCREEN_W - MARGIN_X, SCREEN_H // 2),

    (MARGIN_X, SCREEN_H - MARGIN_Y),

    (SCREEN_W // 2, SCREEN_H - MARGIN_Y),

    (
        SCREEN_W - MARGIN_X,
        SCREEN_H - MARGIN_Y
    )
]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("UNIFIED GAZE EVALUATION")
    print("=" * 80)

    print()

    print(f"TRIAL: {TRIAL_NAME}")

    print()

    print(
        "ALL SIX METHODS USE THE SAME WEBCAM FRAMES."
    )

    print()

    print(
        "1. Baseline   : Raw gaze"
    )

    print(
        "2. Method 2   : Raw + Head-Eye Fusion"
    )

    print(
        "3. Fixed KF   : Fusion + Fixed Kalman"
    )

    print(
        "4. Adaptive Q : Fusion + Adaptive-Q Kalman"
    )

    print(
        "5. Adaptive R : Fusion + Adaptive-R Kalman (confidence-based)"
    )

    print(
        "6. Proposed   : Fusion + Adaptive Q+R Kalman (AKF)"
    )

    print()

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()

    print(
        "Press ESC to stop."
    )

    print()


    # ========================================================
    # MODELS
    # ========================================================

    estimator = GazeEstimator(
        MODEL_PATH,
        device="cpu"
    )


    fusion = HeadEyeFusion(

        gaze_weight=FUSION_GAZE_WEIGHT,

        head_weight=FUSION_HEAD_WEIGHT,

        yaw_sign=FUSION_YAW_SIGN,

        pitch_sign=FUSION_PITCH_SIGN
    )


    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(
        CAMERA_ID
    )


    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return


    # ========================================================
    # WINDOW
    # ========================================================

    window = "Unified Gaze Evaluation"

    cv2.namedWindow(
        window,
        cv2.WND_PROP_FULLSCREEN
    )

    cv2.setWindowProperty(
        window,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )


    # ========================================================
    # CSV
    # ========================================================

    csv_file = open(
        OUTPUT_FILE,
        "w",
        newline=""
    )

    writer = csv.writer(
        csv_file
    )


    writer.writerow([

        "frame",
        "timestamp",

        "point_idx",

        "target_x",
        "target_y",

        # Baseline
        "raw_gaze_yaw",
        "raw_gaze_pitch",
        "raw_confidence",

        # Head
        "head_yaw",
        "head_pitch",
        "head_roll",
        "head_reprojection_error",

        # Method 2
        "fused_yaw",
        "fused_pitch",

        # Fixed Kalman
        "kalman_yaw",
        "kalman_pitch",

        # Adaptive Q
        "adaptive_q_yaw",
        "adaptive_q_pitch",

        # Adaptive R
        "adaptive_r_yaw",
        "adaptive_r_pitch",

        # Adaptive Q+R (Proposed)
        "adaptive_qr_yaw",
        "adaptive_qr_pitch"

    ])


    # ========================================================
    # RECORDING
    # ========================================================

    frame_number = 0

    experiment_start = time.perf_counter()


    for point_idx, (
        target_x,
        target_y
    ) in enumerate(GRID_POINTS):


        print(
            f"Point "
            f"{point_idx + 1}/"
            f"{len(GRID_POINTS)} "
            f"-> "
            f"({target_x}, {target_y})"
        )


        # ----------------------------------------------------
        # NEW FILTERS FOR EACH FIXATION
        # ----------------------------------------------------

        kalman = FixedKalmanGaze(

            process_noise=KALMAN_PROCESS_NOISE,

            measurement_noise=KALMAN_MEASUREMENT_NOISE,

            initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY
        )


        adaptive_q = AdaptiveQKalmanGaze(

            process_noise=KALMAN_PROCESS_NOISE,

            measurement_noise=KALMAN_MEASUREMENT_NOISE,

            initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY,

            q_min=ADAPTIVE_Q_MIN,

            q_max=ADAPTIVE_Q_MAX,

            adaptation_rate=ADAPTIVE_Q_RATE
        )


        adaptive_r = AdaptiveRKalmanGaze(

            process_noise=KALMAN_PROCESS_NOISE,

            measurement_noise=KALMAN_MEASUREMENT_NOISE,

            initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY,

            r_min=ADAPTIVE_R_MIN,

            r_max=ADAPTIVE_R_MAX
        )


        adaptive_qr = AdaptiveQRKalmanGaze(

            process_noise=KALMAN_PROCESS_NOISE,

            measurement_noise=KALMAN_MEASUREMENT_NOISE,

            initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY,

            q_min=ADAPTIVE_Q_MIN,

            q_max=ADAPTIVE_Q_MAX,

            q_adaptation_rate=ADAPTIVE_Q_RATE,

            r_min=ADAPTIVE_R_MIN,

            r_max=ADAPTIVE_R_MAX
        )


        point_start = time.perf_counter()

        point_samples = []


        # ====================================================
        # FIXATION
        # ====================================================

        while (

            time.perf_counter()
            -
            point_start

            <

            FIXATION_SECONDS

        ):


            ret, frame = cap.read()


            if not ret:

                continue


            frame_number += 1


            # ------------------------------------------------
            # MIRROR
            # ------------------------------------------------

            frame = cv2.flip(
                frame,
                1
            )


            timestamp = (

                time.perf_counter()
                -
                experiment_start
            )


            # ------------------------------------------------
            # TARGET
            # ------------------------------------------------

            canvas = np.zeros(

                (
                    SCREEN_H,
                    SCREEN_W,
                    3
                ),

                dtype=np.uint8
            )


            cv2.circle(

                canvas,

                (
                    target_x,
                    target_y
                ),

                18,

                (0, 0, 255),

                -1
            )


            cv2.circle(

                canvas,

                (
                    target_x,
                    target_y
                ),

                18,

                (255, 255, 255),

                2
            )


            cv2.putText(

                canvas,

                (
                    f"Point "
                    f"{point_idx + 1}/"
                    f"{len(GRID_POINTS)}"
                ),

                (30, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (255, 255, 255),

                2
            )


            cv2.imshow(
                window,
                canvas
            )


            if cv2.waitKey(1) & 0xFF == 27:

                cap.release()

                csv_file.close()

                cv2.destroyAllWindows()

                return


            # ------------------------------------------------
            # HEAD POSE
            # ------------------------------------------------

            head_pose = estimate_head_pose(
                frame
            )


            # ------------------------------------------------
            # FACE
            # ------------------------------------------------

            _, face_crop = get_face(

                frame,

                FACE_SIZE
            )


            # ------------------------------------------------
            # DEFAULTS
            # ------------------------------------------------

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


            adaptive_q_yaw = None

            adaptive_q_pitch = None


            adaptive_r_yaw = None

            adaptive_r_pitch = None


            adaptive_qr_yaw = None

            adaptive_qr_pitch = None


            # =================================================
            # HEAD
            # =================================================

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


            # =================================================
            # RAW GAZE
            # =================================================

            if face_crop is not None:

                (
                    raw_yaw,
                    raw_pitch,
                    confidence
                ) = estimator.predict_gaze(
                    face_crop
                )


            # =================================================
            # HEAD-EYE FUSION
            # =================================================

            if (

                raw_yaw is not None

                and raw_pitch is not None

                and head_yaw is not None

                and head_pitch is not None

            ):

                fused_result = fusion.fuse(

                    gaze_yaw=raw_yaw,

                    gaze_pitch=raw_pitch,

                    head_yaw=head_yaw,

                    head_pitch=head_pitch
                )


                if fused_result is not None:

                    fused_yaw = float(
                        fused_result["yaw"]
                    )

                    fused_pitch = float(
                        fused_result["pitch"]
                    )


            # =================================================
            # KALMAN FAMILY -- all four share the same fused
            # measurement; AdaptiveR and AdaptiveQR additionally
            # receive `confidence` from the raw gaze estimate,
            # which is what makes their R adaptation meaningful.
            # =================================================

            if (

                fused_yaw is not None

                and fused_pitch is not None

            ):

                result = kalman.update(
                    fused_yaw,
                    fused_pitch
                )

                if result is not None:
                    kalman_yaw = float(result["yaw"])
                    kalman_pitch = float(result["pitch"])


                result_q = adaptive_q.update(
                    fused_yaw,
                    fused_pitch
                )

                if result_q is not None:
                    adaptive_q_yaw = float(result_q["yaw"])
                    adaptive_q_pitch = float(result_q["pitch"])


                result_r = adaptive_r.update(
                    fused_yaw,
                    fused_pitch,
                    confidence=confidence   # FIX: now wired through
                )

                if result_r is not None:
                    adaptive_r_yaw = float(result_r["yaw"])
                    adaptive_r_pitch = float(result_r["pitch"])


                result_qr = adaptive_qr.update(
                    fused_yaw,
                    fused_pitch,
                    confidence=confidence
                )

                if result_qr is not None:
                    adaptive_qr_yaw = float(result_qr["yaw"])
                    adaptive_qr_pitch = float(result_qr["pitch"])


            # =================================================
            # FAIR SAMPLE -- require every method to have produced
            # a result this frame so all six columns are directly
            # comparable row-for-row
            # =================================================

            if (

                raw_yaw is not None

                and raw_pitch is not None

                and fused_yaw is not None

                and fused_pitch is not None

                and kalman_yaw is not None

                and kalman_pitch is not None

                and adaptive_q_yaw is not None

                and adaptive_q_pitch is not None

                and adaptive_r_yaw is not None

                and adaptive_r_pitch is not None

                and adaptive_qr_yaw is not None

                and adaptive_qr_pitch is not None

            ):

                point_samples.append([

                    frame_number,

                    timestamp,

                    point_idx,

                    target_x,

                    target_y,

                    raw_yaw,

                    raw_pitch,

                    confidence,

                    head_yaw,

                    head_pitch,

                    head_roll,

                    reprojection_error,

                    fused_yaw,

                    fused_pitch,

                    kalman_yaw,

                    kalman_pitch,

                    adaptive_q_yaw,

                    adaptive_q_pitch,

                    adaptive_r_yaw,

                    adaptive_r_pitch,

                    adaptive_qr_yaw,

                    adaptive_qr_pitch

                ])


        # ====================================================
        # DISCARD SETTLING
        # ====================================================

        n_discard = int(

            len(point_samples)
            *
            DISCARD_FRACTION
        )


        kept_samples = point_samples[
            n_discard:
        ]


        # ====================================================
        # WRITE
        # ====================================================

        for row in kept_samples:

            writer.writerow(row)


        csv_file.flush()


        print(

            f"   Collected: "
            f"{len(point_samples)}"
            f" | Kept: "
            f"{len(kept_samples)}"
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    csv_file.close()

    cv2.destroyAllWindows()


    print()
    print("=" * 80)
    print("RECORDING FINISHED")
    print("=" * 80)

    print()

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()


if __name__ == "__main__":

    main()