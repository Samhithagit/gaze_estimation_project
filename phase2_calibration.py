import cv2
import numpy as np
import time
import csv
import os
import sys
import random

from module1.eye_processing import get_face
from module2.gaze_estimation import GazeEstimator
from module3.head_pose import estimate_head_pose


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\majorproject\gaze_project\models\L2CSNet_gaze360.pkl"

CAMERA_ID = 0

FIXATION_SECONDS = 5.0

DISCARD_FRACTION = 0.40

FACE_SIZE = 448


# ============================================================
# TRIAL
# ============================================================

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


CALIBRATION_FILE = os.path.join(
    TRIAL_DIR,
    "calibration_data.csv"
)

EVALUATION_FILE = os.path.join(
    TRIAL_DIR,
    "evaluation_data.csv"
)


# ============================================================
# SCREEN SIZE
# ============================================================

def get_screen_size():

    # --------------------------------------------------------
    # Method 1: tkinter
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Method 2: screeninfo
    # --------------------------------------------------------

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

    # Top row
    (
        MARGIN_X,
        MARGIN_Y
    ),

    (
        SCREEN_W // 2,
        MARGIN_Y
    ),

    (
        SCREEN_W - MARGIN_X,
        MARGIN_Y
    ),

    # Middle row
    (
        MARGIN_X,
        SCREEN_H // 2
    ),

    (
        SCREEN_W // 2,
        SCREEN_H // 2
    ),

    (
        SCREEN_W - MARGIN_X,
        SCREEN_H // 2
    ),

    # Bottom row
    (
        MARGIN_X,
        SCREEN_H - MARGIN_Y
    ),

    (
        SCREEN_W // 2,
        SCREEN_H - MARGIN_Y
    ),

    (
        SCREEN_W - MARGIN_X,
        SCREEN_H - MARGIN_Y
    )
]


# ============================================================
# CSV HEADER
# ============================================================

CSV_HEADER = [

    "frame",

    "timestamp",

    "point_idx",

    "target_x",

    "target_y",

    "raw_gaze_yaw",

    "raw_gaze_pitch",

    "raw_confidence",

    "head_yaw",

    "head_pitch",

    "head_roll",

    "head_reprojection_error"

]


# ============================================================
# CREATE WINDOW
# ============================================================

def create_window():

    window = "Gaze Calibration"

    # IMPORTANT:
    # This is kept exactly like your original working code.

    cv2.namedWindow(

        window,

        cv2.WND_PROP_FULLSCREEN

    )

    cv2.setWindowProperty(

        window,

        cv2.WND_PROP_FULLSCREEN,

        cv2.WINDOW_FULLSCREEN

    )

    return window


# ============================================================
# COLLECT ONE DATASET
# ============================================================

def collect_dataset(
    estimator,
    cap,
    window,
    output_file,
    points,
    dataset_name
):

    print()
    print("=" * 80)
    print(f"{dataset_name.upper()} DATA COLLECTION")
    print("=" * 80)
    print()

    print(
        f"Output: {output_file}"
    )

    print()

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_file = open(

        output_file,

        "w",

        newline=""

    )

    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        CSV_HEADER
    )


    # --------------------------------------------------------
    # Recording
    # --------------------------------------------------------

    frame_number = 0

    experiment_start = (
        time.perf_counter()
    )


    # ========================================================
    # TARGET LOOP
    # ========================================================

    for point_idx, (

        target_x,

        target_y

    ) in enumerate(points):


        print(

            f"Point "
            f"{point_idx + 1}/"
            f"{len(points)} "
            f"-> "
            f"({target_x}, {target_y})"

        )


        point_start = (
            time.perf_counter()
        )

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


            # ------------------------------------------------
            # READ FRAME
            # ------------------------------------------------

            ret, frame = cap.read()


            if not ret:
                continue


            frame_number += 1


            # ------------------------------------------------
            # MIRROR FRAME
            # ------------------------------------------------

            frame = cv2.flip(

                frame,

                1

            )


            # ------------------------------------------------
            # TIMESTAMP
            # ------------------------------------------------

            timestamp = (

                time.perf_counter()
                -
                experiment_start

            )


            # =================================================
            # DISPLAY TARGET
            # =================================================

            canvas = np.zeros(

                (
                    SCREEN_H,
                    SCREEN_W,
                    3
                ),

                dtype=np.uint8

            )


            # -------------------------------------------------
            # Target circle
            # -------------------------------------------------

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


            # -------------------------------------------------
            # Point number
            # -------------------------------------------------

            cv2.putText(

                canvas,

                (
                    f"Point "
                    f"{point_idx + 1}/"
                    f"{len(points)}"
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


            # ------------------------------------------------
            # ESC
            # ------------------------------------------------

            if cv2.waitKey(1) & 0xFF == 27:

                cap.release()

                csv_file.close()

                cv2.destroyAllWindows()

                return False


            # =================================================
            # HEAD POSE
            # =================================================

            head_pose = estimate_head_pose(

                frame

            )


            # =================================================
            # FACE
            # =================================================

            _, face_crop = get_face(

                frame,

                FACE_SIZE

            )


            # =================================================
            # DEFAULT VALUES
            # =================================================

            raw_yaw = None

            raw_pitch = None

            confidence = None

            head_yaw = None

            head_pitch = None

            head_roll = None

            reprojection_error = None


            # =================================================
            # HEAD POSE
            # =================================================

            if head_pose is not None:

                try:

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

                        head_pose[
                            "reprojection_error"
                        ]

                    )

                except (

                    KeyError,

                    TypeError,

                    ValueError

                ):

                    head_yaw = None

                    head_pitch = None

                    head_roll = None

                    reprojection_error = None


            # =================================================
            # RAW GAZE
            # =================================================

            if face_crop is not None:

                try:

                    (

                        raw_yaw,

                        raw_pitch,

                        confidence

                    ) = estimator.predict_gaze(

                        face_crop

                    )

                    if raw_yaw is not None:

                        raw_yaw = float(
                            raw_yaw
                        )

                    if raw_pitch is not None:

                        raw_pitch = float(
                            raw_pitch
                        )

                    if confidence is not None:

                        confidence = float(
                            confidence
                        )

                except Exception:

                    raw_yaw = None

                    raw_pitch = None

                    confidence = None


            # =================================================
            # VALID SAMPLE
            # =================================================

            if (

                raw_yaw is not None

                and raw_pitch is not None

                and head_yaw is not None

                and head_pitch is not None

                and head_roll is not None

                and confidence is not None

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

                    reprojection_error

                ])


        # ====================================================
        # DISCARD INITIAL SETTLING PERIOD
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
        # WRITE SAMPLES
        # ====================================================

        for row in kept_samples:

            writer.writerow(row)


        csv_file.flush()


        print(

            f"   Collected: "
            f"{len(point_samples)}"
            f" | Discarded: "
            f"{n_discard}"
            f" | Kept: "
            f"{len(kept_samples)}"

        )


    csv_file.close()

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("GAZE CALIBRATION + EVALUATION DATA COLLECTION")
    print("=" * 80)
    print()

    print(
        f"TRIAL: {TRIAL_NAME}"
    )

    print()

    print(
        "Two separate datasets will be collected:"
    )

    print(
        "  1. Calibration data"
    )

    print(
        "  2. Evaluation data"
    )

    print()

    print(
        "Ridge regression and Kalman filtering"
    )

    print(
        "are performed later in final_evaluation.py."
    )

    print()

    print(
        "Press ESC to stop."
    )

    print()


    # ========================================================
    # GAZE MODEL
    # ========================================================

    print("Loading gaze model...")

    estimator = GazeEstimator(

        MODEL_PATH,

        device="cpu"

    )

    print("Gaze model loaded.")


    # ========================================================
    # CAMERA
    # ========================================================

    print("Opening webcam...")

    cap = cv2.VideoCapture(

        CAMERA_ID

    )


    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return


    print("Webcam opened.")


    # ========================================================
    # WINDOW
    # ========================================================

    print("Opening calibration window...")

    window = create_window()

    print("Calibration window opened.")


    # ========================================================
    # CALIBRATION
    # ========================================================

    calibration_points = list(
        GRID_POINTS
    )


    success = collect_dataset(

        estimator,

        cap,

        window,

        CALIBRATION_FILE,

        calibration_points,

        "Calibration"

    )


    if not success:
        return


    # ========================================================
    # SHORT BREAK
    # ========================================================

    print()
    print("Calibration complete.")
    print()
    print("Take a short break.")
    print()
    time.sleep(5)


    # ========================================================
    # EVALUATION
    # ========================================================

    # Randomize the order so the evaluation does not always
    # follow the same 1 -> 9 sequence.

    evaluation_points = list(
        GRID_POINTS
    )

    random.shuffle(
        evaluation_points
    )


    print()
    print("Evaluation point order:")

    for i, point in enumerate(
        evaluation_points
    ):

        print(
            f"  {i + 1}: {point}"
        )


    success = collect_dataset(

        estimator,

        cap,

        window,

        EVALUATION_FILE,

        evaluation_points,

        "Evaluation"

    )


    if not success:
        return


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()


    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 80)
    print("DATA COLLECTION FINISHED")
    print("=" * 80)
    print()

    print(
        f"Calibration data:"
    )

    print(
        f"  {CALIBRATION_FILE}"
    )

    print()

    print(
        f"Evaluation data:"
    )

    print(
        f"  {EVALUATION_FILE}"
    )

    print()

    print(
        "Next step:"
    )

    print(
        "Run:"
    )

    print()

    print(
        "python final_evaluation.py "
        f"{CALIBRATION_FILE} "
        f"{EVALUATION_FILE}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()