import cv2
import numpy as np
import time
import csv
import os
import sys
import random
import tkinter as tk

from module1.eye_processing import get_face
from module2.gaze_estimation import GazeEstimator
from module3.head_pose import estimate_head_pose


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Project-relative paths
# ------------------------------------------------------------

# Assumes:
#
# gaze_project/
# ├── phase2_calibration.py
# ├── models/
# │   └── L2CSNet_gaze360.pkl
# ├── module1/
# ├── module2/
# ├── module3/
# └── trials/

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "L2CSNet_gaze360.pkl"
)

CAMERA_ID = 0

# Time given to the participant to settle
# on a new target.
SETTLING_SECONDS = 2.0

# EXACT number of VALID samples collected
# for every point.
SAMPLES_PER_POINT = 20

# Face/eye processing size used by Module 1.
FACE_SIZE = 448

GUI_UPDATE_INTERVAL = 0.10  # Refresh GUI at most 10 times/second
DIAGNOSTIC_INTERVAL = 100   # Report rejected frames every 100 rejects



# ============================================================
# TRIAL
# ============================================================

if len(sys.argv) >= 2:
    TRIAL_NAME = sys.argv[1]
else:
    TRIAL_NAME = "trial_01"


TRIALS_DIR = os.path.join(
    PROJECT_DIR,
    "trials"
)

TRIAL_DIR = os.path.join(
    TRIALS_DIR,
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

# Exact number of random evaluation targets.
RANDOM_EVALUATION_POINTS = 9

# Keep the target fully visible, but otherwise allow it
# anywhere on the usable screen.
RANDOM_EDGE_MARGIN = 25

# Prevent random points from landing almost on top of each other.
# This is only for usability; it does NOT create a grid.
MIN_RANDOM_DISTANCE = 80

# Save the exact random coordinates used for this trial.
EVALUATION_POINTS_FILE = os.path.join(
    TRIAL_DIR,
    "evaluation_points.csv"
)


# ============================================================
# GUI GRID
# ============================================================

def create_grid(width, height):
    """
    Create a 3 x 3 calibration grid using the actual
    Tkinter fullscreen canvas dimensions.

    3 columns x 3 rows = 9 points.

    Coordinates are relative to the calibration canvas,
    so no physical screen-coordinate conversion is needed.
    """

    # Keep targets away from the extreme edges.
    margin_x = int(width * 0.10)
    margin_y = int(height * 0.10)

    # 3 evenly spaced X positions.
    x_positions = np.linspace(
        margin_x,
        width - margin_x,
        3
    ).astype(int)

    # 3 evenly spaced Y positions.
    y_positions = np.linspace(
        margin_y,
        height - margin_y,
        3
    ).astype(int)

    # Create 9 points:
    #
    # Row 1: 3 points
    # Row 2: 3 points
    # Row 3: 3 points
    #
    # Total = 9
    grid_points = []

    for y in y_positions:
        for x in x_positions:
            grid_points.append(
                (
                    int(x),
                    int(y)
                )
            )

    return grid_points


# ============================================================
# RANDOM EVALUATION POINTS
# ============================================================

def create_random_evaluation_points(width, height, number_of_points):
    """
    Generate random evaluation locations across the screen.

    The calibration grid remains fixed 3x3.
    Only the evaluation targets are randomized.

    A small edge margin keeps the complete target visible.
    A minimum distance avoids accidental near-duplicate
    target locations.
    """

    min_x = RANDOM_EDGE_MARGIN
    max_x = width - RANDOM_EDGE_MARGIN

    min_y = RANDOM_EDGE_MARGIN
    max_y = height - RANDOM_EDGE_MARGIN

    if max_x <= min_x or max_y <= min_y:
        raise RuntimeError(
            f"Screen too small for random evaluation: "
            f"{width} x {height}"
        )

    points = []
    attempts = 0
    max_attempts = 50000

    while (
        len(points) < number_of_points
        and attempts < max_attempts
    ):
        attempts += 1

        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)

        candidate = (x, y)

        too_close = False

        for existing in points:
            distance = np.sqrt(
                (candidate[0] - existing[0]) ** 2
                + (candidate[1] - existing[1]) ** 2
            )

            if distance < MIN_RANDOM_DISTANCE:
                too_close = True
                break

        if not too_close:
            points.append(candidate)

    if len(points) != number_of_points:
        raise RuntimeError(
            "Could not generate enough separated random "
            "evaluation points."
        )

    return points


def save_evaluation_points(points, output_file):
    """
    Save the exact random target coordinates used in
    this trial so the evaluation is reproducible.
    """

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "point_idx",
            "target_x",
            "target_y"
        ])

        for point_idx, (x, y) in enumerate(points):
            writer.writerow([
                point_idx,
                x,
                y
            ])


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
# TKINTER FULLSCREEN CALIBRATION WINDOW
# ============================================================

class CalibrationWindow:
    """
    Native fullscreen calibration display.

    This replaces the OpenCV fullscreen display so that
    target coordinates are always coordinates of the actual
    Tkinter canvas. This avoids macOS/OpenCV Retina scaling
    and fullscreen coordinate mismatches.
    """

    def __init__(self):
        self.root = tk.Tk()

        self.root.title("Gaze Calibration")
        self.root.configure(bg="black")

        # Native fullscreen.
        self.root.attributes("-fullscreen", True)

        # Remove normal window borders.
        self.root.overrideredirect(True)

        self.closed = False

        # ESC closes the calibration.
        self.root.bind("<Escape>", self._on_escape)

        # Also handle window close if available.
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self._on_close
        )

        # Force the fullscreen window to be created.
        self.root.update_idletasks()
        self.root.update()

        # IMPORTANT:
        # Use the actual Tkinter window dimensions.
        self.width = int(self.root.winfo_width())
        self.height = int(self.root.winfo_height())

        # Safety fallback for platforms where the window
        # reports 1x1 before fullscreen settles.
        if self.width <= 1 or self.height <= 1:
            self.width = int(self.root.winfo_screenwidth())
            self.height = int(self.root.winfo_screenheight())

        # Canvas uses exactly the same coordinate system
        # used to create the target grid.
        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg="black",
            highlightthickness=0,
            bd=0
        )

        self.canvas.pack(
            fill="both",
            expand=True
        )

        self.root.update()

        # Re-read after canvas has expanded.
        self.width = int(self.canvas.winfo_width())
        self.height = int(self.canvas.winfo_height())

        if self.width <= 1 or self.height <= 1:
            self.width = int(self.root.winfo_screenwidth())
            self.height = int(self.root.winfo_screenheight())

        self.root.update()

    def _on_escape(self, event=None):
        self.closed = True

    def _on_close(self):
        self.closed = True

    def clear(self):
        self.canvas.delete("all")

    def show_settling(self, point_number, total_points,
                      target_x, target_y):
        """
        Display target and settling message.
        """

        self.clear()

        # Target.
        radius = 18

        self.canvas.create_oval(
            target_x - radius,
            target_y - radius,
            target_x + radius,
            target_y + radius,
            fill="red",
            outline="white",
            width=2
        )

        # Point number.
        self.canvas.create_text(
            30,
            40,
            anchor="nw",
            text=f"Point {point_number}/{total_points}",
            fill="white",
            font=("Helvetica", 24, "bold")
        )

        # Settling message.
        self.canvas.create_text(
            self.width // 2,
            self.height // 2,
            text="Get ready...",
            fill="white",
            font=("Helvetica", 30, "bold")
        )

        self.root.update()

    def show_collecting(self, point_number, total_points,
                        target_x, target_y,
                        valid_count, samples_required):
        """
        Display target and collection progress.
        """

        self.clear()

        # Target.
        radius = 18

        self.canvas.create_oval(
            target_x - radius,
            target_y - radius,
            target_x + radius,
            target_y + radius,
            fill="red",
            outline="white",
            width=2
        )

        # Point number.
        self.canvas.create_text(
            30,
            40,
            anchor="nw",
            text=f"Point {point_number}/{total_points}",
            fill="white",
            font=("Helvetica", 24, "bold")
        )

        # Sample progress.
        self.canvas.create_text(
            30,
            80,
            anchor="nw",
            text=f"Samples: {valid_count}/{samples_required}",
            fill="white",
            font=("Helvetica", 22, "bold")
        )

        self.root.update()

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


# ============================================================
# COLLECT ONE DATASET
# ============================================================


def collect_dataset(
    estimator,
    cap,
    gui,
    output_file,
    points,
    dataset_name
):
    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} DATA COLLECTION")
    print("=" * 70)
    print(f"Output: {output_file}")
    print(f"Targets: {len(points)}")
    print(f"Valid samples per target: {SAMPLES_PER_POINT}")
    print(f"Settling time: {SETTLING_SECONDS:.1f}s")

    frame_number = 0
    experiment_start = time.perf_counter()
    gui_closed = False

    rejection_totals = {
        "head_pose_missing": 0,
        "head_pose_invalid": 0,
        "face_missing": 0,
        "gaze_failed": 0,
        "non_finite": 0,
    }

    try:
        with open(
            output_file, "w", newline="", encoding="utf-8"
        ) as csv_file:

            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADER)

            for point_idx, (target_x, target_y) in enumerate(points):
                if gui.closed:
                    return False

                point_number = point_idx + 1

                print(
                    f"\nPoint {point_number}/{len(points)} "
                    f"-> ({target_x}, {target_y})"
                )

                # Show settling screen ONCE per target.
                gui.show_settling(
                    point_number, len(points), target_x, target_y
                )

                settling_start = time.perf_counter()

                while (
                    time.perf_counter() - settling_start
                    < SETTLING_SECONDS
                ):
                    gui.root.update()

                    if gui.closed:
                        return False

                    # Keep the GUI responsive without redrawing it
                    # for every camera frame.
                    time.sleep(0.01)

                point_samples = []
                valid_count = 0
                rejected_count = 0
                last_gui_update = 0.0
                last_diagnostic = 0
                point_start = time.perf_counter()

                timings = {
                    "head": 0.0,
                    "face": 0.0,
                    "gaze": 0.0,
                }

                print(
                    f"Collecting {SAMPLES_PER_POINT} valid samples..."
                )

                while valid_count < SAMPLES_PER_POINT:
                    ret, frame = cap.read()

                    if not ret:
                        rejection_totals["face_missing"] += 1
                        rejected_count += 1
                        gui.root.update()

                        if gui.closed:
                            return False

                        continue

                    frame_number += 1
                    frame = cv2.flip(frame, 1)

                    timestamp = (
                        time.perf_counter() - experiment_start
                    )

                    # Update the GUI at most 10 times per second.
                    now = time.perf_counter()

                    if now - last_gui_update >= GUI_UPDATE_INTERVAL:
                        gui.show_collecting(
                            point_number,
                            len(points),
                            target_x,
                            target_y,
                            valid_count,
                            SAMPLES_PER_POINT
                        )
                        last_gui_update = now

                    if gui.closed:
                        return False

                    # -------------------------------
                    # HEAD POSE
                    # -------------------------------
                    t0 = time.perf_counter()

                    try:
                        head_pose = estimate_head_pose(frame)
                    except Exception as exc:
                        head_pose = None
                        if rejected_count == 0:
                            print(f"Head-pose error: {exc}")

                    timings["head"] += time.perf_counter() - t0

                    head_yaw = head_pitch = head_roll = None
                    reprojection_error = None

                    if head_pose is None:
                        rejection_totals["head_pose_missing"] += 1
                    else:
                        try:
                            head_yaw = float(head_pose["yaw"])
                            head_pitch = float(head_pose["pitch"])
                            head_roll = float(head_pose["roll"])

                            error_value = head_pose.get(
                                "reprojection_error"
                            )
                            if error_value is not None:
                                reprojection_error = float(error_value)

                            if not all(
                                np.isfinite(v)
                                for v in (
                                    head_yaw, head_pitch, head_roll
                                )
                            ):
                                raise ValueError(
                                    "Non-finite head-pose values"
                                )

                        except (KeyError, TypeError, ValueError) as exc:
                            head_yaw = head_pitch = head_roll = None
                            reprojection_error = None
                            rejection_totals["head_pose_invalid"] += 1

                    # -------------------------------
                    # FACE DETECTION
                    # -------------------------------
                    t0 = time.perf_counter()

                    try:
                        _, face_crop = get_face(frame, FACE_SIZE)
                    except Exception as exc:
                        face_crop = None
                        if rejected_count == 0:
                            print(f"Face-detection error: {exc}")

                    timings["face"] += time.perf_counter() - t0

                    raw_yaw = raw_pitch = confidence = None

                    if face_crop is None:
                        rejection_totals["face_missing"] += 1
                    else:
                        # -------------------------------
                        # GAZE INFERENCE
                        # -------------------------------
                        t0 = time.perf_counter()

                        try:
                            raw_yaw, raw_pitch, confidence = (
                                estimator.predict_gaze(face_crop)
                            )
                        except Exception as exc:
                            if rejected_count == 0:
                                print(f"Gaze-inference error: {exc}")
                            rejection_totals["gaze_failed"] += 1

                        timings["gaze"] += time.perf_counter() - t0

                    # -------------------------------
                    # VALIDITY CHECK
                    # -------------------------------
                    values = (
                        raw_yaw, raw_pitch, confidence,
                        head_yaw, head_pitch, head_roll
                    )

                    valid = all(v is not None for v in values)

                    if valid:
                        try:
                            values = tuple(float(v) for v in values)
                            valid = all(np.isfinite(v) for v in values)
                        except (TypeError, ValueError):
                            valid = False

                    if valid:
                        (
                            raw_yaw, raw_pitch, confidence,
                            head_yaw, head_pitch, head_roll
                        ) = values

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

                        valid_count += 1

                    else:
                        rejected_count += 1

                        if not all(v is not None for v in (
                            head_yaw, head_pitch, head_roll
                        )):
                            # Detailed cause was recorded above.
                            pass

                        if (
                            rejected_count - last_diagnostic
                            >= DIAGNOSTIC_INTERVAL
                        ):
                            last_diagnostic = rejected_count
                            elapsed = time.perf_counter() - point_start

                            print(
                                f"  Still collecting: "
                                f"{valid_count}/{SAMPLES_PER_POINT} valid; "
                                f"{rejected_count} rejected; "
                                f"{elapsed:.1f}s elapsed"
                            )
                            print(
                                f"  Rejection totals: {rejection_totals}"
                            )

                    # Save partial progress periodically, so an
                    # interrupted point does not erase earlier points.
                    if valid_count and valid_count % 5 == 0:
                        pass

                # Write exactly 20 valid samples for this point.
                writer.writerows(point_samples)
                csv_file.flush()

                elapsed = time.perf_counter() - point_start

                print(
                    f"  COMPLETE: {valid_count}/{SAMPLES_PER_POINT} "
                    f"samples in {elapsed:.1f}s; "
                    f"{rejected_count} rejected"
                )

                processed_frames = max(1, frame_number)
                print(
                    "  Average processing time/frame: "
                    f"head={timings['head'] / processed_frames:.3f}s, "
                    f"face={timings['face'] / processed_frames:.3f}s, "
                    f"gaze={timings['gaze'] / processed_frames:.3f}s"
                )

    except Exception as exc:
        print(f"\nCollection failed: {exc}")
        raise

    print("\nDataset collection completed.")
    print(f"Rejection totals: {rejection_totals}")
    return True



# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "GAZE CALIBRATION + EVALUATION DATA COLLECTION"
    )
    print("=" * 80)
    print()

    print(
        f"TRIAL: {TRIAL_NAME}"
    )

    print()

    print(
        "Collection protocol:"
    )

    print(
        "  3 columns × 3 rows = 9 calibration points"
    )

    print(
        f"  {SAMPLES_PER_POINT} VALID samples per point"
    )

    print(
        f"  Calibration total = "
        f"{9 * SAMPLES_PER_POINT} valid samples"
    )

    print(
        f"  {SETTLING_SECONDS:.1f} second settling period"
    )

    print(
        "  Calibration = fixed 3x3 grid"
    )

    print(
        "  Evaluation = 9 random screen positions"
    )

    print(
        "  Coordinates are relative to the fullscreen GUI"
    )

    print()

    print(
        "Press ESC to stop."
    )

    print()

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "ERROR: Gaze model not found:"
        )

        print(
            f"  {MODEL_PATH}"
        )

        print()

        print(
            "Make sure L2CSNet_gaze360.pkl is inside:"
        )

        print(
            "  models/"
        )

        return

    # ========================================================
    # GAZE MODEL
    # ========================================================

    print(
        "Loading gaze model..."
    )

    estimator = GazeEstimator(
        MODEL_PATH,
        device="cpu"
    )

    print(
        "Gaze model loaded."
    )

    # ========================================================
    # CAMERA
    # ========================================================

    print(
        "Opening webcam..."
    )

    cap = cv2.VideoCapture(
        CAMERA_ID
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return

    print(
        "Webcam opened."
    )

    # --------------------------------------------------------
    # Print actual camera resolution
    # --------------------------------------------------------

    camera_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    camera_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Camera resolution: "
        f"{camera_width} x {camera_height}"
    )

    # ========================================================
    # FULLSCREEN CALIBRATION GUI
    # ========================================================

    print(
        "Opening native fullscreen calibration window..."
    )

    try:

        gui = CalibrationWindow()

    except Exception as e:

        print(
            f"ERROR: Could not create calibration window: {e}"
        )

        cap.release()
        return

    print(
        "Calibration window opened."
    )

    print(
        f"Using GUI coordinate space: "
        f"{gui.width} x {gui.height}"
    )

    # ========================================================
    # GRID
    # ========================================================

    GRID_POINTS = create_grid(
        gui.width,
        gui.height
    )

    print()
    print(
        "3 × 3 Calibration grid:"
    )

    for i, point in enumerate(
        GRID_POINTS
    ):

        print(
            f"  {i + 1}: {point}"
        )

    print()

    # ========================================================
    # CALIBRATION
    # ========================================================

    calibration_points = list(
        GRID_POINTS
    )

    success = collect_dataset(

        estimator,
        cap,
        gui,

        CALIBRATION_FILE,

        calibration_points,

        "Calibration"
    )

    if not success:

        cap.release()
        gui.close()

        return

    # ========================================================
    # SHORT BREAK
    # ========================================================

    print()
    print(
        "Calibration complete."
    )

    print()
    print(
        "Take a short break."
    )

    print()

    # Keep GUI alive during break and show a message.
    gui.clear()

    gui.canvas.create_text(
        gui.width // 2,
        gui.height // 2 - 30,
        text="Calibration complete",
        fill="white",
        font=("Helvetica", 32, "bold")
    )

    gui.canvas.create_text(
        gui.width // 2,
        gui.height // 2 + 30,
        text="Take a short break...",
        fill="white",
        font=("Helvetica", 24)
    )

    gui.root.update()

    break_start = time.perf_counter()

    while (
        time.perf_counter()
        - break_start
        <
        5.0
    ):

        gui.root.update()

        if gui.closed:

            cap.release()
            gui.close()

            return

        time.sleep(0.01)

    # ========================================================
    # EVALUATION
    # ========================================================

    # IMPORTANT:
    # Calibration stays fixed 3x3.
    #
    # Evaluation uses 9 NEW random screen positions.
    # These positions are NOT the calibration grid.
    # They are generated independently for this trial.

    evaluation_points = create_random_evaluation_points(
        gui.width,
        gui.height,
        RANDOM_EVALUATION_POINTS
    )

    save_evaluation_points(
        evaluation_points,
        EVALUATION_POINTS_FILE
    )

    print()
    print(
        "Random evaluation points:"
    )

    for i, point in enumerate(
        evaluation_points
    ):

        print(
            f"  {i + 1}: {point}"
        )

    print()

    success = collect_dataset(

        estimator,
        cap,
        gui,

        EVALUATION_FILE,

        evaluation_points,

        "Random Evaluation"
    )

    if not success:

        cap.release()
        gui.close()

        return

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    gui.close()

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 80)
    print(
        "DATA COLLECTION FINISHED"
    )
    print("=" * 80)
    print()

    print(
        "Calibration data:"
    )

    print(
        f"  {CALIBRATION_FILE}"
    )

    print()

    print(
        "Evaluation data:"
    )

    print(
        f"  {EVALUATION_FILE}"
    )

    print()

    print(
        "Random evaluation coordinates:"
    )

    print(
        f"  {EVALUATION_POINTS_FILE}"
    )

    print()

    print(
        "GUI coordinate space:"
    )

    print(
        f"  {gui.width} x {gui.height}"
    )

    print()

    print(
        "Expected samples:"
    )

    print(
        f"  Calibration: "
        f"{len(GRID_POINTS)} × "
        f"{SAMPLES_PER_POINT} = "
        f"{len(GRID_POINTS) * SAMPLES_PER_POINT}"
    )

    print(
        f"  Evaluation:  "
        f"{len(evaluation_points)} × "
        f"{SAMPLES_PER_POINT} = "
        f"{len(evaluation_points) * SAMPLES_PER_POINT}"
    )

    print()

    print(
        "Next step:"
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