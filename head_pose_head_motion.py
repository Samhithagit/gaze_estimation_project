import cv2
import csv
import time
import math
import os
import sys
import contextlib

from module3.head_pose import estimate_head_pose
from module3.head_filter import HeadPoseFilter
from module3.head_motion import HeadMotionEstimator


# ============================================================
# FINAL HEAD POSE AND MOTION CONFIGURATION
# ============================================================

CAMERA_ID = 0
TARGET_FPS = 30.0

FILTER_ORDER = 2
CUTOFF_FREQUENCY = 1.0

MAX_GAP_FRAMES = 1

RESULTS_FILE = "head_pose_motion_results.csv"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value):
    """
    Convert a value to float safely.
    Returns None if conversion is not possible.
    """
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass

    return None


def suppress_stdout():
    """
    Context manager used to suppress verbose output from
    the gaze module if it is imported or used elsewhere.
    """
    return contextlib.redirect_stdout(open(os.devnull, "w"))


def calculate_jitter(values):
    """
    Calculate average absolute frame-to-frame change.

    This is used as a simple temporal jitter measure.
    """
    valid_values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]

    if len(valid_values) < 2:
        return None

    differences = [
        abs(valid_values[i] - valid_values[i - 1])
        for i in range(1, len(valid_values))
    ]

    if not differences:
        return None

    return sum(differences) / len(differences)


def calculate_range(values):
    """
    Calculate min/max range of valid values.
    """
    valid_values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]

    if not valid_values:
        return None, None

    return min(valid_values), max(valid_values)


def calculate_mean(values):
    """
    Calculate mean of valid values.
    """
    valid_values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]

    if not valid_values:
        return None

    return sum(valid_values) / len(valid_values)


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

def open_camera():
    """
    Open webcam and configure the requested FPS.
    """

    camera = cv2.VideoCapture(CAMERA_ID)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return None

    camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    return camera


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("FINAL HEAD POSE AND MOTION TEST")
    print("=" * 70)

    print()
    print("Configuration:")
    print(f"  Camera ID          : {CAMERA_ID}")
    print(f"  Target FPS         : {TARGET_FPS}")
    print(f"  Filter Order       : {FILTER_ORDER}")
    print(f"  Cutoff Frequency   : {CUTOFF_FREQUENCY} Hz")
    print(f"  Max Gap Frames     : {MAX_GAP_FRAMES}")
    print(f"  Results File       : {RESULTS_FILE}")
    print()

    # --------------------------------------------------------
    # INITIALIZE CAMERA
    # --------------------------------------------------------

    camera = open_camera()

    if camera is None:
        return

    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    if actual_fps is None or actual_fps <= 0:
        actual_fps = TARGET_FPS

    print(f"Camera reported FPS: {actual_fps:.2f}")
    print()

    # --------------------------------------------------------
    # INITIALIZE HEAD POSE FILTER
    # --------------------------------------------------------

    head_filter = HeadPoseFilter(
        fps=TARGET_FPS,
        cutoff_frequency=CUTOFF_FREQUENCY,
        order=FILTER_ORDER
    )

    # --------------------------------------------------------
    # INITIALIZE HEAD MOTION ESTIMATOR
    # --------------------------------------------------------

    motion_estimator = HeadMotionEstimator(
        max_gap_frames=MAX_GAP_FRAMES
    )

    # --------------------------------------------------------
    # CSV SETUP
    # --------------------------------------------------------

    csv_file = open(
        RESULTS_FILE,
        mode="w",
        newline=""
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame",
        "timestamp",
        "raw_yaw",
        "raw_pitch",
        "raw_roll",
        "filtered_yaw",
        "filtered_pitch",
        "filtered_roll",
        "reprojection_error",
        "head_motion"
    ])

    # --------------------------------------------------------
    # STORAGE FOR STATISTICS
    # --------------------------------------------------------

    raw_yaw_values = []
    raw_pitch_values = []
    raw_roll_values = []

    filtered_yaw_values = []
    filtered_pitch_values = []
    filtered_roll_values = []

    reprojection_errors = []
    head_motion_values = []

    total_frames = 0
    valid_pose_frames = 0
    missing_pose_frames = 0

    start_time = time.time()

    print("Starting webcam test...")
    print()
    print("Controls:")
    print("  Q / ESC : Quit")
    print()
    print("Move your head naturally while looking at the camera.")
    print("The filtered head pose and head motion are displayed.")
    print()

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = camera.read()

        if not ret:
            print("ERROR: Failed to read frame from webcam.")
            break

        total_frames += 1

        # ----------------------------------------------------
        # Preserve the existing horizontal flip behavior.
        # ----------------------------------------------------

        frame = cv2.flip(frame, 1)

        timestamp = time.time() - start_time

        # ----------------------------------------------------
        # HEAD POSE ESTIMATION
        # ----------------------------------------------------

        raw_head_pose = estimate_head_pose(frame)

        # ----------------------------------------------------
        # HANDLE MISSING HEAD POSE
        # ----------------------------------------------------

        if raw_head_pose is None:

            missing_pose_frames += 1

            filtered_head_pose = head_filter.update(None)

            head_motion = motion_estimator.update(None)

            raw_yaw = None
            raw_pitch = None
            raw_roll = None
            reprojection_error = None

            filtered_yaw = None
            filtered_pitch = None
            filtered_roll = None

        # ----------------------------------------------------
        # VALID HEAD POSE
        # ----------------------------------------------------

        else:

            valid_pose_frames += 1

            raw_yaw = safe_float(raw_head_pose["yaw"])
            raw_pitch = safe_float(raw_head_pose["pitch"])
            raw_roll = safe_float(raw_head_pose["roll"])

            reprojection_error = safe_float(
                raw_head_pose["reprojection_error"]
            )

            # ------------------------------------------------
            # TEMPORAL FILTERING
            # ------------------------------------------------

            filtered_head_pose = head_filter.update(
                raw_head_pose
            )

            # ------------------------------------------------
            # HEAD MOTION ESTIMATION
            # ------------------------------------------------

            head_motion = motion_estimator.update(
                filtered_head_pose
            )

            if filtered_head_pose is None:

                filtered_yaw = None
                filtered_pitch = None
                filtered_roll = None

            else:

                filtered_yaw = safe_float(
                    filtered_head_pose["yaw"]
                )

                filtered_pitch = safe_float(
                    filtered_head_pose["pitch"]
                )

                filtered_roll = safe_float(
                    filtered_head_pose["roll"]
                )

        # ----------------------------------------------------
        # STORE VALUES
        # ----------------------------------------------------

        if raw_yaw is not None:
            raw_yaw_values.append(raw_yaw)

        if raw_pitch is not None:
            raw_pitch_values.append(raw_pitch)

        if raw_roll is not None:
            raw_roll_values.append(raw_roll)

        if filtered_yaw is not None:
            filtered_yaw_values.append(filtered_yaw)

        if filtered_pitch is not None:
            filtered_pitch_values.append(filtered_pitch)

        if filtered_roll is not None:
            filtered_roll_values.append(filtered_roll)

        if reprojection_error is not None:
            reprojection_errors.append(
                reprojection_error
            )

        if head_motion is not None:
            head_motion_values.append(
                float(head_motion)
            )

        # ----------------------------------------------------
        # WRITE CSV
        # ----------------------------------------------------

        csv_writer.writerow([
            total_frames,
            f"{timestamp:.6f}",

            "" if raw_yaw is None else f"{raw_yaw:.6f}",
            "" if raw_pitch is None else f"{raw_pitch:.6f}",
            "" if raw_roll is None else f"{raw_roll:.6f}",

            "" if filtered_yaw is None else f"{filtered_yaw:.6f}",
            "" if filtered_pitch is None else f"{filtered_pitch:.6f}",
            "" if filtered_roll is None else f"{filtered_roll:.6f}",

            ""
            if reprojection_error is None
            else f"{reprojection_error:.6f}",

            ""
            if head_motion is None
            else f"{head_motion:.6f}"
        ])

        # ----------------------------------------------------
        # DISPLAY INFORMATION
        # ----------------------------------------------------

        display_frame = frame.copy()

        height, width = display_frame.shape[:2]

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            "Head Pose and Motion Estimation",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # RAW HEAD POSE
        # ----------------------------------------------------

        if raw_head_pose is not None:

            raw_text = (
                f"Raw  Y:{raw_yaw:7.2f}  "
                f"P:{raw_pitch:7.2f}  "
                f"R:{raw_roll:7.2f}"
            )

            cv2.putText(
                display_frame,
                raw_text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                display_frame,
                "Raw Head Pose: NA",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # FILTERED HEAD POSE
        # ----------------------------------------------------

        if filtered_head_pose is not None:

            filtered_text = (
                f"Filtered Y:{filtered_yaw:7.2f}  "
                f"P:{filtered_pitch:7.2f}  "
                f"R:{filtered_roll:7.2f}"
            )

            cv2.putText(
                display_frame,
                filtered_text,
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                display_frame,
                "Filtered Head Pose: NA",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # HEAD MOTION
        # ----------------------------------------------------

        if head_motion is not None:

            motion_text = (
                f"Head Motion: {head_motion:.4f} deg/frame"
            )

            cv2.putText(
                display_frame,
                motion_text,
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

        else:

            cv2.putText(
                display_frame,
                "Head Motion: NA",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # REPROJECTION ERROR
        # ----------------------------------------------------

        if reprojection_error is not None:

            error_text = (
                f"Reprojection Error: "
                f"{reprojection_error:.2f} px"
            )

            cv2.putText(
                display_frame,
                error_text,
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------------
        # FRAME COUNT
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            f"Frames: {total_frames}",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # VALID FRAME COUNT
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            f"Valid Pose: {valid_pose_frames}",
            (20, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.imshow(
            "Head Pose and Motion",
            display_frame
        )

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    # ========================================================
    # CLEANUP
    # ========================================================

    end_time = time.time()

    elapsed_time = end_time - start_time

    camera.release()
    csv_file.close()

    cv2.destroyAllWindows()

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL HEAD POSE AND MOTION TEST COMPLETE")
    print("=" * 70)

    print()

    print("GENERAL STATISTICS")
    print("-" * 70)

    print(f"Total frames           : {total_frames}")
    print(f"Valid head-pose frames : {valid_pose_frames}")
    print(f"Missing head-pose      : {missing_pose_frames}")

    if total_frames > 0:

        valid_percentage = (
            valid_pose_frames / total_frames
        ) * 100.0

        print(
            f"Valid pose percentage  : "
            f"{valid_percentage:.2f}%"
        )

    print(f"Elapsed time           : {elapsed_time:.2f} sec")

    if elapsed_time > 0:

        measured_fps = (
            total_frames / elapsed_time
        )

        print(
            f"Measured FPS           : "
            f"{measured_fps:.2f}"
        )

    # ========================================================
    # RAW HEAD POSE STATISTICS
    # ========================================================

    print()
    print("RAW HEAD POSE")
    print("-" * 70)

    raw_yaw_min, raw_yaw_max = calculate_range(
        raw_yaw_values
    )

    raw_pitch_min, raw_pitch_max = calculate_range(
        raw_pitch_values
    )

    raw_roll_min, raw_roll_max = calculate_range(
        raw_roll_values
    )

    raw_yaw_jitter = calculate_jitter(
        raw_yaw_values
    )

    raw_pitch_jitter = calculate_jitter(
        raw_pitch_values
    )

    raw_roll_jitter = calculate_jitter(
        raw_roll_values
    )

    if raw_yaw_min is not None:
        print(
            f"Yaw   range : "
            f"{raw_yaw_min:.2f} -> {raw_yaw_max:.2f} deg"
        )

    if raw_pitch_min is not None:
        print(
            f"Pitch range : "
            f"{raw_pitch_min:.2f} -> {raw_pitch_max:.2f} deg"
        )

    if raw_roll_min is not None:
        print(
            f"Roll  range : "
            f"{raw_roll_min:.2f} -> {raw_roll_max:.2f} deg"
        )

    if raw_yaw_jitter is not None:
        print(
            f"Yaw   jitter : "
            f"{raw_yaw_jitter:.4f} deg/frame"
        )

    if raw_pitch_jitter is not None:
        print(
            f"Pitch jitter : "
            f"{raw_pitch_jitter:.4f} deg/frame"
        )

    if raw_roll_jitter is not None:
        print(
            f"Roll  jitter : "
            f"{raw_roll_jitter:.4f} deg/frame"
        )

    # ========================================================
    # FILTERED HEAD POSE STATISTICS
    # ========================================================

    print()
    print("FILTERED HEAD POSE")
    print("-" * 70)

    filtered_yaw_min, filtered_yaw_max = calculate_range(
        filtered_yaw_values
    )

    filtered_pitch_min, filtered_pitch_max = calculate_range(
        filtered_pitch_values
    )

    filtered_roll_min, filtered_roll_max = calculate_range(
        filtered_roll_values
    )

    filtered_yaw_jitter = calculate_jitter(
        filtered_yaw_values
    )

    filtered_pitch_jitter = calculate_jitter(
        filtered_pitch_values
    )

    filtered_roll_jitter = calculate_jitter(
        filtered_roll_values
    )

    if filtered_yaw_min is not None:
        print(
            f"Yaw   range : "
            f"{filtered_yaw_min:.2f} -> "
            f"{filtered_yaw_max:.2f} deg"
        )

    if filtered_pitch_min is not None:
        print(
            f"Pitch range : "
            f"{filtered_pitch_min:.2f} -> "
            f"{filtered_pitch_max:.2f} deg"
        )

    if filtered_roll_min is not None:
        print(
            f"Roll  range : "
            f"{filtered_roll_min:.2f} -> "
            f"{filtered_roll_max:.2f} deg"
        )

    if filtered_yaw_jitter is not None:
        print(
            f"Yaw   jitter : "
            f"{filtered_yaw_jitter:.4f} deg/frame"
        )

    if filtered_pitch_jitter is not None:
        print(
            f"Pitch jitter : "
            f"{filtered_pitch_jitter:.4f} deg/frame"
        )

    if filtered_roll_jitter is not None:
        print(
            f"Roll  jitter : "
            f"{filtered_roll_jitter:.4f} deg/frame"
        )

    # ========================================================
    # JITTER REDUCTION
    # ========================================================

    print()
    print("JITTER REDUCTION")
    print("-" * 70)

    reductions = []

    if (
        raw_yaw_jitter is not None
        and filtered_yaw_jitter is not None
        and raw_yaw_jitter > 0
    ):

        yaw_reduction = (
            (raw_yaw_jitter - filtered_yaw_jitter)
            / raw_yaw_jitter
        ) * 100.0

        reductions.append(yaw_reduction)

        print(
            f"Yaw   reduction : "
            f"{yaw_reduction:.2f}%"
        )

    if (
        raw_pitch_jitter is not None
        and filtered_pitch_jitter is not None
        and raw_pitch_jitter > 0
    ):

        pitch_reduction = (
            (raw_pitch_jitter - filtered_pitch_jitter)
            / raw_pitch_jitter
        ) * 100.0

        reductions.append(pitch_reduction)

        print(
            f"Pitch reduction : "
            f"{pitch_reduction:.2f}%"
        )

    if (
        raw_roll_jitter is not None
        and filtered_roll_jitter is not None
        and raw_roll_jitter > 0
    ):

        roll_reduction = (
            (raw_roll_jitter - filtered_roll_jitter)
            / raw_roll_jitter
        ) * 100.0

        reductions.append(roll_reduction)

        print(
            f"Roll  reduction : "
            f"{roll_reduction:.2f}%"
        )

    if reductions:

        average_reduction = (
            sum(reductions) / len(reductions)
        )

        print(
            f"Average reduction: "
            f"{average_reduction:.2f}%"
        )

    # ========================================================
    # REPROJECTION ERROR
    # ========================================================

    print()
    print("HEAD POSE QUALITY")
    print("-" * 70)

    mean_reprojection_error = calculate_mean(
        reprojection_errors
    )

    if mean_reprojection_error is not None:

        print(
            f"Mean reprojection error : "
            f"{mean_reprojection_error:.3f} px"
        )

    # ========================================================
    # HEAD MOTION STATISTICS
    # ========================================================

    print()
    print("HEAD MOTION")
    print("-" * 70)

    mean_head_motion = calculate_mean(
        head_motion_values
    )

    motion_min, motion_max = calculate_range(
        head_motion_values
    )

    if mean_head_motion is not None:

        print(
            f"Mean head motion : "
            f"{mean_head_motion:.4f} deg/frame"
        )

    if motion_min is not None:

        print(
            f"Minimum motion  : "
            f"{motion_min:.4f} deg/frame"
        )

    if motion_max is not None:

        print(
            f"Maximum motion  : "
            f"{motion_max:.4f} deg/frame"
        )

    # ========================================================
    # OUTPUT FILE
    # ========================================================

    print()
    print("OUTPUT")
    print("-" * 70)

    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )

    print()
    print("=" * 70)
    print("HEAD POSE + MOTION PIPELINE FINISHED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()