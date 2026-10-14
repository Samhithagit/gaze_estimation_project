"""
Dataset collector for gaze_estimation_project.

Place this file in the project root beside phase2_calibration.py and main.py.
It does not change Module 1, Module 2, head_pose.py, or gaze_kalman.py.

Run:
    python collect_user_study.py P001 S01

Outputs:
    dataset/P001/S01/calibration_data.csv
    dataset/P001/S01/evaluation_data.csv
    dataset/P001/S01/evaluation_points.csv
    dataset/P001/S01/metadata.json
    dataset/P001/S01/preview/*.jpg

The collector logs every camera read attempt during active target collection,
including invalid frames, and saves a labelled subset of preview frames. It does
not save full face video.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import platform
from datetime import datetime, timezone

import cv2
import numpy as np
import tkinter as tk

from module1.eye_processing import get_face
from module2.gaze_estimation import GazeEstimator
from module3.head_pose import estimate_head_pose


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "L2CSNet_gaze360.pkl")
DATASET_ROOT = os.path.join(PROJECT_DIR, "dataset")

CAMERA_ID = 0
FACE_SIZE = 448
SAMPLES_PER_POINT = 20
SETTLING_SECONDS = 2.0
BREAK_SECONDS = 5.0
CALIBRATION_GRID_SIZE = 3
EVALUATION_POINTS = 9
EDGE_MARGIN = 25
MIN_RANDOM_DISTANCE = 80
PREVIEW_EVERY_N_VALID_SAMPLES = 20  # roughly one saved image per ~20 accepted samples
TARGET_RADIUS = 18

CSV_HEADER = [
    "participant_id", "session_id", "task_name",
    "frame_id", "timestamp_seconds", "timestamp_utc",
    "point_idx", "point_label", "target_x", "target_y",
    "screen_width", "screen_height",
    "raw_gaze_yaw_deg", "raw_gaze_pitch_deg", "raw_confidence",
    "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
    "head_reprojection_error_px",
    "frame_read_success", "face_detected", "gaze_valid", "head_pose_valid",
    "sample_valid", "rejection_reason", "preview_image"
]


def safe_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def create_grid(width, height):
    margin_x = int(width * 0.10)
    margin_y = int(height * 0.10)
    xs = np.linspace(margin_x, width - margin_x, CALIBRATION_GRID_SIZE).astype(int)
    ys = np.linspace(margin_y, height - margin_y, CALIBRATION_GRID_SIZE).astype(int)
    return [(int(x), int(y)) for y in ys for x in xs]


def create_random_points(width, height, count, rng):
    min_x, max_x = EDGE_MARGIN, width - EDGE_MARGIN
    min_y, max_y = EDGE_MARGIN, height - EDGE_MARGIN
    if max_x <= min_x or max_y <= min_y:
        raise RuntimeError(f"Canvas too small: {width}x{height}")
    points, attempts = [], 0
    while len(points) < count and attempts < 50000:
        attempts += 1
        candidate = (rng.randint(min_x, max_x), rng.randint(min_y, max_y))
        if all(math.dist(candidate, existing) >= MIN_RANDOM_DISTANCE for existing in points):
            points.append(candidate)
    if len(points) != count:
        raise RuntimeError("Could not generate enough separated random evaluation points.")
    return points


class TargetWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gaze Dataset Collection")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)
        self.root.overrideredirect(True)
        self.closed = False
        self.root.bind("<Escape>", self.on_escape)
        self.root.protocol("WM_DELETE_WINDOW", self.on_escape)
        self.root.update_idletasks()
        self.root.update()
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.update()
        self.width = int(self.canvas.winfo_width())
        self.height = int(self.canvas.winfo_height())
        if self.width <= 1 or self.height <= 1:
            self.width = int(self.root.winfo_screenwidth())
            self.height = int(self.root.winfo_screenheight())

    def on_escape(self, event=None):
        self.closed = True

    def draw(self, point_idx, total, x, y, status, progress=""):
        self.canvas.delete("all")
        r = TARGET_RADIUS
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="red", outline="white", width=2)
        self.canvas.create_text(25, 25, anchor="nw",
            text=f"Target {point_idx + 1}/{total}   {status} {progress}",
            fill="white", font=("Helvetica", 22, "bold"))
        self.root.update()

    def message(self, title, subtitle=""):
        self.canvas.delete("all")
        self.canvas.create_text(self.width//2, self.height//2-25, text=title,
            fill="white", font=("Helvetica", 30, "bold"))
        if subtitle:
            self.canvas.create_text(self.width//2, self.height//2+25, text=subtitle,
                fill="white", font=("Helvetica", 20))
        self.root.update()

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


def get_face_crop(frame):
    """
    Current project versions of get_face have returned either a crop directly
    or (annotated_frame, crop). Support both without changing Module 1.
    """
    result = get_face(frame, FACE_SIZE)
    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[1]
        return None
    return result


def pose_values(pose):
    if pose is None:
        return None, None, None, None
    return (
        safe_float(pose.get("yaw")),
        safe_float(pose.get("pitch")),
        safe_float(pose.get("roll")),
        safe_float(pose.get("reprojection_error"))
    )


def collect_phase(estimator, cap, gui, out_csv, preview_dir, points,
                  participant_id, session_id, task_name, screen_w, screen_h,
                  session_start, preview_counter):
    """Collect a phase and log every capture attempt during active target collection.

    Valid and invalid camera frames are written to CSV. Only valid samples count
    toward SAMPLES_PER_POINT. Raw webcam video is not saved; selected annotated
    previews are saved as JPG files.
    """
    os.makedirs(preview_dir, exist_ok=True)
    valid_total = 0
    frame_id = 0
    consecutive_camera_failures = 0
    reject_counts = {
        "camera_read_failure": 0,
        "face_missing": 0,
        "gaze_invalid": 0,
        "head_pose_invalid": 0,
        "gaze_and_head_invalid": 0,
    }

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        f.flush()

        for point_idx, (target_x, target_y) in enumerate(points):
            if gui.closed:
                return False, valid_total, reject_counts, preview_counter

            gui.draw(point_idx, len(points), target_x, target_y, "SETTLING")
            settle_start = time.perf_counter()
            while time.perf_counter() - settle_start < SETTLING_SECONDS:
                gui.root.update()
                if gui.closed:
                    return False, valid_total, reject_counts, preview_counter
                # Settling frames are deliberately not part of the labelled task.
                cap.read()
                time.sleep(0.005)

            accepted_for_point = 0
            gui.draw(point_idx, len(points), target_x, target_y, "COLLECTING",
                     f"0/{SAMPLES_PER_POINT}")

            while accepted_for_point < SAMPLES_PER_POINT:
                if gui.closed:
                    return False, valid_total, reject_counts, preview_counter

                ret, frame = cap.read()
                frame_id += 1
                ts = time.perf_counter() - session_start
                timestamp_utc = datetime.now(timezone.utc).isoformat()
                consecutive_camera_failures = 0 if ret else consecutive_camera_failures + 1

                # Common metadata is present for both successful and failed reads.
                row = {
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "task_name": task_name,
                    "frame_id": frame_id,
                    "timestamp_seconds": f"{ts:.6f}",
                    "timestamp_utc": timestamp_utc,
                    "point_idx": point_idx,
                    "point_label": f"{task_name}_target_{point_idx + 1:02d}",
                    "target_x": target_x,
                    "target_y": target_y,
                    "screen_width": screen_w,
                    "screen_height": screen_h,
                    "raw_gaze_yaw_deg": "",
                    "raw_gaze_pitch_deg": "",
                    "raw_confidence": "",
                    "head_yaw_deg": "",
                    "head_pitch_deg": "",
                    "head_roll_deg": "",
                    "head_reprojection_error_px": "",
                    "frame_read_success": int(bool(ret)),
                    "face_detected": 0,
                    "gaze_valid": 0,
                    "head_pose_valid": 0,
                    "sample_valid": 0,
                    "rejection_reason": "",
                    "preview_image": "",
                }

                if not ret or frame is None:
                    reject_counts["camera_read_failure"] += 1
                    row["rejection_reason"] = "camera_read_failure"
                    writer.writerow(row)
                    f.flush()
                    gui.root.update()
                    if consecutive_camera_failures >= MAX_CONSECUTIVE_CAMERA_FAILURES:
                        print(f"ERROR: {consecutive_camera_failures} consecutive camera read failures; stopping phase.")
                        return False, valid_total, reject_counts, preview_counter
                    continue

                # Preserve the project's existing mirrored-frame behavior.
                frame = cv2.flip(frame, 1)

                pose = None
                try:
                    pose = estimate_head_pose(frame.copy())
                except Exception:
                    pose = None
                hy, hp, hr, reproj = pose_values(pose)
                head_valid = all(v is not None for v in (hy, hp, hr))
                if not head_valid:
                    reject_counts["head_pose_invalid"] += 1

                face_crop = None
                try:
                    face_crop = get_face_crop(frame.copy())
                except Exception:
                    face_crop = None

                face_detected = face_crop is not None
                gy = gp = confidence = None
                if not face_detected:
                    reject_counts["face_missing"] += 1
                else:
                    try:
                        prediction = estimator.predict_gaze(face_crop)
                        if prediction is not None and len(prediction) >= 3:
                            gy, gp, confidence = map(safe_float, prediction[:3])
                    except Exception:
                        gy = gp = confidence = None

                gaze_valid = all(v is not None for v in (gy, gp, confidence))
                if not gaze_valid:
                    reject_counts["gaze_invalid"] += 1

                sample_valid = gaze_valid and head_valid
                if not gaze_valid and not head_valid:
                    rejection_reason = "gaze_and_head_invalid"
                    reject_counts["gaze_and_head_invalid"] += 1
                elif not gaze_valid:
                    rejection_reason = "gaze_invalid" if face_detected else "face_missing"
                elif not head_valid:
                    rejection_reason = "head_pose_invalid"
                else:
                    rejection_reason = "accepted"

                row.update({
                    "raw_gaze_yaw_deg": gy if gy is not None else "",
                    "raw_gaze_pitch_deg": gp if gp is not None else "",
                    "raw_confidence": confidence if confidence is not None else "",
                    "head_yaw_deg": hy if hy is not None else "",
                    "head_pitch_deg": hp if hp is not None else "",
                    "head_roll_deg": hr if hr is not None else "",
                    "head_reprojection_error_px": reproj if reproj is not None else "",
                    "face_detected": int(face_detected),
                    "gaze_valid": int(gaze_valid),
                    "head_pose_valid": int(head_valid),
                    "sample_valid": int(sample_valid),
                    "rejection_reason": rejection_reason,
                })

                # Save one labelled preview per target under the current settings.
                if sample_valid and accepted_for_point % PREVIEW_EVERY_N_VALID_SAMPLES == 0:
                    image_name = (
                        f"{participant_id}_{session_id}_{task_name}_"
                        f"frame{frame_id:07d}_point{point_idx:02d}_"
                        f"sample{accepted_for_point:03d}.jpg"
                    )
                    image_path = os.path.join(preview_dir, image_name)
                    annotated = frame.copy()
                    cv2.putText(annotated, f"{participant_id} {session_id} {task_name}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(annotated,
                                f"frame={frame_id} point={point_idx} target=({target_x},{target_y})",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(annotated,
                                f"head yaw={hy:.1f} pitch={hp:.1f} roll={hr:.1f}",
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(annotated,
                                f"gaze=({gy:.1f},{gp:.1f}) conf={confidence:.3f}",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    if cv2.imwrite(image_path, annotated):
                        row["preview_image"] = os.path.relpath(
                            image_path, os.path.dirname(out_csv)
                        ).replace("\\", "/")
                        preview_counter += 1

                # Crucial change: write EVERY successfully read frame, even if invalid.
                writer.writerow(row)
                f.flush()

                if sample_valid:
                    accepted_for_point += 1
                    valid_total += 1
                    gui.draw(point_idx, len(points), target_x, target_y, "COLLECTING",
                             f"{accepted_for_point}/{SAMPLES_PER_POINT}")
                gui.root.update()

    return True, valid_total, reject_counts, preview_counter

def write_csv_points(path, points):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["point_idx", "point_label", "target_x", "target_y"])
        for i, (x, y) in enumerate(points):
            writer.writerow([i, f"evaluation_target_{i+1:02d}", x, y])


def main():
    parser = argparse.ArgumentParser(description="Collect a labelled gaze user-study session.")
    parser.add_argument("participant_id", help="Unique pseudonymous ID, e.g. P001")
    parser.add_argument("session_id", nargs="?", default="S01", help="Session ID, e.g. S01")
    parser.add_argument("--device-id", default="UNSPECIFIED", help="Device label, e.g. LAPTOP_02")
    parser.add_argument("--camera-id", type=int, default=CAMERA_ID)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible random target generation")
    args = parser.parse_args()

    for value, label in ((args.participant_id, "participant_id"), (args.session_id, "session_id")):
        if not value.strip() or any(ch in value for ch in "/\\"):
            parser.error(f"{label} must be a non-empty ID and cannot contain path separators")

    session_dir = os.path.join(DATASET_ROOT, args.participant_id, args.session_id)
    if os.path.exists(session_dir) and os.listdir(session_dir):
        parser.error(f"Session folder already exists and is not empty: {session_dir}. Choose a new session ID; do not overwrite data.")
    preview_dir = os.path.join(session_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)

    model_path = os.path.abspath(args.model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Gaze model not found: {model_path}")

    estimator = GazeEstimator(model_path, device="cpu")
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera ID {args.camera_id}")

    camera_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = safe_float(cap.get(cv2.CAP_PROP_FPS))
    if not reported_fps or reported_fps <= 0:
        reported_fps = None

    gui = None
    try:
        gui = TargetWindow()
        screen_w, screen_h = gui.width, gui.height
        grid_points = create_grid(screen_w, screen_h)
        rng = random.Random(args.seed)
        eval_points = create_random_points(screen_w, screen_h, EVALUATION_POINTS, rng)
        write_csv_points(os.path.join(session_dir, "evaluation_points.csv"), eval_points)

        metadata = {
            "participant_id": args.participant_id,
            "session_id": args.session_id,
            "device_id": args.device_id,
            "collection_started_utc": datetime.now(timezone.utc).isoformat(),
            "computer_platform": platform.platform(),
            "python_version": platform.python_version(),
            "opencv_version": cv2.__version__,
            "camera_id": args.camera_id,
            "camera_resolution_width_px": camera_w,
            "camera_resolution_height_px": camera_h,
            "camera_reported_fps": reported_fps,
            "screen_width_px": screen_w,
            "screen_height_px": screen_h,
            "face_input_size_px": FACE_SIZE,
            "calibration_layout": "3x3 grid",
            "evaluation_layout": "9 random separated targets",
            "samples_per_target": SAMPLES_PER_POINT,
            "settling_seconds": SETTLING_SECONDS,
            "csv_policy": "Log every camera read attempt during active target collection, including invalid frames; settling frames are excluded",
            "preview_policy": "Save selected annotated JPG previews; preview images are not the dataset rows",
            "invalid_frame_policy": "Keep invalid rows with blank unavailable measurements and a rejection_reason",
            "preview_contains_identifiable_face_images": True,
            "full_video_saved": False,
            "gaze_model_path": os.path.relpath(model_path, PROJECT_DIR).replace("\\", "/"),
            "target_coordinates_reference": "Tkinter fullscreen canvas coordinate system",
            "labels": {
                "point_idx": "zero-based target index",
                "point_label": "human-readable target label",
                "target_x": "intended target center x in canvas pixels",
                "target_y": "intended target center y in canvas pixels",
                "sample_valid": "1 when both gaze and head-pose fields are finite",
                "rejection_reason": "accepted or reason the frame was not a valid sample",
                "frame_read_success": "1 when OpenCV successfully returned a camera frame"
            }
        }
        with open(os.path.join(session_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        session_start = time.perf_counter()
        preview_count = 0
        print("\nDATASET COLLECTION")
        print(f"Participant: {args.participant_id} | Session: {args.session_id}")
        print(f"Output folder: {session_dir}")
        print("Calibration: 3x3 grid, 20 accepted samples per target (at least 180 logged rows).")
        print("Evaluation: 9 random targets, 20 accepted samples per target (at least 180 logged rows).")
        print("ESC closes the target window and stops collection.")
        print("Preview images contain faces. Confirm consent/approval before collection.")

        ok, calibration_valid_count, calibration_rejects, preview_count = collect_phase(
            estimator, cap, gui,
            os.path.join(session_dir, "calibration_data.csv"),
            preview_dir, grid_points,
            args.participant_id, args.session_id, "calibration",
            screen_w, screen_h, session_start, preview_count
        )
        if not ok:
            print("Collection stopped during calibration. Partial CSV is retained; do not treat it as complete.")
            return

        gui.message("Calibration complete", f"Break: {BREAK_SECONDS:.0f} seconds")
        break_start = time.perf_counter()
        while time.perf_counter() - break_start < BREAK_SECONDS:
            gui.root.update()
            if gui.closed:
                print("Collection stopped by user.")
                return
            time.sleep(0.01)

        ok, evaluation_valid_count, evaluation_rejects, preview_count = collect_phase(
            estimator, cap, gui,
            os.path.join(session_dir, "evaluation_data.csv"),
            preview_dir, eval_points,
            args.participant_id, args.session_id, "evaluation",
            screen_w, screen_h, session_start, preview_count
        )
        if not ok:
            print("Collection stopped during evaluation. Partial CSV is retained; do not treat it as complete.")
            return

        metadata["collection_finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["preview_image_count"] = preview_count
        metadata["rejection_counts"] = {"calibration": calibration_rejects, "evaluation": evaluation_rejects}
        with open(os.path.join(session_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print("\nCOLLECTION COMPLETE")
        print(f"Calibration CSV: {os.path.join(session_dir, 'calibration_data.csv')}")
        print(f"Evaluation CSV:  {os.path.join(session_dir, 'evaluation_data.csv')}")
        print(f"Target labels:   {os.path.join(session_dir, 'evaluation_points.csv')}")
        print(f"Metadata:        {os.path.join(session_dir, 'metadata.json')}")
        print(f"Preview images:  {preview_dir}")
        print(f"Accepted calibration samples: {calibration_valid_count}")
        print(f"Accepted evaluation samples: {evaluation_valid_count}")
        print(f"Calibration rejection counts: {calibration_rejects}")
        print(f"Evaluation rejection counts: {evaluation_rejects}")
        print("CSV row counts may exceed 180 because invalid frames are retained.")
        print("Run validate_dataset.py on the session folder; it checks accepted samples, not total rows.")
    finally:
        cap.release()
        if gui is not None:
            gui.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
