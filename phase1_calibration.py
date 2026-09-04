"""
Phase 1 — Baseline (raw, unfiltered) accuracy + jitter measurement.

Shows a 9-point calibration grid on screen, one point at a time.
For each point, logs raw (yaw, pitch, confidence) for a fixation
window, then fits a simple mapping from (yaw, pitch) -> screen
pixel coordinates, and reports:

  - Per-point pixel error (predicted vs true fixation point)
  - Per-point jitter (std-dev of gaze position during fixation)
  - Overall mean error / mean jitter -> THIS IS YOUR BASELINE
    NUMBER that Phase 2 (Kalman) and Phase 3 (head-eye fusion)
    must beat.

Screen resolution is auto-detected at runtime (see get_screen_size
below) so this runs unmodified on any laptop -- no hardcoded
SCREEN_W/SCREEN_H to edit per machine.

Protocol:
  1. Run this fullscreen on the same screen/setup you'll use for
     the final system.
  2. A red dot appears at each of 9 grid points in turn. Fixate
     it steadily. The script auto-advances after FIXATION_SECONDS.
  3. First half of samples at each point are DISCARDED (saccade
     landing + settling time); only the last portion is used for
     the mapping fit and error calc.
  4. At the end, results are saved to phase1_results.csv and a
     summary is printed.

Run:
    python phase1_calibration.py
"""

import cv2
import numpy as np
import time
import csv

from module1.eye_processing import get_face
from module2.gaze_estimation import GazeEstimator

MODEL_PATH = r"D:\majorproject\gaze_project\models\L2CSNet_gaze360.pkl"
CAMERA_ID = 0

FIXATION_SECONDS = 3.0
DISCARD_FRACTION = 0.4   # discard first 40% of samples per point (settling time)
LOG_PATH = "phase1_results.csv"


def get_screen_size():
    """
    Auto-detect the actual display resolution so this script runs
    unmodified on any laptop. Tries several methods in order of
    reliability, falling back to a safe default if all else fails.
    """

    # Method 1: tkinter (usually available, no extra install,
    # correctly reports the *actual* screen the window manager
    # will place a fullscreen window on)
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass

    # Method 2: screeninfo package, if installed (handles
    # multi-monitor setups more explicitly -- picks primary)
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        primary = next((m for m in monitors if getattr(m, "is_primary", False)), monitors[0])
        return primary.width, primary.height
    except Exception:
        pass

    # Method 3: fall back to querying the OpenCV window itself
    # after creating it fullscreen (less reliable, platform-
    # dependent, used only as last resort)
    try:
        cv2.namedWindow("probe", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("probe", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        rect = cv2.getWindowImageRect("probe")
        cv2.destroyWindow("probe")
        if rect[2] > 0 and rect[3] > 0:
            return rect[2], rect[3]
    except Exception:
        pass

    # Last-resort default -- common laptop resolution
    print("WARNING: could not auto-detect screen size, defaulting to 1366x768")
    return 1366, 768


SCREEN_W, SCREEN_H = get_screen_size()
print(f"Detected screen resolution: {SCREEN_W}x{SCREEN_H}")

# 3x3 grid, inset from edges by 10% -- scales automatically with
# whatever resolution was detected above
MARGIN_X = int(SCREEN_W * 0.10)
MARGIN_Y = int(SCREEN_H * 0.10)

GRID_POINTS = [
    (MARGIN_X, MARGIN_Y), (SCREEN_W // 2, MARGIN_Y), (SCREEN_W - MARGIN_X, MARGIN_Y),
    (MARGIN_X, SCREEN_H // 2), (SCREEN_W // 2, SCREEN_H // 2), (SCREEN_W - MARGIN_X, SCREEN_H // 2),
    (MARGIN_X, SCREEN_H - MARGIN_Y), (SCREEN_W // 2, SCREEN_H - MARGIN_Y), (SCREEN_W - MARGIN_X, SCREEN_H - MARGIN_Y),
]


def fit_mapping(yaws, pitches, xs, ys):
    """
    Fit two independent 2nd-order polynomial regressions:
        screen_x = f(yaw, pitch)
        screen_y = g(yaw, pitch)
    using least squares. This is the same style of mapping used
    in most webcam gaze-interaction calibration (simple, robust
    with only 9 points; a full 3D geometric model needs more).
    """
    yaws = np.array(yaws)
    pitches = np.array(pitches)
    xs = np.array(xs)
    ys = np.array(ys)

    # Design matrix: [1, yaw, pitch, yaw^2, pitch^2, yaw*pitch]
    A = np.stack([
        np.ones_like(yaws), yaws, pitches,
        yaws ** 2, pitches ** 2, yaws * pitches
    ], axis=1)

    coef_x, _, _, _ = np.linalg.lstsq(A, xs, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)

    def predict(yaw, pitch):
        feat = np.array([1, yaw, pitch, yaw ** 2, pitch ** 2, yaw * pitch])
        px = float(feat @ coef_x)
        py = float(feat @ coef_y)
        return px, py

    return predict


def main():
    estimator = GazeEstimator(MODEL_PATH, device="cpu")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    window = "Phase 1 Calibration"
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    all_records = []  # (point_idx, true_x, true_y, yaw, pitch, confidence, t)

    for idx, (tx, ty) in enumerate(GRID_POINTS):
        print(f"Point {idx + 1}/{len(GRID_POINTS)} at ({tx}, {ty})")
        point_start = time.perf_counter()
        point_samples = []

        while time.perf_counter() - point_start < FIXATION_SECONDS:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            _, face_crop = get_face(frame, face_size=448)

            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 18, (0, 0, 255), -1)
            cv2.circle(canvas, (tx, ty), 18, (255, 255, 255), 2)
            cv2.putText(canvas, f"Point {idx + 1}/{len(GRID_POINTS)}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow(window, canvas)

            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

            if face_crop is None:
                continue

            yaw, pitch, confidence = estimator.predict_gaze(face_crop)
            if yaw is None:
                continue

            point_samples.append((yaw, pitch, confidence, time.perf_counter()))

        # Discard early "settling" samples, keep the tail
        n_discard = int(len(point_samples) * DISCARD_FRACTION)
        kept = point_samples[n_discard:]

        for yaw, pitch, confidence, t in kept:
            all_records.append((idx, tx, ty, yaw, pitch, confidence, t))

        print(f"  kept {len(kept)}/{len(point_samples)} samples")

    cap.release()
    cv2.destroyAllWindows()

    if len(all_records) < 20:
        print("Not enough samples collected -- try again with better lighting/distance.")
        return

    # ---- Fit mapping from all points ----
    yaws = [r[3] for r in all_records]
    pitches = [r[4] for r in all_records]
    txs = [r[1] for r in all_records]
    tys = [r[2] for r in all_records]

    predict = fit_mapping(yaws, pitches, txs, tys)

    # ---- Compute per-point error and jitter ----
    print("\n===== Phase 1 Baseline Results (raw, unfiltered) =====")
    header = f"{'Point':<8}{'TrueXY':<16}{'MeanErr(px)':<14}{'Jitter(px, std)':<18}{'N':<5}"
    print(header)

    per_point_err = []
    per_point_jitter = []

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_idx", "true_x", "true_y", "yaw", "pitch",
                          "confidence", "pred_x", "pred_y", "error_px"])

        for idx, (tx, ty) in enumerate(GRID_POINTS):
            pt_records = [r for r in all_records if r[0] == idx]
            if not pt_records:
                continue

            preds = [predict(r[3], r[4]) for r in pt_records]
            errors = [np.hypot(px - tx, py - ty) for px, py in preds]
            jitter_x = np.std([p[0] for p in preds])
            jitter_y = np.std([p[1] for p in preds])
            jitter = float(np.hypot(jitter_x, jitter_y))

            mean_err = float(np.mean(errors))
            per_point_err.append(mean_err)
            per_point_jitter.append(jitter)

            print(f"{idx:<8}{f'({tx},{ty})':<16}{mean_err:<14.1f}{jitter:<18.1f}{len(pt_records):<5}")

            for r, (px, py), err in zip(pt_records, preds, errors):
                writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5], px, py, err])

    print("\n----- Overall Baseline -----")
    print(f"Mean pixel error : {np.mean(per_point_err):.1f} px "
          f"(std across points: {np.std(per_point_err):.1f})")
    print(f"Mean jitter (std): {np.mean(per_point_jitter):.1f} px")
    print(f"\nSaved detailed log to {LOG_PATH}")
    print("This mean error / jitter is your BASELINE. Phase 2 (Adaptive")
    print("Kalman Filter) should reduce jitter significantly without")
    print("increasing mean error -- that's the comparison table for your paper.")


if __name__ == "__main__":
    main()