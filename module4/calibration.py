# module3/calibration.py
# =============================================================
# 9-point calibration UI.
# Shows dots on screen, collects gaze samples per dot,
# returns structured calibration data ready for ScreenMapper.
# =============================================================

import cv2
import json
import time
import numpy as np
from pathlib import Path

# =========================
# 9-Point Grid (normalised)
# =========================

CALIBRATION_POINTS_NORM = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),   # top row
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),   # middle row
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),   # bottom row
]


# =========================
# Screen Size Helper
# =========================

def _get_screen_size():
    """Get logical screen resolution using pyautogui."""
    import pyautogui
    w, h = pyautogui.size()
    return int(w), int(h)


# =========================
# Dot Renderer
# =========================

def _draw_calibration_frame(canvas, tx, ty, pt_idx, total_pts,
                             phase, progress, n_collected, n_needed):
    """
    Draw one calibration frame onto canvas (in place).

    phase : "fixate" | "collect" | "done"
    """
    canvas[:] = 0  # clear to black

    if phase == "fixate":
        # Shrinking countdown ring
        ring_r = int(45 * (1.0 - progress))
        cv2.circle(canvas, (tx, ty), ring_r, (60, 60, 200), 3)
        cv2.circle(canvas, (tx, ty), 12, (255, 255, 255), -1)
        cv2.putText(canvas, "Look at the dot",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (160, 160, 160), 2)

    elif phase == "collect":
        # Solid green dot while collecting
        cv2.circle(canvas, (tx, ty), 12, (0, 255, 80), -1)
        bar_w = int(200 * n_collected / max(n_needed, 1))
        cv2.rectangle(canvas, (tx - 100, ty + 25),
                      (tx + 100, ty + 40), (50, 50, 50), -1)
        cv2.rectangle(canvas, (tx - 100, ty + 25),
                      (tx - 100 + bar_w, ty + 40), (0, 200, 80), -1)

    elif phase == "done":
        cv2.circle(canvas, (tx, ty), 14, (0, 255, 0), -1)
        cv2.putText(canvas, "OK", (tx - 12, ty + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Point counter (top left)
    cv2.putText(canvas, f"Point {pt_idx + 1} / {total_pts}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (120, 120, 120), 2)
    cv2.putText(canvas, "Press ESC to abort",
                (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (80, 80, 80), 1)


# =========================
# Main Calibration Function
# =========================

def run_calibration(
    gaze_estimator,
    process_frame_fn,
    cap,
    eye_size=224,
    samples_per_point=30,
    fixation_wait_sec=1.5,
    min_confidence=0.5,
):
    """
    Run a 9-point calibration session.

    Parameters
    ----------
    gaze_estimator   : GazeEstimator instance (module2)
    process_frame_fn : function — module1.eye_processing.process_frame
    cap              : cv2.VideoCapture (already opened)
    eye_size         : int — input size for gaze model
    samples_per_point: int — gaze samples to collect per dot
    fixation_wait_sec: float — seconds to wait before collecting
    min_confidence   : float — discard samples below this confidence

    Returns
    -------
    list of dicts:
        [{ screen_x, screen_y, gaze_yaw, gaze_pitch,
           confidence_mean, sample_count }, ...]
    or None if aborted.
    """

    screen_w, screen_h = _get_screen_size()
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    win = "Calibration — 9-Point"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    calibration_data = []
    aborted = False

    # --- Intro screen ---
    intro = canvas.copy()
    cv2.putText(intro, "Gaze Calibration",
                (screen_w // 2 - 200, screen_h // 2 - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    cv2.putText(intro, "9 dots will appear. Look at each dot when it appears.",
                (screen_w // 2 - 420, screen_h // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (170, 170, 170), 2)
    cv2.putText(intro, "Press SPACE to start  |  ESC to quit",
                (screen_w // 2 - 300, screen_h // 2 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 1)
    cv2.imshow(win, intro)

    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == 32:   # SPACE
            break
        if k == 27:   # ESC
            aborted = True
            break

    if aborted:
        cv2.destroyWindow(win)
        return None

    # --- Per-point loop ---
    for pt_idx, (nx, ny) in enumerate(CALIBRATION_POINTS_NORM):

        tx = int(nx * screen_w)
        ty = int(ny * screen_h)

        # ---- Phase 1: Fixation countdown ----
        fixation_start = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - fixation_start
            if elapsed >= fixation_wait_sec:
                break

            progress = elapsed / fixation_wait_sec
            _draw_calibration_frame(canvas, tx, ty, pt_idx,
                                    len(CALIBRATION_POINTS_NORM),
                                    "fixate", progress, 0, samples_per_point)
            cv2.imshow(win, canvas)

            k = cv2.waitKey(16) & 0xFF
            if k == 27:
                aborted = True
                break

        if aborted:
            break

        # ---- Phase 2: Sample collection ----
        yaw_buf, pitch_buf, conf_buf = [], [], []
        collection_start = time.perf_counter()
        timeout = samples_per_point / 10.0 + 5.0   # generous timeout

        # Warm up: discard first few frames (camera buffer may be stale)
        for _ in range(5):
            cap.read()

        while len(yaw_buf) < samples_per_point:

            # Timeout guard
            if time.perf_counter() - collection_start > timeout:
                print(f"[Calibration] Point {pt_idx+1}: timeout "
                      f"({len(yaw_buf)} samples). Moving on.")
                break

            ret, frame = cap.read()
            if not ret:
                print(f"[Calibration] WARNING: cap.read() failed")
                continue

            # process_frame handles mirroring internally
            _, left_eye, right_eye = process_frame_fn(frame, eye_size)

            yaw_l, pitch_l, conf_l = gaze_estimator.predict_gaze(left_eye)
            yaw_r, pitch_r, conf_r = gaze_estimator.predict_gaze(right_eye)

            if yaw_l is None and yaw_r is None:
                if len(yaw_buf) == 0:
                    print(f"[Calibration] No face detected — check camera/lighting")
                continue

            # Average available eyes
            yaws_valid    = [y for y in [yaw_l,   yaw_r]   if y is not None]
            pitches_valid = [p for p in [pitch_l, pitch_r] if p is not None]
            confs_valid   = [c for c in [conf_l,  conf_r]  if c is not None]

            yaw   = float(sum(yaws_valid) / len(yaws_valid))
            pitch = float(sum(pitches_valid) / len(pitches_valid))
            conf  = float(sum(confs_valid) / len(confs_valid))

            # Debug first frame per point
            if len(yaw_buf) == 0:
                print(f"[Calibration] Point {pt_idx+1} first sample: "
                      f"yaw={yaw:.2f}  pitch={pitch:.2f}  conf={conf:.3f}  "
                      f"(threshold={min_confidence})")

            if conf >= min_confidence:
                yaw_buf.append(yaw)
                pitch_buf.append(pitch)
                conf_buf.append(conf)

            # Update display
            _draw_calibration_frame(canvas, tx, ty, pt_idx,
                                    len(CALIBRATION_POINTS_NORM),
                                    "collect", 1.0,
                                    len(yaw_buf), samples_per_point)
            cv2.imshow(win, canvas)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                aborted = True
                break

        if aborted:
            break

        # ---- Require minimum samples ----
        if len(yaw_buf) < max(10, samples_per_point // 3):
            print(f"[Calibration] Point {pt_idx+1}: only {len(yaw_buf)} "
                  f"samples — skipping (check lighting/confidence).")
            continue

        # ---- Store result ----
        mean_yaw   = float(np.mean(yaw_buf))
        mean_pitch = float(np.mean(pitch_buf))
        mean_conf  = float(np.mean(conf_buf))

        calibration_data.append({
            "screen_x":        tx,
            "screen_y":        ty,
            "gaze_yaw":        mean_yaw,
            "gaze_pitch":      mean_pitch,
            "confidence_mean": mean_conf,
            "sample_count":    len(yaw_buf),
        })

        print(f"[Calibration] Point {pt_idx+1}: "
              f"screen=({tx},{ty})  "
              f"yaw={mean_yaw:.2f}  pitch={mean_pitch:.2f}  "
              f"conf={mean_conf:.3f}  n={len(yaw_buf)}")

        # Brief "done" flash
        _draw_calibration_frame(canvas, tx, ty, pt_idx,
                                 len(CALIBRATION_POINTS_NORM),
                                 "done", 1.0, len(yaw_buf), samples_per_point)
        cv2.imshow(win, canvas)
        cv2.waitKey(400)

    cv2.destroyWindow(win)

    if aborted or len(calibration_data) < 6:
        print("[Calibration] Aborted or too few valid points.")
        return None

    return calibration_data


# =========================
# Save / Load Helpers
# =========================

def save_calibration(data, path="data/calibration.json"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Calibration] Saved {len(data)} points to {path}")


def load_calibration(path="data/calibration.json"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No calibration file at {path}.")
    with open(path) as f:
        data = json.load(f)
    print(f"[Calibration] Loaded {len(data)} points from {path}")
    return data
