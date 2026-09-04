#!/usr/bin/env python3
# main.py
# =============================================================
# Live gaze-controlled cursor.
# Requires calibrate.py to have been run first.
#
# Usage:
#   .venv/bin/python main.py
#
# Controls:
#   ESC — quit
#   R   — reset cursor EMA state
# =============================================================

import cv2
import sys
import toml
import time

from module1.eye_processing  import process_frame
from module2.gaze_estimation import GazeEstimator
from module4.screen_mapper   import ScreenMapper
from module4.cursor_controller import CursorController

import pyautogui
SCREEN_W, SCREEN_H = (int(x) for x in pyautogui.size())


# =========================
# Load Config
# =========================

config = toml.load("config.toml")

MODEL_PATH       = config["model"]["weights"]
DEVICE           = config["model"]["device"]
CAMERA_ID        = config["camera"]["camera_id"]
EYE_SIZE         = config["input"]["eye_size"]
SHOW_EYE_WINDOWS = config["display"]["show_eye_windows"]
SHOW_YAW_PITCH   = config["display"]["show_yaw_pitch"]
SMOOTHING_ALPHA  = config["cursor"]["smoothing_alpha"]
DEAD_ZONE_PX     = config["cursor"]["dead_zone_px"]
USE_KALMAN       = config["cursor"]["use_kalman"]

# =========================
# Initialise
# =========================

print("[Main] Loading gaze model...")
gaze_estimator = GazeEstimator(MODEL_PATH, DEVICE)

print("[Main] Loading screen mapper...")
try:
    mapper = ScreenMapper.load("data/screen_mapper.pkl")
except FileNotFoundError as e:
    print(f"[Main] ERROR: {e}")
    print("       Run calibrate.py first.")
    sys.exit(1)

print("[Main] Opening camera...")
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("[Main] ERROR: Cannot open camera.")
    sys.exit(1)

cursor = CursorController(
    screen_w       = SCREEN_W,
    screen_h       = SCREEN_H,
    smoothing_alpha = SMOOTHING_ALPHA,
    dead_zone_px   = DEAD_ZONE_PX,
)

# =========================
# FPS Control
# =========================

TARGET_FPS     = 30
FRAME_TIME_SEC = 1.0 / TARGET_FPS
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

print(f"[Main] Running. Screen={SCREEN_W}×{SCREEN_H}. "
      f"Kalman={'ON (Person 3)' if USE_KALMAN else 'OFF (EMA)'}. "
      f"Press ESC to quit.")

# =========================
# Main Loop
# =========================

while True:

    loop_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    # ---- Module 1: eye extraction ----
    # (process_frame handles mirroring internally)
    frame, left_eye, right_eye = process_frame(frame, EYE_SIZE)

    # ---- Module 2: gaze estimation ----
    yaw_l, pitch_l, conf_l = gaze_estimator.predict_gaze(left_eye)
    yaw_r, pitch_r, conf_r = gaze_estimator.predict_gaze(right_eye)

    # ---- Average available eyes ----
    yaws_v   = [y for y in [yaw_l,   yaw_r]   if y is not None]
    pitches_v = [p for p in [pitch_l, pitch_r] if p is not None]
    confs_v  = [c for c in [conf_l,  conf_r]  if c is not None]

    if yaws_v:
        yaw        = sum(yaws_v) / len(yaws_v)
        pitch      = sum(pitches_v) / len(pitches_v)
        confidence = sum(confs_v) / len(confs_v)

        # ---- Module 3: map to screen ----
        raw_x, raw_y = mapper.predict(yaw, pitch)

        # ---- Module 4a: move cursor ----
        if USE_KALMAN:
            # --------------------------------------------------
            # Person 3 integration point:
            # Replace raw_x/raw_y with Person 3's filtered output.
            #
            # Example (once Person 3 delivers their module):
            #   from interfaces.filter_data import FilteredPosition
            #   fp = person3_filter.update(raw_x, raw_y, confidence)
            #   cursor_x, cursor_y = cursor.move_to_filtered(fp)
            # --------------------------------------------------
            cursor_x, cursor_y = cursor.move_to_raw(raw_x, raw_y)
        else:
            cursor_x, cursor_y = cursor.move_to_raw(raw_x, raw_y)

        # ---- Overlay on webcam frame ----
        if SHOW_YAW_PITCH:
            cv2.putText(frame, f"Yaw:   {yaw:.2f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf:  {confidence:.3f}",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 200, 255), 2)
            cv2.putText(frame, f"Cursor: ({cursor_x:.0f}, {cursor_y:.0f})",
                        (20, 145), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 200, 0), 2)

    # ---- Eye windows ----
    if SHOW_EYE_WINDOWS:
        if left_eye is not None:
            cv2.imshow("Left Eye", left_eye)
        if right_eye is not None:
            cv2.imshow("Right Eye", right_eye)

    cv2.imshow("Integrated Gaze System", frame)

    # ---- FPS limiter ----
    elapsed      = time.perf_counter() - loop_start
    remaining_ms = max(1, int((FRAME_TIME_SEC - elapsed) * 1000))

    key = cv2.waitKey(remaining_ms) & 0xFF
    if key == 27:     # ESC
        break
    elif key == ord('r') or key == ord('R'):
        cursor.reset()
        print("[Main] Cursor EMA state reset.")

# =========================
# Cleanup
# =========================

cap.release()
cv2.destroyAllWindows()
print("[Main] Exited cleanly.")