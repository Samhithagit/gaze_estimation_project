#!/usr/bin/env python3
# calibrate.py
# =============================================================
# Run this ONCE per session before using the cursor.
#
# Usage:
#   .venv/bin/python calibrate.py
#
# Output:
#   data/calibration.json   — raw gaze-screen pairs
#   data/screen_mapper.pkl  — fitted polynomial mapping model
# =============================================================

import cv2
import sys
import toml

from module1.eye_processing import process_frame
from module2.gaze_estimation import GazeEstimator
from module4.calibration     import run_calibration, save_calibration
from module4.screen_mapper   import ScreenMapper

# =========================
# Load Config
# =========================

config = toml.load("config.toml")

MODEL_PATH   = config["model"]["weights"]
DEVICE       = config["model"]["device"]
CAMERA_ID    = config["camera"]["camera_id"]
EYE_SIZE     = config["input"]["eye_size"]

SAMPLES_PER_POINT  = config["calibration"]["samples_per_point"]
FIXATION_WAIT_SEC  = config["calibration"]["fixation_wait_sec"]
MIN_CONFIDENCE     = config["calibration"]["min_confidence"]

# =========================
# Initialise
# =========================

print("[Calibrate] Loading gaze model...")
gaze_estimator = GazeEstimator(MODEL_PATH, DEVICE)

print("[Calibrate] Opening camera...")
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("[Calibrate] ERROR: Cannot open camera.")
    sys.exit(1)

# =========================
# Run Calibration
# =========================

print("[Calibrate] Starting 9-point calibration UI...")
calibration_data = run_calibration(
    gaze_estimator   = gaze_estimator,
    process_frame_fn = process_frame,
    cap              = cap,
    eye_size         = EYE_SIZE,
    samples_per_point = SAMPLES_PER_POINT,
    fixation_wait_sec = FIXATION_WAIT_SEC,
    min_confidence    = MIN_CONFIDENCE,
)

cap.release()

if calibration_data is None:
    print("[Calibrate] Calibration aborted or failed.")
    sys.exit(1)

# =========================
# Save Raw Data
# =========================

save_calibration(calibration_data, "data/calibration.json")

# =========================
# Fit Screen Mapper
# =========================

print("[Calibrate] Fitting polynomial screen mapper...")
mapper = ScreenMapper()
result = mapper.fit(calibration_data)

print(
    f"[Calibrate] Fit complete — "
    f"R²_x={result['r2_x']:.3f}  "
    f"R²_y={result['r2_y']:.3f}  "
    f"Quality: {result['quality']}"
)

if result["quality"] == "POOR — please recalibrate":
    print("[Calibrate] WARNING: Poor fit. Consider re-running calibration.")
    print("            Tips: better lighting, sit closer, look carefully at each dot.")

mapper.save("data/screen_mapper.pkl")

print("\n[Calibrate] ✓ Done! You can now run:")
print("    .venv/bin/python main.py      ← live cursor")
print("    .venv/bin/python evaluate.py  ← accuracy test\n")
