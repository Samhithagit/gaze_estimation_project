#!/usr/bin/env python3
# evaluate.py
# =============================================================
# Run AFTER calibration to measure gaze-to-screen accuracy.
#
# Usage:
#   .venv/bin/python evaluate.py
#
# Output:
#   data/evaluation_results.csv  — per-trial error data
#   Terminal                     — summary stats (mean, std, worst)
# =============================================================

import cv2
import sys
import toml

from module1.eye_processing import process_frame
from module2.gaze_estimation import GazeEstimator
from module4.screen_mapper   import ScreenMapper
from module4.evaluator       import Evaluator

import pyautogui
SCREEN_W, SCREEN_H = (int(x) for x in pyautogui.size())


# =========================
# Load Config
# =========================

config = toml.load("config.toml")

MODEL_PATH   = config["model"]["weights"]
DEVICE       = config["model"]["device"]
CAMERA_ID    = config["camera"]["camera_id"]
EYE_SIZE     = config["input"]["eye_size"]

SAMPLES_PER_TARGET  = config["evaluation"]["samples_per_target"]
FIXATION_WAIT_SEC   = config["calibration"]["fixation_wait_sec"]
MIN_CONFIDENCE      = config["calibration"]["min_confidence"]
NUM_TARGETS         = config["evaluation"]["num_targets"]
VIEWING_DIST_CM     = config["evaluation"]["viewing_distance_cm"]
SCREEN_WIDTH_CM     = config["evaluation"]["screen_width_cm"]

# =========================
# Initialise
# =========================

print("[Evaluate] Loading gaze model...")
gaze_estimator = GazeEstimator(MODEL_PATH, DEVICE)

print("[Evaluate] Loading screen mapper...")
try:
    mapper = ScreenMapper.load("data/screen_mapper.pkl")
except FileNotFoundError as e:
    print(f"[Evaluate] ERROR: {e}")
    print("           Run calibrate.py first.")
    sys.exit(1)

print("[Evaluate] Opening camera...")
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("[Evaluate] ERROR: Cannot open camera.")
    sys.exit(1)

# =========================
# Run Evaluation
# =========================

evaluator = Evaluator(
    screen_mapper      = mapper,
    gaze_estimator     = gaze_estimator,
    process_frame_fn   = process_frame,
    cap                = cap,
    screen_w           = SCREEN_W,
    screen_h           = SCREEN_H,
    eye_size           = EYE_SIZE,
    samples_per_target = SAMPLES_PER_TARGET,
    fixation_wait_sec  = FIXATION_WAIT_SEC,
    min_confidence     = MIN_CONFIDENCE,
    viewing_distance_cm = VIEWING_DIST_CM,
    screen_width_cm    = SCREEN_WIDTH_CM,
    output_path        = "data/evaluation_results.csv",
)

results = evaluator.run(num_targets=NUM_TARGETS)

cap.release()

if results is None:
    print("[Evaluate] Evaluation aborted.")
    sys.exit(1)

print("[Evaluate] ✓ Evaluation complete.")
