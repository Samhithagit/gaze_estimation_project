"""
Phase 0 diagnostic — WEBCAM version.

Runs the exact capture -> face crop -> gaze estimate path your
live system uses, so Phase 0 validates the real pipeline, not a
photo shot under different conditions.

Controls:
  SPACE  - capture current frame and log yaw/pitch/confidence
  ESC    - quit and print summary stats

Protocol:
  1. Sit in your normal seating position, look directly at the
     webcam (not the screen -- the camera, since that's the
     0,0 reference).
  2. Press SPACE ~15-20 times across a few seconds, keeping your
     gaze on the camera lens each time (small natural movement is
     fine and expected).
  3. Press ESC. The script prints mean/std of yaw, pitch, and
     confidence, and saves everything to phase0_log.csv.

What to look for (gate to move to Phase 1):
  - mean |yaw| and mean |pitch| should be within a few degrees
    of 0 (some fixed offset is fine if your camera isn't dead
    center on the screen -- note that offset, you'll subtract it
    later as a calibration bias).
  - mean confidence should be > 0.6.
  - std of yaw/pitch across captures (while trying to hold still)
    tells you the estimator's own noise floor -- this is the
    number the Kalman filter in Phase 2 needs to beat.
"""

import cv2
import csv
import time
import numpy as np

from module1.eye_processing import get_face
from module2.gaze_estimation import GazeEstimator

MODEL_PATH = r"D:\majorproject\gaze_project\models\L2CSNet_gaze360.pkl"
CAMERA_ID = 0
LOG_PATH = "phase0_log.csv"


def main():
    estimator = GazeEstimator(MODEL_PATH, device="cpu")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    records = []

    print("Look at the camera. Press SPACE to log a reading, ESC to finish.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        display_frame, face_crop = get_face(frame, face_size=448)

        status = "Face OK" if face_crop is not None else "No face detected"
        color = (0, 255, 0) if face_crop is not None else (0, 0, 255)
        cv2.putText(display_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_frame, f"Captures: {len(records)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display_frame, "SPACE=capture  ESC=finish", (20, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Phase 0 - Estimator Diagnostic", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key == 32:  # SPACE
            if face_crop is None:
                print("  -> no face detected, not logged")
                continue

            yaw, pitch, confidence = estimator.predict_gaze(face_crop)
            if yaw is None:
                print("  -> prediction failed, not logged")
                continue

            ts = time.time()
            records.append((ts, yaw, pitch, confidence))
            print(f"[{len(records):02d}] yaw={yaw:6.2f} "
                  f"pitch={pitch:6.2f} conf={confidence:.4f}")

    cap.release()
    cv2.destroyAllWindows()

    if not records:
        print("No captures logged.")
        return

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "yaw", "pitch", "confidence"])
        writer.writerows(records)

    yaws = np.array([r[1] for r in records])
    pitches = np.array([r[2] for r in records])
    confs = np.array([r[3] for r in records])

    print("\n===== Phase 0 Summary =====")
    print(f"N captures        : {len(records)}")
    print(f"Yaw   mean/std     : {yaws.mean():.2f} / {yaws.std():.2f} deg")
    print(f"Pitch mean/std     : {pitches.mean():.2f} / {pitches.std():.2f} deg")
    print(f"Confidence mean/std: {confs.mean():.4f} / {confs.std():.4f}")
    print(f"Logged to {LOG_PATH}")

    print("\nGate check:")
    print(f"  |mean yaw| < 10deg?    {'PASS' if abs(yaws.mean()) < 10 else 'FAIL'}")
    print(f"  |mean pitch| < 10deg?  {'PASS' if abs(pitches.mean()) < 10 else 'FAIL'}")
    print(f"  mean confidence > 0.6? {'PASS' if confs.mean() > 0.6 else 'FAIL'}")


if __name__ == "__main__":
    main()