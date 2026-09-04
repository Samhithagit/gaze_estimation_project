# module4/evaluator.py
# =============================================================
# Evaluation framework for gaze-to-screen accuracy.
#
# Protocol:
#   1. Show N target dots on screen (different from calib points).
#   2. Per target: collect gaze samples → map → record error.
#   3. Write data/evaluation_results.csv.
#   4. Print summary stats.
# =============================================================

import cv2
import csv
import math
import time
import numpy as np
from pathlib import Path


# =========================
# Test Target Grid (normalised)
# Offset from calibration grid to avoid overfitting bias
# =========================

EVAL_POINTS_NORM = [
    (0.25, 0.25), (0.75, 0.25),
    (0.50, 0.50),
    (0.25, 0.75), (0.75, 0.75),
    (0.15, 0.50), (0.85, 0.50),
    (0.50, 0.15), (0.50, 0.85),
]


# =========================
# Degree-error conversion
# =========================

def _px_to_deg(error_px, screen_w_px, screen_w_cm, viewing_dist_cm):
    """Convert pixel error to degrees of visual angle."""
    px_per_cm = screen_w_px / screen_w_cm
    error_cm  = error_px / px_per_cm
    angle_rad = math.atan2(error_cm, viewing_dist_cm)
    return math.degrees(angle_rad)


# =========================
# Evaluator
# =========================

class Evaluator:
    """
    Run an accuracy evaluation experiment.

    Parameters
    ----------
    screen_mapper       : ScreenMapper instance (fitted)
    gaze_estimator      : GazeEstimator instance
    process_frame_fn    : module1.eye_processing.process_frame
    cap                 : cv2.VideoCapture (already opened)
    screen_w, screen_h  : int — screen dimensions in pixels
    eye_size            : int — model input size
    samples_per_target  : int — gaze samples per dot
    fixation_wait_sec   : float — pre-collection wait
    min_confidence      : float — sample filter threshold
    viewing_distance_cm : float — participant distance from screen
    screen_width_cm     : float — physical screen width
    output_path         : str — CSV output path
    """

    def __init__(
        self,
        screen_mapper,
        gaze_estimator,
        process_frame_fn,
        cap,
        screen_w,
        screen_h,
        eye_size=224,
        samples_per_target=30,
        fixation_wait_sec=2.0,
        min_confidence=0.5,
        viewing_distance_cm=60.0,
        screen_width_cm=30.4,
        output_path="data/evaluation_results.csv",
    ):
        self.mapper            = screen_mapper
        self.estimator         = gaze_estimator
        self.process_frame_fn  = process_frame_fn
        self.cap               = cap
        self.screen_w          = screen_w
        self.screen_h          = screen_h
        self.eye_size          = eye_size
        self.samples_per_target = samples_per_target
        self.fixation_wait_sec = fixation_wait_sec
        self.min_confidence    = min_confidence
        self.viewing_dist_cm   = viewing_distance_cm
        self.screen_width_cm   = screen_width_cm
        self.output_path       = Path(output_path)

    # =========================
    # Run
    # =========================

    def run(self, num_targets=9):
        """
        Run evaluation and return results list.

        Returns
        -------
        list of dicts, or None if aborted.
        """

        pts = EVAL_POINTS_NORM[:num_targets]
        canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

        win = "Evaluation — Gaze Accuracy"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # --- Intro screen ---
        intro = canvas.copy()
        cv2.putText(intro, "Gaze Accuracy Evaluation",
                    (self.screen_w // 2 - 280, self.screen_h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(intro, f"{num_targets} targets will appear. Look at each dot.",
                    (self.screen_w // 2 - 380, self.screen_h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (170, 170, 170), 2)
        cv2.putText(intro, "Press SPACE to start  |  ESC to quit",
                    (self.screen_w // 2 - 280, self.screen_h // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 1)
        cv2.imshow(win, intro)

        aborted = False
        while True:
            k = cv2.waitKey(30) & 0xFF
            if k == 32:
                break
            if k == 27:
                aborted = True
                break

        if aborted:
            cv2.destroyWindow(win)
            return None

        results = []

        for trial_idx, (nx, ny) in enumerate(pts):

            tx = int(nx * self.screen_w)
            ty = int(ny * self.screen_h)

            # ---- Fixation phase ----
            fix_start = time.perf_counter()
            while True:
                elapsed  = time.perf_counter() - fix_start
                if elapsed >= self.fixation_wait_sec:
                    break

                progress = elapsed / self.fixation_wait_sec
                canvas[:] = 0
                ring_r = int(50 * (1.0 - progress))
                cv2.circle(canvas, (tx, ty), ring_r, (200, 80, 0), 3)
                cv2.circle(canvas, (tx, ty), 12, (255, 100, 0), -1)
                cv2.putText(canvas, f"Target {trial_idx+1} / {num_targets}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
                cv2.imshow(win, canvas)

                k = cv2.waitKey(16) & 0xFF
                if k == 27:
                    aborted = True
                    break

            if aborted:
                break

            # ---- Collection phase ----
            yaw_buf, pitch_buf, conf_buf = [], [], []
            coll_start = time.perf_counter()
            timeout = self.samples_per_target / 10.0 + 3.0

            while len(yaw_buf) < self.samples_per_target:

                if time.perf_counter() - coll_start > timeout:
                    print(f"[Eval] Trial {trial_idx+1}: timeout "
                          f"({len(yaw_buf)} samples).")
                    break

                ret, frame = self.cap.read()
                if not ret:
                    continue

                # process_frame handles mirroring internally
                _, left_eye, right_eye = self.process_frame_fn(
                    frame, self.eye_size
                )

                yaw_l, pitch_l, conf_l = self.estimator.predict_gaze(left_eye)
                yaw_r, pitch_r, conf_r = self.estimator.predict_gaze(right_eye)

                yaws_v  = [y for y in [yaw_l, yaw_r]     if y is not None]
                pitch_v = [p for p in [pitch_l, pitch_r] if p is not None]
                conf_v  = [c for c in [conf_l, conf_r]   if c is not None]

                if not yaws_v:
                    continue

                yaw   = float(np.mean(yaws_v))
                pitch = float(np.mean(pitch_v))
                conf  = float(np.mean(conf_v))

                if conf >= self.min_confidence:
                    yaw_buf.append(yaw)
                    pitch_buf.append(pitch)
                    conf_buf.append(conf)

                # Display collecting
                canvas[:] = 0
                cv2.circle(canvas, (tx, ty), 12, (255, 80, 0), -1)
                bar_w = int(200 * len(yaw_buf) / max(self.samples_per_target, 1))
                cv2.rectangle(canvas, (tx - 100, ty + 25),
                              (tx + 100, ty + 40), (50, 50, 50), -1)
                cv2.rectangle(canvas, (tx - 100, ty + 25),
                              (tx - 100 + bar_w, ty + 40), (0, 160, 255), -1)
                cv2.imshow(win, canvas)

                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    aborted = True
                    break

            if aborted:
                break

            if not yaw_buf:
                print(f"[Eval] Trial {trial_idx+1}: no samples — skipping.")
                continue

            # ---- Map and compute error ----
            mean_yaw   = float(np.mean(yaw_buf))
            mean_pitch = float(np.mean(pitch_buf))
            mean_conf  = float(np.mean(conf_buf))

            pred_x, pred_y = self.mapper.predict(mean_yaw, mean_pitch)

            error_px = float(np.hypot(pred_x - tx, pred_y - ty))
            error_deg = _px_to_deg(
                error_px, self.screen_w,
                self.screen_width_cm, self.viewing_dist_cm
            )

            result = {
                "trial":           trial_idx + 1,
                "target_x":        tx,
                "target_y":        ty,
                "pred_x":          round(pred_x, 2),
                "pred_y":          round(pred_y, 2),
                "gaze_yaw":        round(mean_yaw, 4),
                "gaze_pitch":      round(mean_pitch, 4),
                "confidence_mean": round(mean_conf, 4),
                "error_px":        round(error_px, 2),
                "error_deg":       round(error_deg, 4),
                "sample_count":    len(yaw_buf),
            }
            results.append(result)

            print(
                f"[Eval] Trial {trial_idx+1}: "
                f"target=({tx},{ty})  pred=({pred_x:.0f},{pred_y:.0f})  "
                f"error={error_px:.1f}px / {error_deg:.2f}°"
            )

            # Brief confirmation flash
            canvas[:] = 0
            cv2.circle(canvas, (tx, ty), 14, (0, 255, 0), -1)
            cv2.circle(canvas, (int(pred_x), int(pred_y)), 8, (0, 0, 255), -1)
            cv2.line(canvas, (tx, ty), (int(pred_x), int(pred_y)),
                     (0, 200, 255), 1)
            cv2.putText(canvas, f"Error: {error_px:.0f}px",
                        (tx + 20, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 1)
            cv2.imshow(win, canvas)
            cv2.waitKey(600)

        cv2.destroyWindow(win)

        if not results:
            return None

        self._save_csv(results)
        self._print_summary(results)
        return results

    # =========================
    # CSV Output
    # =========================

    def _save_csv(self, results):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "trial", "target_x", "target_y",
            "pred_x", "pred_y",
            "gaze_yaw", "gaze_pitch", "confidence_mean",
            "error_px", "error_deg", "sample_count",
        ]
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[Eval] Results saved to {self.output_path}")

    # =========================
    # Summary Stats
    # =========================

    def _print_summary(self, results):
        errors_px  = [r["error_px"]  for r in results]
        errors_deg = [r["error_deg"] for r in results]

        mean_px  = np.mean(errors_px)
        std_px   = np.std(errors_px)
        max_px   = np.max(errors_px)
        mean_deg = np.mean(errors_deg)

        worst = max(results, key=lambda r: r["error_px"])

        print("\n" + "=" * 50)
        print("  GAZE ACCURACY EVALUATION — SUMMARY")
        print("=" * 50)
        print(f"  Trials evaluated : {len(results)}")
        print(f"  Mean error       : {mean_px:.1f} px  ({mean_deg:.2f}°)")
        print(f"  Std dev          : {std_px:.1f} px")
        print(f"  Max error        : {max_px:.1f} px  "
              f"(Trial {worst['trial']})")
        print("=" * 50)
        print("  Per-trial breakdown:")
        for r in results:
            print(f"    Trial {r['trial']:2d}  "
                  f"target=({r['target_x']:4d},{r['target_y']:4d})  "
                  f"error={r['error_px']:6.1f}px / {r['error_deg']:.2f}°")
        print("=" * 50 + "\n")
