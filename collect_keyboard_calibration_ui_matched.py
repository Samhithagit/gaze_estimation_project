"""Collect gaze calibration data using the SAME keyboard UI as the live keyboard.

Place this file in the gaze_project root, alongside module1/, module2/, module3/,
extra_trees_gaze_mapper.py, and models/L2CSNet_mpiigaze.pkl.

Example:
    python collect_keyboard_calibration_ui_matched.py --fullscreen --output calibration_keyboard.csv
    python gaze_keyboard_live_ui_matched.py --fullscreen --calibration calibration_keyboard.csv

IMPORTANT - for the calibration to transfer to the live keyboard, run both scripts with:
  * the same window mode (both --fullscreen, or both without it: both are then maximized),
  * the same suggestion-slot count (--suggestion-slots here == --n-suggestions there),
  * the same monitor / camera position.

The widget stack (text box -> status label -> canvas), paddings, fonts, and the key /
suggestion-strip geometry are kept identical to the live keyboard, so every target is
recorded at the exact screen position where that key appears while typing.
"""
import argparse
import csv
import math
import os
import time
import tkinter as tk
from pathlib import Path

import cv2

from module1.eye_processing import get_face
from module2.gaze_estimation import GazeEstimator
from module3.head_pose import estimate_head_pose

try:                                   # use the live keyboard's prefix if available
    from autocomplete_1 import SUG_PREFIX
except Exception:                      # pragma: no cover
    SUG_PREFIX = "SUG"

DEFAULT_MODEL = "models/L2CSNet_mpiigaze.pkl"
ROWS = [list("QWERTYUIOP"), list("ASDFGHJKL"), list("ZXCVBNM") + ["⌫", "SPACE", "ENTER"]]
CSV_FIELDS = [
    "point_idx", "point_label", "target_x", "target_y",
    "screen_width", "screen_height",
    "raw_gaze_yaw_deg", "raw_gaze_pitch_deg", "raw_confidence",
    "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
    "sample_valid", "rejection_reason", "timestamp_seconds",
]

COL_KEY = "#374151"
COL_SUG = "#1d4ed8"
COL_TARGET = "#16a34a"


def finite_float(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def apply_window_mode(root, fullscreen):
    """Identical helper lives in the live keyboard script - keep the two in sync.
    Fullscreen, or a maximized window, so both scripts get the same canvas size."""
    if fullscreen:
        root.attributes("-fullscreen", True)
        return
    try:
        root.state("zoomed")                      # Windows / macOS
    except tk.TclError:
        try:
            root.attributes("-zoomed", True)      # most Linux window managers
        except tk.TclError:
            root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")


class KeyboardCalibrationWindow:
    def __init__(self, fullscreen=True, suggestion_slots=3):
        self.suggestion_keys = [f"{SUG_PREFIX}{i + 1}" for i in range(max(0, suggestion_slots))]
        self.rows = ROWS
        self.root = tk.Tk()
        self.root.title("Gaze Keyboard Calibration")
        self.root.configure(bg="#111827")
        apply_window_mode(self.root, fullscreen)
        self.root.bind("<Escape>", lambda _e: self.stop())
        self.stopped = False

        # Widget stack, order, fonts and paddings are identical to the live keyboard
        # (text box -> status label -> canvas). No extra spacer widgets.
        self.output = tk.Text(self.root, height=3, font=("Arial", 22), takefocus=0,
                              bg="#f3f4f6", fg="#111827")
        self.output.pack(fill="x", padx=12, pady=8)
        self.output.insert("1.0", "Look at the center of the highlighted key")
        self.status = tk.Label(self.root, text="Starting…", bg="#111827", fg="white",
                               font=("Arial", 12))
        self.status.pack(fill="x")
        self.canvas = tk.Canvas(self.root, bg="#111827", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self.rects = {}
        self.centers = {}
        self.key_items = {}
        self.key_h = 100.0
        self.suggestion_h = 0.0
        self.layout_size = None
        self.current_key = None
        self.canvas.bind("<Configure>", lambda _e: self.relayout_if_needed())

        self.root.update_idletasks()
        self.root.update()
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.relayout_if_needed(force=True)

    def stop(self):
        self.stopped = True

    # ── layout: same formulas as App._layout in the live keyboard ──────────
    def layout(self, cw, ch):
        gap = 8
        self.rects.clear()
        self.centers.clear()

        self.suggestion_h = max(52.0, min(82.0, ch * 0.12)) if self.suggestion_keys else 0.0
        sug_gap = 8.0
        if self.suggestion_keys:
            sug_w = (cw - 16 - (len(self.suggestion_keys) - 1) * sug_gap) / len(self.suggestion_keys)
            for i, key in enumerate(self.suggestion_keys):
                x1 = 8 + i * (sug_w + sug_gap)
                x2 = x1 + sug_w
                y1, y2 = 4.0, self.suggestion_h - 4.0
                self.rects[key] = (x1, y1, x2, y2)
                self.centers[key] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        keyboard_top = self.suggestion_h + 4.0 if self.suggestion_keys else 8.0
        keyboard_h = max(1.0, ch - keyboard_top - 8.0)
        rh = keyboard_h / len(self.rows)
        kh = rh - gap
        self.key_h = kh
        for ri, row in enumerate(self.rows):
            factors = [2.0 if k == "SPACE" else
                       1.25 if k in ("ENTER", "⌫") else
                       1.0 for k in row]
            unit = min((cw - 16) / sum(factors), 260)
            x = (cw - sum(factors) * unit) / 2
            y = keyboard_top + ri * rh
            for k, f in zip(row, factors):
                w = unit * f - gap
                self.rects[k] = (x, y, x + w, y + kh)
                self.centers[k] = (x + w / 2.0, y + kh / 2.0)
                x += unit * f

    def draw_keys(self):
        self.canvas.delete("all")
        self.key_items = {}
        fs = max(14, min(34, int(self.key_h * 0.25)))
        suggestion_fs = max(12, min(22, int(self.suggestion_h * 0.28)))
        for key, (x1, y1, x2, y2) in self.rects.items():
            is_sug = key in self.suggestion_keys
            rid = self.canvas.create_rectangle(
                x1, y1, x2, y2, fill=COL_SUG if is_sug else COL_KEY,
                outline="#9ca3af", width=2)
            if is_sug:
                label = f"word {key[len(SUG_PREFIX):]}"
                font = ("Arial", suggestion_fs, "bold")
            else:
                label = key
                font = ("Arial", fs, "bold")
            self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=label, fill="white",
                                    font=font, width=(x2 - x1 - 16) if is_sug else 0)
            self.key_items[key] = rid
        self._paint_target()

    def relayout_if_needed(self, force=False):
        """Re-layout whenever the canvas size changes (fullscreen/maximize settling)."""
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if force or (cw, ch) != self.layout_size:
            self.layout(cw, ch)
            self.draw_keys()
            self.layout_size = (cw, ch)

    def _paint_target(self):
        for k, rid in self.key_items.items():
            default = COL_SUG if k in self.suggestion_keys else COL_KEY
            self.canvas.itemconfigure(rid, fill=COL_TARGET if k == self.current_key else default)

    def highlight(self, key, idx, total, status):
        self.relayout_if_needed()
        self.current_key = key
        self._paint_target()
        self.output.delete("1.0", "end")
        self.output.insert("1.0", f"Look at the CENTER of: {key}     (ESC to stop)")
        self.status.config(text=f"Target {idx + 1}/{total}  |  {status}")
        self.root.update_idletasks()
        self.root.update()

    def target_screen_xy(self, key):
        self.root.update_idletasks()
        self.relayout_if_needed()
        cx, cy = self.centers[key]
        return int(self.canvas.winfo_rootx() + cx), int(self.canvas.winfo_rooty() + cy)

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Collect gaze calibration on the live keyboard layout")
    parser.add_argument("--output", default="calibration4_keyboard.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--face-size", type=int, default=448)
    parser.add_argument("--samples-per-key", type=int, default=50)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--fullscreen", action="store_true",
                        help="Run fullscreen (recommended). Use the same flag on the live keyboard.")
    parser.add_argument("--suggestion-slots", type=int, default=3,
                        help="Suggestion slots above the keyboard; must match the live "
                             "keyboard's --n-suggestions. 0 = original 3-row layout.")
    args = parser.parse_args()
    if args.samples_per_key < 1:
        parser.error("--samples-per-key must be >= 1")
    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Gaze model not found: {model_path}")

    estimator = GazeEstimator(model_path, device="cpu")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera ID {args.camera}")

    gui = None
    try:
        gui = KeyboardCalibrationWindow(fullscreen=args.fullscreen,
                                        suggestion_slots=args.suggestion_slots)
        # Let the window finish going fullscreen/maximized before recording targets.
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.8:
            gui.root.update()
            time.sleep(0.01)
        gui.relayout_if_needed()
        screen_w, screen_h = gui.screen_w, gui.screen_h
        keys = gui.suggestion_keys + [key for row in gui.rows for key in row]
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        accepted_total = 0
        invalid_total = 0
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for idx, key in enumerate(keys):
                if gui.stopped:
                    break
                gui.highlight(key, idx, len(keys), f"settling {args.settle_seconds:.1f}s")
                settle_start = time.perf_counter()
                while time.perf_counter() - settle_start < args.settle_seconds:
                    gui.root.update()
                    if gui.stopped:
                        break
                    cap.read()
                    time.sleep(0.005)
                if gui.stopped:
                    break

                # Read the target AFTER settling so it reflects the final layout.
                target_x, target_y = gui.target_screen_xy(key)

                accepted = 0
                attempts = 0
                while accepted < args.samples_per_key and not gui.stopped:
                    gui.root.update()
                    ret, frame = cap.read()
                    attempts += 1
                    row = {
                        "point_idx": idx,
                        "point_label": f"key_{key}",
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
                        "sample_valid": 0,
                        "rejection_reason": "",
                        "timestamp_seconds": round(time.perf_counter() - start, 6),
                    }
                    if not ret or frame is None:
                        row["rejection_reason"] = "camera_read_failure"
                        invalid_total += 1
                        writer.writerow(row)
                        f.flush()
                        continue

                    # Match the live keyboard's frame preprocessing exactly.
                    frame = cv2.flip(frame, 1)
                    try:
                        pose = estimate_head_pose(frame.copy())
                        face_result = get_face(frame.copy(), args.face_size)
                        face = face_result[1] if isinstance(face_result, tuple) and len(face_result) >= 2 else face_result
                        if pose is None or face is None:
                            raise ValueError("face_or_head_pose_missing")
                        yaw, pitch, conf = estimator.predict_gaze(face)
                        gy, gp, cf = finite_float(yaw), finite_float(pitch), finite_float(conf)
                        hy = finite_float(pose.get("yaw"))
                        hp = finite_float(pose.get("pitch"))
                        hr = finite_float(pose.get("roll"))
                        if any(v is None for v in (gy, gp, cf, hy, hp, hr)):
                            raise ValueError("invalid_gaze_or_head_pose")
                        row.update({
                            "raw_gaze_yaw_deg": gy,
                            "raw_gaze_pitch_deg": gp,
                            "raw_confidence": cf,
                            "head_yaw_deg": hy,
                            "head_pitch_deg": hp,
                            "head_roll_deg": hr,
                            "sample_valid": 1,
                            "rejection_reason": "accepted",
                        })
                        accepted += 1
                        accepted_total += 1
                        gui.highlight(key, idx, len(keys), f"collecting {accepted}/{args.samples_per_key}")
                    except Exception as exc:
                        row["rejection_reason"] = str(exc)[:120]
                        invalid_total += 1
                    writer.writerow(row)
                    f.flush()

                print(f"{key}: {accepted}/{args.samples_per_key} valid samples ({attempts} attempts)")

        print("\nCalibration collection finished")
        print(f"CSV: {out_path}")
        print(f"Accepted samples: {accepted_total}; rejected attempts: {invalid_total}")
        print("If stopped early with ESC, use the CSV only if all key targets were collected.")
    finally:
        cap.release()
        if gui is not None:
            gui.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()