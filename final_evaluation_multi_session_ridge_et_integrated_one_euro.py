from pathlib import Path
import argparse
import csv
import math
import shutil
import subprocess
import sys

import numpy as np

# Run from the project root, beside the dataset/ directory.
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
ONLY_PARTICIPANT = None  # Evaluate ALL participants; set to e.g. "P002" to filter
OUTPUT_DIR = ROOT / "evaluation_outputs"
NORMALIZED_DIR = OUTPUT_DIR / "normalized_inputs"

EVALUATOR_CANDIDATES = [
    ROOT / "final_evaluation_ridge_et_integrated_one_euro.py",
]

COLUMN_ALIASES = {
    "raw_gaze_yaw": "raw_gaze_yaw_deg",
    "raw_gaze_pitch": "raw_gaze_pitch_deg",
    "head_yaw": "head_yaw_deg",
    "head_pitch": "head_pitch_deg",
    "head_roll": "head_roll_deg",
    "timestamp": "timestamp_seconds",
}

METHODS = {
    # Baseline is used for reductions and shown as the reference row.
    "Baseline": ("baseline_x", "baseline_y"),
    "Method 2 (Ridge)": ("ridge_x", "ridge_y"),
    "Extra Trees method": ("et_raw_x", "et_raw_y"),
    "Fixed KF + Ridge": ("fixed_kf_x", "fixed_kf_y"),
    "Fixed KF + Extra Trees": ("et_fixed_kf_x", "et_fixed_kf_y"),
    "Adaptive Q + Extra Trees": ("et_adaptive_q_x", "et_adaptive_q_y"),
    "Adaptive R + Extra Trees": ("et_adaptive_r_x", "et_adaptive_r_y"),
    "Extra Trees + Double Exp.": ("et_double_exp_x", "et_double_exp_y"),
    "Extra Trees + Clipped Holt": ("et_robust_holt_x", "et_robust_holt_y"),
    "Extra Trees + One Euro": ("one_euro_x", "one_euro_y"),
}
REPORT_METHODS = [
    ("Baseline", "Baseline"),
    ("Method 2 (Ridge)", "Method 2 (Ridge)"),
    ("Extra Trees method", "Extra Trees method"),
    ("Fixed KF + Ridge", "Fixed KF + Ridge"),
    ("Fixed KF + Extra Trees", "Fixed KF + Extra Trees"),
    ("Adaptive Q + Extra Trees", "Adaptive Q + Extra Trees"),
    ("Adaptive R + Extra Trees", "Adaptive R + Extra Trees"),
    ("Extra Trees + Double Exp.", "Extra Trees + Double Exp."),
    ("Extra Trees + Clipped Holt", "Extra Trees + Clipped Holt"),
    ("Extra Trees + One Euro", "Extra Trees + One Euro"),
]
REQUIRED_COLUMNS = [
    "raw_gaze_yaw", "raw_gaze_pitch", "raw_confidence",
    "head_yaw", "head_pitch", "target_x", "target_y", "point_idx",
]


def find_evaluator():
    for path in EVALUATOR_CANDIDATES:
        if path.is_file():
            return path
    names = ", ".join(p.name for p in EVALUATOR_CANDIDATES)
    raise FileNotFoundError(
        f"Could not find evaluator. Expected one of: {names}. "
        "Put this wrapper beside your evaluator script."
    )


def discover_sessions():
    """Discover participant/session folders containing both required CSVs."""
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    found = []
    for participant_dir in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        if ONLY_PARTICIPANT is not None and participant_dir.name != ONLY_PARTICIPANT:
            continue
        for session_dir in sorted(p for p in participant_dir.iterdir() if p.is_dir()):
            if (
                (session_dir / "calibration_data.csv").is_file()
                and (session_dir / "evaluation_data.csv").is_file()
            ):
                found.append((participant_dir.name, session_dir.name, session_dir))
    if not found:
        target = ONLY_PARTICIPANT or "any participant"
        raise FileNotFoundError(
            f"No complete sessions found for {target} under {DATASET_DIR}. "
            "Each session must contain calibration_data.csv and evaluation_data.csv."
        )
    return found


def load_and_normalize(src, dst):
    if not src.is_file():
        raise FileNotFoundError(f"Missing dataset file: {src}")
    with src.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {src}")
        fields = list(reader.fieldnames)
        rows = list(reader)

    for expected, original in COLUMN_ALIASES.items():
        if expected not in fields and original in fields:
            fields.append(expected)
            for row in rows:
                row[expected] = row.get(original, "")

    missing = [name for name in REQUIRED_COLUMNS if name not in fields]
    if missing:
        raise ValueError(f"{src} missing required columns: {missing}")

    valid_rows, invalid_count = [], 0
    for row in rows:
        flag = str(row.get("sample_valid", "1")).strip().lower()
        valid_flag = flag not in {"0", "false", "no", ""}
        finite = True
        for col in REQUIRED_COLUMNS:
            try:
                if not math.isfinite(float(row.get(col, ""))):
                    finite = False
                    break
            except (TypeError, ValueError):
                finite = False
                break
        if valid_flag and finite:
            valid_rows.append(row)
        else:
            invalid_count += 1

    if not valid_rows:
        raise ValueError(f"No valid rows remain in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid_rows)
    return len(rows), len(valid_rows), invalid_count


def metrics(rows, x_col, y_col):
    errors, jitter = [], []
    total_variation = 0.0
    groups = {}
    for row in rows:
        try:
            px, py = float(row[x_col]), float(row[y_col])
            tx, ty = float(row["target_x"]), float(row["target_y"])
            if not all(map(math.isfinite, (px, py, tx, ty))):
                continue
            errors.append(math.hypot(px - tx, py - ty))
            group_key = (str(row.get("participant_id", "")),
                         str(row.get("session_id", "")), str(row["point_idx"]))
            groups.setdefault(group_key, []).append((px, py))
        except (KeyError, TypeError, ValueError):
            continue

    for points in groups.values():
        if len(points) > 1:
            movements = [
                math.hypot(points[i][0] - points[i - 1][0],
                           points[i][1] - points[i - 1][1])
                for i in range(1, len(points))
            ]
            jitter.extend(movements)
            total_variation += sum(movements)

    if not errors:
        return {
            "MAE_px": math.nan, "RMSE_px": math.nan, "Median_px": math.nan,
            "P95_px": math.nan, "Jitter_px_per_frame": math.nan,
            "Total_variation_px": math.nan,
        }
    errors = np.asarray(errors, dtype=float)
    return {
        "MAE_px": float(np.mean(errors)),
        "RMSE_px": float(np.sqrt(np.mean(errors ** 2))),
        "Median_px": float(np.median(errors)),
        "P95_px": float(np.percentile(errors, 95)),
        "Jitter_px_per_frame": float(np.mean(jitter)) if jitter else 0.0,
        "Total_variation_px": float(total_variation),
    }


def percent_reduction(baseline, method):
    try:
        baseline, method = float(baseline), float(method)
        if not math.isfinite(baseline) or not math.isfinite(method) or baseline == 0:
            return None
        return (baseline - method) / baseline * 100.0
    except (TypeError, ValueError):
        return None


def fmt_pct(value):
    return "N/A" if value is None else f"{float(value):.2f}%"


def make_reduction_rows(scope, participant, rows):
    method_metrics = {}
    for method_name, (x_col, y_col) in METHODS.items():
        usable = [r for r in rows if x_col in r and y_col in r]
        method_metrics[method_name] = metrics(usable, x_col, y_col)
    base = method_metrics["Baseline"]  # reference only; not printed as a result
    out = []
    for label, method_name in REPORT_METHODS:
        current = method_metrics[method_name]
        if method_name == "Baseline":
            out.append({
                "participant_id": participant,
                "scope": scope,
                "method": label,
                "MAE_px": current["MAE_px"],
                "RMSE_px": current["RMSE_px"],
                "Jitter_px_per_frame": current["Jitter_px_per_frame"],
                "Total_variation_px": current["Total_variation_px"],
                "MAE_reduction_pct": 0.0,
                "RMSE_reduction_pct": 0.0,
                "Jitter_reduction_pct": 0.0,
                "TV_reduction_pct": 0.0,
            })
            continue
        out.append({
            "participant_id": participant,
            "scope": scope,
            "method": label,
            "MAE_px": current["MAE_px"],
            "RMSE_px": current["RMSE_px"],
            "Jitter_px_per_frame": current["Jitter_px_per_frame"],
            "Total_variation_px": current["Total_variation_px"],
            "MAE_reduction_pct": percent_reduction(base["MAE_px"], current["MAE_px"]),
            "RMSE_reduction_pct": percent_reduction(base["RMSE_px"], current["RMSE_px"]),
            "Jitter_reduction_pct": percent_reduction(
                base["Jitter_px_per_frame"], current["Jitter_px_per_frame"]),
            "TV_reduction_pct": percent_reduction(
                base["Total_variation_px"], current["Total_variation_px"]),
        })
    return out


def print_reduction_table(title, rows):
    print("\n" + "=" * 91)
    print(title)
    print("=" * 91)
    print(f"{'Method':<22} {'MAE reduction':>15} {'RMSE reduction':>16} "
          f"{'Jitter reduction':>18} {'TV reduction':>15}")
    for row in rows:
        print(f"{row['method']:<22} "
              f"{fmt_pct(row['MAE_reduction_pct']):>15} "
              f"{fmt_pct(row['RMSE_reduction_pct']):>16} "
              f"{fmt_pct(row['Jitter_reduction_pct']):>18} "
              f"{fmt_pct(row['TV_reduction_pct']):>15}")


def print_final_selected_report():
    """Print reductions from the frozen, tuned final-selected predictions."""
    predictions_dir = (
        OUTPUT_DIR / "adaptive_filter_study" / "final_selected" / "predictions"
    )
    prediction_files = sorted(
        predictions_dir.glob("P*/S*/final_evaluation_results.csv")
    )
    if not prediction_files:
        raise FileNotFoundError(
            "No final-selected predictions found under "
            f"{predictions_dir}. Run the calibration study first."
        )

    combined_predictions = []
    sessions = set()
    for prediction_path in prediction_files:
        participant, session = prediction_path.parts[-3:-1]
        sessions.add((participant, session))
        with prediction_path.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                row["participant_id"] = participant
                row["session_id"] = session
                combined_predictions.append(row)

    participants = sorted({row["participant_id"] for row in combined_predictions})
    print("Source: frozen final-selected tuned predictions")
    print(f"Sessions: {len(sessions)}; prediction rows: {len(combined_predictions)}")

    for participant in participants:
        person_rows = [
            row for row in combined_predictions
            if row["participant_id"] == participant
        ]
        print_reduction_table(
            f"PERCENTAGE REDUCTIONS — {participant} (ALL ITS SESSIONS)",
            make_reduction_rows("PERSON_ALL_SESSIONS", participant, person_rows),
        )

    pooled_rows = make_reduction_rows(
        "POOLED_ALL_PARTICIPANTS", "ALL", combined_predictions
    )
    print_reduction_table(
        "POOLED PERCENTAGE REDUCTIONS — ALL SELECTED DATA",
        pooled_rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-selected",
        action="store_true",
        help="Print the saved tuned final-selected report without rerunning evaluation.",
    )
    args = parser.parse_args()
    if args.final_selected:
        print_final_selected_report()
        return

    evaluator = find_evaluator()
    sessions = discover_sessions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    all_session_metrics, combined_predictions = [], []
    print(f"Using evaluator: {evaluator.name}")
    print(f"Dataset root: {DATASET_DIR}")
    print(f"Participant filter: {ONLY_PARTICIPANT or 'ALL'}")
    print(f"Discovered {len(sessions)} session(s):")
    for participant, session, _ in sessions:
        print(f"  {participant}/{session}")

    for participant, session, session_dir in sessions:
        output_key = f"{participant}_{session}"
        normalized_dir = NORMALIZED_DIR / participant / session
        cal_src = session_dir / "calibration_data.csv"
        eval_src = session_dir / "evaluation_data.csv"
        cal_dst = normalized_dir / "calibration_data.csv"
        eval_dst = normalized_dir / "evaluation_data.csv"

        cal_total, cal_valid, cal_invalid = load_and_normalize(cal_src, cal_dst)
        eval_total, eval_valid, eval_invalid = load_and_normalize(eval_src, eval_dst)
        print(
            f"\n{participant}/{session}: calibration {cal_valid}/{cal_total} usable; "
            f"evaluation {eval_valid}/{eval_total} usable; "
            f"invalid/skipped rows: calibration={cal_invalid}, evaluation={eval_invalid}"
        )

        result_path = normalized_dir / "final_evaluation_results.csv"
        if result_path.exists():
            result_path.unlink()
        completed = subprocess.run(
            [sys.executable, str(evaluator), str(cal_dst), str(eval_dst)],
            cwd=str(ROOT), text=True, capture_output=True
        )
        if completed.stdout:
            print(completed.stdout)
        if completed.returncode != 0:
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)
            raise RuntimeError(
                f"Evaluator failed for {participant}/{session} "
                f"(exit {completed.returncode})"
            )
        if not result_path.is_file():
            raise FileNotFoundError(
                f"Expected evaluator output was not created: {result_path}"
            )

        shutil.copy2(result_path, OUTPUT_DIR / f"{output_key}_predictions.csv")
        with result_path.open("r", newline="", encoding="utf-8-sig") as f:
            prediction_rows = list(csv.DictReader(f))
        for row in prediction_rows:
            row["participant_id"] = participant
            row["session_id"] = session
            combined_predictions.append(row)

        for method, (x_col, y_col) in METHODS.items():
            all_session_metrics.append({
                "participant_id": participant,
                "session_id": session,
                "method": method,
                **metrics(prediction_rows, x_col, y_col),
            })

    session_fields = [
        "participant_id", "session_id", "method", "MAE_px", "RMSE_px",
        "Median_px", "P95_px", "Jitter_px_per_frame", "Total_variation_px",
    ]
    session_metrics_path = OUTPUT_DIR / "metrics_by_session.csv"
    with session_metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=session_fields)
        writer.writeheader()
        writer.writerows(all_session_metrics)

    # One reduction table per participant (all their sessions pooled), then overall pooled.
    reduction_rows = []
    participants = sorted({row["participant_id"] for row in combined_predictions})
    for participant in participants:
        person_rows = [r for r in combined_predictions if r["participant_id"] == participant]
        reduction_rows.extend(make_reduction_rows("PERSON_ALL_SESSIONS", participant, person_rows))
    reduction_rows.extend(make_reduction_rows(
        "POOLED_ALL_PARTICIPANTS", "ALL", combined_predictions
    ))

    reduction_fields = [
        "participant_id", "scope", "method", "MAE_px", "RMSE_px",
        "Jitter_px_per_frame", "Total_variation_px", "MAE_reduction_pct",
        "RMSE_reduction_pct", "Jitter_reduction_pct", "TV_reduction_pct",
    ]
    reduction_path = OUTPUT_DIR / "percentage_reductions_by_person_and_pooled.csv"
    with reduction_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reduction_fields)
        writer.writeheader()
        writer.writerows(reduction_rows)

    for participant in participants:
        rows = [r for r in reduction_rows
                if r["scope"] == "PERSON_ALL_SESSIONS" and r["participant_id"] == participant]
        print_reduction_table(
            f"PERCENTAGE REDUCTIONS — {participant} (ALL ITS SESSIONS)", rows
        )
    pooled_rows = [r for r in reduction_rows if r["scope"] == "POOLED_ALL_PARTICIPANTS"]
    print_reduction_table("POOLED PERCENTAGE REDUCTIONS — ALL SELECTED DATA", pooled_rows)

    combined_path = OUTPUT_DIR / "predictions_all_sessions.csv"
    prediction_fields = list(combined_predictions[0].keys()) if combined_predictions else [
        "participant_id", "session_id"
    ]
    with combined_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=prediction_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined_predictions)

    print("\nSaved output files:")
    print(f"  Session metrics : {session_metrics_path}")
    print(f"  Reduction report: {reduction_path}")
    print(f"  Combined results: {combined_path}")


if __name__ == "__main__":
    main()
