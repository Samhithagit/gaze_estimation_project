"""Validate dataset sessions produced by collect_user_study.py.

The collector now logs every camera read attempt during active target collection.
Therefore each CSV can have more than 180 rows; validation counts only rows with
sample_valid == 1 when checking the expected 9 targets x 20 accepted samples.

Usage:
    python validate_dataset.py
    python validate_dataset.py dataset/P001/S01
"""
import csv
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(PROJECT_DIR, "dataset")
REQUIRED_COLUMNS = [
    "participant_id", "session_id", "task_name", "frame_id",
    "timestamp_seconds", "timestamp_utc", "point_idx", "point_label",
    "target_x", "target_y", "raw_gaze_yaw_deg", "raw_gaze_pitch_deg",
    "raw_confidence", "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
    "head_reprojection_error_px", "frame_read_success", "face_detected",
    "gaze_valid", "head_pose_valid", "sample_valid", "rejection_reason",
    "preview_image"
]


def check_csv(path, expected_targets=9, expected_valid_per_target=20):
    if not os.path.isfile(path):
        return False, f"MISSING: {path}"

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        rows = list(reader)

    if missing:
        return False, f"{path}: missing columns: {missing}"
    if not rows:
        return False, f"{path}: contains no frame rows"

    bad = [i + 2 for i, row in enumerate(rows)
           if not row.get("point_label") or not row.get("participant_id")]
    if bad:
        return False, f"{path}: missing labels/IDs at CSV lines {bad[:10]}"

    accepted = [row for row in rows if row.get("sample_valid", "0").strip() == "1"]
    per_target = {}
    for row in accepted:
        idx = row.get("point_idx", "")
        per_target[idx] = per_target.get(idx, 0) + 1

    if len(per_target) != expected_targets or any(
        n != expected_valid_per_target for n in per_target.values()
    ):
        return False, (
            f"{path}: expected {expected_targets} targets x {expected_valid_per_target} "
            f"valid samples; got {len(accepted)} valid samples across {len(per_target)} targets: {per_target}. "
            f"Total logged frame rows={len(rows)}"
        )

    missing_reason = [i + 2 for i, row in enumerate(rows) if not row.get("rejection_reason", "").strip()]
    if missing_reason:
        return False, f"{path}: missing rejection_reason at CSV lines {missing_reason[:10]}"

    invalid_count = len(rows) - len(accepted)
    return True, (
        f"OK: {path} ({len(rows)} logged frame rows; {len(accepted)} accepted; "
        f"{invalid_count} invalid/failed attempts; {expected_targets} targets x "
        f"{expected_valid_per_target} accepted samples)"
    )


def main():
    if not os.path.exists(ROOT):
        print(f"Dataset path does not exist: {ROOT}")
        raise SystemExit(1)

    sessions = []
    if os.path.isfile(os.path.join(ROOT, "calibration_data.csv")):
        sessions = [ROOT]
    else:
        for path, dirs, files in os.walk(ROOT):
            if "calibration_data.csv" in files or "evaluation_data.csv" in files:
                sessions.append(path)

    if not sessions:
        print(f"No session folders found under {ROOT}")
        raise SystemExit(1)

    failures = 0
    for session in sorted(sessions):
        print(f"\nSession: {session}")
        for filename in ("calibration_data.csv", "evaluation_data.csv"):
            ok, message = check_csv(os.path.join(session, filename))
            print(message)
            failures += 0 if ok else 1

        metadata_path = os.path.join(session, "metadata.json")
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                print(f"Metadata OK: participant={metadata.get('participant_id')} session={metadata.get('session_id')}")
            except Exception as exc:
                print(f"INVALID metadata.json: {exc}")
                failures += 1
        else:
            print("WARNING: metadata.json is missing")

    print("\nVALIDATION RESULT:", "PASS" if failures == 0 else f"FAIL ({failures} issue(s))")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
