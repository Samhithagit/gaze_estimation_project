"""
Compare three Innovation-Clipped Holt candidates across all sessions.

Run from the project root:
    python test_holt_three_candidates.py

This script:
- Changes only the InnovationClippedHoltGaze constructor arguments in
  final_evaluation.py while each run is active.
- Runs final_evaluation_multi_session_formatted.py once per candidate.
- Copies CSV outputs and console logs into separate candidate folders.
- Restores the original final_evaluation.py in a finally block.
- Does NOT edit module3/gaze_kalman.py, Fixed KF, or Method 2.

Back up your project / commit your current changes before running.
"""

from pathlib import Path
import csv
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
EVALUATOR = ROOT / "final_evaluation.py"
RUNNER = ROOT / "final_evaluation_multi_session_formatted.py"
OUTPUT_ROOT = ROOT / "evaluation_outputs"
RESULT_ROOT = OUTPUT_ROOT / "holt_three_candidate_comparison"

CANDIDATES = [
    {
        "name": "A_smoothness_priority",
        "alpha": 0.08,
        "beta": 0.0025,
        "innovation_limit": 8.0,
        "description": "Very smooth; check carefully for suppressed movement.",
    },
    {
        "name": "B_mae_priority",
        "alpha": 0.16,
        "beta": 0.01,
        "innovation_limit": 28.0,
        "description": "Lowest MAE in the previous grid; less smoothing.",
    },
    {
        "name": "C_compromise",
        "alpha": 0.12,
        "beta": 0.005,
        "innovation_limit": 12.0,
        "description": "Compromise between MAE, jitter, and movement.",
    },
]

def patch_constructor(source, candidate):
    # Locate the specific constructor call and replace only its arguments.
    pattern = re.compile(
        r"(robust_holt\s*=\s*InnovationClippedHoltGaze\s*\()"
        r".*?"
        r"(\))",
        re.DOTALL,
    )
    replacement = (
        r"\1\n"
        f"        alpha={candidate['alpha']},\n"
        f"        beta={candidate['beta']},\n"
        f"        innovation_limit={candidate['innovation_limit']},\n"
        "        max_trend_step=22.0,\n"
        r"    \2"
    )
    updated, count = pattern.subn(replacement, source, count=1)
    if count != 1:
        raise RuntimeError(
            "Could not find exactly one 'robust_holt = "
            "InnovationClippedHoltGaze(...)' block in final_evaluation.py. "
            "No evaluation was started. Check the constructor block manually."
        )
    return updated

def snapshot_csv_outputs(candidate_dir, log_path):
    # Preserve all CSVs produced by the runner, including per-session results.
    csv_files = [
        p for p in OUTPUT_ROOT.rglob("*.csv")
        if RESULT_ROOT not in p.parents
    ]
    copied = 0
    for src in csv_files:
        try:
            rel = src.relative_to(OUTPUT_ROOT)
        except ValueError:
            continue
        # Avoid copying previous candidate snapshots back into a new snapshot.
        if rel.parts and rel.parts[0] == RESULT_ROOT.name:
            continue
        dst = candidate_dir / "csv_outputs" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied

def main():
    if not EVALUATOR.is_file():
        raise FileNotFoundError(f"Missing evaluator: {EVALUATOR}")
    if not RUNNER.is_file():
        raise FileNotFoundError(f"Missing multi-session runner: {RUNNER}")

    original_text = EVALUATOR.read_text(encoding="utf-8")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = RESULT_ROOT / "candidate_runs_summary.txt"
    summary_lines = [
        "Three-candidate Innovation-Clipped Holt comparison",
        f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "Fixed KF and Method 2 are not intentionally modified.",
        "",
    ]

    try:
        for candidate in CANDIDATES:
            name = candidate["name"]
            candidate_dir = RESULT_ROOT / name
            candidate_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 78)
            print(f"Running {name}")
            print(
                f"alpha={candidate['alpha']}, beta={candidate['beta']}, "
                f"innovation_limit={candidate['innovation_limit']}, "
                "max_trend_step=22.0"
            )
            print(candidate["description"])
            print("=" * 78)

            patched = patch_constructor(original_text, candidate)
            EVALUATOR.write_text(patched, encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(RUNNER)],
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log_path = candidate_dir / "run.log"
            log_path.write_text(completed.stdout or "", encoding="utf-8")

            copied = snapshot_csv_outputs(candidate_dir, log_path)
            summary_lines.extend([
                f"Candidate: {name}",
                f"alpha={candidate['alpha']}, beta={candidate['beta']}, "
                f"innovation_limit={candidate['innovation_limit']}",
                f"Return code: {completed.returncode}",
                f"CSV files copied: {copied}",
                f"Log: {log_path.relative_to(ROOT)}",
                "",
            ])

            print(completed.stdout or "(No console output)")
            print(f"Saved log: {log_path}")
            print(f"Copied {copied} CSV output(s) to {candidate_dir}")

            if completed.returncode != 0:
                print(
                    f"WARNING: {name} failed. See {log_path} for details. "
                    "The script will continue to the next candidate."
                )

    finally:
        # Always restore the user's exact original evaluator file.
        EVALUATOR.write_text(original_text, encoding="utf-8")
        print("\nRestored the original final_evaluation.py.")

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nComparison outputs: {RESULT_ROOT}")
    print(f"Run summary: {summary_path}")

if __name__ == "__main__":
    main()
