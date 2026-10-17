"""
ONE-SCRIPT joint grid search for InnovationClippedHoltGaze.

Run from the project root:
    python tune_holt_all_parameters.py

Jointly searches alpha, beta, and innovation_limit. It keeps max_trend_step fixed
because the current InnovationClippedHoltGaze.update() implementation does not
actually use that parameter. It evaluates the actual evaluator-generated outputs
for each combination and saves a comparison CSV.

SAFETY:
- Backs up final_evaluation.py in memory and restores it in finally.
- Only patches the robust_holt constructor in final_evaluation.py.
- Does not edit module3/gaze_kalman.py.
- Does not intentionally modify Fixed KF or Method 2 code.
- Make a git commit or backup before running.
"""

from pathlib import Path
import itertools
import math
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EVALUATOR = ROOT / "final_evaluation.py"
RUNNER = ROOT / "final_evaluation_multi_session_formatted.py"
PREDICTIONS = ROOT / "evaluation_outputs" / "predictions_all_sessions.csv"
OUT_DIR = ROOT / "evaluation_outputs" / "holt_joint_tuning"

# Joint grid: 3 alpha values x 3 beta values x 6 limits = 54 full evaluations.
ALPHA_VALUES = [0.08, 0.12, 0.16]
BETA_VALUES = [0.0025, 0.005, 0.01]
LIMIT_VALUES = [8, 10, 12, 16, 20, 28]
MAX_TREND_STEP = 22.0

# Selection guardrail:
# candidates must have MAE no more than 1% worse than Method 2 baseline (ridge)
# and then get ranked by a weighted combination of MAE and jitter.
# Set to None to disable the MAE guardrail.
MAX_MAE_INCREASE_VS_METHOD2 = 0.01

# Keep per-combination logs; full evaluation output can be large.
KEEP_LOGS = True


def patch_constructor(source, alpha, beta, limit):
    pattern = re.compile(
        r"(robust_holt\s*=\s*InnovationClippedHoltGaze\s*\()"
        r"(?P<body>.*?)"
        r"(\))",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        raise RuntimeError(
            "Could not find robust_holt = InnovationClippedHoltGaze(...) "
            "in final_evaluation.py. Nothing was evaluated."
        )

    body = match.group("body")
    args = {
        "alpha": f"alpha={alpha}",
        "beta": f"beta={beta}",
        "innovation_limit": f"innovation_limit={float(limit)}",
        "max_trend_step": f"max_trend_step={MAX_TREND_STEP}",
    }
    for name, replacement in args.items():
        expr = rf"{name}\s*=\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        body, count = re.subn(expr, replacement, body, count=1)
        if count != 1:
            raise RuntimeError(
                f"Could not patch argument {name!r} in robust_holt constructor."
            )

    block = match.group(1) + body + match.group(3)
    return source[:match.start()] + block + source[match.end():]


def metrics_for_predictions(path):
    df = pd.read_csv(path)
    required = [
        "ridge_x", "ridge_y", "robust_holt_x", "robust_holt_y",
        "target_x", "target_y", "point_idx",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} missing columns: {missing}. Available: {list(df.columns)}"
        )

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "participant_id" not in df:
        df["participant_id"] = "ALL"
    if "session_id" not in df:
        df["session_id"] = "ALL"

    groups = ["participant_id", "session_id", "point_idx"]
    sort_cols = groups + (["frame_idx"] if "frame_idx" in df.columns else [])
    df = df.sort_values(sort_cols, kind="stable")

    method2_err, holt_err, holt_sq = [], [], []
    jitter, step = [], []
    for _, part in df.groupby(groups, sort=False):
        target = part[["target_x", "target_y"]].to_numpy(float)
        ridge = part[["ridge_x", "ridge_y"]].to_numpy(float)
        pred = part[["robust_holt_x", "robust_holt_y"]].to_numpy(float)

        valid = (
            np.all(np.isfinite(target), axis=1)
            & np.all(np.isfinite(ridge), axis=1)
            & np.all(np.isfinite(pred), axis=1)
        )
        target, ridge, pred = target[valid], ridge[valid], pred[valid]
        if len(pred) == 0:
            continue

        method2_err.extend(np.linalg.norm(ridge - target, axis=1).tolist())
        e = np.linalg.norm(pred - target, axis=1)
        holt_err.extend(e.tolist())
        holt_sq.extend((e ** 2).tolist())

        if len(pred) >= 2:
            step.extend(np.linalg.norm(np.diff(pred, axis=0), axis=1).tolist())
        if len(pred) >= 3:
            jitter.extend(np.linalg.norm(np.diff(pred, n=2, axis=0), axis=1).tolist())

    if not holt_err:
        raise ValueError("No valid prediction rows available for metrics.")

    return {
        "rows": len(holt_err),
        "Method2_MAE_px": float(np.mean(method2_err)),
        "MAE_px": float(np.mean(holt_err)),
        "RMSE_px": float(math.sqrt(np.mean(holt_sq))),
        "Jitter_second_difference_px": float(np.mean(jitter)) if jitter else 0.0,
        "Mean_step_px": float(np.mean(step)) if step else 0.0,
    }


def main():
    if not EVALUATOR.is_file():
        raise FileNotFoundError(f"Missing {EVALUATOR}")
    if not RUNNER.is_file():
        raise FileNotFoundError(f"Missing {RUNNER}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    original = EVALUATOR.read_text(encoding="utf-8")
    combos = list(itertools.product(ALPHA_VALUES, BETA_VALUES, LIMIT_VALUES))
    rows = []
    start = time.time()

    print(f"Joint parameter combinations: {len(combos)}")
    print(f"alpha: {ALPHA_VALUES}")
    print(f"beta: {BETA_VALUES}")
    print(f"innovation_limit: {LIMIT_VALUES}")
    print("Each combination reruns the complete multi-session evaluator.")
    print("Fixed KF and Method 2 source code are not modified.")

    try:
        for index, (alpha, beta, limit) in enumerate(combos, 1):
            print(
                f"\n[{index}/{len(combos)}] "
                f"alpha={alpha}, beta={beta}, innovation_limit={limit}"
            )
            EVALUATOR.write_text(
                patch_constructor(original, alpha, beta, limit),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [sys.executable, str(RUNNER)],
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log_path = OUT_DIR / f"run_a{alpha}_b{beta}_lim{limit}.log"
            if KEEP_LOGS:
                log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
            print(proc.stdout[-1200:])

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Evaluation failed for alpha={alpha}, beta={beta}, limit={limit}. "
                    f"Inspect {log_path}. Original evaluator will be restored."
                )
            if not PREDICTIONS.is_file():
                raise FileNotFoundError(f"Missing output predictions: {PREDICTIONS}")

            saved_csv = OUT_DIR / f"predictions_a{alpha}_b{beta}_lim{limit}.csv"
            shutil.copy2(PREDICTIONS, saved_csv)
            metrics = metrics_for_predictions(saved_csv)
            row = {
                "alpha": alpha,
                "beta": beta,
                "innovation_limit": limit,
                "max_trend_step": MAX_TREND_STEP,
                **metrics,
                "predictions_file": saved_csv.name,
                "log_file": log_path.name if KEEP_LOGS else "",
            }
            rows.append(row)
            print(
                f"MAE={row['MAE_px']:.4f}, RMSE={row['RMSE_px']:.4f}, "
                f"jitter={row['Jitter_second_difference_px']:.4f}, "
                f"mean_step={row['Mean_step_px']:.4f}"
            )

            # Save partial results continuously, so a later failure doesn't lose work.
            pd.DataFrame(rows).to_csv(OUT_DIR / "joint_tuning_results_partial.csv", index=False)

    finally:
        EVALUATOR.write_text(original, encoding="utf-8")
        print("\nRestored the original final_evaluation.py.")

    result = pd.DataFrame(rows)
    method2_mae = float(result["Method2_MAE_px"].median())
    mae_ceiling = method2_mae * (1.0 + MAX_MAE_INCREASE_VS_METHOD2)
    result["eligible"] = result["MAE_px"] <= mae_ceiling

    # Rank eligible candidates using normalized MAE and jitter. Jitter is given
    # 60% weight, MAE 40%, but the MAE guardrail prevents choosing a very inaccurate
    # near-stationary filter.
    mae_base = max(float(result["Method2_MAE_px"].median()), 1e-9)
    jitter_base = max(float(result["Jitter_second_difference_px"].median()), 1e-9)
    result["score"] = (
        0.40 * result["MAE_px"] / mae_base
        + 0.60 * result["Jitter_second_difference_px"] / jitter_base
    )
    eligible = result[result["eligible"]]
    if eligible.empty:
        print(
            "\nWARNING: No combination met the MAE guardrail versus Method 2. "
            "Ranking all combinations by score, but inspect accuracy carefully."
        )
        eligible = result
    ranked = eligible.sort_values(["score", "MAE_px", "Jitter_second_difference_px"])
    ranked.to_csv(OUT_DIR / "joint_tuning_results.csv", index=False)

    best = ranked.iloc[0].to_dict()
    best_path = OUT_DIR / "best_holt_parameters.txt"
    best_path.write_text(
        "Best candidate according to this script's selection rule\n"
        f"alpha = {best['alpha']}\n"
        f"beta = {best['beta']}\n"
        f"innovation_limit = {best['innovation_limit']}\n"
        f"max_trend_step = {MAX_TREND_STEP}\n"
        f"Method 2 MAE = {best['Method2_MAE_px']:.6f}\n"
        f"Holt MAE = {best['MAE_px']:.6f}\n"
        f"Holt RMSE = {best['RMSE_px']:.6f}\n"
        f"Holt second-difference jitter = {best['Jitter_second_difference_px']:.6f}\n"
        f"Holt mean step = {best['Mean_step_px']:.6f}\n"
        f"Selection score = {best['score']:.6f}\n",
        encoding="utf-8",
    )

    print("\nTOP 10 CANDIDATES:")
    print(
        ranked[
            ["alpha", "beta", "innovation_limit", "Method2_MAE_px", "MAE_px",
             "RMSE_px", "Jitter_second_difference_px", "Mean_step_px", "score"]
        ].head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print(f"\nBest parameters saved to: {best_path}")
    print(f"Full comparison saved to: {OUT_DIR / 'joint_tuning_results.csv'}")
    print(f"Elapsed: {(time.time() - start) / 60:.1f} minutes")
    print(
        "\nIMPORTANT: These are tuning results, not an unbiased final score. "
        "After selecting a candidate, evaluate it once on an untouched participant/session."
    )


if __name__ == "__main__":
    main()
