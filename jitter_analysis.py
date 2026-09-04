import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CSV FILES
# ============================================================

files = {
    "Baseline": r"D:\majorproject\gaze_project\gaze_pipeline_results_baseline.csv",
    "Method 2": r"D:\majorproject\gaze_project\gaze_pipeline_results_fusion.csv",
    "Proposed": r"D:\majorproject\gaze_project\gaze_pipeline_results.csv"
}


# ============================================================
# COLUMNS FOR EACH METHOD
# ============================================================

columns = {
    "Baseline": {
        "yaw": "raw_gaze_yaw",
        "pitch": "raw_gaze_pitch"
    },

    "Method 2": {
        "yaw": "fused_yaw",
        "pitch": "fused_pitch"
    },

    "Proposed": {
        "yaw": "kalman_yaw",
        "pitch": "kalman_pitch"
    }
}


# ============================================================
# FUNCTION TO CALCULATE JITTER
# ============================================================

def calculate_jitter(df, yaw_col, pitch_col):

    yaw = df[yaw_col].dropna().to_numpy()
    pitch = df[pitch_col].dropna().to_numpy()

    # Make sure both have same length
    n = min(len(yaw), len(pitch))

    yaw = yaw[:n]
    pitch = pitch[:n]

    # Frame-to-frame changes
    delta_yaw = np.diff(yaw)
    delta_pitch = np.diff(pitch)

    # Angular displacement between consecutive frames
    angular_change = np.sqrt(
        delta_yaw**2 + delta_pitch**2
    )

    # --------------------------------------------------------
    # Jitter metrics
    # --------------------------------------------------------

    mean_jitter = np.mean(angular_change)

    std_jitter = np.std(angular_change)

    rms_jitter = np.sqrt(
        np.mean(angular_change**2)
    )

    max_jitter = np.max(angular_change)

    # Total amount of frame-to-frame movement
    total_variation = np.sum(
        np.abs(delta_yaw)
    ) + np.sum(
        np.abs(delta_pitch)
    )

    return {
        "Mean Jitter (°)": mean_jitter,
        "Jitter Std (°)": std_jitter,
        "RMS Jitter (°)": rms_jitter,
        "Maximum Jitter (°)": max_jitter,
        "Total Variation (°)": total_variation
    }


# ============================================================
# CALCULATE RESULTS
# ============================================================

results = []

for method, filename in files.items():

    print(f"\nProcessing {method}: {filename}")

    df = pd.read_csv(filename)

    yaw_col = columns[method]["yaw"]
    pitch_col = columns[method]["pitch"]

    # Check required columns
    if yaw_col not in df.columns:
        raise ValueError(
            f"{yaw_col} not found in {filename}"
        )

    if pitch_col not in df.columns:
        raise ValueError(
            f"{pitch_col} not found in {filename}"
        )

    metrics = calculate_jitter(
        df,
        yaw_col,
        pitch_col
    )

    metrics["Method"] = method

    results.append(metrics)


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

# Put Method column first
results_df = results_df[
    [
        "Method",
        "Mean Jitter (°)",
        "Jitter Std (°)",
        "RMS Jitter (°)",
        "Maximum Jitter (°)",
        "Total Variation (°)"
    ]
]


print("\n")
print("=" * 80)
print("GAZE JITTER COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# CALCULATE IMPROVEMENT
# ============================================================

baseline_mean = results_df.loc[
    results_df["Method"] == "Baseline",
    "Mean Jitter (°)"
].iloc[0]

method2_mean = results_df.loc[
    results_df["Method"] == "Method 2",
    "Mean Jitter (°)"
].iloc[0]

proposed_mean = results_df.loc[
    results_df["Method"] == "Proposed",
    "Mean Jitter (°)"
].iloc[0]


method2_improvement = (
    (baseline_mean - method2_mean)
    / baseline_mean
) * 100


proposed_improvement = (
    (baseline_mean - proposed_mean)
    / baseline_mean
) * 100


print("\n")
print("=" * 80)
print("JITTER REDUCTION")
print("=" * 80)

print(
    f"Method 2 improvement over Baseline : "
    f"{method2_improvement:.2f}%"
)

print(
    f"Proposed improvement over Baseline : "
    f"{proposed_improvement:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "jitter_comparison_results.csv",
    index=False
)


# ============================================================
# BAR GRAPH - MEAN JITTER
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Method"],
    results_df["Mean Jitter (°)"]
)

plt.xlabel("Method")
plt.ylabel("Mean Frame-to-Frame Angular Change (°)")
plt.title("Comparison of Gaze Jitter")

plt.tight_layout()

plt.savefig(
    "gaze_jitter_comparison.png",
    dpi=300
)

plt.show()