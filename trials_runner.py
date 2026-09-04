import subprocess
import sys
import os
import pandas as pd
import numpy as np


# ============================================================
# NUMBER OF TRIALS
# ============================================================

if len(sys.argv) >= 2:

    try:

        NUM_TRIALS = int(
            sys.argv[1]
        )

    except ValueError:

        print(
            "ERROR: Number of trials must be an integer."
        )

        sys.exit(1)

else:

    NUM_TRIALS = 5


if NUM_TRIALS <= 0:

    print(
        "ERROR: Number of trials must be greater than 0."
    )

    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

PHASE2_SCRIPT = "phase2_calibration.py"

FINAL_SCRIPT = "final_evaluation.py"

TRIAL_ROOT = "trials"


# ============================================================
# HEADER
# ============================================================

print()

print("=" * 90)

print(
    "MULTI-TRIAL GAZE EVALUATION"
)

print("=" * 90)

print()

print(
    f"Number of trials: {NUM_TRIALS}"
)

print()

print(
    "Each trial evaluates:"
)

print(
    "1. Baseline"
)

print(
    "2. Head-Eye Fusion"
)

print(
    "3. Fixed Kalman"
)

print(
    "4. Adaptive Q"
)

print(
    "5. Adaptive R"
)

print(
    "6. Proposed (Adaptive Q+R)"
)

print()


# ============================================================
# CREATE ROOT
# ============================================================

os.makedirs(
    TRIAL_ROOT,
    exist_ok=True
)


# ============================================================
# RUN TRIALS
# ============================================================

for trial_number in range(
    1,
    NUM_TRIALS + 1
):


    trial_name = (
        f"trial_{trial_number:02d}"
    )


    trial_directory = os.path.join(

        TRIAL_ROOT,

        trial_name

    )


    print()

    print("=" * 90)

    print(
        f"STARTING {trial_name.upper()}"
    )

    print("=" * 90)

    print()


    # --------------------------------------------------------
    # Ask user to prepare
    # --------------------------------------------------------

    input(

        f"Prepare for {trial_name}. "
        f"Press ENTER to start..."

    )


    # --------------------------------------------------------
    # Run recording
    # --------------------------------------------------------

    result = subprocess.run([

        sys.executable,

        PHASE2_SCRIPT,

        trial_name

    ])


    if result.returncode != 0:

        print()

        print(
            f"ERROR during {trial_name}."
        )

        print(
            "Stopping experiment."
        )

        sys.exit(1)


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    csv_file = os.path.join(

        trial_directory,

        "unified_evaluation.csv"

    )


    result = subprocess.run([

        sys.executable,

        FINAL_SCRIPT,

        csv_file

    ])


    if result.returncode != 0:

        print()

        print(
            f"ERROR evaluating {trial_name}."
        )

        sys.exit(1)


# ============================================================
# LOAD ALL TRIAL RESULTS
# ============================================================

print()

print("=" * 90)

print(
    "COMBINING TRIAL RESULTS"
)

print("=" * 90)

print()


all_results = []


for trial_number in range(

    1,

    NUM_TRIALS + 1

):


    trial_name = (

        f"trial_{trial_number:02d}"

    )


    summary_file = os.path.join(

        TRIAL_ROOT,

        trial_name,

        "final_evaluation_summary.csv"

    )


    if not os.path.exists(
        summary_file
    ):

        print(

            f"WARNING: Missing results for "
            f"{trial_name}"

        )

        continue


    trial_df = pd.read_csv(
        summary_file
    )


    trial_df[
        "Trial"
    ] = trial_number


    all_results.append(
        trial_df
    )


if len(all_results) == 0:

    print(
        "ERROR: No trial results found."
    )

    sys.exit(1)


combined_df = pd.concat(

    all_results,

    ignore_index=True

)


# ============================================================
# SAVE RAW COMBINED RESULTS
# ============================================================

combined_file = os.path.join(

    TRIAL_ROOT,

    "all_trial_results.csv"

)


combined_df.to_csv(

    combined_file,

    index=False

)


# ============================================================
# METRICS
# ============================================================

metrics = [

    "MAE_px",

    "RMSE_px",

    "Mean_Jitter_px",

    "Jitter_Std_px",

    "Maximum_Jitter_px",

    "Total_Variation_px"

]


# ============================================================
# FINAL STATISTICS
# ============================================================

summary_rows = []


for method in combined_df[
    "Method"
].unique():


    method_data = combined_df[
        combined_df["Method"] == method
    ]


    row = {

        "Method": method,

        "Trials": len(method_data)

    }


    for metric in metrics:

        values = method_data[
            metric
        ].dropna().values


        if len(values) == 0:

            row[
                metric + "_Mean"
            ] = np.nan

            row[
                metric + "_SD"
            ] = np.nan

        else:

            row[
                metric + "_Mean"
            ] = float(
                np.mean(values)
            )


            row[
                metric + "_SD"
            ] = float(
                np.std(
                    values,
                    ddof=1
                )
            ) if len(values) > 1 else 0.0


    summary_rows.append(
        row
    )


final_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# FINAL SUMMARY FILE
# ============================================================

final_file = os.path.join(

    TRIAL_ROOT,

    "FINAL_MULTI_TRIAL_SUMMARY.csv"

)


final_df.to_csv(

    final_file,

    index=False

)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()

print("=" * 120)

print(
    "FINAL MULTI-TRIAL RESULTS"
)

print("=" * 120)

print()

print(
    f"{'Method':<20}"
    f"{'MAE':>20}"
    f"{'RMSE':>20}"
    f"{'Jitter':>20}"
    f"{'Total Variation':>22}"
)

print(
    "-" * 120
)


# Fixed, deliberate print order (rather than combined_df's
# arbitrary unique() order) so the table always reads
# Baseline -> Fusion -> Fixed KF -> Adaptive Q -> Adaptive R
# -> Proposed, matching the paper's ablation narrative.

METHOD_ORDER = [

    "Baseline",

    "Method 2",

    "Fixed KF",

    "Adaptive Q",

    "Adaptive R",

    "Proposed"

]


for method in METHOD_ORDER:

    matching_rows = final_df[
        final_df["Method"] == method
    ]

    if len(matching_rows) == 0:
        continue

    row = matching_rows.iloc[0]

    mae = (

        f"{row['MAE_px_Mean']:.2f}"
        f" ± "
        f"{row['MAE_px_SD']:.2f}"

    )


    rmse = (

        f"{row['RMSE_px_Mean']:.2f}"
        f" ± "
        f"{row['RMSE_px_SD']:.2f}"

    )


    jitter = (

        f"{row['Mean_Jitter_px_Mean']:.2f}"
        f" ± "
        f"{row['Mean_Jitter_px_SD']:.2f}"

    )


    variation = (

        f"{row['Total_Variation_px_Mean']:.2f}"
        f" ± "
        f"{row['Total_Variation_px_SD']:.2f}"

    )


    print(

        f"{method:<20}"

        f"{mae:>20}"

        f"{rmse:>20}"

        f"{jitter:>20}"

        f"{variation:>22}"

    )


# ============================================================
# IMPROVEMENT USING MEAN VALUES
# ============================================================

baseline = final_df[
    final_df["Method"] == "Baseline"
]


if len(baseline) > 0:

    baseline_mae = float(
        baseline[
            "MAE_px_Mean"
        ].iloc[0]
    )

    baseline_jitter = float(
        baseline[
            "Mean_Jitter_px_Mean"
        ].iloc[0]
    )


    print()

    print("=" * 100)

    print(
        "MEAN IMPROVEMENT OVER BASELINE"
    )

    print("=" * 100)


    for method in [

        "Method 2",

        "Fixed KF",

        "Adaptive Q",

        "Adaptive R",

        "Proposed"

    ]:


        method_rows = final_df[
            final_df["Method"] == method
        ]


        if len(method_rows) == 0:

            continue


        method_mae = float(
            method_rows[
                "MAE_px_Mean"
            ].iloc[0]
        )


        method_jitter = float(
            method_rows[
                "Mean_Jitter_px_Mean"
            ].iloc[0]
        )


        mae_reduction = (

            (
                baseline_mae
                -
                method_mae
            )

            /

            baseline_mae

        ) * 100


        jitter_reduction = (

            (
                baseline_jitter
                -
                method_jitter
            )

            /

            baseline_jitter

        ) * 100


        print()

        print(
            method
        )

        print(

            f"  MAE reduction    : "
            f"{mae_reduction:.2f}%"

        )

        print(

            f"  Jitter reduction : "
            f"{jitter_reduction:.2f}%"

        )


# ============================================================
# DONE
# ============================================================

print()

print("=" * 100)

print(
    "ALL TRIALS COMPLETED"
)

print("=" * 100)

print()

print(
    f"Raw trial results:"
)

print(
    combined_file
)

print()

print(
    f"Final mean ± SD:"
)

print(
    final_file
)

print()