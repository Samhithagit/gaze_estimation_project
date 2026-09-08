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
# FILES
# ============================================================

PHASE2_SCRIPT = "phase2_calibration.py"

FINAL_SCRIPT = "final_evaluation.py"

TRIAL_ROOT = "trials"


# ============================================================
# START
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


    input(

        f"Prepare for {trial_name}. "
        f"Press ENTER to start..."

    )


    # ========================================================
    # COLLECT CALIBRATION + EVALUATION
    # ========================================================

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

        sys.exit(1)


    # ========================================================
    # FILES
    # ========================================================

    calibration_file = os.path.join(

        trial_directory,

        "calibration_data.csv"

    )


    evaluation_file = os.path.join(

        trial_directory,

        "evaluation_data.csv"

    )


    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    result = subprocess.run([

        sys.executable,

        FINAL_SCRIPT,

        calibration_file,

        evaluation_file

    ])


    if result.returncode != 0:

        print()

        print(
            f"ERROR evaluating {trial_name}."
        )

        sys.exit(1)


# ============================================================
# COMBINE
# ============================================================

print()

print("=" * 90)

print(
    "COMBINING TRIAL RESULTS"
)

print("=" * 90)

print()


all_results = []


# ============================================================
# LOAD EACH TRIAL
# ============================================================

for trial_number in range(

    1,

    NUM_TRIALS + 1

):


    trial_name = (

        f"trial_{trial_number:02d}"

    )


    results_file = os.path.join(

        TRIAL_ROOT,

        trial_name,

        "final_evaluation_results.csv"

    )


    if not os.path.exists(
        results_file
    ):

        print(

            f"WARNING: Missing "
            f"{results_file}"

        )

        continue


    df = pd.read_csv(

        results_file

    )


    if len(df) == 0:

        continue


    print(

        f"Loaded {trial_name}: "
        f"{len(df)} samples"

    )


    # ========================================================
    # METHOD COLUMNS
    # ========================================================

    methods = {

        "Baseline": (

            "baseline_x",
            "baseline_y"

        ),

        "Method 2": (

            "ridge_x",
            "ridge_y"

        ),

        "Fixed KF": (

            "fixed_kf_x",
            "fixed_kf_y"

        ),

        "Adaptive Q": (

            "adaptive_q_x",
            "adaptive_q_y"

        ),

        "Adaptive R": (

            "adaptive_r_x",
            "adaptive_r_y"

        ),

        "Proposed": (

            "adaptive_qr_x",
            "adaptive_qr_y"

        )

    }


    # ========================================================
    # METRICS
    # ========================================================

    for method, (

        x_col,
        y_col

    ) in methods.items():


        data = df[

            [
                x_col,
                y_col,
                "target_x",
                "target_y",
                "point_idx"

            ]

        ].dropna()


        if len(data) == 0:

            continue


        # ----------------------------------------------------
        # Position error
        # ----------------------------------------------------

        dx = (

            data[x_col].values
            -
            data["target_x"].values

        )


        dy = (

            data[y_col].values
            -
            data["target_y"].values

        )


        errors = np.sqrt(

            dx ** 2
            +
            dy ** 2

        )


        mae = float(

            np.mean(errors)

        )


        rmse = float(

            np.sqrt(

                np.mean(

                    errors ** 2

                )

            )

        )


        # ----------------------------------------------------
        # Jitter
        # ----------------------------------------------------

        jitter_values = []


        for point_idx in sorted(

            data["point_idx"].unique()

        ):


            point_data = data[

                data["point_idx"]
                ==
                point_idx

            ]


            if len(point_data) < 2:

                continue


            x = point_data[
                x_col
            ].values


            y = point_data[
                y_col
            ].values


            dx_j = np.diff(x)

            dy_j = np.diff(y)


            movement = np.sqrt(

                dx_j ** 2
                +
                dy_j ** 2

            )


            jitter_values.extend(

                movement.tolist()

            )


        if len(jitter_values) > 0:

            mean_jitter = float(

                np.mean(
                    jitter_values
                )

            )

        else:

            mean_jitter = 0.0


        # ----------------------------------------------------
        # Total variation
        # ----------------------------------------------------

        total_variation = 0.0


        for point_idx in sorted(

            data["point_idx"].unique()

        ):


            point_data = data[

                data["point_idx"]
                ==
                point_idx

            ]


            if len(point_data) < 2:

                continue


            x = point_data[
                x_col
            ].values


            y = point_data[
                y_col
            ].values


            dx_tv = np.diff(x)

            dy_tv = np.diff(y)


            movement = np.sqrt(

                dx_tv ** 2
                +
                dy_tv ** 2

            )


            total_variation += float(

                np.sum(movement)

            )


        all_results.append({

            "Trial":
                trial_number,

            "Method":
                method,

            "MAE_px":
                mae,

            "RMSE_px":
                rmse,

            "Mean_Jitter_px":
                mean_jitter,

            "Total_Variation_px":
                total_variation

        })


# ============================================================
# DATAFRAME
# ============================================================

combined_df = pd.DataFrame(

    all_results

)


if len(combined_df) == 0:

    print(
        "ERROR: No results found."
    )

    sys.exit(1)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

raw_file = os.path.join(

    TRIAL_ROOT,

    "all_trial_results.csv"

)


combined_df.to_csv(

    raw_file,

    index=False

)


# ============================================================
# FINAL MEAN + SD
# ============================================================

METHOD_ORDER = [

    "Baseline",

    "Method 2",

    "Fixed KF",

    "Adaptive Q",

    "Adaptive R",

    "Proposed"

]


summary = []


for method in METHOD_ORDER:


    data = combined_df[

        combined_df["Method"]
        ==
        method

    ]


    if len(data) == 0:

        continue


    summary.append({

        "Method":
            method,

        "Trials":
            len(data),

        "MAE_Mean":
            data["MAE_px"].mean(),

        "MAE_SD":
            data["MAE_px"].std(
                ddof=1
            ),

        "RMSE_Mean":
            data["RMSE_px"].mean(),

        "RMSE_SD":
            data["RMSE_px"].std(
                ddof=1
            ),

        "Jitter_Mean":
            data["Mean_Jitter_px"].mean(),

        "Jitter_SD":
            data["Mean_Jitter_px"].std(
                ddof=1
            ),

        "TV_Mean":
            data["Total_Variation_px"].mean(),

        "TV_SD":
            data["Total_Variation_px"].std(
                ddof=1
            )

    })


final_df = pd.DataFrame(
    summary
)


# ============================================================
# SAVE
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
# PRINT
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


for _, row in final_df.iterrows():


    print(

        f"{row['Method']:<20}"

        f"{row['MAE_Mean']:.2f}"
        f" ± "
        f"{row['MAE_SD']:.2f}".rjust(15)

        +

        f"{row['RMSE_Mean']:.2f}"
        f" ± "
        f"{row['RMSE_SD']:.2f}".rjust(15)

        +

        f"{row['Jitter_Mean']:.2f}"
        f" ± "
        f"{row['Jitter_SD']:.2f}".rjust(15)

        +

        f"{row['TV_Mean']:.2f}"
        f" ± "
        f"{row['TV_SD']:.2f}".rjust(17)

    )


# ============================================================
# IMPROVEMENT
# ============================================================

baseline_rows = final_df[

    final_df["Method"]
    ==
    "Baseline"

]


if len(baseline_rows) > 0:


    baseline_mae = float(

        baseline_rows[
            "MAE_Mean"
        ].iloc[0]

    )


    baseline_jitter = float(

        baseline_rows[
            "Jitter_Mean"
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


        rows = final_df[

            final_df["Method"]
            ==
            method

        ]


        if len(rows) == 0:

            continue


        mae = float(

            rows[
                "MAE_Mean"
            ].iloc[0]

        )


        jitter = float(

            rows[
                "Jitter_Mean"
            ].iloc[0]

        )


        mae_reduction = (

            (
                baseline_mae
                -
                mae
            )

            /

            baseline_mae

        ) * 100


        jitter_reduction = (

            (
                baseline_jitter
                -
                jitter
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
    f"Raw trial results: {raw_file}"
)

print()

print(
    f"Final mean ± SD: {final_file}"
)

print()