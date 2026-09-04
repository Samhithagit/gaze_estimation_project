import pandas as pd
import numpy as np
import sys
import os


# ============================================================
# INPUT
# ============================================================

if len(sys.argv) >= 2:

    INPUT_FILE = sys.argv[1]

else:

    INPUT_FILE = "unified_evaluation.csv"


# ============================================================
# READ
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print()
print("=" * 100)
print("FINAL GAZE EVALUATION")
print("=" * 100)

print()

print(
    f"Total evaluation samples: {len(df)}"
)

print(
    f"Number of fixation points: "
    f"{df['point_idx'].nunique()}"
)

print()


# ============================================================
# METHODS
#
# FIX: "Proposed" previously pointed at kalman_yaw/kalman_pitch
# (Fixed Kalman) -- that's an ablation baseline, not your novel
# contribution. Renamed to "Fixed KF" for clarity, and "Proposed"
# now correctly points at adaptive_qr_yaw/adaptive_qr_pitch
# (the combined confidence-gated-R + innovation-gated-Q filter),
# which matches your report's actual AKF architecture.
# ============================================================

METHODS = {

    "Baseline": (
        "raw_gaze_yaw",
        "raw_gaze_pitch"
    ),

    "Method 2": (
        "fused_yaw",
        "fused_pitch"
    ),

    "Fixed KF": (
        "kalman_yaw",
        "kalman_pitch"
    ),

    "Adaptive Q": (
        "adaptive_q_yaw",
        "adaptive_q_pitch"
    ),

    "Adaptive R": (
        "adaptive_r_yaw",
        "adaptive_r_pitch"
    ),

    "Proposed": (
        "adaptive_qr_yaw",
        "adaptive_qr_pitch"
    )
}


# ============================================================
# MAPPING
# ============================================================

def make_design_matrix(
    yaw,
    pitch
):

    yaw = np.asarray(yaw)

    pitch = np.asarray(pitch)

    return np.stack([

        np.ones_like(yaw),

        yaw,
        pitch,

        yaw ** 2,
        pitch ** 2,

        yaw * pitch

    ], axis=1)


def fit_mapping(
    yaw,
    pitch,
    target_x,
    target_y
):

    A = make_design_matrix(
        yaw,
        pitch
    )


    coef_x, _, _, _ = np.linalg.lstsq(

        A,

        np.asarray(target_x),

        rcond=None
    )


    coef_y, _, _, _ = np.linalg.lstsq(

        A,

        np.asarray(target_y),

        rcond=None
    )


    return coef_x, coef_y


def predict_screen_position(
    yaw,
    pitch,
    coef_x,
    coef_y
):

    A = make_design_matrix(
        yaw,
        pitch
    )


    predicted_x = A @ coef_x

    predicted_y = A @ coef_y


    return predicted_x, predicted_y


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    data,
    predicted_x,
    predicted_y
):

    target_x = data[
        "target_x"
    ].values

    target_y = data[
        "target_y"
    ].values


    # --------------------------------------------------------
    # FRAME ERROR
    # --------------------------------------------------------

    errors = np.sqrt(

        (
            predicted_x
            -
            target_x
        ) ** 2

        +

        (
            predicted_y
            -
            target_y
        ) ** 2

    )


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = float(
        np.mean(errors)
    )


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = float(

        np.sqrt(
            np.mean(
                errors ** 2
            )
        )

    )


    # --------------------------------------------------------
    # JITTER
    # --------------------------------------------------------

    jitter_values = []


    for point_idx in data[
        "point_idx"
    ].unique():

        mask = (

            data["point_idx"].values
            ==
            point_idx

        )


        px = predicted_x[mask]

        py = predicted_y[mask]


        if len(px) < 2:

            continue


        jitter_x = np.std(px)

        jitter_y = np.std(py)


        jitter = np.sqrt(

            jitter_x ** 2

            +

            jitter_y ** 2

        )


        jitter_values.append(
            jitter
        )


    if len(jitter_values) > 0:

        mean_jitter = float(
            np.mean(
                jitter_values
            )
        )

        jitter_std = float(
            np.std(
                jitter_values
            )
        )

        maximum_jitter = float(
            np.max(
                jitter_values
            )
        )

    else:

        mean_jitter = np.nan

        jitter_std = np.nan

        maximum_jitter = np.nan


    # --------------------------------------------------------
    # TOTAL VARIATION
    # --------------------------------------------------------

    total_variation_values = []


    for point_idx in data[
        "point_idx"
    ].unique():

        mask = (

            data["point_idx"].values
            ==
            point_idx

        )


        px = predicted_x[mask]

        py = predicted_y[mask]


        if len(px) < 2:

            continue


        dx = np.diff(px)

        dy = np.diff(py)


        movement = np.sqrt(

            dx ** 2

            +

            dy ** 2

        )


        total_variation_values.append(

            np.sum(movement)

        )


    if len(total_variation_values) > 0:

        total_variation = float(

            np.mean(
                total_variation_values
            )

        )

    else:

        total_variation = np.nan


    return {

        "MAE": mae,

        "RMSE": rmse,

        "Mean Jitter": mean_jitter,

        "Jitter Std": jitter_std,

        "Maximum Jitter": maximum_jitter,

        "Total Variation": total_variation

    }


# ============================================================
# EVALUATE
# ============================================================

results = {}


for method_name, (
    yaw_column,
    pitch_column
) in METHODS.items():

    print(
        f"Calculating {method_name}..."
    )


    valid = (

        df[yaw_column].notna()

        &

        df[pitch_column].notna()

        &

        df["target_x"].notna()

        &

        df["target_y"].notna()

    )


    data = df[
        valid
    ].copy()


    yaw = data[
        yaw_column
    ].values


    pitch = data[
        pitch_column
    ].values


    target_x = data[
        "target_x"
    ].values


    target_y = data[
        "target_y"
    ].values


    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    coef_x, coef_y = fit_mapping(

        yaw,

        pitch,

        target_x,

        target_y

    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predicted_x, predicted_y = (

        predict_screen_position(

            yaw,

            pitch,

            coef_x,

            coef_y

        )

    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    results[
        method_name
    ] = calculate_metrics(

        data,

        predicted_x,

        predicted_y

    )


# ============================================================
# PRINT
# ============================================================

print()

print("=" * 110)

print(
    "GAZE ACCURACY + JITTER COMPARISON"
)

print("=" * 110)

print()

print(

    f"{'Method':<20}"

    f"{'MAE (px)':>14}"

    f"{'RMSE (px)':>14}"

    f"{'Jitter (px)':>16}"

    f"{'Jitter Std':>16}"

    f"{'Max Jitter':>16}"

    f"{'Total Var.':>16}"

)

print(
    "-" * 110
)


for method_name in METHODS:

    r = results[
        method_name
    ]


    print(

        f"{method_name:<20}"

        f"{r['MAE']:>14.2f}"

        f"{r['RMSE']:>14.2f}"

        f"{r['Mean Jitter']:>16.2f}"

        f"{r['Jitter Std']:>16.2f}"

        f"{r['Maximum Jitter']:>16.2f}"

        f"{r['Total Variation']:>16.2f}"

    )


# ============================================================
# IMPROVEMENT
# ============================================================

baseline_mae = results[
    "Baseline"
]["MAE"]

baseline_jitter = results[
    "Baseline"
]["Mean Jitter"]


print()

print("=" * 90)

print(
    "IMPROVEMENT OVER BASELINE"
)

print("=" * 90)


for method_name in [

    "Method 2",

    "Fixed KF",

    "Adaptive Q",

    "Adaptive R",

    "Proposed"

]:

    method_mae = results[
        method_name
    ]["MAE"]


    method_jitter = results[
        method_name
    ]["Mean Jitter"]


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
        method_name
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
# SAVE SUMMARY
# ============================================================

summary_rows = []


for method_name in METHODS:

    r = results[
        method_name
    ]


    summary_rows.append({

        "Method": method_name,

        "MAE_px": r["MAE"],

        "RMSE_px": r["RMSE"],

        "Mean_Jitter_px":
            r["Mean Jitter"],

        "Jitter_Std_px":
            r["Jitter Std"],

        "Maximum_Jitter_px":
            r["Maximum Jitter"],

        "Total_Variation_px":
            r["Total Variation"]

    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# OUTPUT PATH
# ============================================================

directory = os.path.dirname(
    INPUT_FILE
)

if directory == "":
    directory = "."


summary_file = os.path.join(

    directory,

    "final_evaluation_summary.csv"

)


summary_df.to_csv(

    summary_file,

    index=False

)


print()

print("=" * 90)

print(
    "Summary saved to:"
)

print(
    summary_file
)

print("=" * 90)