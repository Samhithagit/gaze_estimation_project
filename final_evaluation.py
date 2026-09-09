import os
import sys
import csv
import numpy as np

from ridge_gaze_mapper import RidgeGazeMapper

from module3.gaze_kalman import (
    FixedKalmanGaze,
    AdaptiveQKalmanGaze,
    AdaptiveRKalmanGaze,
    AdaptiveQRKalmanGaze
)


# ============================================================
# INPUT FILES
# ============================================================

if len(sys.argv) >= 3:

    CALIBRATION_FILE = sys.argv[1]
    EVALUATION_FILE = sys.argv[2]

else:

    print(
        "Usage: python final_evaluation.py "
        "calibration_data.csv evaluation_data.csv"
    )

    sys.exit(1)


# ============================================================
# KALMAN PARAMETERS
# ============================================================

KALMAN_PROCESS_NOISE = 25.0
KALMAN_MEASUREMENT_NOISE = 100.0
KALMAN_INITIAL_UNCERTAINTY = 100.0


# ============================================================
# ADAPTIVE Q
# ============================================================

ADAPTIVE_Q_ALPHA = 1.0
ADAPTIVE_Q_MIN = 0.1
ADAPTIVE_Q_MAX = 2500.0


# ============================================================
# ADAPTIVE R
# ============================================================

ADAPTIVE_R_K = 2.0
ADAPTIVE_R_MIN = 25.0
ADAPTIVE_R_MAX = 1000.0


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(filename):

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    rows = []

    with open(
        filename,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# FLOAT
# ============================================================

def to_float(value):

    try:
        return float(value)

    except (
        TypeError,
        ValueError
    ):

        return None


# ============================================================
# VALID ROWS
# ============================================================

def valid_rows(rows):

    output = []

    required = [

        "raw_gaze_yaw",
        "raw_gaze_pitch",
        "raw_confidence",

        "head_yaw",
        "head_pitch",
        "head_roll",

        "target_x",
        "target_y",

        "point_idx"

    ]

    for row in rows:

        values = {}
        valid = True

        for column in required:

            value = to_float(
                row.get(column)
            )

            if value is None:

                valid = False
                break

            values[column] = value

        if valid:
            output.append(values)

    return output


# ============================================================
# FIT RIDGE
# ============================================================

def fit_mappers(rows):

    yaw = np.array(
        [
            r["raw_gaze_yaw"]
            for r in rows
        ],
        dtype=float
    )

    pitch = np.array(
        [
            r["raw_gaze_pitch"]
            for r in rows
        ],
        dtype=float
    )

    target_x = np.array(
        [
            r["target_x"]
            for r in rows
        ],
        dtype=float
    )

    target_y = np.array(
        [
            r["target_y"]
            for r in rows
        ],
        dtype=float
    )

    head_yaw = np.array(
        [
            r["head_yaw"]
            for r in rows
        ],
        dtype=float
    )

    head_pitch = np.array(
        [
            r["head_pitch"]
            for r in rows
        ],
        dtype=float
    )

    head_roll = np.array(
        [
            r["head_roll"]
            for r in rows
        ],
        dtype=float
    )

    # ========================================================
    # BASELINE
    # ========================================================

    baseline_mapper = RidgeGazeMapper(
        feature_mode="gaze_only",
        alpha=1e-6
    )

    baseline_mapper.fit(
        yaw,
        pitch,
        target_x,
        target_y
    )

    # ========================================================
    # METHOD 2
    # ========================================================

    method2_mapper = RidgeGazeMapper(
        feature_mode="gaze_head",
        alpha=0.01
    )

    method2_mapper.fit(
        yaw,
        pitch,
        target_x,
        target_y,
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        head_roll=head_roll
    )

    return (
        baseline_mapper,
        method2_mapper
    )


# ============================================================
# PREDICT ONE ROW
# ============================================================

def predict(
    mapper,
    row,
    use_head
):

    yaw = np.array(
        [row["raw_gaze_yaw"]],
        dtype=float
    )

    pitch = np.array(
        [row["raw_gaze_pitch"]],
        dtype=float
    )

    if use_head:

        prediction = mapper.predict(

            yaw,
            pitch,

            head_yaw=np.array(
                [row["head_yaw"]],
                dtype=float
            ),

            head_pitch=np.array(
                [row["head_pitch"]],
                dtype=float
            ),

            head_roll=np.array(
                [row["head_roll"]],
                dtype=float
            )
        )

    else:

        prediction = mapper.predict(
            yaw,
            pitch
        )

    x = np.asarray(
        prediction[0]
    ).reshape(-1)[0]

    y = np.asarray(
        prediction[1]
    ).reshape(-1)[0]

    return float(x), float(y)


# ============================================================
# CREATE FILTERS
# ============================================================

def create_filters():

    fixed = FixedKalmanGaze(

        process_noise=KALMAN_PROCESS_NOISE,

        measurement_noise=KALMAN_MEASUREMENT_NOISE,

        initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY

    )

    adaptive_q = AdaptiveQKalmanGaze(

        process_noise=KALMAN_PROCESS_NOISE,

        measurement_noise=KALMAN_MEASUREMENT_NOISE,

        initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY,

        q_alpha=ADAPTIVE_Q_ALPHA,

        q_min=ADAPTIVE_Q_MIN,

        q_max=ADAPTIVE_Q_MAX

    )

    adaptive_r = AdaptiveRKalmanGaze(

        process_noise=KALMAN_PROCESS_NOISE,

        measurement_noise=KALMAN_MEASUREMENT_NOISE,

        initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY,

        r_k=ADAPTIVE_R_K,

        r_min=ADAPTIVE_R_MIN,

        r_max=ADAPTIVE_R_MAX

    )

    adaptive_qr = AdaptiveQRKalmanGaze(

        process_noise=KALMAN_PROCESS_NOISE,

        measurement_noise=KALMAN_MEASUREMENT_NOISE,

        initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY,

        q_alpha=ADAPTIVE_Q_ALPHA,

        q_min=ADAPTIVE_Q_MIN,

        q_max=ADAPTIVE_Q_MAX,

        r_k=ADAPTIVE_R_K,

        r_min=ADAPTIVE_R_MIN,

        r_max=ADAPTIVE_R_MAX

    )

    return (
        fixed,
        adaptive_q,
        adaptive_r,
        adaptive_qr
    )


# ============================================================
# UPDATE FILTER
# ============================================================

def update_filter(
    filter_object,
    x,
    y,
    confidence=None,
    head_motion=0.0
):

    result = filter_object.update(

        x,
        y,

        confidence=confidence,

        head_motion=head_motion

    )

    if result is None:
        return None

    return (
        float(result["x"]),
        float(result["y"])
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    rows,
    baseline_mapper,
    method2_mapper
):

    (
        fixed,
        adaptive_q,
        adaptive_r,
        adaptive_qr
    ) = create_filters()

    results = []

    # --------------------------------------------------------
    # Previous head pose
    # --------------------------------------------------------

    previous_head = None

    # ========================================================
    # FRAME LOOP
    # ========================================================

    for row in rows:

        # ----------------------------------------------------
        # CURRENT HEAD POSE
        # ----------------------------------------------------

        current_head = np.array([

            row["head_yaw"],
            row["head_pitch"],
            row["head_roll"]

        ], dtype=float)

        # ----------------------------------------------------
        # HEAD MOTION
        #
        # Delta_head = || H_t - H_(t-1) ||
        # ----------------------------------------------------

        if previous_head is None:

            head_motion = 0.0

        else:

            head_motion = float(

                np.linalg.norm(
                    current_head
                    - previous_head
                )

            )

        previous_head = (
            current_head.copy()
        )

        # ====================================================
        # BASELINE
        # ====================================================

        baseline_x, baseline_y = predict(

            baseline_mapper,

            row,

            use_head=False

        )

        # ====================================================
        # METHOD 2
        # ====================================================

        ridge_x, ridge_y = predict(

            method2_mapper,

            row,

            use_head=True

        )

        # ====================================================
        # FIXED KF
        # ====================================================

        fixed_x, fixed_y = update_filter(

            fixed,

            ridge_x,
            ridge_y

        )

        # ====================================================
        # ADAPTIVE Q
        # ====================================================

        q_x, q_y = update_filter(

            adaptive_q,

            ridge_x,
            ridge_y,

            head_motion=head_motion

        )

        # ====================================================
        # ADAPTIVE R
        # ====================================================

        r_x, r_y = update_filter(

            adaptive_r,

            ridge_x,
            ridge_y,

            confidence=row["raw_confidence"]

        )

        # ====================================================
        # PROPOSED Q + R
        # ====================================================

        qr_x, qr_y = update_filter(

            adaptive_qr,

            ridge_x,
            ridge_y,

            confidence=row["raw_confidence"],

            head_motion=head_motion

        )

        # ====================================================
        # STORE
        # ====================================================

        results.append({

            "point_idx":
                int(row["point_idx"]),

            "target_x":
                row["target_x"],

            "target_y":
                row["target_y"],

            "baseline_x":
                baseline_x,

            "baseline_y":
                baseline_y,

            "ridge_x":
                ridge_x,

            "ridge_y":
                ridge_y,

            "fixed_kf_x":
                fixed_x,

            "fixed_kf_y":
                fixed_y,

            "adaptive_q_x":
                q_x,

            "adaptive_q_y":
                q_y,

            "adaptive_r_x":
                r_x,

            "adaptive_r_y":
                r_y,

            "adaptive_qr_x":
                qr_x,

            "adaptive_qr_y":
                qr_y,

            "confidence":
                row["raw_confidence"],

            "head_motion":
                head_motion

        })

    return results


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    results,
    x_key,
    y_key
):

    errors = []

    for row in results:

        dx = (
            row[x_key]
            - row["target_x"]
        )

        dy = (
            row[y_key]
            - row["target_y"]
        )

        errors.append(
            np.sqrt(
                dx ** 2
                + dy ** 2
            )
        )

    errors = np.asarray(
        errors,
        dtype=float
    )

    return {

        "MAE":
            float(
                np.mean(errors)
            ),

        "RMSE":
            float(
                np.sqrt(
                    np.mean(
                        errors ** 2
                    )
                )
            ),

        "Median":
            float(
                np.median(errors)
            ),

        "P95":
            float(
                np.percentile(
                    errors,
                    95
                )
            )

    }


# ============================================================
# JITTER
# ============================================================

def calculate_jitter(
    results,
    x_key,
    y_key
):

    values = []

    point_ids = sorted(
        set(
            r["point_idx"]
            for r in results
        )
    )

    for point_idx in point_ids:

        rows = [

            r

            for r in results

            if r["point_idx"] == point_idx

        ]

        if len(rows) < 2:
            continue

        x = np.array(
            [
                r[x_key]
                for r in rows
            ],
            dtype=float
        )

        y = np.array(
            [
                r[y_key]
                for r in rows
            ],
            dtype=float
        )

        dx = np.diff(x)
        dy = np.diff(y)

        movement = np.sqrt(
            dx ** 2
            + dy ** 2
        )

        values.extend(
            movement.tolist()
        )

    if len(values) == 0:
        return 0.0

    return float(
        np.mean(values)
    )


# ============================================================
# TOTAL VARIATION
# ============================================================

def calculate_total_variation(
    results,
    x_key,
    y_key
):

    total = 0.0

    point_ids = sorted(
        set(
            r["point_idx"]
            for r in results
        )
    )

    for point_idx in point_ids:

        rows = [

            r

            for r in results

            if r["point_idx"] == point_idx

        ]

        if len(rows) < 2:
            continue

        x = np.array(
            [
                r[x_key]
                for r in rows
            ],
            dtype=float
        )

        y = np.array(
            [
                r[y_key]
                for r in rows
            ],
            dtype=float
        )

        dx = np.diff(x)
        dy = np.diff(y)

        total += float(
            np.sum(
                np.sqrt(
                    dx ** 2
                    + dy ** 2
                )
            )
        )

    return total


# ============================================================
# PRINT METHOD
# ============================================================

def print_method(
    name,
    results,
    x_key,
    y_key
):

    metrics = calculate_metrics(
        results,
        x_key,
        y_key
    )

    jitter = calculate_jitter(
        results,
        x_key,
        y_key
    )

    variation = calculate_total_variation(
        results,
        x_key,
        y_key
    )

    print()
    print(name)

    print(
        f"  MAE:     "
        f"{metrics['MAE']:.2f} px"
    )

    print(
        f"  RMSE:    "
        f"{metrics['RMSE']:.2f} px"
    )

    print(
        f"  Median:  "
        f"{metrics['Median']:.2f} px"
    )

    print(
        f"  P95:     "
        f"{metrics['P95']:.2f} px"
    )

    print(
        f"  Jitter:  "
        f"{jitter:.2f} px/frame"
    )

    print(
        f"  Total variation: "
        f"{variation:.2f} px"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    filename
):

    fields = [

        "point_idx",

        "target_x",
        "target_y",

        "baseline_x",
        "baseline_y",

        "ridge_x",
        "ridge_y",

        "fixed_kf_x",
        "fixed_kf_y",

        "adaptive_q_x",
        "adaptive_q_y",

        "adaptive_r_x",
        "adaptive_r_y",

        "adaptive_qr_x",
        "adaptive_qr_y",

        "confidence",

        "head_motion"

    ]

    with open(
        filename,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("FINAL GAZE EVALUATION")
    print("=" * 80)
    print()

    # ========================================================
    # CALIBRATION
    # ========================================================

    calibration_raw = load_csv(
        CALIBRATION_FILE
    )

    calibration_rows = valid_rows(
        calibration_raw
    )

    print(
        f"Calibration samples: "
        f"{len(calibration_rows)}"
    )

    if len(calibration_rows) == 0:

        raise ValueError(
            "No valid calibration samples."
        )

    # ========================================================
    # FIT RIDGE
    # ========================================================

    print()
    print(
        "Fitting Ridge on CALIBRATION data only..."
    )

    (
        baseline_mapper,
        method2_mapper
    ) = fit_mappers(
        calibration_rows
    )

    print(
        "Ridge fitting complete."
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    evaluation_raw = load_csv(
        EVALUATION_FILE
    )

    evaluation_rows = valid_rows(
        evaluation_raw
    )

    print()

    print(
        f"Evaluation samples: "
        f"{len(evaluation_rows)}"
    )

    if len(evaluation_rows) == 0:

        raise ValueError(
            "No valid evaluation samples."
        )

        # ========================================================
    # HEAD POSE DIAGNOSTIC
    # ========================================================

    for name, rows_ in [
        ("Calibration", calibration_rows),
        ("Evaluation", evaluation_rows)
    ]:

        hy = np.array(
            [r["head_yaw"] for r in rows_],
            dtype=float
        )

        hp = np.array(
            [r["head_pitch"] for r in rows_],
            dtype=float
        )

        hr = np.array(
            [r["head_roll"] for r in rows_],
            dtype=float
        )

        print()
        print(f"{name} head-pose statistics:")

        print(
            f"  Yaw:   "
            f"mean={hy.mean():+.2f}, "
            f"std={hy.std():.2f}, "
            f"range=[{hy.min():.2f}, {hy.max():.2f}]"
        )

        print(
            f"  Pitch: "
            f"mean={hp.mean():+.2f}, "
            f"std={hp.std():.2f}, "
            f"range=[{hp.min():.2f}, {hp.max():.2f}]"
        )

        print(
            f"  Roll:  "
            f"mean={hr.mean():+.2f}, "
            f"std={hr.std():.2f}, "
            f"range=[{hr.min():.2f}, {hr.max():.2f}]"
        )
    # ========================================================
    # EVALUATE
    # ========================================================

    results = evaluate(

        evaluation_rows,

        baseline_mapper,

        method2_mapper

    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print_method(

        "Baseline: L2CS + Ridge (gaze only)",

        results,

        "baseline_x",
        "baseline_y"

    )

    print_method(

        "Method 2: L2CS + Head Pose + Ridge",

        results,

        "ridge_x",
        "ridge_y"

    )

    print_method(

        "Fixed Kalman",

        results,

        "fixed_kf_x",
        "fixed_kf_y"

    )

    print_method(

        "Adaptive Q Kalman",

        results,

        "adaptive_q_x",
        "adaptive_q_y"

    )

    print_method(

        "Adaptive R Kalman",

        results,

        "adaptive_r_x",
        "adaptive_r_y"

    )

    print_method(

        "Proposed: Adaptive Q + R",

        results,

        "adaptive_qr_x",
        "adaptive_qr_y"

    )

    # ========================================================
    # SAVE
    # ========================================================

    output_file = os.path.join(

        os.path.dirname(
            EVALUATION_FILE
        ),

        "final_evaluation_results.csv"

    )

    save_results(
        results,
        output_file
    )

    print()

    print(
        f"Results saved to: "
        f"{output_file}"
    )

    print()

    print("=" * 80)
    print("PIPELINE")
    print("=" * 80)

    print()

    print(
        "Baseline:"
    )

    print(
        "L2CS -> Ridge(gaze only) -> screen"
    )

    print()

    print(
        "Method 2:"
    )

    print(
        "L2CS + Head Pose -> Ridge -> screen"
    )

    print()

    print(
        "Fixed KF:"
    )

    print(
        "Method 2 Ridge -> Fixed Constant-Velocity KF -> screen"
    )

    print()

    print(
        "Adaptive Q:"
    )

    print(
        "Method 2 Ridge -> Head-Motion Adaptive Q KF -> screen"
    )

    print()

    print(
        "Adaptive R:"
    )

    print(
        "Method 2 Ridge -> Confidence Adaptive R KF -> screen"
    )

    print()

    print(
        "Proposed:"
    )

    print(
        "Method 2 Ridge -> Adaptive Q(head) + R(confidence) KF -> screen"
    )

    print()


if __name__ == "__main__":
    main()