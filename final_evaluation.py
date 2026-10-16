import os

import sys

import csv

import numpy as np

from sklearn.linear_model import Ridge



from ridge_gaze_mapper import RidgeGazeMapper



from module3.gaze_kalman import (
    FixedKalmanGaze,
    MovingMedianGaze,
    AdaptiveQKalmanGaze,
    AdaptiveRKalmanGaze,
    AdaptiveQRKalmanGaze,
)





# ============================================================

# INPUT FILES

# ============================================================



if len(sys.argv) < 3:

    print("Usage:")

    print("python final_evaluation_updated.py calibration_data.csv evaluation_data.csv")

    sys.exit(1)



CALIBRATION_FILE = sys.argv[1]

EVALUATION_FILE = sys.argv[2]





# ============================================================

# RIDGE SETTINGS

# ============================================================



# Baseline: raw L2CS gaze only

BASELINE_ALPHA = 1e-6



# Method 2:

# raw L2CS gaze + head yaw + head pitch

#

# IMPORTANT:

# head_roll is intentionally NOT used in Ridge.

# The diagnostic showed that adding roll caused severe

# evaluation degradation.

METHOD2_ALPHA = 0.01





# ============================================================

# KALMAN SETTINGS

# ============================================================



# Shared base parameters for adaptive filters and matched fixed-Q baseline.
KALMAN_PROCESS_NOISE = 25.0
KALMAN_MEASUREMENT_NOISE = 100.0
KALMAN_INITIAL_UNCERTAINTY = 100.0

# Paper-style fixed baseline parameters.
PAPER_PROCESS_NOISE_DENSITY = 8e-4
PAPER_MEASUREMENT_NOISE = 0.2
# The paper does not report P0. This is an explicit implementation choice,
# not a paper-reported parameter.
PAPER_INITIAL_UNCERTAINTY = 100.0
MEDIAN_WINDOW_SIZE = 5

# Used only when the input CSV has no usable timestamp column. In that case,
# the fixed filter uses one sample interval per row (frame-based fallback).
FALLBACK_FRAME_INTERVAL = 1.0





# ------------------------------------------------------------

# Adaptive Q

# Q_t = Q_base  (1 + alpha  head_motion)

# ------------------------------------------------------------



ADAPTIVE_Q_ALPHA = 1.0

ADAPTIVE_Q_MIN = 0.1

ADAPTIVE_Q_MAX = 2500.0





# ------------------------------------------------------------

# Adaptive R

# R is confidence driven

# ------------------------------------------------------------



ADAPTIVE_R_K = 2.0

ADAPTIVE_R_MIN = 25.0

ADAPTIVE_R_MAX = 1000.0





# ------------------------------------------------------------

# Proposed Q + R

# ------------------------------------------------------------



PROPOSED_Q_ALPHA = 1.0

PROPOSED_Q_MIN = 0.1

PROPOSED_Q_MAX = 2500.0



PROPOSED_R_K = 2.0

PROPOSED_R_MIN = 25.0

PROPOSED_R_MAX = 1000.0





# ============================================================

# CSV

# ============================================================



def load_csv(filename):

    if not os.path.exists(filename):

        raise FileNotFoundError(filename)



    with open(filename, "r", newline="") as f:

        return list(csv.DictReader(f))





def to_float(value):

    try:

        return float(value)

    except (TypeError, ValueError):

        return None





def valid_rows(rows):

    required = [

        "raw_gaze_yaw",

        "raw_gaze_pitch",

        "raw_confidence",

        "head_yaw",

        "head_pitch",

        "head_roll",

        "target_x",

        "target_y",

        "point_idx",

    ]



    output = []



    for row in rows:

        values = {}

        valid = True



        for column in required:

            value = to_float(row.get(column))



            if value is None or not np.isfinite(value):

                valid = False

                break



            values[column] = value



        timestamp_value = to_float(row.get("timestamp"))
        if timestamp_value is not None:
            values["timestamp"] = timestamp_value

        if valid:

            output.append(values)



    return output





# ============================================================

# RIDGE MAPPERS

# ============================================================



class SimpleRidgeMapper:

    """

    Local Ridge mapper used only for the updated Method 2.



    Features:

        raw_gaze_yaw

        raw_gaze_pitch

        head_yaw

        head_pitch



    head_roll is deliberately excluded.

    """



    def __init__(self, alpha=0.01):

        self.model_x = Ridge(alpha=alpha)

        self.model_y = Ridge(alpha=alpha)



    def fit(

        self,

        gaze_yaw,

        gaze_pitch,

        head_yaw,

        head_pitch,

        target_x,

        target_y,

    ):

        X = np.column_stack([

            gaze_yaw,

            gaze_pitch,

            head_yaw,

            head_pitch,

        ])



        self.model_x.fit(X, target_x)

        self.model_y.fit(X, target_y)



    def predict(

        self,

        gaze_yaw,

        gaze_pitch,

        head_yaw,

        head_pitch,

    ):

        X = np.column_stack([

            gaze_yaw,

            gaze_pitch,

            head_yaw,

            head_pitch,

        ])



        return (

            self.model_x.predict(X),

            self.model_y.predict(X),

        )





def fit_mappers(rows):

    yaw = np.array(

        [r["raw_gaze_yaw"] for r in rows],

        dtype=float,

    )



    pitch = np.array(

        [r["raw_gaze_pitch"] for r in rows],

        dtype=float,

    )



    target_x = np.array(

        [r["target_x"] for r in rows],

        dtype=float,

    )



    target_y = np.array(

        [r["target_y"] for r in rows],

        dtype=float,

    )



    head_yaw = np.array(

        [r["head_yaw"] for r in rows],

        dtype=float,

    )



    head_pitch = np.array(

        [r["head_pitch"] for r in rows],

        dtype=float,

    )



    # ========================================================

    # BASELINE

    # ========================================================



    baseline_mapper = RidgeGazeMapper(

        feature_mode="gaze_only",

        alpha=BASELINE_ALPHA,

    )



    baseline_mapper.fit(

        yaw,

        pitch,

        target_x,

        target_y,

    )



    # ========================================================

    # METHOD 2

    # ========================================================

    #

    # IMPORTANT:

    # Do NOT use RidgeGazeMapper(feature_mode="gaze_head")

    # here because that mode includes head_roll.

    #

    # We use only:

    #   gaze yaw

    #   gaze pitch

    #   head yaw

    #   head pitch

    #

    # This is the feature set supported by the diagnostic:

    # "Gaze + Head Yaw + Pitch".

    # ========================================================



    method2_mapper = SimpleRidgeMapper(

        alpha=METHOD2_ALPHA,

    )



    method2_mapper.fit(

        gaze_yaw=yaw,

        gaze_pitch=pitch,

        head_yaw=head_yaw,

        head_pitch=head_pitch,

        target_x=target_x,

        target_y=target_y,

    )



    return baseline_mapper, method2_mapper





# ============================================================

# PREDICTION

# ============================================================



def predict_baseline(mapper, row):

    yaw = np.array(

        [row["raw_gaze_yaw"]],

        dtype=float,

    )



    pitch = np.array(

        [row["raw_gaze_pitch"]],

        dtype=float,

    )



    prediction = mapper.predict(

        yaw,

        pitch,

    )



    if isinstance(prediction, dict):

        x = prediction["x"]

        y = prediction["y"]

    else:

        x = prediction[0]

        y = prediction[1]



    x = np.asarray(x).reshape(-1)[0]

    y = np.asarray(y).reshape(-1)[0]



    return float(x), float(y)





def predict_method2(mapper, row):

    yaw = np.array(

        [row["raw_gaze_yaw"]],

        dtype=float,

    )



    pitch = np.array(

        [row["raw_gaze_pitch"]],

        dtype=float,

    )



    head_yaw = np.array(

        [row["head_yaw"]],

        dtype=float,

    )



    head_pitch = np.array(

        [row["head_pitch"]],

        dtype=float,

    )



    prediction = mapper.predict(

        gaze_yaw=yaw,

        gaze_pitch=pitch,

        head_yaw=head_yaw,

        head_pitch=head_pitch,

    )



    x = np.asarray(prediction[0]).reshape(-1)[0]

    y = np.asarray(prediction[1]).reshape(-1)[0]



    return float(x), float(y)





# ============================================================

# KALMAN FILTERS

# ============================================================



def create_filters():
    # Paper-style fixed KF with paper-reported noise parameters.
    paper_fixed = FixedKalmanGaze(
        process_noise_density=PAPER_PROCESS_NOISE_DENSITY,
        measurement_noise=PAPER_MEASUREMENT_NOISE,
        initial_uncertainty=PAPER_INITIAL_UNCERTAINTY,
    )

    # Constant-Q/R comparison baseline matched to the adaptive filters.
    matched_fixed = FixedKalmanGaze(
        process_noise=KALMAN_PROCESS_NOISE,
        measurement_noise=KALMAN_MEASUREMENT_NOISE,
        initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY,
    )

    adaptive_q = AdaptiveQKalmanGaze(
        process_noise=KALMAN_PROCESS_NOISE,
        measurement_noise=KALMAN_MEASUREMENT_NOISE,
        initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY,
        q_alpha=ADAPTIVE_Q_ALPHA, q_min=ADAPTIVE_Q_MIN, q_max=ADAPTIVE_Q_MAX,
    )
    adaptive_r = AdaptiveRKalmanGaze(
        process_noise=KALMAN_PROCESS_NOISE,
        measurement_noise=KALMAN_MEASUREMENT_NOISE,
        initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY,
        r_k=ADAPTIVE_R_K, r_min=ADAPTIVE_R_MIN, r_max=ADAPTIVE_R_MAX,
    )
    proposed = AdaptiveQRKalmanGaze(
        process_noise=KALMAN_PROCESS_NOISE,
        measurement_noise=KALMAN_MEASUREMENT_NOISE,
        initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY,
        q_alpha=PROPOSED_Q_ALPHA, q_min=PROPOSED_Q_MIN, q_max=PROPOSED_Q_MAX,
        r_k=PROPOSED_R_K, r_min=PROPOSED_R_MIN, r_max=PROPOSED_R_MAX,
    )
    return paper_fixed, matched_fixed, adaptive_q, adaptive_r, proposed


def evaluate(

    rows,

    baseline_mapper,

    method2_mapper,

):

    paper_fixed, matched_fixed, adaptive_q, adaptive_r, proposed = create_filters()
    median_filter = MovingMedianGaze(window_size=MEDIAN_WINDOW_SIZE)

    results = []



    previous_head = None
    previous_point_idx = None

    # Use real recorded timestamps when the CSV provides them consistently.
    # Timestamps are expected to be numeric seconds. If no valid timestamp is
    # present, use a frame-index clock as a documented fallback.
    has_recorded_timestamps = bool(rows) and all(
        row.get("timestamp") is not None and np.isfinite(float(row["timestamp"]))
        for row in rows
    )

    for frame_idx, row in enumerate(rows):
        current_point_idx = int(row["point_idx"])
        if previous_point_idx is not None and current_point_idx != previous_point_idx:
            # Reset state between distinct target points.
            median_filter.reset()
            paper_fixed.reset()
            matched_fixed.reset()
            adaptive_q.reset()
            adaptive_r.reset()
            proposed.reset()
            previous_head = None
        previous_point_idx = current_point_idx



        # ----------------------------------------------------

        # HEAD MOTION

        # ----------------------------------------------------

        #

        # IMPORTANT:

        # For adaptive Q / Proposed, use only yaw and pitch.

        # Roll is excluded because the Ridge diagnostic showed

        # that roll was the unstable feature.

        #

        # Delta_head =

        # sqrt((yaw_t-yaw_t-1)^2 +

        #      (pitch_t-pitch_t-1)^2)

        # ----------------------------------------------------



        current_head = np.array([

            row["head_yaw"],

            row["head_pitch"],

        ], dtype=float)



        if previous_head is None:

            head_motion = 0.0

        else:

            head_motion = float(

                np.linalg.norm(

                    current_head - previous_head

                )

            )



        previous_head = current_head.copy()



        # ====================================================

        # BASELINE

        # ====================================================



        baseline_x, baseline_y = predict_baseline(

            baseline_mapper,

            row,

        )



        # ====================================================

        # METHOD 2

        # ====================================================



        ridge_x, ridge_y = predict_method2(

            method2_mapper,

            row,

        )



        # ====================================================
        # Shared 5-frame median preprocessing for every Kalman variant
        # ====================================================
        median_x, median_y = median_filter.update(ridge_x, ridge_y)

        # Paper-style and matched fixed filters use the recorded capture time
        # when available. Otherwise, frame index provides a one-sample-per-step
        # fallback. Adaptive filters deliberately receive no timestamp or dt.
        if has_recorded_timestamps:
            filter_timestamp = float(row["timestamp"])
        else:
            filter_timestamp = frame_idx * FALLBACK_FRAME_INTERVAL

        paper_result = paper_fixed.update(
            median_x, median_y, timestamp=filter_timestamp
        )
        matched_result = matched_fixed.update(
            median_x, median_y, timestamp=filter_timestamp
        )
        paper_x, paper_y = map(float, paper_result)
        matched_x, matched_y = map(float, matched_result)



        # ====================================================

        # ADAPTIVE Q

        # ====================================================



        q_result = adaptive_q.update(

            median_x,

            median_y,

            head_motion=head_motion,

        )



        q_x = float(q_result["x"])

        q_y = float(q_result["y"])



        # ====================================================

        # ADAPTIVE R

        # ====================================================



        r_result = adaptive_r.update(

            median_x,

            median_y,

            confidence=row["raw_confidence"],

        )



        r_x = float(r_result["x"])

        r_y = float(r_result["y"])



        # ====================================================

        # PROPOSED Q + R

        # ====================================================



        proposed_result = proposed.update(

            median_x,

            median_y,

            confidence=row["raw_confidence"],

            head_motion=head_motion,

        )



        proposed_x = float(proposed_result["x"])

        proposed_y = float(proposed_result["y"])



        # ====================================================

        # STORE

        # ====================================================



        results.append({

            "point_idx": int(row["point_idx"]),
            "timestamp": row.get("timestamp"),

            "target_x": row["target_x"],

            "target_y": row["target_y"],



            "baseline_x": baseline_x,

            "baseline_y": baseline_y,



            "ridge_x": ridge_x,

            "ridge_y": ridge_y,



            "fixed_kf_x": paper_x,
            "fixed_kf_y": paper_y,
            "matched_fixed_kf_x": matched_x,
            "matched_fixed_kf_y": matched_y,



            "adaptive_q_x": q_x,

            "adaptive_q_y": q_y,



            "adaptive_r_x": r_x,

            "adaptive_r_y": r_y,



            "adaptive_qr_x": proposed_x,

            "adaptive_qr_y": proposed_y,



            "confidence": row["raw_confidence"],

            "head_motion": head_motion,

        })



    return results





# ============================================================

# METRICS

# ============================================================



def calculate_metrics(results, x_key, y_key):



    errors = []



    for row in results:

        dx = row[x_key] - row["target_x"]

        dy = row[y_key] - row["target_y"]



        errors.append(

            np.sqrt(dx ** 2 + dy ** 2)

        )



    errors = np.asarray(errors, dtype=float)



    return {

        "MAE": float(np.mean(errors)),

        "RMSE": float(

            np.sqrt(np.mean(errors ** 2))

        ),

        "Median": float(np.median(errors)),

        "P95": float(np.percentile(errors, 95)),

    }





def calculate_jitter(results, x_key, y_key):
    """Mean magnitude of second differences (frame-to-frame acceleration)."""
    values = []
    for point_idx in sorted(set(r["point_idx"] for r in results)):
        rows = [r for r in results if r["point_idx"] == point_idx]
        if len(rows) < 3:
            continue
        x = np.asarray([r[x_key] for r in rows], dtype=float)
        y = np.asarray([r[y_key] for r in rows], dtype=float)
        ddx, ddy = np.diff(x, n=2), np.diff(y, n=2)
        values.extend(np.sqrt(ddx ** 2 + ddy ** 2).tolist())
    return float(np.mean(values)) if values else 0.0


def calculate_total_variation(

    results,

    x_key,

    y_key,

):



    total = 0.0



    point_ids = sorted(

        set(r["point_idx"] for r in results)

    )



    for point_idx in point_ids:



        rows = [

            r for r in results

            if r["point_idx"] == point_idx

        ]



        if len(rows) < 2:

            continue



        x = np.array(

            [r[x_key] for r in rows],

            dtype=float,

        )



        y = np.array(

            [r[y_key] for r in rows],

            dtype=float,

        )



        dx = np.diff(x)

        dy = np.diff(y)



        total += float(

            np.sum(

                np.sqrt(

                    dx ** 2 + dy ** 2

                )

            )

        )



    return total





# ============================================================

# PRINT

# ============================================================



def print_method(

    name,

    results,

    x_key,

    y_key,

):



    metrics = calculate_metrics(

        results,

        x_key,

        y_key,

    )



    jitter = calculate_jitter(

        results,

        x_key,

        y_key,

    )



    variation = calculate_total_variation(

        results,

        x_key,

        y_key,

    )



    print()

    print(name)

    print(f"  MAE:             {metrics['MAE']:.2f} px")

    print(f"  RMSE:            {metrics['RMSE']:.2f} px")

    print(f"  Median:          {metrics['Median']:.2f} px")

    print(f"  P95:             {metrics['P95']:.2f} px")

    print(f"  Jitter:          {jitter:.2f} px/frame")

    print(f"  Total variation: {variation:.2f} px")





# ============================================================

# SAVE

# ============================================================



def save_results(results, filename):



    fields = [

        "point_idx",
        "timestamp",

        "target_x",

        "target_y",



        "baseline_x",

        "baseline_y",



        "ridge_x",

        "ridge_y",



        "fixed_kf_x",
        "fixed_kf_y",
        "matched_fixed_kf_x",
        "matched_fixed_kf_y",



        "adaptive_q_x",

        "adaptive_q_y",



        "adaptive_r_x",

        "adaptive_r_y",



        "adaptive_qr_x",

        "adaptive_qr_y",



        "confidence",

        "head_motion",

    ]



    with open(

        filename,

        "w",

        newline="",

    ) as f:



        writer = csv.DictWriter(

            f,

            fieldnames=fields,

        )



        writer.writeheader()

        writer.writerows(results)





# ============================================================

# MAIN

# ============================================================



def main():



    print()

    print("=" * 90)

    print("FINAL GAZE EVALUATION — YAW/PITCH HEAD + ADAPTIVE KF")

    print("=" * 90)



    # --------------------------------------------------------

    # Load calibration

    # --------------------------------------------------------



    calibration = valid_rows(

        load_csv(CALIBRATION_FILE)

    )



    print(

        f"Calibration samples: {len(calibration)}"

    )



    if not calibration:

        raise ValueError(

            "No valid calibration samples."

        )



    # --------------------------------------------------------

    # Fit Ridge

    # --------------------------------------------------------



    print()

    print("Fitting Ridge on calibration data only...")

    print()

    print("Baseline features:")

    print("  raw_gaze_yaw")

    print("  raw_gaze_pitch")



    print()

    print("Method 2 features:")

    print("  raw_gaze_yaw")

    print("  raw_gaze_pitch")

    print("  head_yaw")

    print("  head_pitch")

    print("  head_roll: EXCLUDED")



    baseline_mapper, method2_mapper = fit_mappers(

        calibration

    )



    print()

    print("Ridge fitting complete.")



    # --------------------------------------------------------

    # Load evaluation

    # --------------------------------------------------------



    evaluation = valid_rows(

        load_csv(EVALUATION_FILE)

    )



    print(

        f"Evaluation samples: {len(evaluation)}"

    )



    if not evaluation:

        raise ValueError(

            "No valid evaluation samples."

        )



    # --------------------------------------------------------

    # Evaluate

    # --------------------------------------------------------



    results = evaluate(

        evaluation,

        baseline_mapper,

        method2_mapper,

    )



    # --------------------------------------------------------

    # Results

    # --------------------------------------------------------



    print()

    print("=" * 90)

    print("EVALUATION RESULTS")
    print("All Kalman methods use the same 5-frame median preprocessing.")
    print("Matched fixed-Q and adaptive variants share base Q=25, R=100, P0=100; dt=1 frame.")

    print("=" * 90)



    print_method(

        "Baseline: L2CS + Ridge (gaze only)",

        results,

        "baseline_x",

        "baseline_y",

    )



    print_method(

        "Method 2: L2CS + Head Yaw/Pitch + Ridge",

        results,

        "ridge_x",

        "ridge_y",

    )



    print_method(
        "Paper-style Fixed Kalman",
        results,
        "fixed_kf_x",
        "fixed_kf_y",
    )



    print_method(
        "Matched fixed-Q Kalman (Q=25, R=100, P0=100)",
        results,
        "matched_fixed_kf_x",
        "matched_fixed_kf_y",
    )

    print_method(

        "Adaptive Q Kalman",

        results,

        "adaptive_q_x",

        "adaptive_q_y",

    )



    print_method(

        "Adaptive R Kalman",

        results,

        "adaptive_r_x",

        "adaptive_r_y",

    )



    print_method(

        "Proposed: Adaptive Q + R",

        results,

        "adaptive_qr_x",

        "adaptive_qr_y",

    )



    # --------------------------------------------------------

    # Save

    # --------------------------------------------------------



    output_file = os.path.join(

        os.path.dirname(EVALUATION_FILE),

        "final_evaluation_results.csv",

    )



    save_results(

        results,

        output_file,

    )



    print()

    print("=" * 90)

    print("SAVED")

    print("=" * 90)

    print(output_file)



    # --------------------------------------------------------

    # Pipeline

    # --------------------------------------------------------



    print()

    print("=" * 90)

    print("PIPELINE")

    print("=" * 90)



    print()

    print("Baseline:")

    print("  L2CS -> Ridge(gaze only) -> screen")



    print()

    print("Method 2:")

    print("  L2CS + Head Yaw/Pitch -> Ridge -> screen")



    print()

    print("Paper-style Fixed KF:")

    print("  Method 2 Ridge -> 5-frame moving median -> paper Fixed KF -> screen")
    print("  Timing: recorded numeric timestamps in seconds when available")
    print("  Fallback: frame-index timing if timestamps are unavailable")
    print("  P0 = 100 I is an implementation choice; the paper does not specify P0")



    print()

    print("Adaptive Q:")

    print("  Method 2 Ridge -> same 5-frame moving median -> Adaptive Q KF -> screen")

    print("  Q adapts using head yaw/pitch motion")



    print()

    print("Adaptive R:")

    print("  Method 2 Ridge -> same 5-frame moving median -> Adaptive R KF -> screen")

    print("  R adapts using L2CS confidence")



    print()

    print("Proposed:")

    print("  Method 2 Ridge -> same 5-frame moving median -> Adaptive Q+R KF -> screen")

    print("  Q: head yaw/pitch motion")

    print("  R: L2CS confidence")





if __name__ == "__main__":

    main()