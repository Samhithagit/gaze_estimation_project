import os

import sys

import csv

import numpy as np

from sklearn.linear_model import Ridge



from ridge_gaze_mapper import RidgeGazeMapper
from extra_trees_gaze_mapper import ExtraTreesGazeMapper



from module3.gaze_kalman import (
    FixedKalmanGaze,
    MovingMedianGaze,
    AdaptiveQKalmanGaze,
    AdaptiveRKalmanGaze,
    AdaptiveQRKalmanGaze,
    DoubleExponentialSmoothingGaze,
    ConfidenceAdaptiveEMAGaze,
    InnovationClippedHoltGaze,
    AdaptiveAlphaBetaGaze,
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



    # Dedicated Extra Trees mapper. Uses the same four features as Method 2;
    # head roll is intentionally excluded.
    regression_mappers = {}
    et_mapper = ExtraTreesGazeMapper()
    et_mapper.fit(
        gaze_yaw=yaw,
        gaze_pitch=pitch,
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        target_x=target_x,
        target_y=target_y,
    )
    regression_mappers["extra_trees"] = et_mapper

    return baseline_mapper, method2_mapper, regression_mappers





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

    regression_mappers,

):

    paper_fixed, matched_fixed, adaptive_q, adaptive_r, proposed = create_filters()
    median_filter = MovingMedianGaze(window_size=MEDIAN_WINDOW_SIZE)

    # Additional methods. Existing baseline, Method 2, and Fixed KF logic
    # remains copied verbatim from the originally supplied evaluator.
    double_exp = DoubleExponentialSmoothingGaze(
        alpha=0.28, beta=0.045, max_trend_step=35.0
    )
    adaptive_ab = AdaptiveAlphaBetaGaze(
        alpha_min=0.08, alpha_max=0.55, beta=0.045,
        confidence_power=1.8, outlier_scale=65.0, max_speed=45.0
    )
    adaptive_ema = ConfidenceAdaptiveEMAGaze(
        min_alpha=0.12, max_alpha=0.52, confidence_power=1.5,
        movement_scale=35.0, max_step=45.0
    )
    robust_holt = InnovationClippedHoltGaze(
        alpha=0.16,
        beta=0.01,
        innovation_limit=28.0,
        max_trend_step=22.0,
    )

    # ------------------------------------------------------------
    # Independent copy of the existing filter chain for Extra Trees.
    # This does not change or share state with the original Ridge chain.
    # ------------------------------------------------------------
    et_paper_fixed, et_matched_fixed, et_adaptive_q, et_adaptive_r, et_proposed = create_filters()
    et_median_filter = MovingMedianGaze(window_size=MEDIAN_WINDOW_SIZE)
    et_double_exp = DoubleExponentialSmoothingGaze(
        alpha=0.28, beta=0.045, max_trend_step=35.0
    )
    et_adaptive_ab = AdaptiveAlphaBetaGaze(
        alpha_min=0.08, alpha_max=0.55, beta=0.045,
        confidence_power=1.8, outlier_scale=65.0, max_speed=45.0
    )
    et_adaptive_ema = ConfidenceAdaptiveEMAGaze(
        min_alpha=0.12, max_alpha=0.52, confidence_power=1.5,
        movement_scale=35.0, max_step=45.0
    )
    et_robust_holt = InnovationClippedHoltGaze(
        alpha=0.16, beta=0.01, innovation_limit=28.0, max_trend_step=22.0
    )

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
            double_exp.reset()
            adaptive_ab.reset()
            adaptive_ema.reset()
            robust_holt.reset()
            # Reset the independent Extra Trees filter chain at the same target boundary.
            et_median_filter.reset()
            et_paper_fixed.reset()
            et_matched_fixed.reset()
            et_adaptive_q.reset()
            et_adaptive_r.reset()
            et_proposed.reset()
            et_double_exp.reset()
            et_adaptive_ab.reset()
            et_adaptive_ema.reset()
            et_robust_holt.reset()
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

        # New methods consume the same median-filtered Method 2 coordinates.
        des_x, des_y = double_exp.update(median_x, median_y)
        ab_x, ab_y = adaptive_ab.update(
            median_x, median_y, confidence=row["raw_confidence"],
            head_motion=head_motion
        )
        ema_x, ema_y = adaptive_ema.update(
            median_x, median_y, confidence=row["raw_confidence"]
        )
        robust_holt_x, robust_holt_y = robust_holt.update(median_x, median_y)

        # Candidate regressors are evaluated as raw mapping outputs first.
        # This isolates regression quality before choosing a model for filtering.
        regression_predictions = {}
        for model_name, mapper in regression_mappers.items():
            pred_x, pred_y = mapper.predict(
                [row["raw_gaze_yaw"]],
                [row["raw_gaze_pitch"]],
                head_yaw=[row["head_yaw"]],
                head_pitch=[row["head_pitch"]],
            )
            regression_predictions[f"reg_{model_name}_x"] = float(pred_x[0])
            regression_predictions[f"reg_{model_name}_y"] = float(pred_y[0])

        # ------------------------------------------------------------
        # Full-pipeline evaluation for Extra Trees.
        # Use its own predictions, median buffer, and filter states.
        # Every filter uses the same parameters as the Ridge pipeline.
        # ------------------------------------------------------------
        et_raw_x = regression_predictions["reg_extra_trees_x"]
        et_raw_y = regression_predictions["reg_extra_trees_y"]
        et_median_x, et_median_y = et_median_filter.update(et_raw_x, et_raw_y)

        et_paper_result = et_paper_fixed.update(
            et_median_x, et_median_y, timestamp=filter_timestamp
        )
        et_matched_result = et_matched_fixed.update(
            et_median_x, et_median_y, timestamp=filter_timestamp
        )
        et_fixed_x, et_fixed_y = map(float, et_paper_result)
        et_matched_x, et_matched_y = map(float, et_matched_result)

        et_q_result = et_adaptive_q.update(
            et_median_x, et_median_y, head_motion=head_motion
        )
        et_q_x, et_q_y = float(et_q_result["x"]), float(et_q_result["y"])

        et_r_result = et_adaptive_r.update(
            et_median_x, et_median_y, confidence=row["raw_confidence"]
        )
        et_r_x, et_r_y = float(et_r_result["x"]), float(et_r_result["y"])

        et_proposed_result = et_proposed.update(
            et_median_x, et_median_y,
            confidence=row["raw_confidence"], head_motion=head_motion
        )
        et_proposed_x = float(et_proposed_result["x"])
        et_proposed_y = float(et_proposed_result["y"])

        et_des_x, et_des_y = et_double_exp.update(et_median_x, et_median_y)
        et_ab_x, et_ab_y = et_adaptive_ab.update(
            et_median_x, et_median_y, confidence=row["raw_confidence"],
            head_motion=head_motion
        )
        et_ema_x, et_ema_y = et_adaptive_ema.update(
            et_median_x, et_median_y, confidence=row["raw_confidence"]
        )
        et_holt_x, et_holt_y = et_robust_holt.update(et_median_x, et_median_y)

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

            "double_exp_x": des_x,
            "double_exp_y": des_y,
            "adaptive_ab_x": ab_x,
            "adaptive_ab_y": ab_y,
            "adaptive_ema_x": ema_x,
            "adaptive_ema_y": ema_y,
            "robust_holt_x": robust_holt_x,
            "robust_holt_y": robust_holt_y,

            # Extra Trees full-pipeline predictions (independent filter states).
            "et_raw_x": et_raw_x,
            "et_raw_y": et_raw_y,
            "et_median_x": float(et_median_x),
            "et_median_y": float(et_median_y),
            "et_fixed_kf_x": et_fixed_x,
            "et_fixed_kf_y": et_fixed_y,
            "et_matched_fixed_kf_x": et_matched_x,
            "et_matched_fixed_kf_y": et_matched_y,
            "et_adaptive_q_x": et_q_x,
            "et_adaptive_q_y": et_q_y,
            "et_adaptive_r_x": et_r_x,
            "et_adaptive_r_y": et_r_y,
            "et_adaptive_qr_x": et_proposed_x,
            "et_adaptive_qr_y": et_proposed_y,
            "et_double_exp_x": float(et_des_x),
            "et_double_exp_y": float(et_des_y),
            "et_adaptive_ab_x": float(et_ab_x),
            "et_adaptive_ab_y": float(et_ab_y),
            "et_adaptive_ema_x": float(et_ema_x),
            "et_adaptive_ema_y": float(et_ema_y),
            "et_robust_holt_x": float(et_holt_x),
            "et_robust_holt_y": float(et_holt_y),

            "confidence": row["raw_confidence"],

            "head_motion": head_motion,

        } | regression_predictions)



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



    fields = list(results[0].keys()) if results else []

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

    print("FINAL GAZE EVALUATION — RIDGE + EXTRA TREES (NO HEAD ROLL)")

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
    print("Fitting Extra Trees pipeline on calibration data only...")

    baseline_mapper, method2_mapper, regression_mappers = fit_mappers(

        calibration

    )



    print()

    print("Extra Trees fitting complete.")



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

        regression_mappers,

    )



    # --------------------------------------------------------

    # Results

    # --------------------------------------------------------



    print()
    print("=" * 90)
    print("INTEGRATED RIDGE + EXTRA TREES PIPELINE RESULTS")
    print("Ridge and Extra Trees both use gaze yaw/pitch + head yaw/pitch; head roll is excluded.")
    print("All filters use independent states and the existing filter parameters.")
    print("=" * 90)

    for label, x_key, y_key in [
        ("Baseline", "baseline_x", "baseline_y"),
        ("Method 2 (Ridge)", "ridge_x", "ridge_y"),
        ("Extra Trees method", "et_raw_x", "et_raw_y"),
        ("Fixed KF + Ridge", "fixed_kf_x", "fixed_kf_y"),
        ("Fixed KF + Extra Trees", "et_fixed_kf_x", "et_fixed_kf_y"),
        ("Adaptive Q + Extra Trees", "et_adaptive_q_x", "et_adaptive_q_y"),
        ("Adaptive R + Extra Trees", "et_adaptive_r_x", "et_adaptive_r_y"),
        ("Extra Trees + Double Exp.", "et_double_exp_x", "et_double_exp_y"),
        ("Extra Trees + Clipped Holt", "et_robust_holt_x", "et_robust_holt_y"),
    ]:
        print_method(label, results, x_key, y_key)

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



    print()
    print("=" * 90)
    print("PIPELINE")
    print("=" * 90)
    print("L2CS gaze yaw/pitch + head yaw/pitch -> Extra Trees -> screen coordinates")
    print("Post-processing comparisons: 5-frame median, Fixed KF, adaptive filters, and Holt/EMA methods")
    print("Head roll is excluded. Existing module3/gaze_kalman.py is not modified by this script.")


if __name__ == "__main__":

    main()
