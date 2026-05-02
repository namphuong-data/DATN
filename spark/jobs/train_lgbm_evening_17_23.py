"""
train_lgbm_evening_17_23.py — Evening specialist (17h–23h): Tweedie + quantile ensemble (v9).

- Main head: Tweedie (variance_power 1.25) — giữ MAE/WAPE tổng thể tốt như bản reg_v9.
- Aux head: quantile alpha 0.45 — kéo dự báo xuống ở tail / giờ muộn khi cần.
- Trên val: chọn trọng số w trong [0, 0.5] tối thiểu hóa WAPE trên blend (1-w)*pred_tw + w*pred_qt.
- Mỗi head có pipeline hậu xử lý riêng (calibration, weekend, residual tùy gate); Sun-late expert chỉ trên Tweedie.

Artifacts: lgbm_evening_model.pkl (Tweedie), lgbm_evening_model_quantile.pkl, postprocess_quantile trong postprocess_params,
ensemble_quantile_weight. Metrics slice: metrics_*_by_hour_dow, metrics_*_late_22_23_by_dow, zone tertile.
"""
from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import holidays as hols
import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from deltalake import DeltaTable
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_lgbm_evening_17_23_v9")

GOLD_SRC = "s3://lakehouse/gold/demand_by_zone"
MODEL_DIR = Path("/app/models") / f"lgbm_evening_17_23_ensemble_v9_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
STORAGE_OPTIONS = {
    "endpoint_url": "http://minio:9000",
    "access_key_id": "minioadmin",
    "secret_access_key": "minioadmin123",
    "region": "us-east-1",
    "allow_http": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
ZONE_DEMAND_MIN = 1.0
EVENING_HOURS = {17, 18, 19, 20, 21, 22, 23}

FEATURE_COLS = [
    "PULocationID", "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h",
    "rolling_avg_3h", "rolling_avg_6h", "rolling_avg_24h",
    "zone_hour_mean", "zone_mean", "hour_mean", "baseline_blend",
    "evening_slot", "is_h17", "is_h18", "is_h19", "is_h20", "is_h21", "is_h22", "is_h23",
    "is_evening_segment_early", "is_evening_segment_late", "evening_decay",
    "lag_ratio_1h_24h", "lag_diff_1h_24h", "rolling_ratio_3h_24h",
    "momentum_1h_2h", "momentum_1h_3h", "baseline_gap_1h", "baseline_gap_24h",
    "baseline_ratio_1h", "baseline_ratio_24h", "evening_baseline_gap",
    "decline_2h_to_1h", "decline_ratio_2h_to_1h", "late_decline_signal",
]

# Quantile aux: alpha < 0.5 → conservative vs mean (giảm over-predict).
QUANTILE_ALPHA_AUX = 0.45
QUANTILE_ALPHA_SUN = 0.40
ENSEMBLE_QUANT_W_GRID = np.linspace(0.0, 0.5, 21)

REG_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.25,
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": 10,
    "min_child_samples": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 3,
    "lambda_l1": 0.2,
    "lambda_l2": 0.8,
    "max_bin": 255,
    "num_threads": -1,
    "verbose": -1,
    "random_state": 42,
}
REG_QUANT_PARAMS = {
    "objective": "quantile",
    "alpha": QUANTILE_ALPHA_AUX,
    "metric": ["quantile", "mae"],
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": 10,
    "min_child_samples": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 3,
    "lambda_l1": 0.2,
    "lambda_l2": 0.8,
    "max_bin": 255,
    "num_threads": -1,
    "verbose": -1,
    "random_state": 43,
}
N_ESTIMATORS = 5000
EARLY_STOPPING = 150
LATE_HOURS = {22, 23}
WEEKEND_LATE_HOURS = {21, 22, 23}
SUN_LATE_DOW = 6
SUN_LATE_HOURS = {21, 22, 23}
SUN_EXPERT_MIN_TRAIN = 400
SUN_EXPERT_MIN_VAL = 80
N_ESTIMATORS_SUN = 4000
EARLY_STOPPING_SUN = 120
SUN_EXPERT_PARAMS = {
    "objective": "quantile",
    "alpha": QUANTILE_ALPHA_SUN,
    "metric": ["quantile", "mae"],
    "learning_rate": 0.03,
    "num_leaves": 96,
    "max_depth": 10,
    "min_child_samples": 80,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 3,
    "lambda_l1": 0.2,
    "lambda_l2": 0.8,
    "max_bin": 255,
    "num_threads": -1,
    "verbose": -1,
    "random_state": 44,
}
RESIDUAL_N_ESTIMATORS = 1200
RESIDUAL_EARLY_STOPPING = 80
RESIDUAL_PARAMS = {
    "objective": "regression_l1",
    "metric": ["l1", "rmse"],
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 120,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.75,
    "bagging_freq": 2,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
    "min_gain_to_split": 0.05,
    "max_bin": 255,
    "num_threads": -1,
    "verbose": -1,
    "random_state": 42,
}


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)) * 100)


def wape_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    num = float(np.sum(np.abs(y_true - y_pred)))
    den = float(max(np.sum(y_true), 1.0))
    return 100.0 * num / den


DOW_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def compute_segment_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics for a slice (no logging). Adds rates useful for diagnosing systematic bias."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64).reshape(-1), 0.0, None)
    n = int(len(y_true))
    if n == 0:
        return {
            "n": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
            "MAPE": float("nan"),
            "WAPE": float("nan"),
            "SMAPE": float("nan"),
            "over_pred_rate": float("nan"),
            "under_pred_rate": float("nan"),
            "equal_pred_rate": float("nan"),
            "mean_signed_error": float("nan"),
        }
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")
    wape = float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1.0) * 100)
    mape = float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1.0)) * 100)
    s = smape(y_true, y_pred)
    over = float(np.mean(y_pred > y_true))
    under = float(np.mean(y_pred < y_true))
    eq = float(np.mean(y_pred == y_true))
    bias = float(np.mean(y_pred - y_true))
    return {
        "n": n,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "WAPE": wape,
        "SMAPE": s,
        "over_pred_rate": over,
        "under_pred_rate": under,
        "equal_pred_rate": eq,
        "mean_signed_error": bias,
    }


def _save_metrics_hour_dow(diag: pd.DataFrame, path: Path, split_label: str) -> None:
    rows = []
    for (h, dow), g in diag.groupby(["hour", "dow"], sort=True):
        st = compute_segment_stats(g["actual"].values, g["predicted"].values)
        rows.append({
            "split": split_label,
            "hour": int(h),
            "dow": int(dow),
            "dow_name": DOW_NAMES.get(int(dow), str(int(dow))),
            **st,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_metrics_late_22_23_by_dow(diag: pd.DataFrame, path: Path, split_label: str) -> None:
    sub = diag[diag["hour"].isin([22, 23])].copy()
    rows = []
    for dow, g in sub.groupby("dow", sort=True):
        st = compute_segment_stats(g["actual"].values, g["predicted"].values)
        rows.append({"split": split_label, "dow": int(dow), "dow_name": DOW_NAMES.get(int(dow), str(int(dow))), **st})
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_metrics_late_zone_tertiles(diag: pd.DataFrame, path: Path, split_label: str) -> None:
    sub = diag[diag["hour"].isin([22, 23])].copy()
    if sub.empty or len(sub) < 30:
        pd.DataFrame([{"split": split_label, "note": "insufficient_rows_for_tertile"}]).to_csv(path, index=False)
        return
    z = sub["zone_hour_mean"].astype(float)
    try:
        sub = sub.copy()
        sub["zone_vol_tertile"] = pd.qcut(z, q=3, labels=["low", "mid", "high"], duplicates="drop")
    except Exception:
        pd.DataFrame([{"split": split_label, "note": "qcut_failed"}]).to_csv(path, index=False)
        return
    rows = []
    for t, g in sub.groupby("zone_vol_tertile", sort=True):
        st = compute_segment_stats(g["actual"].values, g["predicted"].values)
        rows.append({"split": split_label, "zone_vol_tertile": str(t), **st})
    pd.DataFrame(rows).to_csv(path, index=False)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    wape = float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1.0) * 100)
    mape = float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1.0)) * 100)
    s = smape(y_true, y_pred)
    log.info("[%s] MAE=%.3f RMSE=%.3f R2=%.4f MAPE=%.2f%% WAPE=%.2f%% SMAPE=%.2f%%", label, mae, rmse, r2, mape, wape, s)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "WAPE": wape, "SMAPE": s}


def load_preprocess() -> tuple[pd.DataFrame, list[str]]:
    dt = DeltaTable(GOLD_SRC, storage_options=STORAGE_OPTIONS)
    df = dt.to_pandas()
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True).dt.tz_convert(None)
    if "target_demand" not in df.columns and "trip_count" in df.columns:
        df["target_demand"] = df["trip_count"]
    df["target_demand"] = pd.to_numeric(df["target_demand"], errors="coerce").fillna(0.0)
    df["hour"] = df["window_start"].dt.hour.astype(int)
    df["day_of_week"] = df["window_start"].dt.dayofweek.astype(int)
    df["month"] = df["window_start"].dt.month.astype(int)
    zone_means = df.groupby("PULocationID")["target_demand"].mean()
    active = zone_means[zone_means >= ZONE_DEMAND_MIN].index
    df = df[df["PULocationID"].isin(active)].copy().sort_values(["PULocationID", "window_start"]).reset_index(drop=True)
    g = df.groupby("PULocationID")["target_demand"]
    for c, s in {"lag_1h": 1, "lag_2h": 2, "lag_3h": 3, "lag_24h": 24, "lag_168h": 168}.items():
        df[c] = g.shift(s)
    for c, w in {"rolling_avg_3h": 3, "rolling_avg_6h": 6, "rolling_avg_24h": 24}.items():
        df[c] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    lag_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("rolling_avg_")]
    df[lag_cols] = df[lag_cols].fillna(0.0)
    years = df["window_start"].dt.year.unique().tolist()
    ny_cal = hols.country_holidays("US", subdiv="NY", years=years)
    hol_dates = {pd.Timestamp(d).normalize() for d in ny_cal.keys()}
    df["is_holiday"] = df["window_start"].dt.normalize().isin(hol_dates).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["zone_mean"] = df.groupby("PULocationID")["target_demand"].transform(lambda x: x.expanding().mean().shift(1))
    df["zone_hour_mean"] = df.groupby(["PULocationID", "hour"])["target_demand"].transform(lambda x: x.expanding().mean().shift(1))
    global_mean = float(df["target_demand"].mean())
    df["zone_mean"] = df["zone_mean"].fillna(global_mean)
    df["zone_hour_mean"] = df["zone_hour_mean"].fillna(df["zone_mean"])
    df["hour_mean"] = df.groupby("hour")["target_demand"].transform(lambda x: x.expanding().mean().shift(1)).fillna(global_mean)
    df["baseline_blend"] = 0.70 * df["zone_hour_mean"] + 0.30 * df["hour_mean"]
    for hh in range(17, 24):
        df[f"is_h{hh}"] = (df["hour"] == hh).astype(int)
    df["evening_slot"] = np.where(df["hour"].between(17, 23), df["hour"] - 17, -1).astype(int)
    df["is_evening_segment_early"] = df["hour"].between(17, 19).astype(int)
    df["is_evening_segment_late"] = df["hour"].between(21, 23).astype(int)
    df["evening_decay"] = np.where(df["hour"].between(17, 23), (23 - df["hour"]) / 6.0, 0.0)
    df["lag_ratio_1h_24h"] = (df["lag_1h"] + 1.0) / (df["lag_24h"] + 1.0)
    df["lag_diff_1h_24h"] = df["lag_1h"] - df["lag_24h"]
    df["rolling_ratio_3h_24h"] = (df["rolling_avg_3h"] + 1.0) / (df["rolling_avg_24h"] + 1.0)
    df["momentum_1h_2h"] = df["lag_1h"] - df["lag_2h"]
    df["momentum_1h_3h"] = df["lag_1h"] - df["lag_3h"]
    df["baseline_gap_1h"] = df["lag_1h"] - df["baseline_blend"]
    df["baseline_gap_24h"] = df["lag_24h"] - df["baseline_blend"]
    df["baseline_ratio_1h"] = (df["lag_1h"] + 1.0) / (df["baseline_blend"] + 1.0)
    df["baseline_ratio_24h"] = (df["lag_24h"] + 1.0) / (df["baseline_blend"] + 1.0)
    df["evening_baseline_gap"] = df["baseline_gap_1h"] * df["is_evening_segment_late"]
    df["decline_2h_to_1h"] = np.maximum(df["lag_2h"] - df["lag_1h"], 0.0)
    df["decline_ratio_2h_to_1h"] = df["decline_2h_to_1h"] / (df["lag_2h"] + 1.0)
    df["late_decline_signal"] = df["decline_ratio_2h_to_1h"] * df["is_evening_segment_late"]
    df = df[df["hour"].isin(EVENING_HOURS)].dropna(subset=["target_demand"]).reset_index(drop=True)
    active_cols = [c for c in FEATURE_COLS if c in df.columns]
    return df, active_cols


def fit_evening_calibration(
    y_true: np.ndarray, y_pred: np.ndarray, hour_arr: np.ndarray, dow_arr: np.ndarray
) -> tuple[dict[int, float], dict[str, float], dict[int, float]]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    hour_arr = np.asarray(hour_arr, dtype=np.int64).reshape(-1)
    dow_arr = np.asarray(dow_arr, dtype=np.int64).reshape(-1)

    hour_factors: dict[int, float] = {}
    for h in sorted(EVENING_HOURS):
        m = hour_arr == h
        if not np.any(m):
            hour_factors[h] = 1.0
            continue
        p = float(np.sum(y_pred[m]))
        t = float(np.sum(y_true[m]))
        if p <= 1e-6 or t <= 1e-6:
            hour_factors[h] = 1.0
        else:
            hour_factors[h] = float(np.clip(t / p, 0.75, 1.25))

    dow_hour_factors: dict[str, float] = {}
    for dow in range(7):
        for h in sorted(EVENING_HOURS):
            m = (dow_arr == dow) & (hour_arr == h)
            if not np.any(m):
                continue
            p = float(np.sum(y_pred[m]))
            t = float(np.sum(y_true[m]))
            if p <= 1e-6 or t <= 1e-6:
                f = 1.0
            else:
                f = float(np.clip(t / p, 0.70, 1.30))
            dow_hour_factors[f"{dow}_{h}"] = f

    late_bias: dict[int, float] = {}
    for h in sorted(EVENING_HOURS):
        if h not in LATE_HOURS:
            late_bias[h] = 0.0
            continue
        m = hour_arr == h
        if not np.any(m):
            late_bias[h] = 0.0
            continue
        # Tune affine correction for late-evening (21-23) to reduce overprediction drift.
        y_t = y_true[m]
        y_p = y_pred[m]
        best_wape = np.inf
        best_factor = hour_factors.get(h, 1.0)
        best_bias = 0.0
        for f in np.linspace(0.55, 1.15, 25):
            yp_f = y_p * f
            for b in np.linspace(-15.0, 8.0, 24):
                yp_fb = np.clip(yp_f + b, 0.0, None)
                wape = wape_score(y_t, yp_fb)
                if wape < best_wape:
                    best_wape = wape
                    best_factor = float(f)
                    best_bias = float(b)
        hour_factors[h] = best_factor
        late_bias[h] = best_bias

    return hour_factors, dow_hour_factors, late_bias


def apply_evening_calibration(
    y_pred: np.ndarray,
    hour_arr: np.ndarray,
    dow_arr: np.ndarray,
    hour_factors: dict[int, float],
    dow_hour_factors: dict[str, float],
    late_bias: dict[int, float],
) -> np.ndarray:
    y = np.asarray(y_pred, dtype=np.float64).copy()
    hour_arr = np.asarray(hour_arr, dtype=np.int64).reshape(-1)
    dow_arr = np.asarray(dow_arr, dtype=np.int64).reshape(-1)
    for i in range(len(y)):
        h = int(hour_arr[i])
        d = int(dow_arr[i])
        k = f"{d}_{h}"
        f = float(dow_hour_factors.get(k, hour_factors.get(h, 1.0)))
        b = float(late_bias.get(h, 0.0))
        y[i] = max(y[i] * f + b, 0.0)
    return y


def _weekend_adjust_score(y_t: np.ndarray, y_pred_adj: np.ndarray, label: str, hour: int) -> float:
    """WAPE + penalties aligned with segment logs (Sat underpredict vs Sun 23 over-rate)."""
    wape = wape_score(y_t, y_pred_adj)
    bias = float(np.mean(y_pred_adj - y_t))
    over = float(np.mean(y_pred_adj > y_t))
    under = float(np.mean(y_pred_adj < y_t))
    score = wape
    if label == "sat" and hour in (22, 23):
        score += 0.55 * max(0.0, -bias)
        score += 8.0 * max(0.0, under - 0.55)
    elif label == "sat" and hour == 21:
        score += 0.3 * max(0.0, -bias)
    elif label == "sun" and hour == 23:
        score += 0.6 * max(0.0, bias)
        score += 20.0 * max(0.0, over - 0.48)
    elif label == "sun" and hour in (21, 22):
        score += 0.4 * max(0.0, -bias)
        score += 0.25 * max(0.0, bias)
    return score


def fit_weekend_late_adjustment(
    y_true: np.ndarray, y_pred: np.ndarray, hour_arr: np.ndarray, dow_arr: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit separate late-hour adjustment for Sat/Sun and each hour (21, 22, 23)."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    hour_arr = np.asarray(hour_arr, dtype=np.int64).reshape(-1)
    dow_arr = np.asarray(dow_arr, dtype=np.int64).reshape(-1)
    factors = {"sat_21": 1.0, "sat_22": 1.0, "sat_23": 1.0, "sun_21": 1.0, "sun_22": 1.0, "sun_23": 1.0}
    biases = {"sat_21": 0.0, "sat_22": 0.0, "sat_23": 0.0, "sun_21": 0.0, "sun_22": 0.0, "sun_23": 0.0}
    for label, dow in (("sat", 5), ("sun", 6)):
        for h in sorted(WEEKEND_LATE_HOURS):
            key = f"{label}_{h}"
            m = (dow_arr == dow) & (hour_arr == h)
            n = int(np.sum(m))
            if n == 0:
                continue
            y_t = y_true[m]
            y_p = y_pred[m]
            best_score = np.inf
            best_wape = np.inf
            best_f = 1.0
            best_b = 0.0
            if label == "sat" and h in (22, 23):
                f_min, f_max = 0.88, 1.22
                b_min, b_max = -1.25, 12.0
            elif label == "sat":
                f_min, f_max = 0.85, 1.15
                b_min, b_max = (-10.0, 10.0)
            elif label == "sun" and h == 23:
                f_min, f_max = 0.66, 1.00
                b_min, b_max = (-8.0, 6.0)
            else:
                f_min, f_max = (0.72, 1.00)
                b_min, b_max = (-14.0, 4.0)
            # Shrink search span when samples are limited.
            shrink = min(1.0, n / 300.0)
            f_lo = 1.0 - (1.0 - f_min) * shrink
            f_hi = 1.0 + (f_max - 1.0) * shrink
            b_lo = b_min * shrink
            b_hi = b_max * shrink
            for f in np.linspace(f_lo, f_hi, 17):
                yp_f = y_p * float(f)
                for b in np.linspace(b_lo, b_hi, 17):
                    yp_fb = np.clip(yp_f + float(b), 0.0, None)
                    sc = _weekend_adjust_score(y_t, yp_fb, label, h)
                    wape = wape_score(y_t, yp_fb)
                    if sc < best_score - 1e-12:
                        best_score = sc
                        best_wape = wape
                        best_f = float(f)
                        best_b = float(b)
                    elif abs(sc - best_score) <= 1e-12 and wape < best_wape - 1e-12:
                        best_wape = wape
                        best_f = float(f)
                        best_b = float(b)
            factors[key] = best_f
            biases[key] = best_b
    return factors, biases


def apply_weekend_late_adjustment(
    y_pred: np.ndarray,
    hour_arr: np.ndarray,
    dow_arr: np.ndarray,
    factors: dict[str, float] | None,
    biases: dict[str, float] | None,
) -> np.ndarray:
    y = np.asarray(y_pred, dtype=np.float64).copy()
    hour_arr = np.asarray(hour_arr, dtype=np.int64).reshape(-1)
    dow_arr = np.asarray(dow_arr, dtype=np.int64).reshape(-1)
    f = factors or {}
    b = biases or {}
    for i in range(len(y)):
        h = int(hour_arr[i])
        d = int(dow_arr[i])
        if h not in WEEKEND_LATE_HOURS:
            continue
        if d == 5:
            k = f"sat_{h}"
            y[i] = max(y[i] * float(f.get(k, 1.0)) + float(b.get(k, 0.0)), 0.0)
        elif d == 6:
            k = f"sun_{h}"
            y[i] = max(y[i] * float(f.get(k, 1.0)) + float(b.get(k, 0.0)), 0.0)
    return y


def fit_late_hour_residual_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    pred_train: np.ndarray,
    h_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    pred_val: np.ndarray,
    h_val: np.ndarray,
    feat_cols: list[str],
) -> tuple[lgb.Booster | None, list[str]]:
    late_train = np.isin(h_train, sorted(LATE_HOURS))
    late_val = np.isin(h_val, sorted(LATE_HOURS))
    if int(np.sum(late_train)) < 500 or int(np.sum(late_val)) < 150:
        log.warning("[RES-LATE] Skip residual stage due to insufficient samples")
        return None, []

    xtr = X_train.loc[late_train, feat_cols].copy()
    xva = X_val.loc[late_val, feat_cols].copy()
    xtr["base_pred_cal"] = pred_train[late_train]
    xva["base_pred_cal"] = pred_val[late_val]
    residual_cols = feat_cols + ["base_pred_cal"]
    ytr_res = (y_train[late_train] - pred_train[late_train]).astype(float)
    yva_res = (y_val[late_val] - pred_val[late_val]).astype(float)
    wtr = np.ones_like(ytr_res, dtype=np.float64)
    # Emphasize larger late-hour demand where aggregate error matters most.
    wtr *= np.clip(y_train[late_train] / 8.0, 0.8, 3.0)

    dtr = lgb.Dataset(xtr[residual_cols], label=ytr_res, weight=wtr, free_raw_data=False)
    dva = lgb.Dataset(xva[residual_cols], label=yva_res, reference=dtr, free_raw_data=False)
    res_model = lgb.train(
        RESIDUAL_PARAMS,
        dtr,
        num_boost_round=RESIDUAL_N_ESTIMATORS,
        valid_sets=[dtr, dva],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(RESIDUAL_EARLY_STOPPING, verbose=True), lgb.log_evaluation(100)],
    )
    return res_model, residual_cols


def apply_late_hour_residual(
    base_pred: np.ndarray,
    X: pd.DataFrame,
    h_arr: np.ndarray,
    residual_model: lgb.Booster | None,
    residual_cols: list[str],
) -> np.ndarray:
    if residual_model is None or not residual_cols:
        return np.asarray(base_pred, dtype=np.float64)
    y = np.asarray(base_pred, dtype=np.float64).copy()
    late_mask = np.isin(np.asarray(h_arr, dtype=np.int64), sorted(LATE_HOURS))
    if not np.any(late_mask):
        return y
    x_late = X.loc[late_mask, [c for c in residual_cols if c != "base_pred_cal"]].copy()
    x_late["base_pred_cal"] = y[late_mask]
    delta = np.asarray(residual_model.predict(x_late[residual_cols]), dtype=np.float64).reshape(-1)
    # Guardrail: keep residual correction conservative to avoid destabilizing late-hour predictions.
    max_abs = np.maximum(0.20 * y[late_mask], 3.0)
    delta = np.clip(delta, -max_abs, max_abs)
    delta = np.clip(delta, -8.0, 8.0)
    y[late_mask] = np.clip(y[late_mask] + delta, 0.0, None)
    return y


def _branch_state_to_post_dict(br: dict) -> dict:
    res_model = br["residual_model"]
    sun_m = br["sun_late_expert_model"]
    return {
        "blend_alpha_late": 0.10,
        "cap_mult_evening": 5.0,
        "cap_bias_evening": 6.0,
        "cap_mult_late": 4.0,
        "cap_bias_late": 6.0,
        "hour_calibration": br["hour_factors"],
        "dow_hour_calibration": br["dow_hour_factors"],
        "late_hour_bias": br["late_bias"],
        "weekend_adjust_enabled": br["weekend_adjust_enabled"],
        "weekend_late_factors": br["weekend_late_factors"],
        "weekend_late_bias": br["weekend_late_bias"],
        "late_hour_residual_enabled": res_model is not None,
        "late_hour_residual_hours": sorted(list(LATE_HOURS)),
        "sun_late_expert_enabled": sun_m is not None and float(br["sun_blend_alpha"]) > 1e-8,
        "sun_late_expert_blend_alpha": float(br["sun_blend_alpha"]),
        "sun_late_expert_hours": sorted(list(SUN_LATE_HOURS)),
    }


def _train_evening_branch(
    model: lgb.Booster,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    h_train: np.ndarray,
    h_val: np.ndarray,
    h_test: np.ndarray,
    d_train: np.ndarray,
    d_val: np.ndarray,
    d_test: np.ndarray,
    feat_cols: list[str],
    *,
    apply_sun_expert: bool,
    tag: str,
) -> dict:
    bi = int(model.best_iteration or 0)
    y_val_pred = np.clip(np.asarray(model.predict(X_val, num_iteration=bi)).reshape(-1), 0.0, None)
    y_test_pred = np.clip(np.asarray(model.predict(X_test, num_iteration=bi)).reshape(-1), 0.0, None)
    y_train_pred = np.clip(np.asarray(model.predict(X_train, num_iteration=bi)).reshape(-1), 0.0, None)
    hour_factors, dow_hour_factors, late_bias = fit_evening_calibration(y_val, y_val_pred, h_val, d_val)
    y_val_cal = apply_evening_calibration(y_val_pred, h_val, d_val, hour_factors, dow_hour_factors, late_bias)
    y_test_cal = apply_evening_calibration(y_test_pred, h_test, d_test, hour_factors, dow_hour_factors, late_bias)
    y_train_cal = apply_evening_calibration(y_train_pred, h_train, d_train, hour_factors, dow_hour_factors, late_bias)
    weekend_late_factors, weekend_late_bias = fit_weekend_late_adjustment(y_val, y_val_cal, h_val, d_val)
    y_val_weekend = apply_weekend_late_adjustment(y_val_cal, h_val, d_val, weekend_late_factors, weekend_late_bias)
    weekend_val_mask = np.isin(h_val, [21, 22, 23]) & np.isin(d_val, [5, 6])
    val_weekend_wape_base = wape_score(y_val[weekend_val_mask], y_val_cal[weekend_val_mask]) if np.any(weekend_val_mask) else np.inf
    val_weekend_wape_adj = wape_score(y_val[weekend_val_mask], y_val_weekend[weekend_val_mask]) if np.any(weekend_val_mask) else np.inf
    if np.isfinite(val_weekend_wape_adj) and val_weekend_wape_adj < val_weekend_wape_base:
        y_train_cal = apply_weekend_late_adjustment(y_train_cal, h_train, d_train, weekend_late_factors, weekend_late_bias)
        y_val_cal = y_val_weekend
        y_test_cal = apply_weekend_late_adjustment(y_test_cal, h_test, d_test, weekend_late_factors, weekend_late_bias)
        weekend_adjust_enabled = True
    else:
        weekend_late_factors = {}
        weekend_late_bias = {}
        weekend_adjust_enabled = False
    residual_model_raw, residual_cols_raw = fit_late_hour_residual_model(
        X_train=X_train,
        y_train=y_train,
        pred_train=y_train_cal,
        h_train=h_train,
        X_val=X_val,
        y_val=y_val,
        pred_val=y_val_cal,
        h_val=h_val,
        feat_cols=feat_cols,
    )
    y_val_res = apply_late_hour_residual(y_val_cal, X_val, h_val, residual_model_raw, residual_cols_raw)
    late_val_mask = np.isin(h_val, sorted(LATE_HOURS))
    val_late_wape_base = wape_score(y_val[late_val_mask], y_val_cal[late_val_mask]) if np.any(late_val_mask) else np.inf
    val_late_wape_res = wape_score(y_val[late_val_mask], y_val_res[late_val_mask]) if np.any(late_val_mask) else np.inf
    residual_model = residual_model_raw
    residual_cols = residual_cols_raw
    if not np.isfinite(val_late_wape_res) or val_late_wape_res >= (0.995 * val_late_wape_base):
        log.info(
            "[%s][RES-LATE] disabled: no val late-hour WAPE gain (base=%.4f, res=%.4f)",
            tag,
            float(val_late_wape_base),
            float(val_late_wape_res),
        )
        residual_model = None
        residual_cols = []
        y_val_final = y_val_cal
        y_test_final = y_test_cal
    else:
        log.info(
            "[%s][RES-LATE] enabled: val late-hour WAPE improved (base=%.4f, res=%.4f)",
            tag,
            float(val_late_wape_base),
            float(val_late_wape_res),
        )
        y_val_final = y_val_res
        y_test_final = apply_late_hour_residual(y_test_cal, X_test, h_test, residual_model, residual_cols)
    log.info(
        "[%s][CALIB-EVE] weekend_adjust_enabled=%s res_enabled=%s",
        tag,
        weekend_adjust_enabled,
        bool(residual_model is not None),
    )

    sun_late_expert_model: lgb.Booster | None = None
    sun_blend_alpha = 0.0
    if apply_sun_expert:
        sun_mask_train = (d_train == SUN_LATE_DOW) & np.isin(h_train, sorted(SUN_LATE_HOURS))
        sun_mask_val = (d_val == SUN_LATE_DOW) & np.isin(h_val, sorted(SUN_LATE_HOURS))
        sun_mask_test = (d_test == SUN_LATE_DOW) & np.isin(h_test, sorted(SUN_LATE_HOURS))
        n_tr_sun = int(np.sum(sun_mask_train))
        n_va_sun = int(np.sum(sun_mask_val))
        if n_tr_sun >= SUN_EXPERT_MIN_TRAIN and n_va_sun >= SUN_EXPERT_MIN_VAL:
            ds_sun_tr = lgb.Dataset(
                X_train.loc[sun_mask_train, feat_cols],
                label=y_train[sun_mask_train],
                free_raw_data=False,
            )
            ds_sun_va = lgb.Dataset(
                X_val.loc[sun_mask_val, feat_cols],
                label=y_val[sun_mask_val],
                reference=ds_sun_tr,
                free_raw_data=False,
            )
            sun_late_expert_model = lgb.train(
                SUN_EXPERT_PARAMS,
                ds_sun_tr,
                num_boost_round=N_ESTIMATORS_SUN,
                valid_sets=[ds_sun_tr, ds_sun_va],
                valid_names=["train", "val"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_SUN, verbose=True), lgb.log_evaluation(200)],
            )
            bi_sun = int(sun_late_expert_model.best_iteration or 0)
            ev_val = np.clip(
                np.asarray(sun_late_expert_model.predict(X_val, num_iteration=bi_sun)).reshape(-1),
                0.0,
                None,
            )
            yvf_base = np.asarray(y_val_final, dtype=np.float64)
            best_a = 0.0
            best_wape_sun = np.inf
            for a in np.linspace(0.0, 1.0, 11):
                yv = yvf_base.copy()
                yv[sun_mask_val] = (1.0 - a) * yvf_base[sun_mask_val] + a * ev_val[sun_mask_val]
                w_sun = wape_score(y_val[sun_mask_val], yv[sun_mask_val])
                if w_sun < best_wape_sun - 1e-9:
                    best_wape_sun = w_sun
                    best_a = float(a)
            sun_blend_alpha = best_a
            log.info(
                "[%s][SUN-EXPERT] blend_alpha=%.3f val_WAPE_Sun_late=%.4f%%",
                tag,
                sun_blend_alpha,
                float(best_wape_sun),
            )
            if sun_blend_alpha > 1e-6:
                ev_test = np.clip(
                    np.asarray(sun_late_expert_model.predict(X_test, num_iteration=bi_sun)).reshape(-1),
                    0.0,
                    None,
                )
                y_val_final = yvf_base.copy()
                y_val_final[sun_mask_val] = (1.0 - sun_blend_alpha) * yvf_base[sun_mask_val] + sun_blend_alpha * ev_val[sun_mask_val]
                y_test_final = np.asarray(y_test_final, dtype=np.float64).copy()
                y_test_final[sun_mask_test] = (
                    (1.0 - sun_blend_alpha) * y_test_final[sun_mask_test] + sun_blend_alpha * ev_test[sun_mask_test]
                )
        else:
            log.info("[%s][SUN-EXPERT] skipped (train=%d val=%d)", tag, n_tr_sun, n_va_sun)

    return {
        "y_val_final": y_val_final,
        "y_test_final": y_test_final,
        "hour_factors": hour_factors,
        "dow_hour_factors": dow_hour_factors,
        "late_bias": late_bias,
        "weekend_late_factors": weekend_late_factors,
        "weekend_late_bias": weekend_late_bias,
        "weekend_adjust_enabled": weekend_adjust_enabled,
        "residual_model": residual_model,
        "residual_cols": residual_cols,
        "sun_late_expert_model": sun_late_expert_model,
        "sun_blend_alpha": sun_blend_alpha,
    }


def train() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df, feat_cols = load_preprocess()
    t = df["window_start"]
    t_min, t_max = t.min(), t.max()
    train_end = t_min + (t_max - t_min) * TRAIN_RATIO
    val_end = t_min + (t_max - t_min) * (TRAIN_RATIO + VAL_RATIO)
    m_train = t < train_end
    m_val = (t >= train_end) & (t < val_end)
    m_test = t >= val_end

    X_train = df.loc[m_train, feat_cols]
    y_train = df.loc[m_train, "target_demand"].astype(float).values
    h_train = df.loc[m_train, "hour"].astype(int).values
    X_val = df.loc[m_val, feat_cols]
    y_val = df.loc[m_val, "target_demand"].astype(float).values
    X_test = df.loc[m_test, feat_cols]
    y_test = df.loc[m_test, "target_demand"].astype(float).values
    h_val = df.loc[m_val, "hour"].astype(int).values
    h_test = df.loc[m_test, "hour"].astype(int).values
    d_val = df.loc[m_val, "day_of_week"].astype(int).values
    d_test = df.loc[m_test, "day_of_week"].astype(int).values
    # emphasize late evening
    w_train = np.ones_like(y_train, dtype=np.float64)
    w_train[np.isin(h_train, [21, 22, 23])] *= 1.15
    w_train[np.isin(h_train, [22, 23])] *= 1.10
    # weekend-aware emphasis for difficult late hours
    d_train = df.loc[m_train, "day_of_week"].astype(int).values
    weekend_late = np.isin(h_train, [21, 22, 23]) & np.isin(d_train, [5, 6])
    sunday_late = np.isin(h_train, [21, 22, 23]) & (d_train == 6)
    w_train[weekend_late] *= 1.15
    w_train[sunday_late] *= 1.20

    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    log.info("[ENSEMBLE] Training Tweedie main model...")
    model_tw = lgb.train(
        REG_PARAMS,
        dtrain,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=True), lgb.log_evaluation(100)],
    )
    br_tw = _train_evening_branch(
        model_tw,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        h_train,
        h_val,
        h_test,
        d_train,
        d_val,
        d_test,
        feat_cols,
        apply_sun_expert=True,
        tag="TW",
    )

    log.info("[ENSEMBLE] Training quantile auxiliary model...")
    model_qt = lgb.train(
        REG_QUANT_PARAMS,
        dtrain,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=True), lgb.log_evaluation(100)],
    )
    br_qt = _train_evening_branch(
        model_qt,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        h_train,
        h_val,
        h_test,
        d_train,
        d_val,
        d_test,
        feat_cols,
        apply_sun_expert=False,
        tag="QT",
    )

    best_w = 0.0
    best_wape = wape_score(y_val, br_tw["y_val_final"])
    for w in ENSEMBLE_QUANT_W_GRID:
        yv = (1.0 - w) * br_tw["y_val_final"] + w * br_qt["y_val_final"]
        wp = wape_score(y_val, yv)
        if wp < best_wape - 1e-9:
            best_wape = wp
            best_w = float(w)
    log.info(
        "[ENSEMBLE] ensemble_quantile_weight=%.4f val_WAPE=%.4f%% (Tweedie-only WAPE=%.4f%%)",
        best_w,
        best_wape,
        wape_score(y_val, br_tw["y_val_final"]),
    )

    y_val_final = (1.0 - best_w) * br_tw["y_val_final"] + best_w * br_qt["y_val_final"]
    y_test_final = (1.0 - best_w) * br_tw["y_test_final"] + best_w * br_qt["y_test_final"]

    val_metrics = compute_metrics(y_val, y_val_final, "VAL-EVENING-REG")
    test_metrics = compute_metrics(y_test, y_test_final, "TEST-EVENING-REG")

    diag = pd.DataFrame({
        "window_start": df.loc[m_test, "window_start"].values,
        "PULocationID": df.loc[m_test, "PULocationID"].values,
        "hour": df.loc[m_test, "hour"].values,
        "dow": df.loc[m_test, "day_of_week"].astype(int).values,
        "zone_hour_mean": df.loc[m_test, "zone_hour_mean"].astype(float).values,
        "actual": y_test,
        "predicted": y_test_final,
    })
    diag_val = pd.DataFrame({
        "hour": df.loc[m_val, "hour"].values,
        "dow": df.loc[m_val, "day_of_week"].astype(int).values,
        "zone_hour_mean": df.loc[m_val, "zone_hour_mean"].astype(float).values,
        "actual": y_val,
        "predicted": y_val_final,
    })
    by_hour = []
    for h, g in diag.groupby("hour"):
        by_hour.append({"hour": int(h), **compute_metrics(g["actual"].values, g["predicted"].values, f"TEST-H{int(h):02d}-REG")})
    pd.DataFrame(by_hour).to_csv(MODEL_DIR / "metrics_by_hour.csv", index=False)
    diag.to_csv(MODEL_DIR / "test_predictions.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(MODEL_DIR / "metrics.csv", index=False)

    _save_metrics_hour_dow(diag_val, MODEL_DIR / "metrics_val_by_hour_dow.csv", "val")
    _save_metrics_hour_dow(diag, MODEL_DIR / "metrics_test_by_hour_dow.csv", "test")
    _save_metrics_late_22_23_by_dow(diag_val, MODEL_DIR / "metrics_val_late_22_23_by_dow.csv", "val")
    _save_metrics_late_22_23_by_dow(diag, MODEL_DIR / "metrics_test_late_22_23_by_dow.csv", "test")
    _save_metrics_late_zone_tertiles(diag_val, MODEL_DIR / "metrics_val_late_22_23_zone_vol_tertile.csv", "val")
    _save_metrics_late_zone_tertiles(diag, MODEL_DIR / "metrics_test_late_22_23_zone_vol_tertile.csv", "test")

    for split_name, dframe, y_act, y_hat in (
        ("VAL", diag_val, diag_val["actual"].values, diag_val["predicted"].values),
        ("TEST", diag, diag["actual"].values, diag["predicted"].values),
    ):
        late_mask = dframe["hour"].isin([22, 23]).to_numpy()
        if not late_mask.any():
            continue
        st_late = compute_segment_stats(y_act[late_mask], y_hat[late_mask])
        log.info(
            "[%s-LATE-22-23] n=%d WAPE=%.2f%% MAE=%.3f over=%.1f%% under=%.1f%% bias=%.3f",
            split_name,
            st_late["n"],
            st_late["WAPE"],
            st_late["MAE"],
            st_late["over_pred_rate"] * 100.0,
            st_late["under_pred_rate"] * 100.0,
            st_late["mean_signed_error"],
        )
        for dow in (5, 6):
            for hh in (22, 23):
                mseg = (dframe["dow"] == dow) & (dframe["hour"] == hh)
                if not mseg.any():
                    continue
                st = compute_segment_stats(dframe.loc[mseg, "actual"].values, dframe.loc[mseg, "predicted"].values)
                if st["n"] < 20:
                    continue
                log.info(
                    "[%s-SEG] %s H%02d n=%d WAPE=%.2f%% MAPE=%.1f%% over=%.1f%% under=%.1f%% bias=%.3f",
                    split_name,
                    DOW_NAMES.get(dow, str(dow)),
                    hh,
                    st["n"],
                    st["WAPE"],
                    st["MAPE"],
                    st["over_pred_rate"] * 100.0,
                    st["under_pred_rate"] * 100.0,
                    st["mean_signed_error"],
                )

    bi_tw = int(model_tw.best_iteration or 0)
    bi_qt = int(model_qt.best_iteration or 0)
    model_tw.save_model(str(MODEL_DIR / "lgbm_evening_model.txt"), num_iteration=bi_tw)
    joblib.dump(model_tw, MODEL_DIR / "lgbm_evening_model.pkl")
    model_qt.save_model(str(MODEL_DIR / "lgbm_evening_model_quantile.txt"), num_iteration=bi_qt)
    joblib.dump(model_qt, MODEL_DIR / "lgbm_evening_model_quantile.pkl")
    joblib.dump(feat_cols, MODEL_DIR / "feature_cols.pkl")

    res_tw = br_tw["residual_model"]
    res_cols_tw = br_tw["residual_cols"]
    if res_tw is not None:
        joblib.dump(res_tw, MODEL_DIR / "lgbm_evening_late_residual_model.pkl")
        joblib.dump(res_cols_tw, MODEL_DIR / "late_residual_feature_cols.pkl")
    res_qt = br_qt["residual_model"]
    res_cols_qt = br_qt["residual_cols"]
    if res_qt is not None:
        joblib.dump(res_qt, MODEL_DIR / "lgbm_evening_late_residual_model_quantile.pkl")
        joblib.dump(res_cols_qt, MODEL_DIR / "late_residual_feature_cols_quantile.pkl")

    sun_late_expert_model = br_tw["sun_late_expert_model"]
    sun_blend_alpha = br_tw["sun_blend_alpha"]
    if sun_late_expert_model is not None:
        joblib.dump(sun_late_expert_model, MODEL_DIR / "lgbm_evening_sun_late_expert.pkl")
        sun_late_expert_model.save_model(
            str(MODEL_DIR / "lgbm_evening_sun_late_expert.txt"),
            num_iteration=int(sun_late_expert_model.best_iteration or 0),
        )

    post = _branch_state_to_post_dict(br_tw)
    post["ensemble_mode"] = "tweedie_quantile"
    post["ensemble_quantile_weight"] = float(best_w)
    post["postprocess_quantile"] = _branch_state_to_post_dict(br_qt)
    joblib.dump(post, MODEL_DIR / "postprocess_params.pkl")
    joblib.dump({"tweedie": REG_PARAMS, "quantile_aux": REG_QUANT_PARAMS}, MODEL_DIR / "train_params.pkl")

    segment_artifacts = [
        "metrics_val_by_hour_dow.csv",
        "metrics_test_by_hour_dow.csv",
        "metrics_val_late_22_23_by_dow.csv",
        "metrics_test_late_22_23_by_dow.csv",
        "metrics_val_late_22_23_zone_vol_tertile.csv",
        "metrics_test_late_22_23_zone_vol_tertile.csv",
    ]
    segment_artifacts.extend(
        [
            "lgbm_evening_model_quantile.pkl",
            "lgbm_evening_model_quantile.txt",
        ]
    )
    if res_tw is not None:
        segment_artifacts.append("lgbm_evening_late_residual_model.pkl")
    if res_qt is not None:
        segment_artifacts.append("lgbm_evening_late_residual_model_quantile.pkl")
    if sun_late_expert_model is not None:
        segment_artifacts.append("lgbm_evening_sun_late_expert.pkl")
    info = {
        "model_type": "lgbm_evening_17_23_ensemble_v9",
        "ensemble_mode": "tweedie_quantile",
        "ensemble_quantile_weight": float(best_w),
        "quantile_alpha_auxiliary": QUANTILE_ALPHA_AUX,
        "tweedie_variance_power": float(REG_PARAMS.get("tweedie_variance_power", 1.25)),
        "quantile_alpha_sun_expert": QUANTILE_ALPHA_SUN,
        "hours": sorted(list(EVENING_HOURS)),
        "feature_cols": feat_cols,
        "best_iteration_tweedie": bi_tw,
        "best_iteration_quantile_aux": bi_qt,
        "late_hour_residual_enabled_tweedie": bool(res_tw is not None),
        "late_hour_residual_enabled_quantile": bool(res_qt is not None),
        "late_hour_residual_feature_cols_tweedie": res_cols_tw,
        "late_hour_residual_feature_cols_quantile": res_cols_qt,
        "sun_late_expert_blend_alpha": float(sun_blend_alpha),
        "sun_late_expert_best_iteration": int(sun_late_expert_model.best_iteration or 0) if sun_late_expert_model else 0,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "postprocess_params": post,
        "segment_metrics_artifacts": segment_artifacts,
        "segment_metrics_note": (
            "WAPE/MAE by hour×dow; 22-23 by dow; 22-23 by zone_hour_mean tertile. "
            "over_pred_rate = share(pred>actual); mean_signed_error = mean(pred-actual)."
        ),
        "created_at": datetime.now().isoformat(),
    }
    (MODEL_DIR / "evening_model_info.json").write_text(json.dumps(info, indent=2, default=float))

    imp = pd.Series(model_tw.feature_importance(importance_type="gain"), index=feat_cols).sort_values(ascending=True).tail(25)
    plt.figure(figsize=(10, 8))
    imp.plot(kind="barh")
    plt.title("Evening 17-23 Tweedie (ensemble main) — feature importance")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importance_reg.png", dpi=150)
    plt.close()
    log.info("[SAVE] Artifacts saved to %s", MODEL_DIR)


if __name__ == "__main__":
    train()
