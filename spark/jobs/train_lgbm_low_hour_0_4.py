"""
train_lgbm_low_hour_0_4.py — Zero-inflated LightGBM (V7-style) cho cầu 00h–09h.

Trước đây chỉ 00–04h; giờ 05–09h (ramp sáng) thường tệ nếu chỉ dựa LGBM chính/LSTM.
Pipeline: classifier P(nhu cầu > 0) × regressor (cường độ), tune ngưỡng/cap/blend theo 3 vùng giờ
(deep ≤2, ramp 3–4, morning 5–9) trên val với score ưu tiên WAPE khung 5–9.

Artifacts (tương thích dashboard):
- lgbm_low_hour_classifier.pkl / .txt
- lgbm_low_hour_regressor.pkl / .txt
- feature_cols.pkl, postprocess_params.pkl, low_hour_model_info.json, metrics*.csv

Thư mục: lgbm_low_0_9_zinf_v7_<timestamp>  (dashboard: LOW_HOUR_MODEL_PREFIXES hoặc LOW_HOUR_MODEL_DIR).
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_lgbm_low_hour_0_9_zinf_v7")

GOLD_SRC = "s3://lakehouse/gold/demand_by_zone"
MODEL_DIR = Path("/app/models") / f"lgbm_low_0_9_zinf_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
LOW_HOURS = set(range(10))

FEATURE_COLS = [
    "PULocationID",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "lag_168h",
    "rolling_avg_3h",
    "rolling_avg_6h",
    "rolling_avg_24h",
    "zone_hour_mean",
    "zone_mean",
    "hour_mean",
    "baseline_blend",
    "night_order",
    "is_00",
    "is_01",
    "is_02",
    "is_03",
    "is_04",
    "is_05",
    "is_06",
    "is_07",
    "is_08",
    "is_09",
    "is_deep_night",
    "is_pre_ramp",
    "is_morning_ramp",
    "is_ramp_low",
    "ramp_strength",
    "morning_ramp_strength",
    "lag_ratio_1h_24h",
    "lag_diff_1h_24h",
    "rolling_ratio_3h_24h",
    "momentum_1h_2h",
    "momentum_1h_3h",
    "baseline_gap_1h",
    "baseline_gap_24h",
    "baseline_ratio_1h",
    "baseline_ratio_24h",
    "pre_ramp_gap",
    "morning_gap_1h",
    "ramp_baseline_signal",
]

CLASSIFIER_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 3,
    "lambda_l1": 0.35,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "num_threads": -1,
    "verbose": -1,
    "random_state": 42,
}

REGRESSOR_PARAMS = {
    "objective": "poisson",
    "metric": "poisson",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": 10,
    "min_child_samples": 35,
    "feature_fraction": 0.88,
    "bagging_fraction": 0.85,
    "bagging_freq": 3,
    "lambda_l1": 0.2,
    "lambda_l2": 0.8,
    "max_bin": 255,
    "num_threads": -1,
    "verbose": -1,
    "random_state": 43,
}

N_ESTIMATORS_CLS = 2500
EARLY_STOP_CLS = 120
N_ESTIMATORS_REG = 4000
EARLY_STOP_REG = 150


def load_gold_data() -> pd.DataFrame:
    dt = DeltaTable(GOLD_SRC, storage_options=STORAGE_OPTIONS)
    return dt.to_pandas()


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["PULocationID", "window_start"]).reset_index(drop=True)
    g = df.groupby("PULocationID")["target_demand"]
    for col, shift in {"lag_1h": 1, "lag_2h": 2, "lag_3h": 3, "lag_24h": 24, "lag_168h": 168}.items():
        df[col] = g.shift(shift)
    for col, window in {"rolling_avg_3h": 3, "rolling_avg_6h": 6, "rolling_avg_24h": 24}.items():
        df[col] = (
            df.groupby("PULocationID")["target_demand"]
            .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
    lag_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("rolling_avg_")]
    df[lag_cols] = df[lag_cols].fillna(0.0)
    return df


def add_baselines_no_leak(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["PULocationID", "window_start"]).reset_index(drop=True)
    df["zone_mean"] = (
        df.groupby("PULocationID")["target_demand"].expanding().mean().shift(1).reset_index(level=0, drop=True)
    )
    df["zone_hour_mean"] = (
        df.groupby(["PULocationID", "hour"])["target_demand"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=[0, 1], drop=True)
    )
    global_mean = float(df["target_demand"].mean())
    df["zone_mean"] = df["zone_mean"].fillna(global_mean)
    df["zone_hour_mean"] = df["zone_hour_mean"].fillna(df["zone_mean"])
    hour_mean = df.groupby("hour")["target_demand"].transform(lambda s: s.expanding().mean().shift(1))
    df["hour_mean"] = hour_mean.fillna(global_mean)
    df["baseline_blend"] = 0.70 * df["zone_hour_mean"] + 0.30 * df["hour_mean"]
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True).dt.tz_convert(None)
    if "target_demand" not in df.columns and "trip_count" in df.columns:
        df["target_demand"] = df["trip_count"]
    df["target_demand"] = pd.to_numeric(df["target_demand"], errors="coerce").fillna(0.0)
    df["hour"] = df["window_start"].dt.hour.astype(int)
    df["day_of_week"] = df["window_start"].dt.dayofweek.astype(int)
    df["month"] = df["window_start"].dt.month.astype(int)

    zone_means = df.groupby("PULocationID")["target_demand"].mean()
    active = zone_means[zone_means >= ZONE_DEMAND_MIN].index
    df = df[df["PULocationID"].isin(active)].reset_index(drop=True)

    df = add_lag_features(df)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    years = df["window_start"].dt.year.unique().tolist()
    ny_cal = hols.country_holidays("US", subdiv="NY", years=years)
    holiday_dates = {pd.Timestamp(d).normalize() for d in ny_cal.keys()}
    df["is_holiday"] = df["window_start"].dt.normalize().isin(holiday_dates).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = add_baselines_no_leak(df)

    no_map = {h: h for h in range(10)}
    df["night_order"] = df["hour"].map(no_map).fillna(-1).astype(int)
    for h in range(10):
        df[f"is_{h:02d}"] = (df["hour"] == h).astype(int)

    df["is_deep_night"] = df["hour"].isin([0, 1, 2]).astype(int)
    df["is_pre_ramp"] = df["hour"].isin([3, 4]).astype(int)
    df["is_morning_ramp"] = df["hour"].isin([5, 6, 7, 8, 9]).astype(int)
    df["is_ramp_low"] = df["is_pre_ramp"]
    df["ramp_strength"] = np.clip((df["hour"].astype(float) - 2.0) / 2.0, 0.0, 1.0)
    df["morning_ramp_strength"] = np.where(
        df["hour"].between(0, 9),
        np.clip((df["hour"].astype(float) - 2.0) / 7.0, 0.0, 1.0),
        0.0,
    )

    df["lag_ratio_1h_24h"] = (df["lag_1h"] + 1.0) / (df["lag_24h"] + 1.0)
    df["lag_diff_1h_24h"] = df["lag_1h"] - df["lag_24h"]
    df["rolling_ratio_3h_24h"] = (df["rolling_avg_3h"] + 1.0) / (df["rolling_avg_24h"] + 1.0)
    df["momentum_1h_2h"] = df["lag_1h"] - df["lag_2h"]
    df["momentum_1h_3h"] = df["lag_1h"] - df["lag_3h"]
    df["baseline_gap_1h"] = df["lag_1h"] - df["baseline_blend"]
    df["baseline_gap_24h"] = df["lag_24h"] - df["baseline_blend"]
    df["baseline_ratio_1h"] = (df["lag_1h"] + 1.0) / (df["baseline_blend"] + 1.0)
    df["baseline_ratio_24h"] = (df["lag_24h"] + 1.0) / (df["baseline_blend"] + 1.0)
    df["pre_ramp_gap"] = df["baseline_gap_1h"] * df["is_pre_ramp"]
    df["morning_gap_1h"] = df["baseline_gap_1h"] * df["is_morning_ramp"]
    df["ramp_baseline_signal"] = df["baseline_blend"] * df["ramp_strength"]

    df = df[df["hour"].isin(LOW_HOURS)].copy().reset_index(drop=True)
    df = df.dropna(subset=["target_demand"])
    active_cols = [c for c in FEATURE_COLS if c in df.columns]
    log.info("[PREP] rows=%d | zones~=%d | feats=%d", len(df), df["PULocationID"].nunique(), len(active_cols))
    return df, active_cols


def split_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = df["window_start"]
    t_min, t_max = t.min(), t.max()
    tr_end = t_min + (t_max - t_min) * TRAIN_RATIO
    va_end = t_min + (t_max - t_min) * (TRAIN_RATIO + VAL_RATIO)
    return (t < tr_end).values, ((t >= tr_end) & (t < va_end)).values, (t >= va_end).values


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64).reshape(-1), 0.0, None)
    return float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1.0) * 100.0)


def composite_score(y_true: np.ndarray, y_pred: np.ndarray, hours: np.ndarray) -> float:
    """Ưu tiên WAPE khung 5–9h (nơi user báo tệ), kèm WAPE toàn."""
    h = np.asarray(hours, dtype=int).reshape(-1)
    w_all = wape(y_true, y_pred)
    m = h >= 5
    if not np.any(m) or np.sum(y_true[m]) < 1e-6:
        return w_all
    w_m = wape(y_true[m], y_pred[m])
    return 0.42 * w_all + 0.58 * w_m


def apply_zinf_postprocess(
    prob: np.ndarray,
    reg_pred: np.ndarray,
    X: pd.DataFrame,
    params: dict,
) -> np.ndarray:
    prob = np.asarray(prob, dtype=float).reshape(-1)
    reg_pred = np.clip(np.asarray(reg_pred, dtype=float).reshape(-1), 0.0, None)
    hours = X["hour"].astype(int).values
    baseline = X.get("zone_hour_mean", pd.Series(np.zeros(len(X)))).astype(float).values
    baseline_blend = X.get("baseline_blend", pd.Series(baseline)).astype(float).values

    th_deep = float(params.get("threshold_deep", 0.22))
    th_ramp = float(params.get("threshold_ramp", 0.10))
    th_morning = float(params.get("threshold_morning", th_ramp))
    threshold = np.where(hours <= 2, th_deep, np.where(hours <= 4, th_ramp, th_morning))

    pred = prob * reg_pred
    pred = np.where(prob < threshold, 0.0, pred)

    ramp_alpha = float(params.get("ramp_blend_alpha", 0.0))
    if ramp_alpha > 0:
        m = (hours >= 3) & (hours <= 4)
        pred[m] = (1.0 - ramp_alpha) * pred[m] + ramp_alpha * baseline_blend[m]

    morning_alpha = float(params.get("morning_blend_alpha", 0.0))
    if morning_alpha > 0:
        m = hours >= 5
        pred[m] = (1.0 - morning_alpha) * pred[m] + morning_alpha * baseline_blend[m]

    cap_deep = baseline * float(params.get("cap_mult_deep", 3.0)) + float(params.get("cap_bias_deep", 2.0))
    cap_ramp = baseline * float(params.get("cap_mult_ramp", 3.0)) + float(params.get("cap_bias_ramp", 4.0))
    cap_morning = baseline * float(params.get("cap_mult_morning", 4.5)) + float(params.get("cap_bias_morning", 5.0))
    cap = np.where(hours <= 2, cap_deep, np.where(hours <= 4, cap_ramp, cap_morning))
    pred = np.minimum(pred, np.maximum(cap, 0.0))
    pred = np.maximum(pred, float(params.get("min_floor", 0.0)))
    return np.clip(pred, 0.0, None)


def tune_postprocess(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    reg_val: np.ndarray,
    X_val: pd.DataFrame,
) -> dict:
    hours = X_val["hour"].astype(int).values
    best: dict = {}
    best_sc = float("inf")
    th_d = [0.18, 0.22, 0.26, 0.30, 0.34]
    th_r = [0.08, 0.11, 0.14, 0.17, 0.20]
    th_m = [0.06, 0.09, 0.12, 0.15, 0.18]
    ramp_bl = [0.0, 0.08, 0.14, 0.20]
    morn_bl = [0.0, 0.10, 0.18, 0.26, 0.34]
    caps_d = [(999.0, 0.0), (3.5, 2.0)]
    caps_r = [(999.0, 0.0), (4.0, 3.0), (4.5, 4.0)]
    caps_m = [(999.0, 0.0), (5.0, 5.0), (5.5, 6.0), (4.0, 4.0)]

    for td in th_d:
        for tr in th_r:
            for tm in th_m:
                for rb in ramp_bl:
                    for mb in morn_bl:
                        for cd in caps_d:
                            for cr in caps_r:
                                for cm in caps_m:
                                    p = {
                                        "threshold_deep": td,
                                        "threshold_ramp": tr,
                                        "threshold_morning": tm,
                                        "ramp_blend_alpha": rb,
                                        "morning_blend_alpha": mb,
                                        "cap_mult_deep": cd[0],
                                        "cap_bias_deep": cd[1],
                                        "cap_mult_ramp": cr[0],
                                        "cap_bias_ramp": cr[1],
                                        "cap_mult_morning": cm[0],
                                        "cap_bias_morning": cm[1],
                                        "min_floor": 0.0,
                                    }
                                    pred = apply_zinf_postprocess(prob_val, reg_val, X_val, p)
                                    sc = composite_score(y_val, pred, hours)
                                    if sc < best_sc - 1e-9:
                                        best_sc = sc
                                        best = p
    log.info("[POST] best composite=%.4f params=%s", best_sc, {k: round(float(v), 4) if isinstance(v, float) else v for k, v in best.items()})
    return best


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 0.0, None)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1.0)) * 100.0),
        "WAPE": wape(y_true, y_pred),
    }


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    d = metrics_dict(y_true, y_pred)
    log.info(
        "[%s] MAE=%.3f RMSE=%.3f R2=%.4f MAPE=%.2f%% WAPE=%.2f%%",
        label,
        d["MAE"],
        d["RMSE"],
        d["R2"],
        d["MAPE"],
        d["WAPE"],
    )
    return d


def train() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Low-hour ZINF 00-09h | out=%s", MODEL_DIR)

    df, feat_cols = preprocess(load_gold_data())
    m_tr, m_va, m_te = split_masks(df)

    X = df[feat_cols]
    y = df["target_demand"].astype(float).values
    h = df["hour"].astype(int).values

    y_cls = (y > 0.5).astype(int)
    w_cls = np.ones(len(y), dtype=np.float64)
    w_cls[h >= 5] *= 1.45
    w_cls[np.isin(h, [3, 4])] *= 1.12

    dtr = lgb.Dataset(X.loc[m_tr], label=y_cls[m_tr], weight=w_cls[m_tr], free_raw_data=False)
    dva = lgb.Dataset(X.loc[m_va], label=y_cls[m_va], weight=w_cls[m_va], reference=dtr, free_raw_data=False)
    clf = lgb.train(
        CLASSIFIER_PARAMS,
        dtr,
        num_boost_round=N_ESTIMATORS_CLS,
        valid_sets=[dtr, dva],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOP_CLS, verbose=True), lgb.log_evaluation(200)],
    )
    bi_c = int(clf.best_iteration or 0)
    p_va = np.clip(np.asarray(clf.predict(X.loc[m_va], num_iteration=bi_c)).reshape(-1), 0.0, 1.0)
    p_te = np.clip(np.asarray(clf.predict(X.loc[m_te], num_iteration=bi_c)).reshape(-1), 0.0, 1.0)
    y_va_bin = y_cls[m_va]
    y_te_bin = y_cls[m_te]
    cls_val = {
        "AUC": float(roc_auc_score(y_va_bin, p_va)) if len(np.unique(y_va_bin)) > 1 else float("nan"),
        "ACC": float(accuracy_score(y_va_bin, (p_va >= 0.5).astype(int))),
        "F1": float(f1_score(y_va_bin, (p_va >= 0.5).astype(int), zero_division=0)),
        "Precision": float(precision_score(y_va_bin, (p_va >= 0.5).astype(int), zero_division=0)),
        "Recall": float(recall_score(y_va_bin, (p_va >= 0.5).astype(int), zero_division=0)),
    }
    cls_test = {
        "AUC": float(roc_auc_score(y_te_bin, p_te)) if len(np.unique(y_te_bin)) > 1 else float("nan"),
        "ACC": float(accuracy_score(y_te_bin, (p_te >= 0.5).astype(int))),
        "F1": float(f1_score(y_te_bin, (p_te >= 0.5).astype(int), zero_division=0)),
        "Precision": float(precision_score(y_te_bin, (p_te >= 0.5).astype(int), zero_division=0)),
        "Recall": float(recall_score(y_te_bin, (p_te >= 0.5).astype(int), zero_division=0)),
    }

    pos = y > 0
    if int(np.sum(pos[m_tr])) < 500:
        log.warning("[REG] Very few positive train rows; training regressor on all y>=0 including zeros as Poisson may be weak.")

    y_reg = np.maximum(y, 0.0)
    w_reg = np.ones(len(y), dtype=np.float64)
    w_reg[h >= 5] *= 1.55
    w_reg[np.isin(h, [3, 4])] *= 1.15
    w_reg[pos] *= np.clip(y_reg[pos] / 5.0, 0.85, 2.2)

    dtr_r = lgb.Dataset(X.loc[m_tr], label=y_reg[m_tr], weight=w_reg[m_tr], free_raw_data=False)
    dva_r = lgb.Dataset(X.loc[m_va], label=y_reg[m_va], weight=w_reg[m_va], reference=dtr_r, free_raw_data=False)
    reg = lgb.train(
        REGRESSOR_PARAMS,
        dtr_r,
        num_boost_round=N_ESTIMATORS_REG,
        valid_sets=[dtr_r, dva_r],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOP_REG, verbose=True), lgb.log_evaluation(200)],
    )
    bi_r = int(reg.best_iteration or 0)
    r_va = np.clip(np.asarray(reg.predict(X.loc[m_va], num_iteration=bi_r)).reshape(-1), 0.0, None)
    r_te = np.clip(np.asarray(reg.predict(X.loc[m_te], num_iteration=bi_r)).reshape(-1), 0.0, None)

    post = tune_postprocess(y[m_va], p_va, r_va, X.loc[m_va])

    pred_va = apply_zinf_postprocess(p_va, r_va, X.loc[m_va], post)
    pred_te = apply_zinf_postprocess(p_te, r_te, X.loc[m_te], post)

    val_m = metrics_block(y[m_va], pred_va, "VAL-ZINF-FINAL")
    test_m = metrics_block(y[m_te], pred_te, "TEST-ZINF-FINAL")

    diag = pd.DataFrame({
        "window_start": df.loc[m_te, "window_start"].values,
        "PULocationID": df.loc[m_te, "PULocationID"].values,
        "hour": df.loc[m_te, "hour"].values,
        "actual": y[m_te],
        "predicted": pred_te,
    })
    rows = []
    for hh, g in diag.groupby("hour"):
        rows.append({"hour": int(hh), **metrics_dict(g["actual"].values, g["predicted"].values)})
    pd.DataFrame(rows).to_csv(MODEL_DIR / "metrics_by_hour.csv", index=False)
    diag.to_csv(MODEL_DIR / "test_predictions.csv", index=False)
    pd.DataFrame([test_m]).to_csv(MODEL_DIR / "metrics.csv", index=False)

    clf.save_model(str(MODEL_DIR / "lgbm_low_hour_classifier.txt"), num_iteration=bi_c)
    joblib.dump(clf, MODEL_DIR / "lgbm_low_hour_classifier.pkl")
    reg.save_model(str(MODEL_DIR / "lgbm_low_hour_regressor.txt"), num_iteration=bi_r)
    joblib.dump(reg, MODEL_DIR / "lgbm_low_hour_regressor.pkl")
    joblib.dump(feat_cols, MODEL_DIR / "feature_cols.pkl")
    joblib.dump(post, MODEL_DIR / "postprocess_params.pkl")
    joblib.dump({"classifier": CLASSIFIER_PARAMS, "regressor": REGRESSOR_PARAMS}, MODEL_DIR / "train_params.pkl")

    info = {
        "model_type": "lgbm_low_hour_0_9_zero_inflated_v7",
        "hours": sorted(list(LOW_HOURS)),
        "classifier_best_iteration": bi_c,
        "regressor_best_iteration": bi_r,
        "classifier_val_metrics": cls_val,
        "classifier_test_metrics": cls_test,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "postprocess_params": post,
        "feature_cols": feat_cols,
        "created_at": datetime.now().isoformat(),
        "usage": "Dashboard 00–09h: ENABLE_LOW_HOUR_MODEL=1; set LOW_HOUR_MODEL_DIR to this folder or use latest lgbm_low_0_9_zinf_v7_* / lgbm_low_0_4_zinf_v7_*.",
    }
    (MODEL_DIR / "low_hour_model_info.json").write_text(json.dumps(info, indent=2, default=float))

    imp = pd.Series(reg.feature_importance(importance_type="gain"), index=feat_cols).sort_values(ascending=True).tail(22)
    plt.figure(figsize=(10, 7))
    imp.plot(kind="barh")
    plt.title("Low-hour 00-09 ZINF — regressor gain")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importance_regressor.png", dpi=150)
    plt.close()

    log.info("[SAVE] %s", MODEL_DIR)


if __name__ == "__main__":
    train()
