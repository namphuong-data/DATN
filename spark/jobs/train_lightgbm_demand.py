"""
train_lightgbm_demand.py — LightGBM CPU v3 cho NYC Taxi Demand
──────────────────────────────────────────────────────────────────
Đánh giá LSTM v3 (R²=0.950, MAPE=30.7%, PeakMAPE=13.3%):

  HẠN CHẾ CỦA LSTM:
  ✗ MAPE=30.7% vẫn cao do giờ thấp điểm (mẫu số nhỏ)
  ✗ Training: nhiều giờ, bắt buộc GPU
  ✗ Một model duy nhất cho 54 zones → không học được đặc thù từng zone
  ✗ Inference: phải build sequence 168 bước, chậm hơn tree model

  TẠI SAO LIGHTGBM PHÙ HỢP:
  ✓ Gold layer đã có lag/rolling features → dùng trực tiếp, không cần window 168h
  ✓ Thêm PULocationID làm feature → tree splits học đặc thù từng zone
  ✓ Training: 5-10 phút CPU, không cần GPU
  ✓ Xử lý phân phối lệch taxi demand tự nhiên qua objective 'huber'
  ✓ Sample weight tương đương tiered loss (low×2, normal×1, peak×6)
  ✓ Feature importance cho phép giải thích model

  TẠI SAO LSTM VẪN CÓ LỢI THẾ:
  ✓ Thấy toàn bộ chuỗi 168 bước (không chỉ lag/rolling đã tổng hợp)
  ✓ Temporal Attention học được TIMESTEP quan trọng
  ✓ Tốt hơn LightGBM khi pattern phức tạp không capture được bằng lag

  ENSEMBLE LSTM + LIGHTGBM:
  ✓ LightGBM: feature interactions + zone-specific splits
  ✓ LSTM: temporal context đầy đủ 168h
  ✓ Trung bình có trọng số → giảm variance, cải thiện cả MAPE lẫn PeakMAPE

Cùng protocol với LSTM:
  - Cùng Gold source, zone filter (mean≥5), split 70/15/15 theo timestamp
  - Cùng log1p target transform (taxi demand right-skewed)
  - Cùng metrics: MAE, RMSE, R², MAPE, PeakMAPE
──────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from deltalake import DeltaTable
import holidays as hols

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("train_lgbm")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
GOLD_SRC  = "s3://lakehouse/gold/demand_by_zone"
MODEL_DIR = Path("/app/models") / f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Đường dẫn đến LSTM model để ensemble (None = bỏ qua bước ensemble)
LSTM_MODEL_DIR: Path | None = None  # v3: tắt ensemble mặc định vì LSTM làm MAPE thấp điểm xấu hơn

ZONE_DEMAND_MIN = 5.0   # Lọc zone thưa, đồng bộ với LSTM
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15

# Sample weight — đồng bộ với tiered Huber loss của LSTM
PEAK_WEIGHT     = 4.0
LOW_WEIGHT      = 3.0
RAMP_WEIGHT     = 3.5
TRANSITION_WEIGHT = 2.5
# lgbm_20260427_091216 đang tốt tổng thể nhưng under-predict 00–04h.
# Tăng weight riêng ban đêm vừa phải để model học kỹ vùng demand thấp.
LATE_NIGHT_WEIGHT = 4.0
# Chỉ bật calibration cho 00–04h, không đụng các giờ khác để tránh làm xấu peak/daytime.
ENABLE_CONTEXT_CALIBRATION = True
CALIB_MIN_SAMPLES = 30
CALIB_ALPHA = 120.0  # shrinkage mạnh hơn để tránh overfit low-hour bucket
NIGHT_CALIB_HOURS = {0, 1, 2, 3, 4}
NIGHT_CALIB_CLIP = (0.75, 1.45)
ZERO_LOW_WEIGHT = 6.0

# Trọng số ensemble: lgbm_w và lstm_w phải cộng = 1.0
# Sẽ được tính tự động từ val MAE nếu LSTM model tồn tại
ENSEMBLE_LGBM_W = 0.5
ENSEMBLE_LSTM_W = 0.5

STORAGE_OPTIONS = {
    "endpoint_url"             : "http://minio:9000",
    "access_key_id"            : "minioadmin",
    "secret_access_key"        : "minioadmin123",
    "region"                   : "us-east-1",
    "allow_http"               : "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

# LightGBM dùng thêm PULocationID — tree splits học đặc thù từng zone
# LSTM không dùng được vì sequences từ nhiều zone được trộn lẫn
FEATURE_COLS = [
    "PULocationID",
    "lag_1h", "lag_2h", "lag_3h",
    "lag_24h", "lag_168h",
    "rolling_avg_3h", "rolling_avg_6h", "rolling_avg_24h",
    "hour", "day_of_week",          # raw integers — tree có thể split trực tiếp
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_holiday",
    # Demand regime features — giúp cải thiện low traffic và peak transition
    "is_late_night", "is_early_morning", "is_morning_peak", "is_evening_peak",
    "is_ramp_up_05_08", "is_post_midnight", "is_late_drop_23",
    "is_transition_up", "is_transition_down",
    "lag_ratio_1h_24h", "lag_ratio_3h_24h",
    "rolling_ratio_3h_24h", "lag_diff_1h_24h",
    "prev_1h_growth", "prev_2h_growth", "prev_3h_growth",
    "prev_day_growth_24_168", "ramp_momentum", "night_decay_signal",
    # Growth/acceleration features — hướng 1: bắt tốc độ tăng/giảm của demand
    "growth_1h", "growth_2h", "growth_3h", "growth_6h",
    "growth_24h", "growth_168h",
    "acceleration_1h", "acceleration_3h",
    "ramp_growth_signal", "peak_growth_signal", "night_over_signal",
    "zone_hour_mean", "zone_dow_hour_mean",
]

LGBM_PARAMS = {
    # v3: Huber trên log-target cân bằng hơn regression_l1.
    # regression_l1 tối ưu median nên dễ under-predict 05-08h ramp-up.
    "objective"         : "huber",
    "alpha"             : 0.85,
    "metric"            : ["mae", "l2", "huber"],
    "num_leaves"        : 383,
    "max_bin"           : 511,
    "min_child_samples" : 25,
    "min_data_in_bin"   : 3,
    "learning_rate"     : 0.035,
    "num_threads"       : -1,        # tất cả CPU cores (alias của n_jobs trong LightGBM core)
    # Regularization
    "feature_fraction"  : 0.75,   # giảm dominance của zone_dow_hour_mean
    "bagging_fraction"  : 0.85,
    "bagging_freq"      : 5,
    "lambda_l1"         : 0.35,
    "lambda_l2"         : 0.55,
    "n_jobs"            : -1,
    "verbose"           : -1,
    "random_state"      : 42,
}
N_ESTIMATORS      = 5000   # EarlyStopping tự dừng sớm
EARLY_STOPPING    = 100     # rounds không cải thiện
CHECKPOINT_EVERY  = 500    # lưu checkpoint mỗi N rounds đề phòng crash


# ═══════════════════════════════════════════════════════════════════
# 0. HARDWARE CONFIG
# ═══════════════════════════════════════════════════════════════════
def configure_device() -> dict:
    """LightGBM CPU mode.

    Dockerfile CPU dùng `pip install lightgbm`, vì vậy không thử CUDA nữa.
    CPU ổn định hơn cho deploy và inference; GPU LightGBM chỉ cần thiết khi training rất lớn.
    """
    log.info("[HW] LightGBM CPU mode | max_bin=511 | num_threads=-1")
    return {"device": "cpu", "max_bin": 511, "num_threads": -1, "n_jobs": -1}

def make_checkpoint_callback(model_dir: Path, every_n: int = CHECKPOINT_EVERY):
    """Lưu booster mỗi every_n rounds, đề phòng training crash giữa chừng."""
    ckpt_path = str(model_dir / "lgbm_checkpoint.txt")

    def callback(env):
        if (env.iteration + 1) % every_n == 0:
            env.model.save_model(ckpt_path, num_iteration=env.iteration + 1)
            log.info("[CKPT] Checkpoint saved at round %d → %s", env.iteration + 1, ckpt_path)

    callback.order = 10  # chạy sau built-in callbacks
    return callback


# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA (giống LSTM)
# ═══════════════════════════════════════════════════════════════════
def load_gold_data() -> pd.DataFrame:
    log.info("[DATA] Đọc Gold Delta từ %s ...", GOLD_SRC)
    dt = DeltaTable(GOLD_SRC, storage_options=STORAGE_OPTIONS)
    df = dt.to_pandas()
    log.info("[DATA] Loaded: %d rows, %d cols", len(df), len(df.columns))
    return df


# ═══════════════════════════════════════════════════════════════════
# 2. TIỀN XỬ LÝ
# ═══════════════════════════════════════════════════════════════════
def _compute_extra_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Tính các lag/rolling chưa có trong Gold layer, tránh cross-zone boundary."""
    needed_lags = {"lag_3h": 3, "lag_24h": 24, "lag_168h": 168}
    needed_rolls = {"rolling_avg_6h": 6, "rolling_avg_24h": 24}

    for col, shift in needed_lags.items():
        if col not in df.columns:
            df[col] = (
                df.groupby("PULocationID")["target_demand"]
                .shift(shift)
                .fillna(0)
            )

    for col, window in needed_rolls.items():
        if col not in df.columns:
            df[col] = (
                df.groupby("PULocationID")["target_demand"]
                .apply(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                .reset_index(level=0, drop=True)
                .fillna(0)
            )
    return df


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Tỷ lệ an toàn, tránh chia 0; clip để tree không bị outlier cực lớn."""
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 10)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature tăng cường cho low traffic, peak và vùng chuyển tiếp.

    Mục tiêu:
      - Low traffic: giúp model phân biệt 00–05h, khi mẫu số nhỏ làm MAPE cao.
      - Peak: giúp model nhận diện 07–09h và 16–19h.
      - Transition: 05–07h và 20–22h thường dễ under/over-shoot.
      - Zone-hour prior: baseline lịch sử theo zone × hour, rất hữu ích khi lag nhiễu.
    """
    h = df["hour"].astype(int)
    df["is_late_night"] = h.between(0, 4).astype(int)
    df["is_early_morning"] = h.between(5, 6).astype(int)
    df["is_morning_peak"] = h.between(7, 9).astype(int)
    df["is_evening_peak"] = h.between(16, 19).astype(int)
    df["is_ramp_up_05_08"] = h.between(5, 8).astype(int)
    df["is_post_midnight"] = h.between(0, 1).astype(int)
    df["is_late_drop_23"] = (h == 23).astype(int)
    df["is_transition_up"] = h.between(5, 8).astype(int)
    df["is_transition_down"] = h.between(20, 23).astype(int)

    df["lag_ratio_1h_24h"] = _safe_ratio(df["lag_1h"], df["lag_24h"] + 1)
    df["lag_ratio_3h_24h"] = _safe_ratio(df["lag_3h"], df["lag_24h"] + 1)
    df["rolling_ratio_3h_24h"] = _safe_ratio(df["rolling_avg_3h"], df["rolling_avg_24h"] + 1)
    df["lag_diff_1h_24h"] = df["lag_1h"] - df["lag_24h"]

    # Momentum features: bắt đúng ramp-up 05-08h và giảm over-shoot đêm khuya/23h.
    df["prev_1h_growth"] = df["lag_1h"] - df["lag_2h"]
    df["prev_2h_growth"] = df["lag_2h"] - df["lag_3h"]
    df["prev_3h_growth"] = df["lag_1h"] - df["lag_3h"]
    df["prev_day_growth_24_168"] = df["lag_24h"] - df["lag_168h"]
    df["ramp_momentum"] = df["is_ramp_up_05_08"] * (
        0.55 * df["prev_1h_growth"] + 0.25 * df["prev_2h_growth"] + 0.20 * df["prev_day_growth_24_168"]
    )
    df["night_decay_signal"] = (df["is_post_midnight"] + df["is_late_drop_23"]) * (
        df["lag_1h"] - df["rolling_avg_24h"]
    )

    # Growth/acceleration features — hướng 1.
    # Các feature này chỉ dùng lag/rolling đã shift, nên không leak target hiện tại.
    df["growth_1h"] = df["lag_1h"] - df["lag_2h"]
    df["growth_2h"] = df["lag_2h"] - df["lag_3h"]
    df["growth_3h"] = df["lag_1h"] - df["lag_3h"]
    df["growth_6h"] = df["lag_1h"] - df["rolling_avg_6h"]
    df["growth_24h"] = df["lag_1h"] - df["lag_24h"]
    df["growth_168h"] = df["lag_1h"] - df["lag_168h"]
    df["acceleration_1h"] = df["growth_1h"] - df["growth_2h"]
    df["acceleration_3h"] = df["growth_3h"] - df["prev_day_growth_24_168"]
    df["ramp_growth_signal"] = df["is_ramp_up_05_08"] * (
        0.50 * df["growth_1h"] + 0.30 * df["acceleration_1h"] + 0.20 * df["growth_24h"]
    )
    df["peak_growth_signal"] = (df["is_morning_peak"] + df["is_evening_peak"]) * (
        0.60 * df["growth_1h"] + 0.25 * df["growth_3h"] + 0.15 * df["growth_168h"]
    )
    df["night_over_signal"] = (df["is_late_night"] + df["is_late_drop_23"]) * (
        df["lag_1h"] - df["rolling_avg_24h"]
    )

    # Chỉ dùng shift/expanding để tránh leakage từ target hiện tại vào feature.
    df["zone_hour_mean"] = (
        df.groupby(["PULocationID", "hour"])["target_demand"]
        .transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
        .fillna(df["rolling_avg_24h"])
        .fillna(0)
    )
    df["zone_dow_hour_mean"] = (
        df.groupby(["PULocationID", "day_of_week", "hour"])["target_demand"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
        .fillna(df["zone_hour_mean"])
        .fillna(0)
    )

    # Tín hiệu over-predict ban đêm/23h sau khi đã có zone_hour_mean.
    df["night_over_signal"] = (df["is_late_night"] + df["is_late_drop_23"]) * (
        df["lag_1h"] - df["zone_hour_mean"]
    )
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log.info("[PREP] Tiền xử lý ...")
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)

    # Lọc zone thưa (đồng bộ với LSTM)
    zone_means   = df.groupby("PULocationID")["target_demand"].mean()
    active_zones = zone_means[zone_means >= ZONE_DEMAND_MIN].index
    n_before     = len(df)
    df = df[df["PULocationID"].isin(active_zones)].reset_index(drop=True)
    log.info("[PREP] Zone filter: %d zones | %d→%d rows",
             len(active_zones), n_before, len(df))

    df = df.sort_values(["PULocationID", "window_start"]).reset_index(drop=True)

    # Fill NaN cho lag/rolling đã có trong Gold
    for col in ["lag_1h", "lag_2h", "rolling_avg_3h"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Tính thêm lag dài hạn nếu chưa có
    df = _compute_extra_lags(df)

    # Cyclical encoding
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)

    # Holiday (đồng bộ với LSTM)
    years    = df["window_start"].dt.year.unique().tolist()
    ny_cal   = hols.country_holidays("US", subdiv="NY", years=years)
    hol_dates = {pd.Timestamp(d).normalize() for d in ny_cal.keys()}
    df["is_holiday"] = df["window_start"].dt.normalize().isin(hol_dates).astype(int)
    log.info("[PREP] is_holiday: %d holiday-hours", df["is_holiday"].sum())

    df = add_regime_features(df)

    df = df.dropna(subset=["target_demand"]).reset_index(drop=True)

    active_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(active_cols)
    if missing:
        log.warning("[PREP] Thiếu features: %s", missing)

    log.info("[PREP] Sau xử lý: %d rows | %d features", len(df), len(active_cols))
    log.info("[PREP] target_demand — min=%.0f  max=%.0f  mean=%.1f  p75=%.1f  p95=%.1f",
             df["target_demand"].min(), df["target_demand"].max(),
             df["target_demand"].mean(),
             df["target_demand"].quantile(0.75),
             df["target_demand"].quantile(0.95))
    return df, active_cols


# ═══════════════════════════════════════════════════════════════════
# 3. SPLIT THEO TIMESTAMP (giống LSTM)
# ═══════════════════════════════════════════════════════════════════
def split_data(df: pd.DataFrame):
    timestamps = df["window_start"]
    t_min   = timestamps.min()
    t_range = timestamps.max() - t_min
    train_end = t_min + t_range * TRAIN_RATIO
    val_end   = t_min + t_range * (TRAIN_RATIO + VAL_RATIO)

    mask_train = timestamps < train_end
    mask_val   = (timestamps >= train_end) & (timestamps < val_end)
    mask_test  = timestamps >= val_end

    log.info("[SPLIT] Train<=%s | Val<=%s | Test>%s",
             train_end.date(), val_end.date(), val_end.date())
    log.info("[SPLIT] Train: %d | Val: %d | Test: %d", mask_train.sum(), mask_val.sum(), mask_test.sum())
    return mask_train, mask_val, mask_test


# ═══════════════════════════════════════════════════════════════════
# 4. SAMPLE WEIGHT — tiered, đồng bộ với LSTM tiered Huber
# ═══════════════════════════════════════════════════════════════════
def compute_sample_weights(y: np.ndarray, hours: np.ndarray | None = None) -> np.ndarray:
    """Sample weights v3.

    Mục tiêu:
      - 00-04h: giảm over-predict bằng cách cho model học kỹ vùng low demand.
      - 05-08h: tăng mạnh ramp-up vì export đang under-predict liên tục.
      - 16-19h: giữ peak chiều nhưng không boost quá mức.
      - 23h: tăng học vùng cuối ngày để giảm dự đoán cao.

    Weight được normalize mean=1 để learning-rate/early-stopping ổn định.
    """
    y = np.asarray(y, dtype=float)
    p10, p25, p50, p75, p90 = np.percentile(y, [10, 25, 50, 75, 90])
    w = np.ones(len(y), dtype=float)

    # Low demand: MAPE nhạy với mẫu nhỏ, nhưng clip để không overfit noise.
    low_boost = np.sqrt((p25 + 1.0) / (y + 1.0))
    w *= np.clip(low_boost, 1.0, LOW_WEIGHT)
    w[y <= p10] *= 1.25

    # Peak: giữ độ chính xác giờ đông xe nhưng giảm so với bản cũ để không hi sinh ramp/low.
    w[y >= p75] *= PEAK_WEIGHT
    w[y >= p90] *= 1.10

    if hours is not None:
        h = np.asarray(hours).astype(int)
        late_night = (h >= 0) & (h <= 4)
        ramp = (h >= 5) & (h <= 8)
        morning_peak = (h >= 7) & (h <= 10)
        evening_peak = (h >= 16) & (h <= 19)
        late_drop = (h == 23)
        transition_down = (h >= 20) & (h <= 23)

        w[late_night] *= LATE_NIGHT_WEIGHT
        w[ramp] *= RAMP_WEIGHT
        w[morning_peak] *= 1.35
        w[evening_peak] *= 1.25
        w[late_drop] *= 2.25
        w[transition_down] *= TRANSITION_WEIGHT

    w = np.clip(w, 0.5, 20.0)
    w = w / np.mean(w)

    log.info(
        "[WEIGHT] p10=%.1f p25=%.1f p50=%.1f p75=%.1f p90=%.1f | mean=%.2f max=%.1f",
        p10, p25, p50, p75, p90, w.mean(), w.max(),
    )
    return w


# ═══════════════════════════════════════════════════════════════════
# 5. LOG1P TRANSFORM (giống LSTM Log1pMinMaxScaler)
# ═══════════════════════════════════════════════════════════════════
class Log1pMinMaxScaler:
    """Wrapper log1p + MinMax — giao diện tương thích với LSTM artifact."""
    def __init__(self):
        self._scaler = MinMaxScaler()

    def fit(self, X: np.ndarray) -> "Log1pMinMaxScaler":
        self._scaler.fit(np.log1p(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.transform(np.log1p(X))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.expm1(self._scaler.inverse_transform(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ═══════════════════════════════════════════════════════════════════
# 6. TRAIN LIGHTGBM
# ═══════════════════════════════════════════════════════════════════
def train_lgbm(
    X_train, y_train_log, w_train,
    X_val,   y_val_log,
    params: dict,
    model_dir: Path,
) -> lgb.Booster:
    # free_raw_data=False: giữ raw data trong RAM (56GB đủ thừa)
    # → LightGBM không cần đọc lại từ disk khi cross-validation hoặc rebuild
    dtrain = lgb.Dataset(X_train, label=y_train_log, weight=w_train, free_raw_data=False)
    dval   = lgb.Dataset(X_val,   label=y_val_log,   reference=dtrain, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING, verbose=True),
        lgb.log_evaluation(period=100),
        make_checkpoint_callback(model_dir, every_n=CHECKPOINT_EVERY),
    ]

    log.info(
        "[TRAIN] LightGBM — device=%s | leaves=%d | max_bin=%d | lr=%.3f | estimators=%d | early=%d",
        params.get("device", "cpu"), params["num_leaves"],
        params.get("max_bin", 511), params["learning_rate"],
        N_ESTIMATORS, EARLY_STOPPING,
    )

    booster = lgb.train(
        params          = params,
        train_set       = dtrain,
        num_boost_round = N_ESTIMATORS,
        valid_sets      = [dtrain, dval],
        valid_names     = ["train", "val"],
        callbacks       = callbacks,
    )
    log.info("[TRAIN] Best iteration: %d", booster.best_iteration)
    return booster


# ═══════════════════════════════════════════════════════════════════
# 7. EVALUATE
# ═══════════════════════════════════════════════════════════════════
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(
        np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))
    ) * 100
    p75       = np.percentile(y_true, 75)
    peak_mask = y_true > p75
    peak_mape = (
        np.mean(
            np.abs((y_true[peak_mask] - y_pred[peak_mask]) /
                   np.where(y_true[peak_mask] == 0, 1, y_true[peak_mask]))
        ) * 100
        if peak_mask.sum() > 0 else float("nan")
    )
    log.info("[%s] MAE=%.2f | RMSE=%.2f | R²=%.4f | MAPE=%.2f%% | PeakMAPE=%.2f%%",
             label, mae, rmse, r2, mape, peak_mape)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "PeakMAPE": peak_mape}


def evaluate_and_plot(
    booster: lgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tgt_scaler: Log1pMinMaxScaler,
    test_dates,
    active_cols: list[str],
    model_dir: Path,
    test_context: pd.DataFrame | None = None,
    context_calibration: dict | None = None,
) -> tuple[dict, np.ndarray]:
    log.info("[EVAL] Predicting on test set ...")
    y_pred_log = booster.predict(X_test, num_iteration=booster.best_iteration)

    # Inverse log1p transform
    y_true = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = np.clip(
        tgt_scaler.inverse_transform(y_pred_log.reshape(-1, 1)).flatten(), 0, None
    )
    if test_context is not None and context_calibration is not None:
        y_pred_raw = y_pred.copy()
        y_pred = apply_context_calibration(y_pred, test_context, context_calibration)
        _compute_metrics(y_true, y_pred_raw, "LGBM-TEST-RAW")

    metrics = _compute_metrics(y_true, y_pred, "LGBM-TEST")
    pd.DataFrame([metrics]).to_csv(model_dir / "metrics.csv", index=False)

    _plot_results(booster, y_true, y_pred, test_dates, active_cols, metrics, model_dir)
    return metrics, y_pred


def _plot_results(
    booster, y_true, y_pred, test_dates, active_cols, metrics, model_dir
):
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle("LightGBM Demand Forecast — NYC Taxi", fontsize=14, fontweight="bold")

    # Plot 1: Actual vs Predicted (7 ngày đầu test)
    n_plot = min(7 * 24, len(y_true))
    axes[0].plot(test_dates[:n_plot], y_true[:n_plot],
                 label="Actual",    color="#2196F3", linewidth=1.5)
    axes[0].plot(test_dates[:n_plot], y_pred[:n_plot],
                 label="Predicted", color="#FF5722", linewidth=1.5, linestyle="--")
    axes[0].set_title("Actual vs Predicted (first 7 days of test set)")
    axes[0].set_ylabel("Trip Count")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Plot 2: Scatter
    lim = max(y_true.max(), y_pred.max())
    axes[1].scatter(y_true, y_pred, alpha=0.2, s=5, color="#9C27B0")
    axes[1].plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect fit")
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    axes[1].set_title(
        f"Scatter — R²={metrics['R2']:.3f}  RMSE={metrics['RMSE']:.1f}"
        f"  MAE={metrics['MAE']:.1f}  PeakMAPE={metrics['PeakMAPE']:.1f}%"
    )
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # Plot 3: Feature importance
    importance = pd.Series(
        booster.feature_importance(importance_type="gain"),
        index=active_cols,
    ).sort_values(ascending=True).tail(20)
    importance.plot(kind="barh", ax=axes[2], color="#4CAF50")
    axes[2].set_title("Feature Importance (Gain) — Top 20")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(model_dir / "forecast_result.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("[EVAL] Plot saved → %s/forecast_result.png", model_dir)


# ═══════════════════════════════════════════════════════════════════
# 8. CALIBRATION — chỉnh bias theo hour bucket trên validation
# ═══════════════════════════════════════════════════════════════════
def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100)


def _segment_metrics_frame(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng lỗi theo segment để biết model yếu ở đâu."""
    segments = {
        "late_night_00_04": df_eval["hour"].between(0, 4),
        "transition_up_05_07": df_eval["hour"].between(5, 7),
        "morning_peak_08_10": df_eval["hour"].between(8, 10),
        "day_11_15": df_eval["hour"].between(11, 15),
        "evening_peak_16_20": df_eval["hour"].between(16, 20),
        "late_evening_21_23": df_eval["hour"].between(21, 23),
        "holiday": df_eval["is_holiday"].astype(bool),
        "weekend": df_eval["is_weekend"].astype(bool),
    }
    rows = []
    for name, mask in segments.items():
        sub = df_eval.loc[mask]
        if len(sub) == 0:
            continue
        err = sub["pred"] - sub["true"]
        rows.append({
            "segment": name,
            "n": int(len(sub)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mape": _safe_mape(sub["true"].values, sub["pred"].values),
            "bias": float(np.mean(err)),
        })
    return pd.DataFrame(rows)


def fit_context_calibration(
    booster: lgb.Booster,
    X_val: np.ndarray,
    y_val_scaled: np.ndarray,
    val_df: pd.DataFrame,
    tgt_scaler: Log1pMinMaxScaler,
    model_dir: Path,
) -> dict:
    """Học correction factor chỉ cho khung 00–04h từ validation.

    Lý do: model lgbm_20260427_091216 đã tốt tổng thể, nhưng export hiện tại cho thấy
    00–04h bị under-predict khá đều. Các lần calibration toàn ngày trước đó có thể làm
    lệch ngày thường/peak, nên bản này chỉ tạo factor cho NIGHT_CALIB_HOURS và giữ
    tất cả giờ khác = 1.0. Calibration chỉ được bật nếu cải thiện MAPE/WAPE của 00–04h
    trên validation và không làm MAPE tổng thể xấu đi đáng kể.
    """
    pred_log = booster.predict(X_val, num_iteration=booster.best_iteration)
    y_true = tgt_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
    y_pred = np.clip(tgt_scaler.inverse_transform(pred_log.reshape(-1, 1)).flatten(), 0, None)

    eval_df = val_df[["hour", "is_holiday", "is_weekend"]].copy().reset_index(drop=True)
    eval_df["true"] = y_true
    eval_df["pred_raw"] = y_pred

    # Không dùng global factor để tránh ảnh hưởng các giờ ngoài 00–04h.
    global_factor = 1.0

    rows = []
    factors = {}
    for key_cols in [["is_holiday", "is_weekend", "hour"], ["is_weekend", "hour"], ["hour"]]:
        grouped = eval_df.groupby(key_cols, dropna=False)
        for key, sub in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            hour = int(key[-1])
            if hour not in NIGHT_CALIB_HOURS:
                continue
            n = len(sub)
            if n < CALIB_MIN_SAMPLES:
                continue
            raw_ratio = float(np.median((sub["true"].values + 1.0) / (sub["pred_raw"].values + 1.0)))
            shrink = n / (n + CALIB_ALPHA)
            factor = 1.0 + shrink * (raw_ratio - 1.0)
            factor = float(np.clip(factor, NIGHT_CALIB_CLIP[0], NIGHT_CALIB_CLIP[1]))
            level = "+".join(key_cols)
            factors[(level, tuple(int(x) for x in key))] = factor
            rows.append({
                "level": level,
                "key": "|".join(str(int(x)) for x in key),
                "hour": hour,
                "n": int(n),
                "raw_ratio": raw_ratio,
                "factor": factor,
            })

    cal = {
        "version": "night_only_context_v1",
        "enabled": True,
        "global_factor": global_factor,
        "night_hours": sorted(NIGHT_CALIB_HOURS),
        "factors": factors,
    }

    pred_cal = apply_context_calibration(y_pred, eval_df, cal)
    raw_mape = _safe_mape(y_true, y_pred)
    cal_mape = _safe_mape(y_true, pred_cal)
    raw_mae = float(np.mean(np.abs(y_true - y_pred)))
    cal_mae = float(np.mean(np.abs(y_true - pred_cal)))
    night_mask = eval_df["hour"].astype(int).isin(NIGHT_CALIB_HOURS).values
    raw_night_mape = _safe_mape(y_true[night_mask], y_pred[night_mask]) if night_mask.any() else raw_mape
    cal_night_mape = _safe_mape(y_true[night_mask], pred_cal[night_mask]) if night_mask.any() else cal_mape
    raw_night_wape = float(np.sum(np.abs(y_true[night_mask] - y_pred[night_mask])) / max(np.sum(y_true[night_mask]), 1.0) * 100) if night_mask.any() else raw_mape
    cal_night_wape = float(np.sum(np.abs(y_true[night_mask] - pred_cal[night_mask])) / max(np.sum(y_true[night_mask]), 1.0) * 100) if night_mask.any() else cal_mape

    # Chỉ bật nếu low-hour cải thiện và MAPE tổng thể không xấu quá 0.25 điểm %.
    if (cal_night_mape > raw_night_mape and cal_night_wape > raw_night_wape) or (cal_mape > raw_mape + 0.25):
        cal["enabled"] = False
        pred_cal = y_pred
        log.warning(
            "[CALIB] Night-only disabled: all raw/cal MAPE %.2f→%.2f | night MAPE %.2f→%.2f | night WAPE %.2f→%.2f",
            raw_mape, cal_mape, raw_night_mape, cal_night_mape, raw_night_wape, cal_night_wape,
        )
    else:
        log.info(
            "[CALIB] Night-only enabled: all MAPE %.2f→%.2f MAE %.2f→%.2f | night MAPE %.2f→%.2f | night WAPE %.2f→%.2f",
            raw_mape, cal_mape, raw_mae, cal_mae, raw_night_mape, cal_night_mape, raw_night_wape, cal_night_wape,
        )

    out_rows = pd.DataFrame(rows)
    out_rows.to_csv(model_dir / "context_calibration.csv", index=False)
    joblib.dump(cal, model_dir / "context_calibration.pkl")

    seg_df = eval_df.rename(columns={"pred_raw": "pred"})
    _segment_metrics_frame(seg_df[["hour", "is_holiday", "is_weekend", "true", "pred"]]).to_csv(
        model_dir / "val_segment_metrics_raw.csv", index=False
    )
    seg_df_cal = eval_df.copy()
    seg_df_cal["pred"] = pred_cal
    _segment_metrics_frame(seg_df_cal[["hour", "is_holiday", "is_weekend", "true", "pred"]]).to_csv(
        model_dir / "val_segment_metrics_calibrated.csv", index=False
    )
    log.info("[CALIB] Context calibration saved → %s/context_calibration.csv", model_dir)
    return cal


def apply_context_calibration(y_pred: np.ndarray, context_df: pd.DataFrame, cal: dict | None) -> np.ndarray:
    if not cal or not cal.get("enabled", False):
        return y_pred
    factors = cal.get("factors", {})
    out = []
    for pred, (_, row) in zip(y_pred, context_df.reset_index(drop=True).iterrows()):
        h = int(row["hour"])
        is_hol = int(row.get("is_holiday", 0))
        is_wend = int(row.get("is_weekend", 0))
        f = factors.get(("is_holiday+is_weekend+hour", (is_hol, is_wend, h)))
        if f is None:
            f = factors.get(("is_weekend+hour", (is_wend, h)))
        if f is None:
            f = factors.get(("hour", (h,)))
        if f is None:
            f = cal.get("global_factor", 1.0)
        out.append(pred * float(f))
    return np.clip(np.asarray(out, dtype=float), 0, None)


# Backward-compatible aliases for older inference code.
def fit_hour_calibration(booster, X_val, y_val_scaled, val_hours, tgt_scaler, model_dir):
    val_df = pd.DataFrame({"hour": val_hours, "is_holiday": 0, "is_weekend": 0})
    return fit_context_calibration(booster, X_val, y_val_scaled, val_df, tgt_scaler, model_dir)


def apply_hour_calibration(y_pred: np.ndarray, hours: np.ndarray, cal: dict | None) -> np.ndarray:
    context_df = pd.DataFrame({"hour": hours, "is_holiday": 0, "is_weekend": 0})
    return apply_context_calibration(y_pred, context_df, cal)


# ═══════════════════════════════════════════════════════════════════
# 9. ENSEMBLE VỚI LSTM (optional)
# ═══════════════════════════════════════════════════════════════════
def _try_lstm_ensemble(
    lgbm_pred_test: np.ndarray,
    y_true_test:    np.ndarray,
    df:             pd.DataFrame,
    mask_val:       pd.Series,
    mask_test:      pd.Series,
    active_cols:    list[str],
    lgbm_booster:   lgb.Booster,
    tgt_scaler:     Log1pMinMaxScaler,
    model_dir:      Path,
):
    """
    Tải LSTM model và tính ensemble prediction trên test set.
    Trọng số ensemble được xác định từ val MAE (inverse-MAE weighting).
    Bỏ qua nếu LSTM_MODEL_DIR không tồn tại hoặc thiếu dependency.
    """
    if LSTM_MODEL_DIR is None or not LSTM_MODEL_DIR.exists():
        log.info("[ENSEMBLE] LSTM_MODEL_DIR không tồn tại — bỏ qua ensemble.")
        return

    try:
        import tensorflow as tf
        log.info("[ENSEMBLE] Loading LSTM model từ %s ...", LSTM_MODEL_DIR)

        lstm_model        = tf.keras.models.load_model(str(LSTM_MODEL_DIR / "lstm_demand_v3.keras"), compile=False)
        lstm_feat_scaler  = joblib.load(LSTM_MODEL_DIR / "feature_scaler.pkl")
        lstm_tgt_scaler   = joblib.load(LSTM_MODEL_DIR / "target_scaler.pkl")
        lstm_feat_cols    = joblib.load(LSTM_MODEL_DIR / "feature_cols.pkl")

    except Exception as e:
        log.warning("[ENSEMBLE] Không tải được LSTM model: %s", e)
        return

    LOOK_BACK = 168

    def _build_lstm_sequences(mask):
        zone_ids = df["PULocationID"].values
        global_idx = np.where(mask)[0]
        valid_idx = []
        for i in global_idx:
            if i < LOOK_BACK:
                continue
            window = zone_ids[i - LOOK_BACK: i + 1]
            if np.all(window == window[0]):
                valid_idx.append(i)
        if not valid_idx:
            return None, None
        valid_idx = np.array(valid_idx)

        lstm_cols_avail = [c for c in lstm_feat_cols if c in df.columns]
        feat_raw  = df[lstm_cols_avail].values
        feat_sc   = lstm_feat_scaler.transform(feat_raw)
        tgt_raw   = df["target_demand"].values.reshape(-1, 1)
        tgt_sc    = lstm_tgt_scaler.transform(tgt_raw).flatten()

        X = np.stack([feat_sc[i - LOOK_BACK:i, :] for i in valid_idx]).astype(np.float32)
        y = tgt_sc[valid_idx]
        return X, y, valid_idx

    log.info("[ENSEMBLE] Building LSTM sequences for val/test ...")

    result_val  = _build_lstm_sequences(mask_val)
    result_test = _build_lstm_sequences(mask_test)

    if result_val is None or result_test is None:
        log.warning("[ENSEMBLE] Không đủ dữ liệu để build sequences — bỏ qua.")
        return

    X_val_lstm,  y_val_lstm_sc,  idx_val_lstm  = result_val
    X_test_lstm, y_test_lstm_sc, idx_test_lstm = result_test

    # LSTM predictions (scaled)
    lstm_val_pred_sc  = lstm_model.predict(X_val_lstm,  verbose=0).flatten()
    lstm_test_pred_sc = lstm_model.predict(X_test_lstm, verbose=0).flatten()

    # Inverse transform về demand gốc
    def _inv(sc_arr, scaler):
        return np.clip(scaler.inverse_transform(sc_arr.reshape(-1, 1)).flatten(), 0, None)

    lstm_val_pred   = _inv(lstm_val_pred_sc,  lstm_tgt_scaler)
    lstm_test_pred  = _inv(lstm_test_pred_sc, lstm_tgt_scaler)
    lstm_val_true   = _inv(y_val_lstm_sc,     lstm_tgt_scaler)
    lstm_test_true  = _inv(y_test_lstm_sc,    lstm_tgt_scaler)

    # LightGBM predictions trên cùng idx để so sánh fair
    lgbm_feat_cols_avail = [c for c in active_cols if c in df.columns]
    lgbm_val_pred  = np.clip(
        tgt_scaler.inverse_transform(
            lgbm_booster.predict(df.loc[mask_val].iloc[
                np.searchsorted(np.where(mask_val)[0], idx_val_lstm)
            ][lgbm_feat_cols_avail].values).reshape(-1, 1)
        ).flatten(), 0, None
    )
    lgbm_test_pred = np.clip(
        tgt_scaler.inverse_transform(
            lgbm_booster.predict(df.loc[mask_test].iloc[
                np.searchsorted(np.where(mask_test)[0], idx_test_lstm)
            ][lgbm_feat_cols_avail].values).reshape(-1, 1)
        ).flatten(), 0, None
    )

    # Xác định trọng số từ val MAE (inverse-MAE weighting)
    mae_lgbm_val = mean_absolute_error(lstm_val_true, lgbm_val_pred)
    mae_lstm_val = mean_absolute_error(lstm_val_true, lstm_val_pred)
    inv_sum      = 1.0 / mae_lgbm_val + 1.0 / mae_lstm_val
    w_lgbm       = (1.0 / mae_lgbm_val) / inv_sum
    w_lstm       = (1.0 / mae_lstm_val)  / inv_sum

    log.info("[ENSEMBLE] Val MAE — LGBM=%.2f | LSTM=%.2f", mae_lgbm_val, mae_lstm_val)
    log.info("[ENSEMBLE] Trọng số auto — LGBM=%.3f | LSTM=%.3f", w_lgbm, w_lstm)

    ensemble_test_pred = w_lgbm * lgbm_test_pred + w_lstm * lstm_test_pred

    log.info("[ENSEMBLE] === INDIVIDUAL MODELS ===")
    _compute_metrics(lstm_test_true, lgbm_test_pred,      "LGBM (aligned)")
    _compute_metrics(lstm_test_true, lstm_test_pred,      "LSTM")
    log.info("[ENSEMBLE] === ENSEMBLE (w_lgbm=%.2f, w_lstm=%.2f) ===", w_lgbm, w_lstm)
    ens_metrics = _compute_metrics(lstm_test_true, ensemble_test_pred, "ENSEMBLE")

    pd.DataFrame([{
        **ens_metrics,
        "w_lgbm": w_lgbm, "w_lstm": w_lstm,
        "mae_lgbm_val": mae_lgbm_val, "mae_lstm_val": mae_lstm_val,
    }]).to_csv(model_dir / "ensemble_metrics.csv", index=False)

    # Plot ensemble vs individual
    n_plot = min(7 * 24, len(lstm_test_true))
    dates  = df["window_start"].values[idx_test_lstm]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates[:n_plot], lstm_test_true[:n_plot],     label="Actual",   color="#2196F3", lw=1.5)
    ax.plot(dates[:n_plot], lstm_test_pred[:n_plot],     label="LSTM",     color="#FF9800", lw=1.2, ls="--")
    ax.plot(dates[:n_plot], lgbm_test_pred[:n_plot],     label="LightGBM", color="#4CAF50", lw=1.2, ls="--")
    ax.plot(dates[:n_plot], ensemble_test_pred[:n_plot], label=f"Ensemble (LGBM×{w_lgbm:.2f}+LSTM×{w_lstm:.2f})",
            color="#E91E63", lw=2.0)
    ax.set_title(f"Ensemble vs Individual — PeakMAPE: Ensemble={ens_metrics['PeakMAPE']:.1f}%")
    ax.set_ylabel("Trip Count"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_dir / "ensemble_result.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("[ENSEMBLE] Plot saved → %s/ensemble_result.png", model_dir)


# ═══════════════════════════════════════════════════════════════════
# 9. SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════
def save_artifacts(
    booster:     lgb.Booster,
    tgt_scaler:  Log1pMinMaxScaler,
    active_cols: list[str],
    params:      dict,
    model_dir:   Path,
):
    # lgbm_model.txt — native format, dùng cho inference nhanh (không cần Python)
    # num_iteration=best_iteration: chỉ lưu các cây đến best round, bỏ phần thừa
    booster.save_model(
        str(model_dir / "lgbm_model.txt"),
        num_iteration=booster.best_iteration,
    )
    # lgbm_model.pkl — tiện cho load nhanh trong Python pipeline
    joblib.dump(booster,     model_dir / "lgbm_model.pkl")
    joblib.dump(tgt_scaler,  model_dir / "target_scaler.pkl")
    joblib.dump(active_cols, model_dir / "feature_cols.pkl")
    joblib.dump(params,      model_dir / "train_params.pkl")
    (model_dir / "scaler_info.txt").write_text(
        "target_scaler: Log1pMinMaxScaler (log1p + MinMax)\n"
        "inverse_transform = expm1(minmax_inverse(x))\n"
        f"best_iteration: {booster.best_iteration}\n"
        f"device: {params.get('device', 'cpu')}\n"
    )
    log.info("[SAVE] Artifacts → %s  (best_iteration=%d)", model_dir, booster.best_iteration)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Phát hiện GPU và merge vào params
    device_cfg = configure_device()
    params = {**LGBM_PARAMS, **device_cfg}   # device_cfg override max_bin nếu CUDA

    log.info("=" * 64)
    log.info("LightGBM Demand Forecast  START")
    log.info("  Gold source  : %s", GOLD_SRC)
    log.info("  Zone filter  : mean >= %.1f trips/h", ZONE_DEMAND_MIN)
    log.info("  Split ratio  : Train=%.0f%%  Val=%.0f%%  Test=%.0f%%",
             TRAIN_RATIO * 100, VAL_RATIO * 100, (1 - TRAIN_RATIO - VAL_RATIO) * 100)
    log.info("  Weights      : peak=%.1f | low=%.1f | ramp=%.1f | transition=%.1f | late-night=%.1f", PEAK_WEIGHT, LOW_WEIGHT, RAMP_WEIGHT, TRANSITION_WEIGHT, LATE_NIGHT_WEIGHT)
    log.info("  Device       : %s  |  num_leaves=%d  |  max_bin=%d",
             params.get("device"), params["num_leaves"], params.get("max_bin", 511))
    log.info("  LSTM ensemble: %s", LSTM_MODEL_DIR or "disabled")
    log.info("  Model dir    : %s", MODEL_DIR)
    log.info("=" * 64)

    df_raw, active_cols = preprocess(load_gold_data())
    mask_train, mask_val, mask_test = split_data(df_raw)

    X_train = df_raw.loc[mask_train, active_cols].values
    y_train = df_raw.loc[mask_train, "target_demand"].values
    X_val   = df_raw.loc[mask_val,   active_cols].values
    y_val   = df_raw.loc[mask_val,   "target_demand"].values
    X_test  = df_raw.loc[mask_test,  active_cols].values
    y_test  = df_raw.loc[mask_test,  "target_demand"].values

    # Log1p transform target (đồng bộ LSTM)
    tgt_scaler   = Log1pMinMaxScaler().fit(y_train.reshape(-1, 1))
    y_train_log  = tgt_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_log    = tgt_scaler.transform(y_val.reshape(-1, 1)).flatten()

    # Sample weights (tiered: low=2, peak=6)
    w_train = compute_sample_weights(y_train, df_raw.loc[mask_train, "hour"].values)

    booster = train_lgbm(X_train, y_train_log, w_train, X_val, y_val_log, params, MODEL_DIR)

    context_calibration = None
    if ENABLE_CONTEXT_CALIBRATION:
        context_calibration = fit_context_calibration(
            booster, X_val, y_val_log,
            df_raw.loc[mask_val, ["hour", "is_holiday", "is_weekend"]],
            tgt_scaler, MODEL_DIR
        )
    else:
        log.info("[CALIB] Disabled by config — using raw LightGBM predictions.")

    # Lưu model TRƯỚC khi ensemble — đảm bảo artifact tồn tại dù ensemble lỗi
    save_artifacts(booster, tgt_scaler, active_cols, params, MODEL_DIR)

    test_dates = df_raw["window_start"].values[mask_test]
    metrics, lgbm_pred = evaluate_and_plot(
        booster, X_test, tgt_scaler.transform(y_test.reshape(-1, 1)).flatten(),
        tgt_scaler, test_dates, active_cols, MODEL_DIR,
        test_context=df_raw.loc[mask_test, ["hour", "is_holiday", "is_weekend"]],
        context_calibration=context_calibration,
    )

    _try_lstm_ensemble(
        lgbm_pred_test = lgbm_pred,
        y_true_test    = y_test,
        df             = df_raw,
        mask_val       = mask_val,
        mask_test      = mask_test,
        active_cols    = active_cols,
        lgbm_booster   = booster,
        tgt_scaler     = tgt_scaler,
        model_dir      = MODEL_DIR,
    )

    log.info("=" * 64)
    log.info(
        "TRAINING COMPLETE — MAE=%.2f | RMSE=%.2f | R²=%.4f | MAPE=%.2f%% | PeakMAPE=%.2f%%",
        metrics["MAE"], metrics["RMSE"], metrics["R2"],
        metrics["MAPE"], metrics["PeakMAPE"],
    )
    log.info("  Artifacts → %s", MODEL_DIR)
    log.info("=" * 64)


if __name__ == "__main__":
    main()
