"""
train_lstm_demand_v5.py — LSTM cho NYC Taxi Demand Forecast
══════════════════════════════════════════════════════════════════════
Vấn đề của v4 (phân tích từ forecast_result_v4.png + dữ liệu tuần 02–08/09/2024):

  Khung 00h–08h (đêm/sáng sớm):
    ✗ Over-predict nặng: bias +431 đến +934 trips/h, MAPE ~40–50%
    ✗ TieredHuberLoss không phân biệt giờ → penalize đêm quá nhẹ
    ✗ look_back=72h không bắt được weekly pattern (cùng giờ tuần trước)

  Khung 17h–23h (chiều tối):
    ✗ Under-predict: bias −123 đến −692 trips/h, MAPE ~15–45%
    ✗ Evening peak có variance cao → model thiên về mean
    ✗ Zone embedding chưa phân biệt zone có evening surge khác nhau

Giải pháp v5 — nhắm trực tiếp 2 khung giờ yếu:

  [1] HourBucketLoss (thay TieredHuberLoss):
      Thay vì chia theo p25/p75 của demand, chia theo GIỜ:
        night   (00–08h): weight ×2.5 — phạt over-prediction đêm
        midday  (09–16h): weight ×1.0 — giữ nguyên (đang tốt)
        evening (17–23h): weight ×1.8 — phạt under-prediction tối
      + Quantile auxiliary loss (q10/q90): regularise tail distributions

  [2] Dual-context LSTM (thay single-LSTM):
      Branch SHORT (look_back=72h): giữ nguyên → midday vẫn tốt
      Branch LONG  (look_back=168h = 1 tuần): bắt weekly pattern đêm
      → Merge bằng learned gate (không hard-code concat)

  [3] Hour + DayOfWeek Embedding (bổ sung):
      nn.Embedding(24, 16) + nn.Embedding(7, 8) concat vào FC head
      → Model học "1h sáng thứ 6" ≠ "1h sáng thứ 2"
      → Giữ nguyên zone_emb, cộng thêm time_emb

  [4] Asymmetric Huber per bucket:
      night/evening dùng delta nhỏ hơn (0.5) → nhạy hơn với lỗi nhỏ
      midday giữ delta=1.0 → không ảnh hưởng

BUG FIX v5 (CUDA OOB Embedding Assertion):
  Lỗi gốc: "indexSelectLargeIndex: srcIndex < srcSelectDimSize failed"
  Nguyên nhân: PULocationID trong NYC TLC data có thể có giá trị 264/265
    (zone đặc biệt: Unknown, Out-of-NYC) vượt quá MAX_LOCATION_ID=263
    → CUDA embedding lookup OOB → crash không stacktrace rõ ràng
  Fix đã áp dụng (3 lớp bảo vệ):
    [A] MAX_LOCATION_ID tăng từ 263 → 265 (bao phủ tất cả NYC zone ID)
    [B] preprocess(): filter zone/hour/dow OOB trước khi build dataset
    [C] forward(): clamp zone_id/hour_id/dow_id làm safety net cuối

Giữ nguyên từ v4 (không đổi):
  ✓ Bi-LSTM 2 layers, hidden=256, dropout=0.3
  ✓ TemporalAttention (ATTN_DIM=128)
  ✓ Zone Embedding (265+1, 16)  ← size tăng nhẹ do fix
  ✓ Log1pStandardScaler
  ✓ CosineAnnealingWarmRestarts
  ✓ AMP mixed precision, batch_size=2048
  ✓ DataLoader num_workers=4, pin_memory
  ✓ Checkpoint + Early stopping + save_artifacts

Hardware target:
  RAM 56GB | VRAM 24GB (RTX 3090/4090) | CPU i7-12700F (12P+4E cores)

Features dùng từ Gold layer:
  Temporal sequences (SHORT look_back=72h, LONG look_back=168h):
    trip_count, lag_1h, lag_2h, rolling_avg_3h, rolling_avg_24h
    lag_24h, lag_168h
    hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
    is_weekend, is_holiday
  Static per-zone + per-time (zone + hour + dow embedding):
    PULocationID → Embedding(266, 16)   ← FIX: 266 = 265+1 (padding_idx=0)
    hour         → Embedding(24,  16)   ← MỚI
    day_of_week  → Embedding(7,    8)   ← MỚI

Target: target_demand (trip count giờ t+1)
══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
import math
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from deltalake import DeltaTable
import holidays as hols

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("train_lstm_v4")


# ══════════════════════════════════════════════════════════════════════
# CONFIG — v5: nhắm vào 2 khung giờ yếu (00-08h, 17-23h)
# ══════════════════════════════════════════════════════════════════════
GOLD_SRC  = "s3://lakehouse/gold/demand_by_zone"
MODEL_DIR = Path("/app/models") / f"lstm_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

STORAGE_OPTIONS = {
    "endpoint_url"             : "http://minio:9000",
    "access_key_id"            : "minioadmin",
    "secret_access_key"        : "minioadmin123",
    "region"                   : "us-east-1",
    "allow_http"               : "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

# ── Data config ───────────────────────────────────────────────────────
ZONE_DEMAND_MIN = 5.0
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15

# ── Sequence config ───────────────────────────────────────────────────
# v5: Dual-context — SHORT=72h (giữ như v4), LONG=168h (1 tuần, MỚI)
# LONG branch bắt weekly pattern cho khung đêm (00-08h):
#   "Cùng giờ 2h sáng tuần trước zone 132 có demand = X"
#   → LSTM 72h không thấy được, cần 168h
LOOK_BACK_SHORT = 72    # giữ nguyên v4 → midday không bị ảnh hưởng
LOOK_BACK_LONG  = 168   # 1 tuần → weekly pattern cho đêm/tối
LOOK_BACK       = LOOK_BACK_LONG  # alias cho các hàm dùng LOOK_BACK (backward compat)

FEATURE_COLS = [
    "trip_count",
    "lag_24h",
    "lag_168h",
    "lag_1h", "lag_2h",
    "rolling_avg_3h",
    "rolling_avg_24h",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_holiday",
]

# ── Zone Embedding (FIX v5: tăng lên 265 để bao NYC zone EWR/Unknown) ──
# NYC TLC có zone ID từ 1-263 chính thức, nhưng có các zone đặc biệt:
#   EWR (Newark Airport) = 1 (đã trong range)
#   Unknown / Out-of-NYC = 264, 265 → cần bao phủ để tránh OOB assertion
MAX_LOCATION_ID = 265
ZONE_EMB_DIM    = 16

# ── Time Embedding (MỚI v5) ──────────────────────────────────────────
# Learnable embedding cho hour-of-day và day-of-week:
#   Lý do: sin/cos chỉ biểu diễn "vị trí" trong chu kỳ, nhưng không học
#   được "profile" riêng của từng giờ (1h sáng thứ 6 vs 1h sáng thứ 2).
#   Embedding học được: đêm thứ 6/7 NYC có nightlife cao → khác đêm thứ 2.
#   → Đặc biệt hữu ích cho khung 00-08h và 17-23h.
HOUR_EMB_DIM    = 16    # 24 hours → embedding dim 16
DOW_EMB_DIM     = 8     # 7 days   → embedding dim 8

# ── LSTM Architecture (giữ nguyên v4) ────────────────────────────────
LSTM_HIDDEN     = 256
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.3
LSTM_BIDIR      = True

ATTN_DIM        = 128
FC_HIDDEN       = 128

# ── HourBucket Loss weights (MỚI v5) ─────────────────────────────────
# Thay TieredHuberLoss (p25/p75 của demand) bằng bucket theo GIỜ:
#   Phân tích bias tuần 02-08/09/2024:
#     night   (00-08h): over-predict +431→+934 trips/h, MAPE 40-50%
#     midday  (09-16h): gần cân bằng, MAPE  3-14% → KHÔNG THAY ĐỔI
#     evening (17-23h): under-predict -123→-692 trips/h, MAPE 15-45%
NIGHT_HOURS     = list(range(0, 9))    # 00h–08h inclusive
MIDDAY_HOURS    = list(range(9, 17))   # 09h–16h
EVENING_HOURS   = list(range(17, 24))  # 17h–23h

NIGHT_WEIGHT    = 3.2   # tăng: over-predict 0h–8h
MIDDAY_WEIGHT   = 1.0
EVENING_WEIGHT  = 2.4   # tăng: under-predict 17h–23h

# Phạt bổ sung bất đối xứng (ngoài Huber có trọng số): đêm ưu tiên phạt pred>actual, tối phạt pred<actual
NIGHT_OVER_PENALTY   = 0.4   # * mean(relu(ŷ-y)²) tại 0–8h
EVENING_UNDER_PENALTY = 0.4  # * mean(relu(y-ŷ)²) tại 17–23h

# Quantile auxiliary loss: regularise tail distributions
# q10/q90 giúp model calibrate tốt ở giờ có variance cao (đêm, tối)
QUANTILE_Q      = (0.10, 0.90)
QUANTILE_WEIGHT = 0.18  # tăng nhẹ: tail đêm/tối

# Huber delta per bucket: đêm/tối dùng delta nhỏ → nhạy hơn lỗi nhỏ
HUBER_DELTA_NIGHT   = 0.5   # MỚI: nhạy hơn với over-prediction đêm
HUBER_DELTA_MIDDAY  = 1.0   # giữ nguyên v4
HUBER_DELTA_EVENING = 0.7   # MỚI: nhạy hơn với under-prediction tối

# (backward compat — dùng ở các hàm cũ không đổi)
HUBER_DELTA     = HUBER_DELTA_MIDDAY
PEAK_WEIGHT     = 6.0
LOW_WEIGHT      = 2.0

# ── Training config ───────────────────────────────────────────────────
# 24GB VRAM: batch 4096 dễ OOM (đặc biệt với pin_memory + workers). Mặc định 2048 ổn định;
# tăng dần: BATCH_SIZE=2560 / 3072. Nếu lỗi OOM ở pin_memory thread: PIN_MEMORY=0
def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2048"))
# Val/Test forward — cap để không vượt VRAM (train 2048 → eval tối đa 4096)
EVAL_BATCH_CAP = int(os.environ.get("EVAL_BATCH_CAP", "4096"))
MAX_EPOCHS      = 100
PATIENCE        = 7
INIT_LR         = 3e-3
MIN_LR          = 1e-5
COSINE_T0       = 10
GRAD_CLIP       = 1.0
USE_AMP         = True
# Tắt nếu OOM trong DataLoader pin_memory: PIN_MEMORY=0 (chậm hơn chút, host RAM dễ chịu hơn)
PIN_MEMORY = _env_bool("PIN_MEMORY", True)
# Docker: shm đủ (compose) — bus error → DATALOADER_WORKERS=0
DATALOADER_WORKERS = int(os.environ.get("DATALOADER_WORKERS", "4"))
PREFETCH_FACTOR = int(os.environ.get("PREFETCH_FACTOR", "2"))
CHECKPOINT_EVERY   = 10


# ══════════════════════════════════════════════════════════════════════
# 0. HARDWARE DETECT
# ══════════════════════════════════════════════════════════════════════
def configure_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1024**3
        log.info("[HW] GPU: %s | VRAM: %.1f GB", props.name, vram)
        log.info("[HW] Mixed precision AMP: %s", "ON" if USE_AMP else "OFF")
        # Ước lượng thô tensor đầu vào (dual branch); thực tế VRAM còn gradients + optimizer states
        rough_act = (
            BATCH_SIZE
            * (LOOK_BACK_SHORT + LOOK_BACK_LONG)
            * len(FEATURE_COLS)
            * 4
            / 1024**3
        )
        log.info(
            "[HW] batch=%d | workers=%d prefetch=%d pin_memory=%s | ~input tensor lower bound ≈ %.2f GB",
            BATCH_SIZE,
            DATALOADER_WORKERS,
            PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else 0,
            PIN_MEMORY,
            rough_act,
        )
    else:
        dev = torch.device("cpu")
        log.info("[HW] GPU không khả dụng → CPU mode | batch_size giảm về 512")
    return dev


# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════
def load_gold_data() -> pd.DataFrame:
    log.info("[DATA] Đọc Gold Delta từ %s ...", GOLD_SRC)
    dt = DeltaTable(GOLD_SRC, storage_options=STORAGE_OPTIONS)
    df = dt.to_pandas()
    log.info("[DATA] Loaded: %d rows × %d cols", len(df), len(df.columns))
    return df


# ══════════════════════════════════════════════════════════════════════
# 2. TIỀN XỬ LÝ
# ══════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log.info("[PREP] Tiền xử lý ...")
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)

    # Lọc zone thưa
    zone_means   = df.groupby("PULocationID")["target_demand"].mean()
    active_zones = zone_means[zone_means >= ZONE_DEMAND_MIN].index
    n_before     = len(df)
    df = df[df["PULocationID"].isin(active_zones)].reset_index(drop=True)
    log.info("[PREP] Zone filter: %d active zones | %d→%d rows",
             len(active_zones), n_before, len(df))

    df = df.sort_values(["PULocationID", "window_start"]).reset_index(drop=True)

    # Fill NaN lag/rolling từ Gold
    for col in ["lag_1h", "lag_2h", "rolling_avg_3h"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Cyclical encoding (chưa có trong Gold → tạo tại đây)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)

    # Holiday
    years    = df["window_start"].dt.year.unique().tolist()
    ny_cal   = hols.country_holidays("US", subdiv="NY", years=years)
    hol_dates = {pd.Timestamp(d).normalize() for d in ny_cal.keys()}
    df["is_holiday"] = df["window_start"].dt.normalize().isin(hol_dates).astype(int)
    log.info("[PREP] is_holiday: %d holiday-hours", df["is_holiday"].sum())

    df = df.dropna(subset=["target_demand"]).reset_index(drop=True)

    # FIX v5: Loại bỏ zone ID nằm ngoài embedding range [1, MAX_LOCATION_ID]
    # Nguyên nhân crash CUDA: "indexSelectLargeIndex: srcIndex < srcSelectDimSize"
    # xảy ra khi zone_id (hoặc hour_id/dow_id) vượt quá kích thước embedding.
    # Các zone đặc biệt của NYC TLC (ID > 265) không có ý nghĩa địa lý → loại bỏ.
    n_before_zone = len(df)
    df = df[df["PULocationID"].between(1, MAX_LOCATION_ID)].reset_index(drop=True)
    if len(df) < n_before_zone:
        log.warning("[PREP] Zone OOB filter: loại %d rows có PULocationID > %d",
                    n_before_zone - len(df), MAX_LOCATION_ID)

    # Sanity check hour và day_of_week (phòng data corruption)
    if "hour" in df.columns:
        bad_hour = (~df["hour"].between(0, 23)).sum()
        if bad_hour:
            log.warning("[PREP] Phát hiện %d rows có hour ngoài [0,23] → loại bỏ", bad_hour)
            df = df[df["hour"].between(0, 23)].reset_index(drop=True)
    if "day_of_week" in df.columns:
        bad_dow = (~df["day_of_week"].between(0, 6)).sum()
        if bad_dow:
            log.warning("[PREP] Phát hiện %d rows có day_of_week ngoài [0,6] → loại bỏ", bad_dow)
            df = df[df["day_of_week"].between(0, 6)].reset_index(drop=True)

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


# ══════════════════════════════════════════════════════════════════════
# 3. SPLIT THEO TIMESTAMP
# ══════════════════════════════════════════════════════════════════════
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
    log.info("[SPLIT] Train: %d | Val: %d | Test: %d",
             mask_train.sum(), mask_val.sum(), mask_test.sum())
    return mask_train, mask_val, mask_test


# ══════════════════════════════════════════════════════════════════════
# 4. LOG1P + STANDARD SCALER
#
# Tại sao StandardScaler (không phải MinMaxScaler như LightGBM):
#   - MinMaxScaler nhạy cảm với outlier (taxi demand có đỉnh cực cao)
#   - StandardScaler robust hơn: outlier không kéo toàn bộ scale
#   - Log1p trước: biến phân phối right-skewed → gần Gaussian
#   - Kết quả: LSTM nhận input phân phối đều → gradient ổn định hơn
# ══════════════════════════════════════════════════════════════════════
class Log1pStandardScaler:
    """
    Scaler: log1p → StandardScaler.
    Khác với LightGBM dùng Log1pMinMaxScaler vì:
      - LSTM cần input có mean≈0, std≈1 để gradient ổn định
      - MinMax bound [0,1] không giúp gì cho LSTM initialization
      - Standard normalization là best practice cho neural nets
    """
    def __init__(self):
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "Log1pStandardScaler":
        self._scaler.fit(np.log1p(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.transform(np.log1p(X))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.expm1(self._scaler.inverse_transform(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ══════════════════════════════════════════════════════════════════════
# 5. PYTORCH DATASET — sequence builder
# ══════════════════════════════════════════════════════════════════════
class TaxiDemandDataset(Dataset):
    """
    Xây dựng dual-context sequences per-zone.

    v5: Trả về HAI sequences cho mỗi sample:
      seq_short: (LOOK_BACK_SHORT=72h, n_features)  — giữ nguyên v4
      seq_long:  (LOOK_BACK_LONG=168h, n_features)  — MỚI: weekly context

    Lý do dual-context:
      - SHORT branch: đủ ngắn để LSTM không vanish, tốt cho midday
      - LONG  branch: bắt "cùng giờ tuần trước" cho đêm/tối
      - Merge bằng learned gate trong model → tự động chọn nguồn nào quan trọng hơn

    Thêm: hour_id và dow_id (integer) cho Time Embedding layer.
    """
    def __init__(
        self,
        df:          pd.DataFrame,
        feat_cols:   list[str],
        feat_scaler: StandardScaler,
        tgt_scaler:  "Log1pStandardScaler",
        look_back:   int = LOOK_BACK,  # backward compat, không dùng nội tại
    ):
        self.look_back_short = LOOK_BACK_SHORT
        self.look_back_long  = LOOK_BACK_LONG
        # samples: (seq_short, seq_long, hour_id, dow_id, zone_id, target)
        self.samples: list = []

        feat_raw   = df[feat_cols].values.astype(np.float32)
        feat_sc    = feat_scaler.transform(feat_raw)
        tgt_raw    = df["target_demand"].values.reshape(-1, 1).astype(np.float32)
        tgt_sc     = tgt_scaler.transform(tgt_raw).flatten().astype(np.float32)
        zone_ids   = df["PULocationID"].values.astype(np.int64)

        # hour_id và dow_id cho Time Embedding (v5)
        hour_ids   = df["hour"].values.astype(np.int64)
        dow_ids    = df["day_of_week"].values.astype(np.int64)

        for zone in df["PULocationID"].unique():
            mask = df["PULocationID"].values == zone
            idx  = np.where(mask)[0]
            # Cần ít nhất LOOK_BACK_LONG bước trước target
            for j in range(self.look_back_long, len(idx)):
                i_target = idx[j]
                i_short_start = idx[j - self.look_back_short]
                i_long_start  = idx[j - self.look_back_long]

                # Kiểm tra continuity
                if (i_target - i_short_start != self.look_back_short or
                        i_target - i_long_start != self.look_back_long):
                    continue

                seq_short = feat_sc[i_short_start:i_target, :]  # (72, n_feats)
                seq_long  = feat_sc[i_long_start:i_target,  :]  # (168, n_feats)
                zone_id   = int(zone_ids[i_target])
                hour_id   = int(hour_ids[i_target])
                dow_id    = int(dow_ids[i_target])
                target    = float(tgt_sc[i_target])

                self.samples.append((seq_short, seq_long, hour_id, dow_id, zone_id, target))

        log.info("[DATASET] Tổng sequences: %d (dual-context short=%dh long=%dh)",
                 len(self.samples), self.look_back_short, self.look_back_long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq_short, seq_long, hour_id, dow_id, zone_id, target = self.samples[idx]
        return (
            torch.from_numpy(seq_short),                        # (72,  n_feats)
            torch.from_numpy(seq_long),                         # (168, n_feats)
            torch.tensor(hour_id,  dtype=torch.long),           # scalar
            torch.tensor(dow_id,   dtype=torch.long),           # scalar
            torch.tensor(zone_id,  dtype=torch.long),           # scalar
            torch.tensor(target,   dtype=torch.float32),        # scalar
        )


# ══════════════════════════════════════════════════════════════════════
# 6. LSTM MODEL với Zone Embedding + Temporal Attention
# ══════════════════════════════════════════════════════════════════════
class TemporalAttention(nn.Module):
    """
    Additive attention over LSTM hidden states.

    Tại sao cần Attention (không chỉ lấy hidden state cuối):
      - LSTM cuối cùng bị "dominated" bởi timestep gần nhất
      - Attention cho phép model học: "giờ cao điểm 2 ngày trước"
        ảnh hưởng nhiều hơn "nửa đêm hôm qua"
      - Kết quả: PeakMAPE giảm 3-5% trong thực nghiệm
    """
    def __init__(self, lstm_output_dim: int, attn_dim: int = ATTN_DIM):
        super().__init__()
        self.W = nn.Linear(lstm_output_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (batch, seq_len, lstm_output_dim)
        scores = self.v(torch.tanh(self.W(hidden)))     # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)              # (batch, seq_len, 1)
        context = (weights * hidden).sum(dim=1)         # (batch, lstm_output_dim)
        return context


class DualContextLSTMModel(nn.Module):
    """
    v5 Architecture — Dual-context LSTM với Time Embedding:

      [seq_short: (72,  n_feats)] → Bi-LSTM → Attention → ctx_short (512)
      [seq_long:  (168, n_feats)] → Bi-LSTM → Attention → ctx_long  (512)
                                                                ↓
                                              Learned gate: α·ctx_short + (1-α)·ctx_long
                                                                ↓
      [zone_id]  → ZoneEmbedding(263,16)  ──→ concat → FC(512+16+16+8 → 128 → 1)
      [hour_id]  → HourEmbedding(24, 16)  ──↗
      [dow_id]   → DowEmbedding(7,    8)  ──↗

    Gate mechanism (MỚI v5):
      α = sigmoid(W·[ctx_short; ctx_long]) ∈ (0,1) per sample
      → Tự học: giờ đêm → α nhỏ (long context quan trọng hơn)
                giờ trưa → α lớn (short context đủ)
      → Không hard-code cách merge → linh hoạt hơn concat cứng

    Hai branch LSTM CHIA SẺ cùng kiến trúc (Bi-LSTM 2 layers, hidden=256)
    nhưng TRỌNG SỐ ĐỘC LẬP → mỗi branch tự học pattern của độ dài mình.

    Giữ nguyên v4:
      - TemporalAttention (ATTN_DIM=128)
      - LayerNorm trước FC
      - GELU + Dropout(0.2) trong FC head
    """
    def __init__(self, n_feats: int, n_locations: int = MAX_LOCATION_ID):
        super().__init__()
        lstm_out_dim = LSTM_HIDDEN * (2 if LSTM_BIDIR else 1)  # 512

        # ── Embeddings ─────────────────────────────────────────────
        self.zone_emb = nn.Embedding(n_locations + 1, ZONE_EMB_DIM, padding_idx=0)
        self.hour_emb = nn.Embedding(24, HOUR_EMB_DIM)   # MỚI v5
        self.dow_emb  = nn.Embedding(7,  DOW_EMB_DIM)    # MỚI v5

        # ── SHORT branch (72h) — giữ như v4, học daily pattern ────
        self.lstm_short = nn.LSTM(
            input_size   = n_feats,
            hidden_size  = LSTM_HIDDEN,
            num_layers   = LSTM_LAYERS,
            batch_first  = True,
            bidirectional= LSTM_BIDIR,
            dropout      = LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )
        self.attn_short = TemporalAttention(lstm_out_dim, ATTN_DIM)

        # ── LONG branch (168h) — MỚI v5, học weekly pattern ───────
        self.lstm_long = nn.LSTM(
            input_size   = n_feats,
            hidden_size  = LSTM_HIDDEN,
            num_layers   = LSTM_LAYERS,
            batch_first  = True,
            bidirectional= LSTM_BIDIR,
            dropout      = LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )
        self.attn_long = TemporalAttention(lstm_out_dim, ATTN_DIM)

        # ── Learned gate: α·short + (1-α)·long ────────────────────
        # Input: concat [ctx_short, ctx_long] → scalar gate per sample
        self.gate = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # ── FC head: concat [gated_ctx + zone_emb + hour_emb + dow_emb] ──
        fc_in = lstm_out_dim + ZONE_EMB_DIM + HOUR_EMB_DIM + DOW_EMB_DIM
        self.fc = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Linear(fc_in, FC_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(FC_HIDDEN, 1),
        )

        # ── Quantile head (auxiliary, MỚI v5) ─────────────────────
        # q10 và q90: regularise tails ở giờ có variance cao
        self.fc_quantile = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Linear(fc_in, FC_HIDDEN // 2),
            nn.GELU(),
            nn.Linear(FC_HIDDEN // 2, len(QUANTILE_Q)),
        )

    def forward(
        self,
        seq_short: torch.Tensor,   # (B, 72,  n_feats)
        seq_long:  torch.Tensor,   # (B, 168, n_feats)
        hour_id:   torch.Tensor,   # (B,)
        dow_id:    torch.Tensor,   # (B,)
        zone_id:   torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # FIX v5: Clamp embedding indices để tránh CUDA OOB assertion
        # "indexSelectLargeIndex: srcIndex < srcSelectDimSize failed"
        # Safety net: ngay cả khi data lọt qua preprocess filter, model không crash.
        zone_id = zone_id.clamp(1, self.zone_emb.num_embeddings - 1)
        hour_id = hour_id.clamp(0, 23)
        dow_id  = dow_id.clamp(0, 6)

        # ── SHORT branch ───────────────────────────────────────────
        out_s, _ = self.lstm_short(seq_short)      # (B, 72,  lstm_out_dim)
        ctx_s    = self.attn_short(out_s)           # (B, lstm_out_dim)

        # ── LONG branch ────────────────────────────────────────────
        out_l, _ = self.lstm_long(seq_long)         # (B, 168, lstm_out_dim)
        ctx_l    = self.attn_long(out_l)            # (B, lstm_out_dim)

        # ── Learned gate ───────────────────────────────────────────
        alpha    = self.gate(torch.cat([ctx_s, ctx_l], dim=-1))  # (B, 1)
        ctx      = alpha * ctx_s + (1 - alpha) * ctx_l           # (B, lstm_out_dim)

        # ── Embeddings ─────────────────────────────────────────────
        zone_vec = self.zone_emb(zone_id)           # (B, 16)
        hour_vec = self.hour_emb(hour_id)           # (B, 16)
        dow_vec  = self.dow_emb(dow_id)             # (B,  8)

        combined = torch.cat([ctx, zone_vec, hour_vec, dow_vec], dim=-1)

        # ── Heads ──────────────────────────────────────────────────
        point    = self.fc(combined).squeeze(-1)                  # (B,)
        quantile = self.fc_quantile(combined)                     # (B, 2)

        return point, quantile


# Alias để không phải đổi tên ở main()
LSTMDemandModel = DualContextLSTMModel


# ══════════════════════════════════════════════════════════════════════
# 7. TIERED HUBER LOSS
#
# LightGBM dùng sample_weight ngoài model.
# PyTorch LSTM: implement loss nội tại để gradient phản ánh đúng priority.
# Giữ nguyên ngưỡng p25/p75 và trọng số LOW/PEAK để đồng bộ LightGBM.
# ══════════════════════════════════════════════════════════════════════
class HourBucketLoss(nn.Module):
    """
    v5: Loss theo KHUNG GIỜ thay vì theo percentile demand.

    Phân tích từ dữ liệu tuần 02-08/09/2024:
      - night   (00-08h): model over-predict 431-934 trips/h, MAPE 40-50%
        → weight cao + delta nhỏ: phạt nặng kể cả lỗi nhỏ
      - midday  (09-16h): gần cân bằng, MAPE 3-14%
        → giữ nguyên v4 (weight=1, delta=1.0)
      - evening (17-23h): model under-predict 123-692 trips/h, MAPE 15-45%
        → weight vừa + delta nhỏ: kéo model bắt peak tốt hơn

    Ngoài ra thêm Quantile auxiliary loss (q10/q90):
      - Giờ đêm và tối có variance cao (phụ thuộc event, thời tiết)
      - Pinball loss ép model calibrate đúng ở tail, không chỉ ở mean
      - λ=0.15: nhỏ đủ để không át point head

    So với TieredHuberLoss v4 (p25/p75 của demand):
      - v4: "demand thấp" và "demand cao" → không biết ĐÓ LÀ GIỜ NÀO
      - v5: "giờ 2h sáng" và "giờ 19h tối" → biết chính xác pattern muốn fix
    """
    def __init__(self):
        super().__init__()
        # Precompute hour → (weight, delta) lookup
        self._build_lookup()

    def _build_lookup(self):
        weights = torch.ones(24)
        deltas  = torch.ones(24)
        for h in NIGHT_HOURS:
            weights[h] = NIGHT_WEIGHT
            deltas[h]  = HUBER_DELTA_NIGHT
        for h in MIDDAY_HOURS:
            weights[h] = MIDDAY_WEIGHT
            deltas[h]  = HUBER_DELTA_MIDDAY
        for h in EVENING_HOURS:
            weights[h] = EVENING_WEIGHT
            deltas[h]  = HUBER_DELTA_EVENING
        self.register_buffer("hour_weights", weights)
        self.register_buffer("hour_deltas",  deltas)

    def _huber(self, pred: torch.Tensor, target: torch.Tensor,
               delta: torch.Tensor) -> torch.Tensor:
        """Element-wise Huber với delta per-sample."""
        diff     = (pred - target).abs()
        quadratic = 0.5 * (pred - target) ** 2
        linear    = delta * (diff - 0.5 * delta)
        return torch.where(diff <= delta, quadratic, linear)

    def _pinball(self, pred: torch.Tensor, target: torch.Tensor,
                 q: float) -> torch.Tensor:
        err = target - pred
        return torch.max(q * err, (q - 1) * err)

    def forward(
        self,
        pred_point:    torch.Tensor,   # (B,)
        pred_quantile: torch.Tensor,   # (B, 2)  q10, q90
        target:        torch.Tensor,   # (B,)
        hour_ids:      torch.Tensor,   # (B,)  integer hour 0-23
    ) -> torch.Tensor:
        # FIX: register_buffer mặc định ở CPU, nhưng hour_ids đang ở GPU.
        # Phải đưa buffer về cùng device với tensor trước khi index.
        dev   = hour_ids.device
        w     = self.hour_weights.to(dev)[hour_ids]   # (B,)
        delta = self.hour_deltas.to(dev)[hour_ids]    # (B,)

        # Point loss: weighted Huber per bucket
        huber = self._huber(pred_point, target, delta)
        point_loss = (w * huber).mean()

        # Bất đối xứng: 0–8h phạt thêm khi dự đoán cao hơn thực tế; 17–23h khi dự đoán thấp hơn
        is_night = (hour_ids >= 0) & (hour_ids <= 8)
        is_evening = (hour_ids >= 17) & (hour_ids <= 23)
        over = F.relu(pred_point - target)
        under = F.relu(target - pred_point)
        asym = (
            NIGHT_OVER_PENALTY * (is_night.float() * over.pow(2)).mean()
            + EVENING_UNDER_PENALTY * (is_evening.float() * under.pow(2)).mean()
        )

        # Quantile auxiliary loss (q10, q90)
        q_loss = torch.zeros(1, device=pred_point.device)
        for i, q in enumerate(QUANTILE_Q):
            q_loss = q_loss + (w * self._pinball(pred_quantile[:, i], target, q)).mean()
        q_loss = q_loss / len(QUANTILE_Q)

        return point_loss + QUANTILE_WEIGHT * q_loss + asym


# Alias backward compat (run_epoch vẫn dùng criterion=TieredHuberLoss)
TieredHuberLoss = HourBucketLoss


# ══════════════════════════════════════════════════════════════════════
# 8. BUILD SCALERS + DATASETS + DATALOADERS
# ══════════════════════════════════════════════════════════════════════
def build_data_pipeline(
    df:         pd.DataFrame,
    feat_cols:  list[str],
    mask_train, mask_val, mask_test,
    device:     torch.device,
):
    df_train = df[mask_train].reset_index(drop=True)
    df_val   = df[mask_val].reset_index(drop=True)
    df_test  = df[mask_test].reset_index(drop=True)

    # Fit scalers CHỈ trên train (tránh data leakage)
    feat_scaler = StandardScaler()
    feat_scaler.fit(df_train[feat_cols].values.astype(np.float32))

    tgt_scaler = Log1pStandardScaler()
    tgt_scaler.fit(df_train["target_demand"].values.reshape(-1, 1).astype(np.float32))

    log.info("[SCALE] feature StandardScaler fitted | target Log1pStandardScaler fitted")

    # v5: TaxiDemandDataset không cần look_back kwarg (dùng LOOK_BACK_SHORT/LONG global)
    kwargs_ds = dict(feat_cols=feat_cols, feat_scaler=feat_scaler, tgt_scaler=tgt_scaler)
    ds_train = TaxiDemandDataset(df_train, **kwargs_ds)
    ds_val   = TaxiDemandDataset(df_val,   **kwargs_ds)
    ds_test  = TaxiDemandDataset(df_test,  **kwargs_ds)

    pin_mem = (device.type == "cuda") and PIN_MEMORY
    nw      = DATALOADER_WORKERS if device.type == "cuda" else 0
    bs_eval = min(BATCH_SIZE * 2, EVAL_BATCH_CAP)
    dl_kw: dict = dict(num_workers=nw, pin_memory=pin_mem, persistent_workers=nw > 0)
    if nw > 0:
        dl_kw["prefetch_factor"] = PREFETCH_FACTOR

    dl_train = DataLoader(
        ds_train, batch_size=BATCH_SIZE, shuffle=True, **dl_kw
    )
    dl_val = DataLoader(
        ds_val, batch_size=bs_eval, shuffle=False, **dl_kw
    )
    dl_test = DataLoader(
        ds_test, batch_size=bs_eval, shuffle=False, **dl_kw
    )

    log.info("[DATA] Train samples=%d | Val=%d | Test=%d",
             len(ds_train), len(ds_val), len(ds_test))
    return dl_train, dl_val, dl_test, feat_scaler, tgt_scaler


# ══════════════════════════════════════════════════════════════════════
# 9. TRAIN LOOP
# ══════════════════════════════════════════════════════════════════════
def run_epoch(
    model:      DualContextLSTMModel,
    loader:     DataLoader,
    criterion:  HourBucketLoss,
    optimizer:  torch.optim.Optimizer | None,
    scaler_amp: GradScaler,
    device:     torch.device,
    is_train:   bool = True,
) -> float:
    model.train() if is_train else model.eval()
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        # v5: DataLoader trả về 6-tuple (seq_short, seq_long, hour_id, dow_id, zone_id, target)
        for seq_short, seq_long, hour_id, dow_id, zone_id, target in loader:
            seq_short = seq_short.to(device, non_blocking=True)
            seq_long  = seq_long.to(device,  non_blocking=True)
            hour_id   = hour_id.to(device,   non_blocking=True)
            dow_id    = dow_id.to(device,    non_blocking=True)
            zone_id   = zone_id.to(device,   non_blocking=True)
            target    = target.to(device,    non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=USE_AMP and device.type == "cuda"):
                pred_point, pred_q = model(seq_short, seq_long, hour_id, dow_id, zone_id)
                # v5: HourBucketLoss nhận thêm hour_ids để weighting theo giờ
                loss = criterion(pred_point, pred_q, target, hour_id)

            if is_train:
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler_amp.step(optimizer)
                scaler_amp.update()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


def train_model(
    model:      LSTMDemandModel,
    dl_train:   DataLoader,
    dl_val:     DataLoader,
    device:     torch.device,
    model_dir:  Path,
) -> tuple[LSTMDemandModel, list[float], list[float]]:
    criterion = HourBucketLoss()

    # Adam với weight_decay nhẹ (L2 regularization cho embedding + FC)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=INIT_LR, weight_decay=1e-4
    )

    # CosineAnnealingWarmRestarts: tự restart mỗi T_0 epochs
    # Lý do dùng thay vì ReduceLROnPlateau:
    #   - Cosine không stuck tại LR quá thấp
    #   - Warm restarts giúp thoát local minima
    #   - LightGBM không có scheduler → neural net cần
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=1, eta_min=MIN_LR
    )

    # AMP scaler cho mixed precision
    amp_scaler = GradScaler(enabled=USE_AMP and device.type == "cuda")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    best_val_loss = float("inf")
    patience_cnt  = 0
    train_losses  = []
    val_losses    = []
    best_epoch    = 0

    log.info("[TRAIN] Bắt đầu training — epochs=%d | patience=%d | lr=%.4f",
             MAX_EPOCHS, PATIENCE, INIT_LR)
    log.info("[TRAIN] Gradient clip=%.1f | AMP=%s | batch=%d | look_back=%d",
             GRAD_CLIP, USE_AMP, BATCH_SIZE, LOOK_BACK)

    for epoch in range(1, MAX_EPOCHS + 1):
        t_loss = run_epoch(model, dl_train, criterion, optimizer,
                           amp_scaler, device, is_train=True)
        v_loss = run_epoch(model, dl_val,   criterion, None,
                           amp_scaler, device, is_train=False)

        scheduler.step()  # cosine step sau mỗi epoch

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        log.info("[EPOCH %3d/%d] train_loss=%.4f  val_loss=%.4f  lr=%.6f",
                 epoch, MAX_EPOCHS, t_loss, v_loss, lr_now)

        # Early stopping
        if v_loss < best_val_loss - 1e-5:
            best_val_loss = v_loss
            best_epoch    = epoch
            patience_cnt  = 0
            torch.save(model.state_dict(), model_dir / "best_lstm.pt")
            log.info("[CKPT] ✓ Best model saved (epoch %d, val_loss=%.4f)", epoch, v_loss)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                log.info("[TRAIN] Early stopping tại epoch %d (best=%d)", epoch, best_epoch)
                break

        # Periodic checkpoint
        if epoch % CHECKPOINT_EVERY == 0:
            ckpt_path = model_dir / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": v_loss,
            }, ckpt_path)
            log.info("[CKPT] Periodic checkpoint → %s", ckpt_path)

    # Load best weights
    log.info("[TRAIN] Load best weights từ epoch %d", best_epoch)
    model.load_state_dict(torch.load(model_dir / "best_lstm.pt", map_location=device))
    return model, train_losses, val_losses


# ══════════════════════════════════════════════════════════════════════
# 10. EVALUATE
# ══════════════════════════════════════════════════════════════════════
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    
    # Tính WAPE (Chính xác nhất cho taxi demand)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    
    # Tính MAPE an toàn (thêm 1 vào mẫu số để tránh chia cho 0)
    # sMAPE: Giới hạn lỗi trong khoảng [0, 200%]
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-5)) * 100
    
    p75       = np.percentile(y_true, 75)
    peak_mask = y_true > p75
    peak_mape = (
        np.mean(
            np.abs((y_true[peak_mask] - y_pred[peak_mask]) /
                   np.where(y_true[peak_mask] == 0, 1, y_true[peak_mask]))
        ) * 100
        if peak_mask.sum() > 0 else float("nan")
    )
    
    # Cập nhật dòng log (thêm WAPE vào để theo dõi)
    log.info("[%s] MAE=%.2f | RMSE=%.2f | R²=%.4f | WAPE=%.2f%% | SMAPE=%.2f%% | PeakMAPE=%.2f%%",
             label, mae, rmse, r2, wape, smape, peak_mape)
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "WAPE": wape, "SMAPE": smape, "PeakMAPE": peak_mape}

def evaluate(
    model:      DualContextLSTMModel,
    dl_test:    DataLoader,
    tgt_scaler: Log1pStandardScaler,
    df_test:    pd.DataFrame,
    device:     torch.device,
    model_dir:  Path,
) -> dict:
    model.eval()
    preds, trues, hours_all = [], [], []

    with torch.no_grad():
        for seq_short, seq_long, hour_id, dow_id, zone_id, target in dl_test:
            seq_short = seq_short.to(device, non_blocking=True)
            seq_long  = seq_long.to(device,  non_blocking=True)
            hour_id   = hour_id.to(device,   non_blocking=True)
            dow_id    = dow_id.to(device,    non_blocking=True)
            zone_id   = zone_id.to(device,   non_blocking=True)
            pred_point, _ = model(seq_short, seq_long, hour_id, dow_id, zone_id)
            preds.append(pred_point.cpu().numpy())
            trues.append(target.numpy())
            hours_all.extend(hour_id.cpu().tolist())

    # Last batch có thể nhỏ hơn, nên dùng concatenate theo batch thay vì np.array(list_of_arrays)
    preds = np.concatenate([p.reshape(-1, 1) for p in preds], axis=0)
    trues = np.concatenate([t.reshape(-1, 1) for t in trues], axis=0)
    hours_arr = np.array(hours_all)

    y_pred = np.clip(tgt_scaler.inverse_transform(preds).flatten(), 0, None)
    y_true = tgt_scaler.inverse_transform(trues).flatten()

    metrics = _compute_metrics(y_true, y_pred, "LSTM-v5-TEST")

    # v5: thêm per-bucket metrics để theo dõi cải thiện 2 khung giờ
    for bucket, hours in [("night_00-08h",   NIGHT_HOURS),
                           ("midday_09-16h",  MIDDAY_HOURS),
                           ("evening_17-23h", EVENING_HOURS)]:
        mask = np.isin(hours_arr, hours)
        if mask.sum() == 0:
            continue
        bm = _compute_metrics(y_true[mask], y_pred[mask], f"LSTM-v5-{bucket}")
        for k, v in bm.items():
            metrics[f"{bucket}_{k}"] = v

    pd.DataFrame([metrics]).to_csv(model_dir / "metrics.csv", index=False)
    return metrics, y_true, y_pred


# ══════════════════════════════════════════════════════════════════════
# 11. PLOT
# ══════════════════════════════════════════════════════════════════════
def plot_results(
    train_losses: list[float],
    val_losses:   list[float],
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    metrics:      dict,
    model_dir:    Path,
):
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    fig.suptitle("LSTM v5 Demand Forecast — NYC Taxi", fontsize=14, fontweight="bold")

    # Plot 1: Loss curves
    axes[0].plot(train_losses, label="Train Loss", color="#2196F3", lw=1.5)
    axes[0].plot(val_losses,   label="Val Loss",   color="#FF5722", lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Tiered Huber Loss")
    axes[0].set_title("Training Curve")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Plot 2: Actual vs Predicted (7 ngày đầu test)
    n_plot = min(7 * 24, len(y_true))
    axes[1].plot(range(n_plot), y_true[:n_plot],
                 label="Actual",    color="#2196F3", lw=1.5)
    axes[1].plot(range(n_plot), y_pred[:n_plot],
                 label="Predicted", color="#FF5722", lw=1.5, ls="--")
    axes[1].set_title("Actual vs Predicted (first 7 days of test set)")
    axes[1].set_ylabel("Trip Count")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # Plot 3: Scatter
    lim = max(float(y_true.max()), float(y_pred.max()))
    axes[2].scatter(y_true, y_pred, alpha=0.2, s=4, color="#9C27B0")
    axes[2].plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect fit")
    axes[2].set_xlabel("Actual"); axes[2].set_ylabel("Predicted")
    axes[2].set_title(
        f"Scatter — R²={metrics['R2']:.3f}  MAE={metrics['MAE']:.1f}"
        f"  RMSE={metrics['RMSE']:.1f}  PeakMAPE={metrics['PeakMAPE']:.1f}%"
    )
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = model_dir / "forecast_result_v5.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("[PLOT] Saved → %s", path)


# ══════════════════════════════════════════════════════════════════════
# 12. SAVE ARTIFACTS (tương thích với _try_lstm_ensemble trong LightGBM script)
# ══════════════════════════════════════════════════════════════════════
def save_artifacts(
    model:        LSTMDemandModel,
    feat_scaler:  StandardScaler,
    tgt_scaler:   Log1pStandardScaler,
    feat_cols:    list[str],
    metrics:      dict,
    model_dir:    Path,
    n_feats:      int,
):
    # Lưu model dưới dạng TorchScript để inference không cần code nguồn
    model.eval()
    torch.save(model.state_dict(), model_dir / "lstm_weights.pt")

    # Lưu config để rebuild model khi load
    model_cfg = {
        "n_feats"         : n_feats,
        "n_locations"     : MAX_LOCATION_ID,   # FIX v5: 265 (bao NYC zone đặc biệt)
        "lstm_hidden"     : LSTM_HIDDEN,
        "lstm_layers"     : LSTM_LAYERS,
        "lstm_dropout"    : LSTM_DROPOUT,
        "lstm_bidir"      : LSTM_BIDIR,
        "attn_dim"        : ATTN_DIM,
        "zone_emb_dim"    : ZONE_EMB_DIM,
        "hour_emb_dim"    : HOUR_EMB_DIM,   # MỚI v5
        "dow_emb_dim"     : DOW_EMB_DIM,    # MỚI v5
        "fc_hidden"       : FC_HIDDEN,
        "look_back_short" : LOOK_BACK_SHORT, # MỚI v5
        "look_back_long"  : LOOK_BACK_LONG,  # MỚI v5
        "feat_cols"       : feat_cols,
    }
    joblib.dump(model_cfg,   model_dir / "model_config.pkl")
    joblib.dump(feat_scaler, model_dir / "feature_scaler.pkl")
    joblib.dump(tgt_scaler,  model_dir / "target_scaler.pkl")
    joblib.dump(feat_cols,   model_dir / "feature_cols.pkl")
    pd.DataFrame([metrics]).to_csv(model_dir / "metrics.csv", index=False)

    (model_dir / "model_info.txt").write_text(
        f"LSTM v5 — Dual-Context Bi-LSTM + HourBucketLoss + Time Embedding\n"
        f"look_back_short : {LOOK_BACK_SHORT}h (daily context)\n"
        f"look_back_long  : {LOOK_BACK_LONG}h  (weekly context)\n"
        f"hidden          : {LSTM_HIDDEN} × {'Bi' if LSTM_BIDIR else ''}{LSTM_LAYERS} layers\n"
        f"zone_emb_dim    : {ZONE_EMB_DIM}\n"
        f"hour_emb_dim    : {HOUR_EMB_DIM}  (MỚI v5)\n"
        f"dow_emb_dim     : {DOW_EMB_DIM}   (MỚI v5)\n"
        f"loss            : HourBucketLoss (night×{NIGHT_WEIGHT}, midday×{MIDDAY_WEIGHT}, evening×{EVENING_WEIGHT})\n"
        f"quantile_weight : {QUANTILE_WEIGHT} (q10/q90 auxiliary)\n"
        f"batch_size      : {BATCH_SIZE}\n"
        f"scaler          : Log1pStandardScaler\n"
        f"MAE             : {metrics['MAE']:.2f}\n"
        f"R2              : {metrics['R2']:.4f}\n"
        f"PeakMAPE        : {metrics['PeakMAPE']:.2f}%\n"
    )
    log.info("[SAVE] Artifacts → %s", model_dir)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# MAIN TRAIN — tạo thư mục mới theo datetime, train từ đầu, lưu artifacts
# Chạy: python train_lstm_demand_v5.py --mode train
# ══════════════════════════════════════════════════════════════════════
def main_train():
    global MODEL_DIR, BATCH_SIZE
    MODEL_DIR = Path("/app/models") / f"lstm_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = configure_device()
    if device.type == "cpu":
        BATCH_SIZE = 512
        log.warning("[CFG] CPU mode → batch_size giảm về %d", BATCH_SIZE)

    log.info("=" * 64)
    log.info("LSTM v5 Demand Forecast  TRAIN")
    log.info("  Gold source      : %s", GOLD_SRC)
    log.info("  look_back_short  : %dh (daily context)", LOOK_BACK_SHORT)
    log.info("  look_back_long   : %dh  (weekly context)", LOOK_BACK_LONG)
    log.info("  LSTM hidden      : %d | layers=%d | Bi=%s",
             LSTM_HIDDEN, LSTM_LAYERS, LSTM_BIDIR)
    log.info("  Zone emb dim     : %d | Hour emb: %d | DoW emb: %d",
             ZONE_EMB_DIM, HOUR_EMB_DIM, DOW_EMB_DIM)
    log.info("  Loss             : HourBucketLoss (night×%.1f δ=%.1f | midday×%.1f δ=%.1f | evening×%.1f δ=%.1f)",
             NIGHT_WEIGHT, HUBER_DELTA_NIGHT,
             MIDDAY_WEIGHT, HUBER_DELTA_MIDDAY,
             EVENING_WEIGHT, HUBER_DELTA_EVENING)
    log.info("  Quantile aux     : λ=%.2f, q=%s", QUANTILE_WEIGHT, QUANTILE_Q)
    log.info("  Batch size       : %d | AMP=%s", BATCH_SIZE, USE_AMP)
    log.info("  LR               : %.4f → %.6f (CosineWarmRestart T0=%d)",
             INIT_LR, MIN_LR, COSINE_T0)
    log.info("  Model dir        : %s", MODEL_DIR)
    log.info("=" * 64)

    # 1. Load + preprocess
    df_raw, feat_cols = preprocess(load_gold_data())

    # 2. Split
    mask_train, mask_val, mask_test = split_data(df_raw)

    # 3. Data pipeline
    dl_train, dl_val, dl_test, feat_scaler, tgt_scaler = build_data_pipeline(
        df_raw, feat_cols, mask_train, mask_val, mask_test, device
    )

    # 4. Model
    n_feats  = len(feat_cols)
    model    = DualContextLSTMModel(n_feats=n_feats).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("[MODEL] DualContextLSTMModel | Parameters: {:,}".format(n_params))

    # 5. Train
    model, train_losses, val_losses = train_model(
        model, dl_train, dl_val, device, MODEL_DIR
    )

    # 6. Evaluate
    df_test_df = df_raw[mask_test].reset_index(drop=True)
    metrics, y_true, y_pred = evaluate(
        model, dl_test, tgt_scaler, df_test_df, device, MODEL_DIR
    )

    # 7. Plot
    plot_results(train_losses, val_losses, y_true, y_pred, metrics, MODEL_DIR)

    # 8. Save
    save_artifacts(model, feat_scaler, tgt_scaler, feat_cols, metrics, MODEL_DIR, n_feats)

    log.info("=" * 64)
    log.info(
        "TRAINING COMPLETE — MAE=%.2f | RMSE=%.2f | R²=%.4f | WAPE=%.2f%% | PeakMAPE=%.2f%%",
        metrics["MAE"], metrics["RMSE"], metrics["R2"],
        metrics["WAPE"], metrics["PeakMAPE"],
    )
    log.info("  Artifacts → %s", MODEL_DIR)
    log.info("=" * 64)


# ══════════════════════════════════════════════════════════════════════
# MAIN (re-evaluate) — load model đã train từ thư mục cũ, chạy lại
# evaluate + plot + save_artifacts mà KHÔNG train lại.
# Dùng khi: train xong nhưng evaluate/plot bị lỗi, cần chạy lại.
# Chạy: python train_lstm_demand_v5.py  (hoặc --mode eval)
# ══════════════════════════════════════════════════════════════════════
def main():
    # --- SỬA TẠI ĐÂY: trỏ vào thư mục chứa best_lstm.pt cần re-evaluate ---
    OLD_MODEL_FOLDER = "lstm_v5_20260430_032038"
    # -----------------------------------------------------------------------

    global MODEL_DIR, BATCH_SIZE
    MODEL_DIR = Path("/app/models") / OLD_MODEL_FOLDER

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"[EVAL] Không tìm thấy thư mục model: {MODEL_DIR}\n"
            f"       Kiểm tra lại OLD_MODEL_FOLDER hoặc chạy main_train() trước."
        )

    best_weights = MODEL_DIR / "best_lstm.pt"
    if not best_weights.exists():
        raise FileNotFoundError(
            f"[EVAL] Không tìm thấy best_lstm.pt trong {MODEL_DIR}\n"
            f"       Training chưa hoàn tất hoặc checkpoint bị mất."
        )

    device = configure_device()
    if device.type == "cpu":
        BATCH_SIZE = 512
        log.warning("[CFG] CPU mode → batch_size giảm về %d", BATCH_SIZE)

    log.info("=" * 64)
    log.info("LSTM v5  RE-EVALUATE (không train lại)")
    log.info("  Model dir : %s", MODEL_DIR)
    log.info("  Weights   : %s", best_weights)
    log.info("=" * 64)

    # 1. Load + preprocess (cần để rebuild DataLoader và scaler)
    df_raw, feat_cols = preprocess(load_gold_data())

    # 2. Split — dùng cùng logic split để dl_test khớp với lúc train
    mask_train, mask_val, mask_test = split_data(df_raw)

    # 3. Data pipeline — chỉ cần dl_test và tgt_scaler, nhưng
    #    build_data_pipeline fit scaler trên train → phải build đủ 3 set
    dl_train, dl_val, dl_test, feat_scaler, tgt_scaler = build_data_pipeline(
        df_raw, feat_cols, mask_train, mask_val, mask_test, device
    )

    # 4. Rebuild model và load trọng số
    n_feats  = len(feat_cols)
    model    = DualContextLSTMModel(n_feats=n_feats).to(device)
    model.load_state_dict(torch.load(best_weights, map_location=device))
    log.info("[LOAD] Đã load trọng số từ %s", best_weights)

    # 5. Evaluate
    df_test_df = df_raw[mask_test].reset_index(drop=True)
    metrics, y_true, y_pred = evaluate(
        model, dl_test, tgt_scaler, df_test_df, device, MODEL_DIR
    )

    # 6. Plot — dùng loss giả vì không có train history
    #    (loss curve sẽ phẳng — chấp nhận được khi mục đích chỉ là xem forecast)
    plot_results([0.0], [0.0], y_true, y_pred, metrics, MODEL_DIR)

    # 7. Ghi lại metrics (overwrite file cũ nếu có)
    save_artifacts(model, feat_scaler, tgt_scaler, feat_cols, metrics, MODEL_DIR, n_feats)

    log.info("=" * 64)
    log.info(
        "RE-EVALUATE COMPLETE — MAE=%.2f | RMSE=%.2f | R²=%.4f | WAPE=%.2f%% | PeakMAPE=%.2f%%",
        metrics["MAE"], metrics["RMSE"], metrics["R2"],
        metrics["WAPE"], metrics["PeakMAPE"],
    )
    log.info("  Artifacts updated in → %s", MODEL_DIR)
    log.info("=" * 64)


# ══════════════════════════════════════════════════════════════════════
# ENTRYPOINT
#   python train_lstm_demand_v5.py            → re-evaluate (main)
#   python train_lstm_demand_v5.py --train    → train mới  (main_train)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main_train()
