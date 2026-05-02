"""
train_lstm_demand_v4.py — LSTM thuần cho NYC Taxi Demand Forecast
══════════════════════════════════════════════════════════════════════
Vấn đề của LSTM v3 (dùng lại params từ LightGBM):
  ✗ look_back=168 quá dài → gradient vanishing qua 168 bước LSTM
  ✗ Không có zone embedding → model xử lý tất cả zones đồng nhất
  ✗ batch_size quá nhỏ → GPU RTX 3090/4090 24GB bị underutilize
  ✗ learning_rate cố định → không decay → không hội tụ tốt
  ✗ Dùng MinMaxScaler thuần → taxi demand right-skewed cần log1p trước
  ✗ Không có Attention → tất cả timestep có weight bằng nhau

Giải pháp v4 — LSTM-native design:
  ✓ look_back=48 (2 ngày): đủ bắt daily cycle, tránh vanishing gradient
  ✓ Zone Embedding (dim=16): học đặc thù từng zone, không trộn lẫn
  ✓ Stacked Bi-LSTM: 2 layers, dropout giữa layer
  ✓ Temporal Attention: học timestep nào quan trọng nhất
  ✓ Log1p + StandardScaler: xử lý phân phối lệch taxi demand
  ✓ CosineAnnealingWarmRestarts: learning rate tự adapt
  ✓ Tiered Huber Loss: low×2, normal×1, peak×6 (đồng bộ LightGBM)
  ✓ batch_size=2048 cho 24GB VRAM: tối đa throughput GPU
  ✓ Mixed precision (fp16): tăng tốc 2× trên GPU CUDA
  ✓ DataLoader num_workers=8: pipeline data không bị GPU chờ CPU
  ✓ Checkpoint + Early stopping: đề phòng crash

Hardware target:
  RAM 56GB | VRAM 24GB (RTX 3090/4090) | CPU i7-12700F (12P+4E cores)

Features dùng từ Gold layer:
  Temporal sequences (look_back=48 bước):
    trip_count, lag_1h, lag_2h, rolling_avg_3h
    hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
    is_weekend, is_holiday
  Static per-zone (zone embedding):
    PULocationID → Embedding(263, 16) → concat vào LSTM output

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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
# CONFIG — LSTM-native (không dùng params từ LightGBM)
# ══════════════════════════════════════════════════════════════════════
GOLD_SRC  = "s3://lakehouse/gold/demand_by_zone"
HOUR_BUCKET = os.getenv("HOUR_BUCKET", "all").strip().lower()
_BUCKET_MAP = {
    "all": None,
    "00_09": set(range(0, 10)),
    "10_16": set(range(10, 17)),
    "17_23": set(range(17, 24)),
}
if HOUR_BUCKET not in _BUCKET_MAP:
    raise ValueError(f"HOUR_BUCKET={HOUR_BUCKET!r} không hợp lệ. Dùng: {list(_BUCKET_MAP.keys())}")
_bucket_suffix = "" if HOUR_BUCKET == "all" else f"_{HOUR_BUCKET}"
MODEL_DIR = Path("/app/models") / f"lstm_v4{_bucket_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

STORAGE_OPTIONS = {
    "endpoint_url"             : "http://minio:9000",
    "access_key_id"            : "minioadmin",
    "secret_access_key"        : "minioadmin123",
    "region"                   : "us-east-1",
    "allow_http"               : "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

# ── Data config ───────────────────────────────────────────────────────
ZONE_DEMAND_MIN = 5.0     # Lọc zone thưa (đồng bộ LightGBM)
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15

# ── Sequence config ───────────────────────────────────────────────────
# 48 bước = 2 ngày × 24h:
#   - Đủ bắt daily cycle (peak sáng + chiều)
#   - Đủ ngắn để gradient lan ngược không bị vanish
#   - 168 bước (1 tuần) trong v3 quá dài → LSTM cells forget quá nhiều
LOOK_BACK   = 96

# Features từ Gold layer (không thêm lag dài hạn vì LSTM đã xử lý chuỗi)
# LightGBM cần lag_24h/lag_168h vì nó chỉ thấy 1 điểm tại thời điểm t.
# LSTM thấy toàn bộ 48 bước nên không cần lag tổng hợp thêm.
FEATURE_COLS = [
    "trip_count",           # chuỗi demand chính — context cho prediction
    "lag_24h",    # thêm — cùng giờ hôm qua
    "lag_168h",   # thêm — cùng giờ tuần trước (quan trọng nhất)
    "lag_1h", "lag_2h",     # Gold đã có, dùng trực tiếp
    "rolling_avg_3h",       # short-term trend
    "rolling_avg_24h",  # thêm — trend ngày
    "hour_sin", "hour_cos", # cyclical time features
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_holiday",
]

# ── Zone Embedding ────────────────────────────────────────────────────
# Zone embedding học đặc thù từng khu vực NYC (≈54 active zones).
# LightGBM dùng PULocationID làm integer feature → tree splits.
# LSTM không thể dùng integer trực tiếp → dùng Embedding layer:
#   - 263 zones (max LocationID NYC)
#   - dim=16: đủ biểu diễn đặc thù zone mà không overfit
MAX_LOCATION_ID = 263
ZONE_EMB_DIM    = 16      # kích thước zone embedding

# ── LSTM Architecture ─────────────────────────────────────────────────
# num_layers=2: stacked LSTM bắt pattern phức tạp hơn 1 layer
# hidden_size=256: cân bằng capacity vs overfit (128 quá nhỏ, 512 quá lớn)
# dropout=0.3: regularization giữa các LSTM layers
# Bidirectional=True: Bi-LSTM đọc chuỗi 2 chiều → tốt hơn vanilla LSTM
#   → output size = hidden_size * 2 = 512 (concat forward+backward)
LSTM_HIDDEN     = 256
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.3
LSTM_BIDIR      = True    # Bidirectional LSTM

# Attention: Temporal attention học timestep nào quan trọng nhất
# → cải thiện đáng kể so với chỉ lấy hidden state cuối (v3)
ATTN_DIM        = 128

# FC head sau khi concat [LSTM output + zone embedding]
FC_HIDDEN       = 128
MODEL_ARCH      = "bucket_heads"   # single_head | bucket_heads

# ── Training config ───────────────────────────────────────────────────
# batch_size=2048: tối đa throughput GPU 24GB VRAM
#   Ước tính VRAM: 2048 × 48 × 12 × float32 × 2 (forward+back) ≈ 9MB
#   Model parameters (LSTM 256-hidden, 2 layers, Bi) ≈ 3-5MB
#   Tổng ≈ 14MB << 24GB VRAM → có thể tăng thêm nếu muốn
BATCH_SIZE      = 2048

# epochs=100 với early stopping patience=15:
#   Cosine LR đảm bảo model khám phá đủ không gian trước khi dừng
MAX_EPOCHS      = 100
PATIENCE        = 7     # early stopping

# Learning rate: 3e-3 thường là điểm xuất phát tốt cho Adam + LSTM
#   (LightGBM dùng 0.05 → không áp dụng cho neural net)
INIT_LR         = 3e-3
MIN_LR          = 1e-5   # cosine annealing floor

# CosineAnnealingWarmRestarts: T_0=10 (restart mỗi 10 epochs)
# Giúp model thoát local minima, không cần manual LR schedule
COSINE_T0       = 10

# Gradient clipping: quan trọng cho LSTM — ngăn exploding gradient
GRAD_CLIP       = 1.0

# Tiered Huber Loss (đồng bộ LightGBM sample weights)
# delta=1.0: chuyển từ L2 → L1 tại |error|=1 (robust với outlier demand)
HUBER_DELTA     = 1.0
PEAK_WEIGHT     = 6.0    # y > p75 (giờ cao điểm)
LOW_WEIGHT      = 2.0    # y < p25 (giờ thấp điểm — cải thiện MAPE)
# Hour-aware weighting + asymmetric penalty:
# Tập trung giờ khó thực tế: 00-03, 04-08 và 22-23.
DEEP_NIGHT_HOUR_WEIGHT = 2.2   # 00-03
DAWN_HOUR_WEIGHT = 1.8         # 04-08
LATE_NIGHT_HOUR_WEIGHT = 2.0   # 22-23
# Asym penalties
NIGHT_OVER_PENALTY = 0.45
EVENING_EARLY_OVER_PENALTY = 0.15
EVENING_LATE_OVER_PENALTY = 0.15
EVENING_LATE_UNDER_PENALTY = 0.55

# Mixed precision: tăng tốc ~2× trên GPU CUDA (fp16 computation, fp32 master weights)
USE_AMP         = True

# DataLoader workers: i7-12700F có 12P+4E cores, nhưng trong Docker /dev/shm
# bị giới hạn. Mỗi worker cần shared memory để truyền tensor qua pin_memory.
# Ước tính: 4 workers × batch=2048 × 48 × 12 × float32 × 2 buffers ≈ 2.4GB shm
# → 4 workers là điểm cân bằng tốt: GPU vẫn không idle, shm không quá tải.
# (docker-compose.yml đã set shm_size=8g, nhưng giữ workers=4 làm safety margin)
DATALOADER_WORKERS = 4

# Checkpoint
CHECKPOINT_EVERY = 10    # lưu checkpoint mỗi N epochs
# Two-phase loss schedule:
# - Phase 1: ổn định hóa backbone trên toàn bộ phân phối
# - Phase 2: tập trung giờ khó (00-09, 17-23) bằng hour/asym terms mạnh hơn
PHASE1_EPOCHS = 6
CRITICAL_FINE_TUNE_EPOCHS = int(os.getenv("CRITICAL_FINE_TUNE_EPOCHS", "6"))
CRITICAL_FINE_TUNE_LR = float(os.getenv("CRITICAL_FINE_TUNE_LR", "8e-4"))
CRITICAL_SAMPLER_MULT = float(os.getenv("CRITICAL_SAMPLER_MULT", "3.0"))


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
        log.info("[HW] batch_size=%d → estimated VRAM usage ≈ %.1f GB",
                 BATCH_SIZE,
                 BATCH_SIZE * LOOK_BACK * len(FEATURE_COLS) * 4 * 2 / 1024**3)
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

    # Train theo bucket giờ nếu được chỉ định
    bucket_hours = _BUCKET_MAP.get(HOUR_BUCKET)
    if bucket_hours is not None:
        n_before_bucket = len(df)
        df = df[df["hour"].isin(bucket_hours)].reset_index(drop=True)
        log.info("[PREP] Hour bucket=%s (%s) | %d→%d rows",
                 HOUR_BUCKET, sorted(bucket_hours), n_before_bucket, len(df))

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
    Xây dựng sequences (look_back, n_features) per-zone.

    Tại sao per-zone (không shuffle toàn bộ):
      - LSTM học temporal pattern → phải đảm bảo tính liên tục chuỗi
      - Nếu trộn zones, window [i-48:i] có thể span 2 zones khác nhau
      - LightGBM không cần điều này vì mỗi sample độc lập

    Zone ID được encode dưới dạng integer để feed vào Embedding layer.
    """
    def __init__(
        self,
        df:          pd.DataFrame,
        feat_cols:   list[str],
        feat_scaler: StandardScaler,
        tgt_scaler:  Log1pStandardScaler,
        look_back:   int = LOOK_BACK,
    ):
        self.look_back = look_back
        self.samples: list[tuple[np.ndarray, int, int, int, float]] = []

        feat_raw   = df[feat_cols].values.astype(np.float32)
        feat_sc    = feat_scaler.transform(feat_raw)
        tgt_raw    = df["target_demand"].values.reshape(-1, 1).astype(np.float32)
        tgt_sc     = tgt_scaler.transform(tgt_raw).flatten().astype(np.float32)
        zone_ids   = df["PULocationID"].values.astype(np.int64)
        hour_ids   = df["hour"].values.astype(np.int64)
        dow_ids    = df["day_of_week"].values.astype(np.int64)

        for zone in df["PULocationID"].unique():
            mask = df["PULocationID"].values == zone
            idx  = np.where(mask)[0]
            for j in range(look_back, len(idx)):
                i_start  = idx[j - look_back]
                i_end    = idx[j]
                # Kiểm tra continuity trong zone
                if i_end - i_start != look_back:
                    continue
                seq      = feat_sc[i_start:i_end, :]         # (look_back, n_feats)
                zone_id  = int(zone_ids[i_end])
                hour_id  = int(hour_ids[i_end])
                dow_id   = int(dow_ids[i_end])
                target   = float(tgt_sc[i_end])
                self.samples.append((seq, zone_id, hour_id, dow_id, target))

        log.info("[DATASET] Tổng sequences: %d", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, zone_id, hour_id, dow_id, target = self.samples[idx]
        return (
            torch.from_numpy(seq),            # (look_back, n_feats)
            torch.tensor(zone_id, dtype=torch.long),
            torch.tensor(hour_id, dtype=torch.long),
            torch.tensor(dow_id, dtype=torch.long),
            torch.tensor(target,  dtype=torch.float32),
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


class LSTMDemandModel(nn.Module):
    """
    Architecture:
      [seq: (look_back, n_feats)] → Bi-LSTM × 2 layers → Attention → FC
                    ↑
      [zone_id: int] → Embedding(263, 16) → concat với attention output
                    ↓
                 FC(lstm_out + emb_dim → 128 → 1)

    Tại sao Bidirectional:
      - Trong training, model thấy toàn bộ sequence 48 bước
      - Bi-LSTM đọc từ trái qua phải VÀ phải qua trái
      - Pattern "demand giảm trước khi tăng đột biến" → bi-LSTM bắt được
      - LightGBM không có khái niệm này → đây là lợi thế LSTM thuần

    Tại sao Zone Embedding:
      - Zone 1 (downtown) vs Zone 200 (airport): pattern hoàn toàn khác
      - Embedding dim=16 học "profile" của mỗi zone
      - LightGBM dùng integer PULocationID trực tiếp → tree splits
      - LSTM concat embedding → cùng FC head nhưng zone-aware
    """
    def __init__(self, n_feats: int, n_locations: int = MAX_LOCATION_ID, architecture: str = "bucket_heads"):
        super().__init__()
        self.architecture = architecture
        lstm_out_dim = LSTM_HIDDEN * (2 if LSTM_BIDIR else 1)

        # Zone Embedding
        self.zone_emb = nn.Embedding(n_locations + 1, ZONE_EMB_DIM, padding_idx=0)

        # Stacked Bi-LSTM
        self.lstm = nn.LSTM(
            input_size   = n_feats,
            hidden_size  = LSTM_HIDDEN,
            num_layers   = LSTM_LAYERS,
            batch_first  = True,
            bidirectional= LSTM_BIDIR,
            dropout      = LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )

        # Temporal Attention
        self.attention = TemporalAttention(lstm_out_dim, ATTN_DIM)

        # FC head: concat [attention_out + zone_emb]
        fc_in = lstm_out_dim + ZONE_EMB_DIM
        self.fc = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Linear(fc_in, FC_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(FC_HIDDEN, 1),
        )
        if self.architecture == "bucket_heads":
            self.head_00_09 = nn.Sequential(
                nn.LayerNorm(fc_in),
                nn.Linear(fc_in, FC_HIDDEN),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(FC_HIDDEN, 1),
            )
            self.head_10_16 = nn.Sequential(
                nn.LayerNorm(fc_in),
                nn.Linear(fc_in, FC_HIDDEN),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(FC_HIDDEN, 1),
            )
            self.head_17_23 = nn.Sequential(
                nn.LayerNorm(fc_in),
                nn.Linear(fc_in, FC_HIDDEN),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(FC_HIDDEN, 1),
            )

    def forward(self, seq: torch.Tensor, zone_id: torch.Tensor, hour_id: torch.Tensor | None = None) -> torch.Tensor:
        # seq: (batch, look_back, n_feats)
        lstm_out, _ = self.lstm(seq)          # (batch, look_back, lstm_out_dim)
        context     = self.attention(lstm_out) # (batch, lstm_out_dim)
        zone_vec    = self.zone_emb(zone_id)   # (batch, emb_dim)
        combined    = torch.cat([context, zone_vec], dim=-1)  # (batch, lstm_out+emb)
        if self.architecture != "bucket_heads" or hour_id is None:
            out = self.fc(combined)
            return out.squeeze(-1)

        out = self.fc(combined).squeeze(-1)  # fallback head
        m_00_09 = (hour_id >= 0) & (hour_id <= 9)
        m_10_16 = (hour_id >= 10) & (hour_id <= 16)
        m_17_23 = (hour_id >= 17) & (hour_id <= 23)
        if m_00_09.any():
            out[m_00_09] = self.head_00_09(combined[m_00_09]).squeeze(-1)
        if m_10_16.any():
            out[m_10_16] = self.head_10_16(combined[m_10_16]).squeeze(-1)
        if m_17_23.any():
            out[m_17_23] = self.head_17_23(combined[m_17_23]).squeeze(-1)
        return out


# ══════════════════════════════════════════════════════════════════════
# 7. TIERED HUBER LOSS
#
# LightGBM dùng sample_weight ngoài model.
# PyTorch LSTM: implement loss nội tại để gradient phản ánh đúng priority.
# Giữ nguyên ngưỡng p25/p75 và trọng số LOW/PEAK để đồng bộ LightGBM.
# ══════════════════════════════════════════════════════════════════════
class TieredHuberLoss(nn.Module):
    """
    Huber loss với sample weight 3 tầng:
      LOW  (y < p25): ×2.0 — cải thiện MAPE giờ thấp điểm
      MID  (p25–p75): ×1.0
      PEAK (y > p75): ×6.0 — giữ PeakMAPE thấp

    Đây là phần thay thế cho sample_weight trong LightGBM.
    Quan trọng: p25/p75 tính trên batch (không toàn dataset) để gradient
    ổn định hơn khi demand phân bố khác nhau giữa train/val.
    """
    def __init__(self, delta: float = HUBER_DELTA):
        super().__init__()
        self.delta = delta
        self.phase = 1

    def set_phase(self, phase: int):
        # phase=1: nền tổng quát, phase=2: tập trung giờ khó
        self.phase = 1 if phase <= 1 else 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor, hour_ids: torch.Tensor) -> torch.Tensor:
        huber = F.huber_loss(pred, target, delta=self.delta, reduction="none")
        # Tính percentile trên batch
        p25 = target.quantile(0.25)
        p75 = target.quantile(0.75)
        weight = torch.ones_like(target)
        weight[target < p25] = LOW_WEIGHT
        weight[target > p75] = PEAK_WEIGHT
        is_deep_night = ((hour_ids >= 0) & (hour_ids <= 3)).float()
        is_night = ((hour_ids >= 0) & (hour_ids <= 8)).float()
        is_dawn = ((hour_ids >= 4) & (hour_ids <= 8)).float()
        is_evening_late = ((hour_ids >= 22) & (hour_ids <= 23)).float()
        hour_weight = torch.ones_like(target)
        hour_weight = hour_weight + (DEEP_NIGHT_HOUR_WEIGHT - 1.0) * is_deep_night
        hour_weight = hour_weight + (DAWN_HOUR_WEIGHT - 1.0) * is_dawn
        hour_weight = hour_weight + (LATE_NIGHT_HOUR_WEIGHT - 1.0) * is_evening_late
        base = (huber * weight * hour_weight).mean()

        # Asymmetric hour-bucket correction
        is_evening_early = ((hour_ids >= 17) & (hour_ids <= 20)).float()
        is_evening_late = ((hour_ids >= 21) & (hour_ids <= 23)).float()
        over = F.relu(pred - target)
        under = F.relu(target - pred)
        asym = (
            NIGHT_OVER_PENALTY * (is_night * over.pow(2)).mean()
            + EVENING_EARLY_OVER_PENALTY * (is_evening_early * over.pow(2)).mean()
            + EVENING_LATE_OVER_PENALTY * (is_evening_late * over.pow(2)).mean()
            + EVENING_LATE_UNDER_PENALTY * (is_evening_late * under.pow(2)).mean()
        )
        if self.phase == 1:
            return base + 0.35 * asym
        return base + asym


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

    # Datasets
    kwargs_ds = dict(feat_cols=feat_cols, feat_scaler=feat_scaler,
                     tgt_scaler=tgt_scaler, look_back=LOOK_BACK)
    ds_train = TaxiDemandDataset(df_train, **kwargs_ds)
    ds_val   = TaxiDemandDataset(df_val,   **kwargs_ds)
    ds_test  = TaxiDemandDataset(df_test,  **kwargs_ds)

    pin_mem = device.type == "cuda"
    nw      = DATALOADER_WORKERS if device.type == "cuda" else 0

    critical_hours = set(range(0, 9)) | {21, 22, 23}
    sample_weights = []
    for _seq, _zone_id, hour_id, _dow_id, _target in ds_train.samples:
        w = CRITICAL_SAMPLER_MULT if int(hour_id) in critical_hours else 1.0
        sample_weights.append(w)
    train_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler,
                          num_workers=nw, pin_memory=pin_mem, persistent_workers=nw > 0)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE * 2, shuffle=False,
                          num_workers=nw, pin_memory=pin_mem, persistent_workers=nw > 0)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE * 2, shuffle=False,
                          num_workers=nw, pin_memory=pin_mem, persistent_workers=nw > 0)

    log.info("[DATA] Train samples=%d | Val=%d | Test=%d",
             len(ds_train), len(ds_val), len(ds_test))
    return dl_train, dl_val, dl_test, feat_scaler, tgt_scaler


# ══════════════════════════════════════════════════════════════════════
# 9. TRAIN LOOP
# ══════════════════════════════════════════════════════════════════════
def run_epoch(
    model:      LSTMDemandModel,
    loader:     DataLoader,
    criterion:  TieredHuberLoss,
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
        for seq, zone_id, hour_id, _dow_id, target in loader:
            seq     = seq.to(device, non_blocking=True)
            zone_id = zone_id.to(device, non_blocking=True)
            hour_id = hour_id.to(device, non_blocking=True)
            target  = target.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)  # set_to_none=True: tiết kiệm VRAM

            with autocast(enabled=USE_AMP and device.type == "cuda"):
                pred = model(seq, zone_id, hour_id)
                loss = criterion(pred, target, hour_id)

            if is_train:
                scaler_amp.scale(loss).backward()
                # Gradient clipping: bắt buộc cho LSTM (tránh exploding gradient)
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
    criterion = TieredHuberLoss(delta=HUBER_DELTA)

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

    best_val_loss = float("inf")
    patience_cnt  = 0
    train_losses  = []
    val_losses    = []
    best_epoch    = 0

    log.info("[TRAIN] Bắt đầu training — epochs=%d | patience=%d | lr=%.4f",
             MAX_EPOCHS, PATIENCE, INIT_LR)
    log.info("[TRAIN] Gradient clip=%.1f | AMP=%s | batch=%d | look_back=%d",
             GRAD_CLIP, USE_AMP, BATCH_SIZE, LOOK_BACK)
    log.info("[TRAIN] Two-phase loss: phase1_epochs=%d (base-heavy) -> phase2 (hard-hour focus)",
             PHASE1_EPOCHS)

    for epoch in range(1, MAX_EPOCHS + 1):
        phase = 1 if epoch <= PHASE1_EPOCHS else 2
        criterion.set_phase(phase)
        t_loss = run_epoch(model, dl_train, criterion, optimizer,
                           amp_scaler, device, is_train=True)
        v_loss = run_epoch(model, dl_val,   criterion, None,
                           amp_scaler, device, is_train=False)

        scheduler.step()  # cosine step sau mỗi epoch

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        log.info("[EPOCH %3d/%d|phase=%d] train_loss=%.4f  val_loss=%.4f  lr=%.6f",
                 epoch, MAX_EPOCHS, phase, t_loss, v_loss, lr_now)

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

    # Stage-2 fine-tune chuyên biệt giờ khó, không tách script/model riêng
    if CRITICAL_FINE_TUNE_EPOCHS > 0:
        log.info(
            "[FT] Critical-hour fine-tune: epochs=%d | lr=%.6f | sampler_mult=%.2f",
            CRITICAL_FINE_TUNE_EPOCHS, CRITICAL_FINE_TUNE_LR, CRITICAL_SAMPLER_MULT
        )
        criterion.set_phase(2)
        ft_optimizer = torch.optim.Adam(
            model.parameters(), lr=CRITICAL_FINE_TUNE_LR, weight_decay=1e-4
        )
        ft_best = float("inf")
        ft_best_path = model_dir / "best_lstm_critical.pt"
        for ft_epoch in range(1, CRITICAL_FINE_TUNE_EPOCHS + 1):
            t_loss = run_epoch(model, dl_train, criterion, ft_optimizer, amp_scaler, device, is_train=True)
            v_loss = run_epoch(model, dl_val, criterion, None, amp_scaler, device, is_train=False)
            train_losses.append(t_loss)
            val_losses.append(v_loss)
            log.info("[FT %2d/%d] train_loss=%.4f val_loss=%.4f", ft_epoch, CRITICAL_FINE_TUNE_EPOCHS, t_loss, v_loss)
            if v_loss < ft_best - 1e-5:
                ft_best = v_loss
                torch.save(model.state_dict(), ft_best_path)
                log.info("[FT] ✓ Best critical model saved (val_loss=%.4f)", v_loss)
        if ft_best_path.exists():
            model.load_state_dict(torch.load(ft_best_path, map_location=device))
            log.info("[FT] Loaded best critical fine-tuned weights")

    return model, train_losses, val_losses


# ══════════════════════════════════════════════════════════════════════
# 10. EVALUATE
# ══════════════════════════════════════════════════════════════════════
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    
    # Tính WAPE (Chính xác nhất cho taxi demand)
    wape = np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1e-9) * 100
    
    # MAPE cổ điển rất dễ "nổ" khi y_true gần 0.
    # Dùng ngưỡng denominator >= 10 để phản ánh tốt hơn cho taxi demand theo giờ.
    mape_denom = np.maximum(y_true, 10.0)
    mape = np.mean(np.abs((y_true - y_pred) / mape_denom)) * 100

    # sMAPE ổn định hơn khi demand thấp
    smape = np.mean(
        2.0 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-9)
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
    
    # Cập nhật dòng log (thêm WAPE vào để theo dõi)
    log.info("[%s] MAE=%.2f | RMSE=%.2f | R²=%.4f | WAPE=%.2f%% | MAPE=%.2f%% | SMAPE=%.2f%% | PeakMAPE=%.2f%%",
             label, mae, rmse, r2, wape, mape, smape, peak_mape)
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "WAPE": wape, "MAPE": mape, "SMAPE": smape, "PeakMAPE": peak_mape}


def fit_hour_calibration(
    model: LSTMDemandModel,
    dl_val: DataLoader,
    tgt_scaler: Log1pStandardScaler,
    device: torch.device,
) -> tuple[dict[int, float], dict[str, float], dict[int, float], dict[int, dict[str, float]]]:
    """
    Fit multiplicative calibration factor theo giờ trên validation:
      factor[h] = sum(y_true_h) / sum(y_pred_h)
    Clamp factor để tránh overfit quá mạnh.
    """
    model.eval()
    preds, trues, hours, dows = [], [], [], []
    with torch.no_grad():
        for seq, zone_id, hour_id, dow_id, target in dl_val:
            seq = seq.to(device, non_blocking=True)
            zone_id = zone_id.to(device, non_blocking=True)
            pred = model(seq, zone_id, hour_id.to(device, non_blocking=True)).cpu().numpy()
            preds.append(pred)
            trues.append(target.numpy())
            hours.append(hour_id.numpy())
            dows.append(dow_id.numpy())

    preds = np.concatenate(preds).reshape(-1, 1)
    trues = np.concatenate(trues).reshape(-1, 1)
    hrs = np.concatenate(hours).astype(np.int64).reshape(-1)
    dws = np.concatenate(dows).astype(np.int64).reshape(-1)

    y_pred = np.clip(tgt_scaler.inverse_transform(preds).flatten(), 0, None)
    y_true = np.clip(tgt_scaler.inverse_transform(trues).flatten(), 0, None)

    factors: dict[int, float] = {}
    for h in range(24):
        m = hrs == h
        if not np.any(m):
            factors[h] = 1.0
            continue
        p = float(np.sum(y_pred[m]))
        t = float(np.sum(y_true[m]))
        if p <= 1e-6 or t <= 1e-6:
            factors[h] = 1.0
        else:
            factors[h] = float(np.clip(t / p, 0.55, 1.45))
    # Chi tiết hơn: factor theo (day_of_week, hour)
    dow_hour_factors: dict[str, float] = {}
    for dow in range(7):
        for h in range(24):
            m = (dws == dow) & (hrs == h)
            if not np.any(m):
                continue
            p = float(np.sum(y_pred[m]))
            t = float(np.sum(y_true[m]))
            if p <= 1e-6 or t <= 1e-6:
                f = 1.0
            else:
                f = float(np.clip(t / p, 0.45, 1.65))
            dow_hour_factors[f"{dow}_{h}"] = f

    # Additive bias chỉ cho giờ khó, để sửa lệch lớn mà không làm nhiễu giờ ổn định
    hour_bias: dict[int, float] = {}
    critical_hours = set(list(range(0, 9)) + [21, 22, 23])
    residual = y_true - y_pred
    for h in range(24):
        if h not in critical_hours:
            hour_bias[h] = 0.0
            continue
        m = hrs == h
        if not np.any(m):
            hour_bias[h] = 0.0
            continue
        b = float(np.median(residual[m]))
        hour_bias[h] = float(np.clip(b, -1200.0, 1200.0))

    # Affine calibration cho giờ khó: pred' = a * pred + b
    # Mục tiêu: sửa bias phi tuyến ở 00-08 và 21-23 tốt hơn factor đơn thuần.
    hour_affine: dict[int, dict[str, float]] = {}
    for h in sorted(critical_hours):
        m = hrs == h
        if not np.any(m):
            continue
        x = y_pred[m].astype(np.float64)
        y = y_true[m].astype(np.float64)
        if len(x) < 16:
            continue
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        var_x = float(np.mean((x - x_mean) ** 2))
        if var_x < 1e-9:
            a = 1.0
        else:
            cov_xy = float(np.mean((x - x_mean) * (y - y_mean)))
            a = cov_xy / var_x
        b = y_mean - a * x_mean
        hour_affine[h] = {
            "a": float(np.clip(a, 0.70, 1.20)),
            "b": float(np.clip(b, -1500.0, 1500.0)),
        }

    log.info("[CALIB] Hour factors fitted (val): %s", factors)
    log.info("[CALIB] DOW×hour factors fitted (count=%d)", len(dow_hour_factors))
    log.info("[CALIB] Hour bias fitted (critical hours): %s", {k: hour_bias[k] for k in sorted(critical_hours)})
    log.info("[CALIB] Hour affine fitted (critical hours): %s", hour_affine)
    return factors, dow_hour_factors, hour_bias, hour_affine


def evaluate(
    model:      LSTMDemandModel,
    dl_test:    DataLoader,
    tgt_scaler: Log1pStandardScaler,
    df_test:    pd.DataFrame,
    device:     torch.device,
    model_dir:  Path,
    hour_factors: dict[int, float] | None = None,
    hour_bias: dict[int, float] | None = None,
    hour_affine: dict[int, dict[str, float]] | None = None,
) -> dict:
    model.eval()
    preds, trues, hours = [], [], []

    with torch.no_grad():
        for seq, zone_id, hour_id, _dow_id, target in dl_test:
            seq     = seq.to(device, non_blocking=True)
            zone_id = zone_id.to(device, non_blocking=True)
            pred    = model(seq, zone_id, hour_id.to(device, non_blocking=True)).cpu().numpy()
            preds.append(pred)
            trues.append(target.numpy())
            hours.append(hour_id.numpy())

    preds = np.concatenate(preds).reshape(-1, 1)
    trues = np.concatenate(trues).reshape(-1, 1)
    hour_arr = np.concatenate(hours).astype(np.int64).reshape(-1)

    # Inverse transform (log1p + standard)
    y_pred = np.clip(tgt_scaler.inverse_transform(preds).flatten(), 0, None)
    y_true = np.clip(tgt_scaler.inverse_transform(trues).flatten(), 0, None)
    critical_hours = set(range(0, 9)) | {21, 22, 23}
    if hour_affine:
        y_adj = y_pred.copy()
        for h, params in hour_affine.items():
            hh = int(h)
            m = hour_arr == hh
            if not np.any(m):
                continue
            a = float(params.get("a", 1.0))
            b = float(params.get("b", 0.0))
            y_adj[m] = a * y_adj[m] + b
        y_pred = np.maximum(y_adj, 0.0)
    if hour_factors or hour_bias:
        calib = np.ones_like(y_pred, dtype=np.float32)
        bias = np.zeros_like(y_pred, dtype=np.float32)
        if hour_factors:
            calib = np.array([float(hour_factors.get(int(h), 1.0)) for h in hour_arr], dtype=np.float32)
        if hour_bias:
            bias = np.array([float(hour_bias.get(int(h), 0.0)) for h in hour_arr], dtype=np.float32)
        non_critical = np.array([int(h) not in critical_hours for h in hour_arr], dtype=bool)
        y_pred[non_critical] = np.maximum(y_pred[non_critical] * calib[non_critical] + bias[non_critical], 0.0)

    metrics = _compute_metrics(y_true, y_pred, "LSTM-v4-TEST")
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
    fig.suptitle("LSTM v4 Demand Forecast — NYC Taxi", fontsize=14, fontweight="bold")

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
    path = model_dir / "forecast_result_v4.png"
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
    hour_calibration: dict[int, float] | None = None,
    dow_hour_calibration: dict[str, float] | None = None,
    hour_bias: dict[int, float] | None = None,
    hour_affine: dict[int, dict[str, float]] | None = None,
):
    # Lưu model dưới dạng TorchScript để inference không cần code nguồn
    model.eval()
    torch.save(model.state_dict(), model_dir / "lstm_weights.pt")

    # Lưu config để rebuild model khi load
    model_cfg = {
        "n_feats"       : n_feats,
        "n_locations"   : MAX_LOCATION_ID,
        "lstm_hidden"   : LSTM_HIDDEN,
        "lstm_layers"   : LSTM_LAYERS,
        "lstm_dropout"  : LSTM_DROPOUT,
        "lstm_bidir"    : LSTM_BIDIR,
        "attn_dim"      : ATTN_DIM,
        "zone_emb_dim"  : ZONE_EMB_DIM,
        "fc_hidden"     : FC_HIDDEN,
        "architecture"  : MODEL_ARCH,
        "look_back"     : LOOK_BACK,
        "feat_cols"     : feat_cols,
        "hour_bucket"   : HOUR_BUCKET,
        "bucket_hours"  : sorted(_BUCKET_MAP[HOUR_BUCKET]) if _BUCKET_MAP[HOUR_BUCKET] is not None else list(range(24)),
    }
    joblib.dump(model_cfg,   model_dir / "model_config.pkl")
    joblib.dump(feat_scaler, model_dir / "feature_scaler.pkl")
    joblib.dump(tgt_scaler,  model_dir / "target_scaler.pkl")
    joblib.dump(feat_cols,   model_dir / "feature_cols.pkl")
    if hour_calibration:
        joblib.dump(hour_calibration, model_dir / "hour_calibration.pkl")
    if dow_hour_calibration:
        joblib.dump(dow_hour_calibration, model_dir / "dow_hour_calibration.pkl")
    if hour_bias:
        joblib.dump(hour_bias, model_dir / "hour_bias.pkl")
    if hour_affine:
        joblib.dump(hour_affine, model_dir / "hour_affine.pkl")
    pd.DataFrame([metrics]).to_csv(model_dir / "metrics.csv", index=False)

    (model_dir / "model_info.txt").write_text(
        f"LSTM v4 — Bi-LSTM + Temporal Attention + Zone Embedding\n"
        f"look_back     : {LOOK_BACK}\n"
        f"hidden        : {LSTM_HIDDEN} × {'Bi' if LSTM_BIDIR else ''}{LSTM_LAYERS} layers\n"
        f"zone_emb_dim  : {ZONE_EMB_DIM}\n"
        f"batch_size    : {BATCH_SIZE}\n"
        f"scaler        : Log1pStandardScaler (log1p + StandardScaler)\n"
        f"MAE           : {metrics['MAE']:.2f}\n"
        f"R2            : {metrics['R2']:.4f}\n"
        f"PeakMAPE      : {metrics['PeakMAPE']:.2f}%\n"
    )
    log.info("[SAVE] Artifacts → %s", model_dir)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = configure_device()

    # Giảm batch_size trên CPU để không OOM
    global BATCH_SIZE
    if device.type == "cpu":
        BATCH_SIZE = 512
        log.warning("[CFG] CPU mode → batch_size giảm về %d", BATCH_SIZE)

    log.info("=" * 64)
    log.info("LSTM v4 Demand Forecast  START")
    log.info("  Hour bucket  : %s", HOUR_BUCKET)
    log.info("  Gold source  : %s", GOLD_SRC)
    log.info("  look_back    : %d (2 ngày × 24h)", LOOK_BACK)
    log.info("  LSTM hidden  : %d | layers=%d | Bi=%s",
             LSTM_HIDDEN, LSTM_LAYERS, LSTM_BIDIR)
    log.info("  Zone emb dim : %d", ZONE_EMB_DIM)
    log.info("  Batch size   : %d | AMP=%s", BATCH_SIZE, USE_AMP)
    log.info("  LR           : %.4f → %.6f (CosineWarmRestart T0=%d)",
             INIT_LR, MIN_LR, COSINE_T0)
    log.info("  Loss         : TieredHuber(delta=%.1f, peak×%.0f, low×%.0f) + HourWeight(deep_night=%.2f, dawn=%.2f, late_night=%.2f) + Asym(night_over=%.2f, eve17_20_over=%.2f, eve21_23_over=%.2f, eve21_23_under=%.2f)",
             HUBER_DELTA, PEAK_WEIGHT, LOW_WEIGHT,
             DEEP_NIGHT_HOUR_WEIGHT, DAWN_HOUR_WEIGHT, LATE_NIGHT_HOUR_WEIGHT,
             NIGHT_OVER_PENALTY, EVENING_EARLY_OVER_PENALTY, EVENING_LATE_OVER_PENALTY, EVENING_LATE_UNDER_PENALTY)
    log.info("  Model dir    : %s", MODEL_DIR)
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
    n_feats = len(feat_cols)
    model   = LSTMDemandModel(n_feats=n_feats, architecture=MODEL_ARCH).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("[MODEL] Parameters: {:,}".format(n_params))

    # 5. Train
    model, train_losses, val_losses = train_model(
        model, dl_train, dl_val, device, MODEL_DIR
    )

    # 5.1 Hour calibration fit trên validation để giảm bias theo giờ
    hour_calibration, dow_hour_calibration, hour_bias, hour_affine = fit_hour_calibration(model, dl_val, tgt_scaler, device)

    # 6. Evaluate
    df_test_df = df_raw[mask_test].reset_index(drop=True)
    metrics, y_true, y_pred = evaluate(
        model, dl_test, tgt_scaler, df_test_df, device, MODEL_DIR,
        hour_factors=hour_calibration, hour_bias=hour_bias, hour_affine=hour_affine
    )

    # 7. Plot
    plot_results(train_losses, val_losses, y_true, y_pred, metrics, MODEL_DIR)

    # 8. Save
    save_artifacts(
        model,
        feat_scaler,
        tgt_scaler,
        feat_cols,
        metrics,
        MODEL_DIR,
        n_feats,
        hour_calibration=hour_calibration,
        dow_hour_calibration=dow_hour_calibration,
        hour_bias=hour_bias,
        hour_affine=hour_affine,
    )

    log.info("=" * 64)
    log.info(
        "TRAINING COMPLETE — MAE=%.2f | RMSE=%.2f | R²=%.4f | MAPE=%.2f%% | PeakMAPE=%.2f%%",
        metrics["MAE"], metrics["RMSE"], metrics["R2"],
        metrics["MAPE"], metrics["PeakMAPE"],
    )
    log.info("  Artifacts → %s", MODEL_DIR)
    log.info("=" * 64)
def main1():
    # --- SỬA TẠI ĐÂY ---
    # Thay vì dùng thư mục mới tạo theo datetime, hãy trỏ thẳng vào thư mục cũ của bạn
    # Lưu ý: Dùng đường dẫn Linux bên trong Docker (/app/models/...)
    OLD_MODEL_FOLDER = "lstm_v4_20260430_075139" # Tên thư mục bạn muốn load
    
    global MODEL_DIR
    MODEL_DIR = Path("/app/models") / OLD_MODEL_FOLDER
    # ------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = configure_device()

    # ... (giữ nguyên các phần log cấu hình) ...

    # 1. Load + preprocess
    df_raw, feat_cols = preprocess(load_gold_data())

    # 2. Split
    mask_train, mask_val, mask_test = split_data(df_raw)

    # 3. Data pipeline
    dl_train, dl_val, dl_test, feat_scaler, tgt_scaler = build_data_pipeline(
        df_raw, feat_cols, mask_train, mask_val, mask_test, device
    )

    # 4. Model
    n_feats = len(feat_cols)
    model   = LSTMDemandModel(n_feats=n_feats, architecture=MODEL_ARCH).to(device)
    
    # --- LOGIC RESUME ---
    best_weights = MODEL_DIR / "best_lstm.pt"
    train_losses, val_losses = [], []

    if best_weights.exists():
        log.info("[LOAD] Tìm thấy trọng số tại: %s", best_weights)
        model.load_state_dict(torch.load(best_weights, map_location=device))
        # Tạo dữ liệu loss giả để không lỗi hàm vẽ biểu đồ
        train_losses, val_losses = [0.05], [0.05] 
    else:
        # Nếu vẫn không tìm thấy, nó sẽ tự động train lại vào thư mục này
        log.info("[TRAIN] Không tìm thấy file tại %s. Bắt đầu train mới...", best_weights)
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
        "COMPLETE — MAE=%.2f | R²=%.4f | WAPE=%.2f%% | PeakMAPE=%.2f%%",
        metrics["MAE"], metrics["R2"], metrics["WAPE"], metrics["PeakMAPE"]
    )
    log.info("  Artifacts stored in → %s", MODEL_DIR)
    log.info("=" * 64)
    
if __name__ == "__main__":
    main()