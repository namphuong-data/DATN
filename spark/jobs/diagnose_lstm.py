"""
diagnose_lstm_v3.py — Kiểm tra toàn diện tham số & chất lượng dữ liệu
═══════════════════════════════════════════════════════════════════════════════
Mục đích:
  Chạy TRƯỚC hoặc SAU khi train để chẩn đoán các vấn đề tiềm ẩn trong:
    1. Phân phối dữ liệu & target (spike, outlier, imbalance)
    2. Tham số scaling & peak threshold trong loss
    3. Feature correlation & leakage risk
    4. Sequence generator (zone boundary, valid indices ratio)
    5. Kiến trúc model & tham số huấn luyện
    6. (Nếu có model đã train) Đánh giá lại bias hệ thống

Cách dùng:
  # Chế độ 1 — Chỉ cần CSV/Parquet (không cần Delta Lake):
  python diagnose_lstm_v3.py --data path/to/data.parquet

  # Chế độ 2 — Dùng Delta Lake (giống train script):
  python diagnose_lstm_v3.py --delta

  # Chế độ 3 — Kèm model đã train:
  python diagnose_lstm_v3.py --data path/to/data.parquet --model /app/models/lstm_v3_xxx

  # Chế độ 4 — Dùng metrics.csv có sẵn:
  python diagnose_lstm_v3.py --data path/to/data.parquet --metrics metrics.csv

Kết quả:
  - In report chi tiết ra console với màu sắc (PASS / WARN / FAIL)
  - Lưu report vào diagnose_report_<timestamp>.txt
  - Lưu plots PNG vào diagnose_plots_<timestamp>.png
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
# ANSI màu cho console
# ══════════════════════════════════════════════════════════
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _c(text, color): return f"{color}{text}{RESET}"
def PASS(msg):  return f"  {_c('✅ PASS', GREEN)}  {msg}"
def WARN(msg):  return f"  {_c('⚠️  WARN', YELLOW)}  {msg}"
def FAIL(msg):  return f"  {_c('❌ FAIL', RED)}  {msg}"
def INFO(msg):  return f"  {_c('ℹ️  INFO', CYAN)}  {msg}"
def HEAD(msg):  return f"\n{_c('═'*60, CYAN)}\n{_c(msg, BOLD)}\n{_c('═'*60, CYAN)}"

# ══════════════════════════════════════════════════════════
# CONFIG — copy từ train_lstm_demand.py (giữ sync)
# ══════════════════════════════════════════════════════════
LOOK_BACK        = 72
BATCH_SIZE       = 1024
LEARNING_RATE    = 1e-3
EPOCHS           = 200
LSTM_UNITS       = [256, 128]
DROPOUT_RATE     = 0.2
GRAD_CLIP        = 0.5
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
PEAK_LOSS_WEIGHT = 3.0
HUBER_DELTA      = 0.1
USE_ROBUST_SCALER = True

FEATURE_COLS = [
    "lag_1h", "lag_2h", "lag_3h",
    "lag_24h", "lag_168h",
    "rolling_avg_3h", "rolling_avg_6h", "rolling_avg_24h",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin","month_cos",
    "is_weekend", "is_holiday",
]
OPTIONAL_COLS = {"lag_3h", "lag_168h", "rolling_avg_6h", "rolling_avg_24h", "is_holiday"}

# Ngưỡng kiểm tra (có thể điều chỉnh)
THRESHOLDS = {
    "peak_pct_min"       : 3.0,   # % samples > 0.5 scaled: ít hơn → peak weight vô hiệu
    "leak_corr_max"      : 0.85,  # Pearson corr tối đa trước khi coi là leakage
    "valid_ratio_min"    : 0.50,  # % valid sequences tối thiểu
    "peak_mape_warn"     : 20.0,  # PeakMAPE > ngưỡng này → WARN
    "peak_mape_fail"     : 35.0,  # PeakMAPE > ngưỡng này → FAIL
    "r2_warn"            : 0.90,  # R² < → WARN
    "train_val_gap_warn" : 0.20,  # val_loss / train_loss > 1+gap → WARN
    "spike_ratio_warn"   : 10.0,  # max/p95 > ratio → spike outlier nặng
}

# ══════════════════════════════════════════════════════════
# Log1pMinMaxScaler (copy từ train script)
# ══════════════════════════════════════════════════════════
class Log1pMinMaxScaler:
    def __init__(self):
        self._scaler = MinMaxScaler()
    def fit(self, X):
        self._scaler.fit(np.log1p(X)); return self
    def transform(self, X):
        return self._scaler.transform(np.log1p(X))
    def inverse_transform(self, X):
        return np.expm1(self._scaler.inverse_transform(X))
    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ══════════════════════════════════════════════════════════
# REPORT COLLECTOR
# ══════════════════════════════════════════════════════════
class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    def add(self, line: str):
        self.lines.append(line)
        print(line)

    def head(self, title: str):
        h = HEAD(title)
        self.lines.append(h)
        print(h)

    def ok(self, msg: str):
        self.counts["PASS"] += 1
        self.add(PASS(msg))

    def warn(self, msg: str):
        self.counts["WARN"] += 1
        self.add(WARN(msg))

    def fail(self, msg: str):
        self.counts["FAIL"] += 1
        self.add(FAIL(msg))

    def info(self, msg: str):
        self.add(INFO(msg))

    def save(self, path: Path):
        # Lưu bản plain text (không ANSI)
        import re
        ansi_escape = re.compile(r'\033\[[0-9;]*m')
        plain = [ansi_escape.sub('', l) for l in self.lines]
        path.write_text("\n".join(plain), encoding="utf-8")
        print(f"\n{_c('📄 Report saved →', CYAN)} {path}")

    def summary(self):
        self.head("📋 TỔNG KẾT")
        self.add(f"  PASS: {_c(self.counts['PASS'], GREEN)}  "
                 f"WARN: {_c(self.counts['WARN'], YELLOW)}  "
                 f"FAIL: {_c(self.counts['FAIL'], RED)}")
        if self.counts["FAIL"] > 0:
            self.add(_c("\n  → Có vấn đề nghiêm trọng cần xử lý trước khi retrain!", RED))
        elif self.counts["WARN"] > 0:
            self.add(_c("\n  → Một số điểm cần xem xét, xem chi tiết ở trên.", YELLOW))
        else:
            self.add(_c("\n  → Tất cả tham số ổn định. Model sẵn sàng!", GREEN))


# ══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════
def load_data(args) -> pd.DataFrame:
    if args.delta:
        try:
            from deltalake import DeltaTable
            STORAGE_OPTIONS = {
                "endpoint_url": "http://minio:9000",
                "access_key_id": "minioadmin",
                "secret_access_key": "minioadmin123",
                "region": "us-east-1",
                "allow_http": "true",
                "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
            }
            GOLD_SRC = "s3://lakehouse/gold/demand_by_zone"
            dt = DeltaTable(GOLD_SRC, storage_options=STORAGE_OPTIONS)
            df = dt.to_pandas()
            print(INFO(f"Loaded từ Delta Lake: {len(df):,} rows"))
            return df
        except Exception as e:
            print(FAIL(f"Không thể load Delta Lake: {e}"))
            sys.exit(1)
    elif args.data:
        p = Path(args.data)
        if not p.exists():
            print(FAIL(f"File không tồn tại: {p}"))
            sys.exit(1)
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
        else:
            print(FAIL(f"Định dạng không hỗ trợ: {p.suffix}"))
            sys.exit(1)
        print(INFO(f"Loaded từ {p.name}: {len(df):,} rows"))
        return df
    else:
        print(FAIL("Phải cung cấp --data hoặc --delta"))
        sys.exit(1)


# ══════════════════════════════════════════════════════════
# 2. KIỂM TRA DỮ LIỆU & CỘT
# ══════════════════════════════════════════════════════════
def check_columns(df: pd.DataFrame, rpt: Report):
    rpt.head("1. KIỂM TRA CỘT DỮ LIỆU")

    required_base = ["target_demand", "window_start", "PULocationID"]
    for col in required_base:
        if col in df.columns:
            rpt.ok(f"Cột bắt buộc '{col}' tồn tại")
        else:
            rpt.fail(f"Thiếu cột bắt buộc: '{col}'")

    present  = [c for c in FEATURE_COLS if c in df.columns]
    missing  = [c for c in FEATURE_COLS if c not in df.columns]
    optional_missing = [c for c in missing if c in OPTIONAL_COLS]
    required_missing = [c for c in missing if c not in OPTIONAL_COLS]

    rpt.info(f"Features có sẵn ({len(present)}/{len(FEATURE_COLS)}): {present}")

    if required_missing:
        rpt.fail(f"Thiếu required features: {required_missing}")
    else:
        rpt.ok("Tất cả required features có mặt")

    if optional_missing:
        rpt.warn(f"Thiếu optional features (sẽ bỏ qua): {optional_missing}")
    else:
        rpt.ok("Tất cả optional features có mặt")

    # NaN check
    nan_counts = df[present + ["target_demand"]].isnull().sum()
    nan_cols   = nan_counts[nan_counts > 0]
    if nan_cols.empty:
        rpt.ok("Không có NaN trong features & target")
    else:
        for col, cnt in nan_cols.items():
            pct = cnt / len(df) * 100
            msg = f"'{col}' có {cnt:,} NaN ({pct:.1f}%)"
            rpt.warn(msg) if pct < 5 else rpt.fail(msg)


# ══════════════════════════════════════════════════════════
# 3. PHÂN PHỐI TARGET & SPIKE
# ══════════════════════════════════════════════════════════
def check_target_distribution(df: pd.DataFrame, rpt: Report) -> dict:
    rpt.head("2. PHÂN PHỐI TARGET (target_demand)")

    tgt = df["target_demand"].dropna().values
    stats = {
        "count": len(tgt),
        "min"  : tgt.min(),
        "max"  : tgt.max(),
        "mean" : tgt.mean(),
        "std"  : tgt.std(),
        "p50"  : np.percentile(tgt, 50),
        "p75"  : np.percentile(tgt, 75),
        "p90"  : np.percentile(tgt, 90),
        "p95"  : np.percentile(tgt, 95),
        "p99"  : np.percentile(tgt, 99),
    }

    rpt.info(f"Count  : {stats['count']:,}")
    rpt.info(f"Min    : {stats['min']:.1f}  |  Max : {stats['max']:.1f}")
    rpt.info(f"Mean   : {stats['mean']:.1f}  |  Std : {stats['std']:.1f}")
    rpt.info(f"p50    : {stats['p50']:.1f}  |  p75 : {stats['p75']:.1f}  |  p90 : {stats['p90']:.1f}  |  p95 : {stats['p95']:.1f}  |  p99 : {stats['p99']:.1f}")

    spike_ratio = stats["max"] / (stats["p95"] + 1e-9)
    if spike_ratio > THRESHOLDS["spike_ratio_warn"]:
        rpt.warn(f"Spike ratio (max/p95) = {spike_ratio:.1f}x — outlier rất lớn, log1p transform là bắt buộc")
    else:
        rpt.ok(f"Spike ratio (max/p95) = {spike_ratio:.1f}x — phân phối chấp nhận được")

    zero_pct = np.mean(tgt == 0) * 100
    rpt.info(f"Zero demand : {zero_pct:.1f}% của tổng samples")
    if zero_pct > 30:
        rpt.warn(f"Zero demand > 30% — cân nhắc zero-inflated model hoặc filter zone thưa")

    return stats


# ══════════════════════════════════════════════════════════
# 4. KIỂM TRA SCALING & PEAK THRESHOLD
# ══════════════════════════════════════════════════════════
def check_scaling(df: pd.DataFrame, rpt: Report) -> dict:
    rpt.head("3. KIỂM TRA SCALING & PEAK THRESHOLD")

    tgt = df["target_demand"].dropna().values.reshape(-1, 1)

    # Split train để fit scaler đúng cách (không look-ahead)
    n = len(tgt)
    n_train = int(n * TRAIN_RATIO)
    tgt_train = tgt[:n_train]

    # Log1pMinMaxScaler (cách v3 làm)
    log_scaler = Log1pMinMaxScaler().fit(tgt_train)
    tgt_scaled = log_scaler.transform(tgt).flatten()

    # Thống kê scaled
    stats_scaled = {
        "min": tgt_scaled.min(),
        "max": tgt_scaled.max(),
        "mean": tgt_scaled.mean(),
        "p50": np.percentile(tgt_scaled, 50),
        "p90": np.percentile(tgt_scaled, 90),
        "p95": np.percentile(tgt_scaled, 95),
    }

    rpt.info(f"Sau Log1pMinMax — min={stats_scaled['min']:.4f}  max={stats_scaled['max']:.4f}  "
             f"mean={stats_scaled['mean']:.4f}  p50={stats_scaled['p50']:.4f}  p90={stats_scaled['p90']:.4f}")

    # Kiểm tra ceiling của scaler
    max_recoverable = log_scaler.inverse_transform(np.array([[1.0]]))[0, 0]
    rpt.info(f"Giá trị gốc tương ứng scaled=1.0 : {max_recoverable:.1f} trips")
    if abs(max_recoverable - tgt.max()) > tgt.max() * 0.05:
        rpt.warn(f"Scaler ceiling ({max_recoverable:.1f}) ≠ data max ({tgt.max():.1f}) — có thể bị cắt ngọn spike")
    else:
        rpt.ok(f"Scaler ceiling khớp data max (±5%): {max_recoverable:.1f} ≈ {tgt.max():.1f}")

    # ── Vấn đề cốt lõi: peak threshold 0.5
    pct_above_05 = np.mean(tgt_scaled > 0.5) * 100
    rpt.info(f"% samples có scaled target > 0.5 (hardcode threshold trong loss) : {pct_above_05:.2f}%")

    if pct_above_05 < THRESHOLDS["peak_pct_min"]:
        rpt.fail(
            f"Chỉ {pct_above_05:.2f}% samples > 0.5 → peak_weight={PEAK_LOSS_WEIGHT}x gần như VÔ HIỆU.\n"
            f"         Khuyến nghị: đổi threshold sang percentile động (p90 = {stats_scaled['p90']:.3f})"
        )
    elif pct_above_05 < 10:
        rpt.warn(
            f"{pct_above_05:.2f}% samples > 0.5 — peak weight ít tác dụng.\n"
            f"         Xem xét dùng: y_true > {stats_scaled['p90']:.3f} (p90 của scaled) thay vì 0.5"
        )
    else:
        rpt.ok(f"{pct_above_05:.2f}% samples vượt threshold 0.5 — peak weight hoạt động")

    # Gợi ý threshold tốt hơn
    suggested_threshold = stats_scaled["p90"]
    pct_with_suggested  = np.mean(tgt_scaled > suggested_threshold) * 100
    rpt.info(f"Ngưỡng gợi ý (p90 scaled = {suggested_threshold:.3f}) → bắt được {pct_with_suggested:.1f}% samples làm 'peak'")

    # So sánh MinMax thuần vs Log1p+MinMax
    mm_scaler  = MinMaxScaler().fit(tgt_train)
    tgt_mm     = mm_scaler.transform(tgt).flatten()
    pct_mm_05  = np.mean(tgt_mm > 0.5) * 100
    rpt.info(f"[So sánh] MinMax thuần: {pct_mm_05:.2f}% > 0.5  |  Log1p+MinMax: {pct_above_05:.2f}% > 0.5")
    if pct_above_05 > pct_mm_05:
        rpt.ok("Log1p transform đang giúp phân phối đều hơn so với MinMax thuần")
    else:
        rpt.warn("Log1p không cải thiện phân phối so với MinMax thuần — kiểm tra lại data")

    return {"tgt_scaled": tgt_scaled, "log_scaler": log_scaler, "stats_scaled": stats_scaled,
            "suggested_threshold": suggested_threshold, "n_train": n_train}


# ══════════════════════════════════════════════════════════
# 5. FEATURE CORRELATION & LEAKAGE
# ══════════════════════════════════════════════════════════
def check_leakage(df: pd.DataFrame, rpt: Report) -> pd.Series:
    rpt.head("4. FEATURE CORRELATION & LEAKAGE CHECK")

    active_cols = [c for c in FEATURE_COLS if c in df.columns]
    target = df["target_demand"]
    corr_results = {}

    for col in active_cols:
        try:
            c = df[col].corr(target)
            corr_results[col] = c
        except Exception:
            corr_results[col] = np.nan

    corr_series = pd.Series(corr_results).sort_values(ascending=False)

    for col, c in corr_series.items():
        if np.isnan(c):
            rpt.warn(f"'{col}' — không tính được correlation")
        elif abs(c) > THRESHOLDS["leak_corr_max"]:
            rpt.fail(f"'{col}' corr={c:.3f} — NGHI NGỜ LEAKAGE (>{THRESHOLDS['leak_corr_max']})")
        elif abs(c) > 0.70:
            rpt.warn(f"'{col}' corr={c:.3f} — cao, cần kiểm tra kỹ shift logic")
        else:
            rpt.ok(f"'{col}' corr={c:.3f}")

    top3 = corr_series.abs().nlargest(3)
    rpt.info(f"Top 3 correlated features: {dict(top3.round(3))}")

    # Kiểm tra lag shift có đúng không
    if "lag_1h" in df.columns:
        sample_diff = (df["target_demand"] - df["lag_1h"].shift(-1)).abs().mean()
        if sample_diff < 1e-6:
            rpt.fail("lag_1h == target_demand shifted — BUG: lag chưa shift đúng, leakage!")
        else:
            rpt.ok(f"lag_1h shift logic hợp lệ (mean diff với target: {sample_diff:.2f})")

    return corr_series


# ══════════════════════════════════════════════════════════
# 6. SPLIT & SEQUENCE GENERATOR
# ══════════════════════════════════════════════════════════
def check_splits_and_sequences(df: pd.DataFrame, rpt: Report):
    rpt.head("5. SPLIT & SEQUENCE VALIDITY")

    df = df.copy()
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
    df = df.sort_values(["PULocationID", "window_start"]).reset_index(drop=True)

    timestamps = df["window_start"]
    t_min, t_max = timestamps.min(), timestamps.max()
    t_range  = t_max - t_min
    train_end = t_min + t_range * TRAIN_RATIO
    val_end   = t_min + t_range * (TRAIN_RATIO + VAL_RATIO)

    mask_train = timestamps < train_end
    mask_val   = (timestamps >= train_end) & (timestamps < val_end)
    mask_test  = timestamps >= val_end

    n_train, n_val, n_test = mask_train.sum(), mask_val.sum(), mask_test.sum()
    rpt.info(f"Train: {n_train:,} ({n_train/len(df)*100:.1f}%)  |  "
             f"Val: {n_val:,} ({n_val/len(df)*100:.1f}%)  |  "
             f"Test: {n_test:,} ({n_test/len(df)*100:.1f}%)")
    rpt.info(f"Train range : {t_min.date()} → {train_end.date()}")
    rpt.info(f"Val range   : {train_end.date()} → {val_end.date()}")
    rpt.info(f"Test range  : {val_end.date()} → {t_max.date()}")

    if n_test < LOOK_BACK * 2:
        rpt.fail(f"Test set {n_test} rows < 2×LOOK_BACK={2*LOOK_BACK} — quá nhỏ để evaluate")
    else:
        rpt.ok(f"Test set đủ lớn: {n_test:,} rows")

    # Kiểm tra peak có trong test không
    if "target_demand" in df.columns:
        train_max = df.loc[mask_train, "target_demand"].max()
        test_max  = df.loc[mask_test,  "target_demand"].max()
        test_pct_peak = (df.loc[mask_test, "target_demand"] >
                         df["target_demand"].quantile(0.90)).mean() * 100
        rpt.info(f"Test max demand: {test_max:.1f}  |  Train max: {train_max:.1f}")
        rpt.info(f"Peak samples (>p90) trong test set: {test_pct_peak:.1f}%")
        if test_pct_peak < 5:
            rpt.warn("Test set có rất ít peak — PeakMAPE metric sẽ không đại diện")
        else:
            rpt.ok(f"Test set có đủ peak samples ({test_pct_peak:.1f}%)")

    # Kiểm tra valid sequences (zone boundary)
    if "PULocationID" in df.columns:
        zone_ids   = df["PULocationID"].values
        global_idx = np.where(mask_train)[0]
        valid = []
        for i in global_idx:
            if i < LOOK_BACK:
                continue
            window = zone_ids[i - LOOK_BACK : i + 1]
            if np.all(window == window[0]):
                valid.append(i)

        valid_ratio = len(valid) / max(len(global_idx), 1) * 100
        rpt.info(f"Valid train sequences: {len(valid):,} / {len(global_idx):,} ({valid_ratio:.1f}%)")

        if valid_ratio < THRESHOLDS["valid_ratio_min"] * 100:
            rpt.warn(
                f"Chỉ {valid_ratio:.1f}% sequences hợp lệ — nhiều sequences bị loại do zone boundary.\n"
                f"         Xem xét sort data theo (zone, time) hoặc train single-zone."
            )
        else:
            rpt.ok(f"{valid_ratio:.1f}% sequences hợp lệ sau zone boundary check")

        # Phân tích số zone
        n_zones = df["PULocationID"].nunique()
        rpt.info(f"Số zone: {n_zones} | LOOK_BACK: {LOOK_BACK}h — cần {n_zones * LOOK_BACK:,} rows tối thiểu để cover all zones")
    else:
        rpt.warn("Không có cột 'PULocationID' — bỏ qua zone boundary check")


# ══════════════════════════════════════════════════════════
# 7. THAM SỐ MODEL & TRAINING
# ══════════════════════════════════════════════════════════
def check_model_params(rpt: Report):
    rpt.head("6. THAM SỐ MODEL & TRAINING")

    rpt.info(f"LOOK_BACK        = {LOOK_BACK}h")
    rpt.info(f"LSTM_UNITS       = {LSTM_UNITS}")
    rpt.info(f"DROPOUT_RATE     = {DROPOUT_RATE}")
    rpt.info(f"BATCH_SIZE       = {BATCH_SIZE}")
    rpt.info(f"LEARNING_RATE    = {LEARNING_RATE}")
    rpt.info(f"GRAD_CLIP        = {GRAD_CLIP}")
    rpt.info(f"EPOCHS           = {EPOCHS}")
    rpt.info(f"PEAK_LOSS_WEIGHT = {PEAK_LOSS_WEIGHT}")
    rpt.info(f"HUBER_DELTA      = {HUBER_DELTA}")
    rpt.info(f"USE_ROBUST_SCALER= {USE_ROBUST_SCALER}")

    # Kiểm tra từng tham số
    if LOOK_BACK < 24:
        rpt.warn(f"LOOK_BACK={LOOK_BACK}h ngắn — không capture daily pattern (cần ≥ 24h)")
    elif LOOK_BACK > 168:
        rpt.warn(f"LOOK_BACK={LOOK_BACK}h dài quá — tốn bộ nhớ, có thể inject noise xa")
    else:
        rpt.ok(f"LOOK_BACK={LOOK_BACK}h hợp lý cho taxi demand (24–168h)")

    if HUBER_DELTA > 1.0:
        rpt.fail(f"HUBER_DELTA={HUBER_DELTA} quá lớn cho target ∈ [0,1] — sẽ hoạt động như MSE thuần")
    elif HUBER_DELTA < 0.05:
        rpt.warn(f"HUBER_DELTA={HUBER_DELTA} rất nhỏ — có thể quá sensitive với noise nhỏ")
    else:
        rpt.ok(f"HUBER_DELTA={HUBER_DELTA} phù hợp với [0,1] scaled range")

    if PEAK_LOSS_WEIGHT < 2.0:
        rpt.warn(f"PEAK_LOSS_WEIGHT={PEAK_LOSS_WEIGHT} thấp — ít tác dụng penalize peak miss")
    elif PEAK_LOSS_WEIGHT > 10.0:
        rpt.warn(f"PEAK_LOSS_WEIGHT={PEAK_LOSS_WEIGHT} rất cao — training có thể mất ổn định")
    else:
        rpt.ok(f"PEAK_LOSS_WEIGHT={PEAK_LOSS_WEIGHT} trong range hợp lý [2, 10]")

    if DROPOUT_RATE > 0.5:
        rpt.warn(f"DROPOUT_RATE={DROPOUT_RATE} cao — có thể underfit")
    else:
        rpt.ok(f"DROPOUT_RATE={DROPOUT_RATE} ổn")

    if LEARNING_RATE > 5e-3:
        rpt.warn(f"LEARNING_RATE={LEARNING_RATE} cao — kết hợp cosine decay có thể diverge ban đầu")
    else:
        rpt.ok(f"LEARNING_RATE={LEARNING_RATE} ổn với Adam + cosine decay")

    # Ước tính tham số model
    n_feat = len([c for c in FEATURE_COLS])
    params_lstm1 = 4 * (LSTM_UNITS[0] * (n_feat + LSTM_UNITS[0] + 1))
    params_lstm2 = 4 * (LSTM_UNITS[1] * (LSTM_UNITS[0] + LSTM_UNITS[1] + 1))
    params_dense = 256 * LSTM_UNITS[1] + 256 + 64 * 256 + 64 + 1 * 64 + 1
    total_params  = params_lstm1 + params_lstm2 + params_dense
    rpt.info(f"Ước tính tổng tham số: ~{total_params/1e6:.2f}M (LSTM1≈{params_lstm1/1e6:.2f}M, LSTM2≈{params_lstm2/1e6:.2f}M)")

    if total_params > 5e6:
        rpt.warn("Model > 5M params — cân nhắc giảm LSTM_UNITS nếu data nhỏ")
    else:
        rpt.ok(f"Kích thước model hợp lý: ~{total_params/1e6:.2f}M params")


# ══════════════════════════════════════════════════════════
# 8. KIỂM TRA METRICS (nếu có file metrics.csv)
# ══════════════════════════════════════════════════════════
def check_metrics(metrics_path: str, rpt: Report):
    rpt.head("7. KIỂM TRA KẾT QUẢ METRICS")

    try:
        m = pd.read_csv(metrics_path).iloc[0]
    except Exception as e:
        rpt.warn(f"Không đọc được metrics file: {e}")
        return

    rpt.info(f"MAE      = {m.get('MAE', 'N/A'):.4f}")
    rpt.info(f"RMSE     = {m.get('RMSE', 'N/A'):.4f}")
    rpt.info(f"R²       = {m.get('R2', 'N/A'):.4f}")
    rpt.info(f"MAPE     = {m.get('MAPE', 'N/A'):.2f}%")
    rpt.info(f"PeakMAPE = {m.get('PeakMAPE', 'N/A'):.2f}%")

    r2 = m.get("R2", 0)
    if r2 < THRESHOLDS["r2_warn"]:
        rpt.fail(f"R²={r2:.4f} < {THRESHOLDS['r2_warn']} — model chất lượng kém tổng thể")
    else:
        rpt.ok(f"R²={r2:.4f} đạt ngưỡng (>{THRESHOLDS['r2_warn']})")

    peak_mape = m.get("PeakMAPE", 0)
    if peak_mape > THRESHOLDS["peak_mape_fail"]:
        rpt.fail(f"PeakMAPE={peak_mape:.1f}% > {THRESHOLDS['peak_mape_fail']}% — model không bắt được peak")
    elif peak_mape > THRESHOLDS["peak_mape_warn"]:
        rpt.warn(f"PeakMAPE={peak_mape:.1f}% > {THRESHOLDS['peak_mape_warn']}% — peak prediction yếu")
    else:
        rpt.ok(f"PeakMAPE={peak_mape:.1f}% trong ngưỡng chấp nhận được")

    # RMSE/MAE ratio
    rmse = m.get("RMSE", 0); mae = m.get("MAE", 1)
    ratio = rmse / (mae + 1e-9)
    rpt.info(f"RMSE/MAE ratio = {ratio:.1f}x (>5 → spike error chiếm dominant)")
    if ratio > 5:
        rpt.warn(f"RMSE/MAE={ratio:.1f}x rất cao — lỗi tập trung ở spike, tăng PEAK_LOSS_WEIGHT")
    else:
        rpt.ok(f"RMSE/MAE ratio = {ratio:.1f}x ổn")


# ══════════════════════════════════════════════════════════
# 9. KIỂM TRA MODEL ĐÃ TRAIN (nếu có)
# ══════════════════════════════════════════════════════════
def check_trained_model(model_dir: str, df: pd.DataFrame, rpt: Report):
    rpt.head("8. KIỂM TRA MODEL ĐÃ TRAIN")

    model_path = Path(model_dir)
    if not model_path.exists():
        rpt.warn(f"Model dir không tồn tại: {model_path}")
        return

    # Kiểm tra artifacts
    artifacts = {
        "lstm_demand_v3.keras": "Model weights",
        "feature_scaler.pkl"  : "Feature scaler",
        "target_scaler.pkl"   : "Target scaler",
        "feature_cols.pkl"    : "Feature columns list",
        "scaler_info.txt"     : "Scaler info note",
        "metrics.csv"         : "Metrics file",
    }
    for fname, desc in artifacts.items():
        fpath = model_path / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            rpt.ok(f"{desc} ({fname}) tồn tại — {size_kb:.1f} KB")
        else:
            rpt.warn(f"Thiếu artifact: {fname} ({desc})")

    # Load và kiểm tra scaler
    tgt_scaler_path = model_path / "target_scaler.pkl"
    feat_cols_path  = model_path / "feature_cols.pkl"

    if tgt_scaler_path.exists():
        try:
            tgt_scaler = joblib.load(tgt_scaler_path)
            max_val = tgt_scaler.inverse_transform(np.array([[1.0]]))[0, 0]
            rpt.ok(f"Target scaler load thành công — scaled=1.0 → {max_val:.1f} trips")
            data_max = df["target_demand"].max() if "target_demand" in df.columns else None
            if data_max and abs(max_val - data_max) / (data_max + 1e-9) > 0.1:
                rpt.warn(f"Scaler max ({max_val:.1f}) lệch >10% so với data max ({data_max:.1f}) — "
                         "scaler có thể fit trên subset khác")
        except Exception as e:
            rpt.fail(f"Không load được target scaler: {e}")

    if feat_cols_path.exists():
        try:
            saved_cols = joblib.load(feat_cols_path)
            missing_in_data = [c for c in saved_cols if c not in df.columns]
            rpt.info(f"Feature cols được lưu ({len(saved_cols)}): {saved_cols}")
            if missing_in_data:
                rpt.fail(f"Features trong model không có trong data hiện tại: {missing_in_data}")
            else:
                rpt.ok("Tất cả saved features có trong data hiện tại")
        except Exception as e:
            rpt.fail(f"Không load được feature_cols.pkl: {e}")


# ══════════════════════════════════════════════════════════
# 10. KIẾN NGHỊ CẢI THIỆN
# ══════════════════════════════════════════════════════════
def print_recommendations(scaling_info: dict, rpt: Report):
    rpt.head("9. KIẾN NGHỊ CẢI THIỆN")

    suggested_thr = scaling_info.get("suggested_threshold", 0.5)
    pct_above_05  = scaling_info.get("stats_scaled", {})

    rpt.add(f"""
  {_c('A. Sửa Peak Threshold trong loss (ưu tiên cao nhất):', BOLD)}
     Thay dòng trong make_peak_weighted_huber():
       TRƯỚC: weights = tf.where(y_true > 0.5, peak_weight, 1.0)
       SAU  : threshold = tf.reduce_mean(y_true) + tf.math.reduce_std(y_true)
              weights = tf.where(y_true > threshold, peak_weight, 1.0)

  {_c('B. Oversample peak sequences:', BOLD)}
     Trong build_generators(), sau khi tính idx_train:
       peak_mask = tgt_scaled[idx_train] > np.percentile(tgt_scaled[idx_train], 90)
       idx_train_aug = np.concatenate([idx_train, np.tile(idx_train[peak_mask], 4)])

  {_c('C. Thêm peak-sensitive features:', BOLD)}
     df["rolling_max_3h"]  = df["lag_1h"].rolling(3).max().fillna(0)
     df["rolling_max_6h"]  = df["lag_1h"].rolling(6).max().fillna(0)
     df["ewm_alpha03"]     = df["target_demand"].shift(1).ewm(alpha=0.3).mean().fillna(0)
     df["demand_pct_ch1h"] = (df["lag_1h"] - df["lag_2h"]) / (df["lag_2h"] + 1)

  {_c('D. Điều chỉnh hyperparameter gợi ý:', BOLD)}
     PEAK_LOSS_WEIGHT = 6.0   # tăng từ 3.0
     LOOK_BACK        = 48    # thử giảm từ 72
     DROPOUT_RATE     = 0.25  # tăng nhẹ để giảm overfit low-demand
""")


# ══════════════════════════════════════════════════════════
# 11. VẼ DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════════════════
def plot_diagnostics(df: pd.DataFrame, tgt_scaled: np.ndarray, corr_series: pd.Series,
                     output_path: Path):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("LSTM v3 — Diagnostic Plots", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    tgt = df["target_demand"].dropna().values

    # 1. Target distribution (raw)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(tgt, bins=80, color="#2196F3", alpha=0.8, edgecolor="none")
    ax1.axvline(np.percentile(tgt, 90), color="red", linestyle="--", linewidth=1.5, label="p90")
    ax1.axvline(np.percentile(tgt, 99), color="orange", linestyle="--", linewidth=1.5, label="p99")
    ax1.set_title("Target Distribution (raw)", fontsize=10)
    ax1.set_xlabel("trip_count"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 2. Target distribution (log1p)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(np.log1p(tgt), bins=80, color="#4CAF50", alpha=0.8, edgecolor="none")
    ax2.set_title("Target Distribution (log1p)", fontsize=10)
    ax2.set_xlabel("log1p(trip_count)"); ax2.grid(alpha=0.3)

    # 3. Scaled target distribution + peak threshold
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(tgt_scaled, bins=80, color="#9C27B0", alpha=0.8, edgecolor="none")
    ax3.axvline(0.5, color="red", linestyle="--", linewidth=2, label=f"hardcode 0.5\n({np.mean(tgt_scaled>0.5)*100:.1f}%)")
    p90_scaled = np.percentile(tgt_scaled, 90)
    ax3.axvline(p90_scaled, color="orange", linestyle="--", linewidth=2,
                label=f"p90={p90_scaled:.3f}\n({np.mean(tgt_scaled>p90_scaled)*100:.1f}%)")
    ax3.set_title("Scaled Target + Peak Threshold", fontsize=10)
    ax3.set_xlabel("scaled value"); ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    # 4. Feature correlation bar chart
    ax4 = fig.add_subplot(gs[1, :2])
    active = corr_series.dropna()
    colors = ["#ef4444" if abs(v) > 0.85 else "#f59e0b" if abs(v) > 0.70 else "#4ade80"
              for v in active.values]
    bars = ax4.barh(active.index, active.values, color=colors, edgecolor="none", height=0.6)
    ax4.axvline(0.85, color="red", linestyle="--", linewidth=1, alpha=0.7, label="leakage threshold")
    ax4.axvline(-0.85, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax4.set_title("Feature Correlation với target_demand", fontsize=10)
    ax4.set_xlabel("Pearson r"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3, axis="x")

    # 5. Spike analysis — CDF
    ax5 = fig.add_subplot(gs[1, 2])
    sorted_tgt = np.sort(tgt)
    cdf = np.arange(1, len(sorted_tgt)+1) / len(sorted_tgt)
    ax5.plot(sorted_tgt, cdf, color="#2196F3", linewidth=1.5)
    ax5.axhline(0.90, color="orange", linestyle="--", linewidth=1, label="p90")
    ax5.axhline(0.99, color="red",    linestyle="--", linewidth=1, label="p99")
    ax5.set_title("CDF của target_demand", fontsize=10)
    ax5.set_xlabel("trip_count"); ax5.set_ylabel("CDF")
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    # 6. Theo giờ nếu có 'hour' hoặc window_start
    ax6 = fig.add_subplot(gs[2, :])
    if "hour" in df.columns:
        hourly = df.groupby("hour")["target_demand"].agg(["mean", "max", "std"])
        ax6.bar(hourly.index, hourly["mean"], color="#2196F3", alpha=0.7, label="Mean demand")
        ax6.plot(hourly.index, hourly["max"], color="red", linewidth=2, marker="o",
                 markersize=4, label="Max demand")
        ax6.fill_between(hourly.index,
                         hourly["mean"] - hourly["std"],
                         hourly["mean"] + hourly["std"],
                         alpha=0.2, color="#2196F3")
        ax6.set_title("Demand theo giờ trong ngày (mean / max / ±std)", fontsize=10)
        ax6.set_xlabel("Hour of day"); ax6.set_ylabel("trip_count")
        ax6.legend(fontsize=8); ax6.grid(alpha=0.3)
    elif "window_start" in df.columns:
        try:
            df2 = df.copy()
            df2["hour"] = pd.to_datetime(df2["window_start"], utc=True).dt.hour
            hourly = df2.groupby("hour")["target_demand"].agg(["mean", "max"])
            ax6.bar(hourly.index, hourly["mean"], color="#2196F3", alpha=0.7, label="Mean")
            ax6.plot(hourly.index, hourly["max"], color="red", linewidth=2, marker="o", markersize=4, label="Max")
            ax6.set_title("Demand theo giờ trong ngày", fontsize=10)
            ax6.set_xlabel("Hour of day"); ax6.set_ylabel("trip_count")
            ax6.legend(fontsize=8); ax6.grid(alpha=0.3)
        except Exception:
            ax6.text(0.5, 0.5, "Không vẽ được hourly plot", ha="center", va="center",
                     transform=ax6.transAxes, color="gray")
    else:
        ax6.text(0.5, 0.5, "Không có cột 'hour' hoặc 'window_start'", ha="center", va="center",
                 transform=ax6.transAxes, color="gray")

    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(INFO(f"Plots saved → {output_path}"))


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Chẩn đoán tham số LSTM v3 demand forecast"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data",    type=str, help="Path tới file .parquet hoặc .csv")
    group.add_argument("--delta",   action="store_true", help="Load từ Delta Lake (MinIO)")
    parser.add_argument("--model",   type=str, default=None, help="Path tới model dir đã train")
    parser.add_argument("--metrics", type=str, default=None, help="Path tới metrics.csv")
    parser.add_argument("--out",     type=str, default=".",  help="Thư mục lưu report & plots")
    return parser.parse_args()


def main():
    args = parse_args()

    # Nếu không có argument nào → chạy chế độ demo với dữ liệu giả lập
    if not args.data and not args.delta:
        print(_c("\n[MODE] Không có --data / --delta → chạy với dữ liệu giả lập (synthetic)\n", YELLOW))
        np.random.seed(42)
        n = 20_000
        hours = np.arange(n)
        base  = 50 + 30 * np.sin(2 * np.pi * hours / 24) + 20 * np.sin(2 * np.pi * hours / 168)
        spike_idx = np.random.choice(n, size=100, replace=False)
        demand = np.abs(base + np.random.normal(0, 10, n))
        demand[spike_idx] *= np.random.uniform(8, 20, 100)

        df = pd.DataFrame({
            "target_demand" : demand.astype(int),
            "window_start"  : pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
            "PULocationID"  : np.repeat(np.arange(1, 6), n // 5 + 1)[:n],
            "hour"          : hours % 24,
            "day_of_week"   : (hours // 24) % 7,
            "month"         : 1,
            "lag_1h"        : np.roll(demand, 1).astype(int),
            "lag_2h"        : np.roll(demand, 2).astype(int),
            "lag_3h"        : np.roll(demand, 3).astype(int),
            "lag_24h"       : np.roll(demand, 24).astype(int),
            "rolling_avg_3h": pd.Series(demand).rolling(3).mean().fillna(0).values,
            "rolling_avg_6h": pd.Series(demand).rolling(6).mean().fillna(0).values,
            "is_weekend"    : ((hours // 24) % 7 >= 5).astype(int),
        })
    else:
        df = load_data(args)

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rpt  = Report()

    rpt.add(HEAD(f"LSTM v3 — PARAMETER DIAGNOSTIC REPORT\n  {ts}"))
    rpt.info(f"Data shape : {df.shape}")
    rpt.info(f"Columns    : {list(df.columns)}")

    # ── Chạy tất cả các kiểm tra
    check_columns(df, rpt)
    target_stats  = check_target_distribution(df, rpt)
    scaling_info  = check_scaling(df, rpt)
    corr_series   = check_leakage(df, rpt)
    check_splits_and_sequences(df, rpt)
    check_model_params(rpt)

    if args.metrics:
        check_metrics(args.metrics, rpt)
    elif (Path(args.model or ".") / "metrics.csv").exists():
        check_metrics(str(Path(args.model) / "metrics.csv"), rpt)

    if args.model:
        check_trained_model(args.model, df, rpt)

    print_recommendations(scaling_info, rpt)

    # ── Summary & save
    rpt.summary()
    report_path = out / f"diagnose_report_{ts}.txt"
    rpt.save(report_path)

    # ── Plots
    try:
        plot_path = out / f"diagnose_plots_{ts}.png"
        plot_diagnostics(df, scaling_info["tgt_scaled"], corr_series, plot_path)
    except Exception as e:
        print(WARN(f"Không vẽ được plots: {e}"))

    # ── JSON summary
    json_path = out / f"diagnose_summary_{ts}.json"
    summary_data = {
        "timestamp"          : ts,
        "data_rows"          : len(df),
        "pass"               : rpt.counts["PASS"],
        "warn"               : rpt.counts["WARN"],
        "fail"               : rpt.counts["FAIL"],
        "target_max"         : float(df["target_demand"].max()) if "target_demand" in df.columns else None,
        "target_mean"        : float(df["target_demand"].mean()) if "target_demand" in df.columns else None,
        "pct_above_05_scaled": float(np.mean(scaling_info["tgt_scaled"] > 0.5) * 100),
        "suggested_peak_thr" : float(scaling_info["suggested_threshold"]),
        "config": {
            "LOOK_BACK"       : LOOK_BACK,
            "PEAK_LOSS_WEIGHT": PEAK_LOSS_WEIGHT,
            "HUBER_DELTA"     : HUBER_DELTA,
            "BATCH_SIZE"      : BATCH_SIZE,
            "LEARNING_RATE"   : LEARNING_RATE,
            "DROPOUT_RATE"    : DROPOUT_RATE,
        }
    }
    json_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False))
    print(INFO(f"JSON summary saved → {json_path}"))


if __name__ == "__main__":
    main()