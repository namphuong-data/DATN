"""
predict_demo.py — Demo dự đoán NYC taxi demand: ngày thường vs ngày lễ
═══════════════════════════════════════════════════════════════════════
Cách dùng:
  python spark/jobs/predict_demo.py \
    --model /app/models/lstm_v3_20260418_064120 \
    --date1 2024-09-10 \
    --date2 2024-11-28 \
    --out /app/project

US Holidays trong test set (2024-07-20 → 2024-12-31):
  Labor Day     : 2024-09-02  (Thứ 2)
  Columbus Day  : 2024-10-14  (Thứ 2)
  Veterans Day  : 2024-11-11  (Thứ 2)
  Thanksgiving  : 2024-11-28  (Thứ 5) ← default ngày lễ
  Christmas     : 2024-12-25  (Thứ 4)
"""
from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path

import holidays as hols
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("predict_demo")


# ── Phải định nghĩa ở module level để joblib unpickle đúng ──────────
# Scaler được pickle khi train với __main__.Log1pMinMaxScaler
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


# ── Cấu hình ─────────────────────────────────────────────────────────
STORAGE_OPTIONS = {
    "endpoint_url"             : "http://minio:9000",
    "access_key_id"            : "minioadmin",
    "secret_access_key"        : "minioadmin123",
    "region"                   : "us-east-1",
    "allow_http"               : "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

ZONE_DEMAND_MIN = 5.0   # phải khớp với config lúc train


# ════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL
# ════════════════════════════════════════════════════════════════════
def load_artifacts(model_dir: str):
    import tensorflow as tf

    p = Path(model_dir)
    model = tf.keras.models.load_model(str(p / "lstm_demand_v3.keras"), compile=False)
    feat_scaler = joblib.load(p / "feature_scaler.pkl")
    tgt_scaler  = joblib.load(p / "target_scaler.pkl")
    feat_cols   = joblib.load(p / "feature_cols.pkl")
    look_back   = model.input_shape[1]

    log.info("[MODEL] %s | look_back=%d | features=%d", p.name, look_back, len(feat_cols))
    log.info("[MODEL] Features: %s", feat_cols)
    return model, feat_scaler, tgt_scaler, feat_cols, look_back


# ════════════════════════════════════════════════════════════════════
# 2. LOAD & PREPROCESS DATA
# ════════════════════════════════════════════════════════════════════
def load_and_preprocess(feat_cols: list) -> pd.DataFrame:
    from deltalake import DeltaTable

    log.info("[DATA] Đọc từ Delta Lake ...")
    dt = DeltaTable("s3://lakehouse/gold/demand_by_zone", storage_options=STORAGE_OPTIONS)
    df = dt.to_pandas()

    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    df["window_end"]   = pd.to_datetime(df["window_end"],   utc=True)

    # Filter zone thưa (khớp với training)
    zone_means   = df.groupby("PULocationID")["target_demand"].mean()
    active_zones = zone_means[zone_means >= ZONE_DEMAND_MIN].index
    df = df[df["PULocationID"].isin(active_zones)].reset_index(drop=True)
    log.info("[DATA] Active zones: %d | Rows: %d", len(active_zones), len(df))

    # Sort
    df = df.sort_values(["PULocationID", "window_start"]).reset_index(drop=True)

    # Fill NaN lag cols gốc
    for col in ["lag_1h", "lag_2h", "lag_3h", "rolling_avg_3h", "rolling_avg_6h"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Tính lag dài hạn nếu chưa có
    for col, shift in [("lag_24h", 24), ("lag_168h", 168)]:
        if col not in df.columns:
            df[col] = (
                df.groupby("PULocationID")["target_demand"]
                .shift(shift).fillna(0)
            )

    if "rolling_avg_24h" not in df.columns:
        df["rolling_avg_24h"] = (
            df.groupby("PULocationID")["target_demand"]
            .apply(lambda x: x.shift(1).rolling(24, min_periods=1).mean())
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    # Temporal encoding
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)

    # is_holiday: dùng thư viện holidays (US/NY)
    years = df["window_start"].dt.year.unique().tolist()
    ny_cal = hols.country_holidays("US", subdiv="NY", years=years)
    holiday_dates = {pd.Timestamp(d).normalize() for d in ny_cal.keys()}
    df["is_holiday"] = df["window_start"].dt.normalize().isin(holiday_dates).astype(int)

    # Chỉ giữ features có trong model
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.warning("[DATA] Thiếu features: %s → set =0", missing)
        for c in missing:
            df[c] = 0

    log.info("[DATA] Preprocessing xong")
    return df


# ════════════════════════════════════════════════════════════════════
# 3. PREDICT 1 NGÀY
# ════════════════════════════════════════════════════════════════════
def predict_day(
    date_str: str,
    df: pd.DataFrame,
    feat_scaled: np.ndarray,
    model,
    tgt_scaler,
    feat_cols: list,
    look_back: int,
    top_n_zones: int = 5,
) -> dict:
    """
    Trả về dict chứa:
      hours        : list 24 timestamps
      actual_total : array 24 — tổng NYC demand thực
      pred_total   : array 24 — tổng NYC demand dự đoán
      actual_zones : dict {zone: array 24}
      pred_zones   : dict {zone: array 24}
      zone_means   : dict {zone: mean_actual} — để chọn top zones
    """
    target_date = pd.Timestamp(date_str, tz="UTC")
    hours = [target_date + pd.Timedelta(hours=h) for h in range(24)]

    zones    = df["PULocationID"].values
    ts_arr   = df["window_start"].values
    tgt_arr  = df["target_demand"].values

    # Lookup (zone, ts_np) -> global row index
    log.info("[PRED] Xây dựng lookup index cho %s ...", date_str)
    lookup: dict[tuple, int] = {}
    for i in range(len(df)):
        lookup[(zones[i], ts_arr[i])] = i

    # Build batch cho tất cả (zone, hour)
    zone_list = sorted(df["PULocationID"].unique())
    batch_X, batch_meta = [], []

    for h_idx, ts in enumerate(hours):
        ts_np = np.datetime64(ts)
        for z in zone_list:
            t_pos = lookup.get((z, ts_np))
            if t_pos is None or t_pos < look_back:
                continue
            if not np.all(zones[t_pos - look_back : t_pos + 1] == z):
                continue
            batch_X.append(feat_scaled[t_pos - look_back : t_pos])
            batch_meta.append((h_idx, z, t_pos))

    if not batch_X:
        log.error("[PRED] Không có sequence hợp lệ cho ngày %s", date_str)
        return {}

    log.info("[PRED] Chạy model.predict — batch size=%d ...", len(batch_X))
    X = np.stack(batch_X).astype(np.float32)
    preds_scaled = model.predict(X, batch_size=512, verbose=0).flatten()
    preds = np.clip(
        tgt_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten(), 0, None
    )

    # Tổng hợp theo giờ
    actual_total = np.zeros(24)
    pred_total   = np.zeros(24)
    actual_zones: dict[int, np.ndarray] = {z: np.zeros(24) for z in zone_list}
    pred_zones:   dict[int, np.ndarray] = {z: np.zeros(24) for z in zone_list}
    zone_count   = np.zeros(24, dtype=int)

    for i, (h_idx, z, t_pos) in enumerate(batch_meta):
        actual_total[h_idx]     += tgt_arr[t_pos]
        pred_total[h_idx]       += preds[i]
        actual_zones[z][h_idx]  += tgt_arr[t_pos]
        pred_zones[z][h_idx]    += preds[i]
        zone_count[h_idx]       += 1

    # Top N zones theo tổng demand thực
    zone_totals = {z: actual_zones[z].sum() for z in zone_list}
    top_zones   = sorted(zone_totals, key=lambda x: zone_totals[x], reverse=True)[:top_n_zones]

    mae  = np.mean(np.abs(actual_total - pred_total))
    mape = np.mean(
        np.abs((actual_total - pred_total) / np.where(actual_total == 0, 1, actual_total))
    ) * 100
    p75  = np.percentile(actual_total, 75)
    peak_mask = actual_total > p75
    peak_mape = (
        np.mean(np.abs((actual_total[peak_mask] - pred_total[peak_mask]) /
                        np.where(actual_total[peak_mask] == 0, 1, actual_total[peak_mask]))) * 100
        if peak_mask.sum() > 0 else float("nan")
    )

    log.info("[PRED] %s → MAE=%.1f | MAPE=%.1f%% | PeakMAPE=%.1f%% | zones=%d",
             date_str, mae, mape, peak_mape, zone_count.max())

    return dict(
        date        = date_str,
        hours       = hours,
        actual      = actual_total,
        predicted   = pred_total,
        mae         = mae,
        mape        = mape,
        peak_mape   = peak_mape,
        top_zones   = top_zones,
        actual_top  = {z: actual_zones[z] for z in top_zones},
        pred_top    = {z: pred_zones[z]   for z in top_zones},
    )


# ════════════════════════════════════════════════════════════════════
# 4. PLOT
# ════════════════════════════════════════════════════════════════════
DAY_OF_WEEK_VN = ["Thứ 2","Thứ 3","Thứ 4","Thứ 5","Thứ 6","Thứ 7","Chủ nhật"]

def _day_label(date_str: str) -> str:
    ts  = pd.Timestamp(date_str, tz="UTC")
    dow = DAY_OF_WEEK_VN[ts.dayofweek]
    hol = NYC_HOLIDAYS.get(ts, "")
    return f"{date_str}  ({dow}{' — ' + hol if hol else ''})"


def plot_results(res1: dict, res2: dict, out_path: Path):
    hours_label = [f"{h:02d}:00" for h in range(24)]
    x = np.arange(24)
    w = 0.35

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ── Màu sắc ────────────────────────────────────────────────────
    C_ACT   = "#2196F3"
    C_PRED  = "#FF5722"
    C_ERR1  = "#4CAF50"
    C_ERR2  = "#FF9800"
    ZONE_COLORS = ["#9C27B0","#00BCD4","#8BC34A","#FF5722","#795548"]

    def draw_day_panel(ax, res, color_pred, title):
        ax.bar(x, res["actual"],    width=w, label="Actual",    color=C_ACT,   alpha=0.75, align="center")
        ax.bar(x + w, res["predicted"], width=w, label="Predicted", color=color_pred, alpha=0.75, align="center")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Số chuyến đi / giờ", fontsize=9)
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(hours_label, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("#FAFAFA")
        # Stats box
        stats = (f"MAE={res['mae']:.1f}  MAPE={res['mape']:.1f}%  "
                 f"PeakMAPE={res['peak_mape']:.1f}%")
        ax.text(0.01, 0.97, stats, transform=ax.transAxes, fontsize=8,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Panel 1 & 2: Actual vs Predicted ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    draw_day_panel(ax1, res1, C_PRED,
                   f"Ngày thường\n{_day_label(res1['date'])}")
    draw_day_panel(ax2, res2, C_ERR2,
                   f"Ngày lễ\n{_day_label(res2['date'])}")

    # ── Panel 3: So sánh demand theo giờ (2 ngày overlaid) ─────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(x, res1["actual"],    color=C_ACT,  lw=2.0, marker="o", ms=4,
             label=f"Actual — {res1['date']}")
    ax3.plot(x, res1["predicted"], color=C_PRED, lw=1.8, ls="--", marker="s", ms=3,
             label=f"Predicted — {res1['date']}")
    ax3.plot(x, res2["actual"],    color="#1565C0", lw=2.0, marker="^", ms=4,
             label=f"Actual — {res2['date']} ({NYC_HOLIDAYS.get(pd.Timestamp(res2['date'],tz='UTC'),'Lễ')})")
    ax3.plot(x, res2["predicted"], color=C_ERR2, lw=1.8, ls="--", marker="v", ms=3,
             label=f"Predicted — {res2['date']}")
    ax3.fill_between(x, res1["actual"], res2["actual"],
                     alpha=0.08, color="gray", label="Chênh lệch hai ngày")
    ax3.set_title("So sánh demand theo giờ — Ngày thường vs Ngày lễ", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Tổng số chuyến (tất cả zones)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(hours_label, rotation=45, ha="right", fontsize=8)
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(alpha=0.3)
    ax3.set_facecolor("#FAFAFA")

    # ── Panel 4 & 5: Top 5 zones ─────────────────────────────────
    def draw_top_zones(ax, res, title):
        for i, z in enumerate(res["top_zones"]):
            ax.plot(x, res["actual_top"][z],
                    color=ZONE_COLORS[i % len(ZONE_COLORS)],
                    lw=1.8, marker="o", ms=3, label=f"Zone {z} (actual)")
            ax.plot(x, res["pred_top"][z],
                    color=ZONE_COLORS[i % len(ZONE_COLORS)],
                    lw=1.2, ls="--", alpha=0.7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Trips/giờ")
        ax.set_xticks(x)
        ax.set_xticklabels(hours_label, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        ax.set_facecolor("#FAFAFA")
        ax.text(0.99, 0.97, "-- = Predicted", transform=ax.transAxes,
                fontsize=7, va="top", ha="right", color="gray")

    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    draw_top_zones(ax4, res1, f"Top 5 zones bận nhất — {res1['date']}")
    draw_top_zones(ax5, res2, f"Top 5 zones bận nhất — {res2['date']}")

    # ── Tiêu đề tổng ───────────────────────────────────────────────
    fig.suptitle(
        "NYC Taxi Demand Forecast — Ngày thường vs Ngày lễ\n"
        f"Model: LSTM v3 | {len(res1['top_zones'])} zones hiển thị | "
        f"Dữ liệu thực tế (test set 2024)",
        fontsize=13, fontweight="bold", y=1.01
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("[PLOT] Saved → %s", out_path)


# ════════════════════════════════════════════════════════════════════
# 5. SUMMARY TEXT
# ════════════════════════════════════════════════════════════════════
def print_summary(res1: dict, res2: dict):
    print("\n" + "═" * 64)
    print("  PREDICTION SUMMARY — NYC Taxi Demand")
    print("═" * 64)
    for res, label in [(res1, "Ngày thường"), (res2, "Ngày lễ")]:
        print(f"\n  {label}: {_day_label(res['date'])}")
        print(f"    {'Giờ':<8} {'Actual':>8} {'Predicted':>10} {'Error%':>8}")
        print(f"    {'-'*40}")
        for h in range(24):
            act  = res["actual"][h]
            pred = res["predicted"][h]
            err  = abs(act - pred) / max(act, 1) * 100
            flag = " ◀ peak" if act >= np.percentile(res["actual"], 75) else ""
            print(f"    {h:02d}:00   {act:8.0f} {pred:10.0f} {err:7.1f}%{flag}")
        print(f"\n    MAE={res['mae']:.1f} | MAPE={res['mape']:.1f}% | PeakMAPE={res['peak_mape']:.1f}%")

    # So sánh
    diff = res2["actual"].sum() - res1["actual"].sum()
    pct  = diff / max(res1["actual"].sum(), 1) * 100
    print(f"\n  So sánh tổng ngày:")
    print(f"    Ngày thường  : {res1['actual'].sum():,.0f} chuyến")
    print(f"    Ngày lễ      : {res2['actual'].sum():,.0f} chuyến")
    print(f"    Chênh lệch   : {diff:+,.0f} ({pct:+.1f}%)")
    print("═" * 64 + "\n")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Demo dự đoán NYC taxi demand")
    p.add_argument("--model",  default="/app/models/lstm_v3_20260418_064120",
                   help="Đường dẫn tới model dir")
    p.add_argument("--date1",  default="2024-09-10",
                   help="Ngày thường (YYYY-MM-DD), mặc định: 2024-09-10 Thứ 3")
    p.add_argument("--date2",  default="2024-11-28",
                   help="Ngày lễ (YYYY-MM-DD), mặc định: 2024-11-28 Thanksgiving")
    p.add_argument("--out",    default=".", help="Thư mục lưu output")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    model, feat_scaler, tgt_scaler, feat_cols, look_back = load_artifacts(args.model)

    # 2. Load & preprocess data
    df = load_and_preprocess(feat_cols)

    # Kiểm tra ngày có trong test set không
    t_min    = df["window_start"].min()
    t_max    = df["window_start"].max()
    t_range  = t_max - t_min
    val_end  = t_min + t_range * 0.85
    log.info("[INFO] Test set: %s → %s", val_end.date(), t_max.date())

    for label, date_str in [("date1", args.date1), ("date2", args.date2)]:
        ts = pd.Timestamp(date_str, tz="UTC")
        if ts < val_end:
            log.warning("[WARN] %s=%s nằm ngoài test set (test bắt đầu %s)",
                        label, date_str, val_end.date())
        if ts < t_min + pd.Timedelta(hours=look_back):
            raise ValueError(
                f"{date_str} quá sớm — cần ít nhất {look_back}h dữ liệu trước đó"
            )

    # 3. Scale features (1 lần cho toàn bộ dataset)
    log.info("[PREP] Scaling features ...")
    feat_scaled = feat_scaler.transform(df[feat_cols].values).astype(np.float32)

    # 4. Predict
    log.info("[PRED] === Ngày 1: %s ===", args.date1)
    res1 = predict_day(args.date1, df, feat_scaled, model, tgt_scaler, feat_cols, look_back)

    log.info("[PRED] === Ngày 2: %s ===", args.date2)
    res2 = predict_day(args.date2, df, feat_scaled, model, tgt_scaler, feat_cols, look_back)

    if not res1 or not res2:
        raise RuntimeError("Prediction thất bại — kiểm tra ngày và dữ liệu")

    # 5. In kết quả
    print_summary(res1, res2)

    # 6. Plot
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = out / f"prediction_{args.date1}_vs_{args.date2}_{ts_str}.png"
    plot_results(res1, res2, plot_path)

    print(f"  Plot saved → {plot_path}\n")


if __name__ == "__main__":
    main()
