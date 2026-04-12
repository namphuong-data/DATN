"""
train_lstm_demand.py
─────────────────────────────────────────────────────────────────
LSTM Model: Dự báo trip_count (target_demand) tại từng zone NYC
Đọc Gold Delta Lake bằng deltalake (pure Python, không cần Java)
─────────────────────────────────────────────────────────────────
"""

import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from deltalake import DeltaTable

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("train_lstm")

# ═════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════
GOLD_SRC = "s3://lakehouse/gold/demand_by_zone"
MODEL_DIR = Path("models") / f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TARGET_ZONE = None
LOOK_BACK = 24
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LSTM_UNITS = [128, 64]
DROPOUT_RATE = 0.2

STORAGE_OPTIONS = {
    "endpoint_url": "http://minio:9000",
    "access_key_id": "minioadmin",
    "secret_access_key": "minioadmin123",
    "region": "us-east-1",
    "allow_http": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

FEATURE_COLS = [
    "trip_count",
    "lag_1h",
    "lag_2h",
    "rolling_avg_3h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]


# ═════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════════
def load_gold_data(target_zone=None):
    log.info("[DATA] Đọc Gold Delta từ %s ...", GOLD_SRC)
    dt = DeltaTable(GOLD_SRC, storage_options=STORAGE_OPTIONS)
    df = dt.to_pandas()
    if target_zone is not None:
        df = df[df["PULocationID"] == target_zone].reset_index(drop=True)
    log.info("[DATA] Loaded: %d rows, %d cols", len(df), len(df.columns))
    return df


# ═════════════════════════════════════════════════════════════
# 2. TIỀN XỬ LÝ
# ═════════════════════════════════════════════════════════════
def preprocess(df, target_zone):
    log.info("[PREP] Tiền xử lý dữ liệu ...")
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True)

    if target_zone is None:
        df = df.sort_values(["PULocationID", "window_start"]).reset_index(drop=True)
    else:
        df = df.sort_values("window_start").reset_index(drop=True)

    for col in ["lag_1h", "lag_2h", "rolling_avg_3h"]:
        df[col] = df[col].fillna(0)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.dropna(subset=["target_demand"]).reset_index(drop=True)
    log.info("[PREP] Sau xử lý: %d rows", len(df))
    log.info(
        "[PREP] trip_count   — min: %.0f, max: %.0f, mean: %.1f",
        df["trip_count"].min(),
        df["trip_count"].max(),
        df["trip_count"].mean(),
    )
    log.info(
        "[PREP] target_demand — min: %.0f, max: %.0f, mean: %.1f",
        df["target_demand"].min(),
        df["target_demand"].max(),
        df["target_demand"].mean(),
    )
    return df


# ═════════════════════════════════════════════════════════════
# 3. SEQUENCES + SPLIT + SCALE
# ═════════════════════════════════════════════════════════════
def make_sequences(features, targets, look_back):
    X, y = [], []
    for i in range(look_back, len(features)):
        X.append(features[i - look_back : i, :])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_and_scale(df):
    log.info("[SPLIT] Tạo train/val/test split theo thời gian ...")
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    features = df[FEATURE_COLS].values
    targets = df["target_demand"].values

    feat_train, feat_val, feat_test = (
        features[:n_train],
        features[n_train : n_train + n_val],
        features[n_train + n_val :],
    )
    tgt_train, tgt_val, tgt_test = (
        targets[:n_train],
        targets[n_train : n_train + n_val],
        targets[n_train + n_val :],
    )

    feat_scaler = MinMaxScaler().fit(feat_train)
    tgt_scaler = MinMaxScaler().fit(tgt_train.reshape(-1, 1))

    X_train, y_train = make_sequences(
        feat_scaler.transform(feat_train),
        tgt_scaler.transform(tgt_train.reshape(-1, 1)).flatten(),
        LOOK_BACK,
    )
    X_val, y_val = make_sequences(
        feat_scaler.transform(feat_val),
        tgt_scaler.transform(tgt_val.reshape(-1, 1)).flatten(),
        LOOK_BACK,
    )
    X_test, y_test = make_sequences(
        feat_scaler.transform(feat_test),
        tgt_scaler.transform(tgt_test.reshape(-1, 1)).flatten(),
        LOOK_BACK,
    )

    log.info(
        "[SPLIT] Train: %d | Val: %d | Test: %d", len(X_train), len(X_val), len(X_test)
    )
    test_dates = df["window_start"].iloc[n_train + n_val + LOOK_BACK :].values
    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feat_scaler,
        tgt_scaler,
        test_dates,
    )


# ═════════════════════════════════════════════════════════════
# 4. BUILD MODEL
# ═════════════════════════════════════════════════════════════
def build_lstm(input_shape):
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(LSTM_UNITS[0], return_sequences=True),
            Dropout(DROPOUT_RATE),
            BatchNormalization(),
            LSTM(LSTM_UNITS[1], return_sequences=False),
            Dropout(DROPOUT_RATE),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dense(1),
        ],
        name="LSTM_DemandForecast",
    )
    model.compile(optimizer=Adam(LEARNING_RATE), loss="huber", metrics=["mae"])
    model.summary(print_fn=log.info)
    return model


# ═════════════════════════════════════════════════════════════
# 5. TRAIN
# ═════════════════════════════════════════════════════════════
def train(model, X_train, y_train, X_val, y_val, model_dir):
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            str(model_dir / "best_model.keras"), monitor="val_loss", save_best_only=True
        ),
    ]
    log.info("[TRAIN] Bắt đầu training (epochs=%d, batch=%d) ...", EPOCHS, BATCH_SIZE)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    log.info("[TRAIN] Hoàn tất (best val_loss=%.5f)", min(history.history["val_loss"]))
    return history


# ═════════════════════════════════════════════════════════════
# 6. EVALUATE
# ═════════════════════════════════════════════════════════════
def evaluate_and_plot(model, X_test, y_test, tgt_scaler, test_dates, model_dir):
    y_pred_s = model.predict(X_test, verbose=0).flatten()
    y_true = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = np.clip(
        tgt_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten(), 0, None
    )

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    log.info("[EVAL] MAE=%.2f | RMSE=%.2f | R²=%.4f | MAPE=%.2f%%", mae, rmse, r2, mape)
    pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}]).to_csv(
        model_dir / "metrics.csv", index=False
    )

    n_plot = min(7 * 24, len(y_true))
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("LSTM Demand Forecast — NYC Taxi", fontsize=14, fontweight="bold")
    axes[0].plot(
        test_dates[:n_plot],
        y_true[:n_plot],
        label="Actual",
        color="#2196F3",
        linewidth=1.5,
    )
    axes[0].plot(
        test_dates[:n_plot],
        y_pred[:n_plot],
        label="Predicted",
        color="#FF5722",
        linewidth=1.5,
        linestyle="--",
    )
    axes[0].set_title("Actual vs Predicted (first 7 days of test set)")
    axes[0].set_ylabel("Trip Count")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    axes[0].grid(alpha=0.3)
    lim = max(y_true.max(), y_pred.max())
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=8, color="#9C27B0")
    axes[1].plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect fit")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"Scatter — R²={r2:.3f}  RMSE={rmse:.1f}  MAE={mae:.1f}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_dir / "forecast_result.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("Training History", fontsize=13, fontweight="bold")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def plot_training_history(history, model_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")
    ax1.plot(history.history["loss"], label="Train")
    ax1.plot(history.history["val_loss"], label="Val")
    ax1.set_title("Huber Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(history.history["mae"], label="Train")
    ax2.plot(history.history["val_mae"], label="Val")
    ax2.set_title("MAE")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_dir / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close()


# ═════════════════════════════════════════════════════════════
# 7. SAVE
# ═════════════════════════════════════════════════════════════
def save_artifacts(model, feat_scaler, tgt_scaler, model_dir):
    model.save(model_dir / "lstm_demand.keras")
    joblib.dump(feat_scaler, model_dir / "feature_scaler.pkl")
    joblib.dump(tgt_scaler, model_dir / "target_scaler.pkl")
    log.info("[SAVE] Artifacts → %s", model_dir)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("LSTM Demand Forecast Training START")
    log.info("  Gold source : %s", GOLD_SRC)
    log.info("  Target zone : %s", TARGET_ZONE or "ALL zones")
    log.info("  Look-back   : %d hours", LOOK_BACK)
    log.info("  Features    : %d  %s", len(FEATURE_COLS), FEATURE_COLS)
    log.info("  Model dir   : %s", MODEL_DIR)
    log.info("=" * 60)

    df_raw = load_gold_data(target_zone=TARGET_ZONE)
    df = preprocess(df_raw, target_zone=TARGET_ZONE)

    if len(df) < LOOK_BACK * 3:
        raise ValueError(f"Không đủ dữ liệu: {len(df)} rows")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feat_scaler,
        tgt_scaler,
        test_dates,
    ) = split_and_scale(df)
    model = build_lstm(input_shape=(LOOK_BACK, len(FEATURE_COLS)))
    history = train(model, X_train, y_train, X_val, y_val, MODEL_DIR)
    metrics = evaluate_and_plot(
        model, X_test, y_test, tgt_scaler, test_dates, MODEL_DIR
    )
    plot_training_history(history, MODEL_DIR)
    save_artifacts(model, feat_scaler, tgt_scaler, MODEL_DIR)

    log.info("=" * 60)
    log.info(
        "TRAINING COMPLETE — MAE=%.2f | RMSE=%.2f | R²=%.4f | MAPE=%.2f%%",
        metrics["MAE"],
        metrics["RMSE"],
        metrics["R2"],
        metrics["MAPE"],
    )
    log.info("  Artifacts → %s", MODEL_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
