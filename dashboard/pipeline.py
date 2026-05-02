"""
dashboard/pipeline.py
─────────────────────────────────────────────────────────────────
Background pipeline: Parquet → Kafka → Bronze → Silver → Gold
Chạy trong thread riêng, cập nhật trạng thái qua JSON files.
─────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import calendar
import json
import logging
import random
import re
import threading
import time
from datetime import datetime, date
from pathlib import Path

import holidays as hols
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _torch_available = True
except Exception:
    torch = None
    _torch_available = False
from confluent_kafka import Consumer, KafkaError, Producer
from confluent_kafka.admin import AdminClient, NewTopic
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger("pipeline")

# ── Paths & constants ─────────────────────────────────────────
import os
STATUS_DIR = Path("/tmp/datn_pipeline")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
TOPIC = "taxi_trips_streaming"
BATCH_SIZE = 10_000   # ~5MB/message (Arrow IPC) — safe dưới 100MB broker limit

STORAGE_OPTIONS = {
    "endpoint_url":              "http://minio:9000",
    "access_key_id":             "minioadmin",
    "secret_access_key":         "minioadmin123",
    "region":                    "us-east-1",
    "allow_http":                "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}
BRONZE_PATH = "s3://lakehouse/bronze/streaming"
SILVER_PATH = "s3://lakehouse/silver/streaming"
GOLD_PATH   = "s3://lakehouse/gold/predictions"
# Historical training/evaluation gold table. Used to show actual values
# for historical dates that are not present in silver/streaming.
GOLD_DEMAND_PATH = os.getenv("GOLD_DEMAND_PATH", "s3://lakehouse/gold/demand_by_zone")

LOOK_BACK = 48   # LSTM v4: look_back=48 (2 ngày × 24h), đồng bộ với train_lstm_demand_v4.py

# Low-hour hybrid inference:
# - Main LightGBM handles 05h-23h (when dashboard uses LightGBM).
# - When dashboard uses LSTM, hours 00h-04h still use the low-hour LGBM if loaded;
#   hours 05h-23h use LSTM as usual.
# Set LOW_HOUR_MODEL_DIR to a specific folder name/path, or leave empty to use latest v7.
LOW_HOUR_MODEL_ENV = "LOW_HOUR_MODEL_DIR"
# Ưu tiên 00–09h (zinf mới); fallback 00–04h bản cũ.
LOW_HOUR_MODEL_PREFIXES = ("lgbm_low_0_9_zinf_v7_", "lgbm_low_0_4_zinf_v7_")
LOW_HOUR_ENABLED = os.getenv("ENABLE_LOW_HOUR_MODEL", "1").lower() in ("1", "true", "yes")
# Buổi tối 17–23h: LightGBM bucket V8 (train_lgbm_evening_17_23.py). Tắt mặc định cho tới khi có artifact.
EVENING_BUCKET_MODEL_ENV = "EVENING_BUCKET_MODEL_DIR"
EVENING_BUCKET_MODEL_PREFIX = "lgbm_evening_17_23_bucket_v8_"
EVENING_BUCKET_ENABLED = os.getenv("ENABLE_EVENING_BUCKET_MODEL", "0").lower() in ("1", "true", "yes")
LSTM_MODEL_DIR_00_09 = os.getenv("LSTM_MODEL_DIR_00_09", "").strip()
LSTM_MODEL_DIR_17_23 = os.getenv("LSTM_MODEL_DIR_17_23", "").strip()



# ── Log1pMinMaxScaler (LightGBM — must match train script) ────
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


# ── Log1pStandardScaler (LSTM v4 — must match train_lstm_demand_v4.py) ──
from sklearn.preprocessing import StandardScaler as _StandardScaler

class Log1pStandardScaler:
    """log1p → StandardScaler. Dùng cho LSTM v4 (không phải MinMaxScaler)."""
    def __init__(self):
        self._scaler = _StandardScaler()
    def fit(self, X):
        self._scaler.fit(np.log1p(X)); return self
    def transform(self, X):
        return self._scaler.transform(np.log1p(X))
    def inverse_transform(self, X):
        return np.expm1(self._scaler.inverse_transform(X))
    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ── LSTM v4 PyTorch Architecture (phải khớp với train_lstm_demand_v4.py) ──
# Các hằng số architecture — KHÔNG được tự ý thay đổi; phải đồng bộ với
# model_config.pkl được lưu lúc train.
_LSTM_HIDDEN   = 256
_LSTM_LAYERS   = 2
_LSTM_DROPOUT  = 0.3
_LSTM_BIDIR    = True
_ATTN_DIM      = 128
_ZONE_EMB_DIM  = 16
_HOUR_EMB_DIM  = 16  # v5 — khớp train_lstm_demand.py
_DOW_EMB_DIM   = 8
_QUANTILE_OUT  = 2   # len(QUANTILE_Q) trong train script
_FC_HIDDEN     = 128
_MAX_LOC_ID    = 263


class _TemporalAttention(nn.Module if _torch_available else object):
    def __init__(self, lstm_output_dim: int, attn_dim: int = _ATTN_DIM):
        super().__init__()
        self.W = nn.Linear(lstm_output_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, hidden):
        scores  = self.v(torch.tanh(self.W(hidden)))   # (B, seq, 1)
        weights = F.softmax(scores, dim=1)
        context = (weights * hidden).sum(dim=1)        # (B, lstm_out)
        return context


class LSTMDemandModel(nn.Module if _torch_available else object):
    """Bi-LSTM + Temporal Attention + Zone Embedding — đúng với LSTM v4."""
    def __init__(self, n_feats: int, n_locations: int = _MAX_LOC_ID,
                 lstm_hidden: int = _LSTM_HIDDEN, lstm_layers: int = _LSTM_LAYERS,
                 lstm_dropout: float = _LSTM_DROPOUT, lstm_bidir: bool = _LSTM_BIDIR,
                 attn_dim: int = _ATTN_DIM, zone_emb_dim: int = _ZONE_EMB_DIM,
                 fc_hidden: int = _FC_HIDDEN, architecture: str = "single_head"):
        super().__init__()
        self.architecture = architecture
        lstm_out_dim = lstm_hidden * (2 if lstm_bidir else 1)
        self.zone_emb = nn.Embedding(n_locations + 1, zone_emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size    = n_feats,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = lstm_bidir,
            dropout       = lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = _TemporalAttention(lstm_out_dim, attn_dim)
        fc_in = lstm_out_dim + zone_emb_dim
        self.fc = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Linear(fc_in, fc_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fc_hidden, 1),
        )
        if self.architecture == "bucket_heads":
            self.head_00_09 = nn.Sequential(
                nn.LayerNorm(fc_in), nn.Linear(fc_in, fc_hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(fc_hidden, 1)
            )
            self.head_10_16 = nn.Sequential(
                nn.LayerNorm(fc_in), nn.Linear(fc_in, fc_hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(fc_hidden, 1)
            )
            self.head_17_23 = nn.Sequential(
                nn.LayerNorm(fc_in), nn.Linear(fc_in, fc_hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(fc_hidden, 1)
            )

    def forward(self, seq, zone_id, hour_id=None):
        lstm_out, _ = self.lstm(seq)
        context     = self.attention(lstm_out)
        zone_vec    = self.zone_emb(zone_id)
        combined    = torch.cat([context, zone_vec], dim=-1)
        if self.architecture != "bucket_heads" or hour_id is None:
            return self.fc(combined).squeeze(-1)
        out = self.fc(combined).squeeze(-1)
        m0 = (hour_id >= 0) & (hour_id <= 9)
        m1 = (hour_id >= 10) & (hour_id <= 16)
        m2 = (hour_id >= 17) & (hour_id <= 23)
        if m0.any():
            out[m0] = self.head_00_09(combined[m0]).squeeze(-1)
        if m1.any():
            out[m1] = self.head_10_16(combined[m1]).squeeze(-1)
        if m2.any():
            out[m2] = self.head_17_23(combined[m2]).squeeze(-1)
        return out

    def predict_numpy(self, X: np.ndarray, zone_ids: np.ndarray,
                      hour_ids: np.ndarray | None,
                      device, batch_size: int = 256) -> np.ndarray:
        """Inference helper: nhận numpy arrays, trả về numpy array."""
        self.eval()
        results = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i+batch_size]).to(device)
                zb = torch.from_numpy(zone_ids[i:i+batch_size].astype(np.int64)).to(device)
                hb = None
                if hour_ids is not None:
                    hb = torch.from_numpy(hour_ids[i:i+batch_size].astype(np.int64)).to(device)
                out = self(xb, zb, hb).cpu().numpy()
                results.append(out)
        return np.concatenate(results)


class DualContextLSTMModel(nn.Module if _torch_available else object):
    """
    LSTM v5 — khớp `spark/jobs/train_lstm_demand.py` (DualContextLSTMModel).
    Checkpoint có lstm_short / lstm_long / gate / hour_emb / dow_emb / fc_quantile.
    """

    def __init__(self, n_feats: int, n_locations: int = 265):
        super().__init__()
        lstm_out_dim = _LSTM_HIDDEN * (2 if _LSTM_BIDIR else 1)

        self.zone_emb = nn.Embedding(n_locations + 1, _ZONE_EMB_DIM, padding_idx=0)
        self.hour_emb = nn.Embedding(24, _HOUR_EMB_DIM)
        self.dow_emb = nn.Embedding(7, _DOW_EMB_DIM)

        self.lstm_short = nn.LSTM(
            input_size=n_feats,
            hidden_size=_LSTM_HIDDEN,
            num_layers=_LSTM_LAYERS,
            batch_first=True,
            bidirectional=_LSTM_BIDIR,
            dropout=_LSTM_DROPOUT if _LSTM_LAYERS > 1 else 0.0,
        )
        self.attn_short = _TemporalAttention(lstm_out_dim, _ATTN_DIM)

        self.lstm_long = nn.LSTM(
            input_size=n_feats,
            hidden_size=_LSTM_HIDDEN,
            num_layers=_LSTM_LAYERS,
            batch_first=True,
            bidirectional=_LSTM_BIDIR,
            dropout=_LSTM_DROPOUT if _LSTM_LAYERS > 1 else 0.0,
        )
        self.attn_long = _TemporalAttention(lstm_out_dim, _ATTN_DIM)

        self.gate = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        fc_in = lstm_out_dim + _ZONE_EMB_DIM + _HOUR_EMB_DIM + _DOW_EMB_DIM
        self.fc = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Linear(fc_in, _FC_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(_FC_HIDDEN, 1),
        )
        self.fc_quantile = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Linear(fc_in, _FC_HIDDEN // 2),
            nn.GELU(),
            nn.Linear(_FC_HIDDEN // 2, _QUANTILE_OUT),
        )

    def forward(self, seq_short, seq_long, hour_id, dow_id, zone_id):
        zone_id = zone_id.clamp(1, self.zone_emb.num_embeddings - 1)
        hour_id = hour_id.clamp(0, 23)
        dow_id = dow_id.clamp(0, 6)

        out_s, _ = self.lstm_short(seq_short)
        ctx_s = self.attn_short(out_s)
        out_l, _ = self.lstm_long(seq_long)
        ctx_l = self.attn_long(out_l)

        alpha = self.gate(torch.cat([ctx_s, ctx_l], dim=-1))
        ctx = alpha * ctx_s + (1 - alpha) * ctx_l

        zone_vec = self.zone_emb(zone_id)
        hour_vec = self.hour_emb(hour_id)
        dow_vec = self.dow_emb(dow_id)
        combined = torch.cat([ctx, zone_vec, hour_vec, dow_vec], dim=-1)

        point = self.fc(combined).squeeze(-1)
        quantile = self.fc_quantile(combined)
        return point, quantile

    def predict_numpy_v5(
        self,
        X_short: np.ndarray,
        X_long: np.ndarray,
        hour_ids: np.ndarray,
        dow_ids: np.ndarray,
        zone_ids: np.ndarray,
        device,
        batch_size: int = 256,
    ) -> np.ndarray:
        """X_short (N, lb_s, F), X_long (N, lb_l, F) — trả về scaled prediction (N,)"""
        self.eval()
        results = []
        n = len(X_short)
        with torch.no_grad():
            for i in range(0, n, batch_size):
                xs = torch.from_numpy(X_short[i : i + batch_size].astype(np.float32)).to(device)
                xl = torch.from_numpy(X_long[i : i + batch_size].astype(np.float32)).to(device)
                h = torch.from_numpy(hour_ids[i : i + batch_size].astype(np.int64)).to(device)
                d = torch.from_numpy(dow_ids[i : i + batch_size].astype(np.int64)).to(device)
                z = torch.from_numpy(zone_ids[i : i + batch_size].astype(np.int64)).to(device)
                pred_point, _ = self(xs, xl, h, d, z)
                results.append(pred_point.cpu().numpy())
        return np.concatenate(results)


# ── Status helpers ────────────────────────────────────────────
def write_status(key: str, data: dict):
    STATUS_DIR.mkdir(exist_ok=True)
    (STATUS_DIR / f"{key}.json").write_text(json.dumps(data, default=str))


def read_status(key: str) -> dict:
    f = STATUS_DIR / f"{key}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return {}
    return {}


def reset_all_status():
    STATUS_DIR.mkdir(exist_ok=True)
    for key in ("pipeline", "bronze", "silver", "gold"):
        p = STATUS_DIR / f"{key}.json"
        if p.exists():
            p.unlink()


# ════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════

def run_pipeline(parquet_path: str, model_type: str = "lstm", model_dir_name: str | None = None, target_date: str | None = None):
    """Entry point — runs Bronze, Silver and Gold as near-real-time workers."""
    model_type = (model_type or "lstm").lower()
    try:
        t0 = time.time()
        write_status("pipeline", {
            "state": "RUNNING",
            "file": Path(parquet_path).name,
            "model_type": model_type,
            "model_dir": model_dir_name,
            "target_date": target_date,
            "started_at": t0,
        })

        errors: list[tuple[str, Exception]] = []

        def _wrap(name, fn, *args):
            try:
                fn(*args)
            except Exception as exc:
                log.exception("%s failed: %s", name, exc)
                errors.append((name, exc))
                write_status(name, {"state": "error", "error": str(exc)})

        threads = [
            threading.Thread(target=_wrap, args=("bronze", stage_bronze, parquet_path), daemon=True),
            threading.Thread(target=_wrap, args=("silver", stage_silver_realtime, parquet_path), daemon=True),
            threading.Thread(target=_wrap, args=("gold", stage_gold_realtime, parquet_path, model_type, model_dir_name, target_date), daemon=True),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        if errors:
            name, exc = errors[0]
            raise RuntimeError(f"{name} failed: {exc}")

        write_status("pipeline", {
            "state": "DONE",
            "file": Path(parquet_path).name,
            "model_type": model_type,
            "model_dir": model_dir_name,
            "target_date": target_date,
            "elapsed_sec": round(time.time() - t0, 1),
        })

    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        write_status("pipeline", {"state": "ERROR", "error": str(exc), "model_type": model_type, "model_dir": model_dir_name, "target_date": target_date})

# ════════════════════════════════════════════════════════════════
# STAGE 1 — BRONZE  (Kafka producer + consumer → Delta Lake)
# ════════════════════════════════════════════════════════════════

def _wait_kafka(max_wait: int = 120):
    """Retry until Kafka broker is reachable (Kafka takes ~30s to start)."""
    log.info("[KAFKA] Waiting for broker at %s ...", KAFKA_BOOTSTRAP)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            admin = AdminClient({
                "bootstrap.servers": KAFKA_BOOTSTRAP,
                "socket.timeout.ms": 3000,
            })
            admin.list_topics(timeout=5)
            log.info("[KAFKA] Broker ready.")
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"Kafka broker not reachable at {KAFKA_BOOTSTRAP} after {max_wait}s")


def _ensure_topic():
    _wait_kafka()
    admin = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})
    meta  = admin.list_topics(timeout=15)
    if TOPIC not in meta.topics:
        fut = admin.create_topics([NewTopic(TOPIC, num_partitions=1, replication_factor=1)])
        for _, f in fut.items():
            try:
                f.result()
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise


def stage_bronze(parquet_path: str):
    _ensure_topic()

    df_raw     = pd.read_parquet(parquet_path)
    total_rows = len(df_raw)
    n_batches  = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

    consumer_state = {"written_rows": 0, "batches_written": 0, "history": []}
    consumer_done  = threading.Event()

    write_status("bronze", {
        "state": "producing", "total_rows": total_rows, "sent_rows": 0,
        "written_rows": 0, "n_batches": n_batches, "batches_sent": 0,
        "batches_written": 0, "throughput_rps": 0, "elapsed_sec": 0, "history": [], "sample_rows": [],
    })

    # Consumer thread
    threading.Thread(
        target=_bronze_consumer,
        args=(total_rows, consumer_state, consumer_done),
        daemon=True,
    ).start()

    # Producer
    t_start  = time.time()
    producer = Producer({
        "bootstrap.servers":            KAFKA_BOOTSTRAP,
        "message.max.bytes":            104857600,   # 100 MB — phải khớp broker
        "queue.buffering.max.kbytes":   524288,      # 512 MB local buffer
        "queue.buffering.max.messages": 20,
    })
    history: list[int] = []

    for i in range(n_batches):
        batch_df = df_raw.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].copy()

        # Serialize as Arrow IPC
        table = pa.Table.from_pandas(batch_df, preserve_index=False)
        buf   = pa.BufferOutputStream()
        with pa.ipc.new_stream(buf, table.schema) as w:
            w.write_table(table)
        producer.produce(TOPIC, value=buf.getvalue().to_pybytes())
        producer.poll(0)

        sent_rows = min((i + 1) * BATCH_SIZE, total_rows)
        elapsed   = time.time() - t_start
        history.append(len(batch_df))
        if len(history) > 40:
            history = history[-40:]

        write_status("bronze", {
            "state":          "producing",
            "total_rows":     total_rows,
            "sent_rows":      sent_rows,
            "written_rows":   consumer_state["written_rows"],
            "n_batches":      n_batches,
            "batches_sent":   i + 1,
            "batches_written": consumer_state["batches_written"],
            "throughput_rps": int(sent_rows / elapsed) if elapsed > 0 else 0,
            "elapsed_sec":    round(elapsed, 1),
            "history":        history,
            "sample_rows":    _sample_records(batch_df),
        })
        time.sleep(0.08)   # ~12 batches/sec visual effect

    # EOF sentinel
    producer.produce(TOPIC, value=b"__EOF__")
    producer.flush()
    consumer_done.wait(timeout=300)

    write_status("bronze", {
        "state": "done", "total_rows": total_rows, "sent_rows": total_rows,
        "written_rows": consumer_state["written_rows"],
        "n_batches": n_batches, "batches_sent": n_batches,
        "batches_written": consumer_state["batches_written"],
        "elapsed_sec": round(time.time() - t_start, 1),
        "history": consumer_state["history"],
        "sample_rows": _sample_records(df_raw),
    })


def _bronze_consumer(total_expected: int, state: dict, done: threading.Event):
    consumer = Consumer({
        "bootstrap.servers":        KAFKA_BOOTSTRAP,
        "group.id":                 f"bronze-writer-{int(time.time())}",
        "auto.offset.reset":        "earliest",
        "enable.auto.commit":       True,
        "fetch.message.max.bytes":   104857600,   # 100 MB
        "max.partition.fetch.bytes": 104857600,
        "receive.message.max.bytes": 209715200,  # 200 MB (>= message.max.bytes)
    })
    consumer.subscribe([TOPIC])
    buffer: list[pa.Table] = []
    FLUSH_ROWS = 400_000

    try:
        while True:
            msg = consumer.poll(timeout=30.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    break
                continue

            val = msg.value()
            if val == b"__EOF__":
                if buffer:
                    _flush_bronze(pa.concat_tables(buffer), state)
                break

            reader = pa.ipc.open_stream(pa.py_buffer(val))
            buffer.append(reader.read_all())

            if sum(len(t) for t in buffer) >= FLUSH_ROWS:
                _flush_bronze(pa.concat_tables(buffer), state)
                buffer = []
    finally:
        consumer.close()
        done.set()


def _flush_bronze(table: pa.Table, state: dict):
    df = table.to_pandas()
    df["_source_file"] = "streaming"
    df["_ingested_at"] = pd.Timestamp.now()
    df["_layer"]       = "bronze"

    # Ensure string columns are actually strings
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str)

    # First flush of a new pipeline run overwrites stale data from previous runs
    mode = "overwrite" if state["batches_written"] == 0 else "append"
    write_deltalake(BRONZE_PATH, df, mode=mode, storage_options=STORAGE_OPTIONS)
    state["written_rows"]    += len(df)
    state["batches_written"] += 1
    state["history"].append(len(df))
    if len(state["history"]) > 40:
        state["history"] = state["history"][-40:]


# ════════════════════════════════════════════════════════════════
# STAGE 2 — SILVER  (pandas clean + transform → Delta Lake)
# ════════════════════════════════════════════════════════════════

def _sample_records(df: pd.DataFrame, limit: int = 8, columns: list[str] | None = None) -> list[dict]:
    """Return small JSON-safe rows for the dashboard live data feed."""
    if df is None or df.empty:
        return []
    cols = columns or [
        c for c in [
            "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID",
            "trip_distance", "fare_amount", "passenger_count", "pickup_hour",
            "pickup_date", "window_start", "trip_count", "predicted", "actual",
        ] if c in df.columns
    ]
    sample = df[cols].tail(limit).copy()
    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]):
            sample[col] = sample[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return sample.astype(object).where(pd.notnull(sample), None).to_dict("records")


def _transform_bronze_dataframe(df: pd.DataFrame, parquet_path: str) -> pd.DataFrame:
    """Clean Bronze taxi rows into Silver rows.

    Bản sửa lỗi tháng 10:
    - Không để một điều kiện clean phụ làm rơi 100% dữ liệu.
    - Passenger count / fare / distance đôi khi null hoặc schema khác theo tháng;
      các điều kiện này được áp dụng theo kiểu "safe filter".
    - Cross-day trip như 2024-10-31 23:50 -> 2024-11-01 00:13 vẫn hợp lệ,
      vì demand được tính theo pickup time.
    - Ghi lại filter_summary để dashboard/debug thấy dữ liệu rơi ở bước nào.
    """
    df0 = df.copy()
    summary: list[dict] = []

    def _mark(step: str, before: int, after: int):
        dropped = before - after
        summary.append({
            "step": step,
            "before": int(before),
            "after": int(after),
            "dropped": int(dropped),
            "dropped_pct": round(dropped / before * 100, 2) if before else 0.0,
        })

    def _safe_numeric(col: str):
        if col in df0.columns:
            df0[col] = pd.to_numeric(df0[col], errors="coerce")

    before = len(df0)
    for col in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
        if col in df0.columns:
            df0[col] = pd.to_datetime(df0[col], errors="coerce")
    _mark("parse_datetime", before, len(df0))

    for col in ("PULocationID", "trip_distance", "fare_amount", "passenger_count"):
        _safe_numeric(col)

    year, month = _parse_ym(parquet_path)

    # Essential columns only. Nếu thiếu dropoff thì vẫn có thể giữ dòng nhưng duration không kiểm tra được.
    required = [c for c in ["tpep_pickup_datetime", "PULocationID"] if c in df0.columns]
    before = len(df0)
    if required:
        df0 = df0.dropna(subset=required)
    _mark("drop_missing_required", before, len(df0))

    # Demand được xác định theo pickup month, không theo dropoff month.
    before = len(df0)
    df0 = df0[(df0["tpep_pickup_datetime"].dt.year == year) &
              (df0["tpep_pickup_datetime"].dt.month == month)]
    _mark(f"pickup_month_{year}-{month:02d}", before, len(df0))

    # PULocationID bắt buộc hợp lệ.
    before = len(df0)
    if "PULocationID" in df0.columns:
        df0 = df0[df0["PULocationID"].between(1, 263)]
    _mark("valid_PULocationID", before, len(df0))

    # Duration: cho phép trip qua ngày/tháng; chỉ loại duration âm hoặc quá dài.
    if "tpep_dropoff_datetime" in df0.columns:
        before = len(df0)
        df0["trip_duration_min"] = (
            (df0["tpep_dropoff_datetime"] - df0["tpep_pickup_datetime"]).dt.total_seconds() / 60
        )
        # Relax hơn bản cũ để tránh rơi sạch dữ liệu vì các tháng có trip dài/cross-day.
        df0 = df0[df0["trip_duration_min"].between(0.5, 360)]
        _mark("duration_0.5_360_min", before, len(df0))
    else:
        df0["trip_duration_min"] = np.nan

    # Các filter chất lượng áp dụng an toàn: nếu cột null quá nhiều thì không ép.
    if "trip_distance" in df0.columns:
        before = len(df0)
        non_null_ratio = float(df0["trip_distance"].notna().mean()) if len(df0) else 0.0
        if non_null_ratio >= 0.50:
            df0 = df0[df0["trip_distance"].between(0.1, 100)]
        _mark("trip_distance_0.1_100_safe", before, len(df0))

    if "fare_amount" in df0.columns:
        before = len(df0)
        non_null_ratio = float(df0["fare_amount"].notna().mean()) if len(df0) else 0.0
        if non_null_ratio >= 0.50:
            df0 = df0[df0["fare_amount"] >= 1.0]
        _mark("fare_amount_ge_1_safe", before, len(df0))

    if "passenger_count" in df0.columns:
        before = len(df0)
        non_null_ratio = float(df0["passenger_count"].notna().mean()) if len(df0) else 0.0
        # Một số tháng passenger_count null nhiều do nguồn dữ liệu/driver app.
        # Chỉ filter nếu cột này thật sự đủ dữ liệu.
        if non_null_ratio >= 0.50:
            df0 = df0[df0["passenger_count"].fillna(1) > 0]
        _mark("passenger_count_positive_safe", before, len(df0))

    # Fallback bảo vệ: nếu các filter phụ làm rơi sạch, dùng bản clean tối thiểu.
    if df0.empty and not df.empty:
        log.warning("[SILVER] Strict/safe cleaning returned 0 rows. Falling back to essential pickup-month cleaning.")
        fallback = df.copy()
        for col in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
            if col in fallback.columns:
                fallback[col] = pd.to_datetime(fallback[col], errors="coerce")
        if "PULocationID" in fallback.columns:
            fallback["PULocationID"] = pd.to_numeric(fallback["PULocationID"], errors="coerce")
        fallback = fallback.dropna(subset=["tpep_pickup_datetime", "PULocationID"])
        fallback = fallback[(fallback["tpep_pickup_datetime"].dt.year == year) &
                            (fallback["tpep_pickup_datetime"].dt.month == month) &
                            (fallback["PULocationID"].between(1, 263))]
        if "tpep_dropoff_datetime" in fallback.columns:
            fallback["trip_duration_min"] = (
                (fallback["tpep_dropoff_datetime"] - fallback["tpep_pickup_datetime"]).dt.total_seconds() / 60
            )
        else:
            fallback["trip_duration_min"] = np.nan
        df0 = fallback
        summary.append({"step": "fallback_essential_clean", "before": int(len(df)), "after": int(len(df0)), "dropped": int(len(df)-len(df0)), "dropped_pct": round((len(df)-len(df0))/len(df)*100,2) if len(df) else 0.0})

    if not df0.empty:
        df0["PULocationID"] = df0["PULocationID"].astype(int)
        df0["pickup_year"] = df0["tpep_pickup_datetime"].dt.year
        df0["pickup_month"] = df0["tpep_pickup_datetime"].dt.month
        df0["pickup_day"] = df0["tpep_pickup_datetime"].dt.day
        df0["pickup_hour"] = df0["tpep_pickup_datetime"].dt.hour
        df0["pickup_dayofweek"] = df0["tpep_pickup_datetime"].dt.dayofweek
        df0["pickup_date"] = df0["tpep_pickup_datetime"].dt.date.astype(str)

    df0["_layer"] = "silver"
    df0["_processed_at"] = pd.Timestamp.now()

    try:
        write_status("silver_filter", {"state": "done", "summary": summary, "output_rows": len(df0)})
    except Exception:
        pass
    log.info("[SILVER] Filter summary: %s", summary)
    return df0



def _fix_delta_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make a pandas DataFrame safe for Delta Lake writes.

    Delta Lake does not support Arrow NullType. This happens when a column is
    completely empty/null after filtering, especially with optional taxi fields.
    We cast all-null columns to a concrete type and normalize object columns to
    strings so schema overwrite can succeed reliably across months.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().all():
            if len(df) == 0:
                df[col] = pd.Series(dtype="string")
            else:
                df[col] = pd.Series([""] * len(df), index=df.index, dtype="string")
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("string").fillna("")
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_convert(None)
    return df

def stage_silver_realtime(parquet_path: str):
    t0 = time.time()
    last_input_rows = -1
    last_state = None
    write_status("silver", {"state": "waiting", "input_rows": 0, "output_rows": 0, "sample_rows": []})

    while True:
        b_status = read_status("bronze")
        b_state = b_status.get("state", "waiting")
        written_rows = int(b_status.get("written_rows", 0) or 0)

        if written_rows <= 0:
            if b_state in ("done", "error"):
                break
            time.sleep(1.0)
            continue

        try:
            dt = DeltaTable(BRONZE_PATH, storage_options=STORAGE_OPTIONS)
            df_b = dt.to_pandas()
        except Exception:
            time.sleep(1.0)
            continue

        input_rows = len(df_b)
        is_final = b_state == "done"
        if input_rows == last_input_rows and last_state == b_state and not is_final:
            time.sleep(1.5)
            continue

        write_status("silver", {
            "state": "transforming",
            "input_rows": input_rows,
            "output_rows": 0,
            "filtered_rows": 0,
            "filtered_pct": 0,
            "elapsed_sec": round(time.time() - t0, 1),
            "sample_rows": _sample_records(df_b),
        })

        df_s = _transform_bronze_dataframe(df_b, parquet_path)
        output_rows = len(df_s)
        filtered = input_rows - output_rows

        write_status("silver", {
            "state": "writing" if not is_final else "finalizing",
            "input_rows": input_rows,
            "output_rows": output_rows,
            "filtered_rows": filtered,
            "filtered_pct": round(filtered / input_rows * 100, 1) if input_rows > 0 else 0,
            "elapsed_sec": round(time.time() - t0, 1),
            "sample_rows": _sample_records(df_s),
        })

        df_s = _fix_delta_null_columns(df_s)
        write_deltalake(
            SILVER_PATH,
            df_s,
            mode="overwrite",
            storage_options=STORAGE_OPTIONS,
            schema_mode="overwrite",
        )
        last_input_rows, last_state = input_rows, b_state

        if is_final:
            write_status("silver", {
                "state": "done",
                "input_rows": input_rows,
                "output_rows": output_rows,
                "filtered_rows": filtered,
                "filtered_pct": round(filtered / input_rows * 100, 1) if input_rows > 0 else 0,
                "elapsed_sec": round(time.time() - t0, 1),
                "sample_rows": _sample_records(df_s),
            })
            return

        time.sleep(1.5)

    write_status("silver", {
        "state": "error",
        "error": "Bronze finished without rows for Silver",
        "input_rows": 0,
        "output_rows": 0,
        "elapsed_sec": round(time.time() - t0, 1),
        "sample_rows": [],
    })


def stage_silver(parquet_path: str):
    """Backward-compatible wrapper."""
    return stage_silver_realtime(parquet_path)

# ════════════════════════════════════════════════════════════════
# STAGE 3 — GOLD  (hourly agg + LSTM inference → Delta Lake)
# ════════════════════════════════════════════════════════════════

def _load_context_parquet(year: int, month: int) -> pd.DataFrame | None:
    """
    Tải context 7 ngày trước tháng (year, month) từ file parquet local.
    Ưu tiên: tháng trước cùng năm → cùng kỳ năm trước.
    Trả về DataFrame raw hoặc None nếu không tìm thấy.
    """
    # Tháng trước
    prev_month = month - 1 if month > 1 else 12
    prev_year  = year      if month > 1 else year - 1

    candidates = [
        # Tháng trước cùng năm (2025)
        Path(f"/app/dataset/streaming/{prev_year}/yellow_tripdata_{prev_year}-{prev_month:02d}.parquet"),
        # Tháng trước từ batch 2022-2024
        Path(f"/app/dataset/batch/yellow_tripdata_{prev_year}-{prev_month:02d}.parquet"),
        # Cùng kỳ năm trước (fallback)
        Path(f"/app/dataset/batch/yellow_tripdata_{year-1}-{month:02d}.parquet"),
        Path(f"/app/dataset/streaming/{year-1}/yellow_tripdata_{year-1}-{month:02d}.parquet"),
    ]
    for p in candidates:
        if p.exists():
            log.info("[GOLD] Context source: %s", p.name)
            return pd.read_parquet(p)
    log.warning("[GOLD] Không tìm thấy context data cho %d-%02d", year, month)
    return None


def _build_context_hourly(year: int, month: int,
                          active_zones: list) -> pd.DataFrame:
    """
    Trả về hourly demand của 7 ngày cuối tháng trước (hoặc cùng kỳ năm trước)
    đã được shift timestamp về đúng 7 ngày trước tháng (year, month).
    """
    df_raw = _load_context_parquet(year, month)
    if df_raw is None:
        return pd.DataFrame(columns=["PULocationID", "window_start", "trip_count"])

    df_raw["tpep_pickup_datetime"] = pd.to_datetime(
        df_raw["tpep_pickup_datetime"], errors="coerce"
    )
    df_raw = df_raw.dropna(subset=["tpep_pickup_datetime"])
    df_raw["window_start"] = df_raw["tpep_pickup_datetime"].dt.floor("h")

    # Lấy 7 ngày cuối của file context (168 giờ)
    ctx_end   = df_raw["window_start"].max()
    ctx_start = ctx_end - pd.Timedelta(hours=167)
    df_ctx    = df_raw[df_raw["window_start"] >= ctx_start]

    hourly_ctx = (
        df_ctx.groupby(["PULocationID", "window_start"])
        .size().reset_index(name="trip_count")
    )
    hourly_ctx["PULocationID"] = hourly_ctx["PULocationID"].astype(int)
    hourly_ctx = hourly_ctx[hourly_ctx["PULocationID"].isin(active_zones)]

    # Shift timestamps: căn về đúng vị trí 7 ngày trước ngày 1 của tháng target
    target_month_start = pd.Timestamp(f"{year}-{month:02d}-01")
    actual_end         = hourly_ctx["window_start"].max()
    shift              = target_month_start - actual_end - pd.Timedelta(hours=1)
    hourly_ctx["window_start"] = hourly_ctx["window_start"] + shift

    return hourly_ctx


def _prepare_hourly_from_silver(parquet_path: str) -> tuple[pd.DataFrame, list[int], int, int, pd.DataFrame]:
    dt = DeltaTable(SILVER_PATH, storage_options=STORAGE_OPTIONS)
    df_s = dt.to_pandas()
    if df_s.empty:
        return pd.DataFrame(), [], 0, 0, df_s
    df_s["tpep_pickup_datetime"] = pd.to_datetime(df_s["tpep_pickup_datetime"])
    df_s["window_start"] = df_s["tpep_pickup_datetime"].dt.floor("h")

    hourly = (
        df_s.groupby(["PULocationID", "window_start"])
        .size().reset_index(name="trip_count")
    )
    hourly["PULocationID"] = hourly["PULocationID"].astype(int)
    hourly["window_start"] = pd.to_datetime(hourly["window_start"])

    zone_means = hourly.groupby("PULocationID")["trip_count"].mean()
    active_zones = zone_means[zone_means >= 5].index.tolist()
    if not active_zones:
        active_zones = zone_means.nlargest(10).index.tolist()
    hourly = hourly[hourly["PULocationID"].isin(active_zones)] if active_zones else hourly
    return hourly, active_zones, len(df_s), int(hourly["window_start"].nunique()) if not hourly.empty else 0, df_s


def stage_gold_realtime(parquet_path: str, model_type: str = "lstm", model_dir_name: str | None = None, target_date: str | None = None):
    t0 = time.time()
    model_type = (model_type or "lstm").lower()
    last_silver_rows = -1
    write_status("gold", {
        "state": "waiting", "zones_done": 0, "total_zones": 0,
        "model_type": model_type, "model_dir": model_dir_name, "target_date": target_date, "sample_rows": [],
    })

    while True:
        s_status = read_status("silver")
        s_state = s_status.get("state", "waiting")
        silver_rows = int(s_status.get("output_rows", 0) or 0)

        if silver_rows <= 0:
            if s_state in ("done", "error"):
                break
            time.sleep(1.0)
            continue

        if silver_rows != last_silver_rows or s_state == "done":
            try:
                hourly, active_zones, input_rows, hours_ready, df_s = _prepare_hourly_from_silver(parquet_path)
            except Exception:
                time.sleep(1.0)
                continue

            write_status("gold", {
                "state": "aggregating" if s_state != "done" else "finalizing",
                "zones_done": 0,
                "total_zones": len(active_zones),
                "silver_rows": input_rows,
                "hours_ready": hours_ready,
                "model_type": model_type,
                "model_dir": model_dir_name,
                "target_date": target_date,
                "elapsed_sec": round(time.time() - t0, 1),
                "sample_rows": _sample_records(hourly),
            })
            last_silver_rows = silver_rows

        if s_state == "done":
            break
        time.sleep(1.5)

    return stage_gold(parquet_path, model_type=model_type, model_dir_name=model_dir_name, target_date=target_date, t0=t0)


def stage_gold(parquet_path: str, model_type: str = "lstm", model_dir_name: str | None = None, target_date: str | None = None, t0: float | None = None):
    t0 = t0 or time.time()
    model_type = (model_type or "lstm").lower()
    write_status("gold", {"state": "aggregating", "zones_done": 0, "total_zones": 54, "model_type": model_type, "model_dir": model_dir_name, "target_date": target_date})

    hourly, active_zones, silver_rows, hours_ready, _df_s = _prepare_hourly_from_silver(parquet_path)
    if not active_zones:
        log.warning("[GOLD] Không có zone nào có dữ liệu, bỏ qua Gold processing.")
        write_status("gold", {"state": "error", "error": "No active zones found in Silver data", "model_type": model_type, "model_dir": model_dir_name, "target_date": target_date})
        return

    year, month = _parse_ym(parquet_path)
    ctx_hourly = _build_context_hourly(year, month, active_zones)
    if not ctx_hourly.empty:
        hourly = pd.concat([ctx_hourly, hourly], ignore_index=True)
        hourly = hourly.drop_duplicates(subset=["PULocationID", "window_start"])
        log.info("[GOLD] Context prepended: %d rows → total hourly: %d", len(ctx_hourly), len(hourly))

    target_day, holiday_name = _resolve_target_day(year, month, hourly, target_date)

    model_bundle = _load_model(model_type, model_dir_name)
    if model_type == "lstm":
        specialists: dict[str, dict] = {}
        for bucket, dir_name in (("00_09", LSTM_MODEL_DIR_00_09), ("17_23", LSTM_MODEL_DIR_17_23)):
            if not dir_name:
                continue
            try:
                b = _load_model("lstm", dir_name)
                # Tránh nạp trùng model chính
                if str(b.get("model_dir")) == str(model_bundle.get("model_dir")):
                    continue
                specialists[bucket] = b
            except Exception as exc:
                log.warning("[MODEL] Cannot load LSTM specialist %s (%s): %s", bucket, dir_name, exc)
        if specialists:
            model_bundle["specialists"] = specialists
            log.info("[MODEL] LSTM specialists loaded: %s",
                     {k: Path(str(v.get('model_dir'))).name for k, v in specialists.items()})
    _cfg = model_bundle.get("cfg") or {}
    _lb = (
        model_bundle.get("look_back_long")
        or _cfg.get("look_back_long")
        or model_bundle.get("look_back")
        or _cfg.get("look_back")
    )
    ctx_hours = int(_lb) if _lb is not None else LOOK_BACK
    if str(model_bundle.get("lstm_version", "")).lower() == "v5":
        ctx_hours = max(ctx_hours, 168)
    log.info(
        "[GOLD] Merge historical context: %d h | lstm_version=%s | model_dir=%s",
        ctx_hours,
        model_bundle.get("lstm_version"),
        model_bundle.get("model_dir"),
    )

    # Merge Gold lịch sử đủ dài cho LSTM v5 (168h), không chỉ 48h
    hourly, active_zones, historical_actual_loaded = _merge_historical_gold_context(
        hourly, target_day, active_zones, context_hours=ctx_hours
    )

    if not hourly.empty and "window_start" in hourly.columns:
        hourly = hourly.copy()
        hourly["window_start"] = _normalize_ts_col(hourly["window_start"]).dt.floor("h")

    context_coverage = _estimate_context_coverage(hourly, target_day, active_zones, ctx_hours)
    if model_type == "lstm":
        log.info(
            "[GOLD] LSTM context coverage = %.1f%% | look_back=%dh",
            context_coverage * 100.0,
            ctx_hours,
        )
        if context_coverage < 0.85:
            msg = (
                f"Insufficient LSTM context coverage: {context_coverage*100:.1f}% "
                f"(required >=85%) for look_back={ctx_hours}h. "
                "Need >=7 days historical rows in gold/demand_by_zone before target date."
            )
            write_status("gold", {
                "state": "error",
                "error": msg,
                "target_date": str(target_day),
                "context_coverage": round(context_coverage * 100.0, 1),
                "required_look_back_hours": int(ctx_hours),
                "actual_source": "gold/demand_by_zone" if historical_actual_loaded else "silver/streaming",
                "model_type": model_type,
                "model_dir": str(model_bundle.get("model_dir", "")),
            })
            raise RuntimeError(msg)

    write_status("gold", {
        "state": "predicting", "zones_done": 0, "total_zones": len(active_zones),
        "target_date": str(target_day),
        "holiday_name": holiday_name,
        "context_loaded": (not ctx_hourly.empty) or historical_actual_loaded,
        "actual_source": "gold/demand_by_zone" if historical_actual_loaded else "silver/streaming",
        "silver_rows": silver_rows,
        "hours_ready": hours_ready,
        "model_type": model_type,
        "model_dir": model_dir_name,
        "sample_rows": _sample_records(hourly),
    })

    custom_result = _predict_day(target_day, hourly, active_zones, model_bundle, model_type)

    rows = []
    for r in custom_result:
        rows.append({**r, "day_type": "custom", "date": str(target_day), "holiday_name": holiday_name, "model_type": model_type})

    df_gold = pd.DataFrame(rows)
    df_gold["_layer"] = "gold"
    df_gold["_processed_at"] = pd.Timestamp.now()

    write_deltalake(GOLD_PATH, df_gold, mode="overwrite", storage_options=STORAGE_OPTIONS, schema_mode="overwrite")

    write_status("gold", {
        "state": "done",
        "zones_done": len(active_zones),
        "total_zones": len(active_zones),
        "target_date": str(target_day),
        "holiday_name": holiday_name,
        "actual_source": "gold/demand_by_zone" if historical_actual_loaded else "silver/streaming",
        "elapsed_sec": round(time.time() - t0, 1),
        "model_type": model_type,
        "model_dir": str(model_bundle.get("model_dir", "")),
        "low_hour_model_dir": str((model_bundle.get("low_hour_model") or {}).get("model_dir", "")),
        "sample_rows": _sample_records(df_gold),
        "predictions": {
            "custom": _fmt_pred(custom_result, str(target_day), "custom", holiday_name),
        },
    })

# ── Helpers ───────────────────────────────────────────────────

def _parse_ym(path: str):
    stem = Path(path).stem            # yellow_tripdata_2025-01
    tail = stem.rsplit("_", 1)[-1]    # 2025-01
    y, m = tail.split("-")
    return int(y), int(m)


def _has_context(target: date, available_dates: set, min_days: int = 5) -> bool:
    """Kiểm tra có đủ LOOK_BACK context không (ít nhất min_days trong 7 ngày trước)."""
    from datetime import timedelta
    ctx_days = [target - timedelta(days=d) for d in range(1, 8)]
    return sum(d in available_dates for d in ctx_days) >= min_days


def _pick_days(year: int, month: int, hourly: pd.DataFrame):
    ny_cal = hols.country_holidays("US", subdiv="NY", years=year)
    _, n_days = calendar.monthrange(year, month)

    available_dates = set(hourly["window_start"].dt.date.unique())

    month_hols = sorted([
        (d, name) for d, name in ny_cal.items()
        if d.year == year and d.month == month
    ])
    hol_set = {d for d, _ in month_hols}

    # Chọn ngày lễ có đủ context (bỏ qua ngày đầu tháng nếu thiếu data trước đó)
    holiday_date, holiday_name = None, "No holiday"
    for d, name in month_hols:
        if _has_context(d, available_dates):
            holiday_date, holiday_name = d, name
            break
    if holiday_date is None and month_hols:
        # Fallback: chọn ngày lễ cuối tháng nhất (nhiều context nhất)
        holiday_date, holiday_name = month_hols[-1]
    if holiday_date is None:
        holiday_date = date(year, month, min(15, n_days))
        holiday_name = "No holiday"

    # Regular: Tue/Wed/Thu, day ≥ 8, có context, không phải ngày lễ
    candidates = [
        date(year, month, d)
        for d in range(8, n_days)
        if date(year, month, d).weekday() in (1, 2, 3)
        and date(year, month, d) not in hol_set
        and date(year, month, d) in available_dates
        and _has_context(date(year, month, d), available_dates)
    ]
    regular_date = random.choice(candidates) if candidates else date(year, month, 15)

    return regular_date, holiday_date, holiday_name


def _resolve_target_day(year: int, month: int, hourly: pd.DataFrame, target_date: str | None) -> tuple[date, str]:
    """Resolve user-selected prediction date and holiday label.

    If target_date is not provided, fallback to previous automatic regular day.
    The selected date must be in the same year/month as the selected parquet file.
    """
    ny_cal = hols.country_holidays("US", subdiv="NY", years=year)

    if target_date:
        try:
            target = pd.to_datetime(target_date).date()
        except Exception as exc:
            raise ValueError(f"Invalid target_date={target_date!r}; expected YYYY-MM-DD") from exc
        if target.year != year or target.month != month:
            log.info(
                "[GOLD] Selected target date %s is outside selected parquet month %04d-%02d; will try historical Gold fallback for actual/context.",
                target, year, month,
            )
    else:
        target, _, _ = _pick_days(year, month, hourly)

    holiday_name = ny_cal.get(target, "")
    return target, str(holiday_name or "")



def _latest_low_hour_model_dir() -> Path | None:
    """Return latest zero-inflated low-hour model directory, if available."""
    models_dir = Path("/app/models")
    if not models_dir.exists():
        return None
    candidates: list[Path] = []
    for pfx in LOW_HOUR_MODEL_PREFIXES:
        candidates.extend([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith(pfx)])
    uniq = list({p.resolve() for p in candidates})
    if not uniq:
        return None
    # Mới nhất theo thời sửa file (tránh sort tên ASCII đẩy 0_9 luôn sau 0_4 bất kể ngày).
    return max(uniq, key=lambda p: p.stat().st_mtime)


def _resolve_low_hour_model_dir() -> Path | None:
    """Resolve low-hour model folder from env or latest v7 artifact."""
    raw = os.getenv(LOW_HOUR_MODEL_ENV, "").strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = Path("/app/models") / raw
        candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"LOW_HOUR_MODEL_DIR does not exist: {candidate}")
        return candidate
    return _latest_low_hour_model_dir()


def _load_low_hour_model() -> dict | None:
    """Load V7 zero-inflated low-hour model (00–04h hoặc 00–09h theo artifact)."""
    if not LOW_HOUR_ENABLED:
        log.info("[MODEL] Low-hour model disabled by ENABLE_LOW_HOUR_MODEL=0")
        return None

    model_dir = _resolve_low_hour_model_dir()
    if model_dir is None:
        log.warning("[MODEL] No low-hour model found with prefixes %s", LOW_HOUR_MODEL_PREFIXES)
        return None

    cls_path = model_dir / "lgbm_low_hour_classifier.pkl"
    reg_path = model_dir / "lgbm_low_hour_regressor.pkl"
    feat_path = model_dir / "feature_cols.pkl"
    post_path = model_dir / "postprocess_params.pkl"
    info_path = model_dir / "low_hour_model_info.json"
    missing = [str(p) for p in (cls_path, reg_path, feat_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Low-hour model artifact missing: " + ", ".join(missing))

    post_params = {}
    if post_path.exists():
        post_params = joblib.load(str(post_path))

    hour_set = {0, 1, 2, 3, 4}
    if info_path.exists():
        try:
            meta = json.loads(info_path.read_text(encoding="utf-8"))
            hrs = meta.get("hours")
            if isinstance(hrs, list) and hrs:
                hour_set = {int(h) for h in hrs}
        except Exception:
            pass
    if "0_9" in model_dir.name:
        hour_set = set(range(10))

    bundle = {
        "type": "low_hour_zinf_v7",
        "model_dir": model_dir,
        "classifier": joblib.load(str(cls_path)),
        "regressor": joblib.load(str(reg_path)),
        "feat_cols": joblib.load(str(feat_path)),
        "postprocess_params": post_params,
        "hours": hour_set,
    }
    log.info("[MODEL] Loaded low-hour zero-inflated model from %s | hours=%s", model_dir, sorted(hour_set))
    return bundle


def _low_hour_effective_hours(low_bundle: dict | None) -> set[int]:
    """Giờ áp dụng low-hour ZINF; trùng logic _load: thư mục 0_9 → luôn 0–9."""
    if not low_bundle:
        return set()
    name_low = Path(str(low_bundle.get("model_dir") or "")).name.lower()
    if "0_9" in name_low or "0-9" in name_low:
        return set(range(10))
    hrs = low_bundle.get("hours")
    if isinstance(hrs, set) and hrs:
        return {int(x) for x in hrs}
    if isinstance(hrs, (list, tuple)) and hrs:
        return {int(x) for x in hrs}
    return {0, 1, 2, 3, 4}


def _latest_evening_bucket_model_dir() -> Path | None:
    models_dir = Path("/app/models")
    if not models_dir.exists():
        return None
    candidates = sorted(
        [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith(EVENING_BUCKET_MODEL_PREFIX)],
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def _resolve_evening_bucket_model_dir() -> Path | None:
    raw = os.getenv(EVENING_BUCKET_MODEL_ENV, "").strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = Path("/app/models") / raw
        candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"EVENING_BUCKET_MODEL_DIR does not exist: {candidate}")
        return candidate
    return _latest_evening_bucket_model_dir()


def _load_evening_bucket_model() -> dict | None:
    """LightGBM multiclass bucket cho 17h–23h (train_lgbm_evening_17_23.py)."""
    if not EVENING_BUCKET_ENABLED:
        return None
    model_dir = _resolve_evening_bucket_model_dir()
    if model_dir is None:
        log.info("[MODEL] No evening bucket model (prefix %s)", EVENING_BUCKET_MODEL_PREFIX)
        return None
    m_path = model_dir / "lgbm_evening_bucket_model.pkl"
    reg_path = model_dir / "lgbm_evening_model.pkl"
    sun_late_path = model_dir / "lgbm_evening_sun_late_expert.pkl"
    late_residual_path = model_dir / "lgbm_evening_late_residual_model.pkl"
    late_residual_feat_path = model_dir / "late_residual_feature_cols.pkl"
    feat_path = model_dir / "feature_cols.pkl"
    bv_path = model_dir / "bucket_values.pkl"
    post_path = model_dir / "postprocess_params.pkl"
    post_params = joblib.load(str(post_path)) if post_path.exists() else {}
    if reg_path.exists():
        late_residual_model = joblib.load(str(late_residual_path)) if late_residual_path.exists() else None
        late_residual_feat_cols = joblib.load(str(late_residual_feat_path)) if late_residual_feat_path.exists() else []
        sun_late_model = joblib.load(str(sun_late_path)) if sun_late_path.exists() else None
        quant_path = model_dir / "lgbm_evening_model_quantile.pkl"
        quant_model = joblib.load(str(quant_path)) if quant_path.exists() else None
        lr_q_path = model_dir / "lgbm_evening_late_residual_model_quantile.pkl"
        lr_q_feat_path = model_dir / "late_residual_feature_cols_quantile.pkl"
        late_residual_model_q = joblib.load(str(lr_q_path)) if lr_q_path.exists() else None
        late_residual_feat_cols_q = joblib.load(str(lr_q_feat_path)) if lr_q_feat_path.exists() else []
        bundle = {
            "type": "evening_reg_v9",
            "model_dir": model_dir,
            "model": joblib.load(str(reg_path)),
            "feat_cols": joblib.load(str(feat_path)),
            "sun_late_expert": sun_late_model,
            "late_residual_model": late_residual_model,
            "late_residual_feat_cols": late_residual_feat_cols,
            "quantile_model": quant_model,
            "late_residual_model_quantile": late_residual_model_q,
            "late_residual_feat_cols_quantile": late_residual_feat_cols_q,
            "postprocess_params": post_params,
            "hours": set(range(17, 24)),
        }
        log.info("[MODEL] Loaded evening 17-23h regression model from %s", model_dir)
        if quant_model is not None:
            wq = float(post_params.get("ensemble_quantile_weight", 0.0)) if isinstance(post_params, dict) else 0.0
            log.info("[MODEL] Loaded evening quantile aux + ensemble weight=%.4f from %s", wq, model_dir)
        if sun_late_model is not None:
            log.info("[MODEL] Loaded evening Sun-late expert from %s", model_dir)
        return bundle

    missing = [str(p) for p in (m_path, feat_path, bv_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Evening bucket/reg model artifact missing: " + ", ".join(missing))
    bv_raw = joblib.load(str(bv_path))
    if isinstance(bv_raw, dict) and "bucket_values" in bv_raw:
        bvals = np.asarray(bv_raw["bucket_values"], dtype=np.float64)
    else:
        bvals = np.asarray(bv_raw, dtype=np.float64)
    bundle = {
        "type": "evening_bucket_v8",
        "model_dir": model_dir,
        "model": joblib.load(str(m_path)),
        "feat_cols": joblib.load(str(feat_path)),
        "bucket_values": bvals,
        "postprocess_params": post_params,
        "hours": set(range(17, 24)),
    }
    log.info("[MODEL] Loaded evening 17-23h bucket model from %s", model_dir)
    return bundle


def _list_model_dirs(model_type: str) -> list[Path]:
    models_dir = Path("/app/models")
    model_type = (model_type or "lstm").lower()
    if not models_dir.exists():
        return []
    if model_type == "lstm":
        pattern = re.compile(r"^lstm(?:_v\d+)?_\d{8}_\d{6}$")
        return sorted([d for d in models_dir.iterdir() if d.is_dir() and pattern.match(d.name)], key=lambda p: p.name)
    if model_type in ("lightgbm", "lgbm"):
        return sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("lgbm_")], key=lambda p: p.name)
    raise ValueError(f"Unsupported model_type={model_type!r}; expected 'lstm' or 'lightgbm'")


def _latest_model_dir(model_type: str) -> Path:
    candidates = _list_model_dirs(model_type)
    if not candidates:
        raise FileNotFoundError(f"No {model_type} model found in /app/models/")
    return candidates[-1]


def _resolve_model_dir(model_type: str, model_dir_name: str | None = None) -> Path:
    """Trả về đúng thư mục model được chọn; fallback latest nếu không truyền."""
    if not model_dir_name:
        return _latest_model_dir(model_type)

    models_dir = Path("/app/models")
    candidate = Path(model_dir_name)
    if not candidate.is_absolute():
        candidate = models_dir / model_dir_name
    candidate = candidate.resolve()

    if not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(f"Selected model directory does not exist: {candidate}")

    model_type = (model_type or "lstm").lower()
    name = candidate.name
    if model_type == "lstm" and not name.startswith("lstm"):
        raise ValueError(f"Selected model {name!r} is not an LSTM model")
    if model_type in ("lightgbm", "lgbm") and not name.startswith("lgbm_"):
        raise ValueError(f"Selected model {name!r} is not a LightGBM model")

    return candidate


def _normalize_lstm_state_dict(raw: object) -> dict:
    """Checkpoint thuần state_dict | hoặc dict có model_state / state_dict | hoặc DataParallel module."""
    sd = raw
    if isinstance(sd, dict):
        if "model_state" in sd:
            sd = sd["model_state"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(raw)}")
    if sd and any(str(k).startswith("module.") for k in sd):
        sd = {
            (str(k)[7:] if str(k).startswith("module.") else str(k)): v
            for k, v in sd.items()
        }
    return sd


def _state_dict_is_lstm_v5(sd: dict) -> bool:
    return any(str(k).startswith("lstm_short.") for k in sd)


def _iter_lstm_weight_paths(model_dir: Path):
    """Thứ tự: best_lstm (thường mới nhất khi train), lstm_weights, checkpoint_epoch*.pt"""
    for name in ("best_lstm.pt", "lstm_weights.pt"):
        p = model_dir / name
        if p.exists():
            yield p
    yield from sorted(model_dir.glob("checkpoint_epoch*.pt"))


def _load_model(model_type: str = "lstm", model_dir_name: str | None = None) -> dict:
    model_type = (model_type or "lstm").lower()
    model_dir = _resolve_model_dir(model_type, model_dir_name)
    log.info("[MODEL] Loading %s model from %s", model_type, model_dir)

    if model_type == "lstm":
        if not _torch_available:
            raise ImportError("PyTorch is required for LSTM inference but is not available")

        cfg_path = model_dir / "model_config.pkl"
        if not cfg_path.exists():
            raise FileNotFoundError(f"model_config.pkl not found in {model_dir}")

        cfg = joblib.load(str(cfg_path))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        cfg_indicates_v5 = (
            cfg.get("look_back_short") is not None
            and cfg.get("look_back_long") is not None
        )

        weight_paths = list(_iter_lstm_weight_paths(model_dir))
        if not weight_paths:
            raise FileNotFoundError(
                f"No LSTM weights in {model_dir}. "
                "Add best_lstm.pt, lstm_weights.pt, or checkpoint_epoch*.pt"
            )

        state_dict: dict | None = None
        weights_path: Path | None = None

        if cfg_indicates_v5:
            for wp in weight_paths:
                try:
                    raw = torch.load(str(wp), map_location=device)
                    sd = _normalize_lstm_state_dict(raw)
                except Exception as exc:
                    log.warning("[MODEL] Skip %s: %s", wp.name, exc)
                    continue
                if _state_dict_is_lstm_v5(sd):
                    state_dict, weights_path = sd, wp
                    break
                log.warning(
                    "[MODEL] %s không phải weights v5 (không có lstm_short.*), thử file khác…",
                    wp.name,
                )
        else:
            for wp in weight_paths:
                try:
                    raw = torch.load(str(wp), map_location=device)
                    sd = _normalize_lstm_state_dict(raw)
                except Exception as exc:
                    log.warning("[MODEL] Skip %s: %s", wp.name, exc)
                    continue
                if _state_dict_is_lstm_v5(sd):
                    state_dict, weights_path = sd, wp
                    break
                state_dict, weights_path = sd, wp
                break

        if state_dict is None or weights_path is None:
            raise FileNotFoundError(
                f"Không load được weights LSTM phù hợp trong {model_dir}. "
                f"cfg look_back_short/long={'OK' if cfg_indicates_v5 else 'missing'} | "
                f"Đã thử: {[p.name for p in weight_paths]}"
            )

        is_v5 = _state_dict_is_lstm_v5(state_dict)
        if cfg_indicates_v5 and not is_v5:
            raise RuntimeError(
                f"model_config.pkl là v5 (có look_back_short/long) nhưng không có file weights "
                f"lstm_short.* trong {model_dir}. Kiểm tra best_lstm.pt / lstm_weights.pt."
            )

        if is_v5:
            lb_s = int(cfg.get("look_back_short", 72))
            lb_l = int(cfg.get("look_back_long", 168))
            n_loc = int(cfg.get("n_locations", 265))
            log.info(
                "[MODEL] LSTM v5 (dual-context) — n_feats=%d | short=%dh long=%dh | n_locations=%d",
                cfg.get("n_feats", 12), lb_s, lb_l, n_loc,
            )
            model = DualContextLSTMModel(
                n_feats=int(cfg["n_feats"]),
                n_locations=n_loc,
            ).to(device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            log.info("[MODEL] LSTM v5 weights loaded from %s → device=%s", weights_path.name, device)
            hour_calib = None
            dow_hour_calib = None
            hour_bias = None
            hour_affine = None
            hc_path = model_dir / "hour_calibration.pkl"
            dh_path = model_dir / "dow_hour_calibration.pkl"
            hb_path = model_dir / "hour_bias.pkl"
            ha_path = model_dir / "hour_affine.pkl"
            if hc_path.exists():
                try:
                    hour_calib = joblib.load(str(hc_path))
                    log.info("[MODEL] Loaded LSTM hour calibration: %s", hc_path)
                except Exception as exc:
                    log.warning("[MODEL] Cannot load LSTM hour calibration %s: %s", hc_path, exc)
            if dh_path.exists():
                try:
                    dow_hour_calib = joblib.load(str(dh_path))
                    log.info("[MODEL] Loaded LSTM dow-hour calibration: %s", dh_path)
                except Exception as exc:
                    log.warning("[MODEL] Cannot load LSTM dow-hour calibration %s: %s", dh_path, exc)
            if hb_path.exists():
                try:
                    hour_bias = joblib.load(str(hb_path))
                    log.info("[MODEL] Loaded LSTM hour bias: %s", hb_path)
                except Exception as exc:
                    log.warning("[MODEL] Cannot load LSTM hour bias %s: %s", hb_path, exc)
            if ha_path.exists():
                try:
                    hour_affine = joblib.load(str(ha_path))
                    log.info("[MODEL] Loaded LSTM hour affine: %s", ha_path)
                except Exception as exc:
                    log.warning("[MODEL] Cannot load LSTM hour affine %s: %s", ha_path, exc)
            lstm_bundle = {
                "type":           "lstm",
                "lstm_version":   "v5",
                "model_dir":      model_dir,
                "model":          model,
                "device":         device,
                "look_back":      lb_l,
                "look_back_short": lb_s,
                "look_back_long":  lb_l,
                "feat_scaler":    joblib.load(str(model_dir / "feature_scaler.pkl")),
                "tgt_scaler":     joblib.load(str(model_dir / "target_scaler.pkl")),
                "feat_cols":      joblib.load(str(model_dir / "feature_cols.pkl")),
                "hour_calibration": hour_calib,
                "dow_hour_calibration": dow_hour_calib,
                "hour_bias": hour_bias,
                "hour_affine": hour_affine,
                "cfg":            cfg,
            }
            try:
                lstm_bundle["low_hour_model"] = _load_low_hour_model()
            except Exception as exc:
                log.warning("[MODEL] Cannot load low-hour LGBM for LSTM hybrid (00-04h): %s", exc)
                lstm_bundle["low_hour_model"] = None
            try:
                lstm_bundle["evening_bucket_model"] = _load_evening_bucket_model()
            except Exception as exc:
                log.warning("[MODEL] Cannot load evening bucket model for LSTM hybrid (17-23h): %s", exc)
                lstm_bundle["evening_bucket_model"] = None
            return lstm_bundle

        log.info(
            "[MODEL] LSTM v4 config — n_feats=%d | look_back=%d | hidden=%d | layers=%d | bidir=%s | zone_emb=%d",
            cfg.get("n_feats", 12), cfg.get("look_back", 48),
            cfg.get("lstm_hidden", _LSTM_HIDDEN), cfg.get("lstm_layers", _LSTM_LAYERS),
            cfg.get("lstm_bidir", _LSTM_BIDIR), cfg.get("zone_emb_dim", _ZONE_EMB_DIM),
        )

        model = LSTMDemandModel(
            n_feats     = cfg["n_feats"],
            n_locations = cfg.get("n_locations", _MAX_LOC_ID),
            lstm_hidden = cfg.get("lstm_hidden",  _LSTM_HIDDEN),
            lstm_layers = cfg.get("lstm_layers",  _LSTM_LAYERS),
            lstm_dropout= cfg.get("lstm_dropout", _LSTM_DROPOUT),
            lstm_bidir  = cfg.get("lstm_bidir",   _LSTM_BIDIR),
            attn_dim    = cfg.get("attn_dim",     _ATTN_DIM),
            zone_emb_dim= cfg.get("zone_emb_dim", _ZONE_EMB_DIM),
            fc_hidden   = cfg.get("fc_hidden",    _FC_HIDDEN),
            architecture= cfg.get("architecture", "single_head"),
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()
        log.info("[MODEL] LSTM v4 weights loaded from %s → device=%s", weights_path.name, device)
        hour_calib = None
        dow_hour_calib = None
        hour_bias = None
        hour_affine = None
        hc_path = model_dir / "hour_calibration.pkl"
        dh_path = model_dir / "dow_hour_calibration.pkl"
        hb_path = model_dir / "hour_bias.pkl"
        ha_path = model_dir / "hour_affine.pkl"
        if hc_path.exists():
            try:
                hour_calib = joblib.load(str(hc_path))
                log.info("[MODEL] Loaded LSTM hour calibration: %s", hc_path)
            except Exception as exc:
                log.warning("[MODEL] Cannot load LSTM hour calibration %s: %s", hc_path, exc)
        if dh_path.exists():
            try:
                dow_hour_calib = joblib.load(str(dh_path))
                log.info("[MODEL] Loaded LSTM dow-hour calibration: %s", dh_path)
            except Exception as exc:
                log.warning("[MODEL] Cannot load LSTM dow-hour calibration %s: %s", dh_path, exc)
        if hb_path.exists():
            try:
                hour_bias = joblib.load(str(hb_path))
                log.info("[MODEL] Loaded LSTM hour bias: %s", hb_path)
            except Exception as exc:
                log.warning("[MODEL] Cannot load LSTM hour bias %s: %s", hb_path, exc)
        if ha_path.exists():
            try:
                hour_affine = joblib.load(str(ha_path))
                log.info("[MODEL] Loaded LSTM hour affine: %s", ha_path)
            except Exception as exc:
                log.warning("[MODEL] Cannot load LSTM hour affine %s: %s", ha_path, exc)

        lstm_bundle = {
            "type":           "lstm",
            "lstm_version":   "v4",
            "model_dir":      model_dir,
            "model":          model,
            "device":         device,
            "look_back":      cfg.get("look_back", 48),
            "feat_scaler":    joblib.load(str(model_dir / "feature_scaler.pkl")),
            "tgt_scaler":     joblib.load(str(model_dir / "target_scaler.pkl")),
            "feat_cols":      joblib.load(str(model_dir / "feature_cols.pkl")),
            "hour_calibration": hour_calib,
            "dow_hour_calibration": dow_hour_calib,
            "hour_bias": hour_bias,
            "hour_affine": hour_affine,
            "cfg":            cfg,
        }
        try:
            lstm_bundle["low_hour_model"] = _load_low_hour_model()
        except Exception as exc:
            log.warning("[MODEL] Cannot load low-hour LGBM for LSTM hybrid (00-04h): %s", exc)
            lstm_bundle["low_hour_model"] = None
        try:
            lstm_bundle["evening_bucket_model"] = _load_evening_bucket_model()
        except Exception as exc:
            log.warning("[MODEL] Cannot load evening bucket model for LSTM hybrid (17-23h): %s", exc)
            lstm_bundle["evening_bucket_model"] = None
        return lstm_bundle

    bundle = {
        "type": "lightgbm",
        "model_dir": model_dir,
        "model": joblib.load(str(model_dir / "lgbm_model.pkl")),
        "tgt_scaler": joblib.load(str(model_dir / "target_scaler.pkl")),
        "feat_cols": joblib.load(str(model_dir / "feature_cols.pkl")),
    }
    # Calibration tạm thời tắt mặc định vì calibration theo validation có thể
    # làm under-predict ngày thường / peak khi phân phối ngày dự đoán khác validation.
    # Bật lại bằng biến môi trường: APPLY_LGBM_CALIBRATION=1
    apply_calib = os.getenv("APPLY_LGBM_CALIBRATION", "0").lower() in ("1", "true", "yes")
    if apply_calib:
        calib_path = model_dir / "context_calibration.pkl"
        hour_calib_path = model_dir / "hour_calibration.pkl"
        if calib_path.exists():
            try:
                bundle["context_calibration"] = joblib.load(str(calib_path))
                log.info("[MODEL] Loaded context calibration: %s", calib_path)
            except Exception as exc:
                log.warning("[MODEL] Cannot load context calibration %s: %s", calib_path, exc)
        if hour_calib_path.exists():
            try:
                bundle["hour_calibration"] = joblib.load(str(hour_calib_path))
                log.info("[MODEL] Loaded hour calibration: %s", hour_calib_path)
            except Exception as exc:
                log.warning("[MODEL] Cannot load hour calibration %s: %s", hour_calib_path, exc)
    else:
        log.info("[MODEL] LightGBM calibration disabled. Set APPLY_LGBM_CALIBRATION=1 to enable.")

    # Attach optional low-hour model for hybrid prediction: 00h-04h use low-hour, 05h-23h use main model.
    try:
        bundle["low_hour_model"] = _load_low_hour_model()
    except Exception as exc:
        log.warning("[MODEL] Cannot load low-hour model; fallback to main LightGBM for 00-04h: %s", exc)
        bundle["low_hour_model"] = None
    try:
        bundle["evening_bucket_model"] = _load_evening_bucket_model()
    except Exception as exc:
        log.warning("[MODEL] Cannot load evening bucket model; fallback LSTM/main for 17-23h: %s", exc)
        bundle["evening_bucket_model"] = None
    return bundle

def _zone_features(demand_window: np.ndarray, timestamps: pd.DatetimeIndex,
                   holiday_dates: set, feat_cols: list, feat_scaler,
                   look_back: int = LOOK_BACK) -> np.ndarray:
    """
    Tính feature vector cho 1 zone từ cửa sổ look_back giờ.
    demand_window : (look_back,) giá trị demand thực/dự đoán
    Trả về        : (look_back, n_feat) đã scale
    """
    df = pd.DataFrame({"target_demand": demand_window, "window_start": timestamps})
    df["hour"]        = df["window_start"].dt.hour
    df["day_of_week"] = df["window_start"].dt.dayofweek
    df["month"]       = df["window_start"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"]  = df["window_start"].dt.normalize().isin(holiday_dates).astype(int)

    s = df["target_demand"]
    df["lag_1h"]          = s.shift(1).fillna(0)
    df["lag_2h"]          = s.shift(2).fillna(0)
    df["lag_24h"]         = s.shift(24).fillna(0)
    df["lag_168h"]        = s.shift(168).fillna(0)
    df["rolling_avg_3h"]  = s.shift(1).rolling(3,  min_periods=1).mean().fillna(0)
    df["rolling_avg_24h"] = s.shift(1).rolling(24, min_periods=1).mean().fillna(0)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)

    # Rename target_demand → trip_count nếu feat_cols dùng "trip_count" (LSTM v4)
    if "trip_count" in feat_cols and "trip_count" not in df.columns:
        df["trip_count"] = df["target_demand"]

    avail = [c for c in feat_cols if c in df.columns]
    vals  = df[avail].values.astype(np.float32)
    if len(avail) < len(feat_cols):
        pad  = np.zeros((look_back, len(feat_cols) - len(avail)), dtype=np.float32)
        vals = np.hstack([vals, pad])
    return feat_scaler.transform(vals)



def _safe_div(num, den, default: float = 0.0):
    """Vector-safe division used by LightGBM inference feature engineering."""
    den = np.asarray(den, dtype=np.float32)
    num = np.asarray(num, dtype=np.float32)
    return np.divide(num, den, out=np.full_like(num, default, dtype=np.float32), where=np.abs(den) > 1e-6)



def _build_lgbm_baselines(hourly: pd.DataFrame, active_zones: list, target_start: pd.Timestamp) -> dict:
    """
    Build historical baseline features for LightGBM inference from all available
    rows BEFORE the predicted day. This is safer than computing zone_hour_mean
    from the 168h rolling window only, and avoids leaking actual values from
    the target day.
    """
    if hourly is None or hourly.empty:
        return {
            "zone_hour_mean": {},
            "zone_dow_hour_mean": {},
            "zone_mean": {},
            "hour_mean": {},
            "global_mean": 0.0,
        }

    hist = hourly.copy()
    hist["window_start"] = pd.to_datetime(hist["window_start"])
    hist = hist[hist["window_start"] < target_start]
    if active_zones:
        hist = hist[hist["PULocationID"].isin(active_zones)]

    if hist.empty:
        return {
            "zone_hour_mean": {},
            "zone_dow_hour_mean": {},
            "zone_mean": {},
            "hour_mean": {},
            "global_mean": 0.0,
        }

    hist["PULocationID"] = hist["PULocationID"].astype(int)
    hist["hour"] = hist["window_start"].dt.hour.astype(int)
    hist["day_of_week"] = hist["window_start"].dt.dayofweek.astype(int)
    hist["trip_count"] = pd.to_numeric(hist["trip_count"], errors="coerce").fillna(0.0)

    zone_hour = hist.groupby(["PULocationID", "hour"])["trip_count"].mean().to_dict()
    zone_dow_hour = hist.groupby(["PULocationID", "day_of_week", "hour"])["trip_count"].mean().to_dict()
    zone_mean = hist.groupby("PULocationID")["trip_count"].mean().to_dict()
    hour_mean = hist.groupby("hour")["trip_count"].mean().to_dict()
    global_mean = float(hist["trip_count"].mean())

    return {
        "zone_hour_mean": zone_hour,
        "zone_dow_hour_mean": zone_dow_hour,
        "zone_mean": zone_mean,
        "hour_mean": hour_mean,
        "global_mean": global_mean,
    }


def _normalize_ts_col(series: pd.Series) -> pd.Series:
    """Normalize timestamps from Delta tables to timezone-naive pandas timestamps."""
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def _hour_floor_ts(ts) -> pd.Timestamp:
    """Khóa lookup (zone, window_start) — luôn floor theo giờ, naive TS."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(None)
    return t.floor("h")


def _build_zone_hour_lookup(hourly: pd.DataFrame) -> dict[tuple[int, pd.Timestamp], float]:
    """Gộp trip_count theo (zone, giờ) — tránh ghi đè khi merge Silver + Gold trùng khóa."""
    if hourly is None or hourly.empty or "trip_count" not in hourly.columns:
        return {}
    h = hourly.copy()
    h["window_start"] = _normalize_ts_col(h["window_start"]).dt.floor("h")
    h["PULocationID"] = pd.to_numeric(h["PULocationID"], errors="coerce").fillna(0).astype(int)
    h["trip_count"] = pd.to_numeric(h["trip_count"], errors="coerce").fillna(0.0)
    h = h.dropna(subset=["window_start"])
    g = h.groupby(["PULocationID", "window_start"], as_index=False)["trip_count"].sum()
    out: dict[tuple[int, pd.Timestamp], float] = {}
    for r in g.itertuples(index=False):
        ts = _hour_floor_ts(r.window_start)
        out[(int(r.PULocationID), ts)] = float(r.trip_count)
    return out


def _load_historical_gold_hourly(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Load historical hourly demand from s3://lakehouse/gold/demand_by_zone.

    This table is the train/val/test source. Dashboard streaming tables are overwritten
    for each run, so historical actuals may exist here even when silver/streaming has 0 rows.
    Returns columns: PULocationID, window_start, trip_count.
    """
    try:
        dt = DeltaTable(GOLD_DEMAND_PATH, storage_options=STORAGE_OPTIONS)
        try:
            df = dt.to_pandas(columns=["PULocationID", "window_start", "target_demand"])
        except TypeError:
            df = dt.to_pandas()
    except Exception as exc:
        log.warning("[GOLD] Cannot load historical gold demand table %s: %s", GOLD_DEMAND_PATH, exc)
        return pd.DataFrame(columns=["PULocationID", "window_start", "trip_count"])

    if df.empty or "window_start" not in df.columns:
        return pd.DataFrame(columns=["PULocationID", "window_start", "trip_count"])

    df = df.copy()
    df["window_start"] = _normalize_ts_col(df["window_start"])
    if "target_demand" in df.columns:
        df["trip_count"] = pd.to_numeric(df["target_demand"], errors="coerce").fillna(0.0)
    elif "trip_count" in df.columns:
        df["trip_count"] = pd.to_numeric(df["trip_count"], errors="coerce").fillna(0.0)
    else:
        return pd.DataFrame(columns=["PULocationID", "window_start", "trip_count"])

    df = df.dropna(subset=["window_start", "PULocationID"])
    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_convert(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert(None)
    df = df[(df["window_start"] >= start_ts) & (df["window_start"] <= end_ts)]
    if df.empty:
        return pd.DataFrame(columns=["PULocationID", "window_start", "trip_count"])

    df["PULocationID"] = df["PULocationID"].astype(int)
    return (
        df.groupby(["PULocationID", "window_start"], as_index=False)["trip_count"]
        .sum()
        .sort_values(["PULocationID", "window_start"])
    )


def _merge_historical_gold_context(
    hourly: pd.DataFrame,
    target_day: date,
    current_active_zones: list[int],
    context_hours: int = 168,
) -> tuple[pd.DataFrame, list[int], bool]:
    """Merge historical Gold demand cho ngữ cảnh + ngày target.

    **context_hours** phải ≥ look_back_long của LSTM v5 (168). Trước đây dùng LOOK_BACK=48
    → thiếu ~120h lịch sử → lookup toàn 0 → dự báo sai nặng (vd 2024-09-09).
    """
    target_start = pd.Timestamp(target_day)
    hist_start = target_start - pd.Timedelta(hours=max(context_hours, 24))
    hist_end = target_start + pd.Timedelta(hours=23)
    hist = _load_historical_gold_hourly(hist_start, hist_end)
    if hist.empty:
        log.warning(
            "[GOLD] Không có dòng Gold trong [%s → %s] tại %s. "
            "Pipeline chỉ dùng Silver — LSTM v5 thiếu 168h ngữ cảnh → dự báo có thể sai rất lớn. "
            "Chạy batch Gold (demand_by_zone) lên MinIO hoặc kiểm tra ngày có trong bảng.",
            hist_start,
            hist_end,
            GOLD_DEMAND_PATH,
        )
        return hourly, current_active_zones, False

    target_mask = (hist["window_start"] >= target_start) & (hist["window_start"] <= hist_end)
    has_target_actual = bool(hist.loc[target_mask, "trip_count"].sum() > 0)
    if not has_target_actual:
        merged = pd.concat([hist, hourly], ignore_index=True) if hourly is not None and not hourly.empty else hist
        merged = merged.drop_duplicates(subset=["PULocationID", "window_start"], keep="first")
        return merged, current_active_zones, False

    hist_before_target = hist[hist["window_start"] < target_start]
    zone_means = (hist_before_target if not hist_before_target.empty else hist).groupby("PULocationID")["trip_count"].mean()
    hist_active_zones = zone_means[zone_means >= 5].index.astype(int).tolist()
    if not hist_active_zones:
        hist_active_zones = zone_means.nlargest(10).index.astype(int).tolist()

    merged = pd.concat([hist, hourly], ignore_index=True) if hourly is not None and not hourly.empty else hist
    merged = merged.drop_duplicates(subset=["PULocationID", "window_start"], keep="first")
    log.info(
        "[GOLD] Historical actual/context loaded from %s | rows=%d | active_zones=%d | target_actual=%.1f",
        GOLD_DEMAND_PATH, len(hist), len(hist_active_zones), float(hist.loc[target_mask, "trip_count"].sum())
    )
    return merged, hist_active_zones, True


def _estimate_context_coverage(
    hourly: pd.DataFrame,
    target_day: date,
    active_zones: list[int],
    look_back_hours: int,
) -> float:
    """Tỉ lệ keys (zone, hour) có dữ liệu trong [T-look_back, T-1]."""
    if hourly is None or hourly.empty or not active_zones or look_back_hours <= 0:
        return 0.0
    lookup = _build_zone_hour_lookup(hourly)
    target_start = pd.Timestamp(target_day)
    hist_ts = pd.date_range(
        target_start - pd.Timedelta(hours=look_back_hours),
        periods=look_back_hours,
        freq="h",
    )
    total = len(active_zones) * len(hist_ts)
    if total == 0:
        return 0.0
    present = 0
    for z in active_zones:
        for h in hist_ts:
            if (z, _hour_floor_ts(h)) in lookup:
                present += 1
    return present / total


def _build_lgbm_feature_frame(zone_id: int, demand_window: np.ndarray,
                              timestamps: pd.DatetimeIndex,
                              holiday_dates: set,
                              baselines: dict | None = None) -> pd.DataFrame:
    """
    Tạo feature LightGBM giống train script.

    Lưu ý quan trọng:
    - DataFrame truyền vào có thể bao gồm 1 dòng placeholder ở cuối cho giờ cần dự đoán.
    - target_demand của placeholder không được dùng trực tiếp, nhưng các lag/rolling sẽ lấy từ
      các giờ trước đó. Vì vậy hour/day/month của dòng cuối chính là giờ cần predict.
    """
    df = pd.DataFrame({
        "target_demand": np.asarray(demand_window, dtype=np.float32),
        "window_start": pd.to_datetime(timestamps),
    })
    df["PULocationID"] = int(zone_id)
    df["hour"] = df["window_start"].dt.hour.astype(int)
    df["day_of_week"] = df["window_start"].dt.dayofweek.astype(int)
    df["month"] = df["window_start"].dt.month.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = df["window_start"].dt.normalize().isin(holiday_dates).astype(int)

    s = df["target_demand"].astype(float)

    # Lag/rolling: tuyệt đối không dùng giá trị target của dòng hiện tại.
    df["lag_1h"] = s.shift(1).fillna(0)
    df["lag_2h"] = s.shift(2).fillna(0)
    df["lag_3h"] = s.shift(3).fillna(0)
    df["lag_24h"] = s.shift(24).fillna(0)
    df["lag_168h"] = s.shift(168).fillna(0)

    df["rolling_avg_3h"] = s.shift(1).rolling(3, min_periods=1).mean().fillna(0)
    df["rolling_avg_6h"] = s.shift(1).rolling(6, min_periods=1).mean().fillna(0)
    df["rolling_avg_24h"] = s.shift(1).rolling(24, min_periods=1).mean().fillna(0)

    # Một số train script có thể dùng tên rolling_mean_* thay vì rolling_avg_*.
    df["rolling_mean_3h"] = df["rolling_avg_3h"]
    df["rolling_mean_6h"] = df["rolling_avg_6h"]
    df["rolling_mean_24h"] = df["rolling_avg_24h"]

    # Cyclical time features.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Feature theo khung giờ / chuyển pha. Tạo nhiều alias để tương thích các bản train.
    df["is_late_night"] = df["hour"].between(0, 4).astype(int)
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["is_midnight"] = df["hour"].between(0, 1).astype(int)
    df["is_early_morning"] = df["hour"].between(5, 7).astype(int)

    # Ramp-up sáng là điểm yếu hiện tại 05–08h.
    df["is_ramp_up"] = df["hour"].between(5, 8).astype(int)
    df["is_morning_ramp"] = df["is_ramp_up"]
    df["is_transition_up"] = df["hour"].between(5, 8).astype(int)

    df["is_morning_peak"] = df["hour"].between(7, 10).astype(int)
    df["is_am_peak"] = df["is_morning_peak"]
    df["is_evening_peak"] = df["hour"].between(16, 19).astype(int)
    df["is_pm_peak"] = df["is_evening_peak"]
    df["is_peak"] = ((df["is_morning_peak"] == 1) | (df["is_evening_peak"] == 1)).astype(int)

    df["is_transition_down"] = df["hour"].between(20, 22).astype(int)
    df["is_end_of_day"] = df["hour"].between(22, 23).astype(int)
    df["is_late_evening"] = df["hour"].between(21, 23).astype(int)

    # Ratio / diff / momentum features. Tạo alias theo nhiều naming convention.
    df["lag_ratio_1h_24h"] = _safe_div(df["lag_1h"], df["lag_24h"] + 1.0)
    df["lag_ratio_3h_24h"] = _safe_div(df["rolling_avg_3h"], df["rolling_avg_24h"] + 1.0)
    df["rolling_ratio_3h_24h"] = _safe_div(df["rolling_avg_3h"], df["rolling_avg_24h"] + 1.0)

    df["lag_diff_1h_24h"] = df["lag_1h"] - df["lag_24h"]
    df["lag_diff_3h_24h"] = df["rolling_avg_3h"] - df["rolling_avg_24h"]
    df["lag_1h_diff_24h"] = df["lag_diff_1h_24h"]
    df["lag_3h_diff_24h"] = df["lag_diff_3h_24h"]

    df["momentum_1h_2h"] = df["lag_1h"] - df["lag_2h"]
    df["momentum_1h_3h"] = df["lag_1h"] - df["lag_3h"]
    df["momentum_3h_24h"] = df["rolling_avg_3h"] - df["rolling_avg_24h"]
    df["ramp_momentum"] = df["momentum_1h_3h"] * df["is_ramp_up"]

    # Baseline lịch sử tại inference: ưu tiên thống kê từ toàn bộ dữ liệu trước ngày dự đoán.
    baselines = baselines or {}
    zh = baselines.get("zone_hour_mean", {})
    zdh = baselines.get("zone_dow_hour_mean", {})
    zm = baselines.get("zone_mean", {})
    hm = baselines.get("hour_mean", {})
    global_mean = float(baselines.get("global_mean", float(s.mean()) if len(s) else 0.0))

    zone_id_int = int(zone_id)
    df["zone_mean"] = float(zm.get(zone_id_int, global_mean))
    df["hour_mean"] = [float(hm.get(int(h), global_mean)) for h in df["hour"].tolist()]
    df["zone_hour_mean"] = [
        float(zh.get((zone_id_int, int(h)), zm.get(zone_id_int, hm.get(int(h), global_mean))))
        for h in df["hour"].tolist()
    ]
    df["zone_dow_hour_mean"] = [
        float(zdh.get((zone_id_int, int(dow), int(h)), zh.get((zone_id_int, int(h)), zm.get(zone_id_int, hm.get(int(h), global_mean)))))
        for dow, h in zip(df["day_of_week"].tolist(), df["hour"].tolist())
    ]

    # More aliases sometimes used in train variants.
    df["zone_hour_baseline"] = df["zone_hour_mean"]
    df["zone_dow_hour_baseline"] = df["zone_dow_hour_mean"]

    # Features required by low-hour zero-inflated V7 (00–04) và bản mở rộng 00–09.
    df["baseline_blend"] = 0.70 * df["zone_hour_mean"] + 0.30 * df["hour_mean"]
    df["night_order"] = np.where(df["hour"].between(0, 9), df["hour"], -1).astype(int)
    for h in range(10):
        df[f"is_{h:02d}"] = (df["hour"] == h).astype(int)
    df["is_deep_night"] = df["hour"].isin([0, 1, 2]).astype(int)
    df["is_pre_ramp"] = df["hour"].isin([3, 4]).astype(int)
    df["is_morning_ramp"] = df["hour"].isin([5, 6, 7, 8, 9]).astype(int)
    df["is_ramp_low"] = df["is_pre_ramp"]
    df["ramp_strength"] = np.clip((df["hour"] - 2) / 2.0, 0.0, 1.0)
    df["morning_ramp_strength"] = np.where(
        df["hour"].between(0, 9),
        np.clip((df["hour"].astype(float) - 2.0) / 7.0, 0.0, 1.0),
        0.0,
    )
    df["baseline_gap_1h"] = df["lag_1h"] - df["baseline_blend"]
    df["baseline_gap_24h"] = df["lag_24h"] - df["baseline_blend"]
    df["baseline_ratio_1h"] = _safe_div(df["lag_1h"] + 1.0, df["baseline_blend"] + 1.0)
    df["baseline_ratio_24h"] = _safe_div(df["lag_24h"] + 1.0, df["baseline_blend"] + 1.0)
    df["pre_ramp_gap"] = df["baseline_gap_1h"] * df["is_pre_ramp"]
    df["morning_gap_1h"] = df["baseline_gap_1h"] * df["is_morning_ramp"]
    df["ramp_baseline_signal"] = df["baseline_blend"] * df["ramp_strength"]

    # Evening specialist (17–23h) — train_lgbm_evening_17_23.py
    for hh in range(17, 24):
        df[f"is_h{hh}"] = (df["hour"] == hh).astype(int)
    df["evening_slot"] = np.where(
        df["hour"].between(17, 23),
        df["hour"] - 17,
        -1,
    ).astype(int)
    df["is_evening_segment_early"] = df["hour"].between(17, 19).astype(int)
    df["is_evening_segment_late"] = df["hour"].between(21, 23).astype(int)
    df["evening_decay"] = np.where(
        df["hour"].between(17, 23),
        (23 - df["hour"]) / 6.0,
        0.0,
    )
    df["evening_baseline_gap"] = df["baseline_gap_1h"] * df["is_evening_segment_late"]

    return df


def _lgbm_features(zone_id: int, demand_window: np.ndarray, timestamps: pd.DatetimeIndex,
                   holiday_dates: set, feat_cols: list, baselines: dict | None = None) -> pd.DataFrame:
    """
    Tạo đúng 1 dòng feature cho giờ cần dự đoán.

    Bug quan trọng đã sửa:
    - Bản cũ lấy tail(1) của cửa sổ lịch sử 168h, nghĩa là feature hour/holiday đang là
      giờ lịch sử cuối cùng (t-1), không phải giờ cần dự đoán (t).
    - Bản này append thêm 1 dòng placeholder tại pred_ts=timestamps[-1]+1h.
      Khi đó lag_1h/rolling vẫn lấy từ lịch sử, còn hour/day/month là của giờ t.
    """
    timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(timestamps) == 0:
        raise ValueError("_lgbm_features received empty timestamps")

    pred_ts = timestamps[-1] + pd.Timedelta(hours=1)
    ext_ts = timestamps.append(pd.DatetimeIndex([pred_ts]))

    demand_window = np.asarray(demand_window, dtype=np.float32)
    # Placeholder target_demand cho giờ cần predict. Không ảnh hưởng lag/rolling vì các feature dùng shift(1).
    ext_demand = np.concatenate([demand_window, np.array([0.0], dtype=np.float32)])

    df = _build_lgbm_feature_frame(zone_id, ext_demand, ext_ts, holiday_dates, baselines)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.warning("[GOLD] Missing LGBM features filled with 0: %s", missing)
    return df.reindex(columns=feat_cols, fill_value=0).tail(1)




def _low_hour_features(zone_id: int, demand_window: np.ndarray, timestamps: pd.DatetimeIndex,
                       holiday_dates: set, feat_cols: list, baselines: dict | None = None) -> pd.DataFrame:
    """Build one-row feature frame for the V7 low-hour model."""
    timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(timestamps) == 0:
        raise ValueError("_low_hour_features received empty timestamps")
    pred_ts = timestamps[-1] + pd.Timedelta(hours=1)
    ext_ts = timestamps.append(pd.DatetimeIndex([pred_ts]))
    demand_window = np.asarray(demand_window, dtype=np.float32)
    ext_demand = np.concatenate([demand_window, np.array([0.0], dtype=np.float32)])
    df = _build_lgbm_feature_frame(zone_id, ext_demand, ext_ts, holiday_dates, baselines)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.warning("[GOLD] Missing low-hour features filled with 0: %s", missing)
    return df.reindex(columns=feat_cols, fill_value=0).tail(1)


def _apply_low_hour_postprocess(prob: np.ndarray, reg_pred: np.ndarray, X: pd.DataFrame, params: dict) -> np.ndarray:
    """Apply V7 zero-inflated low-hour postprocess (deep / ramp / morning)."""
    prob = np.asarray(prob, dtype=float)
    reg_pred = np.clip(np.asarray(reg_pred, dtype=float), 0.0, None)
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
        ramp_mask = (hours >= 3) & (hours <= 4)
        pred[ramp_mask] = (1.0 - ramp_alpha) * pred[ramp_mask] + ramp_alpha * baseline_blend[ramp_mask]

    morning_alpha = float(params.get("morning_blend_alpha", 0.0))
    if morning_alpha > 0:
        morn_mask = hours >= 5
        pred[morn_mask] = (1.0 - morning_alpha) * pred[morn_mask] + morning_alpha * baseline_blend[morn_mask]

    cap_deep = baseline * float(params.get("cap_mult_deep", 3.0)) + float(params.get("cap_bias_deep", 2.0))
    cap_ramp = baseline * float(params.get("cap_mult_ramp", 3.0)) + float(params.get("cap_bias_ramp", 4.0))
    cap_morning = baseline * float(params.get("cap_mult_morning", 999.0)) + float(params.get("cap_bias_morning", 0.0))
    cap = np.where(hours <= 2, cap_deep, np.where(hours <= 4, cap_ramp, cap_morning))
    pred = np.minimum(pred, np.maximum(cap, 0.0))

    min_floor = float(params.get("min_floor", 0.0))
    pred = np.maximum(pred, min_floor)
    return np.clip(pred, 0.0, None)


def _predict_low_hour_zinf(X: pd.DataFrame, low_bundle: dict) -> np.ndarray:
    """Predict per-zone demand for low-hour window (00–04 hoặc 00–09) using V7 zero-inflated bundle."""
    prob = np.asarray(low_bundle["classifier"].predict(X)).reshape(-1)
    reg_pred = np.asarray(low_bundle["regressor"].predict(X)).reshape(-1)
    return _apply_low_hour_postprocess(prob, reg_pred, X, low_bundle.get("postprocess_params", {}))


def _probs_to_pred_evening(
    prob: np.ndarray, bucket_values: np.ndarray, X: pd.DataFrame, params: dict,
) -> np.ndarray:
    """Khớp train_lgbm_evening_17_23.py — probs_to_pred_evening."""
    prob = np.asarray(prob, dtype=float)
    bv = np.asarray(bucket_values, dtype=np.float64).reshape(-1)
    pred = prob @ bv
    hours = X["hour"].astype(int).values
    baseline = X.get("zone_hour_mean", pd.Series(np.zeros(len(X)))).astype(float).values
    baseline_blend = X.get("baseline_blend", pd.Series(baseline)).astype(float).values
    p_zero = prob[:, 0]
    z_early = float(params.get("zero_gate_early_ev", 1.10))
    z_late = float(params.get("zero_gate_late_ev", 1.10))
    gate = np.where(hours <= 19, z_early, z_late)
    pred = np.where(p_zero >= gate, 0.0, pred)
    blend_a = float(params.get("evening_blend_alpha", 0.0))
    if blend_a > 0:
        late_m = np.isin(hours, [21, 22, 23])
        pred[late_m] = (1.0 - blend_a) * pred[late_m] + blend_a * baseline_blend[late_m]
    cap_e = baseline * float(params.get("cap_mult_evening", 999.0)) + float(params.get("cap_bias_evening", 0.0))
    cap_l = baseline * float(params.get("cap_mult_late_ev", 999.0)) + float(params.get("cap_bias_late_ev", 0.0))
    cap = np.where(np.isin(hours, [21, 22, 23]), cap_l, cap_e)
    pred = np.minimum(pred, np.maximum(cap, 0.0))
    return np.clip(pred, 0.0, None)


def _evening_bucket_features(
    zone_id: int, demand_window: np.ndarray, timestamps: pd.DatetimeIndex,
    holiday_dates: set, feat_cols: list, baselines: dict | None = None,
) -> pd.DataFrame:
    timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(timestamps) == 0:
        raise ValueError("_evening_bucket_features received empty timestamps")
    pred_ts = timestamps[-1] + pd.Timedelta(hours=1)
    ext_ts = timestamps.append(pd.DatetimeIndex([pred_ts]))
    demand_window = np.asarray(demand_window, dtype=np.float32)
    ext_demand = np.concatenate([demand_window, np.array([0.0], dtype=np.float32)])
    df = _build_lgbm_feature_frame(zone_id, ext_demand, ext_ts, holiday_dates, baselines)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.warning("[GOLD] Missing evening-bucket features filled with 0: %s", missing)
    return df.reindex(columns=feat_cols, fill_value=0).tail(1)


def _predict_evening_reg_v9_core(
    X: pd.DataFrame,
    model,
    params: dict,
    *,
    late_residual_model,
    late_residual_feat_cols: list,
    sun_late_expert,
) -> np.ndarray:
    """Single-branch evening regression forward (calibration → weekend → residual → Sun expert → blend/cap)."""
    pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
    pred = np.clip(pred, 0.0, None)
    hours = X["hour"].astype(int).values
    dows = X.get("day_of_week", pd.Series(np.zeros(len(X), dtype=int))).astype(int).values
    baseline = X.get("zone_hour_mean", pd.Series(np.zeros(len(X)))).astype(float).values
    baseline_blend = X.get("baseline_blend", pd.Series(baseline)).astype(float).values
    hour_calib = params.get("hour_calibration", {}) if isinstance(params, dict) else {}
    dow_hour_calib = params.get("dow_hour_calibration", {}) if isinstance(params, dict) else {}
    late_bias = params.get("late_hour_bias", {}) if isinstance(params, dict) else {}
    weekend_late_factors = params.get("weekend_late_factors", {}) if isinstance(params, dict) else {}
    weekend_late_bias = params.get("weekend_late_bias", {}) if isinstance(params, dict) else {}
    if isinstance(hour_calib, dict) or isinstance(dow_hour_calib, dict):
        for i in range(len(pred)):
            h = int(hours[i])
            d = int(dows[i])
            key = f"{d}_{h}"
            f = 1.0
            if isinstance(dow_hour_calib, dict) and key in dow_hour_calib:
                f = float(dow_hour_calib[key])
            elif isinstance(hour_calib, dict):
                f = float(hour_calib.get(h, hour_calib.get(str(h), 1.0)))
            b = float(late_bias.get(h, late_bias.get(str(h), 0.0))) if isinstance(late_bias, dict) else 0.0
            pred[i] = max(pred[i] * f + b, 0.0)
    weekend_adjust_enabled = bool(params.get("weekend_adjust_enabled", True)) if isinstance(params, dict) else True
    if weekend_adjust_enabled and (isinstance(weekend_late_factors, dict) or isinstance(weekend_late_bias, dict)):
        for i in range(len(pred)):
            h = int(hours[i])
            if h not in [21, 22, 23]:
                continue
            d = int(dows[i])
            if d == 5:
                key = f"sat_{h}"
                f = float(weekend_late_factors.get(key, 1.0)) if isinstance(weekend_late_factors, dict) else 1.0
                b = float(weekend_late_bias.get(key, 0.0)) if isinstance(weekend_late_bias, dict) else 0.0
                pred[i] = max(pred[i] * f + b, 0.0)
            elif d == 6:
                key = f"sun_{h}"
                f = float(weekend_late_factors.get(key, 1.0)) if isinstance(weekend_late_factors, dict) else 1.0
                b = float(weekend_late_bias.get(key, 0.0)) if isinstance(weekend_late_bias, dict) else 0.0
                pred[i] = max(pred[i] * f + b, 0.0)
    residual_enabled = bool(params.get("late_hour_residual_enabled", False))
    residual_cols = late_residual_feat_cols if isinstance(late_residual_feat_cols, list) else []
    residual_hours = params.get("late_hour_residual_hours", [22, 23]) if isinstance(params, dict) else [22, 23]
    if residual_enabled and late_residual_model is not None and residual_cols:
        late_mask = np.isin(hours, np.asarray(residual_hours, dtype=int))
        if np.any(late_mask):
            x_late = X.loc[late_mask, [c for c in residual_cols if c != "base_pred_cal"]].copy()
            x_late["base_pred_cal"] = pred[late_mask]
            delta = np.asarray(late_residual_model.predict(x_late[residual_cols]), dtype=float).reshape(-1)
            max_abs = np.maximum(0.20 * pred[late_mask], 3.0)
            delta = np.clip(delta, -max_abs, max_abs)
            delta = np.clip(delta, -8.0, 8.0)
            pred[late_mask] = np.clip(pred[late_mask] + delta, 0.0, None)
    sun_hours = params.get("sun_late_expert_hours", [21, 22, 23]) if isinstance(params, dict) else [21, 22, 23]
    sun_blend = float(params.get("sun_late_expert_blend_alpha", 0.0)) if isinstance(params, dict) else 0.0
    sun_enabled = bool(params.get("sun_late_expert_enabled", False)) if isinstance(params, dict) else False
    if sun_enabled and sun_late_expert is not None and sun_blend > 1e-8:
        exp = np.clip(np.asarray(sun_late_expert.predict(X), dtype=float).reshape(-1), 0.0, None)
        for i in range(len(pred)):
            if int(dows[i]) == 6 and int(hours[i]) in sun_hours:
                pred[i] = (1.0 - sun_blend) * pred[i] + sun_blend * exp[i]
    alpha = float(params.get("blend_alpha_late", 0.0))
    if alpha > 0:
        late_m = np.isin(hours, [21, 22, 23])
        pred[late_m] = (1.0 - alpha) * pred[late_m] + alpha * baseline_blend[late_m]
    cap_e = baseline * float(params.get("cap_mult_evening", 999.0)) + float(params.get("cap_bias_evening", 0.0))
    cap_l = baseline * float(params.get("cap_mult_late", 999.0)) + float(params.get("cap_bias_late", 0.0))
    cap = np.where(np.isin(hours, [21, 22, 23]), cap_l, cap_e)
    return np.clip(np.minimum(pred, np.maximum(cap, 0.0)), 0.0, None)


def _predict_evening_bucket(X: pd.DataFrame, eve_bundle: dict) -> np.ndarray:
    params = eve_bundle.get("postprocess_params", {}) or {}
    if str(eve_bundle.get("type")) == "evening_reg_v9":
        w_ens = float(params.get("ensemble_quantile_weight", 0.0)) if isinstance(params, dict) else 0.0
        skip = frozenset({"postprocess_quantile", "ensemble_quantile_weight", "ensemble_mode"})
        post_tw = {k: v for k, v in params.items() if k not in skip} if isinstance(params, dict) else {}
        pred_tw = _predict_evening_reg_v9_core(
            X,
            eve_bundle["model"],
            post_tw,
            late_residual_model=eve_bundle.get("late_residual_model"),
            late_residual_feat_cols=eve_bundle.get("late_residual_feat_cols", []) or [],
            sun_late_expert=eve_bundle.get("sun_late_expert"),
        )
        mq = eve_bundle.get("quantile_model")
        post_qt = params.get("postprocess_quantile") if isinstance(params, dict) else None
        if w_ens > 1e-8 and mq is not None and isinstance(post_qt, dict):
            cols_q = eve_bundle.get("late_residual_feat_cols_quantile") or eve_bundle.get("late_residual_feat_cols", []) or []
            pred_qt = _predict_evening_reg_v9_core(
                X,
                mq,
                post_qt,
                late_residual_model=eve_bundle.get("late_residual_model_quantile"),
                late_residual_feat_cols=cols_q,
                sun_late_expert=None,
            )
            return (1.0 - w_ens) * pred_tw + w_ens * pred_qt
        return pred_tw

    raw = eve_bundle["model"].predict(X)
    prob = np.asarray(raw, dtype=float)
    if prob.ndim == 1:
        prob = prob.reshape(1, -1)
    return _probs_to_pred_evening(
        prob,
        eve_bundle["bucket_values"],
        X,
        params,
    )


def _apply_lgbm_calibration(y_pred: np.ndarray, X: pd.DataFrame, model_bundle: dict) -> np.ndarray:
    """Áp dụng calibration được lưu từ train, nếu có."""
    y = np.asarray(y_pred, dtype=np.float32).copy()

    ctx_calib = model_bundle.get("context_calibration")
    if ctx_calib is not None:
        factors = []
        for _, row in X.iterrows():
            key = (int(row.get("hour", 0)), int(row.get("is_weekend", 0)), int(row.get("is_holiday", 0)))
            factor = None
            if isinstance(ctx_calib, dict):
                factor = ctx_calib.get(key)
                if factor is None:
                    factor = ctx_calib.get(str(key))
                if factor is None:
                    factor = ctx_calib.get(int(row.get("hour", 0)))
            factors.append(float(factor) if factor is not None else 1.0)
        y *= np.asarray(factors, dtype=np.float32)

    hour_calib = model_bundle.get("hour_calibration")
    if hour_calib is not None:
        factors = []
        for h in X.get("hour", pd.Series(np.zeros(len(X), dtype=int))).astype(int).tolist():
            factor = None
            if isinstance(hour_calib, dict):
                factor = hour_calib.get(h)
                if factor is None:
                    factor = hour_calib.get(str(h))
            factors.append(float(factor) if factor is not None else 1.0)
        y *= np.asarray(factors, dtype=np.float32)

    return np.maximum(y, 0.0)

def _predict_day(target_date: date, hourly: pd.DataFrame, active_zones: list,
                 model_bundle: dict, model_type: str) -> list[dict]:
    """Rolling 24-step inference for either LSTM or LightGBM."""
    year = target_date.year
    ny_cal = hols.country_holidays("US", subdiv="NY", years=year)
    holiday_dates = {pd.Timestamp(d).normalize() for d in ny_cal.keys()}

    target_start = pd.Timestamp(target_date)

    _mb_cfg = model_bundle.get("cfg") or {}
    _lb_long = (
        model_bundle.get("look_back_long")
        or _mb_cfg.get("look_back_long")
        or model_bundle.get("look_back")
        or _mb_cfg.get("look_back")
    )
    if str(model_bundle.get("lstm_version", "")).lower() == "v5":
        model_look_back = max(int(_lb_long) if _lb_long is not None else 168, 168)
    else:
        model_look_back = int(model_bundle.get("look_back") or _mb_cfg.get("look_back") or LOOK_BACK)

    hist_start = target_start - pd.Timedelta(hours=model_look_back)

    lookup = _build_zone_hour_lookup(hourly)

    n_zones = len(active_zones)
    if n_zones == 0:
        log.warning("[GOLD] _predict_day: active_zones rỗng, không có dữ liệu để dự đoán.")
        return []

    hist_timestamps = pd.date_range(hist_start, periods=model_look_back, freq="h")
    zone_window = np.array([
        [lookup.get((z, _hour_floor_ts(h)), 0.0) for h in hist_timestamps]
        for z in active_zones
    ], dtype=np.float32)

    actuals = np.array([
        sum(
            lookup.get((z, _hour_floor_ts(target_start + pd.Timedelta(hours=hi))), 0.0)
            for z in active_zones
        )
        for hi in range(24)
    ], dtype=np.float32)

    has_actual_day = bool(np.nansum(actuals) > 0)
    use_actual_history = (
        os.getenv("LGBM_USE_ACTUAL_HISTORY", "1").lower() in ("1", "true", "yes")
        and has_actual_day
    )

    pred_by_hour = np.zeros(24, dtype=np.float32)
    model_type = (model_type or model_bundle.get("type", "lstm")).lower()

    lgbm_baselines = None
    if model_type in ("lightgbm", "lgbm"):
        lgbm_baselines = _build_lgbm_baselines(hourly, active_zones, target_start)
        log.info(
            "[GOLD] LGBM baselines: zones=%d | zone_hour=%d | zone_dow_hour=%d | global_mean=%.2f",
            n_zones,
            len(lgbm_baselines.get("zone_hour_mean", {})),
            len(lgbm_baselines.get("zone_dow_hour_mean", {})),
            float(lgbm_baselines.get("global_mean", 0.0)),
        )
        log.info(
            "[GOLD] LGBM actual-history mode: %s | has_actual_day=%s | total_actual=%.1f",
            "ON" if use_actual_history else "OFF", has_actual_day, float(np.nansum(actuals)),
        )
    elif model_type == "lstm" and (
        model_bundle.get("low_hour_model") is not None
        or model_bundle.get("evening_bucket_model") is not None
    ):
        # Baseline Gold cho feature LGBM (low-hour v7 hoặc evening bucket 17–23h).
        lgbm_baselines = _build_lgbm_baselines(hourly, active_zones, target_start)
        log.info(
            "[GOLD] LGBM baselines for LSTM hybrid (low-hour / evening): zones=%d | zone_hour keys=%d | global_mean=%.2f",
            n_zones,
            len(lgbm_baselines.get("zone_hour_mean", {})),
            float(lgbm_baselines.get("global_mean", 0.0)),
        )

    for hi in range(24):
        infer_model_name = "MAIN-LGBM" if model_type in ("lightgbm", "lgbm") else "LSTM"
        ctx_start = hist_start + pd.Timedelta(hours=hi)
        ctx_ts = pd.date_range(ctx_start, periods=model_look_back, freq="h")

        if use_actual_history:
            zone_window_for_pred = np.array([
                [lookup.get((z, _hour_floor_ts(h)), 0.0) for h in ctx_ts]
                for z in active_zones
            ], dtype=np.float32)
        else:
            zone_window_for_pred = zone_window

        if model_type == "lstm":
            # ── 00h-09h (artifact 0_9): optional LightGBM ZINF — ưu tiên trước LSTM / specialist 00_09 ──
            low_lgbm = model_bundle.get("low_hour_model")
            _low_hrs = _low_hour_effective_hours(low_lgbm)
            use_low_hour_lgbm = low_lgbm is not None and hi in _low_hrs
            eve_b = model_bundle.get("evening_bucket_model")
            use_evening_bucket = eve_b is not None and 17 <= hi <= 23
            if use_low_hour_lgbm:
                rows = [
                    _low_hour_features(
                        z, zone_window_for_pred[zi], ctx_ts, holiday_dates,
                        low_lgbm["feat_cols"], lgbm_baselines,
                    )
                    for zi, z in enumerate(active_zones)
                ]
                X_low = pd.concat(rows, ignore_index=True)
                y_pred = _predict_low_hour_zinf(X_low, low_lgbm)
                infer_model_name = f"LOW-HOUR:{Path(str(low_lgbm.get('model_dir', ''))).name}"
            elif use_evening_bucket:
                rows = [
                    _evening_bucket_features(
                        z, zone_window_for_pred[zi], ctx_ts, holiday_dates,
                        eve_b["feat_cols"], lgbm_baselines,
                    )
                    for zi, z in enumerate(active_zones)
                ]
                X_e = pd.concat(rows, ignore_index=True)
                y_pred = _predict_evening_bucket(X_e, eve_b)
                infer_model_name = f"EVENING-BUCKET:{Path(str(eve_b.get('model_dir', ''))).name}"
            else:
                # ── LSTM PyTorch: v5 dual-context (train_lstm_demand.py) hoặc v4 một nhánh ──
                active_bundle = model_bundle
                specialists = model_bundle.get("specialists") or {}
                if 0 <= hi <= 9 and "00_09" in specialists:
                    active_bundle = specialists["00_09"]
                elif 17 <= hi <= 23 and "17_23" in specialists:
                    active_bundle = specialists["17_23"]

                device = active_bundle["device"]
                feat_cols = active_bundle["feat_cols"]
                feat_scaler = active_bundle["feat_scaler"]
                pytorch_model = active_bundle["model"]
                lstm_ver = active_bundle.get("lstm_version", "v4")

                if lstm_ver == "v5":
                    lb_short = int(active_bundle.get("look_back_short", 72))
                    lb_long = int(active_bundle.get("look_back_long", 168))
                    T_pred = target_start + pd.Timedelta(hours=hi)
                    ts_long = pd.date_range(
                        T_pred - pd.Timedelta(hours=lb_long),
                        periods=lb_long,
                        freq="h",
                    )
                    X_short_list = []
                    X_long_list = []
                    for zi in range(n_zones):
                        z = active_zones[zi]
                        demand_long = np.array(
                            [lookup.get((z, _hour_floor_ts(t)), 0.0) for t in ts_long],
                            dtype=np.float32,
                        )
                        X_full = _zone_features(
                            demand_long,
                            ts_long,
                            holiday_dates,
                            feat_cols,
                            feat_scaler,
                            look_back=lb_long,
                        )
                        X_long_list.append(X_full)
                        X_short_list.append(X_full[-lb_short:, :])
                    Xs = np.stack(X_short_list, axis=0)
                    Xl = np.stack(X_long_list, axis=0)
                    hour_ids = np.full(n_zones, int(T_pred.hour), dtype=np.int64)
                    dow_ids = np.full(n_zones, int(T_pred.dayofweek), dtype=np.int64)
                    zone_ids = np.array(active_zones, dtype=np.int64)
                    y_scaled_flat = pytorch_model.predict_numpy_v5(
                        Xs, Xl, hour_ids, dow_ids, zone_ids, device, batch_size=256,
                    )
                    y_scaled = y_scaled_flat.reshape(-1, 1).astype(np.float32)
                    y_pred = np.maximum(
                        active_bundle["tgt_scaler"].inverse_transform(y_scaled).flatten(), 0.0
                    )
                    dow_hour_calib = active_bundle.get("dow_hour_calibration")
                    hour_calib = active_bundle.get("hour_calibration")
                    hour_bias = active_bundle.get("hour_bias")
                    hour_affine = active_bundle.get("hour_affine")
                    hh = int(T_pred.hour)
                    critical_hours = set(range(0, 9)) | {21, 22, 23}
                    if isinstance(hour_affine, dict) and hh in critical_hours and hh in hour_affine:
                        params = hour_affine.get(hh, hour_affine.get(str(hh), {}))
                        a = float(params.get("a", 1.0))
                        b = float(params.get("b", 0.0))
                        y_pred = np.maximum(a * y_pred + b, 0.0)
                    else:
                        factor = 1.0
                        if isinstance(dow_hour_calib, dict):
                            key = f"{int(T_pred.dayofweek)}_{hh}"
                            if key in dow_hour_calib:
                                factor = float(dow_hour_calib[key])
                        if factor == 1.0 and isinstance(hour_calib, dict):
                            factor = float(hour_calib.get(hh, hour_calib.get(str(hh), 1.0)))
                        bias = 0.0
                        if isinstance(hour_bias, dict):
                            bias = float(hour_bias.get(hh, hour_bias.get(str(hh), 0.0)))
                        y_pred = np.maximum(y_pred * factor + bias, 0.0)
                else:
                    X = np.stack([
                        _zone_features(
                            zone_window_for_pred[zi], ctx_ts, holiday_dates,
                            feat_cols, feat_scaler, look_back=model_look_back,
                        )
                        for zi in range(n_zones)
                    ])
                    zone_ids = np.array(active_zones, dtype=np.int64)
                    hour_ids = np.full(n_zones, int((target_start + pd.Timedelta(hours=hi)).hour), dtype=np.int64)
                    y_scaled_flat = pytorch_model.predict_numpy(X, zone_ids, hour_ids, device, batch_size=256)
                    y_scaled = y_scaled_flat.reshape(-1, 1).astype(np.float32)
                    y_pred = np.maximum(
                        active_bundle["tgt_scaler"].inverse_transform(y_scaled).flatten(), 0.0
                    )
                    dow_hour_calib = active_bundle.get("dow_hour_calibration")
                    hour_calib = active_bundle.get("hour_calibration")
                    hour_bias = active_bundle.get("hour_bias")
                    hour_affine = active_bundle.get("hour_affine")
                    hour_for_pred = int((target_start + pd.Timedelta(hours=hi)).hour)
                    dow_for_pred = int((target_start + pd.Timedelta(hours=hi)).dayofweek)
                    critical_hours = set(range(0, 9)) | {21, 22, 23}
                    if isinstance(hour_affine, dict) and hour_for_pred in critical_hours and hour_for_pred in hour_affine:
                        params = hour_affine.get(hour_for_pred, hour_affine.get(str(hour_for_pred), {}))
                        a = float(params.get("a", 1.0))
                        b = float(params.get("b", 0.0))
                        y_pred = np.maximum(a * y_pred + b, 0.0)
                    else:
                        factor = 1.0
                        if isinstance(dow_hour_calib, dict):
                            key = f"{dow_for_pred}_{hour_for_pred}"
                            if key in dow_hour_calib:
                                factor = float(dow_hour_calib[key])
                        if factor == 1.0 and isinstance(hour_calib, dict):
                            factor = float(hour_calib.get(hour_for_pred, hour_calib.get(str(hour_for_pred), 1.0)))
                        bias = 0.0
                        if isinstance(hour_bias, dict):
                            bias = float(hour_bias.get(hour_for_pred, hour_bias.get(str(hour_for_pred), 0.0)))
                        y_pred = np.maximum(y_pred * factor + bias, 0.0)
        else:
            low_bundle = model_bundle.get("low_hour_model")
            eve_bundle = model_bundle.get("evening_bucket_model")
            use_low_hour = low_bundle is not None and hi in _low_hour_effective_hours(low_bundle)
            use_evening_main = eve_bundle is not None and 17 <= hi <= 23
            if use_low_hour:
                rows = [
                    _low_hour_features(z, zone_window_for_pred[zi], ctx_ts, holiday_dates, low_bundle["feat_cols"], lgbm_baselines)
                    for zi, z in enumerate(active_zones)
                ]
                X = pd.concat(rows, ignore_index=True)
                y_pred = _predict_low_hour_zinf(X, low_bundle)
                infer_model_name = f"LOW-HOUR:{Path(str(low_bundle.get('model_dir', ''))).name}"
            elif use_evening_main:
                rows = [
                    _evening_bucket_features(z, zone_window_for_pred[zi], ctx_ts, holiday_dates, eve_bundle["feat_cols"], lgbm_baselines)
                    for zi, z in enumerate(active_zones)
                ]
                X = pd.concat(rows, ignore_index=True)
                y_pred = _predict_evening_bucket(X, eve_bundle)
                infer_model_name = f"EVENING-BUCKET:{Path(str(eve_bundle.get('model_dir', ''))).name}"
            else:
                rows = [
                    _lgbm_features(z, zone_window_for_pred[zi], ctx_ts, holiday_dates, model_bundle["feat_cols"], lgbm_baselines)
                    for zi, z in enumerate(active_zones)
                ]
                X = pd.concat(rows, ignore_index=True)
                y_scaled = np.asarray(model_bundle["model"].predict(X)).reshape(-1, 1)
                y_pred = np.maximum(model_bundle["tgt_scaler"].inverse_transform(y_scaled).flatten(), 0.0)
                y_pred = _apply_lgbm_calibration(y_pred, X, model_bundle)
                infer_model_name = "MAIN-LGBM"

        pred_by_hour[hi] = float(y_pred.sum())
        log_this_hour = model_type in ("lightgbm", "lgbm") or (
            model_type == "lstm"
            and (
                str(infer_model_name).startswith("LOW-HOUR")
                or str(infer_model_name).startswith("EVENING-BUCKET")
            )
        )
        if log_this_hour:
            actual_sum_hi = float(actuals[hi]) if hi < len(actuals) else 0.0
            log.info(
                "[GOLD][%s] %s %02d:00 | zones=%d | pred_sum=%.1f | actual_sum=%.1f | ratio=%.3f",
                infer_model_name, target_date, hi, n_zones, pred_by_hour[hi], actual_sum_hi,
                pred_by_hour[hi] / max(actual_sum_hi, 1.0),
            )

        # Cập nhật rolling window cho chế độ dự đoán tương lai.
        # Với use_actual_history=True, vòng sau sẽ dựng lại context từ lookup nên
        # phần này không ảnh hưởng kết quả, nhưng vẫn giữ để fallback an toàn.
        if has_actual_day:
            ts_hi = _hour_floor_ts(target_start + pd.Timedelta(hours=hi))
            actual_hi = np.array([
                lookup.get((z, ts_hi), float(y_pred[zi]))
                for zi, z in enumerate(active_zones)
            ], dtype=np.float32)
            zone_window = np.hstack([zone_window[:, 1:], actual_hi.reshape(-1, 1)])
        else:
            zone_window = np.hstack([zone_window[:, 1:], y_pred.reshape(-1, 1)])

        gs = read_status("gold")
        if gs.get("state") == "predicting":
            gs.update({"zones_done": n_zones, "hours_predicted": hi + 1})
            write_status("gold", gs)

    actual_available = bool(np.nansum(actuals) > 0)
    return [
        {
            "hour": hi,
            "actual": round(float(actuals[hi])),
            "predicted": round(float(pred_by_hour[hi])),
            "error_pct": (
                round(abs(pred_by_hour[hi] - actuals[hi]) / max(actuals[hi], 1) * 100, 1)
                if actual_available else None
            ),
            "actual_available": actual_available,
        }
        for hi in range(24)
    ]

def _fmt_pred(result: list[dict], date_str: str, day_type: str, holiday_name: str) -> dict:
    hours     = [r["hour"]       for r in result]
    actual    = [r["actual"]     for r in result]
    predicted = [r["predicted"]  for r in result]
    errors    = [r.get("error_pct") for r in result]
    actual_available = any(bool(r.get("actual_available", False)) for r in result) and sum(actual) > 0
    valid_errors = [float(e) for e in errors if e is not None]

    # MAPE trung bình theo giờ: mỗi khung 1/24 — giờ đêm volume thấp nhưng % sai có trọng số như giờ cao điểm → dễ "méo" so với cảm nhận nghiệp vụ
    mape_avg_hourly = (
        round(sum(valid_errors) / len(valid_errors), 1)
        if actual_available and valid_errors else None
    )

    # WAPE (volume-weighted): sum|error|/sum(actual) — khớp metrics training, ưu tiên hiển thị
    wape = None
    if actual_available and sum(actual) > 0:
        tot_a = float(sum(actual))
        tot_e = float(
            sum(abs(float(predicted[i]) - float(actual[i])) for i in range(len(actual)))
        )
        wape = round(tot_e / max(tot_a, 1e-9) * 100, 1)

    # sMAPE theo ngày (ổn định khi actual hoặc pred gần 0)
    smape_day = None
    if actual_available and len(actual) > 0:
        smape_day = round(
            sum(
                200.0
                * abs(float(predicted[i]) - float(actual[i]))
                / max(abs(float(actual[i])) + abs(float(predicted[i])), 1e-9)
                for i in range(len(actual))
            )
            / len(actual),
            1,
        )

    # Trung vị % sai theo giờ — ít nhạy outlier hơn MAPE trung bình
    mdape_hourly = (
        round(float(np.median(np.asarray(valid_errors, dtype=np.float64))), 1)
        if valid_errors else None
    )

    return {
        "date":         date_str,
        "day_type":     day_type,
        "holiday_name": holiday_name,
        "hours":        hours,
        "actual":       actual,
        "predicted":    predicted,
        "errors":       errors,
        # backward compat: "mape" = WAPE (khuyến nghị); MAPE trung bình theo giờ = mape_avg_hourly
        "mape":         wape if wape is not None else mape_avg_hourly,
        "wape":         wape,
        "mape_avg_hourly": mape_avg_hourly,
        "smape_day":    smape_day,
        "mdape_hourly": mdape_hourly,
        "actual_available": actual_available,
        "total_actual": sum(actual),
        "total_predicted": sum(predicted),
    }
