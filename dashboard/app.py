"""
dashboard/app.py
────────────────────────────────────────────────────────────────
NYC Taxi Demand — Pipeline Dashboard
Layout:
  Hàng trên  : [Bronze] [Silver] [Gold]   ← real-time pipeline
  Hàng dưới  : [Ngày thường] [Ngày lễ]   ← predictions
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler as _StandardScaler
# ── PHẢI định nghĩa ở module level của __main__ ──────────────
# Scaler được pickle với tên __main__.Log1pMinMaxScaler khi train
# joblib sẽ tìm class này trong module __main__ (= app.py khi Streamlit chạy)
class Log1pStandardScaler:
    """Phải định nghĩa ở __main__ để joblib.load target_scaler.pkl của LSTM v4 hoạt động."""
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
class Log1pMinMaxScaler:
    """Dùng để load scaler của LightGBM (đã train trước đó)"""
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
# ── Import pipeline (runs in same container) ──────────────────
import sys
sys.path.insert(0, "/app")
from dashboard.pipeline import run_pipeline, reset_all_status, read_status, write_status

logging.basicConfig(level=logging.INFO)

# ── Constants ─────────────────────────────────────────────────
DATASET_DIRS = [
    Path("/app/dataset/streaming/2025"),
    Path("/app/dataset/streaming/2024"),
    Path("/app/dataset/streaming"),
    Path("/app/dataset/batch"),
]
MODEL_DIR   = Path("/app/models")
STATUS_DIR  = Path("/tmp/datn_pipeline")

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NYC Taxi Pipeline Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
  .panel-box {
      border: 1px solid #333; border-radius: 8px;
      padding: 12px 16px; min-height: 220px;
  }
  .panel-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 6px; }
  .metric-row  { display: flex; gap: 20px; margin-bottom: 8px; }
  .metric-item { text-align: center; }
  .metric-val  { font-size: 1.4rem; font-weight: 700; }
  .metric-lbl  { font-size: 0.75rem; color: #888; }
</style>
""", unsafe_allow_html=True)

# ── Auto-refresh every 2 seconds while pipeline active ────────
pipeline_state = read_status("pipeline").get("state", "IDLE")
active_states  = {"RUNNING", "BRONZE_INGESTING", "SILVER_PROCESSING", "GOLD_PROCESSING"}
refresh_ms     = 2000 if pipeline_state in active_states else 10000
st_autorefresh(interval=refresh_ms, key="auto_refresh")


# ════════════════════════════════════════════════════════════════
# HEADER & CONTROLS
# ════════════════════════════════════════════════════════════════
st.markdown("## 🚕 NYC Taxi Demand — Pipeline Dashboard")

def _find_parquet_files() -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for root in DATASET_DIRS:
        if not root.exists():
            continue
        for f in root.rglob("*.parquet"):
            key = str(f.resolve())
            if key not in seen:
                files.append(f)
                seen.add(key)
    return sorted(files, key=lambda p: p.name)

available = _find_parquet_files()
file_map  = {f.name: str(f) for f in available}


def parse_ym_from_filename(name: str) -> tuple[int, int] | None:
    """Parse yellow_tripdata_YYYY-MM.parquet style filenames."""
    import re
    m = re.search(r"(20\d{2})-(\d{2})", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def list_model_dirs(model_type: str) -> list[str]:
    """Liệt kê các model artifact có thể chọn theo loại model."""
    if not MODEL_DIR.exists():
        return []
    model_type = (model_type or "lstm").lower()
    if model_type == "lstm":
        dirs = [d.name for d in MODEL_DIR.iterdir() if d.is_dir() and d.name.startswith("lstm")]
    else:
        dirs = [d.name for d in MODEL_DIR.iterdir() if d.is_dir() and d.name.startswith("lgbm_")]
    # Tên thư mục có timestamp nên sort giảm dần để model mới ở trên.
    return sorted(dirs, reverse=True)


def render_live_rows(rows: list[dict], caption: str):
    if rows:
        st.caption(caption)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=250)
    else:
        st.caption("Chờ dữ liệu...")

c1, cdate, c2, c3, c4 = st.columns([2.6, 1.5, 2.0, 1.3, 2.0])
with c1:
    file_options = list(file_map.keys()) or ["Không có file parquet trong /app/dataset"]
    selected_file = st.selectbox(
        "Chọn file dữ liệu",
        options=file_options,
        label_visibility="collapsed",
    )

with cdate:
    ym = parse_ym_from_filename(selected_file)
    if ym:
        import calendar
        from datetime import date
        y, m = ym
        default_day = min(15, calendar.monthrange(y, m)[1])
        selected_target_date = st.date_input(
            "Chọn ngày dự đoán",
            value=date(y, m, default_day),
            min_value=date(y, m, 1),
            max_value=date(y, m, calendar.monthrange(y, m)[1]),
            label_visibility="collapsed",
            key=f"target_date_{selected_file}",
        )
    else:
        selected_target_date = st.date_input(
            "Chọn ngày dự đoán",
            label_visibility="collapsed",
            key=f"target_date_{selected_file}",
        )

with c2:
    model_choice_label = st.selectbox(
        "Chọn loại model dự đoán",
        options=["LSTM", "LightGBM"],
        label_visibility="collapsed",
        key="model_type_select",
    )
    model_choice = "lightgbm" if model_choice_label == "LightGBM" else "lstm"

    model_options = list_model_dirs(model_choice)
    if model_options:
        selected_model_dir = st.selectbox(
            "Chọn phiên bản model",
            options=model_options,
            label_visibility="collapsed",
            key=f"model_version_select_{model_choice}",
        )
        st.caption(f"Model đang chọn: `{selected_model_dir}`")
    else:
        selected_model_dir = None
        st.caption("Chưa có model phù hợp trong `/app/models`")

with c3:
    can_start = (
        pipeline_state not in active_states
        and selected_file in file_map
        and selected_model_dir is not None
    )
    if st.button("▶ Bắt đầu Pipeline", disabled=not can_start, use_container_width=True):
        reset_all_status()
        write_status("pipeline", {
            "state": "BRONZE_INGESTING",
            "file": selected_file,
            "model_type": model_choice,
            "model_dir": selected_model_dir,
            "target_date": str(selected_target_date),
            "started_at": time.time(),
        })
        threading.Thread(
            target=run_pipeline,
            args=(file_map[selected_file], model_choice, selected_model_dir, str(selected_target_date)),
            daemon=True,
        ).start()
        st.rerun()

with c4:
    color_map = {
        "IDLE": "#888", "RUNNING": "#f0a500", "BRONZE_INGESTING": "#f0a500",
        "SILVER_PROCESSING": "#4fa3e0", "GOLD_PROCESSING": "#4fa3e0",
        "DONE": "#4caf50", "ERROR": "#f44336",
    }
    clr = color_map.get(pipeline_state, "#888")
    elapsed = read_status("pipeline").get("elapsed_sec", "")
    extra   = f" — {elapsed}s" if elapsed and pipeline_state == "DONE" else ""
    st.markdown(
        f"<div style='padding-top:8px;font-size:0.95rem'>"
        f"Trạng thái: <b><span style='color:{clr}'>{pipeline_state}{extra}</span></b></div>",
        unsafe_allow_html=True,
    )
    if pipeline_state == "ERROR":
        err = read_status("pipeline").get("error", "")
        st.error(err[:120])

st.divider()


# ════════════════════════════════════════════════════════════════
# TOP ROW — 3 PIPELINE PANELS
# ════════════════════════════════════════════════════════════════
top1, top2, top3 = st.columns(3, gap="medium")

# ── BRONZE panel ──────────────────────────────────────────────
with top1:
    bs = read_status("bronze")
    done_ratio = (bs.get("written_rows", 0) / max(bs.get("total_rows", 1), 1))
    b_state    = bs.get("state", "waiting")

    if b_state == "done":
        icon = "🟫✅"
    elif b_state == "producing":
        icon = "🟫⏳"
    else:
        icon = "🟫"

    st.markdown(f"**{icon} Bronze — Raw Ingest (Kafka → Delta)**")
    st.progress(min(done_ratio, 1.0))

    m1, m2, m3 = st.columns(3)
    m1.metric("Đã ghi",      f"{bs.get('written_rows', 0):,}")
    m2.metric("Tổng cộng",   f"{bs.get('total_rows', 0):,}")
    m3.metric("Throughput",  f"{bs.get('throughput_rps', 0):,} r/s")

    m4, m5 = st.columns(2)
    m4.metric("Batches đã gửi",  f"{bs.get('batches_sent', 0)}/{bs.get('n_batches', 0)}")
    m5.metric("Thời gian",       f"{bs.get('elapsed_sec', 0)}s")

    render_live_rows(bs.get("sample_rows", []), "Các dòng raw mới nhất đang được ingest vào Bronze")

# ── SILVER panel ──────────────────────────────────────────────
with top2:
    ss = read_status("silver")
    s_state = ss.get("state", "waiting")
    s_ratio = (ss.get("output_rows", 0) / max(ss.get("input_rows", 1), 1))

    if s_state == "done":
        icon = "🥈✅"
    elif s_state in ("transforming", "writing"):
        icon = "🥈⏳"
    else:
        icon = "🥈"

    st.markdown(f"**{icon} Silver — Transform & Clean**")
    st.progress(min(s_ratio, 1.0) if s_state == "done" else
                (0.5 if s_state == "transforming" else
                 (0.8 if s_state == "writing" else 0.0)))

    m1, m2, m3 = st.columns(3)
    m1.metric("Input rows",   f"{ss.get('input_rows',  0):,}")
    m2.metric("Output rows",  f"{ss.get('output_rows', 0):,}")
    m3.metric("Filtered",     f"{ss.get('filtered_pct', 0):.1f}%")
    st.metric("Thời gian",    f"{ss.get('elapsed_sec', 0)}s")

    if s_state in ("done", "writing", "transforming", "finalizing") and ss.get("input_rows", 0) > 0:
        render_live_rows(ss.get("sample_rows", []), "Các dòng clean mới nhất đang được ghi vào Silver")
    else:
        st.caption("Chờ Bronze có dữ liệu...")

# ── GOLD panel ────────────────────────────────────────────────
with top3:
    gs = read_status("gold")
    g_state    = gs.get("state", "waiting")
    total_z    = gs.get("total_zones", 54)
    done_z     = gs.get("zones_done", 0)
    g_ratio    = done_z / max(total_z, 1)

    if g_state == "done":
        icon = "🥇✅"
    elif g_state in ("aggregating", "finalizing", "predicting"):
        icon = "🥇⏳"
    else:
        icon = "🥇"

    st.markdown(f"**{icon} Gold — Predictions**")
    st.progress(min(g_ratio, 1.0) if g_state == "done" else
                (0.3 if g_state == "aggregating" else
                 (0.7 if g_state == "predicting" else 0.0)))

    m1, m2 = st.columns(2)
    m1.metric("Zones dự đoán", f"{done_z}/{total_z}")
    m2.metric("Thời gian",     f"{gs.get('elapsed_sec', 0)}s")
    m3, m4 = st.columns(2)
    m3.metric("Model", gs.get("model_type", "-"))
    m4.metric("Giờ đã gom/predict", gs.get("hours_predicted", gs.get("hours_ready", 0)))
    if gs.get("model_dir"):
        st.caption(f"🧠 Model version: `{Path(str(gs.get('model_dir'))).name}`")

    if gs.get("target_date"):
        st.caption(f"📌 Ngày dự đoán: **{gs['target_date']}**")
    if gs.get("actual_source"):
        st.caption(f"📊 Actual source: `{gs['actual_source']}`")

    # Gauge-style progress
    if g_state != "waiting":
        render_live_rows(gs.get("sample_rows", []), "Các dòng hourly/prediction mới nhất trong Gold")

    if g_state == "done":
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=100,
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color="#FFD700"),
                bgcolor="rgba(0,0,0,0)",
                steps=[dict(range=[0, 100], color="#1a1a1a")],
            ),
            number=dict(suffix="%", font=dict(size=28)),
            domain=dict(x=[0.1, 0.9], y=[0, 1]),
        ))
        fig.update_layout(
            height=180, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="gold_progress_gauge")
    else:
        st.caption("Gold đang chờ Silver có dữ liệu...")

st.divider()


# ════════════════════════════════════════════════════════════════
# BOTTOM ROW — PREDICTION PANEL
# ════════════════════════════════════════════════════════════════
bot1 = st.container()

gold_status  = read_status("gold")
predictions  = gold_status.get("predictions", {})
gold_done    = gold_status.get("state") == "done"


def render_prediction(col, pred_data: dict | None, title: str, icon: str, panel_key: str):
    with col:
        st.markdown(f"**{icon} {title}**")
        if not gold_done or pred_data is None:
            st.info("Đang chờ pipeline hoàn tất..." if not gold_done
                    else "Không có dữ liệu dự đoán.")
            return

        date_str     = pred_data.get("date", "")
        holiday_name = pred_data.get("holiday_name", "")
        # mape = WAPE (trọng số theo nhu cầu thực) — ưu tiên hơn MAPE trung bình theo giờ
        wape         = pred_data.get("wape", pred_data.get("mape"))
        mape_avg_h   = pred_data.get("mape_avg_hourly")
        smape_d      = pred_data.get("smape_day")
        mdape_h      = pred_data.get("mdape_hourly")
        actual_available = bool(pred_data.get("actual_available", True))
        total_act    = pred_data.get("total_actual", 0)
        total_pred   = pred_data.get("total_predicted", 0)
        hours        = pred_data.get("hours", list(range(24)))
        actual       = pred_data.get("actual", [0] * 24)
        predicted    = pred_data.get("predicted", [0] * 24)
        errors_raw   = pred_data.get("errors", [None] * 24)
        errors       = [float(e) if e is not None else 0.0 for e in errors_raw]

        # Sub-header
        sub = f"{date_str}"
        if holiday_name:
            sub += f"  —  🎉 {holiday_name}"
        st.caption(sub)

        # Key metrics: WAPE = tổng trọng số volume; MAPE_tb/giờ = mỗi khung 1/24 (dễ lệch so cảm nhận đêm vs trưa)
        ma, mb, mc, md = st.columns(4)
        if actual_available and wape is not None:
            ma.metric("WAPE", f"{wape}%")
        else:
            ma.metric("WAPE", "N/A")
            st.caption("Không có actual cho ngày này trong Silver/Gold nên không tính sai số.")
        parts = []
        if mape_avg_h is not None:
            parts.append(f"MAPE(tb/giờ) {mape_avg_h}%")
        if mdape_h is not None:
            parts.append(f"MdAPE {mdape_h}%")
        if smape_d is not None:
            parts.append(f"sMAPE {smape_d}%")
        mb.metric("Theo từng giờ", " · ".join(parts) if parts else "N/A")
        mc.metric("Thực tế (tổng)",   f"{total_act:,}" if actual_available else "N/A")
        md.metric("Dự đoán (tổng)",   f"{total_pred:,}")

        # Grouped bar chart
        hour_labels = [f"{h:02d}:00" for h in hours]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Thực tế",
            x=hour_labels, y=actual,
            marker_color="#4fa3e0",
            hovertemplate="%{x}<br>Thực tế: %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Dự đoán",
            x=hour_labels, y=predicted,
            marker_color="#f0a500",
            hovertemplate="%{x}<br>Dự đoán: %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            name="Sai số %",
            x=hour_labels, y=errors,
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#f44336", width=1.5, dash="dot"),
            marker=dict(size=4),
            hovertemplate="%{x}<br>Sai số: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            barmode="group",
            height=300,
            margin=dict(l=0, r=40, t=10, b=40),
            legend=dict(orientation="h", y=1.05, x=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(
                title="Số chuyến",
                gridcolor="#333",
                showgrid=True,
            ),
            yaxis2=dict(
                title="Sai số %",
                overlaying="y",
                side="right",
                range=[0, max(float(max(errors, default=0.0)) * 1.5, 30.0)],
                showgrid=False,
            ),
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(size=10),
            ),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"prediction_chart_{panel_key}",
        )

        # Hourly table (compact)
        df_tbl = pd.DataFrame({
            "Giờ":      hour_labels,
            "Thực tế":  [f"{v:,}" for v in actual],
            "Dự đoán":  [f"{v:,}" for v in predicted],
            "Sai số":   [f"{float(e):.1f}%" if actual_available and e is not None else "N/A" for e in errors_raw],
        })
        with st.expander(f"Xem bảng chi tiết theo giờ — {title}"):
            st.dataframe(df_tbl, use_container_width=True, hide_index=True)


render_prediction(bot1, predictions.get("custom"), "Ngày tự chọn", "📌", "custom")
