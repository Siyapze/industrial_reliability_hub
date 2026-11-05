import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from copilot import generate_copilot_summary
from models import train_anomaly_model, summarize_asset_health
from pipelines import (
    build_quality_report,
    load_dataset,
    prepare_feature_table,
)


st.set_page_config(page_title="Industrial Reliability Hub", page_icon="🏭", layout="wide")


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for asset in ["Pump-A", "Pump-B", "Pump-C"]:
        for i in range(240):
            timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i)
            vibration = 0.8 + (i % 80) * 0.04 + (0.15 if asset == "Pump-B" else 0.0)
            temperature = 62 + (i % 70) * 0.12 + (3 if asset == "Pump-C" else 0)
            flow = 120 + 6 * (i % 40) / 10
            pressure = 70 + (i % 50) * 0.8
            if i > 180 and asset == "Pump-B":
                vibration += 1.4
                temperature += 8
                pressure += 12
            rows.append(
                {
                    "timestamp": timestamp,
                    "asset_id": asset,
                    "vibration_mm_s": round(vibration, 3),
                    "temperature_c": round(temperature, 2),
                    "flow_lpm": round(flow, 2),
                    "pressure_bar": round(pressure, 2),
                }
            )
    return pd.DataFrame(rows)


st.title("Industrial Reliability Hub")
st.caption("Predictive maintenance and operational intelligence for industrial assets")

with st.sidebar:
    st.header("Controls")
    st.markdown("### Data source")
    source = st.radio("Select input", ["Sample dataset", "Upload CSV"], index=0)

    uploaded_file = None
    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload sensor telemetry", type=["csv", "xlsx"])

    st.markdown("### Monitoring settings")
    contamination = st.slider("Anomaly contamination", min_value=0.01, max_value=0.2, value=0.08, step=0.01)
    target_asset = st.selectbox("Focus asset", ["All", "Pump-A", "Pump-B", "Pump-C"])


if uploaded_file is not None:
    df = load_dataset(uploaded_file)
else:
    df = load_sample_data()

if df.empty:
    st.warning("No data available. Please upload a CSV file with telemetry columns.")
    st.stop()

quality_report = build_quality_report(df)
feature_df = prepare_feature_table(df)

col1, col2, col3 = st.columns(3)
col1.metric("Assets", feature_df["asset_id"].nunique())
col2.metric("Rows", len(feature_df))
col3.metric("Quality score", f"{quality_report['quality_score']:.0f}%")

with st.expander("Quality report", expanded=False):
    st.json(quality_report)


st.subheader("Telemetry overview")
chart_df = feature_df.copy()
chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
metrics = ["vibration_mm_s", "temperature_c", "flow_lpm", "pressure_bar"]
selected_metric = st.selectbox("Metric to visualize", metrics)

if selected_metric in chart_df.columns:
    if target_asset != "All":
        chart_df = chart_df[chart_df["asset_id"] == target_asset]
    st.line_chart(chart_df.set_index("timestamp")[selected_metric])

st.subheader("Anomaly detection")
model, anomaly_scores, insights = train_anomaly_model(feature_df, contamination=contamination)

if model is not None:
    filtered = feature_df.copy()
    if target_asset != "All":
        filtered = filtered[filtered["asset_id"] == target_asset]
    filtered["anomaly_score"] = anomaly_scores[filtered.index]
    filtered["is_anomaly"] = filtered["anomaly_score"] >= filtered["anomaly_score"].quantile(0.9)

    st.dataframe(
        filtered[["timestamp", "asset_id", "anomaly_score", "is_anomaly"]].sort_values("anomaly_score", ascending=False).head(30),
        use_container_width=True,
    )

    st.caption("Higher anomaly scores indicate increasing deviation from the normal operating envelope.")

    health_summary = summarize_asset_health(filtered)
    st.json(health_summary)

    copilot_summary = generate_copilot_summary(
        df=filtered,
        quality_report=quality_report,
        anomaly_scores=filtered["anomaly_score"].to_dict(),
        insight_list=insights,
    )
    st.subheader("AI operations summary")
    st.markdown(copilot_summary)

else:
    st.info("The model could not be trained with the current feature set. Please ensure telemetry columns are present.")

st.subheader("Raw data preview")
preview = feature_df.head(20).copy()
preview = preview.drop(columns=[col for col in preview.columns if col.startswith("_")], errors="ignore")
st.dataframe(preview, use_container_width=True)


footer = """
This app demonstrates a four-layer industrial reliability stack:
1) Presentation & UI (Streamlit)
2) Data preprocessing & integrity checks
3) Quantitative AI & DSP anomaly scoring
4) Qualitative LLM summarization
"""
st.markdown(f"<div style='margin-top: 2rem; color: #6b7280; font-size: 0.9rem;'>{footer}</div>", unsafe_allow_html=True)
