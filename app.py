"""Streamlit presentation and routing layer for the reliability hub."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from copilot import generate_engineering_report
from models import analyze_reliability
from pipelines import ASSET_CONFIG, build_quality_report, load_dataset, prepare_feature_table

st.set_page_config(page_title="Industrial Reliability Hub", page_icon="🏭", layout="wide")


def _sample_data() -> pd.DataFrame:
    """Build deterministic example telemetry for a quick first run."""
    rows: list[dict[str, Any]] = []
    for asset_id, asset_class in [("Pump-A", "pump"), ("Pump-B", "pump"), ("Motor-A", "motor")]:
        for index in range(480):
            timestamp = pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5 * index)
            vibration = 2.0 + 0.15 * (index % 16) + (0.25 if asset_id == "Pump-B" else 0)
            temperature = 55.0 + 0.08 * (index % 60)
            pressure = 4.0 + 0.03 * (index % 30)
            flow = 100.0 + 0.4 * (index % 40)
            if asset_id == "Pump-B" and index >= 400:
                vibration += (index - 399) * 0.08
                temperature += (index - 399) * 0.25
            rows.append({
                "timestamp": timestamp,
                "asset_id": asset_id,
                "asset_class": asset_class,
                "vibration_mm_s": vibration,
                "temperature_c": temperature,
                "pressure_bar": pressure,
                "flow_lpm": flow,
            })
    return pd.DataFrame(rows)


st.title("Industrial Reliability Hub")
st.caption("Telemetry integrity, anomaly detection, and maintenance decision support")

with st.sidebar:
    st.header("Input and analysis")
    input_mode = st.radio("Telemetry source", ["Built-in sample", "Upload CSV or Excel"])
    upload = st.file_uploader("Telemetry file", type=["csv", "xlsx", "xls"]) if input_mode.startswith("Upload") else None
    window = st.number_input("Moving-statistics window (samples)", min_value=3, max_value=500, value=24)
    threshold_sigma = st.slider("Anomaly threshold (standard deviations)", 1.5, 6.0, 3.0, 0.25)
    selected_asset = st.text_input("Filter asset ID (optional)").strip()

try:
    if upload is not None:
        raw_df = load_dataset(upload)
    elif input_mode.startswith("Upload"):
        st.info("Upload a telemetry file, or choose the built-in sample.")
        st.stop()
    else:
        raw_df = load_dataset(_sample_data())
    feature_df = prepare_feature_table(raw_df)
except (ValueError, TypeError) as exc:
    st.error(f"Could not prepare telemetry: {exc}")
    st.stop()

if feature_df.empty:
    st.warning("No usable telemetry rows were found.")
    st.stop()

quality_report = build_quality_report(raw_df, feature_df)
analysis_df = feature_df
if selected_asset:
    analysis_df = feature_df[feature_df["asset_id"].astype(str) == selected_asset]
    if analysis_df.empty:
        st.warning(f"No telemetry found for asset {selected_asset!r}.")
        st.stop()

analysis = analyze_reliability(analysis_df, window=int(window), threshold_sigma=float(threshold_sigma))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Assets", int(analysis_df["asset_id"].nunique()))
m2.metric("Cleaned records", len(analysis_df))
m3.metric("Data quality", f"{quality_report['quality_score']:.1f}%")
m4.metric("Detected events", len(analysis["anomalies"]))

with st.expander("Data integrity report"):
    st.json(quality_report)

st.subheader("Telemetry trends")
numeric_metrics = [column for column in analysis_df.select_dtypes(include="number").columns if column not in {"_"}]
if numeric_metrics:
    metric = st.selectbox("Sensor measurement", numeric_metrics)
    chart_data = analysis_df[["timestamp", "asset_id", metric]].copy()
    chart_data["series"] = chart_data["asset_id"].astype(str)
    st.line_chart(chart_data.pivot_table(index="timestamp", columns="series", values=metric, aggfunc="mean"))
else:
    st.info("No numeric sensor measurements are available to chart.")

st.subheader("Anomalies and remaining useful life")
if analysis["anomalies"]:
    st.dataframe(pd.DataFrame(analysis["anomalies"]), use_container_width=True)
else:
    st.success("No moving-standard-deviation anomalies detected at the configured threshold.")
if analysis["rul_estimates"]:
    st.markdown("**Estimated operating time to configured thresholds**")
    st.dataframe(pd.DataFrame(analysis["rul_estimates"]), use_container_width=True)
    st.caption("RUL is a trend-based estimate, not a guarantee or substitute for engineering judgment.")

st.subheader("Engineering copilot report")
with st.spinner("Preparing structured report..."):
    report = generate_engineering_report(
        quality_report=quality_report,
        anomalies=analysis["anomalies"],
        rul_estimates=analysis["rul_estimates"],
        asset_config=ASSET_CONFIG,
    )
st.json(report)

with st.expander("Prepared telemetry preview"):
    st.dataframe(feature_df.head(100), use_container_width=True)
