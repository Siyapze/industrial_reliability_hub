"""Streamlit presentation and routing layer for the reliability hub."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from copilot import converse_with_asset, generate_engineering_report
from models import analyze_reliability, is_asset_running, trace_event_cascade
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


def _markdown_work_order(report: dict[str, Any]) -> str:
    work_order = report.get("recommended_sap_work_order", {})
    inspections = work_order.get("required_inspections", [])
    cascade = report.get("chronological_cascade", [])
    uncertainties = report.get("uncertainties", [])
    lines = [
        "# SAP PM Work Order Recommendation",
        "",
        f"- **Functional location:** {report.get('functional_location', 'Not provided')}",
        f"- **Incident summary:** {report.get('incident_summary', 'Not provided')}",
        f"- **Primary failure mode:** {report.get('primary_failure_mode', 'Not provided')}",
        "",
        "## Recommended work order",
        f"- **Maintenance activity type:** {work_order.get('maintenance_activity_type', 'Not provided')}",
        f"- **Priority:** {work_order.get('priority', 'Not provided')}",
        f"- **Required craft:** {work_order.get('required_craft', 'Not provided')}",
        "",
        "## Required inspections",
    ]
    lines.extend(f"- {item}" for item in inspections)
    lines.extend(["", "## Chronological event cascade"])
    lines.extend(f"- {item}" for item in cascade)
    lines.extend(["", "## Uncertainties"])
    lines.extend(f"- {item}" for item in uncertainties)
    return "\n".join(lines) + "\n"


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
asset_classes = sorted(feature_df["asset_class"].dropna().astype(str).unique()) if "asset_class" in feature_df else []
with st.sidebar:
    selected_class = st.selectbox("Asset class", ["All asset classes", *asset_classes])

analysis_df = feature_df
if selected_class != "All asset classes" and "asset_class" in analysis_df:
    analysis_df = analysis_df[analysis_df["asset_class"].astype(str) == selected_class]
if selected_asset:
    analysis_df = analysis_df[analysis_df["asset_id"].astype(str) == selected_asset]
if analysis_df.empty:
    st.warning("No telemetry matches the selected asset class and asset ID filters.")
    st.stop()

analysis = analyze_reliability(analysis_df, window=int(window), threshold_sigma=float(threshold_sigma))
info_events = [event for event in analysis["anomalies"] if event.get("severity") == "INFO"]
alarm_events = [event for event in analysis["anomalies"] if event.get("severity") != "INFO"]
critical_events = [event for event in alarm_events if event.get("severity") == "CRITICAL"]

# Determine the current operating badge from each asset's latest telemetry row.
latest_rows = analysis_df.sort_values("timestamp").groupby("asset_id", sort=False).tail(1)
latest_statuses: list[str] = []
for _, row in latest_rows.iterrows():
    asset_data = analysis_df[analysis_df["asset_id"] == row["asset_id"]]
    asset_class = str(row.get("asset_class", "unknown"))
    latest_statuses.append(str(is_asset_running(asset_data, asset_class).iloc[-1]))
if critical_events:
    operational_badge, badge_color = "Critical Alarm", "#b91c1c"
elif any(status in {"STANDBY", "OFFLINE"} for status in latest_statuses):
    operational_badge, badge_color = "Standby", "#d97706"
else:
    operational_badge, badge_color = "Operational", "#15803d"

# Compact context supplied to the interactive copilot. It contains summaries,
# not the entire telemetry table, so conversation requests remain bounded.
data_integrity_report = {
    **quality_report,
    "active_asset_filter": selected_asset or "all assets",
    "active_asset_class_filter": selected_class,
    "operational_state": operational_badge,
    "active_alarm_count": len(alarm_events),
    "operational_info_count": len(info_events),
    "recent_anomalies": analysis["anomalies"][:20],
    "rul_estimates": analysis["rul_estimates"][:20],
}

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Assets", int(analysis_df["asset_id"].nunique()))
m2.metric("Cleaned records", len(analysis_df))
m3.metric("Data quality", f"{quality_report['quality_score']:.1f}%")
m4.metric("Active alarms", len(alarm_events))
m5.markdown(
    "<div style='padding-top:0.4rem'><div style='font-size:0.875rem;color:#666'>Operational state</div>"
    f"<span style='display:inline-block;margin-top:0.35rem;padding:0.25rem 0.65rem;border-radius:1rem;"
    f"background:{badge_color};color:white;font-weight:600'>{operational_badge}</span></div>",
    unsafe_allow_html=True,
)
if info_events:
    st.info(f"{len(info_events)} informational event(s): possible asset idle or powered-down state. These are not treated as alarms.")

with st.expander("Data integrity report"):
    st.json(quality_report)

st.subheader("Telemetry trends")
numeric_metrics = [column for column in analysis_df.select_dtypes(include="number").columns if not column.startswith("_")]
if numeric_metrics:
    metric = st.selectbox("Sensor measurement", numeric_metrics)
    chart_data = analysis_df[["timestamp", "asset_id", metric]].copy()
    chart_data["series"] = chart_data["asset_id"].astype(str)
    st.line_chart(chart_data.pivot_table(index="timestamp", columns="series", values=metric, aggfunc="mean"))

    st.markdown("**Multi-variable overlay**")
    overlay_cols = st.columns(3)
    primary_tag = overlay_cols[0].selectbox("Primary Y-axis tag", numeric_metrics, index=0, key="overlay_primary")
    secondary_options = [column for column in numeric_metrics if column != primary_tag]
    if secondary_options:
        preferred = next((name for name in ("basin_water_level", "makeup_valve_pct") if name in secondary_options), secondary_options[0])
        secondary_tag = overlay_cols[1].selectbox(
            "Secondary Y-axis tag", secondary_options,
            index=secondary_options.index(preferred), key="overlay_secondary",
        )
        overlay_assets = sorted(analysis_df["asset_id"].astype(str).unique())
        if len(overlay_assets) > 1:
            overlay_asset = overlay_cols[2].selectbox("Overlay asset", overlay_assets, key="overlay_asset")
            overlay_data = analysis_df[analysis_df["asset_id"].astype(str) == overlay_asset].sort_values("timestamp")
        else:
            overlay_data = analysis_df.sort_values("timestamp")
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=overlay_data["timestamp"], y=overlay_data[primary_tag], name=primary_tag,
            mode="lines", yaxis="y1",
        ))
        figure.add_trace(go.Scatter(
            x=overlay_data["timestamp"], y=overlay_data[secondary_tag], name=secondary_tag,
            mode="lines", yaxis="y2",
        ))
        figure.update_layout(
            xaxis={"title": "Timestamp"},
            yaxis={"title": primary_tag},
            yaxis2={"title": secondary_tag, "overlaying": "y", "side": "right"},
            legend={"orientation": "h", "y": 1.12},
            margin={"t": 35, "r": 70},
        )
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.caption("At least two numeric sensor tags are required for an overlay.")
else:
    st.info("No numeric sensor measurements are available to chart.")

st.subheader("Anomalies and remaining useful life")
if analysis["anomalies"]:
    st.dataframe(pd.DataFrame(analysis["anomalies"]), use_container_width=True)
else:
    st.success("No directional anomalies detected at the configured threshold.")
if analysis["rul_estimates"]:
    st.markdown("**Estimated operating time to configured thresholds**")
    st.dataframe(pd.DataFrame(analysis["rul_estimates"]), use_container_width=True)
    st.caption("RUL is a trend-based estimate, not a guarantee or substitute for engineering judgment.")

# Event clusters and chronological evidence appear immediately before the report.
cascade_clusters = trace_event_cascade(pd.DataFrame(analysis["anomalies"])) if analysis["anomalies"] else []
if cascade_clusters:
    st.warning(f"Event Cascade: {len(cascade_clusters)} incident cluster(s) detected.")
    with st.expander("View Event Cascade timeline", expanded=False):
        for cluster in cascade_clusters:
            st.markdown(
                f"**Incident Cluster — {cluster.get('asset_id') or 'Unknown asset'}**  \n"
                f"Initiating sensor: **{cluster['initiating_sensor']}**  \n"
                f"{cluster['transition_path']}"
            )
else:
    with st.expander("Event Cascade timeline", expanded=False):
        st.caption("No timestamped anomaly events are available to cluster.")

st.subheader("Engineering Copilot report")
with st.spinner("Preparing structured report..."):
    report = generate_engineering_report(
        quality_report=quality_report,
        anomalies=analysis["anomalies"],
        rul_estimates=analysis["rul_estimates"],
        asset_config=ASSET_CONFIG,
    )
report_col, download_col_json, download_col_md = st.columns([6, 1, 1])
with report_col:
    st.json(report)
with download_col_json:
    st.download_button(
        "Download JSON",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="sap_pm_work_order_recommendation.json",
        mime="application/json",
        use_container_width=True,
    )
with download_col_md:
    st.download_button(
        "Download MD",
        data=_markdown_work_order(report),
        file_name="sap_pm_work_order_recommendation.md",
        mime="text/markdown",
        use_container_width=True,
    )

# Sidebar conversational interface. Chat history persists across Streamlit reruns.
if "asset_copilot_messages" not in st.session_state:
    st.session_state.asset_copilot_messages = []

with st.sidebar:
    with st.expander("AI Copilot Interactive Chat Loop", expanded=False):
        st.caption("Ask questions about the active telemetry, data quality, anomalies, and RUL estimates.")
        for message in st.session_state.asset_copilot_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        chat_question = st.chat_input("Ask the Plant Reliability Assistant", key="asset_copilot_input")
        if chat_question:
            prior_history = list(st.session_state.asset_copilot_messages)
            st.session_state.asset_copilot_messages.append({"role": "user", "content": chat_question})
            with st.chat_message("user"):
                st.markdown(chat_question)
            with st.chat_message("assistant"):
                with st.spinner("Reviewing the active reliability data..."):
                    try:
                        answer = converse_with_asset(
                            user_question=chat_question,
                            data_summary_json=data_integrity_report,
                            chat_history=prior_history,
                        )
                        st.markdown(answer)
                        st.session_state.asset_copilot_messages.append({"role": "assistant", "content": answer})
                    except Exception as exc:
                        st.error(f"Copilot is unavailable: {exc}")
