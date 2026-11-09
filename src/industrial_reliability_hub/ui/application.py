"""Interactive Streamlit workbench for telemetry-led reliability investigations."""
from __future__ import annotations

import json
import hashlib
import html
import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from industrial_reliability_hub.core import plant_core
from industrial_reliability_hub.serialization import json_safe
from industrial_reliability_hub.agent import chat_with_agent, run_reliability_investigation
from industrial_reliability_hub.core.plant_core import (
    ASSET_CONFIG,
    audit_sensor_health,
    build_quality_report,
    clean_historian_telemetry,
    compute_degradation_rul,
    detect_sensor_anomalies,
    get_operational_status,
    is_steady_state,
    prepare_feature_table,
)




def _sample_data() -> pd.DataFrame:
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
            rows.append({"timestamp": timestamp, "asset_id": asset_id, "asset_class": asset_class,
                         "vibration_mm_s": vibration, "temperature_c": temperature,
                         "pressure_bar": pressure, "flow_lpm": flow, "motor_amps": 24.0})
    return pd.DataFrame(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp): return value.isoformat()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if pd.isna(value): return None
    return str(value)


def _shop_package(report: dict[str, Any]) -> str:
    order = report.get("recommended_sap_work_order", {})
    lines = [
        "CMMS WORK PACKAGE — SHOP FLOOR COPY",
        "=" * 42,
        f"Functional location: {report.get('functional_location', 'Not provided')}",
        f"Failure mode: {report.get('primary_failure_mode', 'Unconfirmed')}",
        f"ISO severity zone: {report.get('iso_severity_zone', 'Unknown')}",
        f"Degradation stage: {report.get('degradation_stage', 'Unknown')}",
        f"Order type / priority: {order.get('order_type', 'PM02')} / {order.get('priority', '3-Normal')}",
        f"Object part / damage code: {order.get('object_part', 'PU-BRG')} / {order.get('damage_code', 'Not specified')}",
        f"Assigned craft: {order.get('assigned_craft', 'MECH-MILLWRIGHT')}",
        "",
        "Incident summary:",
        report.get("incident_summary", "No summary supplied."),
        "",
        "Required interventions:",
    ]
    lines.extend(f"  {i}. {action}" for i, action in enumerate(order.get("required_interventions", []), 1))
    lines.extend(["", "Uncertainties / checks before execution:"])
    lines.extend(f"  - {item}" for item in report.get("uncertainties", []))
    lines.extend(["", "Execute only under approved site work-control and isolation procedures."])
    return "\n".join(lines) + "\n"


def _active_asset_data(df: pd.DataFrame, asset_id: str) -> pd.DataFrame:
    data = df[df["asset_id"].astype(str).eq(str(asset_id))].copy()
    return data.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _operational_badge(asset_data: pd.DataFrame, asset_id: str) -> tuple[str, str]:
    state = get_operational_status(asset_data, asset_id)
    if state in {"STANDBY", "OFFLINE", "UNKNOWN"}:
        return state, "#d97706"
    statuses = plant_core.is_asset_running(asset_data, "unknown").reset_index(drop=True)
    transient = plant_core._transient_mask(asset_data.reset_index(drop=True), statuses)
    if len(transient) and bool(transient.iloc[-1]):
        return "TRANSIENT (Startup Mask Active)", "#b45309"
    if is_steady_state(asset_data, asset_id):
        return "RUNNING (Steady State)", "#15803d"
    return "RUNNING (Stabilizing)", "#2563eb"


def _trend_chart(asset_data: pd.DataFrame, asset_id: str) -> go.Figure | None:
    vib_col = next((c for c in ("vibration_rms", "vibration_mm_s") if c in asset_data), None)
    temp_col = next((c for c in ("bearing_temperature", "temperature_c") if c in asset_data), None)
    if vib_col is None and temp_col is None:
        return None
    fig = go.Figure()
    if vib_col:
        fig.add_trace(go.Scatter(x=asset_data.timestamp, y=pd.to_numeric(asset_data[vib_col], errors="coerce"),
                                 name=f"{vib_col} (mm/s RMS)", mode="lines", yaxis="y"))
        # ISO group-2 screening bands. Actual acceptance depends on machine type,
        # support, power and measurement position; show them as decision support.
        fig.add_hrect(y0=2.8, y1=4.5, fillcolor="rgba(245,158,11,0.12)", line_width=0,
                      annotation_text="Zone C — Alert", annotation_position="top left", yref="y")
        fig.add_hrect(y0=4.5, y1=8, fillcolor="rgba(220,38,38,0.12)", line_width=0,
                      annotation_text="Zone D — Danger", annotation_position="top left", yref="y")
        values = pd.to_numeric(asset_data[vib_col], errors="coerce").dropna()
        critical_limit = 4.5
        hours = compute_degradation_rul(asset_data, asset_id, vib_col, critical_limit)
        if hours is not None and hours != -1 and math.isfinite(float(hours)) and float(hours) > 0 and not values.empty:
            latest = values.iloc[-1]
            if latest > 0 and latest < critical_limit:
                latest_time = pd.to_datetime(asset_data.loc[values.index[-1], "timestamp"], errors="coerce")
                if pd.notna(latest_time):
                    # Draw the curve implied by the same exponential crossing
                    # horizon used by the deterministic core RUL estimate.
                    forecast_hours = np.linspace(0.0, float(hours), 40)
                    rate = math.log(critical_limit / float(latest)) / float(hours)
                    projection = float(latest) * np.exp(rate * forecast_hours)
                    projection_times = [latest_time + pd.Timedelta(hours=float(h)) for h in forecast_hours]
                    fig.add_trace(go.Scatter(x=projection_times, y=projection, name="Exponential degradation projection",
                                             mode="lines", line={"dash": "dash", "color": "#7c3aed"}, yaxis="y"))
                    fig.add_trace(go.Scatter(x=[projection_times[-1]], y=[critical_limit], name="Projected Zone D crossing",
                                             mode="markers", marker={"symbol": "x", "size": 11, "color": "#b91c1c"}, yaxis="y"))
    if temp_col:
        fig.add_trace(go.Scatter(x=asset_data.timestamp, y=pd.to_numeric(asset_data[temp_col], errors="coerce"),
                                 name=f"{temp_col} (°C)", mode="lines", yaxis="y2"))
    fig.update_layout(
        title=f"Asset trends and degradation projection — {asset_id}",
        xaxis={"title": "Timestamp"}, yaxis={"title": "Vibration velocity (mm/s RMS)"},
        yaxis2={"title": "Temperature (°C)", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": 1.12}, margin={"t": 65, "r": 75}, hovermode="x unified",
    )
    return fig


def main() -> None:
    """Render one Streamlit run; importing this module does not render the UI."""
    st.set_page_config(page_title="Industrial Reliability Workbench", page_icon="🏭", layout="wide")
    st.title("Industrial Reliability Workbench")
    st.caption("Interactive telemetry diagnostics, agent investigation trace, and SAP PM dispatch preparation")
    st.warning("Decision support only. Verify sensor units, machine-specific limits, and site work-control requirements before acting. ISO bands are screening references, not certification.")

    with st.sidebar:
        st.header("Telemetry and filters")
        input_mode = st.radio("Telemetry source", ["Built-in sample", "Upload CSV or Excel"])
        upload = st.file_uploader("Telemetry file", type=["csv", "xlsx", "xls"]) if input_mode.startswith("Upload") else None
        window = st.number_input("Anomaly window (samples)", min_value=3, max_value=500, value=24)
        threshold_sigma = st.slider("Anomaly threshold (standard deviations)", 1.5, 6.0, 3.0, 0.25)

    try:
        if upload is not None:
            raw_df = clean_historian_telemetry(upload)
        elif input_mode.startswith("Upload"):
            st.info("Upload a telemetry file, or choose the built-in sample.")
            st.stop()
        else:
            raw_df = clean_historian_telemetry(_sample_data())
        if "asset_id" not in raw_df:
            raw_df["asset_id"] = "Uploaded asset"
        feature_df = prepare_feature_table(raw_df)
    except (ValueError, TypeError, ImportError, OSError) as exc:
        st.error(f"Could not prepare telemetry: {exc}")
        st.stop()

    if feature_df.empty or "asset_id" not in feature_df:
        st.warning("No usable telemetry rows or asset identifiers were found.")
        st.stop()

    # Bind investigation and chat state to both data and analysis controls.
    context_key = hashlib.sha256(pd.util.hash_pandas_object(feature_df, index=True).values.tobytes()
                                 + repr((list(feature_df.columns), window, threshold_sigma)).encode()).hexdigest()
    if st.session_state.get("telemetry_context_key") != context_key:
        for key in ("reliability_report", "reliability_report_asset", "messages", "chat_asset_id"):
            st.session_state.pop(key, None)
        st.session_state["telemetry_context_key"] = context_key
    quality_report = build_quality_report(raw_df, feature_df)
    sensor_health = audit_sensor_health(feature_df)
    asset_classes = sorted(feature_df["asset_class"].dropna().astype(str).unique()) if "asset_class" in feature_df else []
    with st.sidebar:
        selected_class = st.selectbox("Asset class", ["All asset classes", *asset_classes])

    filtered = feature_df
    if selected_class != "All asset classes" and "asset_class" in filtered:
        filtered = filtered[filtered.asset_class.astype(str).eq(selected_class)]
    assets = sorted(filtered.asset_id.dropna().astype(str).unique())
    if not assets:
        st.warning("No assets match the selected class filter.")
        st.stop()
    with st.sidebar:
        selected_asset = st.selectbox("Active asset", assets)

    asset_data = _active_asset_data(filtered, selected_asset)
    anomalies_all = detect_sensor_anomalies(filtered, window=int(window), sigma=float(threshold_sigma))
    anomalies = [event for event in anomalies_all if str(event.get("asset_id")) == str(selected_asset)]
    anomalies_df = pd.DataFrame(anomalies)
    operational_badge, badge_color = _operational_badge(asset_data, selected_asset)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets in view", int(filtered.asset_id.nunique()))
    c2.metric("Cleaned records", int(len(filtered)))
    c3.metric("Data quality", f"{quality_report.get('quality_score', 0):.1f}%")
    c4.markdown(
        "<div style='padding-top:.4rem'><div style='font-size:.875rem;color:#666'>Asset operating state</div>"
        f"<span style='display:inline-block;margin-top:.35rem;padding:.3rem .7rem;border-radius:1rem;background:{badge_color};color:white;font-weight:700'>"
        f"{operational_badge}</span></div>", unsafe_allow_html=True,
    )

    with st.expander("Telemetry quality and sensor health", expanded=False):
        q1, q2 = st.columns(2)
        q1.json(quality_report)
        q2.json(sensor_health)

    st.subheader("Asset trends and degradation")
    trend = _trend_chart(asset_data, selected_asset)
    if trend is not None:
        st.plotly_chart(trend, use_container_width=True)
        st.caption("ISO 20816-3 group-2 bands are screening references. A dashed forecast is shown only when the deterministic exponential model returns a qualifying RUL estimate.")
    else:
        st.info("This asset has no recognized vibration or temperature measurements to chart.")

    st.subheader("Anomaly evidence")
    if anomalies:
        st.dataframe(pd.DataFrame(anomalies), use_container_width=True, hide_index=True)
    else:
        st.success("No anomaly events were detected for this asset at the selected threshold.")

    st.subheader("Autonomous investigation and CMMS dispatch")
    if st.button("Run Autonomous Reliability Investigation", type="primary", use_container_width=False):
        with st.status("AI Agent Investigating Machinery Telemetry...", expanded=True) as status:
            report = run_reliability_investigation(
                telemetry_df=asset_data,
                anomalies=anomalies,
                asset_id=selected_asset,
                callback=lambda step, det: status.write(f"⚙️ **{step}**: {det}"),
            )
            st.session_state["reliability_report"] = report
            st.session_state["reliability_report_asset"] = selected_asset
            status.update(label="Investigation Complete", state="complete", expanded=False)

    report = st.session_state.get("reliability_report")
    if report and st.session_state.get("reliability_report_asset") == selected_asset:
        order = report.get("recommended_sap_work_order", {})
        st.markdown("### SAP PM work-order recommendation — review before dispatch")
        loc_col, priority_col, craft_col, zone_col = st.columns(4)
        loc_col.markdown(f"**Functional location**\n\n{report.get('functional_location', 'Not provided')}")
        priority = order.get("priority", "3-Normal")
        priority_color = {"1-Emergency": "#b91c1c", "2-Urgent": "#c2410c", "3-Normal": "#15803d"}.get(priority, "#6b7280")
        priority_col.markdown(f"**Priority**\n\n<span style='color:white;background:{priority_color};padding:.35rem .65rem;border-radius:.5rem;font-weight:700'>{html.escape(str(priority))}</span>", unsafe_allow_html=True)
        craft_col.markdown(f"**Assigned craft**\n\n{order.get('assigned_craft', 'Not specified')}")
        zone_col.markdown(f"**ISO zone / stage**\n\n{report.get('iso_severity_zone', 'Unknown')} / {report.get('degradation_stage', 'Unknown')}")
        st.markdown(f"**Incident summary:** {report.get('incident_summary', 'Not provided')}")
        st.markdown(f"**Primary failure mode:** {report.get('primary_failure_mode', 'Not confirmed')}")
        actions = order.get("required_interventions", [])
        with st.container(border=True):
            st.markdown(f"**Shop actions · {order.get('order_type', 'PM02')} · {order.get('object_part', 'PU-BRG')} · {order.get('damage_code', 'Not specified')}**")
            if actions:
                for action in actions: st.markdown(f"- {action}")
            else:
                st.info("No shop actions were provided; review uncertainties before dispatch.")
        with st.expander("Investigation evidence and uncertainties"):
            st.json(report.get("investigation_evidence", {}))
            st.markdown("**Uncertainties**")
            for item in report.get("uncertainties", []): st.markdown(f"- {item}")
        d1, d2 = st.columns(2)
        report_json = json.dumps(json_safe(report), ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
        d1.download_button("Export SAP PM Notification (.json)", report_json,
                           file_name=f"{selected_asset}_sap_pm_notification.json", mime="application/json", use_container_width=True)
        d2.download_button("Download CMMS Work Package (.txt)", _shop_package(report),
                           file_name=f"{selected_asset}_cmms_work_package.txt", mime="text/plain", use_container_width=True)
    else:
        st.info("Run an autonomous investigation to collect deterministic evidence and prepare a CMMS work package.")

    # Technician chat uses only a compact summary and evidence from the selected asset.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.session_state.get("chat_asset_id") != selected_asset:
        st.session_state.messages = []
        st.session_state.chat_asset_id = selected_asset

    report_context = report if report and st.session_state.get("reliability_report_asset") == selected_asset else {}
    chat_context = {
        "asset_id": selected_asset,
        "operational_state": operational_badge,
        "telemetry_quality": build_quality_report(asset_data),
        "sensor_health": audit_sensor_health(asset_data),
        "recent_anomalies": anomalies[:30],
        "investigation_evidence": report_context.get("investigation_evidence", {}),
        "work_order": {k: v for k, v in report_context.items() if k != "investigation_evidence"},
    }
    with st.sidebar:
        st.divider()
        st.subheader("Technician chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        question = st.chat_input("Ask about this asset's evidence", key="technician_chat_input")
        if question:
            history = list(st.session_state.messages)
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Checking the diagnostic facts..."):
                    try:
                        answer = chat_with_agent(question, chat_context, history)
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as exc:
                        st.error(f"Local agent is unavailable: {exc}")
