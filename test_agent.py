"""Standalone local verification for the Ollama reliability-agent loop.

Run from the repository root with Ollama serving ``llama3.1:8b``::

    python test_agent.py

This intentionally calls the real local model. It does not import or launch Streamlit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import agent
import plant_core

ASSET_ID = "PUMP-101A"
ANSI = {"reset": "\033[0m", "bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m"}


def _color(text: str, color: str) -> str:
    return f"{ANSI[color]}{text}{ANSI['reset']}"


def _synthetic_telemetry() -> pd.DataFrame:
    """Create a deterministic running pump with a hydraulic/vibration event."""
    count = 100
    index = np.arange(count)
    pressure = np.full(count, 4.2, dtype=float)
    pressure[65:] = np.linspace(4.2, 2.4, count - 65)
    vibration = np.full(count, 2.0, dtype=float)
    vibration[65:] = 2.0 + 4.0 * np.expm1(np.linspace(0.0, 1.8, count - 65)) / np.expm1(1.8)
    # Tiny deterministic variation avoids a completely flat baseline while
    # preserving a clear event and stable electrical loading.
    vibration += 0.015 * np.sin(index / 3.0)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=count, freq="5min"),
        "asset_id": ASSET_ID,
        "asset_class": "pump",
        "suction_pressure_bar": pressure,
        "discharge_pressure_bar": np.full(count, 6.0),
        "vibration_rms": vibration,
        "bearing_temperature": 62.0 + 0.015 * index,
        "motor_amps": np.full(count, 28.0),
        "flow_lpm": np.full(count, 115.0),
    })


def _load_telemetry() -> pd.DataFrame:
    path = Path(__file__).with_name("pump_telemetry.csv")
    if not path.exists():
        print("pump_telemetry.csv not found; using deterministic 100-row pump scenario.")
        return _synthetic_telemetry()
    try:
        data = plant_core.clean_historian_telemetry(path)
        # The repository example is useful for ingestion checks but may not have
        # pressure/current or an asset ID. Use the specified scenario in that case
        # so the hydraulic and operating-state tools have meaningful inputs.
        expected = {"timestamp", "vibration_rms", "suction_pressure_bar", "discharge_pressure_bar", "motor_amps"}
        if not expected.issubset(data.columns):
            print("pump_telemetry.csv lacks the required diagnostic tags; using synthetic pump scenario.")
            return _synthetic_telemetry()
        if "asset_id" not in data:
            data["asset_id"] = ASSET_ID
        elif ASSET_ID not in data["asset_id"].astype(str).values:
            data["asset_id"] = ASSET_ID
        return data
    except Exception as exc:
        print(f"Could not use pump_telemetry.csv ({exc}); using synthetic pump scenario.")
        return _synthetic_telemetry()


def main() -> None:
    df = _load_telemetry()
    if "asset_id" not in df:
        df["asset_id"] = ASSET_ID
    if not df["asset_id"].astype(str).eq(ASSET_ID).any():
        df["asset_id"] = ASSET_ID

    health = plant_core.audit_sensor_health(df)
    anomalies = plant_core.detect_sensor_anomalies(df, window=24, sigma=3.0)
    print(f"Pre-flight: {len(df)} telemetry rows; sensor health {'OK' if health.get('healthy') else 'flags present'}; {len(anomalies)} anomaly event(s).")

    tool_calls: set[str] = set()

    def cli_callback(step_name: str, detail: Any):
        if step_name not in {"fact_finding", "offline_fallback"}:
            tool_calls.add(step_name)
        rendered = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False, default=str)
        print(f"  [AGENT TOOL CALL] {step_name} -> {rendered}")

    start_time = time.perf_counter()
    report = agent.run_reliability_investigation(
        df, anomalies, asset_id=ASSET_ID, callback=cli_callback,
    )
    elapsed = time.perf_counter() - start_time

    assert isinstance(report, dict), "Agent result must be a dictionary."
    for key in ("functional_location", "primary_failure_mode", "recommended_sap_work_order"):
        assert key in report, f"Agent report is missing required key: {key}"
    order = report["recommended_sap_work_order"]
    assert isinstance(order, dict), "recommended_sap_work_order must be a dictionary."
    for key in ("order_type", "priority", "assigned_craft"):
        assert key in order, f"SAP work order is missing required key: {key}"
    assert len(tool_calls) >= 2, (
        "Expected at least two distinct diagnostic tools to be executed by the model; "
        f"observed {sorted(tool_calls)}. Confirm Ollama/model tool-calling support."
    )

    status = plant_core.get_operational_status(df, ASSET_ID)
    chronological = sorted(
        (event for event in anomalies if str(event.get("asset_id")) == ASSET_ID),
        key=lambda event: str(event.get("timestamp", "")),
    )
    cascade_tag = chronological[0].get("sensor", "No anomaly cascade detected") if chronological else "No anomaly cascade detected"

    print("\n" + _color("=" * 68, "cyan"))
    print(_color("LOCAL AUTONOMOUS RELIABILITY AGENT VERIFICATION", "bold"))
    print(_color("=" * 68, "cyan"))
    print(f"Asset Operating Status       : {_color(status, 'green' if status == 'RUNNING' else 'yellow')}")
    print(f"Initiating Sensor Cascade Tag: {_color(str(cascade_tag), 'cyan')}")
    print(f"Primary Failure Mode         : {report['primary_failure_mode']}")
    print(f"SAP PM Work Order            : {order['order_type']} | {order['priority']}")
    print(f"Assigned Craft               : {order['assigned_craft']}")
    print(f"Distinct diagnostic tools    : {', '.join(sorted(tool_calls))}")
    print(f"Wall-Clock Runtime            : {elapsed:.2f} seconds")
    if elapsed > 25:
        print(_color("Warning: runtime exceeded 25 seconds on this hardware.", "yellow"))
    print(_color("Verification assertions passed.", "green"))


if __name__ == "__main__":
    main()
