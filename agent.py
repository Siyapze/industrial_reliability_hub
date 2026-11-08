"""Local Ollama-powered autonomous reliability investigation agent.

The language model orchestrates deterministic diagnostics from ``plant_core``;
raw telemetry is never sent to the model. The final work-order recommendation is
strictly grounded in the collected tool evidence.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import plant_core

load_dotenv()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama3.1:8b")
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

INVESTIGATION_SYSTEM_PROMPT = (
    "You are a Senior Rotating Equipment Reliability Specialist. Call tools to systematically check "
    "operating state, rule out startup transients, verify hydraulic stability, and trace "
    "thermal/vibration cascades. Use tools before drawing conclusions. Treat their outputs as facts, "
    "do not invent measurements, and finish tool use when sufficient evidence is gathered."
)

SAP_WORK_ORDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "functional_location": {"type": "string"},
        "incident_summary": {"type": "string", "description": "Exactly two sentences describing the observed mechanical/process problem and evidence."},
        "primary_failure_mode": {"type": "string"},
        "iso_severity_zone": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "degradation_stage": {"type": "string", "enum": ["Early P-F", "Late P-F", "Critical Impending Failure"]},
        "recommended_sap_work_order": {
            "type": "object",
            "properties": {
                "order_type": {"type": "string", "enum": ["PM01", "PM02"]},
                "priority": {"type": "string", "enum": ["1-Emergency", "2-Urgent", "3-Normal"]},
                "object_part": {"type": "string", "enum": ["PU-BRG", "PU-MCH-SL", "PU-IMP", "CP-FLX"]},
                "damage_code": {"type": "string"},
                "assigned_craft": {"type": "string", "enum": ["MECH-MILLWRIGHT", "ELEC-TECH", "OPS-OPERATOR"]},
                "required_interventions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["order_type", "priority", "object_part", "damage_code", "assigned_craft", "required_interventions"],
            "additionalProperties": False,
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["functional_location", "incident_summary", "primary_failure_mode", "iso_severity_zone", "degradation_stage", "recommended_sap_work_order", "uncertainties"],
    "additionalProperties": False,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)): return value.isoformat()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if pd.isna(value): return None
    if isinstance(value, np.ndarray): return value.tolist()
    return str(value)


def _clean_model_text(text: str | None) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def _callback(callback: Callable[[str, Any], None] | None, step_name: str, detail: Any) -> None:
    if callback is not None:
        try: callback(step_name, detail)
        except Exception: pass


def _asset_rows(df: pd.DataFrame, asset_id: str) -> pd.DataFrame:
    if "asset_id" not in df: return df.copy()
    return df.loc[df["asset_id"].astype(str).eq(str(asset_id))].copy()


def _baseline_load(df: pd.DataFrame, asset_id: str) -> dict[str, Any]:
    rows = _asset_rows(df, asset_id)
    candidates = ("motor_amps", "motor_current_a", "current_a", "current", "amps", "flow_lpm", "circulation_flow_gpm", "rpm", "speed_rpm")
    result: dict[str, Any] = {}
    if rows.empty: return result
    status = plant_core.is_asset_running(rows, "unknown")
    running = rows.loc[status.eq("RUNNING")] if status.eq("RUNNING").any() else rows
    for col in candidates:
        if col in rows:
            vals = pd.to_numeric(running[col], errors="coerce").dropna()
            if not vals.empty: result[col] = {"baseline_median": float(vals.median()), "latest": float(pd.to_numeric(rows[col], errors="coerce").dropna().iloc[-1]) if pd.to_numeric(rows[col], errors="coerce").notna().any() else None}
    return result


def _tool_definitions() -> list[dict[str, Any]]:
    def tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}
    return [
        tool("check_asset_operating_state", "Check the latest asset operating state, whether the latest observation is within the 15-minute startup transient, and baseline load.", {"asset_id": {"type": "string"}}, ["asset_id"]),
        tool("diagnose_hydraulic_vs_mechanical", "Compare vibration, suction/discharge pressure, head, current and temperature evidence for hydraulic versus mechanical signatures.", {"asset_id": {"type": "string"}}, ["asset_id"]),
        tool("trace_thermal_vibration_sequence", "Determine whether temperature or vibration leads the other and return timing evidence.", {"asset_id": {"type": "string"}}, ["asset_id"]),
        tool("evaluate_iso_severity", "Classify vibration velocity RMS using the plant core ISO 20816-3 screening limits.", {"vibration_rms": {"type": "number"}, "machine_group": {"type": "integer", "default": 2}}, ["vibration_rms"]),
        tool("calculate_exponential_rul", "Estimate operating hours to a sensor limit with an exponential trend fit; -1 means no qualifying acceleration.", {"asset_id": {"type": "string"}, "sensor": {"type": "string"}, "limit": {"type": "number"}}, ["asset_id", "sensor", "limit"]),
    ]


def _offline_fallback_report(asset_id: str, anomalies: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a deterministic PM02 field-inspection recommendation without Ollama."""
    relevant = [a for a in anomalies if not a.get("asset_id") or str(a.get("asset_id")) == str(asset_id)]
    ordered = sorted(relevant, key=lambda a: str(a.get("timestamp", "")))
    anomaly_text = "; ".join(f"{a.get('sensor', 'sensor')} deviation at {a.get('timestamp', 'unknown time')}" for a in ordered[:3])
    sentence1 = f"Telemetry review for {asset_id} identified {anomaly_text or 'no supplied anomaly events'} in the available anomaly records."
    sentence2 = "The suspected condition is unconfirmed and requires an on-equipment inspection before corrective work is authorized."
    return {
        "functional_location": str(asset_id),
        "incident_summary": f"{sentence1} {sentence2}",
        "primary_failure_mode": "Unconfirmed rotating-equipment condition; inspection required",
        "iso_severity_zone": "A",
        "degradation_stage": "Early P-F",
        "recommended_sap_work_order": {
            "order_type": "PM02", "priority": "3-Normal", "object_part": "PU-BRG",
            "damage_code": "UNCONFIRMED-CONDITION", "assigned_craft": "MECH-MILLWRIGHT",
            "required_interventions": [
                "Verify vibration and bearing-temperature readings with calibrated instruments.",
                "Inspect bearing lubrication, coupling alignment, and mounting condition under site isolation procedures.",
                "Record findings and confirm equipment identity and SAP functional location before follow-up work.",
            ],
        },
        "uncertainties": ["Ollama analysis was unavailable; this deterministic report does not establish a failure diagnosis.", "Confirm vibration severity from validated RMS units and machine-specific ISO support classification.", "Available anomaly records may omit process conditions, maintenance history, and unmonitored failure modes."],
    }


def _run_local_tool(name: str, args: dict[str, Any], telemetry_df: pd.DataFrame, anomalies: list[dict[str, Any]], asset_id: str) -> dict[str, Any]:
    target_asset = str(args.get("asset_id", asset_id))
    if name == "check_asset_operating_state":
        rows = _asset_rows(telemetry_df, target_asset)
        state = plant_core.get_operational_status(telemetry_df, target_asset)
        timestamps = pd.to_datetime(rows["timestamp"], errors="coerce", utc=True).dropna() if "timestamp" in rows else pd.Series(dtype="datetime64[ns, UTC]")
        transient = False
        if len(timestamps) and state == "RUNNING":
            run_rows = rows.copy()
            status = plant_core.is_asset_running(run_rows, "unknown").reset_index(drop=True)
            ordered_times = pd.to_datetime(run_rows["timestamp"], errors="coerce", utc=True).reset_index(drop=True)
            starts = status.eq("RUNNING") & ~status.shift(1, fill_value=False)
            start_times = ordered_times[starts]
            transient = bool(len(start_times) and timestamps.max() < start_times.iloc[-1] + pd.Timedelta(minutes=15))
        return {"asset_id": target_asset, "operating_state": state, "transient": transient, "steady_state": bool(plant_core.is_steady_state(telemetry_df, target_asset)), "baseline_load": _baseline_load(telemetry_df, target_asset)}
    if name == "diagnose_hydraulic_vs_mechanical": return plant_core.diagnose_hydraulic_vs_mechanical(telemetry_df, target_asset)
    if name == "trace_thermal_vibration_sequence": return plant_core.trace_thermal_vibration_sequence(telemetry_df, target_asset)
    if name == "evaluate_iso_severity": return plant_core.evaluate_iso_vibration_zone(float(args["vibration_rms"]), int(args.get("machine_group", 2)))
    if name == "calculate_exponential_rul":
        hours = plant_core.compute_degradation_rul(telemetry_df, target_asset, str(args["sensor"]), float(args["limit"]))
        return {"asset_id": target_asset, "sensor": str(args["sensor"]), "critical_limit": float(args["limit"]), "estimated_operating_hours": float(hours) if hours != -1 else -1, "projection_available": bool(hours != -1)}
    raise ValueError(f"Unknown diagnostic tool: {name}")


def run_reliability_investigation(
    telemetry_df: pd.DataFrame,
    anomalies: list[dict[str, Any]],
    asset_id: str,
    callback: Callable[[str, Any], None] | None = None,
) -> dict[str, Any]:
    """Run tool-driven diagnostics then produce a constrained SAP PM work order."""
    facts: dict[str, Any] = {"asset_id": str(asset_id), "anomalies": [a for a in anomalies if not a.get("asset_id") or str(a.get("asset_id")) == str(asset_id)]}
    tools = _tool_definitions()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": INVESTIGATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Investigate asset {asset_id}. Use available diagnostic tools systematically; do not generate the work order in this phase."},
    ]
    functions = {item["function"]["name"] for item in tools}
    try:
        for iteration in range(4):
            response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, tools=tools, tool_choice="auto", temperature=0.1)
            message = response.choices[0].message
            assistant_message: dict[str, Any] = {"role": "assistant", "content": _clean_model_text(message.content)}
            calls = getattr(message, "tool_calls", None) or []
            if not calls:
                messages.append(assistant_message)
                break
            assistant_message["tool_calls"] = [call.model_dump(exclude_none=True) for call in calls]
            messages.append(assistant_message)
            for call in calls:
                name = call.function.name
                if name not in functions: continue
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    result = _run_local_tool(name, arguments, telemetry_df, anomalies, str(asset_id))
                except Exception as exc:
                    result = {"error": f"Diagnostic tool failed: {type(exc).__name__}: {exc}"}
                facts[name] = result
                _callback(callback, name, result)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False, default=_json_default, allow_nan=False)})
            _callback(callback, "fact_finding", {"iteration": iteration + 1, "tools_executed": len(calls)})

        # Phase 2 is intentionally a fresh, single-turn context. It receives only
        # deterministic investigation evidence and the output contract, not chat history.
        final_prompt = {
            "investigation_evidence": facts,
            "required_cmms_schema": SAP_WORK_ORDER_SCHEMA,
            "instructions": "Return only a JSON object matching the schema. The incident_summary must be exactly two sentences. Use only supplied evidence; identify uncertainty rather than inventing measurements, diagnoses, SAP master data, or completed work. If evidence is insufficient, select a PM02 inspection order with normal priority.",
        }
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": json.dumps(final_prompt, ensure_ascii=False, default=_json_default, allow_nan=False)}],
            response_format={"type": "json_object"}, temperature=0.1,
        )
        content = _clean_model_text(response.choices[0].message.content)
        report = json.loads(content)
        _validate_sap_report(report)
        report["investigation_evidence"] = facts
        return report
    except Exception as exc:
        fallback = _offline_fallback_report(str(asset_id), anomalies)
        fallback["investigation_evidence"] = facts
        fallback["uncertainties"].append(f"Local model or report generation failed: {type(exc).__name__}.")
        _callback(callback, "offline_fallback", {"reason": type(exc).__name__})
        return fallback


def _validate_sap_report(report: Any) -> None:
    if not isinstance(report, dict): raise ValueError("Report must be a JSON object")
    required = set(SAP_WORK_ORDER_SCHEMA["required"])
    if not required.issubset(report): raise ValueError(f"Report is missing required fields: {sorted(required-set(report))}")
    order = report.get("recommended_sap_work_order")
    if not isinstance(order, dict): raise ValueError("recommended_sap_work_order must be an object")
    if order.get("order_type") not in {"PM01", "PM02"}: raise ValueError("Invalid SAP order_type")
    if order.get("priority") not in {"1-Emergency", "2-Urgent", "3-Normal"}: raise ValueError("Invalid SAP priority")
    if order.get("object_part") not in {"PU-BRG", "PU-MCH-SL", "PU-IMP", "CP-FLX"}: raise ValueError("Invalid object_part")
    if order.get("assigned_craft") not in {"MECH-MILLWRIGHT", "ELEC-TECH", "OPS-OPERATOR"}: raise ValueError("Invalid assigned_craft")
    if report.get("iso_severity_zone") not in {"A", "B", "C", "D"}: raise ValueError("Invalid ISO severity zone")
    if report.get("degradation_stage") not in {"Early P-F", "Late P-F", "Critical Impending Failure"}: raise ValueError("Invalid degradation_stage")
    if not isinstance(order.get("required_interventions"), list) or not isinstance(report.get("uncertainties"), list): raise ValueError("Interventions and uncertainties must be arrays")


def chat_with_agent(user_question: str, context: dict, chat_history: list) -> str:
    """Answer technician questions using only provided diagnostic facts and history."""
    if not isinstance(user_question, str) or not user_question.strip(): raise ValueError("user_question must be a non-empty string")
    if not isinstance(context, dict): raise TypeError("context must be a dictionary")
    if not isinstance(chat_history, list): raise TypeError("chat_history must be a list")
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a local plant reliability assistant. Answer only from the provided diagnostic facts and conversation. Explicitly say when information is unavailable; do not invent readings, diagnoses, procedures, or SAP data. Follow site safety requirements."},
        {"role": "user", "content": "Diagnostic facts (JSON):\n" + json.dumps(context, ensure_ascii=False, default=_json_default, allow_nan=False)},
    ]
    for item in chat_history[-12:]:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str):
            messages.append({"role": item["role"], "content": item["content"][:4000]})
    messages.append({"role": "user", "content": user_question.strip()})
    response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.2, max_tokens=600)
    answer = _clean_model_text(response.choices[0].message.content)
    if not answer: raise RuntimeError("The local reliability agent returned an empty response")
    return answer


# Friendly compatibility names for callers migrating from copilot.py.
generate_reliability_copilot_report = run_reliability_investigation
converse_with_asset = chat_with_agent

__all__ = ["OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "client", "SAP_WORK_ORDER_SCHEMA", "run_reliability_investigation", "chat_with_agent", "_offline_fallback_report", "_tool_definitions"]
