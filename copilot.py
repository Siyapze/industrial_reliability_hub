"""OpenAI-backed structured engineering reports and reliability chat."""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecommendedSAPWorkOrder(BaseModel):
    """SAP PM notification/work-order recommendation fields."""

    model_config = ConfigDict(extra="forbid")

    maintenance_activity_type: StrictStr
    priority: Literal["1-Emergency", "2-Urgent", "3-Normal"]
    required_craft: StrictStr
    required_inspections: list[StrictStr]


class ReliabilityReport(BaseModel):
    """Strict, evidence-based plant reliability report using SAP PM terminology."""

    model_config = ConfigDict(extra="forbid")

    functional_location: StrictStr
    incident_summary: StrictStr
    primary_failure_mode: StrictStr
    chronological_cascade: list[StrictStr]
    recommended_sap_work_order: RecommendedSAPWorkOrder
    uncertainties: list[StrictStr]


REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "functional_location": {"type": "string"},
        "incident_summary": {"type": "string"},
        "primary_failure_mode": {"type": "string"},
        "chronological_cascade": {"type": "array", "items": {"type": "string"}},
        "recommended_sap_work_order": {
            "type": "object",
            "properties": {
                "maintenance_activity_type": {"type": "string"},
                "priority": {"type": "string", "enum": ["1-Emergency", "2-Urgent", "3-Normal"]},
                "required_craft": {"type": "string"},
                "required_inspections": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["maintenance_activity_type", "priority", "required_craft", "required_inspections"],
            "additionalProperties": False,
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "functional_location", "incident_summary", "primary_failure_mode",
        "chronological_cascade", "recommended_sap_work_order", "uncertainties",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are an industrial reliability engineering assistant producing a formal plant reliability "
    "assessment aligned with SAP PM notification and work-order practice. Return only JSON matching "
    "the supplied schema, including the functional location, concise incident summary, evidence-based "
    "primary failure mode, chronological sensor cascade, and recommended SAP PM work-order fields. "
    "Use the exact priority values 1-Emergency, 2-Urgent, or 3-Normal. Distinguish observed facts "
    "from hypotheses, state missing evidence in uncertainties, and never fabricate measurements, SAP "
    "master data, confirmed failures, or work already completed. Recommend safe inspections subject "
    "to site procedures; statistical anomalies alone do not prove component failure."
)

RELIABILITY_FMEA_SYSTEM_PROMPT = (
    "Act as a Principal Plant Reliability Engineer preparing a formal, SAP PM-aligned reliability "
    "notification recommendation. Analyze the supplied quantitative telemetry, anomaly cascade, and "
    "prognostic indicators. Return exactly the ReliabilityReport structure: functional_location, "
    "incident_summary, primary_failure_mode, chronological_cascade, recommended_sap_work_order "
    "(maintenance_activity_type, priority, required_craft, required_inspections), and uncertainties. "
    "Use only priority values 1-Emergency, 2-Urgent, or 3-Normal. Base conclusions only on supplied "
    "evidence, distinguish confirmed observations from hypotheses, and list uncertainties explicitly. "
    "Do not invent SAP codes, measurements, or confirmed failures, and do not advise bypassing site "
    "safety procedures."
)

CHAT_SYSTEM_PROMPT = (
    "You are a helpful Plant Reliability Assistant. Answer the technician's question using the active "
    "data summary and previous chat history. Be concise and evidence-based, distinguish observed data "
    "from hypotheses, and do not invent measurements or recommend bypassing site safety procedures."
)


def converse_with_asset(user_question: str, data_summary_json: dict, chat_history: list) -> str:
    """Answer a technician using active telemetry context and recent conversation."""
    if not isinstance(user_question, str) or not user_question.strip():
        raise ValueError("user_question must be a non-empty string.")
    if not isinstance(data_summary_json, dict):
        raise TypeError("data_summary_json must be a dictionary.")
    if not isinstance(chat_history, list):
        raise TypeError("chat_history must be a list.")

    from openai import OpenAI

    messages: list[dict[str, str]] = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "Active data integrity and reliability summary (JSON):\n"
            + json.dumps(data_summary_json, ensure_ascii=False, default=str),
        },
    ]
    # Only accept a bounded window of plain user/assistant messages.
    for item in chat_history[-12:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content[:8000]})
    messages.append({"role": "user", "content": user_question.strip()})

    response = OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("The reliability assistant returned an empty response.")
    return answer.strip()


def generate_reliability_copilot_report(quantitative_payload: dict) -> dict:
    """Generate a Pydantic-validated SAP PM-aligned reliability report."""
    if not isinstance(quantitative_payload, dict):
        raise TypeError("quantitative_payload must be a dictionary.")
    from openai import OpenAI

    response = OpenAI().beta.chat.completions.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": RELIABILITY_FMEA_SYSTEM_PROMPT},
            {"role": "user", "content": "Assess this quantitative reliability payload:\n" + json.dumps(quantitative_payload, ensure_ascii=False, default=str)},
        ],
        response_format=ReliabilityReport,
        temperature=0.2,
    )
    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"OpenAI declined to generate the report: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("OpenAI returned no parsed ReliabilityReport.")
    return message.parsed.model_dump(mode="json")


def _fallback_report(
    quality_report: dict[str, Any],
    anomalies: list[dict[str, Any]],
    rul_estimates: list[dict[str, Any]],
    asset_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = asset_config or {}
    functional_location = str(
        config.get("functional_location") or config.get("functionalLocation") or "Not provided"
    )
    ordered_anomalies = sorted(anomalies, key=lambda item: str(item.get("timestamp", "")))
    if ordered_anomalies:
        top = ordered_anomalies[0]
        sensor = str(top.get("sensor", "unknown sensor"))
        asset_id = str(top.get("asset_id", "unknown asset"))
        timestamp = str(top.get("timestamp", "unknown time"))
        incident_summary = (
            f"Detected {len(ordered_anomalies)} statistical anomaly event(s) for {asset_id}. "
            f"The earliest reported deviation is {sensor} at {timestamp}; verify against operating "
            "conditions and maintenance history."
        )
        chronological_cascade = [
            f"{item.get('sensor', 'unknown sensor')} "
            f"{('drifted' if item.get('direction') in {'increasing', 'decreasing'} else 'deviated')} "
            f"at {item.get('timestamp', 'unknown time')}"
            for item in ordered_anomalies
        ]
        primary_failure_mode = "Unconfirmed statistical deviation; inspection required"
        if "vibration" in sensor.lower():
            craft = "Millwright"
        elif any(term in sensor.lower() for term in ("current", "motor", "voltage", "electrical")):
            craft = "I&E Technician"
        else:
            craft = "Maintenance Technician"
        priority = "1-Emergency" if str(top.get("severity", "")).upper() == "CRITICAL" else (
            "2-Urgent" if str(top.get("severity", "")).upper() == "HIGH" else "3-Normal"
        )
        inspections = [
            f"Verify {sensor} reading, calibration, and signal quality against a trusted reference.",
            "Inspect the associated equipment and operating conditions using approved site procedures.",
            "Review recent maintenance history and process operating mode before confirming a failure mode.",
        ]
    else:
        incident_summary = (
            "No statistical anomaly events were detected in the analyzed telemetry window. "
            "This does not rule out faults outside measured signals or detection settings."
        )
        chronological_cascade = []
        primary_failure_mode = "No confirmed failure mode identified"
        craft = "Maintenance Technician"
        priority = "3-Normal"
        inspections = [
            "Confirm sensor coverage, sampling cadence, and configured alarm limits.",
            "Continue routine condition monitoring and review operating history.",
        ]

    if rul_estimates:
        nearest = rul_estimates[0]
        inspections.insert(
            0,
            f"Review the trend estimate for {nearest.get('asset_id', 'unknown asset')} "
            f"{nearest.get('sensor', 'unknown sensor')}; it is not a guaranteed failure date.",
        )
    return {
        "functional_location": functional_location,
        "incident_summary": incident_summary,
        "primary_failure_mode": primary_failure_mode,
        "chronological_cascade": chronological_cascade,
        "recommended_sap_work_order": {
            "maintenance_activity_type": "PM01 - Corrective Maintenance" if ordered_anomalies else "PM03 - Preventive Maintenance",
            "priority": priority,
            "required_craft": craft,
            "required_inspections": inspections,
        },
        "uncertainties": [
            "Statistical deviations and linear trend projections are decision-support signals, not confirmed diagnoses or guaranteed remaining-life predictions.",
            f"Data quality score is {float(quality_report.get('quality_score', 0)):.1f}%; verify data completeness and operating context before work-order execution.",
            "Functional location is shown as provided; confirm it against the SAP PM equipment hierarchy.",
        ],
    }


def _validate_report(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Report must be a JSON object.")
    return ReliabilityReport.model_validate(value).model_dump(mode="json")


def generate_engineering_report(
    *,
    quality_report: dict[str, Any],
    anomalies: list[dict[str, Any]],
    rul_estimates: list[dict[str, Any]],
    asset_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a SAP PM-aligned engineering report, falling back without an API key."""
    fallback = _fallback_report(quality_report, anomalies, rul_estimates, asset_config)
    if not os.getenv("OPENAI_API_KEY"):
        return fallback
    payload = {
        "quality_report": quality_report,
        "anomalies": anomalies[:50],
        "rul_estimates": rul_estimates[:20],
        "asset_config": asset_config or {},
    }
    try:
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "engineering_report", "strict": True, "schema": REPORT_SCHEMA}},
            temperature=0.2,
            max_tokens=700,
        )
        return _validate_report(response.choices[0].message.content)
    except Exception:
        return fallback


__all__ = [
    "SeverityLevel",
    "RecommendedSAPWorkOrder",
    "ReliabilityReport",
    "REPORT_SCHEMA",
    "converse_with_asset",
    "generate_reliability_copilot_report",
    "generate_engineering_report",
]
