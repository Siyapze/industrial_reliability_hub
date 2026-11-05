"""OpenAI-backed structured engineering reports and reliability chat."""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ReliabilityReport(BaseModel):
    """Strict structured FMEA-style reliability assessment."""

    model_config = ConfigDict(extra="forbid")

    primary_fault_diagnosis: StrictStr
    failure_mechanism_explanation: StrictStr
    suspected_root_cause: StrictStr
    severity_level: SeverityLevel
    immediate_corrective_actions: list[StrictStr]


REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "root_cause_hypotheses": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "root_cause_hypotheses", "recommended_actions", "evidence", "uncertainties"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are an industrial reliability engineering assistant. Analyze only the supplied evidence. "
    "Distinguish observations from hypotheses, never claim certainty without evidence, and provide "
    "safe, practical inspection recommendations. Return a JSON engineering report matching the schema. "
    "Do not fabricate measurements or declare a component failed based only on statistical anomalies."
)

RELIABILITY_FMEA_SYSTEM_PROMPT = (
    "Act as a Principal Plant Reliability Engineer conducting a root-cause Failure Modes and Effects "
    "Analysis (FMEA). Evaluate the supplied quantitative telemetry and prognostic indicators. Base the "
    "diagnosis only on the provided evidence, distinguish confirmed observations from hypotheses, and "
    "explain the plausible failure mechanism. Assign severity conservatively using LOW, MEDIUM, HIGH, "
    "or CRITICAL. Recommend immediate, safe, practical corrective actions; do not invent measurements, "
    "claim certainty unsupported by the payload, or advise bypassing site safety procedures. Return the "
    "response in the required ReliabilityReport structure."
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
    """Generate a Pydantic-validated FMEA report using the modern OpenAI SDK."""
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


def _fallback_report(quality_report: dict[str, Any], anomalies: list[dict[str, Any]], rul_estimates: list[dict[str, Any]]) -> dict[str, Any]:
    if anomalies:
        top = anomalies[0]
        summary = (
            f"Detected {len(anomalies)} statistical anomaly event(s). The strongest signal is "
            f"{top.get('sensor', 'unknown sensor')} on {top.get('asset_id', 'unknown asset')} "
            f"at {top.get('timestamp', 'unknown time')} (z-score {float(top.get('z_score', 0)):.2f})."
        )
        hypotheses = ["The deviation may indicate a developing operating or sensor condition; verify it against operating context and maintenance history."]
        evidence = [
            f"{item.get('asset_id', 'unknown asset')} {item.get('sensor', 'unknown sensor')}="
            f"{item.get('value', 'unknown')} at {item.get('timestamp', 'unknown time')} "
            f"(z={float(item.get('z_score', 0)):.2f})"
            for item in anomalies[:5]
        ]
    else:
        summary = "No statistical anomaly events were detected in the analyzed telemetry window. This does not rule out faults outside the measured signals or detection settings."
        hypotheses = []
        evidence = ["No values crossed the configured moving-statistics threshold."]
    actions = [
        "Review flagged measurements against process setpoints, operating mode, and recent maintenance records.",
        "If deviations persist, inspect the relevant sensor and equipment using approved site procedures.",
    ] if anomalies else ["Continue monitoring and confirm sensor coverage, sampling cadence, and configured alarm limits."]
    if rul_estimates:
        nearest = rul_estimates[0]
        evidence.append(
            f"Trend estimate for {nearest.get('asset_id', 'unknown asset')} "
            f"{nearest.get('sensor', 'unknown sensor')}: "
            f"{float(nearest.get('estimated_hours_to_threshold', 0)):.1f} operating hours to configured limit."
        )
        actions.insert(0, "Prioritize the nearest trend-based threshold estimate for engineering review; it is not a guaranteed failure date.")
    return {
        "summary": summary,
        "root_cause_hypotheses": hypotheses,
        "recommended_actions": actions,
        "evidence": evidence + [f"Data quality score: {float(quality_report.get('quality_score', 0)):.1f}%"],
        "uncertainties": ["Statistical deviations and linear trend projections are decision-support signals, not confirmed diagnoses or guaranteed remaining-life predictions."],
    }


def _validate_report(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Report must be a JSON object.")
    required = REPORT_SCHEMA["required"]
    if set(value) != set(required):
        raise ValueError("Report keys do not match the engineering report schema.")
    if not isinstance(value["summary"], str) or any(
        not isinstance(value[key], list) or not all(isinstance(item, str) for item in value[key])
        for key in required[1:]
    ):
        raise ValueError("Report fields have invalid types.")
    return value


def generate_engineering_report(*, quality_report: dict[str, Any], anomalies: list[dict[str, Any]], rul_estimates: list[dict[str, Any]], asset_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate a legacy engineering summary, falling back when no API key exists."""
    fallback = _fallback_report(quality_report, anomalies, rul_estimates)
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
    "ReliabilityReport",
    "REPORT_SCHEMA",
    "converse_with_asset",
    "generate_reliability_copilot_report",
    "generate_engineering_report",
]
