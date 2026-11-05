"""OpenAI-backed structured engineering report generation."""

from __future__ import annotations

import json
import os
from typing import Any

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


def _fallback_report(quality_report: dict[str, Any], anomalies: list[dict[str, Any]], rul_estimates: list[dict[str, Any]]) -> dict[str, Any]:
    if anomalies:
        top = anomalies[0]
        summary = f"Detected {len(anomalies)} statistical anomaly event(s). The strongest signal is {top['sensor']} on {top['asset_id']} at {top['timestamp']} (z-score {top['z_score']:.2f})."
        hypotheses = ["The observed deviation may indicate a developing operating or sensor condition; verify against operating context and maintenance history."]
        evidence = [f"{item['asset_id']} {item['sensor']}={item['value']:.3g} at {item['timestamp']} (z={item['z_score']:.2f})" for item in anomalies[:5]]
    else:
        summary = "No statistical anomaly events were detected in the analyzed telemetry window. This does not rule out faults outside the measured signals or detection settings."
        hypotheses = []
        evidence = ["No values crossed the configured moving-standard-deviation threshold."]
    actions = ["Review flagged measurements against process setpoints, operating mode, and recent maintenance records.", "If deviations persist, inspect the relevant sensor and equipment using approved site procedures."] if anomalies else ["Continue monitoring and confirm sensor coverage, sampling cadence, and configured alarm limits."]
    if rul_estimates:
        nearest = rul_estimates[0]
        evidence.append(f"Trend estimate for {nearest['asset_id']} {nearest['sensor']}: {nearest['estimated_hours_to_threshold']:.1f} operating hours to configured limit.")
        actions.insert(0, "Prioritize the nearest trend-based threshold estimate for engineering review; it is not a guaranteed failure date.")
    return {
        "summary": summary,
        "root_cause_hypotheses": hypotheses,
        "recommended_actions": actions,
        "evidence": evidence + [f"Data quality score: {quality_report.get('quality_score', 0):.1f}%"],
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
    if not isinstance(value["summary"], str) or any(not isinstance(value[key], list) or not all(isinstance(item, str) for item in value[key]) for key in required[1:]):
        raise ValueError("Report fields have invalid types.")
    return value


def generate_engineering_report(*, quality_report: dict[str, Any], anomalies: list[dict[str, Any]], rul_estimates: list[dict[str, Any]], asset_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create and validate a JSON report. Uses deterministic fallback without an API key."""
    fallback = _fallback_report(quality_report, anomalies, rul_estimates)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback
    payload = {
        "quality_report": quality_report,
        "anomalies": anomalies[:50],
        "rul_estimates": rul_estimates[:20],
        "asset_config": asset_config or {},
    }
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "engineering_report", "strict": True, "schema": REPORT_SCHEMA}},
            temperature=0.2,
            max_tokens=700,
        )
        content = response.choices[0].message.content
        return _validate_report(content)
    except Exception:
        # Do not expose keys or provider details; preserve a useful report on API/schema errors.
        return fallback


__all__ = ["REPORT_SCHEMA", "generate_engineering_report"]
