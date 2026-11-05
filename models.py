"""Statistical anomaly detection, sensor drift analysis, and maintenance prognostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PRIMARY_STRESS_SENSOR: dict[str, str] = {
    "Centrifugal Pump": "vibration_rms",
    "Cooling Tower": "basin_water_level",
}

# Directional rules for the quantitative-alert API.
_POSITIVE_ONLY_SENSORS = {
    "vibration_mm_s", "vibration_rms", "temperature_c", "bearing_temperature",
}
_NEGATIVE_ONLY_SENSORS = {"pressure_bar", "flow_lpm"}
_TEMPERATURE_SENSORS = {"temperature_c", "bearing_temperature"}

# Recognized operating-load indicators. Equivalent column names are accepted so
# callers can pass telemetry using either canonical or common field names.
_ACTIVITY_SENSOR_ALIASES = (
    ("motor_amps", "motor_current_a", "current_a", "current", "amps"),
    ("rpm", "speed_rpm", "speed"),
    ("flow_lpm", "circulation_flow_gpm"),
)


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include="number").columns if not c.startswith("_")]


def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    return data.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)


def is_asset_running(df: pd.DataFrame, asset_class: str) -> pd.Series:
    """Classify each row as RUNNING, STANDBY, OFFLINE, or UNKNOWN.

    For available motor-current, speed, and flow indicators, a reading below 5%
    of that indicator's positive-reading median baseline is considered inactive.
    Rows with at least one but not all available indicators inactive are STANDBY;
    rows where every available indicator is inactive are OFFLINE. ``asset_class``
    is accepted for API clarity and future class-specific operating rules.
    """
    del asset_class  # Indicators are inferred from the columns present in telemetry.
    activity_columns: list[str] = []
    for aliases in _ACTIVITY_SENSOR_ALIASES:
        column = next((name for name in aliases if name in df.columns), None)
        if column is not None:
            activity_columns.append(column)

    status = pd.Series("UNKNOWN", index=df.index, dtype="string", name="operational_status")
    if not activity_columns:
        return status

    inactive = pd.DataFrame(index=df.index)
    available = pd.DataFrame(index=df.index)
    for column in activity_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        positive = values[values > 0]
        if positive.empty:
            # Zero-only telemetry provides no positive baseline, but measured
            # zeros still indicate inactivity rather than a missing observation.
            inactive[column] = values.notna() & values.le(0)
        else:
            threshold = float(positive.median()) * 0.05
            inactive[column] = values.notna() & values.lt(threshold)
        available[column] = values.notna()

    observed_count = available.sum(axis=1)
    inactive_count = (inactive & available).sum(axis=1)
    has_observation = observed_count.gt(0)
    all_inactive = has_observation & inactive_count.eq(observed_count)
    any_inactive = has_observation & inactive_count.gt(0)
    status.loc[has_observation & ~any_inactive] = "RUNNING"
    status.loc[any_inactive & ~all_inactive] = "STANDBY"
    status.loc[all_inactive] = "OFFLINE"
    return status


def _directional_breach(sensor: str, z_score: float, threshold: float) -> bool:
    """Apply positive-only degradation or negative-only delivery loss rules."""
    if sensor in _POSITIVE_ONLY_SENSORS:
        return z_score > threshold
    if sensor in _NEGATIVE_ONLY_SENSORS:
        return z_score < -threshold
    return abs(z_score) > threshold


def _signed_z_scores(values: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
    """Calculate signed z-scores, retaining zero-variance step changes as infinities."""
    delta = values - mean
    z = delta / std.replace(0, np.nan)
    zero_variance = mean.notna() & std.eq(0) & values.notna() & delta.ne(0)
    z.loc[zero_variance & delta.gt(0)] = np.inf
    z.loc[zero_variance & delta.lt(0)] = -np.inf
    return z


def _severity(sensor: str, z_score: float) -> str:
    if sensor in _TEMPERATURE_SENSORS and z_score < 0:
        return "INFO"
    return "CRITICAL" if abs(z_score) >= 4.0 else "HIGH"


def extract_quantitative_alerts(df: pd.DataFrame, asset_type: str) -> dict[str, Any]:
    """Find the first directionally valid primary-sensor 3-sigma breach and order secondary drifts."""
    if asset_type not in PRIMARY_STRESS_SENSOR:
        raise ValueError(f"Unsupported asset_type {asset_type!r}; expected one of {list(PRIMARY_STRESS_SENSOR)}.")
    primary = PRIMARY_STRESS_SENSOR[asset_type]
    missing = {"timestamp", primary}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = _normalize_timestamps(df)
    if data.empty:
        return {"asset_type": asset_type, "primary_sensor": primary, "anomaly_timestamp": None, "drifting_indicators": []}

    groups = data.groupby("asset_id", sort=False, dropna=False) if "asset_id" in data else [(None, data)]
    breaches: list[dict[str, Any]] = []
    for asset_id, group in groups:
        group = group.reset_index(drop=True)
        values = pd.to_numeric(group[primary], errors="coerce")
        mean = values.shift(1).rolling(15, min_periods=5).mean()
        std = values.shift(1).rolling(15, min_periods=5).std(ddof=0)
        z_scores = _signed_z_scores(values, mean, std)
        eligible = np.array([
            _directional_breach(primary, float(z), 3.0) if pd.notna(z) else False
            for z in z_scores
        ], dtype=bool)
        hit_indices = np.flatnonzero(eligible)
        if len(hit_indices):
            i = int(hit_indices[0])
            breaches.append({
                "asset_id": asset_id, "group": group, "index": i,
                "timestamp": group.loc[i, "timestamp"], "value": float(values.iloc[i]),
                "mean": float(mean.iloc[i]), "std": float(std.iloc[i]),
                "sigma": float(z_scores.iloc[i]),
            })
    if not breaches:
        return {"asset_type": asset_type, "primary_sensor": primary, "anomaly_timestamp": None, "drifting_indicators": []}

    hit = min(breaches, key=lambda x: x["timestamp"])
    group, i = hit["group"], hit["index"]
    start, stop = max(0, i - 7), min(len(group), i + 8)
    context = group.iloc[start:stop].reset_index(drop=True)
    before = context.iloc[:i - start]
    drift: list[dict[str, Any]] = []
    for sensor in [c for c in _sensor_columns(context) if c != primary]:
        values = pd.to_numeric(context[sensor], errors="coerce")
        base_values = pd.to_numeric(before[sensor], errors="coerce").dropna()
        available = values.dropna()
        if base_values.empty or len(available) < 2:
            continue
        baseline = float(base_values.mean())
        noise = float(base_values.std(ddof=0)) if len(base_values) > 1 else 0.0
        last = float(available.iloc[-1])
        change = last - baseline
        tolerance = max(noise, abs(baseline) * 1e-6, 1e-12)
        if abs(change) <= tolerance:
            continue
        direction = "increasing" if change > 0 else "decreasing"
        threshold = baseline + tolerance if change > 0 else baseline - tolerance
        onset = next((j for j, value in enumerate(values) if pd.notna(value) and (value >= threshold if change > 0 else value <= threshold)), None)
        if onset is None:
            continue
        drift.append({
            "sensor": sensor, "direction": direction,
            "first_drift_timestamp": context.loc[onset, "timestamp"].isoformat(),
            "baseline_value": baseline, "window_end_value": last, "net_change": change,
        })
    drift.sort(key=lambda item: item["first_drift_timestamp"])
    severity = _severity(primary, hit["sigma"])
    return {
        "asset_type": asset_type,
        "asset_id": None if pd.isna(hit["asset_id"]) else str(hit["asset_id"]),
        "primary_sensor": primary,
        "anomaly_timestamp": hit["timestamp"].isoformat(),
        "anomaly_value": hit["value"],
        "rolling_mean": hit["mean"],
        "rolling_std": hit["std"],
        "sigma_deviation": hit["sigma"],
        "severity": severity,
        "state": "operational_info" if severity == "INFO" else "alarm",
        "drift_analysis_window": {
            "start_timestamp": context["timestamp"].iloc[0].isoformat(),
            "end_timestamp": context["timestamp"].iloc[-1].isoformat(),
            "data_points": len(context),
        },
        "drifting_indicators": drift,
    }


def trace_event_cascade(
    anomalies_df: pd.DataFrame, time_window_minutes: float = 30,
) -> list[dict[str, Any]]:
    """Group nearby sensor anomalies into per-asset incident clusters.

    Accepts long-form rows with ``timestamp`` and ``sensor`` columns (optional
    boolean ``is_anomaly``, ``anomaly``, ``flagged``, or ``flag`` column), or
    wide-form boolean/flag columns named for sensors. Events are clustered when
    each consecutive event is no more than ``time_window_minutes`` after the
    previous event. Each result includes the first deviating sensor and a
    timestamp-ordered transition path for use as root-cause evidence.
    """
    try:
        window = float(time_window_minutes)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_window_minutes must be a non-negative finite number") from exc
    if not np.isfinite(window) or window < 0:
        raise ValueError("time_window_minutes must be a non-negative finite number")
    if "timestamp" not in anomalies_df.columns:
        raise ValueError("anomalies_df must contain a 'timestamp' column")
    if anomalies_df.empty:
        return []

    flag_columns = ("is_anomaly", "anomaly", "flagged", "flag")
    events: list[dict[str, Any]] = []
    if "sensor" in anomalies_df.columns:
        flag_column = next((name for name in flag_columns if name in anomalies_df.columns), None)
        for _, row in anomalies_df.iterrows():
            if flag_column is not None:
                flag = row[flag_column]
                if pd.isna(flag) or not (bool(flag) if isinstance(flag, (bool, np.bool_)) else str(flag).strip().lower() in {"1", "true", "yes", "anomaly", "flagged"}):
                    continue
            if pd.isna(row["sensor"]):
                continue
            events.append({
                "timestamp": row["timestamp"], "sensor": str(row["sensor"]),
                "asset_id": row.get("asset_id", None),
                "direction": row.get("direction", None),
                "value": row.get("value", None),
            })
    else:
        metadata = {"timestamp", "asset_id", "asset_class", "severity", "state", "message", "z_score", "value"}
        sensor_flags = [
            column for column in anomalies_df.columns
            if column not in metadata and (
                pd.api.types.is_bool_dtype(anomalies_df[column])
                or column.lower().endswith(("_anomaly", "_flag", "_alert"))
            )
        ]
        for _, row in anomalies_df.iterrows():
            for column in sensor_flags:
                flag = row[column]
                if pd.isna(flag) or not (bool(flag) if isinstance(flag, (bool, np.bool_)) else str(flag).strip().lower() in {"1", "true", "yes", "anomaly", "flagged"}):
                    continue
                sensor = column
                for suffix in ("_anomaly", "_flag", "_alert"):
                    if sensor.lower().endswith(suffix):
                        sensor = sensor[:-len(suffix)]
                        break
                events.append({
                    "timestamp": row["timestamp"], "sensor": sensor,
                    "asset_id": row.get("asset_id", None), "direction": None, "value": None,
                })

    normalized: list[dict[str, Any]] = []
    for event in events:
        timestamp = pd.to_datetime(event["timestamp"], errors="coerce", utc=True)
        if pd.isna(timestamp):
            continue
        event["timestamp"] = timestamp
        normalized.append(event)
    if not normalized:
        return []

    event_frame = pd.DataFrame(normalized)
    grouping = event_frame.groupby("asset_id", sort=False, dropna=False) if "asset_id" in event_frame else [(None, event_frame)]
    clusters: list[dict[str, Any]] = []
    max_gap = pd.Timedelta(minutes=window)
    for asset_id, asset_events in grouping:
        asset_events = asset_events.sort_values("timestamp", kind="stable").reset_index(drop=True)
        current: list[dict[str, Any]] = []
        for _, row in asset_events.iterrows():
            event = row.to_dict()
            if current and event["timestamp"] - current[-1]["timestamp"] > max_gap:
                clusters.append(_build_incident_cluster(asset_id, current))
                current = []
            current.append(event)
        if current:
            clusters.append(_build_incident_cluster(asset_id, current))
    clusters.sort(key=lambda cluster: cluster["start_timestamp"])
    return clusters


def _build_incident_cluster(asset_id: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a serializable incident summary from chronologically ordered events."""
    transitions: list[str] = []
    public_events: list[dict[str, Any]] = []
    for event in events:
        timestamp = event["timestamp"]
        sensor = event["sensor"]
        direction = event.get("direction")
        verb = str(direction) if direction is not None and not pd.isna(direction) else "deviated"
        transitions.append(f"{sensor} {verb} at {timestamp.strftime('%H:%M')}")
        item: dict[str, Any] = {"sensor": sensor, "timestamp": timestamp.isoformat()}
        if direction is not None and not pd.isna(direction):
            item["direction"] = str(direction)
        if event.get("value") is not None and not pd.isna(event["value"]):
            item["value"] = event["value"].item() if isinstance(event["value"], np.generic) else event["value"]
        public_events.append(item)
    first = events[0]
    normalized_asset_id = None if asset_id is None or pd.isna(asset_id) else str(asset_id)
    start = first["timestamp"]
    end = events[-1]["timestamp"]
    return {
        "incident_cluster": f"{normalized_asset_id or 'unknown'}:{start.isoformat()}",
        "asset_id": normalized_asset_id,
        "initiating_sensor": first["sensor"],
        "start_timestamp": start.isoformat(),
        "end_timestamp": end.isoformat(),
        "events": public_events,
        "transition_path": " -> ".join(transitions),
    }


def predict_maintenance_window(df: pd.DataFrame, anomaly_timestamp: Any, critical_limit: float) -> tuple[str, float]:
    """Project a degrading sensor trend to its critical limit using linear regression."""
    if df.empty:
        raise ValueError("Cannot project maintenance window from an empty DataFrame.")
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")
    try:
        limit = float(critical_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("critical_limit must be a finite number.") from exc
    if not np.isfinite(limit):
        raise ValueError("critical_limit must be a finite number.")
    anomaly_time = pd.to_datetime(anomaly_timestamp, errors="coerce", utc=True)
    if pd.isna(anomaly_time):
        raise ValueError("anomaly_timestamp must be a valid timestamp.")
    anomaly_time = anomaly_time.tz_convert(None)
    data = _normalize_timestamps(df)
    data = data.loc[data["timestamp"] >= anomaly_time].copy()
    if data.empty:
        raise ValueError("No observations exist at or after anomaly_timestamp.")
    sensor = next((name for name in PRIMARY_STRESS_SENSOR.values() if name in data.columns), None)
    if sensor is None:
        numeric = _sensor_columns(data)
        if not numeric:
            raise ValueError("No numeric sensor column is available for prognostics.")
        sensor = numeric[0]
    data[sensor] = pd.to_numeric(data[sensor], errors="coerce")
    data = data.dropna(subset=[sensor]).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    if len(data) < 2:
        raise ValueError("At least two valid readings from the anomaly timestamp are required.")
    origin, latest_time = data["timestamp"].iloc[0], data["timestamp"].iloc[-1]
    x_hours = (data["timestamp"] - origin).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    y = data[sensor].to_numpy(dtype=float)
    if len(np.unique(x_hours)) < 2:
        raise ValueError("At least two distinct timestamps are required to fit the degradation trend.")
    slope, intercept = np.polyfit(x_hours, y, 1)
    latest_value, latest_elapsed = float(y[-1]), float(x_hours[-1])
    if latest_value == limit:
        return latest_time.isoformat(), 0.0
    if (latest_value > limit and slope > 0) or (latest_value < limit and slope < 0):
        return latest_time.isoformat(), 0.0
    if slope == 0 or not np.isfinite(slope) or (limit - latest_value) / slope <= 0:
        raise ValueError("Fitted trend does not project a future crossing toward the critical limit.")
    crossing_elapsed = (limit - intercept) / slope
    hours_left = float(crossing_elapsed - latest_elapsed)
    if not np.isfinite(hours_left) or hours_left <= 0:
        raise ValueError("The fitted trend does not project a future critical-limit crossing.")
    return (latest_time + pd.to_timedelta(hours_left, unit="h")).isoformat(), hours_left


def detect_anomalies(
    df: pd.DataFrame, *, window: int = 24, threshold_sigma: float = 3.0,
    negative_deviations_as_warnings: bool = False,
) -> list[dict[str, Any]]:
    """Detect asymmetric deviations and mark standby deviations as informational.

    Positive deviations generate warnings when they exceed the threshold. Negative
    deviations are emitted as ``INFO: Below Baseline`` by default; set
    ``negative_deviations_as_warnings=True`` to promote them to warnings. Any
    statistical deviations observed while an asset is not RUNNING are retained as
    informational events instead of alarms.
    """
    if window < 2 or threshold_sigma <= 0:
        raise ValueError("window must be >= 2 and threshold_sigma must be positive")
    if df.empty or not {"timestamp", "asset_id"}.issubset(df.columns):
        return []
    events: list[dict[str, Any]] = []
    for asset_id, group in df.sort_values("timestamp").groupby("asset_id", sort=False):
        group = group.reset_index(drop=True)
        asset_class = str(group["asset_class"].dropna().iloc[0]) if "asset_class" in group and group["asset_class"].notna().any() else "unknown"
        operating_status = is_asset_running(group, asset_class)
        for sensor in _sensor_columns(group):
            values = pd.to_numeric(group[sensor], errors="coerce")
            mean = values.shift(1).rolling(window, min_periods=max(3, window // 2)).mean()
            std = values.shift(1).rolling(window, min_periods=max(3, window // 2)).std(ddof=0)
            z_scores = _signed_z_scores(values, mean, std)
            for i, z_value in enumerate(z_scores):
                if pd.isna(z_value):
                    continue
                z_score = float(z_value)
                if abs(z_score) <= threshold_sigma:
                    continue
                is_negative = z_score < 0
                status = str(operating_status.iloc[i])
                in_standby = status in {"STANDBY", "OFFLINE"}
                below_baseline_info = is_negative and not negative_deviations_as_warnings
                if in_standby or below_baseline_info:
                    severity = "INFO"
                else:
                    severity = _severity(sensor, z_score)
                message = (
                    f"Statistical deviation during {status.lower()}; alarm suppressed"
                    if in_standby else
                    "INFO: Below Baseline" if below_baseline_info else ""
                )
                events.append({
                    "asset_id": str(asset_id),
                    "timestamp": pd.Timestamp(group.loc[i, "timestamp"]).isoformat(),
                    "sensor": sensor,
                    "value": float(values.iloc[i]),
                    "moving_mean": float(mean.iloc[i]),
                    "moving_std": float(std.iloc[i]),
                    "z_score": z_score,
                    "severity": severity,
                    "state": "standby_info" if in_standby else ("operational_info" if severity == "INFO" else "alarm"),
                    "operational_status": status,
                    "message": message,
                })
    return sorted(events, key=lambda event: (event["severity"] == "INFO", -abs(event["z_score"])))


def estimate_rul(df: pd.DataFrame, thresholds: dict[str, Any] | None = None, *, min_points: int = 5) -> list[dict[str, Any]]:
    """Estimate time to configured thresholds using linear least-squares trends."""
    output: list[dict[str, Any]] = []
    if df.empty or not {"timestamp", "asset_id"}.issubset(df.columns):
        return output
    for asset_id, group in df.groupby("asset_id", sort=False):
        group = group.sort_values("timestamp")
        times = pd.to_datetime(group["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
        for sensor, configured in (thresholds or {}).items():
            if sensor not in group:
                continue
            values = pd.to_numeric(group[sensor], errors="coerce")
            valid = times.notna() & values.notna()
            if int(valid.sum()) < min_points:
                continue
            x = (times[valid] - times[valid].iloc[0]).dt.total_seconds().to_numpy() / 3600
            y = values[valid].to_numpy(dtype=float)
            if len(np.unique(x)) < 2:
                continue
            slope, _ = np.polyfit(x, y, 1)
            latest = float(y[-1])
            now = pd.Timestamp(times[valid].iloc[-1])
            if isinstance(configured, dict):
                limit = configured.get("max")
                direction = 1
                if limit is None:
                    limit, direction = configured.get("min"), -1
            else:
                limit, direction = configured, 1
            if limit is None:
                continue
            limit = float(limit)
            if (direction == 1 and latest >= limit) or (direction == -1 and latest <= limit):
                hours = 0.0
            elif (direction == 1 and slope > 0) or (direction == -1 and slope < 0):
                hours = max(0.0, (limit - latest) / slope)
            else:
                continue
            output.append({
                "asset_id": str(asset_id), "sensor": sensor, "threshold": limit,
                "current_value": latest, "trend_per_hour": float(slope),
                "estimated_hours_to_threshold": float(hours),
                "estimated_threshold_time": (now + pd.Timedelta(hours=hours)).isoformat(),
                "method": "linear trend extrapolation",
            })
    return sorted(output, key=lambda row: row["estimated_hours_to_threshold"])


def analyze_reliability(
    df: pd.DataFrame, *, window: int = 24, threshold_sigma: float = 3.0,
    thresholds: dict[str, Any] | None = None, negative_deviations_as_warnings: bool = False,
) -> dict[str, Any]:
    """Run anomaly detection and RUL estimation, returning UI-ready results."""
    return {
        "anomalies": detect_anomalies(
            df, window=window, threshold_sigma=threshold_sigma,
            negative_deviations_as_warnings=negative_deviations_as_warnings,
        ),
        "rul_estimates": estimate_rul(df, thresholds=thresholds),
    }


__all__ = [
    "PRIMARY_STRESS_SENSOR", "is_asset_running", "extract_quantitative_alerts",
    "predict_maintenance_window", "trace_event_cascade", "detect_anomalies", "estimate_rul", "analyze_reliability",
]
