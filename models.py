"""Statistical anomaly detection, sensor drift analysis, and maintenance prognostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PRIMARY_STRESS_SENSOR: dict[str, str] = {
    "Centrifugal Pump": "vibration_rms",
    "Cooling Tower": "basin_water_level",
}


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include="number").columns if not c.startswith("_")]


def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    return data.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)


def extract_quantitative_alerts(df: pd.DataFrame, asset_type: str) -> dict[str, Any]:
    """Find the first primary-sensor 3-sigma breach and order secondary drifts."""
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
        delta = values - mean
        mask = mean.notna() & std.notna() & values.notna() & (
            (std.eq(0) & delta.ne(0)) | (std.gt(0) & (delta.abs() > 3 * std))
        )
        hit = np.flatnonzero(mask.to_numpy())
        if len(hit):
            i = int(hit[0])
            breaches.append({"asset_id": asset_id, "group": group, "index": i,
                             "timestamp": group.loc[i, "timestamp"], "value": float(values.iloc[i]),
                             "mean": float(mean.iloc[i]), "std": float(std.iloc[i]),
                             "sigma": float("inf") if std.iloc[i] == 0 else float(delta.iloc[i] / std.iloc[i])})
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
        drift.append({"sensor": sensor, "direction": direction,
                      "first_drift_timestamp": context.loc[onset, "timestamp"].isoformat(),
                      "baseline_value": baseline, "window_end_value": last, "net_change": change})
    drift.sort(key=lambda item: item["first_drift_timestamp"])
    return {"asset_type": asset_type,
            "asset_id": None if pd.isna(hit["asset_id"]) else str(hit["asset_id"]),
            "primary_sensor": primary, "anomaly_timestamp": hit["timestamp"].isoformat(),
            "anomaly_value": hit["value"], "rolling_mean": hit["mean"],
            "rolling_std": hit["std"], "sigma_deviation": hit["sigma"],
            "drift_analysis_window": {"start_timestamp": context["timestamp"].iloc[0].isoformat(),
                                      "end_timestamp": context["timestamp"].iloc[-1].isoformat(),
                                      "data_points": len(context)},
            "drifting_indicators": drift}


def predict_maintenance_window(df: pd.DataFrame, anomaly_timestamp: Any, critical_limit: float) -> tuple[str, float]:
    """Project a degrading sensor trend to its critical limit.

    The degradation segment starts at ``anomaly_timestamp`` and uses the primary
    stress sensor for the asset type when present (vibration_rms for pumps,
    basin_water_level for cooling towers). Otherwise the first numeric sensor
    column is used. A least-squares linear fit is made against elapsed hours.
    The returned ISO timestamp is the projected threshold crossing; the float is
    the elapsed operating hours from the latest observation to that crossing.

    Raises ``ValueError`` when the data or fitted trend cannot support a forward
    crossing. This is a statistical projection, not a guaranteed failure date.
    """
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

    origin = data["timestamp"].iloc[0]
    latest_time = data["timestamp"].iloc[-1]
    x_hours = (data["timestamp"] - origin).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    y = data[sensor].to_numpy(dtype=float)
    if len(np.unique(x_hours)) < 2:
        raise ValueError("At least two distinct timestamps are required to fit the degradation trend.")
    slope, intercept = np.polyfit(x_hours, y, 1)
    latest_value = float(y[-1])
    latest_elapsed = float(x_hours[-1])
    if latest_value == limit or (latest_value > limit and slope >= 0) or (latest_value < limit and slope <= 0):
        if latest_value == limit or (latest_value > limit and slope > 0) or (latest_value < limit and slope < 0):
            return latest_time.isoformat(), 0.0
        raise ValueError("Fitted trend is not moving toward the critical limit.")
    if slope == 0 or not np.isfinite(slope):
        raise ValueError("Fitted degradation trend has no usable slope toward the critical limit.")
    crossing_elapsed = (limit - intercept) / slope
    hours_left = float(crossing_elapsed - latest_elapsed)
    if not np.isfinite(hours_left) or hours_left <= 0:
        raise ValueError("The fitted trend does not project a future critical-limit crossing.")
    projected_timestamp = latest_time + pd.to_timedelta(hours_left, unit="h")
    return projected_timestamp.isoformat(), hours_left


def detect_anomalies(df: pd.DataFrame, *, window: int = 24, threshold_sigma: float = 3.0) -> list[dict[str, Any]]:
    """Find readings beyond a trailing moving mean +/- threshold standard deviations."""
    if window < 2 or threshold_sigma <= 0:
        raise ValueError("window must be >= 2 and threshold_sigma must be positive")
    if df.empty or not {"timestamp", "asset_id"}.issubset(df.columns):
        return []
    events: list[dict[str, Any]] = []
    for asset_id, group in df.sort_values("timestamp").groupby("asset_id", sort=False):
        group = group.reset_index(drop=True)
        for sensor in _sensor_columns(group):
            values = pd.to_numeric(group[sensor], errors="coerce")
            mean = values.shift(1).rolling(window, min_periods=max(3, window // 2)).mean()
            std = values.shift(1).rolling(window, min_periods=max(3, window // 2)).std(ddof=0)
            z = (values - mean) / std.replace(0, np.nan)
            for i in np.flatnonzero((z.abs() >= threshold_sigma).fillna(False).to_numpy()):
                events.append({"asset_id": str(asset_id), "timestamp": pd.Timestamp(group.loc[i, "timestamp"]).isoformat(),
                               "sensor": sensor, "value": float(values.iloc[i]), "moving_mean": float(mean.iloc[i]),
                               "moving_std": float(std.iloc[i]), "z_score": float(z.iloc[i]),
                               "severity": "high" if abs(float(z.iloc[i])) >= threshold_sigma * 1.5 else "warning"})
    return sorted(events, key=lambda event: abs(event["z_score"]), reverse=True)


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
            slope, intercept = np.polyfit(x, y, 1)
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
            output.append({"asset_id": str(asset_id), "sensor": sensor, "threshold": limit,
                           "current_value": latest, "trend_per_hour": float(slope),
                           "estimated_hours_to_threshold": float(hours),
                           "estimated_threshold_time": (now + pd.Timedelta(hours=hours)).isoformat(),
                           "method": "linear trend extrapolation"})
    return sorted(output, key=lambda row: row["estimated_hours_to_threshold"])


def analyze_reliability(df: pd.DataFrame, *, window: int = 24, threshold_sigma: float = 3.0,
                        thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run anomaly detection and RUL estimation, returning UI-ready results."""
    return {"anomalies": detect_anomalies(df, window=window, threshold_sigma=threshold_sigma),
            "rul_estimates": estimate_rul(df, thresholds=thresholds)}


__all__ = ["PRIMARY_STRESS_SENSOR", "extract_quantitative_alerts", "predict_maintenance_window",
           "detect_anomalies", "estimate_rul", "analyze_reliability"]
