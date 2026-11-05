"""Moving-statistics anomaly detection and trend-based RUL estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PRIMARY_STRESS_SENSOR: dict[str, str] = {
    "Centrifugal Pump": "vibration_rms",
    "Cooling Tower": "basin_water_level",
}


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.select_dtypes(include="number").columns if not column.startswith("_")]


def extract_quantitative_alerts(df: pd.DataFrame, asset_type: str) -> dict[str, Any]:
    """Extract the first 3-sigma stress-sensor breach and surrounding sensor drifts.

    A 15-reading trailing baseline (excluding the current observation) is used
    to calculate the stress sensor's rolling mean and standard deviation. The
    first chronological value outside the mean +/- 3 standard deviations is
    selected. For that breach, up to 15 surrounding records (7 before, the
    breach, and 7 after) are examined for directional movement in other numeric
    sensors. Drift indicators are ordered by their first departure from the
    pre-breach baseline.

    Returns a dictionary with ``anomaly_timestamp``, ``primary_sensor``,
    ``drifting_indicators``, and related breach statistics. If no qualifying
    breach exists, ``anomaly_timestamp`` is ``None`` and the indicators list is
    empty.
    """
    if asset_type not in PRIMARY_STRESS_SENSOR:
        supported = ", ".join(PRIMARY_STRESS_SENSOR)
        raise ValueError(f"Unsupported asset_type {asset_type!r}. Expected one of: {supported}.")
    primary_sensor = PRIMARY_STRESS_SENSOR[asset_type]
    required = {"timestamp", primary_sensor}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for {asset_type}: {sorted(missing)}")
    if df.empty:
        return {
            "asset_type": asset_type,
            "primary_sensor": primary_sensor,
            "anomaly_timestamp": None,
            "anomaly_value": None,
            "rolling_mean": None,
            "rolling_std": None,
            "sigma_deviation": None,
            "drifting_indicators": [],
            "message": "No data rows were provided.",
        }

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    data[primary_sensor] = pd.to_numeric(data[primary_sensor], errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid timestamps are available for anomaly extraction.")

    # Evaluate each asset separately where an asset identifier is supplied.
    group_key = "asset_id" if "asset_id" in data.columns else None
    groups = data.groupby(group_key, sort=False, dropna=False) if group_key else [(None, data)]
    candidates: list[dict[str, Any]] = []
    window = 15
    min_periods = 5
    for asset_id, group in groups:
        group = group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        values = pd.to_numeric(group[primary_sensor], errors="coerce")
        baseline_mean = values.shift(1).rolling(window=window, min_periods=min_periods).mean()
        baseline_std = values.shift(1).rolling(window=window, min_periods=min_periods).std(ddof=0)
        deviation = values - baseline_mean
        # When a baseline is perfectly constant, any finite nonzero departure
        # is beyond its zero-width 3-sigma boundary.
        breach_mask = baseline_mean.notna() & baseline_std.notna() & values.notna() & (
            (baseline_std.eq(0) & deviation.ne(0))
            | (baseline_std.gt(0) & (deviation.abs() > 3.0 * baseline_std))
        )
        for index in np.flatnonzero(breach_mask.to_numpy()):
            sigma_deviation = (
                float("inf") if baseline_std.iloc[index] == 0
                else float(deviation.iloc[index] / baseline_std.iloc[index])
            )
            candidates.append({
                "asset_id": None if asset_id is None or pd.isna(asset_id) else str(asset_id),
                "group": group,
                "index": int(index),
                "timestamp": pd.Timestamp(group.loc[index, "timestamp"]),
                "value": float(values.iloc[index]),
                "mean": float(baseline_mean.iloc[index]),
                "std": float(baseline_std.iloc[index]),
                "sigma_deviation": sigma_deviation,
            })
            break

    if not candidates:
        return {
            "asset_type": asset_type,
            "primary_sensor": primary_sensor,
            "anomaly_timestamp": None,
            "anomaly_value": None,
            "rolling_mean": None,
            "rolling_std": None,
            "sigma_deviation": None,
            "drifting_indicators": [],
            "message": "No primary-sensor values exceeded the rolling 3-sigma boundary.",
        }

    breach = min(candidates, key=lambda item: item["timestamp"])
    group = breach["group"]
    breach_index = breach["index"]
    context_start = max(0, breach_index - 7)
    context_end = min(len(group), breach_index + 8)
    context = group.iloc[context_start:context_end].reset_index(drop=True)
    context_breach_index = breach_index - context_start
    secondary_sensors = [
        column for column in _sensor_columns(context)
        if column != primary_sensor
    ]

    drifting_indicators: list[dict[str, Any]] = []
    # Use up to seven observations preceding the breach as the local baseline.
    pre_breach = context.iloc[:context_breach_index]
    if pre_breach.empty:
        pre_breach = context.iloc[:1]
    for sensor in secondary_sensors:
        series = pd.to_numeric(context[sensor], errors="coerce")
        baseline_values = pd.to_numeric(pre_breach[sensor], errors="coerce").dropna()
        observed = series.dropna()
        if baseline_values.empty or len(observed) < 2:
            continue
        baseline = float(baseline_values.mean())
        baseline_std = float(baseline_values.std(ddof=0)) if len(baseline_values) > 1 else 0.0
        tail = pd.to_numeric(context.iloc[-1:][sensor], errors="coerce")
        if tail.empty or pd.isna(tail.iloc[0]):
            tail = observed.iloc[-1:]
        net_change = float(tail.iloc[0]) - baseline
        tolerance = max(baseline_std, abs(baseline) * 1e-6, 1e-12)
        if abs(net_change) <= tolerance:
            continue
        direction = "increasing" if net_change > 0 else "decreasing"
        threshold = baseline + tolerance if direction == "increasing" else baseline - tolerance
        onset_index: int | None = None
        for index, value in enumerate(series):
            if pd.isna(value):
                continue
            moved = float(value) >= threshold if direction == "increasing" else float(value) <= threshold
            if moved:
                onset_index = index
                break
        if onset_index is None:
            continue
        slope = float(np.polyfit(np.arange(len(observed), dtype=float), observed.to_numpy(dtype=float), 1)[0])
        drifting_indicators.append({
            "sensor": sensor,
            "direction": direction,
            "first_drift_timestamp": pd.Timestamp(context.loc[onset_index, "timestamp"]).isoformat(),
            "baseline_value": baseline,
            "window_end_value": float(tail.iloc[0]),
            "net_change": net_change,
            "trend_per_interval": slope,
        })

    drifting_indicators.sort(key=lambda item: item["first_drift_timestamp"])
    return {
        "asset_type": asset_type,
        "asset_id": breach["asset_id"],
        "primary_sensor": primary_sensor,
        "anomaly_timestamp": breach["timestamp"].isoformat(),
        "anomaly_value": breach["value"],
        "rolling_mean": breach["mean"],
        "rolling_std": breach["std"],
        "sigma_deviation": breach["sigma_deviation"],
        "drift_analysis_window": {
            "start_timestamp": pd.Timestamp(context["timestamp"].iloc[0]).isoformat(),
            "end_timestamp": pd.Timestamp(context["timestamp"].iloc[-1]).isoformat(),
            "data_points": int(len(context)),
        },
        "drifting_indicators": drifting_indicators,
    }


def detect_anomalies(df: pd.DataFrame, *, window: int = 24, threshold_sigma: float = 3.0) -> list[dict[str, Any]]:
    """Find points beyond a trailing moving mean +/- threshold times moving std.

    The current point is excluded from its baseline to reduce masking of abrupt events.
    Each returned record identifies the exact asset, timestamp, sensor, measurement,
    baseline, and signed z-score for an actionable alert.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    if threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be positive")
    if df.empty or "timestamp" not in df or "asset_id" not in df:
        return []
    events: list[dict[str, Any]] = []
    for asset_id, group in df.sort_values("timestamp").groupby("asset_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for sensor in _sensor_columns(group):
            values = pd.to_numeric(group[sensor], errors="coerce")
            baseline = values.shift(1).rolling(window=window, min_periods=max(3, min(window, window // 2))).mean()
            moving_std = values.shift(1).rolling(window=window, min_periods=max(3, min(window, window // 2))).std(ddof=0)
            z_scores = (values - baseline) / moving_std.replace(0, np.nan)
            mask = z_scores.abs() >= threshold_sigma
            for index in np.flatnonzero(mask.fillna(False).to_numpy()):
                events.append({
                    "asset_id": str(asset_id),
                    "timestamp": pd.Timestamp(group.loc[index, "timestamp"]).isoformat(),
                    "sensor": sensor,
                    "value": float(values.iloc[index]),
                    "moving_mean": float(baseline.iloc[index]),
                    "moving_std": float(moving_std.iloc[index]),
                    "z_score": float(z_scores.iloc[index]),
                    "severity": "high" if abs(float(z_scores.iloc[index])) >= threshold_sigma * 1.5 else "warning",
                })
    return sorted(events, key=lambda event: abs(event["z_score"]), reverse=True)


def estimate_rul(df: pd.DataFrame, thresholds: dict[str, Any] | None = None, *, min_points: int = 5) -> list[dict[str, Any]]:
    """Estimate hours to a configured sensor limit using a linear least-squares trend.

    Only increasing values toward an upper limit (or decreasing values toward a lower
    limit, when configured as ``{"min": value}``) produce an estimate. This is a
    simple extrapolation and should be treated as decision support, not a guarantee.
    """
    thresholds = thresholds or {}
    results: list[dict[str, Any]] = []
    if df.empty or "timestamp" not in df or "asset_id" not in df:
        return results
    for asset_id, group in df.groupby("asset_id", sort=False):
        group = group.sort_values("timestamp")
        times = pd.to_datetime(group["timestamp"], errors="coerce")
        for sensor, configured in thresholds.items():
            if sensor not in group.columns:
                continue
            values = pd.to_numeric(group[sensor], errors="coerce")
            valid = values.notna() & times.notna()
            if int(valid.sum()) < min_points:
                continue
            x = (times[valid] - times[valid].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
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
                    limit = configured.get("min")
                    direction = -1
            else:
                limit, direction = configured, 1
            if limit is None:
                continue
            limit = float(limit)
            toward_limit = (direction == 1 and slope > 0) or (direction == -1 and slope < 0)
            if direction == 1 and latest >= limit or direction == -1 and latest <= limit:
                hours = 0.0
                crossing = now
            elif toward_limit:
                hours = max(0.0, (limit - latest) / slope) if direction == 1 else max(0.0, (limit - latest) / slope)
                crossing = now + pd.Timedelta(hours=hours)
            else:
                continue
            results.append({
                "asset_id": str(asset_id),
                "sensor": sensor,
                "threshold": limit,
                "current_value": latest,
                "trend_per_hour": float(slope),
                "estimated_hours_to_threshold": float(hours),
                "estimated_threshold_time": crossing.isoformat(),
                "method": "linear trend extrapolation",
            })
    return sorted(results, key=lambda row: row["estimated_hours_to_threshold"])


def analyze_reliability(df: pd.DataFrame, *, window: int = 24, threshold_sigma: float = 3.0, thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run analytical operations and return UI-ready results."""
    return {
        "anomalies": detect_anomalies(df, window=window, threshold_sigma=threshold_sigma),
        "rul_estimates": estimate_rul(df, thresholds=thresholds),
    }


__all__ = ["PRIMARY_STRESS_SENSOR", "extract_quantitative_alerts", "detect_anomalies", "estimate_rul", "analyze_reliability"]
