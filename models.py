"""Moving-statistics anomaly detection and trend-based RUL estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.select_dtypes(include="number").columns if not column.startswith("_")]


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


__all__ = ["detect_anomalies", "estimate_rul", "analyze_reliability"]
