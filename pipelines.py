"""Telemetry ingestion, normalization, interpolation, and integrity metrics."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

# Canonical measurement names and aliases expected for supported asset classes.
ASSET_CONFIG: dict[str, dict[str, Any]] = {
    "pump": {
        "columns": {
            "vibration_mm_s": ["vibration_mm_s", "vibration", "vibration_velocity", "vib_mm_s"],
            "temperature_c": ["temperature_c", "temperature", "temp_c", "bearing_temp_c"],
            "pressure_bar": ["pressure_bar", "pressure", "discharge_pressure_bar"],
            "flow_lpm": ["flow_lpm", "flow", "flow_rate_lpm"],
        },
        "thresholds": {"vibration_mm_s": 7.1, "temperature_c": 90.0},
    },
    "motor": {
        "columns": {
            "vibration_mm_s": ["vibration_mm_s", "vibration", "vibration_velocity", "vib_mm_s"],
            "temperature_c": ["temperature_c", "temperature", "temp_c", "winding_temp_c"],
            "current_a": ["current_a", "current", "motor_current_a"],
            "speed_rpm": ["speed_rpm", "speed", "rpm"],
        },
        "thresholds": {"vibration_mm_s": 7.1, "temperature_c": 120.0},
    },
    "default": {
        "columns": {
            "vibration_mm_s": ["vibration_mm_s", "vibration", "vibration_velocity", "vib_mm_s"],
            "temperature_c": ["temperature_c", "temperature", "temp_c"],
        },
        "thresholds": {},
    },
}

TIMESTAMP_ALIASES = ("timestamp", "time", "datetime", "event_time", "date")
ASSET_ID_ALIASES = ("asset_id", "asset", "equipment_id", "device_id")


def _read_source(source: Any) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if hasattr(source, "getvalue"):
        data = source.getvalue()
        name = str(getattr(source, "name", "")).lower()
        if name.endswith((".xlsx", ".xls")) or data[:2] == b"PK":
            return pd.read_excel(io.BytesIO(data))
        return pd.read_csv(io.BytesIO(data))
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        return pd.read_excel(io.BytesIO(data)) if data[:2] == b"PK" else pd.read_csv(io.BytesIO(data))
    if isinstance(source, str):
        lower = source.lower()
        if lower.endswith(".csv"):
            return pd.read_csv(source)
        if lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(source)
        raise ValueError("File must be CSV or Excel (.xlsx/.xls).")
    return pd.DataFrame(source)


def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {column: str(column).strip().lower().replace(" ", "_") for column in df.columns}
    df = df.rename(columns=normalized)
    renames: dict[str, str] = {}
    for target, aliases in (("timestamp", TIMESTAMP_ALIASES), ("asset_id", ASSET_ID_ALIASES)):
        for alias in aliases:
            if alias in df.columns:
                renames[alias] = target
                break
    df = df.rename(columns=renames)
    all_sensor_aliases: dict[str, str] = {}
    for config in ASSET_CONFIG.values():
        for canonical, aliases in config["columns"].items():
            for alias in aliases:
                all_sensor_aliases[alias] = canonical
    sensor_renames = {alias: canonical for alias, canonical in all_sensor_aliases.items() if alias in df.columns and canonical not in df.columns}
    return df.rename(columns=sensor_renames)


def load_dataset(source: Any) -> pd.DataFrame:
    """Read and normalize telemetry; requires timestamp and asset identifier."""
    df = _rename_aliases(_read_source(source))
    if df.empty:
        return df
    missing = {"timestamp", "asset_id"}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after alias mapping: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    df["asset_id"] = df["asset_id"].astype("string").str.strip()
    df = df.dropna(subset=["timestamp", "asset_id"])
    df = df[df["asset_id"] != ""].sort_values(["asset_id", "timestamp"])
    for column in df.columns:
        if column not in {"timestamp", "asset_id", "asset_class"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.reset_index(drop=True)


def _asset_class(group: pd.DataFrame) -> str:
    if "asset_class" in group.columns and group["asset_class"].notna().any():
        candidate = str(group["asset_class"].dropna().iloc[0]).strip().lower()
        if candidate in ASSET_CONFIG:
            return candidate
    return "default"


def prepare_feature_table(df: pd.DataFrame, *, interpolation_limit: int = 3, resample_frequency: str | None = None) -> pd.DataFrame:
    """Sort and clean telemetry, optionally regularize sampling and interpolate short gaps.

    If no resampling interval is specified, timestamps and rows are preserved; interior
    missing sensor values are interpolated by row for up to ``interpolation_limit`` rows.
    """
    if df.empty:
        return df.copy()
    clean = _rename_aliases(df.copy())
    if not {"timestamp", "asset_id"}.issubset(clean.columns):
        raise ValueError("Telemetry requires timestamp and asset_id columns.")
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    clean = clean.dropna(subset=["timestamp", "asset_id"]).sort_values(["asset_id", "timestamp"])
    sensor_columns = [c for c in clean.select_dtypes(include="number").columns]
    prepared: list[pd.DataFrame] = []
    for _, group in clean.groupby("asset_id", sort=False):
        group = group.sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
        if resample_frequency:
            group = group.set_index("timestamp")
            numeric = group.select_dtypes(include="number").columns.tolist()
            non_numeric = [c for c in group.columns if c not in numeric]
            group = group[numeric].resample(resample_frequency).mean()
            for col in non_numeric:
                group[col] = clean.loc[clean["asset_id"] == group.iloc[0].get("asset_id"), col].iloc[0] if not group.empty else None
            group["asset_id"] = str(_)
            group.index.name = "timestamp"
            group = group.reset_index()
        numeric_columns = [c for c in sensor_columns if c in group.columns]
        if numeric_columns:
            group[numeric_columns] = group[numeric_columns].interpolate(method="linear", limit=interpolation_limit, limit_area="inside")
        prepared.append(group)
    if not prepared:
        return clean.reset_index(drop=True)
    result = pd.concat(prepared, ignore_index=True).sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    return result


def build_quality_report(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Measure missingness, duplicate timestamps, dropouts, and interpolation impact."""
    raw = raw_df.copy()
    cleaned = cleaned_df if cleaned_df is not None else raw
    sensor_columns = [c for c in raw.select_dtypes(include="number").columns]
    missing = {str(c): int(raw[c].isna().sum()) for c in raw.columns}
    missing_cells = sum(missing.values())
    total_cells = max(int(raw.size), 1)
    duplicate_count = int(raw.duplicated([c for c in ("asset_id", "timestamp") if c in raw.columns]).sum()) if {"asset_id", "timestamp"}.issubset(raw.columns) else 0
    dropout_counts: dict[str, int] = {}
    if {"asset_id", "timestamp"}.issubset(raw.columns):
        times = pd.to_datetime(raw["timestamp"], errors="coerce")
        for asset, indexes in raw.groupby("asset_id").groups.items():
            ordered = times.loc[indexes].dropna().sort_values()
            if len(ordered) >= 3:
                diffs = ordered.diff().dropna()
                expected = diffs.median()
                dropout_counts[str(asset)] = int((diffs > expected * 1.5).sum()) if expected > pd.Timedelta(0) else 0
            else:
                dropout_counts[str(asset)] = 0
    return {
        "rows": int(len(raw)),
        "cleaned_rows": int(len(cleaned)),
        "columns": int(len(raw.columns)),
        "sensor_columns": sensor_columns,
        "missing_by_column": missing,
        "completeness_percent": {str(c): round(100 * (1 - raw[c].isna().mean()), 2) for c in raw.columns},
        "quality_score": round(100 * (1 - missing_cells / total_cells), 2),
        "duplicate_asset_timestamps": duplicate_count,
        "estimated_dropout_intervals_by_asset": dropout_counts,
        "interpolated_cells_estimate": max(0, int(cleaned[sensor_columns].notna().sum().sum() - raw[sensor_columns].notna().sum().sum())) if sensor_columns and all(c in cleaned for c in sensor_columns) else 0,
        "time_range": {
            "start": pd.to_datetime(raw["timestamp"], errors="coerce").min().isoformat() if "timestamp" in raw and raw["timestamp"].notna().any() else None,
            "end": pd.to_datetime(raw["timestamp"], errors="coerce").max().isoformat() if "timestamp" in raw and raw["timestamp"].notna().any() else None,
        },
    }


def detect_signal_drift(df: pd.DataFrame) -> dict[str, Any]:
    """Return per-sensor summary statistics and first-to-last mean shift."""
    output: dict[str, Any] = {}
    for column in df.select_dtypes(include="number").columns:
        series = df[column].dropna()
        midpoint = max(1, len(series) // 2)
        first_mean = float(series.iloc[:midpoint].mean()) if len(series) else None
        last_mean = float(series.iloc[midpoint:].mean()) if len(series) > midpoint else first_mean
        output[column] = {
            "mean": float(series.mean()) if len(series) else None,
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "min": float(series.min()) if len(series) else None,
            "max": float(series.max()) if len(series) else None,
            "second_half_mean_shift": last_mean - first_mean if first_mean is not None and last_mean is not None else None,
        }
    return output


__all__ = ["ASSET_CONFIG", "load_dataset", "prepare_feature_table", "build_quality_report", "detect_signal_drift"]
