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

# Exact required schemas for the data-integrity entry point.
DATA_INTEGRITY_SCHEMAS: dict[str, tuple[str, ...]] = {
    "Centrifugal Pump": (
        "timestamp", "vibration_rms", "bearing_temperature", "ambient_humidity"
    ),
    "Cooling Tower": (
        "timestamp", "basin_water_level", "makeup_valve_position_pct", "outflow_rate"
    ),
}

# Physical upper limits for sensors whose units imply a percentage. A repeated
# exact maximum is treated as suspected clipping, not proof of a sensor fault.
CLIPPING_LIMITS: dict[str, tuple[float, ...]] = {
    "ambient_humidity": (100.0, 99.99),
    "basin_water_level": (100.0, 99.99),
    "makeup_valve_position_pct": (100.0, 99.99),
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


def check_data_integrity(file_path_or_buffer: Any, asset_type: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and clean telemetry for a supported asset type.

    NaN sensor readings are interpolated linearly (interior gaps only). Flatlines
    are reported when the same exact numeric value persists for more than 30
    consecutive intervals (at least 32 identical readings). Suspected clipping
    is reported when a percentage-like sensor repeatedly equals a known physical
    maximum such as 100.0 or 99.99.

    Raises:
        ValueError: If ``asset_type`` is unsupported or required schema columns
            are missing.
    """
    if asset_type not in DATA_INTEGRITY_SCHEMAS:
        supported = ", ".join(DATA_INTEGRITY_SCHEMAS)
        raise ValueError(f"Unsupported asset_type {asset_type!r}. Expected one of: {supported}.")

    df = _read_source(file_path_or_buffer).copy()
    df.columns = [str(column).strip().lower() for column in df.columns]
    required_columns = DATA_INTEGRITY_SCHEMAS[asset_type]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns for {asset_type}: {missing_columns}. "
            f"Required schema: {list(required_columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    sensor_columns = [column for column in required_columns if column != "timestamp"]
    for column in sensor_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Stable timestamp ordering makes consecutive-interval checks deterministic.
    df = df.sort_values("timestamp", kind="stable", na_position="last").reset_index(drop=True)
    missing_before = {column: int(df[column].isna().sum()) for column in sensor_columns}
    df[sensor_columns] = df[sensor_columns].interpolate(
        method="linear", limit_area="inside", limit_direction="both"
    )
    missing_after = {column: int(df[column].isna().sum()) for column in sensor_columns}
    interpolated_by_column = {
        column: missing_before[column] - missing_after[column] for column in sensor_columns
    }

    flatline_events: list[dict[str, Any]] = []
    for sensor in sensor_columns:
        values = df[sensor]
        run_start = 0
        while run_start < len(values):
            value = values.iloc[run_start]
            if pd.isna(value):
                run_start += 1
                continue
            run_end = run_start + 1
            while run_end < len(values) and pd.notna(values.iloc[run_end]) and float(values.iloc[run_end]) == float(value):
                run_end += 1
            run_length = run_end - run_start
            intervals = run_length - 1
            if intervals > 30:
                flatline_events.append({
                    "sensor": sensor,
                    "value": float(value),
                    "start_timestamp": df.loc[run_start, "timestamp"].isoformat() if pd.notna(df.loc[run_start, "timestamp"]) else None,
                    "end_timestamp": df.loc[run_end - 1, "timestamp"].isoformat() if pd.notna(df.loc[run_end - 1, "timestamp"]) else None,
                    "consecutive_intervals": int(intervals),
                    "reading_count": int(run_length),
                })
            run_start = run_end

    clipping_events: list[dict[str, Any]] = []
    for sensor in sensor_columns:
        limits = CLIPPING_LIMITS.get(sensor, ())
        if not limits:
            continue
        values = df[sensor]
        for limit in limits:
            positions = np.flatnonzero(values.eq(limit).fillna(False).to_numpy())
            # A single valid boundary value is not enough to call clipping.
            if len(positions) >= 2:
                clipping_events.append({
                    "sensor": sensor,
                    "limit": float(limit),
                    "reading_count": int(len(positions)),
                    "timestamps": [
                        df.loc[index, "timestamp"].isoformat()
                        for index in positions
                        if pd.notna(df.loc[index, "timestamp"])
                    ],
                })

    report: dict[str, Any] = {
        "asset_type": asset_type,
        "schema": list(required_columns),
        "row_count": int(len(df)),
        "missing_data": {
            "found": any(missing_before.values()),
            "missing_by_sensor_before": missing_before,
            "interpolated_by_sensor": interpolated_by_column,
            "interpolated_count": int(sum(interpolated_by_column.values())),
            "remaining_missing_by_sensor": missing_after,
        },
        "sensor_flatlines": {
            "found": bool(flatline_events),
            "threshold_consecutive_intervals": 30,
            "events": flatline_events,
        },
        "sensor_clipping": {
            "found": bool(clipping_events),
            "events": clipping_events,
        },
        "healthy": not (any(missing_before.values()) or flatline_events or clipping_events),
    }
    return df, report


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


__all__ = [
    "ASSET_CONFIG",
    "DATA_INTEGRITY_SCHEMAS",
    "load_dataset",
    "check_data_integrity",
    "prepare_feature_table",
    "build_quality_report",
    "detect_signal_drift",
]
