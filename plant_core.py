"""Deterministic telemetry ingestion and reliability-analysis core.

This module combines the former ``pipelines`` and ``models`` APIs and exposes
stable, JSON-friendly diagnostics for autonomous reliability tooling.
"""
from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
import pandas as pd

ASSET_CONFIG: dict[str, dict[str, Any]] = {
    "pump": {"columns": {
        "vibration_mm_s": ["vibration_mm_s", "vibration", "vibration_velocity", "vib_mm_s"],
        "vibration_rms": ["vibration_rms", "vibration_rms_mm_s"],
        "temperature_c": ["temperature_c", "temperature", "temp_c", "bearing_temp_c"],
        "bearing_temperature": ["bearing_temperature", "bearing_temp", "bearing_temperature_c"],
        "pressure_bar": ["pressure_bar", "pressure", "discharge_pressure_bar"],
        "suction_pressure_bar": ["suction_pressure_bar", "suction_pressure", "suction_bar"],
        "discharge_pressure_bar": ["discharge_pressure_bar", "discharge_pressure", "pressure_discharge_bar"],
        "flow_lpm": ["flow_lpm", "flow", "flow_rate_lpm"],
        "motor_amps": ["motor_amps", "motor_current_a", "current_a", "current", "amps"],
        "rpm": ["rpm", "speed_rpm", "speed"],
    }, "thresholds": {"vibration_mm_s": 7.1, "temperature_c": 90.0}},
    "motor": {"columns": {
        "vibration_mm_s": ["vibration_mm_s", "vibration", "vibration_velocity", "vib_mm_s"],
        "vibration_rms": ["vibration_rms", "vibration_rms_mm_s"],
        "temperature_c": ["temperature_c", "temperature", "temp_c", "winding_temp_c"],
        "bearing_temperature": ["bearing_temperature", "bearing_temp", "bearing_temperature_c"],
        "current_a": ["current_a", "current", "motor_current_a", "motor_amps", "amps"],
        "speed_rpm": ["speed_rpm", "speed", "rpm"],
    }, "thresholds": {"vibration_mm_s": 7.1, "temperature_c": 120.0}},
    "cooling_tower": {"columns": {
        "basin_water_level": ["basin_water_level", "level_m", "tower_level", "water_level"],
        "makeup_valve_pct": ["makeup_valve_pct", "valve_pos", "makeup_valve_pos", "lcv_pos", "makeup_valve_position_pct"],
        "basin_temp_c": ["basin_temp_c"],
        "circulation_flow_gpm": ["circulation_flow_gpm", "outflow_rate"],
    }, "thresholds": {}},
    "heat_exchanger": {"columns": {
        "inlet_temp_c": ["inlet_temp_c"], "outlet_temp_c": ["outlet_temp_c"],
        "shell_dp_bar": ["shell_dp_bar"], "tube_dp_bar": ["tube_dp_bar"],
    }, "thresholds": {}},
    "default": {"columns": {
        "vibration_mm_s": ["vibration_mm_s", "vibration", "vibration_velocity", "vib_mm_s"],
        "vibration_rms": ["vibration_rms"],
        "temperature_c": ["temperature_c", "temperature", "temp_c"],
    }, "thresholds": {}},
}
DATA_INTEGRITY_SCHEMAS: dict[str, tuple[str, ...]] = {
    "Centrifugal Pump": ("timestamp", "vibration_rms", "bearing_temperature", "ambient_humidity"),
    "Cooling Tower": ("timestamp", "basin_water_level", "makeup_valve_pct", "circulation_flow_gpm"),
    "Heat Exchanger": ("timestamp", "inlet_temp_c", "outlet_temp_c", "shell_dp_bar", "tube_dp_bar"),
}
CLIPPING_LIMITS: dict[str, tuple[float, ...]] = {
    "ambient_humidity": (0.0, 100.0, 99.99), "basin_water_level": (0.0, 100.0, 99.99),
    "makeup_valve_pct": (0.0, 100.0, 99.99), "makeup_valve_position_pct": (0.0, 100.0, 99.99),
}
TIMESTAMP_ALIASES = ("timestamp", "time", "datetime", "event_time", "date")
ASSET_ID_ALIASES = ("asset_id", "asset", "equipment_id", "device_id")
STATUS_ARTIFACTS = {"bad", "calc error", "scan off"}
FLATLINE_SAMPLE_THRESHOLD = 20
PRIMARY_STRESS_SENSOR = {"Centrifugal Pump": "vibration_rms", "Cooling Tower": "basin_water_level"}
_POSITIVE_ONLY_SENSORS = {"vibration_mm_s", "vibration_rms", "temperature_c", "bearing_temperature"}
_NEGATIVE_ONLY_SENSORS = {"pressure_bar", "flow_lpm"}
_TEMPERATURE_SENSORS = {"temperature_c", "bearing_temperature"}
_ACTIVITY_SENSOR_ALIASES = (
    ("motor_amps", "motor_current_a", "current_a", "current", "amps"),
    ("rpm", "speed_rpm", "speed"), ("flow_lpm", "circulation_flow_gpm"),
)


def _read_source(source: Any) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if hasattr(source, "getvalue"):
        data = source.getvalue()
        name = str(getattr(source, "name", "")).lower()
        return pd.read_excel(io.BytesIO(data)) if name.endswith((".xlsx", ".xls")) or data[:2] == b"PK" else pd.read_csv(io.BytesIO(data))
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        return pd.read_excel(io.BytesIO(data)) if data[:2] == b"PK" else pd.read_csv(io.BytesIO(data))
    if isinstance(source, str):
        if source.lower().endswith(".csv"): return pd.read_csv(source)
        if source.lower().endswith((".xlsx", ".xls")): return pd.read_excel(source)
        raise ValueError("File must be CSV or Excel (.xlsx/.xls).")
    return pd.DataFrame(source)


def _alias_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for config in ASSET_CONFIG.values():
        for canonical, aliases in config["columns"].items():
            for alias in aliases: result[alias] = canonical
    extra = {
        "bearing_temp": "bearing_temperature", "bearing_temp_c": "bearing_temperature",
        "bearing_temperature_c": "bearing_temperature", "vib_rms": "vibration_rms",
        "suction_pressure": "suction_pressure_bar", "suction_bar": "suction_pressure_bar",
        "discharge_pressure": "discharge_pressure_bar", "motor_current": "motor_amps",
        "differential_head": "differential_head_bar", "head_bar": "differential_head_bar",
        "sensor_name": "sensor", "tag": "sensor", "tag_name": "sensor",
        "reading": "value", "measurement": "value", "equipment": "asset_id",
    }
    result.update(extra)
    return result


def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    renames: dict[str, str] = {}
    for canonical, aliases in (("timestamp", TIMESTAMP_ALIASES), ("asset_id", ASSET_ID_ALIASES)):
        found = next((alias for alias in aliases if alias in data.columns), None)
        if found and canonical not in data.columns: renames[found] = canonical
    aliases = _alias_map()
    for col in data.columns:
        if col in aliases and aliases[col] not in data.columns and aliases[col] not in renames.values():
            renames[col] = aliases[col]
    return data.rename(columns=renames)


def clean_historian_telemetry(source: Any) -> pd.DataFrame:
    """Read and normalize wide/long historian data; interpolate interior gaps up to 3 rows."""
    data = _rename_aliases(_read_source(source))
    if data.empty: return data
    if "sensor" in data.columns and "value" in data.columns:
        keys = [c for c in ("timestamp", "asset_id", "asset_class") if c in data]
        if "timestamp" not in keys: raise ValueError("Telemetry requires a timestamp column.")
        data["sensor"] = data["sensor"].astype("string").str.strip().str.lower().str.replace(" ", "_", regex=False)
        data["sensor"] = data["sensor"].map(lambda x: _alias_map().get(str(x), str(x)))
        status = data["value"].astype("string").str.strip().str.casefold()
        data.loc[status.isin(STATUS_ARTIFACTS), "value"] = np.nan
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data.pivot_table(index=keys, columns="sensor", values="value", aggfunc="mean", dropna=False).reset_index()
        data.columns.name = None
    if "timestamp" not in data:
        found = next((c for c in TIMESTAMP_ALIASES if c in data), None)
        if found: data = data.rename(columns={found: "timestamp"})
    if "timestamp" not in data: raise ValueError("Telemetry requires a timestamp column.")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    for col in data.columns:
        if col in {"timestamp", "asset_id", "asset_class", "sensor", "quality", "status", "quality_flag", "status_flag"}: continue
        text = data[col].astype("string").str.strip().str.casefold()
        data.loc[text.isin(STATUS_ARTIFACTS), col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if "asset_id" in data:
        data["asset_id"] = data["asset_id"].astype("string").str.strip()
        data = data.dropna(subset=["timestamp", "asset_id"])
        data = data[data["asset_id"] != ""]
        group_cols = ["asset_id"]
    else:
        data = data.dropna(subset=["timestamp"])
        group_cols = []
    data = data.sort_values(group_cols + ["timestamp"], kind="stable").drop_duplicates(group_cols + ["timestamp"] if group_cols else ["timestamp"], keep="last")
    numeric = data.select_dtypes(include="number").columns.tolist()
    for _, idx in (data.groupby(group_cols, sort=False).groups.items() if group_cols else [(None, data.index)]):
        data.loc[idx, numeric] = data.loc[idx, numeric].interpolate(method="linear", limit=3, limit_area="inside")
    return data.reset_index(drop=True)


def load_dataset(source: Any) -> pd.DataFrame:
    return clean_historian_telemetry(source)


def prepare_feature_table(df: pd.DataFrame, *, interpolation_limit: int = 3, resample_frequency: str | None = None) -> pd.DataFrame:
    data = clean_historian_telemetry(df)
    if data.empty or not resample_frequency: return data
    if "asset_id" not in data: return data.set_index("timestamp").resample(resample_frequency).mean(numeric_only=True).reset_index()
    parts = []
    for aid, group in data.groupby("asset_id", sort=False):
        indexed = group.set_index("timestamp")
        numeric = indexed.select_dtypes(include="number").resample(resample_frequency).mean()
        for col in indexed.columns.difference(numeric.columns):
            if col not in ("asset_id",): numeric[col] = indexed[col].dropna().iloc[0] if indexed[col].notna().any() else None
        numeric["asset_id"] = str(aid)
        numeric = numeric.interpolate(method="linear", limit=interpolation_limit, limit_area="inside")
        parts.append(numeric.reset_index())
    return pd.concat(parts, ignore_index=True).sort_values(["asset_id", "timestamp"]).reset_index(drop=True)


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include="number").columns if not c.startswith("_")]


def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    return data.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)


def audit_sensor_health(df: pd.DataFrame) -> dict[str, Any]:
    """Report sensor flatlines (>20 unchanged intervals), clipping, and range violations."""
    data = df.copy()
    if "timestamp" in data: data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    flatlines, clipping, ranges = [], [], []
    for sensor in _sensor_columns(data):
        values = pd.to_numeric(data[sensor], errors="coerce")
        start = 0
        while start < len(values):
            value = values.iloc[start]
            if pd.isna(value): start += 1; continue
            end = start + 1
            while end < len(values) and pd.notna(values.iloc[end]) and values.iloc[end] == value: end += 1
            if end - start > FLATLINE_SAMPLE_THRESHOLD:
                flatlines.append({"sensor": sensor, "value": float(value), "start_timestamp": data.iloc[start]["timestamp"].isoformat() if "timestamp" in data and pd.notna(data.iloc[start]["timestamp"]) else None, "end_timestamp": data.iloc[end-1]["timestamp"].isoformat() if "timestamp" in data and pd.notna(data.iloc[end-1]["timestamp"]) else None, "consecutive_intervals": int(end-start-1), "reading_count": int(end-start)})
            start = end
        limits = CLIPPING_LIMITS.get(sensor, ())
        for limit in limits:
            positions = np.flatnonzero(values.eq(limit).fillna(False).to_numpy())
            if len(positions) >= 2:
                clipping.append({"sensor": sensor, "boundary": float(limit), "reading_count": int(len(positions)), "timestamps": [data.iloc[i]["timestamp"].isoformat() for i in positions if "timestamp" in data and pd.notna(data.iloc[i]["timestamp"])]})
        # Broad physical sanity limits for common measurements.
        bounds = {"vibration_mm_s": (0, 100), "vibration_rms": (0, 100), "temperature_c": (-50, 300), "bearing_temperature": (-50, 300), "pressure_bar": (-1, 1000), "suction_pressure_bar": (-1, 1000), "discharge_pressure_bar": (-1, 1000), "motor_amps": (0, 100000), "flow_lpm": (0, 1e8)}.get(sensor)
        if bounds:
            bad = values.lt(bounds[0]) | values.gt(bounds[1])
            if bad.any(): ranges.append({"sensor": sensor, "minimum": float(bounds[0]), "maximum": float(bounds[1]), "count": int(bad.sum()), "row_indices": [int(i) for i in np.flatnonzero(bad.fillna(False).to_numpy())]})
    return {"healthy": not (flatlines or clipping or ranges), "flatlines": flatlines, "clipping": clipping, "out_of_range": ranges}


def assess_data_health(file_path_or_buffer: Any, asset_type: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if asset_type not in DATA_INTEGRITY_SCHEMAS: raise ValueError(f"Unsupported asset_type {asset_type!r}.")
    df = _rename_aliases(_read_source(file_path_or_buffer))
    required = DATA_INTEGRITY_SCHEMAS[asset_type]
    missing = [c for c in required if c not in df]
    if missing: raise ValueError(f"Missing required columns for {asset_type}: {missing}. Required schema: {list(required)}")
    df = clean_historian_telemetry(df)
    for c in required:
        if c != "timestamp" and c not in df: df[c] = np.nan
    df = df[list(required)].copy()
    report_health = audit_sensor_health(df)
    sensors = [c for c in required if c != "timestamp"]
    before = {c: int(df[c].isna().sum()) for c in sensors}
    df[sensors] = df[sensors].interpolate(method="linear", limit=3, limit_area="inside")
    after = {c: int(df[c].isna().sum()) for c in sensors}
    interpolated = {c: before[c] - after[c] for c in sensors}
    report = {"asset_type": asset_type, "schema": list(required), "row_count": int(len(df)), "missing_data": {"found": any(before.values()), "missing_by_sensor_before": before, "interpolated_by_sensor": interpolated, "interpolated_count": int(sum(interpolated.values())), "remaining_missing_by_sensor": after}, "sensor_flatlines": {"found": bool(report_health["flatlines"]), "threshold_consecutive_samples": FLATLINE_SAMPLE_THRESHOLD, "events": report_health["flatlines"]}, "sensor_clipping": {"found": bool(report_health["clipping"]), "events": report_health["clipping"]}, "healthy": not (any(before.values()) or not report_health["healthy"])}
    return df, report


def check_data_integrity(file_path_or_buffer: Any, asset_type: str): return assess_data_health(file_path_or_buffer, asset_type)


def build_quality_report(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame | None = None) -> dict[str, Any]:
    raw, clean = raw_df.copy(), cleaned_df if cleaned_df is not None else raw_df
    missing = {str(c): int(raw[c].isna().sum()) for c in raw.columns}
    cells = max(int(raw.size), 1)
    keys = [c for c in ("asset_id", "timestamp") if c in raw]
    duplicate = int(raw.duplicated(keys).sum()) if len(keys) == 2 else 0
    start = end = None
    if "timestamp" in raw and raw["timestamp"].notna().any():
        ts = pd.to_datetime(raw["timestamp"], errors="coerce")
        start, end = ts.min().isoformat(), ts.max().isoformat()
    sensors = _sensor_columns(raw)
    return {"rows": int(len(raw)), "cleaned_rows": int(len(clean)), "columns": int(len(raw.columns)), "sensor_columns": sensors, "missing_by_column": missing, "completeness_percent": {str(c): round(100*(1-raw[c].isna().mean()), 2) for c in raw.columns}, "quality_score": round(100*(1-sum(missing.values())/cells), 2), "duplicate_asset_timestamps": duplicate, "estimated_dropout_intervals_by_asset": {}, "interpolated_cells_estimate": int(max(0, clean[sensors].notna().sum().sum()-raw[sensors].notna().sum().sum())) if sensors and all(c in clean for c in sensors) else 0, "time_range": {"start": start, "end": end}}


def detect_signal_drift(df: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for col in _sensor_columns(df):
        s = pd.to_numeric(df[col], errors="coerce").dropna(); mid = max(1, len(s)//2)
        a = float(s.iloc[:mid].mean()) if len(s) else None; b = float(s.iloc[mid:].mean()) if len(s)>mid else a
        result[col] = {"mean": float(s.mean()) if len(s) else None, "std": float(s.std()) if len(s)>1 else 0.0, "min": float(s.min()) if len(s) else None, "max": float(s.max()) if len(s) else None, "second_half_mean_shift": b-a if a is not None and b is not None else None}
    return result


def _activity_status(group: pd.DataFrame) -> pd.Series:
    found = [next((n for n in aliases if n in group), None) for aliases in _ACTIVITY_SENSOR_ALIASES]
    cols = list(dict.fromkeys(c for c in found if c))
    status = pd.Series("UNKNOWN", index=group.index, dtype="string", name="operational_status")
    if not cols: return status
    vals, inactive, available = {}, {}, {}
    for c in cols:
        v = pd.to_numeric(group[c], errors="coerce"); vals[c] = v
        pos = v[v > 0]; threshold = float(pos.median())*.05 if len(pos) else 0.0
        inactive[c] = v.notna() & (v.lt(threshold) if len(pos) else v.le(0)); available[c] = v.notna()
    avail = pd.DataFrame(available).sum(axis=1); off = pd.DataFrame(inactive).sum(axis=1)
    has = avail.gt(0); all_off = has & off.eq(avail); some = has & off.gt(0)
    status.loc[has & ~some] = "RUNNING"; status.loc[some & ~all_off] = "STANDBY"; status.loc[all_off] = "OFFLINE"
    return status


def is_asset_running(df: pd.DataFrame, asset_class: str = "unknown") -> pd.Series:
    del asset_class
    return _activity_status(df)


def _asset_frame(df: pd.DataFrame, asset_id: Any) -> pd.DataFrame:
    data = df.copy()
    if "asset_id" in data: data = data[data["asset_id"].astype(str) == str(asset_id)]
    return data.sort_values("timestamp", kind="stable").reset_index(drop=True) if "timestamp" in data else data.reset_index(drop=True)


def get_operational_status(df: pd.DataFrame, asset_id: Any) -> str:
    group = _asset_frame(df, asset_id)
    if group.empty: return "UNKNOWN"
    statuses = _activity_status(group)
    return str(statuses.iloc[-1])


def _transient_mask(group: pd.DataFrame, status: pd.Series | None = None) -> pd.Series:
    if "timestamp" not in group: return pd.Series(False, index=group.index)
    status = status if status is not None else _activity_status(group)
    running = status.eq("RUNNING")
    starts = running & ~running.shift(1, fill_value=False)
    times = pd.to_datetime(group["timestamp"], errors="coerce")
    mask = pd.Series(False, index=group.index)
    for idx in group.index[starts]:
        t = times.loc[idx]
        if pd.notna(t): mask |= running & times.ge(t) & times.lt(t + pd.Timedelta(minutes=15))
    return mask


def is_steady_state(df: pd.DataFrame, asset_id: Any) -> bool:
    group = _asset_frame(df, asset_id)
    if len(group) < 2 or "timestamp" not in group: return False
    status = _activity_status(group)
    times = pd.to_datetime(group["timestamp"], errors="coerce")
    now = times.iloc[-1]
    if pd.isna(now) or status.iloc[-1] != "RUNNING": return False
    start = now - pd.Timedelta(minutes=15)
    recent = group.loc[times.ge(start)]
    recent_status = status.loc[recent.index]
    return bool(len(recent) >= 2 and recent_status.eq("RUNNING").all() and (times.loc[recent.index].iloc[-1]-times.loc[recent.index].iloc[0]) >= pd.Timedelta(minutes=15))


def _directional_breach(sensor: str, z: float, threshold: float) -> bool:
    if sensor in _POSITIVE_ONLY_SENSORS: return z > threshold
    if sensor in _NEGATIVE_ONLY_SENSORS: return z < -threshold
    return abs(z) > threshold


def _signed_z_scores(values: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
    delta = values-mean; z = delta/std.replace(0, np.nan)
    zero = mean.notna() & std.eq(0) & values.notna() & delta.ne(0)
    z.loc[zero & delta.gt(0)] = np.inf; z.loc[zero & delta.lt(0)] = -np.inf
    return z


def _severity(sensor: str, z: float) -> str:
    if sensor in _TEMPERATURE_SENSORS and z < 0: return "INFO"
    return "CRITICAL" if abs(z) >= 4 else "HIGH"


def detect_sensor_anomalies(df: pd.DataFrame, window: int = 24, sigma: float = 3.0) -> list[dict[str, Any]]:
    if window < 2 or sigma <= 0: raise ValueError("window must be >= 2 and sigma must be positive")
    if df.empty: return []
    if "timestamp" not in df or "asset_id" not in df: return []
    events = []
    for aid, group in df.sort_values("timestamp", kind="stable").groupby("asset_id", sort=False):
        group = group.reset_index(drop=True); status = _activity_status(group); transient = _transient_mask(group, status)
        for sensor in _sensor_columns(group):
            v = pd.to_numeric(group[sensor], errors="coerce")
            mean = v.shift(1).rolling(window, min_periods=max(3, window//2)).mean()
            std = v.shift(1).rolling(window, min_periods=max(3, window//2)).std(ddof=0)
            zs = _signed_z_scores(v, mean, std)
            for i, zval in enumerate(zs):
                if pd.isna(zval) or abs(float(zval)) <= sigma: continue
                z = float(zval); state = str(status.iloc[i]); standby = state in {"STANDBY", "OFFLINE"}; trans = bool(transient.iloc[i])
                severity = "INFO" if standby or trans or (z < 0 and sensor in _POSITIVE_ONLY_SENSORS) else _severity(sensor, z)
                events.append({"asset_id": str(aid), "timestamp": pd.Timestamp(group.loc[i, "timestamp"]).isoformat(), "sensor": sensor, "value": float(v.iloc[i]), "moving_mean": float(mean.iloc[i]), "moving_std": float(std.iloc[i]), "z_score": z, "severity": severity, "state": "transient_info" if trans else "standby_info" if standby else "operational_info" if severity == "INFO" else "alarm", "operational_status": "TRANSIENT" if trans else state, "message": "Transient startup deviation; alarm suppressed" if trans else f"Statistical deviation during {state.lower()}; alarm suppressed" if standby else ""})
    return sorted(events, key=lambda e: (e["severity"] == "INFO", -abs(e["z_score"])))


def detect_anomalies(df: pd.DataFrame, *, window: int = 24, threshold_sigma: float = 3.0, negative_deviations_as_warnings: bool = False) -> list[dict[str, Any]]:
    events = detect_sensor_anomalies(df, window, threshold_sigma)
    if negative_deviations_as_warnings:
        for e in events:
            if e["z_score"] < 0 and e["state"] == "operational_info": e["severity"], e["state"] = _severity(e["sensor"], e["z_score"]), "alarm"
    return events


def diagnose_hydraulic_vs_mechanical(df: pd.DataFrame, asset_id: Any) -> dict[str, Any]:
    g = _asset_frame(df, asset_id)
    if g.empty: return {"asset_id": str(asset_id), "failure_mode": "INSUFFICIENT_DATA", "evidence": []}
    def col(*names): return next((n for n in names if n in g), None)
    vib = col("vibration_rms", "vibration_mm_s"); suction = col("suction_pressure_bar"); discharge = col("discharge_pressure_bar", "pressure_bar"); amps = col("motor_amps", "motor_current_a", "current_a", "current"); temp = col("bearing_temperature", "temperature_c")
    evidence = []
    if not vib or not suction or not discharge: return {"asset_id": str(asset_id), "failure_mode": "INSUFFICIENT_DATA", "evidence": evidence}
    v = pd.to_numeric(g[vib], errors="coerce"); s = pd.to_numeric(g[suction], errors="coerce"); d = pd.to_numeric(g[discharge], errors="coerce")
    baseline = max(3, len(g)//2); latest = max(1, len(g)//4)
    def changed(series, pos):
        a, b = series.iloc[:baseline].median(), series.iloc[-latest:].median()
        if pd.isna(a) or pd.isna(b): return False
        return (b > a*1.25 if pos else b < a*.8)
    vib_up, suction_down = changed(v, True), changed(s, False)
    head = d-s; head_down = changed(head, False)
    current_spike = changed(pd.to_numeric(g[amps], errors="coerce"), True) if amps else False
    temp_spike = changed(pd.to_numeric(g[temp], errors="coerce"), True) if temp else False
    evidence += (["vibration increased"] if vib_up else []) + (["suction pressure decreased"] if suction_down else []) + (["differential head decreased"] if head_down else []) + (["motor current increased"] if current_spike else []) + (["bearing temperature increased"] if temp_spike else [])
    hydraulic = vib_up and (suction_down or head_down) and not current_spike and not temp_spike
    return {"asset_id": str(asset_id), "failure_mode": "Hydraulic Cavitation/Recirculation" if hydraulic else "Mechanical Unbalance/Looseness" if vib_up else "NO_CONFIRMED_PATTERN", "confidence": "high" if hydraulic else "moderate" if vib_up else "low", "evidence": evidence}


def trace_thermal_vibration_sequence(df: pd.DataFrame, asset_id: Any) -> dict[str, Any]:
    g = _asset_frame(df, asset_id)
    if g.empty or "timestamp" not in g: return {"asset_id": str(asset_id), "sequence": "INSUFFICIENT_DATA", "lead_lag_minutes": None}
    temp = next((c for c in ("bearing_temperature", "temperature_c", "bearing_temp_c") if c in g), None)
    vib = next((c for c in ("vibration_rms", "vibration_mm_s") if c in g), None)
    if not temp or not vib: return {"asset_id": str(asset_id), "sequence": "INSUFFICIENT_DATA", "lead_lag_minutes": None}
    times = pd.to_datetime(g.timestamp, errors="coerce")
    def onset(col):
        x = pd.to_numeric(g[col], errors="coerce"); base = x.iloc[:max(3, len(x)//2)].median(); sd = x.iloc[:max(3, len(x)//2)].std()
        if pd.isna(base): return None
        threshold = base + max(float(sd) if pd.notna(sd) else 0, abs(float(base))*.05, 1e-9)
        ids = np.flatnonzero((x > threshold).fillna(False).to_numpy())
        return times.iloc[int(ids[0])] if len(ids) else None
    t0, v0 = onset(temp), onset(vib)
    if t0 is None or v0 is None: seq, lag = "INSUFFICIENT_SIGNAL", None
    else:
        lag = float((v0-t0).total_seconds()/60)
        seq = "LUBRICATION_STARVATION_TEMPERATURE_LEADS_VIBRATION" if lag > 0 else "STRUCTURAL_OR_COUPLING_MISALIGNMENT_VIBRATION_LEADS_TEMPERATURE" if lag < 0 else "SIMULTANEOUS_ONSET"
    return {"asset_id": str(asset_id), "sequence": seq, "temperature_onset": t0.isoformat() if t0 is not None else None, "vibration_onset": v0.isoformat() if v0 is not None else None, "lead_lag_minutes": lag}


def trace_incident_cascade(anomalies_df: pd.DataFrame, window_minutes: float = 30) -> list[dict[str, Any]]:
    if not np.isfinite(float(window_minutes)) or float(window_minutes) < 0: raise ValueError("window_minutes must be a non-negative finite number")
    if anomalies_df.empty or "timestamp" not in anomalies_df: return []
    data = anomalies_df.copy()
    if "sensor" not in data:
        flags = [c for c in data if c not in {"timestamp", "asset_id", "asset_class", "severity", "state", "message", "value"} and (pd.api.types.is_bool_dtype(data[c]) or c.lower().endswith(("_anomaly", "_flag", "_alert")))]
        records = []
        for _, row in data.iterrows():
            for c in flags:
                if bool(row[c]): records.append({"timestamp": row["timestamp"], "sensor": c.removesuffix("_anomaly").removesuffix("_flag").removesuffix("_alert"), "asset_id": row.get("asset_id"), "value": row.get("value")})
        data = pd.DataFrame(records)
    if data.empty: return []
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    if "is_anomaly" in data: data = data[data.is_anomaly.fillna(False).astype(bool)]
    if "severity" in data: data = data[data.severity.astype(str).str.upper() != "INFO"]
    clusters = []
    groupby = data.groupby("asset_id", dropna=False, sort=False) if "asset_id" in data else [(None, data)]
    gap = pd.Timedelta(minutes=float(window_minutes))
    for aid, group in groupby:
        current = []
        for _, row in group.iterrows():
            item = row.to_dict()
            if current and item["timestamp"]-current[-1]["timestamp"] > gap:
                clusters.append(_incident(aid, current)); current=[]
            current.append(item)
        if current: clusters.append(_incident(aid, current))
    return sorted(clusters, key=lambda c: c["start_timestamp"])


def _incident(aid: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
    public=[]
    for e in events:
        item={"sensor": str(e.get("sensor")), "timestamp": pd.Timestamp(e["timestamp"]).isoformat()}
        for k in ("direction", "value", "severity"):
            v=e.get(k)
            if v is not None and not pd.isna(v): item[k]=v.item() if isinstance(v, np.generic) else v
        public.append(item)
    start=pd.Timestamp(events[0]["timestamp"])
    return {"incident_cluster": f"{str(aid) if aid is not None else 'unknown'}:{start.isoformat()}", "asset_id": None if aid is None or pd.isna(aid) else str(aid), "initiating_sensor": public[0]["sensor"], "start_timestamp": start.isoformat(), "end_timestamp": pd.Timestamp(events[-1]["timestamp"]).isoformat(), "events": public, "transition_path": " -> ".join(f"{e['sensor']} at {pd.Timestamp(e['timestamp']).strftime('%H:%M')}" for e in events)}


def trace_event_cascade(anomalies_df: pd.DataFrame, time_window_minutes: float = 30):
    return trace_incident_cascade(anomalies_df, time_window_minutes)


def evaluate_iso_vibration_zone(vibration_rms: float, machine_group: int = 2) -> dict[str, Any]:
    """Evaluate velocity RMS (mm/s) against ISO 20816-3 group-2 zone boundaries.

    ISO limits depend on machine type, support, power, and measurement position;
    this deterministic default is a screening interpretation, not certification.
    """
    value=float(vibration_rms)
    if not math.isfinite(value) or value < 0: raise ValueError("vibration_rms must be a finite non-negative value")
    if machine_group not in (1, 2): raise ValueError("machine_group must be 1 or 2")
    # Group 2 boundaries (mm/s RMS): A/B 1.4, B/C 2.8, C/D 4.5.
    a_b, b_c, c_d = (1.4, 2.8, 4.5) if machine_group == 2 else (1.1, 2.3, 3.5)
    zone = "A" if value < a_b else "B" if value < b_c else "C" if value < c_d else "D"
    return {"vibration_rms": float(value), "machine_group": int(machine_group), "zone": zone, "classification": "Acceptable" if zone in {"A", "B"} else "Alert/Restricted" if zone == "C" else "Danger/Trip", "standard": "ISO 20816-3", "boundaries_mm_s_rms": {"A_B": float(a_b), "B_C": float(b_c), "C_D": float(c_d)}}


def compute_degradation_rul(df: pd.DataFrame, asset_id: Any, sensor: str, critical_limit: float):
    """Fit an exponential degradation curve; return projected operating hours or -1 if unsupported."""
    if sensor not in df or "timestamp" not in df: return -1
    g = _asset_frame(df, asset_id)
    if len(g) < 5: return -1
    times=pd.to_datetime(g.timestamp, errors="coerce", utc=True).dt.tz_convert(None); vals=pd.to_numeric(g[sensor], errors="coerce")
    valid=times.notna() & vals.notna(); times=times[valid]; vals=vals[valid]
    if len(vals)<5: return -1
    load = _activity_status(g.loc[valid.index[valid] if False else g.index[:len(g)]]) if False else None
    # Prognostics use operating samples only when load indicators are available.
    status=_activity_status(g)
    if status.ne("UNKNOWN").any():
        keep=status.eq("RUNNING").to_numpy()[:len(valid)]
        times=times.iloc[np.flatnonzero(keep)[:len(times)]] if len(keep)==len(times) else times
        vals=vals.iloc[np.flatnonzero(keep)[:len(vals)]] if len(keep)==len(vals) else vals
    if len(vals)<5: return -1
    limit=float(critical_limit); y=vals.to_numpy(dtype=float); x=(times-times.iloc[0]).dt.total_seconds().to_numpy(dtype=float)/3600
    if not np.isfinite(limit) or len(np.unique(x))<3 or np.any(y<=0) or y[-1] >= limit: return -1
    # Log-linear exponential fit; only accelerating, positive-growth hazards qualify.
    slope, intercept=np.polyfit(x, np.log(y), 1)
    if not np.isfinite(slope) or slope <= 0: return -1
    predicted=float(np.exp(intercept+slope*x[-1]))
    # Reject weak/non-accelerating fits; require positive slope and improved fit over a constant.
    fitted=np.exp(intercept+slope*x); ss_res=float(np.sum((y-fitted)**2)); ss_tot=float(np.sum((y-y.mean())**2))
    r2=1-ss_res/ss_tot if ss_tot>0 else 0
    if r2 < .25 or predicted <= 0: return -1
    hours=float((math.log(limit)-math.log(predicted))/slope)
    if not math.isfinite(hours) or hours <= 0: return -1
    return float(hours)


def estimate_rul(df: pd.DataFrame, thresholds: dict[str, Any] | None = None, *, min_points: int = 5) -> list[dict[str, Any]]:
    output=[]
    if df.empty or not {"timestamp", "asset_id"}.issubset(df): return output
    for aid, group in df.groupby("asset_id", sort=False):
        for sensor, configured in (thresholds or {}).items():
            if sensor not in group: continue
            limit = configured.get("max") if isinstance(configured, dict) else configured
            if limit is None and isinstance(configured, dict): limit=configured.get("min")
            if limit is None: continue
            hours=compute_degradation_rul(group.assign(asset_id=str(aid)), str(aid), sensor, float(limit))
            if hours == -1: continue
            ts=pd.to_datetime(group.timestamp, errors="coerce", utc=True).dropna()
            if ts.empty: continue
            now=ts.max().tz_convert(None)
            output.append({"asset_id": str(aid), "sensor": sensor, "threshold": float(limit), "estimated_hours_to_threshold": float(hours), "estimated_threshold_time": (now+pd.Timedelta(hours=hours)).isoformat(), "method": "exponential hazard fit"})
    return sorted(output, key=lambda x:x["estimated_hours_to_threshold"])


def predict_maintenance_window(df: pd.DataFrame, anomaly_timestamp: Any, critical_limit: float) -> tuple[str, float]:
    if df.empty: raise ValueError("Cannot project maintenance window from an empty DataFrame.")
    data=_normalize_timestamps(df); start=pd.to_datetime(anomaly_timestamp, errors="coerce", utc=True)
    if pd.isna(start): raise ValueError("anomaly_timestamp must be a valid timestamp.")
    data=data[data.timestamp >= start.tz_convert(None)]
    sensor=next((s for s in PRIMARY_STRESS_SENSOR.values() if s in data), next(iter(_sensor_columns(data)), None))
    if sensor is None: raise ValueError("No numeric sensor column is available for prognostics.")
    aid=data.asset_id.iloc[0] if "asset_id" in data and len(data) else None
    hours=compute_degradation_rul(data, aid, sensor, critical_limit)
    if hours == -1: raise ValueError("Trend is non-accelerating or below warning limits; no reliable RUL projection.")
    latest=pd.to_datetime(data.timestamp.max())
    return (latest+pd.Timedelta(hours=hours)).isoformat(), float(hours)


def extract_quantitative_alerts(df: pd.DataFrame, asset_type: str) -> dict[str, Any]:
    if asset_type not in PRIMARY_STRESS_SENSOR: raise ValueError(f"Unsupported asset_type {asset_type!r}; expected one of {list(PRIMARY_STRESS_SENSOR)}.")
    sensor=PRIMARY_STRESS_SENSOR[asset_type]
    if sensor not in df or "timestamp" not in df: raise ValueError(f"Missing required columns: ['timestamp', '{sensor}']")
    events=detect_sensor_anomalies(df, window=15, sigma=3)
    hits=[e for e in events if e["sensor"]==sensor and e["severity"]!="INFO"]
    if not hits: return {"asset_type": asset_type, "primary_sensor": sensor, "anomaly_timestamp": None, "drifting_indicators": []}
    hit=min(hits, key=lambda e:e["timestamp"])
    return {"asset_type": asset_type, "asset_id": hit["asset_id"], "primary_sensor": sensor, "anomaly_timestamp": hit["timestamp"], "anomaly_value": float(hit["value"]), "sigma_deviation": float(hit["z_score"]), "severity": hit["severity"], "state": hit["state"], "drifting_indicators": []}


def analyze_reliability(df: pd.DataFrame, *, window: int = 24, threshold_sigma: float = 3.0, thresholds: dict[str, Any] | None = None, negative_deviations_as_warnings: bool = False) -> dict[str, Any]:
    return {"anomalies": detect_anomalies(df, window=window, threshold_sigma=threshold_sigma, negative_deviations_as_warnings=negative_deviations_as_warnings), "rul_estimates": estimate_rul(df, thresholds=thresholds)}

__all__ = ["ASSET_CONFIG", "DATA_INTEGRITY_SCHEMAS", "CLIPPING_LIMITS", "PRIMARY_STRESS_SENSOR", "clean_historian_telemetry", "load_dataset", "prepare_feature_table", "audit_sensor_health", "assess_data_health", "check_data_integrity", "build_quality_report", "detect_signal_drift", "get_operational_status", "is_steady_state", "is_asset_running", "detect_sensor_anomalies", "detect_anomalies", "diagnose_hydraulic_vs_mechanical", "trace_thermal_vibration_sequence", "trace_incident_cascade", "trace_event_cascade", "evaluate_iso_vibration_zone", "compute_degradation_rul", "estimate_rul", "predict_maintenance_window", "extract_quantitative_alerts", "analyze_reliability"]
