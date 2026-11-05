from __future__ import annotations

import io
from typing import Any, Dict, Iterable, List

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "asset_id",
}


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.Series(pd.NaT, index=series.index)


def load_dataset(source: Any) -> pd.DataFrame:
    """Load a CSV or Excel file into a normalized dataframe."""
    if isinstance(source, (str, bytes, io.BytesIO)):
        if isinstance(source, str):
            if source.lower().endswith(".csv"):
                df = pd.read_csv(source)
            elif source.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(source)
            else:
                raise ValueError("Unsupported file type. Please upload CSV or Excel data.")
        else:
            if hasattr(source, "read"):
                uploaded = source.read()
                if uploaded.startswith(b"PK"):
                    df = pd.read_excel(io.BytesIO(uploaded))
                else:
                    df = pd.read_csv(io.BytesIO(uploaded))
            else:
                df = pd.read_csv(source)
    else:
        df = pd.DataFrame(source)

    if df.empty:
        return df

    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    if "timestamp" not in df.columns:
        for candidate in ["time", "datetime", "event_time", "date"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "timestamp"})
                break

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = _coerce_timestamp(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).copy()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    return df.sort_values("timestamp").reset_index(drop=True)


def build_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a data-quality overview used in the UI and LLM summary."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    missing_by_col = {col: int(df[col].isna().sum()) for col in df.columns}
    completeness = {
        col: round((1 - (df[col].isna().mean() if col in df.columns else 0)) * 100, 2)
        for col in df.columns
    }
    quality_score = round((1 - sum(missing_by_col.values()) / max(1, df.size)) * 100, 2)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric_cols,
        "missing_by_column": missing_by_col,
        "completeness_percent": completeness,
        "quality_score": quality_score,
        "time_range": {
            "start": df["timestamp"].min().isoformat() if not df["timestamp"].empty else None,
            "end": df["timestamp"].max().isoformat() if not df["timestamp"].empty else None,
        },
    }


def prepare_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a feature table for modeling and UI output."""
    feature_df = df.copy()
    feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"])

    numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    feature_df = feature_df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    grouped = (
        feature_df.groupby(["asset_id", pd.Grouper(key="timestamp", freq="15min")], as_index=False)
        .agg({col: "mean" for col in numeric_cols})
    )

    if grouped.empty:
        return feature_df

    grouped["timestamp"] = pd.to_datetime(grouped["timestamp"])
    if "asset_id" not in grouped.columns:
        grouped["asset_id"] = feature_df["asset_id"]
    return grouped


def detect_signal_drift(df: pd.DataFrame) -> Dict[str, Any]:
    """Provide a basic drift signature summary."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    drift = {}
    for col in numeric:
        drift[col] = {
            "mean": round(float(df[col].mean()), 4) if not df[col].empty else None,
            "std": round(float(df[col].std()), 4) if not df[col].empty else None,
            "max": round(float(df[col].max()), 4) if not df[col].empty else None,
        }
    return drift


__all__ = [
    "load_dataset",
    "build_quality_report",
    "prepare_feature_table",
    "detect_signal_drift",
]
