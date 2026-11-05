from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])] + [
        col for col in ["vibration_mm_s", "temperature_c", "pressure_bar", "flow_lpm"] if col in df.columns
    ]


def train_anomaly_model(df: pd.DataFrame, contamination: float = 0.08) -> Tuple[Any, np.ndarray, List[str]]:
    """Train an Isolation Forest model for anomaly detection."""
    if df.empty:
        return None, np.array([]), []

    feature_cols = [c for c in _numeric_columns(df) if c not in {"timestamp", "asset_id"}]
    if not feature_cols:
        return None, np.array([]), ["No numeric sensor columns available for anomaly detection."]

    matrix = df[feature_cols].copy()
    matrix = matrix.fillna(matrix.median())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    model.fit(scaled)
    scores = -model.decision_function(scaled)
    return model, scores, [f"Isolation Forest trained using {len(feature_cols)} features."]


def summarize_asset_health(df: pd.DataFrame) -> Dict[str, Any]:
    """Produce a health summary per asset."""
    if df.empty:
        return {"assets": []}

    numeric_cols = [c for c in _numeric_columns(df) if c not in {"timestamp", "asset_id"}]
    health = []
    for asset, group in df.groupby("asset_id"):
        row = {"asset_id": asset}
        for col in numeric_cols:
            series = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = round(float(series.mean()), 3) if not series.empty else None
            row[f"{col}_max"] = round(float(series.max()), 3) if not series.empty else None
        row["anomaly_events"] = int((group.get("is_anomaly", pd.Series(False, index=group.index))).sum()) if "is_anomaly" in group.columns else 0
        health.append(row)
    return {"assets": health}


def estimate_failure_risk(df: pd.DataFrame) -> Dict[str, float]:
    """Heuristic risk estimate based on sensor deviation from baseline."""
    if df.empty:
        return {}
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {"anomaly_score"}]
    risk = {}
    for col in numeric:
        mean = float(df[col].mean())
        max_value = float(df[col].max())
        risk[col] = min(1.0, max_value / (abs(mean) * 2 + 1e-6))
    return risk


__all__ = ["train_anomaly_model", "summarize_asset_health", "estimate_failure_risk"]
