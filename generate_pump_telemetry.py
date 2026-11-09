"""Generate realistic multi-sensor telemetry for industrial pump diagnostics.

Produces operational load, suction/discharge pressure, and vibration progression
matching plant_core.py diagnostic signatures and ISO 20816-3 screening criteria.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def generate_telemetry(
    output_path: str = "pump_telemetry.csv",
    n_samples: int = 288,  # 24 hours at 5-minute sampling
) -> pd.DataFrame:
    np.random.seed(42)
    timestamps = pd.date_range("2026-03-01 00:00:00", periods=n_samples, freq="5min")

    records = []

    # --- Asset 1: PUMP-101A (Developing Suction Cavitation / Strainer Clogging) ---
    # Normal Baseline: Suction ~2.2 bar, Discharge ~6.5 bar, Current ~28.0 A, Vib ~1.8 mm/s
    # Failure onset around sample 190: Suction drops, differential head drops,
    # vibration surges past ISO Zone D (>4.5 mm/s), while motor current and temperature remain steady.
    suction_a = 2.2 + np.random.normal(0, 0.04, n_samples)
    discharge_a = 6.5 + np.random.normal(0, 0.06, n_samples)
    amps_a = 28.0 + np.random.normal(0, 0.25, n_samples)
    temp_a = 62.0 + np.random.normal(0, 0.4, n_samples)
    vib_a = 1.8 + np.random.normal(0, 0.06, n_samples)
    flow_a = 120.0 + np.random.normal(0, 1.2, n_samples)

    for i in range(190, n_samples):
        drift = (i - 189) / (n_samples - 190)
        suction_a[i] -= 1.35 * drift + np.random.normal(0, 0.03)  # Drops to ~0.85 bar
        discharge_a[i] -= 0.65 * drift  # Head degrades
        vib_a[i] += 3.85 * (drift ** 1.6)  # Surges to ~5.7 mm/s (Zone D breach)
        flow_a[i] -= 12.0 * drift

    for i in range(n_samples):
        records.append({
            "timestamp": timestamps[i],
            "asset_id": "PUMP-101A",
            "asset_class": "pump",
            "motor_amps": round(float(amps_a[i]), 2),
            "suction_pressure_bar": round(float(suction_a[i]), 2),
            "discharge_pressure_bar": round(float(discharge_a[i]), 2),
            "vibration_rms": round(float(vib_a[i]), 2),
            "bearing_temperature": round(float(temp_a[i]), 1),
            "flow_lpm": round(float(flow_a[i]), 1),
        })

    # --- Asset 2: PUMP-102B (Healthy Baseline Asset) ---
    suction_b = 2.3 + np.random.normal(0, 0.04, n_samples)
    discharge_b = 6.6 + np.random.normal(0, 0.05, n_samples)
    amps_b = 27.5 + np.random.normal(0, 0.2, n_samples)
    temp_b = 58.5 + np.random.normal(0, 0.3, n_samples)
    vib_b = 1.4 + np.random.normal(0, 0.05, n_samples)  # Steady Zone A/B
    flow_b = 122.0 + np.random.normal(0, 1.0, n_samples)

    for i in range(n_samples):
        records.append({
            "timestamp": timestamps[i],
            "asset_id": "PUMP-102B",
            "asset_class": "pump",
            "motor_amps": round(float(amps_b[i]), 2),
            "suction_pressure_bar": round(float(suction_b[i]), 2),
            "discharge_pressure_bar": round(float(discharge_b[i]), 2),
            "vibration_rms": round(float(vib_b[i]), 2),
            "bearing_temperature": round(float(temp_b[i]), 1),
            "flow_lpm": round(float(flow_b[i]), 1),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} telemetry rows across 2 assets -> {output_path}")
    return df


if __name__ == "__main__":
    generate_telemetry()