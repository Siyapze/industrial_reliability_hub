"""Generate seven days of simulated pump telemetry at five-minute intervals.

The first five days model normal operation. The final two days model a
fluid-ingress event followed by progressive bearing degradation.
"""

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_FILE = Path("pump_telemetry.csv")
START_TIME = "2025-01-01 00:00:00"
INTERVAL = "5min"
DAYS = 7
NORMAL_DAYS = 5
SAMPLES_PER_DAY = 24 * 60 // 5
RANDOM_SEED = 42


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    timestamps = pd.date_range(
        start=START_TIME,
        periods=DAYS * SAMPLES_PER_DAY,
        freq=INTERVAL,
    )
    failure_start = NORMAL_DAYS * SAMPLES_PER_DAY
    sample_index = np.arange(len(timestamps))
    failure_samples = len(timestamps) - failure_start

    # Normal operation: small sensor noise around the specified baselines.
    vibration_rms = 2.1 + rng.normal(0.0, 0.08, len(timestamps))
    bearing_temperature = 65.0 + rng.normal(0.0, 0.4, len(timestamps))
    ambient_humidity = 45.0 + rng.normal(0.0, 1.0, len(timestamps))

    # At the start of day 6, humidity steps up to indicate fluid ingress.
    ambient_humidity[failure_start:] = 98.0 + rng.normal(
        0.0, 0.3, failure_samples
    )

    # Model grease wash-out: a rapid temperature drop at event onset, then
    # a gradual climb over the two-day failure period.
    failure_progress = np.linspace(0.0, 1.0, failure_samples)
    bearing_temperature[failure_start:] = (
        53.0 + 22.0 * failure_progress + rng.normal(0.0, 0.5, failure_samples)
    )

    # Exponential rise from the normal vibration level toward 12.5 mm/s.
    # The normalized curve starts at 2.1 and reaches 12.5 at the final sample.
    curvature = 4.0
    exponential_progress = (
        np.expm1(curvature * failure_progress) / np.expm1(curvature)
    )
    vibration_rms[failure_start:] = (
        2.1
        + (12.5 - 2.1) * exponential_progress
        + rng.normal(0.0, 0.12, failure_samples)
    )
    vibration_rms[-1] = 12.5  # Preserve the requested terminal severity.

    telemetry = pd.DataFrame(
        {
            "timestamp": timestamps,
            "vibration_rms": vibration_rms,
            "bearing_temperature": bearing_temperature,
            "ambient_humidity": ambient_humidity,
        }
    )
    telemetry.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(telemetry)} rows to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
