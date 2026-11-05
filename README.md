# Industrial Reliability Hub

A four-layer Streamlit application for industrial telemetry integrity checks, moving-statistics anomaly detection, trend-based remaining useful life (RUL) estimates, and structured AI-assisted engineering reports.

> **Important:** RUL and anomaly outputs are decision-support estimates. They are not confirmed diagnoses, safety alarms, or guaranteed failure dates. Validate findings with site procedures and qualified engineers.

## Architecture

| Layer | Module | Responsibility |
| --- | --- | --- |
| 1. Presentation and routing | `app.py` | Streamlit widgets, input routing, visualization, and passing data between layers. No analytical calculations. |
| 2. Ingestion and integrity | `pipelines.py` | Asset-class sensor aliases and limits, file ingestion, normalization, gap/dropout checks, interpolation, and data-health metrics. |
| 3. Analytical core | `models.py` | Trailing moving mean/std anomaly detection with exact timestamps and linear trend extrapolation to configured thresholds. |
| 4. Reasoning copilot | `copilot.py` | OpenAI client, system prompt, compact context assembly, strict JSON schema validation, and deterministic offline fallback. |

## Quick start

Requires Python 3.10 or later.

```bash
git clone https://github.com/Siyapze/industrial_reliability_hub.git
cd industrial_reliability_hub
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app starts with built-in sample telemetry. Select **Upload CSV or Excel** to analyze your own file.

## Telemetry input

Required columns:

- A timestamp column named `timestamp`, `time`, `datetime`, `event_time`, or `date`.
- An asset identifier named `asset_id`, `asset`, `equipment_id`, or `device_id`.

An optional `asset_class` column supports the built-in `pump` and `motor` configurations. Sensor aliases are normalized, for example `vibration`, `vib_mm_s`, and `vibration_mm_s` map to `vibration_mm_s`; `temperature`, `temp_c`, and `temperature_c` map to `temperature_c`.

Each row should represent one timestamp for one asset. Supported files are CSV and Excel (`.xlsx` / `.xls`). Interior sensor gaps up to three consecutive rows are linearly interpolated by default; leading and trailing gaps are left missing. The data-health report estimates irregular sampling/dropout intervals from each asset's median cadence.

## Configuration and analytical behavior

Edit `ASSET_CONFIG` in `pipelines.py` to add asset classes, canonical sensor names, aliases, and upper operating thresholds. The app uses the configured pump and motor thresholds for trend-based RUL estimates.

Anomaly detection compares a sensor reading with its preceding rolling window, excluding the current value from the baseline. A point is flagged when its absolute deviation reaches the selected number of moving standard deviations. The initial window has no score until enough history is available.

RUL fits a simple least-squares line against elapsed time and extrapolates toward an upper configured threshold. It reports no estimate when the sensor trend is not moving toward that threshold, or when there are too few valid observations. Treat it as a rough trend indicator; use validated degradation models and maintenance data for operational deployment.

## OpenAI copilot (optional)

Without an API key, the app returns a deterministic local report. To enable OpenAI-generated reports, set the key in your environment; it is never stored in the repository:

```bash
export OPENAI_API_KEY="your-key"          # macOS/Linux
export OPENAI_MODEL="gpt-4o-mini"          # optional
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "gpt-4o-mini"
```

The model response is constrained to and validated against a JSON schema containing `summary`, `root_cause_hypotheses`, `recommended_actions`, `evidence`, and `uncertainties`. Provider or response errors use the offline fallback. Avoid sending sensitive or regulated telemetry to an external provider unless authorized by your organization.

## Modules

- `app.py` — Streamlit interface and orchestration.
- `pipelines.py` — ingestion, normalization, interpolation, configuration, and quality report.
- `models.py` — timestamped anomaly events and RUL estimates.
- `copilot.py` — OpenAI integration and structured report validation.
- `requirements.txt` — Python dependencies.

## Limitations

- Sample thresholds are illustrative and must be replaced with asset/vendor/site-specific engineering limits.
- Dropout detection is an estimate based on observed sampling cadence, not a guarantee that every missing interval is found.
- Linear extrapolation assumes a locally linear trend and does not model operating state, load, maintenance interventions, or failure modes.
- Validate input units and sensor calibration before interpreting results.
