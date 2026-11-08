# Industrial Reliability Hub

An interactive Streamlit workbench for industrial telemetry ingestion, signal-quality checks, transient-aware anomaly detection, rotating-equipment diagnostics, non-linear remaining useful life (RUL) estimates, and evidence-grounded SAP PM work-order recommendations. The local agent uses Ollama through the OpenAI Python client; deterministic diagnostics are provided by `plant_core.py`.

> **Safety note:** Anomaly flags, ISO screening zones, diagnoses, and RUL projections are decision-support only. They are not confirmed diagnoses, safety alarms, or guaranteed failure dates. Confirm units, instrument quality, equipment details, and site procedures before taking action or dispatching work.

## Architecture

| Layer | Module | Responsibility |
| --- | --- | --- |
| Interactive workbench | `app.py` | Streamlit input/filter controls, asset trends, status badges, investigation trace, SAP PM presentation/export, and technician chat. |
| Deterministic analytical core | `plant_core.py` | CSV/Excel and wide/long historian ingestion, aliases, data cleaning, sensor-health audits, operational state/transient detection, anomaly detection, diagnostics, ISO vibration screening, and exponential RUL estimates. |
| Local reasoning agent | `agent.py` | Ollama/OpenAI-compatible client, diagnostic function calling, two-phase investigation and SAP PM JSON generation, fallback report, and facts-grounded technician chat. |
| Standalone verification | `test_agent.py` | CLI pre-flight checks and live local agent-loop/report verification without starting Streamlit. |
| Compatibility modules | `pipelines.py`, `models.py` | Legacy import shims re-exporting the `plant_core` API. New code should import from `plant_core`. |

## Quick start

Requires Python 3.10 or later and a local Ollama installation.

```bash
git clone https://github.com/Siyapze/industrial_reliability_hub.git
cd industrial_reliability_hub
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama serve
```

In another terminal, launch the workbench:

```bash
streamlit run app.py
```

The app starts with built-in sample telemetry. Select **Upload CSV or Excel** to analyze a CSV or Excel historian export. Choose an asset, inspect its trends and anomalies, then select **Run Autonomous Reliability Investigation** to run diagnostics and prepare a work-order recommendation. The agent callback streams tool activity into the investigation status container. If Ollama is unavailable or generation fails, the agent returns a deterministic PM02 inspection recommendation and records that limitation in uncertainties.

### Ollama configuration

`agent.py` loads these optional variables from the environment or a `.env` file in the project directory:

```dotenv
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=llama3.1:8b
```

The default endpoint and key are for local Ollama's OpenAI-compatible API. Change `OPENAI_BASE_URL` when Ollama runs at a different address; set `OPENAI_MODEL` to a locally available model that supports tool calling and JSON responses. The configured API key is sent to that endpoint; the default placeholder `ollama` is not a secret. Do not commit credentials or sensitive telemetry.

## Telemetry input and processing

Telemetry should include a timestamp (`timestamp`, `time`, `datetime`, `event_time`, or `date`) and, for multi-asset inputs, an asset identifier (`asset_id`, `asset`, `equipment_id`, or `device_id`). An optional `asset_class` helps describe equipment. CSV and Excel (`.xlsx` / `.xls`) files are supported.

Both wide-form data (one row per asset and timestamp with sensor columns) and long-form data (timestamp, asset, sensor/tag, and value/reading) are supported. Recognized aliases are normalized, for example `vibration`, `vib_mm_s`, and `vibration_mm_s` map to `vibration_mm_s`; `vibration_rms` is kept distinct. Historian quality strings such as `Bad`, `Calc Error`, and `Scan Off` are treated as missing measurements. Interior gaps up to three consecutive rows are interpolated; leading and trailing gaps are left missing.

`plant_core.audit_sensor_health(df)` reports long unchanged runs, known clipping boundaries, and configured physical range violations. The quality report and sensor-health report are decision aids; review the reported raw values and sensor context.

## Analysis and visualization

The workbench displays asset-level vibration and temperature trends. When the deterministic exponential degradation fit returns a qualifying estimate, it overlays the projection to the Zone D vibration limit. The group-2 ISO 20816-3 Zone C and D bands are illustrative screening references; actual vibration acceptance criteria depend on machine type, support, power, and measurement position. Validate the applicable standard limits for the installed machine.

Operational status is inferred from available load indicators such as motor current, speed, and flow. Startup transitions are marked transient for the initial 15-minute operating window; statistical deviations in that interval are informational rather than alarm cascades. `plant_core.is_steady_state` checks that the asset has remained in running status for at least 15 minutes.

Anomaly detection compares a reading against its preceding rolling window. Positive deviations are emphasized for vibration and temperature signals; standby and transient readings are informational. Hydraulic/mechanical and thermal/vibration sequence diagnostics are evidence-based pattern checks, not definitive failure diagnoses.

`plant_core.compute_degradation_rul` fits an exponential trend to valid operating data and projects operating hours to the supplied critical limit. It returns `-1` when there is insufficient qualifying accelerating trend evidence; do not treat any estimate as a guaranteed time to failure.

## Autonomous investigation and CMMS export

The agent's fact-finding tools call deterministic functions in `plant_core.py` to check operating state and transient conditions, compare hydraulic and mechanical indicators, trace thermal/vibration lead-lag, screen ISO severity, and estimate exponential RUL. The final model call receives the structured diagnostic evidence and required CMMS schema, rather than the raw telemetry table.

The workbench presents the generated SAP PM recommendation, evidence, and uncertainties. Use **Export SAP PM Notification (.json)** for the JSON payload or **Download CMMS Work Package (.txt)** for a technician-oriented work package. Confirm the functional location, SAP codes, priority, craft, and interventions against local master data and site work-control practices before use.

The technician chat uses the selected asset's compact telemetry-quality summary, anomalies, and investigation evidence. It is intended to answer questions from those facts and should state when evidence is unavailable.

## Standalone CLI verification

With Ollama running and the configured model available, run:

```bash
python test_agent.py
```

The script loads `pump_telemetry.csv` when it contains the diagnostic tags needed for the scenario, or synthesizes a deterministic 100-row running-pump dataset with suction pressure drop, rising vibration, and steady motor current. It runs `audit_sensor_health` and `detect_sensor_anomalies`, invokes the agent with terminal tool-call logging, validates the report fields, checks for at least two distinct diagnostic tools, and prints a colorized summary and elapsed time. It warns when runtime exceeds 25 seconds. The check is intentionally a live local-model verification, not a mocked unit test; tool-calling support is required for the tool-count assertion.

## Modules and direct use

- `plant_core.py` — import ingestion, analytical, health-audit, and diagnostic functions here.
- `agent.py` — `run_reliability_investigation(...)` and `chat_with_agent(...)` local-agent entry points.
- `app.py` — interactive Streamlit workbench.
- `test_agent.py` — standalone Ollama CLI verification.
- `pipelines.py`, `models.py` — compatibility imports for older callers.
- `requirements.txt` — Python dependencies, including the OpenAI client and `python-dotenv`.

## Limitations

- Sensor aliases, range boundaries, operating thresholds, and ISO vibration screening values are illustrative; validate them for the actual equipment, installation, and site.
- Flatline, clipping, and dropout checks can identify suspicious signal quality but cannot verify instrument calibration or historian configuration.
- Operational state depends on available load indicators and fixed inferred thresholds; missing or atypical tags may result in `UNKNOWN` or inaccurate classification.
- Model output and tool selection can vary by Ollama model/version. Review all generated diagnoses, uncertainties, and SAP PM fields before operational use.
- Exponential trend fits are sensitive to data quality and operating regime and do not account for maintenance interventions or every failure mode.
