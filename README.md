# Industrial Reliability Hub

A Streamlit workbench for industrial telemetry ingestion, signal-quality checks,
statistical anomalies, illustrative vibration screening, trend forecasts, and
reviewable maintenance recommendations. Local Ollama provides optional narrative
investigation and technician chat.

**Development status:** `agenticflow` implements roadmap Steps 1–2: a verified
baseline and Python packaging. The v3.1 profile rules, ETTT contracts, LangGraph,
human approval workflow, and offline deployment controls are future work. Current
outputs are decision support, not diagnoses, protection alarms, approved SAP
orders, or standards certification.

## Windows / VS Code quick start

Requires Python **3.11 or 3.12**. In a PowerShell terminal:

```powershell
git clone --branch agenticflow https://github.com/Siyapze/industrial_reliability_hub.git
cd industrial_reliability_hub
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Replace `-3.11` with `-3.12` if needed. These commands avoid PowerShell activation
policy changes. In VS Code, select `.venv\Scripts\python.exe` via **Python: Select
Interpreter**. If activation is permitted, run:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

For an existing clean clone, use `git fetch origin` then `git switch agenticflow`
before installation. Preserve any local edits before switching branches.

Expect a dashboard with 3 sample assets and 1,440 records. The bundled
`pump_telemetry.csv` is a separate synthetic upload fixture with 2,016 records;
uploads without IDs use `Uploaded asset`. Uploads, filters, plots, and offline
inspection reports work without Ollama. Settings/data changes invalidate prior
investigation and chat state. The UI does not submit anything to SAP.

## macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
streamlit run app.py
```

`pip install -r requirements.txt` remains supported and installs this editable
package. Runtime dependencies are defined only in `pyproject.toml`. Build a wheel
with `python -m build`. An installed wheel also provides `reliability-hub` to
launch its bundled UI from outside a checkout; `reliability-hub --help` shows
Streamlit options. This is not yet an offline wheelhouse or locked deployment.

## Optional local Ollama (16 GB RAM)

A 4,096-token context is a reasonable starting point for a quantized 8B model on
16 GB RAM, but available memory and latency must be measured on the target
machine. CPU inference is supported and may be slow. The context budget includes
system instructions, schemas, tool history, evidence, and output. A <200-token
individual payload is a target for later evidence contracts, **not enforced by
this baseline** and not a guarantee against context overflow.

The `llama3.1:8b-lowctx` name is a local custom tag, not a model assumed to exist:

```powershell
ollama pull llama3.1:8b
ollama create llama3.1:8b-lowctx -f config/ollama/llama3.1-lowctx.Modelfile
Copy-Item .env.example .env
```

Do not overwrite an existing `.env`; edit it to match the example instead. Start
Ollama if its desktop service is not already running. The example selects the
local endpoint, placeholder key `ollama`, and custom tag. Without `.env`, the
legacy default remains `llama3.1:8b` at localhost. Calls currently have a 5-second
connection timeout and a 120-second read timeout per request, without automatic
retries. Configure these in `.env` and restart Streamlit:

```dotenv
OLLAMA_CONNECT_TIMEOUT_SECONDS=5
OLLAMA_READ_TIMEOUT_SECONDS=120
```

On a slow CPU you can try `180` for the read timeout. This is a per-request
network wait, not a total investigation deadline; the existing agent can issue
up to five sequential requests. It does not guarantee the model fits its context
or that inference will succeed. A slow/cold model can still fall back to a
clearly marked deterministic inspection report. The UI and exported evidence
distinguish a model-generated draft from fallback, including the failure type. Live model behavior has not been verified
on the target Windows laptop. Run `python test_agent.py` for the optional live
model check (requires at least two model-selected diagnostic tools).

`.env` is ignored; never commit real credentials or proprietary telemetry.
Streamlit binds to loopback and usage statistics are disabled in
`.streamlit/config.toml`. These settings alone do not establish air-gap security;
the current agent still permits configurable model endpoints.

## Package layout and migration

```text
app.py                                  # streamlit run app.py launcher
pyproject.toml                          # package/dependencies/console launcher
src/industrial_reliability_hub/
    agent.py                            # existing local agent; graph comes later
    serialization.py                    # strict JSON evidence conversion
    cli.py / streamlit_app.py            # installed wheel launcher
    core/plant_core.py                   # existing analytical core
    ui/application.py                   # presentation helpers and main()
tests/                                  # offline runtime/packaging checks
scripts/smoke_browser.py                 # optional running-app browser check
config/ollama/llama3.1-lowctx.Modelfile   # optional local model definition
```

Import new code from `industrial_reliability_hub`. Root `agent.py`,
`plant_core.py`, and `serialization.py` are transitional import shims;
`pipelines.py` and `models.py` retain their older public imports. The unused
cloud-backed `copilot.py` has been removed; callers should migrate to the local
agent API (the old report payloads are not interchangeable). Core module splits,
validated schemas, and graph modules will be introduced as their steps are
implemented, rather than adding empty placeholders.

## Troubleshooting local inference

`ollama ps` showing `100% CPU` means inference uses the CPU. A brief `READY`
response proves availability, but not that a multi-call investigation will finish
quickly. If the trace shows a tool call followed by `APITimeoutError`, at least
one request succeeded and a later request exceeded its timeout. Warm the model
with `ollama run llama3.1:8b-lowctx "Reply with READY only."`, then rerun the
investigation. Fallback reports remain unconfirmed decision support.

Streamlit 1.51 or newer is required for the current `width` UI arguments; the
legacy `use_container_width` options have been removed.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m build
```

The suite tests ingestion, missing-data filtering, startup detection, strict JSON,
report schema checks, mocked tool orchestration, UI state, installed imports, and
console launch. GitHub Actions is configured for Windows/Linux with Python
3.11/3.12; adding the workflow does not mean those hosted runs have passed.

For an optional browser check, install `.[browser]`, run
`python -m playwright install chromium`, start Streamlit with Ollama unavailable,
and run `python scripts/smoke_browser.py`. Test outputs and virtual environments
are ignored and excluded from the package.

## Current limitations and next steps

- ISO bands, operating thresholds, and SAP codes remain illustrative; no approved
  plant configuration or human approval mechanism exists yet.
- Existing trend forecasts retain legacy RUL naming and fitting rules; validated
  ETTT math and gates arrive in Step 6.
- Cleaning still interpolates per pass; strict idempotence arrives in Step 4.
- Model-selected hypotheses and work packages require engineering review.
- All bundled samples are synthetic, with no claimed Arkema plant provenance.
  Future chemical-plant demonstrations will use fictional assets and operating
  scenarios, not inferred private site measurements.

See [implementation progress](docs/implementation-progress.md) for per-step checks.
