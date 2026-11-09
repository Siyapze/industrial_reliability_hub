# Agenticflow implementation progress

## Step 1 — Branch and verified baseline

- Branch: `agenticflow`, starting from `08098d2` on the existing checkout.
- Retained the earlier runtime fixes, strict JSON helper, regression tests, and documentation.
- Generated CSV outputs, screenshots, virtual environments, and caches are not added.
- Validation on Linux / Python 3.12.14: 10 tests passed; `pip check` passed;
  Chromium verified startup, bundled CSV upload, offline investigation, and JSON export.
- Live Ollama and Windows execution were not tested. These checks do not certify ISO/SAP compliance.

### Running Step 1 on Windows (PowerShell)

From the cloned repository on branch `agenticflow`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Python 3.12 can be used by replacing `-3.11` with `-3.12`.
Expect the existing sample dashboard, asset filtering, upload, and offline inspection report.
Ollama is not needed for those checks; technician chat and live model investigation require it.

## Step 2 — Installable package and retained Streamlit launcher

- Runtime code now lives under `src/industrial_reliability_hub`; root `app.py`
  remains the supported Streamlit launcher.
- `pyproject.toml` declares Python 3.11/3.12, runtime dependencies, dev/browser
  extras, package discovery, and the callable `reliability-hub` console entry.
- Kept small root import shims for existing callers. Removed unused cloud-backed
  `copilot.py`; it was not imported by the current UI or test suite.
- Added an explicit source manifest, loopback/usage-reporting defaults, a reusable
  browser smoke script, local Ollama Modelfile, and a non-secret `.env.example`.
- No empty future graph/schema packages, proprietary plant data, generated outputs,
  screenshots, virtual environments, or build artifacts are committed.
- Validation: 13 offline tests passed against both editable and wheel installs;
  `pip check` and source/wheel builds passed. Chromium verified upload, offline
  investigation, and JSON export through the root launcher and installed wheel.
- The wheel was installed in the existing validated dependency environment;
  this is not a fresh offline machine restoration test. Actual checks ran on
  Linux/Python 3.12.14. Windows and Python 3.11 CI are configured, not claimed passed.
- Live Ollama, <200-token evidence enforcement, approved profiles, and synthetic
  chemical-plant scenario development remain for subsequent roadmap steps.

### Running Step 2 on Windows (PowerShell)

From your clean clone, fetch and switch to `agenticflow`. For a fresh clone:

```powershell
git clone --branch agenticflow https://github.com/Siyapze/industrial_reliability_hub.git
cd industrial_reliability_hub
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Choose `.venv\Scripts\python.exe` as the VS Code interpreter. After optional
activation (`.\.venv\Scripts\Activate.ps1`), `streamlit run app.py` works directly.
Expect the same dashboard and fallback capabilities as Step 1, now backed by the
installed package. No LangGraph or approved profile behavior is introduced yet.
See README for optional `llama3.1:8b-lowctx` creation and context-budget caveats.

## Step 2 follow-up — CPU inference timeout and UI status

The Windows printout showed a successful model tool call followed by
`APITimeoutError`; CLI checks showed a 4096-token model using 100% CPU.

- Replaced the fixed 20-second timeout with configurable connection (5 seconds)
  and read (120 seconds) timeouts; automatic retries remain disabled.
- These are per-request network timeouts, not a total investigation deadline.
- Replaced deprecated Streamlit width arguments and raised its minimum to 1.51.
- Reports retain explicit MODEL_GENERATED / DETERMINISTIC_FALLBACK provenance in
  exported evidence, with fallback status visible in the UI and text package.
- Validation on Linux/Python 3.12: 20 tests passed, including a tool-call-then-
  timeout regression; pip check passed. Chromium verified upload, fallback
  status, and JSON export. No width deprecation messages appeared during that run.
- Live multi-call inference on the Windows laptop is still to be rechecked.
  This patch does not enforce the future evidence token budget or start Step 3.

On the laptop, stop Streamlit with Ctrl+C, then run:

```powershell
git switch agenticflow
git pull --ff-only
python -m pip install -e ".[dev]"
python -m pytest -q
streamlit run app.py
```

The new timeouts apply by default. Optionally edit the existing `.env` to set
`OLLAMA_CONNECT_TIMEOUT_SECONDS=5` and `OLLAMA_READ_TIMEOUT_SECONDS=120` (or 180
for a slower CPU); restart Streamlit after changing them. Do not overwrite `.env`.
