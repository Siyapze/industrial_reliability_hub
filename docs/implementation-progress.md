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
