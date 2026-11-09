"""Exercise installed entry points and compatibility during migration."""
import os
from pathlib import Path
import subprocess
import sys


def test_installed_imports_outside_checkout(tmp_path):
    result = subprocess.run([
        sys.executable, '-c',
        'from industrial_reliability_hub.core import plant_core; '
        'from industrial_reliability_hub.ui import application; '
        'assert callable(application.main); '
        'assert plant_core.evaluate_iso_vibration_zone(2)["zone"] == "B"',
    ], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'missing ScriptRunContext' not in result.stderr


def test_console_launcher_help_outside_checkout(tmp_path):
  executable = Path(sys.executable).parent / (
      "reliability-hub.exe" if os.name == "nt" else "reliability-hub"
  )
  env = os.environ.copy()
  env["PYTHONIOENCODING"] = "utf-8"
  env["PYTHONUTF8"] = "1"

  result = subprocess.run(
      [str(executable), "--help"],
      cwd=tmp_path,
      capture_output=True,
      text=True,
      encoding="utf-8",
      env=env,
      timeout=30,
  )
  assert result.returncode == 0, result.stderr


def test_legacy_imports_share_implementation():
    import agent
    import plant_core
    import serialization
    import pipelines
    import models
    from industrial_reliability_hub import agent as canonical_agent
    from industrial_reliability_hub.core import plant_core as canonical_core
    from industrial_reliability_hub import serialization as canonical_serialization
    assert agent is canonical_agent
    assert plant_core is canonical_core
    assert serialization is canonical_serialization
    assert pipelines.detect_sensor_anomalies is canonical_core.detect_sensor_anomalies
    assert models.compute_degradation_rul is canonical_core.compute_degradation_rul
