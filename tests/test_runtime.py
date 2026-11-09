import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from jsonschema import ValidationError
from streamlit.testing.v1 import AppTest

from industrial_reliability_hub import agent
from industrial_reliability_hub.core import plant_core as core
from industrial_reliability_hub.serialization import json_safe
from test_agent import _synthetic_telemetry

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_csv_path():
    data = core.clean_historian_telemetry(ROOT / 'pump_telemetry.csv')
    assert len(data) == 2016
    assert data.timestamp.is_monotonic_increasing


def test_long_form_does_not_create_cross_asset_rows():
    data = pd.DataFrame({'timestamp': ['2025-01-01', '2025-01-02'],
                         'asset_id': ['A', 'B'], 'asset_class': ['pump', 'motor'],
                         'sensor': ['vibration', 'vibration'], 'value': [2., 3.]})
    result = core.clean_historian_telemetry(data)
    assert len(result) == 2
    assert list(result.asset_id) == ['A', 'B']


def test_excel_and_invalid_input():
    buffer = io.BytesIO()
    pd.DataFrame({'timestamp': ['2025-01-01'], 'vibration': [2.]}).to_excel(buffer, index=False)
    assert core.clean_historian_telemetry(buffer.getvalue()).vibration_mm_s.iloc[0] == 2.
    with pytest.raises(ValueError, match='timestamp'):
        core.clean_historian_telemetry(pd.DataFrame({'reading': [1]}))


def test_missing_data_and_offline_samples_are_excluded_from_rul():
    df = _synthetic_telemetry()
    df.loc[5:15, 'vibration_rms'] = np.nan
    df.loc[20:30, 'motor_amps'] = 0
    df.loc[20:30, 'flow_lpm'] = 0
    df.loc[20:30, 'vibration_rms'] = 80
    valid = df.vibration_rms.notna() & df.motor_amps.gt(0)
    assert core.compute_degradation_rul(df, 'PUMP-101A', 'vibration_rms', 10) == pytest.approx(
        core.compute_degradation_rul(df.loc[valid], 'PUMP-101A', 'vibration_rms', 10))


def test_transient_detection_and_small_anomaly_window():
    df = _synthetic_telemetry().iloc[:2]
    assert agent._run_local_tool('check_asset_operating_state', {}, df, [], 'PUMP-101A')['transient']
    core.detect_sensor_anomalies(_synthetic_telemetry(), window=2)


def test_offline_report_retains_evidence_and_real_severity():
    df = _synthetic_telemetry()
    anomalies = core.detect_sensor_anomalies(df)
    with patch.object(agent.client.chat.completions, 'create', side_effect=ConnectionError):
        report = agent.run_reliability_investigation(df, anomalies, 'PUMP-101A')
    assert report['iso_severity_zone'] == 'D'
    assert report['investigation_evidence']['check_asset_operating_state']['operating_state'] == 'RUNNING'
    agent._validate_sap_report({k: v for k, v in report.items() if k != 'investigation_evidence'})
    json.dumps(json_safe(report), allow_nan=False)


def test_schema_rejects_invalid_model_field_types():
    report = agent._offline_fallback_report('A', [])
    report['recommended_sap_work_order']['required_interventions'] = [42]
    with pytest.raises(ValidationError):
        agent._validate_sap_report(report)


def test_nonfinite_json_values():
    assert json_safe({'x': [np.inf, np.nan, -np.inf, np.int64(2), pd.NaT]}) == {'x': [None, None, None, 2, None]}


def test_streamlit_start_filters_investigation_and_state_invalidation():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == '3'
    app.sidebar.selectbox[1].select('Pump-B').run()
    assert not app.exception
    with patch.object(agent.client.chat.completions, 'create', side_effect=ConnectionError):
        app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.session_state['reliability_report_asset'] == 'Pump-B'
    app.sidebar.slider[0].set_value(4.0).run()
    assert not app.exception
    assert 'reliability_report' not in app.session_state


def test_model_tool_loop_and_report_generation():
    from types import SimpleNamespace
    class ToolCall:
        id = 'call-1'
        function = SimpleNamespace(name='check_asset_operating_state', arguments='{}')
        def model_dump(self, **kwargs):
            return {'id': self.id, 'type': 'function', 'function': vars(self.function)}
    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[ToolCall()]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Evidence collected', tool_calls=[]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(agent._offline_fallback_report('PUMP-101A', []))))]),
    ]
    steps = []
    with patch.object(agent.client.chat.completions, 'create', side_effect=responses) as create:
        report = agent.run_reliability_investigation(_synthetic_telemetry(), [], 'PUMP-101A', callback=lambda name, detail: steps.append(name))
    assert 'offline_fallback' not in steps
    assert 'check_asset_operating_state' in steps
    assert create.call_count == 3
    assert report['investigation_evidence']['check_asset_operating_state']['steady_state'] is True
