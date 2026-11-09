"""Run against a running local app: python scripts/smoke_browser.py --port 8501.

Requires pip install -e ".[browser]" and playwright install chromium.
Uses deterministic offline fallback; start the app with Ollama unavailable.
"""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--chromium', help='Optional existing Chromium executable')
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=args.chromium)
        try:
            page = browser.new_page()
            page.goto(f'http://127.0.0.1:{args.port}')
            page.get_by_role('heading', name='Industrial Reliability Workbench', exact=True).wait_for(timeout=30000)
            page.get_by_text('Upload CSV or Excel', exact=True).click()
            page.locator('input[type=file]').set_input_files(str(Path(__file__).resolve().parents[1] / 'pump_telemetry.csv'))
            page.get_by_text('2016', exact=True).first.wait_for(timeout=30000)
            page.get_by_role('button', name='Run Autonomous Reliability Investigation', exact=True).click()
            page.get_by_text('SAP PM work-order recommendation — review before dispatch', exact=True).wait_for(timeout=40000)
            with page.expect_download() as download:
                page.get_by_role('button', name='Export SAP PM Notification (.json)').click()
            report = json.loads(Path(download.value.path()).read_text(encoding='utf-8'))
            assert report['iso_severity_zone'] == 'D'
            assert report['investigation_evidence']['synthesis']['status'] == 'DETERMINISTIC_FALLBACK'
            assert page.get_by_text('AI synthesis unavailable — this is a deterministic inspection report.', exact=False).count() == 1
            assert report['investigation_evidence']['asset_id'] == 'Uploaded asset'
            assert page.locator('[data-testid="stException"]').count() == 0
            print('PASS: startup, CSV upload, offline investigation, JSON export')
        finally:
            browser.close()


if __name__ == '__main__':
    main()
