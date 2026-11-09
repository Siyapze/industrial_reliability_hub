"""Console launcher for installed wheels and editable checkouts."""
from pathlib import Path
import subprocess
import sys


def main() -> int:
    return subprocess.call([
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).with_name("streamlit_app.py")),
        "--server.address=127.0.0.1", "--browser.gatherUsageStats=false",
        *sys.argv[1:]
    ])
