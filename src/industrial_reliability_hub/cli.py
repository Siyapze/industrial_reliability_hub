"""Console launcher for installed wheels and editable checkouts."""
from pathlib import Path
import subprocess
import sys


def main() -> int:

    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    return subprocess.call([
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).with_name("streamlit_app.py")),
        "--server.address=127.0.0.1", "--browser.gatherUsageStats=false",
        *sys.argv[1:]
    ])
