"""Run agent-generated Python (pandas/numpy/requests) in a subprocess with a timeout."""
import os
import subprocess
import sys
import tempfile

TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 4000


def run_python(code: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-MAX_OUTPUT_CHARS:],
            "stderr": result.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"Timed out after {TIMEOUT_SECONDS}s"}
    finally:
        os.unlink(path)
