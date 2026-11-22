import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from dotenv import load_dotenv

SHUTDOWN_TIMEOUT_SECONDS = 10
MONITORING_SHUTDOWN_TIMEOUT_SECONDS = 45


def _wait_for_monitoring(
    process: subprocess.Popen,
    proxy_url: str,
    api_key: str,
    timeout: float = 90,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    health_url = f"{proxy_url.rstrip('/')}/health/liveliness"
    request = urllib.request.Request(
        health_url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Monitoring exited with status {process.returncode} before the proxy became ready."
            )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return
                last_error = RuntimeError(
                    f"LiteLLM readiness endpoint returned {response.status}."
                )
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(
        f"LiteLLM proxy did not become ready at {health_url}: {last_error}"
    )


def _stop_monitoring(process: subprocess.Popen, stop_file: Path) -> None:
    try:
        if process.poll() is None:
            stop_file.touch()
            try:
                process.wait(timeout=MONITORING_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                print(
                    "Monitoring did not stop after the shutdown request; terminating it.",
                    file=sys.stderr,
                )
                process.terminate()
                try:
                    process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    finally:
        stop_file.unlink(missing_ok=True)


def main() -> int:
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")

    port = os.getenv("LITELLM_PORT", "4000")
    if not port.isdigit():
        raise ValueError("LITELLM_PORT must be a numeric port.")
    model = os.getenv("MONITORING_MODEL") or os.getenv("LLM_MODEL_NAME", "qwen3:4b")
    api_key = os.getenv("LITELLM_MASTER_KEY") or f"sk-{uuid.uuid4().hex}"
    if not api_key.startswith("sk-"):
        raise ValueError("LITELLM_MASTER_KEY must start with 'sk-'.")

    proxy_url = f"http://127.0.0.1:{port}"
    os.environ["MONITORING_MODEL"] = model
    os.environ["LITELLM_MASTER_KEY"] = api_key
    os.environ["LITELLM_API_BASE"] = proxy_url
    os.environ["MONITORING_DB_PATH"] = str(
        Path(
            os.getenv(
                "MONITORING_DB_PATH",
                str(root / "Monitoring" / "metrics.db"),
            )
        ).expanduser().resolve()
    )

    monitor_stop_file = (
        Path(tempfile.gettempdir()) / f"rag-monitor-stop-{uuid.uuid4().hex}.flag"
    )
    monitoring_process = None
    process = None
    try:
        monitoring_env = os.environ.copy()
        monitoring_env["MONITORING_STOP_FILE"] = str(monitor_stop_file)
        monitoring_process = subprocess.Popen(
            [
                sys.executable,
                str(root / "Monitoring" / "launch_dashboard.py"),
            ],
            cwd=root,
            env=monitoring_env,
        )
        _wait_for_monitoring(monitoring_process, proxy_url, api_key)

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(root / "User_Interface" / "streamlit_ui.py"),
            ],
            cwd=root,
        )
        try:
            return process.wait()
        except KeyboardInterrupt:
            # Ctrl+C is also delivered to the child on a shared console.
            print("Shutting down Streamlit...")
            try:
                return process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                process.terminate()
                try:
                    return process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    process.kill()
                    return process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if monitoring_process is not None:
            _stop_monitoring(monitoring_process, monitor_stop_file)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Application startup failed: {exc}", file=sys.stderr)
        sys.exit(1)