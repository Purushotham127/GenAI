"""Start Ollama's LiteLLM monitoring proxy and its separate Streamlit UI."""

import argparse
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests
import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
MONITORING_DIR = Path(__file__).resolve().parent
OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"


def _validate_ollama(base_url: str, model: str) -> None:
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
    response.raise_for_status()
    installed_models = {
        entry["name"] for entry in response.json().get("models", [])
    }
    if model not in installed_models:
        available = ", ".join(sorted(installed_models)) or "(none)"
        raise RuntimeError(
            f"Ollama model {model!r} is not installed at {base_url}. "
            f"Available models: {available}. Set MONITORING_MODEL to an installed tag."
        )


def _make_runtime_config(
    model: str,
    ollama_url: str,
    langfuse_enabled: bool,
) -> Path:
    with (MONITORING_DIR / "litellm_config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["model_list"][0]["model_name"] = model
    config["model_list"][0]["litellm_params"]["model"] = f"ollama/{model}"
    config["model_list"][0]["litellm_params"]["api_base"] = ollama_url

    if langfuse_enabled:
        config["litellm_settings"]["success_callback"] = ["langfuse"]
        config["litellm_settings"]["failure_callback"] = ["langfuse"]

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        prefix="litellm-monitoring-",
        delete=False,
    ) as file:
        yaml.safe_dump(config, file, sort_keys=False)
        return Path(file.name)


def _wait_for_service(
    process: subprocess.Popen,
    url: str,
    api_key: str,
    timeout: float,
    stop_file: Optional[Path] = None,
) -> None:
    deadline = time.monotonic() + timeout
    headers = {"Authorization": f"Bearer {api_key}"}
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        if stop_file is not None and stop_file.exists():
            raise KeyboardInterrupt
        if process.poll() is not None:
            raise RuntimeError(
                f"Service exited with status {process.returncode} before becoming ready."
            )
        try:
            response = requests.get(url, headers=headers, timeout=2)
            if response.ok:
                return
            last_error = RuntimeError(f"Readiness endpoint returned {response.status_code}.")
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"Service did not become ready at {url}: {last_error}")


def _terminate(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-only", action="store_true", help="Run the proxy without Streamlit.")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    load_dotenv(ROOT_DIR / ".env")

    model = os.getenv(
        "MONITORING_MODEL",
        os.getenv("LLM_MODEL_NAME", "qwen3:4b"),
    ).strip()
    ollama_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_URL).rstrip("/")
    port = os.getenv("LITELLM_PORT", "4000")
    ui_port = os.getenv("MONITORING_UI_PORT", "8502")
    if not port.isdigit() or not ui_port.isdigit():
        raise ValueError("LITELLM_PORT and MONITORING_UI_PORT must be numeric ports.")

    api_key = os.getenv("LITELLM_MASTER_KEY") or f"sk-{secrets.token_urlsafe(32)}"
    if not api_key.startswith("sk-"):
        raise ValueError("LITELLM_MASTER_KEY must start with 'sk-'.")
    os.environ["LITELLM_MASTER_KEY"] = api_key
    os.environ["MONITORING_MODEL"] = model
    os.environ["OLLAMA_BASE_URL"] = ollama_url
    os.environ["LITELLM_API_BASE"] = f"http://127.0.0.1:{port}"
    database_path = Path(
        os.getenv(
            "MONITORING_DB_PATH",
            str(MONITORING_DIR / "metrics.db"),
        )
    ).expanduser().resolve()
    os.environ["MONITORING_DB_PATH"] = str(database_path)
    stop_file_value = os.getenv("MONITORING_STOP_FILE")
    stop_file = Path(stop_file_value) if stop_file_value else None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if bool(public_key) != bool(secret_key):
        raise ValueError("Set both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, or neither.")
    langfuse_enabled = bool(public_key and secret_key)
    if langfuse_enabled:
        langfuse_host = (
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
            or "https://cloud.langfuse.com"
        )
        os.environ["LANGFUSE_HOST"] = langfuse_host
        os.environ["MONITORING_LANGFUSE_HOST"] = langfuse_host
    else:
        os.environ.pop("MONITORING_LANGFUSE_HOST", None)

    _validate_ollama(ollama_url, model)

    runtime_config = _make_runtime_config(model, ollama_url, langfuse_enabled)
    proxy = None
    dashboard = None
    try:
        proxy = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from litellm_cli.main import proxy_server; proxy_server()",
                "--config",
                str(runtime_config),
                "--host",
                "127.0.0.1",
                "--port",
                port,
            ],
            cwd=ROOT_DIR,
            env=os.environ.copy(),
        )
        proxy_url = f"http://127.0.0.1:{port}"
        print(f"Starting LiteLLM proxy for Ollama model {model} at {proxy_url}")
        _wait_for_service(
            proxy,
            f"{proxy_url}/health/liveliness",
            api_key,
            timeout=90,
            stop_file=stop_file,
        )

        if not args.proxy_only:
            dashboard = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(MONITORING_DIR / "metrics_dashboard.py"),
                    "--server.port",
                    ui_port,
                    "--server.address",
                    "127.0.0.1",
                ],
                cwd=ROOT_DIR,
                env=os.environ.copy(),
            )
            ui_url = f"http://127.0.0.1:{ui_port}"
            _wait_for_service(
                dashboard,
                f"{ui_url}/_stcore/health",
                api_key,
                timeout=60,
                stop_file=stop_file,
            )
            print(f"Metrics dashboard ready: {ui_url}")

        if langfuse_enabled:
            print(f"Langfuse tracing enabled: {os.environ['LANGFUSE_HOST']}")
        else:
            print("Langfuse tracing disabled; add both Langfuse keys to the existing .env to enable it.")
        print("Press Ctrl+C to stop.")

        if args.proxy_only:
            while proxy.poll() is None:
                if stop_file is not None and stop_file.exists():
                    return 0
                time.sleep(1)
            return proxy.returncode or 0

        while proxy.poll() is None and dashboard.poll() is None:
            if stop_file is not None and stop_file.exists():
                return 0
            time.sleep(1)
        return dashboard.returncode or 0
    except KeyboardInterrupt:
        return 0
    finally:
        _terminate(dashboard)
        _terminate(proxy)
        runtime_config.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, requests.RequestException, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Monitoring startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
