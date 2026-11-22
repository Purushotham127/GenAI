# Standalone Ollama monitoring

The monitoring implementation remains isolated in this folder. Running the project entry point starts it as a separate process alongside the RAG app:

```text
RAG app ──────────────┐
                      ├──> LiteLLM proxy ───> local Ollama
Metrics UI ───────────┘           ├─────────> Langfuse (optional)
                                  └─────────> metrics.db
```

The proxy exposes LiteLLM Prometheus metrics; the Streamlit page reads those metrics from the proxy process. When Langfuse keys are configured, LiteLLM also sends request traces to Langfuse, whose UI provides persistent per-request inspection.

Successful and failed requests are also persisted locally in `Monitoring\metrics.db`. The dashboard displays request rows newest-first and supports showing the latest 50, 100, 500, 1,000, or all retained rows. The source column distinguishes RAG app requests from dashboard tests. Full model input messages and generated output are stored in the database for tracing but are deliberately excluded from the UI table.

## Run on Windows

Use the project venv or another Python environment, then install only the monitoring dependencies:

```powershell
.\rag_venv\Scripts\python.exe -m pip install -r Monitoring\requirements_monitoring.txt
```

Add the desired settings from `Monitoring\.env.example` to the existing project-root `.env` (do not replace its RAG settings). The default model is `qwen3:4b`; it must already be present in Ollama.

Start Ollama, then launch the RAG app and monitor together:

```powershell
.\rag_venv\Scripts\python.exe main.py
```

The RAG UI is at <http://127.0.0.1:8501>; the metrics UI is at <http://127.0.0.1:8502>. Stopping the RAG server from its terminal also stops the monitor.

To run only the monitoring stack, use:

```powershell
.\rag_venv\Scripts\python.exe Monitoring\launch_dashboard.py
```

Open the UI at <http://127.0.0.1:8502>. Use **Send streaming test request** to exercise Ollama through LiteLLM and populate TTFT metrics. The LiteLLM API listens only on `127.0.0.1:4000`. Press Ctrl+C in the terminal to stop both services.

The proxy generates a random local master key when `LITELLM_MASTER_KEY` is not set. Its authenticated API key is passed only to its child processes and is not printed. To use Langfuse, set both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in the existing `.env`; `LANGFUSE_HOST` is optional and defaults to Langfuse Cloud.

## Metrics

- TTFT: LiteLLM's streaming TTFT histogram (shown only after a streaming request).
- Output tokens/second: aggregate output tokens divided by model time after first token; this is meaningful for streaming calls.
- Model and end-to-end request latency histograms.
- Request failures and input/output token counters.

Prometheus values are cumulative for the lifetime of the LiteLLM process. SQLite retains per-request metrics locally; Langfuse traces are also sent when configured.

SQLite request history persists across proxy and dashboard restarts. Set `MONITORING_DB_PATH` in `.env` to use a different database file.

## RAG application requests

When started with `main.py`, the RAG app's chat model uses LiteLLM's OpenAI-compatible local endpoint, which routes requests to the configured Ollama model. App requests and dashboard test requests appear together in Prometheus metrics and `metrics.db`. The chat client streams internally so LiteLLM can record TTFT; the app still displays the completed answer as before. Embedding, retrieval, and reranking operations do not pass through LiteLLM and are not included.

## Tests

```powershell
.\rag_venv\Scripts\python.exe -m unittest Monitoring.test_monitoring
```
