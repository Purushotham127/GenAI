"""Standalone Streamlit dashboard for the LiteLLM proxy's live metrics."""

import json
import os
import time
from typing import Iterator

import requests
import pandas as pd
import streamlit as st

from Monitoring.llm_monitor import MetricsSnapshot, fetch_metrics
from Monitoring.sqlite_metrics import count_requests, fetch_request_rows


st.set_page_config(page_title="Ollama Metrics", page_icon="📊", layout="wide")
st.title("Ollama model metrics")
st.caption("Metrics include RAG application queries routed through LiteLLM and dashboard test requests.")

proxy_url = os.getenv("LITELLM_API_BASE", "http://127.0.0.1:4000").rstrip("/")
api_key = os.getenv("LITELLM_MASTER_KEY", "")
model = os.getenv("MONITORING_MODEL", "qwen3:4b")
langfuse_host = os.getenv("MONITORING_LANGFUSE_HOST")


def stream_test_completion(prompt: str) -> Iterator[str]:
    """Send a streaming test request through the proxy."""
    response = requests.post(
        f"{proxy_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Request-Source": "dashboard-test",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 128,
        },
        stream=True,
        timeout=(5, 600),
    )
    response.raise_for_status()
    response.encoding = "utf-8"
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if isinstance(content, str) and content:
                    yield content
    finally:
        response.close()


with st.sidebar:
    st.subheader("Proxy")
    st.code(proxy_url)
    st.caption(f"Model: `{model}`")
    if langfuse_host:
        st.link_button("Open Langfuse", langfuse_host)
    prompt = st.text_area("Test prompt", "Reply with a short greeting.")
    send_test = st.button("Send streaming test request", type="primary")
    row_limit = st.selectbox(
        "Request rows",
        options=[50, 100, 500, 1000, "All"],
        index=2,
    )
    auto_refresh = st.checkbox("Refresh automatically", value=True)
    refresh_seconds = st.slider("Refresh interval (seconds)", 3, 30, 5)

if send_test:
    if not prompt.strip():
        st.warning("Enter a prompt before sending a test request.")
    else:
        st.subheader("Test response")
        try:
            st.write_stream(stream_test_completion(prompt.strip()))
        except requests.RequestException as exc:
            st.error(f"Test request to LiteLLM failed: {exc}")

try:
    snapshot: MetricsSnapshot = fetch_metrics(proxy_url, api_key)
except requests.RequestException as exc:
    st.error(
        f"Cannot read LiteLLM metrics at {proxy_url}/metrics. "
        f"Start the monitoring stack and check the proxy logs. Details: {exc}"
    )
    st.stop()

st.subheader("Since proxy startup")
columns = st.columns(4)
columns[0].metric("Proxy requests", f"{snapshot.requests:,.0f}")
columns[1].metric("Failed requests", f"{snapshot.failed_requests:,.0f}")
columns[2].metric("Input tokens", f"{snapshot.input_tokens:,.0f}")
columns[3].metric("Output tokens", f"{snapshot.output_tokens:,.0f}")

columns = st.columns(4)
columns[0].metric(
    "Average TTFT",
    f"{snapshot.average_ttft_seconds:.3f} s"
    if snapshot.average_ttft_seconds is not None
    else "—",
    help="LiteLLM reports TTFT for streaming requests.",
)
columns[1].metric(
    "Output tokens / second",
    f"{snapshot.output_tokens_per_second:.1f}"
    if snapshot.output_tokens_per_second is not None
    else "—",
    help="Aggregate output tokens divided by model time after first token; meaningful for streaming requests.",
)
columns[2].metric(
    "Average model latency",
    f"{snapshot.average_llm_latency_seconds:.3f} s"
    if snapshot.average_llm_latency_seconds is not None
    else "—",
)
columns[3].metric(
    "Average request latency",
    f"{snapshot.average_request_latency_seconds:.3f} s"
    if snapshot.average_request_latency_seconds is not None
    else "—",
)

if snapshot.requests == 0:
    st.info("No requests have passed through LiteLLM yet. Ask a question in the RAG app or use the test button.")
if snapshot.average_ttft_seconds is None:
    st.caption("TTFT appears after at least one successful streaming request.")
st.caption(
    "LiteLLM metrics are cumulative for the lifetime of the proxy process. "
    "When Langfuse credentials are configured, individual request traces are also sent to Langfuse."
)

st.subheader("Per-request history")
total_rows = count_requests()
limit = None if row_limit == "All" else int(row_limit)
request_rows = fetch_request_rows(limit=limit)
st.caption(
    f"Retained in metrics.db: {total_rows:,} requests. "
    f"Showing {len(request_rows):,}, newest first. Source distinguishes app queries from dashboard tests. Full input/output traces are stored locally but hidden from this table."
)
if request_rows:
    request_frame = pd.DataFrame(request_rows)[
        [
            "timestamp",
            "request_id",
            "model",
            "request_source",
            "status",
            "is_streaming",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "ttft_seconds",
            "tokens_per_second",
            "latency_seconds",
            "error_type",
        ]
    ]
    request_frame["is_streaming"] = request_frame["is_streaming"].astype(bool)
    st.dataframe(
        request_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Timestamp (UTC)"),
            "request_id": st.column_config.TextColumn("Request ID"),
            "model": st.column_config.TextColumn("Model"),
            "request_source": st.column_config.TextColumn("Source"),
            "status": st.column_config.TextColumn("Status"),
            "is_streaming": st.column_config.CheckboxColumn("Streaming"),
            "prompt_tokens": st.column_config.NumberColumn("Prompt tokens"),
            "completion_tokens": st.column_config.NumberColumn("Completion tokens"),
            "total_tokens": st.column_config.NumberColumn("Total tokens"),
            "ttft_seconds": st.column_config.NumberColumn("TTFT (s)", format="%.3f"),
            "tokens_per_second": st.column_config.NumberColumn(
                "Tokens/s", format="%.1f"
            ),
            "latency_seconds": st.column_config.NumberColumn(
                "Latency (s)", format="%.3f"
            ),
            "error_type": st.column_config.TextColumn("Error type"),
        },
    )
else:
    st.info("No request records have been saved yet.")

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
