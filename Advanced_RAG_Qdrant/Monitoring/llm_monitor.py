"""Read LiteLLM's Prometheus metrics from the separate monitoring process."""

from dataclasses import dataclass
from typing import Optional

import requests
from prometheus_client.parser import text_string_to_metric_families


@dataclass(frozen=True)
class MetricsSnapshot:
    requests: float
    failed_requests: float
    input_tokens: float
    output_tokens: float
    average_ttft_seconds: Optional[float]
    average_llm_latency_seconds: Optional[float]
    average_request_latency_seconds: Optional[float]
    output_tokens_per_second: Optional[float]


def _read_samples(metrics_text: str):
    return [
        sample
        for family in text_string_to_metric_families(metrics_text)
        for sample in family.samples
    ]


def _counter_total(samples, metric_name: str) -> float:
    exact_names = {metric_name, f"{metric_name}_total"}
    values = [sample.value for sample in samples if sample.name in exact_names]
    return sum(values)


def _histogram_average(samples, metric_name: str) -> Optional[float]:
    total = _histogram_total(samples, metric_name)
    count = sum(
        sample.value for sample in samples if sample.name == f"{metric_name}_count"
    )
    return total / count if count else None


def _histogram_total(samples, metric_name: str) -> float:
    return sum(
        sample.value for sample in samples if sample.name == f"{metric_name}_sum"
    )


def parse_metrics(metrics_text: str) -> MetricsSnapshot:
    """Parse LiteLLM's Prometheus exposition into dashboard-ready aggregates."""
    samples = _read_samples(metrics_text)

    ttft_total = _histogram_total(
        samples, "litellm_llm_api_time_to_first_token_metric"
    )
    model_latency_total = _histogram_total(
        samples, "litellm_llm_api_latency_metric"
    )
    output_tokens = _counter_total(samples, "litellm_output_tokens_metric")
    generation_seconds = model_latency_total - ttft_total
    output_tokens_per_second = (
        output_tokens / generation_seconds
        if ttft_total > 0 and generation_seconds > 0
        else None
    )

    return MetricsSnapshot(
        requests=_counter_total(samples, "litellm_proxy_total_requests_metric"),
        failed_requests=_counter_total(samples, "litellm_proxy_failed_requests_metric"),
        input_tokens=_counter_total(samples, "litellm_input_tokens_metric"),
        output_tokens=output_tokens,
        average_ttft_seconds=_histogram_average(
            samples, "litellm_llm_api_time_to_first_token_metric"
        ),
        average_llm_latency_seconds=_histogram_average(
            samples, "litellm_llm_api_latency_metric"
        ),
        average_request_latency_seconds=_histogram_average(
            samples, "litellm_request_total_latency_metric"
        ),
        output_tokens_per_second=output_tokens_per_second,
    )


def fetch_metrics(
    base_url: str = "http://127.0.0.1:4000",
    api_key: Optional[str] = None,
    timeout: float = 5,
) -> MetricsSnapshot:
    """Fetch metrics from a running, authenticated LiteLLM proxy."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.get(
        f"{base_url.rstrip('/')}/metrics",
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_metrics(response.text)
