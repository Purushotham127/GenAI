import json
import unittest
import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from Monitoring.llm_monitor import parse_metrics
from Monitoring import sqlite_metrics
from Monitoring.sqlite_metrics import (
    SQLiteMetricsCallback,
    count_requests,
    fetch_request_rows,
    initialize_database,
)


PROMETHEUS_FIXTURE = """
# TYPE litellm_proxy_total_requests_metric counter
litellm_proxy_total_requests_metric_total{requested_model="qwen3:4b"} 3
# TYPE litellm_proxy_failed_requests_metric counter
litellm_proxy_failed_requests_metric_total{requested_model="qwen3:4b"} 1
# TYPE litellm_input_tokens_metric counter
litellm_input_tokens_metric_total{model="qwen3:4b"} 30
# TYPE litellm_output_tokens_metric counter
litellm_output_tokens_metric_total{model="qwen3:4b"} 20
# TYPE litellm_llm_api_time_to_first_token_metric histogram
litellm_llm_api_time_to_first_token_metric_sum{model="qwen3:4b"} 1.2
litellm_llm_api_time_to_first_token_metric_count{model="qwen3:4b"} 3
# TYPE litellm_llm_api_latency_metric histogram
litellm_llm_api_latency_metric_sum{model="qwen3:4b"} 6
litellm_llm_api_latency_metric_count{model="qwen3:4b"} 3
# TYPE litellm_request_total_latency_metric histogram
litellm_request_total_latency_metric_sum{model="qwen3:4b"} 6.6
litellm_request_total_latency_metric_count{model="qwen3:4b"} 3
# TYPE litellm_deployment_latency_per_output_token gauge
litellm_deployment_latency_per_output_token{model="qwen3:4b"} 0.05
"""


class ParseMetricsTests(unittest.TestCase):
    def test_parses_counters_histograms_and_output_rate(self):
        snapshot = parse_metrics(PROMETHEUS_FIXTURE)

        self.assertEqual(snapshot.requests, 3)
        self.assertEqual(snapshot.failed_requests, 1)
        self.assertEqual(snapshot.input_tokens, 30)
        self.assertEqual(snapshot.output_tokens, 20)
        self.assertAlmostEqual(snapshot.average_ttft_seconds, 0.4)
        self.assertAlmostEqual(snapshot.average_llm_latency_seconds, 2)
        self.assertAlmostEqual(snapshot.average_request_latency_seconds, 2.2)
        self.assertAlmostEqual(snapshot.output_tokens_per_second, 20 / 4.8)

    def test_missing_metrics_are_not_reported_as_zero_latency(self):
        snapshot = parse_metrics("")

        self.assertEqual(snapshot.requests, 0)
        self.assertIsNone(snapshot.average_ttft_seconds)
        self.assertIsNone(snapshot.output_tokens_per_second)


class SQLiteMetricsCallbackTests(unittest.TestCase):
    def test_persists_request_metrics_and_reads_newest_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "test-metrics.db"
            initialize_database(database)
            callback = SQLiteMetricsCallback(database)
            now = datetime.now(timezone.utc).timestamp()

            async def write_requests():
                await callback.async_log_success_event(
                    {
                        "model": "qwen3:4b",
                        "stream": True,
                        "messages": [{"role": "user", "content": "full input query"}],
                        "litellm_params": {
                            "proxy_server_request": {
                                "headers": {"X-Request-Source": "rag-app"}
                            }
                        },
                        "standard_logging_object": {
                            "id": "request-older",
                            "model": "qwen3:4b",
                            "stream": True,
                            "startTime": now - 2,
                            "completionStartTime": now - 1.5,
                            "endTime": now,
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                        },
                    },
                    {
                        "choices": [
                            {"message": {"content": "full generated response"}}
                        ]
                    },
                    None,
                    None,
                )
                await callback.async_log_failure_event(
                    {
                        "model": "qwen3:4b",
                        "exception": RuntimeError("model unavailable"),
                        "standard_logging_object": {
                            "id": "request-newer",
                            "model": "qwen3:4b",
                            "startTime": now - 0.5,
                            "endTime": now,
                            "error_information": {"exception_class": "RuntimeError"},
                        },
                    },
                    None,
                    None,
                    None,
                )

            asyncio.run(write_requests())
            rows = fetch_request_rows(path=database)

            self.assertEqual(count_requests(database), 2)
            self.assertEqual(
                [row["request_id"] for row in rows],
                ["request-newer", "request-older"],
            )
            self.assertEqual(rows[1]["status"], "success")
            self.assertEqual(rows[1]["request_source"], "rag-app")
            self.assertEqual(
                json.loads(rows[1]["input_query"]),
                [{"role": "user", "content": "full input query"}],
            )
            self.assertEqual(rows[1]["output_response"], "full generated response")
            self.assertEqual(rows[1]["prompt_tokens"], 10)
            self.assertEqual(rows[1]["completion_tokens"], 20)
            self.assertEqual(rows[1]["ttft_seconds"], 0.5)
            self.assertAlmostEqual(rows[1]["tokens_per_second"], 20 / 1.5)
            self.assertEqual(rows[0]["status"], "failure")
            self.assertEqual(rows[0]["error_type"], "RuntimeError")

    def test_migrates_legacy_database_when_creating_default_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "metrics.db"
            legacy_database = Path(temp_dir) / "metrics.sqlite3"
            connection = sqlite_metrics.sqlite3.connect(legacy_database)
            try:
                connection.execute(
                    """
                    CREATE TABLE llm_requests (
                        request_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        model TEXT NOT NULL,
                        status TEXT NOT NULL,
                        is_streaming INTEGER NOT NULL,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        total_tokens INTEGER,
                        ttft_seconds REAL,
                        tokens_per_second REAL,
                        latency_seconds REAL,
                        error_type TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO llm_requests (
                        request_id, timestamp, model, status, is_streaming
                    ) VALUES ('legacy-request', '2026-01-01T00:00:00+00:00',
                              'qwen3:4b', 'success', 0)
                    """
                )
                connection.commit()
            finally:
                connection.close()

            with patch.object(sqlite_metrics, "DATABASE_PATH", database):
                with patch.dict(
                    sqlite_metrics.os.environ,
                    {"MONITORING_DB_PATH": str(database)},
                    clear=False,
                ):
                    initialized_path = initialize_database()

            rows = fetch_request_rows(path=initialized_path)
            self.assertEqual(initialized_path.name, "metrics.db")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["request_id"], "legacy-request")
            self.assertEqual(rows[0]["request_source"], "other")


if __name__ == "__main__":
    unittest.main()
