"""Persist one metrics row per LiteLLM request in a local SQLite database."""

import asyncio
import json
import os
import sqlite3
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from litellm.integrations.custom_logger import CustomLogger


DATABASE_PATH = Path(__file__).resolve().parent / "metrics.db"


def _database_path(path: Optional[Path] = None) -> Path:
    configured_path = path or os.getenv("MONITORING_DB_PATH")
    if configured_path is None:
        return DATABASE_PATH
    return Path(configured_path).expanduser().resolve()


def _connect(path: Optional[Path] = None) -> sqlite3.Connection:
    database_path = _database_path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize_database(path: Optional[Path] = None) -> Path:
    """Create the request table and index if they do not already exist."""
    database_path = _database_path(path)
    uses_default_path = database_path == DATABASE_PATH.resolve()
    database_preexisted = database_path.exists()
    connection = _connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_requests (
                request_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                request_source TEXT NOT NULL DEFAULT 'other',
                input_query TEXT,
                output_response TEXT,
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
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(llm_requests)")
        }
        if "request_source" not in columns:
            connection.execute(
                "ALTER TABLE llm_requests ADD COLUMN request_source TEXT NOT NULL DEFAULT 'other'"
            )
        if "input_query" not in columns:
            connection.execute("ALTER TABLE llm_requests ADD COLUMN input_query TEXT")
        if "output_response" not in columns:
            connection.execute("ALTER TABLE llm_requests ADD COLUMN output_response TEXT")
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_llm_requests_timestamp
            ON llm_requests(timestamp DESC)
            """
        )
        legacy_path = database_path.with_name("metrics.sqlite3")
        if uses_default_path and not database_preexisted and legacy_path.exists():
            legacy_connection = sqlite3.connect(legacy_path)
            legacy_connection.row_factory = sqlite3.Row
            try:
                legacy_table = legacy_connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table' AND name = 'llm_requests'
                    """
                ).fetchone()
                if legacy_table:
                    legacy_columns = {
                        row["name"]
                        for row in legacy_connection.execute(
                            "PRAGMA table_info(llm_requests)"
                        )
                    }
                    fields = (
                        "request_id",
                        "timestamp",
                        "model",
                        "request_source",
                        "input_query",
                        "output_response",
                        "status",
                        "is_streaming",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "ttft_seconds",
                        "tokens_per_second",
                        "latency_seconds",
                        "error_type",
                    )
                    legacy_rows = legacy_connection.execute(
                        "SELECT * FROM llm_requests"
                    ).fetchall()
                    values = [
                        tuple(
                            (
                                row[field]
                                if field in legacy_columns
                                else "other" if field == "request_source" else None
                            )
                            for field in fields
                        )
                        for row in legacy_rows
                    ]
                    placeholders = ", ".join("?" for _ in fields)
                    connection.executemany(
                        f"""
                        INSERT OR IGNORE INTO llm_requests ({", ".join(fields)})
                        VALUES ({placeholders})
                        """,
                        values,
                    )
            finally:
                legacy_connection.close()
        connection.commit()
    finally:
        connection.close()
    return database_path


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _epoch_seconds(value: Any) -> Optional[float]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
    return None


def _timestamp_text(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat(
        timespec="milliseconds"
    )


def _optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _error_type(kwargs: Mapping[str, Any], payload: Mapping[str, Any], response: Any) -> Optional[str]:
    exception = kwargs.get("exception") or kwargs.get("traceback_exception")
    if isinstance(exception, BaseException):
        return type(exception).__name__
    if isinstance(response, BaseException):
        return type(response).__name__

    error_information = _as_mapping(payload.get("error_information"))
    error_class = error_information.get("exception_class") or error_information.get(
        "error_type"
    )
    return str(error_class) if error_class else None


def _request_source(kwargs: Mapping[str, Any]) -> str:
    litellm_params = _as_mapping(kwargs.get("litellm_params"))
    proxy_request = _as_mapping(litellm_params.get("proxy_server_request"))
    headers = _as_mapping(proxy_request.get("headers"))
    source = next(
        (
            value
            for name, value in headers.items()
            if str(name).lower() == "x-request-source"
        ),
        None,
    )
    return source if source in {"rag-app", "dashboard-test"} else "other"


def _trace_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _output_text(response: Any) -> Optional[str]:
    response_mapping = _as_mapping(response)
    choices = response_mapping.get("choices")
    if isinstance(choices, list):
        output_messages = []
        for choice in choices:
            message = _as_mapping(_as_mapping(choice).get("message"))
            content = message.get("content")
            if content is not None:
                output_messages.append(_trace_text(content))
        if output_messages:
            return "\n".join(message for message in output_messages if message is not None)
    return _trace_text(response)


def _request_record(
    kwargs: Mapping[str, Any],
    response: Any,
    callback_start: Any,
    callback_end: Any,
    status: str,
) -> dict[str, Any]:
    payload = _as_mapping(kwargs.get("standard_logging_object"))
    start = _epoch_seconds(payload.get("startTime"))
    if start is None:
        start = _epoch_seconds(callback_start)
    end = _epoch_seconds(payload.get("endTime"))
    if end is None:
        end = _epoch_seconds(callback_end)
    if end is None:
        end = datetime.now(timezone.utc).timestamp()

    latency = max(end - start, 0.0) if start is not None else None
    is_streaming = bool(payload.get("stream", kwargs.get("stream", False)))

    ttft = None
    completion_start = _epoch_seconds(payload.get("completionStartTime"))
    if is_streaming and start is not None and completion_start is not None:
        ttft = max(completion_start - start, 0.0)
    elif is_streaming and payload.get("response_time") is not None:
        response_time = payload["response_time"]
        if isinstance(response_time, (int, float)):
            ttft = max(float(response_time), 0.0)

    prompt_tokens = _optional_int(payload.get("prompt_tokens"))
    completion_tokens = _optional_int(payload.get("completion_tokens"))
    total_tokens = _optional_int(payload.get("total_tokens"))
    generation_seconds = latency - ttft if latency is not None and ttft is not None else None
    tokens_per_second = (
        completion_tokens / generation_seconds
        if completion_tokens is not None
        and completion_tokens > 0
        and generation_seconds is not None
        and generation_seconds > 0
        else None
    )

    response_id = getattr(response, "id", None)
    request_id = (
        payload.get("id")
        or kwargs.get("litellm_call_id")
        or response_id
        or str(uuid.uuid4())
    )
    model = payload.get("model") or kwargs.get("model") or "unknown"
    input_messages = kwargs.get("messages", payload.get("messages"))

    return {
        "request_id": str(request_id),
        "timestamp": _timestamp_text(end),
        "model": str(model),
        "request_source": _request_source(kwargs),
        "input_query": _trace_text(input_messages),
        "output_response": _output_text(response) if status == "success" else None,
        "status": status,
        "is_streaming": int(is_streaming),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "ttft_seconds": ttft,
        "tokens_per_second": tokens_per_second,
        "latency_seconds": latency,
        "error_type": _error_type(kwargs, payload, response) if status == "failure" else None,
    }


def persist_request(record: Mapping[str, Any], path: Optional[Path] = None) -> None:
    """Insert or update request metrics and trace text."""
    initialize_database(path)
    connection = _connect(path)
    try:
        connection.execute(
            """
            INSERT INTO llm_requests (
                request_id, timestamp, model, request_source, status, is_streaming,
                input_query, output_response,
                prompt_tokens, completion_tokens, total_tokens, ttft_seconds,
                tokens_per_second, latency_seconds, error_type
            ) VALUES (
                :request_id, :timestamp, :model, :request_source, :status, :is_streaming,
                :input_query, :output_response,
                :prompt_tokens, :completion_tokens, :total_tokens, :ttft_seconds,
                :tokens_per_second, :latency_seconds, :error_type
            )
            ON CONFLICT(request_id) DO UPDATE SET
                timestamp = excluded.timestamp,
                model = excluded.model,
                request_source = excluded.request_source,
                input_query = excluded.input_query,
                output_response = excluded.output_response,
                status = excluded.status,
                is_streaming = excluded.is_streaming,
                prompt_tokens = excluded.prompt_tokens,
                completion_tokens = excluded.completion_tokens,
                total_tokens = excluded.total_tokens,
                ttft_seconds = excluded.ttft_seconds,
                tokens_per_second = excluded.tokens_per_second,
                latency_seconds = excluded.latency_seconds,
                error_type = excluded.error_type
            """,
            dict(record),
        )
        connection.commit()
    finally:
        connection.close()


def fetch_request_rows(
    limit: Optional[int] = 500,
    path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Read request metrics newest-first; the database itself has no row cap."""
    initialize_database(path)
    connection = _connect(path)
    try:
        if limit is None:
            rows = connection.execute(
                """
                SELECT * FROM llm_requests
                ORDER BY timestamp DESC, rowid DESC
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM llm_requests
                ORDER BY timestamp DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def count_requests(path: Optional[Path] = None) -> int:
    initialize_database(path)
    connection = _connect(path)
    try:
        result = connection.execute("SELECT COUNT(*) FROM llm_requests").fetchone()
        return int(result[0])
    finally:
        connection.close()


class SQLiteMetricsCallback(CustomLogger):
    """LiteLLM callback that saves one row after each completed request."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        super().__init__()
        self.db_path = db_path

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        record = _request_record(kwargs, response_obj, start_time, end_time, "success")
        await asyncio.to_thread(persist_request, record, self.db_path)

    async def async_log_failure_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        record = _request_record(kwargs, response_obj, start_time, end_time, "failure")
        await asyncio.to_thread(persist_request, record, self.db_path)


proxy_handler_instance = SQLiteMetricsCallback()
