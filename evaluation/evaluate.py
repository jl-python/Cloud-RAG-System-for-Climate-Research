"""Persist RAG query metrics to Postgres (app.eval_metrics)."""
from __future__ import annotations
from typing import Any, Mapping, Optional
import psycopg2.extras


def log_metrics_to_postgres(
    log_data: Mapping[str, Any],
    conn: Optional[Any] = None,
) -> None:
    """Insert one row into app.eval_metrics. Caller may pass an existing connection."""
    close_after = False
    if conn is None:
        from scripts.db_connect import get_conn

        conn = get_conn()
        close_after = True

    cur = conn.cursor()
    try:
        tool_calls = log_data.get("tool_calls") or []
        cur.execute(
            """
            INSERT INTO app.eval_metrics (
                log_id, question, generated_response, context_used,
                retrieval_mode, confidence, latency_ms, tool_calls, num_iterations
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                log_data["log_id"],
                log_data["question"],
                log_data["answer"],
                log_data.get("context_used"),
                log_data["retrieval_mode"],
                log_data["confidence"],
                log_data["latency_ms"],
                psycopg2.extras.Json(tool_calls),
                log_data["num_iterations"],
            ),
        )
    finally:
        cur.close()
        if close_after:
            conn.close()
