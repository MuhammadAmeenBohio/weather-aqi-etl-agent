from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict


class SQLResult(TypedDict):
    sql:       str
    rows:      list[dict]
    columns:   list[str]
    row_count: int


class AgentState(TypedDict):
    # ── input ──────────────────────────────────────────────
    user_query:    str
    session_id:    Optional[str]

    # ── intent (set by supervisor) ─────────────────────────
    # "sql_only" | "plot_only" | "sql_and_plot" | "direct"
    intent:        Optional[str]

    # ── sql agent output ───────────────────────────────────
    sql_result:    Optional[SQLResult]
    sql_error:     Optional[str]

    # ── plot agent output ──────────────────────────────────
    plot_b64:      Optional[str]   # base64 PNG
    plot_error:    Optional[str]

    # ── final response ─────────────────────────────────────
    answer:        Optional[str]

    # ── internal ───────────────────────────────────────────
    retry_count:   int
    messages:      list[Any]       # full LangChain message history for the agent