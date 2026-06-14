from __future__ import annotations

import os
import re
import logging
import json
from typing import Any

import geonamescache
import psycopg
from psycopg.rows import dict_row
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

DB_URL = os.environ.get('PIPELINE_DB_URL')

# ── blocked keywords (write-op guard) ─────────────────────────────────────────
_WRITE_OPS = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|MERGE)\b',
    re.IGNORECASE,
)

# ── geonamescache (no DB round-trip) ──────────────────────────────────────────
_gc = geonamescache.GeonamesCache()


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_connection():
    url = os.environ.get('PIPELINE_DB_URL')
    return psycopg.connect(url, row_factory=dict_row)


def _is_safe(sql: str) -> tuple[bool, str]:
    stripped = sql.strip().lstrip(';').upper()
    # Allow CTEs (WITH ... SELECT)
    if not (stripped.startswith('SELECT') or stripped.startswith('WITH')):
        return False, "Only SELECT statements are permitted."
    match = _WRITE_OPS.search(sql)
    if match:
        return False, f"Forbidden keyword detected: {match.group().upper()}"
    return True, ""


# ── tools ─────────────────────────────────────────────────────────────────────

@tool
def list_cities() -> list[str]:
    """
    Returns all Pakistani city names available in the database.
    Call this before filtering by city to ensure the city name is valid
    and matches what is stored in dim_location.
    """
    cities = sorted(
        city['name']
        for city in _gc.get_cities().values()
        if city['countrycode'] == 'PK'
    )
    logger.debug("list_cities returned %d cities", len(cities))
    return cities


@tool
def validate_sql(sql: str) -> dict[str, Any]:
    """
    Validates a SELECT query before execution.
    Checks for forbidden write operations and runs EXPLAIN to catch syntax errors.
    Returns {"valid": bool, "error": str | None}.
    Always call this before execute_sql.
    """
    is_safe, reason = _is_safe(sql)
    if not is_safe:
        return {"valid": False, "error": reason}

    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"EXPLAIN {sql}")
    except Exception as exc:
        return {"valid": False, "error": str(exc)}

    return {"valid": True, "error": None}


@tool
def execute_sql(sql: str) -> dict[str, Any]:
    """
    Executes a validated SELECT query against the TimescaleDB pipeline database.
    Returns {"sql": str, "columns": list, "rows": list[dict], "row_count": int}.
    Only call this after validate_sql confirms the query is valid.
    Raises ValueError if the query fails the safety check.
    """
    is_safe, reason = _is_safe(sql)
    if not is_safe:
        raise ValueError(f"Unsafe query blocked: {reason}")

    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        if not rows:
            columns = []
        else:
            columns = list(rows[0].keys())

        logger.info("execute_sql: %d rows returned", len(rows))
        result = {
            "sql":       sql,
            "columns":   columns,
            "rows":      [dict(r) for r in rows],
            "row_count": len(rows),
        }
        return json.dumps(result, default=str)

    except Exception as exc:
        logger.error("execute_sql failed: %s", exc)
        return {
            "sql":       sql,
            "columns":   [],
            "rows":      [],
            "row_count": 0,
            "error":     str(exc),
        }


# ── tool registry (imported by agent node) ────────────────────────────────────
SQL_TOOLS = [list_cities, validate_sql, execute_sql]