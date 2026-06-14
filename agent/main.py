from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.graph import pipeline
from agent.state import AgentState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pakistan Weather & AQI Agent",
    description="Natural language querying over the Pakistan weather/AQI pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_UI_DIR = Path(__file__).parent / "ui"
app.mount("/ui", StaticFiles(directory=_UI_DIR), name="ui")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(_UI_DIR / "index.html")

# ── in-memory session store (message history per session_id) ──────────────────
# For production swap this out for Redis or a DB-backed store.
_sessions: dict[str, list] = {}


# ── request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:      str
    session_id: Optional[str] = None


class SQLResultOut(BaseModel):
    sql:       str
    columns:   list[str]
    rows:      list[dict]
    row_count: int


class QueryResponse(BaseModel):
    session_id: str
    intent:     str
    answer:     Optional[str]
    sql_result: Optional[SQLResultOut]
    plot_b64:   Optional[str]        # base64 PNG, None if no chart
    error:      Optional[str]


# ── endpoint ──────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    session_id = request.session_id or str(uuid.uuid4())
    prior_messages = _sessions.get(session_id, [])

    logger.info("Session %s | query: %s", session_id, request.query)

    initial_state: AgentState = {
        "user_query":  request.query,
        "session_id":  session_id,
        "intent":      None,
        "sql_result":  None,
        "sql_error":   None,
        "plot_b64":    None,
        "plot_error":  None,
        "answer":      None,
        "retry_count": 0,
        "messages":    prior_messages,
    }

    try:
        final_state: AgentState = pipeline.invoke(initial_state)
    except Exception as exc:
        logger.exception("Graph execution error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Persist messages for next turn
    if final_state.get("messages"):
        _sessions[session_id] = final_state["messages"]

    # Collect any errors
    error = final_state.get("sql_error") or final_state.get("plot_error")

    # Build sql_result output model
    sql_out = None
    if sr := final_state.get("sql_result"):
        sql_out = SQLResultOut(
            sql=sr["sql"],
            columns=sr["columns"],
            rows=sr["rows"],
            row_count=sr["row_count"],
        )

    return QueryResponse(
        session_id=session_id,
        intent=final_state.get("intent", "unknown"),
        answer=final_state.get("answer"),
        sql_result=sql_out,
        plot_b64=final_state.get("plot_b64"),
        error=error,
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str) -> dict:
    """Clear conversation history for a session."""
    _sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}