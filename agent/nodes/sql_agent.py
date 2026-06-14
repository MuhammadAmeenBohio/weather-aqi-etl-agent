from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode

from agent.state import AgentState, SQLResult
from agent.tools.db_tools import SQL_TOOLS
from agent.schema_context import build_system_prompt

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

# ── LLM + tools ───────────────────────────────────────────────────────────────
_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
_llm_with_tools = _llm.bind_tools(SQL_TOOLS)
_tool_node = ToolNode(SQL_TOOLS)

# ── system prompt (built once at import, introspects DB schema) ───────────────
_SYSTEM_PROMPT = build_system_prompt()


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_sql_result(messages: list[Any]) -> SQLResult | None:
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        logger.info("ToolMessage content: %s", str(msg.content)[:200])
        try:
            content = msg.content
            if isinstance(content, dict):
                data = content
            elif isinstance(content, str):
                data = json.loads(content)
            else:
                continue
        except (json.JSONDecodeError, TypeError):
            continue
        if all(k in data for k in ("sql", "columns", "rows", "row_count")):
            return SQLResult(
                sql=data["sql"],
                columns=data["columns"],
                rows=data["rows"],
                row_count=data["row_count"],
            )
    return None


def _last_ai_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content.strip()
    return ""


# ── main agent node ───────────────────────────────────────────────────────────

def sql_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node.  Runs a ReAct loop:
      1. LLM decides which tools to call (list_cities → validate_sql → execute_sql)
      2. Tools execute via ToolNode
      3. Loop until LLM produces a final text answer or MAX_RETRIES exceeded
    Updates state with sql_result (on success) or sql_error (on failure).
    """
    retry_count = state.get("retry_count", 0)

    # Build message history for this turn
    messages: list[Any] = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=state["user_query"]),
    ]

    # Append any prior messages from state (multi-turn or retry context)
    if state.get("messages"):
        clean_history = [
            m for m in state["messages"]
            if isinstance(m, (HumanMessage, AIMessage))
            and not getattr(m, 'tool_calls', None)  # exclude tool call messages
        ]
        messages = [SystemMessage(content=_SYSTEM_PROMPT)] + clean_history + [HumanMessage(content=state["user_query"])]

    # ── ReAct loop ────────────────────────────────────────────────────────────
    for attempt in range(MAX_RETRIES + 1):
        logger.info("SQL agent attempt %d/%d", attempt + 1, MAX_RETRIES + 1)

        ai_msg = _llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # No tool calls → LLM gave a direct answer (shouldn't happen for SQL
        # queries, but handle gracefully)
        if not ai_msg.tool_calls:
            logger.info("SQL agent: LLM returned direct answer without tool calls")
            return {
                **state,
                "messages":   messages,
                "sql_result": _extract_sql_result(messages),
                "answer":     _last_ai_text(messages),
                "sql_error":  None,
                "retry_count": attempt,
            }

        # Execute tools
        tool_result_state = _tool_node.invoke({"messages": messages})
        tool_messages: list[ToolMessage] = tool_result_state["messages"]
        messages.extend(tool_messages)

        # Check if any tool returned a validation error
        validation_error = _check_validation_error(tool_messages)
        if validation_error:
            if attempt < MAX_RETRIES:
                # Inject error feedback and retry
                logger.warning("Validation failed (attempt %d): %s", attempt + 1, validation_error)
                messages.append(
                    HumanMessage(
                        content=(
                            f"The SQL you wrote failed validation:\n{validation_error}\n"
                            "Please fix the query and try again."
                        )
                    )
                )
                continue
            else:
                logger.error("SQL agent: max retries exceeded. Last error: %s", validation_error)
                return {
                    **state,
                    "messages":  messages,
                    "sql_result": None,
                    "sql_error":  f"Query validation failed after {MAX_RETRIES + 1} attempts: {validation_error}",
                    "retry_count": attempt,
                }

        # Check if execute_sql was called successfully
        sql_result = _extract_sql_result(messages)
        if sql_result:
            # One final LLM call to generate a natural language answer
            messages.append(
                HumanMessage(
                    content=(
                        f"The query returned {sql_result['row_count']} rows with columns {sql_result['columns']}. "
                        f"Here are the results: {sql_result['rows'][:10]}. "
                        "Write a clear 2-3 sentence summary of what these results show. "
                        "You MUST write a text response."
                    )
                )
            )
            final_msg = _llm_with_tools.invoke(messages)
            messages.append(final_msg)

            return {
                **state,
                "messages":   messages,
                "sql_result": sql_result,
                "answer":     _last_ai_text(messages),
                "sql_error":  None,
                "retry_count": attempt,
            }

        # Tools ran but execute_sql wasn't called yet — continue the loop
        # (e.g. LLM called list_cities or validate_sql, needs another turn)

    # Exhausted retries without a result
    return {
        **state,
        "messages":   messages,
        "sql_result": None,
        "sql_error":  "SQL agent did not produce a result within the retry limit.",
        "retry_count": MAX_RETRIES,
    }


# ── private helpers ───────────────────────────────────────────────────────────

def _check_validation_error(tool_messages: list[ToolMessage]) -> str | None:
    """
    Looks for a validate_sql ToolMessage that returned valid=False.
    Returns the error string or None.
    """
    for msg in tool_messages:
        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and data.get("valid") is False:
            return data.get("error", "Unknown validation error")
    return None