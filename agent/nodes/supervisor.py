from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agent.state import AgentState

logger = logging.getLogger(__name__)

_SUPERVISOR_PROMPT = """You are a routing assistant for a Pakistan Weather & AQI data system.

Your only job is to classify the user's query into one of three intents:

1. "sql_only"     — user wants data, numbers, statistics, comparisons, rankings,
                    or any factual answer that requires querying the database.
                    Examples: "What was the average temperature in Karachi last week?",
                    "Which city had the worst AQI in January?"

2. "sql_and_plot" — user wants data AND a chart, graph, plot, or visualization.
                    Examples: "Plot AQI trends for Lahore", "Show me a chart of
                    temperature over the last month", "Visualize rainfall by province"

3. "direct"       — user is asking a conceptual or explanatory question that does NOT
                    need any database query. Answer from general knowledge.
                    Examples: "What does PM2.5 mean?", "Explain what AQI is",
                    "What is a hypertable?", "How does temperature affect air quality?"

Respond with ONLY a valid JSON object, no explanation, no markdown:
{"intent": "<sql_only|sql_and_plot|direct>", "reason": "<one short sentence>"}
"""


_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def supervisor_node(state: AgentState) -> AgentState:
    """
    Classifies user intent and sets state["intent"].
    Downstream graph edges route based on this value.
    """
    logger.info("Supervisor classifying query: %s", state["user_query"])

    messages = [
        SystemMessage(content=_SUPERVISOR_PROMPT),
        HumanMessage(content=state["user_query"]),
    ]

    response = _llm.invoke(messages)

    try:
        raw = response.content.strip()
        # Strip markdown fences if model wraps in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        intent = data.get("intent", "sql_only")
        reason = data.get("reason", "")
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning("Supervisor parse error: %s — defaulting to sql_only", exc)
        intent = "sql_only"
        reason = "Parse error, defaulted to sql_only"

    # Guard against hallucinated intents
    if intent not in ("sql_only", "sql_and_plot", "direct"):
        logger.warning("Unknown intent '%s' — defaulting to sql_only", intent)
        intent = "sql_only"

    logger.info("Supervisor intent: %s | reason: %s", intent, reason)

    return {
        **state,
        "intent": intent,
    }


def route_after_supervisor(state: AgentState) -> str:
    """
    LangGraph conditional edge function.
    Returns the name of the next node to execute.
    """
    intent = state.get("intent", "sql_only")
    route_map = {
        "sql_only":     "sql_agent",
        "sql_and_plot": "sql_agent",   # plot_agent runs after sql_agent
        "direct":       "direct_answer",
    }
    next_node = route_map.get(intent, "sql_agent")
    logger.info("Routing to: %s", next_node)
    return next_node