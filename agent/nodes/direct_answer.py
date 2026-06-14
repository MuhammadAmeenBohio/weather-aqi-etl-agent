from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agent.state import AgentState

logger = logging.getLogger(__name__)

_DIRECT_PROMPT = """You are a helpful data analyst assistant specialising in
weather and air quality data for Pakistan. Answer the user's conceptual or
explanatory question clearly and concisely. You do not have access to live data
for this query — answer from domain knowle dge only."""

_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def direct_answer_node(state: AgentState) -> AgentState:
    """
    LangGraph node for intent='direct'.
    Answers conceptual questions without touching the database.
    """
    logger.info("Direct answer node: %s", state["user_query"])

    messages = [
        SystemMessage(content=_DIRECT_PROMPT),
        HumanMessage(content=state["user_query"]),
    ]

    response = _llm.invoke(messages)

    return {
        **state,
        "answer":     response.content.strip(),
        "sql_result": None,
        "plot_b64":   None,
    }