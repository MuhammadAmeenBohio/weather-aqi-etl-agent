from __future__ import annotations

from langgraph.graph import StateGraph, END

from dotenv import load_dotenv
load_dotenv()  # must run before nodes are imported

from agent.state import AgentState
from agent.nodes.supervisor import supervisor_node, route_after_supervisor
from agent.nodes.sql_agent import sql_agent_node
from agent.nodes.plot_agent import plot_agent_node
from agent.nodes.direct_answer import direct_answer_node


def _route_after_sql(state: AgentState) -> str:
    """
    After sql_agent runs:
    - If intent is sql_and_plot AND we have data → go to plot_agent
    - Otherwise → END
    """
    if state.get("intent") == "sql_and_plot" and state.get("sql_result"):
        return "plot_agent"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── register nodes ────────────────────────────────────────────────────────
    graph.add_node("supervisor",    supervisor_node)
    graph.add_node("sql_agent",     sql_agent_node)
    graph.add_node("plot_agent",    plot_agent_node)
    graph.add_node("direct_answer", direct_answer_node)

    # ── entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── supervisor → conditional branch ──────────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "sql_agent":     "sql_agent",
            "direct_answer": "direct_answer",
        },
    )

    # ── sql_agent → conditional branch ───────────────────────────────────────
    graph.add_conditional_edges(
        "sql_agent",
        _route_after_sql,
        {
            "plot_agent": "plot_agent",
            END:          END,
        },
    )

    # ── terminal nodes ────────────────────────────────────────────────────────
    graph.add_edge("plot_agent",    END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


# Compiled graph — imported by main.py
pipeline = build_graph()