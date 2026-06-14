from __future__ import annotations

import base64
import io
import logging
import textwrap
import traceback
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agent.state import AgentState, SQLResult

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

_PLOT_SYSTEM_PROMPT = """You are a data visualization expert. You receive a SQL query result
(column names + rows as JSON) and the user's original question, and you write Python code
to produce a clear, professional matplotlib/seaborn chart.

Available in the sandbox (already imported — do NOT import anything):
- plt      : matplotlib.pyplot
- sns      : seaborn
- pd       : pandas
- df       : pandas DataFrame containing the query results
- buf      : io.BytesIO buffer — save your figure here
- BytesIO  : io.BytesIO (available if needed)

Rules:
- Do NOT import anything. All variables above are pre-loaded.
- Save the figure using EXACTLY: plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
- Do NOT call plt.show(). Do NOT call buf = BytesIO() — buf already exists.
- Use seaborn whitegrid style: sns.set_theme(style="whitegrid")
- Always set a clear title, axis labels, and rotate x-axis ticks if there are many categories.
- Datetime columns in df are already parsed as pandas datetime64 — use them directly.
  For x-axis tick formatting use: plt.gcf().autofmt_xdate()
- Never call pd.to_datetime() or pd.to_numeric() on datetime columns.
- Choose the right chart type:
    - Time series          → line chart (use df['col'] directly on x-axis)
    - Category comparison  → bar chart (horizontal if many categories)
    - Distribution         → histogram or boxplot
    - Correlation          → scatter or heatmap
- Keep the code clean and concise. No comments needed.
- Return ONLY raw Python code — no explanation, no markdown fences, no ```python wrapper.
"""


_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def _build_plot_prompt(sql_result: SQLResult, user_query: str) -> str:
    # Send at most 500 rows to keep the prompt lean
    sample_rows = sql_result["rows"][:500]
    return (
        f"User question: {user_query}\n\n"
        f"Columns: {sql_result['columns']}\n"
        f"Row count: {sql_result['row_count']}\n"
        f"Data (JSON):\n{sample_rows}\n\n"
        "Write the Python plotting code."
    )


def _execute_plot_code(code: str, sql_result: SQLResult) -> str:
    """
    Executes LLM-generated plotting code in a sandboxed context.
    Returns base64-encoded PNG string.
    Raises RuntimeError on execution failure.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    buf = io.BytesIO()

    # Build DataFrame and pre-parse datetime columns
    df = pd.DataFrame(sql_result["rows"])
    for col in df.columns:
        if df[col].dtype == object:
            try:
                converted = pd.to_datetime(df[col])
                df[col] = converted
            except Exception:
                pass

    sandbox: dict[str, Any] = {
        "pd":      pd,
        "plt":     plt,
        "sns":     sns,
        "df":      df,
        "buf":     buf,
        "BytesIO": io.BytesIO,
        "io":      io,
    }

    try:
        exec(textwrap.dedent(code), sandbox)   # noqa: S102
    except Exception as exc:
        raise RuntimeError(f"Plot code execution failed: {exc}\n{traceback.format_exc()}")

    # Use sandbox buf in case LLM reassigned it
    final_buf = sandbox.get("buf", buf)
    final_buf.seek(0)
    png_bytes = final_buf.read()
    if not png_bytes:
        raise RuntimeError("Plot code ran but produced no output (buf is empty).")

    plt.close("all")
    return base64.b64encode(png_bytes).decode("utf-8")


def _strip_fences(code: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` wrappers if present."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        # Drop first line (```python or ```) and last line (```)
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        code = "\n".join(
            line for line in inner if line.strip() != "```"
        )
    return code.strip()


def plot_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node. Receives sql_result from state, asks the LLM to write
    matplotlib code, executes it in a sandbox, and stores base64 PNG in state.
    Retries up to MAX_RETRIES times with error feedback on failure.
    """
    sql_result = state.get("sql_result")

    if not sql_result or sql_result["row_count"] == 0:
        logger.warning("Plot agent: no data to plot.")
        return {
            **state,
            "plot_b64":   None,
            "plot_error": "No data available to generate a plot.",
        }

    user_query = state["user_query"]
    plot_prompt = _build_plot_prompt(sql_result, user_query)

    messages = [
        SystemMessage(content=_PLOT_SYSTEM_PROMPT),
        HumanMessage(content=plot_prompt),
    ]

    last_error: str | None = None

    for attempt in range(MAX_RETRIES + 1):
        logger.info("Plot agent attempt %d/%d", attempt + 1, MAX_RETRIES + 1)

        response = _llm.invoke(messages)
        code = _strip_fences(response.content)

        try:
            b64_png = _execute_plot_code(code, sql_result)
            logger.info("Plot agent: chart generated successfully.")
            return {
                **state,
                "plot_b64":   b64_png,
                "plot_error": None,
            }

        except RuntimeError as exc:
            last_error = str(exc)
            logger.warning("Plot attempt %d failed: %s", attempt + 1, last_error)

            if attempt < MAX_RETRIES:
                # Feed the error back so the LLM can self-correct
                messages.append(response)
                messages.append(
                    HumanMessage(
                        content=(
                            f"The code you wrote failed with this error:\n{last_error}\n"
                            "Fix the code and try again. Return only the corrected Python code."
                        )
                    )
                )

    logger.error("Plot agent: max retries exceeded.")
    return {
        **state,
        "plot_b64":   None,
        "plot_error": f"Failed to generate plot after {MAX_RETRIES + 1} attempts: {last_error}",
    }