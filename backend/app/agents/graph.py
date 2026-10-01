"""
The LangGraph agent workflow.

Graph shape
-----------

    START -> agent -> (conditional) -> tools -> validate -> agent -> ... -> END
                    \\-> (no tool calls) -> END

  1. **agent** node: the LLM (Groq, via agents/llm.py) is given the running
     conversation plus a set of tools bound with `.bind_tools(...)`. The LLM
     decides -- on its own, from the user's message and the system prompt --
     whether it needs to call one or more tools, and if so, which ones and
     with what arguments. This is "tool selection": the model reasons about
     intent (e.g. "the user is asking about seasonality" -> call
     `analyze_seasonality`) rather than us hard-coding an intent classifier.

  2. **tools** node (LangGraph's prebuilt `ToolNode`): executes whichever
     tool calls the agent emitted, against the *real* Python functions in
     agents/tools.py. Each tool call's result is appended to the message
     history as a `ToolMessage`.

  3. **validate** node: a lightweight guard that inspects the ToolMessages
     just produced for error markers or obviously malformed output (NaNs,
     empty results) and records a note. This is what "result validation"
     means in this project -- it's intentionally simple (not another LLM
     call) so validation is fast, deterministic, and never itself a source
     of hallucination.

  4. Control returns to **agent**, which now has the tool results in its
     context and either calls more tools (e.g. it analyzed trend, and now
     decides it also needs seasonality) or produces a final natural-language
     answer with no further tool calls, which ends the graph.

This loop is capped at `MAX_STEPS` round-trips to guarantee termination even
if the LLM gets stuck calling tools repeatedly.
"""
from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.agents.llm import get_chat_model
from app.agents.state import AgentState
from app.agents.tools import build_tools
from app.core.logging_config import get_logger

logger = get_logger(__name__)

MAX_STEPS = 6

SYSTEM_PROMPT = """You are ForecastGPT, a conversational assistant for time-series analysis and forecasting.

Ground rules (do not break these):
1. You NEVER invent numeric results (forecast values, trend slopes, anomaly scores, error metrics)
   yourself. Every number in your answer must come from a tool call you made in this conversation.
2. If the user asks something that requires a number you don't already have from a tool result in
   this conversation, call the appropriate tool first, then answer using its output.
3. If no dataset is loaded and the user asks about "the data", tell them to upload a CSV first.
4. Use `search_knowledge_base` for conceptual questions about forecasting methods/metrics, not for
   numeric results about the user's own data.
5. Be concise and concrete: state the key number(s), then a short plain-language interpretation.
6. If a tool result contains an "error" field or a validation note flags a problem, say so honestly
   instead of glossing over it.

Available tools let you: profile the dataset, analyze trend, analyze seasonality, detect anomalies,
generate a forecast with a specific model, compare all models, create a visualization, search
forecasting documentation, and run a simplified what-if scenario.
"""


def _validate_tool_messages(messages: list) -> list[str]:
    """Inspect the most recent ToolMessages for problems. Deterministic
    (string/JSON checks only) on purpose -- validation should not itself be
    a place where hallucination can creep in."""
    notes = []
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            break
        try:
            payload = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("error"):
            notes.append(f"Tool '{msg.name}' reported an error: {payload['error']}")
        if isinstance(payload, dict) and payload.get("n_rows") == 0:
            notes.append(f"Tool '{msg.name}' returned zero rows.")
    return notes


def build_graph():
    llm = get_chat_model()

    def agent_node(state: AgentState) -> dict:
        dataset_id = state.get("dataset_id")
        tools = build_tools(dataset_id)
        llm_with_tools = llm.bind_tools(tools)

        # Strip any previously-injected system message and rebuild it fresh
        # every turn. This matters because the LLM has no other way of
        # knowing whether a dataset is currently loaded -- that fact lives
        # in graph state, not in the conversation text, so it must be
        # re-stated explicitly on every call or the model will (wrongly)
        # assume no data has been uploaded.
        if dataset_id:
            dataset_status = (
                f"\n\nA dataset IS currently loaded (dataset_id: {dataset_id}). "
                "Do not tell the user to upload a file. Call the relevant tool "
                "to answer their question."
            )
        else:
            dataset_status = (
                "\n\nNo dataset is currently loaded. If the user asks a question "
                "about data, ask them to upload a CSV first."
            )
        system_message = SystemMessage(content=SYSTEM_PROMPT + dataset_status)

        conversation = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        messages = [system_message, *conversation]

        response: AIMessage = llm_with_tools.invoke(messages)

        tool_calls_made = list(state.get("tool_calls_made", []))
        for call in getattr(response, "tool_calls", None) or []:
            tool_calls_made.append({"tool": call["name"], "args": call["args"]})
            logger.info(f"Agent decided to call tool '{call['name']}' with args {call['args']}")

        return {
            "messages": [response],
            "tool_calls_made": tool_calls_made,
            "step_count": state.get("step_count", 0) + 1,
        }

    def tools_node(state: AgentState) -> dict:
        tools = build_tools(state.get("dataset_id"))
        node = ToolNode(tools)
        result = node.invoke(state)
        return result

    def validate_node(state: AgentState) -> dict:
        notes = _validate_tool_messages(state["messages"])
        existing = list(state.get("validation_notes", []))
        if notes:
            logger.warning(f"Validation flagged: {notes}")
            existing.extend(notes)
            # Feed the validation note back in as a system-ish tool message so
            # the next agent turn actually sees it and can caveat its answer.
            return {"validation_notes": existing}
        return {"validation_notes": existing}

    def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if state.get("step_count", 0) >= MAX_STEPS:
            return END
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "validate")
    graph.add_edge("validate", "agent")

    return graph.compile()


_compiled_graph = None


def get_agent_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph