"""
Chat service.

Holds conversation history in memory, keyed by conversation_id (a simple
dict is fine for a local, single-process portfolio app; swap for
Redis/Postgres if you productionize this). Each call to `chat()` invokes
the compiled LangGraph agent with the accumulated message history plus the
new user message, and returns its final natural-language reply along with
which tools it decided to call -- so the API/UI can show that "under the
hood, the agent called analyze_seasonality() and generate_forecast()".
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.graph import get_agent_graph
from app.core.logging_config import get_logger

logger = get_logger(__name__)

_conversations: dict[str, list] = {}


def chat(message: str, dataset_id: Optional[str], conversation_id: Optional[str]) -> dict[str, Any]:
    conversation_id = conversation_id or str(uuid.uuid4())[:8]
    history = _conversations.get(conversation_id, [])

    graph = get_agent_graph()
    initial_state = {
        "messages": [*history, HumanMessage(content=message)],
        "dataset_id": dataset_id,
        "tool_calls_made": [],
        "validation_notes": [],
        "step_count": 0,
    }

    final_state = graph.invoke(initial_state)
    _conversations[conversation_id] = final_state["messages"]

    last_ai_message = next(
        (m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage) and m.content),
        None,
    )
    reply = last_ai_message.content if last_ai_message else "I wasn't able to generate a response."

    return {
        "conversation_id": conversation_id,
        "reply": reply,
        "tool_calls_made": final_state.get("tool_calls_made", []),
        "validation_notes": final_state.get("validation_notes", []),
    }
