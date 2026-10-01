"""
Shared state object that flows through every node of the LangGraph graph.

LangGraph passes this dict-like object between nodes; each node reads what
it needs and returns a partial update that gets merged in. Keeping it a
plain TypedDict (rather than scattering extra args through function
signatures) is what makes the graph's nodes composable and easy to test in
isolation.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Full chat history for this conversation turn (LangChain message objects).
    # `add_messages` is a LangGraph reducer: instead of overwriting this list on
    # every node's return value, it appends new messages to it.
    messages: Annotated[list[BaseMessage], add_messages]

    # The dataset the user currently has "loaded" in this conversation.
    dataset_id: Optional[str]

    # Bookkeeping the API layer surfaces back to the client so the user (and
    # this portfolio's reviewers) can see exactly which tools the agent
    # decided to call and with what arguments -- this is what makes tool
    # selection visible/explainable instead of a black box.
    tool_calls_made: list[dict[str, Any]]

    # Set by the validation node if a tool call's output looked malformed
    # (e.g. NaNs, an error field). Surfaced to the agent so it can caveat
    # its answer instead of confidently explaining broken numbers.
    validation_notes: list[str]

    # Number of agent<->tool round trips so far, used to hard-cap the loop.
    step_count: int
