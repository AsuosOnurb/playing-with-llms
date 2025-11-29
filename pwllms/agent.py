# agent.py

from typing import List, TypedDict

from langchain_core.messages import (
    BaseMessage,
)


class AgentState(TypedDict):
    # Use LangChain message objects directly, not dicts
    messages: List[BaseMessage]
    plan: List[dict]
    executing_plan: bool
