# graph.py

from typing import List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph

from pwllms.agent import AgentState
from pwllms.tools import AVAILABLE_TOOLS, LLM_WITH_TOOLS


def create_agent_node(state: AgentState) -> AgentState:
    """
    Run the main agent LLM. It can decide to call tools or answer directly.
    """
    messages = state["messages"]
    try:
        response: AIMessage = LLM_WITH_TOOLS.invoke(messages)
    except Exception as e:
        response = AIMessage(content=f"ERROR: agent_node failed: {e}")

    new_state = dict(state)
    new_state["messages"] = messages + [response]
    return new_state


def create_tool_node(state: AgentState):
    """
    Execute any tool calls requested by the last assistant message.
    """
    messages = state["messages"]
    if not messages:
        return state

    last = messages[-1]

    # The last message must be an AIMessage with tool_calls
    if not isinstance(last, AIMessage):
        return state

    tool_calls = last.tool_calls or []
    if not tool_calls:
        return state

    new_messages: List[BaseMessage] = messages.copy()

    for tc in tool_calls:
        # For ChatOpenAI tool calls, tc is dict-like: {"name", "args", "id"}
        tool_name = tc["name"]
        # Make a mutable copy of args so we can sanitize
        args = dict(tc["args"])
        call_id = tc["id"]

        # --- SANITIZE ARGS FOR KNOWN TOOLS ---
        if tool_name in ("fetch_url", "read_text_file"):
            # Ollama sometimes sends "null" (string) or None for max_chars.
            if "max_chars" in args and args["max_chars"] in (None, "null", ""):
                # Remove it so Pydantic uses the default from the function signature.
                args.pop("max_chars")

        if tool_name == "web_search":
            if "max_results" in args and args["max_results"] in (None, "null", ""):
                args.pop("max_results")

        matching_tool = next((t for t in AVAILABLE_TOOLS if t.name == tool_name), None)
        if not matching_tool:
            new_messages.append(
                ToolMessage(
                    content=f"Tool {tool_name} not found.",
                    tool_call_id=call_id,
                )
            )
            continue

        result = matching_tool.invoke(args)

        # Add the ToolMessage so the LLM can see the tool output next step
        new_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call_id,
            )
        )
    new_state = dict(state)
    new_state["messages"] = new_messages

    return new_state


def _should_continue(state: AgentState):
    """
    Decide whether to continue to tools or end.

    If the last assistant message has tool_calls, go to tools; else, end.
    """
    messages = state["messages"]
    if not messages:
        return END

    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ---------- GRAPH ----------


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", create_agent_node)
    workflow.add_node("tools", create_tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def build_planner_graph(planner_node, executor_node):
    """
    Graph for hierarchical planning:
      - planner_node: builds plan
      - executor_node: runs plan and writes final answer
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", END)

    return workflow.compile()
