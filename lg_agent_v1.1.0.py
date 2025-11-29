import os
from typing import List, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# ---------- STATE ----------


class AgentState(TypedDict):
    # Use LangChain message objects directly, not dicts
    messages: List[BaseMessage]


# ---------- LLM ----------

llm = ChatOpenAI(
    model="llama3.2:3b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # dummy, required by client
    temperature=0,
)


# ---------- TOOLS ----------


@tool
def calculate(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression, like '2 + 2 * 3'."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Expression contains invalid characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def list_files(path: str) -> str:
    """List files in the given directory path."""
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {e}"


tools = [calculate, list_files]
llm_with_tools = llm.bind_tools(tools)


# ---------- NODES ----------


def agent_node(state: AgentState) -> AgentState:
    """
    Run the main agent LLM. It can decide to call tools or answer directly.
    """
    messages = state["messages"]
    response: AIMessage = llm_with_tools.invoke(messages)
    # Append the AIMessage directly
    return {"messages": messages + [response]}


def tool_node(state: AgentState) -> AgentState:
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
        args = tc["args"]
        call_id = tc["id"]

        matching_tool = next((t for t in tools if t.name == tool_name), None)
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

    return {"messages": new_messages}


# ---------- ROUTER ----------


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

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

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


# ---------- CLI ----------

if __name__ == "__main__":
    graph = build_graph()
    print("LangGraph agent with memory. Type 'exit' to quit.\n")

    system_prompt = (
        "You are a helpful local agent.\n"
        "You have access to tools: calculate, list_files.\n"
        "When the user asks you to do math or inspect directories, "
        "prefer to call a tool instead of guessing."
    )

    state: AgentState = {"messages": [SystemMessage(content=system_prompt)]}

    while True:
        user = input("You: ")
        if user.strip().lower() in {"exit", "quit"}:
            break

        # Append new user message to existing convo
        state["messages"].append(HumanMessage(content=user))

        # Run the graph
        state = graph.invoke(state)

        # Find last ai message
        last_ai = None
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                last_ai = m
                break

        if last_ai is None:
            print("Agent: (no answer)\n")
        else:
            print(f"Agent: {last_ai.content}\n")
