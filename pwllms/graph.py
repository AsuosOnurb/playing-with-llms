# graph.py

from typing import List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph

from pwllms.agent import AgentState
from pwllms.tools import AVAILABLE_TOOLS, LLM_WITH_TOOLS, llm

# ---------- AGENT NODE ----------


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


# ---------- HELPER: SHOULD ROUTE FROM AGENT TO GUARD? ----------


def _route_from_agent(state: AgentState):
    """
    Decide whether to go from agent -> guard or agent -> END.

    If the last assistant message has tool_calls, go to 'guard';
    else, end the turn.
    """
    messages = state["messages"]
    if not messages:
        return END

    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "guard"
    return END


def should_use_planner(user_input: str) -> bool:
    """
    Decide if a user message should go through the planner+executor
    instead of the normal chat graph.

    This is a simple heuristic version of `/plan` without the slash.
    """
    text = user_input.strip().lower()

    # Ignore explicit commands (/plan, /show-plan, /exec, etc.)
    if text.startswith("/"):
        return False

    # Obvious planning keywords
    planning_keywords = [
        "make a plan",
        "create a plan",
        "plan this",
        "plan out",
        "roadmap",
        "step by step",
        "step-by-step",
        "outline the steps",
        "outline steps",
        "break it down",
        "detailed steps",
        "implementation plan",
        "project plan",
    ]
    if any(k in text for k in planning_keywords):
        return True

    if "how to" in text:
        # If it's long-ish or clearly mentions sequencing
        if len(text.split()) >= 10 or " and " in text or " then " in text:
            return True

    return False


# ---------- HELPER: TOOL ALLOW / BLOCK HEURISTICS ----------


def _should_allow_tools(state: AgentState) -> bool:
    """
    Look at the latest user message and decide whether it makes sense
    to actually execute the proposed tools.
    """
    messages = state["messages"]
    if not messages:
        return False

    # Last AI with proposed tool calls
    last_ai = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            last_ai = m
            break

    if last_ai is None or not last_ai.tool_calls:
        return False

    # Latest human message
    last_user = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user = m
            break

    if last_user is None:
        return False

    text = last_user.content.strip().lower()

    # 1) Hard-block for obvious chit-chat / identity stuff
    chit_chat_keywords = [
        "hi",
        "hello",
        "hey",
        "heya",
        "how are you",
        "your name",
        "who are you",
        "what is your name",
        "how's it going",
        "how you doin",
    ]
    if any(k in text for k in chit_chat_keywords):
        return False

    # 2) Decide per tool type
    def _allow_tool(tool_name: str) -> bool:
        if tool_name == "list_files":
            return any(
                kw in text
                for kw in [
                    "list files",
                    "show files",
                    "what files",
                    "ls ",
                    "directory",
                    "folder",
                ]
            )
        if tool_name == "read_text_file":
            return any(
                kw in text
                for kw in [
                    "read file",
                    "open file",
                    "show file",
                    "view file",
                ]
            )
        if tool_name == "write_text_file":
            return any(
                kw in text
                for kw in [
                    "write file",
                    "save to file",
                    "create file",
                    "append to file",
                    "log to file",
                ]
            )
        if tool_name == "search_text_in_file":
            return any(
                kw in text
                for kw in [
                    "search in file",
                    "find in file",
                    "grep",
                ]
            )
        if tool_name == "web_search":
            return any(
                kw in text
                for kw in [
                    "search the web",
                    "look up",
                    "google",
                    "duckduckgo",
                    "find online",
                    "latest",
                    "news",
                ]
            )
        if tool_name == "fetch_url":
            return (
                "http://" in text
                or "https://" in text
                or "www." in text
                or "url" in text
                or any(
                    kw in text
                    for kw in [
                        "open this",
                        "fetch this",
                        "download",
                        "visit",
                        "go to",
                    ]
                )
            )
        if tool_name == "calculate":
            if any(
                kw in text
                for kw in [
                    "calculate",
                    "compute",
                    "sum",
                    "add",
                    "subtract",
                    "multiply",
                    "divide",
                    "math",
                ]
            ):
                return True
            has_digit = any(ch.isdigit() for ch in text)
            has_op = any(op in text for op in ["+", "-", "*", "/"])
            return has_digit and has_op

        # Default: disallow unknown tools
        return False

    # If ANY requested tool is clearly allowed, allow tools
    for tc in last_ai.tool_calls:
        tool_name = tc.get("name") or ""
        if _allow_tool(tool_name):
            return True

    return False


# ---------- GUARD NODE ----------


def guard_node(state: AgentState) -> AgentState:
    """
    Guard node: either let tool calls through unchanged, or
    block tools and replace the last AI tool-call message with a normal answer.
    """
    messages = state["messages"]
    if not messages:
        return state

    # Find last AI with tool_calls
    last_ai_index = None
    last_ai = None
    for idx in range(len(messages) - 1, -1, -1):
        m = messages[idx]
        if isinstance(m, AIMessage):
            last_ai_index = idx
            last_ai = m
            break

    if last_ai is None or not last_ai.tool_calls:
        # Nothing to guard
        return state

    # Decide if we allow tools
    if _should_allow_tools(state):
        # Do nothing; router will send us to 'tools'
        return state

    # Tools NOT allowed: call plain llm to get a normal answer instead.
    # We drop the tool-calling AI message and regenerate.
    base_messages = messages[:last_ai_index]  # up to but not including last_ai
    try:
        safe_response: AIMessage = llm.invoke(base_messages)
    except Exception as e:
        safe_response = AIMessage(content=f"ERROR: guard_node failed to re-answer: {e}")

    new_messages: List[BaseMessage] = base_messages + [safe_response]

    new_state = dict(state)
    new_state["messages"] = new_messages
    return new_state


def _route_from_guard(state: AgentState):
    """
    After guard_node runs, decide whether to execute tools.

    If the last assistant message STILL has tool_calls, go to 'tools';
    otherwise, end the turn.
    """
    messages = state["messages"]
    if not messages:
        return END

    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ---------- TOOL EXECUTION NODE ----------


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
        tool_name = tc["name"]
        args = dict(tc["args"])
        call_id = tc["id"]

        # --- SANITIZE ARGS FOR KNOWN TOOLS ---
        if tool_name in ("fetch_url", "read_text_file"):
            if "max_chars" in args and args["max_chars"] in (None, "null", ""):
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

        new_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call_id,
            )
        )

    new_state = dict(state)
    new_state["messages"] = new_messages

    return new_state


# ---------- GRAPH ----------


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", create_agent_node)
    workflow.add_node("guard", guard_node)
    workflow.add_node("tools", create_tool_node)

    workflow.set_entry_point("agent")

    # First routing: agent -> guard or END (if no tool_calls)
    workflow.add_conditional_edges(
        "agent",
        _route_from_agent,
        {
            "guard": "guard",
            END: END,
        },
    )

    # Second routing: guard -> tools or END (guard logic)
    workflow.add_conditional_edges(
        "guard",
        _route_from_guard,
        {
            "tools": "tools",
            END: END,
        },
    )

    # After tools, go back to agent for another round
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
