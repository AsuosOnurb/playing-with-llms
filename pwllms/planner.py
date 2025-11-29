# planner.py

import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from pwllms.agent import AgentState
from pwllms.tools import llm

PLANNER_SYSTEM_PROMPT = """
You are a hierarchical task planner for a local agent that has tools.

You must output ONLY JSON (no extra text) with this structure:

{
  "goal": "short description of the overall goal",
  "steps": [
    {
      "id": 1,
      "kind": "subplan",
      "description": "High-level phase, e.g. 'Gather information about X'",
      "substeps": [
        {
          "id": 2,
          "kind": "tool",
          "tool_name": "web_search",
          "description": "Search web for X",
          "args": {"query": "X", "max_results": 3}
        },
        {
          "id": 3,
          "kind": "tool",
          "tool_name": "fetch_url",
          "description": "Fetch most relevant result",
          "args": {"url": "https://example.com"}
        }
      ]
    },
    {
      "id": 4,
      "kind": "tool",
      "tool_name": "write_text_file",
      "description": "Write final report to a file",
      "args": {"path": "report.md", "content": "report goes here"}
    }
  ]
}

Rules:
- kind must be one of: "tool", "subplan", "note".
- For kind="tool", you MUST provide "tool_name" and "args".
- Allowed tool_name values: "calculate", "list_files", "read_text_file",
  "write_text_file", "search_text_in_file", "fetch_url", "web_search".
- For kind="subplan", you MUST provide "substeps" as a non-empty list.
- For kind="note", you just provide a description (no tool_name, no args).

Be concrete and realistic in args, avoid placeholders like "URL_HERE".
""".strip()


def create_planner_node(state: AgentState) -> AgentState:
    """
    Create a hierarchical plan for the latest user request.
    Stores it in state["plan"].
    """
    # Find the latest user message as the task/goal
    goal = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            goal = m.content
            break

    if not goal:
        # Nothing to plan
        return state

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Create a hierarchical plan for this task:\n{goal}"),
    ]

    resp: AIMessage = llm.invoke(messages)
    raw = resp.content

    plan: dict
    try:
        plan = json.loads(raw)
    except Exception:
        # Fallback: simple one-step plan
        plan = {
            "goal": goal,
            "steps": [
                {
                    "id": 1,
                    "kind": "note",
                    "description": "Planner failed to produce valid JSON. Just answer directly.",
                }
            ],
        }

    new_state = dict(state)
    new_state["plan"] = plan.get("steps", [])
    new_state["executing_plan"] = True

    # Optionally add a summary message of the plan
    plan_summary = (
        f"Created plan for goal: {plan.get('goal', goal)} "
        f"with {len(new_state['plan'])} top-level steps."
    )
    new_state["messages"] = state["messages"] + [SystemMessage(content=plan_summary)]

    return new_state
