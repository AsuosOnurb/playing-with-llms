# executor.py

from typing import List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from pwllms.agent import AgentState
from pwllms.tools import AVAILABLE_TOOLS, llm


def _execute_plan_steps(steps: List[dict], execution_log: List[str], depth: int = 0):
    indent = "  " * depth
    for step in steps:
        kind = step.get("kind", "note")
        desc = step.get("description", "")
        step_id = step.get("id", "?")

        if kind == "note":
            execution_log.append(f"{indent}- NOTE [{step_id}]: {desc}")
            continue

        if kind == "subplan":
            execution_log.append(f"{indent}- SUBPLAN [{step_id}]: {desc}")
            substeps = step.get("substeps", []) or []
            _execute_plan_steps(substeps, execution_log, depth + 1)
            continue

        if kind == "tool":
            tool_name = step.get("tool_name")
            args = step.get("args", {}) or {}
            execution_log.append(f"{indent}- TOOL [{step_id}]: {tool_name}({args})")

            matching_tool = next(
                (t for t in AVAILABLE_TOOLS if t.name == tool_name), None
            )
            if not matching_tool:
                execution_log.append(f"{indent}  -> ERROR: tool {tool_name} not found")
                continue

            # Reuse the same sanitization logic for known tools
            if tool_name in ("fetch_url", "read_text_file"):
                if "max_chars" in args and args["max_chars"] in (None, "null", ""):
                    args.pop("max_chars")
            if tool_name == "web_search":
                if "max_results" in args and args["max_results"] in (None, "null", ""):
                    args.pop("max_results")

            try:
                result = matching_tool.invoke(args)
                # Keep result short in log
                res_str = str(result)
                if len(res_str) > 300:
                    res_str = res_str[:300] + "... [truncated in log]"
                execution_log.append(f"{indent}  -> RESULT: {res_str}")
            except Exception as e:
                execution_log.append(
                    f"{indent}  -> ERROR running tool {tool_name}: {e}"
                )
            continue

        # Unknown kind
        execution_log.append(f"{indent}- UNKNOWN STEP KIND [{step_id}]: {step}")


def executor_node(state: AgentState) -> AgentState:
    """
    Execute the hierarchical plan stored in state["plan"],
    then produce a final answer using the LLM.
    """
    plan_steps = state.get("plan") or []

    # Get the original goal / latest user message
    goal = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            goal = m.content
            break

    execution_log: List[str] = []
    if not plan_steps:
        execution_log.append("No plan steps found. Nothing executed.")
    else:
        _execute_plan_steps(plan_steps, execution_log, depth=0)

    # Ask LLM to summarize what was done and answer the user
    log_text = "\n".join(execution_log)
    summary_prompt = (
        "You are an assistant that has just executed a hierarchical plan "
        "for the user.\n\n"
        f"User goal:\n{goal}\n\n"
        "Execution log (steps and tool results):\n"
        "--------------------------------------\n"
        f"{log_text}\n\n"
        "Using ONLY the information in the execution log, produce a helpful "
        "final answer for the user. Also briefly summarize the steps you took."
    )

    resp: AIMessage = llm.invoke(
        [
            SystemMessage(content="You summarize execution logs into final answers."),
            HumanMessage(content=summary_prompt),
        ]
    )

    new_state = dict(state)
    new_state["messages"] = state["messages"] + [resp]
    new_state["plan"] = []
    new_state["executing_plan"] = False

    return new_state
