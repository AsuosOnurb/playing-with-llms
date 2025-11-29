import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict

from openai import OpenAI

# ---------- LLM CLIENT ----------

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# ---------- TOOLS ----------


def tool_calculate(expression: str) -> str:
    """Safely evaluate a simple math expression, like '2 + 2 * 3'."""
    # VERY simple sandbox: only allow digits, + - * / ( ) and spaces
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Expression contains invalid characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def tool_list_files(path: str) -> str:
    """List files in a directory."""
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {e}"


@dataclass
class Tool:
    name: str
    description: str
    arg_schema: Dict[str, Any]
    func: Callable[..., str]


TOOLS: Dict[str, Tool] = {
    "calculate": Tool(
        name="calculate",
        description="Evaluate basic arithmetic expressions like '2 + 2 * 3'.",
        arg_schema={"expression": "string, required"},
        func=lambda args: tool_calculate(args["expression"]),
    ),
    "list_files": Tool(
        name="list_files",
        description="List files in a directory.",
        arg_schema={"path": "string, required, directory path"},
        func=lambda args: tool_list_files(args["path"]),
    ),
}

TOOLS_JSON_SPEC = [
    {
        "name": t.name,
        "description": t.description,
        "args": t.arg_schema,
    }
    for t in TOOLS.values()
]


# ---------- CORE AGENT LOOP ----------

SYSTEM_PROMPT = f"""
You are a helpful local agent with access to tools.

You MUST respond in one of two JSON formats ONLY (no extra text):

1) To answer directly (no tool needed):
{{
  "type": "final",
  "answer": "your answer here"
}}

2) To call a tool:
{{
  "type": "tool_call",
  "tool_name": "<one of: {list(TOOLS.keys())}>",
  "tool_args": {{ ... arguments as JSON ... }}
}}

Available tools (with arg hints) are:
{json.dumps(TOOLS_JSON_SPEC, indent=2)}
""".strip()


def call_llm(messages):
    resp = client.chat.completions.create(
        model="llama3.2:3b",
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content


def run_agent(user_input: str) -> str:
    # 1) Ask LLM what to do (final answer or tool_call)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    raw = call_llm(messages)

    # 2) Parse JSON
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        return f"Model did not return valid JSON: {raw}"

    if decision.get("type") == "final":
        return decision.get("answer", "")

    if decision.get("type") == "tool_call":
        tool_name = decision.get("tool_name")
        tool_args = decision.get("tool_args", {})

        tool = TOOLS.get(tool_name)
        if tool is None:
            return f"Unknown tool requested: {tool_name}"

        # 3) Execute tool
        tool_result = tool.func(tool_args)

        # 4) Ask model to produce final answer using tool_result
        followup_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    f"Original question: {user_input}\n"
                    f"Tool used: {tool_name}\n"
                    f"Tool result:\n{tool_result}\n\n"
                    "Now give the final answer to the user, using this info."
                ),
            },
        ]
        final_answer = call_llm(followup_messages)
        return final_answer

    return f"Unknown decision type: {decision}"


if __name__ == "__main__":
    print("Simple local agent. Type 'exit' to quit.\n")
    while True:
        user = input("You: ")
        if user.strip().lower() in {"exit", "quit"}:
            break
        answer = run_agent(user)
        print(f"Agent: {answer}\n")
