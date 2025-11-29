# 🚀 Local Agent Architecture with Hierarchical Planning

*A LangGraph-based multi-node design for tool-using autonomous agents*

## 1. Overview

This architecture implements a **two-mode local agent**:

1.  **Normal Chat Mode** -- runs the basic LangGraph agent\
2.  **Hierarchical Planner Mode (Level 4)** -- triggered with
    `/plan <task>`

The system is modular: chat mode and planner mode are separate graphs.

## 2. Components

### Tools

-   calculate\
-   list_files\
-   read_text_file\
-   write_text_file\
-   search_text_in_file\
-   fetch_url\
-   web_search

### State

``` python
class AgentState(TypedDict):
    messages: List[BaseMessage]
    plan: List[dict]
    executing_plan: bool
```

## 3. Chat Graph

Two nodes: - agent → LLM with tools\
- tools → executes tool calls

## 4. Hierarchical Planner Graph

### planner_node

Generates a structured JSON plan with: - kind: tool, subplan, note\
- args for tools\
- nested substeps

### executor_node

-   Executes plan recursively\
-   Logs every step\
-   Produces final response via LLM

Flow:

    planner → executor → END

## 5. Example Plan Schema

``` json
{
  "goal": "Create a news report",
  "steps": [
    {
      "id": 1,
      "kind": "subplan",
      "description": "Gather information",
      "substeps": [
        {
          "id": 2,
          "kind": "tool",
          "tool_name": "web_search",
          "args": {"query": "Python latest news", "max_results": 3}
        }
      ]
    }
  ]
}
```

