# main.py


from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from rich import box

# --- RICH VISUALIZATION ---
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from pwllms.agent import AgentState
from pwllms.executor import executor_node
from pwllms.graph import build_graph, build_planner_graph, should_use_planner
from pwllms.planner import create_planner_node


def render_file_list(output: str) -> Table:
    """Render file list output as a rich Table."""
    lines = [ln for ln in output.splitlines() if ln.strip()]
    table = Table(title="Directory Listing", box=box.SIMPLE)
    table.add_column("Name", style="bold cyan")
    for name in lines:
        table.add_row(name)
    return table


def render_search_results(output: str) -> Table:
    """Render search results as a rich Table."""
    lines = [ln for ln in output.splitlines() if ln.strip()]
    table = Table(title="Search Results", box=box.SIMPLE)
    table.add_column("Line", style="bold yellow")
    table.add_column("Content", style="white")
    for ln in lines:
        if ln.startswith("Line "):
            parts = ln.split(":", 1)
            if len(parts) == 2:
                table.add_row(parts[0].replace("Line ", ""), parts[1].strip())
            else:
                table.add_row("?", ln)
        else:
            table.add_row("", ln)
    return table


def render_markdown(output: str) -> Markdown:
    """Render markdown content using rich."""
    return Markdown(output)


def render_code(output: str, language: str = "python") -> Syntax:
    """Render code content using rich syntax highlighting."""
    return Syntax(output, language, theme="monokai", line_numbers=True)


def main():
    chat_graph = build_graph()
    planner_graph = build_planner_graph(
        planner_node=create_planner_node, executor_node=executor_node
    )

    console = Console()
    console.print(
        Panel(
            "[bold cyan]LangGraph agent with internet search and planner mode.[/bold cyan]",
            box=box.ROUNDED,
        )
    )
    console.print("[bold]Type 'exit' to quit.[/bold]")
    console.print("[bold]Use '/plan <task>' to invoke hierarchical planning.[/bold]")
    console.print(
        "[bold]Use '/show-plan <task>' to preview a plan, then '/exec' to execute it.[/bold]\n"
    )

    system_prompt = (
        "You are an agent, much like Jarvis from Iron Man.\n"
        "You have access to tools: calculate, list_files, read_text_file, "
        "write_text_file, search_text_in_file, fetch_url, web_search.\n"
        "Only call tools when the user explicitly asks you to perform an action that "
        "requires them (doing math, inspecting directories or files, reading/writing "
        "local files, or accessing web pages / web search).\n"
        "For greetings, general chit-chat, or simple questions that do not require "
        "file or web access, answer directly and DO NOT call any tools.\n"
        "Use web_search when the user clearly asks you to look something up on the "
        "internet or wants up-to-date information; then, if needed, call fetch_url on "
        "one of the returned URLs for more detail.\n"
        "When you use tools, always tell the user clearly what you did and summarize results."
    )

    # Chat state
    chat_state: AgentState = {
        "messages": [SystemMessage(content=system_prompt)],
        "plan": [],
        "executing_plan": False,
    }

    # Stored plan for interactive review
    pending_plan_state: AgentState | None = None

    while True:
        user = console.input("[bold magenta]You:[/bold magenta] ")
        cmd = user.strip().lower()

        if cmd in {"exit", "quit"}:
            break

        # ---- SHOW PLAN (INTERACTIVE REVIEW) ----
        if cmd.startswith("/show-plan"):
            task = user[len("/show-plan") :].strip()
            if not task:
                console.print(
                    "[red]Agent: Please provide a task after /show-plan[/red]\n"
                )
                continue

            # Create plan state
            pending_plan_state = {
                "messages": [
                    SystemMessage(content="You are a planning agent."),
                    HumanMessage(content=task),
                ],
                "plan": [],
                "executing_plan": False,
            }

            # Run only the planner node
            pending_plan_state = create_planner_node(pending_plan_state)

            # Extract goal from messages
            goal = task
            for m in pending_plan_state["messages"]:
                if isinstance(m, HumanMessage):
                    goal = m.content
                    break

            # Pretty-print the plan using rich Tree
            plan = pending_plan_state.get("plan", [])
            tree = Tree(
                f'[bold green]Plan for goal:[/bold green] "{goal}"',
                guide_style="bold bright_blue",
            )

            def add_plan_step(tree_node, step):
                kind = step.get("kind", "note")
                desc = step.get("description", "")
                step_id = step.get("id", "?")
                if kind == "subplan":
                    subnode = tree_node.add(
                        f"[yellow]{step_id}. SUBPLAN:[/yellow] {desc}"
                    )
                    substeps = step.get("substeps", [])
                    for sub in substeps:
                        add_plan_step(subnode, sub)
                elif kind == "tool":
                    tool_name = step.get("tool_name", "?")
                    args = step.get("args", {})
                    subnode = tree_node.add(
                        f"[cyan]{step_id}. TOOL:[/cyan] [bold]{tool_name}[/bold] [white]{args}[/white]"
                    )
                    if desc:
                        subnode.add(f"[dim]→ {desc}[/dim]")
                elif kind == "note":
                    tree_node.add(f"[magenta]{step_id}. NOTE:[/magenta] {desc}")
                else:
                    tree_node.add(f"[red]{step_id}. UNKNOWN:[/red] {step}")

            for step in plan:
                add_plan_step(tree, step)

            console.print(tree)
            console.print(
                Panel(
                    "Type '/exec' to execute this plan, or '/show-plan <new task>' to create another plan.",
                    style="bold white",
                    box=box.ROUNDED,
                )
            )
            continue

        # ---- EXECUTE PLAN ----
        if cmd in {"/exec", "/run-plan"}:
            if not pending_plan_state or not pending_plan_state.get("plan"):
                console.print(
                    "[red]Agent: No plan in memory. Use /show-plan <task> first.[/red]\n"
                )
                continue

            # Execute the stored plan
            final_state = executor_node(pending_plan_state)

            # Print final AIMessage
            last_ai = None
            for m in reversed(final_state["messages"]):
                if isinstance(m, AIMessage):
                    last_ai = m
                    break

            if last_ai is None:
                console.print(
                    Panel("Agent (executor): (no answer)", style="red", box=box.ROUNDED)
                )
            else:
                console.print(
                    Panel(
                        last_ai.content,
                        title="Agent (executor)",
                        style="bold green",
                        box=box.ROUNDED,
                    )
                )

            # Clear the pending plan
            pending_plan_state = None
            continue

        # ---- PLANNER MODE (original /plan command) ----
        if cmd.startswith("/plan"):
            task = user[len("/plan") :].strip()
            if not task:
                console.print("[red]Agent: Please provide a task after /plan[/red]\n")
                continue

            planner_state: AgentState = {
                "messages": [
                    SystemMessage(content="You are a planning agent."),
                    HumanMessage(content=task),
                ],
                "plan": [],
                "executing_plan": False,
            }

            final_state = planner_graph.invoke(planner_state)
            # Print final AIMessage
            last_ai = None
            for m in reversed(final_state["messages"]):
                if isinstance(m, AIMessage):
                    last_ai = m
                    break

            if last_ai is None:
                console.print(
                    Panel("Agent (planner): (no answer)", style="red", box=box.ROUNDED)
                )
            else:
                console.print(
                    Panel(
                        last_ai.content,
                        title="Agent (planner)",
                        style="bold blue",
                        box=box.ROUNDED,
                    )
                )

            continue

        # ---- AUTO PLANNER INSIDE NORMAL CHAT ----
        if should_use_planner(user):
            # Build a fresh planner state for this request
            planner_state: AgentState = {
                "messages": [
                    SystemMessage(content="You are a planning agent."),
                    HumanMessage(content=user),
                ],
                "plan": [],
                "executing_plan": False,
            }

            final_state = planner_graph.invoke(planner_state)

            # Extract final AI answer from planner+executor
            last_ai = None
            for m in reversed(final_state["messages"]):
                if isinstance(m, AIMessage):
                    last_ai = m
                    break

            if last_ai is None or not str(getattr(last_ai, "content", "")).strip():
                console.print(
                    Panel("Agent (planner): (no answer)", style="red", box=box.ROUNDED)
                )
            else:
                console.print(
                    Panel(
                        last_ai.content,
                        title="Agent (planner)",
                        style="bold blue",
                        box=box.ROUNDED,
                    )
                )

            # OPTIONAL: also append this interaction into main chat history
            chat_state["messages"].append(HumanMessage(content=user))
            if last_ai is not None:
                chat_state["messages"].append(last_ai)

            continue

        # ---- NORMAL CHAT MODE ----
        prev_len = len(chat_state["messages"])
        chat_state["messages"].append(HumanMessage(content=user))

        chat_state = chat_graph.invoke(chat_state)

        new_messages = chat_state["messages"][prev_len:]

        for m in new_messages:
            if isinstance(m, ToolMessage):
                full = m.content or ""
                # Detect tool type from output
                preview = None
                if full.startswith("SUCCESS: wrote") or full.startswith("ERROR:"):
                    preview = full
                elif full.startswith("Top search results:"):
                    # DuckDuckGo web_search
                    table = Table(title="Web Search Results", box=box.SIMPLE)
                    table.add_column("Result", style="bold green")
                    for res in full.split("\n\n")[1:]:
                        table.add_row(res.strip())
                    preview = table
                elif full.startswith("No matches for") or full.startswith(
                    "Error searching file:"
                ):
                    preview = full
                elif full.startswith("Line "):
                    preview = render_search_results(full)
                elif full.startswith("SUCCESS: fetched"):
                    # If content is markdown, render as markdown
                    if "CONTENT_START" in full:
                        content = full.split("CONTENT_START", 1)[-1].strip()
                        if (
                            content.startswith("#")
                            or content.startswith("-")
                            or "\n#" in content
                        ):
                            preview = render_markdown(content)
                        elif content.startswith("import ") or content.startswith(
                            "def "
                        ):
                            preview = render_code(content)
                        else:
                            preview = content
                    else:
                        preview = full
                elif full.startswith("No search results found") or full.startswith(
                    "ERROR: web_search"
                ):
                    preview = full
                elif full.startswith("SUCCESS: wrote"):
                    preview = full
                elif full and all("/" not in ln for ln in full.splitlines()):
                    # Likely a file list
                    preview = render_file_list(full)
                else:
                    preview = full
                console.print(
                    Panel(
                        preview,
                        title=f"[TOOL] {m.tool_call_id}",
                        style="bold magenta",
                        box=box.ROUNDED,
                    )
                )

        last_ai = None
        for m in reversed(chat_state["messages"]):
            if isinstance(m, AIMessage):
                last_ai = m
                break

        if last_ai is None or not str(getattr(last_ai, "content", "")).strip():
            console.print(Panel("Agent: (no answer)", style="red", box=box.ROUNDED))
        else:
            console.print(
                Panel(
                    last_ai.content, title="Agent", style="bold cyan", box=box.ROUNDED
                )
            )


if __name__ == "__main__":
    main()
