# tools.py

import os
from typing import Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

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


BASE_DIR = os.path.abspath(".")  # restrict file access under this directory


def _safe_join(path: str) -> str:
    """
    Resolve path safely under BASE_DIR to avoid writing/reading outside.
    """
    joined = os.path.abspath(os.path.join(BASE_DIR, path))
    if not joined.startswith(BASE_DIR):
        raise ValueError("Access outside of base directory is not allowed.")
    return joined


@tool
def read_text_file(path: str, max_chars: Optional[int] = 4000) -> str:
    """
    Read a text file (UTF-8) under the project directory.

    Args:
        path: Relative path to the file.
        max_chars: Max characters to return (to avoid huge dumps).
    """
    try:
        if max_chars is None:
            max_chars = 4000

        full_path = _safe_join(path)
        with open(full_path, "r", encoding="utf-8") as f:
            data = f.read(max_chars)
        if len(data) == max_chars:
            return data + "\n\n[Truncated output]"
        return data
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_text_file(path: str, content: str, mode: str = "overwrite") -> str:
    """
    Write text to a file under the project directory.

    Args:
        path: Relative path to the file.
        content: Text content to write.
        mode: 'overwrite' or 'append'.
    """
    try:
        full_path = _safe_join(path)
        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)

        write_mode = "w" if mode == "overwrite" else "a"
        with open(full_path, write_mode, encoding="utf-8") as f:
            f.write(content)

        # Short, explicit, and something the LLM can copy/paraphrase
        return (
            f"SUCCESS: wrote {len(content)} characters to file '{path}' (mode={mode})."
        )
    except Exception as e:
        return f"ERROR: failed to write file '{path}': {e}"


@tool
def search_text_in_file(path: str, query: str, max_matches: int = 5) -> str:
    """
    Search for a text query inside a file and return matching lines.

    Args:
        path: Relative path to the file.
        query: Substring to search for.
        max_matches: Maximum number of matching lines to return.
    """
    try:
        full_path = _safe_join(path)
        matches = []
        with open(full_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if query in line:
                    matches.append(f"Line {i}: {line.rstrip()}")
                    if len(matches) >= max_matches:
                        break
        if not matches:
            return f"No matches for '{query}' in {path}."
        return "\n".join(matches)
    except Exception as e:
        return f"Error searching file: {e}"


@tool
def fetch_url(url: str, max_chars: Optional[int] = 8000) -> str:
    """
    Fetch a web page via HTTP GET and return cleaned text content.
    """
    try:
        if max_chars is None:
            max_chars = 8000

        if not (url.startswith("http://") or url.startswith("https://")):
            return "ERROR: URL must start with http:// or https://"

        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "local-agent/0.1"},
        )
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        text = resp.text

        if "html" in content_type.lower():
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            page_text = soup.get_text(separator="\n")
        else:
            page_text = text

        page_text = page_text.strip()
        if not page_text:
            return f"SUCCESS: fetched '{url}', but no text content was found."

        truncated = False
        if len(page_text) > max_chars:
            page_text = page_text[:max_chars]
            truncated = True

        header = f"SUCCESS: fetched '{url}'. Length={len(page_text)} characters."
        if truncated:
            header += " Content truncated to max_chars."

        return header + "\n\nCONTENT_START\n" + page_text
    except Exception as e:
        return f"ERROR: failed to fetch URL '{url}': {e}"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo's HTML results page and return top links.

    Args:
        query: Search query text.
        max_results: Maximum number of results to return.
    """
    try:
        if max_results <= 0:
            max_results = 5

        resp = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "local-agent/0.1"},
            timeout=10,
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        # DuckDuckGo's HTML layout can change; this selector works for the current classic HTML
        for res in soup.select(".result"):
            title_tag = res.select_one("a.result__a") or res.select_one("a[href]")
            snippet_tag = res.select_one(".result__snippet")

            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            url = title_tag.get("href", "").strip()
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            if not url:
                continue

            results.append(f"- {title}\n  {url}\n  {snippet}")
            if len(results) >= max_results:
                break

        if not results:
            return f"No search results found for query: '{query}'."

        return "Top search results:\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"ERROR: web_search failed for query '{query}': {e}"


llm = ChatOpenAI(
    model="llama3.2:3b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # dummy, required by client
    temperature=0,
)


AVAILABLE_TOOLS = [
    calculate,
    list_files,
    read_text_file,
    write_text_file,
    search_text_in_file,
    fetch_url,
    web_search,
]

LLM_WITH_TOOLS = llm.bind_tools(AVAILABLE_TOOLS)
