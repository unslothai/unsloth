# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Tool definitions and executors for LLM tool calling.

Currently supports web search via DuckDuckGo (ddgs package, no API key needed).
"""

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information, recent events, or facts you are uncertain about.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
}

ALL_TOOLS = [WEB_SEARCH_TOOL]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with the given arguments. Returns result as a string."""
    if name == "web_search":
        return _web_search(arguments.get("query", ""))
    return f"Unknown tool: {name}"


def _web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return formatted results."""
    if not query.strip():
        return "No query provided."
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results = max_results)
        if not results:
            return "No results found."
        parts = []
        for r in results:
            parts.append(
                f"Title: {r.get('title', '')}\n"
                f"URL: {r.get('href', '')}\n"
                f"Snippet: {r.get('body', '')}"
            )
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Search failed: {e}"
