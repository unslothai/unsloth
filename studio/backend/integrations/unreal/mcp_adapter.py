# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Smooths the calling conventions of Epic's Unreal MCP plugin, which exposes its ~50 toolsets through three meta-tools
(list_toolsets, describe_toolset, call_tool). Models trip on the same few things every time:

- passing the dotted name ``<toolset>.<tool>`` (as describe_toolset prints it) as ``tool_name`` with no ``toolset_name``;
- leaving out optional parameters, which Epic's server still rejects ("needs a default value",
  "is required by the function input schema" for a parameter the schema does not list as required);
- guessing tool or toolset names, which Epic answers with "Unknown tool X" and no list of the real ones.

The first is rewritten before the call; the second is filled from the toolset's own schema (its declared default,
else null) and retried once; the third gets the real names appended to the error so the next call can use them.
"""

import json
import re
import threading
import time
from typing import Callable, Optional

CALL_TOOL = "call_tool"
_SCHEMA_TTL_S = 300.0

_UNKNOWN_TOOL = re.compile(
    r"^Error: (?:Unknown tool (?P<a>\S+)|Tool '(?P<b>[^']+)' not found)", re.M
)
_UNKNOWN_TOOLSET = re.compile(r"^Error: Toolset '(?P<ts>[^']+)' not found", re.M)
_NEEDS_DEFAULT = re.compile(r'input param "(?P<p>[^"]+)" needs a default value')
_MISSING_PARAM = re.compile(
    r'input param "(?P<p>[^"]+)" is required by the function input schema Json, but is missing'
)

# (server url, toolset) -> (fetched at, {tool short name: input schema}); toolset list per server url.
_schemas: dict[tuple[str, str], tuple[float, dict[str, dict]]] = {}
_toolsets: dict[str, tuple[float, list[str]]] = {}
_lock = threading.Lock()

Describe = Callable[
    [str, dict], str
]  # (meta tool name, arguments) -> raw result text from the same server


def prepare(arguments: dict) -> dict:
    """Rewrites a dotted tool_name into toolset_name + tool_name; leaves anything else as it was."""
    if not isinstance(arguments, dict):
        return arguments
    tool = arguments.get("tool_name")
    toolset = arguments.get("toolset_name")
    if not isinstance(tool, str) or "." not in tool:
        return arguments
    if isinstance(toolset, str) and toolset and tool.startswith(toolset + "."):
        return {**arguments, "tool_name": tool[len(toolset) + 1 :]}
    if not toolset:
        head, _, short = tool.rpartition(".")
        if head and short:
            return {**arguments, "toolset_name": head, "tool_name": short}
    return arguments


def _parse_describe(text: str) -> dict[str, dict]:
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return {}
    tools = data.get("tools") if isinstance(data, dict) else None
    out = {}
    for tool in tools or ():
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            out[tool["name"].rpartition(".")[2]] = tool.get("inputSchema") or {}
    return out


def _toolset_schemas(url: str, toolset: str, describe: Describe) -> dict[str, dict]:
    key = (url, toolset)
    with _lock:
        hit = _schemas.get(key)
        if hit and time.monotonic() - hit[0] < _SCHEMA_TTL_S:
            return hit[1]
    schemas = _parse_describe(describe("describe_toolset", {"toolset_name": toolset}))
    if schemas:
        with _lock:
            _schemas[key] = (time.monotonic(), schemas)
    return schemas


def _toolset_names(url: str, describe: Describe) -> list[str]:
    with _lock:
        hit = _toolsets.get(url)
        if hit and time.monotonic() - hit[0] < _SCHEMA_TTL_S:
            return hit[1]
    text = describe("list_toolsets", {})
    names = re.findall(r"^- ([A-Za-z0-9_.]+):", text or "", re.M)
    if names:
        with _lock:
            _toolsets[url] = (time.monotonic(), names)
    return names


def _fill_optional(schema: dict, arguments: dict) -> Optional[dict]:
    """Every property the schema does not require and the call left out: its default, else null. None if nothing to add."""
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return None
    required = set(schema.get("required") or ())
    filled = dict(arguments)
    for name, spec in props.items():
        if name in filled or name in required:
            continue
        filled[name] = spec.get("default") if isinstance(spec, dict) else None
    return filled if len(filled) > len(arguments) else None


def repair(
    url: str, arguments: dict, result: str, describe: Describe
) -> tuple[Optional[dict], str]:
    """For a failed call_tool: (arguments to retry with, or None; text to append to the error, or '')."""
    if (
        not isinstance(result, str)
        or not result.startswith("Error")
        or not isinstance(arguments, dict)
    ):
        return None, ""
    toolset = arguments.get("toolset_name")
    tool = arguments.get("tool_name")
    missing = _NEEDS_DEFAULT.search(result) or _MISSING_PARAM.search(result)
    if missing:
        if isinstance(toolset, str) and isinstance(tool, str):
            schema = _toolset_schemas(url, toolset, describe).get(tool)
            inner = (
                arguments.get("arguments") if isinstance(arguments.get("arguments"), dict) else {}
            )
            filled = _fill_optional(schema or {}, inner)
            # Retry only when the parameter Epic named is one we can supply; a truly required one is the model's to send.
            if filled is not None and missing.group("p") in filled:
                return {**arguments, "arguments": filled}, ""
        return None, ""
    if _UNKNOWN_TOOLSET.search(result):
        names = _toolset_names(url, describe)
        if names:
            return None, "\nToolsets on this server: " + ", ".join(names)
        return None, ""
    if _UNKNOWN_TOOL.search(result) and isinstance(toolset, str) and toolset:
        names = sorted(_toolset_schemas(url, toolset, describe))
        if names:
            return None, f"\nTools in {toolset}: " + ", ".join(names)
    return None, ""


def call(url: str, arguments: dict, invoke: Callable[[dict], str], describe: Describe) -> str:
    """Runs one Unreal call_tool through the fixes above: rewrite, call, then at most one schema-filled retry."""
    arguments = prepare(arguments)
    result = invoke(arguments)
    retry, hint = repair(url, arguments, result, describe)
    if retry is not None:
        return invoke(retry)
    return result + hint if hint else result
