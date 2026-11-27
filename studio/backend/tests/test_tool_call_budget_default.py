# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every tool loop and request default share one tool-call budget (50) instead of scattered literals."""

import ast
from pathlib import Path

from core.inference.tool_budget import DEFAULT_MAX_TOOL_CALLS_PER_MESSAGE
from models.inference import ChatCompletionRequest

BACKEND = Path(__file__).resolve().parents[1]
BUDGET_NAMES = {"max_tool_iterations", "max_tool_calls_per_message", "max_calls"}


def test_request_default_is_fifty():
    assert DEFAULT_MAX_TOOL_CALLS_PER_MESSAGE == 50
    request = ChatCompletionRequest(messages = [{"role": "user", "content": "hi"}])
    assert request.max_tool_calls_per_message == 50


def _literal_budgets(path: Path) -> list[str]:
    """Integer literals given as a tool budget: a parameter default, keyword argument or `else` fallback."""
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.args + node.args.kwonlyargs
            defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(
                node.args.defaults
            )
            defaults += node.args.kw_defaults
            for arg, default in zip(args, defaults):
                if (
                    arg.arg in BUDGET_NAMES
                    and isinstance(default, ast.Constant)
                    and isinstance(default.value, int)
                ):
                    found.append(f"{path.name}:{node.lineno} def {arg.arg} = {default.value}")
        elif isinstance(node, ast.keyword) and node.arg in BUDGET_NAMES:
            value = node.value
            if isinstance(value, ast.IfExp):
                value = value.orelse
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, int)
                and value.value not in (0,)
            ):
                found.append(f"{path.name}:{node.value.lineno} {node.arg}= {value.value}")
    return found


def test_no_tool_loop_hard_codes_its_own_default():
    sources = [
        BACKEND / "models" / "inference.py",
        BACKEND / "routes" / "inference.py",
        BACKEND / "routes" / "managed_engine_chat.py",
        *(
            BACKEND / "core" / "inference" / name
            for name in (
                "inference.py",
                "llama_cpp.py",
                "orchestrator.py",
                "safetensors_agentic.py",
            )
        ),
    ]
    offenders = [hit for path in sources for hit in _literal_budgets(path)]
    assert offenders == []
