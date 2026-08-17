# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Load the GRPO hidden-states forward wrapper out of ``unsloth/models/rl.py``.

Same trick as ``_grpo_dispatch_source``: lift the module-level defs with ``ast``
rather than importing ``unsloth``, so the tests stay CPU-only and import-free
while still tracking the shipped code.
"""

from __future__ import annotations

import ast
import collections
import inspect
import logging
import os
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"

WRAPPER_NAMES = (
    "_module_returns_logits",
    "_grpo_hidden_states_wrap_target",
    "_model_supports_unsloth_return_hidden_states",
    "_drop_forward_kwargs_consumed_positionally",
    "_get_num_logits_to_keep",
    "_warn_grpo_hidden_states_fallback_once",
    "_note_grpo_hidden_states_success",
    "_replace_outputs_logits",
    "_minimise_logits_kwarg",
    "_drop_spare_hidden_states",
    "_install_grpo_hidden_states_forward_wrapper",
)

# present only once the per-call degradation fix has landed
OPTIONAL_NAMES = ("_note_grpo_hidden_states_success",)

CONSTANT_NAMES = (
    "_UNSLOTH_RETURN_HIDDEN_STATES_SUPPORT_MARKER",
    "_UNSLOTH_GRPO_HIDDEN_STATES_WRAPPED_ATTR",
    "_UNSLOTH_GRPO_HIDDEN_STATES_WARNING_ATTR",
    "_UNSLOTH_GRPO_HIDDEN_STATES_DEGRADED_ATTR",
)


def load_rl_wrapper(names = WRAPPER_NAMES):
    """Return ``{name: object}`` for the wrapper helpers, exec'd from live source."""
    text = SOURCE_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(text, filename = str(SOURCE_PATH))

    wanted = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            wanted.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in CONSTANT_NAMES for t in node.targets
        ):
            wanted.append(node)

    found = {
        node.name for node in wanted if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = set(names) - found - set(OPTIONAL_NAMES)
    names = tuple(name for name in names if name in found)
    if missing:
        raise AssertionError(f"missing module-level defs in {SOURCE_PATH}: {sorted(missing)}")

    namespace: dict = {
        "os": os,
        "collections": collections,
        "inspect": inspect,
        "logger": logging.getLogger("unsloth-repro"),
    }
    exec(compile(ast.Module(body = wanted, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return {name: namespace[name] for name in names}
