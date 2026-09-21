# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Load the GRPO hidden-states dispatch helpers straight out of the live source.

``_unsloth_grpo_returns_hidden_states`` and ``_unsloth_grpo_hidden_states_signal``
are shipped to the generated trainer as text (``RL_PRE_ITEMS``), so the tests
that ``exec`` a block of ``_get_per_token_logps_and_entropies`` need them in the
namespace exactly as the generated module would have them.

Lifting them with ``ast`` instead of importing ``unsloth`` keeps these tests
CPU-only and import-free, and keeps them tracking the shipped code rather than a
copy of it.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"

HELPER_NAMES = (
    "_unsloth_grpo_returns_hidden_states",
    "_unsloth_grpo_hidden_states_signal",
)


def load_dispatch_helpers():
    """Return ``{name: function}`` for the helpers, exec'd from the live source."""
    text = SOURCE_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(text, filename = str(SOURCE_PATH))
    wanted = []
    for name in HELPER_NAMES:
        found = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        ]
        if len(found) != 1:
            raise AssertionError(
                f"expected exactly one module-level def {name} in {SOURCE_PATH}, found {len(found)}"
            )
        wanted.append(found[0])

    namespace: dict = {}
    exec(compile(ast.Module(body = wanted, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return {name: namespace[name] for name in HELPER_NAMES}


def load_padded_loop_source():
    """Dedented source of the padded logprob loop, located structurally.

    The one ``with`` statement inside ``_get_per_token_logps_and_entropies``
    whose direct body holds ``for ... in zipped_inputs``. No text search, so a
    comment quoting the same code cannot match.
    """
    text = SOURCE_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(text, filename = str(SOURCE_PATH))
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_get_per_token_logps_and_entropies"
    ]
    if len(functions) != 1:
        raise AssertionError(
            f"expected exactly one def _get_per_token_logps_and_entropies, found {len(functions)}"
        )
    loops = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.With)
        and any(
            isinstance(stmt, ast.For)
            and isinstance(stmt.iter, ast.Name)
            and stmt.iter.id == "zipped_inputs"
            for stmt in node.body
        )
    ]
    if len(loops) != 1:
        raise AssertionError(f"expected exactly one padded loop, found {len(loops)}")
    segment = ast.get_source_segment(text, loops[0], padded = True)
    if segment is None:
        raise AssertionError("could not recover the padded-loop source segment")
    return textwrap.dedent(segment)
