# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A runtime field must not be passed twice into an inference response (#8007).

`_llama_runtime_fields()` returns one entry per field on `_InferenceRuntimeFields`, and
callers splat it. Passing any of those same names as an explicit keyword in the same call
is `TypeError: got multiple values for keyword argument`, whatever the two values are, so
pinning the duplicate to None does not help -- the key has to be absent from the dict.

#8007 added `chat_template_override` to `_InferenceRuntimeFields` while `get_status` was
already passing it explicitly. Every `/api/inference/status` poll with a GGUF loaded then
raised, the chat UI showed "Failed to get status", and the model picker rendered empty.

Checked at source level: importing the routes module pulls in the whole studio stack, and
this is a call-shape property that AST can see directly.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_PATH = REPO_ROOT / "studio" / "backend" / "models" / "inference.py"
ROUTE_PATH = REPO_ROOT / "studio" / "backend" / "routes" / "inference.py"

SPLAT_HELPER = "_llama_runtime_fields"
RUNTIME_MODEL = "_InferenceRuntimeFields"


def _runtime_field_names() -> set[str]:
    """Field names declared on `_InferenceRuntimeFields`, read without importing it."""
    tree = ast.parse(MODELS_PATH.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == RUNTIME_MODEL:
            # Direct BaseModel subclass, so its own AnnAssign nodes are the whole field set.
            bases = {ast.unparse(b) for b in node.bases}
            assert bases == {"BaseModel"}, (
                f"{RUNTIME_MODEL} now inherits from {sorted(bases)}; this test reads only its "
                "own annotated fields and would miss inherited ones"
            )
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    raise AssertionError(f"{RUNTIME_MODEL} not found in {MODELS_PATH}")


def _splat_call_sites() -> list[tuple[int, set[str]]]:
    """(line, explicit keyword names) for every call that splats the runtime-fields helper."""
    tree = ast.parse(ROUTE_PATH.read_text(encoding = "utf-8"))
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        splats = any(
            kw.arg is None
            and isinstance(kw.value, ast.Call)
            and isinstance(kw.value.func, ast.Name)
            and kw.value.func.id == SPLAT_HELPER
            for kw in node.keywords
        )
        if splats:
            sites.append((node.lineno, {kw.arg for kw in node.keywords if kw.arg is not None}))
    return sites


def test_runtime_fields_are_not_also_passed_explicitly():
    """No call may splat the runtime fields and name one of them as a keyword too."""
    runtime_fields = _runtime_field_names()
    sites = _splat_call_sites()
    assert sites, f"no `**{SPLAT_HELPER}(...)` call sites found; did the helper get renamed?"

    collisions = {
        line: sorted(explicit & runtime_fields)
        for line, explicit in sites
        if explicit & runtime_fields
    }
    assert not collisions, (
        f"{ROUTE_PATH.name} splats **{SPLAT_HELPER}(...) and passes the same field(s) "
        f"explicitly, which raises TypeError at call time: "
        + "; ".join(f"line {line}: {names}" for line, names in sorted(collisions.items()))
        + f". Assign into the dict before splatting instead of adding a second keyword."
    )


def test_chat_template_override_is_a_runtime_field():
    """Guards the premise: if this moves off the model, the regression above changes shape."""
    assert "chat_template_override" in _runtime_field_names()
