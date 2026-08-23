"""Smoke test for unsloth.models.llama4.

The module currently only re-exports Llama4 patching from unsloth_studio
(commented out) and carries no local logic, but it had zero test
references at all -- this locks in that it stays importable without
raising, and stays a thin passthrough rather than silently growing
untested logic.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LLAMA4 = REPO_ROOT / "unsloth" / "models" / "llama4.py"


def test_llama4_module_is_importable():
    import unsloth.models.llama4  # noqa: F401


def test_llama4_module_defines_no_untested_logic():
    """If real functions/classes land here, they need their own tests --
    this fails loudly instead of silently shipping untested code."""
    tree = ast.parse(LLAMA4.read_text(encoding = "utf-8"))
    top_level_defs = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert top_level_defs == [], (
        f"unsloth/models/llama4.py now defines {top_level_defs}; "
        "add dedicated tests for this logic and update/remove this guard"
    )
