# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The GGUF chat request path lives in llama_chat.py and chat compaction in chat_compaction.py."""

import ast
from pathlib import Path

INFERENCE = Path(__file__).resolve().parent.parent / "core" / "inference"


def _imported_modules(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out |= {f"{node.module}.{alias.name}" for alias in node.names}
        elif isinstance(node, ast.Import):
            out |= {alias.name for alias in node.names}
    return out


def test_the_chat_modules_never_import_the_backend_module():
    """llama_cpp imports llama_chat, which imports chat_compaction: an import back, even inside a
    function, would bring back the circular coupling the split removed."""
    for name in ("llama_chat.py", "chat_compaction.py"):
        assert "core.inference.llama_cpp" not in _imported_modules(INFERENCE / name), name
    assert "core.inference.llama_chat" not in _imported_modules(INFERENCE / "chat_compaction.py")


def test_the_backend_class_does_not_shadow_a_mixin_method():
    """A copy of a chat method left in LlamaCppBackend, say by a merge resolution against a branch
    that still edits it in llama_cpp.py, would silently win over the one in llama_chat.py."""
    from core.inference.llama_chat import LlamaChatMixin
    from core.inference.llama_cpp import LlamaCppBackend

    mixin_methods = {name for name in vars(LlamaChatMixin) if not name.startswith("__")}
    assert mixin_methods == {
        "generate_chat_completion",
        "generate_chat_completion_with_tools",
        "count_chat_tokens",
    }
    assert LlamaChatMixin in LlamaCppBackend.__mro__
    assert not mixin_methods & set(vars(LlamaCppBackend))
    for name in mixin_methods:
        assert getattr(LlamaCppBackend, name) is vars(LlamaChatMixin)[name]
