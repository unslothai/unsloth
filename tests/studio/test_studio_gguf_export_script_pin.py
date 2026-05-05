"""Tests for Studio GGUF export pinning convert_hf_to_gguf.py via
UNSLOTH_LLAMA_CPP_SCRIPTS_DIR with graceful fallback when unsloth_zoo
lacks the local-script resolver.

Verifies:
  - export.py imports LLAMA_CPP_DEFAULT_DIR and _resolve_local_convert_script
    from unsloth_zoo.llama_cpp inside a single try/except ImportError so a
    zoo missing either symbol degrades to a warning instead of crashing.
  - os.environ.setdefault("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", LLAMA_CPP_DEFAULT_DIR)
    is called inside the try; setdefault preserves explicit user overrides
    and assigns the default when unset.
  - The compatibility warning is gated on a module-level flag so it fires
    once per process rather than on every export call.
"""

from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "core"
    / "export"
    / "export.py"
)
SRC = SOURCE_PATH.read_text()
TREE = ast.parse(SRC)


def _module_level_assignments(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    yield target.id, node.value


def _find_pin_try(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.ImportFrom)
                and stmt.module == "unsloth_zoo.llama_cpp"
                and any(
                    alias.name == "_resolve_local_convert_script"
                    for alias in stmt.names
                )
            ):
                return node
    return None


def test_warning_flag_defined_at_module_scope():
    flags = {
        name: value
        for name, value in _module_level_assignments(TREE)
        if name == "_LLAMA_CPP_SCRIPTS_WARNING_EMITTED"
    }
    assert flags, "expected module-level _LLAMA_CPP_SCRIPTS_WARNING_EMITTED flag"
    init = flags["_LLAMA_CPP_SCRIPTS_WARNING_EMITTED"]
    assert isinstance(init, ast.Constant) and init.value is False


def test_constant_and_resolver_imported_in_same_try():
    try_node = _find_pin_try(TREE)
    assert try_node is not None
    imported = []
    for stmt in try_node.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "unsloth_zoo.llama_cpp":
            imported.extend(alias.name for alias in stmt.names)
    assert "LLAMA_CPP_DEFAULT_DIR" in imported
    assert "_resolve_local_convert_script" in imported


def test_setdefault_inside_try_block():
    try_node = _find_pin_try(TREE)
    assert try_node is not None
    setdefault_calls = []
    for stmt in try_node.body:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"
            ):
                setdefault_calls.append(node)
    assert setdefault_calls
    second = setdefault_calls[0].args[1]
    assert isinstance(second, ast.Name) and second.id == "LLAMA_CPP_DEFAULT_DIR"


def test_warning_handler_gated_on_module_flag():
    try_node = _find_pin_try(TREE)
    assert try_node is not None
    handlers = [
        h
        for h in try_node.handlers
        if isinstance(h.type, ast.Name) and h.type.id == "ImportError"
    ]
    assert handlers
    handler = handlers[0]
    flag_reads = []
    flag_writes = []
    warning_calls = []
    for node in ast.walk(ast.Module(body = handler.body, type_ignores = [])):
        if (
            isinstance(node, ast.Name)
            and node.id == "_LLAMA_CPP_SCRIPTS_WARNING_EMITTED"
        ):
            if isinstance(node.ctx, ast.Load):
                flag_reads.append(node)
            elif isinstance(node.ctx, ast.Store):
                flag_writes.append(node)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "warning"
        ):
            warning_calls.append(node)
    assert flag_reads
    assert flag_writes
    assert warning_calls
    msg = ast.dump(warning_calls[0])
    assert "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" in msg
    assert "unsloth_zoo" in msg


def test_default_dir_is_string_for_setdefault_compat():
    from unsloth_zoo.llama_cpp import LLAMA_CPP_DEFAULT_DIR

    assert isinstance(LLAMA_CPP_DEFAULT_DIR, str)


def test_setdefault_preserves_explicit_user_override(monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", "/explicit/override")
    from unsloth_zoo.llama_cpp import LLAMA_CPP_DEFAULT_DIR

    os.environ.setdefault("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", LLAMA_CPP_DEFAULT_DIR)
    assert os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] == "/explicit/override"


def test_setdefault_assigns_default_when_unset(monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    from unsloth_zoo.llama_cpp import LLAMA_CPP_DEFAULT_DIR

    os.environ.setdefault("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", LLAMA_CPP_DEFAULT_DIR)
    assert os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] == LLAMA_CPP_DEFAULT_DIR


def _simulate_pin_block(emit_records, set_value):
    fake = types.ModuleType("unsloth_zoo.llama_cpp")
    if set_value is not None:
        fake.LLAMA_CPP_DEFAULT_DIR = set_value
    sys.modules["unsloth_zoo.llama_cpp"] = fake

    state = {"emitted": False}

    def run_once():
        try:
            from unsloth_zoo.llama_cpp import (
                LLAMA_CPP_DEFAULT_DIR,
                _resolve_local_convert_script,  # noqa: F401
            )

            os.environ.setdefault(
                "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", LLAMA_CPP_DEFAULT_DIR
            )
        except ImportError:
            if not state["emitted"]:
                emit_records.append("warned")
                state["emitted"] = True

    return run_once


def test_warning_fires_at_most_once_across_calls(monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    emits = []
    runner = _simulate_pin_block(emits, set_value = "/fake/default")
    runner()
    runner()
    runner()
    assert emits == ["warned"]


def test_missing_default_dir_degrades_to_warning(monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    emits = []
    runner = _simulate_pin_block(emits, set_value = None)
    runner()
    assert emits == ["warned"]
    assert "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" not in os.environ


def test_no_warning_when_both_symbols_present(monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    fake = types.ModuleType("unsloth_zoo.llama_cpp")
    fake.LLAMA_CPP_DEFAULT_DIR = "/fake/dir"
    fake._resolve_local_convert_script = lambda: None
    monkeypatch.setitem(sys.modules, "unsloth_zoo.llama_cpp", fake)

    emits = []
    state = {"emitted": False}
    try:
        from unsloth_zoo.llama_cpp import (
            LLAMA_CPP_DEFAULT_DIR,
            _resolve_local_convert_script,  # noqa: F401
        )

        os.environ.setdefault("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", LLAMA_CPP_DEFAULT_DIR)
    except ImportError:
        if not state["emitted"]:
            emits.append("warned")
            state["emitted"] = True

    assert emits == []
    assert os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR") == "/fake/dir"
