"""Unsloth GGUF export pins convert_hf_to_gguf.py for the conversion and takes the pin
back afterwards, with a once-per-process warning fallback when unsloth_zoo lacks the
local-script resolver.

The pin used to be an ``os.environ.setdefault`` that was never unwound. unsloth_zoo reads
UNSLOTH_LLAMA_CPP_SCRIPTS_DIR as the user's own choice: it outranks
UNSLOTH_LLAMA_CPP_CONVERTER_TAG, and it exempts the converter from the
UNSLOTH_CONVERTER_SCAN_STRICT refusal. Neither is true of a directory Studio pinned for
its own routing, so the pin now goes through unsloth_zoo's internal_scripts_dir_pin and is
scoped to the conversion.

The behaviour cases below execute the real helper, lifted out of export.py with ast, rather
than a copy written here: a hand-written copy of the block passes whatever the block does.
"""

from __future__ import annotations

import ast
import contextlib
import os
import sys
import threading
import types
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[2] / "studio" / "backend" / "core" / "export" / "export.py"
)
SRC = SOURCE_PATH.read_text(encoding = "utf-8")
TREE = ast.parse(SRC)
SCRIPTS_DIR = "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"
CONVERTER_TAG = "UNSLOTH_LLAMA_CPP_CONVERTER_TAG"
PIN_HELPER = "_llama_cpp_scripts_pin"


def _module_level_assignments(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    yield target.id, node.value


def _pin_helper_node(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == PIN_HELPER:
            return node
    return None


def _find_pin_try(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.ImportFrom)
                and stmt.module == "unsloth_zoo.llama_cpp"
                and any(alias.name == "_resolve_local_convert_script" for alias in stmt.names)
            ):
                return node
    return None


# The pin catches Exception, not ImportError: a half-built unsloth_zoo raises RuntimeError or
# AttributeError too. Anything that still catches an ImportError counts, so widening the handler
# again does not break this test.
_CATCHES_IMPORT_ERROR = ("ImportError", "Exception", "BaseException")


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in _CATCHES_IMPORT_ERROR for n in names)


# A half-built unsloth_zoo imports and then raises RuntimeError or AttributeError, which ImportError alone does not
# cover.
_CATCHES_EVERYTHING = ("Exception", "BaseException")


def _covers_half_built_zoo(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    caught = {n.id for n in names if isinstance(n, ast.Name)}
    return bool(caught & set(_CATCHES_EVERYTHING)) or {"RuntimeError", "AttributeError"} <= caught


def _load_pin_helper(*, is_mlx = False, logger = None):
    """The real helper, executed on its own so these cases run the shipped code."""
    node = _pin_helper_node(TREE)
    assert node is not None, f"expected a {PIN_HELPER} helper in export.py"
    namespace = {
        "os": os,
        "contextlib": contextlib,
        "logger": logger if logger is not None else types.SimpleNamespace(
            warning = lambda *args, **kwargs: None
        ),
        "_LLAMA_CPP_SCRIPTS_WARNING_EMITTED": False,
        "_IS_MLX": is_mlx,
    }
    exec(
        compile(ast.Module(body = [node], type_ignores = []), str(SOURCE_PATH), "exec"),
        namespace,
    )
    return namespace[PIN_HELPER], namespace


def _install_fake_zoo(monkeypatch, *, default_dir = "/fake/llama.cpp", resolver = True,
                      internal_pin = True, incomplete = False):
    """A stand-in unsloth_zoo.llama_cpp whose pin is as non-reentrant as the real one."""
    calls = {"internal": []}
    fake = types.ModuleType("unsloth_zoo.llama_cpp")
    if default_dir is not None:
        fake.LLAMA_CPP_DEFAULT_DIR = default_dir
    if resolver:
        fake._resolve_local_convert_script = lambda *args, **kwargs: None
    fake._converter_dir_is_incomplete = lambda folder: incomplete
    held = threading.Lock()

    @contextlib.contextmanager
    def _internal_scripts_dir_pin(folder):
        if not held.acquire(blocking = False):
            raise RuntimeError("internal_scripts_dir_pin re-entered: the real one deadlocks here")
        calls["internal"].append(folder)
        existing = os.environ.get(SCRIPTS_DIR)
        if existing is None:
            os.environ[SCRIPTS_DIR] = folder
        try:
            yield
        finally:
            if existing is None:
                os.environ.pop(SCRIPTS_DIR, None)
            else:
                os.environ[SCRIPTS_DIR] = existing
            held.release()

    if internal_pin:
        fake.internal_scripts_dir_pin = _internal_scripts_dir_pin
    package = types.ModuleType("unsloth_zoo")
    monkeypatch.setitem(sys.modules, "unsloth_zoo", package)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.llama_cpp", fake)
    monkeypatch.delenv(SCRIPTS_DIR, raising = False)
    monkeypatch.delenv(CONVERTER_TAG, raising = False)
    return fake, calls


# ---------------------------------------------------------------- source contract


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


def test_the_pin_is_never_left_in_the_environment():
    """No setdefault, and every direct write is answered by a pop in a finally. The
    setdefault this replaced is what made the converter tag inert and turned
    UNSLOTH_CONVERTER_SCAN_STRICT into a warning for every Studio export."""
    setdefaults = []
    for node in ast.walk(TREE):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == SCRIPTS_DIR
        ):
            setdefaults.append(node)
    assert not setdefaults, "the scripts pin must not be left in the environment"

    helper = _pin_helper_node(TREE)
    assert helper is not None
    writes = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "environ"
    ]
    pops = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pop"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "environ"
    ]
    assert len(writes) <= len(pops), "a direct write to os.environ needs an unwind"


def test_pin_prefers_the_internal_helper():
    """unsloth_zoo exempts a user pin from the strict converter scan, so Studio's own
    routing has to be marked as internal or the exemption is taken by a converter nobody
    reviewed."""
    helper = _pin_helper_node(TREE)
    assert helper is not None
    imported = [
        alias.name
        for node in ast.walk(helper)
        if isinstance(node, ast.ImportFrom) and node.module == "unsloth_zoo.llama_cpp"
        for alias in node.names
    ]
    assert "internal_scripts_dir_pin" in imported
    used = [
        item
        for node in ast.walk(helper)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Name)
        and item.context_expr.func.id == "internal_scripts_dir_pin"
    ]
    assert used, "internal_scripts_dir_pin must be entered, not just imported"


def test_warning_handler_gated_on_module_flag():
    try_node = _find_pin_try(TREE)
    assert try_node is not None
    handlers = [h for h in try_node.handlers if _catches_import_error(h)]
    assert handlers
    # And it has to keep covering the half-built cases, not just the missing-module one. That is
    # what #8603 widened the handler for: an unsloth_zoo that imports but raises RuntimeError or
    # AttributeError aborts the export otherwise, and a revert to ImportError alone still
    # satisfies _catches_import_error above.
    covering = [h for h in handlers if _covers_half_built_zoo(h)]
    assert (
        covering
    ), "the scripts pin must fall back on a half-built unsloth_zoo, not just a missing one"
    handler = covering[0]
    flag_reads = []
    flag_writes = []
    warning_calls = []
    for node in ast.walk(ast.Module(body = handler.body, type_ignores = [])):
        if isinstance(node, ast.Name) and node.id == "_LLAMA_CPP_SCRIPTS_WARNING_EMITTED":
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
    assert SCRIPTS_DIR in msg
    assert "unsloth_zoo" in msg


def test_default_dir_is_string():
    from unsloth_zoo.llama_cpp import LLAMA_CPP_DEFAULT_DIR
    assert isinstance(LLAMA_CPP_DEFAULT_DIR, str)


# ---------------------------------------------------------------- behaviour


def test_pin_is_in_force_only_while_converting(monkeypatch):
    fake, calls = _install_fake_zoo(monkeypatch)
    pin, _ns = _load_pin_helper()

    with pin():
        assert os.environ[SCRIPTS_DIR] == fake.LLAMA_CPP_DEFAULT_DIR
    assert SCRIPTS_DIR not in os.environ
    assert calls["internal"] == [fake.LLAMA_CPP_DEFAULT_DIR]


def test_pin_preserves_an_explicit_user_override(monkeypatch):
    _fake, _calls = _install_fake_zoo(monkeypatch)
    monkeypatch.setenv(SCRIPTS_DIR, "/explicit/override")
    pin, _ns = _load_pin_helper()

    with pin():
        assert os.environ[SCRIPTS_DIR] == "/explicit/override"
    assert os.environ[SCRIPTS_DIR] == "/explicit/override"


def test_converter_tag_is_not_overridden(monkeypatch):
    """UNSLOTH_LLAMA_CPP_SCRIPTS_DIR outranks the tag, so pinning while a tag is set is
    what made the tag do nothing."""
    _fake, calls = _install_fake_zoo(monkeypatch)
    monkeypatch.setenv(CONVERTER_TAG, "b9999")
    pin, _ns = _load_pin_helper()

    with pin():
        assert SCRIPTS_DIR not in os.environ
    assert calls["internal"] == []


def test_incomplete_install_is_not_pinned(monkeypatch):
    _fake, calls = _install_fake_zoo(monkeypatch, incomplete = True)
    pin, _ns = _load_pin_helper()

    with pin():
        assert SCRIPTS_DIR not in os.environ
    assert calls["internal"] == []


def test_mlx_does_not_nest_the_pin(monkeypatch):
    """unsloth_zoo's MLX save path pins the converter itself and its pin holds a plain
    threading.Lock across the conversion, so a second entry hangs the export."""
    fake, calls = _install_fake_zoo(monkeypatch)
    pin, _ns = _load_pin_helper(is_mlx = True)

    with pin():
        with fake.internal_scripts_dir_pin(fake.LLAMA_CPP_DEFAULT_DIR):
            pass
    assert calls["internal"] == [fake.LLAMA_CPP_DEFAULT_DIR]
    assert SCRIPTS_DIR not in os.environ


def test_older_zoo_without_the_internal_helper_still_unwinds(monkeypatch):
    fake, _calls = _install_fake_zoo(monkeypatch, internal_pin = False)
    pin, _ns = _load_pin_helper()

    with pin():
        assert os.environ[SCRIPTS_DIR] == fake.LLAMA_CPP_DEFAULT_DIR
    assert SCRIPTS_DIR not in os.environ


def test_warning_fires_at_most_once_across_calls(monkeypatch):
    """A zoo with no local-script resolver: warn once, keep exporting."""
    _fake, _calls = _install_fake_zoo(monkeypatch, resolver = False)
    emits = []
    logger = types.SimpleNamespace(warning = lambda message, *a, **k: emits.append(message))
    pin, _ns = _load_pin_helper(logger = logger)

    for _ in range(3):
        with pin():
            pass
    assert len(emits) == 1
    assert SCRIPTS_DIR in emits[0]
    assert SCRIPTS_DIR not in os.environ


def test_missing_default_dir_degrades_to_warning(monkeypatch):
    _fake, _calls = _install_fake_zoo(monkeypatch, default_dir = None)
    emits = []
    logger = types.SimpleNamespace(warning = lambda message, *a, **k: emits.append(message))
    pin, _ns = _load_pin_helper(logger = logger)

    with pin():
        assert SCRIPTS_DIR not in os.environ
    assert len(emits) == 1
    assert SCRIPTS_DIR not in os.environ


def test_no_warning_when_both_symbols_present(monkeypatch):
    fake, _calls = _install_fake_zoo(monkeypatch)
    emits = []
    logger = types.SimpleNamespace(warning = lambda message, *a, **k: emits.append(message))
    pin, _ns = _load_pin_helper(logger = logger)

    with pin():
        assert os.environ.get(SCRIPTS_DIR) == fake.LLAMA_CPP_DEFAULT_DIR
    assert emits == []
