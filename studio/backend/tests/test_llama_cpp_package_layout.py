# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``core.inference.llama_cpp`` was one 43,000-line module and is now a package.

Everything that depended on it being one module is pinned here, deterministically, from the
source and the live objects rather than from a reading of the diff:

* every definition is exported by the package once, and the facade and the owning module
  hand out the same object;
* a patch on the package reaches every submodule alias of the patched object, and its undo
  restores them, so the 160-odd names the suite patches on the module keep landing;
* a global a function rebinds with ``global`` is read live through the package and is never
  copied into another module;
* submodules import siblings relatively and never the package, so the import graph has no
  cycle through ``__init__``;
* ``LlamaCppBackend`` is bound in every submodule whose code names it, and no two mixins
  define the same method;
* every ``from core.inference.llama_cpp import X`` in production code, routes included and
  lazy imports included, names an attribute the package has;
* the two ``__file__``-relative lookups and the ``_vulkan_probe.py`` sibling still resolve
  from the package's depth.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from unittest import mock

import pytest

from llama_cpp_source import PACKAGE_DIR, class_source, module_paths, module_source

import core.inference.llama_cpp as llama_cpp
from core.inference.llama_cpp import LlamaCppBackend

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "core.inference.llama_cpp"
SUBMODULES = llama_cpp._SUBMODULES


def _defined_names(tree: ast.Module) -> list[str]:
    """Names a module binds at top level by definition or assignment, not by import."""
    names: list[str] = []

    def record(target):
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.append(node.id)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                record(target)
        elif isinstance(node, ast.AnnAssign):
            record(node.target)
        elif isinstance(node, (ast.Try, ast.If)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        record(target)
    return names


def _module_trees() -> dict[str, ast.Module]:
    return {
        path.stem: ast.parse(path.read_text(encoding = "utf-8"))
        for path in module_paths()
        if path.name != "__init__.py"
    }


def _module_by_stem(stem: str):
    return next(module for module in SUBMODULES if module.__name__ == f"{PACKAGE}.{stem}")


# ── exports ──────────────────────────────────────────────────────────────────────────


def test_every_definition_is_exported_once_and_is_the_owners_object():
    owners: dict[str, str] = {}
    for stem, tree in _module_trees().items():
        module = _module_by_stem(stem)
        for name in _defined_names(tree):
            assert (
                name not in owners or owners[name] == stem
            ), f"{name} is defined in both {owners[name]} and {stem}"
            owners[name] = stem
            if name.endswith("Mixin") and inspect.isclass(getattr(module, name)):
                continue  # the composed class's parts: internal, not part of the old surface
            assert getattr(llama_cpp, name) is getattr(
                module, name
            ), f"llama_cpp.{name} is not the object {stem} defines"
    assert owners["LlamaCppBackend"] == "backend"
    # The refactor guard's baseline is the old module's symbol list, recorded before the split.
    recorded = json.loads(
        (BACKEND_ROOT / "tests" / "data" / "refactor_guard" / "ast_inventory.json").read_text(
            encoding = "utf-8"
        )
    )[PACKAGE]["symbols"]
    assert set(recorded) <= set(owners), sorted(set(recorded) - set(owners))


def test_the_old_modules_imports_are_still_attributes_of_the_package():
    """``llama_cpp.subprocess`` and the names tests patch by string lived on the module."""
    for name in (
        "os",
        "sys",
        "shutil",
        "subprocess",
        "httpx",
        "logger",
        "hf_hub_download_with_xet_fallback",
    ):
        assert hasattr(llama_cpp, name), name
    assert 'get_logger("core.inference.llama_cpp")' in module_source("log")


def test_the_logger_keeps_the_old_module_name_in_every_submodule():
    for module in SUBMODULES:
        if "logger" in vars(module):
            assert module.logger is llama_cpp.logger, module.__name__


# ── patches reach their readers ──────────────────────────────────────────────────────


def _holders(name):
    target = getattr(llama_cpp, name)
    return [module for module in SUBMODULES if module.__dict__.get(name) is target]


@pytest.mark.parametrize(
    "name",
    [
        "cached_gguf_for_load",
        "_metal_device_is_paravirtual",
        "_extra_args_set_any_flag",
        "logger",
    ],
)
def test_a_patch_on_the_package_rebinds_every_alias_and_its_undo_restores_them(name):
    original = getattr(llama_cpp, name)
    holders = _holders(name)
    assert len(holders) >= 2, f"{name} is imported into fewer than two modules; pick another"
    sentinel = object()

    # A private patcher: the fixture-provided one is shared with the autouse fixtures, and
    # undoing it here would undo theirs too.
    patcher = pytest.MonkeyPatch()
    patcher.setattr(llama_cpp, name, sentinel)
    assert getattr(llama_cpp, name) is sentinel
    for module in holders:
        assert module.__dict__[name] is sentinel, module.__name__

    patcher.undo()
    assert getattr(llama_cpp, name) is original
    for module in holders:
        assert module.__dict__[name] is original, module.__name__


def test_a_string_patch_target_reaches_the_module_that_calls_it():
    name = "hf_hub_download_with_xet_fallback"
    holders = _holders(name)
    assert holders, "no submodule imports the download helper"
    with mock.patch(f"{PACKAGE}.{name}") as fake:
        for module in holders:
            assert module.__dict__[name] is fake, module.__name__
    for module in holders:
        assert module.__dict__[name] is llama_cpp.hf_hub_download_with_xet_fallback


def test_a_patch_only_touches_aliases_of_the_same_object(monkeypatch):
    """Two modules may bind one name to different objects; only the package's is rebound."""
    victim = _module_by_stem("compaction")
    monkeypatch.setattr(victim, "_PROBE_ONLY_HERE", "theirs", raising = False)
    monkeypatch.setattr(llama_cpp, "_PROBE_ONLY_HERE", "mine", raising = False)
    assert victim._PROBE_ONLY_HERE == "theirs"


# ── live globals ─────────────────────────────────────────────────────────────────────


def test_live_globals_are_read_through_the_owner_and_written_to_it(monkeypatch):
    assert set(llama_cpp._LIVE) == {"_SWA_CACHE", "_CHAT_LOADS_IN_FLIGHT"}
    for name, owner in llama_cpp._LIVE.items():
        assert name not in vars(llama_cpp), f"{name} was copied into the package"
        assert getattr(llama_cpp, name) is getattr(owner, name)
        probe = {}
        monkeypatch.setattr(owner, name, probe)
        assert getattr(llama_cpp, name) is probe, f"{name} is stale after the owner rebound it"
        other = {}
        monkeypatch.setattr(llama_cpp, name, other)
        assert getattr(owner, name) is other, f"writing llama_cpp.{name} missed the owner"
        assert name in dir(llama_cpp)


def test_a_global_statement_rebinds_a_name_only_its_own_module_holds():
    trees = _module_trees()
    live: dict[str, str] = {}
    for stem, tree in trees.items():
        defined = set(_defined_names(tree))
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                for name in node.names:
                    assert name in defined, f"{stem} rebinds {name} with global but does not own it"
                    live[name] = stem
    assert set(live) == set(llama_cpp._LIVE)
    for stem, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                leaked = sorted({alias.name for alias in node.names} & set(live))
                assert not leaked, f"{stem} imports {leaked}; a copy would go stale on rebind"


# ── import graph ─────────────────────────────────────────────────────────────────────


def test_submodules_import_siblings_relatively_and_never_the_package():
    for stem, tree in _module_trees().items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target = node.module or ""
                assert not (
                    node.level == 0 and target.startswith(PACKAGE)
                ), f"{stem} imports the package by its absolute name: {ast.unparse(node)}"
                assert not (
                    node.level == 1 and node.module is None
                ), f"{stem} uses `from . import ...`, which reads the half-built package"
                assert node.level <= 1, f"{stem}: {ast.unparse(node)}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(PACKAGE), f"{stem}: {ast.unparse(node)}"


def test_the_class_is_bound_in_every_submodule_that_names_it():
    for stem, tree in _module_trees().items():
        if stem == "backend":
            continue
        names_it = any(
            isinstance(node, ast.Name) and node.id == "LlamaCppBackend" for node in ast.walk(tree)
        )
        module = _module_by_stem(stem)
        if names_it:
            assert (
                module.__dict__.get("LlamaCppBackend") is LlamaCppBackend
            ), f"{stem} names LlamaCppBackend but __init__ never bound it there"


def test_binding_the_class_into_submodules_is_also_a_patch_seam(monkeypatch):
    holders = [m for m in SUBMODULES if m.__dict__.get("LlamaCppBackend") is LlamaCppBackend]
    assert len(holders) >= 10

    class Fake:
        pass

    monkeypatch.setattr(llama_cpp, "LlamaCppBackend", Fake)
    assert all(m.__dict__["LlamaCppBackend"] is Fake for m in holders)
    monkeypatch.undo()
    assert all(m.__dict__["LlamaCppBackend"] is LlamaCppBackend for m in holders)


# ── the composed class ───────────────────────────────────────────────────────────────


def test_no_two_mixins_define_the_same_attribute():
    seen: dict[str, type] = {}
    for klass in LlamaCppBackend.__mro__[1:]:
        if klass is object:
            continue
        assert klass.__module__.startswith(PACKAGE + "."), klass
        for name in vars(klass):
            if name.startswith("__"):
                continue
            assert name not in seen, f"{name} is defined on both {seen[name]} and {klass}"
            seen[name] = klass
    # The composed class owns __init__ and the locks, and defines nothing a mixin defines. Read
    # from the source: at runtime the suite's fixtures patch methods onto the composed class.
    backend_tree = _module_trees()["backend"]
    composed = next(n for n in backend_tree.body if isinstance(n, ast.ClassDef))
    own = {n.name for n in composed.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    own |= {
        t.id
        for n in composed.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    assert "__init__" in own
    assert not (own & set(seen)), f"the composed class redefines {sorted(own & set(seen))}"
    # Every method the sources define is reachable on the composed class from its own mixin:
    # a method lost between two mixins would be silent at import.
    trees = _module_trees()
    defined: dict[str, str] = {}
    for stem, tree in trees.items():
        for klass in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            if klass.name != "LlamaCppBackend" and not klass.name.endswith("Mixin"):
                continue  # the helper classes: exceptions, intents, named tuples
            for node in klass.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined[node.name] = stem
    assert len(defined) >= 500, len(defined)
    for name, stem in defined.items():
        holder = seen.get(name, LlamaCppBackend)
        assert holder.__module__ == f"{PACKAGE}.{stem}", (name, stem, holder)
        assert name in vars(holder), name


def test_getattr_static_reaches_the_mixin_descriptor():
    """``LlamaCppBackend.__dict__[name]`` no longer works for a mixin method; this does."""
    finder = inspect.getattr_static(LlamaCppBackend, "_find_llama_server_binary")
    assert isinstance(finder, staticmethod)
    assert "_find_llama_server_binary" not in vars(LlamaCppBackend)


def test_class_source_covers_the_mixins():
    source = class_source(LlamaCppBackend)
    for needle in ("def load_model(", "def _find_llama_server_binary(", "def unload_model("):
        assert needle in source, needle
    assert needle not in inspect.getsource(LlamaCppBackend)


# ── the rest of the backend ──────────────────────────────────────────────────────────


def _production_imports():
    """Every ``from core.inference.llama_cpp import ...`` in production code, lazy ones too."""
    out = []
    for path in sorted(BACKEND_ROOT.rglob("*.py")):
        rel = path.relative_to(BACKEND_ROOT).as_posix()
        if (
            rel.startswith(("tests/", "vendor/"))
            or "/tests/" in rel
            or rel.startswith("core/inference/llama_cpp/")
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            absolute = node.level == 0 and node.module == PACKAGE
            relative = (
                node.level == 1
                and node.module == "llama_cpp"
                and rel.startswith("core/inference/")
                and rel.count("/") == 2
            )
            if absolute or relative:
                out.append((rel, node.lineno, [alias.name for alias in node.names]))
    return out


def test_every_production_import_from_the_package_names_an_attribute_it_has():
    imports = _production_imports()
    assert any(rel.startswith("routes/") for rel, _, _ in imports), "no route imports the backend?"
    assert len(imports) >= 30
    missing = [
        (rel, line, name)
        for rel, line, names in imports
        for name in names
        if not hasattr(llama_cpp, name)
    ]
    assert not missing, missing


# ── file-relative lookups ────────────────────────────────────────────────────────────


def test_file_relative_lookups_resolve_from_the_package_depth():
    """Two call sites take ``parents[5]`` for the checkout root; one looks for a sibling probe."""
    for stem in ("backend_binary", "backend_shutdown"):
        source = (PACKAGE_DIR / f"{stem}.py").read_text(encoding = "utf-8")
        assert "Path(__file__).resolve().parents[5]" in source, stem
        assert "parents[4]" not in source, stem
        root = Path(_module_by_stem(stem).__file__).resolve().parents[5]
        assert (root / "studio" / "backend" / "core" / "inference" / "llama_cpp").is_dir()
    gpu_memory = _module_by_stem("backend_gpu_memory")
    assert 'Path(__file__).resolve().parent.with_name("_vulkan_probe.py")' in (
        PACKAGE_DIR / "backend_gpu_memory.py"
    ).read_text(encoding = "utf-8")
    assert Path(gpu_memory.__file__).resolve().parent.with_name("_vulkan_probe.py").is_file()
    for path in module_paths():
        text = path.read_text(encoding = "utf-8")
        assert 'with_name("_vulkan_probe.py")' not in text or path.stem == "backend_gpu_memory"
