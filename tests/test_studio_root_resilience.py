"""Unsloth install-root inference must not crash under hostile filesystem conditions (PermissionError/OSError swallowed; custom root kept when resolve() fails)."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOTS = REPO_ROOT / "studio" / "backend" / "utils" / "paths" / "storage_roots.py"
LLAMA_CPP = REPO_ROOT / "studio" / "backend" / "core" / "inference" / "llama_cpp.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_infer_studio_home_swallows_permission_error(tmp_path, monkeypatch):
    candidate = tmp_path / "fake_root"
    venv = candidate / "unsloth_studio"
    venv.mkdir(parents = True)
    monkeypatch.setattr(sys, "prefix", str(venv))
    sys.modules.pop("sr_perm", None)
    mod = _load("sr_perm", STORAGE_ROOTS)
    with mock.patch.object(Path, "is_file", side_effect = PermissionError("denied")):
        # Must NOT raise.
        assert mod._infer_studio_home_from_venv() is None


def test_studio_root_does_not_crash_on_permission_error(tmp_path, monkeypatch):
    """studio_root() falls through to the legacy default on a restricted filesystem."""
    candidate = tmp_path / "fake_root"
    venv = candidate / "unsloth_studio"
    venv.mkdir(parents = True)
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    sys.modules.pop("sr_studio_perm", None)
    mod = _load("sr_studio_perm", STORAGE_ROOTS)
    with mock.patch.object(Path, "is_file", side_effect = OSError("ebusy")):
        result = mod.studio_root()
    assert result == Path.home() / ".unsloth" / "studio"


def _method_body(src: str, name: str) -> str:
    """Whole method body (def to next sibling def at the same indent)."""
    start = src.index(f"def {name}")
    indent = " " * (start - src.rfind("\n", 0, start) - 1)
    nxt = src.find(f"\n{indent}def ", start + 1)
    return src[start : nxt if nxt != -1 else len(src)]


def test_kill_orphan_catches_oserror_from_studio_root():
    """Cleanup must not crash when studio_root() raises. _kill_orphaned_servers
    resolves the install root through the shared _resolved_studio_root_and_is_legacy()
    classifier, which swallows (ImportError, OSError, ValueError) on the probe."""
    src = LLAMA_CPP.read_text()
    # Cleanup delegates to the shared classifier rather than importing studio_root inline.
    assert "LlamaCppBackend._resolved_studio_root_and_is_legacy()" in _method_body(
        src, "_kill_orphaned_servers"
    ), "_kill_orphaned_servers must resolve the root via _resolved_studio_root_and_is_legacy()"
    # The shared classifier catches both the resolve() failure and the outer studio_root() probe.
    classifier = _method_body(src, "_resolved_studio_root_and_is_legacy")
    assert "studio_root as _sr" in classifier, "classifier must probe studio_root()"
    assert "except (OSError, ValueError):" in classifier, "inner resolve() probe must be guarded"
    assert "except (ImportError, OSError, ValueError):" in classifier, (
        "_resolved_studio_root_and_is_legacy must catch (ImportError, OSError, ValueError) "
        "from studio_root()"
    )


def _exec_search_roots_block(
    home: Path, studio_root_value: Path, resolve_raises: bool
) -> list[Path]:
    """Run _find_llama_server_binary's search_roots derivation -- plus the shared
    _resolved_studio_root_and_is_legacy() classifier it delegates to -- with a
    controlled studio_root() and resolve(), without importing the heavy module."""
    src = LLAMA_CPP.read_text()
    # Shared root classifier (holds the defensive try/except for studio_root()).
    # End the slice at the next sibling def/decorator at the same indent rather
    # than the literal "@staticmethod" string, so a future docstring mentioning a
    # decorator can't truncate the helper mid-body and break exec().
    helper_start = src.index("def _resolved_studio_root_and_is_legacy")
    indent = " " * (helper_start - src.rfind("\n", 0, helper_start) - 1)
    nxt_def = src.find(f"\n{indent}def ", helper_start + 1)
    nxt_dec = src.find(f"\n{indent}@", helper_start + 1)
    sibling = [idx for idx in (nxt_def, nxt_dec) if idx != -1]
    helper_end = min(sibling) if sibling else len(src)
    helper = textwrap.dedent(src[helper_start:helper_end])
    # search_roots derivation inside _find_llama_server_binary (delegates to the classifier).
    block_start = src.index('legacy_llama = Path.home() / ".unsloth" / "llama.cpp"')
    block_end = src.index("for unsloth_home in search_roots:", block_start)
    block = textwrap.dedent(" " * 8 + src[block_start:block_end])
    fake_module = type(sys)("fake_storage_roots")
    fake_module.studio_root = lambda: studio_root_value
    sys.modules["utils.paths.storage_roots"] = fake_module
    try:
        original_resolve = Path.resolve

        def _resolve(self, *a, **k):
            if resolve_raises:
                raise OSError("ebusy")
            return original_resolve(self, *a, **k)

        with (
            mock.patch.object(Path, "home", classmethod(lambda cls: home)),
            mock.patch.object(Path, "resolve", _resolve),
        ):
            ns: dict = {"Path": Path}
            exec(helper, ns)  # noqa: S102  -- defines _resolved_studio_root_and_is_legacy
            ns["LlamaCppBackend"] = type(
                "LlamaCppBackend",
                (),
                {
                    "_resolved_studio_root_and_is_legacy": staticmethod(
                        ns["_resolved_studio_root_and_is_legacy"]
                    )
                },
            )
            exec(block, ns)  # noqa: S102  -- defines search_roots
        return ns["search_roots"]
    finally:
        sys.modules.pop("utils.paths.storage_roots", None)


def test_search_roots_keeps_custom_when_resolve_fails(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    custom = tmp_path / "custom_studio"
    custom.mkdir()
    roots = _exec_search_roots_block(home = home, studio_root_value = custom, resolve_raises = True)
    # On resolve() failure, the inner except falls back to direct equality;
    # custom != legacy_studio so the custom root must remain in search_roots.
    assert custom / "llama.cpp" in roots, f"custom root dropped on resolve() failure: {roots}"
    # custom-mode discovery excludes the legacy tree to match _kill_orphaned_servers.
    assert (
        home / ".unsloth" / "llama.cpp"
    ) not in roots, f"legacy llama path must not appear in custom-mode search_roots: {roots}"


def test_search_roots_default_mode_uses_legacy_only(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    legacy = home / ".unsloth" / "studio"
    legacy.mkdir(parents = True)
    roots = _exec_search_roots_block(home = home, studio_root_value = legacy, resolve_raises = False)
    # Default mode: only legacy_llama.
    assert roots == [home / ".unsloth" / "llama.cpp"]
