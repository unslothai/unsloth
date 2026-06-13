"""Resilience checks for Studio install-root inference under hostile
filesystem conditions:
- _infer_studio_home_from_venv must NOT propagate PermissionError /
  OSError out through studio_root() (it would crash module import in
  run.py / main.py / transformers_version.py / model_config.py).
- _kill_orphaned_servers must catch (ImportError, OSError, ValueError)
  on the studio_root() probe so a transient resolve / sentinel failure
  cannot crash server startup.
- _find_llama_server_binary must keep the custom-root in search_roots
  when the inner resolve() comparison itself fails."""

from __future__ import annotations

import importlib.util
import re
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOTS = (
    REPO_ROOT / "studio" / "backend" / "utils" / "paths" / "storage_roots.py"
)
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
    """studio_root() must remain callable even when the venv inference
    encounters a restricted filesystem; it should fall through to the
    legacy default."""
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


def test_kill_orphan_catches_oserror_from_studio_root():
    """_kill_orphaned_servers must catch (ImportError, OSError, ValueError)
    on the studio_root() probe specifically; the sister function
    _find_llama_server_binary uses the same broader catch on its own probe."""
    src = LLAMA_CPP.read_text()
    fn_start = src.index("def _kill_orphaned_servers")
    fn_body = src[fn_start : fn_start + 4000]
    # The studio_root() probe in this fn is the one that imports as `_sr`
    # and assigns `_resolved_sr = _sr()`. Find the except that closes it.
    probe_idx = fn_body.index("storage_roots import studio_root as _sr")
    # The matching except is the next `except ...:` after the inner
    # OSError/ValueError block that wraps resolve().
    after = fn_body[probe_idx:]
    # Skip over the inner `except (OSError, ValueError):` that wraps resolve().
    inner_idx = after.index("except (OSError, ValueError):")
    after_inner = after[inner_idx + len("except (OSError, ValueError):") :]
    outer_match = re.search(r"except\s*\(?[^)]*?\)?:", after_inner)
    assert outer_match, "outer except for studio_root probe missing"
    clause = outer_match.group(0)
    assert (
        "OSError" in clause and "ValueError" in clause
    ), f"_kill_orphaned_servers studio_root probe catch too narrow: {clause!r}"


def _exec_search_roots_block(
    home: Path, studio_root_value: Path, resolve_raises: bool
) -> list[Path]:
    """Extract _find_llama_server_binary's env-mode search_roots block
    and execute it with controlled inputs."""
    src = LLAMA_CPP.read_text()
    block_start = src.index('legacy_llama = Path.home() / ".unsloth" / "llama.cpp"')
    block_end = src.index("_seen_roots: set[str]", block_start)
    raw = src[block_start:block_end]
    indent = " " * 8
    block = textwrap.dedent(indent + raw)
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
            exec(block, ns)  # noqa: S102
        return ns["search_roots"]
    finally:
        sys.modules.pop("utils.paths.storage_roots", None)


def test_search_roots_keeps_custom_when_resolve_fails(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    custom = tmp_path / "custom_studio"
    custom.mkdir()
    roots = _exec_search_roots_block(
        home = home, studio_root_value = custom, resolve_raises = True
    )
    # On resolve() failure, the inner except falls back to direct equality;
    # custom != legacy_studio so the custom root must remain in search_roots.
    assert (
        custom / "llama.cpp" in roots
    ), f"custom root dropped on resolve() failure: {roots}"
    # custom-mode discovery excludes the legacy tree to match _kill_orphaned_servers.
    assert (
        (home / ".unsloth" / "llama.cpp") not in roots
    ), f"legacy llama path must not appear in custom-mode search_roots: {roots}"


def test_search_roots_default_mode_uses_legacy_only(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    legacy = home / ".unsloth" / "studio"
    legacy.mkdir(parents = True)
    roots = _exec_search_roots_block(
        home = home, studio_root_value = legacy, resolve_raises = False
    )
    # Default mode: only legacy_llama.
    assert roots == [home / ".unsloth" / "llama.cpp"]
