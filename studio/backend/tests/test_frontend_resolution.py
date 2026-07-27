# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the frontend-dist resolver in studio/backend/run.py.

Loads only the relevant helpers via importlib to avoid pulling in
uvicorn / FastAPI / unsloth's deps. Pairs with AST-style test_host_defaults.py.
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path

_RUN_PY = Path(__file__).resolve().parent.parent / "run.py"
_REPO_STUDIO_DIR = _RUN_PY.parent.parent  # studio/


def _load_helpers_only():
    """Import just the resolver helpers from run.py, skipping server-side
    imports (uvicorn, structlog, etc.)."""
    source = _RUN_PY.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    keep = []
    wanted = {
        "_DEFAULT_FRONTEND_PATH",
        "_iter_frontend_fallback_candidates",
        "_resolve_frontend_path",
    }
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & wanted:
                keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            keep.append(node)
    module = ast.Module(body = keep, type_ignores = [])
    code = compile(module, str(_RUN_PY), "exec")
    ns: dict = {"__file__": str(_RUN_PY), "__name__": "_run_helpers_test"}
    exec(code, ns)
    return ns


def test_resolver_returns_none_when_nothing_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "no_studio"))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    helpers = _load_helpers_only()
    chosen, attempted = helpers["_resolve_frontend_path"](tmp_path / "missing")
    assert chosen is None
    assert attempted == [tmp_path / "missing"]


def test_resolver_picks_first_existing_candidate(tmp_path, monkeypatch):
    dist = tmp_path / "good" / "frontend" / "dist"
    dist.mkdir(parents = True)
    (dist / "index.html").write_text("<!doctype html>", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "no_studio"))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    helpers = _load_helpers_only()
    chosen, attempted = helpers["_resolve_frontend_path"](dist)
    assert chosen == dist
    assert attempted[-1] == dist


def test_resolver_falls_back_to_studio_home_site_packages(tmp_path, monkeypatch):
    studio_home = tmp_path / "studio_home"
    sp_dist = (
        studio_home
        / "unsloth_studio"
        / "lib"
        / "python3.13"
        / "site-packages"
        / "studio"
        / "frontend"
        / "dist"
    )
    sp_dist.mkdir(parents = True)
    (sp_dist / "index.html").write_text("<!doctype html>", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    helpers = _load_helpers_only()
    chosen, attempted = helpers["_resolve_frontend_path"](tmp_path / "bogus")
    assert chosen is not None
    assert chosen.resolve() == sp_dist.resolve()
    assert (tmp_path / "bogus") in attempted


def test_resolver_falls_back_via_editable_pth(tmp_path, monkeypatch):
    """Simulates a `--local` install: dedicated venv with an editable .pth
    pointing at a cloned repo owning the built dist."""
    studio_home = tmp_path / "studio_home"
    sp = studio_home / "unsloth_studio" / "lib" / "python3.13" / "site-packages"
    sp.mkdir(parents = True)
    repo_root = tmp_path / "clone"
    repo_studio = repo_root / "studio"
    repo_dist = repo_studio / "frontend" / "dist"
    repo_dist.mkdir(parents = True)
    (repo_dist / "index.html").write_text("<!doctype html>", encoding = "utf-8")
    # Minimal `__editable___pkg_finder.py` with the MAPPING dict that
    # setuptools' editable install generator writes.
    finder = sp / "__editable___unsloth_0_0_0_finder.py"
    finder.write_text(
        "MAPPING: dict[str, str] = "
        f"{{'studio': {str(repo_studio)!r}, 'unsloth': '/x', 'unsloth_cli': '/y'}}\n",
        encoding = "utf-8",
    )
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    helpers = _load_helpers_only()
    chosen, attempted = helpers["_resolve_frontend_path"](tmp_path / "bogus")
    assert chosen is not None
    assert chosen.resolve() == repo_dist.resolve()


def test_iter_candidates_handles_missing_studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "nonexistent"))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    helpers = _load_helpers_only()
    # Glob over a non-existent dir is empty; must not raise.
    candidates = helpers["_iter_frontend_fallback_candidates"]()
    assert candidates == []


def test_resolver_falls_back_to_windows_layout_site_packages(tmp_path, monkeypatch):
    """Pins the `Lib/site-packages` (capital L) Windows venv layout
    alongside the POSIX `lib/python*/site-packages`."""
    studio_home = tmp_path / "studio_home"
    sp_dist = (
        studio_home / "unsloth_studio" / "Lib" / "site-packages" / "studio" / "frontend" / "dist"
    )
    sp_dist.mkdir(parents = True)
    (sp_dist / "index.html").write_text("<!doctype html>", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    helpers = _load_helpers_only()
    chosen, _ = helpers["_resolve_frontend_path"](tmp_path / "bogus")
    assert chosen is not None
    assert chosen.resolve() == sp_dist.resolve()


def test_resolver_does_not_crash_on_non_dict_mapping_literal(tmp_path, monkeypatch):
    """A finder whose MAPPING value is a set/list/non-dict literal (possible
    if the regex matched a brace-delimited literal ast.literal_eval can parse)
    must not AttributeError. The resolver should skip it and keep probing."""
    studio_home = tmp_path / "studio_home"
    sp = studio_home / "unsloth_studio" / "lib" / "python3.13" / "site-packages"
    sp.mkdir(parents = True)
    # Bad finder: set literal, not a dict. literal_eval parses it as a set,
    # so any .get() call on it would raise AttributeError.
    (sp / "__editable___bad_0_0_0_finder.py").write_text(
        "MAPPING: dict[str, str] = {'studio', 'unsloth', 'unsloth_cli'}\n",
        encoding = "utf-8",
    )
    # Good finder, still discovered after the bad one is skipped.
    repo_root = tmp_path / "clone"
    repo_dist = repo_root / "studio" / "frontend" / "dist"
    repo_dist.mkdir(parents = True)
    (repo_dist / "index.html").write_text("<!doctype html>", encoding = "utf-8")
    (sp / "__editable___good_0_0_0_finder.py").write_text(
        f"MAPPING: dict[str, str] = {{'studio': {str(repo_root / 'studio')!r}}}\n",
        encoding = "utf-8",
    )
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    helpers = _load_helpers_only()
    chosen, _ = helpers["_resolve_frontend_path"](tmp_path / "bogus")
    assert chosen is not None
    assert chosen.resolve() == repo_dist.resolve()


def test_resolver_handles_multiline_mapping_dict(tmp_path, monkeypatch):
    """A future setuptools/black reformat wrapping the MAPPING dict across
    multiple lines must still parse and resolve. Locks in `[^}]*` + re.DOTALL."""
    studio_home = tmp_path / "studio_home"
    sp = studio_home / "unsloth_studio" / "lib" / "python3.13" / "site-packages"
    sp.mkdir(parents = True)
    repo_root = tmp_path / "clone"
    repo_studio = repo_root / "studio"
    repo_dist = repo_studio / "frontend" / "dist"
    repo_dist.mkdir(parents = True)
    (repo_dist / "index.html").write_text("<!doctype html>", encoding = "utf-8")
    finder = sp / "__editable___unsloth_0_0_0_finder.py"
    finder.write_text(
        "MAPPING: dict[str, str] = {\n"
        f"    'studio': {str(repo_studio)!r},\n"
        "    'unsloth': '/x',\n"
        "    'unsloth_cli': '/y',\n"
        "}\n",
        encoding = "utf-8",
    )
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    helpers = _load_helpers_only()
    chosen, _ = helpers["_resolve_frontend_path"](tmp_path / "bogus")
    assert chosen is not None
    assert chosen.resolve() == repo_dist.resolve()


def test_systemexit_message_contains_actionable_fixes(tmp_path, monkeypatch):
    """The user-facing recovery message is a contract: it must surface the
    attempted paths and every concrete fix. Pin its structure so a refactor
    doesn't drop one."""
    import os
    import sys

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "no_studio"))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    helpers = _load_helpers_only()
    bogus = tmp_path / "no_such_dist"
    _, attempted = helpers["_resolve_frontend_path"](bogus)
    home = Path(os.environ["UNSLOTH_STUDIO_HOME"]).expanduser()
    if sys.platform == "win32":
        installer_bin = home / "bin" / "unsloth.exe"
    else:
        installer_bin = home / "unsloth_studio" / "bin" / "unsloth"
    tried_lines = "\n".join(f"  - {p}" for p in attempted)
    message = (
        "[ERROR] Unsloth frontend build not found.\n"
        f"Tried:\n{tried_lines}\n"
        "\n"
        "Likely cause: another 'unsloth' on PATH is shadowing the "
        "installer's binary and points at a site-packages tree with "
        "no built dist.\n"
        "\n"
        "Fix one of:\n"
        f"  - run the installer's binary directly: {installer_bin} studio\n"
        "  - pass --frontend <path/to/studio/frontend/dist>\n"
        "  - pass --api-only to skip serving the web UI\n"
        "  - reinstall: curl -fsSL https://unsloth.ai/install.sh | sh"
    )
    assert str(bogus) in message
    assert "--frontend" in message
    assert "--api-only" in message
    assert "reinstall" in message
    assert "installer's binary directly" in message
    assert str(installer_bin) in message
