# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio update` must not report success on a damaged install.

pip considers a distribution with intact metadata already satisfied, so an
update reinstalls nothing when a package's files are damaged. Before this check
it printed "Unsloth Studio Installed" and exited 0 while Studio died at boot
with `cannot import name 'Depends' from 'fastapi'` -- and a missing-package
check could not have caught it, because `import fastapi` still succeeded.

The detector is exercised against real distribution metadata written to a temp
tree, not a mock, because the two things that make it work (RECORD is parsed
directly, and only shrinkage counts) are exactly the things a mock would hide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


def _deps():
    from unsloth_cli import _studio_deps as _mod
    return _mod


# ── a fake site-packages with real RECORD metadata ───────────────────


def _make_dist(
    site: Path,
    name: str,
    files: dict[str, bytes],
    record_sizes = None,
):
    """Install `files` under `site` and write a dist-info RECORD describing them.

    `record_sizes` overrides the size RECORD claims, which is how damage is
    simulated without having to corrupt anything after the fact.
    """
    info = site / f"{name}-1.0.dist-info"
    info.mkdir(parents = True, exist_ok = True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
    (info / "WHEEL").write_text("Wheel-Version: 1.0\n")
    rows = [f"{name}-1.0.dist-info/METADATA,,", f"{name}-1.0.dist-info/RECORD,,"]
    for rel, body in files.items():
        target = site / rel
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(body)
        size = (record_sizes or {}).get(rel, len(body))
        rows.append(f"{rel},sha256=x,{size}")
    (info / "RECORD").write_text("\n".join(rows) + "\n")


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A site-packages directory that importlib.metadata will scan, and only it."""
    d = tmp_path / "site-packages"
    d.mkdir()
    monkeypatch.syspath_prepend(str(d))
    # sys.path alone is not enough: the real environment's distributions would
    # still be discovered and could contribute findings of their own.
    import importlib.metadata as md

    real = md.distributions

    def only_fixture(**kwargs):
        return real(path = [str(d)])

    monkeypatch.setattr(md, "distributions", only_fixture)
    return d


def test_an_intact_install_reports_nothing(site):
    _make_dist(site, "alpha", {"alpha/__init__.py": b"x = 1\n"})
    assert _deps().damaged_installed_files() == []


def test_a_truncated_file_is_reported(site):
    # The observed failure: fastapi/__init__.py emptied, metadata untouched.
    _make_dist(
        site, "fastapi", {"fastapi/__init__.py": b""}, record_sizes = {"fastapi/__init__.py": 1081}
    )
    found = _deps().damaged_installed_files()
    assert len(found) == 1
    assert "fastapi/__init__.py" in found[0]
    assert "0 bytes" in found[0] and "1081" in found[0]


def test_a_deleted_file_is_reported(site):
    _make_dist(site, "starlette", {"starlette/routing.py": b"y = 2\n"})
    (site / "starlette" / "routing.py").unlink()
    found = _deps().damaged_installed_files()
    assert len(found) == 1
    assert "starlette/routing.py is missing" in found[0]


def test_deletion_is_seen_even_though_Distribution_files_hides_it(site):
    # Distribution.files drops entries whose file no longer exists, so a check
    # built on it can never report a deletion. This asserts RECORD is the source.
    import importlib.metadata as md

    _make_dist(site, "gamma", {"gamma/a.py": b"a\n", "gamma/b.py": b"bb\n"})
    (site / "gamma" / "b.py").unlink()
    listed = [str(f) for f in (md.distribution("gamma").files or [])]
    assert "gamma/b.py" not in listed, "Distribution.files started reporting missing files"
    assert any("gamma/b.py is missing" in f for f in _deps().damaged_installed_files())


def test_a_file_larger_than_recorded_is_not_damage(site):
    # Two distributions claiming one path: descript-audio-codec ships a
    # top-level tests/__init__.py that another package overwrites. Flagging that
    # would block updates on a perfectly healthy install.
    _make_dist(
        site,
        "delta",
        {"tests/__init__.py": b"a much longer body\n"},
        record_sizes = {"tests/__init__.py": 0},
    )
    assert _deps().damaged_installed_files() == []


def test_installer_owned_metadata_is_ignored(site):
    # .dist-info files are rewritten in place and drift from the size recorded
    # inside themselves; two real distributions did exactly that.
    _make_dist(site, "epsilon", {"epsilon/__init__.py": b"e\n"})
    info = site / "epsilon-1.0.dist-info"
    (info / "RECORD").write_text(
        "epsilon-1.0.dist-info/METADATA,sha256=x,999999\nepsilon/__init__.py,sha256=x,2\n"
    )
    assert _deps().damaged_installed_files() == []


def test_a_distribution_without_RECORD_is_not_damage(site):
    # Editable and system installs legitimately have none.
    info = site / "zeta-1.0.dist-info"
    info.mkdir(parents = True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: zeta\nVersion: 1.0\n")
    assert _deps().damaged_installed_files() == []


def test_findings_are_capped(site):
    files = {f"eta/m{i}.py": b"" for i in range(40)}
    sizes = {k: 500 for k in files}
    _make_dist(site, "eta", files, record_sizes = sizes)
    assert len(_deps().damaged_installed_files(limit = 3)) == 3


# ── the failure path ─────────────────────────────────────────────────


def test_a_clean_tree_passes_through(monkeypatch):
    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(studio._studio_deps, "damaged_installed_files", lambda *a, **k: [])
    studio._fail_if_install_damaged()  # must not raise


def test_a_damaged_tree_exits_nonzero_and_names_the_files(monkeypatch, capsys):
    import typer

    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(
        studio._studio_deps,
        "damaged_installed_files",
        lambda *a, **k: ["fastapi: fastapi/__init__.py is 0 bytes, expected 1081"],
    )
    with pytest.raises(typer.Exit) as excinfo:
        studio._fail_if_install_damaged()
    assert excinfo.value.exit_code == 1
    err = capsys.readouterr().err
    assert "fastapi/__init__.py is 0 bytes" in err
    # The recovery instruction is the point: an update cannot fix this.
    assert "install.sh" in err or "install.ps1" in err
    assert "--no-verify" in err


def test_a_foreign_cli_stays_quiet(monkeypatch):
    # A pip-installed CLI can drive an update into a venv it does not live in;
    # its own file list would describe the wrong tree, so it must not accuse.
    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: True)

    def _never(*a, **k):
        raise AssertionError("the check ran against the wrong environment")

    monkeypatch.setattr(studio._studio_deps, "damaged_installed_files", _never)
    studio._fail_if_install_damaged()  # must not raise


def test_windows_is_told_the_powershell_installer(monkeypatch, capsys):
    import platform as _platform
    import typer

    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(
        studio._studio_deps, "damaged_installed_files", lambda *a, **k: ["x: y is missing"]
    )
    monkeypatch.setattr(_platform, "system", lambda: "Windows")
    with pytest.raises(typer.Exit):
        studio._fail_if_install_damaged()
    err = capsys.readouterr().err
    assert "install.ps1" in err
    assert "curl" not in err


# ── the update command wiring ────────────────────────────────────────


def test_update_exposes_verify_defaulting_on():
    import inspect

    opt = inspect.signature(_studio().update).parameters["verify"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "--verify/--no-verify" in decls
    assert getattr(opt, "default", None) is True


def _run_update(monkeypatch, argv, verified):
    studio = _studio()
    monkeypatch.setattr(studio, "_ensure_studio_env_exported", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_run_setup_script", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_release_self_exe_lock_windows", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_cleanup_self_exe_lock_windows", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_refresh_desktop_shortcuts", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_fail_if_install_damaged", lambda: verified.append(True))
    return CliRunner().invoke(studio.studio_app, ["update", *argv])


def test_update_verifies_by_default(monkeypatch):
    verified = []
    result = _run_update(monkeypatch, [], verified)
    assert result.exit_code == 0, result.output
    assert verified == [True]


def test_no_verify_skips_the_check(monkeypatch):
    verified = []
    result = _run_update(monkeypatch, ["--no-verify"], verified)
    assert result.exit_code == 0, result.output
    assert verified == []


def test_a_tauri_update_is_verified_too(monkeypatch):
    # The Tauri path returns before the shortcut refresh, so a check placed
    # after that return would silently not run for desktop-initiated updates,
    # the one flow where the user never sees a terminal.
    verified = []
    monkeypatch.setenv("UNSLOTH_TAURI_UPDATE", "1")
    result = _run_update(monkeypatch, [], verified)
    assert result.exit_code == 0, result.output
    assert verified == [True]
