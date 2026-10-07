# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio update` must not report success on a damaged install.

pip considers a distribution with intact metadata already satisfied, so an
update reinstalls nothing when a package's files are damaged. Before this check
it printed "Unsloth Studio Installed" and exited 0 while Unsloth died at boot
with `cannot import name 'Depends' from 'fastapi'` -- and a missing-package
check could not have caught it, because `import fastapi` still succeeded.

The detector is exercised against real distribution metadata written to a temp
tree, not a mock, because the two things that make it work (RECORD is parsed
directly, and only shrinkage counts) are exactly the things a mock would hide.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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


def _make_dist(
    site: Path,
    name: str,
    files: dict[str, bytes],
    record_sizes = None,
    version: str = "1.0",
):
    """Install `files` under `site` and write a dist-info RECORD describing them.

    `record_sizes` overrides the size RECORD claims, which is how damage is
    simulated without having to corrupt anything after the fact.
    """
    info = site / f"{name}-{version}.dist-info"
    info.mkdir(parents = True, exist_ok = True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    (info / "WHEEL").write_text("Wheel-Version: 1.0\n")
    rows = [
        f"{name}-{version}.dist-info/METADATA,,",
        f"{name}-{version}.dist-info/RECORD,,",
    ]
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
    # sys.path alone is not enough: the real environment's distributions would still be discovered.
    import importlib.metadata as md

    real = md.distributions

    def only_fixture(**kwargs):
        return real(path = [str(d)])

    monkeypatch.setattr(md, "distributions", only_fixture)
    return d


def test_an_intact_install_reports_nothing(site):
    _make_dist(site, "alpha", {"alpha/__init__.py": b"x = 1\n"})
    assert _deps().damaged_installed_files() == []


def test_superseded_metadata_is_not_treated_as_file_damage(site):
    removed = "studio/frontend/dist/assets/removed-hash.js"
    _make_dist(site, "unsloth", {removed: b"old\n"}, version = "1.0")
    _make_dist(site, "unsloth", {"unsloth/__init__.py": b"new\n"}, version = "2.0")
    (site / removed).unlink()

    assert _deps().damaged_installed_files() == []
    conflicts = _deps().installed_metadata_conflicts()
    assert len(conflicts) == 1
    assert "unsloth" in conflicts[0]
    assert "1.0" in conflicts[0] and "2.0" in conflicts[0]


def test_duplicate_metadata_names_are_canonicalized(site):
    _make_dist(site, "foo_bar", {"foo_bar/old.py": b"old\n"}, version = "1.0")
    info = site / "foo_bar-1.0.dist-info"
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: foo.bar\nVersion: 1.0\n")
    _make_dist(site, "foo_bar", {"foo_bar/new.py": b"new\n"}, version = "2.0")
    info = site / "foo_bar-2.0.dist-info"
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: foo-bar\nVersion: 2.0\n")

    conflicts = _deps().installed_metadata_conflicts()
    assert len(conflicts) == 1 and conflicts[0].startswith("foo-bar:")


def test_duplicate_metadata_conflicts_can_be_scoped_by_canonical_name(site):
    _make_dist(site, "unsloth", {"unsloth/old.py": b"old\n"}, version = "1.0")
    _make_dist(site, "unsloth", {"unsloth/new.py": b"new\n"}, version = "2.0")
    _make_dist(site, "foo_bar", {"foo_bar/old.py": b"old\n"}, version = "1.0")
    _make_dist(site, "foo_bar", {"foo_bar/new.py": b"new\n"}, version = "2.0")

    deps = _deps()
    included = deps.installed_metadata_conflicts(names = ("foo.bar",))
    excluded = deps.installed_metadata_conflicts(exclude_names = ("foo-bar",))

    assert len(included) == 1 and included[0].startswith("foo-bar:")
    assert len(excluded) == 1 and excluded[0].startswith("unsloth:")


def test_duplicate_metadata_does_not_hide_another_packages_damage(site):
    _make_dist(site, "unsloth", {"studio/old.py": b"old\n"}, version = "1.0")
    _make_dist(site, "unsloth", {"unsloth/__init__.py": b"new\n"}, version = "2.0")
    _make_dist(
        site,
        "fastapi",
        {"fastapi/__init__.py": b""},
        record_sizes = {"fastapi/__init__.py": 1081},
    )

    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "fastapi/__init__.py" in found[0]


def test_a_truncated_file_is_reported(site):
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


def test_deletion_is_seen_whatever_Distribution_files_does(site):
    # Distribution.files differs across CPython versions for missing files, so RECORD is parsed directly.
    import importlib.metadata as md

    _make_dist(site, "gamma", {"gamma/a.py": b"a\n", "gamma/b.py": b"bb\n"})
    (site / "gamma" / "b.py").unlink()
    stale = [f for f in (md.distribution("gamma").files or []) if str(f) == "gamma/b.py"]
    assert not stale or not stale[0].locate().exists()
    assert any("gamma/b.py is missing" in f for f in _deps().damaged_installed_files())


def test_a_file_larger_than_recorded_is_not_damage(site):
    _make_dist(
        site,
        "delta",
        {"tests/__init__.py": b"a much longer body\n"},
        record_sizes = {"tests/__init__.py": 0},
    )
    assert _deps().damaged_installed_files() == []


def test_a_shared_file_shorter_than_recorded_is_not_damage(site):
    _make_dist(
        site, "iota", {"shared/__init__.py": b"short\n"}, record_sizes = {"shared/__init__.py": 900}
    )
    _make_dist(
        site, "kappa", {"shared/__init__.py": b"short\n"}, record_sizes = {"shared/__init__.py": 5}
    )
    assert _deps().damaged_installed_files() == []


def test_a_singly_owned_short_file_is_still_damage(site):
    _make_dist(site, "lam", {"lam/a.py": b"x"}, record_sizes = {"lam/a.py": 900})
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "lam/a.py" in found[0]


def test_the_scan_is_limited_to_this_interpreters_site_packages(monkeypatch, tmp_path):
    external = tmp_path / "elsewhere"
    (external / "ext-1.0.dist-info").mkdir(parents = True)
    (external / "ext").mkdir()
    (external / "ext-1.0.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: ext\nVersion: 1.0\n"
    )
    (external / "ext-1.0.dist-info" / "RECORD").write_text("ext/mod.py,sha256=x,9999\n")
    (external / "ext" / "mod.py").write_text("x\n")

    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.setattr(_deps(), "_scan_paths", lambda: {"path": [str(site)]})
    monkeypatch.syspath_prepend(str(external))
    assert _deps().damaged_installed_files() == []

    monkeypatch.setattr(_deps(), "_scan_paths", lambda: {"path": [str(external)]})
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "ext/mod.py" in found[0]


def test_a_deleted_shared_file_is_still_reported(site):
    _make_dist(site, "mu", {"shared/x.py": b"hello\n"}, record_sizes = {"shared/x.py": 10})
    _make_dist(site, "nu", {}, record_sizes = {})
    (site / "nu-1.0.dist-info" / "RECORD").write_text(
        "nu-1.0.dist-info/METADATA,,\nshared/x.py,sha256=x,10\n"
    )
    (site / "shared" / "x.py").unlink()
    found = _deps().damaged_installed_files()
    assert len(found) == 2
    assert all("shared/x.py is missing" in line for line in found)


def test_a_row_without_a_recorded_size_is_still_checked(site):
    _make_dist(site, "xi", {"xi/__init__.py": b"y\n"})
    (site / "xi-1.0.dist-info" / "RECORD").write_text("xi/__init__.py,,\n")
    (site / "xi" / "__init__.py").unlink()
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "xi/__init__.py is missing" in found[0]


def test_a_directory_standing_in_for_a_module_is_damage(site):
    _make_dist(site, "omicron", {}, record_sizes = {})
    (site / "omicron-1.0.dist-info" / "RECORD").write_text("omicron/mod.py,sha256=x,10\n")
    (site / "omicron" / "mod.py").mkdir(parents = True)
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "not a regular file" in found[0]


def test_installer_owned_metadata_is_ignored(site):
    _make_dist(site, "epsilon", {"epsilon/__init__.py": b"e\n"})
    info = site / "epsilon-1.0.dist-info"
    (info / "RECORD").write_text(
        "epsilon-1.0.dist-info/METADATA,sha256=x,999999\nepsilon/__init__.py,sha256=x,2\n"
    )
    assert _deps().damaged_installed_files() == []


def test_a_distribution_without_RECORD_is_not_damage(site):
    info = site / "zeta-1.0.dist-info"
    info.mkdir(parents = True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: zeta\nVersion: 1.0\n")
    assert _deps().damaged_installed_files() == []


def test_findings_are_capped(site):
    files = {f"eta/m{i}.py": b"" for i in range(40)}
    sizes = {k: 500 for k in files}
    _make_dist(site, "eta", files, record_sizes = sizes)
    assert len(_deps().damaged_installed_files(limit = 3)) == 3


def test_findings_are_capped_when_the_files_are_deleted(site):
    files = {f"theta/m{i}.py": b"x" * 500 for i in range(40)}
    _make_dist(site, "theta", files)
    for rel in files:
        (site / rel).unlink()
    found = _deps().damaged_installed_files(limit = 3)
    assert len(found) == 3
    assert all("is missing" in line for line in found)


def test_a_clean_tree_passes_through(monkeypatch):
    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(studio._studio_deps, "damaged_installed_files", lambda *a, **k: [])
    studio._fail_if_install_damaged()


def test_duplicate_metadata_gets_its_own_actionable_failure(monkeypatch, capsys):
    import typer

    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(
        studio._studio_deps,
        "installed_metadata_conflicts",
        lambda *a, **k: [
            "unsloth: multiple metadata records "
            "(2026.8.12 at unsloth-2026.8.12.dist-info, "
            "2026.8.15 at unsloth-2026.8.15.dist-info)"
        ],
    )

    def _file_scan_must_not_run(*_args, **_kwargs):
        raise AssertionError("ambiguous RECORDs reached the file-damage scan")

    monkeypatch.setattr(studio._studio_deps, "damaged_installed_files", _file_scan_must_not_run)
    with pytest.raises(typer.Exit) as excinfo:
        studio._fail_if_install_damaged()

    assert excinfo.value.exit_code == 1
    err = capsys.readouterr().err
    assert "Unsloth package metadata is inconsistent" in err
    assert "cannot safely choose" in err
    assert "Recreate the managed environment before" in err
    assert "pip install" not in err
    assert "installed files are damaged" not in err
    assert "Unsloth will keep failing to start" not in err


@pytest.mark.parametrize("package", ["typer", "torch"])
def test_other_duplicate_metadata_warns_without_an_unsafe_command(monkeypatch, capsys, package):
    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)

    def conflicts(
        *_args,
        names = None,
        exclude_names = (),
    ):
        if names is not None:
            return []
        assert "unsloth" in exclude_names and "unsloth-zoo" in exclude_names
        return [
            f"{package}: multiple metadata records "
            f"(1.0 at {package}-1.0.dist-info, 2.0 at {package}-2.0.dist-info)"
        ]

    monkeypatch.setattr(studio._studio_deps, "installed_metadata_conflicts", conflicts)
    monkeypatch.setattr(studio._studio_deps, "damaged_installed_files", lambda: [])

    studio._fail_if_install_damaged()

    err = capsys.readouterr().err
    assert "Warning: some other packages have duplicate metadata" in err
    assert f"{package}: multiple metadata records" in err
    assert "skipped file verification" in err
    assert "original package source" in err
    assert "pip install" not in err


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
    assert "install.sh" in err or "install.ps1" in err
    assert "--no-verify" in err


def test_a_foreign_cli_stays_quiet(monkeypatch):
    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: True)

    def _never(*a, **k):
        raise AssertionError("the check ran against the wrong environment")

    monkeypatch.setattr(studio._studio_deps, "damaged_installed_files", _never)
    studio._fail_if_install_damaged()


def test_a_system_python_is_not_treated_as_the_managed_venv(monkeypatch, tmp_path):
    prefix = tmp_path / "usr"
    prefix.mkdir()
    monkeypatch.setattr(sys, "prefix", str(prefix))
    assert _deps().running_outside_managed_venv() is True

    (prefix / "pyvenv.cfg").write_text("home = /usr/bin\n")
    assert _deps().running_outside_managed_venv() is (_deps()._managed_root(()) is not None)


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


def test_update_exposes_verify_defaulting_on():
    import inspect

    opt = inspect.signature(_studio().update).parameters["verify"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "--verify/--no-verify" in decls
    assert getattr(opt, "default", None) is True


def test_the_verify_help_does_not_promise_an_import_check():
    import inspect

    opt = inspect.signature(_studio().update).parameters["verify"].default
    help_text = (getattr(opt, "help", "") or "").lower()
    assert "import" not in help_text
    assert "files" in help_text


def _run_update(monkeypatch, argv, verified):
    studio = _studio()

    class _NoopLauncherUpdate:
        def __enter__(self):
            return self

        def validate_launcher(self):
            pass

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(studio, "_ensure_studio_env_exported", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_WindowsLauncherUpdateTransaction", _NoopLauncherUpdate)
    monkeypatch.setattr(studio, "_run_setup_script", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_refresh_desktop_shortcuts", lambda *a, **k: None)
    monkeypatch.setattr(
        studio, "_fail_if_install_damaged", lambda package: verified.append(package)
    )
    return CliRunner().invoke(studio.studio_app, ["update", *argv])


def test_update_verifies_by_default(monkeypatch):
    verified = []
    result = _run_update(monkeypatch, [], verified)
    assert result.exit_code == 0, result.output
    assert verified == ["unsloth"]


def test_no_verify_skips_the_check(monkeypatch):
    verified = []
    result = _run_update(monkeypatch, ["--no-verify"], verified)
    assert result.exit_code == 0, result.output
    assert verified == []


def test_a_tauri_update_is_verified_too(monkeypatch):
    verified = []
    monkeypatch.setenv("UNSLOTH_TAURI_UPDATE", "1")
    result = _run_update(monkeypatch, [], verified)
    assert result.exit_code == 0, result.output
    assert verified == ["unsloth"]


@pytest.mark.parametrize(
    "system, expected",
    [
        ("Linux", "| UNSLOTH_STUDIO_HOME=/srv/studios/a sh"),
        ("Windows", "$env:UNSLOTH_STUDIO_HOME = '/srv/studios/a'; irm"),
    ],
)
def test_a_custom_root_is_carried_into_the_reinstall_command(monkeypatch, capsys, system, expected):
    import platform as _platform
    import typer

    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(
        studio._studio_deps, "damaged_installed_files", lambda *a, **k: ["x: y is missing"]
    )
    monkeypatch.setattr(studio, "STUDIO_HOME", Path("/srv/studios/a"))
    monkeypatch.setattr(studio, "_STUDIO_HOME_IS_CUSTOM", True)
    monkeypatch.setattr(_platform, "system", lambda: system)
    with pytest.raises(typer.Exit):
        studio._fail_if_install_damaged()
    assert expected in capsys.readouterr().err


def test_a_root_with_spaces_is_quoted(monkeypatch, capsys):
    import platform as _platform
    import typer

    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(
        studio._studio_deps, "damaged_installed_files", lambda *a, **k: ["x: y is missing"]
    )
    monkeypatch.setattr(studio, "STUDIO_HOME", Path("/srv/my studios/a"))
    monkeypatch.setattr(studio, "_STUDIO_HOME_IS_CUSTOM", True)
    monkeypatch.setattr(_platform, "system", lambda: "Linux")
    with pytest.raises(typer.Exit):
        studio._fail_if_install_damaged()
    assert "UNSLOTH_STUDIO_HOME='/srv/my studios/a' sh" in capsys.readouterr().err


@pytest.mark.parametrize(
    "system, expected",
    [
        ("Linux", "| UNSLOTH_NO_TORCH=1 sh"),
        ("Windows", "$env:UNSLOTH_NO_TORCH = '1'; irm"),
    ],
)
def test_a_no_torch_install_keeps_that_mode_in_the_reinstall(monkeypatch, capsys, system, expected):
    import platform as _platform
    import typer

    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(
        studio._studio_deps, "damaged_installed_files", lambda *a, **k: ["x: y is missing"]
    )
    # The manifest and marker live in the venv, not STUDIO_HOME.
    seen = {}

    def _module(*a, **k):
        def _recorded(root = None):
            seen["root"] = root
            return True

        return SimpleNamespace(recorded_no_torch = _recorded)

    monkeypatch.setattr(studio._studio_deps, "load_install_manifest_module", _module)
    monkeypatch.setattr(_platform, "system", lambda: system)
    with pytest.raises(typer.Exit):
        studio._fail_if_install_damaged()
    assert expected in capsys.readouterr().err
    assert seen["root"] is None


@pytest.mark.parametrize("recorded", [False, None])
def test_an_unrecorded_or_torch_install_does_not_gain_the_flag(monkeypatch, capsys, recorded):
    import platform as _platform
    import typer

    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(
        studio._studio_deps, "damaged_installed_files", lambda *a, **k: ["x: y is missing"]
    )
    monkeypatch.setattr(
        studio._studio_deps,
        "load_install_manifest_module",
        lambda *a, **k: SimpleNamespace(recorded_no_torch = lambda **kw: recorded),
    )
    monkeypatch.setattr(_platform, "system", lambda: "Linux")
    with pytest.raises(typer.Exit):
        studio._fail_if_install_damaged()
    assert "UNSLOTH_NO_TORCH" not in capsys.readouterr().err


def test_the_default_root_keeps_the_plain_command(monkeypatch, capsys):
    import platform as _platform
    import typer

    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(
        studio._studio_deps, "damaged_installed_files", lambda *a, **k: ["x: y is missing"]
    )
    monkeypatch.setattr(studio, "_STUDIO_HOME_IS_CUSTOM", False)
    monkeypatch.setattr(_platform, "system", lambda: "Linux")
    with pytest.raises(typer.Exit):
        studio._fail_if_install_damaged()
    err = capsys.readouterr().err
    assert "curl -fsSL https://unsloth.ai/install.sh | sh" in err
    assert "UNSLOTH_STUDIO_HOME" not in err


def test_the_message_covers_packages_the_installer_will_not_repair(monkeypatch, capsys):
    import typer

    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(
        studio._studio_deps,
        "damaged_installed_files",
        lambda *a, **k: ["orphan: o/x.py is missing"],
    )
    with pytest.raises(typer.Exit):
        studio._fail_if_install_damaged()
    err = capsys.readouterr().err
    assert "still listed after that" in err
    assert "--force-reinstall" in err
    assert "--no-deps" in err
    assert "<package>==<installed version>" in err
    assert "--no-verify" in err


@pytest.mark.parametrize(
    "system, exe, expected",
    [
        ("Linux", "/srv/my studios/a/bin/python", "'/srv/my studios/a/bin/python' -m pip"),
        ("Windows", r"C:\my studios\a\python.exe", "& 'C:\\my studios\\a\\python.exe' -m pip"),
    ],
)
def test_the_repair_command_quotes_the_interpreter(monkeypatch, capsys, system, exe, expected):
    import platform as _platform
    import typer

    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(
        studio._studio_deps,
        "damaged_installed_files",
        lambda *a, **k: ["orphan: o/x.py is missing"],
    )
    monkeypatch.setattr(_platform, "system", lambda: system)
    monkeypatch.setattr(sys, "executable", exe)
    with pytest.raises(typer.Exit):
        studio._fail_if_install_damaged()
    assert expected in capsys.readouterr().err


def test_a_shared_top_level_test_tree_is_not_damage(site):
    _make_dist(site, "einx", {"einx/__init__.py": b"e\n"})
    (site / "einx-1.0.dist-info" / "RECORD").write_text(
        "einx/__init__.py,sha256=x,2\ntest/conftest.py,sha256=x,20650\n"
    )
    assert _deps().damaged_installed_files() == []


def test_an_installer_rewritten_lockfile_is_not_damage(site):
    lock = "studio/backend/core/data_recipe/oxc-validator/package-lock.json"
    _make_dist(
        site,
        "unsloth",
        {"unsloth/__init__.py": b"u\n", lock: b"L" * 27225},
        record_sizes = {lock: 28473},
    )
    assert _deps().damaged_installed_files() == []


def test_a_deleted_installer_rewritten_file_is_still_damage(site):
    lock = "studio/backend/core/data_recipe/oxc-validator/package-lock.json"
    _make_dist(site, "unsloth", {"unsloth/__init__.py": b"u\n", lock: b"L" * 100})
    (site / lock).unlink()
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "package-lock.json is missing" in found[0]


def test_a_shared_top_level_scripts_tree_is_not_damage(site):
    _make_dist(site, "upsilon", {"upsilon/__init__.py": b"u\n"})
    (site / "upsilon-1.0.dist-info" / "RECORD").write_text(
        "upsilon/__init__.py,sha256=x,2\nscripts/helper.py,sha256=x,99\n"
    )
    assert _deps().damaged_installed_files() == []


def test_a_package_owned_tests_subdirectory_is_still_checked(site):
    _make_dist(site, "rho", {"rho/tests/helper.py": b"h\n"})
    (site / "rho" / "tests" / "helper.py").unlink()
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "rho/tests/helper.py is missing" in found[0]


def test_a_top_level_module_named_like_a_test_root_is_still_checked(site):
    _make_dist(site, "sigma", {"tests.py": b"t\n"})
    (site / "tests.py").unlink()
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "tests.py is missing" in found[0]


def test_runtime_damage_still_fails_when_ignored_rows_are_present(site):
    lock = "studio/backend/core/data_recipe/oxc-validator/package-lock.json"
    _make_dist(
        site,
        "unsloth",
        {"unsloth/__init__.py": b"u\n", lock: b"L" * 10},
        record_sizes = {lock: 28473},
    )
    (site / "unsloth" / "__init__.py").unlink()
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "unsloth/__init__.py is missing" in found[0]


def test_ignored_rows_do_not_consume_the_finding_budget(site):
    # Filtered while RECORD is read: unfiltered, these 40 would fill limit = 3.
    files = {f"test/t{i}.py": b"x" for i in range(40)}
    files["tau/__init__.py"] = b"t\n"
    _make_dist(site, "tau", files)
    for rel in files:
        (site / rel).unlink()
    found = _deps().damaged_installed_files(limit = 3)
    assert len(found) == 1 and "tau/__init__.py is missing" in found[0]


def test_our_own_shared_top_level_trees_are_exempt_too(site):
    conftest = "tests/conftest.py"
    _make_dist(
        site,
        "unsloth_zoo",
        {"unsloth_zoo/__init__.py": b"z\n", conftest: b"c" * 8107},
        record_sizes = {conftest: 11429},
    )
    assert _deps().damaged_installed_files() == []


def test_two_distributions_claiming_one_shared_path(site):
    rel = "tests/conftest.py"
    _make_dist(site, "unsloth_zoo", {rel: b"u" * 11429})
    _make_dist(site, "upsilon", {rel: b"c" * 8107})
    assert (site / rel).stat().st_size == 8107
    assert _deps().damaged_installed_files() == []
    (site / rel).unlink()
    assert _deps().damaged_installed_files() == []


def test_our_own_shared_top_level_trees_may_also_vanish(site):
    _make_dist(site, "unsloth_zoo", {"unsloth_zoo/__init__.py": b"z\n"})
    (site / "unsloth_zoo-1.0.dist-info" / "RECORD").write_text(
        "unsloth_zoo/__init__.py,sha256=x,2\nscripts/helper.py,sha256=x,99\n"
    )
    assert _deps().damaged_installed_files() == []


def test_our_own_runtime_trees_are_still_checked(site):
    _make_dist(
        site,
        "unsloth_zoo",
        {"unsloth_zoo/__init__.py": b"z\n", "unsloth_zoo/tests/helper.py": b"h\n"},
    )
    (site / "unsloth_zoo" / "tests" / "helper.py").unlink()
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "unsloth_zoo/tests/helper.py is missing" in found[0]
