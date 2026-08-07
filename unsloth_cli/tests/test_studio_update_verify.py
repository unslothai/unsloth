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


def test_deletion_is_seen_whatever_Distribution_files_does(site):
    # Distribution.files is not a usable basis for this check, and it is not
    # consistent either: newer CPython filters out entries whose file is gone
    # (so a deletion becomes invisible), older CPython lists them but with a
    # path that does not exist. Which one you get depends on the interpreter,
    # and this project supports >= 3.9, so pinning one behaviour would make the
    # test fail on the other. RECORD is parsed directly instead, which reports
    # the deletion on every version.
    import importlib.metadata as md

    _make_dist(site, "gamma", {"gamma/a.py": b"a\n", "gamma/b.py": b"bb\n"})
    (site / "gamma" / "b.py").unlink()
    stale = [f for f in (md.distribution("gamma").files or []) if str(f) == "gamma/b.py"]
    # Either it was dropped, or it is listed and locate() does not resolve.
    # Both mean files() cannot tell you the file is gone.
    assert not stale or not stale[0].locate().exists()
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


def test_a_shared_file_shorter_than_recorded_is_not_damage(site):
    # The mirror of the case above. Two distributions claiming one path is a
    # packaging collision, and whichever copy landed is the one on disk, so its
    # size says nothing about either RECORD -- in either direction. Only the
    # larger direction was excluded, so a collision that overwrote with a
    # shorter file was reported as corruption and blocked every update.
    _make_dist(
        site, "iota", {"shared/__init__.py": b"short\n"}, record_sizes = {"shared/__init__.py": 900}
    )
    _make_dist(
        site, "kappa", {"shared/__init__.py": b"short\n"}, record_sizes = {"shared/__init__.py": 5}
    )
    assert _deps().damaged_installed_files() == []


def test_a_singly_owned_short_file_is_still_damage(site):
    # The collision rule must not become a blanket exemption.
    _make_dist(site, "lam", {"lam/a.py": b"x"}, record_sizes = {"lam/a.py": 900})
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "lam/a.py" in found[0]


def test_the_scan_is_limited_to_this_interpreters_site_packages(monkeypatch, tmp_path):
    # distributions() searches every sys.path entry, so a damaged distribution
    # reachable only through an inherited PYTHONPATH failed every update while
    # sitting outside the installation, where neither printed repair command can
    # reach it. Only --no-verify broke the loop.
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
    # The external tree is on sys.path but not in the scan paths.
    assert _deps().damaged_installed_files() == []

    # And the same tree IS reported once it is what the scan points at.
    monkeypatch.setattr(_deps(), "_scan_paths", lambda: {"path": [str(external)]})
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "ext/mod.py" in found[0]


def test_a_deleted_shared_file_is_still_reported(site):
    # Multiple ownership makes the recorded SIZES ambiguous; it cannot explain
    # the file being gone. Skipping shared paths outright hid real deletions.
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
    # The size field is optional and real wheels leave it blank. Dropping those
    # rows meant a deleted file was never reported.
    _make_dist(site, "xi", {"xi/__init__.py": b"y\n"})
    (site / "xi-1.0.dist-info" / "RECORD").write_text("xi/__init__.py,,\n")
    (site / "xi" / "__init__.py").unlink()
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "xi/__init__.py is missing" in found[0]


def test_a_directory_standing_in_for_a_module_is_damage(site):
    # An empty directory is commonly 4096 bytes on POSIX, so it sails past the
    # shrinkage test while importing as something other than the recorded module.
    _make_dist(site, "omicron", {}, record_sizes = {})
    (site / "omicron-1.0.dist-info" / "RECORD").write_text("omicron/mod.py,sha256=x,10\n")
    (site / "omicron" / "mod.py").mkdir(parents = True)
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "not a regular file" in found[0]


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


def test_findings_are_capped_when_the_files_are_deleted(site):
    # Truncation and deletion take different branches, and only truncation was
    # covered. A wiped package -- `rm -rf` on a venv's torch, the shape a user
    # actually hits -- takes the deletion branch, so an uncapped one floods the
    # caller with a line per RECORD entry (~11.8k for torch), each of which the
    # desktop updater turns into its own IPC event.
    files = {f"theta/m{i}.py": b"x" * 500 for i in range(40)}
    _make_dist(site, "theta", files)
    for rel in files:
        (site / rel).unlink()
    found = _deps().damaged_installed_files(limit = 3)
    assert len(found) == 3
    assert all("is missing" in line for line in found)


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


def test_a_system_python_is_not_treated_as_the_managed_venv(monkeypatch, tmp_path):
    # Colab has no Unsloth venv: studio/setup.sh installs the backend into the
    # system Python on purpose. Distro-packaged RECORDs there list files the
    # distro never installed (PEP 627), so running the file check would accuse
    # the distro of damaging Studio. Reproduced on Ubuntu system Python, which
    # reports an apt-owned `markdown-it-py: ../scripts/markdown-it is missing`.
    prefix = tmp_path / "usr"
    prefix.mkdir()
    monkeypatch.setattr(sys, "prefix", str(prefix))
    assert _deps().running_outside_managed_venv() is True

    (prefix / "pyvenv.cfg").write_text("home = /usr/bin\n")
    # With a real venv the answer goes back to the managed-root question.
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


# ── the update command wiring ────────────────────────────────────────


def test_update_exposes_verify_defaulting_on():
    import inspect

    opt = inspect.signature(_studio().update).parameters["verify"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "--verify/--no-verify" in decls
    assert getattr(opt, "default", None) is True


def test_the_verify_help_does_not_promise_an_import_check():
    # The scan compares RECORD entries against the filesystem and imports
    # nothing, so it cannot see same-size corruption or an intact but
    # incompatible package. Saying it checks that the backend still imports
    # promises a stronger guarantee than it delivers.
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


@pytest.mark.parametrize(
    "system, expected",
    [
        ("Linux", "| UNSLOTH_STUDIO_HOME=/srv/studios/a sh"),
        ("Windows", "$env:UNSLOTH_STUDIO_HOME = '/srv/studios/a'; irm"),
    ],
)
def test_a_custom_root_is_carried_into_the_reinstall_command(monkeypatch, capsys, system, expected):
    # The CLI shim is a bare symlink and _ensure_studio_env_exported only sets
    # os.environ for this process, so the shell that runs the printed command
    # has no UNSLOTH_STUDIO_HOME. Unqualified, it would build a fresh
    # ~/.unsloth/studio and leave the damaged custom root broken.
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
    # install.sh derives SKIP_TORCH from its flag or UNSLOTH_NO_TORCH only, so
    # following the plain command on a GGUF-only install pulls the whole PyTorch
    # stack -- multiple GB the user deliberately opted out of.
    import platform as _platform
    import typer

    studio = _studio()
    monkeypatch.setattr(studio._studio_deps, "running_outside_managed_venv", lambda *a: False)
    monkeypatch.setattr(
        studio._studio_deps, "damaged_installed_files", lambda *a, **k: ["x: y is missing"]
    )
    # The stub records what root it was asked for: the manifest and marker live
    # in the venv, and an earlier attempt at this passed STUDIO_HOME, which is
    # one directory too high, so it read None and never fired in production.
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
    # Default root, i.e. sys.prefix, which the early return guarantees is the venv.
    assert seen["root"] is None


@pytest.mark.parametrize("recorded", [False, None])
def test_an_unrecorded_or_torch_install_does_not_gain_the_flag(monkeypatch, capsys, recorded):
    # recorded_no_torch() returns None when nothing recorded the mode, and its
    # contract is that None is not False. Adding the flag on a guess would leave
    # a torch install without torch, so the flag is added only on an explicit True.
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
    # install_python_stack installs the current requirement sets and prunes
    # nothing, and the installer never recreates the venv, so damage in an
    # orphan from an older release survives the reinstall it recommends and
    # would report the same failure forever. The scan is deliberately not
    # scoped to Studio's dependency closure: under-including there would let
    # real damage through, which is the failure this whole check exists to
    # catch. So the message has to carry the fallback instead.
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
    # Without --no-deps, pip resolves the damaged package's graph and
    # --force-reinstall can swap the pinned CUDA/ROCm torch build.
    assert "--no-deps" in err
    # A bare name would let --force-reinstall upgrade the orphan rather than
    # repair it, which --no-deps does not prevent.
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
    # Custom roots with spaces are supported, so an unquoted sys.executable
    # would split into several shell tokens and the command would not run.
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


# ── runtime-irrelevant rows must not fail an update ──────────────────


def test_a_shared_top_level_test_tree_is_not_damage(site):
    # Reported as `einx: test/conftest.py is missing`. einx and torchao both
    # ship it, and install_python_stack.py force-reinstalls torchao every
    # update, so pip removes the file and the pinned torchao does not ship it.
    # Nothing imports another project's fixtures, and no reinstall repairs it.
    _make_dist(site, "einx", {"einx/__init__.py": b"e\n"})
    (site / "einx-1.0.dist-info" / "RECORD").write_text(
        "einx/__init__.py,sha256=x,2\ntest/conftest.py,sha256=x,20650\n"
    )
    assert _deps().damaged_installed_files() == []


def test_an_installer_rewritten_lockfile_is_not_damage(site):
    # Reported as `package-lock.json is 27225 bytes, expected 28473`.
    # setup.ps1/setup.sh run `npm install` inside the installed tree, and npm
    # dedupes hoisted entries under legacy-peer-deps, shrinking the file.
    lock = "studio/backend/core/data_recipe/oxc-validator/package-lock.json"
    _make_dist(site, "unsloth", {"unsloth/__init__.py": b"u\n", lock: b"L" * 27225},
               record_sizes = {lock: 28473})
    assert _deps().damaged_installed_files() == []


def test_a_package_owned_tests_subdirectory_is_still_checked(site):
    # Only the shared top-level namespace is exempt; a tests/ tree inside a
    # package is that package's alone, so a deletion there is real.
    _make_dist(site, "rho", {"rho/tests/helper.py": b"h\n"})
    (site / "rho" / "tests" / "helper.py").unlink()
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "rho/tests/helper.py is missing" in found[0]


def test_a_top_level_module_named_like_a_test_root_is_still_checked(site):
    # The exemption is for a shared directory, not for a name prefix.
    _make_dist(site, "sigma", {"tests.py": b"t\n"})
    (site / "tests.py").unlink()
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "tests.py is missing" in found[0]


def test_runtime_damage_still_fails_when_ignored_rows_are_present(site):
    # The exemption must not blind the scan to a torn runtime module.
    lock = "studio/backend/core/data_recipe/oxc-validator/package-lock.json"
    _make_dist(site, "unsloth", {"unsloth/__init__.py": b"u\n", lock: b"L" * 10},
               record_sizes = {lock: 28473})
    (site / "unsloth" / "__init__.py").unlink()
    found = _deps().damaged_installed_files()
    assert len(found) == 1 and "unsloth/__init__.py is missing" in found[0]


def test_ignored_rows_do_not_consume_the_finding_budget(site):
    # Filtering happens while RECORD is read, so harmless rows cannot crowd a
    # real one off a capped list. Unfiltered, these 40 fill limit = 3.
    files = {f"test/t{i}.py": b"x" for i in range(40)}
    files["tau/__init__.py"] = b"t\n"
    _make_dist(site, "tau", files)
    for rel in files:
        (site / rel).unlink()
    found = _deps().damaged_installed_files(limit = 3)
    assert len(found) == 1 and "tau/__init__.py is missing" in found[0]
