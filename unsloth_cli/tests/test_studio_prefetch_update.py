# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ast
import json
import os
import platform
import subprocess
import time
import sys
import threading
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli import _studio_prefetch  # noqa: E402

STUDIO_COMMAND = _REPO_ROOT / "unsloth_cli" / "commands" / "studio.py"
INSTALL_PYTHON_STACK = _REPO_ROOT / "studio" / "install_python_stack.py"
# The source form of the no-torch core step's label, as ast.unparse renders it.
NO_TORCH_CORE_LABEL = "f'Updating {package_name} + unsloth-zoo (no-torch mode)'"


# ── Dry-run plan parsing ──


def test_plan_lines_are_read_from_either_stream_and_either_line_ending():
    output = (
        "Using Python 3.12.12 environment at: unsloth_studio\r\n"
        "Resolved 4 packages in 108ms\r\n"
        "Would download 1 package\r\n"
        " - unsloth==2026.8.1\r\n"
        " + unsloth==2026.9.2\r\n"
        " + unsloth-zoo==2026.9.1\r"
    )

    assert _studio_prefetch.parse_dry_run_plan(output) == {
        "unsloth": "2026.9.2",
        "unsloth-zoo": "2026.9.1",
    }


def test_removals_and_local_tag_pins_are_left_out_of_the_plan():
    output = "\n".join(
        [
            " - numpy==2.1.0",
            " + numpy==2.2.0",
            # A pinned index or a --torch-backend produced this; it cannot be asked
            # for again as a bare name==version, so it is not prefetchable.
            " + torch==2.9.0+cu128",
            " + nvidia-cublas-cu12==12.8.4.1",
        ]
    )

    assert _studio_prefetch.parse_dry_run_plan(output) == {
        "numpy": "2.2.0",
        "nvidia-cublas-cu12": "12.8.4.1",
    }


def test_a_direct_url_entry_keeps_only_its_pin():
    output = " + unsloth-zoo==2026.9.1 (from file:///tmp/wheels/unsloth_zoo.whl)\n"

    assert _studio_prefetch.parse_dry_run_plan(output) == {"unsloth-zoo": "2026.9.1"}


def test_plan_names_are_normalised_so_the_pin_matches_the_index():
    assert _studio_prefetch.parse_dry_run_plan(" + unsloth_zoo==1.0\n") == {"unsloth-zoo": "1.0"}
    assert _studio_prefetch.pins_from_plan({"unsloth-zoo": "1.0"}) == ["unsloth-zoo==1.0"]


# ── The core command must not drift from the installer's ──


def _installer_core_step_arguments(label: str) -> list[str]:
    """The positional arguments of one core `pip_install` call in the installer.

    `label` is the SOURCE form of the call's first argument, because one of the two
    branches names itself with an f-string.

    Read out of the installer rather than copied, so a change to that call site
    fails this test instead of silently making the prefetch warm the wrong wheels.
    """
    tree = ast.parse(INSTALL_PYTHON_STACK.read_text(encoding = "utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "pip_install"
        and node.args
        # The no-torch label is an f-string, so compare the unparsed source form.
        and ast.unparse(node.args[0]) == label
        # The --local branch passes the literal "unsloth"; the branches the desktop
        # runs pass the floor-aware spec.
        and any(isinstance(arg, ast.Name) and arg.id == "unsloth_spec" for arg in node.args)
    ]
    assert len(calls) == 1, f"the installer's {label!r} core step moved or was duplicated"
    arguments: list[str] = []
    for argument in calls[0].args[1:]:
        if isinstance(argument, ast.Constant):
            arguments.append(argument.value)
        elif isinstance(argument, ast.Name) and argument.id in ("unsloth_spec", "package_name"):
            arguments.append("<spec>" if argument.id == "unsloth_spec" else "unsloth")
        else:  # pragma: no cover - a new argument shape needs a decision, not a guess
            raise AssertionError(f"unhandled core step argument: {ast.dump(argument)}")
    return arguments


def _expected_tail(label: str, floor: str) -> list[str]:
    installer = _installer_core_step_arguments(label)
    # _translate_pip_args_for_uv drops this on the uv path, so the prefetch does too.
    assert "--no-cache-dir" in installer
    return [
        f"unsloth>={floor}" if argument == "<spec>" else argument
        for argument in installer
        if argument != "--no-cache-dir"
    ]


def test_the_core_dry_run_is_the_installers_core_step_plus_dry_run(tmp_path):
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("numpy<3\n", encoding = "utf-8")
    python = tmp_path / "unsloth_studio" / "bin" / "python"

    command = _studio_prefetch.core_dry_run_command(
        python, floor = "2026.9.2", constraints = constraints
    )

    assert command == (
        ["uv", "pip", "install", "--python", str(python), "--dry-run"]
        + _expected_tail("'Updating core packages'", "2026.9.2")
        + ["-c", str(constraints)]
    )


def test_a_no_torch_install_resolves_the_core_step_with_no_deps(tmp_path):
    """Without it the resolver plans torch and every nvidia wheel behind it.

    unsloth's PyPI metadata makes torch a hard dependency, and a GGUF-only venv has
    none of it installed to satisfy that, so the plain resolve returns the whole
    CUDA stack. Measured before this branch pinned it: 24 packages, 2.7 GB
    downloaded, for an update that installs two wheels.
    """
    python = tmp_path / "unsloth_studio" / "bin" / "python"

    command = _studio_prefetch.core_dry_run_command(
        python, floor = "2026.9.2", constraints = None, no_torch = True
    )

    assert command == (
        ["uv", "pip", "install", "--python", str(python), "--dry-run"]
        + _expected_tail(NO_TORCH_CORE_LABEL, "2026.9.2")
    )
    assert "--no-deps" in command


def test_a_requirement_file_is_filtered_the_way_the_installer_filters_it(tmp_path):
    requirement = tmp_path / "extras.txt"
    requirement.write_text(
        "# audio\n"
        "-r base.txt\n"
        "librosa>=0.10\n"
        "openai_whisper==20250625\n"
        "soundfile\n"
        "timm ; sys_platform != 'darwin'\n",
        encoding = "utf-8",
    )
    work = tmp_path / "req"

    filtered = _studio_prefetch.effective_requirements(
        requirement, _studio_prefetch.NO_TORCH_SKIP_PACKAGES, work
    )

    assert filtered != requirement
    # Beside the source, as install_python_stack._filter_requirements does it, so the
    # `-r base.txt` copied through still resolves.
    assert filtered.parent == requirement.parent
    assert not work.exists()
    assert filtered.read_text(encoding = "utf-8") == "# audio\n-r base.txt\nsoundfile\n"
    # openai_whisper and openai-whisper are the same distribution; timm carries a marker.
    assert "librosa" not in filtered.read_text(encoding = "utf-8")


def test_a_requirement_file_with_nothing_to_skip_is_used_as_it_stands(tmp_path):
    requirement = tmp_path / "studio.txt"
    requirement.write_text("fastapi\nuvicorn\n", encoding = "utf-8")

    assert _studio_prefetch.effective_requirements(requirement, (), tmp_path / "req") is requirement
    assert (
        _studio_prefetch.effective_requirements(
            requirement, _studio_prefetch.NO_TORCH_SKIP_PACKAGES, tmp_path / "req"
        )
        is requirement
    )
    assert not (tmp_path / "req").exists()


def test_the_core_dry_run_falls_back_to_a_bare_name_without_a_floor(tmp_path):
    command = _studio_prefetch.core_dry_run_command(tmp_path / "python", floor = "", constraints = None)

    assert "unsloth" in command
    assert not any(argument.startswith("unsloth>=") for argument in command)
    assert "-c" not in command


def test_the_torch_backend_follows_the_installers_rule(monkeypatch, tmp_path):
    monkeypatch.setenv("UV_TORCH_BACKEND", "auto")

    command = _studio_prefetch.core_dry_run_command(
        tmp_path / "python", floor = "1.0", constraints = None
    )
    assert "--torch-backend=auto" in command

    # _build_uv_cmd never adds it on a pinned index; neither does this.
    assert "--torch-backend=auto" not in _studio_prefetch._torch_backend_argument(
        ["uv", "pip", "install", "--index-url", "https://example.invalid/simple"]
    )


def test_the_fetch_keeps_every_byte_out_of_the_venv(tmp_path):
    command = _studio_prefetch.fetch_command(
        tmp_path / "python", tmp_path / "site", ["unsloth==1.0"], only_binary = True
    )

    assert command[:3] == ["uv", "pip", "install"]
    assert "--target" in command
    assert command[command.index("--target") + 1] == str(tmp_path / "site")
    assert "--no-deps" in command
    assert command[command.index("--only-binary") + 1] == ":all:"
    assert "unsloth==1.0" in command
    # No --system, no --break-system-packages, nothing that could write to a venv.
    assert "--system" not in command


def test_a_wheel_less_requirement_is_left_to_swap_time_rather_than_losing_the_file():
    """uv refuses the whole `--only-binary :all:` command over one such pin.

    openai-whisper and friends have no wheel at any version, so leaving them in the
    pin list loses every other package in the file, which is what the first end-to-end
    run showed: `extras.txt` skipped with "no usable wheel" and nothing warmed.
    """
    planned = {"openai-whisper": "20250625", "soundfile": "0.13.1", "argbind": "0.3.9"}

    assert _studio_prefetch.pins_from_plan(planned, only_binary = True) == ["soundfile==0.13.1"]
    # The core fetch has no such flag and builds nothing, so it keeps every pin.
    assert len(_studio_prefetch.pins_from_plan(planned)) == 3


def test_a_plan_uv_announced_but_this_parser_could_not_read_is_not_an_empty_plan():
    # The shape uv prints today.
    assert _studio_prefetch.plan_is_readable(
        "Would install 2 packages\n + a==1\n + b==2\n", {"a": "1", "b": "2"}
    )
    # Fewer pins than announced is fine: local tags are dropped on purpose.
    assert _studio_prefetch.plan_is_readable(
        "Would install 2 packages\n + a==1\n + torch==2.9.0+cu128\n", {"a": "1"}
    )
    assert _studio_prefetch.plan_is_readable("Resolved 4 packages in 8ms\n", {})
    # A plan that is nothing but local-tag pins reads fine and prepares nothing.
    assert _studio_prefetch.plan_is_readable("Would install 1 package\n + torch==2.9.0+cu128\n", {})
    # Announced installs and nothing parsed: the format moved.
    assert not _studio_prefetch.plan_is_readable("Would install 2 packages\n> a 1\n> b 2\n", {})
    assert _studio_prefetch.planned_install_count("Would install 1 package\n") == 1
    assert _studio_prefetch.planned_install_count("nothing to say") is None


# ── Floors ──


def test_a_post_release_of_the_floor_still_meets_it():
    assert _studio_prefetch.version_meets_floor("2026.9.2", "2026.9.2")
    assert _studio_prefetch.version_meets_floor("2026.9.2.post1", "2026.9.2")
    assert _studio_prefetch.version_meets_floor("2026.9.3", "2026.9.2")
    assert not _studio_prefetch.version_meets_floor("2026.9.1", "2026.9.2")


# ── run() ──


class _Recorder:
    def __init__(self, responses):
        self.responses = responses
        self.commands: list[list[str]] = []
        self.environments: list[dict] = []

    def __call__(self, cmd, env):
        self.commands.append(list(cmd))
        self.environments.append(dict(env or {}))
        for match, response in self.responses:
            if match(list(cmd)):
                return response
        return subprocess.CompletedProcess(list(cmd), 0, "", "")


def _completed(
    returncode = 0,
    stderr = "",
    stdout = "",
):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


@pytest.fixture
def managed(monkeypatch, tmp_path):
    """A Studio home whose venv is the interpreter this test pretends to be."""
    home = tmp_path / "studio"
    venv = home / _studio_prefetch.VENV_NAME
    site = venv / "lib" / "python3.12" / "site-packages"
    (site / "studio" / "backend" / "requirements" / "single-env").mkdir(parents = True)
    (site / "studio" / "backend" / "requirements" / "single-env" / "constraints.txt").write_text(
        "anyio<4.14\n", encoding = "utf-8"
    )
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding = "utf-8")
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(_studio_prefetch.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(_studio_prefetch, "_is_editable_install", lambda name = "unsloth": False)
    monkeypatch.setattr(_studio_prefetch, "_installed_version", lambda name: "2026.8.1")
    monkeypatch.delenv("UV_NO_CACHE", raising = False)
    monkeypatch.delenv("STUDIO_LOCAL_INSTALL", raising = False)
    return home


def _plan_response(text):
    return _completed(0, stderr = text)


def _install_new_wheel_tree(target: Path) -> None:
    """What the core fetch leaves behind: the NEW wheel's studio/ tree."""
    requirements = target / "studio" / "backend" / "requirements" / "single-env"
    requirements.mkdir(parents = True, exist_ok = True)
    (requirements.parent / "studio.txt").write_text("fastapi\n", encoding = "utf-8")
    (requirements.parent / "base.txt").write_text("numpy\n", encoding = "utf-8")
    (requirements / "constraints.txt").write_text("anyio<4.14\n", encoding = "utf-8")
    dist_info = target / "unsloth-2026.9.2.dist-info"
    dist_info.mkdir(parents = True, exist_ok = True)
    (dist_info / "RECORD").write_text("unsloth/__init__.py,,\n", encoding = "utf-8")


def test_a_successful_prefetch_writes_a_ready_marker_and_never_touches_the_venv(
    managed, monkeypatch
):
    target = _studio_prefetch.site_dir(managed)

    def respond(cmd):
        if "--dry-run" in cmd and any(a.startswith("unsloth>=") for a in cmd):
            return _plan_response(" + unsloth==2026.9.2\n + unsloth-zoo==2026.9.1\n")
        if "--dry-run" in cmd:
            return _plan_response(" + fastapi==0.120.0\n")
        if "--target" in cmd:
            _install_new_wheel_tree(target)
            return _completed(0)
        return _completed(0)

    recorder = _Recorder([(lambda cmd: True, None)])
    monkeypatch.setattr(_studio_prefetch, "_run", lambda cmd, env: respond(list(cmd)))
    monkeypatch.setattr(_studio_prefetch, "_uv_version", lambda uv, env: "uv 0.12.1")

    payload = _studio_prefetch.run(
        studio_home = managed,
        floor = "2026.9.2",
        shell_version = "0.1.900-beta",
        env = {"UV_CACHE_DIR": "/cache/uv"},
        echo = lambda line: None,
    )

    assert payload["state"] == "ready"
    assert payload["backend_version"] == "2026.9.2"
    assert payload["zoo_version"] == "2026.9.1"
    assert payload["cache_dir"] == "/cache/uv"
    assert payload["shell_version"] == "0.1.900-beta"
    assert payload["core_records"] == {
        "unsloth-2026.9.2.dist-info": _studio_prefetch.digest_file(
            target / "unsloth-2026.9.2.dist-info" / "RECORD"
        )
    }
    assert set(payload["requirement_digests"]) == {"studio.txt", "base.txt"}

    on_disk = json.loads(_studio_prefetch.marker_path(managed).read_text(encoding = "utf-8"))
    assert on_disk == payload
    assert (_studio_prefetch.prefetch_root(managed) / _studio_prefetch.OWNED_MARKER).is_file()
    assert recorder.commands == []
    # site-packages of the live venv is untouched: everything landed under site/.
    live = managed / _studio_prefetch.VENV_NAME / "lib" / "python3.12" / "site-packages"
    assert sorted(entry.name for entry in live.iterdir()) == ["studio"]


def test_a_gguf_only_install_prepares_two_wheels_and_not_the_cuda_stack(managed, monkeypatch):
    """The regression this branch was measured into: 24 packages and 2.7 GB.

    A no-torch venv satisfies none of unsloth's torch dependency, so a plain
    resolve plans torch and every nvidia wheel behind it, and the requirement
    files the installer filters plan the rest.
    """
    (managed / _studio_prefetch.VENV_NAME / _studio_prefetch.NO_TORCH_MARKER).write_text(
        "", encoding = "utf-8"
    )
    target = _studio_prefetch.site_dir(managed)
    recorder = _Recorder([])

    def respond(cmd, env):
        recorder(cmd, env)
        cmd = list(cmd)
        if "--dry-run" in cmd:
            return _plan_response(" + unsloth==2026.9.2\n")
        if "--target" in cmd:
            _install_new_wheel_tree(target)
            (target / "studio" / "backend" / "requirements" / "extras.txt").write_text(
                "librosa>=0.10\nopenai_whisper==20250625\nsoundfile\n", encoding = "utf-8"
            )
            return _completed(0)
        return _completed(0)

    monkeypatch.setattr(_studio_prefetch, "_run", respond)
    _studio_prefetch.run(studio_home = managed, floor = "2026.9.2", echo = lambda line: None)

    core = recorder.commands[0]
    assert "--no-deps" in core, core
    # base.txt is the torch file; a no-torch install never runs it.
    assert not any(Path(c[-1]).name.startswith("base") for c in recorder.commands)
    extras = [c for c in recorder.commands if Path(c[-1]).name.startswith((".extras", "extras"))]
    assert extras, recorder.commands
    filtered = Path(extras[0][-1]).read_text(encoding = "utf-8")
    assert "librosa" not in filtered and "openai_whisper" not in filtered
    assert "soundfile" in filtered


def test_a_plan_without_unsloth_records_noop_and_downloads_nothing(managed, monkeypatch):
    recorder = _Recorder(
        [(lambda cmd: "--dry-run" in cmd, _plan_response("Resolved 4 packages\n"))]
    )
    monkeypatch.setattr(_studio_prefetch, "_run", recorder)

    payload = _studio_prefetch.run(studio_home = managed, floor = "2026.8.1", echo = lambda line: None)

    assert payload["state"] == "noop"
    assert payload["backend_version"] is None
    assert not any("--target" in command for command in recorder.commands)
    assert _studio_prefetch.marker_path(managed).is_file()


def test_a_zoo_only_bump_is_prepared_rather_than_recorded_as_nothing_to_do(managed, monkeypatch):
    """unsloth and unsloth-zoo release independently.

    Keying "nothing to prepare" off unsloth alone made a zoo-only update report a
    warm cache and then download unsloth-zoo at restart.
    """
    target = _studio_prefetch.site_dir(managed)

    def respond(cmd, env):
        cmd = list(cmd)
        if "--dry-run" in cmd:
            return _plan_response(" + unsloth-zoo==2026.9.5\n")
        if "--target" in cmd:
            _install_new_wheel_tree(target)
            return _completed(0)
        return _completed(0)

    recorder = _Recorder([])
    monkeypatch.setattr(
        _studio_prefetch, "_run", lambda cmd, env: (recorder(cmd, env), respond(cmd, env))[1]
    )

    payload = _studio_prefetch.run(studio_home = managed, echo = lambda line: None)

    assert payload["state"] in ("ready", "partial")
    assert payload["backend_version"] is None
    assert payload["zoo_version"] == "2026.9.5"
    fetches = [c for c in recorder.commands if "--target" in c]
    assert fetches, recorder.commands
    assert "unsloth-zoo==2026.9.5" in fetches[0]


def test_a_core_plan_this_parser_cannot_read_is_a_failure_not_a_noop(managed, monkeypatch):
    monkeypatch.setattr(
        _studio_prefetch,
        "_run",
        lambda cmd, env: _plan_response("Would install 2 packages\n> unsloth 2026.9.5\n"),
    )

    with pytest.raises(_studio_prefetch.PrefetchError) as failure:
        _studio_prefetch.run(studio_home = managed, echo = lambda line: None)

    assert "could not read the core plan" in str(failure.value)
    assert not _studio_prefetch.marker_path(managed).is_file()


def test_the_requirement_pass_stops_at_its_budget_instead_of_running_for_hours(
    managed, monkeypatch
):
    target = _studio_prefetch.site_dir(managed)

    def respond(cmd, env):
        cmd = list(cmd)
        if "--dry-run" in cmd:
            return _plan_response(" + unsloth==2026.9.5\n")
        if "--target" in cmd:
            _install_new_wheel_tree(target)
            return _completed(0)
        return _completed(0)

    monkeypatch.setattr(_studio_prefetch, "_run", respond)
    monkeypatch.setattr(_studio_prefetch, "BUDGET_SECONDS", -1)

    payload = _studio_prefetch.run(studio_home = managed, echo = lambda line: None)

    # The core packages are cached; the rest is what the update downloads anyway.
    assert payload["state"] == "partial"
    assert payload["backend_version"] == "2026.9.5"
    assert all(
        record.get("skipped_reason") == "out of time" for record in payload["requirements"].values()
    )


def test_a_plan_below_the_floor_is_an_error_and_leaves_no_marker(managed, monkeypatch):
    monkeypatch.setattr(
        _studio_prefetch,
        "_run",
        lambda cmd, env: _plan_response(" + unsloth==2026.9.1\n"),
    )

    with pytest.raises(_studio_prefetch.PrefetchError) as failure:
        _studio_prefetch.run(studio_home = managed, floor = "2026.9.2", echo = lambda line: None)

    assert "below the required 2026.9.2" in str(failure.value)
    assert not _studio_prefetch.marker_path(managed).is_file()


def test_a_failing_requirement_file_degrades_to_partial_rather_than_failing(managed, monkeypatch):
    target = _studio_prefetch.site_dir(managed)

    def respond(cmd):
        cmd = list(cmd)
        if "--dry-run" in cmd and any(a.startswith("unsloth>=") for a in cmd):
            return _plan_response(" + unsloth==2026.9.2\n")
        if "--dry-run" in cmd:
            return _completed(1, stderr = "no solution found")
        if "--target" in cmd:
            _install_new_wheel_tree(target)
            return _completed(0)
        return _completed(0)

    monkeypatch.setattr(_studio_prefetch, "_run", lambda cmd, env: respond(cmd))

    payload = _studio_prefetch.run(studio_home = managed, floor = "2026.9.2", echo = lambda line: None)

    assert payload["state"] == "partial"
    assert "resolve failed" in payload["requirements"]["studio.txt"]["skipped_reason"]


# ── Finding uv ──


UV_NAME = "uv.exe" if platform.system() == "Windows" else "uv"

# Every variable the installers read when they choose where to put uv, so a test
# that means "PATH has no uv" is not answered by the developer's own ~/.local/bin.
_UV_LOCATION_VARS = (
    "UV_INSTALL_DIR",
    "UV_UNMANAGED_INSTALL",
    "XDG_BIN_HOME",
    "XDG_DATA_HOME",
    "LOCALAPPDATA",
)


def _install_uv_at(directory: Path) -> Path:
    directory.mkdir(parents = True, exist_ok = True)
    uv = directory / UV_NAME
    uv.write_text("", encoding = "utf-8")
    return uv


def _without_uv_on_path(monkeypatch, home: Path) -> None:
    """A shell whose PATH predates the install, which is the Windows failure."""
    monkeypatch.setattr(_studio_prefetch.shutil, "which", lambda name: None)
    for name in _UV_LOCATION_VARS:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    # locate_uv runs a candidate before believing in it; nothing here is a real
    # binary, so the probe is what says whether the file counts.
    monkeypatch.setattr(
        _studio_prefetch,
        "_run",
        lambda cmd, env: _completed(0, stdout = "uv 0.12.1")
        if list(cmd)[1:] == ["--version"] and Path(list(cmd)[0]).is_file()
        else _completed(1),
    )


def _arrange_without_any_uv(monkeypatch, home: Path) -> None:
    _without_uv_on_path(monkeypatch, home.parent / "empty-home")


def test_uv_on_path_is_the_one_the_core_step_would_run(monkeypatch):
    monkeypatch.setattr(_studio_prefetch.shutil, "which", lambda name: "/usr/bin/uv")
    found, searched = _studio_prefetch.locate_uv({})
    assert found == "/usr/bin/uv"
    # Searched is what the message would name; PATH won before any of it was read.
    assert searched


def test_a_path_without_uv_still_finds_the_one_the_installer_put_in_local_bin(
    monkeypatch, tmp_path
):
    """setup.ps1 puts uv in %USERPROFILE%\\.local\\bin and prepends it to PATH.

    A desktop shell started before that registry write has a PATH without it, and
    the prefetch used to give up there while the update, which reinstalls uv,
    carried on. Both have to end up on the same binary or the cache is warmed for
    a resolver the swap will not use.
    """
    home = tmp_path / "home"
    _without_uv_on_path(monkeypatch, home)
    uv = _install_uv_at(home / ".local" / "bin")

    found, _ = _studio_prefetch.locate_uv({"HOME": str(home), "USERPROFILE": str(home)})

    assert found == str(uv)


def test_the_installer_destination_variables_outrank_the_default(monkeypatch, tmp_path):
    """astral's priority, which install.ps1 and install.sh both compute this way."""
    home = tmp_path / "home"
    _without_uv_on_path(monkeypatch, home)
    _install_uv_at(home / ".local" / "bin")
    pinned = _install_uv_at(tmp_path / "pinned")
    xdg = _install_uv_at(tmp_path / "xdg")

    env = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "UV_INSTALL_DIR": str(pinned.parent),
        "XDG_BIN_HOME": str(xdg.parent),
    }

    assert _studio_prefetch.locate_uv(env)[0] == str(pinned)
    del env["UV_INSTALL_DIR"]
    assert _studio_prefetch.locate_uv(env)[0] == str(xdg)


def test_a_file_that_cannot_run_is_not_taken_for_uv(monkeypatch, tmp_path):
    home = tmp_path / "home"
    _without_uv_on_path(monkeypatch, home)
    _install_uv_at(home / ".local" / "bin")
    monkeypatch.setattr(_studio_prefetch, "_run", lambda cmd, env: _completed(1))

    assert _studio_prefetch.locate_uv({"HOME": str(home), "USERPROFILE": str(home)})[0] is None


def test_the_skip_names_the_places_it_looked(managed, monkeypatch, tmp_path):
    home = tmp_path / "nowhere"
    _without_uv_on_path(monkeypatch, home)

    with pytest.raises(_studio_prefetch.PrefetchSkipped) as skipped:
        _studio_prefetch.run(
            studio_home = managed,
            env = {"HOME": str(home), "USERPROFILE": str(home)},
            echo = lambda line: None,
        )

    # "uv is not available" on its own leaves the reader guessing on the one
    # platform where this fires.
    reason = str(skipped.value)
    assert "looked on PATH and in" in reason
    assert str(home / ".local" / "bin") in reason


def test_the_prefetch_runs_the_uv_it_found_rather_than_the_bare_token(
    managed, monkeypatch, tmp_path
):
    home = tmp_path / "home"
    _without_uv_on_path(monkeypatch, home)
    uv = _install_uv_at(home / ".local" / "bin")
    recorder = _Recorder([(lambda cmd: "--dry-run" in cmd, _plan_response(""))])
    monkeypatch.setattr(_studio_prefetch, "_run", recorder)

    _studio_prefetch.run(
        studio_home = managed,
        env = {"HOME": str(home), "USERPROFILE": str(home)},
        echo = lambda line: None,
    )

    assert recorder.commands
    assert {command[0] for command in recorder.commands} == {str(uv)}


def test_the_environment_the_prefetch_runs_uv_in_is_the_callers(managed, monkeypatch):
    recorder = _Recorder([(lambda cmd: "--dry-run" in cmd, _plan_response(""))])
    monkeypatch.setattr(_studio_prefetch, "_run", recorder)

    _studio_prefetch.run(
        studio_home = managed,
        env = {"UV_CACHE_DIR": "/cache/uv", "UV_OVERRIDE": "/live/overrides-darwin-arm64.txt"},
        echo = lambda line: None,
    )

    assert recorder.environments[0]["UV_CACHE_DIR"] == "/cache/uv"
    # macOS arm64: install_python_stack.py sets UV_OVERRIDE at module load, and the
    # plan is only the plan the update will run if it resolves under the same one.
    assert recorder.environments[0]["UV_OVERRIDE"] == "/live/overrides-darwin-arm64.txt"


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(
            lambda monkeypatch, home: monkeypatch.setenv("UV_NO_CACHE", "1"),
            id = "uv-no-cache",
        ),
        pytest.param(
            _arrange_without_any_uv,
            id = "no-uv",
        ),
        pytest.param(
            lambda monkeypatch, home: monkeypatch.setattr(
                _studio_prefetch, "_is_editable_install", lambda name = "unsloth": True
            ),
            id = "editable",
        ),
        pytest.param(
            lambda monkeypatch, home: monkeypatch.setenv("STUDIO_LOCAL_INSTALL", "1"),
            id = "local",
        ),
        pytest.param(
            lambda monkeypatch, home: monkeypatch.setattr(sys, "prefix", str(home / "other")),
            id = "foreign-venv",
        ),
    ],
)
def test_an_install_with_nothing_to_prepare_is_skipped_without_a_marker(
    managed, monkeypatch, arrange
):
    arrange(monkeypatch, managed)
    monkeypatch.setattr(
        _studio_prefetch,
        "_run",
        lambda cmd, env: pytest.fail("a skipped prefetch must not run uv"),
    )

    with pytest.raises(_studio_prefetch.PrefetchSkipped):
        _studio_prefetch.run(studio_home = managed, echo = lambda line: None)

    assert not _studio_prefetch.marker_path(managed).is_file()


def test_a_full_disk_stops_the_prefetch_before_it_writes_anything(managed, monkeypatch):
    monkeypatch.setattr(_studio_prefetch, "_free_bytes", lambda path: 512 * 1024 * 1024)
    monkeypatch.setattr(
        _studio_prefetch,
        "_run",
        lambda cmd, env: pytest.fail("the disk floor must be checked before uv runs"),
    )

    with pytest.raises(_studio_prefetch.PrefetchError) as failure:
        _studio_prefetch.run(studio_home = managed, echo = lambda line: None)

    assert "free space" in str(failure.value)
    assert not _studio_prefetch.prefetch_root(managed).exists()


def test_a_directory_unsloth_did_not_create_is_refused_not_deleted(managed, monkeypatch):
    root = _studio_prefetch.prefetch_root(managed)
    root.mkdir(parents = True)
    keep = root / "someone-elses-file"
    keep.write_text("data", encoding = "utf-8")

    with pytest.raises(_studio_prefetch.PrefetchError):
        _studio_prefetch.run(studio_home = managed, echo = lambda line: None)

    assert keep.is_file()
    assert not _studio_prefetch.discard(managed)
    assert keep.is_file()


def test_discard_removes_only_an_owned_directory(tmp_path):
    home = tmp_path / "studio"
    root = _studio_prefetch.prefetch_root(home)
    root.mkdir(parents = True)
    (root / _studio_prefetch.OWNED_MARKER).write_text("", encoding = "utf-8")
    (root / "site").mkdir()

    assert _studio_prefetch.discard(home) is True
    assert not root.exists()
    # Idempotent: a second update after a first one must not fail on the absence.
    assert _studio_prefetch.discard(home) is False
    assert _studio_prefetch.discard_after_update(home) is False


# ── Locking ──


@pytest.mark.skipif(os.name == "nt", reason = "the POSIX flock branch is under test")
def test_a_second_prefetch_is_refused_rather_than_queued(tmp_path):
    home = tmp_path / "studio"
    holding = threading.Event()
    release = threading.Event()
    busy: list[bool] = []

    def hold():
        with _studio_prefetch.prefetch_lock(home):
            holding.set()
            release.wait(10)

    worker = threading.Thread(target = hold)
    worker.start()
    try:
        assert holding.wait(10)
        # A different process is what flock separates, so ask one.
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
                "from unsloth_cli import _studio_prefetch as p\n"
                "from pathlib import Path\n"
                "try:\n"
                f"    with p.prefetch_lock(Path({str(home)!r})):\n"
                "        sys.exit(0)\n"
                "except p.PrefetchBusy:\n"
                f"    sys.exit({_studio_prefetch.EXIT_BUSY})\n",
            ],
            capture_output = True,
            text = True,
        )
        busy.append(probe.returncode == _studio_prefetch.EXIT_BUSY)
    finally:
        release.set()
        worker.join(10)

    assert busy == [True]
    # Released: the next prefetch has to be able to take it.
    with _studio_prefetch.prefetch_lock(home):
        pass


# ── Markers ──


def test_a_marker_is_only_current_for_the_python_cache_and_floor_it_recorded():
    marker = {
        "schema": _studio_prefetch.MARKER_SCHEMA,
        "state": "ready",
        "backend_version": "2026.9.2",
        "python": "/studio/unsloth_studio/bin/python",
        "cache_dir": "/cache/uv",
    }

    assert _studio_prefetch.marker_is_current(
        marker,
        floor = "2026.9.2",
        python = "/studio/unsloth_studio/bin/python",
        cache_dir = "/cache/uv",
    )
    assert not _studio_prefetch.marker_is_current(marker, floor = "2026.9.3")
    assert not _studio_prefetch.marker_is_current(marker, python = "/other/bin/python")
    assert not _studio_prefetch.marker_is_current(marker, cache_dir = "/other/uv")
    assert not _studio_prefetch.marker_is_current({**marker, "schema": 2})
    assert not _studio_prefetch.marker_is_current(None)


def test_a_noop_marker_is_current_when_the_installed_version_meets_the_floor():
    marker = {
        "schema": _studio_prefetch.MARKER_SCHEMA,
        "state": "noop",
        "backend_version": None,
        "installed_backend_version": "2026.9.2",
    }

    assert _studio_prefetch.marker_is_current(marker, floor = "2026.9.2")
    assert not _studio_prefetch.marker_is_current(marker, floor = "2026.9.3")


def test_the_marker_is_replaced_atomically(tmp_path, monkeypatch):
    home = tmp_path / "studio"
    _studio_prefetch.prefetch_root(home).mkdir(parents = True)
    _studio_prefetch.write_marker(home, {"schema": 1, "state": "ready"})

    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def record(source, destination):
        replaced.append((str(source), str(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(_studio_prefetch.os, "replace", record)
    _studio_prefetch.write_marker(home, {"schema": 1, "state": "noop"})

    assert len(replaced) == 1
    assert replaced[0][0].endswith(".tmp")
    assert replaced[0][1] == str(_studio_prefetch.marker_path(home))
    assert _studio_prefetch.read_marker(home)["state"] == "noop"
    assert not list(_studio_prefetch.prefetch_root(home).glob("*.tmp"))


# ── The CLI wiring ──


def _command_body(source: str, name: str) -> str:
    """The function's own text, to the next top-level definition of any kind."""
    start = source.index(f"def {name}(")
    ends = [
        end
        for end in (source.find("\ndef ", start + 1), source.find("\nclass ", start + 1))
        if end != -1
    ]
    return source[start : min(ends)] if ends else source[start:]


def test_the_prefetch_command_takes_none_of_the_update_wrappers():
    source = STUDIO_COMMAND.read_text(encoding = "utf-8")
    body = _command_body(source, "prefetch_update")

    assert '@studio_app.command("prefetch-update", hidden = True)' in source
    for forbidden in (
        "_studio_runtime_launch_guard",
        "consume_runtime_gate_handoff",
        "ensure_managed_environment_is_idle",
        "_WindowsLauncherUpdateTransaction",
    ):
        assert forbidden not in body, f"{forbidden} must not wrap a prefetch"
    assert "_studio_prefetch.prefetch_lock(STUDIO_HOME)" in body
    assert "_with_studio_uv_cache(None, cwd = setup_cwd)" in body
    assert "raise typer.Exit(_studio_prefetch.EXIT_BUSY)" in body


def test_the_update_discards_a_prefetch_only_after_it_has_succeeded():
    source = STUDIO_COMMAND.read_text(encoding = "utf-8")
    body = _command_body(source, "update")

    setup = body.index("_run_setup_script(")
    verify = body.index("_fail_if_install_damaged(", setup)
    discard = body.index("_studio_prefetch.discard_after_update(STUDIO_HOME)", verify)
    # Outside the `with`, so an update that raised keeps the prefetch for the retry.
    assert body[:discard].count("    with launcher_transaction as launcher_update:") == 1
    assert body[discard - 200 : discard].count("except") == 0
    assert setup < verify < discard


def test_the_prefetch_module_stays_importable_under_isolated_python():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys\n"
            f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
            "from unsloth_cli import _studio_prefetch\n"
            "print(_studio_prefetch.EXIT_BUSY)\n",
        ],
        capture_output = True,
        text = True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3"


# ── the installer's override, the budget inside each call, and redaction ──


def test_the_dry_run_carries_the_installer_override_on_apple_silicon(managed, monkeypatch):
    """Without the override uv answers for a different resolver than the core step's:
    on the staging matrix it planned mlx-vlm and mlx-audio downgrades the update never
    makes, and the offline swap installed them."""
    site = managed / _studio_prefetch.VENV_NAME / "lib" / "python3.12" / "site-packages"
    overrides = site / "studio" / "backend" / "requirements" / "single-env" / "overrides-darwin-arm64.txt"
    overrides.write_text("transformers>=5.5.0,<=5.5.0\n", encoding = "utf-8")
    monkeypatch.setattr(_studio_prefetch.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_studio_prefetch.platform, "machine", lambda: "arm64")
    seen = []

    def respond(cmd, env):
        seen.append(dict(env or {}))
        return _plan_response("")

    monkeypatch.setattr(_studio_prefetch, "_run", respond)
    monkeypatch.delenv("UV_OVERRIDE", raising = False)
    _studio_prefetch.run(studio_home = managed, echo = lambda line: None)
    assert seen and all(e.get("UV_OVERRIDE") == str(overrides) for e in seen)

    # A caller's own override is kept, as the installer keeps it.
    seen.clear()
    _studio_prefetch.run(studio_home = managed, env = {**os.environ, "UV_OVERRIDE": "/my/overrides.txt"}, echo = lambda line: None)
    assert seen and all(e.get("UV_OVERRIDE") == "/my/overrides.txt" for e in seen)

    # Off Apple silicon nothing is set.
    monkeypatch.setattr(_studio_prefetch.platform, "machine", lambda: "x86_64")
    seen.clear()
    _studio_prefetch.run(studio_home = managed, echo = lambda line: None)
    assert seen and all("UV_OVERRIDE" not in e for e in seen)


def test_every_uv_call_is_bounded_by_what_is_left_of_the_budget(monkeypatch):
    """A stalled index used to hold the UI at Preparing for the full 30 minutes of each
    of the two core calls before the deadline was looked at."""
    timeouts = []

    def fake_run(cmd, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        return _completed(0)

    monkeypatch.setattr(_studio_prefetch.subprocess, "run", fake_run)
    _studio_prefetch._run(["uv", "--version"], None)
    assert timeouts[-1] == _studio_prefetch.SUBPROCESS_TIMEOUT_SECONDS
    with _studio_prefetch._within_budget(time.monotonic() + 5):
        _studio_prefetch._run(["uv", "--version"], None)
        assert 0 < timeouts[-1] <= 5
        with pytest.raises(subprocess.TimeoutExpired):
            with _studio_prefetch._within_budget(time.monotonic() - 1):
                _studio_prefetch._run(["uv", "--version"], None)
    # Restored: the next call outside the budget is unbounded again.
    _studio_prefetch._run(["uv", "--version"], None)
    assert timeouts[-1] == _studio_prefetch.SUBPROCESS_TIMEOUT_SECONDS


def test_a_core_call_that_runs_out_of_time_is_a_bounded_error(managed, monkeypatch):
    def stall(cmd, env):
        raise subprocess.TimeoutExpired(list(cmd), 1)

    monkeypatch.setattr(_studio_prefetch, "_run", stall)
    with pytest.raises(_studio_prefetch.PrefetchError) as failure:
        _studio_prefetch.run(studio_home = managed, echo = lambda line: None)
    assert "out of time" in str(failure.value)


SECRET_INDEX = "https://user:s3cret@index.example/simple?token=t0k3n#frag=f"


def test_index_credentials_never_reach_the_failure_text(managed, monkeypatch):
    """uv echoes the failing index URL, and what is raised here goes to the desktop log
    and the renderer; the installer already redacts the same shape."""
    monkeypatch.setattr(
        _studio_prefetch,
        "_run",
        lambda cmd, env: _completed(1, stderr = f"error: Failed to fetch: `{SECRET_INDEX}`"),
    )
    with pytest.raises(_studio_prefetch.PrefetchError) as failure:
        _studio_prefetch.run(studio_home = managed, echo = lambda line: None)
    text = str(failure.value)
    assert "s3cret" not in text and "t0k3n" not in text and "frag=f" not in text
    assert "<redacted>" in text

    # ...and from a requirement file's recorded reason.
    target = _studio_prefetch.site_dir(managed)

    def respond(cmd, env):
        cmd = list(cmd)
        if "--dry-run" in cmd and any(a.startswith("unsloth>=") for a in cmd):
            return _plan_response(" + unsloth==2026.9.2\n")
        if "--dry-run" in cmd:
            return _completed(1, stderr = f"no solution: {SECRET_INDEX}")
        if "--target" in cmd:
            _install_new_wheel_tree(target)
            return _completed(0)
        return _completed(0)

    monkeypatch.setattr(_studio_prefetch, "_run", respond)
    payload = _studio_prefetch.run(studio_home = managed, floor = "2026.9.2", echo = lambda line: None)
    reason = payload["requirements"]["studio.txt"]["skipped_reason"]
    assert "s3cret" not in reason and "t0k3n" not in reason and "<redacted>" in reason
