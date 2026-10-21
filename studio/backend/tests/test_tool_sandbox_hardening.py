# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Nothing the sandboxed tool code can write decides what runs, or where, outside the sandbox."""

from __future__ import annotations

import errno
import os
import socket
import subprocess
import sys

import pytest

from core.inference import os_sandbox, sandbox_probe, tools
from core.inference.os_sandbox import ToolLaunchPlan

_POSIX = pytest.mark.skipif(sys.platform == "win32", reason = "POSIX launch layout")


def _planted_workdir(tmp_path):
    workdir = tmp_path / "chat"
    packages = workdir / os_sandbox.SESSION_PACKAGES_RELPATH
    packages.mkdir(parents = True)
    for folder in (workdir, packages):
        (folder / "csv.py").write_text("print('PLANTED', __file__)\n", encoding = "utf-8")
    (packages / "chatpkg.py").write_text("VALUE = 'from-session-packages'\n", encoding = "utf-8")
    (workdir / ".gitconfig").write_text("[core]\n\tfsmonitor = planted\n", encoding = "utf-8")
    return workdir


@_POSIX
def test_an_approved_host_call_loads_nothing_from_the_workdir(tmp_path, monkeypatch):
    workdir = _planted_workdir(tmp_path)
    host_file = tmp_path / "host" / "notes.csv"
    host_file.parent.mkdir()
    host_file.write_text("a,b\n", encoding = "utf-8")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    code = (
        "import csv, os, sys\n"
        f"rows = list(csv.reader(open({str(host_file)!r})))\n"
        "print('ROWS', rows)\n"
        "print('HOME', os.environ['HOME'])\n"
        "print('FILE', __file__)\n"
        "print('CWD', os.getcwd())\n"
        "import chatpkg\n"
        "print('PKG', chatpkg.VALUE)\n"
    )
    out = tools._python_exec(code, None, 60, "hardening-chat", host_access_approved = True)
    assert "ROWS [['a', 'b']]" in out, out
    assert "PLANTED" not in out, out
    # A package pip-installed in the chat is still importable, behind the stdlib.
    assert "PKG from-session-packages" in out, out
    lines = dict(line.split(" ", 1) for line in out.splitlines() if " " in line)
    assert os.path.realpath(lines["HOME"]) != os.path.realpath(workdir)
    assert not lines["FILE"].startswith(str(workdir)), "the approved script ran from the workdir"
    # Relative paths in the approved code still resolve in the chat folder.
    assert os.path.realpath(lines["CWD"]) == os.path.realpath(workdir)
    # The private per-call directory is gone afterwards.
    assert not os.path.exists(os.path.dirname(lines["FILE"]))


@_POSIX
def test_an_approved_host_launch_env_drops_every_workdir_input(tmp_path):
    workdir = _planted_workdir(tmp_path)
    script = workdir / "studio_exec_x.py"
    script.write_text("print('swapped')\n", encoding = "utf-8")
    plan = ToolLaunchPlan(
        argv = (sys.executable, "-u", str(script)),
        workdir = str(workdir),
        env = tools._build_safe_env(str(workdir)),
        requested_mode = "auto",
        execution_kind = "python",
    )
    prepared = tools._prepare_tool_launch(
        plan, host_access_approved = True, host_script = "print('approved')\n"
    )
    try:
        assert prepared.execution_record.os_isolation is False
        assert prepared.env["PYTHONSAFEPATH"] == "1"
        assert prepared.env["HOME"] != str(workdir)
        packages = str(workdir / os_sandbox.SESSION_PACKAGES_RELPATH)
        assert packages not in prepared.env.get("PYTHONPATH", "").split(os.pathsep)
        assert not any(part.startswith(packages) for part in prepared.env["PATH"].split(os.pathsep))
        assert not prepared.argv[-1].startswith(str(workdir))
        # The approved code, not whatever the workdir copy was replaced with.
        ran = subprocess.run(
            prepared.argv,
            cwd = prepared.workdir,
            env = prepared.env,
            capture_output = True,
            text = True,
            timeout = 60,
        )
        assert ran.stdout.strip() == "approved", ran
    finally:
        prepared.cleanup()


@_POSIX
def test_launches_that_are_not_approved_host_access_keep_the_workdir_env(tmp_path):
    workdir = _planted_workdir(tmp_path)
    script = str(workdir / "studio_exec_x.py")
    plan = ToolLaunchPlan(
        argv = (sys.executable, "-u", script),
        workdir = str(workdir),
        env = tools._build_safe_env(str(workdir)),
        requested_mode = "auto",
        execution_kind = "python",
    )
    packages = str(workdir / os_sandbox.SESSION_PACKAGES_RELPATH)
    for prepared in (
        tools._software_safeguards_launch(plan, "sandbox_became_unavailable"),
        # A chosen project folder is never a sandbox workdir in auto: its calls keep importing from it.
        tools._prepare_tool_launch(plan, host_access_approved = True, user_folder = True),
    ):
        assert prepared.argv[-1] == script
        assert prepared.env["HOME"] == str(workdir)
        assert "PYTHONSAFEPATH" not in prepared.env
        assert packages in prepared.env["PYTHONPATH"].split(os.pathsep)
        assert prepared.cleanup_paths == []


@pytest.mark.skipif(sys.platform != "linux", reason = "the bubblewrap backend is Linux only")
def test_the_jail_sees_host_sysctls_read_only(tmp_path, monkeypatch):
    from core.inference import sandbox_linux

    if sandbox_linux.shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this host")
    monkeypatch.setattr(sandbox_linux, "empty_proc_layout", lambda bwrap = None: False)
    monkeypatch.setattr(sandbox_linux, "_model_cache_binds", lambda workdir: {})
    launch = sandbox_linux.prepare(
        ToolLaunchPlan(argv = ("/bin/true",), workdir = str(tmp_path), env = {"PATH": "/usr/bin"})
    )
    try:
        argv = list(launch.argv)
        at = next(i for i in range(len(argv) - 1) if argv[i : i + 2] == ["--proc", "/proc"])
        assert argv[at + 2 : at + 5] == ["--ro-bind-try", "/proc/sys", "/proc/sys"]
    finally:
        launch.cleanup()


@pytest.mark.skipif(sys.platform != "linux", reason = "the bubblewrap backend is Linux only")
def test_datasets_and_assets_are_bound_from_the_real_hf_home(tmp_path, monkeypatch):
    """HF_HUB_CACHE=/mnt/models makes the display home the hub itself, which the jail can write."""
    from core.inference import sandbox_linux
    import utils.hf_cache_settings as cache_settings

    hub = tmp_path / "mnt" / "models"
    real_home = tmp_path / "hf_home"
    for path in (hub / "datasets", hub / "assets", real_home / "datasets", real_home / "assets"):
        path.mkdir(parents = True)
    monkeypatch.setenv("HF_HOME", str(real_home))
    monkeypatch.setattr(
        cache_settings,
        "get_hf_cache_paths",
        lambda: cache_settings.HuggingFaceCachePaths(
            cache_home = hub, hub_cache = hub, xet_cache = hub / "xet", source = "environment"
        ),
    )
    monkeypatch.setattr(
        sandbox_linux,
        "_cache_hazard_within_deadline",
        lambda name, path: None if os.path.isdir(path) else sandbox_linux.CACHE_MISSING,
    )
    binds = sandbox_linux._model_cache_binds(str(tmp_path / "session"))
    assert binds["hub"] == str(hub)
    assert binds["datasets"] == str(real_home / "datasets")
    assert binds["assets"] == str(real_home / "assets")


def test_portable_datasets_and_assets_caches_are_bound_where_they_point(tmp_path, monkeypatch):
    """Portable installs set HF_DATASETS_CACHE / HF_ASSETS_CACHE under their root and leave HF_HOME alone."""
    from core.inference import sandbox_linux
    import utils.hf_cache_settings as cache_settings

    default_home = tmp_path / "default_hf"
    portable = tmp_path / "portable" / "huggingface"
    for path in (
        default_home / "hub",
        default_home / "datasets",
        default_home / "assets",
        portable / "datasets",
        portable / "assets",
    ):
        path.mkdir(parents = True)
    monkeypatch.setenv("HF_HOME", str(default_home))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(portable / "datasets"))
    monkeypatch.setenv("HF_ASSETS_CACHE", str(portable / "assets"))
    monkeypatch.setattr(
        cache_settings,
        "get_hf_cache_paths",
        lambda: cache_settings.HuggingFaceCachePaths(
            cache_home = default_home,
            hub_cache = default_home / "hub",
            xet_cache = default_home / "xet",
            source = "environment",
        ),
    )
    monkeypatch.setattr(
        sandbox_linux,
        "_cache_hazard_within_deadline",
        lambda name, path: None if os.path.isdir(path) else sandbox_linux.CACHE_MISSING,
    )
    binds = sandbox_linux._model_cache_binds(str(tmp_path / "session"))
    assert binds["datasets"] == str(portable / "datasets")
    assert binds["assets"] == str(portable / "assets")


def _auto_plan(tmp_path):
    return ToolLaunchPlan(
        argv = (sys.executable, "-c", "print(1)"),
        workdir = str(tmp_path),
        env = {"PATH": "/usr/bin"},
        requested_mode = "auto",
        execution_kind = "python",
    )


@pytest.fixture
def fresh_probe_cache():
    sandbox_probe.reset_probe_cache()
    yield
    sandbox_probe.reset_probe_cache()


@pytest.mark.skipif(sys.platform not in ("linux", "darwin"), reason = "the probe backends")
@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EMFILE, errno.EAGAIN])
def test_a_probe_out_of_resources_refuses_rather_than_running_on_the_host(
    tmp_path, monkeypatch, fresh_probe_cache, code
):
    calls = []

    def exhausted():
        calls.append(True)
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(sandbox_probe, "_probe_base", exhausted)
    for _ in range(2):
        with pytest.raises(os_sandbox.SandboxBuildError, match = "OS_ISOLATION_UNAVAILABLE"):
            os_sandbox.prepare_tool_launch(_auto_plan(tmp_path))
    # Undecided verdicts are not cached: the next call checks again.
    assert len(calls) == 2
    # tools.py does not turn it into a host launch either.
    with pytest.raises(os_sandbox.SandboxBuildError):
        tools._prepare_tool_launch(_auto_plan(tmp_path))


@pytest.mark.skipif(sys.platform not in ("linux", "darwin"), reason = "the probe backends")
def test_a_host_control_that_runs_out_of_disk_is_undecided_too(
    tmp_path, monkeypatch, fresh_probe_cache
):
    monkeypatch.setattr(
        sandbox_probe,
        "_host_payload",
        lambda workdir: "import errno, os\nraise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))\n",
    )
    with pytest.raises(os_sandbox.SandboxBuildError):
        os_sandbox.prepare_tool_launch(_auto_plan(tmp_path))


@pytest.mark.skipif(sys.platform not in ("linux", "darwin"), reason = "the probe backends")
def test_a_host_without_a_sandbox_still_falls_back_in_auto(
    tmp_path, monkeypatch, fresh_probe_cache
):
    monkeypatch.setattr(
        sandbox_probe,
        "_run_probe",
        lambda backend, name, plan_cls: (False, "bubblewrap is not installed"),
    )
    prepared = os_sandbox.prepare_tool_launch(_auto_plan(tmp_path))
    assert prepared.backend == "software-safeguards"
    assert prepared.execution_record.os_isolation is False


@_POSIX
def test_the_stale_ipc_sweep_cannot_be_redirected_onto_the_host(tmp_path, monkeypatch):
    workdir = tmp_path / "w"
    scratch = workdir / os_sandbox.TOOL_TEMP_DIRNAME
    inner = scratch / "d"
    inner.mkdir(parents = True)
    os.chmod(scratch, os_sandbox.TOOL_TEMP_MODE)
    host = tmp_path / "host"
    host.mkdir()
    (host / "sock").write_text("host data", encoding = "utf-8")
    monkeypatch.chdir(inner)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind("sock")
    listener.close()
    monkeypatch.chdir(tmp_path)

    def swap_then_report_dead(path):
        os.rename(inner, scratch / "moved")
        os.symlink(host, inner)
        return True

    monkeypatch.setattr(os_sandbox, "_ipc_endpoint_is_dead", swap_then_report_dead)
    os_sandbox.clear_stale_tool_ipc(str(workdir))
    assert (host / "sock").read_text(encoding = "utf-8") == "host data"
    # The socket the walk actually saw is the one removed.
    assert not os.path.lexists(scratch / "moved" / "sock")


@pytest.fixture
def project(tmp_path, monkeypatch):
    from storage import studio_db

    root = tmp_path / "Projects" / "p1"
    (root / "sandbox").mkdir(parents = True)
    record = {"id": "p1", "rootPath": str(root), "sandboxPath": str(root / "sandbox")}
    monkeypatch.setattr(tools, "is_owner_context", lambda: True)
    monkeypatch.setattr(tools, "_thread_exists", lambda *a, **k: False)
    monkeypatch.setattr(studio_db, "ensure_chat_project_workspace", lambda project_id: record)
    monkeypatch.setattr(studio_db, "available_project_working_folder", lambda project: None)
    return root


def test_a_project_sandbox_folder_is_used_as_it_is(project):
    assert tools._get_project_workdir("project-p1") == os.path.realpath(project / "sandbox")


@_POSIX
@pytest.mark.parametrize("target", [".", "elsewhere"])
def test_a_project_sandbox_swapped_for_a_link_is_not_followed(project, target):
    (project / "elsewhere").mkdir()
    os.rename(project / "sandbox", project / "old")
    os.symlink(target, project / "sandbox")
    assert tools._get_project_workdir("project-p1") is None
