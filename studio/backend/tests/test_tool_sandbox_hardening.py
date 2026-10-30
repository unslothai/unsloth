# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Nothing the sandboxed tool code can write decides what runs, or where, outside the sandbox."""

from __future__ import annotations

import errno
import os
import shlex
import socket
import subprocess
import sys

import pytest

from core.inference import os_sandbox, sandbox_probe, tools
from core.inference.os_sandbox import ToolLaunchPlan

_POSIX = pytest.mark.skipif(sys.platform == "win32", reason = "POSIX launch layout")

# Old importlib.metadata.entry_points() returns a dict of groups; 3.10+ selects.
_PYTEST11_SCAN = (
    "_eps = m.entry_points()\n"
    "_eps = _eps.select(group = 'pytest11') if hasattr(_eps, 'select') else _eps.get('pytest11', ())\n"
    "print('SCAN', sorted(e.name for e in _eps if e.name.startswith('chat')))\n"
)


@pytest.mark.parametrize("shape", ["dict", "selectable"])
def test_the_pytest11_scan_reads_both_entry_point_api_shapes(shape, capsys):
    """The scan the approved-call checks run works on the 3.9 dict API and the 3.10+ select API."""
    import types

    hit = types.SimpleNamespace(name = "chatplugin")
    other = types.SimpleNamespace(name = "hostplugin")

    class Selectable(list):
        def select(self, group):
            return [hit, other] if group == "pytest11" else []

    def entry_points(**kwargs):
        if shape == "dict":
            if kwargs:
                raise TypeError("entry_points() got an unexpected keyword argument 'group'")
            return {"pytest11": (hit, other), "console_scripts": ()}
        return Selectable()

    exec(_PYTEST11_SCAN, {"m": types.SimpleNamespace(entry_points = entry_points)})
    assert capsys.readouterr().out == "SCAN ['chatplugin']\n"


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
def test_an_approved_terminal_call_reuses_chat_packages_without_shadowing(tmp_path, monkeypatch):
    workdir = _planted_workdir(tmp_path)
    # Terminal deliberately retains cwd imports; exercise only the session-package precedence here.
    (workdir / "csv.py").unlink()
    packages = workdir / os_sandbox.SESSION_PACKAGES_RELPATH
    (packages / "usercustomize.py").write_text("print('PLANTED usercustomize')\n", encoding = "utf-8")
    host_file = tmp_path / "notes.csv"
    host_file.write_text("a,b\n", encoding = "utf-8")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    code = (
        "import chatpkg, csv, sys; "
        "print('PKG', chatpkg.VALUE); "
        "print('ROWS', list(csv.reader(open(sys.argv[1]))))"
    )
    command = "python -c " + shlex.quote(code) + " " + shlex.quote(str(host_file))
    assert tools._reaches_host_paths("terminal", command)
    for _ in range(2):
        out = tools._bash_exec(command, None, 60, "hardening-chat", host_access_approved = True)
        assert "PKG from-session-packages" in out, out
        assert "ROWS [['a', 'b']]" in out, out
        assert "PLANTED" not in out, out


def _plant_chat_plugin(packages, marker):
    """A pytest11 entry point a chat pip-installed into its package folder, which copies a host file when loaded."""
    secret = marker.parent / "host-secret.txt"
    secret.write_text("outside-chat-secret", encoding = "utf-8")
    (packages / "chat_autoload.py").write_text(
        f"import pathlib\npathlib.Path({str(marker)!r}).write_text(pathlib.Path({str(secret)!r}).read_text())\n",
        encoding = "utf-8",
    )
    info = packages / "chat_autoload-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: chat-autoload\nVersion: 1.0\n", encoding = "utf-8"
    )
    (info / "entry_points.txt").write_text(
        "[pytest11]\nchat_autoload = chat_autoload\n", encoding = "utf-8"
    )


@_POSIX
def test_an_approved_pytest_call_does_not_autoload_chat_installed_plugins(tmp_path, monkeypatch):
    """Session packages must not be visible to entry-point discovery: pytest would import a planted plugin."""
    workdir = _planted_workdir(tmp_path)
    (workdir / "csv.py").unlink()
    packages = workdir / os_sandbox.SESSION_PACKAGES_RELPATH
    host = tmp_path / "host"
    host.mkdir()
    marker = host / "plugin-ran.txt"
    _plant_chat_plugin(packages, marker)
    (workdir / "test_ok.py").write_text(
        "import chatpkg\n\ndef test_ok():\n    assert chatpkg.VALUE == 'from-session-packages'\n",
        encoding = "utf-8",
    )
    host_file = host / "notes.csv"
    host_file.write_text("a,b\n", encoding = "utf-8")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    command = (
        shlex.quote(sys.executable)
        + " -m pytest -q -o addopts= -p no:cacheprovider test_ok.py && cat "
        + shlex.quote(str(host_file))
    )
    assert tools._reaches_host_paths("terminal", command)
    out = tools._bash_exec(command, None, 60, "hardening-chat", host_access_approved = True)
    assert "1 passed" in out and "a,b" in out, out
    assert not marker.exists(), "a chat-installed pytest plugin ran on the host"


@_POSIX
def test_an_approved_script_still_imports_chat_installed_packages(tmp_path, monkeypatch):
    """Explicit imports (packages, submodules, their own metadata) keep working; name-less scans see nothing."""
    workdir = _planted_workdir(tmp_path)
    packages = workdir / os_sandbox.SESSION_PACKAGES_RELPATH
    (packages / "chatlib").mkdir()
    (packages / "chatlib" / "__init__.py").write_text("from .sub import X\n", encoding = "utf-8")
    (packages / "chatlib" / "sub.py").write_text("X = 'chatlib-sub'\n", encoding = "utf-8")
    info = packages / "chatlib-2.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: chatlib\nVersion: 2.0\n", encoding = "utf-8"
    )
    marker = tmp_path / "plugin-ran.txt"
    _plant_chat_plugin(packages, marker)
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    code = (
        "import importlib.metadata as m, chatlib, chatlib.sub, chatpkg\n"
        "print('LIB', chatlib.X, chatlib.sub.X, chatpkg.VALUE, m.version('chatlib'))\n"
        + _PYTEST11_SCAN
        + f"open({str(tmp_path / 'host.txt')!r}, 'w').close()\n"
    )
    out = tools._python_exec(code, None, 60, "hardening-chat", host_access_approved = True)
    assert "LIB chatlib-sub chatlib-sub from-session-packages 2.0" in out, out
    assert "SCAN []" in out, out
    assert not marker.exists(), out


@_POSIX
def test_an_approved_script_imports_chat_portions_of_host_namespace_packages(tmp_path, monkeypatch):
    """A PEP 420 namespace the host already provides still reaches the portion the chat installed."""
    workdir = _planted_workdir(tmp_path)
    packages = workdir / os_sandbox.SESSION_PACKAGES_RELPATH
    host = tmp_path / "host"
    for root, owner in ((host, "host"), (packages, "chat")):
        (root / "chatns" / "inner").mkdir(parents = True)
        (root / "chatns" / f"{owner}_mod.py").write_text(f"V = '{owner}'\n", encoding = "utf-8")
        (root / "chatns" / "inner" / f"{owner}_leaf.py").write_text(
            f"V = '{owner}-leaf'\n", encoding = "utf-8"
        )
        (root / "chatns" / "shared.py").write_text(f"V = '{owner}-shared'\n", encoding = "utf-8")
    (packages / "chatns" / "chatreg").mkdir()
    (packages / "chatns" / "chatreg" / "__init__.py").write_text("V = 'reg'\n", encoding = "utf-8")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    code = (
        f"import sys; sys.path.insert(0, {str(host)!r})\n"
        "import chatns.host_mod, chatns.chat_mod, chatns.shared, chatns.chatreg\n"
        "import chatns.inner.host_leaf, chatns.inner.chat_leaf\n"
        "print('NS', chatns.host_mod.V, chatns.chat_mod.V, chatns.shared.V, chatns.chatreg.V,"
        " chatns.inner.host_leaf.V, chatns.inner.chat_leaf.V)\n"
        f"open({str(tmp_path / 'host.txt')!r}, 'w').close()\n"
    )
    out = tools._python_exec(code, None, 60, "hardening-chat", host_access_approved = True)
    # The host's own portion still wins for a name both provide.
    assert "NS host chat host-shared reg host-leaf chat-leaf" in out, out


@_POSIX
def test_an_approved_script_imports_chat_portions_of_host_pkgutil_namespaces(tmp_path, monkeypatch):
    """A pkgutil.extend_path namespace (backports, ...) the host provides reaches the chat's portion; a regular one does not."""
    workdir = _planted_workdir(tmp_path)
    packages = workdir / os_sandbox.SESSION_PACKAGES_RELPATH
    host = tmp_path / "host"
    extend = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)\n"
    # Only mentions the namespace helpers (a comment, an unused import, a string, a dead branch):
    # still a regular package.
    mentions = (
        "# not a namespace: no extend_path or declare_namespace here\n"
        "from pkgutil import extend_path\n"
        "NOTE = 'declare_namespace(__name__)'\n"
        "if False:\n    __path__ = extend_path(__path__, __name__)\n"
    )
    for root, owner in ((host, "host"), (packages, "chat")):
        for pkg, init in (("chatpkgutil", extend), ("chatregular", ""), ("chatmention", mentions)):
            (root / pkg).mkdir(parents = True)
            (root / pkg / "__init__.py").write_text(init, encoding = "utf-8")
            (root / pkg / f"{owner}_mod.py").write_text(f"V = '{owner}'\n", encoding = "utf-8")
            (root / pkg / "shared.py").write_text(f"V = '{owner}-shared'\n", encoding = "utf-8")
    marker = tmp_path / "plugin-ran.txt"
    _plant_chat_plugin(packages, marker)
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    code = (
        f"import sys; sys.path.insert(0, {str(host)!r})\n"
        "import importlib.metadata as m, pkgutil\n"
        "import chatpkgutil.host_mod, chatpkgutil.chat_mod, chatpkgutil.shared, chatregular.host_mod\n"
        "print('PKG', chatpkgutil.host_mod.V, chatpkgutil.chat_mod.V, chatpkgutil.shared.V,"
        " sorted(i.name for i in pkgutil.iter_modules(chatpkgutil.__path__)))\n"
        "for name in ('chatregular', 'chatmention'):\n"
        "    try:\n        __import__(name + '.chat_mod')\n        print('REGULAR leaked', name)\n"
        "    except ModuleNotFoundError:\n        print('REGULAR closed', name)\n"
        + _PYTEST11_SCAN
        + f"open({str(tmp_path / 'host.txt')!r}, 'w').close()\n"
    )
    out = tools._python_exec(code, None, 60, "hardening-chat", host_access_approved = True)
    # The host's own portion still wins for a name both provide.
    assert "PKG host chat host-shared ['chat_mod', 'host_mod', 'shared']" in out, out
    assert "REGULAR closed chatregular" in out and "REGULAR closed chatmention" in out, out
    assert "REGULAR leaked" not in out, out
    assert "SCAN []" in out, out
    assert not marker.exists(), out


@_POSIX
def test_approved_calls_import_chat_helpers_without_shadowing_the_stdlib(tmp_path, monkeypatch):
    workdir = _planted_workdir(tmp_path)
    (workdir / "helper.py").write_text("VALUE = 'from-chat-folder'\n", encoding = "utf-8")
    for name in ("json.py", "os.py"):
        (workdir / name).write_text("print('PLANTED', __file__)\n", encoding = "utf-8")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    code = "import helper, json, os; print('HELPER', helper.VALUE); print('JSON', json.dumps([1]), json.__file__)\n"
    for _ in range(2):
        out = tools._python_exec(
            code + f"open({str(tmp_path / 'host.txt')!r}, 'w').close()\n",
            None,
            60,
            "hardening-chat",
            host_access_approved = True,
        )
        assert "HELPER from-chat-folder" in out, out
        assert "JSON [1]" in out and str(workdir) not in out.split("JSON", 1)[1], out
        assert "PLANTED" not in out, out
    # Terminal-started Python keeps importing helpers from the chat folder too.
    (workdir / "json.py").unlink()
    command = (
        "python -c "
        + shlex.quote("import helper; print('HELPER', helper.VALUE)")
        + " "
        + str(tmp_path)
    )
    assert tools._reaches_host_paths("terminal", command)
    out = tools._bash_exec(command, None, 60, "hardening-chat", host_access_approved = True)
    assert "HELPER from-chat-folder" in out, out
    assert "PLANTED" not in out, out


def test_chat_cli_names_match_case_insensitively_on_windows(tmp_path, monkeypatch):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    (scripts / "ruff.exe").write_bytes(b"MZ")
    (scripts / "other.exe").write_bytes(b"MZ")
    monkeypatch.setattr(tools.os, "name", "nt")
    dest = tools._requested_chat_clis(str(scripts), "RUFF check C:\\data", str(tmp_path / "bin"))
    assert dest is not None and sorted(os.listdir(dest)) == ["ruff.exe"]


def test_chat_folder_only_supplies_the_helpers_the_script_imports(tmp_path):
    """Startup hooks and optional imports a library probes never come from the chat folder."""
    assert tools._script_top_level_imports(
        "import helper, os.path\nfrom pkg.sub import x\nfrom . import y\n"
    ) == [
        "helper",
        "os",
        "pkg",
    ]
    assert tools._script_top_level_imports("def (") == []
    workdir = tmp_path / "chat"
    workdir.mkdir()
    (workdir / "helper.py").write_text("import config\nVALUE = config.X\n", encoding = "utf-8")
    (workdir / "config.py").write_text("X = 1\n", encoding = "utf-8")
    for name in ("usercustomize.py", "optional_dep.py"):
        (workdir / name).write_text("print('PLANTED')\n", encoding = "utf-8")
    site_dir = os.path.join(os.path.dirname(tools.__file__), "sandbox_site")
    env = dict(
        os.environ,
        PYTHONPATH = site_dir,
        _UNSLOTH_HOST_WORKDIR = str(workdir),
        _UNSLOTH_HOST_WORKDIR_MODULES = "helper,usercustomize",
    )
    code = (
        "import importlib.util as u, sys, helper; "
        "print('HOOK', u.find_spec('usercustomize'), u.find_spec('optional_dep'), 'HELPER', helper.VALUE, "
        f"{str(workdir)!r} in sys.path)"
    )
    for _ in range(2):
        out = subprocess.run(
            [sys.executable, "-c", code],
            env = env,
            cwd = str(tmp_path),
            capture_output = True,
            text = True,
            timeout = 60,
        ).stdout
        assert "HOOK None None HELPER 1 False" in out, out
        assert "PLANTED" not in out, out


@_POSIX
def test_an_approved_terminal_call_runs_chat_clis_after_system_ones(tmp_path, monkeypatch):
    workdir = _planted_workdir(tmp_path)
    bin_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("chatcli", "echo CHATCLI ok"),
        ("cat", "echo PLANTED cat"),
        ("unsloth-missing-tool", "echo PLANTED probe"),
    ):
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n", encoding = "utf-8")
        (bin_dir / name).chmod(0o755)
    host_file = tmp_path / "notes.csv"
    host_file.write_text("a,b\n", encoding = "utf-8")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    # A probe for a command the call never names (`command -v`, as tools probe for ffmpeg) misses the chat folder.
    # The name is built at run time, so it never appears in the command text.
    command = (
        "chatcli && cat "
        + shlex.quote(str(host_file))
        + " && { command -v $(printf %s unsloth-missing; printf %s -tool) || echo PROBE-MISSED; }"
    )
    assert tools._reaches_host_paths("terminal", command)
    for _ in range(2):
        out = tools._bash_exec(command, None, 60, "hardening-chat", host_access_approved = True)
        assert "CHATCLI ok" in out, out
        assert "a,b" in out, out
        assert "PROBE-MISSED" in out, out
        assert "PLANTED" not in out, out


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
        # The chat bin folder itself is never on PATH; only CLIs the call names are, last.
        path = prepared.env["PATH"].split(os.pathsep)
        assert not any(part.startswith(packages) for part in path)
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
def test_the_empty_proc_layout_binds_nothing_from_host_proc(tmp_path, monkeypatch):
    from core.inference import sandbox_linux

    if sandbox_linux.shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this host")
    monkeypatch.setattr(sandbox_linux, "empty_proc_layout", lambda bwrap = None: True)
    monkeypatch.setattr(sandbox_linux, "_model_cache_binds", lambda workdir: {})
    launch = sandbox_linux.prepare(
        ToolLaunchPlan(argv = ("/bin/true",), workdir = str(tmp_path), env = {"PATH": "/usr/bin"})
    )
    try:
        assert "/proc/sys" not in list(launch.argv)
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
    monkeypatch.delenv("HF_DATASETS_CACHE", raising = False)
    monkeypatch.delenv("HF_ASSETS_CACHE", raising = False)
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


class _FailingLauncher:
    """A backend whose launcher starts, then exits with *message* on stderr."""

    def __init__(self, message):
        self.message = message

    def prepare(self, plan):
        from types import SimpleNamespace
        code = f"import sys; sys.stderr.write({self.message!r}); sys.exit(1)"
        return SimpleNamespace(
            argv = [sys.executable, "-c", code],
            workdir = plan.workdir,
            env = dict(plan.env),
            preexec_fn = None,
            pass_fds = (),
            cleanup = lambda: None,
        )


@pytest.mark.skipif(sys.platform not in ("linux", "darwin"), reason = "the probe backends")
@pytest.mark.parametrize(
    ("message", "undecided"),
    [
        (f"bwrap: Creating new namespace failed: {os.strerror(errno.EAGAIN)}", True),
        (f"bwrap: Creating new namespace failed: {os.strerror(errno.ENOMEM)}", True),
        # The payload itself filling the disk says nothing about the backend.
        (f"OSError: [Errno 28] {os.strerror(errno.ENOSPC)}: 'work/x'", True),
        # bwrap reports a user-namespace limit as ENOSPC: no sandbox only when the quota is zero.
        (f"bwrap: Creating new namespace failed: {os.strerror(errno.ENOSPC)}", "quota"),
    ],
)
def test_a_launcher_that_exits_out_of_resources_is_undecided(message, undecided, monkeypatch):
    if undecided == "quota":
        monkeypatch.setattr(sandbox_probe, "_user_namespaces_disabled", lambda: True)
        launcher = _FailingLauncher(message)
        _available, reason = sandbox_probe._run_probe(launcher, "bubblewrap", ToolLaunchPlan)
        assert not isinstance(reason, sandbox_probe.UndecidedReason), reason
        # A quota other launches used up is exhaustion, not a host without a sandbox.
        monkeypatch.setattr(sandbox_probe, "_user_namespaces_disabled", lambda: False)
        undecided = True
    # bwrap forks inside the namespace it made; running out there is an exit, not an exception.
    launcher = _FailingLauncher(message)
    available, reason = sandbox_probe._run_probe(launcher, "bubblewrap", ToolLaunchPlan)
    assert available is False
    assert isinstance(reason, sandbox_probe.UndecidedReason) is undecided, reason


def test_dedicated_cache_paths_are_expanded(tmp_path, monkeypatch):
    from core.inference import sandbox_linux
    import utils.hf_cache_settings as cache_settings

    home = tmp_path / "home"
    (home / "portable" / "datasets").mkdir(parents = True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HF_DATASETS_CACHE", " ~/portable/datasets ")
    monkeypatch.delenv("HF_ASSETS_CACHE", raising = False)
    monkeypatch.setattr(
        cache_settings,
        "get_hf_cache_paths",
        lambda: cache_settings.HuggingFaceCachePaths(
            cache_home = home / "hf",
            hub_cache = home / "hf" / "hub",
            xet_cache = home / "hf" / "xet",
            source = "environment",
        ),
    )
    monkeypatch.setattr(
        sandbox_linux,
        "_cache_hazard_within_deadline",
        lambda name, path: None if os.path.isdir(path) else sandbox_linux.CACHE_MISSING,
    )
    binds = sandbox_linux._model_cache_binds(str(tmp_path / "session"))
    assert binds["datasets"] == str(home / "portable" / "datasets")


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


def test_a_project_sandbox_that_is_a_windows_junction_is_not_followed(project, monkeypatch):
    # islink() does not report a junction; its reparse-point attribute does.
    monkeypatch.setattr(
        tools,
        "_is_windows_reparse_point",
        lambda path: os.path.basename(path) == "sandbox",
    )
    assert tools._get_project_workdir("project-p1") is None


@pytest.mark.skipif(sys.platform not in ("linux", "darwin"), reason = "the probe backends")
def test_a_probe_out_of_python_memory_is_undecided(tmp_path, monkeypatch, fresh_probe_cache):
    def exhausted():
        raise MemoryError()

    monkeypatch.setattr(sandbox_probe, "_probe_base", exhausted)
    with pytest.raises(os_sandbox.SandboxBuildError, match = "OS_ISOLATION_UNAVAILABLE"):
        os_sandbox.prepare_tool_launch(_auto_plan(tmp_path))
