# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""stdio MCP servers get the managed Node bin dir on PATH.

Run from studio/backend:  python -m pytest tests/test_mcp_stdio_node_path.py -q
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

from core.inference import mcp_client
from utils import node_runtime


@pytest.fixture(autouse = True)
def _reset_managed_node_memo():
    node_runtime._reset_managed_node_check()
    yield
    node_runtime._reset_managed_node_check()


@pytest.fixture
def managed_node(tmp_path, monkeypatch):
    """A managed install that clears the version floor (the probe is stubbed: these
    tests cover PATH assembly, not `node -v`)."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    bin_dir = tmp_path / "studio" / "node" / ("" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(node_runtime, "managed_node_bin_dir", lambda: bin_dir)
    monkeypatch.setattr(node_runtime, "managed_node_usable", lambda: True)

    def _no_usable_node(
        path,
        require_npm = True,
        require_npx = True,
    ):
        return False

    monkeypatch.setattr(node_runtime, "_path_has_usable_node", _no_usable_node)
    return bin_dir


@pytest.fixture
def runtime_free_dir(tmp_path):
    """A base PATH that resolves no runtime on every host. Real system dirs cannot be used
    for this: a developer machine with Node in /usr/bin resolves a complete toolchain there,
    so path_with_managed_node returns it unchanged and a prepend assertion fails, while a CI
    image without Node passes. The empty dir makes the outcome depend on the managed install
    under test rather than on what the host happens to ship."""
    base = tmp_path / "runtime-free"
    base.mkdir()
    return base


@pytest.fixture
def managed_node_install(tmp_path, monkeypatch):
    """The real locator + a stub node binary, so managed_node_usable() is exercised."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    bin_dir = tmp_path / "studio" / "node" / ("" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = node_runtime.managed_node_binary()
    binary.write_text("")
    return bin_dir


def test_bin_dir_none_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    assert node_runtime.managed_node_bin_dir() is None


def test_path_prepends_managed_node(managed_node):
    assert node_runtime.path_with_managed_node("/usr/bin") == f"{managed_node}{os.pathsep}/usr/bin"


def test_path_unchanged_when_already_present(managed_node):
    existing = f"{managed_node}{os.pathsep}/usr/bin"
    assert node_runtime.path_with_managed_node(existing) == existing


def test_path_unchanged_without_managed_node(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    assert node_runtime.path_with_managed_node("/usr/bin") == "/usr/bin"


def test_stdio_env_adds_node_to_inherited_path(managed_node, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    env = mcp_client._stdio_env(None)
    assert env["PATH"] == f"{managed_node}{os.pathsep}/usr/bin"


def test_stdio_env_keeps_server_env_and_extends_its_path(managed_node):
    env = mcp_client._stdio_env({"API_KEY": "sk-1", "PATH": "/opt/bin"})
    assert env["API_KEY"] == "sk-1"
    assert env["PATH"] == f"{managed_node}{os.pathsep}/opt/bin"


def test_stdio_env_is_none_without_managed_node_or_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.delenv("PATH", raising = False)
    assert mcp_client._stdio_env(None) is None


@pytest.mark.parametrize(
    "environment",
    [
        pytest.param({"BAD\x00NAME": "value"}, id = "nul-name"),
        pytest.param({"NAME": "bad\x00value"}, id = "nul-value"),
        pytest.param({"BAD=NAME": "value"}, id = "equals-name"),
    ],
)
def test_stdio_env_rejects_invalid_legacy_values(environment):
    with pytest.raises(ValueError):
        mcp_client._stdio_env(environment, "python")


def test_client_passes_node_path_to_transport(managed_node, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setenv("PATH", "/usr/bin")
    _make_executable(managed_node, "npx")
    client = mcp_client._client("npx -y @modelcontextprotocol/server-filesystem /tmp", None)
    assert client.transport.env["PATH"].split(os.pathsep)[0] == str(managed_node)


def _make_executable(bin_dir, name):
    """A stub the platform's own PATH lookup will accept (.cmd via PATHEXT on Windows)."""
    path = bin_dir / (f"{name}.cmd" if os.name == "nt" else name)
    path.write_text("")
    if os.name == "nt" and name in ("npm", "npx"):
        (bin_dir / "node.exe").write_text("")
        cli = bin_dir / "node_modules" / "npm" / "bin" / f"{name}-cli.js"
        cli.parent.mkdir(parents = True, exist_ok = True)
        cli.write_text("")
    elif os.name != "nt":
        path.chmod(0o755)
    return path


def _expected_node_launcher_argv(launcher, arguments):
    if os.name != "nt":
        return [str(launcher), *arguments]
    bin_dir = launcher.parent
    cli = bin_dir / "node_modules" / "npm" / "bin" / f"{launcher.stem}-cli.js"
    return [str(bin_dir / "node.exe"), str(cli), *arguments]


def test_stdio_argv_resolves_managed_npx(managed_node):
    npx = _make_executable(managed_node, "npx")
    env = mcp_client._stdio_env(None)
    argv = mcp_client._stdio_argv(["npx", "-y", "pkg"], env)
    assert argv == _expected_node_launcher_argv(npx, ["-y", "pkg"])


def test_stdio_argv_keeps_unresolvable_command(managed_node):
    parts = ["definitely-not-on-path-9304", "-y"]
    env = mcp_client._stdio_env(None)
    if os.name == "nt":
        with pytest.raises(ValueError, match = "configured PATH"):
            mcp_client._stdio_argv(parts, env)
        return
    argv = mcp_client._stdio_argv(parts, env)
    assert argv == ["definitely-not-on-path-9304", "-y"]


def test_stdio_argv_prefers_child_path_over_parent(managed_node, monkeypatch, tmp_path):
    """The parent env must not decide the lookup: only the child PATH has the command."""
    npx = _make_executable(managed_node, "npx")
    bare = tmp_path / "empty"
    bare.mkdir()
    monkeypatch.setenv("PATH", str(bare))
    assert shutil.which("npx") is None
    argv = mcp_client._stdio_argv(["npx"], mcp_client._stdio_env(None))
    assert argv == _expected_node_launcher_argv(npx, [])


@pytest.mark.parametrize("launcher", ["npm", "npx"])
def test_windows_stdio_argv_bypasses_batch_launcher(launcher, monkeypatch, tmp_path):
    bin_dir = tmp_path / "node"
    cli = bin_dir / "node_modules" / "npm" / "bin" / f"{launcher}-cli.js"
    cli.parent.mkdir(parents = True)
    cli.write_text("")
    node = bin_dir / "node.exe"
    node.write_text("")
    batch = bin_dir / f"{launcher}.cmd"
    batch.write_text("")
    monkeypatch.setattr(mcp_client, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        mcp_client.shutil,
        "which",
        lambda command, path = None: str(batch) if command == launcher else str(node),
    )
    arguments = ["%TOKEN%", "a&b", "x|y", 'say "hello"', "a b", ""]

    argv = mcp_client._stdio_argv([launcher, *arguments], {"PATH": str(bin_dir)})

    assert argv == [str(node), str(cli), *arguments]


def test_windows_stdio_argv_rejects_batch_when_cli_is_unknown(monkeypatch, tmp_path):
    batch = tmp_path / "custom" / "npx.cmd"
    batch.parent.mkdir()
    batch.write_text("")
    monkeypatch.setattr(mcp_client, "_IS_WINDOWS", True)
    monkeypatch.setattr(mcp_client.shutil, "which", lambda command, path = None: str(batch))

    with pytest.raises(ValueError, match = "npm CLI script"):
        mcp_client._stdio_argv(["npx", "package"], {"PATH": str(tmp_path)})


@pytest.mark.parametrize("argument", ["%TOKEN%", "a&b", "x|y", 'say "hello"', "(group)"])
def test_windows_stdio_argv_rejects_unsafe_batch_arguments(argument, monkeypatch, tmp_path):
    batch = tmp_path / "mcp-server-example.cmd"
    batch.write_text("")
    monkeypatch.setattr(mcp_client, "_IS_WINDOWS", True)
    monkeypatch.setattr(mcp_client.shutil, "which", lambda command, path = None: str(batch))

    with pytest.raises(ValueError, match = "cannot safely preserve these MCP command arguments"):
        mcp_client._stdio_argv(
            ["mcp-server-example", argument],
            {"PATH": str(tmp_path)},
        )


def test_windows_stdio_argv_keeps_safe_batch_arguments(monkeypatch, tmp_path):
    batch = tmp_path / "mcp-server-example.cmd"
    batch.write_text("")
    monkeypatch.setattr(mcp_client, "_IS_WINDOWS", True)
    monkeypatch.setattr(mcp_client.shutil, "which", lambda command, path = None: str(batch))
    arguments = [
        "--port",
        "3000",
        r"C:\Users\me\data",
        r"C:\Program Files (x86)\mcp data",
        "a & b",
        "",
    ]

    assert mcp_client._stdio_argv(["mcp-server-example", *arguments], {"PATH": str(tmp_path)}) == [
        str(batch),
        *arguments,
    ]


def test_windows_stdio_argv_keeps_argument_free_batch(monkeypatch, tmp_path):
    batch = tmp_path / "mcp-server-example.cmd"
    batch.write_text("")
    monkeypatch.setattr(mcp_client, "_IS_WINDOWS", True)
    monkeypatch.setattr(mcp_client.shutil, "which", lambda command, path = None: str(batch))

    assert mcp_client._stdio_argv(["mcp-server-example"], {"PATH": str(tmp_path)}) == [str(batch)]


@pytest.mark.skipif(os.name != "nt", reason = "requires an installed Windows Node runtime")
@pytest.mark.parametrize("launcher", ["npm", "npx"])
def test_windows_installed_node_launcher_avoids_cmd_shell(launcher):
    resolved = shutil.which(launcher)
    if resolved is None:
        pytest.skip(f"{launcher} is not installed")
    assert Path(resolved).suffix.lower() in (".cmd", ".bat")
    arguments = ["%TOKEN%", "a&b", "x|y", 'say "hello"', "a b", ""]

    argv = mcp_client._stdio_argv([launcher, *arguments], {"PATH": os.environ.get("PATH", "")})

    assert Path(argv[0]).name.lower() == "node.exe"
    assert Path(argv[1]).name.lower() == f"{launcher}-cli.js"
    assert Path(argv[1]).is_file()
    assert argv[2:] == arguments


def test_client_spawns_managed_npx_by_full_path(managed_node, monkeypatch, tmp_path):
    npx = _make_executable(managed_node, "npx")
    bare = tmp_path / "empty"
    bare.mkdir()
    monkeypatch.setenv("PATH", str(bare))
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    client = mcp_client._client("npx -y @modelcontextprotocol/server-filesystem /tmp", None)
    expected = _expected_node_launcher_argv(
        npx, ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )
    assert [client.transport.command, *client.transport.args] == expected


def test_runtime_free_dir_resolves_nothing(runtime_free_dir):
    """Pins the precondition the prepend tests below rely on. Asserting against a real
    system dir like /usr/bin instead would make them read the host: where it ships a
    Node the base PATH already resolves a runtime, so path_with_managed_node correctly
    returns it unchanged and the prepend assertions flip."""
    assert node_runtime._path_has_usable_node(str(runtime_free_dir)) is False
    assert node_runtime._path_has_usable_node(str(runtime_free_dir), require_npm = False) is False


def test_stale_managed_node_is_not_prepended(managed_node_install, monkeypatch, runtime_free_dir):
    """A dir left behind after the host moved to a system Node must not win the lookup."""
    monkeypatch.setattr(node_runtime, "_node_version_ok", lambda executable, path = None: False)
    assert node_runtime.managed_node_usable() is False
    base = str(runtime_free_dir)
    assert node_runtime.path_with_managed_node(base) == base


def test_usable_managed_node_is_prepended(managed_node_install, monkeypatch, runtime_free_dir):
    monkeypatch.setattr(node_runtime, "_node_version_ok", lambda executable, path = None: True)
    assert node_runtime.managed_node_usable() is True
    base = str(runtime_free_dir)
    expected = f"{managed_node_install}{os.pathsep}{base}"
    assert node_runtime.path_with_managed_node(base) == expected


def test_stale_managed_node_leaves_stdio_env_alone(
    managed_node_install, monkeypatch, runtime_free_dir
):
    monkeypatch.setattr(node_runtime, "_node_version_ok", lambda executable, path = None: False)
    monkeypatch.setenv("PATH", str(runtime_free_dir))
    assert mcp_client._stdio_env(None)["PATH"] == str(runtime_free_dir)


def test_managed_node_check_is_memoized_on_success(managed_node_install, monkeypatch):
    """One probe per process once usable: _stdio_env runs on every client build."""
    calls = []

    def _record(executable, path = None):
        calls.append(executable)
        return True

    monkeypatch.setattr(node_runtime, "_node_version_ok", _record)
    assert node_runtime.managed_node_usable() is True
    assert node_runtime.managed_node_usable() is True
    assert len(calls) == 1


def test_explicit_empty_path_is_preserved(managed_node, monkeypatch):
    """PATH: "" is a deliberate sandbox; the inherited PATH must not replace it."""
    monkeypatch.setenv("PATH", "/usr/bin")
    env = mcp_client._stdio_env({"API_KEY": "sk-1", "PATH": ""})
    assert env["PATH"] == ""
    assert env["API_KEY"] == "sk-1"


def test_explicit_empty_path_blocks_host_lookup(managed_node, monkeypatch, tmp_path):
    """The host PATH must not resolve argv[0] once the server opted out of PATH."""
    host = tmp_path / "hostbin"
    host.mkdir()
    _make_executable(host, "hostcmd")
    monkeypatch.setenv("PATH", str(host))
    assert shutil.which("hostcmd") is not None
    if os.name == "nt":
        with pytest.raises(ValueError, match = "configured PATH"):
            mcp_client._stdio_argv(["hostcmd"], {"PATH": ""})
    else:
        assert mcp_client._stdio_argv(["hostcmd"], {"PATH": ""}) == ["hostcmd"]


def test_explicit_empty_path_still_allows_absolute_command(managed_node):
    argv = mcp_client._stdio_argv([sys.executable, "-c", "pass"], {"PATH": ""})
    assert argv == [sys.executable, "-c", "pass"]


def test_absent_path_still_inherits(managed_node, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    assert (
        mcp_client._stdio_env({"API_KEY": "sk-1"})["PATH"] == f"{managed_node}{os.pathsep}/usr/bin"
    )


def test_path_preserves_trailing_empty_component(managed_node):
    """An empty component means the working directory on POSIX; keep it verbatim."""
    configured = f"/usr/bin{os.pathsep}"
    expected = f"{managed_node}{os.pathsep}{configured}"
    assert node_runtime.path_with_managed_node(configured) == expected


def test_path_preserves_bare_empty_components(managed_node):
    expected = f"{managed_node}{os.pathsep}{os.pathsep}"
    assert node_runtime.path_with_managed_node(os.pathsep) == expected


def test_stdio_env_preserves_empty_component_from_config(managed_node):
    configured = f"/usr/bin{os.pathsep}"
    env = mcp_client._stdio_env({"PATH": configured})
    assert env["PATH"] == f"{managed_node}{os.pathsep}{configured}"


def _system_node_dir(tmp_path, with_npx = True):
    """A system runtime dir; without npx/npm it mirrors a host where setup picked bundled."""
    sysbin = tmp_path / "sysbin"
    sysbin.mkdir()
    _make_executable(sysbin, "node")
    if with_npx:
        _make_executable(sysbin, "npm")
        _make_executable(sysbin, "npx")
    return sysbin


def _patch_floors(monkeypatch, predicate):
    """Both floors move together: the installers require node AND npm to pass."""

    def _check(executable, path = None):
        return predicate(executable)

    monkeypatch.setattr(node_runtime, "_node_version_ok", _check)
    monkeypatch.setattr(node_runtime, "_npm_version_ok", _check)


def test_adequate_system_node_is_not_shadowed(managed_node_install, monkeypatch, tmp_path):
    """A leftover managed install must not override a Node the PATH already provides."""
    sysbin = _system_node_dir(tmp_path)
    _patch_floors(monkeypatch, lambda executable, path = None: True)
    assert node_runtime.path_with_managed_node(str(sysbin)) == str(sysbin)


def test_managed_node_used_when_system_node_is_below_floor(
    managed_node_install, monkeypatch, tmp_path
):
    sysbin = _system_node_dir(tmp_path)
    _patch_floors(monkeypatch, lambda executable, path = None: "sysbin" not in str(executable))
    expected = f"{managed_node_install}{os.pathsep}{sysbin}"
    assert node_runtime.path_with_managed_node(str(sysbin)) == expected


def test_managed_node_used_when_path_has_no_node(managed_node_install, monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(node_runtime, "_node_version_ok", lambda executable, path = None: True)
    expected = f"{managed_node_install}{os.pathsep}{empty}"
    assert node_runtime.path_with_managed_node(str(empty)) == expected


def test_system_node_probe_is_memoized(managed_node_install, monkeypatch, tmp_path):
    sysbin = _system_node_dir(tmp_path)
    calls = []
    _patch_floors(monkeypatch, lambda executable, path = None: calls.append(executable) or True)
    assert node_runtime.path_with_managed_node(str(sysbin)) == str(sysbin)
    after_first = len(calls)
    assert node_runtime.path_with_managed_node(str(sysbin)) == str(sysbin)
    assert len(calls) == after_first, calls


def test_probe_memo_does_not_answer_across_paths(managed_node_install, monkeypatch, tmp_path):
    """One npm shim, two PATHs, two different runtimes. npm and npx are
    ``#!/usr/bin/env node`` scripts, so the shim clears the floor only under the PATH whose
    node is adequate. The success memo must not let the passing PATH answer for the other."""
    shim = tmp_path / "shim"
    shim.mkdir()
    _make_executable(shim, "npm")
    _make_executable(shim, "npx")
    good = tmp_path / "good"
    good.mkdir()
    _make_executable(good, "node")
    old = tmp_path / "old"
    old.mkdir()
    _make_executable(old, "node")

    # Patched directly rather than through _patch_floors: that helper drops the path
    # argument, which is the whole dimension under test here.
    def _npm_floor(executable, path = None):
        # The shim runs whichever node its PATH reaches, so only the good PATH clears.
        return path is not None and str(good) in path

    monkeypatch.setattr(node_runtime, "_node_version_ok", lambda executable, path = None: True)
    monkeypatch.setattr(node_runtime, "_npm_version_ok", _npm_floor)
    good_path = f"{shim}{os.pathsep}{good}"
    old_path = f"{shim}{os.pathsep}{old}"
    assert node_runtime._path_has_usable_node(good_path) is True
    # Same npm executable, PATH whose node is below the floor: must be re-probed, not served
    # from the entry the good PATH cached.
    assert node_runtime._path_has_usable_node(old_path) is False


def test_managed_node_used_when_system_lacks_npx(managed_node_install, monkeypatch, tmp_path):
    """decide_node_source installs bundled when npm is missing, so node alone is not enough."""
    sysbin = _system_node_dir(tmp_path, with_npx = False)
    monkeypatch.setattr(node_runtime, "_node_version_ok", lambda executable, path = None: True)
    expected = f"{managed_node_install}{os.pathsep}{sysbin}"
    assert node_runtime.path_with_managed_node(str(sysbin)) == expected


def test_complete_system_runtime_is_not_shadowed(managed_node_install, monkeypatch, tmp_path):
    sysbin = _system_node_dir(tmp_path)
    _patch_floors(monkeypatch, lambda executable, path = None: True)
    assert node_runtime.path_with_managed_node(str(sysbin)) == str(sysbin)


def test_shadowed_managed_dir_moves_to_front(managed_node_install, monkeypatch, tmp_path):
    """Already on PATH but behind a stale runtime: it has to move up, not stay put."""
    stale = _system_node_dir(tmp_path)
    _patch_floors(monkeypatch, lambda executable, path = None: "sysbin" not in str(executable))
    configured = f"{stale}{os.pathsep}{managed_node_install}"
    expected = f"{managed_node_install}{os.pathsep}{stale}"
    assert node_runtime.path_with_managed_node(configured) == expected


def test_managed_dir_already_first_is_unchanged(managed_node_install, monkeypatch, tmp_path):
    stale = _system_node_dir(tmp_path)
    _patch_floors(monkeypatch, lambda executable, path = None: "sysbin" not in str(executable))
    configured = f"{managed_node_install}{os.pathsep}{stale}"
    assert node_runtime.path_with_managed_node(configured) == configured


def test_non_node_command_keeps_its_configured_path(managed_node):
    """A Python server pinning its own toolchain must not get the managed Node."""
    env = mcp_client._stdio_env({"PATH": "/proj/node18/bin"}, "python")
    assert env["PATH"] == "/proj/node18/bin"


def test_non_node_command_leaves_inherited_path_alone(managed_node, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    assert mcp_client._stdio_env(None, "uvx") is None


def test_node_family_commands_still_augmented(managed_node, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    for command in ("npx", "node", "npm", "/usr/local/bin/npx", "NPX.CMD", "node.exe"):
        env = mcp_client._stdio_env(None, command)
        assert env["PATH"] == f"{managed_node}{os.pathsep}/usr/bin", command


def test_is_node_command_rejects_lookalikes():
    assert not mcp_client._is_node_command("nodemon")
    assert not mcp_client._is_node_command("python")
    assert not mcp_client._is_node_command("/opt/bin/deno")


def test_client_does_not_touch_env_for_a_python_server(managed_node, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setenv("PATH", "/usr/bin")
    client = mcp_client._client("python -m my_server", {"API_KEY": "sk-1"})
    assert client.transport.env == {"API_KEY": "sk-1"}


def test_windows_npx_sibling_runtime_is_the_one_validated(
    managed_node_install, monkeypatch, tmp_path
):
    """npx.cmd runs the node.exe beside it, so that runtime decides, not PATH order."""
    good = tmp_path / "good"
    good.mkdir()
    _make_executable(good, "node")
    old = tmp_path / "old"
    old.mkdir()
    _make_executable(old, "npm")
    _make_executable(old, "npx")
    (old / "node.exe").write_text("")
    monkeypatch.setattr(node_runtime, "_IS_WINDOWS", True)
    checked = []
    _patch_floors(
        monkeypatch,
        lambda executable, path = None: (
            checked.append(str(executable)) or not str(executable).startswith(str(old))
        ),
    )
    configured = f"{good}{os.pathsep}{old}"
    result = node_runtime.path_with_managed_node(configured)
    assert any(c.endswith("node.exe") for c in checked), checked
    assert result == f"{managed_node_install}{os.pathsep}{configured}"


def test_posix_split_layout_still_trusts_the_path_node(managed_node_install, monkeypatch, tmp_path):
    """On POSIX npx is a shebang script resolving node via PATH, so a split is fine."""
    good = tmp_path / "good"
    good.mkdir()
    _make_executable(good, "node")
    other = tmp_path / "other"
    other.mkdir()
    _make_executable(other, "npm")
    _make_executable(other, "npx")
    _patch_floors(monkeypatch, lambda executable, path = None: True)
    configured = f"{good}{os.pathsep}{other}"
    assert node_runtime.path_with_managed_node(configured) == configured


@pytest.fixture
def windows_env(monkeypatch):
    monkeypatch.setattr(mcp_client, "_IS_WINDOWS", True)


@pytest.fixture
def posix_env(monkeypatch):
    """Pin the platform: these assert POSIX semantics and must not follow the runner."""
    monkeypatch.setattr(mcp_client, "_IS_WINDOWS", False)
    monkeypatch.setattr(node_runtime, "_IS_WINDOWS", False)


def test_windows_lowercase_path_key_is_recognized(managed_node, monkeypatch, windows_env):
    monkeypatch.setenv("PATH", "/usr/bin")
    env = mcp_client._stdio_env({"Path": "/opt/bin"}, "npx")
    assert env["Path"] == f"{managed_node}{os.pathsep}/opt/bin"
    assert "PATH" not in env


def test_windows_lowercase_empty_path_is_still_a_sandbox(managed_node, monkeypatch, windows_env):
    monkeypatch.setenv("PATH", "/usr/bin")
    assert mcp_client._stdio_env({"Path": ""}, "npx") == {"Path": ""}


def test_windows_lowercase_path_blocks_host_lookup(managed_node, windows_env):
    with pytest.raises(ValueError, match = "configured PATH"):
        mcp_client._stdio_argv(["npx"], {"Path": ""})


def test_posix_treats_path_and_lowercase_path_as_distinct(managed_node, monkeypatch, posix_env):
    monkeypatch.setenv("PATH", "/usr/bin")
    env = mcp_client._stdio_env({"Path": "/opt/bin"}, "npx")
    assert env["Path"] == "/opt/bin"
    assert env["PATH"] == f"{managed_node}{os.pathsep}/usr/bin"


def test_npm_below_installer_floor_falls_back_to_managed(
    managed_node_install, monkeypatch, tmp_path
):
    """Node clears its floor but npm is 10, which is why setup installed the managed one."""
    sysbin = _system_node_dir(tmp_path)
    monkeypatch.setattr(node_runtime, "_node_version_ok", lambda executable, path = None: True)
    monkeypatch.setattr(
        node_runtime,
        "_npm_version_ok",
        lambda executable, path = None: not str(executable).startswith(str(sysbin)),
    )
    expected = f"{managed_node_install}{os.pathsep}{sysbin}"
    assert node_runtime.path_with_managed_node(str(sysbin)) == expected


def test_npm_floor_matches_the_installers():
    assert node_runtime._npm_meets_floor("11.0.0")
    assert node_runtime._npm_meets_floor("v12.1.2")
    assert not node_runtime._npm_meets_floor("10.9.3")
    assert not node_runtime._npm_meets_floor("")


def _node_only_dir(tmp_path):
    """Debian/Ubuntu ship nodejs and npm as separate packages, so this is ordinary."""
    nodeonly = tmp_path / "nodeonly"
    nodeonly.mkdir()
    _make_executable(nodeonly, "node")
    return nodeonly


def test_direct_node_server_keeps_a_node_only_path(managed_node_install, monkeypatch, tmp_path):
    nodeonly = _node_only_dir(tmp_path)
    _patch_floors(monkeypatch, lambda executable, path = None: True)
    unchanged = node_runtime.path_with_managed_node(
        str(nodeonly), require_npm = False, require_npx = False
    )
    assert unchanged == str(nodeonly)


def test_npx_server_still_needs_npx_on_a_node_only_path(
    managed_node_install, monkeypatch, tmp_path
):
    """node alone cannot launch an ``npx`` server, so the managed dir still goes on."""
    nodeonly = _node_only_dir(tmp_path)
    _patch_floors(monkeypatch, lambda executable, path = None: True)
    expected = f"{managed_node_install}{os.pathsep}{nodeonly}"
    assert (
        node_runtime.path_with_managed_node(str(nodeonly), require_npm = False, require_npx = True)
        == expected
    )


def test_npx_server_keeps_a_path_with_npx_but_no_npm(managed_node_install, monkeypatch, tmp_path):
    """A curated PATH exposing node and npx without a separate npm launcher runs npx fine:
    npx-cli.js delegates in-process to the npm it ships with and never looks up an ``npm``
    executable. Demanding one would prepend the managed dir and silently swap the
    configured toolchain for a different npx, changing package resolution."""
    curated = tmp_path / "curated"
    curated.mkdir()
    _make_executable(curated, "node")
    _make_executable(curated, "npx")
    _patch_floors(monkeypatch, lambda executable, path = None: True)
    monkeypatch.setenv("PATH", str(curated))
    assert mcp_client._stdio_env(None, "npx")["PATH"] == str(curated)
    assert mcp_client._stdio_argv(["npx", "-y", "server"], {"PATH": str(curated)})[0].startswith(
        str(curated)
    )


def test_npx_only_path_is_still_held_to_the_npm_floor(managed_node_install, monkeypatch, tmp_path):
    """``npx -v`` reports the bundled npm's version, so an npx-only PATH is floor-checked
    through npx rather than waved through."""
    curated = tmp_path / "curated"
    curated.mkdir()
    _make_executable(curated, "node")
    _make_executable(curated, "npx")
    monkeypatch.setattr(node_runtime, "_node_version_ok", lambda executable, path = None: True)
    probed = []

    def _npm_floor(executable, path = None):
        probed.append(str(executable))
        return False  # the bundled npm is below the installers' floor

    monkeypatch.setattr(node_runtime, "_npm_version_ok", _npm_floor)
    expected = f"{managed_node_install}{os.pathsep}{curated}"
    assert (
        node_runtime.path_with_managed_node(str(curated), require_npm = False, require_npx = True)
        == expected
    )
    assert any(os.path.basename(p).startswith("npx") for p in probed), probed


def test_stdio_env_does_not_shadow_node_for_a_direct_node_server(
    managed_node_install, monkeypatch, tmp_path
):
    nodeonly = _node_only_dir(tmp_path)
    _patch_floors(monkeypatch, lambda executable, path = None: True)
    monkeypatch.setenv("PATH", str(nodeonly))
    assert mcp_client._stdio_env(None, "node")["PATH"] == str(nodeonly)


def test_stdio_env_still_augments_for_npx_on_a_node_only_path(
    managed_node_install, monkeypatch, tmp_path
):
    nodeonly = _node_only_dir(tmp_path)
    _patch_floors(monkeypatch, lambda executable, path = None: True)
    monkeypatch.setenv("PATH", str(nodeonly))
    env = mcp_client._stdio_env(None, "npx")
    assert env["PATH"] == f"{managed_node_install}{os.pathsep}{nodeonly}"


def test_runtime_requirements_match_each_launcher():
    assert mcp_client._runtime_requirements("node") == (False, False)
    assert mcp_client._runtime_requirements("/usr/bin/node") == (False, False)
    assert mcp_client._runtime_requirements("node.exe") == (False, False)
    assert mcp_client._runtime_requirements("npm") == (True, False)
    assert mcp_client._runtime_requirements("npm.cmd") == (True, False)
    assert mcp_client._runtime_requirements("npx") == (False, True)
    assert mcp_client._runtime_requirements("npx.cmd") == (False, True)
    assert mcp_client._runtime_requirements(None) == (True, True)


def test_absolute_node_launcher_keeps_its_configured_path(managed_node, monkeypatch):
    """An explicit interpreter runs regardless of PATH; its children must match it."""
    monkeypatch.setenv("PATH", "/usr/bin")
    env = mcp_client._stdio_env({"API_KEY": "sk-1"}, "/opt/node/bin/node")
    assert env == {"API_KEY": "sk-1"}


def test_bare_node_command_is_still_helped(managed_node, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    env = mcp_client._stdio_env(None, "node")
    assert env["PATH"] == f"{managed_node}{os.pathsep}/usr/bin"


def test_absolute_npx_launcher_still_gets_the_runtime(managed_node, monkeypatch):
    """npx needs a node on PATH to run at all, so it keeps the managed dir."""
    monkeypatch.setenv("PATH", "/usr/bin")
    env = mcp_client._stdio_env(None, "/opt/node/bin/npx")
    assert env["PATH"] == f"{managed_node}{os.pathsep}/usr/bin"


def test_command_selects_runtime_only_for_node_paths():
    assert mcp_client._command_selects_runtime("/opt/node/bin/node")
    assert not mcp_client._command_selects_runtime("node")
    assert not mcp_client._command_selects_runtime("/opt/node/bin/npx")
    assert not mcp_client._command_selects_runtime(None)


def test_direct_npm_server_does_not_need_npx(managed_node_install, monkeypatch, tmp_path):
    """The installers gate on node and npm only, so a missing npx is irrelevant here."""
    nonpx = tmp_path / "nonpx"
    nonpx.mkdir()
    _make_executable(nonpx, "node")
    _make_executable(nonpx, "npm")
    _patch_floors(monkeypatch, lambda executable, path = None: True)
    monkeypatch.setenv("PATH", str(nonpx))
    assert mcp_client._stdio_env(None, "npm")["PATH"] == str(nonpx)


def test_npx_server_on_the_same_path_still_gets_managed(
    managed_node_install, monkeypatch, tmp_path
):
    nonpx = tmp_path / "nonpx"
    nonpx.mkdir()
    _make_executable(nonpx, "node")
    _make_executable(nonpx, "npm")
    _patch_floors(monkeypatch, lambda executable, path = None: True)
    monkeypatch.setenv("PATH", str(nonpx))
    env = mcp_client._stdio_env(None, "npx")
    assert env["PATH"] == f"{managed_node_install}{os.pathsep}{nonpx}"


@pytest.mark.skipif(os.name == "nt", reason = "POSIX shebang launcher")
def test_npm_probe_runs_with_the_candidate_path(managed_node_install, monkeypatch, tmp_path):
    """npm is a `#!/usr/bin/env node` script, so the probe needs the candidate's node."""
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()
    node = toolchain / "node"
    node.write_text("#!/bin/sh\necho v22.12.0\n")
    node.chmod(0o755)
    npm = toolchain / "npm"
    npm.write_text("#!/usr/bin/env node\n")  # only runs if node is on the probe PATH
    npm.chmod(0o755)
    _make_executable(toolchain, "npx")
    # a backend PATH with no node at all, which is why this PR exists
    monkeypatch.setenv("PATH", "/nonexistent-9304")
    monkeypatch.setattr(node_runtime, "_npm_meets_floor", lambda version: True)
    # the managed install must be usable, or the function returns before the npm probe
    monkeypatch.setattr(node_runtime, "managed_node_usable", lambda: True)
    assert node_runtime.path_with_managed_node(str(toolchain)) == str(toolchain)


def test_windows_npm_sibling_runtime_is_validated(managed_node_install, monkeypatch, tmp_path):
    """npm.cmd prefers the node.exe beside it just as npx.cmd does."""
    good = tmp_path / "good"
    good.mkdir()
    _make_executable(good, "node")
    old = tmp_path / "old"
    old.mkdir()
    _make_executable(old, "npm")
    (old / "node.exe").write_text("")
    monkeypatch.setattr(node_runtime, "_IS_WINDOWS", True)
    checked = []

    def _record(executable, path = None):
        checked.append(str(executable))
        return not str(executable).startswith(str(old))

    _patch_floors(monkeypatch, lambda executable: _record(executable))
    configured = f"{good}{os.pathsep}{old}"
    result = node_runtime.path_with_managed_node(configured, require_npm = True, require_npx = False)
    assert any(c.endswith("node.exe") for c in checked), checked
    assert result == f"{managed_node_install}{os.pathsep}{configured}"


def _good_node_only(tmp_path, name = "toolchain"):
    d = tmp_path / name
    d.mkdir()
    _make_executable(d, "node")
    return d


def test_pathed_npm_keeps_a_configured_node(managed_node_install, monkeypatch, tmp_path):
    """The pathed launcher runs either way; its shebang must keep the configured node."""
    configured = _good_node_only(tmp_path)
    _patch_floors(monkeypatch, lambda executable: True)
    monkeypatch.setenv("PATH", str(configured))
    assert mcp_client._stdio_env(None, "/opt/toolchain/npm")["PATH"] == str(configured)


def test_pathed_npx_keeps_a_configured_node(managed_node_install, monkeypatch, tmp_path):
    configured = _good_node_only(tmp_path)
    _patch_floors(monkeypatch, lambda executable: True)
    monkeypatch.setenv("PATH", str(configured))
    assert mcp_client._stdio_env(None, "/opt/toolchain/npx")["PATH"] == str(configured)


def test_pathed_npm_still_gets_node_when_path_has_none(managed_node_install, monkeypatch, tmp_path):
    """With no node at all the shebang would fail, so the managed runtime still helps."""
    empty = tmp_path / "empty"
    empty.mkdir()
    _patch_floors(monkeypatch, lambda executable: True)
    monkeypatch.setenv("PATH", str(empty))
    env = mcp_client._stdio_env(None, "/opt/toolchain/npm")
    assert env["PATH"] == f"{managed_node_install}{os.pathsep}{empty}"


def test_bare_npm_still_requires_npm_on_path(managed_node_install, monkeypatch, tmp_path):
    configured = _good_node_only(tmp_path, "bare")
    _patch_floors(monkeypatch, lambda executable: True)
    monkeypatch.setenv("PATH", str(configured))
    env = mcp_client._stdio_env(None, "npm")
    assert env["PATH"] == f"{managed_node_install}{os.pathsep}{configured}"


def test_runtime_requirements_for_pathed_launchers():
    assert mcp_client._runtime_requirements("/opt/toolchain/npm") == (False, False)
    assert mcp_client._runtime_requirements("/opt/toolchain/npx") == (False, False)
    assert mcp_client._runtime_requirements("npm") == (True, False)
    assert mcp_client._runtime_requirements("npx") == (False, True)
