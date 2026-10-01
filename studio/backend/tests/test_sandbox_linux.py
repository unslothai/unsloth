# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the Linux sandbox argv and its seccomp program actually say."""

from __future__ import annotations

import ast
import ctypes
import errno
import inspect
import json
import os
import platform
import shutil
import socket
import stat
import struct
import sys
import sysconfig
import tempfile
import threading
import time

import pytest

if sys.platform != "linux":
    pytest.skip("the bubblewrap backend is Linux only", allow_module_level = True)

from core.inference import (  # noqa: E402
    os_sandbox,
    sandbox_landlock,
    sandbox_linux,
    sandbox_seccomp,
)
from core.inference.os_sandbox import (  # noqa: E402
    SandboxUnavailableError,
    ToolLaunchPlan,
    WorkdirUnsafeError,
)  # noqa: E402


def _plan(
    workdir,
    argv = ("/bin/true",),
    **kwargs,
):
    return ToolLaunchPlan(argv = argv, workdir = str(workdir), env = {"PATH": "/usr/bin"}, **kwargs)


@pytest.fixture
def prepared(tmp_path):
    if sandbox_linux.shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this host")
    launch = sandbox_linux.prepare(_plan(tmp_path))
    yield launch
    launch.cleanup()


def _pairs(argv, flag):
    return [
        (argv[index + 1], argv[index + 2])
        for index, token in enumerate(argv)
        if token == flag and index + 2 < len(argv)
    ]


def test_the_launch_runs_bwrap_and_ends_with_the_payload_after_a_bare_separator(prepared):
    assert os.path.basename(prepared.argv[0]) == "bwrap"
    separator = prepared.argv.index("--")
    assert prepared.argv[separator + 1 :] == ("/bin/true",)
    assert prepared.argv.count("--") == 1


def test_the_network_namespace_is_deliberately_left_alone(prepared):
    assert "--unshare-net" not in prepared.argv
    assert "--unshare-all" not in prepared.argv


def test_every_namespace_except_the_network_one_is_unshared(prepared):
    for option in (
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
    ):
        assert option in prepared.argv


def test_the_session_dies_with_studio_and_detaches_from_the_terminal(prepared):
    assert "--die-with-parent" in prepared.argv
    assert "--new-session" in prepared.argv


def test_capabilities_are_dropped_and_the_filter_arrives_as_an_inherited_descriptor(prepared):
    argv = prepared.argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    fd = int(argv[argv.index("--seccomp") + 1])
    assert prepared.pass_fds == (fd,)
    assert [handle.fileno() for handle in prepared.owned_files] == [fd]
    assert os.lseek(fd, 0, os.SEEK_CUR) == 0


def test_identity_is_synthesised_rather_than_bound_from_the_host(prepared):
    binds = {destination: source for source, destination in _pairs(prepared.argv, "--ro-bind")}
    passwd, group = binds["/etc/passwd"], binds["/etc/group"]
    assert passwd != "/etc/passwd" and group != "/etc/group"
    assert prepared.cleanup_paths == [os.path.dirname(passwd)]
    assert os.stat(os.path.dirname(passwd)).st_mode & 0o777 == 0o700
    with open(passwd, encoding = "utf-8") as stream:
        entries = stream.read().splitlines()
    assert len(entries) == 1 and entries[0].split(":")[2] == str(os.getuid())


def test_the_writable_workdir_bind_lands_after_the_root_goes_read_only(prepared, tmp_path):
    argv = prepared.argv
    workdir = os.path.realpath(tmp_path)
    remount = argv.index("--remount-ro")
    assert argv[remount + 1] == "/"
    assert argv.index("--dir", 0, remount) < remount
    assert (workdir, workdir) in _pairs(argv[remount:], "--bind")
    assert argv[argv.index("--chdir") + 1] == workdir


def test_both_spellings_of_a_symlinked_workdir_get_a_mount_point(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    launch = sandbox_linux.prepare(
        ToolLaunchPlan(
            argv = (sys.executable, "-c", "pass"),
            workdir = str(link),
            env = {"PATH": "/usr/bin:/bin"},
        )
    )
    try:
        argv = launch.argv
        remount = argv.index("--remount-ro")
        made = [argv[i + 1] for i, token in enumerate(argv[:remount]) if token == "--dir"]
        assert str(link) in made, "the caller's spelling has no mount point"
        assert os.path.realpath(link) in made, "the canonical spelling has no mount point"
        bound = _pairs(argv[remount:], "--bind")
        assert (os.path.realpath(link), str(link)) in bound
    finally:
        launch.cleanup()


def test_the_private_tmpfs_replaces_the_shared_directories(prepared):
    argv = prepared.argv
    remount = argv.index("--remount-ro")
    tmpfs = [argv[i + 1] for i, token in enumerate(argv) if token == "--tmpfs"]
    assert tmpfs[:2] == ["/dev/shm", "/tmp"]
    assert argv.index("--tmpfs") > remount


def test_no_private_key_directory_enters_the_jail_at_all(prepared):
    sources = [source for source, _ in _pairs(prepared.argv, "--ro-bind-try")]
    assert "/etc/ssl/certs" in sources
    assert "/etc/ssl" not in sources and "/etc/pki" not in sources
    secret = "/etc/ssl/private"
    assert os.path.isdir(secret), "no private-key directory here, so this proves nothing"
    for source, _ in (*_pairs(prepared.argv, "--ro-bind-try"), *_pairs(prepared.argv, "--ro-bind")):
        assert not sandbox_linux._within(secret, source), source


def test_pip_gets_a_writable_target_inside_the_workdir(prepared):
    argv = prepared.argv
    packages = os.path.join(prepared.workdir, sandbox_linux.SESSION_PACKAGES_RELPATH)
    assert argv[argv.index("PIP_TARGET") + 1] == packages
    pythonpath = argv[argv.index("PYTHONPATH") + 1].split(os.pathsep)
    assert packages in pythonpath
    assert pythonpath[-1] == packages


def test_system_directories_are_bound_whole_and_never_file_by_file(prepared):
    sources = [source for source, _ in _pairs(prepared.argv, "--ro-bind-try")]
    assert "/usr/lib" in sources
    named = {
        *sandbox_linux._ETC_FILES,
        *sandbox_linux._ETC_FILES_IF_TRUSTED,
        *sandbox_linux._NETWORK_FILES,
    }
    identity_dir = prepared.cleanup_paths[0]
    for flag in ("--bind", "--ro-bind", "--ro-bind-try"):
        for source, _ in _pairs(prepared.argv, flag):
            if os.path.isdir(source):
                continue
            if source in named or os.path.dirname(source) == identity_dir:
                continue
            assert os.path.basename(source) == "pyvenv.cfg", source


def test_the_interpreter_contributes_only_a_handful_of_read_only_binds(prepared):
    runtime = [
        source
        for source, destination in _pairs(prepared.argv, "--ro-bind")
        if source == destination
    ]
    assert 0 < len(runtime) < 40, runtime
    system_roots = tuple(path for path in sandbox_linux._SYSTEM_ROOTS if os.path.isdir(path))
    for path in runtime:
        assert not any(sandbox_linux._within(path, root) for root in system_roots)


def test_nothing_binds_or_creates_the_filesystem_root(prepared):
    for flag in ("--bind", "--ro-bind", "--ro-bind-try"):
        for source, destination in _pairs(prepared.argv, flag):
            assert source != "/" and destination != "/"
    directories = [
        prepared.argv[index + 1] for index, token in enumerate(prepared.argv) if token == "--dir"
    ]
    assert "/" not in directories


def test_a_launch_with_no_command_is_refused_rather_than_handed_to_bwrap(tmp_path):
    with pytest.raises(SandboxUnavailableError, match = "needs a command"):
        sandbox_linux.prepare(_plan(tmp_path, argv = ()))


def test_a_path_bwrap_from_a_writable_location_is_never_executed(tmp_path, monkeypatch):
    planted = tmp_path / "bwrap"
    planted.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    planted.chmod(0o777)
    monkeypatch.setattr(sandbox_linux.shutil, "which", lambda _name: str(planted))

    with pytest.raises(SandboxUnavailableError, match = "trusted system installation"):
        sandbox_linux.prepare(_plan(tmp_path))


def test_the_workdir_is_the_only_writable_bind(prepared, tmp_path):
    workdir = os.path.realpath(tmp_path)
    assert _pairs(prepared.argv, "--bind") == [(workdir, workdir)]


def test_the_model_cache_shares_its_data_subdirectories_and_nothing_else(tmp_path, monkeypatch):
    cache = tmp_path / "hostcache"
    for name in ("hub", "datasets", "modules", "xet", "assets"):
        (cache / name).mkdir(parents = True)
    (cache / "token").write_text("hf_A_REAL_LOOKING_TOKEN")
    (cache / "stored_tokens").write_text("{}")
    _share_cache(monkeypatch, cache)

    launch = sandbox_linux.prepare(_plan(tmp_path))
    try:
        workdir = os.path.realpath(tmp_path)
        inner = os.path.join(workdir, ".cache", "huggingface")
        shared = _pairs(launch.argv, "--bind-try")
        assert sorted(shared) == sorted(
            (os.path.join(str(cache), name), os.path.join(inner, name))
            for name in ("hub", "datasets", "xet", "assets")
        )
        assert not any("modules" in destination for _, destination in shared)
        assert launch.argv[launch.argv.index("HF_HOME") + 1] == inner
        for credential in ("token", "stored_tokens"):
            host_path = os.path.join(str(cache), credential)
            assert not any(host_path in token for token in launch.argv), credential
    finally:
        launch.cleanup()


def test_the_cache_mount_points_are_made_here_and_left(tmp_path, cache):
    workdir = tmp_path / "session"
    workdir.mkdir()

    launch = sandbox_linux.prepare(_plan(workdir))
    launch.cleanup()
    assert (workdir / ".cache" / "huggingface" / "hub").is_dir()
    second = sandbox_linux.prepare(_plan(workdir))
    second.cleanup()
    assert (workdir / ".cache" / "huggingface" / "hub").is_dir()


def test_a_cache_directory_the_tool_call_wrote_is_left_alone(tmp_path, cache):
    workdir = tmp_path / "session"
    (workdir / ".cache" / "huggingface").mkdir(parents = True)
    (workdir / ".cache" / "notes.txt").write_text("the user's")

    launch = sandbox_linux.prepare(_plan(workdir))
    launch.cleanup()
    assert (workdir / ".cache" / "notes.txt").read_text() == "the user's"


def test_the_model_cache_bind_is_absent_when_the_host_has_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_linux, "_model_cache_binds", lambda workdir: {})
    launch = sandbox_linux.prepare(_plan(tmp_path))
    try:
        assert "HF_HOME" not in launch.argv
        assert [
            p for p in _pairs(launch.argv, "--bind") if p[0] != os.path.realpath(tmp_path)
        ] == []
    finally:
        launch.cleanup()


def test_the_inner_home_and_tmpdir_are_set_by_bwrap_not_by_the_caller_environment(prepared):
    argv = prepared.argv
    assert argv[argv.index("HOME") + 1] == prepared.workdir
    assert argv[argv.index("TMPDIR") + 1] == "/tmp"
    assert prepared.env == {"PATH": "/usr/bin"}


def test_a_tmpdir_inside_the_workdir_survives_into_the_jail(tmp_path):
    scratch = tmp_path / "unsloth-tmp"
    scratch.mkdir()
    plan = ToolLaunchPlan(
        argv = ("/bin/true",),
        workdir = str(tmp_path),
        env = {"PATH": "/usr/bin", "TMPDIR": str(scratch)},
    )
    launch = sandbox_linux.prepare(plan)
    try:
        assert launch.argv[launch.argv.index("TMPDIR") + 1] == str(scratch)
    finally:
        launch.cleanup()


def test_a_tmpdir_outside_the_workdir_is_replaced_by_the_private_tmpfs(tmp_path):
    plan = ToolLaunchPlan(
        argv = ("/bin/true",),
        workdir = str(tmp_path),
        env = {"PATH": "/usr/bin", "TMPDIR": "/var/tmp/somewhere-else"},
    )
    launch = sandbox_linux.prepare(plan)
    try:
        assert launch.argv[launch.argv.index("TMPDIR") + 1] == "/tmp"
    finally:
        launch.cleanup()


def test_the_outer_setsid_preexec_is_preserved(tmp_path):
    ran = []
    launch = sandbox_linux.prepare(_plan(tmp_path, preexec_fn = lambda: ran.append("plan")))
    try:
        assert launch.preexec_fn is not None
        launch.preexec_fn()
        assert ran == ["plan"]
    finally:
        launch.cleanup()


def test_the_plan_policy_fields_survive_preparation(tmp_path):
    launch = sandbox_linux.prepare(_plan(tmp_path, timeout_seconds = 42, terminate_descendants = False))
    try:
        assert launch.timeout_seconds == 42
        assert launch.terminate_descendants is False
        assert launch.backend == sandbox_linux.BACKEND_NAME
    finally:
        launch.cleanup()


def test_cleanup_releases_the_descriptor_and_the_private_identity_directory(tmp_path):
    launch = sandbox_linux.prepare(_plan(tmp_path))
    identity_dir = launch.cleanup_paths[0]
    handle = launch.owned_files[0]
    launch.cleanup()
    assert handle.closed
    assert not os.path.exists(identity_dir)
    assert launch.cleanup_diagnostics == []


def test_the_backend_declares_the_two_things_it_does_not_confine():
    assert "unrestricted_network" in sandbox_linux.LIMITATIONS
    assert "model_cache_writable" in sandbox_linux.LIMITATIONS
    assert isinstance(sandbox_linux.LIMITATIONS, tuple)
    assert sandbox_linux.BACKEND_NAME and sandbox_linux.PROFILE_ID


def test_a_unix_socket_under_the_workdir_is_refused():
    workdir = tempfile.mkdtemp(prefix = "sbx-")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(os.path.join(workdir, "s"))
        with pytest.raises(SandboxUnavailableError, match = "device or IPC node"):
            sandbox_linux._validate_workdir(workdir)
    finally:
        listener.close()
        shutil.rmtree(workdir, ignore_errors = True)


def test_a_fifo_under_the_workdir_is_refused(tmp_path):
    os.mkfifo(str(tmp_path / "pipe"))
    with pytest.raises(SandboxUnavailableError, match = "device or IPC node"):
        sandbox_linux._validate_workdir(str(tmp_path))


def test_a_file_hard_linked_from_outside_the_workdir_is_refused(tmp_path):
    outside, workdir = tmp_path / "outside", tmp_path / "session"
    outside.mkdir()
    workdir.mkdir()
    (outside / "secret").write_text("x")
    os.link(str(outside / "secret"), str(workdir / "innocent"))
    with pytest.raises(SandboxUnavailableError, match = "hard-linked from outside"):
        sandbox_linux._validate_workdir(str(workdir))


def test_a_hard_link_wholly_inside_the_workdir_is_allowed(tmp_path):
    (tmp_path / "a").write_text("x")
    os.link(str(tmp_path / "a"), str(tmp_path / "b"))
    (tmp_path / "nested").mkdir()
    os.link(str(tmp_path / "a"), str(tmp_path / "nested" / "c"))
    assert sandbox_linux._validate_workdir(str(tmp_path)) == (os.path.realpath(tmp_path), ())


def test_a_nested_host_mount_under_the_workdir_is_refused(tmp_path, monkeypatch):
    nested = tmp_path / "mounted"
    nested.mkdir()
    real = os.path.ismount
    monkeypatch.setattr(
        os.path, "ismount", lambda path: os.path.samefile(path, nested) or real(path)
    )
    with pytest.raises(SandboxUnavailableError, match = "nested host mount"):
        sandbox_linux._validate_workdir(str(tmp_path))


def test_the_workdir_itself_being_a_mount_point_is_allowed(tmp_path, monkeypatch):
    real = os.path.ismount
    monkeypatch.setattr(
        os.path, "ismount", lambda path: os.path.samefile(path, tmp_path) or real(path)
    )
    assert sandbox_linux._validate_workdir(str(tmp_path)) == (os.path.realpath(tmp_path), ())


def test_a_workdir_too_large_to_check_still_launches_and_says_it_was_not_checked(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_ENTRIES", 2)
    for name in ("a", "b", "c", "d"):
        (tmp_path / name).write_text("")
    resolved, limitations = sandbox_linux._validate_workdir(str(tmp_path))
    assert resolved == os.path.realpath(tmp_path)
    assert limitations == ("workdir_scan_incomplete",)


def test_an_overrun_does_not_stop_a_real_finding_from_refusing(tmp_path, monkeypatch):
    os.mkfifo(str(tmp_path / "channel"))
    with pytest.raises(SandboxUnavailableError, match = "device or IPC node"):
        sandbox_linux._validate_workdir(str(tmp_path))


def test_an_incomplete_scan_reaches_the_execution_record(tmp_path, monkeypatch):
    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_ENTRIES", 2)
    for name in ("a", "b", "c", "d"):
        (tmp_path / name).write_text("")
    prepared = sandbox_linux.prepare(
        os_sandbox.ToolLaunchPlan(
            argv = (sys.executable, "-c", "pass"),
            workdir = str(tmp_path),
            env = {},
            requested_mode = "required",
        )
    )
    try:
        assert "workdir_scan_incomplete" in prepared.launch_limitations
    finally:
        prepared.cleanup()


def test_a_workdir_that_is_not_a_directory_is_refused(tmp_path):
    with pytest.raises(SandboxUnavailableError, match = "not a safe canonical directory"):
        sandbox_linux._validate_workdir(str(tmp_path / "missing"))
    with pytest.raises(SandboxUnavailableError, match = "not a safe canonical directory"):
        sandbox_linux._validate_workdir("/")


def test_a_symlinked_directory_under_the_workdir_is_not_followed(tmp_path):
    (tmp_path / "escape").symlink_to("/dev")
    assert sandbox_linux._validate_workdir(str(tmp_path)) == (os.path.realpath(tmp_path), ())


def test_every_interpreter_path_this_python_imports_from_is_bound(prepared, tmp_path):
    bound = [
        source
        for flag in ("--ro-bind", "--ro-bind-try", "--bind")
        for source, _ in _pairs(prepared.argv, flag)
    ]
    prefixes = (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
    paths = sysconfig.get_paths()
    needed = {
        paths[key] for key in ("stdlib", "platstdlib", "purelib", "platlib") if paths.get(key)
    }
    needed |= {
        entry
        for entry in sys.path
        if entry and any(sandbox_linux._within(entry, os.path.join(p, "lib")) for p in prefixes)
    }
    for path in sorted(needed):
        if not os.path.exists(path):
            continue
        assert any(sandbox_linux._within(path, root) for root in bound), path


def test_a_venv_at_a_project_root_does_not_expose_the_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "lib").mkdir(parents = True)
    (project / "bin").mkdir()
    (project / "src").mkdir()
    (project / ".env").write_text("TOKEN=secret")
    for name in ("prefix", "exec_prefix"):
        monkeypatch.setattr(sys, name, str(project))
    roots = tuple(path for path in sandbox_linux._SYSTEM_ROOTS if os.path.isdir(path))
    paths = sandbox_linux._runtime_read_paths(os.path.realpath(tmp_path / "wd"), roots)
    assert os.path.realpath(project) not in paths
    assert os.path.realpath(project / "lib") in paths
    assert not any(sandbox_linux._within(str(project / "src"), path) for path in paths)


def test_an_interpreter_path_resolving_to_the_filesystem_root_is_refused(tmp_path, monkeypatch):
    (tmp_path / "lib").symlink_to("/")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    with pytest.raises(SandboxUnavailableError, match = "filesystem root"):
        sandbox_linux._runtime_read_paths(str(tmp_path / "wd"), ("/usr/lib",))


def test_a_system_interpreter_adds_nothing_the_system_roots_do_not_already_cover(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    roots = tuple(path for path in sandbox_linux._SYSTEM_ROOTS if os.path.isdir(path))
    for path in sandbox_linux._runtime_read_paths(str(tmp_path), roots):
        assert path not in ("/", "/usr", "/usr/local")


def test_runtime_paths_inside_the_workdir_are_left_to_the_writable_bind(tmp_path, monkeypatch):
    inner = tmp_path / "venv"
    (inner / "lib").mkdir(parents = True)
    monkeypatch.setattr(sys, "prefix", str(inner))
    roots = tuple(path for path in sandbox_linux._SYSTEM_ROOTS if os.path.isdir(path))
    paths = sandbox_linux._runtime_read_paths(os.path.realpath(tmp_path), roots)
    assert os.path.realpath(inner / "lib") not in paths


def test_the_probe_builds_its_launch_through_the_real_argv_builder(tmp_path):
    if sandbox_linux.shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this host")
    launch = sandbox_linux.probe_argv(str(tmp_path), ("/bin/true", "-x"), {"PATH": "/usr/bin"})
    try:
        reference = sandbox_linux.prepare(_plan(tmp_path, argv = ("/bin/true", "-x")))
        try:

            def normalise(argv, launch):
                fd = str(argv[argv.index("--seccomp") + 1])
                identity = launch.cleanup_paths[0]
                return tuple(item.replace(identity, "<identity>") for item in argv if item != fd)

            assert normalise(launch.argv, launch) == normalise(reference.argv, reference)
        finally:
            reference.cleanup()
    finally:
        launch.cleanup()


_AUDIT_ARCH = sandbox_seccomp._ABIS.get(platform.machine().lower(), (0,))[0]
_KILL, _ALLOW = 0x80000000, 0x7FFF0000
_EPERM, _ENOSYS = sandbox_seccomp._EPERM, sandbox_seccomp._ENOSYS

if _AUDIT_ARCH == 0:
    pytest.skip(f"no seccomp ABI for {platform.machine()}", allow_module_level = True)


def _evaluate(
    instructions,
    *,
    nr,
    arch = _AUDIT_ARCH,
    args = (0,) * 6,
):
    data = struct.pack("=IIQ6Q", nr, arch, 0, *args)
    accumulator, counter = 0, 0
    for _ in range(len(instructions) * 4):
        code, jt, jf, k = instructions[counter]
        counter += 1
        if code == 0x20:
            (accumulator,) = struct.unpack_from("=I", data, k)
        elif code == 0x15:
            counter += jt if accumulator == k else jf
        elif code == 0x45:
            counter += jt if accumulator & k else jf
        elif code == 0x06:
            return k
        else:
            raise AssertionError(f"unexpected BPF opcode {code:#x}")
    raise AssertionError("the program ran off the end without returning")


def _share_cache(monkeypatch, cache):
    """Share every subdirectory of *cache* that exists, as the real resolver would."""
    monkeypatch.setattr(
        sandbox_linux,
        "_model_cache_binds",
        lambda workdir: {
            name: str(cache / name)
            for name in sandbox_linux._MODEL_CACHE_SUBDIRS
            if (cache / name).is_dir()
        },
    )


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A host cache with one populated subdirectory, shared into the jail."""
    directory = tmp_path / "hostcache"
    (directory / "hub").mkdir(parents = True)
    _share_cache(monkeypatch, directory)
    return directory


@pytest.fixture(params = [False, True], ids = ["userns_by_bwrap", "userns_by_seccomp"])
def program(request):
    return sandbox_seccomp.program(platform.machine(), block_userns = request.param)


def _socket_nr():
    return sandbox_seccomp._ABIS[platform.machine().lower()][1]


def test_a_vsock_socket_is_denied_because_no_mount_namespace_can_hide_it(program):
    assert _evaluate(program, nr = _socket_nr(), args = (40, 0, 0, 0, 0, 0)) == _EPERM


def test_ordinary_sockets_stay_allowed_because_the_network_is_not_confined(program):
    for family in (1, 2, 10):
        assert _evaluate(program, nr = _socket_nr(), args = (family, 0, 0, 0, 0, 0)) == _ALLOW


def test_a_vsock_socketpair_is_denied_too(program):
    socketpair_nr = sandbox_seccomp._ABIS[platform.machine().lower()][2]
    assert _evaluate(program, nr = socketpair_nr, args = (40, 0, 0, 0, 0, 0)) == _EPERM
    assert _evaluate(program, nr = socketpair_nr, args = (1, 0, 0, 0, 0, 0)) == _ALLOW


def test_io_uring_is_denied_on_all_three_entry_points(program):
    for number in (425, 426, 427):
        assert _evaluate(program, nr = number) == _EPERM


def test_the_keyring_syscalls_are_denied(program):
    for number in sandbox_seccomp._KEYRING_SYSCALLS[platform.machine().lower()]:
        assert _evaluate(program, nr = number) == _EPERM


def test_an_unrelated_syscall_is_allowed(program):
    assert _evaluate(program, nr = 1) == _ALLOW


def test_a_foreign_abi_is_killed_rather_than_evaluated(program):
    assert _evaluate(program, nr = 1, arch = 0xDEADBEEF) == _KILL


@pytest.mark.skipif(platform.machine().lower() not in ("x86_64", "amd64"), reason = "x32 is x86 only")
def test_the_x32_syscall_table_is_killed(program):
    assert _evaluate(program, nr = 1 | 0x40000000) == _KILL


def test_nested_user_namespaces_are_refused_only_when_bwrap_cannot_do_it():
    clone_nr, unshare_nr, clone3_nr = sandbox_seccomp._USERNS_SYSCALLS[platform.machine().lower()]
    blocking = sandbox_seccomp.program(platform.machine(), block_userns = True)
    assert _evaluate(blocking, nr = unshare_nr, args = (0x10000000, 0, 0, 0, 0, 0)) == _EPERM
    for flags in (0x00000200, 0x00000400, 0x00040000):
        assert _evaluate(blocking, nr = unshare_nr, args = (flags, 0, 0, 0, 0, 0)) == _ALLOW
    assert _evaluate(blocking, nr = clone3_nr) == _ENOSYS
    assert _evaluate(blocking, nr = clone_nr, args = (0x10000000, 0, 0, 0, 0, 0)) == _EPERM
    assert _evaluate(blocking, nr = clone_nr, args = (0x00000100, 0, 0, 0, 0, 0)) == _ALLOW

    permissive = sandbox_seccomp.program(platform.machine(), block_userns = False)
    assert _evaluate(permissive, nr = unshare_nr, args = (0x10000000, 0, 0, 0, 0, 0)) == _ALLOW
    assert _evaluate(permissive, nr = clone3_nr) == _ALLOW


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def _kernel_verdicts(block_userns):
    """Each probe's errno from a fork child, 0 meaning allowed."""
    clone3_nr, unshare_nr = 435, sandbox_seccomp._USERNS_SYSCALLS[platform.machine().lower()][1]
    keyctl_nr = sandbox_seccomp._KEYRING_SYSCALLS[platform.machine().lower()][2]
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - this child never returns into pytest
        os.close(read_fd)
        try:
            instructions = sandbox_seccomp.program(platform.machine(), block_userns = block_userns)
            block = (_SockFilter * len(instructions))(
                *(_SockFilter(*item) for item in instructions)
            )
            libc = ctypes.CDLL(None, use_errno = True)
            libc.syscall.restype = ctypes.c_long
            if libc.prctl(38, 1, 0, 0, 0) != 0:
                os._exit(97)
            fprog = _SockFprog(len(instructions), block)
            if libc.prctl(22, 2, ctypes.byref(fprog), 0, 0) != 0:
                os._exit(98)
            verdicts = {}
            for name, family in (("inet", socket.AF_INET), ("unix", socket.AF_UNIX), ("vsock", 40)):
                try:
                    socket.socket(family, socket.SOCK_STREAM).close()
                    verdicts[name] = 0
                except OSError as error:
                    verdicts[name] = error.errno
            for name, number, argument in (
                ("io_uring", 425, 0),
                ("keyctl", keyctl_nr, 0),
                ("clone3", clone3_nr, None),
                ("unshare_userns", unshare_nr, 0x10000000),
                ("unshare_fs", unshare_nr, 0x00000200),
            ):
                ctypes.set_errno(0)
                result = libc.syscall(number, argument, 0, 0)
                verdicts[name] = 0 if result >= 0 else ctypes.get_errno()
            child = os.fork()
            if child == 0:
                os._exit(0)
            verdicts["fork"] = os.waitpid(child, 0)[1]
            os.write(write_fd, json.dumps(verdicts).encode())
            os._exit(0)
        except BaseException:
            os._exit(99)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as stream:
        payload = stream.read()
    status = os.waitpid(pid, 0)[1]
    assert status == 0, f"the seccomp child exited with status {status:#x}"
    return json.loads(payload)


def test_the_kernel_loads_the_filter_and_denies_the_channels_it_names():
    for block_userns in (False, True):
        verdicts = _kernel_verdicts(block_userns)
        assert verdicts["vsock"] == errno.EPERM
        assert verdicts["io_uring"] == errno.EPERM
        assert verdicts["keyctl"] == errno.EPERM
        assert verdicts["inet"] == 0 and verdicts["unix"] == 0
        assert verdicts["fork"] == 0


def test_the_kernel_refuses_nested_user_namespaces_only_in_the_fallback_filter():
    assert _kernel_verdicts(block_userns = True)["clone3"] == errno.ENOSYS
    blocking = _kernel_verdicts(block_userns = True)
    assert blocking["unshare_userns"] == errno.EPERM
    assert blocking["unshare_fs"] == 0
    assert _kernel_verdicts(block_userns = False)["clone3"] != errno.ENOSYS


def test_an_unreviewed_architecture_is_refused_rather_than_left_unfiltered():
    with pytest.raises(RuntimeError, match = "little-endian"):
        sandbox_seccomp.program("riscv64")


def test_the_filter_file_holds_exactly_the_program_and_is_rewound():
    stream = sandbox_seccomp.filter_file(block_userns = True)
    try:
        assert stream.tell() == 0
        payload = stream.read()
        assert payload == sandbox_seccomp.program_bytes(block_userns = True)
        assert len(payload) % 8 == 0
        instructions = sandbox_seccomp.program(platform.machine(), block_userns = True)
        assert len(payload) // 8 == len(instructions)
        assert struct.unpack_from("=HBBI", payload, 0) == instructions[0]
    finally:
        stream.close()


def test_a_runtime_path_symlinked_out_of_the_workdir_is_not_bound(tmp_path, monkeypatch):
    workdir = tmp_path / "session"
    workdir.mkdir()
    secret = tmp_path / "secrets"
    secret.mkdir()
    (secret / "id_rsa").write_text("PRIVATE KEY")
    venv = workdir / "venv"
    venv.mkdir()
    (venv / "lib").symlink_to(secret)
    monkeypatch.setattr(sys, "prefix", str(venv))

    paths = sandbox_linux._runtime_read_paths(str(workdir), ("/usr/lib",))
    assert str(secret) not in paths
    assert not any(sandbox_linux._within(str(secret), path) for path in paths)


def test_a_symlinked_cache_ancestor_is_refused_rather_than_written_through(tmp_path, cache):
    """Refused rather than unlinked: symlinking a cache leaf at another volume is a legitimate layout."""
    workdir = tmp_path / "session"
    workdir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workdir / ".cache").symlink_to(outside)

    with pytest.raises(SandboxUnavailableError, match = "not a plain directory"):
        sandbox_linux.prepare(_plan(workdir))
    assert (workdir / ".cache").is_symlink()
    assert sorted(entry.name for entry in outside.iterdir()) == []


def test_the_probe_launch_sets_no_new_privs_like_a_real_one(tmp_path):
    from core.inference import sandbox_probe

    assert sandbox_probe._PR_SET_NO_NEW_PRIVS == 38
    source = inspect.getsource(sandbox_probe._run_probe)
    assert "preexec_fn = _no_new_privs" in source
    assert not [
        node
        for node in ast.walk(ast.parse(inspect.getsource(sandbox_probe._no_new_privs)))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def test_the_pip_targets_script_directory_is_last_on_path(prepared):
    argv = prepared.argv
    entries = argv[argv.index("PATH") + 1].split(os.pathsep)
    packages = os.path.join(prepared.workdir, sandbox_linux.SESSION_PACKAGES_RELPATH)
    assert entries[-1] == os.path.join(packages, "bin")
    assert "/usr/bin" in entries[:-1]


def test_the_compiler_headers_a_source_build_needs_are_readable(prepared):
    sources = [source for source, _ in _pairs(prepared.argv, "--ro-bind-try")]
    assert "/usr/include" in sources
    assert "/usr/local/include" in sources
    include = sysconfig.get_paths().get("include")
    if include and os.path.isdir(include):
        bound = [source for source, _ in _pairs(prepared.argv, "--ro-bind")]
        assert any(sandbox_linux._within(include, path) for path in (*sources, *bound)), include


def test_a_runtime_prefix_contributes_its_git_helpers(tmp_path, monkeypatch):
    prefix = tmp_path / "conda"
    for name in ("bin", "libexec", "lib"):
        (prefix / name).mkdir(parents = True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    paths = sandbox_linux._runtime_read_paths(str(tmp_path / "session"), ("/usr/lib",))
    assert str(prefix / "libexec") in paths
    assert str(prefix) not in paths


def test_the_launch_pre_exec_scopes_abstract_sockets(tmp_path):
    if not sandbox_landlock.abstract_scope_supported():
        pytest.skip("this kernel predates the Landlock abstract-socket scope")
    name = b"\0unsloth-abstract-probe-" + os.urandom(6).hex().encode()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(name)
    listener.listen(1)
    read_fd, write_fd = os.pipe()
    try:
        child = os.fork()
        if child == 0:  # pragma: no cover - runs in the forked child
            try:
                os.close(read_fd)
                sandbox_landlock.with_abstract_scope(None)()
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                probe.settimeout(2)
                try:
                    probe.connect(name)
                    os.write(write_fd, b"connected")
                except OSError as error:
                    os.write(write_fd, str(error.errno).encode())
            finally:
                os._exit(0)
        os.close(write_fd)
        write_fd = -1
        with os.fdopen(read_fd, "rb") as stream:
            verdict = stream.read()
        read_fd = -1
        os.waitpid(child, 0)
        assert verdict == str(errno.EPERM).encode(), verdict
        control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        control.settimeout(2)
        control.connect(name)
        control.close()
    finally:
        for fd in (read_fd, write_fd):
            if fd >= 0:
                os.close(fd)
        listener.close()


def test_the_backend_says_so_when_the_kernel_cannot_scope_them(monkeypatch):
    supported = sandbox_landlock.abstract_scope_supported()
    assert ("host_abstract_sockets_reachable" in sandbox_linux.LIMITATIONS) is not supported


def test_a_landlock_probe_child_that_never_answers_does_not_hang_the_server(monkeypatch):
    if not sandbox_landlock._abi_reports_scope():
        pytest.skip("no Landlock ABI 6 here, so the fork under test never happens")

    def never_answers():
        time.sleep(30)

    monkeypatch.setattr(sandbox_landlock, "apply_abstract_scope", never_answers)
    monkeypatch.setattr(sandbox_landlock, "_PROBE_TIMEOUT_SECONDS", 1.0)
    sandbox_landlock.abstract_scope_supported.cache_clear()
    try:
        started = time.monotonic()
        assert sandbox_landlock.abstract_scope_supported() is False
        assert time.monotonic() - started < 10, "the deadline did not fire"
    finally:
        sandbox_landlock.abstract_scope_supported.cache_clear()


def test_the_plan_pre_exec_still_runs_before_the_scope():
    ran = []
    composed = sandbox_landlock.with_abstract_scope(lambda: ran.append("plan"))
    composed()
    assert ran == ["plan"]


def test_a_standalone_interpreter_in_the_workdir_is_re_bound_read_only(tmp_path, monkeypatch):
    workdir = tmp_path / "session"
    workdir.mkdir()
    executable = workdir / "python"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    for attribute in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix"):
        monkeypatch.setattr(sys, attribute, str(workdir))
    monkeypatch.setattr(sys, "executable", str(executable))

    assert sandbox_linux._runtime_paths_under(str(workdir)) == (str(executable),)
    launch = sandbox_linux.prepare(_plan(workdir))
    try:
        argv = list(launch.argv)
        read_only = [argv[i + 1] for i, item in enumerate(argv) if item == "--ro-bind"]
        assert str(executable) in read_only, read_only
        assert argv.index("--ro-bind", argv.index("--bind")) > argv.index("--bind")
    finally:
        launch.cleanup()


def test_an_interpreter_in_a_home_directory_does_not_bind_the_home(tmp_path, monkeypatch):
    home = tmp_path / "alice"
    (home / ".ssh").mkdir(parents = True)
    (home / ".ssh" / "id_rsa").write_text("SECRET", encoding = "utf-8")
    executable = home / "python"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(executable))

    paths = sandbox_linux._runtime_read_paths(str(tmp_path / "session"), ("/usr/lib",))
    assert str(home) not in paths, paths
    assert not any(_within_for_test(p, str(home / ".ssh")) for p in paths), paths
    assert str(executable) in paths, paths


def _within_for_test(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def test_a_runtime_origin_is_excluded_under_the_workdir_alias_too(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    secret = tmp_path / "secrets"
    secret.mkdir()
    (secret / "id_rsa").write_text("SECRET", encoding = "utf-8")
    venv = real / "venv"
    (venv / "bin").mkdir(parents = True)
    (venv / "lib").symlink_to(secret)
    for attribute in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix"):
        monkeypatch.setattr(sys, attribute, str(alias / "venv"))

    paths = sandbox_linux._runtime_read_paths(os.path.realpath(real), ("/usr/lib",), str(alias))
    assert not any(os.path.realpath(p) == str(secret) for p in paths), paths
    launch = sandbox_linux.prepare(_plan(alias))
    try:
        argv = list(launch.argv)
        bound = [
            argv[i + 1] for i, item in enumerate(argv) if item in ("--ro-bind", "--ro-bind-try")
        ]
        assert not any(os.path.realpath(p) == str(secret) for p in bound), bound
    finally:
        launch.cleanup()


def test_an_editable_source_inside_the_workdir_is_re_bound_read_only(tmp_path, monkeypatch):
    workdir = tmp_path / "session"
    package = workdir / "mypkg"
    package.mkdir(parents = True)
    (package / "__init__.py").write_text("", encoding = "utf-8")
    monkeypatch.setattr(sandbox_linux, "editable_source_roots", lambda: (str(package),))

    assert str(package) in sandbox_linux._runtime_paths_under(str(workdir))
    assert str(package) not in sandbox_linux._runtime_read_paths(str(workdir), ("/usr/lib",))


def test_the_interpreter_spelling_the_launch_execs_is_bound(tmp_path, monkeypatch):
    target_bin = tmp_path / "opt" / "py" / "bin"
    target_bin.mkdir(parents = True)
    target = target_bin / "python3.13"
    target.write_bytes(b"#!/bin/sh\nexit 0\n")
    target.chmod(0o755)
    user_bin = tmp_path / "home" / "bin"
    user_bin.mkdir(parents = True)
    link = user_bin / "python"
    link.symlink_to(target)
    monkeypatch.setattr(sys, "executable", str(link))
    for attribute in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix"):
        monkeypatch.setattr(sys, attribute, str(tmp_path / "opt" / "py"))

    paths = sandbox_linux._runtime_read_paths(str(tmp_path / "session"), ("/usr/lib",))
    assert str(link) in paths, paths
    assert str(target) in paths or str(target_bin) in paths, paths
    assert str(user_bin) not in paths, paths


def test_a_wedged_cache_mount_drops_the_cache_instead_of_hanging_the_launch(tmp_path, monkeypatch):
    cache = tmp_path / "hostcache"
    (cache / "hub").mkdir(parents = True)
    _share_cache_paths(monkeypatch, cache)

    def wedged(path):
        time.sleep(30)
        raise AssertionError("the caller should not have waited for this")

    monkeypatch.setattr(
        sandbox_linux, "_inspect_cache_component", lambda name, path, witness = None: wedged(path)
    )
    monkeypatch.setattr(sandbox_linux, "_CACHE_INSPECT_SECONDS", 1.0)
    started = time.monotonic()
    binds = sandbox_linux._model_cache_binds(str(tmp_path / "session"))
    assert time.monotonic() - started < 10, "the deadline did not fire"
    assert binds == {}, binds


def _share_cache_paths(monkeypatch, cache):
    import types

    class _Paths:
        cache_home = str(cache)
        hub_cache = str(cache / "hub")
        xet_cache = str(cache / "xet")

    module = types.ModuleType("utils.hf_cache_settings")
    module.get_hf_cache_paths = lambda: _Paths()
    monkeypatch.setitem(sys.modules, "utils.hf_cache_settings", module)


def test_a_symlinked_runtime_entry_inside_the_workdir_is_protected_by_name(tmp_path, monkeypatch):
    workdir = tmp_path / "session"
    target = workdir / "real_pkg"
    target.mkdir(parents = True)
    (target / "__init__.py").write_text("", encoding = "utf-8")
    entry = workdir / "mypkg"
    entry.symlink_to(target)
    monkeypatch.setattr(sandbox_linux, "editable_source_roots", lambda: (str(entry),))

    protected = sandbox_linux._runtime_paths_under(str(workdir))
    assert str(entry) in protected, protected
    assert str(target) in protected, protected


def test_a_wedged_cache_path_is_not_re_scanned_by_every_later_launch(tmp_path, monkeypatch):
    cache = tmp_path / "hostcache"
    (cache / "hub").mkdir(parents = True)
    _share_cache_paths(monkeypatch, cache)
    started: list[str] = []

    def wedged(
        name,
        path,
        witness = None,
    ):
        started.append(path)
        time.sleep(30)

    monkeypatch.setattr(sandbox_linux, "_inspect_cache_component", wedged)
    monkeypatch.setattr(sandbox_linux, "_CACHE_INSPECT_SECONDS", 0.5)
    monkeypatch.setattr(sandbox_linux, "_cache_scan_pending", {})
    session = str(tmp_path / "session")
    assert sandbox_linux._model_cache_binds(session) == {}
    first = len(started)
    assert first > 0
    assert sandbox_linux._model_cache_binds(session) == {}
    assert len(started) == first, "a second launch started another worker on the same path"


def test_revalidating_a_cached_verdict_on_a_wedged_mount_is_bounded_too(monkeypatch, tmp_path):
    """A memo hit re-stats every directory the walk saw, which blocks on a stalled NFS/FUSE cache just like the walk."""
    component = tmp_path / "hub"
    component.mkdir()
    monkeypatch.setattr(sandbox_linux, "_cache_scan_pending", {})
    sandbox_linux.reset_cache_verdicts()
    assert sandbox_linux._cache_hazard_within_deadline("hub", str(component)) is None

    release = threading.Event()
    monkeypatch.setattr(
        sandbox_linux, "directory_witness_matches", lambda witness: release.wait(30)
    )
    monkeypatch.setattr(sandbox_linux, "_CACHE_INSPECT_SECONDS", 0.5)
    start = time.monotonic()
    try:
        hazard = sandbox_linux._cache_hazard_within_deadline("hub", str(component))
    finally:
        release.set()

    assert time.monotonic() - start < 10
    assert hazard is not None and "wedged" in hazard, hazard


def test_a_runtime_entry_whose_target_leaves_the_workdir_gets_no_rule(tmp_path, monkeypatch):
    workdir = tmp_path / "session"
    workdir.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    (private / "id_rsa").write_text("SECRET", encoding = "utf-8")
    (workdir / "venv").mkdir()
    (workdir / "venv" / "lib").symlink_to(private)
    monkeypatch.setattr(sys, "prefix", str(workdir / "venv"))
    monkeypatch.setattr(sys, "exec_prefix", str(workdir / "venv"))

    protected = sandbox_linux._runtime_paths_under(str(workdir))
    assert str(workdir / "venv" / "lib") not in protected, protected
    assert not any(os.path.realpath(p) == str(private) for p in protected), protected


def test_an_interpreter_symlinked_out_of_the_workdir_fails_the_call(tmp_path, monkeypatch):
    workdir = tmp_path / "session"
    workdir.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    real = outside / "python"
    real.write_bytes(b"#!/bin/sh\nexit 0\n")
    real.chmod(0o755)
    link = workdir / "python"
    link.symlink_to(real)
    monkeypatch.setattr(sys, "executable", str(link))

    with pytest.raises(WorkdirUnsafeError, match = "Python that runs Studio"):
        sandbox_linux._runtime_paths_under(str(workdir))


def test_concurrent_launches_start_one_cache_worker_and_never_raise(tmp_path, monkeypatch):
    cache = tmp_path / "hostcache"
    (cache / "hub").mkdir(parents = True)
    _share_cache_paths(monkeypatch, cache)
    started: list[str] = []
    gate = threading.Event()

    def wedged(
        name,
        path,
        witness = None,
    ):
        started.append(path)
        gate.wait(30)

    monkeypatch.setattr(sandbox_linux, "_inspect_cache_component", wedged)
    monkeypatch.setattr(sandbox_linux, "_CACHE_INSPECT_SECONDS", 0.5)
    monkeypatch.setattr(sandbox_linux, "_cache_scan_pending", {})
    session = str(tmp_path / "session")
    errors: list[BaseException] = []

    def launch() -> None:
        try:
            sandbox_linux._model_cache_binds(session)
        except BaseException as exc:  # noqa: BLE001 - the point of the test
            errors.append(exc)

    threads = [threading.Thread(target = launch) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    gate.set()
    assert not errors, errors
    assert started, "the inspection never ran"
    assert len(started) == len(set(started)), f"a path was scanned twice: {sorted(started)}"


def test_a_runtime_prefix_contributes_its_certificate_store(tmp_path, monkeypatch):
    prefix = tmp_path / "conda"
    for name in ("bin", "ssl", "lib"):
        (prefix / name).mkdir(parents = True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    paths = sandbox_linux._runtime_read_paths(str(tmp_path / "session"), ("/usr/lib",))
    assert str(prefix / "ssl") in paths


def test_a_workdir_reached_through_a_symlink_is_bound_at_the_spelling_the_caller_used(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    launch = sandbox_linux.prepare(_plan(link))
    try:
        binds = _pairs(launch.argv, "--bind")
        assert (str(real), str(link)) in binds
        assert (str(real), str(real)) in binds
        assert launch.argv[launch.argv.index("--chdir") + 1] == str(link)
        assert launch.argv[launch.argv.index("HOME") + 1] == str(link)
    finally:
        launch.cleanup()


def test_a_cache_leaf_left_behind_as_a_file_is_refused_at_preparation(tmp_path, cache):
    workdir = tmp_path / "session"
    (workdir / ".cache" / "huggingface").mkdir(parents = True)
    (workdir / ".cache" / "huggingface" / "hub").write_text("not a directory")

    with pytest.raises(SandboxUnavailableError, match = "not a plain directory"):
        sandbox_linux.prepare(_plan(workdir))
    assert (workdir / ".cache" / "huggingface" / "hub").read_text() == "not a directory"


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason = "root reads and writes regardless of the mode bits, so the premise is void",
)
def test_an_unreadable_directory_is_refused(tmp_path):
    """A mode-000 directory hides a link out from the scan, and the process that owns it can chmod it back."""
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "inside").write_text("x")
    locked.chmod(0o000)
    try:
        with pytest.raises(SandboxUnavailableError, match = "cannot be fully inspected"):
            sandbox_linux._validate_workdir(str(tmp_path))
    finally:
        locked.chmod(0o700)


def test_the_cache_studio_actually_uses_is_the_one_shared(tmp_path, monkeypatch):
    """HF_HUB_CACHE=/mnt/models is NOT /mnt/models/hub, so each component is bound where it actually is."""
    hub = tmp_path / "mnt" / "models"
    xet = tmp_path / "fast" / "xet"
    home = tmp_path / "cachehome"
    for path in (hub, xet, home / "datasets", home / "assets"):
        path.mkdir(parents = True)
    import utils.hf_cache_settings as cache_settings

    monkeypatch.setattr(
        cache_settings,
        "get_hf_cache_paths",
        lambda: cache_settings.HuggingFaceCachePaths(
            cache_home = home, hub_cache = hub, xet_cache = xet, source = "studio"
        ),
    )
    binds = sandbox_linux._model_cache_binds(str(tmp_path / "session"))
    assert binds["hub"] == str(hub)
    assert binds["xet"] == str(xet)
    assert binds["datasets"] == str(home / "datasets")
    assert "hub" not in sandbox_linux._model_cache_binds(str(hub.parent))


def _real_cache(monkeypatch, home):
    """Point the settings layer at *home* and let the REAL _model_cache_binds run."""
    import types

    paths = types.SimpleNamespace(cache_home = home, hub_cache = home / "hub", xet_cache = home / "xet")
    module = types.ModuleType("utils.hf_cache_settings")
    module.get_hf_cache_paths = lambda: paths
    monkeypatch.setitem(sys.modules, "utils.hf_cache_settings", module)


def test_a_cache_component_holding_an_ipc_node_is_not_shared(tmp_path, monkeypatch):
    host = tmp_path / "hostcache"
    (host / "hub").mkdir(parents = True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    monkeypatch.chdir(host / "hub")
    sock.bind("leftover.sock")
    try:
        _real_cache(monkeypatch, host)
        assert "hub" not in sandbox_linux._model_cache_binds(str(tmp_path / "session"))
    finally:
        sock.close()


def test_a_cache_component_holding_an_external_hard_link_is_not_shared(tmp_path, monkeypatch):
    """Same inode under two names, one of them outside the cache."""
    host = tmp_path / "hostcache"
    (host / "hub").mkdir(parents = True)
    outside = tmp_path / "private.txt"
    outside.write_text("secret")
    os.link(outside, host / "hub" / "innocent.bin")
    _real_cache(monkeypatch, host)
    assert "hub" not in sandbox_linux._model_cache_binds(str(tmp_path / "session"))


def test_a_clean_cache_component_is_still_shared(tmp_path, monkeypatch):
    """The negative control: the check above must not simply drop everything."""
    host = tmp_path / "hostcache"
    (host / "hub" / "models--x").mkdir(parents = True)
    (host / "hub" / "models--x" / "weights.bin").write_text("w")
    _real_cache(monkeypatch, host)
    assert sandbox_linux._model_cache_binds(str(tmp_path / "session"))["hub"] == str(host / "hub")


def test_a_hazardous_cache_drops_the_component_rather_than_failing_the_launch(
    tmp_path, monkeypatch
):
    host = tmp_path / "hostcache"
    (host / "hub").mkdir(parents = True)
    os.mkfifo(host / "hub" / "pipe")
    _real_cache(monkeypatch, host)
    workdir = tmp_path / "session"
    workdir.mkdir()
    launch = sandbox_linux.prepare(_plan(workdir))
    try:
        assert "HF_HOME" not in launch.argv
    finally:
        launch.cleanup()


def test_a_trusted_system_gitconfig_is_bound_and_an_untrusted_one_is_not(tmp_path, monkeypatch):
    good = tmp_path / "gitconfig"
    good.write_text("[http]\n")
    link = tmp_path / "linked"
    link.symlink_to(good)
    assert sandbox_linux._trusted_system_file(str(link)) is False, "a symlink is followed"
    assert sandbox_linux._trusted_system_file(str(tmp_path / "absent")) is False
    os.chmod(good, 0o666)
    assert sandbox_linux._trusted_system_file(str(good)) is False, "world-writable accepted"


def test_a_runtime_under_the_workdir_is_re_bound_read_only(tmp_path, monkeypatch):
    workdir = tmp_path / "session"
    venv = workdir / "venv"
    (venv / "lib").mkdir(parents = True)
    (venv / "bin").mkdir()
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "exec_prefix", str(venv))

    launch = sandbox_linux.prepare(_plan(workdir))
    try:
        argv = launch.argv
        writable = argv.index("--bind")
        ro_after = [
            source
            for index, source in enumerate(argv)
            if index > writable and argv[index - 1] == "--ro-bind"
        ]
        assert str(venv / "lib") in ro_after, "the runtime stayed writable"
        assert str(venv / "bin") in ro_after
    finally:
        launch.cleanup()


def test_a_runtime_symlinked_out_of_the_workdir_is_not_re_bound(tmp_path, monkeypatch):
    """The negative control for the rule above."""
    workdir = tmp_path / "session"
    (workdir / "venv").mkdir(parents = True)
    secret = tmp_path / "home" / ".ssh"
    secret.mkdir(parents = True)
    (workdir / "venv" / "lib").symlink_to(secret)
    monkeypatch.setattr(sys, "prefix", str(workdir / "venv"))

    launch = sandbox_linux.prepare(_plan(workdir))
    try:
        for flag in ("--bind", "--ro-bind", "--ro-bind-try"):
            for source, _ in _pairs(launch.argv, flag):
                assert not sandbox_linux._within(str(secret), source), source
    finally:
        launch.cleanup()


def test_a_nested_bind_mount_in_the_cache_is_caught_by_the_mount_table(tmp_path, monkeypatch):
    host = tmp_path / "hostcache"
    (host / "hub" / "nested").mkdir(parents = True)
    _real_cache(monkeypatch, host)
    monkeypatch.setattr(
        sandbox_linux, "_host_mount_points", lambda: (str(host / "hub" / "nested"),)
    )
    assert "hub" not in sandbox_linux._model_cache_binds(str(tmp_path / "session"))


def test_a_nested_bind_mount_is_caught_through_a_symlinked_cache(tmp_path, monkeypatch):
    """The mount table lists canonical paths, so a cache reached through a symlinked ~/.cache must be compared in its resolved form."""
    real = tmp_path / "volume" / "huggingface"
    (real / "hub" / "nested").mkdir(parents = True)
    (tmp_path / "cache-link").symlink_to(tmp_path / "volume")
    _real_cache(monkeypatch, tmp_path / "cache-link" / "huggingface")
    monkeypatch.setattr(
        sandbox_linux, "_host_mount_points", lambda: (str(real / "hub" / "nested"),)
    )
    assert "hub" not in sandbox_linux._model_cache_binds(str(tmp_path / "session"))


def _fake_editable(
    tmp_path,
    monkeypatch,
    source: str,
    top_level: str | None = None,
):
    """A dist-info recording an editable install, the way an installer writes it."""
    site_dir = tmp_path / "sitepkgs"
    info = site_dir / "demo-1.0.dist-info"
    info.mkdir(parents = True)
    (info / "METADATA").write_text("Name: demo\nVersion: 1.0\n", encoding = "utf-8")
    (info / "RECORD").write_text("", encoding = "utf-8")
    if top_level is not None:
        (info / "top_level.txt").write_text(top_level + "\n", encoding = "utf-8")
    (info / "direct_url.json").write_text(
        json.dumps({"url": f"file://{source}", "dir_info": {"editable": True}}),
        encoding = "utf-8",
    )
    monkeypatch.syspath_prepend(str(site_dir))
    os_sandbox.editable_source_roots.cache_clear()
    return site_dir


def test_an_editable_installs_source_root_is_readable(tmp_path, monkeypatch):
    source = tmp_path / "checkout"
    package = source / "demo"
    package.mkdir(parents = True)
    (package / "__init__.py").write_text("", encoding = "utf-8")
    (source / ".env").write_text("AWS_SECRET_ACCESS_KEY=real\n", encoding = "utf-8")
    (source / "fixtures").mkdir()
    _fake_editable(tmp_path, monkeypatch, str(source))
    try:
        granted = os_sandbox.editable_source_roots()
        assert str(package) in granted, granted
        assert str(source) not in granted, granted
        assert not any("fixtures" in path or ".env" in path for path in granted), granted
        roots = tuple(p for p in sandbox_linux._SYSTEM_ROOTS if os.path.isdir(p))
        read = sandbox_linux._runtime_read_paths(str(tmp_path / "wd"), roots)
        assert str(package) in read
        assert str(source) not in read
    finally:
        os_sandbox.editable_source_roots.cache_clear()


def test_a_guessed_editable_import_root_gives_up_what_it_cannot_confirm(tmp_path, monkeypatch):
    source = tmp_path / "checkout"
    package = source / "demo"
    package.mkdir(parents = True)
    (package / "__init__.py").write_text("", encoding = "utf-8")
    (source / "deploy.py").write_text("AWS_SECRET_ACCESS_KEY = 'real'\n", encoding = "utf-8")
    (source / "conftest.py").write_text("", encoding = "utf-8")
    tests_pkg = source / "tests"
    tests_pkg.mkdir()
    (tests_pkg / "__init__.py").write_text("", encoding = "utf-8")
    (tests_pkg / "fixtures").mkdir()
    (tests_pkg / "fixtures" / "id_rsa").write_text("-----BEGIN-----", encoding = "utf-8")
    _fake_editable(tmp_path, monkeypatch, str(source), top_level = "demo")
    try:
        granted = os_sandbox.editable_source_roots()
        assert granted == (str(package),), granted
    finally:
        os_sandbox.editable_source_roots.cache_clear()


def test_a_declared_package_symlinked_out_of_the_checkout_is_refused(tmp_path, monkeypatch):
    source = tmp_path / "checkout"
    source.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    (private / "id_rsa").write_text("SECRET", encoding = "utf-8")
    (private / "__init__.py").write_text("", encoding = "utf-8")
    (source / "demo").symlink_to(private)
    _fake_editable(tmp_path, monkeypatch, str(source), top_level = "demo")
    try:
        granted = os_sandbox.editable_source_roots()
        assert granted == (), granted
    finally:
        os_sandbox.editable_source_roots.cache_clear()


def test_an_editable_root_at_the_filesystem_root_is_refused(tmp_path, monkeypatch):
    """The negative control."""
    _fake_editable(tmp_path, monkeypatch, "/usr")
    try:
        assert os_sandbox.editable_source_roots() == ()
    finally:
        os_sandbox.editable_source_roots.cache_clear()


def test_a_runtime_under_a_symlinked_workdir_is_read_only_through_both_spellings(
    tmp_path, monkeypatch
):
    real = tmp_path / "real"
    venv = real / "venv"
    (venv / "lib").mkdir(parents = True)
    (venv / "bin").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "exec_prefix", str(venv))

    launch = sandbox_linux.prepare(_plan(alias))
    try:
        argv = list(launch.argv)
        binds = [i for i, item in enumerate(argv) if item == "--bind"]
        writable = [argv[i + 2] for i in binds]
        assert str(alias) in writable and str(real) in writable, writable
        last_bind = max(binds)
        for leg in ("lib", "bin"):
            for spelling in (real / "venv" / leg, alias / "venv" / leg):
                landed = [
                    i
                    for i in range(len(argv))
                    if argv[i] == "--ro-bind" and argv[i + 2] == str(spelling)
                ]
                assert landed, f"{spelling} is not re-bound read-only"
                assert min(landed) > last_bind, f"{spelling} is bound before the last --bind"
    finally:
        launch.cleanup()


def test_a_runtime_is_protected_when_sys_prefix_carries_the_workdir_alias(tmp_path, monkeypatch):
    real = tmp_path / "real"
    (real / "venv" / "lib").mkdir(parents = True)
    (real / "venv" / "bin").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    monkeypatch.setattr(sys, "prefix", str(alias / "venv"))
    monkeypatch.setattr(sys, "exec_prefix", str(alias / "venv"))

    under = sandbox_linux._runtime_paths_under(str(real))
    assert str(real / "venv" / "lib") in under, under
    assert str(real / "venv" / "bin") in under, under

    launch = sandbox_linux.prepare(_plan(alias))
    try:
        argv = list(launch.argv)
        last_bind = max(i for i, item in enumerate(argv) if item == "--bind")
        for leg in ("lib", "bin"):
            for spelling in (real / "venv" / leg, alias / "venv" / leg):
                landed = [
                    i
                    for i in range(len(argv))
                    if argv[i] == "--ro-bind" and argv[i + 2] == str(spelling)
                ]
                assert landed and max(landed) > last_bind, f"{spelling} unprotected"
    finally:
        launch.cleanup()


def test_an_editable_namespace_package_is_granted_without_an_init(tmp_path, monkeypatch):
    source = tmp_path / "checkout"
    namespace = source / "acme"
    (namespace / "widget").mkdir(parents = True)
    (namespace / "widget" / "__init__.py").write_text("", encoding = "utf-8")
    (source / ".env").write_text("AWS_SECRET_ACCESS_KEY=real\n", encoding = "utf-8")
    (source / "fixtures").mkdir()
    _fake_editable(tmp_path, monkeypatch, str(source), top_level = "acme")
    try:
        granted = os_sandbox.editable_source_roots()
        assert str(namespace) in granted, granted
        assert not any("fixtures" in path or ".env" in path for path in granted), granted
        assert str(source) not in granted, granted
    finally:
        os_sandbox.editable_source_roots.cache_clear()


def test_the_cache_verdict_is_memoized_between_launches(tmp_path, monkeypatch):
    """The walk is O(entries) and was re-paid on every launch."""
    calls = []
    real = sandbox_linux._inspect_cache_component
    monkeypatch.setattr(
        sandbox_linux,
        "_inspect_cache_component",
        lambda name, path, witness = None: (calls.append(path), real(name, path, witness))[1],
    )
    sandbox_linux.reset_cache_verdicts()
    component = tmp_path / "hub"
    component.mkdir()
    (component / "a.bin").write_text("")

    first = sandbox_linux._cache_hazard_within_deadline("hub", str(component))
    walked_once = len(calls)
    second = sandbox_linux._cache_hazard_within_deadline("hub", str(component))

    assert second == first
    assert len(calls) == walked_once, "the second launch re-walked the cache"


def test_a_changed_cache_is_re_inspected_rather_than_trusted(tmp_path, monkeypatch):
    """The memo key is the component root's identity and mtime, so a change made through the root must invalidate it."""
    calls = []
    real = sandbox_linux._inspect_cache_component
    monkeypatch.setattr(
        sandbox_linux,
        "_inspect_cache_component",
        lambda name, path, witness = None: (calls.append(path), real(name, path, witness))[1],
    )
    sandbox_linux.reset_cache_verdicts()
    component = tmp_path / "hub"
    component.mkdir()
    sandbox_linux._cache_hazard_within_deadline("hub", str(component))
    before = len(calls)

    os.utime(component, (0, 0))
    (component / "models--org--new").mkdir()
    sandbox_linux._cache_hazard_within_deadline("hub", str(component))

    assert len(calls) > before, "a changed cache reused its old verdict"


def test_a_failed_launch_drops_every_cache_verdict(tmp_path, monkeypatch):
    """tools.py calls this when a launch dies, for the same reason it drops the capability probe: a failed launch is the one signal that something the planner believed about this host has changed."""
    calls = []
    real = sandbox_linux._inspect_cache_component
    monkeypatch.setattr(
        sandbox_linux,
        "_inspect_cache_component",
        lambda name, path, witness = None: (calls.append(path), real(name, path, witness))[1],
    )
    sandbox_linux.reset_cache_verdicts()
    component = tmp_path / "hub"
    component.mkdir()
    sandbox_linux._cache_hazard_within_deadline("hub", str(component))
    before = len(calls)

    sandbox_linux.reset_cache_verdicts()
    sandbox_linux._cache_hazard_within_deadline("hub", str(component))

    assert len(calls) > before, "reset_cache_verdicts did not invalidate the memo"


def test_the_studio_state_directory_is_never_a_system_bind(monkeypatch, tmp_path):
    """/opt is a system root and the Docker layout puts Studio's state in it."""
    from core.inference import sandbox_linux

    opt = tmp_path / "opt"
    state = opt / "unsloth-studio"
    (state / "auth").mkdir(parents = True)
    (state / "auth" / "auth.db").write_text("secret")
    toolchain = opt / "some-toolchain"
    toolchain.mkdir()

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(state))

    kept = sandbox_linux._without_studio_state((str(opt),))

    assert str(state) not in kept
    assert not any(
        sandbox_linux._within(str(state), path) for path in kept
    ), "a bind source still contains the Studio auth database"
    assert str(toolchain) in kept, "unrelated /opt software stopped being readable"


def test_a_system_root_without_studio_state_is_still_bound_whole(monkeypatch, tmp_path):
    """The descent must only happen where it is needed, or every launch pays a listdir of /usr and binds hundreds of paths."""
    from core.inference import sandbox_linux

    opt = tmp_path / "opt"
    (opt / "some-toolchain").mkdir(parents = True)
    elsewhere = tmp_path / "home" / "studio"
    elsewhere.mkdir(parents = True)

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(elsewhere))

    assert sandbox_linux._without_studio_state((str(opt),)) == (str(opt),)


def test_a_deeply_nested_studio_state_root_is_refused_not_restored(monkeypatch, tmp_path):
    """The descent is bounded, and the bound must fail CLOSED."""
    from core.inference import sandbox_linux

    opt = tmp_path / "opt"
    state = opt / "a" / "b" / "c" / "d" / "studio"
    (state / "auth").mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(state))

    kept = sandbox_linux._without_studio_state((str(opt),), depth = 0)

    assert kept == (), "an ancestor of the Studio auth database was bound anyway"


def test_a_symlinked_sibling_is_not_promoted_to_a_bind_source(monkeypatch, tmp_path):
    """Splitting a system root must not mount what the whole-root bind did not."""
    from core.inference import sandbox_linux

    opt = tmp_path / "opt"
    state = opt / "unsloth-studio"
    (state / "auth").mkdir(parents = True)
    private = tmp_path / "home" / "operator" / "private"
    private.mkdir(parents = True)
    (opt / "private").symlink_to(private)
    (opt / "toolchain").mkdir()
    (opt / "inside-link").symlink_to(opt / "toolchain")

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(state))

    kept = sandbox_linux._without_studio_state((str(opt),))

    assert str(opt / "private") not in kept, "a symlink out of the root became a bind source"
    assert str(opt / "toolchain") in kept
    assert str(opt / "inside-link") in kept


def test_a_cache_holding_studio_state_is_not_shared_writable(monkeypatch, tmp_path):
    """HF_HUB_CACHE at or above the Studio root would bind auth/auth.db in WRITABLE, and the hazard scan would not object: an ordinary file is not a host channel."""
    from core.inference import sandbox_linux

    state = tmp_path / "studio"
    (state / "auth").mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(state))

    assert sandbox_linux._holds_studio_state(str(state)) is True
    assert (
        sandbox_linux._holds_studio_state(str(tmp_path)) is True
    ), "an ancestor of the Studio root was not recognised"
    assert sandbox_linux._holds_studio_state(str(tmp_path / "elsewhere")) is False


def test_the_cache_bind_itself_drops_a_component_holding_studio_state(monkeypatch, tmp_path):
    """Through _model_cache_binds, not just the predicate: the guard is only worth anything if the bind list is what changes."""
    import types

    from core.inference import sandbox_linux

    state = tmp_path / "studio"
    (state / "auth").mkdir(parents = True)
    elsewhere = tmp_path / "models"
    for name in ("xet", "datasets", "assets"):
        (elsewhere / name).mkdir(parents = True, exist_ok = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(state))

    settings = types.ModuleType("utils.hf_cache_settings")
    settings.get_hf_cache_paths = lambda: types.SimpleNamespace(
        cache_home = str(elsewhere),
        hub_cache = str(state),
        xet_cache = str(elsewhere / "xet"),
    )
    monkeypatch.setitem(sys.modules, "utils.hf_cache_settings", settings)
    monkeypatch.setattr(sandbox_linux, "_cache_hazard_within_deadline", lambda name, path: None)

    binds = sandbox_linux._model_cache_binds(str(tmp_path / "work"))

    assert "hub" not in binds, "the Studio root was shared into the sandbox as the hub cache"
    assert binds.get("xet") == str(elsewhere / "xet")


def test_a_cache_configured_as_a_home_directory_is_not_shared(monkeypatch, tmp_path):
    """HF_HUB_CACHE pointed at a home is a misconfiguration; the consequence is not."""
    import types

    from core.inference import sandbox_linux

    home = tmp_path / "home" / "alice"
    (home / ".ssh").mkdir(parents = True)
    (home / ".ssh" / "id_ed25519").write_text("private")
    models = tmp_path / "models"
    for name in ("xet", "datasets", "assets"):
        (models / name).mkdir(parents = True, exist_ok = True)

    monkeypatch.setenv("HOME", str(home))
    settings = types.ModuleType("utils.hf_cache_settings")
    settings.get_hf_cache_paths = lambda: types.SimpleNamespace(
        cache_home = str(models),
        hub_cache = str(home),
        xet_cache = str(models / "xet"),
    )
    monkeypatch.setitem(sys.modules, "utils.hf_cache_settings", settings)
    monkeypatch.setattr(sandbox_linux, "_cache_hazard_within_deadline", lambda name, path: None)

    binds = sandbox_linux._model_cache_binds(str(tmp_path / "work"))

    assert "hub" not in binds, "the user's home was shared into the sandbox writable"
    assert binds.get("xet") == str(models / "xet"), "an ordinary cache stopped working"


def test_a_filesystem_root_is_never_a_cache(monkeypatch, tmp_path):
    """Cheap, and the worst case of the same mistake."""
    from core.inference import sandbox_linux

    monkeypatch.setenv("HOME", str(tmp_path / "somewhere-else"))

    assert sandbox_linux._too_broad_for_a_cache("/") is True
    assert sandbox_linux._too_broad_for_a_cache("/home") is True
    assert sandbox_linux._too_broad_for_a_cache(str(tmp_path / "models" / "hub")) is False


def _stub_capable_backend(monkeypatch, limitations):
    """An available capability and a backend that builds, so the test is about prepare_tool_launch's decision rather than about this host's bwrap."""
    from core.inference import os_sandbox, sandbox_linux

    capability = os_sandbox.SandboxCapability(
        backend = "stub",
        available = True,
        reason = "stub",
        environment = "linux",
        protection_state = "qualified",
        profile_id = "stub",
        limitations = (),
    )
    monkeypatch.setattr(os_sandbox, "capability_snapshot", lambda *a, **k: capability)

    def build(plan):
        return os_sandbox.PreparedSandboxLaunch(
            argv = plan.argv,
            workdir = plan.workdir,
            env = dict(plan.env),
            preexec_fn = None,
            backend = "stub",
            launch_limitations = limitations,
        )

    monkeypatch.setattr(sandbox_linux, "prepare", build)


def test_required_refuses_a_workdir_it_could_not_finish_checking(monkeypatch, tmp_path):
    """`auto` degrades here on purpose, and `required` must not."""
    from core.inference import os_sandbox

    workdir = tmp_path / "work"
    workdir.mkdir()
    for i in range(8):
        (workdir / f"f{i}").write_text("")

    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_ENTRIES", 3)

    limitations = os_sandbox.scan_workdir_for_host_channels(str(workdir))
    assert limitations == (
        os_sandbox.WORKDIR_SCAN_INCOMPLETE,
    ), "the budget did not trip, so this test proves nothing"
    _stub_capable_backend(monkeypatch, limitations)

    plan = os_sandbox.ToolLaunchPlan(
        argv = ("/bin/true",),
        workdir = str(workdir),
        env = {},
        requested_mode = "required",
        timeout_seconds = 10,
    )

    with pytest.raises(os_sandbox.WorkdirUnsafeError, match = "too large to check"):
        os_sandbox.prepare_tool_launch(plan)


def test_auto_still_launches_on_the_same_workdir(monkeypatch, tmp_path):
    """The other half of the same decision: the brick fix must survive."""
    from core.inference import os_sandbox

    workdir = tmp_path / "work"
    workdir.mkdir()
    for i in range(8):
        (workdir / f"f{i}").write_text("")

    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_ENTRIES", 3)
    _stub_capable_backend(monkeypatch, (os_sandbox.WORKDIR_SCAN_INCOMPLETE,))

    plan = os_sandbox.ToolLaunchPlan(
        argv = ("/bin/true",),
        workdir = str(workdir),
        env = {},
        requested_mode = "auto",
        timeout_seconds = 10,
    )

    prepared = os_sandbox.prepare_tool_launch(plan)
    try:
        assert prepared.execution_record is not None
        assert os_sandbox.WORKDIR_SCAN_INCOMPLETE in prepared.execution_record.limitations
    finally:
        prepared.cleanup()


def test_a_registered_model_folder_is_bound_read_only(monkeypatch, tmp_path):
    """The same disagreement on Linux: silent at the approval gate, absent from the binds, so the read fails inside the jail."""
    from core.inference import os_sandbox

    library = tmp_path / "library" / "models"
    library.mkdir(parents = True)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setattr(os_sandbox, "model_library_roots", lambda: (str(library),))

    from core.inference import sandbox_linux

    if sandbox_linux.shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this host")
    monkeypatch.setattr(sandbox_linux, "model_library_roots", lambda: (str(library),))

    launch = sandbox_linux.prepare(_plan(workdir))
    try:
        pairs = _pairs(launch.argv, "--ro-bind-try")
    finally:
        launch.cleanup()

    assert (str(library), str(library)) in pairs, "the registered model folder was not bound"


def test_a_model_folder_that_is_a_system_directory_is_refused(monkeypatch):
    """A registered folder that is really /etc or a home is a misconfiguration, and binding it would undo the rest of the profile."""
    from core.inference import os_sandbox, tool_path_approval

    monkeypatch.setattr(tool_path_approval, "_scan_folder_roots", lambda: ("/etc", "/"))
    monkeypatch.setattr(
        "utils.paths.storage_roots.well_known_model_dirs",
        lambda: (os.path.expanduser("~"),),
        raising = False,
    )

    assert os_sandbox.model_library_roots() == ()


def test_a_model_folder_that_is_a_credential_directory_is_refused(monkeypatch, tmp_path):
    """OLLAMA_MODELS=~/.ssh would otherwise grant the keys the approval gate still asks about."""
    from core.inference import os_sandbox, tool_path_approval

    keys = tmp_path / ".ssh"
    models = tmp_path / "models"
    keys.mkdir()
    models.mkdir()
    monkeypatch.setattr(tool_path_approval, "_scan_folder_roots", lambda: (str(keys),))
    monkeypatch.setattr(
        "utils.paths.storage_roots.well_known_model_dirs",
        lambda: (str(models),),
        raising = False,
    )

    assert os_sandbox.model_library_roots() == (os.path.realpath(models),)


def _bind_unix_socket(path):
    """Create a bound AF_UNIX socket at *path*."""
    import socket

    here = os.getcwd()
    os.chdir(os.path.dirname(path))
    try:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(os.path.basename(path))
        finally:
            listener.close()
    finally:
        os.chdir(here)


def test_a_stale_tool_socket_does_not_brick_the_session(tmp_path):
    """A crashed tool's leftover listener must not end the chat."""
    from core.inference import os_sandbox

    workdir = tmp_path / "work"
    scratch = workdir / os_sandbox.TOOL_TEMP_DIRNAME / "pymp-abc"
    scratch.mkdir(parents = True)
    os.chmod(scratch.parent, 0o700)
    _bind_unix_socket(str(scratch / "listener-0"))
    assert (scratch / "listener-0").exists(), "the stale socket was not created"

    assert os_sandbox.scan_workdir_for_host_channels(str(workdir)) == ()
    assert not (scratch / "listener-0").exists(), "the stale socket was left in place"


def test_a_socket_outside_the_scratch_directory_still_fails_the_call(tmp_path):
    """The sweep is an exemption for one Studio-owned directory, not a way past the check: a socket anywhere else in the workdir is still fatal."""
    from core.inference import os_sandbox

    workdir = tmp_path / "work"
    workdir.mkdir()
    _bind_unix_socket(str(workdir / "planted"))

    with pytest.raises(os_sandbox.WorkdirUnsafeError, match = "device or IPC node"):
        os_sandbox.scan_workdir_for_host_channels(str(workdir))


def test_a_scratch_entry_that_cannot_be_removed_still_fails_the_call(tmp_path):
    """Fails closed: what the sweep could not unlink is still a finding."""
    from core.inference import os_sandbox

    if os.geteuid() == 0:
        pytest.skip("root ignores the directory permissions this test relies on")

    workdir = tmp_path / "work"
    scratch = workdir / os_sandbox.TOOL_TEMP_DIRNAME
    held = scratch / "held"
    held.mkdir(parents = True)
    os.chmod(scratch, 0o700)
    _bind_unix_socket(str(held / "listener-0"))
    os.chmod(held, 0o500)
    try:
        with pytest.raises(os_sandbox.WorkdirUnsafeError, match = "device or IPC node"):
            os_sandbox.scan_workdir_for_host_channels(str(workdir))
    finally:
        os.chmod(held, 0o700)


def test_concurrent_launches_share_one_workdir_scan(monkeypatch, tmp_path):
    """A second launch used to see the first one's walk in flight and report the workdir unchecked, which required refuses."""
    import threading
    import time

    from core.inference import os_sandbox

    workdir = tmp_path / "work"
    workdir.mkdir()
    walks = []

    def slow(
        root,
        max_entries,
        seconds,
        witness = None,
    ):
        walks.append(root)
        time.sleep(0.5)
        return None

    monkeypatch.setattr(os_sandbox, "_host_channel_hazard", slow)
    results = []
    callers = [
        threading.Thread(
            target = lambda: results.append(os_sandbox.scan_workdir_for_host_channels(str(workdir)))
        )
        for _ in range(3)
    ]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(20)

    assert results == [(), (), ()], results
    assert len(walks) == 1, "each launch walked the workdir again"


def test_a_workdir_scan_that_blocks_is_given_up_on_rather_than_waited_out(monkeypatch, tmp_path):
    """The budget is checked between entries, which is only a deadline while the walk is running."""
    import threading
    import time

    from core.inference import os_sandbox

    workdir = tmp_path / "work"
    workdir.mkdir()
    released = threading.Event()

    def wedged(root, max_entries, seconds):
        released.wait(8)
        return None

    monkeypatch.setattr(os_sandbox, "_host_channel_hazard", wedged)
    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_SECONDS", 0.3)

    started = time.monotonic()
    try:
        limitations = os_sandbox.scan_workdir_for_host_channels(str(workdir))
        waited = time.monotonic() - started
    finally:
        released.set()

    assert limitations == (os_sandbox.WORKDIR_SCAN_INCOMPLETE,)
    assert waited < 5, f"the caller waited {waited:.1f}s on a wedged scan"


def test_a_windows_drive_root_is_not_a_model_library():
    """`C:\\` is not os.sep, so the filesystem-root check missed it and a folder registered as the whole system drive was granted read access."""
    import ntpath
    import posixpath

    from core.inference import os_sandbox

    assert os_sandbox._is_filesystem_root("C:\\", ntpath)
    assert os_sandbox._is_filesystem_root("\\\\server\\share\\", ntpath)
    assert not os_sandbox._is_filesystem_root("C:\\Models", ntpath)
    assert os_sandbox._is_filesystem_root("/", posixpath)
    assert not os_sandbox._is_filesystem_root("/models", posixpath)


def test_a_windows_system_directory_is_not_a_model_library(monkeypatch, tmp_path):
    """The Windows system directories are not fixed paths, so they are read from the environment rather than spelled in a POSIX-only table."""
    from core.inference import os_sandbox, tool_path_approval

    windows = tmp_path / "Windows"
    windows.mkdir()
    monkeypatch.setenv("SystemRoot", str(windows))
    monkeypatch.setattr(tool_path_approval, "_scan_folder_roots", lambda: (str(windows),))
    monkeypatch.setattr(
        "utils.paths.storage_roots.well_known_model_dirs",
        lambda: (),
        raising = False,
    )

    assert os_sandbox.model_library_roots() == ()


def test_the_directory_holding_every_home_is_not_a_model_library(monkeypatch, tmp_path):
    """/home and /Users were listed literally; the Windows equivalent is C:\\Users under whichever drive Windows was installed on."""
    from core.inference import os_sandbox, tool_path_approval

    homes = tmp_path / "homes"
    (homes / "someone").mkdir(parents = True)
    monkeypatch.setenv("HOME", str(homes / "someone"))
    monkeypatch.setattr(tool_path_approval, "_scan_folder_roots", lambda: (str(homes),))
    monkeypatch.setattr(
        "utils.paths.storage_roots.well_known_model_dirs",
        lambda: (),
        raising = False,
    )

    assert os_sandbox.model_library_roots() == ()


def test_a_live_listener_in_the_scratch_directory_is_not_removed(tmp_path):
    """Two tool calls can share the scratch directory, so "left behind" has to be proved: a listener the other call is still using must survive."""
    import socket

    from core.inference import os_sandbox

    workdir = tmp_path / "work"
    scratch = workdir / os_sandbox.TOOL_TEMP_DIRNAME
    scratch.mkdir(parents = True)
    os.chmod(scratch, 0o700)

    here = os.getcwd()
    os.chdir(scratch)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind("listener-0")
        listener.listen(1)
        os.chdir(here)
        assert os_sandbox.clear_stale_tool_ipc(str(workdir)) == ()
        assert (scratch / "listener-0").exists(), "a live listener was unlinked"
    finally:
        os.chdir(here)
        listener.close()


def test_a_scratch_directory_studio_did_not_create_is_left_alone(tmp_path):
    """_sandbox_temp_dir adopts an existing unsloth-tmp rather than failing, so a project that already had one must not have its own endpoints swept."""
    from core.inference import os_sandbox

    workdir = tmp_path / "work"
    scratch = workdir / os_sandbox.TOOL_TEMP_DIRNAME
    scratch.mkdir(parents = True)
    os.chmod(scratch, 0o755)
    _bind_unix_socket(str(scratch / "theirs"))

    assert os_sandbox.clear_stale_tool_ipc(str(workdir)) == ()
    assert (scratch / "theirs").exists(), "a directory Studio did not create was swept"


def test_a_dormant_fifo_is_never_swept(tmp_path):
    """A FIFO with no reader is at rest, not abandoned: it is meant to outlive the processes at its ends, and 0700 plus ownership is evidence of who made the directory rather than proof."""
    from core.inference import os_sandbox

    workdir = tmp_path / "work"
    scratch = workdir / os_sandbox.TOOL_TEMP_DIRNAME
    scratch.mkdir(parents = True)
    os.chmod(scratch, 0o700)
    os.mkfifo(scratch / "theirs", 0o600)

    assert os_sandbox.clear_stale_tool_ipc(str(workdir)) == ()
    assert (scratch / "theirs").exists(), "an idle named pipe was deleted"
    with pytest.raises(os_sandbox.WorkdirUnsafeError, match = "device or IPC node"):
        os_sandbox.scan_workdir_for_host_channels(str(workdir))


def test_the_stale_ipc_sweep_runs_inside_the_scan_budget(monkeypatch, tmp_path):
    """The sweep walks the same workdir, so it can block on the same wedged mount."""
    import threading
    import time

    from core.inference import os_sandbox

    workdir = tmp_path / "work"
    workdir.mkdir()
    released = threading.Event()

    def wedged_sweep(_workdir, _deadline = None):
        released.wait(8)
        return ()

    monkeypatch.setattr(os_sandbox, "clear_stale_tool_ipc", wedged_sweep)
    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_SECONDS", 0.3)

    started = time.monotonic()
    try:
        limitations = os_sandbox.scan_workdir_for_host_channels(str(workdir))
        waited = time.monotonic() - started
    finally:
        released.set()

    assert limitations == (os_sandbox.WORKDIR_SCAN_INCOMPLETE,)
    assert waited < 5, f"the caller waited {waited:.1f}s on a wedged sweep"


def test_the_sweep_stops_at_its_deadline(tmp_path):
    """Past the deadline it leaves the rest for the walk to refuse rather than spending the call's time on a directory it may never finish."""
    import time

    from core.inference import os_sandbox

    workdir = tmp_path / "work"
    scratch = workdir / os_sandbox.TOOL_TEMP_DIRNAME
    scratch.mkdir(parents = True)
    os.chmod(scratch, 0o700)
    _bind_unix_socket(str(scratch / "listener-0"))

    assert os_sandbox.clear_stale_tool_ipc(str(workdir), time.monotonic() - 1) == ()
    assert (scratch / "listener-0").exists()
    assert os_sandbox.clear_stale_tool_ipc(str(workdir)) == (str(scratch / "listener-0"),)


def test_a_channel_planted_deep_inside_a_cached_component_invalidates_its_verdict(
    monkeypatch, tmp_path
):
    """The memo made the walk cheap and made invalidation wrong: a socket created inside an existing nested directory leaves the component root's mtime alone, so a clean verdict could be reused over exactly the channel the scan exists to reject."""
    from core.inference import sandbox_linux

    cache = tmp_path / "hostcache"
    nested = cache / "hub" / "models--org--name" / "snapshots" / "abc"
    nested.mkdir(parents = True)
    (nested / "config.json").write_text("{}")
    _share_cache_paths(monkeypatch, cache)
    sandbox_linux.reset_cache_verdicts()

    session = str(tmp_path / "session")
    assert "hub" in sandbox_linux._model_cache_binds(session), "the clean cache was not shared"

    _bind_unix_socket(str(nested / "planted"))
    binds = sandbox_linux._model_cache_binds(session)

    assert "hub" not in binds, "a cached verdict was reused over a newly planted socket"


def test_an_unchanged_cache_is_not_walked_again(monkeypatch, tmp_path):
    """The other half: revalidation has to stay cheaper than the walk it replaces, or the memo is pointless."""
    from core.inference import sandbox_linux

    cache = tmp_path / "hostcache"
    (cache / "hub" / "models--org--name").mkdir(parents = True)
    (cache / "hub" / "models--org--name" / "config.json").write_text("{}")
    _share_cache_paths(monkeypatch, cache)
    sandbox_linux.reset_cache_verdicts()

    session = str(tmp_path / "session")
    assert "hub" in sandbox_linux._model_cache_binds(session)

    walks: list[str] = []
    real = sandbox_linux._inspect_cache_component

    def counted(
        name,
        path,
        witness = None,
    ):
        walks.append(path)
        return real(name, path, witness)

    monkeypatch.setattr(sandbox_linux, "_inspect_cache_component", counted)
    assert "hub" in sandbox_linux._model_cache_binds(session)
    assert walks == [], f"the unchanged cache was walked again: {walks}"


def test_a_bwrap_planted_on_path_is_refused_before_it_runs(monkeypatch, tmp_path):
    """bwrap runs on the host before any isolation exists, so a user-writable one must never be executed."""
    planted = tmp_path / "bin"
    planted.mkdir()
    marker = tmp_path / "ran"
    fake = planted / "bwrap"
    fake.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding = "utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{planted}{os.pathsep}{os.environ.get('PATH', '')}")
    workdir = tmp_path / "session"
    workdir.mkdir()
    plan = os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-c", "pass"),
        workdir = str(workdir),
        env = {"PATH": os.environ["PATH"]},
        execution_kind = "python",
        timeout_seconds = 5,
    )

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "trusted system installation"):
        sandbox_linux.prepare(plan)
    assert not marker.exists(), "the planted bwrap was executed"


def test_studio_state_inside_a_runtime_path_is_carved_out(monkeypatch, tmp_path):
    """A runtime root bound whole would hand over a Studio home that lives inside it, auth.db included."""
    lib = tmp_path / "venv" / "lib"
    state = lib / "studio-state"
    (state / "auth").mkdir(parents = True)
    (lib / "python3.12").mkdir()
    managed = tmp_path / "home" / "studio" / "venv"
    managed.mkdir(parents = True)
    monkeypatch.setattr(
        sandbox_linux, "studio_state_roots", lambda: (str(state), str(tmp_path / "home" / "studio"))
    )

    kept = sandbox_linux._without_state_inside((str(lib), str(managed)))

    assert str(lib) not in kept
    assert str(state) not in kept
    assert str(lib / "python3.12") in kept
    assert str(managed) in kept, "the managed venv inside the Studio home must stay readable"
