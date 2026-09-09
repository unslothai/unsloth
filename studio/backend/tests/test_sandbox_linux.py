# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the Linux sandbox argv and its seccomp program actually say.

Structure, not a live sandbox: most CI denies unprivileged user namespaces. The
seccomp program is verified by interpreting it, since a filter with the wrong
jump offset passes every length check and then allows what it meant to deny.
"""

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

import pytest

if sys.platform != "linux":
    pytest.skip("the bubblewrap backend is Linux only", allow_module_level = True)

from core.inference import (  # noqa: E402
    os_sandbox,
    sandbox_landlock,
    sandbox_linux,
    sandbox_seccomp,
)
from core.inference.os_sandbox import SandboxUnavailableError, ToolLaunchPlan  # noqa: E402


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
    # Everything past "--" is argv for the tool, so an appended option there
    # becomes a positional argument.
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
    # --new-session denies TIOCSTI injection back into Studio's terminal.
    assert "--new-session" in prepared.argv


def test_capabilities_are_dropped_and_the_filter_arrives_as_an_inherited_descriptor(prepared):
    argv = prepared.argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    fd = int(argv[argv.index("--seccomp") + 1])
    assert prepared.pass_fds == (fd,)
    assert [handle.fileno() for handle in prepared.owned_files] == [fd]
    # The descriptor is rewound: bwrap reads from the current offset.
    assert os.lseek(fd, 0, os.SEEK_CUR) == 0


def test_identity_is_synthesised_rather_than_bound_from_the_host(prepared):
    binds = {destination: source for source, destination in _pairs(prepared.argv, "--ro-bind")}
    passwd, group = binds["/etc/passwd"], binds["/etc/group"]
    assert passwd != "/etc/passwd" and group != "/etc/group"
    assert prepared.cleanup_paths == [os.path.dirname(passwd)]
    # A private 0700 directory, so no other account can swap the files.
    assert os.stat(os.path.dirname(passwd)).st_mode & 0o777 == 0o700
    with open(passwd, encoding = "utf-8") as stream:
        entries = stream.read().splitlines()
    assert len(entries) == 1 and entries[0].split(":")[2] == str(os.getuid())


def test_the_writable_workdir_bind_lands_after_the_root_goes_read_only(prepared, tmp_path):
    argv = prepared.argv
    workdir = os.path.realpath(tmp_path)
    remount = argv.index("--remount-ro")
    assert argv[remount + 1] == "/"
    # --dir before, so the mount point exists in the read-only root; --bind after,
    # so the one writable directory survives the remount.
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
    # After the remount, or every temp file would fail on a read-only path.
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
    # Appended, never first: the sandbox_site startup shim must stay unshadowable.
    assert pythonpath[-1] == packages


def test_system_directories_are_bound_whole_and_never_file_by_file(prepared):
    sources = [source for source, _ in _pairs(prepared.argv, "--ro-bind-try")]
    assert "/usr/lib" in sources
    # Enumerating shared objects produces hundreds of binds and still misses
    # the one dlopen() wants, so every bind of a *file* has to be a named
    # config file or one of the two synthesised identities.
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
            # pyvenv.cfg is the one interpreter file with no directory of its own,
            # and without it a venv cannot find its base.
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
        # Spelled out rather than built from _MODEL_CACHE_SUBDIRS: deriving the
        # expectation from the constant under test makes both sides move together.
        assert sorted(shared) == sorted(
            (os.path.join(str(cache), name), os.path.join(inner, name))
            for name in ("hub", "datasets", "xet", "assets")
        )
        # "modules" is remote code huggingface_hub writes and then imports, so
        # sharing it would let a sandboxed call leave a module an unisolated one
        # later imports.
        assert not any("modules" in destination for _, destination in shared)
        assert launch.argv[launch.argv.index("HF_HOME") + 1] == inner
        # No bind names the credentials, so HF_HOME/token resolves into the
        # writable session workdir and is empty.
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
    # bwrap --setenv owns the inner values, so rewriting the caller's env
    # here would desynchronise the two.
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
        # --new-session covers the inside of the jail only, and tools.py kills
        # with killpg. Composed with the Landlock scope, so what is asserted is
        # that it RUNS, and first, not that it is the same object.
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
    # Not tmp_path: sun_path is 108 bytes and pytest's per-test directory
    # can exceed it on its own.
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
    # The workdir is bound read-write, so this second name is a writable
    # path out of it.
    os.link(str(outside / "secret"), str(workdir / "innocent"))
    with pytest.raises(SandboxUnavailableError, match = "hard-linked from outside"):
        sandbox_linux._validate_workdir(str(workdir))


def test_a_hard_link_wholly_inside_the_workdir_is_allowed(tmp_path):
    (tmp_path / "a").write_text("x")
    os.link(str(tmp_path / "a"), str(tmp_path / "b"))
    (tmp_path / "nested").mkdir()
    os.link(str(tmp_path / "a"), str(tmp_path / "nested" / "c"))
    assert sandbox_linux._validate_workdir(str(tmp_path)) == os.path.realpath(tmp_path)


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
    assert sandbox_linux._validate_workdir(str(tmp_path)) == os.path.realpath(tmp_path)


def test_a_workdir_too_large_to_check_is_refused_rather_than_accepted_unchecked(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_ENTRIES", 2)
    for name in ("a", "b", "c", "d"):
        (tmp_path / name).write_text("")
    with pytest.raises(SandboxUnavailableError, match = "too large"):
        sandbox_linux._validate_workdir(str(tmp_path))


def test_a_workdir_that_is_not_a_directory_is_refused(tmp_path):
    with pytest.raises(SandboxUnavailableError, match = "not a safe canonical directory"):
        sandbox_linux._validate_workdir(str(tmp_path / "missing"))
    with pytest.raises(SandboxUnavailableError, match = "not a safe canonical directory"):
        sandbox_linux._validate_workdir("/")


def test_a_symlinked_directory_under_the_workdir_is_not_followed(tmp_path):
    # Following it would scan /dev and refuse every launch; the bind does
    # not follow it either, so the link dangles inside the jail.
    (tmp_path / "escape").symlink_to("/dev")
    assert sandbox_linux._validate_workdir(str(tmp_path)) == os.path.realpath(tmp_path)


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
    # Only sys.path entries in a lib directory of the interpreter; whatever
    # else PYTHONPATH inherited is deliberately left outside.
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
        if code == 0x20:  # BPF_LD | BPF_W | BPF_ABS
            (accumulator,) = struct.unpack_from("=I", data, k)
        elif code == 0x15:  # BPF_JMP | BPF_JEQ | BPF_K
            counter += jt if accumulator == k else jf
        elif code == 0x45:  # BPF_JMP | BPF_JSET | BPF_K
            counter += jt if accumulator & k else jf
        elif code == 0x06:  # BPF_RET | BPF_K
            return k
        else:
            raise AssertionError(f"unexpected BPF opcode {code:#x}")
    raise AssertionError("the program ran off the end without returning")


def _share_cache(monkeypatch, cache):
    """Share every subdirectory of *cache* that exists, as the real resolver would.

    The real _model_cache_binds asks the cache-settings layer, which is a
    different unit and not what these tests are about.
    """
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
    for family in (1, 2, 10):  # AF_UNIX, AF_INET, AF_INET6
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
    assert _evaluate(program, nr = 1) == _ALLOW  # write on x86_64, close on aarch64


def test_a_foreign_abi_is_killed_rather_than_evaluated(program):
    # Argument offsets differ per ABI, so a guess inspects the wrong bytes.
    assert _evaluate(program, nr = 1, arch = 0xDEADBEEF) == _KILL


@pytest.mark.skipif(platform.machine().lower() not in ("x86_64", "amd64"), reason = "x32 is x86 only")
def test_the_x32_syscall_table_is_killed(program):
    assert _evaluate(program, nr = 1 | 0x40000000) == _KILL


def test_nested_user_namespaces_are_refused_only_when_bwrap_cannot_do_it():
    clone_nr, unshare_nr, clone3_nr = sandbox_seccomp._USERNS_SYSCALLS[platform.machine().lower()]
    blocking = sandbox_seccomp.program(platform.machine(), block_userns = True)
    assert _evaluate(blocking, nr = unshare_nr) == _EPERM
    # ENOSYS, so glibc falls back to clone() where the flags word is checkable.
    assert _evaluate(blocking, nr = clone3_nr) == _ENOSYS
    assert _evaluate(blocking, nr = clone_nr, args = (0x10000000, 0, 0, 0, 0, 0)) == _EPERM
    assert _evaluate(blocking, nr = clone_nr, args = (0x00000100, 0, 0, 0, 0, 0)) == _ALLOW

    permissive = sandbox_seccomp.program(platform.machine(), block_userns = False)
    assert _evaluate(permissive, nr = unshare_nr) == _ALLOW
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
    """Each probe's errno from a fork child, 0 meaning allowed. Installed through
    prctl, not bwrap, which cannot start where user namespaces are denied."""
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
            if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
                os._exit(97)
            fprog = _SockFprog(len(instructions), block)
            if libc.prctl(22, 2, ctypes.byref(fprog), 0, 0) != 0:  # PR_SET_SECCOMP, FILTER
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
            ):
                ctypes.set_errno(0)
                result = libc.syscall(number, argument, 0, 0)
                verdicts[name] = 0 if result >= 0 else ctypes.get_errno()
            child = os.fork()  # an ordinary fork must survive the userns rules
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
        # A filter that broke sockets would make the "filesystem only" claim
        # false in the other direction.
        assert verdicts["inet"] == 0 and verdicts["unix"] == 0
        assert verdicts["fork"] == 0


def test_the_kernel_refuses_nested_user_namespaces_only_in_the_fallback_filter():
    # clone3 is the honest discriminator: a host that denies user namespaces
    # returns EPERM from unshare anyway, but only this filter makes clone3
    # report ENOSYS.
    assert _kernel_verdicts(block_userns = True)["clone3"] == errno.ENOSYS
    assert _kernel_verdicts(block_userns = True)["unshare_userns"] == errno.EPERM
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
        assert len(payload) % 8 == 0  # struct sock_filter is 8 bytes
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
    """Refused rather than unlinked: symlinking a cache leaf at another volume is a
    legitimate layout."""
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
    # Resolved at import: an import in the forked child can deadlock on a
    # lock a thread held at fork time.
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
        # Positive control: unscoped, this host reaches that socket, so the
        # refusal above is the scope and not an unreachable socket.
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


def test_the_plan_pre_exec_still_runs_before_the_scope():
    ran = []
    composed = sandbox_landlock.with_abstract_scope(lambda: ran.append("plan"))
    composed()
    assert ran == ["plan"]


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
        # The canonical spelling too, so a tool that resolved a path for itself
        # can still open it.
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


def test_an_unreadable_directory_is_refused(tmp_path):
    """A mode-000 directory hides a link out from the scan, and the process that
    owns it can chmod it back."""
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
    """HF_HUB_CACHE=/mnt/models is NOT /mnt/models/hub, so each component is bound
    where it actually is."""
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
    # And a component that would sit inside the workdir is dropped.
    assert "hub" not in sandbox_linux._model_cache_binds(str(hub.parent))


def _real_cache(monkeypatch, home):
    """Point the settings layer at *home* and let the REAL _model_cache_binds run.

    _share_cache replaces _model_cache_binds outright, so a test that used it
    would never reach the hazard check it is about.
    """
    import types

    paths = types.SimpleNamespace(cache_home = home, hub_cache = home / "hub", xet_cache = home / "xet")
    module = types.ModuleType("utils.hf_cache_settings")
    module.get_hf_cache_paths = lambda: paths
    monkeypatch.setitem(sys.modules, "utils.hf_cache_settings", module)


def test_a_cache_component_holding_an_ipc_node_is_not_shared(tmp_path, monkeypatch):
    """The bind is writable and the network namespace is shared, so a pathname
    socket under it is connectable from inside: a read-only mount would not even
    help, since MNT_READONLY governs write() and a socket is reached with send().
    Measured against the real backend before this check existed."""
    host = tmp_path / "hostcache"
    (host / "hub").mkdir(parents = True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Relative, because an AF_UNIX address is capped at ~108 bytes and pytest's
    # tmp_path alone can exceed it.
    monkeypatch.chdir(host / "hub")
    sock.bind("leftover.sock")
    try:
        _real_cache(monkeypatch, host)
        assert "hub" not in sandbox_linux._model_cache_binds(str(tmp_path / "session"))
    finally:
        sock.close()


def test_a_cache_component_holding_an_external_hard_link_is_not_shared(tmp_path, monkeypatch):
    """Same inode under two names, one of them outside the cache. The bind is
    writable, so without this the file outside is writable through the cache name."""
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
    """Dropping, never refusing. The cache is an optimisation, so the degraded
    case is the re-download every call did before it was shared; refusing would
    let anything able to write one socket end every later tool call."""
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
    """git reads /etc/gitconfig for a proxy, a CA path or a URL rewrite, and /etc
    is fresh in the jail. Bound only when root owns it and no one else can write
    it: a symlink into $HOME, or a user-writable file, would carry whatever it
    aimed at back into a jail whose claim is that $HOME is unreadable."""
    good = tmp_path / "gitconfig"
    good.write_text("[http]\n")
    link = tmp_path / "linked"
    link.symlink_to(good)
    assert sandbox_linux._trusted_system_file(str(link)) is False, "a symlink is followed"
    assert sandbox_linux._trusted_system_file(str(tmp_path / "absent")) is False
    os.chmod(good, 0o666)
    assert sandbox_linux._trusted_system_file(str(good)) is False, "world-writable accepted"


def test_a_runtime_under_the_workdir_is_re_bound_read_only(tmp_path, monkeypatch):
    """Studio's own venv living beneath the session workdir must not be writable.

    _runtime_read_paths drops these deliberately, but the recursive workdir bind
    is WRITABLE, so dropping alone let a tool call rewrite site-packages or the
    interpreter and the next server subprocess launched with sys.executable ran
    it with the server's authority. Re-bound read-only AFTER the writable bind.
    """
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
    """The negative control for the rule above. A <workdir>/venv/lib aimed at the
    user's home must NOT be bound by name; inside the jail it simply dangles."""
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
    """os.path.ismount compares device numbers and misses a same-filesystem bind
    mount, which is why _validate_workdir re-reads /proc/self/mountinfo. The
    writable cache bind needs the same check, or it carries the nested mount in."""
    host = tmp_path / "hostcache"
    (host / "hub" / "nested").mkdir(parents = True)
    _real_cache(monkeypatch, host)
    monkeypatch.setattr(
        sandbox_linux, "_host_mount_points", lambda: (str(host / "hub" / "nested"),)
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
    """Its code lives OUTSIDE site-packages, so without this a sandboxed
    `import unsloth` fails where the same environment imported it a moment
    earlier. Read from PEP 610's direct_url.json rather than by parsing .pth
    files, because that record is written whichever mechanism the installer used:
    a PEP 660 finder keeps its mapping in a module and puts nothing on sys.path."""
    source = tmp_path / "checkout"
    package = source / "demo"
    package.mkdir(parents = True)
    (package / "__init__.py").write_text("", encoding = "utf-8")
    # A checkout holds more than its packages, and the sandbox keeps the network.
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


def test_an_editable_root_at_the_filesystem_root_is_refused(tmp_path, monkeypatch):
    """The negative control. An editable install rooted at / or /usr would hand
    back most of the host, which is the guard the runtime paths already apply."""
    _fake_editable(tmp_path, monkeypatch, "/usr")
    try:
        assert os_sandbox.editable_source_roots() == ()
    finally:
        os_sandbox.editable_source_roots.cache_clear()


def test_a_runtime_under_a_symlinked_workdir_is_read_only_through_both_spellings(
    tmp_path, monkeypatch
):
    """A workdir reached through a symlink is bound TWICE, once per spelling, and
    the read-only runtime mounts have to come after both.

    Placed between them, the second bind hides them. Placed at a spelling the jail
    has not bound yet, there is no mount point to land on and bwrap dies with
    "Can't mkdir parents ... Read-only file system", which in `auto` costs the
    session its isolation rather than protecting anything. Measured under
    bubblewrap 0.11 in a container: before this ordering the alias spelling failed
    to launch at all, and now both refuse the write with EROFS.
    """
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
    """The spelling CPython actually reports, which the test above did not use.

    A venv invoked as <alias>/venv/bin/python reports sys.prefix = <alias>/venv,
    not the resolved form; measured on CPython 3.12. The caller hands in the
    canonical workdir, so a lexical containment test rejected every runtime path
    and nothing was re-bound: under bubblewrap 0.11 in a container a tool call
    then overwrote the interpreter's sitecustomize through both spellings.
    """
    real = tmp_path / "real"
    (real / "venv" / "lib").mkdir(parents = True)
    (real / "venv" / "bin").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    # The alias spelling, exactly as an invoked venv reports it.
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
                # The LAST one is what stands: an earlier read-only bind is fine
                # and is covered by the writable bind that follows it, which is
                # exactly why the post-bind one has to exist.
                assert landed and max(landed) > last_bind, f"{spelling} unprotected"
    finally:
        launch.cleanup()


def test_an_editable_namespace_package_is_granted_without_an_init(tmp_path, monkeypatch):
    """A PEP 420 namespace package has no __init__.py by design, so presence of
    one cannot be the only test: the package imports in Studio's environment and
    would fail only inside a tool call. top_level.txt names it."""
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
        # Still only what the distribution declares, so the checkout does not
        # come with it.
        assert not any("fixtures" in path or ".env" in path for path in granted), granted
        assert str(source) not in granted, granted
    finally:
        os_sandbox.editable_source_roots.cache_clear()
