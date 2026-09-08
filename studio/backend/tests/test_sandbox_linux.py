# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the Linux sandbox argv and its seccomp program actually say.

These assert on structure, not on a live sandbox. A host with bubblewrap
installed still cannot build one when the kernel denies unprivileged user
namespaces (Ubuntu's ``kernel.apparmor_restrict_unprivileged_userns=1``), which
is the majority of CI, so a suite that needed a working jail would be a suite
that never ran. The argv is where the policy lives, and the seccomp program is
verified by interpreting it: a filter with the wrong jump offset passes every
length check and then allows the syscall it was written to deny.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import platform
import shutil
import socket
import struct
import sys
import sysconfig
import tempfile

import pytest

if sys.platform != "linux":
    pytest.skip("the bubblewrap backend is Linux only", allow_module_level = True)

from core.inference import sandbox_linux, sandbox_seccomp  # noqa: E402
from core.inference.os_sandbox import SandboxUnavailableError, ToolLaunchPlan  # noqa: E402


def _plan(workdir, argv = ("/bin/true",), **kwargs):
    return ToolLaunchPlan(argv = argv, workdir = str(workdir), env = {"PATH": "/usr/bin"}, **kwargs)


@pytest.fixture
def prepared(tmp_path):
    if sandbox_linux.shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this host")
    launch = sandbox_linux.prepare(_plan(tmp_path))
    yield launch
    launch.cleanup()


def _pairs(argv, flag):
    """Every (source, destination) a bind-like flag names in this argv."""
    return [
        (argv[index + 1], argv[index + 2])
        for index, token in enumerate(argv)
        if token == flag and index + 2 < len(argv)
    ]


# ── the argv ─────────────────────────────────────────────────────────


def test_the_launch_runs_bwrap_and_ends_with_the_payload_after_a_bare_separator(prepared):
    assert os.path.basename(prepared.argv[0]) == "bwrap"
    separator = prepared.argv.index("--")
    assert prepared.argv[separator + 1 :] == ("/bin/true",)
    # Nothing may be appended after the payload: everything past "--" is argv for
    # the tool, so a stray option there would become a positional argument.
    assert prepared.argv.count("--") == 1


def test_the_network_namespace_is_deliberately_left_alone(prepared):
    """The badge says the network is unrestricted, so the argv must not confine it."""
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
    # The descriptor is rewound: bwrap reads the program from the current offset.
    assert os.lseek(fd, 0, os.SEEK_CUR) == 0


def test_identity_is_synthesised_rather_than_bound_from_the_host(prepared):
    binds = {
        destination: source for source, destination in _pairs(prepared.argv, "--ro-bind")
    }
    passwd, group = binds["/etc/passwd"], binds["/etc/group"]
    assert passwd != "/etc/passwd" and group != "/etc/group"
    assert prepared.cleanup_paths == [os.path.dirname(passwd)]
    # A private 0700 directory, so no other account can swap the files under it.
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


def test_the_private_tmpfs_replaces_the_shared_directories(prepared):
    argv = prepared.argv
    remount = argv.index("--remount-ro")
    tmpfs = [argv[i + 1] for i, token in enumerate(argv) if token == "--tmpfs"]
    assert tmpfs == ["/dev/shm", "/tmp"]
    # After the remount, or they would be read-only and every temp file would fail.
    assert argv.index("--tmpfs") > remount


def test_system_directories_are_bound_whole_and_never_file_by_file(prepared):
    sources = [source for source, _ in _pairs(prepared.argv, "--ro-bind-try")]
    assert "/usr/lib" in sources
    # Enumerating shared objects is the failure this backend exists to avoid: it
    # produces hundreds of binds and still misses the one dlopen() wants. So
    # every bind of a *file* has to be one of the named configuration files, or
    # one of the two identities synthesised for this launch.
    named = {*sandbox_linux._ETC_FILES, *sandbox_linux._NETWORK_FILES}
    identity_dir = prepared.cleanup_paths[0]
    for flag in ("--bind", "--ro-bind", "--ro-bind-try"):
        for source, _ in _pairs(prepared.argv, flag):
            if os.path.isdir(source):
                continue
            if source in named or os.path.dirname(source) == identity_dir:
                continue
            # pyvenv.cfg is the one interpreter file with no directory of its own:
            # without it a venv cannot find its base and the jail has no stdlib.
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


def test_the_model_cache_is_the_only_writable_bind_outside_the_workdir(prepared, tmp_path):
    workdir = os.path.realpath(tmp_path)
    for source, destination in _pairs(prepared.argv, "--bind"):
        if source == workdir:
            continue
        assert source == os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        # Mounted at the sandbox's own HOME so the default cache location finds it.
        assert destination == os.path.join(workdir, ".cache", "huggingface")
        assert prepared.argv[prepared.argv.index("HF_HOME") + 1] == destination


def test_the_model_cache_bind_is_absent_when_the_host_has_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_linux, "_model_cache_path", lambda workdir: None)
    launch = sandbox_linux.prepare(_plan(tmp_path))
    try:
        assert "HF_HOME" not in launch.argv
        workdir = os.path.realpath(tmp_path)
        assert _pairs(launch.argv, "--bind") == [(workdir, workdir)]
    finally:
        launch.cleanup()


def test_the_inner_home_and_tmpdir_are_set_by_bwrap_not_by_the_caller_environment(prepared):
    argv = prepared.argv
    assert argv[argv.index("HOME") + 1] == prepared.workdir
    assert argv[argv.index("TMPDIR") + 1] == "/tmp"
    # The caller's environment is handed through untouched: bwrap --setenv owns
    # the inner values, and rewriting them here would desynchronise the two.
    assert prepared.env == {"PATH": "/usr/bin"}


def test_the_outer_setsid_preexec_is_preserved(tmp_path):
    marker = lambda: None  # noqa: E731 - identity is the whole assertion
    launch = sandbox_linux.prepare(_plan(tmp_path, preexec_fn = marker))
    try:
        # tools.py kills a tool call with killpg. --new-session covers the inside
        # of the jail; without this the outer process group never exists.
        assert launch.preexec_fn is marker
    finally:
        launch.cleanup()


def test_the_plan_policy_fields_survive_preparation(tmp_path):
    launch = sandbox_linux.prepare(
        _plan(tmp_path, timeout_seconds = 42, terminate_descendants = False)
    )
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


# ── the workdir a launch will accept ─────────────────────────────────


def test_a_unix_socket_under_the_workdir_is_refused():
    # Not tmp_path: the AF_UNIX address is capped at 108 bytes and pytest's
    # per-test directory can exceed it on its own.
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
    # The workdir is bound read-write, so this second name is a writable path out
    # of the one directory the sandbox is supposed to confine writes to.
    os.link(str(outside / "secret"), str(workdir / "innocent"))
    with pytest.raises(SandboxUnavailableError, match = "hard-linked from outside"):
        sandbox_linux._validate_workdir(str(workdir))


def test_a_hard_link_wholly_inside_the_workdir_is_allowed(tmp_path):
    """Counting the names matters: refusing every st_nlink > 1 breaks normal use.

    pip and uv hard-link wheels into a venv, and `cp -al`, `git clone --local` and
    `rsync --link-dest` all link within a tree. Refusing those would turn OS
    isolation off for the rest of any session whose tool call ran one.
    """
    (tmp_path / "a").write_text("x")
    os.link(str(tmp_path / "a"), str(tmp_path / "b"))
    (tmp_path / "nested").mkdir()
    os.link(str(tmp_path / "a"), str(tmp_path / "nested" / "c"))
    assert sandbox_linux._validate_workdir(str(tmp_path)) == os.path.realpath(tmp_path)


def test_a_nested_host_mount_under_the_workdir_is_refused(tmp_path, monkeypatch):
    nested = tmp_path / "mounted"
    nested.mkdir()
    monkeypatch.setattr(
        sandbox_linux, "_host_mount_points", lambda: ("/", os.path.realpath(nested))
    )
    with pytest.raises(SandboxUnavailableError, match = "nested host mount"):
        sandbox_linux._validate_workdir(str(tmp_path))


def test_the_workdir_itself_being_a_mount_point_is_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sandbox_linux, "_host_mount_points", lambda: ("/", os.path.realpath(tmp_path))
    )
    assert sandbox_linux._validate_workdir(str(tmp_path)) == os.path.realpath(tmp_path)


def test_a_workdir_too_large_to_check_is_refused_rather_than_scanned_forever(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sandbox_linux, "_WORKDIR_SCAN_ENTRIES", 2)
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
    # Following it would scan /dev and refuse every launch; the bind does not
    # follow it either, so the link is dangling inside the jail and harmless.
    (tmp_path / "escape").symlink_to("/dev")
    assert sandbox_linux._validate_workdir(str(tmp_path)) == os.path.realpath(tmp_path)


# ── the interpreter paths a launch exposes ───────────────────────────


def test_every_interpreter_path_this_python_imports_from_is_bound(prepared, tmp_path):
    """The bind set has to cover the interpreter's own sys.path, spelling and all.

    Caught a real hole: a uv-managed base interpreter reports ``base_prefix`` as
    ``cpython-3.12.12-linux-x86_64-gnu`` but ``base_exec_prefix`` as the
    ``cpython-3.12-...`` alias symlink, and lib-dynload sits under the alias. A
    jail built from ``base_prefix`` alone had no C extensions in the standard
    library at all, so nothing importing ``select`` or ``_socket`` would start.
    """
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
    # Every sys.path entry that lives in a lib directory of the interpreter: the
    # standard library, lib-dynload and site-packages. Whatever else PYTHONPATH
    # inherited is deliberately left outside, so it must not be required here.
    needed |= {
        entry
        for entry in sys.path
        if entry
        and any(sandbox_linux._within(entry, os.path.join(p, "lib")) for p in prefixes)
    }
    for path in sorted(needed):
        if not os.path.exists(path):
            continue
        assert any(sandbox_linux._within(path, root) for root in bound), path


def test_a_venv_at_a_project_root_does_not_expose_the_project(tmp_path, monkeypatch):
    """`python -m venv .` makes sys.prefix the project root. Binding it would hand
    the jail the sources, the .git directory and any .env sitting beside them."""
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
    # A read-only bind here would shadow part of the one writable directory.
    assert os.path.realpath(inner / "lib") not in paths


# ── the probe uses this builder, not one of its own ──────────────────


def test_the_probe_builds_its_launch_through_the_real_argv_builder(tmp_path):
    if sandbox_linux.shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this host")
    launch = sandbox_linux.probe_argv(str(tmp_path), ("/bin/true", "-x"), {"PATH": "/usr/bin"})
    try:
        reference = sandbox_linux.prepare(_plan(tmp_path, argv = ("/bin/true", "-x")))
        try:
            # Identical but for the per-launch descriptor and identity directory.
            def normalise(argv, launch):
                fd = str(argv[argv.index("--seccomp") + 1])
                identity = launch.cleanup_paths[0]
                return tuple(item.replace(identity, "<identity>") for item in argv if item != fd)

            assert normalise(launch.argv, launch) == normalise(reference.argv, reference)
        finally:
            reference.cleanup()
    finally:
        launch.cleanup()


# ── the seccomp program, interpreted ─────────────────────────────────

_AUDIT_ARCH = sandbox_seccomp._ABIS.get(platform.machine().lower(), (0,))[0]
_KILL, _ALLOW = 0x80000000, 0x7FFF0000
_EPERM, _ENOSYS = sandbox_seccomp._EPERM, sandbox_seccomp._ENOSYS

if _AUDIT_ARCH == 0:
    pytest.skip(f"no seccomp ABI for {platform.machine()}", allow_module_level = True)


def _evaluate(instructions, *, nr, arch = _AUDIT_ARCH, args = (0,) * 6):
    """Interpret the classic BPF program the way the kernel would."""
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


def test_an_unrelated_syscall_is_allowed(program):
    assert _evaluate(program, nr = 1) == _ALLOW  # write on x86_64, close on aarch64


def test_a_foreign_abi_is_killed_rather_than_evaluated(program):
    # Argument offsets differ per ABI, so a filter that guessed would inspect the
    # wrong bytes and allow the call it meant to deny.
    assert _evaluate(program, nr = 1, arch = 0xDEADBEEF) == _KILL


@pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64"), reason = "x32 is x86 only"
)
def test_the_x32_syscall_table_is_killed(program):
    assert _evaluate(program, nr = 1 | 0x40000000) == _KILL


def test_nested_user_namespaces_are_refused_only_when_bwrap_cannot_do_it():
    clone_nr, unshare_nr, clone3_nr = sandbox_seccomp._USERNS_SYSCALLS[platform.machine().lower()]
    blocking = sandbox_seccomp.program(platform.machine(), block_userns = True)
    assert _evaluate(blocking, nr = unshare_nr) == _EPERM
    # ENOSYS, so glibc falls back to clone() where the flags word is checkable.
    assert _evaluate(blocking, nr = clone3_nr) == _ENOSYS
    assert _evaluate(blocking, nr = clone_nr, args = (0x10000000, 0, 0, 0, 0, 0)) == _EPERM
    # An ordinary thread or fork must still work.
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
    """Install the filter for real in a fork child and report each probe's errno.

    Interpreting the program proves the jumps; only the kernel's own BPF verifier
    proves it loads. bubblewrap cannot start on a host that denies unprivileged
    user namespaces, so the child installs the filter directly through prctl,
    which is the same program bwrap would have installed. 0 means allowed.
    """
    clone3_nr, unshare_nr = 435, sandbox_seccomp._USERNS_SYSCALLS[platform.machine().lower()][1]
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
        # The network is not confined, and a filter that broke sockets would
        # make the "filesystem isolation only" claim false in the other direction.
        assert verdicts["inet"] == 0 and verdicts["unix"] == 0
        assert verdicts["fork"] == 0


def test_the_kernel_refuses_nested_user_namespaces_only_in_the_fallback_filter():
    # clone3 is the honest discriminator: a host that denies user namespaces on
    # its own returns EPERM from unshare either way, but only this filter makes
    # clone3 report ENOSYS so glibc retries through a clone() it can inspect.
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
