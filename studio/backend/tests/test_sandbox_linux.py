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
    binds = {destination: source for source, destination in _pairs(prepared.argv, "--ro-bind")}
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


def test_both_spellings_of_a_symlinked_workdir_get_a_mount_point(tmp_path):
    """A workdir reached through a symlink is TWO paths inside the jail, and both
    are bound. The mount points for both therefore have to be created before the
    root goes read-only, or bwrap dies with "Can't mkdir ...: Read-only file
    system" in exactly the case the second bind exists to serve. Measured in a
    container with working user namespaces before this assertion was written.
    """
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
        # And both still resolve, which is why the pair is needed at all.
        bound = _pairs(argv[remount:], "--bind")
        assert (os.path.realpath(link), str(link)) in bound
    finally:
        launch.cleanup()


def test_the_private_tmpfs_replaces_the_shared_directories(prepared):
    argv = prepared.argv
    remount = argv.index("--remount-ro")
    tmpfs = [argv[i + 1] for i, token in enumerate(argv) if token == "--tmpfs"]
    assert tmpfs[:2] == ["/dev/shm", "/tmp"]
    # After the remount, or they would be read-only and every temp file would fail.
    assert argv.index("--tmpfs") > remount


def test_no_private_key_directory_enters_the_jail_at_all(prepared):
    """The public halves of the trust trees are named one by one, so nothing has to
    remember which distribution called its key directory what. Binding /etc/ssl and
    /etc/pki whole and masking the secrets fails in the wrong direction: the mask
    list is finished only until the next name for one."""
    sources = [source for source, _ in _pairs(prepared.argv, "--ro-bind-try")]
    assert "/etc/ssl/certs" in sources
    assert "/etc/ssl" not in sources and "/etc/pki" not in sources
    # Not by luck: this host has one, and no bind reaches it.
    secret = "/etc/ssl/private"
    assert os.path.isdir(secret), "no private-key directory here, so this proves nothing"
    for source, _ in (*_pairs(prepared.argv, "--ro-bind-try"), *_pairs(prepared.argv, "--ro-bind")):
        assert not sandbox_linux._within(secret, source), source


def test_pip_gets_a_writable_target_inside_the_workdir(prepared):
    """Every runtime path is read-only in here, so pip's default target is not
    writable and `pip install X` failed with a read-only filesystem error. The
    session-local target is on PYTHONPATH so the install imports in the same call."""
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


def test_the_workdir_is_the_only_writable_bind(prepared, tmp_path):
    workdir = os.path.realpath(tmp_path)
    assert _pairs(prepared.argv, "--bind") == [(workdir, workdir)]


def test_the_model_cache_shares_its_data_subdirectories_and_nothing_else(tmp_path, monkeypatch):
    """The cache ROOT is never bound. huggingface_hub keeps the access token at
    $HF_HOME/token and $HF_HOME/stored_tokens, and this sandbox's network is open
    by design, so binding the root and pointing HF_HOME at it would put a live
    credential at the first path a model-authored script reads."""
    cache = tmp_path / "hostcache"
    for name in ("hub", "datasets", "modules", "xet", "assets"):
        (cache / name).mkdir(parents = True)
    (cache / "token").write_text("hf_A_REAL_LOOKING_TOKEN")
    (cache / "stored_tokens").write_text("{}")
    monkeypatch.setattr(sandbox_linux, "_model_cache_path", lambda workdir: str(cache))

    launch = sandbox_linux.prepare(_plan(tmp_path))
    try:
        workdir = os.path.realpath(tmp_path)
        inner = os.path.join(workdir, ".cache", "huggingface")
        shared = _pairs(launch.argv, "--bind-try")
        # Spelled out rather than built from _MODEL_CACHE_SUBDIRS. Deriving the
        # expected value from the constant under test makes both sides move
        # together, so the assertion holds no matter what the constant says:
        # "modules" was dropped from it in d0e30972f and no test changed. A
        # literal is the only version of this that can fail.
        assert sorted(shared) == sorted(
            (os.path.join(str(cache), name), os.path.join(inner, name))
            for name in ("hub", "datasets", "xet", "assets")
        )
        # "modules" is remote code huggingface_hub writes and then imports. It is
        # deliberately NOT shared with the host: a sandboxed call that fetched a
        # trust_remote_code model would otherwise leave a module behind that an
        # unisolated later call imports.
        assert not any("modules" in destination for _, destination in shared)
        # HF_HOME still resolves so the default cache location finds the weights.
        assert launch.argv[launch.argv.index("HF_HOME") + 1] == inner
        # The credentials are named by no bind of any kind, so inside the jail
        # HF_HOME/token is a path in the writable session workdir and empty.
        for credential in ("token", "stored_tokens"):
            host_path = os.path.join(str(cache), credential)
            assert not any(host_path in token for token in launch.argv), credential
    finally:
        launch.cleanup()


def test_the_cache_mount_points_are_made_here_and_left(tmp_path, monkeypatch):
    """bwrap would create a missing bind destination itself, and these sit under
    the workdir bind, so it would create them ON THE HOST -- and behind a .cache an
    earlier call pointed elsewhere. Made here instead, and left: empty directories
    in a dot directory _holds_no_user_files already ignores, and removing them
    would let two overlapping calls in one session unlink each other's."""
    cache = tmp_path / "hostcache"
    (cache / "hub").mkdir(parents = True)
    monkeypatch.setattr(sandbox_linux, "_model_cache_path", lambda workdir: str(cache))
    workdir = tmp_path / "session"
    workdir.mkdir()

    launch = sandbox_linux.prepare(_plan(workdir))
    launch.cleanup()
    assert (workdir / ".cache" / "huggingface" / "hub").is_dir()
    # A second, overlapping preparation finds them and does not disturb them.
    second = sandbox_linux.prepare(_plan(workdir))
    second.cleanup()
    assert (workdir / ".cache" / "huggingface" / "hub").is_dir()


def test_a_cache_directory_the_tool_call_wrote_is_left_alone(tmp_path, monkeypatch):
    """Only the mount points are created, and nothing of the user's is removed to
    make room for them."""
    cache = tmp_path / "hostcache"
    (cache / "hub").mkdir(parents = True)
    monkeypatch.setattr(sandbox_linux, "_model_cache_path", lambda workdir: str(cache))
    workdir = tmp_path / "session"
    (workdir / ".cache" / "huggingface").mkdir(parents = True)
    (workdir / ".cache" / "notes.txt").write_text("the user's")

    launch = sandbox_linux.prepare(_plan(workdir))
    launch.cleanup()
    assert (workdir / ".cache" / "notes.txt").read_text() == "the user's"


def test_the_model_cache_bind_is_absent_when_the_host_has_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_linux, "_model_cache_path", lambda workdir: None)
    launch = sandbox_linux.prepare(_plan(tmp_path))
    try:
        assert "HF_HOME" not in launch.argv
        assert _pairs(launch.argv, "--bind-try") == []
    finally:
        launch.cleanup()


def test_the_inner_home_and_tmpdir_are_set_by_bwrap_not_by_the_caller_environment(prepared):
    argv = prepared.argv
    assert argv[argv.index("HOME") + 1] == prepared.workdir
    # No TMPDIR in the plan's env, so the private tmpfs is the fallback.
    assert argv[argv.index("TMPDIR") + 1] == "/tmp"
    # The caller's environment is handed through untouched: bwrap --setenv owns
    # the inner values, and rewriting them here would desynchronise the two.
    assert prepared.env == {"PATH": "/usr/bin"}


def test_a_tmpdir_inside_the_workdir_survives_into_the_jail(tmp_path):
    """tools.py points TMPDIR at <workdir>/unsloth-tmp so a file a tool call
    writes through tempfile is still there when the call returns and is offered as
    a download. The private /tmp dies with the mount namespace, so overriding
    TMPDIR with it loses every one of those files."""
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
    """A host temp directory is not in the jail at all, so honouring it would
    break every tempfile call rather than preserve an artifact."""
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
        # tools.py kills a tool call with killpg. --new-session covers the inside
        # of the jail; without this the outer process group never exists. Composed
        # with the Landlock scope rather than handed through, so what is asserted
        # is that it RUNS, and first, not that it is the same object.
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


# ── the workdir a launch will accept ─────────────────────────────────


def test_a_unix_socket_under_the_workdir_is_refused():
    """The scan cannot tell a socket a tool call left behind from one a host
    process is serving, and the inode does not record who made it. Refused, which
    costs a littering tool call its next call and not the boundary."""
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
    """Same rule as the socket, and the same reason: no discriminator exists."""
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
    """Somebody else's storage wearing a path inside the one writable directory:
    the workdir bind is recursive and takes it along, and the macOS subpath rule
    grants writes across it, so the check is in the shared scan rather than here."""
    nested = tmp_path / "mounted"
    nested.mkdir()
    real = os.path.ismount
    monkeypatch.setattr(
        os.path, "ismount", lambda path: os.path.samefile(path, nested) or real(path)
    )
    with pytest.raises(SandboxUnavailableError, match = "nested host mount"):
        sandbox_linux._validate_workdir(str(tmp_path))


def test_the_workdir_itself_being_a_mount_point_is_allowed(tmp_path, monkeypatch):
    """Only a mount UNDER the workdir is a boundary problem. The scan walks its
    contents, so the workdir's own mount status is never asked about."""
    real = os.path.ismount
    monkeypatch.setattr(
        os.path, "ismount", lambda path: os.path.samefile(path, tmp_path) or real(path)
    )
    assert sandbox_linux._validate_workdir(str(tmp_path)) == os.path.realpath(tmp_path)


def test_a_workdir_too_large_to_check_is_refused_rather_than_accepted_unchecked(
    tmp_path, monkeypatch
):
    """An unchecked remainder is not a checked one: a link out or a device node
    past the cutoff is exposed exactly as if the scan had never run. Refusing
    costs that session its calls, with the limit named, which is visible and
    actionable where a boundary that quietly is not there is neither."""
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
        if entry and any(sandbox_linux._within(entry, os.path.join(p, "lib")) for p in prefixes)
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


def _evaluate(
    instructions,
    *,
    nr,
    arch = _AUDIT_ARCH,
    args = (0,) * 6,
):
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


def test_the_keyring_syscalls_are_denied(program):
    """Keyrings are not namespaced. A session keyring is a process credential
    carried across fork and exec, so a Kerberos KEYRING: cache or an fscrypt key
    the operator's login session holds is readable from inside the jail without
    touching a host path, and the network in here is open."""
    for number in sandbox_seccomp._KEYRING_SYSCALLS[platform.machine().lower()]:
        assert _evaluate(program, nr = number) == _EPERM


def test_an_unrelated_syscall_is_allowed(program):
    assert _evaluate(program, nr = 1) == _ALLOW  # write on x86_64, close on aarch64


def test_a_foreign_abi_is_killed_rather_than_evaluated(program):
    # Argument offsets differ per ABI, so a filter that guessed would inspect the
    # wrong bytes and allow the call it meant to deny.
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


# ── nothing that originates in the writable workdir crosses the boundary ──


def test_a_runtime_path_symlinked_out_of_the_workdir_is_not_bound(tmp_path, monkeypatch):
    """The workdir is the one place a tool call can write, so a runtime path that
    starts there points wherever the last call pointed it. Excluding only the
    resolved spelling would skip <workdir>/venv/lib and bind the ~/.ssh behind it."""
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


def test_a_cache_ancestor_replaced_during_the_launch_is_not_followed_on_the_way_out(
    tmp_path, monkeypatch
):
    """The other end of the same hazard. When no cache source exists the whole
    generated tree is ordinary writable directories, so a tool call can empty it
    and leave a symlink where .cache was; an rmdir by path afterwards would follow
    that into a matching empty host directory."""
    cache = tmp_path / "hostcache"
    cache.mkdir()  # no subdirectories, so every --bind-try is skipped
    monkeypatch.setattr(sandbox_linux, "_model_cache_path", lambda workdir: str(cache))
    workdir = tmp_path / "session"
    workdir.mkdir()
    outside = tmp_path / "outside"
    (outside / "huggingface" / "hub").mkdir(parents = True)

    launch = sandbox_linux.prepare(_plan(workdir))
    # What a tool call can do from inside: empty the tree and point .cache away.
    shutil.rmtree(workdir / ".cache")
    (workdir / ".cache").symlink_to(outside)
    launch.cleanup()
    assert (outside / "huggingface" / "hub").is_dir(), "cleanup followed the planted symlink"
    assert (workdir / ".cache").is_symlink()


def test_a_symlinked_cache_ancestor_is_refused_rather_than_written_through(tmp_path, monkeypatch):
    """os.mkdir follows an intermediate symlink, and the workdir scan deliberately
    permits directory symlinks, so a .cache a previous call pointed at the user's
    home would have the mount points created out there, on the host, before bwrap
    starts. Refused, and nothing of the user's deleted to make room: symlinking a
    cache leaf at another volume is a legitimate layout. Refusing is safe because a
    refusal no longer de-isolates anything -- on a host that can isolate it is a
    failed call, not an unisolated one."""
    cache = tmp_path / "hostcache"
    (cache / "hub").mkdir(parents = True)
    monkeypatch.setattr(sandbox_linux, "_model_cache_path", lambda workdir: str(cache))
    workdir = tmp_path / "session"
    workdir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workdir / ".cache").symlink_to(outside)

    with pytest.raises(SandboxUnavailableError, match = "not a plain directory"):
        sandbox_linux.prepare(_plan(workdir))
    # Neither followed nor deleted.
    assert (workdir / ".cache").is_symlink()
    assert sorted(entry.name for entry in outside.iterdir()) == []


def test_the_probe_launch_sets_no_new_privs_like_a_real_one(tmp_path):
    """A bubblewrap installed setuid, which is how a host with unprivileged user
    namespaces disabled gets one at all, cannot raise privileges once
    PR_SET_NO_NEW_PRIVS is set. Without it the probe qualifies a backend on which
    every real launch dies after Popen, where auto can no longer fall back."""
    from core.inference import sandbox_probe

    assert sandbox_probe._PR_SET_NO_NEW_PRIVS == 38
    source = inspect.getsource(sandbox_probe._run_probe)
    assert "preexec_fn = _no_new_privs" in source
    # Resolved at import: the pre-exec runs in the forked child, where an import
    # can deadlock on the lock a thread held at fork time.
    assert not [
        node
        for node in ast.walk(ast.parse(inspect.getsource(sandbox_probe._no_new_privs)))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def test_the_pip_targets_script_directory_is_last_on_path(prepared):
    """pip writes a console entry point to <target>/bin, so `pip install black`
    followed by `black .` needs it on PATH. LAST, and that placement is the whole
    safety argument: the directory is writable by the tool call, and
    _build_safe_env drops user-writable PATH entries precisely so a planted binary
    cannot shadow a bare command the approval logic treats as safe."""
    argv = prepared.argv
    entries = argv[argv.index("PATH") + 1].split(os.pathsep)
    packages = os.path.join(prepared.workdir, sandbox_linux.SESSION_PACKAGES_RELPATH)
    assert entries[-1] == os.path.join(packages, "bin")
    assert "/usr/bin" in entries[:-1]


def test_the_compiler_headers_a_source_build_needs_are_readable(prepared):
    """A pip install with no wheel builds from source, which the executor this
    replaces could do: without the include trees it fails at the first #include,
    and Python.h lives under the interpreter's prefix, not /usr/include, for a uv
    or pyenv managed runtime."""
    sources = [source for source, _ in _pairs(prepared.argv, "--ro-bind-try")]
    assert "/usr/include" in sources
    assert "/usr/local/include" in sources
    include = sysconfig.get_paths().get("include")
    if include and os.path.isdir(include):
        bound = [source for source, _ in _pairs(prepared.argv, "--ro-bind")]
        assert any(sandbox_linux._within(include, path) for path in (*sources, *bound)), include


def test_a_runtime_prefix_contributes_its_git_helpers(tmp_path, monkeypatch):
    """A Conda or Homebrew prefix that supplies its own git keeps git-remote-https
    and the rest in <prefix>/libexec, and the sanitized PATH selects that git, so
    an https clone fails at the helper without it."""
    prefix = tmp_path / "conda"
    for name in ("bin", "libexec", "lib"):
        (prefix / name).mkdir(parents = True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    paths = sandbox_linux._runtime_read_paths(str(tmp_path / "session"), ("/usr/lib",))
    assert str(prefix / "libexec") in paths
    assert str(prefix) not in paths


# ── the network namespace is shared, and abstract sockets live in it ──


def test_the_launch_pre_exec_scopes_abstract_sockets(tmp_path):
    """An abstract AF_UNIX socket is in the network namespace, not the filesystem,
    so no mount, bind or seccomp rule in this backend touches one: /proc/net/unix
    names every socket on the host and a connect needs nothing else. On an
    ordinary desktop that reaches the session bus and the X server, which is a way
    out of the boundary this backend claims."""
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
        # The positive control: unscoped, this host can reach that socket, so the
        # refusal above is the scope and not a socket nobody could have connected to.
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
    """A boundary that is not there has to be named. LIMITATIONS is built at import
    from the kernel's Landlock ABI, so the record on an older host carries the
    reachable-sockets entry that this host's does not."""
    supported = sandbox_landlock.abstract_scope_supported()
    assert ("host_abstract_sockets_reachable" in sandbox_linux.LIMITATIONS) is not supported


def test_the_plan_pre_exec_still_runs_before_the_scope():
    """Composed, never replaced: the plan's pre-exec is the os.setsid() every kill
    path in tools.py signals."""
    ran = []
    composed = sandbox_landlock.with_abstract_scope(lambda: ran.append("plan"))
    composed()
    assert ran == ["plan"]


def test_a_runtime_prefix_contributes_its_certificate_store(tmp_path, monkeypatch):
    """A Conda prefix builds OpenSSL against its own <prefix>/ssl/cacert.pem, so
    an https clone reaches git-remote-https and then cannot verify a certificate."""
    prefix = tmp_path / "conda"
    for name in ("bin", "ssl", "lib"):
        (prefix / name).mkdir(parents = True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    paths = sandbox_linux._runtime_read_paths(str(tmp_path / "session"), ("/usr/lib",))
    assert str(prefix / "ssl") in paths


def test_a_workdir_reached_through_a_symlink_is_bound_at_the_spelling_the_caller_used(tmp_path):
    """UNSLOTH_STUDIO_SANDBOX_HOME pointing at another volume is a supported
    override, and tools.py builds the scratch script path, HOME and TMPDIR from
    the spelling it was given. Binding only the canonical form starts bwrap fine
    and then Python cannot open its own argv, after Popen, where auto can no
    longer fall back."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    launch = sandbox_linux.prepare(_plan(link))
    try:
        binds = _pairs(launch.argv, "--bind")
        assert (str(real), str(link)) in binds
        # The canonical spelling too, so a tool that resolved a path for itself is
        # not handed one the jail cannot open either.
        assert (str(real), str(real)) in binds
        assert launch.argv[launch.argv.index("--chdir") + 1] == str(link)
        assert launch.argv[launch.argv.index("HOME") + 1] == str(link)
    finally:
        launch.cleanup()


def test_a_cache_leaf_left_behind_as_a_file_is_refused_at_preparation(tmp_path, monkeypatch):
    """--bind-try onto a leaf that is not a directory fails inside bwrap, after
    Popen, so auto cannot fall back and the session stays broken. Refused at
    preparation instead, and not deleted: these names are not reserved from
    workspace content."""
    cache = tmp_path / "hostcache"
    (cache / "hub").mkdir(parents = True)
    monkeypatch.setattr(sandbox_linux, "_model_cache_path", lambda workdir: str(cache))
    workdir = tmp_path / "session"
    (workdir / ".cache" / "huggingface").mkdir(parents = True)
    (workdir / ".cache" / "huggingface" / "hub").write_text("not a directory")

    with pytest.raises(SandboxUnavailableError, match = "not a plain directory"):
        sandbox_linux.prepare(_plan(workdir))
    assert (workdir / ".cache" / "huggingface" / "hub").read_text() == "not a directory"


def test_an_unreadable_directory_is_refused(tmp_path):
    """A mode-000 directory hides whatever is inside it from the check that exists
    to find a link out, and the process that owns it can chmod it back. Refusing
    costs a tool call that made one its own next call, which is a self-inflicted
    and visible failure; accepting it costs the boundary."""
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
    """Moving the cache through Studio Settings deliberately leaves HF_HOME at the
    default and puts the real paths in the component variables, so reading one
    variable finds an empty default and re-downloads into every session."""
    elsewhere = tmp_path / "models"
    (elsewhere / "hub").mkdir(parents = True)
    import utils.hf_cache_settings as cache_settings

    monkeypatch.setattr(
        cache_settings,
        "get_hf_cache_paths",
        lambda: cache_settings.HuggingFaceCachePaths(
            cache_home = elsewhere,
            hub_cache = elsewhere / "hub",
            xet_cache = elsewhere / "xet",
            source = "studio",
        ),
    )
    assert sandbox_linux._model_cache_path(str(tmp_path / "session")) == str(elsewhere)
    # And still refused when the resolved root would be inside the workdir.
    assert sandbox_linux._model_cache_path(str(elsewhere)) is None
