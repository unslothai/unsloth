# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the macOS Seatbelt profile generator.

These run on every platform, because the thing under test is TEXT: the profile
is built by a pure function with no kernel involvement, so its shape can be
asserted anywhere. That matters -- the failure this file exists to catch is a
rule quietly dropped or a mach service quietly added, and neither needs a Mac
to notice.

What it cannot tell you is whether the kernel accepts the profile or enforces
what the rules say. That lives in the darwin-only tests at the bottom, skipped
with a reason everywhere else, and in the live probe.

``_path_filters`` touches the filesystem twice, through ``os.path.exists`` and
``os.path.isdir``. The dual-spelling assertions need a workdir under ``/tmp``
(the symlink into ``/private`` is the whole point) without creating one on a
host where ``/tmp`` is an ordinary directory, so those two predicates are
stubbed for exactly the two paths named. Nothing else in the generator reads
the disk, and the profiles built here pass ``runtime_paths = ()`` so no
assertion depends on where this checkout happens to live.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from core.inference import sandbox_macos as backend
from core.inference.os_sandbox import SandboxUnavailableError, ToolLaunchPlan

_WORKDIR = "/tmp/unsloth-session-abc123"
_PRIVATE_TMP = "/tmp/us-seatbelt-xyz789"
_READ_PREFIX = '(allow file-read* file-test-existence (literal "/") '
_OPTIONAL_PREFIX = '(allow file-read* file-test-existence (literal "/etc/gitconfig")'
# The devices rule and the /dev/fd regex rules share the file-write* operation.
_WRITE_PREFIX = "(allow file-write* (literal "


@pytest.fixture
def profile(monkeypatch) -> str:
    """A profile for a /tmp-rooted workdir, built without creating either path."""
    real_exists, real_isdir = os.path.exists, os.path.isdir
    named = {_WORKDIR, _PRIVATE_TMP}
    monkeypatch.setattr(os.path, "exists", lambda path: path in named or real_exists(path))
    monkeypatch.setattr(os.path, "isdir", lambda path: path in named or real_isdir(path))
    return backend.build_profile(workdir = _WORKDIR, private_tmp = _PRIVATE_TMP, runtime_paths = ())


@pytest.fixture
def launchable(monkeypatch):
    """Let ``prepare`` past its launcher check on a host with no sandbox-exec.

    Only the availability gate is stubbed. Everything else in ``prepare`` --
    the workdir checks, the profile, the private tmp, the argv -- runs for real,
    which is the point: those are what these tests are about.
    """
    monkeypatch.setattr(backend, "available", lambda: (True, "stubbed for this test"))


def _rule(profile: str, prefix: str) -> str:
    """The single rule line starting with ``prefix`` -- and prove it is single."""
    matches = [line for line in profile.splitlines() if line.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one rule starting {prefix!r}, got {len(matches)}"
    return matches[0]


def _subpaths(rule: str) -> set[str]:
    return set(re.findall(r'\(subpath "([^"]+)"\)', rule))


def _literals(rule: str) -> set[str]:
    return set(re.findall(r'\(literal "([^"]+)"\)', rule))


def test_module_imports_and_exports_the_backend_contract():
    """Importing must not require darwin: os_sandbox reads these on any platform."""
    assert backend.BACKEND_NAME == "macos-seatbelt"
    assert isinstance(backend.PROFILE_ID, str) and backend.PROFILE_ID
    assert isinstance(backend.LIMITATIONS, tuple)
    assert all(isinstance(item, str) for item in backend.LIMITATIONS)
    # The execution record has to keep saying the network is not confined.
    assert "unrestricted_network" in backend.LIMITATIONS
    assert callable(backend.prepare)


def test_profile_is_deny_default(profile):
    lines = profile.splitlines()
    assert lines[0] == "(version 1)"
    assert lines[1] == "(deny default)"
    assert "(allow default)" not in profile


def test_every_rule_is_a_balanced_s_expression(profile):
    """The nearest thing to a parser this host has.

    SBPL is TinyScheme: one unbalanced paren rejects the whole profile, and
    every launch on the affected host then fails at once. Quotes are counted
    too, since every path is interpolated into a quoted string.
    """
    depth = 0
    in_string = False
    for index, char in enumerate(profile):
        if in_string:
            if char == "\\":
                continue
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            assert depth >= 0, f"unbalanced ) at offset {index}"
    assert depth == 0, f"{depth} unclosed ( in the profile"
    assert not in_string, "the profile ends inside a quoted string"
    # Every top-level line opens and closes its own rule.
    for line in profile.splitlines():
        assert line == "" or line.startswith("(") or line.startswith("  ("), line


def test_login_keychain_mach_service_is_absent(profile):
    """com.apple.SecurityServer would undo the exec denial on /usr/bin/security.

    With that service a sandboxed tool reads the login Keychain through
    Security.framework without ever running the binary. Certificate evaluation
    does not need it: trustd and ocspd are what TLS uses, and both are here.
    """
    assert "com.apple.SecurityServer" not in profile
    assert "com.apple.SecurityServer" not in backend._MACH_SERVICES
    assert 'com.apple.trustd"' in profile
    assert 'com.apple.ocspd"' in profile
    assert '(literal "/usr/bin/security")' in _rule(profile, "(deny process-exec ")


def test_denied_executables_are_denied_after_the_exec_allowance(profile):
    """Seatbelt is last-match-wins, so the deny has to come after the allow."""
    lines = profile.splitlines()
    deny_index = next(
        index for index, line in enumerate(lines) if line.startswith("(deny process-exec ")
    )
    assert lines.index("(allow process-exec)") < deny_index
    denied = _literals(_rule(profile, "(deny process-exec "))
    for path in backend._DENIED_EXECUTABLES:
        assert path in denied, f"{path} is no longer exec-denied"


def test_both_private_spellings_are_emitted_for_the_workdir(profile):
    """HAZARD 1: /tmp is a symlink into /private, and Seatbelt judges the spelling given.

    A rule written only against the canonical /private form EPERMs a tool that
    opens the short form, because the short spelling is denied before the
    canonical rule is consulted.
    """
    for rule in (_rule(profile, _READ_PREFIX), _rule(profile, _WRITE_PREFIX)):
        for path in (_WORKDIR, f"/private{_WORKDIR}"):
            assert f'(literal "{path}")' in rule
            assert f'(subpath "{path}")' in rule


def test_optional_literals_are_allowed_even_though_they_do_not_exist(profile):
    """HAZARD 3: a missing file under (deny default) is EPERM, and git aborts on it."""
    literals = _literals(_rule(profile, _OPTIONAL_PREFIX))
    # Every optional literal, present or not, in both spellings.
    for path in backend._OPTIONAL_READ_LITERALS:
        assert path in literals, f"{path} lost its unconditional read allowance"
    assert {"/private/etc/gitconfig", "/private/etc/gitattributes"} <= literals
    # And at least one of them does not exist on this host, which is the point:
    # the existence-filtered path rules could not have carried it.
    absent = [path for path in backend._OPTIONAL_READ_LITERALS if not os.path.exists(path)]
    assert absent, "no optional literal is absent here, so this test proved nothing"


def test_optional_read_literals_never_follow_a_symlink(tmp_path):
    """A host that points /etc/gitconfig at ~/dotfiles must not get a home read.

    The read literals are emitted for the spelling as written. The deny list
    keeps resolving, because covering the symlink target as well is the safe
    direction for a denial.
    """
    target = tmp_path / "secret"
    target.write_text("")
    link = tmp_path / "link"
    link.symlink_to(target)
    # The target's absence is the assertion, not an exact list: tmp_path is under
    # /tmp on Linux and under /private/var on macOS, and _sbpl_spellings emits the
    # /private pair for both, so the unresolved spelling is never alone.
    unresolved = backend._literal_filters((str(link),), resolve = False)
    assert f'(literal "{link}")' in unresolved
    assert not any(str(target) in filter_ for filter_ in unresolved), unresolved
    assert f'(literal "{target}")' in backend._literal_filters((str(link),))


def test_ancestor_metadata_rules_are_emitted(profile):
    """HAZARD 2: path resolution stats every intermediate component on the way down."""
    metadata_rule = _rule(profile, "(allow file-read-metadata ")
    metadata = _literals(metadata_rule)
    assert "/" in metadata
    for ancestor in (
        "/tmp",
        "/private/tmp",
        "/private",
        "/etc",
        "/private/etc",
        "/usr/local/etc",
        # Resolving mDNSResponder's socket for connect() walks these first.
        "/var/run",
        "/private/var/run",
    ):
        assert ancestor in metadata, f"{ancestor} has no file-read-metadata rule"
    # Metadata only: an ancestor must not become readable or listable by this.
    assert not _subpaths(metadata_rule)


def test_workdir_and_private_tmp_are_the_only_writable_subpaths(profile):
    writable = _subpaths(_rule(profile, _WRITE_PREFIX))
    assert writable == {_WORKDIR, f"/private{_WORKDIR}", _PRIVATE_TMP, f"/private{_PRIVATE_TMP}"}
    for forbidden in ("/", "/usr", "/tmp", "/private/tmp", str(Path.home())):
        assert forbidden not in writable
    # Devices get file-write-data by literal, never a writable subpath.
    assert not _subpaths(_rule(profile, "(allow file-read* file-test-existence file-write-data "))


def test_no_read_root_reaches_the_users_home(profile):
    """The static tables must never name $HOME; only the chosen runtime paths may.

    The read roots are existence-filtered, so on a non-macOS host most of them
    do not reach the profile text at all -- hence the assertions against the
    tables themselves, which are what a future edit would touch.
    """
    home = str(Path.home())
    assert f'(subpath "{home}")' not in profile
    assert "/Users/" not in profile
    tables = (
        backend._READ_ROOTS
        + backend._TLS_TRUST_PATHS
        + backend._OPTIONAL_READ_ROOTS
        + backend._OPTIONAL_READ_LITERALS
    )
    for path in tables:
        assert not path.startswith("/Users/"), path
        assert "~" not in path, path
    # The keychains that are readable are the SYSTEM ones, for TLS trust. The
    # login keychain lives under the unreadable home and must stay there.
    assert "/System/Library/Keychains" in backend._TLS_TRUST_PATHS
    assert not any("Library/Keychains" in path and path.startswith(home) for path in tables)


def test_ip_egress_is_unrestricted_but_unix_sockets_are_not(profile):
    """The claim is a filesystem boundary and IP stays open, but an unfiltered
    network-outbound also covers AF_UNIX, which no file rule governs. That is a way
    to /var/run/docker.sock and out of the boundary entirely, so outbound names the
    ip domain and the unix sockets a launch needs are listed one by one."""
    lines = profile.splitlines()
    for rule in ("(allow system-socket)", "(allow network-inbound)"):
        assert rule in lines
    # Neither direction may be unconditional: an unfiltered grant covers AF_UNIX,
    # which no file rule governs, so it is a socket anywhere the user can write
    # and a connect() to any host socket such as Docker's.
    assert "(allow network-outbound)" not in lines
    assert "(allow network-bind)" not in lines
    # Host and port wildcards, so TCP and UDP over v4 and v6 are all still open.
    assert '(allow network-outbound (remote ip "*:*"))' in lines
    assert '(allow network-bind (local ip "*:*"))' in lines
    # Nothing left of the allowlist proxy this backend deliberately does not have.
    assert "localhost" not in profile
    assert "proxy" not in profile.lower()
    # connect()/bind() on a unix socket is network-outbound / network-bind in
    # Seatbelt, not a file operation, so multiprocessing needs its own rule.
    assert f'(allow network-outbound (remote unix-socket (subpath "{_PRIVATE_TMP}")' in profile
    assert f'(allow network-bind (local unix-socket (subpath "{_PRIVATE_TMP}")' in profile
    # Both spellings for the DNS socket: a missed one would read as a name
    # resolution bug on every Mac and no test here can pick the right one.
    assert '(allow network-outbound (literal "/private/var/run/mDNSResponder")' in profile
    assert "(allow network-outbound (remote unix-socket (literal " in profile
    assert '(literal "/var/run/mDNSResponder")' in profile
    # Nothing grants a host socket such as Docker's.
    assert "docker.sock" not in profile


def test_process_substitution_descriptors_are_readable(profile):
    """`diff <(sort a) <(sort b)` hands the child /dev/fd/63. Limiting the rule to
    0, 1 and 2 fails a command that works unisolated and on the Linux backend."""
    assert '(allow file-read* (regex #"^/dev/fd/[0-9]+$"))' in profile
    assert '(allow file-write* (regex #"^/dev/fd/[0-9]+$"))' in profile


def test_pip_gets_a_writable_target_inside_the_workdir():
    """Parity with the Linux backend: the runtime paths are not in the write set,
    so pip's default target is unwritable and the install has to go somewhere the
    launch can write and then import from."""
    env = backend._sandbox_environment(
        {"PATH": "/usr/bin", "PYTHONPATH": "/shim"}, _WORKDIR, _PRIVATE_TMP
    )
    packages = f"{_WORKDIR}/{backend.SESSION_PACKAGES_RELPATH}"
    assert env["PIP_TARGET"] == packages
    # Appended, never first: the sandbox_site startup shim must stay unshadowable.
    assert env["PYTHONPATH"].split(os.pathsep) == ["/shim", packages]


def test_openmp_can_write_its_registration_segment(profile):
    """libomp does not only create and unlink /__KMP_REGISTERED_LIB_<uid>, it
    writes the registration into it, and the live probe never loads an OpenMP
    workload so nothing else here would catch the missing operation."""
    kmp = next(
        block for block in profile.split("(allow ipc-posix-shm") if "__KMP_REGISTERED_LIB_" in block
    )
    assert "ipc-posix-shm-write-data" in kmp, kmp


def test_sysctl_and_shm_rules_survive(profile):
    """torch and multiprocessing die without these, loudly and unhelpfully."""
    sysctl = _rule(profile, "(allow sysctl-read ")
    for name in ("hw.ncpu", "hw.memsize", "kern.osproductversion"):
        assert f'(sysctl-name "{name}")' in sysctl
    assert "(allow ipc-posix-sem)" in profile
    assert "^/torch_[0-9]+_[0-9]+_[0-9]+$" in profile


def test_runtime_read_paths_cover_the_interpreter_and_the_site_shim():
    paths = backend.runtime_read_paths()
    # normpath, because ``backend.__file__`` carries whatever spelling the import
    # that first loaded the module used, and under a full-suite run that is
    # "tests/../core/inference" rather than the canonical form the generator
    # emits. Comparing the two directly makes this test pass or fail on import
    # order, which it did: green on its own file, red in the whole suite.
    shim = os.path.normpath(os.path.join(os.path.dirname(backend.__file__), "sandbox_site"))
    assert shim in paths
    # A read root of "/" or "/usr" would hand back most of the host, so a
    # system interpreter that reports one of them as its prefix is expected to
    # be dropped rather than to appear here.
    assert "/" not in paths and "/usr" not in paths
    for prefix in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
        for name in ("bin", "lib"):
            member = os.path.join(prefix, name)
            if not os.path.isdir(member):
                continue
            assert any(backend._within(member, root) for root in paths), member
    # lib-dynload hangs off the exec pair; a uv interpreter spells it through an
    # alias symlink that base_prefix alone never names.
    dynload = os.path.join(
        sys.base_exec_prefix,
        "lib",
        f"python{sys.version_info.major}." f"{sys.version_info.minor}",
        "lib-dynload",
    )
    if os.path.isdir(dynload):
        assert any(backend._within(dynload, root) for root in paths), dynload


def test_a_venv_at_a_project_root_does_not_put_the_project_in_the_read_set(monkeypatch, tmp_path):
    """`python -m venv .` at a project root makes sys.prefix the project root.
    Granting file-read* on the prefix would hand the sandbox the sources, .git and
    .env to reach one lib directory -- and the network is open, so a readable .env
    is an exportable one. The Linux backend has always taken the subdirectories
    only; this is the same rule."""
    project = tmp_path / "project"
    (project / "bin").mkdir(parents = True)
    (project / "lib").mkdir()
    (project / ".git").mkdir()
    (project / ".env").write_text("OPENAI_API_KEY=sk-real")
    (project / "pyvenv.cfg").write_text("home = /usr/bin\n")
    for name in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix"):
        monkeypatch.setattr(sys, name, str(project))

    paths = backend.runtime_read_paths()
    assert str(project) not in paths
    assert not any(backend._within(str(project / ".env"), root) for root in paths)
    assert not any(backend._within(str(project / ".git"), root) for root in paths)
    # The interpreter still gets what it needs out of that same tree.
    assert str(project / "bin") in paths
    assert str(project / "lib") in paths


def test_prepare_wraps_argv_and_preserves_the_preexec(tmp_path, launchable):
    """The preexec is tools.py's os.setsid(); losing it strands the process group."""
    workdir = tmp_path / "session"
    workdir.mkdir()

    def preexec() -> None:
        return None

    plan = ToolLaunchPlan(
        argv = ("/bin/echo", "hello"),
        workdir = str(workdir),
        env = {"PATH": "/usr/bin", "DYLD_INSERT_LIBRARIES": "/evil.dylib", "KEEP": "1"},
        preexec_fn = preexec,
        timeout_seconds = 30,
    )
    prepared = backend.prepare(plan)
    private_tmp = prepared.cleanup_paths[0]
    try:
        assert prepared.argv[0] == backend.SANDBOX_EXEC
        assert prepared.argv[1] == "-p"
        assert prepared.argv[2].startswith("(version 1)\n(deny default)")
        assert prepared.argv[3] == "--"
        assert prepared.argv[4:] == ("/bin/echo", "hello")
        assert prepared.preexec_fn is preexec
        assert prepared.backend == backend.BACKEND_NAME
        assert prepared.timeout_seconds == 30
        assert prepared.env["HOME"] == str(workdir)
        assert prepared.env["KEEP"] == "1"
        assert "DYLD_INSERT_LIBRARIES" not in prepared.env
        assert len(prepared.cleanup_paths) == 1
        assert prepared.env["TMPDIR"] == private_tmp
        assert os.path.isdir(private_tmp)
        writable = _subpaths(_rule(prepared.argv[2], _WRITE_PREFIX))
        assert str(workdir) in writable and private_tmp in writable
    finally:
        prepared.cleanup()
    assert not os.path.exists(private_tmp)


def test_prepare_refuses_a_workdir_that_does_not_exist(tmp_path, launchable):
    plan = ToolLaunchPlan(argv = ("/bin/echo",), workdir = str(tmp_path / "missing"), env = {})
    with pytest.raises(SandboxUnavailableError, match = "does not exist"):
        backend.prepare(plan)


def test_prepare_refuses_the_filesystem_root_as_a_workdir(launchable):
    """ "/" as the workdir would make the writable set the whole host."""
    plan = ToolLaunchPlan(argv = ("/bin/echo",), workdir = "/", env = {})
    with pytest.raises(SandboxUnavailableError, match = "filesystem root"):
        backend.prepare(plan)


def test_prepare_refuses_when_the_launcher_is_missing(tmp_path):
    """No stub here: argv[0] must never become a sandbox-exec that is not there."""
    if os.path.exists(backend.SANDBOX_EXEC):
        pytest.skip("this host has a real /usr/bin/sandbox-exec, so the gate cannot be observed")
    workdir = tmp_path / "session"
    workdir.mkdir()
    plan = ToolLaunchPlan(argv = ("/bin/echo",), workdir = str(workdir), env = {})
    with pytest.raises(SandboxUnavailableError, match = "Seatbelt launcher"):
        backend.prepare(plan)


def test_a_rule_that_lost_every_filter_raises_instead_of_granting_everything():
    """A filterless (allow file-write* ) is UNCONDITIONAL, so it must never render.

    ``_path_filters`` drops paths that do not exist, and a workdir deleted
    between the check and the build would empty that list. Failing closed here
    is the difference between a refused launch and a writable host. An empty
    deny is refused too: it would deny the whole operation, not nothing.
    """
    with pytest.raises(SandboxUnavailableError, match = "unconditionally"):
        backend._rule("allow file-write*", [])
    with pytest.raises(SandboxUnavailableError, match = "unconditionally"):
        backend._rule("deny process-exec", [])
    rendered = backend._rule("allow file-write*", ['(literal "/x")'])
    assert rendered == '(allow file-write* (literal "/x"))'
    # And the whole builder refuses when the writable paths are gone.
    with pytest.raises(SandboxUnavailableError, match = "unconditionally"):
        backend.build_profile(
            workdir = "/tmp/does-not-exist-workdir",
            private_tmp = "/tmp/does-not-exist-tmp",
            runtime_paths = (),
        )


def test_paths_must_be_absolute_and_free_of_newlines():
    with pytest.raises(SandboxUnavailableError):
        backend._validated("relative/path")
    with pytest.raises(SandboxUnavailableError):
        backend._validated("/tmp/with\nnewline")
    with pytest.raises(SandboxUnavailableError):
        backend._validated("/tmp/with\0nul")


def test_available_never_raises_on_a_host_without_the_launcher():
    ok, reason = backend.available()
    assert isinstance(ok, bool) and isinstance(reason, str) and reason
    if not os.path.exists(backend.SANDBOX_EXEC):
        assert ok is False
        assert backend.SANDBOX_EXEC in reason


# ── needs a real macOS host ──────────────────────────────────────────
# Everything above asserts on text. Nothing above proves the kernel accepts the
# profile, and no Seatbelt profile is loaded by anything in this file off Darwin.

_darwin_only = pytest.mark.skipif(
    sys.platform != "darwin" or not os.path.exists(backend.SANDBOX_EXEC),
    reason = "needs a real macOS host: only a Darwin kernel can compile or enforce an SBPL profile",
)


@_darwin_only
def test_profile_compiles_under_sandbox_exec(tmp_path):
    """SBPL is undocumented; one bad token fails the whole profile, not one rule."""
    workdir = tmp_path / "session"
    workdir.mkdir()
    plan = ToolLaunchPlan(argv = ("/usr/bin/true",), workdir = str(workdir), env = {})
    prepared = backend.prepare(plan)
    try:
        result = subprocess.run(
            prepared.argv, capture_output = True, text = True, timeout = 60, check = False
        )
        assert result.returncode == 0, result.stderr
    finally:
        prepared.cleanup()


@_darwin_only
def test_home_is_unreadable_inside_the_sandbox(tmp_path):
    """The negative control for the whole exercise, with its positive control.

    This used to run ``ls ~/.ssh`` and assert a non-zero exit, which a machine
    with no ``~/.ssh`` passes whether or not anything is confined -- and a CI
    runner is exactly such a machine. A read that fails for an unrelated reason
    proves nothing, so the file is created here and proven readable on the host
    first. Only then does the sandbox failing to read it mean the sandbox.
    """
    workdir = tmp_path / "session"
    workdir.mkdir()
    canary = Path(os.path.expanduser("~")) / ".unsloth-seatbelt-canary"
    canary.write_text("UNSLOTH_CANARY_HOME_READABLE")
    try:
        # Positive control: the same command, unsandboxed, on this host.
        argv = ("/bin/sh", "-c", f"cat {shlex.quote(str(canary))}")
        host = subprocess.run(argv, capture_output = True, text = True, timeout = 60, check = False)
        assert host.returncode == 0 and "UNSLOTH_CANARY_HOME_READABLE" in host.stdout, (
            "the canary is not readable even outside the sandbox, so the negative "
            f"control below would prove nothing: {host.stderr}"
        )

        prepared = backend.prepare(
            ToolLaunchPlan(argv = argv, workdir = str(workdir), env = {"PATH": "/usr/bin:/bin"})
        )
        try:
            result = subprocess.run(
                prepared.argv, capture_output = True, text = True, timeout = 60, check = False
            )
            assert result.returncode != 0
            assert "UNSLOTH_CANARY_HOME_READABLE" not in result.stdout
        finally:
            prepared.cleanup()
    finally:
        canary.unlink(missing_ok = True)


def test_a_runtime_path_symlinked_out_of_the_workdir_is_not_readable(monkeypatch, tmp_path):
    """Parity with the Linux backend. The workdir is the one place a tool call can
    write, so a runtime path that starts there points wherever the last call
    pointed it; dropping only the resolved spelling would grant file-read* on it."""
    workdir = tmp_path / "session"
    workdir.mkdir()
    secret = tmp_path / "secrets"
    secret.mkdir()
    (secret / "id_rsa").write_text("PRIVATE KEY")
    venv = workdir / "venv"
    (venv / "bin").mkdir(parents = True)
    (venv / "lib").symlink_to(secret)
    for name in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix"):
        monkeypatch.setattr(sys, name, str(venv))

    paths = backend.runtime_read_paths(str(workdir))
    assert str(secret) not in paths
    assert not any(backend._within(str(secret), path) for path in paths)
    # Without the workdir it is the resolved spelling that survives, which is the
    # behaviour this excludes; asserted so the parameter cannot be dropped silently.
    assert str(secret) in backend.runtime_read_paths()


def test_pip_console_scripts_are_reachable_and_never_shadow_a_system_command():
    """Parity with the Linux backend: <target>/bin holds the console entry point
    pip writes, and it goes last so a binary the tool call plants there cannot
    shadow a bare command the approval logic treats as safe."""
    env = backend._sandbox_environment(
        {"PATH": "/usr/bin:/bin", "PYTHONPATH": "/shim"}, _WORKDIR, _PRIVATE_TMP
    )
    packages = f"{_WORKDIR}/{backend.SESSION_PACKAGES_RELPATH}"
    assert env["PATH"].split(os.pathsep) == ["/usr/bin", "/bin", f"{packages}/bin"]


def test_the_semaphore_namespace_is_named_rather_than_narrowed(profile):
    """ipc-posix-sem is unfiltered and the namespace is host-wide, so a launch can
    reach another same-user process's semaphore. Narrowing it would need the names
    torch and OpenMP pick on a platform none of this runs on, and a wrong guess
    breaks multiprocessing instead of confining a filesystem, so the record states
    it instead."""
    assert "(allow ipc-posix-sem)" in profile
    assert "posix_semaphore_namespace_shared" in backend.LIMITATIONS


def test_host_process_metadata_is_named_rather_than_withheld(profile):
    """kern.proc.pid. and kern.proc.pgrp. are how ps works, and a Terminal call
    running ps is ordinary; Seatbelt has no PID namespace to hide the host the way
    the Linux backend does. The record has to say so rather than read as though
    process-info* being same-sandbox settled it."""
    sysctl = _rule(profile, "(allow sysctl-read ")
    assert '(sysctl-name-prefix "kern.proc.pid.")' in sysctl
    assert "host_process_metadata_readable" in backend.LIMITATIONS
    assert "(allow process-info* (target same-sandbox))" in profile


def test_a_framework_build_gets_its_dyld_image(monkeypatch, tmp_path):
    """A python.org framework build loads <prefix>/Python, a FILE at the top of a
    prefix this otherwise only descends into, and under no read root either. The
    probe fails at dyld startup without it and the whole backend reads as
    unavailable."""
    prefix = tmp_path / "Python.framework" / "Versions" / "3.13"
    (prefix / "bin").mkdir(parents = True)
    (prefix / "lib").mkdir()
    (prefix / "Python").write_bytes(b"\xcf\xfa\xed\xfe")
    for name in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix"):
        monkeypatch.setattr(sys, name, str(prefix))
    paths = backend.runtime_read_paths()
    assert str(prefix / "Python") in paths
    # And still not the prefix itself.
    assert str(prefix) not in paths
