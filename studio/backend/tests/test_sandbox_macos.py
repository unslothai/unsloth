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
    assert backend._literal_filters((str(link),), resolve = False) == [f'(literal "{link}")']
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


def test_network_is_unrestricted_and_unix_sockets_stay_reachable(profile):
    """The claim is a filesystem boundary; the network is deliberately open."""
    lines = profile.splitlines()
    for rule in ("(allow system-socket)", "(allow network-outbound)", "(allow network-bind)"):
        assert rule in lines
    # Nothing left of the allowlist proxy this backend deliberately does not have.
    assert "localhost" not in profile
    assert "proxy" not in profile.lower()
    # connect()/bind() on a unix socket is network-outbound / network-bind in
    # Seatbelt, not a file operation, so multiprocessing needs its own rule.
    assert f'(allow network-outbound (remote unix-socket (subpath "{_PRIVATE_TMP}")' in profile
    assert f'(allow network-bind (local unix-socket (subpath "{_PRIVATE_TMP}")' in profile
    assert '(literal "/private/var/run/mDNSResponder")' in profile
    assert '(literal "/var/run/mDNSResponder")' in profile


def test_sysctl_and_shm_rules_survive(profile):
    """torch and multiprocessing die without these, loudly and unhelpfully."""
    sysctl = _rule(profile, "(allow sysctl-read ")
    for name in ("hw.ncpu", "hw.memsize", "kern.osproductversion"):
        assert f'(sysctl-name "{name}")' in sysctl
    assert "(allow ipc-posix-sem)" in profile
    assert "^/torch_[0-9]+_[0-9]+_[0-9]+$" in profile


def test_runtime_read_paths_cover_the_interpreter_and_the_site_shim():
    paths = backend.runtime_read_paths()
    shim = os.path.join(os.path.dirname(backend.__file__), "sandbox_site")
    assert shim in paths
    # A read root of "/" or "/usr" would hand back most of the host, so a
    # system interpreter that reports one of them as its prefix is expected to
    # be dropped rather than to appear here.
    assert "/" not in paths and "/usr" not in paths
    for prefix in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
        if prefix in ("/", "/usr"):
            continue
        # Either the prefix itself, or a root already covering it.
        assert any(backend._within(prefix, root) for root in paths), prefix
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
    """The negative control for the whole exercise."""
    workdir = tmp_path / "session"
    workdir.mkdir()
    plan = ToolLaunchPlan(
        argv = ("/bin/sh", "-c", "ls ~/.ssh"),
        workdir = str(workdir),
        env = {"PATH": "/usr/bin:/bin"},
    )
    prepared = backend.prepare(plan)
    try:
        result = subprocess.run(
            prepared.argv, capture_output = True, text = True, timeout = 60, check = False
        )
        assert result.returncode != 0
    finally:
        prepared.cleanup()
