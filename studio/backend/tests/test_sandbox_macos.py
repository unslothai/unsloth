# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the macOS Seatbelt profile generator, asserting on the profile TEXT.

``os.path.exists`` and ``os.path.isdir`` are stubbed for exactly two paths so the
dual-spelling assertions get a ``/tmp``-rooted workdir without creating one.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

if sys.platform == "win32":
    pytest.skip("the Seatbelt profile generator is POSIX only", allow_module_level = True)

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
    real_exists, real_isdir = os.path.exists, os.path.isdir
    named = {_WORKDIR, _PRIVATE_TMP}
    monkeypatch.setattr(os.path, "exists", lambda path: path in named or real_exists(path))
    monkeypatch.setattr(os.path, "isdir", lambda path: path in named or real_isdir(path))
    return backend.build_profile(workdir = _WORKDIR, private_tmp = _PRIVATE_TMP, runtime_paths = ())


@pytest.fixture
def launchable(monkeypatch):
    """Only the availability gate is stubbed; everything else runs for real."""
    monkeypatch.setattr(backend, "available", lambda: (True, "stubbed for this test"))


def _rule(profile: str, prefix: str) -> str:
    matches = [line for line in profile.splitlines() if line.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one rule starting {prefix!r}, got {len(matches)}"
    return matches[0]


def _subpaths(rule: str) -> set[str]:
    return set(re.findall(r'\(subpath "([^"]+)"\)', rule))


def _literals(rule: str) -> set[str]:
    return set(re.findall(r'\(literal "([^"]+)"\)', rule))


def test_module_imports_and_exports_the_backend_contract():
    assert backend.BACKEND_NAME == "macos-seatbelt"
    assert isinstance(backend.PROFILE_ID, str) and backend.PROFILE_ID
    assert isinstance(backend.LIMITATIONS, tuple)
    assert all(isinstance(item, str) for item in backend.LIMITATIONS)
    assert "unrestricted_network" in backend.LIMITATIONS
    assert callable(backend.prepare)


def test_profile_is_deny_default(profile):
    lines = profile.splitlines()
    assert lines[0] == "(version 1)"
    assert lines[1] == "(deny default)"
    assert "(allow default)" not in profile


def test_every_rule_is_a_balanced_s_expression(profile):
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
    for line in profile.splitlines():
        assert line == "" or line.startswith("(") or line.startswith("  ("), line


def test_login_keychain_mach_service_is_absent(profile):
    """com.apple.SecurityServer would make the login Keychain readable through
    Security.framework. TLS uses trustd and ocspd, not it."""
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
    for rule in (_rule(profile, _READ_PREFIX), _rule(profile, _WRITE_PREFIX)):
        for path in (_WORKDIR, f"/private{_WORKDIR}"):
            assert f'(literal "{path}")' in rule
            assert f'(subpath "{path}")' in rule


def test_a_non_ascii_workdir_reaches_the_profile_unescaped(monkeypatch):
    """SBPL is TinyScheme and has no \\u escape, so json's default spelling of an
    accented path is a rule that matches nothing: the workdir would be unwritable
    and the live probe would report the whole backend unavailable."""
    workdir = "/tmp/unsloth-session-caf\u00e9"
    private_tmp = "/tmp/us-seatbelt-\u00fcber"
    real_exists, real_isdir = os.path.exists, os.path.isdir
    named = {workdir, private_tmp}
    monkeypatch.setattr(os.path, "exists", lambda path: path in named or real_exists(path))
    monkeypatch.setattr(os.path, "isdir", lambda path: path in named or real_isdir(path))
    text = backend.build_profile(workdir = workdir, private_tmp = private_tmp, runtime_paths = ())
    assert "\\u00" not in text
    for path in (workdir, private_tmp):
        assert f'(subpath "{path}")' in _rule(text, _WRITE_PREFIX)


def test_optional_literals_are_allowed_even_though_they_do_not_exist(profile):
    literals = _literals(_rule(profile, _OPTIONAL_PREFIX))
    for path in backend._OPTIONAL_READ_LITERALS:
        assert path in literals, f"{path} lost its unconditional read allowance"
    assert {"/private/etc/gitconfig", "/private/etc/gitattributes"} <= literals
    # At least one does not exist here, so the existence-filtered path rules
    # could not have carried it.
    absent = [path for path in backend._OPTIONAL_READ_LITERALS if not os.path.exists(path)]
    assert absent, "no optional literal is absent here, so this test proved nothing"


def test_optional_read_literals_never_follow_a_symlink(tmp_path):
    """Read literals must not resolve: an /etc/gitconfig symlinked at ~/dotfiles
    would put a home path in the read set. Denials still resolve."""
    target = tmp_path / "secret"
    target.write_text("")
    link = tmp_path / "link"
    link.symlink_to(target)
    # The target's absence is the assertion, not an exact list: _sbpl_spellings
    # emits the /private pair on both platforms, so the spelling is never alone.
    unresolved = backend._literal_filters((str(link),), resolve = False)
    assert f'(literal "{link}")' in unresolved
    assert not any(str(target) in filter_ for filter_ in unresolved), unresolved
    assert f'(literal "{target}")' in backend._literal_filters((str(link),))


def test_ancestor_metadata_rules_are_emitted(profile):
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
    # Metadata only: an ancestor must not become readable or listable.
    assert not _subpaths(metadata_rule)


def test_workdir_and_private_tmp_are_the_only_writable_subpaths(profile):
    writable = _subpaths(_rule(profile, _WRITE_PREFIX))
    assert writable == {_WORKDIR, f"/private{_WORKDIR}", _PRIVATE_TMP, f"/private{_PRIVATE_TMP}"}
    for forbidden in ("/", "/usr", "/tmp", "/private/tmp", str(Path.home())):
        assert forbidden not in writable
    assert not _subpaths(_rule(profile, "(allow file-read* file-test-existence file-write-data "))


def test_no_read_root_reaches_the_users_home(profile):
    """Asserted against the tables, not the profile text: the read roots are
    existence-filtered and mostly vanish on a non-macOS host."""
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
    # SYSTEM keychains only; the login one is under the unreadable home.
    assert "/System/Library/Keychains" in backend._TLS_TRUST_PATHS
    assert not any("Library/Keychains" in path and path.startswith(home) for path in tables)


def test_ip_egress_is_unrestricted_but_unix_sockets_are_not(profile):
    lines = profile.splitlines()
    for rule in ("(allow system-socket)", "(allow network-inbound)"):
        assert rule in lines
    # Neither direction may be unconditional: an unfiltered grant covers AF_UNIX.
    assert "(allow network-outbound)" not in lines
    assert "(allow network-bind)" not in lines
    assert '(allow network-outbound (remote ip "*:*"))' in lines
    assert '(allow network-bind (local ip "*:*"))' in lines
    assert "localhost" not in profile
    assert "proxy" not in profile.lower()
    # connect()/bind() on a unix socket is network-outbound / network-bind, not a
    # file operation, so multiprocessing needs its own rule.
    assert f'(allow network-outbound (remote unix-socket (subpath "{_PRIVATE_TMP}")' in profile
    assert f'(allow network-bind (local unix-socket (subpath "{_PRIVATE_TMP}")' in profile
    # Both spellings for the DNS socket: no test here can pick the right one.
    assert '(allow network-outbound (literal "/private/var/run/mDNSResponder")' in profile
    assert "(allow network-outbound (remote unix-socket (literal " in profile
    assert '(literal "/var/run/mDNSResponder")' in profile
    assert "docker.sock" not in profile


def test_process_substitution_descriptors_are_readable(profile):
    """bash process substitution hands the child /dev/fd/63, so narrowing the rule
    to 0, 1 and 2 fails a command that works everywhere else."""
    assert '(allow file-read* (regex #"^/dev/fd/[0-9]+$"))' in profile
    assert '(allow file-write* (regex #"^/dev/fd/[0-9]+$"))' in profile


def test_pip_gets_a_writable_target_inside_the_workdir():
    env = backend._sandbox_environment(
        {"PATH": "/usr/bin", "PYTHONPATH": "/shim"}, _WORKDIR, _PRIVATE_TMP
    )
    packages = f"{_WORKDIR}/{backend.SESSION_PACKAGES_RELPATH}"
    assert env["PIP_TARGET"] == packages
    assert env["PYTHONPATH"].split(os.pathsep) == ["/shim", packages]


def test_openmp_can_write_its_registration_segment(profile):
    kmp = next(
        block for block in profile.split("(allow ipc-posix-shm") if "__KMP_REGISTERED_LIB_" in block
    )
    assert "ipc-posix-shm-write-data" in kmp, kmp


def test_sysctl_and_shm_rules_survive(profile):
    sysctl = _rule(profile, "(allow sysctl-read ")
    for name in ("hw.ncpu", "hw.memsize", "kern.osproductversion"):
        assert f'(sysctl-name "{name}")' in sysctl
    assert "(allow ipc-posix-sem)" in profile
    assert "^/torch_[0-9]+_[0-9]+_[0-9]+$" in profile


def test_runtime_read_paths_cover_the_interpreter_and_the_site_shim():
    paths = backend.runtime_read_paths()
    # normpath, because ``backend.__file__`` carries whatever spelling first
    # imported the module, so a direct comparison passes or fails on import order.
    shim = os.path.normpath(os.path.join(os.path.dirname(backend.__file__), "sandbox_site"))
    assert shim in paths
    # "/" or "/usr" as a read root hands back most of the host, so a system
    # interpreter reporting one as its prefix must be dropped.
    assert "/" not in paths and "/usr" not in paths
    for prefix in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
        for name in ("bin", "lib"):
            member = os.path.join(prefix, name)
            if not os.path.isdir(member):
                continue
            assert any(backend._within(member, root) for root in paths), member
    # lib-dynload hangs off the exec pair, which a uv interpreter spells through
    # an alias symlink base_prefix alone never names.
    dynload = os.path.join(
        sys.base_exec_prefix,
        "lib",
        f"python{sys.version_info.major}." f"{sys.version_info.minor}",
        "lib-dynload",
    )
    if os.path.isdir(dynload):
        assert any(backend._within(dynload, root) for root in paths), dynload


def test_a_venv_at_a_project_root_does_not_put_the_project_in_the_read_set(monkeypatch, tmp_path):
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
    with pytest.raises(SandboxUnavailableError, match = "unconditionally"):
        backend._rule("allow file-write*", [])
    with pytest.raises(SandboxUnavailableError, match = "unconditionally"):
        backend._rule("deny process-exec", [])
    rendered = backend._rule("allow file-write*", ['(literal "/x")'])
    assert rendered == '(allow file-write* (literal "/x"))'
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


# Everything above asserts on text; none of it proves the kernel accepts it.

_darwin_only = pytest.mark.skipif(
    sys.platform != "darwin" or not os.path.exists(backend.SANDBOX_EXEC),
    reason = "needs a real macOS host: only a Darwin kernel can compile or enforce an SBPL profile",
)


@_darwin_only
def test_profile_compiles_under_sandbox_exec(tmp_path):
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
    """The file is created and proven readable on the host first: a read that fails
    for an unrelated reason proves nothing."""
    workdir = tmp_path / "session"
    workdir.mkdir()
    canary = Path(os.path.expanduser("~")) / ".unsloth-seatbelt-canary"
    canary.write_text("UNSLOTH_CANARY_HOME_READABLE", encoding = "utf-8")
    try:
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
    # Asserted so the workdir parameter cannot be dropped silently.
    assert str(secret) in backend.runtime_read_paths()


def test_pip_console_scripts_are_reachable_and_never_shadow_a_system_command():
    env = backend._sandbox_environment(
        {"PATH": "/usr/bin:/bin", "PYTHONPATH": "/shim"}, _WORKDIR, _PRIVATE_TMP
    )
    packages = f"{_WORKDIR}/{backend.SESSION_PACKAGES_RELPATH}"
    assert env["PATH"].split(os.pathsep) == ["/usr/bin", "/bin", f"{packages}/bin"]


def test_the_semaphore_namespace_is_named_rather_than_narrowed(profile):
    assert "(allow ipc-posix-sem)" in profile
    assert "posix_semaphore_namespace_shared" in backend.LIMITATIONS


def test_host_process_metadata_is_named_rather_than_withheld(profile):
    sysctl = _rule(profile, "(allow sysctl-read ")
    assert '(sysctl-name-prefix "kern.proc.pid.")' in sysctl
    assert "host_process_metadata_readable" in backend.LIMITATIONS
    assert "(allow process-info* (target same-sandbox))" in profile


def test_a_framework_build_gets_its_dyld_image(monkeypatch, tmp_path):
    prefix = tmp_path / "Python.framework" / "Versions" / "3.13"
    (prefix / "bin").mkdir(parents = True)
    (prefix / "lib").mkdir()
    (prefix / "Python").write_bytes(b"\xcf\xfa\xed\xfe")
    for name in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix"):
        monkeypatch.setattr(sys, name, str(prefix))
    paths = backend.runtime_read_paths()
    assert str(prefix / "Python") in paths
    assert str(prefix) not in paths


def test_a_runtime_under_the_workdir_is_denied_write_after_the_allowance(tmp_path, monkeypatch):
    """The macOS half of the same rule as the Linux backend.

    runtime_read_paths drops a runtime inside the workdir so a <workdir>/venv/lib
    symlinked at ~/.ssh is not granted by name, but file-write* covers the workdir
    subpath, so dropping alone left Studio's own venv writable when it sits under
    the workdir. A tool call could rewrite site-packages or the interpreter and the
    next server subprocess started from sys.executable would run it with the
    server's authority. Denied AFTER the allowance, because Seatbelt is
    last-match-wins and a deny before it would be overridden.
    """
    workdir = tmp_path / "session"
    venv = workdir / "venv"
    (venv / "lib").mkdir(parents = True)
    (venv / "bin").mkdir()
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "exec_prefix", str(venv))

    profile = backend.build_profile(workdir = str(workdir), private_tmp = "/tmp/pt", runtime_paths = ())
    lines = profile.splitlines()
    allow = next(i for i, line in enumerate(lines) if line.startswith("(allow file-write* "))
    deny = next(i for i, line in enumerate(lines) if line.startswith("(deny file-write* "))
    assert deny > allow, "a deny before the allowance is overridden by it"
    denied = _subpaths(lines[deny]) | _literals(lines[deny])
    assert str(venv / "lib") in denied
    assert str(venv / "bin") in denied


def test_no_write_denial_is_emitted_when_the_runtime_is_outside_the_workdir(tmp_path, monkeypatch):
    """The negative control: the rule above must not fire for an ordinary layout,
    where denying anything under the workdir would take away the one writable place."""
    workdir = tmp_path / "session"
    workdir.mkdir()
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "exec_prefix", "/usr")
    profile = backend.build_profile(workdir = str(workdir), private_tmp = "/tmp/pt", runtime_paths = ())
    assert not any(line.startswith("(deny file-write* ") for line in profile.splitlines())


def test_the_openssl_directory_is_granted_by_component_not_whole(profile):
    """A locally managed OpenSSL keeps private keys in a directory beside the
    certificates, so a recursive rule over /etc/ssl is an exfiltratable key with
    the network open. Linux names the public components one by one; this asserts
    macOS does too. The ancestor `file-read-metadata` literals are the exception:
    they carry no contents, and every allowed path needs them."""
    for spelling in ("/etc/ssl", "/private/etc/ssl"):
        for line in profile.splitlines():
            if line.startswith("(allow file-read-metadata"):
                continue
            assert f'(subpath "{spelling}")' not in line, line
            assert f'(literal "{spelling}")' not in line, line
    # The components are optional paths, dropped from the profile on a host that
    # lacks them, so the trust list itself is what carries the assertion.
    assert "/private/etc/ssl" not in backend._TLS_TRUST_PATHS
    for component in ("cert.pem", "certs", "openssl.cnf"):
        assert f"/private/etc/ssl/{component}" in backend._TLS_TRUST_PATHS


def test_a_toolchain_directory_the_user_can_write_is_not_trusted(tmp_path):
    """`xcode-select -p` honours $DEVELOPER_DIR, so a Studio started with that
    aimed at a directory under $HOME would otherwise hand recursive file-read*
    over a home subtree to a profile whose claim is that $HOME is unreadable.
    The variable is stripped from the subprocess, and the answer is checked
    rather than trusted, which is what this pins."""
    mine = tmp_path / "FakeXcode.app" / "Contents" / "Developer"
    mine.mkdir(parents = True)
    assert backend._trusted_system_dir(str(mine)) is False
    assert backend._trusted_system_dir(str(tmp_path / "absent")) is False
    missing_file = tmp_path / "Developer"
    missing_file.write_text("", encoding = "utf-8")
    assert backend._trusted_system_dir(str(missing_file)) is False
    # The positive control, so the check above is not passing because it always
    # says no: a root-owned system directory is accepted.
    assert backend._trusted_system_dir("/usr") is True


def test_the_developer_dir_variable_never_reaches_xcode_select(monkeypatch, tmp_path):
    mine = tmp_path / "Developer"
    mine.mkdir()
    monkeypatch.setenv("DEVELOPER_DIR", str(mine))
    monkeypatch.setattr(backend.sys, "platform", "darwin")
    monkeypatch.setattr(backend.os.path, "exists", lambda path: True)
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout = str(mine), stderr = "")

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    monkeypatch.setattr(backend, "_developer_paths_cache", None)
    try:
        assert backend._developer_paths() == ()
    finally:
        backend._developer_paths_cache = None
    assert "DEVELOPER_DIR" not in seen["env"]


def _profile_for(workdir, monkeypatch, prefix):
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "exec_prefix", str(prefix))
    return backend.build_profile(workdir = str(workdir), private_tmp = _PRIVATE_TMP, runtime_paths = ())


def test_a_runtime_under_a_symlinked_workdir_is_denied_through_both_spellings(
    tmp_path, monkeypatch
):
    """build_profile is handed the caller's spelling of the workdir, and its write
    allowance covers the resolved form too. Measuring containment against the alias
    alone rejected every runtime path, so NO denial was emitted and a tool could
    rewrite the interpreter a later host subprocess runs."""
    real = tmp_path / "real"
    venv = real / "venv"
    (venv / "lib").mkdir(parents = True)
    (venv / "bin").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real)

    profile = _profile_for(alias, monkeypatch, venv)
    under = backend.runtime_paths_under(str(alias))
    for spelling in (real / "venv" / "lib", alias / "venv" / "lib"):
        assert str(spelling) in under, under
    deny = _rule(profile, "(deny file-write* ")
    for spelling in (real / "venv" / "lib", alias / "venv" / "lib"):
        assert f'(subpath "{spelling}")' in deny, deny
    # Last-match-wins, so the denial is worthless before the allowance.
    lines = profile.splitlines()
    assert lines.index(_rule(profile, _WRITE_PREFIX)) < lines.index(deny)


def test_the_framework_python_image_is_denied_when_the_prefix_is_under_the_workdir(
    tmp_path, monkeypatch
):
    """A python.org framework's top-level `Python` is the dyld image, and
    runtime_read_paths already names it. Left out of the denial it stayed under
    the workdir's write allowance, which is the one file a later host subprocess
    maps."""
    workdir = tmp_path / "session"
    prefix = workdir / "Python.framework" / "Versions" / "3.12"
    (prefix / "lib").mkdir(parents = True)
    image = prefix / "Python"
    image.write_bytes(b"\xcf\xfa\xed\xfe")

    deny = _rule(_profile_for(workdir, monkeypatch, prefix), "(deny file-write* ")
    assert str(image) in backend.runtime_paths_under(str(workdir))
    assert f'(literal "{image}")' in deny, deny


def test_an_optional_search_root_that_resolves_out_of_its_prefix_is_dropped(monkeypatch):
    """Homebrew on Intel chowns /usr/local to the user, so /usr/local/bin aimed at
    the home directory is something a user, or an earlier unisolated tool call,
    can arrange. _path_filters resolves before it emits, so the recursive subpath
    would be over a home subtree."""
    home = os.path.expanduser("~")
    real = os.path.realpath

    def resolves_home(path):
        return home if path == "/usr/local/bin" else real(path)

    monkeypatch.setattr(os.path, "realpath", resolves_home)
    kept = backend._contained_optional_roots()
    assert "/usr/local/bin" not in kept
    # The positive control: the siblings are untouched, so this is not passing by
    # dropping everything.
    assert "/opt/homebrew/bin" in kept and "/usr/local/lib" in kept
    # Through the profile as well, since the filter is worth nothing if
    # build_profile still reaches for the unfiltered list.
    named = {"/usr/local/bin", _WORKDIR, _PRIVATE_TMP}
    real_isdir, real_exists = os.path.isdir, os.path.exists
    monkeypatch.setattr(os.path, "isdir", lambda path: path in named or real_isdir(path))
    monkeypatch.setattr(os.path, "exists", lambda path: path in named or real_exists(path))
    profile = backend.build_profile(workdir = _WORKDIR, private_tmp = _PRIVATE_TMP, runtime_paths = ())
    assert f'(subpath "{home}")' not in profile


def test_an_editable_checkout_is_listable_but_not_readable(tmp_path, monkeypatch):
    """The import root has to be listed for the interpreter to find anything in
    it, and a literal grants exactly that. A subpath would grant the checkout,
    which is the whole point of naming the packages one by one."""
    checkout = tmp_path / "checkout"
    package = checkout / "demo"
    package.mkdir(parents = True)
    (package / "__init__.py").write_text("", encoding = "utf-8")
    (checkout / ".env").write_text("AWS_SECRET_ACCESS_KEY=real\n", encoding = "utf-8")
    monkeypatch.setattr(backend, "editable_source_roots", lambda: (str(package),))
    monkeypatch.setattr(backend, "editable_import_roots", lambda: (str(checkout),))

    named = {_WORKDIR, _PRIVATE_TMP}
    real_isdir, real_exists = os.path.isdir, os.path.exists
    monkeypatch.setattr(os.path, "isdir", lambda path: path in named or real_isdir(path))
    monkeypatch.setattr(os.path, "exists", lambda path: path in named or real_exists(path))
    profile = backend.build_profile(
        workdir = _WORKDIR,
        private_tmp = _PRIVATE_TMP,
        runtime_paths = (str(package),),
    )
    assert f'(literal "{checkout}")' in profile
    assert f'(subpath "{checkout}")' not in profile
    assert f'(subpath "{package}")' in profile


def test_a_runtime_is_denied_when_sys_prefix_carries_the_workdir_alias(tmp_path, monkeypatch):
    """The macOS half of the same miss. A venv invoked through a symlinked path
    reports the alias in sys.prefix, and pairing the two lexical tests per root
    rejected it either way round, so no denial was emitted at all."""
    real = tmp_path / "real"
    (real / "venv" / "lib").mkdir(parents = True)
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    profile = _profile_for(alias, monkeypatch, alias / "venv")
    deny = _rule(profile, "(deny file-write* ")
    for spelling in (real / "venv" / "lib", alias / "venv" / "lib"):
        assert f'(subpath "{spelling}")' in deny, deny
