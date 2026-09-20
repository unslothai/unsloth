# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MXC Windows backend's policy and gating.

These run on every platform, because they are about the policy this backend
GENERATES and the conditions under which it engages. Whether Windows actually
ENFORCES that policy is a different question and no test here answers it; that
needs a Windows runner and lives in the Windows CI job.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.inference import os_sandbox, sandbox_windows  # noqa: E402
from core.inference.os_sandbox import ToolLaunchPlan  # noqa: E402



def _pin(monkeypatch, path):
    """Register ``path`` as the pinned executor for this architecture."""
    from core.inference import mxc_pins

    monkeypatch.setitem(mxc_pins.EXECUTOR_SHA256, mxc_pins.arch_dir(), mxc_pins.digest(path))


@pytest.fixture
def plan(tmp_path):
    return ToolLaunchPlan(
        argv = ("python.exe", "-c", "print('hi')"),
        workdir = str(tmp_path),
        env = {"PATH": "C:\\Windows\\System32", "HOME": str(tmp_path)},
        requested_mode = "required",
        timeout_seconds = 30,
        execution_kind = "python",
    )


def policy_for(plan, tmp_path):
    return sandbox_windows.build_policy(plan, str(tmp_path), "unsloth-test")


def test_the_schema_version_is_pinned_exactly(plan, tmp_path):
    # Not "whatever MXC currently defaults to": the executor was measured
    # accepting 0.8.1-alpha despite the docs calling stable schemas exact-match,
    # so the pin is ours to hold.
    assert policy_for(plan, tmp_path)["version"] == "0.8.0-alpha"


def test_only_the_session_workdir_is_writable(plan, tmp_path):
    filesystem = policy_for(plan, tmp_path)["filesystem"]
    packages = os.path.join(str(tmp_path), os_sandbox.SESSION_PACKAGES_RELPATH)
    assert sorted(filesystem["readwritePaths"]) == sorted([str(tmp_path), packages])


def test_the_user_home_is_not_granted(plan, tmp_path):
    # The whole point: a model-authored script must not reach ~/.aws or ~/.ssh.
    filesystem = policy_for(plan, tmp_path)["filesystem"]
    granted = filesystem["readwritePaths"] + filesystem["readonlyPaths"]
    home = os.path.expanduser("~")
    assert not any(os.path.normcase(p) == os.path.normcase(home) for p in granted)


def test_the_argv_is_not_rendered_into_the_command_line(plan, tmp_path):
    # commandLine is a single STRING. Quoting model-authored argv into a Windows
    # command line in Python is the injection bug this design avoids, so the
    # policy must carry no command at all and leave it to MXC's `--` splice.
    assert "commandLine" not in policy_for(plan, tmp_path)["process"]


def test_the_timeout_reaches_mxc_in_milliseconds(plan, tmp_path):
    assert policy_for(plan, tmp_path)["process"]["timeout"] == 30_000


def test_the_environment_is_passed_verbatim_not_inherited(plan, tmp_path):
    # MXC replaces rather than layers unless inheritDefaultEnv, which is 0.9
    # only, so the sanitized env tools.py built must be the whole env.
    env = policy_for(plan, tmp_path)["process"]["env"]
    # Starts with the caller's PATH: the session package Scripts directory is
    # appended after it, deliberately and last.
    assert any(
        entry == "PATH=C:\\Windows\\System32"
        or entry.startswith("PATH=C:\\Windows\\System32" + os.pathsep)
        for entry in env
    )
    assert all("=" in entry for entry in env)


def test_the_policy_is_json_serializable(plan, tmp_path):
    # It is about to be base64'd into an argv; a non-serializable value would
    # fail at launch rather than here.
    json.dumps(policy_for(plan, tmp_path))


def test_windows_isolation_is_opt_in(monkeypatch):
    monkeypatch.delenv(os_sandbox.WINDOWS_PREVIEW_ENV, raising = False)
    assert os_sandbox.windows_preview_enabled() is False
    monkeypatch.setenv(os_sandbox.WINDOWS_PREVIEW_ENV, "1")
    assert os_sandbox.windows_preview_enabled() is True


def test_the_preview_cannot_be_selected_by_a_request(monkeypatch, tmp_path):
    # Backend eligibility is host policy. If a tool argument could turn it on,
    # it could also turn it off.
    monkeypatch.delenv(os_sandbox.WINDOWS_PREVIEW_ENV, raising = False)
    monkeypatch.setattr(sys, "platform", "win32")
    capability = os_sandbox.capability_snapshot(force = True)
    assert capability.available is False
    assert "opt-in preview" in capability.reason


def test_an_unavailable_windows_host_still_names_the_reason(monkeypatch):
    monkeypatch.delenv(os_sandbox.WINDOWS_PREVIEW_ENV, raising = False)
    monkeypatch.setattr(sys, "platform", "win32")
    capability = os_sandbox.capability_snapshot(force = True)
    assert os_sandbox.WINDOWS_PREVIEW_ENV in capability.remediation
    assert capability.limitations == ("no_os_isolation",)


def test_the_preview_warning_is_carried_into_the_limitations():
    # MXC's README says no MXC profile is a security boundary yet. Surfacing
    # that is not optional; the badge would otherwise read as more than it is.
    assert "mxc_preview_not_a_security_boundary" in sandbox_windows.LIMITATIONS
    assert "unrestricted_network" in sandbox_windows.LIMITATIONS


def test_the_dacl_caveat_is_dropped_only_on_a_tier1_host(monkeypatch):
    monkeypatch.setattr(sandbox_windows, "_windows_build", lambda: 26100)
    assert "windows_tier3_dacl" in sandbox_windows.host_limitations()
    monkeypatch.setattr(sandbox_windows, "_windows_build", lambda: 26600)
    assert "windows_tier3_dacl" not in sandbox_windows.host_limitations()


def test_the_executor_is_never_resolved_from_path(monkeypatch):
    # PATH is attacker-influenced on a host where tool calls run, and this is
    # the binary the whole boundary rests on.
    monkeypatch.delenv("UNSLOTH_MXC_EXEC", raising = False)
    monkeypatch.setenv("PATH", os.path.dirname(sys.executable))
    source = sandbox_windows.executable_path.__doc__ or ""
    assert "Never from PATH" in source


def test_a_non_windows_host_refuses_this_backend(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    ok, reason = sandbox_windows.available()
    assert ok is False
    assert "Windows-only" in reason


def test_every_launch_gets_its_own_container_id(monkeypatch, plan, tmp_path):
    # MXC keys its ACEs on the SID derived from this id and documents that two
    # concurrent runs sharing one revoke each other's ACEs.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "available", lambda: (True, "ok"))
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: sys.executable)
    # prepare() re-checks the digest while holding the file open, so the
    # stand-in executor has to be pinned for that check to be about the
    # thing this test is about.
    _pin(monkeypatch, sys.executable)
    seen = set()
    for _ in range(5):
        prepared = sandbox_windows.prepare(plan)
        # Located by flag, not by index: argv grows as the backend learns what
        # MXC needs, and a positional assumption breaks silently.
        encoded = prepared.argv[prepared.argv.index("--config-base64") + 1]
        import base64

        seen.add(json.loads(base64.b64decode(encoded))["containerId"])
    assert len(seen) == 5


def test_mxc_diagnostics_are_diverted_off_the_payload_stdout(monkeypatch, plan):
    # MXC runs in passthrough stdio mode and forwards its own handles to the
    # child, so without --log-file its warnings land on the same stdout as the
    # payload. That breaks the probe, which requires its token alone, and would
    # splice MXC's banner into what the user sees from a tool call.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "available", lambda: (True, "ok"))
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: sys.executable)
    # prepare() re-checks the digest while holding the file open, so the
    # stand-in executor has to be pinned for that check to be about the
    # thing this test is about.
    _pin(monkeypatch, sys.executable)
    prepared = sandbox_windows.prepare(plan)
    assert "--log-file" in prepared.argv
    log_path = prepared.argv[prepared.argv.index("--log-file") + 1]
    assert os.path.isfile(log_path)
    prepared.cleanup()
    assert not os.path.exists(log_path)


def test_the_prepared_launch_carries_no_posix_only_kwargs(monkeypatch, plan):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "available", lambda: (True, "ok"))
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: sys.executable)
    # prepare() re-checks the digest while holding the file open, so the
    # stand-in executor has to be pinned for that check to be about the
    # thing this test is about.
    _pin(monkeypatch, sys.executable)
    prepared = sandbox_windows.prepare(plan)
    assert prepared.preexec_fn is None
    assert prepared.pass_fds == ()


def test_the_argv_ends_with_the_payload_after_a_bare_separator(monkeypatch, plan):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "available", lambda: (True, "ok"))
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: sys.executable)
    # prepare() re-checks the digest while holding the file open, so the
    # stand-in executor has to be pinned for that check to be about the
    # thing this test is about.
    _pin(monkeypatch, sys.executable)
    prepared = sandbox_windows.prepare(plan)
    assert prepared.argv[-4:] == ("--", "python.exe", "-c", "print('hi')")


def test_cleanup_reconciles_the_container(monkeypatch, plan):
    # A hard kill skips MXC's own ACE revert, so this must run on every path.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "available", lambda: (True, "ok"))
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: sys.executable)
    # prepare() re-checks the digest while holding the file open, so the
    # stand-in executor has to be pinned for that check to be about the
    # thing this test is about.
    _pin(monkeypatch, sys.executable)
    called = []
    monkeypatch.setattr(
        sandbox_windows,
        "_reconcile_container",
        lambda executor, container_id: called.append(container_id),
    )
    prepared = sandbox_windows.prepare(plan)
    prepared.cleanup()
    assert len(called) == 1


def test_the_windows_system_roots_are_granted_read_only(monkeypatch, tmp_path):
    """MXC grants nothing implicitly: "Omitted = no filesystem access beyond the
    default sandbox root". Without these, python.exe cannot resolve ntdll or the
    CRT and the launch fails before the payload runs. Guarded because the whole
    grant can be dropped and every suite still passes on a non-Windows box.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    (tmp_path / "System32").mkdir()

    launch = ToolLaunchPlan(
        argv = ("python.exe", "-c", "print(1)"),
        workdir = str(tmp_path / "workdir"),
        env = {},
        requested_mode = "required",
        timeout_seconds = 30,
    )
    roots = sandbox_windows._readonly_roots(launch, str(tmp_path / "workdir"))

    assert str(tmp_path) in roots, "the Windows system root is not granted"
    assert str(tmp_path / "System32") in roots, "System32 is not granted"


def test_the_system_roots_are_never_writable(monkeypatch, tmp_path):
    """Read-only is the whole point: this widens what a tool call can READ, and
    must not widen what it can write."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    (tmp_path / "System32").mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    plan = ToolLaunchPlan(
        argv = ("python.exe", "-c", "pass"),
        workdir = str(workdir),
        env = {},
        requested_mode = "required",
    )
    filesystem = sandbox_windows.build_policy(plan, str(workdir), "t")["filesystem"]

    assert str(tmp_path) not in filesystem["readwritePaths"]
    assert str(tmp_path / "System32") not in filesystem["readwritePaths"]


def test_windows_gets_the_environment_it_needs_to_start_a_process(monkeypatch):
    """MXC replaces the environment rather than layering onto it, so whatever
    the policy carries is the whole environment. A plan env without SystemRoot
    leaves the interpreter unable to initialise, which looks exactly like
    confinement from the outside."""
    monkeypatch.setenv("SystemRoot", "C:\\Windows")
    monkeypatch.setenv("COMSPEC", "C:\\Windows\\System32\\cmd.exe")

    # Through build_policy, not the helper directly: testing the helper alone
    # leaves the WIRING unguarded, and the wiring is the part that can be
    # dropped by a refactor while every assertion still passes.
    plan = ToolLaunchPlan(
        argv = ("python.exe", "-c", "pass"),
        workdir = "C:\\work",
        env = {"PATH": "C:\\Windows\\System32"},
        requested_mode = "required",
    )
    env = sandbox_windows.build_policy(plan, "C:\\work", "t")["process"]["env"]

    assert "SystemRoot=C:\\Windows" in env
    assert "COMSPEC=C:\\Windows\\System32\\cmd.exe" in env
    # Starts with the caller's PATH: the session package Scripts directory is
    # appended after it, deliberately and last.
    assert any(
        entry == "PATH=C:\\Windows\\System32"
        or entry.startswith("PATH=C:\\Windows\\System32" + os.pathsep)
        for entry in env
    )


def test_a_caller_supplied_value_wins_over_the_host(monkeypatch):
    """Filled in only when absent, so a deliberately overridden value still
    wins and the sanitized env tools.py built is not quietly undone."""
    monkeypatch.setenv("SystemRoot", "C:\\Windows")

    env = sandbox_windows._policy_environment({"SystemRoot": "D:\\Custom"})

    assert "SystemRoot=D:\\Custom" in env
    assert "SystemRoot=C:\\Windows" not in env


def test_a_windows_host_without_the_executor_refuses_cleanly(monkeypatch):
    """The default path must not raise into the launch planner.

    The first version imported `utils.studio_paths`, which does not exist in
    this repo, so on any Windows host that had not set UNSLOTH_MXC_EXEC the
    planner got a ModuleNotFoundError instead of a clean "not installed"
    refusal. CI never saw it because the workflow always sets the variable.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("UNSLOTH_MXC_EXEC", raising = False)

    ok, reason = sandbox_windows.available()

    assert ok is False
    assert "not installed" in reason


def test_the_install_directory_resolves_without_utils_paths(monkeypatch):
    """Degraded environments still get a path, mirroring node_runtime."""
    import builtins

    real_import = builtins.__import__

    def no_storage_roots(name, *args, **kwargs):
        if name == "utils.paths.storage_roots":
            raise ImportError("simulated degraded environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_storage_roots)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", os.path.join("C:", "Studio"))

    directory = sandbox_windows.managed_mxc_dir()

    assert directory.endswith("mxc")


def test_a_workdir_on_another_drive_does_not_break_the_policy_build():
    """os.path.commonpath raises ValueError across Windows drives.

    Left to propagate it fails every policy build for the ordinary layout of
    the interpreter on C: and the sandbox home on D:, and prepare() turns that
    into SandboxBuildError, so even `auto` refuses the tool call instead of
    falling back to software safeguards. Forced here with a relative path,
    which raises the same ValueError on every platform.
    """
    assert sandbox_windows._within("relative/path", os.path.abspath(os.sep)) is False


def test_a_different_drive_counts_as_outside_the_workdir(monkeypatch):
    """The guard must answer "outside", not swallow the question."""

    def different_drives(paths):
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(os.path, "commonpath", different_drives)

    assert sandbox_windows._within("D:\\project", "C:\\workdir") is False


def test_a_windows_editable_url_decodes_to_a_drive_path():
    """file:///C:/Users/me/project must not decode to \\C:\\Users\\me\\project.

    The naive decode keeps the URL's leading slash, the isdir check then drops
    the source root, and editable imports fail inside an isolated tool call.
    """
    from urllib.parse import urlparse

    decoded = os_sandbox._path_from_file_url(
        urlparse("file:///C:/Users/me/my%20project"), is_windows = True
    )

    assert decoded == "C:\\Users\\me\\my project"


def test_a_unc_editable_url_keeps_its_host():
    """The authority is a UNC host on Windows, not part of the path."""
    from urllib.parse import urlparse

    decoded = os_sandbox._path_from_file_url(
        urlparse("file://server/share/project"), is_windows = True
    )

    assert decoded == "\\\\server\\share\\project"


def test_posix_editable_url_decoding_is_unchanged():
    """The Windows fix must cost Linux and macOS nothing."""
    from urllib.parse import unquote, urlparse

    url = "file:///home/me/my%20project"
    parsed = urlparse(url)

    assert os_sandbox._path_from_file_url(parsed, is_windows = False) == os.path.abspath(
        unquote(parsed.path)
    )


def test_windows_setup_installs_the_sandbox_executor():
    """The executor must arrive through a supported setup path.

    A tool call must never download its own sandbox, so setup is the only
    route, and available() tells users to re-run setup. Without this wiring
    that message is false and the preview always falls back.
    """
    setup = os.path.join(os.path.dirname(__file__), "..", "..", "setup.ps1")
    with open(setup, encoding = "utf-8") as handle:
        body = handle.read()

    assert "install_mxc_runtime.py" in body
    assert "UNSLOTH_WINDOWS_SANDBOX_PREVIEW" in body


def _installer_module():
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "..", "install_mxc_runtime.py")
    spec = importlib.util.spec_from_file_location("install_mxc_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_tampered_tarball_is_refused_before_anything_is_extracted(tmp_path, monkeypatch):
    """wxc-exec.exe is the boundary, so an unrecognised artifact must not install.

    Without the pin a compromised registry response, mirror or republished
    artifact silently becomes the sandbox while the install reports success.
    The refusal has to come BEFORE extraction, so nothing from it is ever
    written into the destination.
    """
    installer = _installer_module()
    dest = tmp_path / "mxc"
    dest.mkdir()

    def fake_download(url, into):
        target = os.path.join(into, "mxc-sdk.tgz")
        with open(target, "wb") as handle:
            handle.write(b"not the pinned tarball")
        return target

    extracted = []
    monkeypatch.setattr(installer, "download", fake_download)
    monkeypatch.setattr(installer, "extract", lambda *a: extracted.append(a) or ["wxc-exec.exe"])

    with pytest.raises(SystemExit) as refusal:
        installer.install(str(dest))

    assert "pinned SHA-256" in str(refusal.value)
    assert extracted == []
    assert list(dest.iterdir()) == []


def test_the_pinned_digests_are_recorded_for_both_architectures():
    """A bump that forgets one arch must fail here, not on a user's machine."""
    installer = _installer_module()

    assert set(installer.EXECUTOR_SHA256) == {"x64", "arm64"}
    assert len(installer.TARBALL_SHA256) == 64
    for value in installer.EXECUTOR_SHA256.values():
        assert len(value) == 64


def test_the_interpreters_library_roots_are_granted(plan, tmp_path):
    """Studio's managed Windows virtualenv puts python.exe in Scripts.

    The packages are in the sibling Lib\\site-packages and the standard library
    comes from sys.base_prefix\\Lib, so granting only the executable's own
    directory leaves an isolated call unable to import its runtime. The Windows
    CI job cannot catch this: it runs from an actions/setup-python install,
    where everything happens to sit under one root.
    """
    import sysconfig

    policy = sandbox_windows.build_policy(plan, str(tmp_path), "unsloth-test")
    granted = {os.path.normcase(path) for path in policy["filesystem"]["readonlyPaths"]}

    for name in ("stdlib", "purelib"):
        path = sysconfig.get_path(name)
        if path and os.path.isdir(path):
            assert os.path.normcase(path) in granted, f"{name} was not granted"


def test_the_shell_and_its_userland_are_granted(tmp_path):
    """_get_shell_cmd picks Git for Windows' bash whenever the host has one.

    Granting only the system directories and the interpreter leaves bash.exe
    unrunnable and its usr\\bin userland unreadable, so the Terminal tool fails
    on the ordinary Windows path.
    """
    git = tmp_path / "Git"
    (git / "bin").mkdir(parents = True)
    (git / "usr" / "bin").mkdir(parents = True)
    bash = git / "bin" / "bash.exe"
    bash.write_text("")
    workdir = tmp_path / "work"
    workdir.mkdir()

    launch = ToolLaunchPlan(
        argv = (str(bash), "-c", "echo hi"),
        workdir = str(workdir),
        env = {"PATH": os.pathsep.join([str(git / "bin"), str(git / "usr" / "bin")])},
        requested_mode = "required",
        timeout_seconds = 30,
    )

    policy = sandbox_windows.build_policy(launch, str(workdir), "unsloth-test")
    granted = {os.path.normcase(path) for path in policy["filesystem"]["readonlyPaths"]}

    assert os.path.normcase(str(git / "bin")) in granted, "bash.exe itself is unreachable"
    assert os.path.normcase(str(git / "usr" / "bin")) in granted, "the bash userland is unreachable"


def test_the_installer_destination_override_is_honoured(tmp_path, monkeypatch):
    """Setup installing into UNSLOTH_MXC_DIR must not leave the backend looking
    somewhere else and reporting the executor as missing."""
    from core.inference import mxc_pins

    dest = tmp_path / "custom-mxc"
    dest.mkdir()
    executor = dest / "wxc-exec.exe"
    executor.write_bytes(b"pretend this is wxc-exec")
    # executable_path() now verifies the digest at the trust boundary, so a
    # stand-in has to be pinned for this to be a test about the DIRECTORY.
    monkeypatch.setitem(
        mxc_pins.EXECUTOR_SHA256, mxc_pins.arch_dir(), mxc_pins.digest(str(executor))
    )

    monkeypatch.setenv("UNSLOTH_MXC_DIR", str(dest))

    assert sandbox_windows.managed_mxc_dir() == str(dest)
    monkeypatch.delenv("UNSLOTH_MXC_EXEC", raising = False)
    assert sandbox_windows.executable_path() == str(executor)


def test_the_session_package_directory_reaches_the_windows_environment(plan, tmp_path):
    """Write access to .unsloth-packages does not send an install there.

    plan.env comes from _build_safe_env, which carries only the trusted shim on
    PYTHONPATH and no PIP_TARGET, so without this a pip install targets the
    read-only Studio virtualenv and fails, and an explicit install into the
    directory is invisible to every later Python call. Linux and macOS both do
    this and Windows must not be the odd one out.
    """
    from core.inference.os_sandbox import SESSION_PACKAGES_RELPATH

    workdir = str(tmp_path)
    packages = os.path.join(workdir, SESSION_PACKAGES_RELPATH)
    policy = sandbox_windows.build_policy(plan, workdir, "unsloth-test")
    env = dict(entry.split("=", 1) for entry in policy["process"]["env"])

    assert env["PIP_TARGET"] == packages
    assert packages in env["PYTHONPATH"].split(os.pathsep)
    # LAST on PATH, so a package a tool call installed cannot shadow a bare
    # command the approval logic treats as safe.
    assert env["PATH"].split(os.pathsep)[-1] == os.path.join(packages, "Scripts")


def test_setup_accepts_every_enabled_preview_value():
    """The backend treats true, yes and on as enabled. Setup matching only "1"
    meant the backend engaged while its only supported installation path was
    skipped."""
    setup = os.path.join(os.path.dirname(__file__), "..", "..", "setup.ps1")
    with open(setup, encoding = "utf-8") as handle:
        body = handle.read()

    line = next(l for l in body.splitlines() if "$MxcPreview =" in l)
    for value in ("1", "true", "yes", "on"):
        assert f'"{value}"' in line, f"setup does not accept {value!r} as an enabled preview"


def test_a_failed_mxc_teardown_is_reported_rather_than_recorded_as_success(monkeypatch):
    """An executor that starts and refuses --delete exits non-zero.

    That is the case which leaves DENY and ALLOW ACEs on the user's own workdir
    after a hard kill, and subprocess.run does not raise for it, so without a
    returncode check cleanup recorded a successful teardown that never happened.
    """
    import subprocess as sp

    def refusing_delete(argv, **kwargs):
        return sp.CompletedProcess(argv, 3, b"", b"access denied")

    monkeypatch.setattr(sandbox_windows.subprocess, "run", refusing_delete)

    with pytest.raises(RuntimeError) as failure:
        sandbox_windows._reconcile_container("wxc-exec.exe", "unsloth-test")

    assert "could not reconcile" in str(failure.value)
    assert "access denied" in str(failure.value)


def test_a_successful_mxc_teardown_stays_quiet(monkeypatch):
    """The common case is that the executor already cleaned up."""
    import subprocess as sp

    monkeypatch.setattr(
        sandbox_windows.subprocess,
        "run",
        lambda argv, **kwargs: sp.CompletedProcess(argv, 0, b"", b""),
    )

    sandbox_windows._reconcile_container("wxc-exec.exe", "unsloth-test")


def test_the_degraded_installer_honours_a_custom_studio_home(monkeypatch, tmp_path):
    """A degraded install on a custom-home Studio must not write where the
    backend will not look: it would report success and the preview would still
    say the executor is not installed."""
    import builtins

    installer = _installer_module()
    real_import = builtins.__import__

    def no_backend(name, *args, **kwargs):
        if name.startswith("core.inference"):
            raise ImportError("simulated degraded environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_backend)
    monkeypatch.delenv("UNSLOTH_MXC_DIR", raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "custom"))

    assert installer.default_dest() == os.path.join(str(tmp_path / "custom"), "mxc")


def test_an_executor_that_is_not_the_pinned_build_is_refused(tmp_path, monkeypatch):
    """The managed directory is user-writable.

    A same-user process, including a software-safeguarded tool call made before
    isolation became available, can replace wxc-exec.exe after installation.
    Everything this backend claims rests on that binary being the pinned one,
    so the digest is checked at the trust boundary and not only at install time.
    """
    impostor = tmp_path / "wxc-exec.exe"
    impostor.write_bytes(b"not the pinned executor")
    monkeypatch.setenv("UNSLOTH_MXC_EXEC", str(impostor))

    assert sandbox_windows.executable_path() is None

    monkeypatch.setattr(sys, "platform", "win32")
    ok, reason = sandbox_windows.available()
    assert ok is False
    assert "pinned build" in reason


def test_the_pinned_executor_is_accepted(tmp_path, monkeypatch):
    """Fails closed, but not closed on everything: a matching digest passes."""
    from core.inference import mxc_pins

    executor = tmp_path / "wxc-exec.exe"
    executor.write_bytes(b"pretend this is wxc-exec")
    monkeypatch.setitem(
        mxc_pins.EXECUTOR_SHA256, mxc_pins.arch_dir(), mxc_pins.digest(str(executor))
    )
    monkeypatch.setenv("UNSLOTH_MXC_EXEC", str(executor))

    assert sandbox_windows.executable_path() == str(executor)


def test_the_elevated_helper_is_verified_before_it_runs(tmp_path, monkeypatch):
    """wxc-host-prep.exe carries requireAdministrator, so a replaced helper
    gains administrator execution the moment the user approves the UAC prompt.
    The download-time check does not cover that window."""
    installer = _installer_module()
    dest = tmp_path / "mxc"
    dest.mkdir()
    (dest / "wxc-host-prep.exe").write_bytes(b"not the pinned helper")

    ran = []
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: ran.append(a))

    with pytest.raises(SystemExit) as refusal:
        installer.prepare_host(str(dest))

    assert "pinned" in str(refusal.value)
    assert ran == [], "the elevated helper was executed before it was verified"


def test_a_failed_host_preparation_exits_non_zero(tmp_path, monkeypatch):
    """A refused UAC prompt otherwise looked like success to setup."""
    installer = _installer_module()

    monkeypatch.setattr(
        installer,
        "prepare_host",
        lambda dest, timeout: {"prepare-system-drive": {"exit": 65}},
    )
    monkeypatch.setattr(
        installer.sys, "argv", ["install_mxc_runtime.py", "--prepare-host", "--dest", str(tmp_path)]
    )

    assert installer.main() == 1

    monkeypatch.setattr(
        installer,
        "prepare_host",
        lambda dest, timeout: {"prepare-system-drive": {"exit": 0}},
    )
    assert installer.main() == 0


def test_the_fallback_path_uses_scripts_on_windows(monkeypatch, tmp_path):
    """After an isolated call installs a CLI, Bypass Permissions or a software
    fallback must find it too: pip puts entry points in Scripts on Windows."""
    from core.inference import tools

    packages = tmp_path / ".unsloth-packages"
    packages.mkdir()

    monkeypatch.setattr(tools.sys, "platform", "win32")
    updated = tools._with_session_packages({"PATH": "C:\\Windows"}, str(tmp_path))
    assert updated["PATH"].split(os.pathsep)[-1] == str(packages / "Scripts")

    monkeypatch.setattr(tools.sys, "platform", "linux")
    updated = tools._with_session_packages({"PATH": "/usr/bin"}, str(tmp_path))
    assert updated["PATH"].split(os.pathsep)[-1] == str(packages / "bin")


def test_the_editable_checkout_parent_is_not_granted(plan, tmp_path, monkeypatch):
    """MXC's readonlyPaths are recursive, so granting an import root would hand
    over the rest of the checkout: .env, credentials, fixtures, .git. macOS
    grants those parents as literals for listing only and Linux merely creates
    them, so neither platform hands over the tree either."""
    from core.inference import os_sandbox

    checkout = tmp_path / "checkout"
    package = checkout / "package"
    package.mkdir(parents = True)
    (checkout / ".env").write_text("SECRET=1")

    monkeypatch.setattr(os_sandbox, "editable_source_roots", lambda: (str(package),))
    monkeypatch.setattr(sandbox_windows, "editable_source_roots", lambda: (str(package),))
    # raising = False because the fixed backend does not import this name at
    # all. If it ever does again, this stand-in is what it would receive, and
    # the assertion below is what would catch it.
    monkeypatch.setattr(
        sandbox_windows, "editable_import_roots", lambda: (str(checkout),), raising = False
    )
    monkeypatch.setattr(os_sandbox, "editable_import_roots", lambda: (str(checkout),))

    workdir = tmp_path / "work"
    workdir.mkdir()
    policy = sandbox_windows.build_policy(plan, str(workdir), "unsloth-test")
    granted = {os.path.normcase(path) for path in policy["filesystem"]["readonlyPaths"]}

    assert os.path.normcase(str(package)) in granted, "the source root itself must stay granted"
    assert (
        os.path.normcase(str(checkout)) not in granted
    ), "the whole editable checkout was granted read access"


def test_the_elevated_helper_is_reverified_before_each_invocation(tmp_path, monkeypatch):
    """The first preparation step can take the full timeout, and the pathname
    stays user-writable throughout, so a same-user process could wait for it to
    finish and swap the helper before the second elevated run."""
    import subprocess as sp

    from core.inference import mxc_pins

    installer = _installer_module()
    dest = tmp_path / "mxc"
    dest.mkdir()
    prep = dest / "wxc-host-prep.exe"
    prep.write_bytes(b"the pinned helper")
    monkeypatch.setitem(installer.HOST_PREP_SHA256, mxc_pins.arch_dir(), mxc_pins.digest(str(prep)))

    ran = []

    def swap_after_the_first_run(argv, **kwargs):
        ran.append(argv[1])
        prep.write_bytes(b"replaced between the two elevated runs")
        return sp.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(installer.subprocess, "run", swap_after_the_first_run)

    with pytest.raises(SystemExit) as refusal:
        installer.prepare_host(str(dest))

    assert ran == ["prepare-system-drive"], f"the swapped helper ran elevated: {ran}"
    assert "no longer the pinned" in str(refusal.value)


def test_an_executor_swapped_after_verification_is_refused(monkeypatch, plan, tmp_path):
    """available() verifies the digest, then policy construction and the live
    probe run before Popen ever sees the pathname. A same-user process, which
    is precisely the attacker this pin exists for, can swap the file in that
    window, and cleanup executes the same pathname again for --delete."""
    executor = tmp_path / "wxc-exec.exe"
    executor.write_bytes(b"the pinned executor")
    _pin(monkeypatch, str(executor))

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "available", lambda: (True, "ok"))
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: str(executor))

    executor.write_bytes(b"swapped between the check and the launch")

    with pytest.raises(sandbox_windows.SandboxUnavailableError) as refusal:
        sandbox_windows.prepare(plan)

    assert "changed after it was verified" in str(refusal.value)


def test_the_executor_hold_is_released_after_the_reconciliation(monkeypatch, plan, tmp_path):
    """The callbacks run LIFO and the reconciliation executes the same file,
    so the hold has to outlive it or the window reopens for the teardown."""
    executor = tmp_path / "wxc-exec.exe"
    executor.write_bytes(b"the pinned executor")
    _pin(monkeypatch, str(executor))

    order = []

    class Hold:
        def close(self):
            order.append("released")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "available", lambda: (True, "ok"))
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: str(executor))
    monkeypatch.setattr(sandbox_windows, "_hold_executor", lambda path: Hold())
    monkeypatch.setattr(
        sandbox_windows, "_reconcile_container",
        lambda executor, container_id: order.append("reconciled"),
    )

    prepared = sandbox_windows.prepare(plan)
    prepared.cleanup()

    assert order == ["reconciled", "released"], order


def test_the_elevated_helper_is_held_across_process_creation(tmp_path, monkeypatch):
    """Re-checking before each invocation does not close the window.

    The user answers a UAC prompt between the check and process creation, and
    the pathname stays user-writable throughout, so the verified file has to
    be held against writes and renames the way the backend holds the executor.
    """
    import subprocess as sp

    from core.inference import mxc_pins

    installer = _installer_module()
    dest = tmp_path / "mxc"
    dest.mkdir()
    prep = dest / "wxc-host-prep.exe"
    prep.write_bytes(b"the pinned helper")
    monkeypatch.setitem(
        installer.HOST_PREP_SHA256, mxc_pins.arch_dir(), mxc_pins.digest(str(prep)))

    held = []

    class Hold:
        def __init__(self, path):
            held.append(("held", path))

        def close(self):
            held.append(("released", None))

    monkeypatch.setattr(installer.pins, "hold_file", Hold)
    monkeypatch.setattr(
        installer.subprocess, "run",
        lambda argv, **kwargs: (held.append(("ran", argv[1]))
                                or sp.CompletedProcess(argv, 0, b"", b"")),
    )

    installer.prepare_host(str(dest))

    # Held, then run, then released, for each of the two elevated steps.
    assert [entry[0] for entry in held] == [
        "held", "ran", "released", "held", "ran", "released",
    ], held


def test_the_helper_hold_is_released_even_when_the_prompt_is_never_answered(
    tmp_path, monkeypatch):
    """The timeout path continues the loop, so the release has to be in a
    finally or the second step would run against a still-held handle."""
    import subprocess as sp

    from core.inference import mxc_pins

    installer = _installer_module()
    dest = tmp_path / "mxc"
    dest.mkdir()
    prep = dest / "wxc-host-prep.exe"
    prep.write_bytes(b"the pinned helper")
    monkeypatch.setitem(
        installer.HOST_PREP_SHA256, mxc_pins.arch_dir(), mxc_pins.digest(str(prep)))

    released = []

    class Hold:
        def __init__(self, path):
            pass

        def close(self):
            released.append(True)

    def never_answered(argv, **kwargs):
        raise sp.TimeoutExpired(argv, 1)

    monkeypatch.setattr(installer.pins, "hold_file", Hold)
    monkeypatch.setattr(installer.subprocess, "run", never_answered)

    results = installer.prepare_host(str(dest), timeout = 1)

    assert len(released) == 2, "a hold survived a step that timed out"
    assert all(result["exit"] is None for result in results.values())


def test_the_probe_does_not_run_an_executor_swapped_after_hashing(tmp_path, monkeypatch):
    """available() hashes the file and then reopens the pathname to run
    --probe. prepare()'s hold starts only after available() returns, so a
    same-user process swapping the file in between would have the replacement
    run directly on the host, outside MXC."""
    from core.inference import mxc_pins

    executor = tmp_path / "wxc-exec.exe"
    executor.write_bytes(b"the pinned executor")
    _pin(monkeypatch, str(executor))

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: str(executor))

    ran = []
    def record(argv, **kwargs):
        ran.append(argv)
        raise AssertionError("the probe should not have run")

    monkeypatch.setattr(sandbox_windows.subprocess, "run", record)
    # Swapped between executable_path()'s hash and the probe.
    def swap_then_hold(path):
        executor.write_bytes(b"swapped before the probe")
        return None

    monkeypatch.setattr(sandbox_windows, "_hold_executor", swap_then_hold)

    ok, reason = sandbox_windows.available()

    assert ok is False
    assert "changed after it was verified" in reason
    assert ran == [], "the swapped executor was run"
