# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Windows signing binary is executed with the release signing secrets.

test_release_desktop_signing.py asserts on the workflow text; this file runs the
two step bodies, so a change that reads fine but stops failing closed is caught.
The bodies are extracted, never retyped, then wrapped and invoked the way the
runner does (ScriptHandlerHelpers.cs prepends `$ErrorActionPreference = 'stop'`
and appends the $LASTEXITCODE propagation, then runs `pwsh -command ". '<f>'"`).
That wrapper is what makes `exit 1` mean "the release stops" here too.

A local HTTP server stands in for the release asset, so tampered, truncated and
unreachable downloads are deterministic and offline.

Needs pwsh. Checks that need a bare name to resolve to a `.exe` need Windows
PATHEXT; UNSLOTH_NETWORK_TESTS=1 also pulls the real pinned asset.
"""

import functools
import hashlib
import http.server
import os
import pathlib
import shutil
import sys
import threading

import pytest
import yaml

from unsloth_pwsh_runner import run_pwsh


# Only PATHEXT maps a bare name onto the `.exe` the verify step insists on, so checks needing a *passing* resolution
needs_pathext = pytest.mark.skipif(
    sys.platform != "win32",
    reason = "needs Windows PATHEXT resolution to map a bare name onto the .exe",
)

# Opt in, so an offline or rate limited run does not fail on what the digest already pins.
needs_network = pytest.mark.skipif(
    not os.environ.get("UNSLOTH_NETWORK_TESTS"),
    reason = "set UNSLOTH_NETWORK_TESTS=1 to fetch the pinned release asset",
)

pytestmark = pytest.mark.skipif(
    shutil.which("pwsh") is None,
    reason = "pwsh is required to execute the Windows step bodies",
)


REPO = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "release-desktop.yml"

RUNNER_PREPEND = "$ErrorActionPreference = 'stop'"
RUNNER_APPEND = r"if ((Test-Path -LiteralPath variable:\LASTEXITCODE)) { exit $LASTEXITCODE }"

WINDOWS_GUARD = "matrix.platform == 'windows-latest'"


@functools.lru_cache(maxsize = 1)
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def step(name):
    return next(s for s in workflow()["jobs"]["build"]["steps"] if s.get("name") == name)


@functools.lru_cache(maxsize = 1)
def pinned():
    env = step("Install trusted-signing-cli")["env"]
    return env["TRUSTED_SIGNING_CLI_URL"], env["TRUSTED_SIGNING_CLI_SHA256"]


def write_step_script(directory, name, filename):
    body = step(name)["run"]
    path = pathlib.Path(directory) / filename
    path.write_text(f"{RUNNER_PREPEND}\n{body}\n{RUNNER_APPEND}\n", encoding = "utf-8")
    return path


def run_step(script, env):
    """Invoke a step script the way the runner does and return (code, output)."""
    full = {**os.environ, **env}
    # Every signing check in this file reads the exit code of this one call, so an interpreter that aborts at startup
    result = run_pwsh(
        ["pwsh", "-NoProfile", "-Command", f". '{script}'"],
        capture_output = True,
        text = True,
        env = full,
        timeout = 300,
    )
    return result.returncode, result.stdout + result.stderr


# ── a local stand-in for the GitHub release asset ────────────────────────────
class _Handler(http.server.BaseHTTPRequestHandler):
    payload = b""
    truncate = False
    status = 200

    def do_GET(self):
        if self.status != 200:
            self.send_error(self.status)
            return
        body = self.payload[: len(self.payload) // 2] if self.truncate else self.payload
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope = "session")
def asset_server():
    handler = _Handler
    handler.payload = b"MZ" + b"\x00" * 4094 + b"pretend trusted-signing-cli"
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/trusted-signing-cli.exe", handler
    server.shutdown()


@pytest.fixture
def sandbox(tmp_path):
    install = write_step_script(tmp_path, "Install trusted-signing-cli", "install.ps1")
    verify = write_step_script(tmp_path, "Verify trusted-signing-cli", "verify.ps1")
    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir()
    github_path = tmp_path / "github_path"
    github_path.write_text("", encoding = "utf-8")
    return {
        "install": install,
        "verify": verify,
        "runner_temp": runner_temp,
        "github_path": github_path,
        "env": {
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_PATH": str(github_path),
        },
    }


def good_digest(handler):
    return hashlib.sha256(handler.payload).hexdigest()


def install_env(sandbox, url, digest):
    return {
        **sandbox["env"],
        "TRUSTED_SIGNING_CLI_URL": url,
        "TRUSTED_SIGNING_CLI_SHA256": digest,
    }


def installed_binary(sandbox):
    return sandbox["runner_temp"] / "trusted-signing-cli" / "trusted-signing-cli.exe"


# ── static contracts on the PR ───────────────────────────────────────────────
def test_pin_is_a_full_sha256_and_a_versioned_url():
    url, digest = pinned()
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    assert url.startswith("https://")
    assert "/releases/download/0.10.0/" in url


def test_no_cache_step_restores_the_signing_binary():
    for s in workflow()["jobs"]["build"]["steps"]:
        if str(s.get("uses", "")).startswith("actions/cache"):
            assert "trusted-signing-cli" not in yaml.safe_dump(s), s.get("name")


def test_the_removed_steps_are_gone():
    names = [s.get("name") for s in workflow()["jobs"]["build"]["steps"]]
    assert "Restore trusted-signing-cli" not in names
    assert "Add cargo bin to PATH" not in names


def test_every_signing_cli_step_stays_windows_only():
    for s in workflow()["jobs"]["build"]["steps"]:
        if "trusted-signing-cli" in yaml.safe_dump(s):
            assert s.get("if") == WINDOWS_GUARD, s.get("name")


def test_other_platform_integrity_paths_are_untouched():
    names = [s.get("name") for s in workflow()["jobs"]["build"]["steps"]]
    assert "Pin complete AppImage toolchain" in names
    assert "Import Apple certificate" in names


def test_the_install_step_never_interpolates_workflow_expressions():
    # `${{ }}` in a run body is a shell-injection surface;
    body = step("Install trusted-signing-cli")["run"]
    assert "${{" not in body


# ── the happy path ───────────────────────────────────────────────────────────
def test_a_matching_digest_installs_and_publishes_the_path(sandbox, asset_server):
    url, handler = asset_server
    code, out = run_step(sandbox["install"], install_env(sandbox, url, good_digest(handler)))

    assert code == 0, out
    assert installed_binary(sandbox).is_file()
    assert installed_binary(sandbox).read_bytes() == handler.payload
    assert sandbox["github_path"].read_text().strip().endswith("trusted-signing-cli")
    assert "verified trusted-signing-cli sha256=" in out


def test_the_digest_check_is_case_insensitive(sandbox, asset_server):
    url, handler = asset_server
    code, out = run_step(
        sandbox["install"],
        install_env(sandbox, url, good_digest(handler).upper()),
    )
    assert code == 0, out


def test_installing_twice_is_idempotent(sandbox, asset_server):
    url, handler = asset_server
    env = install_env(sandbox, url, good_digest(handler))
    first = run_step(sandbox["install"], env)
    second = run_step(sandbox["install"], env)

    assert first[0] == 0, first[1]
    assert second[0] == 0, second[1]
    assert sandbox["github_path"].read_text().count("trusted-signing-cli") == 2


def test_a_path_containing_spaces_still_works(tmp_path, asset_server):
    url, handler = asset_server
    spaced = tmp_path / "Program Files" / "runner temp"
    spaced.mkdir(parents = True)
    install = write_step_script(tmp_path, "Install trusted-signing-cli", "install.ps1")
    github_path = tmp_path / "github_path"
    github_path.write_text("", encoding = "utf-8")

    code, out = run_step(
        install,
        {
            "RUNNER_TEMP": str(spaced),
            "GITHUB_PATH": str(github_path),
            "TRUSTED_SIGNING_CLI_URL": url,
            "TRUSTED_SIGNING_CLI_SHA256": good_digest(handler),
        },
    )

    assert code == 0, out
    assert (spaced / "trusted-signing-cli" / "trusted-signing-cli.exe").is_file()


# ── every way it must fail closed ────────────────────────────────────────────
def test_a_tampered_asset_fails_the_release(sandbox, asset_server):
    url, _ = asset_server
    code, out = run_step(sandbox["install"], install_env(sandbox, url, "0" * 64))

    assert code != 0
    assert "digest mismatch" in out
    assert not installed_binary(sandbox).exists(), "a rejected binary was left on disk"
    assert sandbox["github_path"].read_text().strip() == "", "PATH was published anyway"


def test_a_truncated_download_fails_the_release(sandbox, asset_server):
    url, handler = asset_server
    handler.truncate = True
    try:
        code, out = run_step(sandbox["install"], install_env(sandbox, url, good_digest(handler)))
    finally:
        handler.truncate = False

    assert code != 0
    assert "digest mismatch" in out
    assert not installed_binary(sandbox).exists()


def test_an_unreachable_asset_fails_the_release(sandbox, asset_server):
    url, handler = asset_server
    handler.status = 404
    try:
        code, out = run_step(sandbox["install"], install_env(sandbox, url, good_digest(handler)))
    finally:
        handler.status = 200

    assert code != 0
    assert sandbox["github_path"].read_text().strip() == ""


def test_a_dead_host_fails_the_release(sandbox):
    code, out = run_step(
        sandbox["install"],
        install_env(sandbox, "http://127.0.0.1:9/nope.exe", "0" * 64),
    )
    assert code != 0
    assert sandbox["github_path"].read_text().strip() == ""


def test_an_empty_digest_pin_cannot_pass(sandbox, asset_server):
    # A cleared or mistyped pin must not degrade into "accept anything".
    url, _ = asset_server
    code, out = run_step(sandbox["install"], install_env(sandbox, url, ""))
    assert code != 0
    assert not installed_binary(sandbox).exists()


# ── the verify step ──────────────────────────────────────────────────────────
def _fake_on_path(directory, name, script):
    directory.mkdir(parents = True, exist_ok = True)
    target = directory / name
    target.write_text(script, encoding = "utf-8")
    target.chmod(0o755)
    return target


def _real_exe_on_path(directory, source):
    """Put a genuinely runnable executable at the verified name.

    A shebang script cannot stand in: PATHEXT resolves a bare name only to a real
    `.exe`, which is also what the verify step compares against. The branch under
    test decides which real binary is borrowed.
    """
    directory.mkdir(parents = True, exist_ok = True)
    target = directory / "trusted-signing-cli.exe"
    shutil.copy2(source, target)
    return target


def _path_with(*directories):
    return os.pathsep.join([*(str(d) for d in directories), os.environ["PATH"]])


def test_verify_rejects_a_binary_that_was_never_digest_checked(sandbox, tmp_path):
    # The rust-cache scenario: an identically named copy earlier on PATH that no digest gate saw.
    decoy_dir = tmp_path / "cargo_bin"
    _fake_on_path(
        decoy_dir, "trusted-signing-cli", '#!/bin/sh\necho "trusted-signing-cli 0.10.0"\n'
    )

    code, out = run_step(
        sandbox["verify"],
        {
            **sandbox["env"],
            "PATH": _path_with(decoy_dir),
        },
    )

    assert code != 0
    assert "not the verified" in out


def test_verify_fails_when_nothing_is_on_path(sandbox):
    code, out = run_step(sandbox["verify"], sandbox["env"])
    assert code != 0
    assert "not on PATH" in out


@needs_pathext
def test_verify_fails_when_the_binary_cannot_start(sandbox):
    directory = sandbox["runner_temp"] / "trusted-signing-cli"
    directory.mkdir(parents = True)
    (directory / "trusted-signing-cli.exe").write_bytes(b"\x00\x01not an executable")

    code, out = run_step(sandbox["verify"], {**sandbox["env"], "PATH": _path_with(directory)})

    assert code != 0
    assert "could not be started" in out, out


@needs_pathext
def test_verify_fails_when_the_binary_exits_non_zero(sandbox):
    directory = sandbox["runner_temp"] / "trusted-signing-cli"
    _real_exe_on_path(directory, pathlib.Path(os.environ["SystemRoot"], "System32", "where.exe"))

    code, out = run_step(sandbox["verify"], {**sandbox["env"], "PATH": _path_with(directory)})

    assert code != 0
    assert "is on PATH but exited" in out, out


@needs_pathext
def test_verify_accepts_the_binary_it_installed(sandbox):
    # Starts fine, exits non-zero: where.exe given an option it does not take.
    directory = sandbox["runner_temp"] / "trusted-signing-cli"
    verified = _real_exe_on_path(directory, sys.executable)

    code, out = run_step(sandbox["verify"], {**sandbox["env"], "PATH": _path_with(directory)})

    assert code == 0, out
    assert str(verified) in out, out


def test_an_unverified_copy_ahead_on_path_is_rejected(sandbox, tmp_path):
    # verified one must not be accepted. Prepending is what prevents it.
    # Rejecting direction of the ordering proof:
    verified_dir = sandbox["runner_temp"] / "trusted-signing-cli"
    _fake_on_path(
        verified_dir, "trusted-signing-cli.exe", '#!/bin/sh\necho "trusted-signing-cli 0.10.0"\n'
    )
    decoy_dir = tmp_path / "cargo_bin"
    _fake_on_path(
        decoy_dir, "trusted-signing-cli", '#!/bin/sh\necho "trusted-signing-cli 0.10.0"\n'
    )

    code, out = run_step(
        sandbox["verify"],
        {
            **sandbox["env"],
            "PATH": _path_with(decoy_dir, verified_dir),
        },
    )

    assert code != 0, "an unverified copy ahead on PATH was accepted"
    assert "not the verified" in out


def test_the_install_step_publishes_its_directory_for_path_prepending(sandbox, asset_server):
    # $GITHUB_PATH prepends, so publishing the install dir is what puts the verified copy ahead of ~/.cargo/bin.
    url, handler = asset_server
    assert run_step(sandbox["install"], install_env(sandbox, url, good_digest(handler)))[0] == 0
    published = sandbox["github_path"].read_text().strip()
    assert published == str(sandbox["runner_temp"] / "trusted-signing-cli")


@needs_network
def test_the_real_pinned_asset_matches_its_digest(sandbox):
    url, digest = pinned()
    code, out = run_step(sandbox["install"], install_env(sandbox, url, digest))

    assert code == 0, out
    downloaded = installed_binary(sandbox)
    assert hashlib.sha256(downloaded.read_bytes()).hexdigest() == digest
    assert downloaded.read_bytes()[:2] == b"MZ", "not a Windows executable"


@needs_network
def test_the_real_asset_is_a_64_bit_windows_console_binary(sandbox):
    url, digest = pinned()
    assert run_step(sandbox["install"], install_env(sandbox, url, digest))[0] == 0
    data = installed_binary(sandbox).read_bytes()
    offset = int.from_bytes(data[0x3C:0x40], "little")
    assert data[offset : offset + 4] == b"PE\x00\x00"
    assert int.from_bytes(data[offset + 4 : offset + 6], "little") == 0x8664, "not x86-64"


@needs_network
def test_the_real_asset_declares_the_arguments_the_signing_script_passes(sandbox):
    # sign-with-trusted-signing.ps1 passes -e and -d;
    url, digest = pinned()
    assert run_step(sandbox["install"], install_env(sandbox, url, digest))[0] == 0
    blob = installed_binary(sandbox).read_bytes()
    for token in (
        b"endpoint",
        b"description",
        b"AZURE_TRUSTED_SIGNING_ACCOUNT_NAME",
        b"AZURE_CERTIFICATE_PROFILE_NAME",
        b"AZURE_CLIENT_SECRET",
        b"0.10.0",
    ):
        assert token in blob, token


def test_the_signing_script_still_calls_the_tool_by_bare_name():
    # Only reads a file in the tree.
    script = REPO / "studio" / "src-tauri" / "windows" / "sign-with-trusted-signing.ps1"
    text = script.read_text(encoding = "utf-8")
    assert "& trusted-signing-cli @trustedSigningArgs" in text
