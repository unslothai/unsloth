# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The "installed release: ..." line, in both setup twins, against real markers.

Two things this holds. `backend` is absent from every marker written before #8520
(2026-08-13), and reading a missing property is a terminating error under a caller's
Set-StrictMode 2.0+, so an unguarded read aborts setup on those installs. And the twins
must render identical bytes: PowerShell prints @(1, 2) as "1 2", Python as "[1, 2]".

Both functions are sliced out rather than sourced whole, because both scripts run install
steps at load (as tests/studio_setup_ps1/Get-FunctionSource.ps1 and tests/sh/ do).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"
SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"
GET_FUNCTION_SOURCE = PACKAGE_ROOT / "tests" / "studio_setup_ps1" / "Get-FunctionSource.ps1"

requires_pwsh = pytest.mark.skipif(
    shutil.which("pwsh") is None, reason = "pwsh is required to run the setup.ps1 printer"
)


def _usable_bash():
    """A bash that actually runs a command, or None.

    shutil.which("bash") alone is wrong on Windows: C:\\Windows\\System32\\bash.exe is the
    WSL launcher, so it is the first PATH hit even with no distro installed, and running it
    writes "Windows Subsystem for Linux has no installed distributions." to stdout as UTF-16
    -- which this file then compared against a printer line. setup.sh is never executed on
    Windows anyway (_find_setup_script picks setup.ps1 there), so probe before trusting.
    """
    exe = shutil.which("bash")
    if exe is None:
        return None
    try:
        probe = subprocess.run(
            [exe, "-c", "printf ok"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            errors = "replace",
            timeout = 60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return exe if probe.stdout.strip() == "ok" else None


BASH = _usable_bash()
requires_bash = pytest.mark.skipif(
    BASH is None, reason = "a working bash is required to run the setup.sh printer"
)

# 2.0 is where missing-property reads turn fatal; 3.0 and Latest keep that rule.
STRICT_MODES = ["off", "1.0", "2.0", "3.0", "Latest"]

_BASE_MARKER = {
    "published_repo": "unslothai/llama.cpp",
    "release_tag": "b10715-mix-86bd2d3",
    "tag": "b10715",
}
_BASE_LINE = "installed release: unslothai/llama.cpp@b10715-mix-86bd2d3 (tag b10715)"


def _marker(**overrides):
    payload = dict(_BASE_MARKER)
    payload.update(overrides)
    return json.dumps(payload)


def _with_backend(value):
    payload = dict(_BASE_MARKER)
    payload["backend"] = value
    return json.dumps(payload)


def _drop_backend_key():
    return json.dumps(dict(_BASE_MARKER))


# (id, raw file text or None for "no file", expected printed line)
_CASES = [
    # the legacy shape, which is the whole point of the guard
    ("legacy_no_backend_key", _drop_backend_key(), _BASE_LINE),
    ("cuda", _with_backend("cuda"), _BASE_LINE + " -- cuda backend"),
    ("rocm", _with_backend("rocm"), _BASE_LINE + " -- rocm backend"),
    ("vulkan", _with_backend("vulkan"), _BASE_LINE + " -- vulkan backend"),
    ("cpu", _with_backend("cpu"), _BASE_LINE + " -- cpu backend"),
    ("metal", _with_backend("metal"), _BASE_LINE + " -- metal backend"),
    # an unknown name still prints: no allowlist
    ("unknown_future_backend", _with_backend("sycl2"), _BASE_LINE + " -- sycl2 backend"),
    (
        "backend_with_punctuation",
        _with_backend("cuda13.0+x-1"),
        _BASE_LINE + " -- cuda13.0+x-1 backend",
    ),
    ("backend_padded", _with_backend("  vulkan  "), _BASE_LINE + " -- vulkan backend"),
    ("backend_max_length", _with_backend("a" * 32), _BASE_LINE + " -- " + "a" * 32 + " backend"),
    ("backend_null", _with_backend(None), _BASE_LINE),
    ("backend_empty", _with_backend(""), _BASE_LINE),
    ("backend_whitespace", _with_backend("   "), _BASE_LINE),
    ("backend_zero", _with_backend(0), _BASE_LINE),
    ("backend_false", _with_backend(False), _BASE_LINE),
    ("backend_true", _with_backend(True), _BASE_LINE),
    ("backend_int", _with_backend(7), _BASE_LINE),
    ("backend_float", _with_backend(1.5), _BASE_LINE),
    ("backend_empty_list", _with_backend([]), _BASE_LINE),
    ("backend_list", _with_backend(["rocm", "vulkan"]), _BASE_LINE),
    ("backend_empty_object", _with_backend({}), _BASE_LINE),
    ("backend_object", _with_backend({"name": "rocm"}), _BASE_LINE),
    ("backend_too_long", _with_backend("a" * 33), _BASE_LINE),
    ("backend_unicode", _with_backend("vulkän"), _BASE_LINE),
    ("backend_emoji", _with_backend("rocm\U0001f680"), _BASE_LINE),
    ("backend_cjk", _with_backend("显卡"), _BASE_LINE),
    ("backend_1kib", _with_backend("v" * 1024), _BASE_LINE),
    ("backend_64kib", _with_backend("v" * 65536), _BASE_LINE),
    ("backend_with_space", _with_backend("vulkan gpu"), _BASE_LINE),
    ("backend_underscore", _with_backend("windows_rocm"), _BASE_LINE + " -- windows_rocm backend"),
    ("backend_double_quote", _with_backend('vul"kan'), _BASE_LINE),
    ("backend_single_quote", _with_backend("vul'kan"), _BASE_LINE),
    ("backend_backtick", _with_backend("vul`kan"), _BASE_LINE),
    ("backend_ps_subexpression", _with_backend("$(Write-Output pwned)"), _BASE_LINE),
    ("backend_ps_variable", _with_backend("$env:PATH"), _BASE_LINE),
    ("backend_shell_semicolon", _with_backend("vulkan; echo pwned"), _BASE_LINE),
    ("backend_shell_substitution", _with_backend("$(echo pwned)"), _BASE_LINE),
    ("backend_backslash", _with_backend("vul\\kan"), _BASE_LINE),
    ("backend_newline", _with_backend("vulkan\nrocm"), _BASE_LINE),
    ("backend_crlf", _with_backend("vulkan\r\nrocm"), _BASE_LINE),
    ("backend_tab", _with_backend("vul\tkan"), _BASE_LINE),
    ("backend_ansi", _with_backend("\x1b[31mrocm\x1b[0m"), _BASE_LINE),
    ("backend_nul", _with_backend("rocm\x00x"), _BASE_LINE),
    # -ccontains, so a case-folded key is ignored by both rather than only by Python.
    ("backend_key_uppercase", _marker(BACKEND = "vulkan"), _BASE_LINE),
    ("backend_key_mixed_case", _marker(Backend = "vulkan"), _BASE_LINE),
    (
        "no_tag_key",
        json.dumps({"published_repo": "unslothai/llama.cpp", "release_tag": "b10715-mix"}),
        "installed release: unslothai/llama.cpp@b10715-mix",
    ),
    (
        "tag_equals_release_tag",
        json.dumps({"published_repo": "r/l", "release_tag": "b1", "tag": "b1", "backend": "rocm"}),
        "installed release: r/l@b1 -- rocm backend",
    ),
    (
        "upstream_binary_source",
        json.dumps(
            {
                "published_repo": "unslothai/unsloth",
                "release_tag": "b10715-mix",
                "tag": "b10715",
                "source": "ggml-org",
                "binary_repo": "ggml-org/llama.cpp",
                "binary_release_tag": "b10715",
                "backend": "cpu",
            }
        ),
        "installed release: unslothai/unsloth@b10715-mix + ggml-org@b10715 -- cpu backend",
    ),
    ("missing_published_repo", json.dumps({"release_tag": "b1", "backend": "rocm"}), ""),
    ("missing_release_tag", json.dumps({"published_repo": "r/l", "backend": "rocm"}), ""),
    (
        "extra_unknown_keys",
        _marker(backend = "rocm", future_key = {"a": [1, 2]}),
        _BASE_LINE + " -- rocm backend",
    ),
    ("malformed_json", "{not json at all", ""),
    ("truncated_json", '{"published_repo": "r/l", "release_ta', ""),
    ("empty_file", "", ""),
    ("whitespace_file", "   \n  ", ""),
    ("absent_file", None, ""),
]

_IDS = [case[0] for case in _CASES]

# setup.sh has a "+ <source>@<binary_tag>" branch setup.ps1 never has. Pre-existing.
_PS1_EXPECTED_OVERRIDES = {
    "upstream_binary_source": "installed release: unslothai/unsloth@b10715-mix (tag b10715) -- cpu backend",
}
_HISTORICAL_PS1_OVERRIDES = {
    "2026_05_30_61df3aaef": "installed release: unslothai/llama.cpp@b5300-mix (tag b5300)",
}
_KNOWN_TWIN_DIVERGENCES = frozenset(_PS1_EXPECTED_OVERRIDES) | frozenset(_HISTORICAL_PS1_OVERRIDES)


@pytest.fixture
def marker_dir(tmp_path):
    def _write(raw):
        install_dir = tmp_path / "llama.cpp"
        install_dir.mkdir(exist_ok = True)
        if raw is not None:
            (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(raw, encoding = "utf-8")
        return install_dir

    return _write


_PS_HARNESS = """
param([string]$InstallDir, [string]$StrictMode)

if ($StrictMode -ne 'off') {{ Set-StrictMode -Version $StrictMode }}

# Stubbed: the real Get-PathState pulls in half the installer.
function Test-PathQuiet {{
    param([string]$Path, [string]$PathType = "Any")
    return (Test-Path -LiteralPath $Path)
}}

. {get_function_source}
$src = Get-FunctionSource -Path {setup_ps1} -Name Get-InstalledLlamaPrebuiltRelease
if (-not $src) {{ Write-Error "could not slice Get-InstalledLlamaPrebuiltRelease"; exit 2 }}
. ([scriptblock]::Create($src))

$result = Get-InstalledLlamaPrebuiltRelease -InstallDir $InstallDir
if ($null -ne $result) {{ [Console]::Out.Write($result) }}
"""


def _run_ps1_printer(install_dir, strict_mode):
    script = _PS_HARNESS.format(
        get_function_source = f"'{GET_FUNCTION_SOURCE}'",
        setup_ps1 = f"'{SETUP_PS1}'",
    )
    # -File, not -Command: param() only binds named args from a file, and strict mode has
    # to sit in the CALLER's scope.
    script_path = Path(install_dir).parent / f"drive_{strict_mode.replace('.', '_')}.ps1"
    script_path.write_text(script, encoding = "utf-8")
    proc = subprocess.run(
        [
            "pwsh",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(script_path),
            "-InstallDir",
            str(install_dir),
            "-StrictMode",
            strict_mode,
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = 120,
    )
    return proc


_SH_HARNESS = """
set -u
_FUNC_FILE="$1"
_INSTALL_DIR="$2"
# shellcheck disable=SC1090
. "$_FUNC_FILE"
installed_llama_prebuilt_release "$_INSTALL_DIR"
"""


def _sliced_sh_function(tmp_path):
    text = SETUP_SH.read_text(encoding = "utf-8")
    start = text.index("installed_llama_prebuilt_release() {")
    end = text.index("\n}\n", start) + len("\n}\n")
    body = text[start:end]
    assert "UNSLOTH_PREBUILT_INFO.json" in body, "sliced the wrong block out of setup.sh"
    assert body.count("<<'PY'") == 1, "expected exactly one heredoc in the sliced block"
    path = tmp_path / "installed_release_fn.sh"
    path.write_text(body, encoding = "utf-8")
    return path


def _run_sh_printer(install_dir, tmp_path):
    func_file = _sliced_sh_function(tmp_path)
    script = tmp_path / "drive.sh"
    script.write_text(_SH_HARNESS, encoding = "utf-8")
    # The sliced function shells out to `python`; pin it to the test interpreter.
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir(exist_ok = True)
    shim = shim_dir / "python"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding = "utf-8")
    shim.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
    proc = subprocess.run(
        [BASH, str(script), str(func_file), str(install_dir)],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = env,
        timeout = 120,
    )
    return proc


@requires_pwsh
@pytest.mark.parametrize("strict_mode", STRICT_MODES)
@pytest.mark.parametrize(("case_id", "raw", "expected"), _CASES, ids = _IDS)
def test_ps1_printer(marker_dir, strict_mode, case_id, raw, expected):
    expected = _PS1_EXPECTED_OVERRIDES.get(case_id, expected)
    install_dir = marker_dir(raw)
    proc = _run_ps1_printer(install_dir, strict_mode)
    assert proc.returncode == 0, proc.stderr
    assert (
        proc.stdout == expected
    ), f"{case_id} under StrictMode {strict_mode}: {proc.stdout!r} != {expected!r}"
    assert proc.stderr == "", f"{case_id} wrote to the error stream: {proc.stderr!r}"


@requires_pwsh
@pytest.mark.parametrize("strict_mode", ["2.0", "3.0", "Latest"])
def test_a_legacy_marker_does_not_abort_under_strict_mode(marker_dir, strict_mode):
    """The discriminating case: markers between #4562 and #8520 have no backend key, and
    the function's only try/catch wraps ConvertFrom-Json, so the read escapes it.
    """
    install_dir = marker_dir(_drop_backend_key())
    proc = _run_ps1_printer(install_dir, strict_mode)
    assert "PropertyNotFound" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == _BASE_LINE
    assert proc.stderr == ""


@requires_pwsh
@pytest.mark.parametrize(
    "payload",
    [
        "$(Write-Output pwned)",
        "`$(Write-Output pwned)",
        "vulkan; Write-Output pwned",
        "$(1+1)",
    ],
)
def test_ps1_printer_never_evaluates_the_marker(marker_dir, payload):
    install_dir = marker_dir(_with_backend(payload))
    proc = _run_ps1_printer(install_dir, "Latest")
    assert proc.returncode == 0, proc.stderr
    assert "pwned" not in proc.stdout
    assert proc.stdout == _BASE_LINE


@requires_bash
@pytest.mark.parametrize(("case_id", "raw", "expected"), _CASES, ids = _IDS)
def test_sh_printer(marker_dir, tmp_path, case_id, raw, expected):
    install_dir = marker_dir(raw)
    proc = _run_sh_printer(install_dir, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.rstrip("\n") == expected, f"{case_id}: {proc.stdout!r} != {expected!r}"


@requires_bash
@pytest.mark.parametrize(
    "payload", ["vulkan; echo pwned", "$(echo pwned)", "`echo pwned`", "$(id)"]
)
def test_sh_printer_never_evaluates_the_marker(marker_dir, tmp_path, payload):
    install_dir = marker_dir(_with_backend(payload))
    proc = _run_sh_printer(install_dir, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "pwned" not in proc.stdout
    assert proc.stdout.rstrip("\n") == _BASE_LINE


@requires_pwsh
@requires_bash
@pytest.mark.parametrize(("case_id", "raw", "expected"), _CASES, ids = _IDS)
def test_the_two_printers_agree(marker_dir, tmp_path, case_id, raw, expected):
    """Byte-for-byte parity. Without it the twins drift and nobody notices."""
    if case_id in _KNOWN_TWIN_DIVERGENCES:
        pytest.skip("documented pre-existing divergence; see the dedicated test")
    install_dir = marker_dir(raw)
    ps1 = _run_ps1_printer(install_dir, "Latest")
    sh = _run_sh_printer(install_dir, tmp_path)
    assert ps1.returncode == 0, ps1.stderr
    assert sh.returncode == 0, sh.stderr
    assert ps1.stdout == sh.stdout.rstrip(
        "\n"
    ), f"{case_id}: setup.ps1 printed {ps1.stdout!r}, setup.sh printed {sh.stdout!r}"


# Every key set write_prebuilt_metadata has emitted; the 2026-04-01 to 2026-08-13 era is
# on disk today with no backend key.

_HISTORICAL_MARKERS = {
    # no published_repo / release_tag yet, so both printers print nothing.
    "2026_03_25_f4d8a246b": (
        {
            "requested_tag": "b4000",
            "tag": "b4000",
            "asset": "linux-cuda",
            "source": "unsloth",
            "bundle_profile": "full",
            "runtime_line": "cuda",
            "coverage_class": "broad",
            "prebuilt_fallback_used": False,
            "installed_at_utc": "2026-03-25T00:00:00Z",
        },
        "",
    ),
    "2026_04_01_428efc7d9": (
        {
            "requested_tag": "b4100",
            "tag": "b4100",
            "release_tag": "b4100-mix",
            "published_repo": "unslothai/llama.cpp",
            "asset": "linux-cuda",
            "asset_sha256": "0" * 64,
            "source": "unsloth",
            "source_sha256": "1" * 64,
            "source_commit": "a" * 40,
            "install_fingerprint": "fp",
            "bundle_profile": "full",
            "runtime_line": "cuda",
            "coverage_class": "broad",
            "prebuilt_fallback_used": False,
            "installed_at_utc": "2026-04-01T00:00:00Z",
        },
        "installed release: unslothai/llama.cpp@b4100-mix (tag b4100)",
    ),
    "2026_04_02_1ce8a8e7c": (
        {
            "requested_tag": "b4200",
            "tag": "b4200",
            "release_tag": "b4200-mix",
            "published_repo": "unslothai/llama.cpp",
            "asset": "linux-vulkan",
            "source_asset": "src.tar.gz",
            "source_commit_short": "aaaaaaa",
            "source_repo": "ggml-org/llama.cpp",
            "source_repo_url": "https://x",
            "source_ref_kind": "tag",
            "requested_source_ref": "b4200",
            "resolved_source_ref": "b4200",
            "installed_at_utc": "2026-04-02T00:00:00Z",
        },
        "installed release: unslothai/llama.cpp@b4200-mix (tag b4200)",
    ),
    "2026_05_30_61df3aaef": (
        {
            "tag": "b5300",
            "release_tag": "b5300-mix",
            "published_repo": "unslothai/llama.cpp",
            "source": "ggml-org",
            "binary_repo": "ggml-org/llama.cpp",
            "binary_release_tag": "b5300",
            "installed_at_utc": "2026-05-30T00:00:00Z",
        },
        "installed release: unslothai/llama.cpp@b5300-mix + ggml-org@b5300",
    ),
    "2026_07_20_cf912cbd8": (
        {
            "tag": "b6900",
            "release_tag": "b6900-mix",
            "published_repo": "unslothai/llama.cpp",
            "force_cpu": True,
            "asset": "linux-cpu",
            "installed_at_utc": "2026-07-20T00:00:00Z",
        },
        "installed release: unslothai/llama.cpp@b6900-mix (tag b6900)",
    ),
    # llama_backend is the pre-#8520 field, not the key the printer reads.
    "2026_07_27_7917c7828": (
        {
            "tag": "b7100",
            "release_tag": "b7100-mix",
            "published_repo": "unslothai/llama.cpp",
            "llama_backend": "vulkan",
            "installed_at_utc": "2026-07-27T00:00:00Z",
        },
        "installed release: unslothai/llama.cpp@b7100-mix (tag b7100)",
    ),
    "2026_08_04_9b452cb3b": (
        {
            "tag": "b7400",
            "release_tag": "b7400-mix",
            "published_repo": "unslothai/llama.cpp",
            "ggml_tree": "abc123",
            "installed_at_utc": "2026-08-04T00:00:00Z",
        },
        "installed release: unslothai/llama.cpp@b7400-mix (tag b7400)",
    ),
    "2026_08_08_738413ab0": (
        {
            "tag": "b7600",
            "release_tag": "b7600-mix",
            "published_repo": "unslothai/llama.cpp",
            "llama_backend": "auto",
            "rocm_gfx": "gfx1151",
            "installed_at_utc": "2026-08-08T00:00:00Z",
        },
        "installed release: unslothai/llama.cpp@b7600-mix (tag b7600)",
    ),
    "2026_08_13_5426a78c3": (
        {
            "tag": "b7900",
            "release_tag": "b7900-mix",
            "published_repo": "unslothai/llama.cpp",
            "backend": "vulkan",
            "backend_request": "auto",
            "installed_at_utc": "2026-08-13T00:00:00Z",
        },
        "installed release: unslothai/llama.cpp@b7900-mix (tag b7900) -- vulkan backend",
    ),
    # backend_for_install_kind() returns None for an unknown kind, landing as JSON null.
    "backend_written_as_null": (
        {
            "tag": "b7900",
            "release_tag": "b7900-mix",
            "published_repo": "unslothai/llama.cpp",
            "backend": None,
            "backend_request": "auto",
            "installed_at_utc": "2026-08-13T00:00:00Z",
        },
        "installed release: unslothai/llama.cpp@b7900-mix (tag b7900)",
    ),
    "2026_08_13_5a3e9fc7a": (
        {
            "tag": "b8000",
            "release_tag": "b8000-mix",
            "published_repo": "unslothai/llama.cpp",
            "backend": "rocm",
            "gfx_target": "gfx1151",
            "mapped_targets": ["gfx1151", "gfx1200"],
            "installed_at_utc": "2026-08-13T00:00:00Z",
        },
        "installed release: unslothai/llama.cpp@b8000-mix (tag b8000) -- rocm backend",
    ),
    "2026_08_18_9d1dcfe58": (
        {
            "tag": "b8300",
            "release_tag": "b8300-mix",
            "published_repo": "unslothai/llama.cpp",
            "backend": "cuda",
            "supported_sms": ["90", "100"],
            "installed_at_utc": "2026-08-18T00:00:00Z",
        },
        "installed release: unslothai/llama.cpp@b8300-mix (tag b8300) -- cuda backend",
    ),
    # the current shape.
    "2026_08_31_1400031e2_current": (
        {
            "requested_tag": "b10715",
            "tag": "b10715",
            "release_tag": "b10715-mix-86bd2d3",
            "published_repo": "unslothai/llama.cpp",
            "asset": "windows-vulkan",
            "force_cpu": False,
            "llama_backend": "vulkan",
            "backend": "vulkan",
            "backend_request": "auto",
            "asset_sha256": "0" * 64,
            "runtime_asset": "runtime.zip",
            "source": "unsloth",
            "binary_repo": "unslothai/llama.cpp",
            "binary_release_tag": "b10715-mix",
            "source_asset": "src.tar.gz",
            "source_sha256": "1" * 64,
            "source_commit": "a" * 40,
            "source_commit_short": "aaaaaaa",
            "source_repo": "ggml-org/llama.cpp",
            "source_repo_url": "https://x",
            "source_ref_kind": "tag",
            "requested_source_ref": "b10715",
            "resolved_source_ref": "b10715",
            "ggml_tree": "abc123",
            "bundle_profile": "full",
            "runtime_line": "vulkan",
            "coverage_class": "broad",
            "gfx_target": "",
            "mapped_targets": [],
            "supported_sms": [],
            "install_fingerprint": "fp",
            "prebuilt_fallback_used": False,
            "installed_at_utc": "2026-08-31T00:00:00Z",
        },
        "installed release: unslothai/llama.cpp@b10715-mix-86bd2d3 (tag b10715) -- vulkan backend",
    ),
}


@requires_pwsh
@pytest.mark.parametrize("strict_mode", STRICT_MODES)
@pytest.mark.parametrize("shape_id", sorted(_HISTORICAL_MARKERS))
def test_every_historical_marker_shape_on_ps1(marker_dir, strict_mode, shape_id):
    payload, expected = _HISTORICAL_MARKERS[shape_id]
    expected = _HISTORICAL_PS1_OVERRIDES.get(shape_id, expected)
    install_dir = marker_dir(json.dumps(payload, indent = 2) + "\n")
    proc = _run_ps1_printer(install_dir, strict_mode)
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == "", f"{shape_id} under {strict_mode}: {proc.stderr!r}"
    assert proc.stdout == expected, f"{shape_id} under {strict_mode}"


@requires_bash
@pytest.mark.parametrize("shape_id", sorted(_HISTORICAL_MARKERS))
def test_every_historical_marker_shape_on_sh(marker_dir, tmp_path, shape_id):
    payload, expected = _HISTORICAL_MARKERS[shape_id]
    install_dir = marker_dir(json.dumps(payload, indent = 2) + "\n")
    proc = _run_sh_printer(install_dir, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.rstrip("\n") == expected, shape_id


@requires_pwsh
@requires_bash
@pytest.mark.parametrize("shape_id", sorted(_HISTORICAL_MARKERS))
def test_every_historical_marker_shape_agrees(marker_dir, tmp_path, shape_id):
    if shape_id in _KNOWN_TWIN_DIVERGENCES:
        pytest.skip("documented pre-existing divergence; see the dedicated test")
    payload, _expected = _HISTORICAL_MARKERS[shape_id]
    install_dir = marker_dir(json.dumps(payload, indent = 2) + "\n")
    ps1 = _run_ps1_printer(install_dir, "Latest")
    sh = _run_sh_printer(install_dir, tmp_path)
    assert ps1.stdout == sh.stdout.rstrip("\n"), shape_id


def test_the_ps1_guard_checks_property_existence():
    text = SETUP_PS1.read_text(encoding = "utf-8")
    start = text.index("function Get-InstalledLlamaPrebuiltRelease")
    body = text[start : text.index("\nfunction ", start + 1)]
    assert "PSObject.Properties.Name -ccontains 'backend'" in body, (
        "the backend read must stay guarded: a bare $payload.backend is a terminating "
        "error under a caller's Set-StrictMode for every marker written before #8520"
    )


def test_both_printers_share_one_backend_shape_rule():
    shape = "[A-Za-z0-9._+-]{1,32}"
    assert shape in SETUP_PS1.read_text(encoding = "utf-8")
    assert shape in SETUP_SH.read_text(encoding = "utf-8")


def test_the_sh_printer_only_accepts_a_string():
    text = SETUP_SH.read_text(encoding = "utf-8")
    assert (
        "isinstance(_backend_raw, str)" in text
    ), "str() on a non-string diverges from the PowerShell twin"


@requires_pwsh
@requires_bash
def test_the_upstream_source_branch_is_a_known_pre_existing_divergence(marker_dir, tmp_path):
    """Recorded, not fixed: teaching setup.ps1 this branch changes what Windows prints
    for upstream bundles, which is a separate change from naming the backend.
    """
    raw = json.dumps(
        {
            "published_repo": "unslothai/unsloth",
            "release_tag": "b10715-mix",
            "tag": "b10715",
            "source": "ggml-org",
            "binary_repo": "ggml-org/llama.cpp",
            "binary_release_tag": "b10715",
            "backend": "cpu",
        }
    )
    install_dir = marker_dir(raw)
    ps1 = _run_ps1_printer(install_dir, "Latest")
    sh = _run_sh_printer(install_dir, tmp_path)
    assert ps1.stdout == (
        "installed release: unslothai/unsloth@b10715-mix (tag b10715) -- cpu backend"
    )
    assert sh.stdout.rstrip("\n") == (
        "installed release: unslothai/unsloth@b10715-mix + ggml-org@b10715 -- cpu backend"
    )
    assert ps1.stdout.endswith(" -- cpu backend")
    assert sh.stdout.rstrip("\n").endswith(" -- cpu backend")
