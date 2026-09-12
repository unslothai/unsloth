# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared plumbing for the Windows-on-ARM installer tests.

install.ps1 and studio/setup.ps1 cannot dot-source each other and cannot be imported, so a
behavioural test lifts a function body or a live block out of the script under test, pastes
it into a bare pwsh session with the few stubs that block needs, and asserts on what it
writes out. The sources, the lifting and the stub bundles are the same from one test to the
next, so they live here rather than being restated in every module that wants them.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST_PY = PACKAGE_ROOT / "studio" / "install_manifest.py"
SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"
STACK_PY = PACKAGE_ROOT / "studio" / "install_python_stack.py"
STACK_LLAMA = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"

# Read once. Well over a hundred tests want one of these whole files, and none of them
# mutate what they read.
INSTALL_SRC = INSTALL_PS1.read_text(encoding = "utf-8")
SETUP_SRC = SETUP_PS1.read_text(encoding = "utf-8")
STACK_SRC = STACK_PY.read_text(encoding = "utf-8")
LLAMA_SRC = STACK_LLAMA.read_text(encoding = "utf-8")
CONSTRAINTS_SRC = (
    PACKAGE_ROOT / "studio" / "backend" / "requirements" / "single-env" / "constraints.txt"
).read_text(encoding = "utf-8")

PWSH = shutil.which("pwsh")
requires_pwsh = pytest.mark.skipif(PWSH is None, reason = "pwsh not available")

# The channels, mirrors and indexes these tests name over and over. Spelled once so a
# parametrize row stays one readable line instead of six.
NV_GA = "https://pypi.nvidia.com/nvtorch_oot"
NV_NIGHTLY = "https://pypi.nvidia.com/nvtorch_oot_nightly"
PYPI = "https://pypi.org/simple"
CORP_INDEX = "https://pypi.corp.test/simple"


def _script(*lines: str) -> str:
    """The lines of a PowerShell snippet, joined the way pwsh -Command wants them."""
    return "\n".join(lines)


def _ps(
    script,
    timeout = 120,
    **kwargs,
):
    """Run a PowerShell snippet and hand back the completed process."""
    return subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        timeout = timeout,
        **kwargs,
    )


def _ps_ok(
    script,
    timeout = 120,
    **kwargs,
):
    """Same, but fail the test with PowerShell's own stderr when it does not exit 0."""
    done = _ps(script, timeout = timeout, **kwargs)
    assert done.returncode == 0, done.stderr
    return done


def _ps_last(script, **kwargs) -> str:
    """Run a snippet that must succeed and hand back its last line of output.

    Every one of these scripts ends in a Write-Output the assertion is about, and a
    PowerShell prelude can print before it, so the last line is the answer.
    """
    return _ps_ok(script, **kwargs).stdout.strip().splitlines()[-1]


def _ps_kv(script, **kwargs) -> dict:
    """A snippet whose Write-Outputs are NAME=value lines, read back as a dict."""
    return dict(l.split("=", 1) for l in _ps_ok(script, **kwargs).stdout.splitlines() if "=" in l)


def _function_source(text: str, name: str) -> str:
    """Extract a PowerShell function by matching balanced braces."""
    match = re.search(rf"(?im)^[ \t]*function[ \t]+{re.escape(name)}\b", text)
    assert match, f"{name} is not defined in setup.ps1"
    start = text.index("{", match.start())
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[match.start() : index + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def functions(text: str, *names: str) -> str:
    """Several bodies in one lift, in the order given.

    PowerShell does not hoist and a helper the prelude does not lift is a
    command-not-found, not a false answer, so a body's helpers come with it.
    """
    return "\n".join(_function_source(text, name) for name in names)


def _ps_function(path: pathlib.Path, name: str) -> str:
    """A function body by SCRIPT PATH, for the tests that parametrize over the two files."""
    src = {INSTALL_PS1: INSTALL_SRC, SETUP_PS1: SETUP_SRC}.get(path)
    return _function_source(src if src is not None else path.read_text(encoding = "utf-8"), name)


def _ps_copies(name: str) -> tuple:
    """install.ps1's and setup.ps1's copies of one function, bar comments and indentation.

    Neither script can dot-source the other, so each is carried twice and the parity is
    pinned instead. Returned as a pair rather than compared here so a mismatch fails with
    pytest's own diff of the two bodies.
    """

    def normalized(source: str) -> str:
        lines = [
            line.rstrip()
            for line in _function_source(source, name).splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        indent = min(len(line) - len(line.lstrip()) for line in lines)
        return "\n".join(line[indent:] for line in lines)

    return normalized(INSTALL_SRC), normalized(SETUP_SRC)


def slice_between(
    src: str,
    start_marker: str,
    end_marker: str,
    *,
    include_end = False,
) -> str:
    """The live text from one marker to the next, sliced out of the script itself.

    Restating a block in the test instead means a copy that passes forever after the
    original stopped matching it, which is the failure these tests exist to catch.
    """
    start = src.index(start_marker)
    end = src.index(end_marker, start)
    return src[start : end + len(end_marker)] if include_end else src[start:end]


# ── Stub bundles ──────────────────────────────────────────────────────────────────────────

SUBSTEP_NOOP = "function substep { param($m, $c) }"


def substep_collector(var: str = "Messages") -> str:
    """substep, collected into $script:<var> so a test can assert on what was said."""
    return _script(
        f"$script:{var} = @()",
        f"function substep {{ param($m, $c) $script:{var} += $m }}",
    )


# The real one shortens a path to its 8.3 form; a passthrough is what every test that is
# not about 8.3 wants, and it keeps the assertions readable.
UV_SAFE_PATH = "function Get-UvSafePath { param([string]$Path) return $Path }"

JOIN_URL_PATH = _script(
    "function Join-UrlPath { param([string]$Base,[string]$Path)",
    "  return ($Base.TrimEnd('/') + '/' + $Path.TrimStart('/')) }",
)
JOIN_URL_RETURNS_BASE = "function Join-UrlPath { param($Base, $Path) return $Base }"
JOIN_URL_RETURNS_PATH = "function Join-UrlPath { param($Base, $Path) return $Path }"


def invoke_restmethod(body: str) -> str:
    """Invoke-RestMethod serving one synthetic PEP 503 page, so the probes are offline."""
    return (
        "function Invoke-RestMethod { param([Parameter(ValueFromRemainingArguments=$true)]$a) "
        f"return @'\n{body}\n'@ }}"
    )


INVOKE_RESTMETHOD_OFFLINE = (
    "function Invoke-RestMethod { param([Parameter(ValueFromRemainingArguments=$true)]$a) "
    "throw 'offline' }"
)
INVOKE_RESTMETHOD_NO_NETWORK = "function Invoke-RestMethod { throw 'no network in this test' }"

# The resolver settings a lifted block must not inherit from the session running the tests.
UV_INDEX_ENV = (
    "UV_NO_INDEX",
    "PIP_NO_INDEX",
    "UV_DEFAULT_INDEX",
    "UV_INDEX_URL",
    "PIP_INDEX_URL",
    "UV_INDEX",
    "UV_EXTRA_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "UV_CONFIG_FILE",
)
UV_POLICY_ENV = ("UV_OFFLINE",) + UV_INDEX_ENV + ("UV_NO_CONFIG",)
UV_ONLY_INDEX_ENV = (
    "UV_NO_INDEX",
    "UV_DEFAULT_INDEX",
    "UV_INDEX_URL",
    "UV_INDEX",
    "UV_EXTRA_INDEX_URL",
    "UV_CONFIG_FILE",
)


def clear_env(names) -> str:
    """Remove every one of `names` from the environment the lifted block will read."""
    listed = ",".join(f"'{name}'" for name in names)
    return f'foreach ($n in {listed}) {{ Remove-Item "Env:$n" -ErrorAction SilentlyContinue }}'


# The four marker helpers travel together: the path builder, the write guard, the writer and
# the reader. Injecting one without the others is a command-not-found inside the body.
MARKER_FUNCS = functions(
    SETUP_SRC,
    "Get-WoaTorchIndexMarkerPath",
    "Test-WoaPersistableIndex",
    "Save-WoaTorchIndexMarker",
    "Get-WoaTorchIndexMarker",
)

# Tag matching, the version order it needs, the floor and the pyarrow gate built on both.
PYARROW_FLOOR = '$script:WoaPyarrowFloor = "21.0.0"'
WHEEL_TAG_FUNCS = functions(INSTALL_SRC, "Test-WoaWheelTags", "Test-WoaWheelTagsUsable")
PYARROW_USABLE_FUNCS = _script(
    WHEEL_TAG_FUNCS,
    _function_source(INSTALL_SRC, "Test-WoaVersionAtLeast"),
    PYARROW_FLOOR,
    _function_source(INSTALL_SRC, "Test-WoaPyarrowWheelUsable"),
)


def pyarrow_source_script(
    *,
    wheelhouse: str,
    local: str = "$false",
    rest_method: str = INVOKE_RESTMETHOD_NO_NETWORK,
    reaches_pypi: str | None = None,
    lifts: tuple = (),
    preamble: tuple = (),
    tail: tuple = ("Write-Output ('[' + (Get-WoaPyarrowSource -PythonMinor '3.13') + ']')",),
) -> str:
    """Get-WoaPyarrowSource with its network branches stubbed out.

    `wheelhouse` is the PowerShell expression for $script:WoaWheelhouse, `lifts` names any
    extra install.ps1 helper the branch under test reaches for.
    """
    return _script(
        SUBSTEP_NOOP,
        JOIN_URL_RETURNS_BASE,
        f"function Test-WoaWheelhouseIsLocal {{ {local} }}",
        rest_method,
        *((f"function Test-WoaResolveReachesPyPI {{ {reaches_pypi} }}",) if reaches_pypi else ()),
        f"$script:WoaWheelhouse = {wheelhouse}",
        PYARROW_USABLE_FUNCS,
        functions(INSTALL_SRC, "Test-ZipArchiveReadable", *lifts, "Get-WoaPyarrowSource"),
        *preamble,
        *tail,
    )


def native_probe_script(
    *,
    driver: str,
    driver_leaf: str = "$null",
    stubs: tuple = (),
    lifts: tuple = (),
    outputs: tuple = (),
) -> str:
    """Initialize-WoaNativeCudaTorch run against a stubbed host.

    `stubs` carries the index list and the wheel answers, which is all the callers differ
    by; `outputs` names the extra script variables to read back between NATIVE and MSG.
    """
    return _script(
        "$SkipTorch = $false",
        substep_collector(),
        "function Get-HostMachineArch { 'arm64' }",
        "function Get-WoaAbiTag { param($PythonMinor, $FreeThreaded) 'cp313' }",
        "function Test-WoaNvidiaPresent { $true }",
        "function Test-WoaResolverPathsUsable { $true }",
        f"function Get-WoaDriverCudaLeaf {{ {driver_leaf} }}",
        f"function Get-WoaDriverCudaVersion {{ {driver} }}",
        *stubs,
        "function Get-WoaPyarrowSource { param($PythonMinor, $AbiTag) 'pypi' }",
        "function Test-WoaWheelAvailable { $true }",
        functions(
            INSTALL_SRC, *lifts, "Test-WoaAudioMatchesTorch", "Initialize-WoaNativeCudaTorch"
        ),
        "Initialize-WoaNativeCudaTorch -PythonMinor '3.13'",
        "Write-Output ('NATIVE=' + $script:WoaNativeCudaTorch)",
        *outputs,
        "Write-Output ('MSG=' + ($script:Messages -join ' | '))",
    )
