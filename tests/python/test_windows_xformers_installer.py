# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""install.ps1 must install the xFormers wheel built for the torch it installed.

Before this, install.ps1 never mentioned xFormers at all: it installed torch from a
cu126 / cu128 / cu130 index and then plain `unsloth`, whose `windows` extra pulled
xFormers from PyPI -- which publishes only the CUDA-12.8 flavour. On a cu130 host that
produced the NVIDIA QA report "xFormers was built for PyTorch 2.10.0+cu128 with CUDA
1208 (you have 2.10.0+cu130)", with memory-efficient attention silently disabled.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"


def _source() -> str:
    return INSTALL_PS1.read_text(encoding = "utf-8")


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL | re.MULTILINE)
    assert match is not None, f"installer block not found: {pattern}"
    return match.group(0)


def _selector_harness() -> str:
    """`$script:XformersWheelVersions` + the two pure helpers, ready to dot-source."""
    source = _source()
    return "\n".join(
        (
            _extract(r"^    \$script:XformersWheelVersions = @\{.*?^    \}", source),
            _extract(r"^    \$script:XformersStableAbiFloor = [^\n]*", source),
            _extract(r"^    \$script:XformersStableAbiVersion = [^\n]*", source),
            _extract(r"^    \$script:XformersStableAbiFamilies = [^\n]*", source),
            _extract(r"^    function ConvertTo-TorchFlavorTag \{.*?^    \}", source),
            _extract(r"^    function ConvertTo-TorchReleaseVersion \{.*?^    \}", source),
            _extract(r"^    function Get-XformersWheelVersion \{.*?^    \}", source),
            _extract(r"^    function Get-XformersExpectedTorchBuild \{.*?^    \}", source),
            _extract(r"^    function Get-XformersFilenamePythonTag \{.*?^    \}", source),
        )
    )


def _run_pwsh(script: str) -> str:
    # run_pwsh, not subprocess.run: a pwsh killed by a signal never ran this script, and
    # which reads as the selector being wrong. See tests/_shared/unsloth_pwsh_runner.py.
    result = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
    )
    return result.stdout.strip()


# (torch.__version__, expected xFormers version or "" for "no wheel, install nothing").
# The live wheels behind each row were HEAD-verified on download.pytorch.org and their xformers/cpp_lib.json read back
# cu130/xformers-0.0.34 reports {"torch": "2.10.0+cu130"}, cu128/xformers-0.0.34 reports {"torch": "2.10.0+cu128"}.
SELECTION_CASES = [
    ("2.10.0+cu130", "0.0.34"),
    ("2.10.0+cu128", "0.0.34"),
    ("2.10.0+cu126", "0.0.34"),
    ("2.9.1+cu130", "0.0.33.post2"),
    ("2.9.1+cu128", "0.0.33.post2"),
    ("2.9.0+cu130", "0.0.33.post1"),
    ("2.9.0+cu126", "0.0.33.post1"),
    ("2.8.0+cu129", "0.0.32.post2"),
    ("2.7.1+cu128", "0.0.31.post1"),
    # 0.0.30 predates the abi3 switch and has no cp313 wheel;
    ("2.7.0+cu128", ""),
    # No cu130 build of xFormers exists for torch 2.8 or earlier, and no cu118 / cu124 win_amd64 build exists at all --
    ("2.8.0+cu130", ""),
    ("2.9.0+cu118", ""),
    ("2.10.0+cu124", ""),
    # torch 2.11+ resolves to 0.0.35.
    # It is built against 2.10.0 and loads there by design: xFormers moved to the PyTorch stable API/ABI in 0.0.34,
    # whose notes state such builds are "compatible with any later version".
    ("2.11.0+cu130", "0.0.35"),
    ("2.13.0+cu128", "0.0.35"),
    # And so does every release above the floor that the table cannot list, because they
    ("2.10.1+cu130", "0.0.35"),
    ("2.11.1+cu128", "0.0.35"),
    ("2.12.4+cu126", "0.0.35"),
    ("2.14.0+cu130", "0.0.35"),
    # Bounded in both directions:
    ("2.9.2+cu130", ""),
    ("2.12.0+cu124", ""),
    # Non-CUDA and nightly builds must miss the table outright.
    ("2.10.0+cpu", ""),
    ("2.10.0+rocm6.4", ""),
    ("2.10.0.dev20260101+cu130", ""),
]


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(("torch_version", "expected"), SELECTION_CASES)
def test_selector_picks_the_matching_wheel(torch_version: str, expected: str):
    out = _run_pwsh(
        f"{_selector_harness()}\n"
        f"$v = '{torch_version}'\n"
        "$tag = ConvertTo-TorchFlavorTag $v\n"
        "$sel = Get-XformersWheelVersion -TorchVersion $v -CudaTag $tag\n"
        'Write-Output "[$sel]"\n'
    )
    assert out == f"[{expected}]", f"{torch_version} selected {out}, expected [{expected}]"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_selector_refuses_a_missing_or_blank_input():
    out = _run_pwsh(
        f"{_selector_harness()}\n"
        "Write-Output \"[$(Get-XformersWheelVersion -TorchVersion '' -CudaTag 'cu130')]\"\n"
        "Write-Output \"[$(Get-XformersWheelVersion -TorchVersion '2.10.0+cu130' -CudaTag '')]\"\n"
    )
    assert out.splitlines() == ["[]", "[]"]


def test_installer_installs_xformers_from_the_torch_index():
    source = _source()
    block = _extract(r"    # ── Pin xFormers to the wheel built for the torch.*?^    \}\n", source)
    assert "--no-deps" in block, (
        "the xFormers wheel declares torch==<exact release>; without --no-deps uv can pull "
        "a PyPI (CUDA 12.8) torch over the CUDA build just installed"
    )
    assert "--reinstall-package xformers" in block, (
        "cu126/cu128/cu130 publish the SAME xformers version string, so a wrong-CUDA wheel "
        "is invisible to a version check and must be force-replaced"
    )
    assert "--default-index $_xfIndexUrl" in block
    assert "xformers==$_xfVersion" in block
    assert "UNSLOTH_SKIP_XFORMERS" in block
    # A full-URL index pin is authoritative and is reused WHOLE.
    # Its leaf is not required to name the CUDA family: a documented full-URL override can be an authenticated mirror,
    # and rebuilding a download.pytorch.org URL over it strands an air-gapped host.
    assert (
        "if ($TorchIndexUrl -and -not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL))"
        in block
    )
    # The leaf is read to BUILD a URL under the override, never to throw the override away: a full-URL override can be
    # an authenticated mirror whose leaf is not a family, and rebuilding a download.pytorch.org URL over it strands an
    # air-gapped host.
    assert "$_xfIndexUrl = $TorchIndexUrl" in block
    assert '$_xfWheelUrl = Join-UrlPath $_xfBase "$_xfCudaTag/$_xfWheelName"' in block
    assert "--reinstall-package xformers $_xfWheelUrl" in block
    # The already-installed check compares against the wheel's OWN build target, not the resident torch:
    assert "Get-XformersExpectedTorchBuild -Version $_xfVersion" in block


def test_a_family_pin_still_gets_the_direct_wheel_url():
    """UNSLOTH_TORCH_INDEX_FAMILY sets $TorchIndexPinned too, so keying the index branch off
    that flag sent a plain `cu130` pin down --default-index -- and uv's --index / UV_INDEX
    are used "in addition to" the default one, so a machine-level UV_INDEX carrying the same
    xFormers version could satisfy the pin from the wrong CUDA family, which is the failure
    this step exists to prevent. A family pin names a leaf, so a direct URL can be built for
    it; only a full-URL override (possibly an authenticated mirror we cannot rebuild) has to
    go through the index."""
    block = _extract(
        r"    # ── Pin xFormers to the wheel built for the torch.*?^    \}\n", _source()
    )
    condition = re.search(r"if \(\$TorchIndexUrl[^\n]*\) \{", block)
    assert condition is not None, "the index branch must be gated on the full-URL override"
    assert "UNSLOTH_TORCH_INDEX_URL" in condition.group(0)
    assert "$TorchIndexPinned" not in condition.group(0)
    # Unpinned, install the direct wheel URL: --default-index does not make an index exclusive, and cu126/cu128/cu130
    # share a version string, so a machine with UV_INDEX set can satisfy the pin from the wrong CUDA family.
    assert '$_xfWheelUrl = Join-UrlPath $_xfBase "$_xfCudaTag/$_xfWheelName"' in block


def test_a_full_url_override_naming_a_cuda_leaf_gets_a_direct_url():
    """`--default-index` is not exclusive -- uv reads UV_INDEX "in addition to" it -- and every
    CUDA family publishes the same xFormers version, so an index resolve can always be
    satisfied from the wrong one. A documented `.../cu130` override names the leaf, so the
    wheel is addressed under it directly, which nothing can substitute for."""
    block = _extract(
        r"    # ── Pin xFormers to the wheel built for the torch.*?^    \}\n", _source()
    )
    assert "$_xfOverrideLeaf = Get-TorchIndexLeafName $TorchIndexUrl" in block
    assert "'^cu\\d+$'" in block
    assert "$_xfWheelUrl = Join-UrlPath $TorchIndexUrl $_xfWheelName" in block


def test_the_index_fallback_hides_the_machine_level_indexes():
    """The one path that still resolves rather than fetching a URL: a mirror root whose leaf
    is not a CUDA family. UV_INDEX / UV_EXTRA_INDEX_URL are cleared around that call so the
    chosen index really is the only one, and restored afterwards."""
    block = _extract(
        r"    # ── Pin xFormers to the wheel built for the torch.*?^    \}\n", _source()
    )
    assert "$env:UV_INDEX = $null" in block
    assert "$env:UV_EXTRA_INDEX_URL = $null" in block
    assert "$env:UV_INDEX = $_xfSavedIndex" in block
    assert "$env:UV_EXTRA_INDEX_URL = $_xfSavedExtra" in block
    assert "} finally {" in block, "the restore must survive a failing install"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_the_url_join_keeps_a_tokenized_mirrors_query():
    """Executed, not read: a private mirror may authenticate by query string, and appending
    after the query put the wheel path inside the token value -- request path still /whl,
    token unusable. Same rule as the Python side's join_wheel_url."""
    joiner = _extract(r"^    function Join-UrlPath \{.*?^    \}", _source())
    out = _run_pwsh(
        f"{joiner}\n"
        "Write-Output (Join-UrlPath 'https://m/whl?token=abc' 'cu130/x.whl')\n"
        "Write-Output (Join-UrlPath 'https://m/whl/' 'cu130/x.whl')\n"
        "Write-Output (Join-UrlPath 'https://m/whl#f' 'cu130/x.whl')\n"
    )
    assert out.splitlines() == [
        "https://m/whl/cu130/x.whl?token=abc",
        "https://m/whl/cu130/x.whl",
        "https://m/whl/cu130/x.whl#f",
    ]


def test_installer_never_installs_an_unpinned_xformers():
    """A bare `xformers` resolves to whatever the index serves newest, which is not
    necessarily built for the CUDA family the resident torch came from. The installer
    picks the exact version for (torch, cuda) instead, so every spec must be pinned."""
    source = _source()
    for match in re.finditer(r'"xformers[^"]*"', source):
        spec = match.group(0)
        # A wheel FILENAME is not a spec: it names one exact file and cannot resolve to
        if spec.endswith('.whl"'):
            continue
        assert spec == '"xformers==$_xfVersion"', f"unpinned xFormers spec in install.ps1: {spec}"


def test_xformers_step_runs_after_the_torch_flavor_repair():
    """The repair can reinstall torch from a different index; selecting the wheel before it
    would pin against a torch build that is about to be replaced."""
    source = _source()
    repair = source.index("Enforce the installed torch flavor matches the detected GPU build")
    xformers = source.index("Pin xFormers to the wheel built for the torch")
    overlay = source.index("CI only: overlay a source checkout")
    assert repair < xformers < overlay


def test_xformers_step_is_skipped_for_no_torch_installs():
    block = _extract(
        r"    # ── Pin xFormers to the wheel built for the torch.*?^    \}\n", _source()
    )
    assert "if (-not $SkipTorch" in block
    # cu<digits> only: cpu / rocm / xpu torch has no xFormers wheel on any index.
    assert "'^cu\\d+$'" in block


def test_installed_build_probe_reads_cpp_lib_json():
    """The version string cannot distinguish cu128 from cu130, so the repair check has to read
    the build metadata xFormers itself reports in its error message."""
    block = _extract(r"    function Get-InstalledXformersBuild \{.*?^    \}", _source())
    assert "cpp_lib.json" in block
    assert "find_spec" in block
    # Reading the file rather than importing xformers keeps a mismatched .pyd from writing its own warning into the
    # probe output.
    # Every release from 0.0.31 on ships cpp_lib.json, 0.0.35 included, so absence means an unbuilt or source install.
    assert "import xformers" not in block
    # Invoke-BoundedPythonProbe interpolates the code into a double-quoted -c argument.
    assert '"' not in block.split("$code = ", 1)[1].split("\n", 1)[0].strip("'")


# Torch releases the exact tables cannot list, and the answer both selectors must give.
# The two implementations resolve the same machine (install.ps1 during install, wheel_utils on demand from Unsloth), so
# a fallback that lives in only one of them is a machine whose answer changes depending on which one asked.
STABLE_ABI_PARITY_CASES = [
    ("2.10.1+cu130", "13.0", "0.0.35"),
    ("2.11.1+cu128", "12.8", "0.0.35"),
    ("2.12.4+cu126", "12.6", "0.0.35"),
    ("2.14.0+cu130", "13.0", "0.0.35"),
    ("2.9.2+cu130", "13.0", ""),
    ("2.12.0+cu124", "12.4", ""),
]


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(("torch_version", "cuda_version", "expected"), STABLE_ABI_PARITY_CASES)
def test_both_selectors_agree_outside_the_exact_tables(torch_version, cuda_version, expected):
    import sys

    sys.path.insert(0, str(REPO_ROOT / "studio" / "backend"))
    try:
        from utils import wheel_utils
    finally:
        sys.path.pop(0)

    family = wheel_utils.xformers_cuda_family(cuda_version)
    python_side = wheel_utils.xformers_wheel_version(torch_version, family) or ""
    ps_side = _run_pwsh(
        f"{_selector_harness()}\n"
        f"$v = '{torch_version}'\n"
        "$tag = ConvertTo-TorchFlavorTag $v\n"
        'Write-Output "[$(Get-XformersWheelVersion -TorchVersion $v -CudaTag $tag)]"\n'
    )
    assert python_side == expected
    assert ps_side == f"[{expected}]"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_the_already_installed_check_uses_the_wheels_own_build_target():
    """A resident 0.0.35 records the torch it was COMPILED against (the stable-ABI floor),
    not the torch it is running under. Comparing against the resident torch made every
    correct install look mismatched, so each run force-reinstalled a good wheel -- and
    warned about a failed xFormers install whenever the index was unreachable."""
    out = _run_pwsh(
        f"{_selector_harness()}\n"
        "Write-Output \"[$(Get-XformersExpectedTorchBuild -Version '0.0.35' "
        "-TorchVersion '2.12.1+cu130' -CudaTag 'cu130')]\"\n"
        "Write-Output \"[$(Get-XformersExpectedTorchBuild -Version '0.0.34' "
        "-TorchVersion '2.10.0+cu128' -CudaTag 'cu128')]\"\n"
    )
    # The stable-ABI wheel reports the floor release;
    assert out.splitlines() == ["[2.10.0+cu130]", "[2.10.0+cu128]"]


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_the_direct_wheel_filename_tag_matches_the_python_resolver():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "studio" / "backend"))
    try:
        from utils import wheel_utils
    finally:
        sys.path.pop(0)

    versions = ["0.0.31.post1", "0.0.32.post2", "0.0.33.post2", "0.0.34", "0.0.35"]
    ps_out = _run_pwsh(
        f"{_selector_harness()}\n"
        + "\n".join(f"Write-Output \"[$(Get-XformersFilenamePythonTag '{v}')]\"" for v in versions)
    )
    assert ps_out.splitlines() == [
        f"[{wheel_utils.xformers_filename_python_tag(v)}]" for v in versions
    ]
