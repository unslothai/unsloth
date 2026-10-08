# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#12798: setup.sh / setup.ps1 pass -DLLAMA_BUILD_NUMBER=<N> for a bNNNN tag source build, and only then."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"
SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"

BASH = shutil.which("bash")
PWSH = shutil.which("pwsh")
requires_bash = pytest.mark.skipif(BASH is None, reason = "bash not available")
requires_pwsh = pytest.mark.skipif(PWSH is None, reason = "pwsh not available")

REFS = [
    ("b11408", "", "11408"),
    ("b11408", "12345", None),  # a llama.cpp PR build is not the tagged release
    ("latest", "", None),
    ("master", "", None),
    ("pr-5", "", None),
    ("b11408-mix-abc", "", None),
    ("", "", None),
]


def _sh_block() -> str:
    text = SETUP_SH.read_text(encoding = "utf-8")
    m = re.search(
        r'^[ \t]*if \[ -z "\$_LLAMA_PR" \] && \[ "\$_RESOLVED_SOURCE_REF_KIND" != "commit" \] \\\n.*?^[ \t]*fi\n',
        text,
        re.M | re.S,
    )
    assert m, "LLAMA_BUILD_NUMBER block missing from setup.sh"
    return m.group(0)


def test_setup_sh_stamps_before_cpu_fallback_copy():
    text = SETUP_SH.read_text(encoding = "utf-8")
    base = text.index('CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release')
    stamp = text.index("-DLLAMA_BUILD_NUMBER=", base)
    copy = text.index('CPU_FALLBACK_CMAKE_ARGS="$CMAKE_ARGS"', base)
    assert base < stamp < copy


def _run_sh_block(ref: str, pr: str, kind: str) -> str:
    script = (
        f'CMAKE_ARGS="-DBASE=1"\n_LLAMA_PR="{pr}"\n_RESOLVED_SOURCE_REF="{ref}"\n'
        f'_RESOLVED_SOURCE_REF_KIND="{kind}"\n{_sh_block()}printf "%s" "$CMAKE_ARGS"\n'
    )
    return subprocess.run([BASH, "-c", script], capture_output = True, text = True, check = True).stdout


@requires_bash
def test_setup_sh_skips_commit_ref_that_looks_like_a_tag():
    assert _run_sh_block("b1234567", "", "commit") == "-DBASE=1"


@requires_bash
@pytest.mark.parametrize("ref,pr,expected", REFS)
def test_setup_sh_gating(ref, pr, expected):
    out = _run_sh_block(ref, pr, "tag")
    if expected is None:
        assert out == "-DBASE=1"
    else:
        assert out == f"-DBASE=1 -DLLAMA_BUILD_NUMBER={expected}"


def test_setup_ps1_sets_number_only_after_concrete_checkout():
    text = SETUP_PS1.read_text(encoding = "utf-8")
    assert "$LlamaBuildNumber = $null" in text
    sets = [m.start() for m in re.finditer(r"\$LlamaBuildNumber = \$TagBuildNumber", text)]
    assert len(sets) == 2
    reuse = text.index("} elseif ($UseConcreteRef) {")
    fetch_failed = text.index('substep "git fetch failed -- using existing source"', reuse)
    clean = text.index("git -C $LlamaCppDir clean -fdx", fetch_failed)
    assert fetch_failed < clean < sets[0] < text.index("} else {", sets[0])
    fresh = text.index('$cloneArgs += @("--branch", $ResolvedSourceRef)')
    assert fresh < sets[1]
    assert re.search(
        r"\} elseif \(\$UseConcreteRef\) \{\s*\$LlamaBuildNumber = \$TagBuildNumber", text[fresh:]
    )
    assert re.search(
        r"if \(\$LlamaBuildNumber\) \{\s*\$CmakeArgs \+= \"-DLLAMA_BUILD_NUMBER=\$LlamaBuildNumber\"",
        text,
    )


@requires_pwsh
# PR builds never reach the assignment sites (checked above).
@pytest.mark.parametrize("ref,_pr,expected", [r for r in REFS if not r[1]])
def test_setup_ps1_tag_parse(ref, _pr, expected):
    line = next(
        line.strip()
        for line in SETUP_PS1.read_text(encoding = "utf-8").splitlines()
        if line.strip().startswith("$TagBuildNumber = ")
    )
    script = f"$ResolvedSourceRef = '{ref}'\n{line}\nif ($null -eq $TagBuildNumber) {{ 'NULL' }} else {{ $TagBuildNumber }}"
    out = subprocess.run(
        [PWSH, "-NoProfile", "-Command", script],
        capture_output = True,
        text = True,
        check = True,
        timeout = 120,
    ).stdout.strip()
    assert out == (expected or "NULL")
