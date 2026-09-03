# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for the Windows-on-ARM / WSL2 CUDA provisioning review follow-ups.

Six independent bugs, each with a behavioural test where the shell allows one and a
source-level assert where it does not:

1. provision_llama_cuda.sh read the driver's CUDA major from a "CUDA Version:"-only
   banner. R580+/610.x renamed the field to "CUDA UMD Version:" (issue #5812), which
   left the value empty and silently disabled the stale-toolkit upgrade AND the
   toolkit-vs-driver sanity check.
2. provision_llama_cuda.sh's "already provisioned" fast path returned before
   UNSLOTH_LLAMA_PR was honoured, so a rerun with a PR pin reported success while
   keeping the old binary.
3. provision_llama_cuda.sh fetched "pull/N/head:_unsloth_pr_N" without force; git
   rejects that refspec on a rerun (checked-out branch) and after a force-push
   (non-fast-forward), so the stale revision was rebuilt and stamped.
4. setup.sh's CUDA compute-capability probe resolved nvidia-smi via PATH + /usr/bin
   only, missing WSL2 GPU-PV's /usr/lib/wsl/lib and dropping a CUDA host to CPU.
5. setup.sh cleared _NEED_LLAMA_SOURCE_BUILD before the background-staging guard and
   then provisioned CUDA (apt/sudo + a full source build) inside a staged update.
6. install.sh's UNSLOTH_INSTALL_REF branch dropped the torch --overrides file that
   every other with-dependencies branch passes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
PROVISION_SH = PACKAGE_ROOT / "studio" / "scripts" / "provision_llama_cuda.sh"
SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"
INSTALL_SH = PACKAGE_ROOT / "install.sh"


def _provision_src() -> str:
    return PROVISION_SH.read_text(encoding = "utf-8")


# ── 1. driver CUDA major must parse both banner spellings ────────────────────

# Real banners. The second is the R610 spelling reported in unslothai/unsloth#5812.
_BANNER_CLASSIC = (
    "Thu Sep  3 17:50:05 2026\n"
    "+-----------------------------------------------------------------+\n"
    "| NVIDIA-SMI 590.48.01    Driver Version: 590.48.01  CUDA Version: 13.1 |\n"
    "+-----------------------------------------------------------------+\n"
)
_BANNER_UMD = (
    "Thu Sep  3 17:50:05 2026\n"
    "+-----------------------------------------------------------------+\n"
    "| NVIDIA-SMI 610.47   KMD Version: 610.47   CUDA UMD Version: 13.3 |\n"
    "+-----------------------------------------------------------------+\n"
)
_BANNER_UMD_CU12 = _BANNER_UMD.replace("CUDA UMD Version: 13.3", "CUDA UMD Version: 12.8")


def _driver_major_sed() -> str:
    """The exact `sed` expression provision_llama_cuda.sh uses for _DRV_CUDA_MAJOR."""
    line = next(
        ln for ln in _provision_src().splitlines() if ln.startswith("_DRV_CUDA_MAJOR=")
    )
    match = re.search(r"sed (-\w+) '([^']+)'", line)
    assert match, f"could not extract the sed expression from: {line}"
    return f"sed {match.group(1)} '{match.group(2)}'"


@pytest.mark.parametrize(
    "banner, expected",
    [
        (_BANNER_CLASSIC, "13"),
        (_BANNER_UMD, "13"),
        (_BANNER_UMD_CU12, "12"),
        ("no version here at all\n", ""),
    ],
)
def test_driver_cuda_major_parses_both_banner_spellings(banner, expected):
    out = subprocess.run(
        ["bash", "-c", f"{_driver_major_sed()} | head -1"],
        input = banner,
        capture_output = True,
        text = True,
        check = True,
    )
    assert out.stdout.strip() == expected, (
        f"driver CUDA major parsed as {out.stdout.strip()!r}, expected {expected!r}"
    )


def test_setup_sh_and_provisioner_agree_on_the_banner_spelling():
    # setup.sh's _cuda_driver_max_version already accepted " UMD"; the provisioner must too.
    assert "CUDA( UMD)? Version:" in SETUP_SH.read_text(encoding = "utf-8")
    assert "CUDA( UMD)? Version:" in _provision_src()


# ── 2 + 3. UNSLOTH_LLAMA_PR must survive the fast path and a rerun ───────────


def test_pr_pin_bypasses_the_already_provisioned_fast_path():
    src = _provision_src()
    assert 'if [ -z "$_PR_PIN" ] && is_cuda_server "$SERVER"; then' in src, (
        "the already-provisioned early exit must not short-circuit an explicit PR pin"
    )
    # The pin is still validated as numeric before it is used anywhere.
    assert re.search(r'case "\$\{UNSLOTH_LLAMA_PR:-\}" in\s*\n\s*\'\'\|\*\[!0-9\]\*\)', src)


@pytest.mark.skipif(sys.platform.startswith("win"), reason = "bash required")
def test_already_provisioned_fast_path_runs_only_without_a_pr_pin(tmp_path):
    """Run the script's step-0 section against a fake CUDA install.

    Split at the NVIDIA gate so the test never reaches apt/nvcc/cmake.
    """
    src = _provision_src()
    head = src[: src.index("# 1. Require an NVIDIA GPU")] + '\nlog "REACHED_STEP_1"\n'
    script = tmp_path / "step0.sh"
    script.write_text(head, encoding = "utf-8")

    llama_dir = tmp_path / "llama.cpp"
    bin_dir = llama_dir / "build" / "bin"
    bin_dir.mkdir(parents = True)
    server = bin_dir / "llama-server"
    server.write_text("#!/bin/sh\n")
    server.chmod(0o755)
    (bin_dir / "libggml-cuda.so").write_text("")       # structural CUDA marker
    (bin_dir / ".unsloth-cuda-ok").write_text("")      # completion stamp

    def run(**extra):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
               "UNSLOTH_LLAMA_CPP_PATH": str(llama_dir)}
        env.update(extra)
        return subprocess.run(
            ["bash", str(script)], capture_output = True, text = True, env = env, timeout = 120,
        ).stdout

    assert "already present" in run(), "an unpinned rerun must still short-circuit"
    assert "REACHED_STEP_1" not in run()

    pinned = run(UNSLOTH_LLAMA_PR = "17453")
    assert "already present" not in pinned, "an explicit PR pin must not be short-circuited"
    assert "REACHED_STEP_1" in pinned

    # Junk pins are still ignored, exactly as the ref handling below treats them.
    assert "already present" in run(UNSLOTH_LLAMA_PR = "not-a-number")


def test_pr_fetch_uses_fetch_head_not_a_branch_refspec():
    src = _provision_src()
    assert 'origin "pull/${_PR_PIN}/head"' in src, "fetch must target FETCH_HEAD"
    assert "pull/${_PR_PIN}/head:_unsloth_pr_" not in src, (
        "a branch refspec is rejected on rerun and after a force-push"
    )
    assert 'checkout -q -B "_unsloth_pr_${_PR_PIN}" FETCH_HEAD' in src


@pytest.mark.skipif(shutil.which("git") is None, reason = "git required")
def test_pr_fetch_survives_rerun_and_force_push(tmp_path):
    """Replay the script's fetch/checkout against a local remote.

    Old form ("pull/N/head:_unsloth_pr_N"): fails both when the branch is checked out
    ("refusing to fetch into branch") and after a force-push ("non-fast-forward").
    New form (FETCH_HEAD + checkout -B): succeeds in both cases.
    """
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
    }

    def git(cwd, *args, check = True):
        return subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output = True, text = True, env = env, check = check,
        )

    upstream = tmp_path / "upstream.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(upstream)], check = True, env = env)
    subprocess.run(["git", "init", "-q", str(work)], check = True, env = env)
    (work / "f").write_text("v1")
    git(work, "add", "f"), git(work, "commit", "-qm", "v1")
    git(work, "push", "-q", str(upstream), "HEAD:refs/pull/7/head")

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{upstream}", str(clone)],
        check = False, env = env, capture_output = True,
    )
    subprocess.run(["git", "init", "-q", str(clone)], check = True, env = env)
    git(clone, "remote", "add", "origin", f"file://{upstream}", check = False)

    def provision_fetch(pr = "7"):
        """The new script logic."""
        fetched = git(clone, "fetch", "--depth", "1", "origin", f"pull/{pr}/head", check = False)
        if fetched.returncode != 0:
            return False
        out = git(clone, "checkout", "-q", "-B", f"_unsloth_pr_{pr}", "FETCH_HEAD", check = False)
        return out.returncode == 0

    assert provision_fetch(), "first provision must check the PR out"
    assert (clone / "f").read_text() == "v1"

    # Upstream force-pushes a new head (the routine llama.cpp PR update).
    (work / "f").write_text("v2")
    git(work, "commit", "-qam", "v2")
    git(work, "push", "-qf", str(upstream), "HEAD:refs/pull/7/head")

    # The old refspec fails while _unsloth_pr_7 is the checked-out branch...
    old = git(
        clone, "fetch", "--depth", "1", "origin", "pull/7/head:_unsloth_pr_7", check = False,
    )
    assert old.returncode != 0
    assert "refusing to fetch into branch" in old.stderr or "non-fast-forward" in old.stderr

    # ...while the new form updates the checkout to the real PR head.
    assert provision_fetch(), "rerun after a force-push must refresh the PR checkout"
    assert (clone / "f").read_text() == "v2", "the stale PR revision was rebuilt"


# ── 4 + 5. setup.sh: WSL nvidia-smi resolver and staged-update guards ────────


def test_compute_cap_probe_uses_the_shared_nvsmi_resolver():
    src = SETUP_SH.read_text(encoding = "utf-8")
    probe = src[src.index("Resolve the arch list before committing") : src.index("_resolve_cuda_archs \"$_raw_caps\"")]
    assert '_smi_bin="$(_resolve_nvsmi)"' in probe, (
        "the compute-cap probe must reuse _resolve_nvsmi (it covers /usr/lib/wsl/lib)"
    )
    assert 'elif [ -x "/usr/bin/nvidia-smi" ]' not in probe, (
        "the inlined PATH+/usr/bin resolver misses WSL2 GPU-PV's only nvidia-smi"
    )


def test_staged_updates_never_reach_the_cuda_provisioner():
    src = SETUP_SH.read_text(encoding = "utf-8")

    defer = src[src.index("# ── Native Linux aarch64 + NVIDIA, no nvcc yet") : src.index("Background staging cannot install system build tools")]
    assert '[ -z "$STAGE_ROOT" ]' in defer, (
        "clearing _NEED_LLAMA_SOURCE_BUILD under STAGE_ROOT slips past the staging guard"
    )

    provision_start = src.index("aarch64 + NVIDIA (DGX Spark / GB10 / N1X \"RTX Spark\"): provision a CUDA")
    provision = src[provision_start : src.index("_PROV_SH=\"\"", provision_start)]
    assert '[ -z "$STAGE_ROOT" ]' in provision, (
        "the CUDA provisioner (apt/sudo + a full source build) must stay on foreground runs"
    )


# ── 6. install.sh: the torch trio override must reach every with-deps install ─


def test_every_with_deps_unsloth_install_passes_the_torch_overrides():
    src = INSTALL_SH.read_text(encoding = "utf-8")
    # The git-ref (pre-merge testing) branch used to omit it, letting unsloth-zoo's
    # resolution replace the CUDA/ROCm torch trio installed in Step 1.
    ref_branch = src[src.index("installing unsloth from git ref") : src.index("unsloth-zoo\n", src.index("installing unsloth from git ref"))]
    assert "${_UNSLOTH_TORCH_OVERRIDES:+--overrides \"$_UNSLOTH_TORCH_OVERRIDES\"}" in ref_branch

    # Every `uv pip install` that resolves unsloth WITH dependencies carries the flag.
    for match in re.finditer(r'run_install_cmd(?:_retry)? "install unsloth[^"]*"(?:[^\n]*\\\n)*[^\n]*\n', src):
        block = match.group(0)
        if "--no-deps" in block or "--torch-backend=auto" in block:
            continue
        assert "_UNSLOTH_TORCH_OVERRIDES" in block, f"missing --overrides in:\n{block}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
