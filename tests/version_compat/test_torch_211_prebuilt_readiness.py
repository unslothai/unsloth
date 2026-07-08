# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Canary for raising the CUDA torch pin to 2.11.

The Studio installer selects torch-version-tagged prebuilt CUDA wheels for three
accelerators: flash-attn (wheel_utils.flash_attn_package_version) and causal-conv1d
/ mamba-ssm (worker._install_package_wheel_first). None of them publish torch 2.11
wheels yet, so the CUDA torch pin stays < 2.11 (_CUDA_TORCH_PKG_SPEC in
studio/install_python_stack.py); moving to 2.11 today would drop those wheels and
force slow source builds. This test stays green while any of the three is missing
torch 2.11 Linux wheels and fails once all three ship them, i.e. when it is finally
safe to bump the pin.
"""

from __future__ import annotations

import re

from tests.version_compat._fetch import fetch_json

# human name -> GitHub repo
_REPOS = {
    "flash-attn": "Dao-AILab/flash-attention",
    "causal-conv1d": "Dao-AILab/causal-conv1d",
    "mamba-ssm": "state-spaces/mamba",
}
_TORCH211_LINUX = re.compile(r"torch2\.11.*linux", re.IGNORECASE)


def _has_torch211_linux_wheel(repo: str) -> bool:
    """True if a recent release of ``repo`` publishes a torch 2.11 Linux wheel."""
    releases = fetch_json(f"https://api.github.com/repos/{repo}/releases?per_page=5")
    if not releases:
        return False
    return any(
        _TORCH211_LINUX.search(asset.get("name", ""))
        for rel in releases
        for asset in rel.get("assets", [])
    )


def test_torch_211_prebuilt_wheels_not_all_ready():
    status = {name: _has_torch211_linux_wheel(repo) for name, repo in _REPOS.items()}
    ready = sorted(n for n, ok in status.items() if ok)
    pending = sorted(n for n, ok in status.items() if not ok)
    print(f"torch 2.11 wheel readiness -> ready: {ready or 'none'}; pending: {pending}")
    assert pending, (
        f"torch 2.11 Linux wheels are now published for all of {ready}. It is time "
        "to raise the CUDA torch pin: bump _CUDA_TORCH_PKG_SPEC to <2.12.0 (with the "
        "matching torchvision/torchaudio bounds) in studio/install_python_stack.py, "
        "add torch 2.11 to wheel_utils.flash_attn_package_version, and bump the "
        "causal-conv1d / mamba release tags in studio/backend/core/training/worker.py."
    )
