# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Readiness canary: fail once flash-attn ships a torch >= 2.12 wheel.

Background: torch 2.12 changed the c10 CUDA-check ABI
(``c10::cuda::c10_cuda_check_implementation`` gained an ``unsigned int`` arg),
so every pre-2.12 flash-attn prebuilt wheel (newest is ``cu13torch2.10``) fails
to import on torch 2.12 with an undefined-symbol error. That is why the Studio
installer caps modern-CUDA torch at ``<2.12.0``. causal-conv1d and mamba already
ship ``cu13torch2.10`` wheels that load fine on torch 2.12, so flash-attn is the
one blocker for raising the ceiling.

This canary polls the Dao-AILab/flash-attention GitHub releases and FAILS the
moment a wheel built against torch >= 2.12 appears. Wire the live test into a
scheduled CI job (see flash-attn-torch212-canary.yml) so the red build tells us
to add torch 2.12 support to unsloth. The parser is unit-tested offline so the
detection mechanism itself is verified on every run.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

import pytest

RELEASES_URL = "https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=100"

# flash-attn wheels embed their build target as e.g. "cu13torch2.10" in the name.
_TORCH_TAG_RE = re.compile(r"cu(\d+)torch(\d+)\.(\d+)")

# torch 2.12 is the first release with the new c10 CUDA-check ABI.
_MIN_SUPPORTING = (2, 12)


def torch_ge_212_wheels(asset_names):
    """Return the wheel names built against torch >= 2.12 (sorted, unique)."""
    hits = []
    for name in asset_names:
        for cu, major, minor in _TORCH_TAG_RE.findall(name):
            if (int(major), int(minor)) >= _MIN_SUPPORTING:
                hits.append(name)
                break
    return sorted(set(hits))


def fetch_release_asset_names(timeout = 30):
    """All release-asset filenames for Dao-AILab/flash-attention (network)."""
    req = urllib.request.Request(
        RELEASES_URL,
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "unsloth-flash-attn-torch212-canary",
        },
    )
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout = timeout) as resp:
        releases = json.loads(resp.read().decode("utf-8"))
    return [
        asset["name"]
        for release in releases
        for asset in release.get("assets", [])
        if asset.get("name")
    ]


# The live network poll runs only in the scheduled canary job, so it never turns
# unrelated PRs red the day a torch 2.12 wheel ships. The offline parser tests
# below still run in every suite, keeping the detection logic covered.
_RUN_LIVE = os.environ.get("UNSLOTH_RUN_FLASH_ATTN_CANARY") == "1"


_ACTION = (
    "\n\ntorch 2.12 broke the pre-2.12 c10 CUDA-check ABI, so this is the trigger "
    "to add torch 2.12 support to unsloth:\n"
    "  1. Extend studio/backend/utils/wheel_utils.py prebuilt_wheel_torch_mm to map "
    "torch 2.12 onto this flash-attn build tag.\n"
    "  2. Raise the install.sh modern-CUDA ceiling from <2.12.0 toward <2.13.0 once "
    "causal-conv1d and mamba also ship torch 2.12 wheels.\n"
    "  3. Re-run the Blackwell accelerator smoke on torch 2.12, then bump this "
    "canary's threshold."
)


@pytest.mark.skipif(
    not _RUN_LIVE,
    reason = "live canary runs only in the scheduled job (set UNSLOTH_RUN_FLASH_ATTN_CANARY=1)",
)
def test_no_flash_attn_torch212_wheel_yet():
    """Live canary: fails when flash-attn publishes a torch >= 2.12 wheel."""
    try:
        names = fetch_release_asset_names()
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            pytest.skip("GitHub API rate-limited (%s); cannot check canary" % exc.code)
        pytest.skip("GitHub API HTTP %s; skipping canary" % exc.code)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip("network unavailable for canary: %s" % exc)

    hits = torch_ge_212_wheels(names)
    assert not hits, (
        "flash-attn now publishes prebuilt wheels built against torch >= 2.12:\n"
        + "\n".join("  " + h for h in hits)
        + _ACTION
    )


def test_parser_flags_torch_212_and_above():
    fake = [
        "flash_attn-3.0.0+cu13torch2.12cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "flash_attn-3.0.0+cu12torch2.13cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
    ]
    assert torch_ge_212_wheels(fake) == sorted(fake)


def test_parser_ignores_pre_212():
    fake = [
        "flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "flash_attn-2.8.3+cu12torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
        "some-source-tarball.tar.gz",
    ]
    assert torch_ge_212_wheels(fake) == []


if __name__ == "__main__":
    import sys

    _names = fetch_release_asset_names()
    _hits = torch_ge_212_wheels(_names)
    if _hits:
        print("torch 2.12+ flash-attn wheels detected:")
        for _h in _hits:
            print("  " + _h)
        sys.exit(1)
    print("No flash-attn wheel built against torch >= 2.12 yet (canary green).")
    sys.exit(0)
