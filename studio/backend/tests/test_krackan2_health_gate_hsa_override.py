# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""lone Linux gfx1153 (Krackan Point 2), nameless DRM record, no rocminfo/amd-smi,
CPU torch. With HSA_OVERRIDE_GFX_VERSION=11.0.0 the installers judge gfx1100 (covered) and
install generic ROCm torch (install_python_stack._linux_host_only_uncovered_amd), so the
health gate must keep the adapter eligible for a torch_cpu_build mismatch. Without the
override the gfx1153 exemption (#13931/#14022) must still apply.

"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "studio" / "backend"))
import utils.hardware.amd as amd  # noqa: E402
import utils.hardware.hardware as hw  # noqa: E402
import utils.hardware.nvidia as nvidia  # noqa: E402


def _kfd_dir(versions):
    d = Path(tempfile.mkdtemp(dir = os.environ.get("TMPDIR")))
    for i, v in enumerate(("0", *versions)):
        n = d / str(i)
        n.mkdir()
        (n / "properties").write_text(
            f"simd_count {0 if i == 0 else 16}\nvendor_id {0 if i == 0 else 4098}\n"
            f"gfx_target_version {v}\n"
        )
    return d


def _eligible(hsa):
    d = _kfd_dir(["110503"])  # gfx1153
    real = amd.amd_kfd_gpu_gfx_targets
    rec = {"vendor": "amd", "index": 0, "name": None, "memory_total_gb": None}
    env = {k: v for k, v in os.environ.items() if k not in ("HSA_OVERRIDE_GFX_VERSION", "UNSLOTH_ROCM_GFX_ARCH")}
    if hsa:
        env["HSA_OVERRIDE_GFX_VERSION"] = hsa
    with patch.dict(os.environ, env, clear = True), \
         patch.object(amd, "amd_kfd_gpu_gfx_targets", lambda real = real: real(str(d))), \
         patch.object(hw.platform, "system", lambda: "Linux"), \
         patch.object(nvidia, "get_physical_gpu_inventory", lambda: {"devices": [], "absent": True}), \
         patch.object(hw, "_linux_drm_sysfs_records", lambda **_: [dict(rec)], create = True), \
         patch.object(hw, "_linux_amd_gfx_candidates", lambda: [], create = True), \
         patch.object(hw, "_linux_kfd_reports_an_amd_gpu", lambda: True, create = True):
        devs = [x for x in hw._probe_physical_gpu_inventory()["devices"] if x.get("vendor") == "amd"]
        assert len(devs) == 1
        return hw._amd_device_can_establish_a_mismatch(devs[0]), devs[0].get("gfx_candidates")


def test_hsa_override_to_covered_arch_keeps_mismatch_eligible():
    ok, cand = _eligible("11.0.0")
    print("override 11.0.0 ->", ok, cand)
    assert ok, f"gfx1153 under HSA_OVERRIDE_GFX_VERSION=11.0.0 hidden from health (candidates={cand})"


def test_plain_gfx1153_still_exempt():
    ok, cand = _eligible(None)
    print("no override ->", ok, cand)
    assert not ok, f"plain gfx1153 should be exempt (candidates={cand})"


@pytest.mark.parametrize("bad", ["", "garbage", "11.0"])
def test_unreadable_override_keeps_exemption(bad):
    ok, cand = _eligible(bad)
    assert not ok, f"unreadable override {bad!r} must not lift the exemption (candidates={cand})"
