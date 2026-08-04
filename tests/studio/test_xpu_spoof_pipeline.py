# SPDX-License-Identifier: AGPL-3.0-only
"""Full Intel XPU spoof pipeline: fake torch.xpu on a GPU-less/NVIDIA runner so
Studio's hardware selection + training-device path (detect -> select -> apply ->
device_map -> cache clear) runs exactly as the CUDA path does, with no real
Intel hardware. The XPU sibling of tests/_zoo_aggressive_cuda_spoof.py.

State-sensitive: it fresh-imports the Studio hardware module under the spoof and
mutates its module globals, so studio-backend-ci.yml runs it in the isolated
"Hardware-spoof tests" step (never alongside tests that import hardware).

torch.xpu surface faked here mirrors the PyTorch 2.6+ API hardware.py calls:
is_available, device_count, current_device, get_device_name,
get_device_properties(idx).total_memory, memory_allocated/reserved, mem_get_info
(incl. the Arc B580 / Lunar Lake RuntimeError), is_initialized, synchronize,
empty_cache, plus torch.version.xpu.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"


def _make_fake_xpu(
    *,
    available: bool = True,
    device_count: int = 2,
    total_gb: float = 16.0,
    used_gb: float = 1.0,
    device_name: str = "Intel(R) Arc(TM) B580 Graphics (spoofed)",
    mem_get_info: str = "ok",  # "ok" | "raise" | "absent"
    is_initialized: bool = False,
):
    """Build a fake torch.xpu namespace + a call counter for synchronize/empty_cache.

    mem_get_info: "ok" returns (free, total); "raise" models the Arc B580 / Lunar
    Lake "device doesn't support querying free memory" RuntimeError; "absent"
    omits the attribute so the memory_allocated fallback path is exercised.
    """
    total_bytes = int(total_gb * 1024**3)
    used_bytes = int(used_gb * 1024**3)
    calls = {"synchronize": 0, "empty_cache": 0}
    props = types.SimpleNamespace(name = device_name, total_memory = total_bytes)

    def _mem_get_info(idx = 0):
        if mem_get_info == "raise":
            raise RuntimeError(
                "The device (Intel(R) Arc(TM) B580 Graphics) doesn't support "
                "querying the available free memory."
            )
        return (total_bytes - used_bytes, total_bytes)

    def _sync(*a, **k):
        calls["synchronize"] += 1

    def _empty(*a, **k):
        calls["empty_cache"] += 1

    xpu = types.SimpleNamespace(
        is_available = lambda: available,
        device_count = lambda: device_count,
        current_device = lambda: 0,
        get_device_name = lambda idx = 0: device_name,
        get_device_properties = lambda idx = 0: props,
        memory_allocated = lambda idx = 0: used_bytes,
        memory_reserved = lambda idx = 0: used_bytes,
        is_initialized = lambda: is_initialized,
        synchronize = _sync,
        empty_cache = _empty,
    )
    if mem_get_info != "absent":
        xpu.mem_get_info = _mem_get_info
    return xpu, calls


def _import_studio_hardware_module():
    """Fresh-import Studio's hardware module so detect_hardware re-runs under the
    current spoofs (mirrors test_hardware_dispatch_matrix.py)."""
    if str(STUDIO_BACKEND) not in sys.path:
        sys.path.insert(0, str(STUDIO_BACKEND))
    sys.modules.pop("utils.hardware.hardware", None)
    sys.modules.pop("utils.hardware", None)
    from utils.hardware import hardware as hw  # type: ignore

    return hw


@pytest.fixture
def spoof_xpu(monkeypatch):
    """Apply a full torch.xpu spoof and return (hardware_module, xpu_call_counter).

    Defaults present an unambiguous "prefer XPU" host: CUDA hidden, a numeric
    ZE_AFFINITY_MASK, and torch.xpu reporting devices. Override cuda_available /
    cuda_visible / force_xpu / ze_mask to model hybrid or canary hosts.
    """

    def _apply(
        *,
        cuda_available: bool = False,
        cuda_visible: str = "",  # "" hides CUDA; None unsets; else passthrough
        ze_mask: str = "0,1",  # None unsets the mask
        force_xpu: bool = False,
        xpu_version = "2.7",
        **xpu_kwargs,
    ):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
        if cuda_available:
            monkeypatch.setattr(
                torch.cuda,
                "get_device_properties",
                lambda i = 0: types.SimpleNamespace(name = "Stub NVIDIA GPU"),
                raising = False,
            )
        fake_xpu, calls = _make_fake_xpu(**xpu_kwargs)
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising = False)
        monkeypatch.setattr(torch.version, "xpu", xpu_version, raising = False)

        if ze_mask is None:
            monkeypatch.delenv("ZE_AFFINITY_MASK", raising = False)
        else:
            monkeypatch.setenv("ZE_AFFINITY_MASK", ze_mask)
        if cuda_visible is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visible)
        if force_xpu:
            monkeypatch.setenv("UNSLOTH_FORCE_XPU", "1")
        else:
            monkeypatch.delenv("UNSLOTH_FORCE_XPU", raising = False)
        # FLAT is the oneAPI default; pin it so the test is host-independent.
        monkeypatch.delenv("ZE_FLAT_DEVICE_HIERARCHY", raising = False)

        hw = _import_studio_hardware_module()
        hw._visible_gpu_count = None
        return hw, calls

    return _apply


# ---------- detection ----------


def test_detect_hardware_routes_to_xpu(spoof_xpu):
    hw, _ = spoof_xpu()
    assert hw.detect_hardware() == hw.DeviceType.XPU
    assert hw.CHAT_ONLY is False
    assert hw.IS_ROCM is False


def test_force_xpu_env_routes_to_xpu_even_without_mask(spoof_xpu):
    hw, _ = spoof_xpu(force_xpu = True, ze_mask = None, cuda_visible = None)
    assert hw.detect_hardware() == hw.DeviceType.XPU


def test_bare_mask_with_cuda_present_stays_cuda(spoof_xpu):
    # Canary: a stray inherited ZE_AFFINITY_MASK must NOT steal a CUDA host.
    hw, _ = spoof_xpu(cuda_available = True, cuda_visible = None, ze_mask = "0,1")
    assert hw.detect_hardware() == hw.DeviceType.CUDA


def test_force_xpu_on_hybrid_hides_cuda_for_workers(spoof_xpu):
    # Forced XPU with CUDA still visible must hide CUDA: unsloth's
    # device_type picks CUDA before XPU and ignores UNSLOTH_FORCE_XPU,
    # so workers would otherwise silently train on CUDA.
    hw, _ = spoof_xpu(force_xpu = True, cuda_available = True, cuda_visible = None, ze_mask = None)
    assert hw.detect_hardware() == hw.DeviceType.XPU
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_force_xpu_without_working_xpu_leaves_cuda_untouched(spoof_xpu):
    # Canary: FORCE_XPU on a CUDA host with no working XPU must fall
    # through to CUDA and must NOT hide it.
    hw, _ = spoof_xpu(
        force_xpu = True,
        cuda_available = True,
        cuda_visible = None,
        ze_mask = None,
        available = False,
    )
    assert hw.detect_hardware() == hw.DeviceType.CUDA
    import os

    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_apply_gpu_ids_predetect_never_probes_torch(spoof_xpu, monkeypatch):
    # Workers call apply_gpu_ids() BEFORE detect_hardware(); a lazy detect
    # would probe torch.cuda against the unmasked parent env, latching device
    # enumeration before the mask is written. Pre-detect it must decide from
    # env/build attributes only.
    import torch

    hw, _ = spoof_xpu(ze_mask = None, cuda_visible = None)
    assert hw.DEVICE is None  # fresh import, pre-detect

    def _poisoned_detect():
        raise AssertionError("apply_gpu_ids triggered detect_hardware pre-mask")

    monkeypatch.setattr(hw, "detect_hardware", _poisoned_detect)
    monkeypatch.setattr(torch.cuda, "is_available", _poisoned_detect, raising = False)
    # CUDA-build torch (torch.version.cuda set on this box or spoofed):
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising = False)
    monkeypatch.setattr(torch.version, "xpu", None, raising = False)
    hw.apply_gpu_ids([1])
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert "ZE_AFFINITY_MASK" not in os.environ


def test_apply_gpu_ids_predetect_xpu_build_writes_ze_mask(spoof_xpu, monkeypatch):
    # Pre-detect on an XPU-build torch (version.xpu set, no cuda/hip):
    # the mask must go to ZE_AFFINITY_MASK without any runtime probe.
    import torch

    hw, _ = spoof_xpu(ze_mask = None, cuda_visible = None)
    assert hw.DEVICE is None

    def _poisoned_detect():
        raise AssertionError("apply_gpu_ids triggered detect_hardware pre-mask")

    monkeypatch.setattr(hw, "detect_hardware", _poisoned_detect)
    monkeypatch.setattr(torch.version, "cuda", None, raising = False)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch.version, "xpu", "2.7", raising = False)
    hw.apply_gpu_ids([0])
    import os

    assert os.environ["ZE_AFFINITY_MASK"] == "0"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_apply_gpu_ids_predetect_xpu_compiled_with_null_version(spoof_xpu, monkeypatch):
    # version.xpu can be None on a working XPU build; torch.xpu._is_compiled()
    # must be accepted as the build signal so the mask still goes to
    # ZE_AFFINITY_MASK.
    import torch

    hw, _ = spoof_xpu(ze_mask = None, cuda_visible = None)
    assert hw.DEVICE is None
    monkeypatch.setattr(
        hw, "detect_hardware", lambda: (_ for _ in ()).throw(AssertionError("detect ran"))
    )
    monkeypatch.setattr(torch.version, "cuda", None, raising = False)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch.version, "xpu", None, raising = False)
    monkeypatch.setattr(torch.xpu, "_is_compiled", lambda: True, raising = False)
    hw.apply_gpu_ids([0])
    import os

    assert os.environ["ZE_AFFINITY_MASK"] == "0"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_apply_gpu_ids_predetect_force_on_cuda_build_writes_cvd(spoof_xpu, monkeypatch):
    # UNSLOTH_FORCE_XPU=1 on a CUDA build (no XPU compiled in): detect falls
    # back to CUDA, so the pre-detect mask must go to CUDA_VISIBLE_DEVICES,
    # not ZE_AFFINITY_MASK.
    import torch

    hw, _ = spoof_xpu(force_xpu = True, ze_mask = None, cuda_visible = None)
    assert hw.DEVICE is None
    monkeypatch.setattr(
        hw, "detect_hardware", lambda: (_ for _ in ()).throw(AssertionError("detect ran"))
    )
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising = False)
    monkeypatch.setattr(torch.version, "xpu", None, raising = False)
    monkeypatch.setattr(torch.xpu, "_is_compiled", lambda: False, raising = False)
    hw.apply_gpu_ids([1])
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert "ZE_AFFINITY_MASK" not in os.environ


def test_apply_gpu_ids_predetect_dual_build_honors_xpu_hint(spoof_xpu, monkeypatch):
    # Dual CUDA+XPU build launched the documented XPU way (CUDA hidden + ZE
    # mask): the mask must narrow ZE_AFFINITY_MASK, not re-expose the hidden
    # CUDA via CUDA_VISIBLE_DEVICES. Mirrors detect_hardware's hint.
    import torch

    hw, _ = spoof_xpu(ze_mask = "0,1", cuda_visible = "")
    assert hw.DEVICE is None
    monkeypatch.setattr(
        hw, "detect_hardware", lambda: (_ for _ in ()).throw(AssertionError("detect ran"))
    )
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising = False)
    monkeypatch.setattr(torch.version, "xpu", "2.7", raising = False)
    hw.apply_gpu_ids([0])
    import os

    assert os.environ["ZE_AFFINITY_MASK"] == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""  # stays hidden


def test_apply_gpu_ids_predetect_dual_build_cuda_active_writes_cvd(spoof_xpu, monkeypatch):
    # Canary: dual build with CUDA active (no hint) keeps CUDA masking, same
    # as detect_hardware picking CUDA on a hybrid host.
    import torch

    hw, _ = spoof_xpu(ze_mask = "0,1", cuda_visible = None)
    assert hw.DEVICE is None
    monkeypatch.setattr(
        hw, "detect_hardware", lambda: (_ for _ in ()).throw(AssertionError("detect ran"))
    )
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising = False)
    monkeypatch.setattr(torch.version, "xpu", "2.7", raising = False)
    hw.apply_gpu_ids([1])
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["ZE_AFFINITY_MASK"] == "0,1"  # untouched


def test_apply_gpu_ids_trusts_parent_backend_param(spoof_xpu, monkeypatch):
    # Workers pass the parent's detected backend (config["device_backend"]):
    # it must win over build heuristics in both directions, mirroring
    # detect_hardware's availability check and CUDA fallback exactly.
    import torch

    hw, _ = spoof_xpu(ze_mask = None, cuda_visible = None, force_xpu = True)
    assert hw.DEVICE is None
    monkeypatch.setattr(
        hw, "detect_hardware", lambda: (_ for _ in ()).throw(AssertionError("detect ran"))
    )
    # Forced XPU + XPU build, but the parent detected CUDA (xpu had no
    # device): backend="cuda" must route to CUDA_VISIBLE_DEVICES.
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising = False)
    monkeypatch.setattr(torch.version, "xpu", "2.7", raising = False)
    hw.apply_gpu_ids([1], backend = "cuda")
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert "ZE_AFFINITY_MASK" not in os.environ

    # And backend="xpu" routes to ZE_AFFINITY_MASK even on a CUDA build.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setattr(torch.version, "xpu", None, raising = False)
    hw.apply_gpu_ids([0], backend = "xpu")
    assert os.environ["ZE_AFFINITY_MASK"] == "0"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_apply_gpu_ids_predetect_hidden_cuda_without_mask_prefers_xpu(spoof_xpu, monkeypatch):
    # Hidden CUDA on an XPU-capable build prefers XPU even with NO ZE mask
    # set (detection falls through to XPU in that state); writing the ids to
    # CUDA_VISIBLE_DEVICES would re-expose the hidden CUDA.
    import torch

    hw, _ = spoof_xpu(ze_mask = None, cuda_visible = "")
    assert hw.DEVICE is None
    monkeypatch.setattr(
        hw, "detect_hardware", lambda: (_ for _ in ()).throw(AssertionError("detect ran"))
    )
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising = False)
    monkeypatch.setattr(torch.version, "xpu", "2.7", raising = False)
    hw.apply_gpu_ids([0])
    import os

    assert os.environ["ZE_AFFINITY_MASK"] == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""  # stays hidden


# ---------- visibility / selection ----------


def test_apply_gpu_ids_writes_ze_affinity_mask(spoof_xpu, monkeypatch):
    hw, _ = spoof_xpu()
    hw.detect_hardware()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    hw.apply_gpu_ids([0, 1])
    import os

    assert os.environ["ZE_AFFINITY_MASK"] == "0,1"
    # XPU pinning must not touch CUDA_VISIBLE_DEVICES (hybrid-host safety).
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "sentinel"


def test_get_visible_gpu_count_uses_device_count(spoof_xpu):
    hw, _ = spoof_xpu(ze_mask = "0,1", device_count = 2)
    hw.detect_hardware()
    assert hw.get_visible_gpu_count() == 2


def test_get_visible_gpu_count_empty_mask_is_zero(spoof_xpu):
    hw, _ = spoof_xpu(ze_mask = "")
    hw.detect_hardware()
    assert hw.get_visible_gpu_count() == 0


def test_flat_numeric_mask_reports_relative_ordinals(spoof_xpu):
    hw, _ = spoof_xpu(ze_mask = "4,7", device_count = 2)
    hw.detect_hardware()

    spec = hw._get_parent_visible_gpu_spec()
    assert spec["numeric_ids"] is None
    assert spec["supports_explicit_gpu_ids"] is False

    for result in (hw.get_visible_gpu_utilization(), hw.get_backend_visible_gpu_info()):
        assert result["available"] is True
        assert result["index_kind"] == "relative"
        assert result["parent_visible_gpu_ids"] == []
        assert [device["index"] for device in result["devices"]] == [0, 1]


def test_composite_numeric_mask_reports_physical_ids(spoof_xpu, monkeypatch):
    hw, _ = spoof_xpu(ze_mask = "4,7", device_count = 2)
    monkeypatch.setenv("ZE_FLAT_DEVICE_HIERARCHY", "COMPOSITE")
    hw.detect_hardware()

    spec = hw._get_parent_visible_gpu_spec()
    assert spec["numeric_ids"] == [4, 7]
    assert spec["supports_explicit_gpu_ids"] is True

    for result in (hw.get_visible_gpu_utilization(), hw.get_backend_visible_gpu_info()):
        assert result["available"] is True
        assert result["index_kind"] == "physical"
        assert result["parent_visible_gpu_ids"] == [4, 7]
        assert [device["index"] for device in result["devices"]] == [4, 7]


def test_get_device_map_multi_is_balanced(spoof_xpu):
    hw, _ = spoof_xpu(ze_mask = "0,1", device_count = 2)
    hw.detect_hardware()
    assert hw.get_device_map([0, 1]) == "balanced"


def test_get_device_map_explicit_single_is_sequential(spoof_xpu):
    hw, _ = spoof_xpu(ze_mask = "0,1", device_count = 2)
    hw.detect_hardware()
    # Explicit gpu_ids=[0] is a deliberate single-device request.
    assert hw.get_device_map([0]) == "sequential"


# ---------- cache / telemetry / versions ----------


def test_clear_gpu_cache_calls_xpu(spoof_xpu):
    hw, calls = spoof_xpu()
    hw.detect_hardware()
    hw.clear_gpu_cache()
    assert calls["synchronize"] >= 1
    assert calls["empty_cache"] >= 1


def test_package_versions_survive_broken_xpu_runtime(spoof_xpu, monkeypatch):
    # A broken Intel runtime raising in is_available() must not blank the
    # CUDA/ROCm versions on NVIDIA/AMD hosts.
    import torch

    hw, _ = spoof_xpu(cuda_available = True, cuda_visible = None, ze_mask = None)
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising = False)

    def _broken():
        raise RuntimeError("Level Zero init failed")

    monkeypatch.setattr(torch.xpu, "is_available", _broken)
    versions = hw.get_package_versions()
    assert versions["cuda"] == "12.8"
    assert versions.get("xpu") is None


def test_package_versions_reports_xpu(spoof_xpu):
    hw, _ = spoof_xpu(xpu_version = "2.7")
    hw.detect_hardware()
    assert hw.get_package_versions().get("xpu") == "2.7"


def test_package_versions_xpu_available_fallback(spoof_xpu):
    hw, _ = spoof_xpu(xpu_version = None)
    hw.detect_hardware()
    assert hw.get_package_versions().get("xpu") == "available"


def test_per_device_info_mem_get_info_ok(spoof_xpu):
    hw, _ = spoof_xpu(total_gb = 16.0, used_gb = 1.0)
    hw.detect_hardware()
    info = hw._torch_get_per_device_info([0])
    assert len(info) == 1
    assert info[0]["total_gb"] == pytest.approx(16.0, abs = 0.1)
    assert info[0]["used_gb"] == pytest.approx(1.0, abs = 0.1)


def test_mem_get_info_runtimeerror_keeps_device_with_unknown_usage(spoof_xpu):
    # Arc B580 and Lunar Lake can reject mem_get_info while remaining usable.
    hw, _ = spoof_xpu(mem_get_info = "raise", total_gb = 16.0, device_count = 2)
    hw.detect_hardware()

    info = hw._torch_get_per_device_info([0])
    assert len(info) == 1
    assert info[0]["total_gb"] == pytest.approx(16.0, abs = 0.1)
    assert info[0]["used_gb"] is None

    utilization = hw.get_visible_gpu_utilization()
    assert utilization["available"] is True
    assert len(utilization["devices"]) == 2
    assert all(device["vram_total_gb"] == pytest.approx(16.0) for device in utilization["devices"])
    assert all(device["vram_used_gb"] is None for device in utilization["devices"])

    visibility = hw.get_backend_visible_gpu_info()
    assert visibility["available"] is True
    assert len(visibility["devices"]) == 2
    assert all(device["memory_total_gb"] == pytest.approx(16.0) for device in visibility["devices"])


def test_per_device_info_no_mem_get_info_uses_none(spoof_xpu):
    hw, _ = spoof_xpu(mem_get_info = "absent")
    hw.detect_hardware()
    info = hw._torch_get_per_device_info([0])
    assert len(info) == 1
    assert info[0]["used_gb"] is None


# ---------- training-device wiring ----------


def test_get_torch_device_str_is_xpu(spoof_xpu):
    hw, _ = spoof_xpu()
    hw.detect_hardware()
    assert hw.get_torch_device_str() == "xpu"


def test_dataset_map_num_proc_none_after_xpu_init(spoof_xpu):
    # os.fork() after Level-Zero init corrupts the XPU context -> force in-process.
    hw, _ = spoof_xpu(is_initialized = True)
    hw.detect_hardware()
    assert hw.dataset_map_num_proc(4) is None
