# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""R570+ MIG: CUDA enumerates one device per GPU instance, so Studio offers, sizes and pins each
one instead of collapsing a card's instances to a single slice. Older drivers keep the old rule.
CPU fixtures only: every NVIDIA boundary (nvidia-smi, the NVML / CUDA driver probe, torch) is faked."""

from __future__ import annotations

import contextlib
import copy
import ctypes
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

import utils.hardware.hardware as hw
from core.inference import llama_cpp as lc
from utils.hardware import gpu_query, mig, nvidia

B = lc.LlamaCppBackend

GPU = "GPU-00000000-0000-0000-0000-000000000000"
A, B_UUID, C = [f"MIG-{c * 8}-0000-0000-0000-000000000000" for c in "abc"]
LISTING = f"""GPU 0: NVIDIA A100-SXM4-80GB (UUID: {GPU})
  MIG 3g.40gb Device 0: (UUID: {A})
  MIG 2g.20gb Device 1: (UUID: {B_UUID})
  MIG 2g.20gb Device 2: (UUID: {C})
"""


def row(
    uuid,
    free,
    total,
    gi = None,
):
    r = dict(
        index = "0",
        uuid = uuid,
        name = "NVIDIA A100-SXM4-80GB",
        compute_cap = "8.0",
        memory_free_mib = str(free),
        memory_total_mib = str(total),
    )
    if gi is not None:
        r.update(mig = "1", gpu_instance_id = str(gi))
    return r


ROWS = [
    row(GPU, 78000, 81920),
    row(A, 36000, 40192, 0),
    row(B_UUID, 18000, 19968, 1),
    row(C, 17000, 19968, 2),
]


def payload(
    rows = None,
    uuids = None,
    driver = "570.86.15",
):
    return dict(
        source = "nvml",
        driver_version = driver,
        devices = copy.deepcopy(ROWS if rows is None else rows),
        cuda_devices = [
            dict(uuid = u[4:], index = str(i))
            for i, u in enumerate([A, B_UUID, C] if uuids is None else uuids)
        ],
    )


@contextlib.contextmanager
def host(
    mask = None,
    driver = "570.86.15",
    rows = None,
):
    """One A100 80GB with three GPU instances; CUDA enumerates the mask's instances."""
    probe = dict(
        source = "nvml",
        driver_version = driver,
        cuda_driver_version = [12, 8],
        devices = copy.deepcopy(ROWS if rows is None else rows),
    )
    visible = (
        [A, B_UUID, C]
        if mask is None
        else [
            [A, B_UUID, C][int(t)] if t.isdigit() else t
            for t in (x.strip() for x in mask.split(","))
            if t and t != "-1"
        ]
    )
    probe["cuda_devices"] = [{"index": str(i), "uuid": u[4:]} for i, u in enumerate(visible)]

    def result(text):
        return types.SimpleNamespace(returncode = 0, stdout = text, stderr = "")

    def run(cmd, *args, **kw):
        if os.path.basename(str(cmd[0])) == "nvidia-smi":
            if "-L" in cmd:
                return result(LISTING)
            query = next((x for x in cmd if x.startswith("--query-gpu=")), "")
            if query == "--query-gpu=index,uuid":
                return result(f"0, {GPU}\n")
            if "memory.free" in query:
                return result(
                    "0, 78000, 81920" + (f", {GPU}" if query.endswith(",uuid") else "") + "\n"
                )
            return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "unsupported")
        if "--json" in cmd:
            return result(json.dumps(probe))
        raise AssertionError(f"Unmocked subprocess: {cmd}")

    torch_count = 3 if mask is None else len(mask.split(",")) if mask else 0
    torch = types.SimpleNamespace(
        cuda = types.SimpleNamespace(device_count = lambda: torch_count),
        version = types.SimpleNamespace(hip = None, cuda = "12.8"),
        __version__ = "2.9.0",
    )
    with contextlib.ExitStack() as s:
        s.enter_context(
            patch.dict(
                os.environ,
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "UNSLOTH_GPU_QUERY_CACHE": "0",
                    # conftest turns the NVML probe off for every test; this host's answers come from it.
                    "UNSLOTH_NVIDIA_LIBRARY_PROBE": "1",
                },
            )
        )
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            mig._ENV,
        ):
            os.environ.pop(key, None)
        if mask is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = mask
        s.enter_context(patch.dict(sys.modules, {"torch": torch}))
        s.enter_context(patch.object(subprocess, "run", run))
        s.enter_context(patch.object(mig, "possible", lambda: True))
        for name, value in [
            ("DEVICE", hw.DeviceType.CUDA),
            ("IS_ROCM", False),
            ("_physical_gpu_count", None),
            ("_physical_gpu_count_from_smi", False),
            ("_physical_gpu_count_checked_at", None),
            ("_visible_gpu_count", None),
        ]:
            s.enter_context(patch.object(hw, name, value, create = True))
        s.enter_context(patch.object(hw, "get_device", lambda: hw.DeviceType.CUDA))
        s.enter_context(patch.object(hw, "_torch_get_physical_gpu_count", lambda: torch_count))
        s.enter_context(patch.object(hw, "_mask_names_cuda_ordinals", lambda *a: False))
        s.enter_context(patch.object(hw, "rocm_gpu_ids_without_torch_kernels", lambda: set()))
        s.enter_context(
            patch.object(
                hw,
                "_smi_visible_utilization",
                lambda spec: {
                    "devices": [{"index": 0, "vram_total_gb": 80.0, "vram_used_gb": 2.0}]
                },
            )
        )
        for name, value in [
            ("_NVML_ROWS", []),
            ("_SMI_UUID_ROWS", []),
            ("_SMI_MIG_INDICES", None),
            ("_GPU_IDS_ARE_PCI_INDICES", None),
            ("_MIG_DEVICE_SNAPSHOT", None),
            ("_is_vulkan_backend", staticmethod(lambda *a: False)),
            ("_find_llama_server_binary", staticmethod(lambda: "/mock/llama-server")),
            ("_nvidia_probe_script", staticmethod(lambda: Path("mock-probe.py"))),
            ("_get_gpu_memory_amd_smi", staticmethod(lambda *a, **k: [])),
            ("_get_gpu_memory_torch", staticmethod(lambda *a, **k: [])),
            ("_widen_integrated_cuda_rows", staticmethod(lambda rows: rows)),
        ]:
            s.enter_context(patch.object(B, name, value, create = True))
        gpu_query.invalidate_static("MIG fixture host")
        s.enter_context(patch.object(nvidia, "_mig_entry", None))
        s.enter_context(patch.object(mig, "_cache", {}))
        yield probe


@pytest.fixture(autouse = True)
def _probe_script(monkeypatch):
    import utils.prebuilt.update_flow as update_flow
    monkeypatch.setattr(update_flow, "find_installer_script", lambda **k: Path("mock-probe.py"))


def test_training_and_gguf_offer_and_pin_every_gpu_instance():
    for mask, count in ((None, 3), (f"{A},{B_UUID}", 2)):
        with host(mask):
            assert hw.get_parent_visible_gpu_ids() == list(range(count))
            selection, _ = hw.auto_select_gpu_ids("fixture", required_override_gb = 65)
            assert selection == list(range(count))
            expected = ",".join([A, B_UUID, C][:count])
            before = os.environ.get("CUDA_VISIBLE_DEVICES")
            hw.apply_gpu_ids(selection, backend = "cuda")
            assert os.environ["CUDA_VISIBLE_DEVICES"] == expected
            if before is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = before
            gpus = B._get_gpu_memory()
            assert len(gpus) == count
            assert B._pinned_child_visibility([g[0] for g in gpus]) == expected


def test_each_instance_is_sized_by_its_own_memory():
    with host():
        assert B._get_gpu_memory() == [(0, 36000, 40192), (1, 18000, 19968), (2, 17000, 19968)]
        assert B._pinned_child_visibility([2, 0]) == f"{C},{A}"


def test_utilization_lists_every_instance():
    with host():
        util = hw.get_visible_gpu_utilization()
        assert util["index_kind"] == "cuda_device"
        assert util["parent_visible_gpu_ids"] == [0, 1, 2]
        assert [d["uuid"] for d in util["devices"]] == [A, B_UUID, C]
        assert util["devices"][0]["vram_total_gb"] == round(40192 / 1024, 2)


def test_reordered_explicit_mig_mask_keeps_order_and_membership():
    with host(f"{B_UUID},{A}"):
        assert B._get_gpu_memory() == [(0, 18000, 19968), (1, 36000, 40192)]
        assert B._pinned_child_visibility([0, 1]) == f"{B_UUID},{A}"
        hw.apply_gpu_ids([0, 1], backend = "cuda")
        assert os.environ["CUDA_VISIBLE_DEVICES"] == f"{B_UUID},{A}"


def test_a_pick_outside_the_inherited_mask_is_refused():
    with host(f"{A},{B_UUID}"):
        assert hw.get_parent_visible_gpu_ids() == [0, 1]
        with pytest.raises(ValueError):
            hw.apply_gpu_ids([2], backend = "cuda")
        assert os.environ["CUDA_VISIBLE_DEVICES"] == f"{A},{B_UUID}"


def test_a_fresh_worker_pins_the_instances_the_parent_sized_without_probing():
    with host():
        assert hw.get_parent_visible_gpu_ids() == [0, 1, 2]
        inherited = os.environ[mig._ENV]
        # A spawned worker: cold caches, the parent's environment, and no probe may run.
        with (
            patch.object(mig, "_cache", {}),
            patch.object(nvidia, "_mig_entry", None),
            patch.object(subprocess, "run", side_effect = AssertionError("worker probed")),
        ):
            assert os.environ[mig._ENV] == inherited
            hw.apply_gpu_ids([1, 2], backend = "cuda")
            assert os.environ["CUDA_VISIBLE_DEVICES"] == f"{B_UUID},{C}"


def test_an_inherited_map_from_another_mask_is_ignored():
    with host():
        hw.get_parent_visible_gpu_ids()
        with patch.object(mig, "_cache", {}), patch.object(nvidia, "_mig_entry", None):
            os.environ["CUDA_DEVICE_ORDER"] = "FASTEST_FIRST"
            assert mig._inherited_uuids() is None
            assert mig.mask_for([0]) is None


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason = "main never runs the NVML probe on macOS, so the pre-R570 slice reading this checks does not exist there",
)
def test_old_drivers_keep_the_one_slice_rule():
    with host(driver = "565.57.01"):
        assert hw.get_parent_visible_gpu_ids() == [0]
        assert B._get_gpu_memory() == [(0, 17000, 19968)]
        assert B._pinned_child_visibility([0]) == C
        assert mig._ENV not in os.environ


def test_an_instance_without_a_memory_reading_keeps_its_identity():
    rows = copy.deepcopy(ROWS)
    rows[2]["memory_free_mib"] = rows[2]["memory_total_mib"] = "0"
    with host(rows = rows):
        assert len(hw.get_parent_visible_gpu_ids()) == 3
        assert B._get_gpu_memory() == [(0, 36000, 40192), (1, 0, 0), (2, 17000, 19968)]
        assert B._pinned_child_visibility([0, 1, 2]) == f"{A},{B_UUID},{C}"


def _age_cache(seconds):
    for key, entry in list(mig._cache.items()):
        mig._cache[key] = (entry[0] - seconds, *entry[1:])


def test_a_failed_refresh_keeps_known_identities_without_stale_memory():
    with host() as probe:
        assert len(hw.get_parent_visible_gpu_ids()) == 3
        probe["source"] = "failed"
        _age_cache(mig._FRESH_SHARE_S + 1)
        assert B._get_gpu_memory() == [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        assert B._pinned_child_visibility([0, 1, 2]) == f"{A},{B_UUID},{C}"


def test_full_numeric_mask_keeps_its_ids_for_training_and_gguf():
    with host("2,0"):
        assert hw.get_parent_visible_gpu_ids() == [2, 0]
        assert B._get_gpu_memory() == [(0, 36000, 40192), (2, 17000, 19968)]
        assert B._pinned_child_visibility([2, 0]) == f"{C},{A}"
        hw.apply_gpu_ids([2, 0], backend = "cuda")
        assert os.environ["CUDA_VISIBLE_DEVICES"] == f"{C},{A}"


def test_several_compute_instances_of_one_gpu_instance_count_once():
    rows = copy.deepcopy(ROWS)
    rows[2]["gpu_instance_id"] = rows[1]["gpu_instance_id"]
    # The CUDA driver, not NVML's number of CI handles, picks the representatives.
    result = mig.rows_from_payload(payload(rows = rows, uuids = [A, C]))
    assert [r["uuid"] for r in result] == [A, C]


def test_mixed_mig_and_whole_gpu_host():
    other = "GPU-dddddddd-0000-0000-0000-000000000000"
    rows = copy.deepcopy(ROWS) + [{**row(other, 72000, 81920), "index": "1"}]
    result = mig.rows_from_payload(payload(rows = rows, uuids = [A, B_UUID, C, other]))
    assert [r["uuid"] for r in result] == [A, B_UUID, C, other]
    assert [r["index"] for r in result] == ["0", "1", "2", "3"]


def test_numeric_mask_ids_stay_mask_ids():
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,0"}):
        result = mig.rows_from_payload(payload(uuids = [C, A]))
        assert [r["index"] for r in result] == ["2", "0"]
        assert mig.visibility_for(result, [2, 0]) == f"{C},{A}"


def test_no_driver_enumeration_means_the_old_rule():
    p = payload()
    p["cuda_devices"] = None
    assert mig.rows_from_payload(p) is None
    p["cuda_devices"] = [{"uuid": ""}]
    assert mig.rows_from_payload(p) is None
    assert mig.rows_from_payload(payload(driver = "565.57.01")) is None
    assert mig.rows_from_payload(payload(rows = [ROWS[0]], uuids = [GPU])) is None  # no MIG


def test_other_hosts_never_start_the_probe():
    with (
        patch.object(mig, "possible", lambda: False),
        patch.object(mig.subprocess, "run", side_effect = AssertionError),
    ):
        assert mig.visible_devices() is None
    # An NVIDIA Linux host whose cached listing shows no MIG instance, or could not answer.
    for listed in (set(), None):
        with (
            patch.object(mig, "possible", lambda: True),
            patch.dict(os.environ, {}, clear = True),
            patch.object(nvidia, "get_mig_gpu_indices_or_none", lambda: listed),
            patch.object(mig.subprocess, "run", side_effect = AssertionError),
        ):
            assert mig.visible_devices() is None


def test_a_worker_never_probes_for_its_pin():
    with (
        patch.dict(os.environ, {}, clear = True),
        patch.object(nvidia, "get_mig_gpu_indices_or_none", side_effect = AssertionError("listed")),
        patch.object(mig.subprocess, "run", side_effect = AssertionError("probed")),
    ):
        assert mig.mask_for([0]) is None


def test_gguf_memory_skips_the_mig_listing_off_nvidia_linux(monkeypatch):
    monkeypatch.setattr(mig, "possible", lambda: False)
    monkeypatch.setattr(
        nvidia,
        "get_mig_gpu_indices_or_none",
        lambda: (_ for _ in ()).throw(AssertionError("listed")),
    )
    monkeypatch.setattr(B, "_is_vulkan_backend", staticmethod(lambda *a: False))
    monkeypatch.setattr(B, "_find_llama_server_binary", staticmethod(lambda: "/mock/llama-server"))
    monkeypatch.setattr(
        gpu_query, "run_nvidia_smi", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())
    )
    monkeypatch.setattr(B, "_get_gpu_memory_amd_smi", staticmethod(lambda *a, **k: []))
    monkeypatch.setattr(B, "_get_gpu_memory_torch", staticmethod(lambda *a, **k: []))
    monkeypatch.setattr(B, "_get_gpu_memory_nvml", staticmethod(lambda *a, **k: []))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    assert B._get_gpu_memory() == []


def test_compute_caps_follow_the_instances():
    with host():
        with patch.object(
            B, "_installed_llama_cuda_sms", return_value = frozenset([80]), create = True
        ):
            assert len(B._get_gpu_memory(for_llama_server = True)) == 3
            assert B._cuda_compute_caps() == {0: 80, 1: 80, 2: 80}


def test_mig_instances_cannot_use_the_physical_nvlink_matrix():
    with (
        host(),
        patch.dict(os.environ, {"UNSLOTH_DISABLE_DC_P2P": "0", "UNSLOTH_FORCE_DC_P2P": "0"}),
    ):
        B._get_gpu_memory()
        assert "MIG" in B._p2p_veto_reason_inner([0, 1, 2], True, True)


def _probe_module():
    path = Path(__file__).resolve().parents[2] / "nvidia_probe.py"
    spec = importlib.util.spec_from_file_location("mig_test_nvidia_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _set_int(
    ptr,
    value,
    kind = ctypes.c_int,
):
    ctypes.cast(ptr, ctypes.POINTER(kind))[0] = value
    return 0


class _Cuda:
    raw = bytes.fromhex(A[4:].replace("-", ""))

    def cuInit(self, _):
        return 0

    def cuDriverGetVersion(self, p):
        return _set_int(p, 12080)

    def cuDeviceGetCount(self, p):
        return _set_int(p, 1)

    def cuDeviceGet(self, p, i):
        return _set_int(p, i)

    def cuDeviceGetName(self, p, n, d):
        p.value = b"A100 MIG"
        return 0

    def cuDeviceGetAttribute(self, p, a, d):
        return _set_int(p, 8 if a == 75 else 0)

    def cuDeviceGetUuid_v2(self, p, d):
        ctypes.memmove(p, self.raw, 16)
        return 0


def test_strict_cuda_enumeration_reads_instance_uuids():
    module = _probe_module()
    with patch.object(module, "_load", return_value = _Cuda()):
        result = module._probe_cuda_driver(strict = True)
    assert result["devices"][0]["uuid"] == A[4:]
    assert result["devices"][0]["memory_total_mib"] == "0"


def test_the_cuda_driver_fallback_is_unchanged():
    module = _probe_module()
    with patch.object(module, "_load", return_value = _Cuda()):
        assert module._probe_cuda_driver()["devices"][0]["uuid"] == ""

    class NoCount(_Cuda):
        def cuDeviceGetCount(self, p):
            return 1

    with patch.object(module, "_load", return_value = NoCount()):
        assert module._probe_cuda_driver()["devices"] == []
        assert module._probe_cuda_driver(strict = True) is None


def test_nvml_probe_enumerates_cuda_uuids_only_for_r570_mig():
    module = _probe_module()

    class NVML:
        version = b"570.86.15"

        def nvmlInit_v2(self):
            return 0

        def nvmlShutdown(self):
            return 0

        def nvmlSystemGetDriverVersion(self, p, n):
            p.value = self.version
            return 0

        def nvmlSystemGetCudaDriverVersion(self, p):
            return _set_int(p, 12080, ctypes.c_uint)

        def nvmlDeviceGetCount_v2(self, p):
            return _set_int(p, 1, ctypes.c_uint)

        def nvmlDeviceGetHandleByIndex_v2(self, i, p):
            return 0

        def nvmlDeviceGetName(self, h, p, n):
            p.value = b"A100"
            return 0

        def nvmlDeviceGetUUID(self, h, p, n):
            p.value = GPU.encode()
            return 0

        def nvmlDeviceGetCudaComputeCapability(self, h, a, b):
            _set_int(a, 8)
            return _set_int(b, 0)

        def nvmlDeviceGetMemoryInfo(self, h, p):
            return 0

    nvml = NVML()
    memory = module._NvmlMemory(total = 40192 * 1024 * 1024, free = 36000 * 1024 * 1024, used = 0)
    cuda = dict(devices = [dict(uuid = A[4:], index = "0")])
    with (
        patch.object(module, "_load", return_value = nvml),
        patch.object(module, "_mig_children", return_value = [(None, A, memory)]),
        patch.object(module, "_probe_cuda_driver", return_value = cuda) as reader,
        patch.object(module.sys, "platform", "linux"),
    ):
        result = module._probe_nvml()
        assert result["cuda_devices"] == cuda["devices"]
        reader.assert_called_once_with(strict = True)
        reader.reset_mock()
        nvml.version = b"565.57.01"
        assert module._probe_nvml()["cuda_devices"] is None
        reader.assert_not_called()
    with (
        patch.object(module, "_load", return_value = nvml),
        patch.object(module, "_mig_children", return_value = []),
        patch.object(module, "_probe_cuda_driver", return_value = cuda) as reader,
    ):
        nvml.version = b"570.86.15"
        assert module._probe_nvml()["cuda_devices"] is None  # no MIG: no extra enumeration
        reader.assert_not_called()


def test_only_placement_relists_after_a_listing_found_no_instance():
    # The last listing (any age, here from before a hardware re-detect) found no MIG instance.
    stale = (gpu_query._static_gen - 1, 0, 0.0, frozenset())
    with (
        patch.object(mig, "possible", lambda: True),
        patch.dict(os.environ, {"UNSLOTH_NVIDIA_LIBRARY_PROBE": "1"}, clear = True),
        patch.object(nvidia, "_mig_entry", stale),
        patch.object(mig.subprocess, "run", side_effect = AssertionError("probed")),
    ):
        with patch.object(
            nvidia, "get_mig_gpu_indices_or_none", side_effect = AssertionError("listed")
        ):
            assert mig.visible_devices() is None
            assert mig.visible_devices(fresh = True) is None
        listed = []
        with patch.object(nvidia, "get_mig_gpu_indices_or_none", lambda: listed.append(1) or set()):
            assert mig.visible_devices(fresh = True, listing = True) is None
        assert listed == [1]


@contextlib.contextmanager
def _counting_probe(gate = None):
    """Count NVML probe children (``--json``); optionally hold each one until ``gate`` is set."""
    inner = subprocess.run
    calls = []

    def run(cmd, *args, **kw):
        if "--json" in cmd:
            calls.append(cmd)
            if gate is not None:
                gate.wait(10)
        return inner(cmd, *args, **kw)

    with patch.object(subprocess, "run", run):
        yield calls


def test_the_backend_visible_inventory_lists_every_gpu_instance():
    with host():
        info = hw._probe_backend_visible_gpu_info(hw.DeviceType.CUDA)
    assert info["index_kind"] == "cuda_device"
    assert [d["index"] for d in info["devices"]] == [0, 1, 2]
    assert [d["uuid"] for d in info["devices"]] == [A, B_UUID, C]
    assert [d["memory_total_gb"] for d in info["devices"]] == [39.25, 19.5, 19.5]


def test_a_manual_split_under_an_inherited_uuid_mask_pins_those_uuids():
    with host(f"{B_UUID},{A}"):
        assert len(B._get_gpu_memory()) == 2
        env = dict(os.environ)
        assert B._pin_visible_gpu_order_for_split(env) == (0, 1)
        assert env["CUDA_VISIBLE_DEVICES"] == f"{B_UUID},{A}"
        # CUDA device ids never pass for physical card ids.
        assert B._resolve_visible_physical_ids() is None


def test_p2p_vetoes_mig_instances_but_whole_cards_keep_the_physical_matrix(monkeypatch):
    rows = [
        dict(index = "0", physical_index = "1", uuid = GPU, name = "NVIDIA A100-SXM4-80GB"),
        dict(index = "1", physical_index = "0", uuid = A, name = "NVIDIA A100-SXM4-80GB", mig = "1"),
    ]
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_P2P", raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_DC_P2P", raising = False)
    monkeypatch.setattr(
        B,
        "_MIG_DEVICE_SNAPSHOT",
        (os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_DEVICE_ORDER"), rows),
        raising = False,
    )
    seen = []

    def match(pattern, gpu_indices):
        seen.append(list(gpu_indices))
        return False

    monkeypatch.setattr(B, "_all_selected_gpus_match", staticmethod(match))
    mig_reason = "MIG device UUIDs cannot be verified by the physical-card NVLink matrix"
    assert B._p2p_veto_reason_inner([1]) == mig_reason
    assert B._p2p_veto_reason_inner(None) == mig_reason
    assert B._p2p_veto_reason_inner([0]) != mig_reason
    assert seen == [[1]]  # the whole card's nvidia-smi index, not its CUDA device id


def test_a_failed_refresh_backs_off_for_fresh_callers():
    with host() as probe:
        assert len(B._get_gpu_memory()) == 3
        probe["source"] = "failed"
        _age_cache(mig._FRESH_SHARE_S + 1)
        with _counting_probe() as calls:
            for _ in range(5):
                rows = mig.visible_devices(fresh = True)
                assert [r["memory_total_mib"] for r in rows] == ["0", "0", "0"]
        assert len(calls) == 1


def test_concurrent_fresh_callers_share_one_probe_child():
    import threading
    with host():
        assert len(B._get_gpu_memory()) == 3
        _age_cache(mig._FRESH_SHARE_S + 1)
        gate = threading.Event()
        with _counting_probe(gate) as calls:
            results = []
            threads = [
                threading.Thread(target = lambda: results.append(mig.visible_devices(fresh = True)))
                for _ in range(8)
            ]
            for t in threads:
                t.start()
            while not calls:
                threading.Event().wait(0.01)
            threading.Event().wait(0.2)
            gate.set()
            for t in threads:
                t.join(10)
        assert len(calls) == 1
        assert len(results) == 8 and all(r is not None and len(r) == 3 for r in results)


def test_polling_inside_the_share_window_starts_no_child():
    with host():
        assert len(B._get_gpu_memory()) == 3
        with _counting_probe() as calls:
            for _ in range(10):
                mig.visible_devices(fresh = True)
        assert calls == []


def test_a_legacy_answer_is_not_re_probed_by_fresh_callers():
    with host(driver = "565.57.01"):
        with _counting_probe() as calls:
            for _ in range(3):
                assert mig.visible_devices(fresh = True, listing = True) is None
                _age_cache(mig._FAILED_BACKOFF_S + 1)
        assert len(calls) == 1


def test_the_physical_card_correction_never_touches_instance_rows(monkeypatch):
    def correction(devices):
        raise AssertionError("physical-card MIG correction ran on CUDA device rows")

    with host():
        monkeypatch.setattr(hw, "_with_mig_instance_memory", correction)
        selection, _ = hw.auto_select_gpu_ids("fixture", required_override_gb = 30)
        assert selection == [0]


def test_diffusion_gemma_pins_the_instance_uuid_not_a_numeric_id():
    with host(f"{B_UUID},{A}"):
        assert len(B._get_gpu_memory()) == 2
        assert B._mig_pinned_gpu_token("0") == B_UUID
        assert B._mig_pinned_gpu_token("1,0") == f"{A},{B_UUID}"
        assert B._mig_pinned_gpu_token("") == ""
        assert B._mig_pinned_gpu_token(B_UUID) == B_UUID


def test_diffusion_gemma_launch_uses_the_pinned_token():
    import inspect
    assert "gpu = self._mig_pinned_gpu_token(gpu)" in inspect.getsource(lc)


def test_optional_engines_pin_the_instance_uuid_and_budget_its_slice(monkeypatch):
    from core.inference import engine_adapters, engine_install, managed_engine

    monkeypatch.setattr(engine_install, "gpu_platform", lambda: "cuda")
    checked = []

    def support_reason(engine, gpu_id, **kwargs):
        checked.append(gpu_id)
        return "stop here"

    monkeypatch.setattr(managed_engine, "support_reason", support_reason)
    request = types.SimpleNamespace(gpu_ids = [1], gguf_variant = None, model_path = "org/model")
    with host(f"{B_UUID},{A}"):
        with pytest.raises(ValueError, match = "GPU 1: stop here"):
            managed_engine.validate_load("vllm", request)
        # CUDA device 1 is instance A on physical card 0: the driver check reads that card.
        assert checked == [0]
        assert managed_engine._child_cuda_mask([1]) == A
        assert managed_engine._child_cuda_mask([1, 0]) == f"{A},{B_UUID}"
        # The budget is the selected slice's memory, not nvidia-smi --id on a physical card.
        assert engine_adapters._nvidia_memory_rows([1]) == [(40192.0, 36000.0)]


def test_a_reordered_numeric_mask_keeps_its_order_for_a_manual_split():
    with host("2,0"):
        assert [g[0] for g in B._get_gpu_memory()] == [0, 2]
        env = dict(os.environ)
        assert B._pin_visible_gpu_order_for_split(env) == (2, 0)
        assert env["CUDA_VISIBLE_DEVICES"] == f"{C},{A}"


def test_an_r570_mig_payload_without_its_cuda_enumeration_is_a_failed_refresh():
    with host() as probe:
        assert len(B._get_gpu_memory()) == 3
        probe["cuda_devices"] = None
        _age_cache(mig._FRESH_SHARE_S + 1)
        rows = mig.visible_devices(fresh = True)
        assert [r["uuid"] for r in rows] == [A, B_UUID, C]
        assert all(r["memory_total_mib"] == "0" for r in rows)
        assert next(iter(mig._cache.values()))[2] == "failed"


def test_saved_picks_carry_the_cuda_device_namespace():
    from utils.openai_auto_switch_settings import stored_gpu_index_kind

    assert stored_gpu_index_kind({"gpu_index_kind": "cuda_device"}) == "cuda_device"
    with host():
        assert hw.current_gpu_index_kind() == "cuda_device"
        assert hw.current_gpu_index_kind(is_vulkan = True) == "vulkan"
    with host(driver = "565.57.01"):
        assert hw.current_gpu_index_kind() == "physical"


def test_a_physical_pin_saved_before_is_not_reused_on_a_mig_host(monkeypatch):
    import asyncio

    from routes import inference

    monkeypatch.setattr(B, "_is_vulkan_backend", staticmethod(lambda *a: False))
    with host():
        assert asyncio.run(inference._override_gpu_ids_still_resolve([1], "physical")) is False
        assert asyncio.run(inference._override_gpu_ids_still_resolve([1], "cuda_device")) is True


def test_a_retired_probe_does_not_replace_the_worker_map():
    with host() as probe:
        assert len(B._get_gpu_memory()) == 3
        current = os.environ[mig._ENV]
        _age_cache(mig._FRESH_SHARE_S + 1)
        probe["cuda_devices"] = probe["cuda_devices"][:1]
        inner = subprocess.run

        def run(cmd, *args, **kw):
            if "--json" in cmd:
                # Hardware re-detection lands while this probe runs.
                gpu_query.invalidate_static("re-detect during probe")
            return inner(cmd, *args, **kw)

        with patch.object(subprocess, "run", run):
            assert len(mig.visible_devices(fresh = True)) == 1
        assert os.environ[mig._ENV] == current


def test_untranslatable_optional_engine_ids_keep_the_physical_behaviour(monkeypatch):
    from core.inference import engine_install, managed_engine

    monkeypatch.setattr(engine_install, "gpu_platform", lambda: "cuda")
    checked = []

    def support_reason(engine, gpu_id, **kwargs):
        checked.append(gpu_id)
        return "stop here"

    monkeypatch.setattr(managed_engine, "support_reason", support_reason)
    request = types.SimpleNamespace(gpu_ids = [1], gguf_variant = None, model_path = "org/model")
    with host():
        monkeypatch.setattr(mig, "mask_for", lambda ids: None)
        # No refusal: the check and the mask are exactly the pre-MIG ones for this launch.
        with pytest.raises(ValueError, match = "GPU 1: stop here"):
            managed_engine.validate_load("vllm", request)
        assert checked == [1]
        assert managed_engine._child_cuda_mask([1]) == "1"


def test_a_mig_slice_without_a_memory_reading_fails_the_reservation_clearly():
    from core.inference import engine_adapters

    # Imported here, not first inside host(): its sys.modules patch would drop the module on exit.
    from utils import vram_budget_settings  # noqa: F401
    with host() as probe:
        assert len(B._get_gpu_memory()) == 3
        probe["source"] = "failed"
        _age_cache(mig._FRESH_SHARE_S + 1)
        inner = subprocess.run
        queried = []

        def run(cmd, *args, **kw):
            if "--query-gpu=memory.total,memory.free" in cmd:
                queried.append(cmd)
            return inner(cmd, *args, **kw)

        with patch.object(subprocess, "run", run):
            with pytest.raises(RuntimeError, match = "MIG GPU instance has no memory reading.*Retry"):
                engine_adapters.gpu_memory_fraction([0])
        # CUDA device ids are not physical cards: nvidia-smi --id never stands in for a slice.
        assert queried == []
