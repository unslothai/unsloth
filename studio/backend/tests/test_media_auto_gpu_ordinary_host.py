# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automatic image / video GPU placement (#8235) leaves ordinary hosts as they were.

Zero / one GPU, Apple, CPU and ROCm run no device, dtype or capability probe for an
automatic request; the current device (a tie included) stays None so generation gains
no explicit set_device pins; identical cards under FASTEST_FIRST keep nvidia-smi
telemetry; a /v1/models catalog refresh does not rank live on homogeneous hosts."""

import asyncio
import os
import types
from unittest.mock import patch

import pytest

from core.inference import diffusion_device as dd
from utils.hardware import hardware as hw
from utils.hardware.hardware import DeviceType


def _dev(index, total, used):
    return {"index": index, "vram_total_gb": total, "vram_used_gb": used}


def _counting_target(
    monkeypatch,
    module,
    device = "cuda",
):
    calls = []

    def _target(**_kw):
        calls.append(1)
        return types.SimpleNamespace(device = device)

    monkeypatch.setattr(module, "resolve_diffusion_device_target", _target)
    return calls


def _counting_rank(monkeypatch, answer = 1):
    calls = []

    def _rank():
        calls.append(1)
        return answer

    monkeypatch.setattr(dd, "resolve_auto_cuda_ordinal", _rank)
    return calls


def _ordinary(
    monkeypatch,
    device = DeviceType.CUDA,
    visible = (0,),
    rocm = False,
):
    monkeypatch.setattr(hw, "IS_ROCM", rocm)
    monkeypatch.setattr(hw, "get_device", lambda: device)
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: list(visible))


# Item 1: the current device stays None.


@pytest.mark.parametrize(
    "devices",
    [
        [_dev(0, 80.0, 0.5), _dev(1, 80.0, 0.5)],  # idle identical cards
        [_dev(0, 80.0, 0.5), _dev(1, 80.0, 40.0)],  # card 0 is the freest
    ],
)
def test_auto_returns_none_when_the_current_device_wins(monkeypatch, devices):
    _two_cards(monkeypatch, devices, {0: (8, 0), 1: (8, 0)})
    assert dd.resolve_auto_cuda_ordinal() is None


def test_auto_keeps_card_zero_explicit_beside_a_mig_card(monkeypatch):
    # Identical physical rows, but card 1 is MIG: its slice size is not the row's.
    _two_cards(
        monkeypatch, [_dev(0, 80.0, 0.5), _dev(1, 80.0, 0.5)], {0: (8, 0), 1: (8, 0)}, mig = {1}
    )
    assert dd.resolve_auto_cuda_ordinal() == 0


@pytest.mark.parametrize(
    "devices, caps",
    [
        ([_dev(0, 80.0, 0.5), _dev(1, 80.0, 40.0)], {0: (8, 0), 1: (9, 0)}),  # mixed capability
        ([_dev(0, 80.0, 0.5), _dev(1, 40.0, 30.0)], {0: (8, 0), 1: (8, 0)}),  # mixed size
    ],
)
def test_auto_keeps_card_zero_explicit_on_mixed_cards(monkeypatch, devices, caps):
    # A pooled thread left on card 1 must not preflight a card-0 load against card 1.
    _two_cards(monkeypatch, devices, caps)
    assert dd.resolve_auto_cuda_ordinal() == 0


def _two_cards(
    monkeypatch,
    devices,
    caps,
    mig = (),
):
    from utils.hardware import nvidia

    monkeypatch.setattr(nvidia, "get_compute_capabilities", lambda: dict(caps))
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: set(mig))
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0, 1])
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: True)
    monkeypatch.setattr(hw, "_integrated_cuda_rows", lambda ids: ({}, "index"))
    monkeypatch.setattr(hw, "_get_parent_visible_gpu_spec", lambda: {"numeric_ids": [0, 1]})
    monkeypatch.setattr(
        hw, "_smi_visible_utilization", lambda spec: {"available": True, "devices": devices}
    )


@pytest.mark.parametrize("gpu_ordinal, expected", [(None, [("pin", 0)]), (0, ["apply"])])
def test_video_generation_sets_the_device_once(monkeypatch, gpu_ordinal, expected):
    # Auto (None) keeps the single placed-device pin; an explicit 0 pins once, not twice.
    from core.inference import video

    pins = []
    monkeypatch.setattr(video, "resolve_diffusion_device_target", lambda **_k: object())
    monkeypatch.setattr(video, "apply_diffusion_device_ordinal", lambda t: pins.append("apply"))
    monkeypatch.setattr(video, "pin_cuda_ordinal", lambda o: pins.append(("pin", o)))
    state = types.SimpleNamespace(gpu_ordinal = gpu_ordinal, placed_ordinal = 0)
    video.VideoBackend._state_device_target(video.VideoBackend.__new__(video.VideoBackend), state)
    assert pins == expected


@pytest.mark.parametrize("gpu_ordinal, expected", [(None, ["apply", ("pin", 0)]), (0, ["apply"])])
def test_image_generation_sets_the_device_once(monkeypatch, gpu_ordinal, expected):
    from core.inference import diffusion

    pins = []
    backend = diffusion.DiffusionBackend.__new__(diffusion.DiffusionBackend)
    monkeypatch.setattr(backend, "_target_for_ordinal", lambda fam, o: object(), raising = False)
    monkeypatch.setattr(diffusion, "apply_diffusion_device_ordinal", lambda t: pins.append("apply"))
    monkeypatch.setattr(diffusion, "pin_cuda_ordinal", lambda o: pins.append(("pin", o)))
    state = types.SimpleNamespace(family = None, gpu_ordinal = gpu_ordinal, placed_ordinal = 0)
    diffusion.DiffusionBackend._state_device_target(backend, state)
    assert pins == expected


# Item 2: no probe on ordinary hosts, and validation first.


@pytest.mark.parametrize(
    "device, visible, rocm",
    [
        (DeviceType.CUDA, (0,), False),  # one GPU
        (DeviceType.CUDA, (), False),  # zero visible
        (DeviceType.MLX, (), False),  # Apple Silicon
        (DeviceType.CPU, (), False),
        (DeviceType.CUDA, (0, 1), True),  # ROCm
    ],
)
def test_route_auto_runs_no_device_probe_on_ordinary_hosts(monkeypatch, device, visible, rocm):
    from routes import inference as routes

    _ordinary(monkeypatch, device = device, visible = visible, rocm = rocm)
    targets = _counting_target(monkeypatch, dd)
    ranks = _counting_rank(monkeypatch)
    assert asyncio.run(routes._selected_gpu_ordinal(None)) is None
    assert (targets, ranks) == ([], [])


def test_route_auto_still_ranks_on_a_multi_gpu_cuda_host(monkeypatch):
    from routes import inference as routes

    _ordinary(monkeypatch, visible = (0, 1))
    targets = _counting_target(monkeypatch, dd)
    ranks = _counting_rank(monkeypatch)

    async def load():
        return await routes._selected_gpu_ordinal(None)

    assert asyncio.run(load()) == 1
    assert (len(targets), len(ranks)) == (1, 1)


@pytest.mark.parametrize(
    "module_name, cls_name", [("diffusion", "DiffusionBackend"), ("video", "VideoBackend")]
)
def test_direct_invalid_auto_load_is_refused_before_any_probe(monkeypatch, module_name, cls_name):
    import importlib

    module = importlib.import_module(f"core.inference.{module_name}")
    _ordinary(monkeypatch, visible = (0, 1))
    targets = _counting_target(monkeypatch, module)
    dd_targets = _counting_target(monkeypatch, dd)
    ranks = _counting_rank(monkeypatch)
    backend = getattr(module, cls_name)()
    with pytest.raises(ValueError):
        backend.begin_load("nobody/not-a-diffusion-model-at-all")
    assert (targets, dd_targets, ranks) == ([], [], [])


@pytest.mark.parametrize("device, visible", [(DeviceType.CUDA, (0,)), (DeviceType.MLX, ())])
def test_backend_auto_fallback_runs_no_probe_on_ordinary_hosts(monkeypatch, device, visible):
    _ordinary(monkeypatch, device = device, visible = visible)
    targets = _counting_target(monkeypatch, dd)
    ranks = _counting_rank(monkeypatch)
    assert dd.auto_cuda_ordinal_for_load() is None
    assert (targets, ranks) == ([], [])


# Item 3: FASTEST_FIRST numeric masks read nvidia-smi rows joined by PCI bus id, not torch.


def _fastest_first_host(monkeypatch, rows):
    torch_calls = []

    def _torch_rows(indices):
        torch_calls.append(list(indices))
        return [
            {"index": idx, "visible_ordinal": o, "total_gb": 80.0, "used_gb": 10.0 + o * 60.0}
            for o, idx in enumerate(indices)
        ]

    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_device", lambda: DeviceType.CUDA)
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: 2)
    monkeypatch.setattr(hw, "_physical_gpu_count_from_smi", True)
    monkeypatch.setattr(hw, "rocm_gpu_ids_without_torch_kernels", lambda: set())
    # Masked SMI rows (PCI order): row 0 busy, row 1 free.
    smi_rows = [_dev(0, 80.0, 70.0), _dev(1, 80.0, 10.0)]
    monkeypatch.setattr(hw, "get_visible_gpu_utilization", lambda: {"devices": smi_rows})
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda name, ids, **_k: {
            "available": True,
            "devices": [d for d in smi_rows if d["index"] in ids],
        },
    )
    monkeypatch.setattr(hw, "_torch_get_per_device_info", _torch_rows)
    from utils.hardware import nvidia

    reads = []
    monkeypatch.setattr(nvidia, "get_cuda_ordinal_smi_rows", lambda: reads.append(1) or rows)
    monkeypatch.setattr(hw, "_cuda_ordinal_rows_cache", {})
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: set())
    return torch_calls, reads


def _auto(env):
    with patch.dict(os.environ, env, clear = True):
        ids, _meta = hw.auto_select_gpu_ids("m", required_override_gb = 16.0)
    return ids


_FASTEST = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST", "CUDA_VISIBLE_DEVICES": "0,1"}


@pytest.mark.parametrize("rows, expected", [([0, 1], [1]), ([1, 0], [0])])
def test_fastest_first_mask_ranks_smi_rows_by_bus_id(monkeypatch, rows, expected):
    # rows = the nvidia-smi row behind each CUDA ordinal; [1, 0] is a tie order that differs from PCI.
    torch_calls, reads = _fastest_first_host(monkeypatch, rows)
    assert _auto(_FASTEST) == expected
    assert _auto(_FASTEST) == expected
    assert torch_calls == []
    assert reads == [1]  # mapping found once per environment


def test_fastest_first_mask_without_a_mapping_keeps_torch(monkeypatch):
    torch_calls, _reads = _fastest_first_host(monkeypatch, None)
    assert _auto(_FASTEST) == [0]
    assert torch_calls == [[0, 1]]


@pytest.mark.parametrize("mig", [{1}, None])  # a mapped MIG card, or MIG discovery failing
def test_fastest_first_mask_falls_back_to_torch_around_mig(monkeypatch, mig):
    # nvidia-smi shows a MIG card's whole memory; torch reports the slice, so it ranks instead.
    torch_calls, _reads = _fastest_first_host(monkeypatch, [1, 0])
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: mig)
    with patch.dict(os.environ, _FASTEST, clear = True):
        assert hw._smi_rows_for_cuda_ordinals([0, 1]) is None
        hw.auto_select_gpu_ids("m", required_override_gb = 8.0)
    assert torch_calls == [[0, 1]]


def test_fastest_first_mask_without_mig_keeps_nvidia_smi(monkeypatch):
    torch_calls, _reads = _fastest_first_host(monkeypatch, [1, 0])
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: {5})  # MIG elsewhere only
    with patch.dict(os.environ, _FASTEST, clear = True):
        assert hw._smi_rows_for_cuda_ordinals([0, 1]) is not None
    assert torch_calls == []


def test_torch_fallback_rows_skip_the_physical_mig_correction(monkeypatch):
    # No bus-id mapping: torch rows carry mask ids and already report slice memory. Physical row 0 is a
    # MIG card, but mask id 0 is another card, so correcting it would hide the card that has room.
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.hardware import nvidia

    torch_calls, _reads = _fastest_first_host(monkeypatch, None)
    monkeypatch.setattr(nvidia, "get_mig_gpu_indices", lambda: {0})
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_memory_nvml", staticmethod(lambda: [(0, 1024, 5120)])
    )
    assert _auto(_FASTEST) == [0]  # torch: mask id 0 has 70 GB free
    assert torch_calls == [[0, 1]]


def test_bus_ids_from_both_tools_join():
    from utils.hardware import nvidia
    assert nvidia._normalize_bus_id("00000000:3B:00.0") == nvidia._normalize_bus_id("0000:3b:00.0")
    assert nvidia._normalize_bus_id("garbage") is None


def test_card_identities_are_read_once(monkeypatch):
    from utils.hardware import nvidia

    reads = []
    row = ("NVIDIA A100-SXM4-80GB", "81920 MiB", "8.0", "1410 MHz")
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: 2)
    monkeypatch.setattr(nvidia, "get_card_identities", lambda: reads.append(1) or {0: row, 1: row})
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: set())
    monkeypatch.setattr(hw, "_cuda_card_identities", {})
    assert hw._cuda_cards_identical() is True
    assert hw._cuda_cards_identical([0, 1]) is True
    assert reads == [1]


# Item 4: catalog refreshes do not rank live on single-GPU / homogeneous hosts.


def _catalog(monkeypatch):
    from core.inference import media_locality, media_model_index
    from routes import inference as routes

    planned = []
    monkeypatch.setattr(media_model_index, "available_media_model_ids", lambda task: ["a", "b"])
    monkeypatch.setattr(
        media_model_index,
        "resolve_local_media_model",
        lambda model_id, task: types.SimpleNamespace(model_id = model_id),
    )
    monkeypatch.setattr(media_locality, "is_edit_only", lambda pick: False)
    monkeypatch.setattr(
        media_locality,
        "missing_download_bytes",
        lambda owner, pick, hf_token = None: planned.append(media_locality.plan_gpu_ordinal()) or 0,
    )
    monkeypatch.setattr(routes, "_MEDIA_PICK_CACHE", {"at": None, "picks": {}})
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: True)
    return routes, planned


@pytest.mark.parametrize("visible, identical", [((0,), True), ((0,), None), ((0, 1), True)])
def test_catalog_refresh_does_not_rank_on_ordinary_hosts(monkeypatch, visible, identical):
    routes, planned = _catalog(monkeypatch)
    _ordinary(monkeypatch, visible = visible)
    monkeypatch.setattr(hw, "_cuda_cards_identical", lambda gpu_ids = None: identical)
    _counting_target(monkeypatch, dd)
    ranks = _counting_rank(monkeypatch)
    routes._validated_media_picks(1.0)
    assert ranks == []
    assert planned and set(planned) == {None}


@pytest.mark.parametrize("identical", [False, None])
def test_catalog_refresh_ranks_once_on_mixed_or_unknown_cards(monkeypatch, identical):
    routes, planned = _catalog(monkeypatch)
    _ordinary(monkeypatch, visible = (0, 1))
    monkeypatch.setattr(hw, "_cuda_cards_identical", lambda gpu_ids = None: identical)
    _counting_target(monkeypatch, dd)
    ranks = _counting_rank(monkeypatch, answer = 1)
    routes._validated_media_picks(1.0)
    assert len(ranks) == 1
    assert set(planned) == {1}
    from core.inference.media_locality import planned_gpu_ordinal

    assert planned_gpu_ordinal() == (False, None)


# The route's gate answers from memory, or runs on a worker thread.


@pytest.mark.parametrize(
    "device, count, mask, cached",
    [
        (DeviceType.CUDA, 1, None, False),  # counted single GPU: answered inline
        (DeviceType.MLX, None, None, False),  # Apple: answered inline
        (None, None, None, None),  # detection not run yet
        (DeviceType.CUDA, None, None, None),  # unmasked, never counted
        (DeviceType.CUDA, 2, "GPU-abc,GPU-def", None),  # UUID mask may run nvidia-smi
    ],
)
def test_cached_gate_defers_anything_that_could_probe(monkeypatch, device, count, mask, cached):
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "DEVICE", device)
    monkeypatch.setattr(hw, "_physical_gpu_count", count)
    monkeypatch.setattr(hw, "get_device", lambda: device)
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: list(range(count or 0)))
    env = {} if mask is None else {"CUDA_VISIBLE_DEVICES": mask}
    with patch.dict(os.environ, env, clear = True):
        assert dd.auto_cuda_ranking_possible_cached() is cached


def test_route_runs_an_uncached_gate_off_the_event_loop(monkeypatch):
    from routes import inference as routes

    monkeypatch.setattr(dd, "auto_cuda_ranking_possible_cached", lambda: None)
    loop_thread = []

    def _gate():
        import threading
        loop_thread.append(threading.current_thread() is threading.main_thread())
        return False

    monkeypatch.setattr(dd, "auto_cuda_ranking_possible", _gate)
    targets = _counting_target(monkeypatch, dd)
    assert asyncio.run(routes._selected_gpu_ordinal(None)) is None
    assert loop_thread == [False]
    assert targets == []


def test_identical_rows_with_a_mig_card_are_not_identical(monkeypatch):
    from utils.hardware import nvidia

    row = ("NVIDIA A100-SXM4-80GB", "81920 MiB", "8.0", "1410 MHz")
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: 2)
    monkeypatch.setattr(nvidia, "get_card_identities", lambda: {0: row, 1: row})
    monkeypatch.setattr(hw, "_cuda_card_identities", {})
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: {1})
    assert hw._cuda_cards_identical() is False
    assert hw._cuda_cards_identical([0]) is True


def test_catalog_compares_every_card_when_mask_ids_are_not_smi_rows(monkeypatch):
    # FASTEST_FIRST subset mask "0,2": those are CUDA ordinals, not nvidia-smi rows 0 and 2.
    routes, planned = _catalog(monkeypatch)
    _ordinary(monkeypatch, visible = (0, 2))
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: False)
    asked = []
    # Rows 0 and 2 match, row 1 differs: only the all-cards answer is trustworthy.
    monkeypatch.setattr(
        hw,
        "_cuda_cards_identical",
        lambda gpu_ids = None: asked.append(gpu_ids) or gpu_ids is not None,
    )
    _counting_target(monkeypatch, dd)
    ranks = _counting_rank(monkeypatch, answer = 0)
    routes._validated_media_picks(1.0)
    assert asked == [None]
    assert len(ranks) == 1


@pytest.mark.parametrize(
    "listing",
    [
        types.SimpleNamespace(returncode = 1, stdout = ""),  # nvidia-smi -L failed
        types.SimpleNamespace(returncode = 0, stdout = ""),  # empty output
        "timeout",
    ],
)
def test_failed_mig_discovery_is_unknown_not_identical(monkeypatch, listing):
    import subprocess

    from utils.hardware import nvidia

    def _run(argv, **_k):
        if listing == "timeout":
            raise subprocess.TimeoutExpired(argv, 5)
        return listing

    row = ("NVIDIA A100-SXM4-80GB", "81920 MiB", "8.0", "1410 MHz")
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: 2)
    monkeypatch.setattr(nvidia, "get_card_identities", lambda: {0: row, 1: row})
    monkeypatch.setattr(hw, "_cuda_card_identities", {})
    monkeypatch.setattr(nvidia.gpu_query, "run_nvidia_smi", _run)
    assert hw._nvidia_mig_gpu_indices_or_none() is None
    assert hw._cuda_cards_identical() is None
    # The load path keeps card 0 explicit when MIG is unknown.
    _two_cards(monkeypatch, [_dev(0, 80.0, 0.5), _dev(1, 80.0, 0.5)], {0: (8, 0), 1: (8, 0)})
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: None)
    assert dd.resolve_auto_cuda_ordinal() == 0


def test_mig_listing_parses_slices():
    from unittest.mock import patch as _patch

    from utils.hardware import nvidia

    out = "GPU 0: A100 (UUID: GPU-a)\n  MIG 3g.40gb Device 0: (UUID: MIG-x)\nGPU 1: A100 (UUID: GPU-b)\n"
    with _patch.object(
        nvidia.gpu_query,
        "run_nvidia_smi",
        lambda argv, **_k: types.SimpleNamespace(returncode = 0, stdout = out),
    ):
        assert hw._nvidia_mig_gpu_indices_or_none() == {0}


def test_fastest_first_mask_with_an_integrated_card_keeps_torch(monkeypatch):
    # [] = mapping known but an integrated card is mapped: its nvidia-smi row is a carve-out.
    torch_calls, reads = _fastest_first_host(monkeypatch, [])
    with patch.dict(os.environ, _FASTEST, clear = True):
        assert hw._smi_rows_for_cuda_ordinals([0, 1]) is None
        assert hw._smi_rows_for_cuda_ordinals([0, 1]) is None
        hw.auto_select_gpu_ids("m", required_override_gb = 8.0)
    assert torch_calls == [[0, 1]]
    assert reads == [1]  # cached, the child is not re-run


@pytest.mark.parametrize(
    "child_out, expected",
    [
        ("0000:3b:00.0 0\n0000:1a:00.0 0\n", [1, 0]),
        ("0000:3b:00.0 0\n0000:1a:00.0 1\n", []),  # integrated card mapped
        ("0000:3b:00.0\n", None),  # no attribute: unreadable
    ],
)
def test_cuda_ordinal_rows_parse_bus_ids_and_integrated(monkeypatch, child_out, expected):
    import subprocess

    from utils.hardware import nvidia

    child = types.SimpleNamespace(returncode = 0, stdout = child_out)
    smi = types.SimpleNamespace(returncode = 0, stdout = "0, 00000000:1A:00.0\n1, 00000000:3B:00.0\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: child)
    monkeypatch.setattr(nvidia.gpu_query, "run_nvidia_smi", lambda *a, **k: smi)
    assert nvidia.get_cuda_ordinal_smi_rows() == expected


def test_identity_compares_every_listed_row_not_a_stale_count(monkeypatch):
    # A torch-fallback count of 2 must not hide a third, different card nvidia-smi lists.
    from utils.hardware import nvidia

    row = ("NVIDIA A100-SXM4-80GB", "81920 MiB", "8.0", "1410 MHz")
    other = ("NVIDIA RTX A6000", "49140 MiB", "8.6", "2100 MHz")
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: 2)
    monkeypatch.setattr(nvidia, "get_card_identities", lambda: {0: row, 1: row, 2: other})
    monkeypatch.setattr(hw, "_cuda_card_identities", {})
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: set())
    assert hw._cuda_cards_identical() is False
    assert hw._cuda_cards_identical([0, 1]) is True


def test_hardware_redetection_retires_the_ordinal_mapping(monkeypatch):
    from utils.hardware import gpu_query

    _torch_calls, reads = _fastest_first_host(monkeypatch, [0, 1])
    assert _auto(_FASTEST) == [1]
    gpu_query.invalidate_static("test: eGPU attached")
    assert _auto(_FASTEST) == [1]
    assert reads == [1, 1]


def test_cold_rocm_detection_is_settled_before_the_gate_reads_it(monkeypatch):
    _ordinary(monkeypatch, visible = (0, 1))
    monkeypatch.setattr(hw, "IS_ROCM", False)

    def cold_detect():
        hw.IS_ROCM = True  # ROCm reports as CUDA once detection has run
        return DeviceType.CUDA

    monkeypatch.setattr(hw, "get_device", cold_detect)
    assert dd.auto_cuda_ranking_possible() is False
