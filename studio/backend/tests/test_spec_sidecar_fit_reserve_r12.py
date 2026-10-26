# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Twelfth-round cells for the DSpark/DFlash sidecar reserve and the Metal OOM latch.

Planning and spawn agree on where a positional reserve is provable (Vulkan, an unresolvable
draft pin, a torch-ordinal parent whose child switches device order), host-resident draft
KV is judged by the cards the child really runs on, and a server retired after its second
Metal OOM is neither "loading" nor able to re-arm the latch after an explicit reload.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_metal_poisoned_respawn as _respawn  # noqa: E402
import test_llama_cpp_placement as _placement  # noqa: E402
import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r4 as _r4  # noqa: E402
import test_spec_sidecar_fit_reserve_r7 as _r7  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402

harness = _respawn.harness  # the respawn fixture, reused as-is

_GIB = 1024 * 1024 * 1024
_TWO = ((0, 24_000, 24_000), (1, 24_000, 24_000))
_MANUAL_AUTO = {"gpu_memory_mode": "manual", "gpu_layers": -1}


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)
    monkeypatch.delenv("LLAMA_ARG_SPLIT_MODE", raising = False)
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(
        _lc.LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0)
    )


_spy_fit_sized = _r2._spy_fit_sized  # planning plus the spawn's report


def _two_cards(
    tmp_path,
    memory = _TWO,
    vulkan = False,
):
    backend, gguf = _r2._cuda_backend(tmp_path, memory = memory, vulkan = vulkan)
    backend._arch_gate_survivors = lambda binary = None: []
    return backend, gguf


# Planning and spawn agree on Vulkan.


def test_an_unpinned_multi_card_vulkan_launch_gets_no_positional_reserve(tmp_path):
    """Vulkan orders its devices its own way, so planning already reports no reserve; the
    spawn must not then emit per-card shares in the parent's order."""
    backend, gguf = _two_cards(tmp_path, vulkan = True)
    seen = _spy_fit_sized(backend)
    # Manual with Auto layers: no Studio --device pin, the child sees every card.
    cmd = _r7._dspark(backend, gguf, tmp_path, **_MANUAL_AUTO)
    assert "--device" not in cmd
    assert seen["fit_sized"] is False
    assert len(_r7._margins(cmd)) == 1


# Planning reports a draft pin as reserved only where the spawn resolves it.


@pytest.mark.parametrize(
    "extras, vulkan, reserved",
    [
        # Off the target's two cards.
        (["--device-draft", "CUDA2"], False, False),
        # A repeated target entry leaves the pin's position unknown.
        (["--device", "CUDA0,CUDA1,CUDA0", "--device-draft", "CUDA0"], False, False),
        # A Vulkan name with no target list to find it in (Manual with Auto layers).
        (["--device-draft", "Vulkan0"], True, False),
        # Controls: a pin the spawn resolves.
        (["--device-draft", "CUDA1"], False, True),
        (["--device", "CUDA0,CUDA1", "--device-draft", "CUDA1"], False, True),
    ],
)
def test_planning_reports_only_a_draft_pin_the_spawn_resolves(
    tmp_path, monkeypatch, extras, vulkan, reserved
):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    backend, gguf = _two_cards(tmp_path, vulkan = vulkan)
    seen = _spy_fit_sized(backend)
    cmd = _r7._dspark(
        backend, gguf, tmp_path, extra_args = list(extras), **(_MANUAL_AUTO if vulkan else {})
    )
    assert (len(_r7._margins(cmd)) == 2) is reserved
    assert seen["fit_sized"] is reserved


def test_planning_resolves_a_draft_pin_against_studios_vulkan_pin(tmp_path):
    """gpu_ids on Vulkan reaches the child as --device Vulkan0,Vulkan1 (added after planning),
    so a pin on one of those cards is reserved in both."""
    backend, gguf = _two_cards(tmp_path, vulkan = True)
    seen = _spy_fit_sized(backend)
    cmd = _r7._dspark(
        backend, gguf, tmp_path, gpu_ids = [0, 1], extra_args = ["--device-draft", "Vulkan1"]
    )
    assert len(_r7._margins(cmd)) == 2
    assert seen["fit_sized"] is True


# A torch-ordinal parent whose child is switched to PCI_BUS_ID.


@pytest.mark.parametrize("parent_order, reserved", [("FASTEST_FIRST", False), ("PCI_BUS_ID", True)])
def test_torch_ordinals_in_another_order_than_the_child_get_no_positional_reserve(
    tmp_path, monkeypatch, parent_order, reserved
):
    """No nvidia-smi: the ids are torch ordinals in the parent's CUDA_DEVICE_ORDER, while a
    GPU pick pins the child to PCI_BUS_ID, so shares by position could land on other cards."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", parent_order)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    B = _lc.LlamaCppBackend
    monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", False)
    monkeypatch.setattr(B, "_host_torch_is_rocm", staticmethod(lambda: False))
    backend, gguf = _two_cards(tmp_path, memory = ((0, 80_000, 80_000), (1, 8_000, 8_000)))
    seen = _spy_fit_sized(backend)
    cmd = _r7._dspark(backend, gguf, tmp_path, gpu_ids = [0, 1])
    assert (len(_r7._margins(cmd)) == 2) is reserved
    assert seen["fit_sized"] is reserved


def test_the_order_check_is_one_helper_for_planning_and_spawn(monkeypatch):
    B = _lc.LlamaCppBackend
    monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", False)
    monkeypatch.setattr(B, "_host_torch_is_rocm", staticmethod(lambda: False))
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)
    backend = B()
    # Unset is CUDA's default, FASTEST_FIRST.
    assert backend._positional_child_order_unproven({}) is False
    assert backend._positional_child_order_unproven({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}) is False
    assert backend._positional_child_order_unproven({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}) is True
    # HIP ignores CUDA_DEVICE_ORDER.
    monkeypatch.setattr(B, "_host_torch_is_rocm", staticmethod(lambda: True))
    assert backend._positional_child_order_unproven({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}) is False


# Host-resident draft KV follows the cards the child runs on.


def _nkvo_margins(
    tmp_path,
    monkeypatch,
    *,
    memory,
    vulkan,
    apu = None,
    **kw,
):
    out = []
    for name, extras in (("on", []), ("off", ["-nkvo"])):
        d = tmp_path / name
        d.mkdir()
        backend, gguf = _r2._cuda_backend(d, memory = memory, vulkan = vulkan)
        backend._arch_gate_survivors = lambda binary = None: []
        backend._mtp_draft_kv_bytes = lambda *a, **k: 768 * 1024 * 1024
        if apu is not None:
            backend._amd_apu_wants_unified_memory = (
                lambda gpu_indices = None: gpu_indices is None or apu in gpu_indices
            )
        backend._integrated_cuda_unified_memory = lambda gpu_indices = None: False
        # What --list-devices reports: a --device the build never enumerated is not followed.
        names = [f"{'Vulkan' if vulkan else 'CUDA'}{idx}" for idx, _f, _t in memory]
        monkeypatch.setattr(
            _lc.LlamaCppBackend,
            "_enumerated_gpu_devices",
            classmethod(lambda cls, binary = None, env = None, _n = names: list(_n)),
        )
        extra = list(kw.get("extra_args", [])) + extras
        args = {k: v for k, v in kw.items() if k != "extra_args"}
        out.append(_r4._margin(_r7._dspark(backend, gguf, d, extra_args = extra, **args)))
    return out


def test_a_discrete_card_picked_beside_an_apu_leaves_host_kv_out(tmp_path, monkeypatch):
    """APU 0 plus discrete card 1, the launch pinned to card 1: -nkvo keeps the draft KV in
    host memory, which that card does not hold."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    on, off = _nkvo_margins(
        tmp_path,
        monkeypatch,
        memory = ((0, 24_000, 24_000), (1, 24_000, 24_000)),
        vulkan = False,
        apu = 0,
        gpu_ids = [1],
    )
    assert on - off >= 760


@pytest.mark.parametrize("device, shared", [("Vulkan0", True), ("Vulkan1", False)])
def test_a_vulkan_igpu_picked_by_device_keeps_host_kv(tmp_path, monkeypatch, device, shared):
    """Vulkan iGPU 0 (reports total 0) plus dGPU 1: host memory is the iGPU's VRAM, so its
    margin keeps the draft KV under -nkvo; the dGPU's does not."""
    on, off = _nkvo_margins(
        tmp_path,
        monkeypatch,
        memory = ((0, 24_000, 0), (1, 24_000, 24_000)),
        vulkan = True,
        extra_args = ["--device", device],
        **_MANUAL_AUTO,
    )
    if shared:
        assert on == off
    else:
        assert on - off >= 760


# The terminal Metal OOM latch.


def test_a_terminally_retired_server_is_not_active(harness):
    """After the second Metal OOM the child is reaped and never respawned, so delete,
    Always mode and the training guard must not see a model still loading."""
    intent = _respawn._Intent(speculative_type = "off")
    backend, first, _loads = harness(intent, "none")
    backend._retire_device_lost_server(first, metal_poison = {"draft": False})
    # Strike one: the next request respawns it.
    assert backend._respawn_if_dead() is True
    assert backend.is_active
    second = backend._process
    backend._retire_device_lost_server(second, metal_poison = {"draft": False})
    assert second.poll() is not None
    # Nothing respawns it, so it is unloaded like any other model (r41).
    assert backend._process is None and backend._model_identifier is None
    assert backend._respawn_if_dead() is False
    assert backend.is_loaded is False
    # routes/models.py refuses a delete with 409 while is_active or is_loaded.
    assert backend.is_active is False


def _stub_child():
    return type(
        "Process",
        (),
        {
            "pid": 123,
            "poll": lambda self: 0,
            "kill": lambda self: None,
            "wait": lambda self, timeout = None: 0,
        },
    )()


def test_an_explicit_reload_with_changed_settings_clears_the_terminal_latch(tmp_path):
    """Two Metal OOM strikes, then an explicit reload that changes the GPU pick: the latch
    must not survive and re-arm once a matching request writes gpu_ids back."""
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._speculative_type = "none"
    _placement._launch(backend, gguf, n_ctx = 4096)
    committed = backend._last_load_intent
    assert committed is not None and committed.gpu_ids is None
    # Two strikes on the committed load: the second is terminal.
    for _strike in range(2):
        dead = _stub_child()
        backend._process = dead
        backend._retired_process = dead
        backend._after_metal_poisoned_server(False)
    assert backend._metal_oom_is_terminal(committed)
    # The user's explicit reload, now pinned to GPU 0.
    _placement._launch(backend, gguf, n_ctx = 4096, gpu_ids = [0])
    assert backend.is_active and backend.is_loaded
    reloaded = backend._last_load_intent
    assert list(reloaded.gpu_ids) == [0]
    # Not vacuous: writing gpu_ids back reproduces the latched load exactly.
    assert backend._metal_oom_key(replace(reloaded, gpu_ids = None)) == backend._metal_oom_key(
        committed
    )
    assert backend._metal_oom_terminal_key is None
    assert backend._metal_oom_strikes is None
    backend._record_matching_gpu_request(None)
    assert backend._metal_oom_is_terminal(backend._last_load_intent) is False
