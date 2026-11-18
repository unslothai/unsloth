# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sidecar reserve across several cards: per-card shares, draft pins, split modes, host KV."""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pytest

from . import _sidecar_fit_helpers as h
from ._sidecar_fit_helpers import FLOOR, GIB, MANUAL_AUTO, TWO, lc

B = lc.LlamaCppBackend
_MIXED = ((0, 49_140, 49_140), (1, 24_576, 24_576))
_VK_IGPU = ((0, 24_000, 0), (1, 24_000, 24_000))
_COMPUTE_FLOOR = math.ceil(B._MTP_DRAFT_COMPUTE_BYTES / h.MIB)


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    for k in ("CUDA_VISIBLE_DEVICES", "LLAMA_ARG_DEVICE", "LLAMA_ARG_SPLIT_MODE", "LLAMA_ARG_RPC"):
        monkeypatch.delenv(k, raising = False)
    # A discrete host on every runner, class-level budget reads included.
    monkeypatch.setattr(lc, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(B, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0))


@pytest.fixture
def _pci_ids(monkeypatch):
    """Studio's GPU ids are nvidia-smi's PCI indices and no mask needs a UUID resolver."""
    monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(B, "_smi_uuid_mask_order", staticmethod(lambda: None))
    monkeypatch.setattr(B, "_visibility_mask_is_unmappable", staticmethod(lambda: False))


@pytest.fixture
def _budget(monkeypatch, _pci_ids):
    """A 30% VRAM budget, so each card gets its own --fit-target margin."""
    monkeypatch.setattr(lc, "_active_vram_fraction", lambda: 0.3)
    monkeypatch.setattr(B, "_SMI_MIG_INDICES", frozenset())
    monkeypatch.setattr(B, "_host_torch_is_rocm", staticmethod(lambda: False))


def _band(m, lo, hi):
    return 1024 + lo * 1024 <= m < 1024 + hi * 1024


def test_multi_gpu_cuda_does_not_broadcast_the_reserve(tmp_path):
    """A scalar --fit-target applies to every device: adding the whole sidecar to it would
    withhold the sidecar on each card and push target layers to the host."""
    backend, gguf = h.cuda_backend(tmp_path, memory = TWO)
    cmd = h.dspark(backend, gguf, tmp_path)
    assert h.last(cmd, "--model-draft") == str(tmp_path / "dspark-model-Q8_0.gguf")
    assert h.margin(cmd) < FLOOR


@pytest.mark.parametrize(
    "args, env",
    [(["--device", "Vulkan0,Vulkan1"], {}), ([], {"LLAMA_ARG_DEVICE": "CUDA0,CUDA1"})],
    ids = ["argv", "env"],
)
def test_no_scalar_reserve_on_a_multi_device_launch(args, env):
    argv = ["--fit", "on", "-md", "/d.gguf", *args]
    assert lc._sidecar_reserve_margin_flags(argv, (), env, 1024, 8192) == []


def test_a_single_named_device_keeps_the_reserve():
    argv = ["--fit", "on", "-md", "/d.gguf", "--device", "CUDA0"]
    assert lc._sidecar_reserve_margin_flags(argv, (), {}, 1024, 8192) == ["--fit-target", "9216"]


def test_extras_widening_the_device_list_get_no_scalar_reserve(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path, memory = TWO, vulkan = True)
    backend._select_gpus = lambda *a, **k: ([0], True)
    cmd = h.dspark(backend, gguf, tmp_path, extra_args = ("--device", "Vulkan0,Vulkan1"))
    assert h.margin(cmd) < FLOOR


def test_one_named_device_on_a_two_gpu_host_keeps_the_reserve(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path, memory = TWO, vulkan = True)
    cmd = h.dspark(backend, gguf, tmp_path, extra_args = ("--device", "Vulkan0"))
    assert h.last(cmd, "--device") == "Vulkan0"
    assert h.margin(cmd) >= FLOOR


def test_two_detected_gpus_without_a_device_pin_stay_unreserved(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path, memory = TWO, vulkan = True)
    assert h.margin(h.dspark(backend, gguf, tmp_path)) < FLOOR


@pytest.mark.parametrize("devices", ["CUDA0,CUDA0", "CUDA0, cuda0"])
def test_a_repeated_device_is_one_card_for_the_reserve(tmp_path, devices):
    backend, gguf = h.cuda_backend(tmp_path)
    assert h.margin(h.dspark(backend, gguf, tmp_path, extra_args = ["--device", devices])) >= FLOOR


def test_a_launch_the_arch_gate_narrows_to_one_card_gets_the_reserve(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path, memory = TWO)
    backend._arch_gate_survivors = lambda binary = None: [1]
    assert h.margin(h.dspark(backend, gguf, tmp_path)) >= FLOOR


def test_an_arch_retry_onto_several_gpus_drops_the_scalar_reserve(tmp_path):
    rows = ((0, 24_000, 24_000), (1, 12_000, 12_000), (2, 12_000, 12_000))
    backend, gguf = h.cuda_backend(tmp_path, memory = rows)
    backend._select_gpus = lambda *a, **k: ([0], True)
    backend._arch_crash_retry_gpu_ids = lambda *a, **k: [1, 2]
    crash = ("ggml_cuda_compute_forward: ROCm error: device kernel image is invalid",)
    cmds = h.spawns(
        backend,
        gguf,
        stdout = lambda i, cmd: crash if i == 0 else (),
        healthy = lambda i, cmd: i > 0,
        crash = lambda i, cmd: None if i > 0 else 1,
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = h.sidecar(tmp_path),
    )
    assert len(cmds) >= 2, [c[-6:] for c in cmds]
    assert h.margin(cmds[0]) >= FLOOR
    assert "--model-draft" in cmds[1]
    assert h.margin(cmds[1]) < FLOOR


def test_a_launch_left_on_two_cards_reserves_each_cards_share(tmp_path):
    backend, gguf = h.cards(tmp_path)
    margins = h.margins(h.dspark(backend, gguf, tmp_path))
    assert len(margins) == 2
    assert all(_band(m, 4, 5) for m in margins)


@pytest.mark.parametrize(
    "memory, extras",
    [
        # 30,000 of 40,000 MiB free: three quarters of the ~8.3 GiB reserve on card 0.
        (((0, 30_000, 30_000), (1, 10_000, 10_000)), []),
        # The drafter inherits the target's --tensor-split even on equally free cards.
        (TWO, ["--tensor-split", "3,1"]),
    ],
    ids = [
        "test_unequal_cards_carry_their_own_shares",
        "test_an_explicit_tensor_split_decides_the_draft_shares",
    ],
)
def test_draft_shares_three_to_one(tmp_path, memory, extras):
    backend, gguf = h.cards(tmp_path, memory)
    big, small = h.margins(h.dspark(backend, gguf, tmp_path, extra_args = extras))
    assert _band(big, 6, 7) and _band(small, 2, 3)


def test_a_small_card_beside_a_large_one_is_not_over_reserved(tmp_path):
    """A broadcast of the large card's share (~7.6 GiB) would exceed the 8,000 MiB card."""
    backend, gguf = h.cards(tmp_path, ((0, 80_000, 80_000), (1, 8_000, 8_000)))
    big, small = h.margins(h.dspark(backend, gguf, tmp_path))
    assert big >= 1024 + 7 * 1024
    assert small < 1024 + 1024


def test_per_device_margins_follow_an_inherited_mask_order(tmp_path, monkeypatch):
    """CUDA_VISIBLE_DEVICES=1,0 makes physical 1 the child's device 0."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    backend, gguf = h.cards(tmp_path, ((0, 30_000, 30_000), (1, 10_000, 10_000)))
    first, second = h.margins(h.dspark(backend, gguf, tmp_path))
    assert first < second


def _three_blocks(
    tmp_path,
    graph_gib = None,
    target_layers = None,
):
    backend, gguf = h.cards(tmp_path)
    h.three_block_drafter(backend)
    if graph_gib is not None:
        backend._mtp_draft_compute_bytes = lambda *a, **k: graph_gib * GIB
    if target_layers is not None:
        backend._n_layers = target_layers
    return backend, gguf


@pytest.mark.parametrize(
    "extras, bands",
    [
        # Equal cards: blocks 0-1 below 2/4, block 2 at it, so two on card 0 and one on card 1.
        ([], [(5, 6), (2, 3)]),
        # 3:2 keeps every block under 0.6 of n_layer + 1, so all three on card 0.
        (["--tensor-split", "3,2"], [(8, 9), (0, 1)]),
    ],
)
def test_draft_shares_follow_llama_cpps_whole_layer_split(tmp_path, extras, bands):
    backend, gguf = _three_blocks(tmp_path)
    margins = h.margins(h.dspark(backend, gguf, tmp_path, extra_args = extras))
    for m, (lo, hi) in zip(margins, bands):
        assert _band(m, lo, hi), margins


def test_the_output_card_keeps_the_draft_compute_buffers_without_a_block(tmp_path):
    """A 3:2 split gives the output card no block, yet the draft still runs on it."""
    backend, gguf = _three_blocks(tmp_path)
    _big, small = h.margins(h.dspark(backend, gguf, tmp_path, extra_args = ["--tensor-split", "3,2"]))
    assert small >= 1024 + _COMPUTE_FLOOR


def test_the_output_card_keeps_the_whole_vocabulary_graph(tmp_path):
    """Borrowed-vocabulary draft logits are sized on the output card, not block shares."""
    backend, gguf = _three_blocks(tmp_path, graph_gib = 3)
    _big, small = h.margins(h.dspark(backend, gguf, tmp_path, extra_args = ["--tensor-split", "3,2"]))
    assert small >= 1024 + _COMPUTE_FLOOR + 3 * 1024


def test_a_split_that_leaves_the_last_card_out_keeps_the_graph_on_the_used_one(tmp_path):
    backend, gguf = _three_blocks(tmp_path, graph_gib = 3)
    used, idle = h.margins(h.dspark(backend, gguf, tmp_path, extra_args = ["--tensor-split", "1,0"]))
    assert used >= 1024 + 3 * 1024
    assert idle < 2048


def test_the_graph_follows_the_targets_output_layer_not_the_drafters(tmp_path):
    """DSpark borrows the target's output tensor: with one target block and a 3:2 split the
    target's output layer (index 1 of 2 units = 0.5 < 0.6) is on card 0, so the graph is too."""
    backend, gguf = _three_blocks(tmp_path, graph_gib = 3, target_layers = 1)
    used, idle = h.margins(h.dspark(backend, gguf, tmp_path, extra_args = ["--tensor-split", "3,2"]))
    assert used >= 1024 + 3 * 1024 + 4 * 1024
    assert idle < 2048


def test_a_draft_with_its_own_head_keeps_its_graph_on_its_own_output_card(tmp_path, monkeypatch):
    """Same split, but the draft owns its head: its output layer (3 / 4 = 0.75) is on card 1."""
    monkeypatch.setattr(lc, "_sidecar_borrows_output", lambda path: False)
    backend, gguf = _three_blocks(tmp_path, graph_gib = 3, target_layers = 1)
    _used, out = h.margins(h.dspark(backend, gguf, tmp_path, extra_args = ["--tensor-split", "3,2"]))
    assert out >= 1024 + 3 * 1024


def _owned_head_margins(tmp_path, monkeypatch, head_gib):
    monkeypatch.setattr(lc, "_sidecar_borrows_output", lambda path: False)
    backend, gguf = _three_blocks(tmp_path, target_layers = 1)
    read = backend._tensor_spill_layout

    def layout(path, **kw):
        if "dspark" in str(path):
            return SimpleNamespace(complete = True, lm_head_bytes = int(head_gib * GIB))
        return read(path, **kw)

    backend._tensor_spill_layout = layout
    return h.margins(h.dspark(backend, gguf, tmp_path, extra_args = ["--tensor-split", "3,2"]))


def test_an_owned_head_is_reserved_on_its_output_card(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    no_head = _owned_head_margins(tmp_path / "a", monkeypatch, 0)
    head = _owned_head_margins(tmp_path / "b", monkeypatch, 2)
    assert head[1] - no_head[1] >= 2000
    assert no_head[0] - head[0] >= 2000


def test_a_row_split_places_whole_draft_layers_like_a_layer_split(tmp_path):
    """--split-mode row assigns blocks to devices like a layer split (llama-model.cpp)."""
    (tmp_path / "row").mkdir()
    backend, gguf = _three_blocks(tmp_path / "row")
    row = h.margins(h.dspark(backend, gguf, tmp_path, extra_args = ["--split-mode", "row"]))
    backend, gguf = _three_blocks(tmp_path)
    layer = h.margins(h.dspark(backend, gguf, tmp_path))
    assert row == layer
    assert row[0] > row[1]


def test_the_underscore_tensor_split_spelling_places_the_draft(tmp_path):
    backend, gguf = _three_blocks(tmp_path)
    margins = h.margins(h.dspark(backend, gguf, tmp_path, extra_args = ["--tensor_split", "1,0"]))
    assert len(margins) == 2, margins
    assert margins[0] >= FLOOR
    assert margins[1] < 2048


def test_a_tensor_split_launch_gets_no_per_device_reserve(tmp_path):
    """--split-mode tensor folds cards into one meta device; no per-card --fit-target applies."""
    backend, gguf = _three_blocks(tmp_path)
    margins = h.margins(h.dspark(backend, gguf, tmp_path, extra_args = ["--split-mode", "tensor"]))
    assert all(m < 2048 for m in margins), margins


def test_shares_follow_each_cards_free_memory_past_its_own_margin(tmp_path):
    """A user margin of 20,000 MiB leaves card 0 about 4 GB for weights against card 1's 23 GB."""
    backend, gguf = h.cards(tmp_path)
    margins = h.margins(
        h.dspark(backend, gguf, tmp_path, extra_args = ["--fit-target", "20000,1024"])
    )
    assert len(margins) == 2
    assert margins[0] < 24_000
    assert margins[1] > 1024 + 4 * 1024


@pytest.mark.parametrize("order, per_card", [("FASTEST_FIRST", False), ("PCI_BUS_ID", True)])
def test_per_card_shares_need_a_provable_device_order(tmp_path, monkeypatch, order, per_card):
    """nvidia-smi keys free memory by PCI order, so per-card shares need PCI_BUS_ID ordering."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(B, "_smi_uuid_mask_order", staticmethod(lambda: None))
    backend, gguf = h.cards(tmp_path, ((0, 80_000, 80_000), (1, 8_000, 8_000)))
    assert (len(h.margins(h.dspark(backend, gguf, tmp_path))) == 2) is per_card


@pytest.mark.parametrize("parent_order, reserved", [("FASTEST_FIRST", False), ("PCI_BUS_ID", True)])
def test_torch_ordinals_in_another_order_than_the_child_get_no_positional_reserve(
    tmp_path, monkeypatch, parent_order, reserved
):
    """No nvidia-smi: the ids are torch ordinals in the parent's CUDA_DEVICE_ORDER, while a
    GPU pick pins the child to PCI_BUS_ID."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", parent_order)
    monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", False)
    monkeypatch.setattr(B, "_host_torch_is_rocm", staticmethod(lambda: False))
    backend, gguf = h.cards(tmp_path, ((0, 80_000, 80_000), (1, 8_000, 8_000)))
    seen = h.spy_fit_sized(backend)
    cmd = h.dspark(backend, gguf, tmp_path, gpu_ids = [0, 1])
    assert (len(h.margins(cmd)) == 2) is reserved
    assert seen["fit_sized"] is reserved


def test_the_order_check_is_one_helper_for_planning_and_spawn(monkeypatch):
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


def _mig_two_cards(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(B, "_smi_uuid_mask_order", staticmethod(lambda: None))
    # A MIG mask (child order: card 1's slice, then card 0's) no resolver can map.
    monkeypatch.setattr(B, "_visibility_mask_is_unmappable", staticmethod(lambda: True))
    return h.cards(tmp_path, ((0, 16_000, 16_000), (1, 4_000, 4_864)))


def test_an_unmappable_mig_mask_gets_no_positional_reserve(tmp_path, monkeypatch):
    backend, gguf = _mig_two_cards(tmp_path, monkeypatch)
    assert len(h.margins(h.dspark(backend, gguf, tmp_path))) == 1


def test_an_unmappable_mig_mask_is_not_reported_as_reserved(tmp_path, monkeypatch):
    backend, gguf = _mig_two_cards(tmp_path, monkeypatch)
    seen = h.spy_fit_sized(backend)
    h.dspark(backend, gguf, tmp_path)
    assert seen["fit_sized"] is False


def test_a_users_per_device_margins_stay_per_device_across_cards():
    flags = lc._sidecar_reserve_margin_flags(
        ["--model-draft", "/d.gguf", "--fit", "on"],
        ["--fit-target", "1024,20000"],
        {},
        1024,
        4213,
        per_device = [4213, 4213],
    )
    assert flags == ["--fit-target", "5237,24213"]


def test_a_scalar_user_margin_is_broadcast_under_per_device_shares():
    flags = lc._sidecar_reserve_margin_flags(
        ["--model-draft", "/d.gguf", "--fit", "on"],
        ["--fit-target", "2048"],
        {},
        1024,
        5000,
        per_device = [5000, 500],
    )
    assert flags == ["--fit-target", "7048,2548"]


@pytest.mark.parametrize(
    "user, env, want",
    [
        # A short list sets only the entries it names; the rest keep the earlier value.
        (["--fit-target", "4096", "--fit-target", "2048,3072"], {}, "3048,4072,5096"),
        # The env applies first, the argv after it.
        ([], {"LLAMA_ARG_FIT_TARGET": "4096"}, "5096,5096,5096"),
        (["--fit-target", "2048"], {"LLAMA_ARG_FIT_TARGET": "4096,4096,8192"}, "3048,3048,3048"),
        (["--fit-target=2048,3072"], {"LLAMA_ARG_FIT_TARGET": "8192"}, "3048,4072,9192"),
        (["--fit_target", "4096"], {}, "5096,5096,5096"),
        (["--fit_target=2048,3072", "-fitt", "4096"], {}, "5096,5096,5096"),
    ],
)
def test_per_device_margins_follow_llama_cpps_update_order(user, env, want):
    flags = lc._sidecar_reserve_margin_flags(
        ["--model-draft", "/d.gguf", "--fit", "on"],
        user,
        env,
        1024,
        1000,
        per_device = [1000, 1000, 1000],
    )
    assert flags == ["--fit-target", want]


@pytest.mark.parametrize(
    "user, base, reserve, per_device, by_device, want",
    [
        # The smaller card's lower margin is not raised to the scalar.
        ([], 4000, 5000, [5000, 500], [4000, 1000], "9000,1500"),
        (["--fit-target", "2048"], 1024, 100, [100, 100], [1024, 4096], "2148,4196"),
        # A child masked to the smaller card carries that card's own margin.
        ([], 4000, 5000, None, [1000], "6000"),
    ],
    ids = [
        "test_studios_per_card_margins_are_kept_under_the_reserve",
        "test_a_users_margin_still_wins_over_a_lower_per_card_base",
        "test_one_card_keeps_its_own_emitted_base_under_the_reserve",
    ],
)
def test_the_reserve_sits_on_each_cards_own_base(user, base, reserve, per_device, by_device, want):
    kw = {} if per_device is None else {"per_device": per_device}
    flags = lc._sidecar_reserve_margin_flags(
        ["--model-draft", "/d.gguf", "--fit", "on"],
        user,
        {},
        base,
        reserve,
        base_by_device = by_device,
        **kw,
    )
    assert flags == ["--fit-target", want]


def _base_margins(
    tmp_path,
    memory = _MIXED,
    **kw,
):
    (tmp_path / "base").mkdir()
    backend, gguf = h.cards(tmp_path / "base", memory)
    return h.margins(h.placement._launch(backend, gguf, **kw)["cmd"])


def test_each_card_keeps_its_budget_margin_under_the_reserve(tmp_path, _budget):
    base = _base_margins(tmp_path)
    assert len(base) == 2 and base[0] != base[1]
    backend, gguf = h.cards(tmp_path, _MIXED)
    reserved = h.margins(h.dspark(backend, gguf, tmp_path))
    assert len(reserved) == 2
    assert all(r > b for r, b in zip(reserved, base))
    assert reserved[1] < 24_576


@pytest.mark.parametrize("main_gpu", [[], ["--main-gpu", "1"]])
def test_split_mode_none_keeps_the_other_cards_margins(tmp_path, _budget, main_gpu):
    """llama.cpp keeps only devices[main_gpu]; the reserve goes in entry 0 for any main card."""
    extras = ["--split-mode", "none", *main_gpu]
    base = _base_margins(tmp_path, extra_args = extras)
    backend, gguf = h.cards(tmp_path, _MIXED)
    reserved = h.margins(h.dspark(backend, gguf, tmp_path, extra_args = extras))
    assert len(reserved) == 2
    assert reserved[0] >= base[0] + 8 * 1024
    assert reserved[1] == base[1]


def test_one_local_card_with_rpc_gets_no_broadcast_reserve(tmp_path, _budget):
    """A scalar margin is broadcast to the remote devices too."""
    extras = ["--rpc", "10.0.0.2:50052"]
    (tmp_path / "base").mkdir()
    backend, gguf = h.cuda_backend(tmp_path / "base", memory = _MIXED[:1])
    base = h.margins(h.placement._launch(backend, gguf, extra_args = extras)["cmd"])
    backend, gguf = h.cuda_backend(tmp_path, memory = _MIXED[:1])
    assert h.margins(h.dspark(backend, gguf, tmp_path, extra_args = extras)) == base


def test_an_rpc_launch_gets_no_positional_reserve_list(tmp_path, _budget):
    backend, gguf = h.cards(tmp_path, _MIXED)
    cmd = h.dspark(backend, gguf, tmp_path, extra_args = ["--rpc", "10.0.0.2:50052"])
    assert len(h.margins(cmd)) == 1


def test_the_text_only_retry_leaves_the_reserve_pair_alone():
    source = "".join(inspect.getsource(B.load_model).split())
    at = source.index("cmd=self._strip_mmproj_args(_vision_gpu_cmd)")
    window = source[at : at + 400]
    assert "notin_sidecar_override_pairs" in window
    assert window.index("notin_sidecar_override_pairs") < window.index(
        "_per_device_fit_target_argv("
    )


@pytest.mark.parametrize("draft_device, reserved", [("Vulkan1", False), ("Vulkan0", True)])
def test_a_draft_pinned_to_another_vulkan_device_is_not_reserved_on_the_target(
    tmp_path, draft_device, reserved
):
    backend, gguf = h.cuda_backend(tmp_path, memory = TWO, vulkan = True)
    cmd = h.dspark(
        backend, gguf, tmp_path, gpu_ids = [0], extra_args = ["--device-draft", draft_device]
    )
    assert (h.margin(cmd) >= FLOOR) is reserved


@pytest.mark.parametrize("draft_device, reserved", [("Vulkan0", True), ("Vulkan1", False)])
def test_a_stale_device_extra_does_not_decide_the_draft_device(tmp_path, draft_device, reserved):
    """gpu_ids=[0] owns placement and strips the stale --device Vulkan1 at emission."""
    backend, gguf = h.cuda_backend(tmp_path, memory = TWO, vulkan = True)
    extras = ["--device", "Vulkan1", "--device-draft", draft_device]
    cmd = h.dspark(backend, gguf, tmp_path, gpu_ids = [0], extra_args = extras)
    assert (h.margin(cmd) >= FLOOR) is reserved


@pytest.mark.parametrize(
    "pin, reserved", [("CUDA0", True), ("CUDA1", False), ("Vulkan0", False), ("RPC0", False)]
)
def test_a_cuda_draft_pin_on_the_only_target_card_keeps_the_reserve(tmp_path, pin, reserved):
    """The child's mask numbers the single target card 0, so CUDA0 is the target."""
    backend, gguf = h.cuda_backend(tmp_path)
    cmd = h.dspark(backend, gguf, tmp_path, extra_args = ["--device-draft", pin])
    assert "--model-draft" in cmd
    assert (h.margin(cmd) >= FLOOR) is reserved


@pytest.mark.parametrize(
    "extras, pin, reserved",
    [
        ([], "Vulkan1", True),
        ([], "Vulkan0", True),
        (["--split-mode", "none"], "Vulkan1", False),
        (["--split-mode", "none"], "Vulkan0", True),
    ],
)
def test_a_pin_on_any_studio_chosen_vulkan_card_is_on_the_target(tmp_path, extras, pin, reserved):
    backend, gguf = h.cuda_backend(tmp_path, memory = TWO, vulkan = True)
    cmd = h.dspark(
        backend, gguf, tmp_path, gpu_ids = [0, 1], extra_args = [*extras, "--device-draft", pin]
    )
    assert (h.margin(cmd) >= FLOOR) is reserved


@pytest.mark.parametrize("pin, reserved", [("CUDA1", True), ("CUDA0", False)])
def test_an_inherited_target_device_decides_the_draft_pin(tmp_path, monkeypatch, pin, reserved):
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA1")
    backend, gguf = h.cuda_backend(tmp_path, memory = TWO)
    cmd = h.dspark(backend, gguf, tmp_path, extra_args = ["--spec-draft-device", pin])
    assert (h.margin(cmd) >= FLOOR) is reserved


@pytest.mark.parametrize(
    "argv, devices, on_target",
    [
        # The masked child's device 0 is the target.
        (["--spec-draft-device", "CUDA0"], None, True),
        (["--spec-draft-device", "CUDA1"], None, False),
        # A companion widening: the target is --device CUDA0, the drafter its CUDA1.
        (["--device", "CUDA0", "--spec-draft-device", "CUDA1"], "CUDA0", False),
        (["--device", "CUDA1", "--spec-draft-device", "CUDA1"], "CUDA1", True),
        # Settled at planning, where the backend is known.
        (["--spec-draft-device", "RPC0"], None, True),
        (["--spec-draft-device", "none"], None, True),
    ],
)
def test_the_spawned_argv_decides_a_cuda_draft_pin(argv, devices, on_target):
    assert lc._spawn_draft_on_target(argv, devices) is on_target


def test_a_repeated_device_is_one_card_for_the_draft_pin():
    argv = ["--device", "CUDA0,CUDA0", "--device-draft", "CUDA0"]
    assert lc._spawn_draft_on_target(argv, "CUDA0,CUDA0")
    assert not lc._spawn_draft_on_target(["--device-draft", "CUDA1"], "CUDA0,CUDA0")


def test_a_repeated_draft_pin_is_one_card():
    assert lc._spawn_draft_on_target(
        ["--device", "CUDA0", "--device-draft", "CUDA0,CUDA0"], "CUDA0"
    )


_NONE_MAIN_1 = ["--split-mode", "none", "--main-gpu", "1"]
_BOTH = ["--device", "CUDA0,CUDA1"]


@pytest.mark.parametrize(
    "argv, devices, on_target",
    [
        # --split-mode none puts the target on --main-gpu, not device 0.
        ([*_NONE_MAIN_1, "--spec-draft-device", "CUDA1"], None, True),
        ([*_NONE_MAIN_1, "--spec-draft-device", "CUDA0"], None, False),
        ([*_BOTH, *_NONE_MAIN_1, "--spec-draft-device", "CUDA1"], "CUDA0,CUDA1", True),
        ([*_BOTH, *_NONE_MAIN_1, "--spec-draft-device", "CUDA0"], "CUDA0,CUDA1", False),
    ],
)
def test_split_mode_none_compares_the_pin_with_the_main_gpu(argv, devices, on_target):
    assert lc._spawn_draft_on_target(argv, devices, {}) is on_target


@pytest.mark.parametrize("pin, on_target", [("CUDA0", True), ("CUDA1", False)])
def test_split_mode_none_defaults_to_the_lists_first_device(pin, on_target):
    argv = [*_BOTH, "--split-mode", "none", "--spec-draft-device", pin]
    assert lc._spawn_draft_on_target(argv, "CUDA0,CUDA1", {}) is on_target


def test_split_mode_none_keeps_the_whole_reserve_on_the_main_card(tmp_path):
    backend, gguf = h.cards(tmp_path)
    assert h.margin(h.dspark(backend, gguf, tmp_path, extra_args = ["--split-mode", "none"])) >= FLOOR


@pytest.mark.parametrize("pin, reserved", [("CUDA0", True), ("CUDA1", False)])
def test_split_mode_none_narrows_an_inherited_device_list_at_planning(
    tmp_path, monkeypatch, pin, reserved
):
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA0,CUDA1")
    backend, gguf = h.cuda_backend(tmp_path, memory = TWO)
    extras = ["--split-mode", "none", "--spec-draft-device", pin]
    assert (h.margin(h.dspark(backend, gguf, tmp_path, extra_args = extras)) >= FLOOR) is reserved


def test_planning_drops_an_inherited_split_mode_the_launch_clears(tmp_path, monkeypatch):
    """A layer launch clears an inherited non-layer LLAMA_ARG_SPLIT_MODE before spawning, so
    planning must not narrow the target to the main card on it."""
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA0,CUDA1")
    monkeypatch.setenv("LLAMA_ARG_SPLIT_MODE", "none")
    backend, gguf = h.cards(tmp_path)
    assert (
        h.margin(h.dspark(backend, gguf, tmp_path, extra_args = ["--device-draft", "CUDA1"])) >= FLOOR
    )


@pytest.mark.parametrize(
    "extras, at",
    [
        ([*_BOTH, "--spec-draft-device", "CUDA0"], 0),
        ([*_BOTH, "--spec-draft-device", "CUDA1"], 1),
        (["--spec-draft-device", "CUDA1"], 1),
    ],
)
def test_a_draft_pinned_to_one_target_card_reserves_there(tmp_path, extras, at):
    backend, gguf = h.cards(tmp_path)
    margins = h.margins(h.dspark(backend, gguf, tmp_path, extra_args = extras))
    assert len(margins) == 2, margins
    assert margins[at] >= FLOOR, margins
    assert margins[1 - at] < 1024 + 1024, margins


@pytest.mark.parametrize(
    "pin, devices, count, at",
    [
        ("Vulkan1", "Vulkan0,Vulkan1", 2, 1),
        ("vulkan0", "Vulkan0,Vulkan1", 2, 0),
        ("Vulkan2", "Vulkan0,Vulkan1", 2, None),
        ("Vulkan1", None, 2, None),
        ("CUDA1", None, 2, 1),
        ("CUDA0,CUDA1", "CUDA0,CUDA1", 2, None),
    ],
)
def test_a_draft_pin_resolves_to_its_target_card(pin, devices, count, at):
    assert lc._draft_pin_device_index(pin, devices, count) == at


@pytest.mark.parametrize(
    "args, want",
    [
        (["--split_mode=tensor"], "tensor"),
        (["--split_mode", "row"], "row"),
        (["-sm", "layer"], "layer"),
    ],
)
def test_the_underscore_split_mode_spelling_is_read(args, want):
    assert lc._extra_args_split_mode(args, {}) == want


# The fit-sized report: planning and spawn agree on where a reserve is provable.


@pytest.mark.parametrize(
    "extras, caps_fit_target, fit_sized",
    [
        ([], True, True),
        (["--rpc", "10.0.0.2:50052"], True, False),
        (_BOTH, True, False),
        ([], False, False),
    ],
)
def test_fit_sized_is_reported_only_where_the_spawn_can_price(
    tmp_path, extras, caps_fit_target, fit_sized
):
    backend, gguf = h.cards(tmp_path)
    caps = dict(h.CAPS, supports_fit_target = caps_fit_target)
    backend.probe_server_capabilities = lambda _binary = None: dict(caps)
    seen = h.spy_fit_sized(backend)
    h.dspark(backend, gguf, tmp_path, extra_args = extras)
    assert seen["fit_sized"] is fit_sized


def _past_one_card(tmp_path):
    backend, gguf = h.cards(tmp_path)
    backend._get_gguf_size_bytes = lambda path: (8 if "dspark" in str(path) else 30) * GIB
    return backend, gguf


def test_device_extras_a_gpu_ids_pin_strips_do_not_block_the_report(tmp_path):
    backend, gguf = _past_one_card(tmp_path)
    seen = h.spy_fit_sized(backend)
    cmd = h.dspark(backend, gguf, tmp_path, gpu_ids = [0, 1], extra_args = _BOTH)
    assert "--device" not in cmd
    assert len(h.margins(cmd)) == 2
    assert seen["fit_sized"] is True


@pytest.mark.parametrize("pin, reserved", [("CUDA0,CUDA1", False), ("CUDA0", True)])
def test_a_multi_device_draft_pin_is_not_reported_as_reserved(tmp_path, pin, reserved):
    backend, gguf = _past_one_card(tmp_path)
    seen = h.spy_fit_sized(backend)
    cmd = h.dspark(backend, gguf, tmp_path, extra_args = [*_BOTH, "--device-draft", pin])
    assert (len(h.margins(cmd)) == 2) is reserved
    assert seen["fit_sized"] is reserved


def test_an_unpinned_multi_card_vulkan_launch_gets_no_positional_reserve(tmp_path):
    """Vulkan orders its devices its own way, so neither planning nor spawn reserves."""
    backend, gguf = h.cards(tmp_path, vulkan = True)
    seen = h.spy_fit_sized(backend)
    cmd = h.dspark(backend, gguf, tmp_path, **MANUAL_AUTO)
    assert "--device" not in cmd
    assert seen["fit_sized"] is False
    assert len(h.margins(cmd)) == 1


@pytest.mark.parametrize(
    "extras, vulkan, reserved",
    [
        (["--device-draft", "CUDA2"], False, False),
        # A repeated target entry leaves the pin's position unknown.
        (["--device", "CUDA0,CUDA1,CUDA0", "--device-draft", "CUDA0"], False, False),
        # A Vulkan name with no target list to find it in.
        (["--device-draft", "Vulkan0"], True, False),
        (["--device-draft", "CUDA1"], False, True),
        ([*_BOTH, "--device-draft", "CUDA1"], False, True),
    ],
)
def test_planning_reports_only_a_draft_pin_the_spawn_resolves(tmp_path, extras, vulkan, reserved):
    backend, gguf = h.cards(tmp_path, vulkan = vulkan)
    seen = h.spy_fit_sized(backend)
    cmd = h.dspark(
        backend, gguf, tmp_path, extra_args = list(extras), **(MANUAL_AUTO if vulkan else {})
    )
    assert (len(h.margins(cmd)) == 2) is reserved
    assert seen["fit_sized"] is reserved


def test_planning_resolves_a_draft_pin_against_studios_vulkan_pin(tmp_path):
    """gpu_ids on Vulkan reaches the child as --device Vulkan0,Vulkan1 (added after planning)."""
    backend, gguf = h.cards(tmp_path, vulkan = True)
    seen = h.spy_fit_sized(backend)
    cmd = h.dspark(
        backend, gguf, tmp_path, gpu_ids = [0, 1], extra_args = ["--device-draft", "Vulkan1"]
    )
    assert len(h.margins(cmd)) == 2
    assert seen["fit_sized"] is True


@pytest.mark.parametrize(
    "memory, extras, caps_fit_target",
    [
        # --rpc devices come first in the child's list: no reserve on one local card.
        (h.ONE, ["--rpc", "10.0.0.2:50052"], True),
        (h.ONE, [], False),
        # --split-mode none keeps the target on card 0, the drafter on card 1.
        (TWO, ["--split-mode", "none", "--device-draft", "CUDA1"], True),
    ],
)
def test_a_reserve_the_spawn_drops_is_reported_as_missing(
    tmp_path, monkeypatch, memory, extras, caps_fit_target
):
    backend, gguf = h.cards(tmp_path, memory)
    caps = dict(h.CAPS, supports_fit_target = caps_fit_target)
    backend.probe_server_capabilities = lambda _binary = None: dict(caps)
    seen = h.spy_fit_sized(backend)
    messages = h.info_messages(monkeypatch)
    cmd = h.dspark(backend, gguf, tmp_path, extra_args = list(extras))
    assert h.margin(cmd) < 1024 + 4 * 1024
    assert seen["fit_sized"] is False
    assert not any("keeps" in m and "free for it" in m for m in messages)
    assert any("is not reserved" in m for m in messages)


def test_a_dflash_reserve_the_spawn_drops_is_reported_as_missing(tmp_path, monkeypatch):
    backend, gguf = h.cards(tmp_path, h.ONE)
    messages = h.info_messages(monkeypatch)
    cmd = h.placement._launch(
        backend,
        gguf,
        speculative_type = "dflash",
        dflash_draft_path = h.sidecar(tmp_path, "dflash"),
        extra_args = ["--rpc", "10.0.0.2:50052"],
    )["cmd"]
    assert "draft-dflash" in cmd
    assert any(m.startswith("DFlash") and "is not reserved" in m for m in messages)
    assert not any("keeps" in m and "free for it" in m for m in messages)


def test_a_renumbered_companion_pin_is_reported_as_reserved(tmp_path):
    """gpu_ids [1, 2] masks the child, so the request's CUDA2 arrives there as CUDA1."""
    backend, gguf = h.cards(tmp_path, (*TWO, (2, 24_000, 24_000)))
    seen = h.spy_fit_sized(backend)
    cmd = h.dspark(backend, gguf, tmp_path, gpu_ids = [1, 2], extra_args = ["--device-draft", "CUDA2"])
    margins = h.margins(cmd)
    assert len(margins) == 2 and max(margins) >= FLOOR
    assert seen["fit_sized"] is True


def test_the_start_log_shows_the_argv_spawned(tmp_path, monkeypatch):
    """The per-card reserve is spelled at spawn, so the logged command must be that one."""
    backend, gguf = h.cards(tmp_path)
    messages = h.info_messages(monkeypatch)
    spelled = h.last(h.dspark(backend, gguf, tmp_path), "--fit-target")
    assert "," in str(spelled)
    starts = [m for m in messages if m.startswith("Starting llama-server:")]
    assert starts and f"--fit-target {spelled}" in starts[-1]


def test_auto_drops_a_drafter_pinned_across_both_target_cards(tmp_path):
    backend, gguf, path = h.placement._tight_vram_backend(tmp_path, drafter_gb = 12.0)
    backend._get_gpu_memory = lambda _binary = None, **_kw: [(0, 12288, 12288), (1, 12288, 12288)]
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: [(0, 12288), (1, 12288)]
    backend._select_gpus = lambda *a, **k: ([0, 1], False)
    backend._select_gpus_split_aware = lambda *a, **k: ([0, 1], False)
    backend._arch_gate_survivors = lambda binary = None: []
    cmd = h.placement._launch_auto_8k(
        backend, gguf, dspark_draft_path = str(path), extra_args = ["--device-draft", "CUDA0,CUDA1"]
    )["cmd"]
    assert "--model-draft" not in cmd
    assert backend.spec_fallback_reason == "drafter_no_vram"


# Host-resident draft KV (-nkvo) leaves a card's reserve only on positive evidence that
# the drafter's card is discrete; host memory is an APU's / iGPU's VRAM.

_DGPU = ["CUDA0", "CUDA1"]
_VULKAN = ["Vulkan0", "Vulkan1"]


def test_a_discrete_card_picked_beside_an_apu_leaves_host_kv_out(tmp_path, monkeypatch):
    delta = h.host_kv_delta(tmp_path, monkeypatch, memory = TWO, listed = _DGPU, apu = 0, gpu_ids = [1])
    assert delta >= 760


@pytest.mark.parametrize("device, shared", [("Vulkan0", True), ("Vulkan1", False)])
def test_a_vulkan_igpu_picked_by_device_keeps_host_kv(tmp_path, monkeypatch, device, shared):
    """Vulkan iGPU 0 (reports total 0) plus dGPU 1."""
    delta = h.host_kv_delta(
        tmp_path,
        monkeypatch,
        memory = _VK_IGPU,
        vulkan = True,
        listed = _VULKAN,
        extras = ["--device", device],
        **MANUAL_AUTO,
    )
    assert delta == 0 if shared else delta >= 760


@pytest.mark.parametrize("device, discrete", [("CUDA1", True), ("ROCm1", True), ("CUDA0", False)])
def test_a_named_card_beside_an_apu_decides_host_kv(tmp_path, monkeypatch, device, discrete):
    delta = h.host_kv_delta(
        tmp_path,
        monkeypatch,
        memory = TWO,
        listed = [f"{device[:-1]}0", f"{device[:-1]}1"],
        apu = 0,
        extras = ["--device", device],
        **MANUAL_AUTO,
    )
    assert delta >= 760 if discrete else delta == 0


@pytest.mark.parametrize(
    "listed",
    [_VULKAN, None],
    ids = [
        # Vulkan Auto drops iGPU 0 before the shared-card set is built: unknown, not discrete.
        "test_a_vulkan_igpu_auto_filtered_out_keeps_host_kv",
        # No --list-devices answer: the planned discrete card says nothing about the child's.
        "test_an_override_the_build_does_not_list_keeps_host_kv",
    ],
)
def test_an_unproven_vulkan_device_keeps_host_kv(tmp_path, monkeypatch, listed):
    delta = h.host_kv_delta(
        tmp_path,
        monkeypatch,
        memory = _VK_IGPU,
        vulkan = True,
        listed = listed,
        apu = None,
        extras = ["--device", "Vulkan0"],
    )
    assert delta == 0


def test_a_one_card_draft_pin_on_the_discrete_card_leaves_host_kv_out(tmp_path, monkeypatch):
    delta = h.host_kv_delta(
        tmp_path, monkeypatch, memory = TWO, listed = _DGPU, apu = 0, extras = ["--device-draft", "CUDA1"]
    )
    assert delta >= 760


@pytest.mark.parametrize("devices, left_out", [("CUDA1,CUDA0", True), ("CUDA0,CUDA1", False)])
def test_split_mode_none_narrows_to_the_main_card(tmp_path, monkeypatch, devices, left_out):
    """--split-mode none keeps devices[main_gpu], 0 by default: the first listed card."""
    delta = h.host_kv_delta(
        tmp_path,
        monkeypatch,
        memory = TWO,
        listed = _DGPU,
        apu = 0,
        extras = ["--device", devices, "--split-mode", "none"],
    )
    assert delta >= 760 if left_out else delta == 0


@pytest.mark.parametrize(
    "order, left_out",
    [
        # CUDA's default order: the child's CUDA1 need not be nvidia-smi's card 1.
        ("FASTEST_FIRST", False),
        ("PCI_BUS_ID", True),
    ],
)
def test_a_mixed_host_narrows_only_through_a_proven_child_order(
    tmp_path, monkeypatch, _pci_ids, order, left_out
):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    delta = h.host_kv_delta(
        tmp_path, monkeypatch, memory = TWO, listed = _DGPU, apu = 0, extras = ["--device", "CUDA1"]
    )
    assert delta >= 760 if left_out else delta == 0


def test_a_uuid_mask_is_not_read_as_physical_ids(tmp_path, monkeypatch, _pci_ids):
    """CUDA_VISIBLE_DEVICES=GPU-<dgpu>,GPU-<apu>: the child's CUDA1 is the APU, which
    _physical_ordinals (numeric masks only) would leave as card 1, the discrete one."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-11111111,GPU-00000000")
    delta = h.host_kv_delta(
        tmp_path, monkeypatch, memory = TWO, listed = _DGPU, apu = 0, extras = ["--device", "CUDA1"]
    )
    assert delta == 0


def test_another_backends_device_name_is_not_read_as_cuda(tmp_path, monkeypatch, _pci_ids):
    """A CUDA + Vulkan build lists Vulkan1 too, but it is not CUDA card 1."""
    delta = h.host_kv_delta(
        tmp_path,
        monkeypatch,
        memory = TWO,
        listed = [*_DGPU, *_VULKAN],
        apu = 0,
        extras = ["--device", "Vulkan1"],
    )
    assert delta == 0


def test_an_all_discrete_host_needs_no_ordinal_mapping(tmp_path, monkeypatch, _pci_ids):
    """Control: no shared card anywhere, even under an order the child does not share."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    delta = h.host_kv_delta(
        tmp_path, monkeypatch, memory = TWO, listed = _DGPU, apu = None, extras = ["--device", "CUDA1"]
    )
    assert delta >= 760
