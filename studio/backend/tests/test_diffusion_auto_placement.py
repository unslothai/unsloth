# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automatic multi-GPU placement end to end on fake card topologies, CPU only.

One fake host feeds every reader the real decision uses: the nvidia-smi rows (``resolve_auto_cuda_ordinal`` ranks the
primary from them, ``smi_free_total_mib`` sizes the other cards without a CUDA context), the primary's own
``trusted_mem_get_info``, torch's device count / names / capabilities and an optional CUDA_VISIBLE_DEVICES mask. The
one-card plan comes from ``plan_diffusion_memory`` with the primary's real budget rule, and the multi-GPU refinement,
the layer split packer and the CFG-parallel gate run unpatched. Only the loaded component sizes, the block layout and
the layer split engine are fakes (no weights)."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Optional

import pytest
import torch

from core.inference import diffusion_device as dd
from core.inference import diffusion_layer_split_plan as lsp
from core.inference import diffusion_memory as dm
from core.inference import diffusion_multigpu as mg

GIB = 1024


def _replica_need_stub(mib):
    """Stands in for ``mg.replica_need_mib`` with its real signature: a call the real one rejects fails here too,
    and a keyword it gains (``activations=``, #19209) is accepted instead of raising inside the placement try."""
    signature = inspect.signature(mg.replica_need_mib)

    def stub(*args, **kwargs):
        signature.bind(*args, **kwargs)
        return int(mib)

    return stub


@dataclasses.dataclass
class Card:
    index: int  # physical (nvidia-smi) index
    total_mib: int
    used_mib: int = 0
    cap: tuple = (7, 5)
    name: str = "Tesla T4"

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib


T4 = 15360  # MiB nvidia-smi reports for a T4


class Host:
    def __init__(
        self,
        cards: list[Card],
        mask: Optional[list[int]] = None,
    ):
        self.cards = {c.index: c for c in cards}
        self.visible = list(mask) if mask is not None else sorted(self.cards)
        self.mem_get_info_calls: list[int] = []

    def card(self, ordinal: int) -> Card:
        return self.cards[self.visible[int(ordinal)]]

    def install(self, monkeypatch) -> None:
        from utils import hardware as hw_pkg
        from utils.hardware import hardware as hw
        from utils.hardware import nvidia

        rows = {
            "devices": [
                {
                    "index": c.index,
                    "vram_total_gb": c.total_mib / 1024,
                    "vram_used_gb": c.used_mib / 1024,
                }
                for c in self.cards.values()
            ]
        }
        monkeypatch.setattr(hw, "IS_ROCM", False)
        monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
        monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: list(self.visible))
        monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: True)
        monkeypatch.setattr(hw, "_integrated_cuda_rows", lambda visible: ({}, ""))
        monkeypatch.setattr(hw, "_get_parent_visible_gpu_spec", lambda: {})
        monkeypatch.setattr(hw, "_smi_visible_utilization", lambda spec: rows)
        monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: set())
        monkeypatch.setattr(
            nvidia,
            "get_compute_capabilities",
            lambda: {c.index: c.cap for c in self.cards.values()},
        )

        def mem_get_info(ordinal, *, module = None):
            self.mem_get_info_calls.append(int(ordinal))
            c = self.card(int(ordinal))
            return c.free_mib << 20, c.total_mib << 20

        monkeypatch.setattr(hw_pkg, "trusted_mem_get_info", mem_get_info)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: len(self.visible))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda o = 0: self.card(int(o)).name)
        monkeypatch.setattr(mg, "_capability", lambda o: tuple(self.card(int(o)).cap))
        monkeypatch.setattr(mg, "_rocm_torch", lambda: False)
        monkeypatch.setattr(lsp, "_is_rocm", lambda: False)
        monkeypatch.setattr(
            dd,
            "resolve_diffusion_device_target",
            lambda **kw: dd.DiffusionDeviceTarget(
                "cuda", torch.float16, "cuda", "nvidia", True, True, True
            ),
        )
        monkeypatch.setattr(dd, "_PINNED_AWAY_FROM_ZERO", False)


@dataclasses.dataclass
class _Target:
    supports_model_cpu_offload: bool = True
    ordinal: Optional[int] = 0
    device: str = "cuda"
    backend: str = "cuda"
    is_cuda_torch_device: bool = True


class _Pipe:
    components: dict = {}
    transformer = None

    def to(self, *a, **k):
        return self


class _Engine:
    def __init__(self):
        self.placed = []

    def place(
        self,
        pipe,
        *,
        primary,
        split,
        logger = None,
    ):
        self.placed.append(split)
        return {}

    def undo(self, pipe):
        pass


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for var in (mg.MULTIGPU_ENV, mg.CFG_PARALLEL_ENV, lsp.LAYER_SPLIT_ENV, "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(mg, "models_kept_loaded", lambda: False)
    # The fake cards below are sized against the Linux per-card reserve (1200 MiB). Windows keeps 300 MiB more,
    # so on a Windows runner the same T4 splits do not fit; that value is pinned in
    # test_diffusion_layer_split_plan.py::test_reserve_follows_comfyui_minimum_inference_memory.
    monkeypatch.setattr(lsp, "secondary_reserve_mib", lambda: 1200)
    eng = _Engine()
    monkeypatch.setattr(lsp, "engine_available", lambda: True)
    monkeypatch.setattr(lsp, "place_layer_split", eng.place)
    monkeypatch.setattr(lsp, "undo_layer_split", eng.undo)
    monkeypatch.setattr("core.inference.diffusion_bg_compile.load_time_enabled", lambda: False)
    yield eng


def _sizes(
    dit: int,
    te: int,
    vae: int = 160,
    dit2: int = 0,
) -> dict:
    out = {"transformer": (dit, "dit")}
    if dit2:
        out["transformer_2"] = (dit2, "dit")
    out["text_encoder"] = (te, "text_encoder")
    out["vae"] = (vae, "other")
    return out


@dataclasses.dataclass
class Decision:
    primary: Optional[int]
    selection: Optional[mg.MultiGpuSelection]
    one_card: dm.MemoryPlan
    plan: dm.MemoryPlan
    status: Optional[dict]
    cfg_parallel: Optional[str]
    pipe: object = None


def decide(
    monkeypatch,
    host: Host,
    sizes: dict,
    *,
    headroom: int = 2500,
    blocks: Optional[dict] = None,
    gpu_ids: Optional[list[int]] = None,
    preload_split: bool = True,
    place: bool = True,
    calibrated: Optional[dm.CalibratedImageActivation] = None,
    pipe: Optional[object] = None,
) -> Decision:
    """What an Automatic load (``gpu_ids`` None) or an explicit selection decides on ``host`` for ``sizes``."""
    host.install(monkeypatch)
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: sizes)
    if blocks is None:
        dits = {n: m for n, (m, r) in sizes.items() if r == "dit"}
        blocks = {n: lsp.DenoiserLayout(max(1, m // 40), (m // 40,) * 39) for n, m in dits.items()}
    monkeypatch.setattr(lsp, "_denoiser_layouts", lambda pipe: blocks)
    if gpu_ids is None:
        primary = dd.auto_cuda_ordinal_for_load()
        selection = mg.automatic_selection(primary)
    else:
        primary = min(gpu_ids)
        selection = mg.make_selection(primary, gpu_ids)
    ordinal = 0 if primary is None else primary
    card = host.card(ordinal)
    memory = dm.DeviceMemory(
        backend = "cuda",
        device = "cuda",
        memory_kind = "discrete_vram",
        free_mib = card.free_mib,
        total_mib = card.total_mib,
    )
    dits = sum(m for m, r in sizes.values() if r == "dit")
    companions = sum(m for m, r in sizes.values() if r != "dit")
    tes = sum(m for m, r in sizes.values() if r == "text_encoder")
    pipe = _Pipe() if pipe is None else pipe
    with mg.selection_scope(selection if preload_split else None):
        one_card = dm.plan_diffusion_memory(
            target = _Target(ordinal = primary),
            device_memory = memory,
            model_dense_mib = dits + companions,
            companion_dense_mib = companions,
            text_encoder_dense_mib = tes,
            runtime_headroom_mib = headroom,
            requested_mode = "auto",
            calibrated_activation = calibrated,
        )
    with mg.selection_scope(selection):
        plan = mg.refine_plan_for_multigpu(pipe, one_card)
        if not place:
            pass  # placement failed: nothing engaged, the one-card fallback runs
        elif plan.layer_split is not None:
            lsp.apply_layer_split_plan(pipe, plan, placement = f"cuda:{ordinal}")
        elif plan.secondary_ordinal is not None:
            # what place_split records: the component-to-device map under "placement"
            mg._PLACEMENTS[pipe] = {
                "primary": f"cuda:{ordinal}",
                "secondary": f"cuda:{plan.secondary_ordinal}",
                "placement": {
                    name: f"cuda:{plan.secondary_ordinal}"
                    if role == "text_encoder"
                    else f"cuda:{ordinal}"
                    for name, (_mib, role) in sizes.items()
                },
            }
        cfg = mg.install_cfg_parallel(pipe, offload_policy = plan.offload_policy)
    return Decision(primary, selection, one_card, plan, mg.placement_status(pipe), cfg, pipe)


def _placement(d: Decision) -> dict:
    assert d.status is not None and "decision" in d.status, d.status
    return d.status["decision"]


# A: fits one card -> one card, no CFG-parallel, no CUDA context on the other card.
def test_a_fits_one_card_uses_one_card_and_no_cfg_parallel(monkeypatch):
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 3900, te = 1600))
    assert d.plan is d.one_card and d.plan.offload_policy == dm.OFFLOAD_NONE
    assert d.plan.secondary_ordinal is None and d.plan.layer_split is None
    # Automatic now offers CFG-parallel on an idle card of the same model: declined here only because the fake
    # pipeline has no denoiser module (UNSLOTH_DIFFUSION_CFG_PARALLEL_AUTO=0 restores the automatic decline)
    assert d.cfg_parallel is not None and "no DiT denoiser" in d.cfg_parallel
    p = _placement(d)
    assert p["chosen"] == "one_card"
    assert p["summary"].startswith("placement: one card: everything resident on cuda:0")
    kinds = {r["kind"]: r["reason"] for r in p["rejected"]}
    assert kinds["companion_split"].startswith("not needed")
    # Automatic now borrows an idle card of the same model for CFG-parallel (every guided model that fits ran
    # 1.46-1.79x faster on 2x T4); the fake pipeline declines only because it has no denoiser module
    assert "no DiT denoiser" in kinds["cfg_parallel"]
    assert set(host.mem_get_info_calls) <= {0}


def test_a_explicit_selection_of_a_model_that_fits_still_offers_cfg_parallel(monkeypatch):
    # Explicit picks keep one-card; CFG-parallel is only skipped as the fake has no denoiser.
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 3900, te = 1600), gpu_ids = [0, 1])
    assert d.plan is d.one_card and not d.selection.automatic
    assert d.cfg_parallel is not None and "automatic" not in d.cfg_parallel


# B: denoiser fits card 0, text encoder does not fit beside it -> text encoder on card 1.
def test_b_companion_split_puts_the_text_encoder_on_the_second_card(monkeypatch):
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 7400, te = 7600))
    assert d.one_card.offload_policy != dm.OFFLOAD_NONE
    assert d.plan.offload_policy == dm.OFFLOAD_NONE and d.plan.secondary_ordinal == 1
    assert "text_encoder" in d.plan.secondary_components
    assert d.plan.layer_split is None
    # Automatic now borrows an idle card of the same model for CFG-parallel (every guided model that fits ran
    # 1.46-1.79x faster on 2x T4); the fake pipeline declines only because it has no denoiser module
    assert d.cfg_parallel is not None and "no DiT denoiser" in d.cfg_parallel
    p = _placement(d)
    assert p["chosen"] == "companion_split"
    assert "text_encoder" in p["summary"] and "on cuda:1" in p["summary"]
    assert "fits cuda:0" in p["summary"] and "does not fit beside it" in p["summary"]
    assert p["rejected"][0]["kind"] == "one_card"


# C: the denoiser fits no single card but fits two -> layer split.
def test_c_denoiser_too_big_for_one_card_is_layer_split(monkeypatch, _env):
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 14000, te = 7600))
    assert d.plan.layer_split is not None and d.plan.layer_split.cards == (0, 1)
    assert d.plan.secondary_ordinal is None
    assert _env.placed
    assert "layer split" in d.cfg_parallel
    p = _placement(d)
    assert p["chosen"] == "layer_split" and "cuda:0, cuda:1" in p["summary"]
    rejected = [r["kind"] for r in p["rejected"]]
    assert rejected[0] == "one_card" and "companion_split" in rejected


# D: nothing fits -> one-card offload, with the layer split's reason.
def test_d_nothing_fits_offloads_on_one_card(monkeypatch, _env):
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 32000, te = 7600))
    assert d.plan is d.one_card and d.plan.offload_policy != dm.OFFLOAD_NONE
    assert not _env.placed
    p = _placement(d)
    assert p["chosen"] == "offload"
    layer = [r for r in p["rejected"] if r["kind"] == "layer_split"]
    assert layer and "fit" in layer[0]["reason"]


# E: the primary is busy (a foreign process) -> the free card.
def test_e_busy_first_card_moves_the_load_to_the_free_card(monkeypatch):
    host = Host([Card(0, T4, 13000), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 3900, te = 1600))
    assert d.primary == 1 and d.selection.primary == 1
    assert d.plan.offload_policy == dm.OFFLOAD_NONE
    assert _placement(d)["summary"].startswith("placement: one card: everything resident on cuda:1")
    assert set(host.mem_get_info_calls) <= {1}


# F: mixed sizes.
def test_f_mixed_sizes_put_the_denoiser_on_the_big_card(monkeypatch):
    for order in ((24576, 12288), (12288, 24576)):
        host = Host([Card(0, order[0], 400, name = "RTX"), Card(1, order[1], 400, name = "RTX")])
        d = decide(monkeypatch, host, _sizes(dit = 15000, te = 5000))
        big = 0 if order[0] > order[1] else 1
        assert d.selection.primary == big
        assert d.plan.secondary_ordinal == 1 - big
        assert "text_encoder" in d.plan.secondary_components


def test_f_mixed_sizes_layer_split_cuts_in_proportion(monkeypatch, _env):
    host = Host([Card(0, 24576, 400, name = "RTX"), Card(1, 12288, 400, name = "RTX")])
    d = decide(monkeypatch, host, _sizes(dit = 19500, te = 1500))
    split = d.plan.layer_split
    assert split is not None and split.cards == (0, 1)
    blocks = {o: sum(r.stop - r.start for r in split.ranges if r.card == o) for o in split.cards}
    assert blocks[0] > blocks[1]


def test_f_older_second_card_takes_no_component(monkeypatch):
    host = Host([Card(0, 24576, 400, cap = (8, 9)), Card(1, 24576, 400, cap = (7, 5))])
    d = decide(monkeypatch, host, _sizes(dit = 15000, te = 9000))
    # the T4 stays in the selection; the planner keeps the (bf16, sm 8.0) text encoder off it
    assert d.selection is not None and d.selection.secondaries == (1,)
    assert d.plan.secondary_ordinal is None and d.plan.offload_policy != dm.OFFLOAD_NONE


# G: three cards.
def test_g_three_cards_companion_goes_to_the_freest_secondary(monkeypatch):
    host = Host([Card(0, T4, 300), Card(1, T4, 6000), Card(2, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 7400, te = 7600))
    assert d.plan.secondary_ordinal == 2


def test_g_three_cards_layer_split_uses_two_when_two_suffice(monkeypatch, _env):
    host = Host([Card(0, T4, 300), Card(1, T4, 300), Card(2, T4, 300)])
    two = decide(monkeypatch, host, _sizes(dit = 14000, te = 7600))
    assert two.plan.layer_split is not None and len(two.plan.layer_split.cards) == 2
    three = decide(monkeypatch, host, _sizes(dit = 22000, te = 7600))
    assert three.plan.layer_split is not None and three.plan.layer_split.cards == (0, 1, 2)


# H: a CUDA_VISIBLE_DEVICES mask: torch ordinals are positions in the mask, smi rows are physical.
def test_h_mask_maps_torch_ordinals_to_physical_rows(monkeypatch):
    cards = [Card(0, T4, 300), Card(1, T4, 300), Card(2, T4, 300), Card(3, T4, 13000)]
    host = Host(cards, mask = [3, 1])
    d = decide(monkeypatch, host, _sizes(dit = 3900, te = 1600))
    # physical 3 (ordinal 0) is busy: ordinal 1 = physical 1 wins, and only the two masked cards are candidates
    assert d.primary == 1 and d.selection.secondaries == (0,)
    assert dd.smi_free_total_mib(0) == (T4 - 13000, T4)
    split = decide(monkeypatch, Host(cards, mask = [1, 2]), _sizes(dit = 7400, te = 7600))
    assert split.selection.primary == 0 and split.plan.secondary_ordinal == 1


def test_one_visible_card_or_the_kill_switch_is_today_one_card_path(monkeypatch):
    host = Host([Card(0, T4, 300), Card(1, T4, 300)], mask = [1])
    d = decide(monkeypatch, host, _sizes(dit = 7400, te = 7600))
    assert d.selection is None and d.status is None
    assert d.plan is d.one_card
    monkeypatch.setenv(mg.MULTIGPU_ENV, "0")
    off = decide(monkeypatch, Host([Card(0, T4, 300), Card(1, T4, 300)]), _sizes(dit = 7400, te = 7600))
    assert off.selection is None and off.status is None and off.plan is off.one_card
    assert off.plan == d.plan


# Video families share the refinement: a two-expert denoiser and a large video DiT.
def test_video_two_experts_that_do_not_fit_one_card_share_two(monkeypatch):
    # Wan 2.2 A14B on 2x A100 80 GB: the experts stay together on the primary, every companion moves.
    a100 = dict(total_mib = 81920, used_mib = 1000, cap = (8, 0), name = "A100")
    host = Host([Card(0, **a100), Card(1, **a100)])
    d = decide(monkeypatch, host, _sizes(dit = 28600, dit2 = 28600, te = 11400), headroom = 6000)
    assert d.plan.offload_policy == dm.OFFLOAD_NONE and d.plan.secondary_ordinal == 1
    assert _placement(d)["chosen"] == "companion_split"


def test_video_second_expert_moves_when_the_companions_are_not_enough(monkeypatch):
    l40 = dict(total_mib = 46068, used_mib = 500, cap = (8, 9), name = "L40S")
    host = Host([Card(0, **l40), Card(1, **l40)])
    d = decide(monkeypatch, host, _sizes(dit = 19000, dit2 = 19000, te = 3000), headroom = 4000)
    assert d.plan.secondary_ordinal == 1
    assert "transformer_2" in (d.plan.secondary_components or ())
    assert _placement(d)["chosen"] == "companion_split"


def test_video_wan5b_on_two_t4s(monkeypatch):
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 9500, te = 10800, vae = 1400), headroom = 3500)
    p = _placement(d)
    assert p["chosen"] in ("companion_split", "layer_split")
    assert d.plan.offload_policy == dm.OFFLOAD_NONE


# Kept on purpose: SDXL's U-Net is tagged 'other', so refinement keeps the one-card plan; counting
# it as denoiser makes the pre-compile split preview predict a split the whole-compiled U-Net
# refuses, and the compile is set up for a resident denoiser under model offload (Kaggle 2x T4).
def test_unet_family_keeps_todays_refinement(monkeypatch):
    card = dict(total_mib = 12288, used_mib = 300, cap = (8, 6), name = "RTX 3060")
    host = Host([Card(0, **card), Card(1, **card)])
    sdxl = {
        "unet": (4900, "other"),
        "text_encoder": (235, "text_encoder"),
        "text_encoder_2": (1324, "text_encoder"),
        "vae": (160, "other"),
    }
    d = decide(monkeypatch, host, sdxl, headroom = 1200, preload_split = False)
    assert d.plan is d.one_card
    assert d.plan.secondary_ordinal is None and d.plan.layer_split is None


def test_video_umt5_text_encoder_moves_to_a_second_t4(monkeypatch):
    # Wan 2.1 1.3B: the 10.6 GiB umt5-xxl encoder fits a free 15 GB T4 once that card is priced for an encoder
    # forward (weights + 1 GiB + ComfyUI's 1.2 GiB floor) instead of a denoising card's overhead and margin.
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 2700, te = 10834, vae = 485), headroom = 3500)
    assert d.one_card.offload_policy != dm.OFFLOAD_NONE
    assert d.plan.secondary_ordinal == 1 and d.plan.offload_policy == dm.OFFLOAD_NONE
    assert set(d.plan.secondary_components) == {"text_encoder"}
    assert _placement(d)["chosen"] == "companion_split"


def test_encoder_card_pricing_never_admits_an_encoder_that_does_not_fit_the_free_memory(
    monkeypatch,
):
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 2700, te = 14000, vae = 485), headroom = 3500)
    assert d.plan.secondary_ordinal is None


# The discrete reserve: a partly used big card.
@pytest.mark.parametrize("total", [15360, 24576, 40960, 81920, 182272])
def test_discrete_reserve_is_fixed(total):
    assert dm._reserve_mib("discrete_vram", total) == 2048


def test_partly_used_big_cards_get_the_companion_split(monkeypatch):
    # 2x B200 (178 GiB) with 22 / 17 GiB free: the uncapped 10% reserve (17.8 GiB) left 4 GiB on the primary and
    # nothing on the secondary, so this streamed the denoiser on one card.
    b200 = 182272
    host = Host(
        [
            Card(0, b200, b200 - 22 * GIB, cap = (10, 0), name = "B200"),
            Card(1, b200, b200 - 17 * GIB, cap = (10, 0), name = "B200"),
        ]
    )
    d = decide(monkeypatch, host, _sizes(dit = 7400, te = 7600), headroom = 3000)
    assert d.plan.secondary_ordinal == 1 and d.plan.offload_policy == dm.OFFLOAD_NONE


def test_partly_used_big_card_places_the_companion_split_with_the_fixed_reserve(monkeypatch):
    # 2x B200, 20 / 12 GiB free: the old 4 GiB reserve offloaded; 2 GiB puts the encoder on cuda:1.
    b200 = 182272
    host = Host(
        [
            Card(0, b200, b200 - 20 * GIB, cap = (10, 0), name = "B200"),
            Card(1, b200, b200 - 12 * GIB, cap = (10, 0), name = "B200"),
        ]
    )
    d = decide(monkeypatch, host, _sizes(dit = 12000, te = 9000), headroom = 3000)
    assert d.plan.secondary_ordinal == 1 and d.plan.offload_policy == dm.OFFLOAD_NONE
    assert _placement(d)["chosen"] == "companion_split"


def test_the_apu_pool_keeps_the_uncapped_reserve(monkeypatch):
    memory = dm.DeviceMemory("cuda", "cuda:0", "unified_memory", 31 * GIB, 96 * GIB)
    monkeypatch.setattr(dm, "_rocm_linux_apu_os_room_outside_pool", lambda m: True)
    assert dm._safe_device_budget_mib(memory) == 31 * GIB - int(96 * GIB * 0.10)


# Decision must not overwrite the component map; a fitting forced split reports one card.
def test_decision_keeps_the_companion_component_map(monkeypatch):
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 7400, te = 7600))
    assert d.plan.secondary_ordinal == 1
    assert d.status["placement"]["text_encoder"] == "cuda:1"
    assert d.status["placement"]["transformer"] == "cuda:0"
    assert d.status["decision"]["chosen"] == "companion_split"


def test_video_card_floors_still_read_the_component_map(monkeypatch):
    from core.inference import video as vid

    class _Module(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(n))

    pipe = _Pipe()
    pipe.components = {"transformer": _Module(4), "text_encoder": _Module(8), "vae": _Module(2)}
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    host.install(monkeypatch)
    sel = mg.make_selection(0, [0, 1], automatic = True)
    mg._PLACEMENTS[pipe] = {
        "primary": "cuda:0",
        "secondary": "cuda:1",
        "placement": {"transformer": "cuda:0", "text_encoder": "cuda:1", "vae": "cuda:0"},
    }
    mg._record_decision(pipe, sel, "companion_split", "companion split: text_encoder on cuda:1")
    floors = vid._video_split_card_floors(pipe)
    assert floors is not None
    assert {int(card[0]) for card in floors} == {0, 1}


def test_forced_layer_split_of_a_resident_model_that_does_not_cut_reports_one_card(monkeypatch):
    monkeypatch.setenv(lsp.LAYER_SPLIT_ENV, "force")
    monkeypatch.setattr(lsp, "engine_available", lambda: False)
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 3900, te = 1600))
    assert d.plan is d.one_card and d.plan.offload_policy == dm.OFFLOAD_NONE
    p = _placement(d)
    assert p["chosen"] == "one_card" and "everything resident" in p["summary"]


def test_failed_forced_companion_split_of_a_resident_model_reports_one_card(monkeypatch):
    monkeypatch.setenv(mg.MULTIGPU_ENV, "force")
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 3900, te = 1600), preload_split = False, place = False)
    assert d.one_card.offload_policy == dm.OFFLOAD_NONE
    assert d.plan.secondary_ordinal == 1
    p = _placement(d)
    assert p["chosen"] == "one_card", p
    assert p["rejected"][0]["kind"] == "companion_split"


def test_failed_companion_split_of_an_offloading_model_reports_offload(monkeypatch):
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 7400, te = 7600), place = False)
    assert d.plan.secondary_ordinal == 1
    assert _placement(d)["chosen"] == "offload"


def test_layer_split_decline_reports_the_encode_activation_it_charged():
    layout = {"transformer": lsp.DenoiserLayout(500, (900,) * 20)}
    split, reason = lsp.plan_layer_split(
        denoisers = layout,
        companions = {"text_encoder": (7000, "text_encoder"), "vae": (160, "other")},
        primary = 0,
        card_limits = {0: 9000, 1: 9000},
        headroom_mib = 2000,
        overhead_mib = 2048,
        phase_headroom_mib = (5555, 1000, 1200),
    )
    assert split is None
    assert "5555 MiB of activations" in reason


# Z-Image-Turbo fp16 on two cards: a split priced with the flat estimate declines when each card is held to 85% of
# its safe budget; with the primary at the one-card budget it fits, every card within its limit.
_ZIMAGE_T4 = dict(dit = 11739, te = 7672, vae = 211)
_ZIMAGE_FLAT = 6963


def test_zimage_on_two_t4s_flat_estimate_fits_the_layer_split_at_the_one_card_budget(monkeypatch):
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(**_ZIMAGE_T4), headroom = _ZIMAGE_FLAT, place = False)
    split = d.plan.layer_split
    assert split is not None and split.cards == (0, 1)
    budget = int(d.one_card.estimates["safe_device_budget_mib"])
    for ordinal, required in split.card_required_mib:
        assert required <= dict(split.card_limit_mib)[ordinal] <= budget
    # the primary still carries the flat activation figure beside its weights
    assert dict(split.card_required_mib)[0] >= dict(split.card_weights_mib)[0] + _ZIMAGE_FLAT


def test_zimage_on_two_t4s_layer_split_fits_with_the_measured_activations(monkeypatch):
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    act = dm.calibrated_image_activation("z-image", max_speed = False)
    assert act is not None and act.headroom(False) < _ZIMAGE_FLAT
    d = decide(monkeypatch, host, _sizes(**_ZIMAGE_T4), headroom = _ZIMAGE_FLAT, calibrated = act)
    assert d.one_card.estimates["calibrated_split_headroom_mib"] == act.headroom(False)
    assert d.plan.layer_split is not None and d.plan.layer_split.cards == (0, 1)
    assert _placement(d)["chosen"] == "layer_split"
    for ordinal, required in d.plan.layer_split.card_required_mib:
        assert required <= dict(d.plan.layer_split.card_limit_mib)[ordinal]


def test_image_activation_estimate_scales_with_resolution():
    hint = "z-image Tongyi-MAI/Z-Image-Turbo"
    at_1024 = dm.estimate_image_runtime_mib(width = 1024, height = 1024, family = hint)
    at_768 = dm.estimate_image_runtime_mib(width = 768, height = 768, family = hint)
    assert (
        at_1024
        == _ZIMAGE_FLAT
        == dm.estimate_image_runtime_mib(width = None, height = None, family = hint)
    )
    assert at_768 == int(8192 * (768 * 768) / (1024 * 1024) * 0.85) < at_1024


def test_a_split_priced_with_the_measured_row_is_a_calibrated_placement(monkeypatch):
    # The generate guard must treat the split as calibrated (it never reserved the flat figure).
    from core.inference.diffusion import plan_is_calibrated

    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    act = dm.calibrated_image_activation("z-image", max_speed = False)
    split = decide(monkeypatch, host, _sizes(**_ZIMAGE_T4), headroom = _ZIMAGE_FLAT, calibrated = act)
    assert split.plan.layer_split is not None and plan_is_calibrated(split.plan)
    # the one-card fallback that carries the key but was decided from the flat estimate stays uncalibrated
    assert "calibrated_split_headroom_mib" in split.one_card.estimates
    assert not plan_is_calibrated(split.one_card)
    flat = decide(monkeypatch, host, _sizes(**_ZIMAGE_T4), headroom = _ZIMAGE_FLAT)
    # a split priced with the flat figure is not a calibrated placement
    assert not plan_is_calibrated(flat.plan)


# Wan2.2-TI2V-5B fp16 with a forced two-card split: an 85% primary cap left no room for a block
# beside stems, VAE, overhead and headroom; a one-card resident plan is priced at full budget.
_WAN5B_T4 = dict(dit = 9582, te = 10836, vae = 1631)
_WAN5B_BLOCKS = {"transformer": lsp.DenoiserLayout(192, (313,) * 30)}
_WAN5B_HEADROOM = 7839  # estimate_video_runtime_mib at the family default 1280x704, 121 frames
_WAN5B_USED = (554, 732)  # nvidia-smi used on each card once torch holds its context


def _wan5b_forced_split(monkeypatch):
    monkeypatch.setenv(lsp.LAYER_SPLIT_ENV, "force")
    host = Host([Card(0, T4, _WAN5B_USED[0]), Card(1, T4, _WAN5B_USED[1])])
    return decide(
        monkeypatch,
        host,
        _sizes(**_WAN5B_T4),
        headroom = _WAN5B_HEADROOM,
        blocks = _WAN5B_BLOCKS,
        gpu_ids = [0, 1],
        place = False,
    )


def test_wan5b_forced_split_on_two_t4s_prices_the_primary_with_the_one_card_budget(monkeypatch):
    d = _wan5b_forced_split(monkeypatch)
    assert d.one_card.offload_policy != dm.OFFLOAD_NONE
    split = d.plan.layer_split
    assert split is not None and split.cards == (0, 1), d.status["layer_split"]
    budget = int(d.one_card.estimates["safe_device_budget_mib"])
    limits = dict(split.card_limit_mib)
    required = dict(split.card_required_mib)
    # the primary is held to the budget a one-card resident plan gets there, never more
    assert limits[0] == budget and required[0] <= budget
    # the secondary keeps 85% of its safe budget
    assert limits[1] == mg.secondary_resident_limit_mib(1, "auto") and required[1] <= limits[1]
    # expected cut: 3 blocks beside the stems and the VAE, 27 on cuda:1
    assert [(r.start, r.stop, r.card) for r in split.ranges] == [(0, 3, 0), (3, 30, 1)]
    weights = dict(split.card_weights_mib)
    assert required[0] >= weights[0] + _WAN5B_HEADROOM


def test_layer_split_primary_never_exceeds_the_one_card_budget(monkeypatch):
    # a primary already carrying a model too large for its budget is still refused: nothing above the budget is lent
    monkeypatch.setenv(lsp.LAYER_SPLIT_ENV, "force")
    host = Host([Card(0, T4, _WAN5B_USED[0]), Card(1, T4, _WAN5B_USED[1])])
    d = decide(
        monkeypatch,
        host,
        _sizes(dit = 9582, te = 10836, vae = 6000),
        headroom = _WAN5B_HEADROOM,
        blocks = _WAN5B_BLOCKS,
        gpu_ids = [0, 1],
        place = False,
    )
    assert d.plan.layer_split is None


class WanTransformer3DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class FluxTransformer2DModel(WanTransformer3DModel):
    pass


class HiDreamImageTransformer2DModel(WanTransformer3DModel):
    pass


class _DitPipe(_Pipe):
    def __init__(self, denoiser):
        self.transformer = denoiser


def _priced(
    monkeypatch,
    *,
    need_mib,
    compute_s,
    tokens,
    h2d = 11.0,
):
    monkeypatch.setattr(mg, "replica_need_mib", _replica_need_stub(need_mib))
    monkeypatch.setattr(dm, "_offload_cost_measurements", lambda: (float(h2d), 30.0))
    monkeypatch.setattr(mg, "_gemm_tflops", lambda o, d: 25.0)
    priced = []

    def compute(params, n, tflops):
        priced.append((n, tflops))
        return float(compute_s) * n / tokens

    monkeypatch.setattr(mg, "denoiser_compute_s", compute)
    return priced


def _wan5b(
    monkeypatch,
    *,
    gpu_ids = (0, 1),
    guided = True,
    cls = WanTransformer3DModel,
    need = 9582 + 0,
    tokens = 5070,
    text_tokens = None,
    names = ("Tesla T4", "Tesla T4"),
):
    _priced(monkeypatch, need_mib = need, compute_s = 2.65, tokens = 5070)
    host = Host(
        [
            Card(0, T4, _WAN5B_USED[0], name = names[0]),
            Card(1, T4, _WAN5B_USED[1], name = names[1]),
        ]
    )
    pipe = _DitPipe(cls())
    sized = {} if text_tokens is None else {"text_tokens": text_tokens}
    mg.note_default_request(
        pipe, guided = guided, steps = 10, tokens = tokens, why = "default guidance 5", **sized
    )
    return decide(
        monkeypatch,
        host,
        _sizes(**_WAN5B_T4),
        headroom = _WAN5B_HEADROOM,
        blocks = _WAN5B_BLOCKS,
        gpu_ids = None if gpu_ids is None else list(gpu_ids),
        place = False,
        pipe = pipe,
    )


@pytest.mark.parametrize("gpu_ids", [(0, 1), None], ids = ["explicit", "automatic"])
def test_wan5b_on_two_t4s_runs_one_card_plus_a_cfg_replica_instead_of_the_layer_split(
    monkeypatch, gpu_ids
):
    d = _wan5b(monkeypatch, gpu_ids = gpu_ids)
    assert d.plan is d.one_card and d.plan.offload_policy != dm.OFFLOAD_NONE
    assert d.plan.layer_split is None and d.plan.secondary_ordinal is None
    recorded = _recorded(d)
    assert recorded is not None and recorded.chosen == "cfg_replica", recorded
    assert "CFG replica on cuda:1" in recorded.summary and "s/step" in recorded.summary
    kinds = {c.kind: c.reason for c in recorded.rejected}
    assert "predicted faster" in kinds["layer_split"]


def _recorded(d: Decision) -> mg.PlacementDecision:
    return mg.placement_decision(d.pipe)


def test_wan5b_kill_switch_keeps_the_layer_split(monkeypatch):
    monkeypatch.setenv(mg.CFG_REPLICA_PLACEMENT_ENV, "0")
    d = _wan5b(monkeypatch)
    assert d.plan.layer_split is not None and d.plan.layer_split.cards == (0, 1)
    assert _recorded(d).chosen == "layer_split"


@pytest.mark.parametrize("gpu_ids", [(0, 1), None], ids = ["explicit", "automatic"])
def test_forced_multigpu_split_is_not_replaced_by_the_replica(monkeypatch, gpu_ids):
    monkeypatch.setenv(mg.MULTIGPU_ENV, "force")
    _priced(monkeypatch, need_mib = 3900, compute_s = 1.0, tokens = 4096)
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    pipe = _DitPipe(WanTransformer3DModel())
    mg.note_default_request(pipe, guided = True, steps = 20)
    ids = None if gpu_ids is None else list(gpu_ids)
    d = decide(
        monkeypatch,
        host,
        _sizes(dit = 3900, te = 1600),
        pipe = pipe,
        gpu_ids = ids,
        preload_split = False,
        place = False,
    )
    assert d.plan.secondary_ordinal == 1
    assert _recorded(d).chosen == "companion_split"
    w = _wan5b(monkeypatch, gpu_ids = gpu_ids)
    assert w.plan.layer_split is not None and w.plan.layer_split.cards == (0, 1)
    assert _recorded(w).chosen == "layer_split"


def test_wan5b_status_reports_the_replica_choice_and_its_fallback(monkeypatch):
    d = _wan5b(monkeypatch)
    p = _placement(d)
    assert p["chosen"] == "offload", p
    first = p["rejected"][0]
    assert first["kind"] == "cfg_replica" and first["reason"].startswith(
        "the CFG replica fell back to one card"
    )
    assert "not on a CUDA card" in first["reason"]
    est = d.status["mode_estimate"]
    assert est["chosen"] == "cfg_replica" and est["replica_card"] == "cuda:1"
    assert est["replica_render_s"] < est["alternative_render_s"]


def test_unguided_default_request_keeps_the_layer_split(monkeypatch):
    d = _wan5b(monkeypatch, guided = False, cls = FluxTransformer2DModel)
    assert d.plan.layer_split is not None
    assert _recorded(d).chosen == "layer_split"
    reasons = {c.kind: c.reason for c in _recorded(d).rejected}
    assert "one denoiser branch per step" in reasons["cfg_replica"]


class Flux2Transformer2DModel(WanTransformer3DModel):
    pass


def _flux2_plan(monkeypatch):
    # stands in for the check of diffusers' context-parallel plan (importing diffusers here would touch CUDA)
    from core.inference import diffusion_sequence_parallel as sp
    monkeypatch.setattr(sp, "read_plan", lambda cls: sp.SP_EXEC_PLANS.get(cls.__name__))


def test_unguided_flux2_default_request_takes_the_replica_to_split_its_tokens(monkeypatch):
    # a distilled FLUX.2 klein render has one branch per step: the replica runs half of each step's tokens
    _flux2_plan(monkeypatch)
    d = _wan5b(monkeypatch, guided = False, cls = Flux2Transformer2DModel, **_KLEIN_1024)
    assert d.plan.layer_split is None and d.plan.secondary_ordinal is None
    recorded = _recorded(d)
    assert recorded.chosen == "cfg_replica", recorded
    assert "half of each step's tokens" in recorded.summary
    kinds = {c.kind: c.reason for c in recorded.rejected}
    assert "half of each step's tokens on each card" in kinds["layer_split"]
    est = d.status["mode_estimate"]
    assert est["replica_render_s"] < est["alternative_render_s"]


def test_unguided_flux2_keeps_the_layer_split_with_sequence_parallel_off(monkeypatch):
    from core.inference import diffusion_sequence_parallel as sp

    _flux2_plan(monkeypatch)
    monkeypatch.setenv(sp.SP_ENV, "0")
    d = _wan5b(monkeypatch, guided = False, cls = Flux2Transformer2DModel, **_KLEIN_1024)
    assert d.plan.layer_split is not None
    reasons = {c.kind: c.reason for c in _recorded(d).rejected}
    assert (
        "one denoiser branch per step" in reasons["cfg_replica"]
        and sp.SP_ENV in reasons["cfg_replica"]
    )


def test_unguided_flux2_with_compiled_blocks_keeps_the_layer_split(monkeypatch):
    # speed default / max compile the blocks before placement; a compiled call never splits by tokens, so a replica
    # placed for the split would only cost a second copy of the model
    _flux2_plan(monkeypatch)

    def compiled_init(self):
        Flux2Transformer2DModel.__init__(self)
        self.proj._compiled_call_impl = object()

    compiled = type(
        "Flux2Transformer2DModel", (Flux2Transformer2DModel,), {"__init__": compiled_init}
    )
    d = _wan5b(monkeypatch, guided = False, cls = compiled, **_KLEIN_1024)
    assert d.plan.layer_split is not None
    reasons = {c.kind: c.reason for c in _recorded(d).rejected}
    assert "compiled submodules" in reasons["cfg_replica"], reasons


def test_unguided_flux2_with_per_call_activation_scaling_keeps_the_layer_split(monkeypatch):
    # dynamic NVFP4 linears scale by each call's amax, so the router refuses every token split of such a denoiser
    _flux2_plan(monkeypatch)

    class ComfyNVFP4DynamicLinear(torch.nn.Linear):
        pass

    def scaled_init(self):
        Flux2Transformer2DModel.__init__(self)
        self.proj = ComfyNVFP4DynamicLinear(4, 4)

    scaled = type("Flux2Transformer2DModel", (Flux2Transformer2DModel,), {"__init__": scaled_init})
    d = _wan5b(monkeypatch, guided = False, cls = scaled, **_KLEIN_1024)
    assert d.plan.layer_split is not None
    reasons = {c.kind: c.reason for c in _recorded(d).rejected}
    assert "per-call activation scaling" in reasons["cfg_replica"], reasons


# a default request whose size clears MIN_TOKENS, priced like the Wan 5B fixture (5070 tokens) plus a 512-token prompt
_KLEIN_1024 = {"tokens": 5070, "text_tokens": 512}


@pytest.mark.parametrize(
    "sized, why",
    [
        (
            {"tokens": 2304, "text_tokens": 512},
            "2816 tokens, under 4096",
        ),  # a 768 px default request
        ({"tokens": None, "text_tokens": None}, "token count is unknown"),
        ({"tokens": 4096, "text_tokens": None}, "token count is unknown"),
    ],
    ids = ["768px", "unsized", "no_text_length"],
)
def test_unguided_flux2_under_the_threshold_or_unsized_keeps_the_layer_split(
    monkeypatch, sized, why
):
    # the router never splits such calls: giving the layer split up for a replica would buy nothing
    _flux2_plan(monkeypatch)
    d = _wan5b(monkeypatch, guided = False, cls = Flux2Transformer2DModel, **sized)
    assert d.plan.layer_split is not None
    reasons = {c.kind: c.reason for c in _recorded(d).rejected}
    assert why in reasons["cfg_replica"], reasons


def test_unguided_flux2_on_two_different_cards_keeps_the_layer_split(monkeypatch):
    # CFG-parallel force takes any card for a replica; the router still refuses a token split across two card models
    # (it keeps the CFG force only for guided branches), so placement does too
    _flux2_plan(monkeypatch)
    monkeypatch.setattr(mg, "_NAMES", {})  # card names are cached per process
    monkeypatch.setenv(mg.CFG_PARALLEL_ENV, "force")
    d = _wan5b(
        monkeypatch,
        guided = False,
        cls = Flux2Transformer2DModel,
        names = ("NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 5080"),
        **_KLEIN_1024,
    )
    assert d.plan.layer_split is not None
    reasons = {c.kind: c.reason for c in _recorded(d).rejected}
    assert "different cards" in reasons["cfg_replica"], reasons


def test_unguided_wan_default_request_keeps_main_placement(monkeypatch):
    # the unguided token-split replica is FLUX.2 only: an unguided Wan default request places as on main
    from core.inference import diffusion_sequence_parallel as sp

    monkeypatch.setattr(sp, "read_plan", lambda cls: sp.SP_EXEC_PLANS.get(cls.__name__))
    d = _wan5b(monkeypatch, guided = False)
    assert d.plan.layer_split is not None
    reasons = {c.kind: c.reason for c in _recorded(d).rejected}
    assert "one denoiser branch per step" in reasons["cfg_replica"], reasons


def test_sequence_split_pricing_takes_a_share_of_the_step():
    replica, alternative = mg.price_replica_layout(
        guided = False, compute_s = 1.0, primary_stream_s = 0.0, steps = 4, sequence_split = True
    )
    assert alternative == 4.0 and abs(replica - 4 * mg.SEQUENCE_SPLIT_STEP_SHARE) < 1e-9


def test_replica_that_does_not_fit_the_second_card_keeps_the_layer_split(monkeypatch):
    d = _wan5b(monkeypatch, need = 14500)
    assert d.plan.layer_split is not None
    reasons = {c.kind: c.reason for c in _recorded(d).rejected}
    assert "no card for a replica" in reasons["cfg_replica"]


def test_family_whose_branches_cannot_overlap_keeps_the_layer_split(monkeypatch):
    d = _wan5b(monkeypatch, cls = HiDreamImageTransformer2DModel)
    assert d.plan.layer_split is not None
    reasons = {c.kind: c.reason for c in _recorded(d).rejected}
    assert "llama states" in reasons["cfg_replica"]


def test_heavy_stream_on_the_primary_keeps_the_layer_split(monkeypatch):
    d = _wan5b(monkeypatch)
    _priced(monkeypatch, need_mib = 9582, compute_s = 0.3, tokens = 5070, h2d = 2.0)
    host = Host([Card(0, T4, _WAN5B_USED[0]), Card(1, T4, _WAN5B_USED[1])])
    pipe = _DitPipe(WanTransformer3DModel())
    mg.note_default_request(pipe, guided = True, steps = 10, tokens = 5070)
    d = decide(
        monkeypatch,
        host,
        _sizes(**_WAN5B_T4),
        headroom = _WAN5B_HEADROOM,
        blocks = _WAN5B_BLOCKS,
        gpu_ids = [0, 1],
        place = False,
        pipe = pipe,
    )
    assert d.plan.layer_split is not None
    reasons = {c.kind: c.reason for c in _recorded(d).rejected}
    assert "predicted slower" in reasons["cfg_replica"]


class ZImageTransformer2DModel(WanTransformer3DModel):
    pass


def test_streamed_zimage_plan_keeps_the_split_its_replica_could_not_use(monkeypatch):
    d = _wan5b(monkeypatch, cls = ZImageTransformer2DModel)
    assert d.plan.layer_split is not None and d.plan.layer_split.cards == (0, 1)
    recorded = _recorded(d)
    assert recorded.chosen == "layer_split"
    reasons = {c.kind: c.reason for c in recorded.rejected}
    assert (
        "Z-Image splits guidance only while no group-offload block streams"
        in reasons["cfg_replica"]
    )


def test_unpriced_card_keeps_the_layer_split(monkeypatch):
    d = _wan5b(monkeypatch)
    monkeypatch.setattr(dm, "_offload_cost_measurements", lambda: None)
    pipe = _DitPipe(WanTransformer3DModel())
    mg.note_default_request(pipe, guided = True, steps = 10, tokens = 5070)
    host = Host([Card(0, T4, _WAN5B_USED[0]), Card(1, T4, _WAN5B_USED[1])])
    d = decide(
        monkeypatch,
        host,
        _sizes(**_WAN5B_T4),
        headroom = _WAN5B_HEADROOM,
        blocks = _WAN5B_BLOCKS,
        gpu_ids = [0, 1],
        place = False,
        pipe = pipe,
    )
    assert d.plan.layer_split is not None


def test_guided_model_that_fits_one_card_reports_the_cfg_replica(monkeypatch):
    _priced(monkeypatch, need_mib = 3900, compute_s = 1.0, tokens = 4096)
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    pipe = _DitPipe(WanTransformer3DModel())
    mg.note_default_request(pipe, guided = True, steps = 20)
    d = decide(monkeypatch, host, _sizes(dit = 3900, te = 1600), pipe = pipe)
    assert d.plan is d.one_card and d.plan.offload_policy == dm.OFFLOAD_NONE
    recorded = mg.placement_decision(pipe)
    assert recorded.chosen == "cfg_replica" and "everything resident on cuda:0" in recorded.summary
    assert _placement(d)["chosen"] == "one_card"


def test_guided_model_whose_text_encoder_would_move_runs_the_replica_instead(monkeypatch):
    _priced(monkeypatch, need_mib = 2700, compute_s = 1.5, tokens = 32760)
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    pipe = _DitPipe(WanTransformer3DModel())
    mg.note_default_request(pipe, guided = True, steps = 50, tokens = 32760)
    d = decide(monkeypatch, host, _sizes(dit = 2700, te = 10834, vae = 485), headroom = 3500, pipe = pipe)
    assert d.plan is d.one_card and d.plan.secondary_ordinal is None
    recorded = mg.placement_decision(pipe)
    assert recorded.chosen == "cfg_replica"
    assert "predicted faster" in {c.kind: c.reason for c in recorded.rejected}["companion_split"]
    pipe2 = _DitPipe(WanTransformer3DModel())
    mg.note_default_request(pipe2, guided = False, steps = 50)
    d2 = decide(monkeypatch, host, _sizes(dit = 2700, te = 10834, vae = 485), headroom = 3500, pipe = pipe2)
    assert d2.plan.secondary_ordinal == 1 and _placement(d2)["chosen"] == "companion_split"


# NVIDIA feedback (2x RTX PRO 4000 / 4500, Wan 2.2 TI2V 5B with CFG at 1.01-1.03x): a companion split planned before
# the load returned early, so a CFG replica was never priced; the encoder card counted as busy; unpriced kept the split.
_BIG = 24564  # a 24 GB card


def _planned_split(
    monkeypatch,
    *,
    dit = 9000,
    te = 10000,
    vae = 160,
    need = 9400,
    priced = True,
    guided = True,
    gpu_ids = None,
):
    _priced(monkeypatch, need_mib = need, compute_s = 1.0, tokens = 4096)
    if not priced:
        monkeypatch.setattr(dm, "_offload_cost_measurements", lambda: None)
    host = Host([Card(0, _BIG, 300, name = "RTX PRO 4000"), Card(1, _BIG, 300, name = "RTX PRO 4000")])
    pipe = _DitPipe(WanTransformer3DModel())
    mg.note_default_request(pipe, guided = guided, steps = 20, tokens = 4096, why = "default guidance 5")
    sizes = _sizes(dit = dit, te = te, vae = vae)
    return decide(monkeypatch, host, sizes, pipe = pipe, gpu_ids = gpu_ids)


@pytest.mark.parametrize("gpu_ids", [None, [0, 1]], ids = ["automatic", "explicit"])
def test_planned_companion_split_gives_way_to_a_faster_cfg_replica(monkeypatch, gpu_ids):
    d = _planned_split(monkeypatch, gpu_ids = gpu_ids)
    assert d.one_card.secondary_ordinal == 1  # the split was planned before the load
    assert d.plan is d.one_card.fallback_plan and d.plan.secondary_ordinal is None
    recorded = mg.placement_decision(d.pipe)
    assert recorded.chosen == "cfg_replica", recorded
    assert "CFG replica on cuda:1" in recorded.summary and "s/step" in recorded.summary
    assert "predicted faster" in {c.kind: c.reason for c in recorded.rejected}["companion_split"]
    placement = d.status["cfg_parallel"]["placement"]
    assert placement["reason"], placement


def test_planned_encoder_split_keeps_the_encoders_and_runs_a_replica_beside_them(monkeypatch):
    # text encoders + VAE do not fit cuda:1, the encoders alone do: the replica fits there beside them
    d = _planned_split(monkeypatch, dit = 6000, te = 12000, vae = 6000, need = 6400)
    assert d.plan is d.one_card and d.plan.secondary_ordinal == 1
    assert d.plan.secondary_components == mg.TEXT_ENCODERS_ONLY
    recorded = mg.placement_decision(d.pipe)
    assert recorded.chosen == "cfg_replica", recorded
    assert (
        "CFG replica on cuda:1" in recorded.summary
        and "the replica sits beside text_encoder on cuda:1" in recorded.summary
    )
    # the fake denoiser is on the CPU, so the replica declines at install: the placed split is what runs
    live = _placement(d)
    assert live["chosen"] == "companion_split", live
    assert live["rejected"][0]["kind"] == "cfg_replica"
    placement = d.status["cfg_parallel"]["placement"]
    assert placement["chosen"] == "companion_split" and "fell back" in placement["reason"]


def test_planned_split_without_pricing_takes_the_replica_for_a_guided_request(monkeypatch):
    d = _planned_split(monkeypatch, priced = False)
    recorded = mg.placement_decision(d.pipe)
    assert recorded.chosen == "cfg_replica", recorded
    assert "unpriced" in recorded.summary and "default request is guided" in recorded.summary
    assert d.status["mode_estimate"]["priced"] is False


def test_planned_split_is_kept_for_an_unguided_request(monkeypatch):
    d = _planned_split(monkeypatch, guided = False)
    assert d.plan is d.one_card and d.plan.secondary_ordinal == 1
    recorded = mg.placement_decision(d.pipe)
    assert recorded.chosen == "companion_split"
    assert (
        "one denoiser branch per step"
        in {c.kind: c.reason for c in recorded.rejected}["cfg_replica"]
    )
    assert d.status["cfg_parallel"]["placement"]["chosen"] == "companion_split"


def test_unpriced_guided_request_takes_the_replica_over_a_post_load_companion_split(monkeypatch):
    _priced(monkeypatch, need_mib = 7600, compute_s = 1.0, tokens = 4096)
    monkeypatch.setattr(dm, "_offload_cost_measurements", lambda: None)
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    pipe = _DitPipe(WanTransformer3DModel())
    mg.note_default_request(pipe, guided = True, steps = 20, tokens = 4096)
    d = decide(monkeypatch, host, _sizes(dit = 7400, te = 7600), pipe = pipe)
    assert d.plan is d.one_card and d.plan.secondary_ordinal is None
    recorded = mg.placement_decision(pipe)
    assert recorded.chosen == "cfg_replica" and "unpriced" in recorded.summary


def test_wan5b_on_two_32gb_cards_runs_the_replica_instead_of_the_planned_split(monkeypatch):
    # The reported 2x RTX PRO 4500 case at the recorded Wan 2.2 TI2V 5B sizes: the plan before the load moves the
    # companions to cuda:1, which on main returned before any replica was priced (both branches on cuda:0)
    _priced(monkeypatch, need_mib = 9558 + 3400, compute_s = 2.0, tokens = 27280)
    cards = [Card(o, 32607, 600, cap = (12, 0), name = "NVIDIA RTX PRO 4500") for o in (0, 1)]
    pipe = _DitPipe(WanTransformer3DModel())
    mg.note_default_request(pipe, guided = True, steps = 50, tokens = 27280, why = "default guidance 5")
    d = decide(
        monkeypatch,
        Host(cards),
        _sizes(**_WAN5B_T4),
        headroom = _WAN5B_HEADROOM,
        blocks = _WAN5B_BLOCKS,
        pipe = pipe,
    )
    assert d.one_card.secondary_ordinal == 1
    assert d.plan is d.one_card.fallback_plan and d.plan.secondary_ordinal is None
    recorded = mg.placement_decision(pipe)
    assert recorded.chosen == "cfg_replica", recorded
    assert "0.0 GiB streamed per step on cuda:0" in recorded.summary
    assert d.status["mode_estimate"]["alternative"] == "companion_split"


def test_planned_encoder_split_without_room_for_the_next_encode_runs_the_replica_instead(
    monkeypatch,
):
    # Codex P2 on #19671: beside the encoders only when their next encode also fits; else the one-card plan + replica
    monkeypatch.setattr(mg, "encoder_working_mib", lambda pipe: 16000)
    d = _planned_split(monkeypatch, dit = 6000, te = 12000, vae = 6000, need = 6400)
    assert (
        d.one_card.secondary_ordinal == 1
        and d.one_card.secondary_components == mg.TEXT_ENCODERS_ONLY
    )
    assert d.plan is d.one_card.fallback_plan and d.plan.secondary_ordinal is None
    recorded = mg.placement_decision(d.pipe)
    assert recorded.chosen == "cfg_replica", recorded
    reasons = [c.reason for c in recorded.rejected if c.kind == "cfg_replica"]
    assert any(r.startswith("beside the encoders on cuda:1") for r in reasons), recorded.rejected


def test_planned_encoder_split_records_its_encode_headroom(monkeypatch):
    d = _planned_split(monkeypatch, dit = 6000, te = 12000, vae = 6000, need = 6400)
    assert d.one_card.estimates["multigpu_secondary_headroom_mib"] == mg.TEXT_ENCODER_HEADROOM_MIB
    assert mg.encoder_working_mib(d.pipe) == mg.TEXT_ENCODER_HEADROOM_MIB
