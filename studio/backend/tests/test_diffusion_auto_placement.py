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
from typing import Optional

import pytest
import torch

from core.inference import diffusion_device as dd
from core.inference import diffusion_layer_split_plan as lsp
from core.inference import diffusion_memory as dm
from core.inference import diffusion_multigpu as mg

GIB = 1024


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
    pipe = _Pipe()
    with mg.selection_scope(selection if preload_split else None):
        one_card = dm.plan_diffusion_memory(
            target = _Target(ordinal = primary),
            device_memory = memory,
            model_dense_mib = dits + companions,
            companion_dense_mib = companions,
            text_encoder_dense_mib = tes,
            runtime_headroom_mib = headroom,
            requested_mode = "auto",
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
    return Decision(primary, selection, one_card, plan, mg.placement_status(pipe), cfg)


def _placement(d: Decision) -> dict:
    assert d.status is not None and "decision" in d.status, d.status
    return d.status["decision"]


# A: fits one card -> one card, no CFG-parallel, no CUDA context on the other card.
def test_a_fits_one_card_uses_one_card_and_no_cfg_parallel(monkeypatch):
    host = Host([Card(0, T4, 300), Card(1, T4, 300)])
    d = decide(monkeypatch, host, _sizes(dit = 3900, te = 1600))
    assert d.plan is d.one_card and d.plan.offload_policy == dm.OFFLOAD_NONE
    assert d.plan.secondary_ordinal is None and d.plan.layer_split is None
    assert d.cfg_parallel is not None and "automatic" in d.cfg_parallel
    p = _placement(d)
    assert p["chosen"] == "one_card"
    assert p["summary"].startswith("placement: one card: everything resident on cuda:0")
    kinds = {r["kind"]: r["reason"] for r in p["rejected"]}
    assert kinds["companion_split"].startswith("not needed")
    assert "automatic" in kinds["cfg_parallel"]
    assert set(host.mem_get_info_calls) <= {0}


def test_a_explicit_selection_of_a_model_that_fits_still_offers_cfg_parallel(monkeypatch):
    # The user picked both cards: the one-card plan stays, CFG-parallel is the user's lever (declined here only
    # because the fake pipeline has no denoiser module).
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
    assert d.cfg_parallel is not None and "automatic" in d.cfg_parallel
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
    assert d.selection is None
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


# A U-Net family (SDXL): the loaded-size reader tags ``unet`` 'other', so the post-load refinement finds no denoiser
# to keep on the primary and leaves the one-card plan. Kept that way on purpose: with the U-Net counted as the
# denoiser, the pre-compile layer split preview of an SDXL load predicts a split that the whole-compiled U-Net then
# refuses, and the compile is set up for a resident denoiser under model offload (Kaggle 2x T4 run).
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
    # 2x B200 with 20 / 12 GiB free, a 12 GiB denoiser and a 9 GiB encoder: the 4 GiB capped reserve left the primary
    # short and the load offloaded on one card; the fixed 2 GiB reserve puts the encoder on cuda:1.
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


# Review follow-ups: the decision must not overwrite the component map, and forced splits of a model one card holds
# must report their resident fallback as one card.
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
