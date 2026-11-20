# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A LoRA set attached at load keeps the base build's measured rows and adds its own forward; CPU only.

Regression: on two T4s a FLUX.2 klein LoRA load keyed ``model:unknown``, skipped the learned peak and lost the CFG
replica (63.7 vs 35 s/img)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

import core.inference.diffusion as diffusion  # noqa: E402
import core.inference.diffusion_layer_split_plan as lsp  # noqa: E402
import core.inference.diffusion_measured_peaks as mp  # noqa: E402
import core.inference.diffusion_memory as dm  # noqa: E402
import core.inference.diffusion_multigpu as mg  # noqa: E402

try:
    import core.inference.diffusion_lora_memory as lm  # noqa: E402
except ImportError:  # pragma: no cover - base tree
    lm = None

needs_lm = pytest.mark.skipif(lm is None, reason = "diffusion_lora_memory is not in this tree")
LORA_MEASURED_ENV = "UNSLOTH_DIFFUSION_LORA_MEASURED"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_diffusion_auto_placement import Card, Host, _Pipe  # noqa: E402
from test_diffusion_dense_tier_activation import ENVS, KLEIN_FLAT, _flat_plan  # noqa: E402

# FLUX.2 klein-base 4B, int8 DiT + dense Qwen3 encoder, as loaded on a Kaggle T4 (MiB)
F2KB_LOADED = {
    "transformer": (3890, "dit"),
    "text_encoder": (5940, "text_encoder"),
    "vae": (160, "other"),
}
WIDE = 27648  # Flux2 single block fused qkv + SwiGLU projection
HIDDEN = 3072
LABELS = dict(attention = "native", vae = "full", te = "int8", cache = "off", graphs = "off")
LEARNED_PEAK = 5500


def _denoiser(
    rank: int = 16,
    adapters: int = 1,
    dora: bool = False,
):
    from peft import LoraConfig, inject_adapter_in_model

    class Dit(torch.nn.Module):
        pass

    dit = Dit()
    with torch.device("meta"):
        dit.wide = torch.nn.Linear(HIDDEN, WIDE, bias = False, dtype = torch.float16)
        dit.narrow = torch.nn.Linear(HIDDEN, HIDDEN, bias = False, dtype = torch.float16)
        names = [f"style{i}" for i in range(adapters)]
        for name in names:
            config = LoraConfig(
                r = rank, lora_alpha = rank, target_modules = ["wide", "narrow"], use_dora = dora
            )
            inject_adapter_in_model(config, dit, adapter_name = name)
    if adapters:
        for layer in (dit.wide, dit.narrow):
            layer.set_adapter(names)
            for name in names:
                layer.lora_A[name].half()
                layer.lora_B[name].half()
    return dit, tuple((n, f"/loras/{n}.safetensors", 1.0) for n in names)


def _pipe(adapters: int = 1, **kw):
    pipe = _Pipe()
    if adapters:
        pipe.transformer, pipe._unsloth_loras = _denoiser(adapters = adapters, **kw)
    else:
        pipe.transformer, _ = _denoiser(adapters = 0)
        pipe._unsloth_loras = ()
    pipe.components = {"transformer": pipe.transformer}
    pipe._unsloth_phase_seeds = True
    pipe._unsloth_peak_expects_cfg = True
    return pipe


@pytest.fixture
def two_t4(monkeypatch, tmp_path):
    for var in (
        mg.MULTIGPU_ENV,
        mg.CFG_PARALLEL_ENV,
        "CUDA_VISIBLE_DEVICES",
        LORA_MEASURED_ENV,
        *ENVS,
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv(lsp.LAYER_SPLIT_ENV, "0")
    monkeypatch.setattr(mg, "models_kept_loaded", lambda: False)
    monkeypatch.setattr(lsp, "engine_available", lambda: True)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda _: True)
    monkeypatch.setattr(dm, "_torchao_weight_classes", lambda module: {"Int8Tensor"})
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda _: dict(F2KB_LOADED))
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(tmp_path / "peaks.json"))
    mp._cache.update(path = None, mtime = None, data = None)
    cards = [Card(0, 15095, 289), Card(1, 15095, 289)]
    Host(cards).install(monkeypatch)
    mp.record(
        family = "flux.2-klein",
        quant = _load_key(_pipe(adapters = 0)),
        speed_mode = "default",
        width = 1024,
        height = 1024,
        peak_mib = LEARNED_PEAK,
    )
    return cards


def _load_rule_source() -> str:
    """diffusion.py's load statements from ``_peak_model_identity(...)`` to the next ``try:``."""
    import textwrap

    lines = Path(diffusion.__file__).read_text(encoding = "utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if "peak_model = _peak_model_identity(" in line)
    i = start
    while lines[i].strip() != ")":
        i += 1
    end = next(j for j in range(i + 1, len(lines)) if lines[j].strip() == "try:")
    return textwrap.dedent("\n".join(lines[i + 1 : end]))


class _Fam:
    name = "flux.2-klein"
    uses_negative_prompt = True


class _Target:
    dtype = torch.float16
    backend = "cuda"
    vendor = "nvidia"


def _load_model(pipe) -> str:
    """Run the tree's own load rule on ``pipe`` (base: any adapter names no build; head: unpriceable sets only)."""
    scope = {
        "pipe": pipe,
        "fam": _Fam(),
        "target": _Target(),
        "peak_model": "black-forest-labs/FLUX.2-klein-base-4B",
    }
    exec(_load_rule_source(), vars(diffusion), scope)  # noqa: S102 - the tree's own source
    return scope["peak_model"]


def _load_key(pipe) -> str:
    return dm.learned_peak_label(pipe, model = _load_model(pipe), **LABELS)


def _place(cards, pipe):
    memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", cards[0].free_mib, cards[0].total_mib)
    flat = _flat_plan(
        KLEIN_FLAT,
        "flux.2-klein",
        budget_mib = dm._safe_device_budget_mib(memory),
        total_mib = cards[0].total_mib,
    )
    refined = dm.refine_plan_from_loaded_weights(
        pipe, flat, family = "flux.2-klein", speed_mode = "default", model = _load_model(pipe), **LABELS
    )
    with mg.selection_scope(mg.MultiGpuSelection(0, (1,), automatic = False)):
        return refined, mg.refine_plan_for_multigpu(pipe, refined)


@needs_lm
def test_pricing_counts_the_widest_unfused_layer_per_branch():
    pipe = _pipe()
    pricing = lm.price_attached_loras(pipe, cfg = True, compute_bytes = 2)
    assert pricing is not None and getattr(pipe, lm.PRICED_ATTR) is pricing
    # (2 x 27648 + 16) fp16 bytes per token, (64 x 64 + 1024) tokens, 2 branches, rounded up to 64 MiB
    assert pricing.bytes_per_token == (2 * WIDE + 16) * 2
    assert pricing.mib == 1088
    assert lm.price_attached_loras(_pipe(), cfg = False, compute_bytes = 2).mib == 576
    assert lm.price_attached_loras(_pipe(adapters = 2), cfg = True, compute_bytes = 2).mib == 1088
    assert lm.price_attached_loras(_pipe(), cfg = True, compute_bytes = 4).mib == 2176


@needs_lm
@pytest.mark.parametrize("window, mib", [(512, 1088), (2048, 1344), (8192, 2624)])
def test_pricing_covers_the_pipelines_text_window(window, mib):
    class _LongText(_Pipe):
        def __call__(
            self,
            prompt = None,
            max_sequence_length: int = window,
        ):
            raise AssertionError("not rendered")

    pipe = _pipe()
    pipe.__class__ = _LongText
    pricing = lm.price_attached_loras(pipe, cfg = True, compute_bytes = 2)
    assert pricing.text_tokens == max(1024, window)
    assert pricing.mib == mib
    assert lm.request_extra_mib(pipe, width = 1024, height = 1024) == 0
    assert lm.request_extra_mib(pipe, width = 2048, height = 2048) > 0


@needs_lm
def test_unpriceable_sets_keep_the_unknown_build(monkeypatch):
    assert lm.price_attached_loras(_pipe(dora = True), cfg = True) is None
    merged = _pipe()
    for layer in (merged.transformer.wide, merged.transformer.narrow):
        layer.merged_adapters = ["style0"]
    assert lm.price_attached_loras(merged, cfg = True) is None
    assert lm.price_attached_loras(_pipe(adapters = 0), cfg = True) is None
    monkeypatch.setattr("core.inference.diffusion_lora_files.kron_names", lambda pipe: ("kron",))
    assert lm.price_attached_loras(_pipe(), cfg = True) is None
    monkeypatch.undo()
    monkeypatch.setenv(LORA_MEASURED_ENV, "0")
    assert lm.price_attached_loras(_pipe(), cfg = True) is None


@pytest.mark.parametrize("adapter_dtype", [torch.float16, torch.float32])
def test_lora_load_builds_the_key_the_plain_render_recorded(two_t4, adapter_dtype):
    """A LoRA load reads the plain build's learned-peak key (named model, adapter-free quant digest)."""
    plain = _pipe(adapters = 0)
    lora = _pipe()
    for layer in (lora.transformer.wide, lora.transformer.narrow):
        layer.lora_A["style0"].to(adapter_dtype)
        layer.lora_B["style0"].to(adapter_dtype)
    assert _load_model(lora) == _load_model(plain) != "unknown"
    assert _load_key(lora) == _load_key(plain)
    assert (
        mp.lookup(
            family = "flux.2-klein",
            quant = _load_key(lora),
            speed_mode = "default",
            width = 1024,
            height = 1024,
        )
        == LEARNED_PEAK
    )
    from core.inference.diffusion_compile_cache import graph_variant

    assert graph_variant(plain.transformer, skip_adapters = True) == graph_variant(plain.transformer)


def test_lora_load_reads_the_measured_row_and_frees_the_replica_card(two_t4):
    without = _pipe(adapters = 0)
    refined_plain, plain = _place(two_t4, without)
    with_lora = _pipe()
    refined, placed = _place(two_t4, with_lora)
    assert (
        refined.estimates["measured_runtime_headroom_mib"]
        == refined_plain.estimates["measured_runtime_headroom_mib"]
    )
    assert refined.estimates["measured_runtime_headroom_mib"] > dm.measured_image_runtime_mib(
        "flux.2-klein", "default", phase_seeds = True
    )
    assert refined.estimates["lora_activation_mib"] == 1088
    assert "lora_activation_mib" not in refined_plain.estimates
    # nothing moves to cuda:1, so the CFG replica can take it, as without the adapters
    assert placed.secondary_components is None and placed.layer_split is None
    assert plain.secondary_components is None and plain.layer_split is None
    assert int(refined.resident_text_encoder_mib or 0) <= int(
        refined_plain.resident_text_encoder_mib or 0
    )
    assert refined.resident_transformer_mib == F2KB_LOADED["transformer"][0]
    assert refined.resident_text_encoder_mib is None
    assert dm._resident_dit_fits(
        refined.device_memory,
        F2KB_LOADED["transformer"][0],
        refined.estimates["measured_runtime_headroom_mib"]
        + refined.estimates["lora_activation_mib"],
        F2KB_LOADED["vae"][0],
    )


def test_unpriced_lora_load_is_unchanged(two_t4):
    dora = _pipe(dora = True)
    assert _load_model(dora) == "unknown"
    refined, _placed = _place(two_t4, dora)
    assert "lora_activation_mib" not in refined.estimates
    assert refined.estimates.get("measured_runtime_headroom_mib") != LEARNED_PEAK


@needs_lm
def test_non_lora_placement_is_byte_identical_with_the_switch_off(two_t4, monkeypatch):
    on_refined, on = _place(two_t4, _pipe(adapters = 0))
    monkeypatch.setenv(LORA_MEASURED_ENV, "0")
    off_refined, off = _place(two_t4, _pipe(adapters = 0))
    assert on_refined == off_refined and on == off


def _reserved(pipe, headroom: int):
    pipe._unsloth_measured_reserve = (
        headroom,
        "flux.2-klein",
        "default",
        None,
        None,
        LEARNED_PEAK,
        1 << 20,
        False,
    )
    pipe._unsloth_peak_identity = ("flux.2-klein", "torchao:Int8Tensor", "default")
    return pipe


@needs_lm
def test_request_guard_keeps_the_measured_sizing_for_a_priced_set():
    plain = _reserved(_pipe(adapters = 0), 6400)
    expected = dm.measured_request_extra_mib(plain, width = 1024, height = 1024)
    pipe = _reserved(_pipe(), 6400)
    lm.price_attached_loras(pipe, cfg = True, compute_bytes = 2)
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024) == expected
    # a 2048^2 request needs its LoRA forward past the 1024^2 reservation on top of the plain build's figure
    big_plain = dm.measured_request_extra_mib(plain, width = 2048, height = 2048)
    big = dm.measured_request_extra_mib(pipe, width = 2048, height = 2048)
    assert big - big_plain == lm.request_extra_mib(pipe, width = 2048, height = 2048) > 0
    pipe._unsloth_loras = (("other", "/loras/other.safetensors", 1.0),)
    flat = dm.estimate_image_runtime_mib(
        width = 1024, height = 1024, batch_size = 1, family = "flux.2-klein"
    )
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024) == max(0, flat - 6400)


def test_unpriced_runtime_lora_keeps_the_flat_guard():
    pipe = _reserved(_pipe(), 6400)  # adapters attached, never priced (applied after the load)
    flat = dm.estimate_image_runtime_mib(
        width = 1024, height = 1024, batch_size = 1, family = "flux.2-klein"
    )
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024) == max(0, flat - 6400)


@needs_lm
def test_load_prices_the_set_before_naming_the_build():
    src = (Path(dm.__file__).parent / "diffusion.py").read_text(encoding = "utf-8")
    at = src.index("price_attached_loras(\n")
    assert src.index('peak_model = "unknown"', at) > at
    assert "renders still record nothing" in src
