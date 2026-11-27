# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unset text-encoder request on qwen-image keeps the bf16 encoder when the card holds the whole load resident
with room to spare, and the int8 ConvRot file otherwise. Explicit picks and the kill switch always win."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_precision as prec
from core.inference.diffusion_memory import (
    OFFLOAD_GROUP,
    OFFLOAD_NONE,
    DeviceMemory,
)


@pytest.fixture(autouse = True)
def _no_kill_switch(monkeypatch):
    monkeypatch.delenv(prec.TE_BF16_AUTO_ENV, raising = False)


@pytest.mark.parametrize(
    "raw, family, auto_scheme, kind, expected",
    [
        (None, "qwen-image", "int8", "pipeline", True),
        ("auto", "qwen-image", "int8", "pipeline", True),
        ("", "qwen-image", "int8", "pipeline", True),
        ("none", "qwen-image", "int8", "pipeline", False),
        ("off", "qwen-image", "int8", "pipeline", False),
        ("int8", "qwen-image", "int8", "pipeline", False),
        ("fp8", "qwen-image", "int8", "pipeline", False),
        (None, "qwen-image", "fp8", "pipeline", False),
        (None, "qwen-image", None, "pipeline", False),
        (None, "qwen-image", "int8", "gguf", False),
        (None, "qwen-image", "int8", "single_file", False),
        (None, "flux.3-klein", "int8", "pipeline", False),
        (None, "qwen-image-edit", "int8", "pipeline", False),
    ],
)
def test_wanted_table(raw, family, auto_scheme, kind, expected):
    assert prec.te_bf16_auto_wanted(raw, family, auto_scheme, kind) is expected


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_kill_switch(monkeypatch, value):
    monkeypatch.setenv(prec.TE_BF16_AUTO_ENV, value)
    assert prec.te_bf16_auto_wanted(None, "qwen-image", "int8", "pipeline") is False


def _plan(
    required,
    budget,
    *,
    policy = OFFLOAD_NONE,
    kind = "discrete_vram",
    stream = False,
    secondary = None,
):
    return types.SimpleNamespace(
        offload_policy = policy,
        stream_text_encoders = stream,
        secondary_ordinal = secondary,
        device_memory = types.SimpleNamespace(memory_kind = kind),
        estimates = {"resident_required_mib": required, "safe_device_budget_mib": budget},
    )


@pytest.mark.parametrize(
    "plan, expected",
    [
        (None, False),
        (_plan(50_000, 90_000), True),
        # Margin is 10% above 5120 MiB of requirement: 50000 + 5000 is the edge.
        (_plan(50_000, 55_000), True),
        (_plan(50_000, 54_999), False),
        # And 512 MiB below it.
        (_plan(4_000, 4_512), True),
        (_plan(4_000, 4_511), False),
        (_plan(50_000, 90_000, policy = OFFLOAD_GROUP), False),
        (_plan(50_000, 90_000, stream = True), False),
        (_plan(50_000, 90_000, secondary = 1), False),
        (_plan(50_000, 90_000, kind = "unified_memory"), False),
        (_plan(None, 90_000), False),
        (_plan(50_000, None), False),
    ],
)
def test_fits_table(plan, expected):
    assert prec.te_bf16_auto_fits(plan)[0] is expected


GIB = 1024


@pytest.fixture
def backend(monkeypatch):
    """The real planners on a stub card: only the memory reading, the device and the Hub probes are faked."""
    import torch

    from core.inference import diffusion as dmod
    from core.inference import diffusion_auto_policy as ap
    from core.inference import diffusion_memory as mem
    from core.inference import diffusion_transformer_quant as tq

    monkeypatch.setattr(mem, "_installed_torchao_version", lambda: (0, 18))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (12, 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        tq,
        "_scheme_supported",
        lambda scheme, device = None, unproven_ok = False: scheme in ("int8", "fp8"),
    )
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda *_a, **_k: True)
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: 10**7)
    card: dict = {}
    for module in (dmod, mem):
        for fn in (
            "snapshot_device_memory",
            "settled_snapshot_device_memory",
            "reclaimable_snapshot_device_memory",
        ):
            if hasattr(module, fn):
                monkeypatch.setattr(module, fn, lambda *_a, **_k: card["memory"])
    target = types.SimpleNamespace(
        device = "cuda",
        dtype = torch.bfloat16,
        backend = "cuda",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = True,
        ordinal = None,
        capability = (12, 0),
    )
    b = dmod.DiffusionBackend.__new__(dmod.DiffusionBackend)
    b._resolve_device_target = lambda *_a, **_k: target
    b._target_for_ordinal = lambda *_a, **_k: target
    b._cache_bytes = lambda *_a, **_k: 0
    b._released_transformer_cached = lambda *_a, **_k: False
    # The seed probe reads the Hub; the bf16-table branch is the one an unseeded default load prices.
    b._pipeline_planned_denoiser_scheme = lambda *_a, **_k: card.get("seed")

    def room(gib: int, seed = None) -> bool:
        from core.inference.diffusion_families import _FAMILIES

        fam = next(f for f in _FAMILIES if f.name == "qwen-image")
        base = "Qwen/Qwen-Image-2512"
        card["seed"] = seed
        card["memory"] = DeviceMemory("cuda", "cuda", "discrete_vram", gib * GIB - 600, gib * GIB)
        return b._te_bf16_auto_room(fam, base, "pipeline", {"repo_id": base})

    return room


def test_big_card_keeps_the_bf16_encoder_and_small_cards_do_not(backend):
    # Qwen-Image 2512 bf16 DiT + bf16 Qwen2.5-VL-7B + VAE priced from the family table (~64 GiB with activations):
    # 80 and 96 GiB cards hold it with the margin; smaller cards would offload, so they keep the int8 ConvRot file.
    assert backend(96) is True
    assert backend(80) is True
    assert backend(64) is False
    assert backend(24) is False


def test_seeded_int8_denoiser_is_priced_on_its_artifact(backend):
    # The default G4 load seeds the hosted int8 denoiser: ~47 GiB with the bf16 encoder, so 64 GiB keeps it and a
    # 56 GiB card, where the int8 encoder load is still resident, does not.
    assert backend(96, "int8") is True
    assert backend(64, "int8") is True
    assert backend(56, "int8") is False
    assert backend(24, "int8") is False


class _PlannedTe(Exception):
    pass


@pytest.mark.parametrize("room, expected", [(True, None), (False, "int8")])
def test_download_plan_stages_the_encoder_the_load_will_open(monkeypatch, room, expected):
    # The Images page stages /images/download-plan before it loads: the plan must take the same bf16 decision as
    # _run_load, or a fresh download fetches the unused int8 file and the load then pulls the dense encoder too.
    from core.inference import diffusion as dmod

    b = object.__new__(dmod.DiffusionBackend)
    for name in ("flux2_pick_mismatch", "flux3_pick_mismatch", "speech_pick_refusal"):
        monkeypatch.setattr(dmod, name, lambda *_a, **_k: None)
    monkeypatch.setattr(b, "_comfy_text_encoders", lambda *_a, **_k: None, raising = False)
    monkeypatch.setattr(b, "_te_bf16_auto_room", lambda *_a, **_k: room, raising = False)

    def planned(_fam, te_quant, *_a, **_k):
        raise _PlannedTe(te_quant)

    monkeypatch.setattr(b, "_te_prequant_plan_files", planned, raising = False)
    with pytest.raises(_PlannedTe) as seen:
        b.download_plan("Qwen/Qwen-Image-2512", memory_verdict = False)
    assert seen.value.args[0] == expected
