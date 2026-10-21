# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An explicit ``fast`` LTX-2.3 single-file load stays resident when it fits, whatever the speed mode."""

import json
import struct
import sys
import types

import pytest

from core.inference.diffusion_memory import (
    MEMORY_MODE_AUTO,
    MEMORY_MODE_FAST,
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    DeviceMemory,
    plan_diffusion_memory,
)

MIB = 1024 * 1024
B200_TOTAL_MIB = 182_633
# What torch reported with another tenant holding ~91 GiB of the card.
TENANT_FREE_MIB = 89_106
# The live LTX-2.3 single-file inputs (ltx-2 table companions 24.4 + 5.5 GB, 121-frame 768x512 headroom).
LTX23_DIT_MIB = 40_073
LTX23_FILE_MIB = 44_011
COMPANION_MIB = 28_514
RUNTIME_MIB = 5_729


def _target():
    return types.SimpleNamespace(device = "cuda", supports_model_cpu_offload = True)


def _b200(free_mib):
    return DeviceMemory("cuda", "cuda", "discrete_vram", free_mib, B200_TOTAL_MIB)


def _plan(
    free_mib,
    transformer_mib,
    mode = MEMORY_MODE_FAST,
):
    return plan_diffusion_memory(
        target = _target(),
        device_memory = _b200(free_mib),
        model_dense_mib = transformer_mib + COMPANION_MIB,
        runtime_headroom_mib = RUNTIME_MIB,
        companion_dense_mib = COMPANION_MIB,
        requested_mode = mode,
    )


def _write_header(path, tensors):
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, mib in tensors.items():
        header[name] = {
            "dtype": "BF16",
            "shape": [mib * MIB // 2],
            "data_offsets": [offset, offset + mib * MIB],
        }
        offset += mib * MIB
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_fast_stays_resident_when_the_dit_fits_free_memory():
    plan = _plan(TENANT_FREE_MIB, LTX23_DIT_MIB)
    assert plan.offload_policy == OFFLOAD_NONE
    assert plan.reasons == ("fast requested; weights resident on device",)


def test_fast_still_offloads_when_resident_would_not_fit():
    plan = _plan(60 * 1024, LTX23_DIT_MIB)
    assert plan.offload_policy == OFFLOAD_GROUP
    assert "do not fit resident" in plan.reasons[0]
    assert _plan(20 * 1024, LTX23_DIT_MIB).offload_policy == OFFLOAD_MODEL


def test_fast_reserve_never_drops_below_two_gib():
    """Half the reserve on a small card would be 1 GiB; the floor keeps the old 2 GiB."""
    small = DeviceMemory("cuda", "cuda", "discrete_vram", 12_000, 12_000)
    fits = plan_diffusion_memory(
        target = _target(),
        device_memory = small,
        model_dense_mib = 12_000 - 2048 - 1000 - 1000,
        runtime_headroom_mib = 1000,
        base_overhead_mib = 1000,
        requested_mode = MEMORY_MODE_FAST,
    )
    over = plan_diffusion_memory(
        target = _target(),
        device_memory = small,
        model_dense_mib = 12_000 - 2048 - 1000 - 1000 + 1,
        runtime_headroom_mib = 1000,
        base_overhead_mib = 1000,
        requested_mode = MEMORY_MODE_FAST,
    )
    assert fits.offload_policy == OFFLOAD_NONE
    assert over.offload_policy != OFFLOAD_NONE


def test_auto_keeps_its_own_headroom():
    assert _plan(TENANT_FREE_MIB, LTX23_DIT_MIB, MEMORY_MODE_AUTO).offload_policy != OFFLOAD_NONE


def test_prefix_mib_reads_only_the_dit_tensors(tmp_path):
    from core.inference.diffusion_memory import safetensors_prefix_mib

    path = tmp_path / "ltx.safetensors"
    _write_header(
        path,
        {
            "model.diffusion_model.a.weight": 30,
            "model.diffusion_model.b.weight": 10,
            "vae.decoder.weight": 5,
            "text_embedding_projection.weight": 2,
        },
    )
    assert safetensors_prefix_mib(path, "model.diffusion_model.") == 40
    assert safetensors_prefix_mib(path, "absent.") is None
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"\xff" * 64)
    assert safetensors_prefix_mib(junk, "model.diffusion_model.") is None
    assert safetensors_prefix_mib(tmp_path / "missing.safetensors", "x") is None


class _Planned(Exception):
    pass


def _plan_inputs_for_single_file_load(monkeypatch, tmp_path, speed_mode, free_mib):
    """Drive ``load_pipeline`` for the LTX-2.3 single file up to its first memory plan; return that plan's inputs."""
    import core.inference.diffusion_te_prequant as te_prequant
    import core.inference.video as vid

    ckpt = tmp_path / "ltx-2.3-22b-distilled.safetensors"
    _write_header(
        ckpt,
        {
            "model.diffusion_model.blocks.weight": LTX23_DIT_MIB,
            "vae.decoder.weight": 1385,
            "audio_vae.weight": 102,
            "vocoder.weight": 246,
            "text_embedding_projection.weight": 2205,
        },
    )
    target = types.SimpleNamespace(
        device = "cuda",
        dtype = None,
        torch_device = "cuda",
        ordinal = None,
        supports_model_cpu_offload = True,
    )
    backend = vid.VideoBackend()
    monkeypatch.setattr(backend, "_device_target", lambda *a, **k: target)
    monkeypatch.setattr(backend, "_resolve_checkpoint_path", lambda *a, **k: ckpt)
    monkeypatch.setattr(vid, "file_size_mib", lambda _p: LTX23_FILE_MIB)
    monkeypatch.setattr(vid, "settled_snapshot_device_memory", lambda _t: _b200(free_mib))
    monkeypatch.setattr(te_prequant, "te_prequant_budget_scale", lambda *a, **k: 1.0)
    # The plan is drawn before any diffusers class is touched; CPU CI installs no diffusers.
    diffusers = types.ModuleType("diffusers")
    diffusers.__version__ = "0.40.0"
    diffusers.LTX2Pipeline = type("LTX2Pipeline", (), {})
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    seen: list = []
    real = plan_diffusion_memory

    def capture(**kw):
        plan = real(**kw)
        seen.append(
            (
                kw["model_dense_mib"],
                kw["companion_dense_mib"],
                kw["requested_mode"],
                plan.offload_policy,
            )
        )
        raise _Planned()

    monkeypatch.setattr(vid, "plan_diffusion_memory", capture)
    with pytest.raises(_Planned):
        backend.load_pipeline(
            "Lightricks/LTX-2.3",
            local_files_only = True,
            gguf_filename = ckpt.name,
            model_kind = "single_file",
            memory_mode = MEMORY_MODE_FAST,
            speed_mode = speed_mode,
        )
    return seen[0]


@pytest.mark.parametrize("speed_mode", [None, "eager", "max"])
def test_ltx23_single_file_prices_the_dit_once_and_stays_resident(
    monkeypatch, tmp_path, speed_mode
):
    dense, companions, mode, policy = _plan_inputs_for_single_file_load(
        monkeypatch, tmp_path, speed_mode, TENANT_FREE_MIB
    )
    assert companions == COMPANION_MIB
    assert dense == LTX23_DIT_MIB + COMPANION_MIB
    assert mode == MEMORY_MODE_FAST
    assert policy == OFFLOAD_NONE


def test_ltx23_plan_does_not_depend_on_the_speed_mode(monkeypatch, tmp_path):
    for free_mib in (TENANT_FREE_MIB, 60 * 1024, B200_TOTAL_MIB):
        plans = {
            speed: _plan_inputs_for_single_file_load(monkeypatch, tmp_path, speed, free_mib)
            for speed in (None, "eager")
        }
        assert plans[None] == plans["eager"], free_mib


def test_fast_that_offloaded_is_recorded_as_a_fallback_naming_the_policy():
    from core.inference.diffusion_auto_policy import build_resolved_record
    from core.inference.video import _memory_mode_resolved

    fell = _plan(60 * 1024, LTX23_DIT_MIB)
    entry = build_resolved_record(
        {"memory_mode": _memory_mode_resolved("fast", fell, fell.offload_policy)}
    )["memory_mode"]
    assert (entry["status"], entry["requested"], entry["value"]) == (
        "fell_back",
        "fast",
        OFFLOAD_GROUP,
    )
    assert "do not fit resident" in entry["reason"]

    kept = _plan(TENANT_FREE_MIB, LTX23_DIT_MIB)
    entry = build_resolved_record(
        {"memory_mode": _memory_mode_resolved("fast", kept, kept.offload_policy)}
    )["memory_mode"]
    assert (entry["status"], entry["value"]) == ("applied", "fast")


def test_resident_proof_uses_the_budget_fast_was_placed_with():
    """Auto precision keeps bf16 only on a proven fit; a fast plan placed resident under its own budget is one."""
    from core.inference.diffusion import _plan_proves_resident

    fast = _plan(TENANT_FREE_MIB, LTX23_DIT_MIB)
    assert fast.offload_policy == OFFLOAD_NONE
    assert fast.estimates["resident_required_mib"] > fast.estimates["safe_device_budget_mib"]
    assert _plan_proves_resident(fast)
    auto = _plan(B200_TOTAL_MIB, LTX23_DIT_MIB, MEMORY_MODE_AUTO)
    assert "resident_budget_mib" not in auto.estimates
    assert _plan_proves_resident(auto)
