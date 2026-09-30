# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An explicit ``fast`` LTX-2.3 single-file load stays resident when it fits, whatever the speed mode.

Measured on a B200 (183 GB): the 22B distilled single file peaks at 72 GiB resident at its 121-frame default. With
89 GB free the planner priced it at 80.3 GB (the file's own VAE / audio VAE / vocoder / connectors counted on top of
the family table's companions) against free minus an 18 GB reserve, and streamed the DiT at 5-8x the step time."""

import json
import struct
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


def _plan(free_mib, transformer_mib, mode = MEMORY_MODE_FAST):
    return plan_diffusion_memory(
        target = _target(),
        device_memory = _b200(free_mib),
        model_dense_mib = transformer_mib + COMPANION_MIB,
        runtime_headroom_mib = RUNTIME_MIB,
        companion_dense_mib = COMPANION_MIB,
        requested_mode = mode,
    )


def _write_header(path, tensors):
    """A safetensors header naming ``tensors`` (name -> MiB); only the header is ever read."""
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
    # A card too small for even the companions falls through to whole-module offload.
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
    """Only an explicit fast is widened: auto still streams the same load at the same free memory."""
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
    seen: list = []
    real = plan_diffusion_memory

    def capture(**kw):
        plan = real(**kw)
        seen.append(
            (kw["model_dense_mib"], kw["companion_dense_mib"], kw["requested_mode"], plan.offload_policy)
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
def test_ltx23_single_file_prices_the_dit_once_and_stays_resident(monkeypatch, tmp_path, speed_mode):
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
