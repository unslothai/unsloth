# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.3 distilled vs dev from the checkpoint's weights (a ``proj_out.bias`` fingerprint), the name only as the
fallback, so a renamed distilled file keeps the distilled recipe and companions."""

from __future__ import annotations

import hashlib

import pytest
import torch
from safetensors.torch import save_file

from core.inference import video_ltx2
from core.inference.video_families import default_video_generation_params
from core.inference.video_ltx2 import (
    checkpoint_variant,
    ltx2_distilled_ids,
    ltx23_checkpoint_variant,
    ltx23_variant_identifier,
)


def _fingerprint(bias: torch.Tensor) -> str:
    return hashlib.sha256(bias.to(torch.bfloat16).view(torch.int16).numpy().tobytes()).hexdigest()[
        :16
    ]


@pytest.fixture
def biases(monkeypatch):
    """Two bf16-exact biases standing in for the distilled and dev releases."""
    gen = torch.Generator().manual_seed(0)
    distilled = torch.randn(128, generator = gen).to(torch.bfloat16)
    dev = torch.randn(128, generator = gen).to(torch.bfloat16)
    monkeypatch.setattr(
        video_ltx2,
        "LTX23_PROJ_OUT_BIAS_SHA",
        {_fingerprint(distilled): "distilled", _fingerprint(dev): "dev"},
    )
    video_ltx2._CONTENT_VARIANT_CACHE.clear()
    yield {"distilled": distilled, "dev": dev}
    video_ltx2._CONTENT_VARIANT_CACHE.clear()


def _ltx_file(
    path,
    bias,
    *,
    dtype = torch.bfloat16,
    prefix = "model.diffusion_model.",
):
    tensors = {
        f"{prefix}proj_out.bias": bias.to(dtype),
        # A quantized weight beside it, as in an fp8 / int8 ComfyUI repack; never read.
        f"{prefix}transformer_blocks.0.attn1.to_q.weight": torch.zeros(
            4, 4, dtype = torch.float8_e4m3fn
        ),
        f"{prefix}transformer_blocks.0.scale_shift_table": torch.zeros(9, 8),
        f"{prefix}audio_proj_out.bias": torch.ones(16),
    }
    save_file(tensors, str(path), metadata = {"model_version": "2.3.0"})
    return path


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("prefix", ["model.diffusion_model.", ""])
def test_a_renamed_distilled_file_reads_distilled_from_its_weights(tmp_path, biases, dtype, prefix):
    path = _ltx_file(
        tmp_path / "LTX-2.3-FP8-ComfyUI.safetensors",
        biases["distilled"],
        dtype = dtype,
        prefix = prefix,
    )
    assert ltx23_checkpoint_variant(path) == "distilled"
    assert ltx23_variant_identifier(path) == "ltx-2.3-22b-distilled"
    assert checkpoint_variant(path) == "distilled"


def test_content_outranks_the_name_both_ways(tmp_path, biases):
    dev_named_distilled = _ltx_file(tmp_path / "my-distilled-copy.safetensors", biases["dev"])
    assert checkpoint_variant(dev_named_distilled) == "dev"
    distilled_named_dev = _ltx_file(tmp_path / "devbox_ltx.safetensors", biases["distilled"])
    assert checkpoint_variant(distilled_named_dev) == "distilled"


def test_an_unknown_fingerprint_or_missing_file_falls_back_to_the_name(tmp_path, biases):
    finetune = _ltx_file(tmp_path / "ltx-2.3-22b-dev-finetune.safetensors", torch.zeros(128))
    assert ltx23_checkpoint_variant(finetune) is None
    assert ltx23_variant_identifier(finetune) is None
    assert checkpoint_variant(finetune) == "dev"
    assert ltx23_checkpoint_variant(tmp_path / "absent.safetensors") is None
    assert ltx23_checkpoint_variant(None) is None
    assert checkpoint_variant(tmp_path / "absent-distilled.safetensors") == "distilled"
    garbage = tmp_path / "garbage.safetensors"
    garbage.write_bytes(b"\x00" * 64)
    assert ltx23_checkpoint_variant(garbage) is None


def test_gguf_bias_is_fingerprinted(tmp_path, biases):
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "LTX-2.3-Q4_K_M.gguf"
    writer = gguf.GGUFWriter(str(path), "ltxv")
    writer.add_tensor("proj_out.bias", biases["dev"].float().numpy())
    writer.add_tensor("audio_proj_out.bias", torch.ones(16).numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert ltx23_checkpoint_variant(path) == "dev"


def test_the_identifier_drives_defaults_and_the_distilled_recipe(tmp_path, biases):
    neutral = _ltx_file(tmp_path / "LTX-2.3-FP8-ComfyUI.safetensors", biases["distilled"])
    variant_id = ltx23_variant_identifier(neutral)
    ids = (variant_id, neutral.name, "local/LTX-2.3-FP8-ComfyUI", "Lightricks/LTX-2")
    assert default_video_generation_params(*ids) == (8, 1.0)
    assert ltx2_distilled_ids(*ids)
    # Without content the neutral name gets the dev recipe: the regression this closes.
    assert default_video_generation_params(*ids[1:]) == (40, 4.0)
    assert not ltx2_distilled_ids(*ids[1:])
    # A dev file under a "...distilled..." name stays dev.
    dev = _ltx_file(tmp_path / "ltx-distilled-mirror.safetensors", biases["dev"])
    dev_ids = (ltx23_variant_identifier(dev), dev.name, "Lightricks/LTX-2")
    assert default_video_generation_params(*dev_ids) == (40, 4.0)
    assert not ltx2_distilled_ids(*dev_ids)


def test_the_real_release_fingerprints_are_registered():
    table = video_ltx2.LTX23_PROJ_OUT_BIAS_SHA
    assert sorted(table.values()) == ["dev", "distilled", "distilled"]
    assert all(len(k) == 16 for k in table)
