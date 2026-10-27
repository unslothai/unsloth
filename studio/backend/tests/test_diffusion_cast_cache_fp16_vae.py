# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A VAE rescaled in place for fp16 decode is never cached (a reload would rescale it twice)."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("safetensors")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_cast_cache as CC  # noqa: E402
from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference.diffusion_vae_fp16 import enable_fp16_vae_decode  # noqa: E402

_CFG = dict(
    latent_channels = 4,
    layers_per_block = 1,
    norm_num_groups = 8,
    block_out_channels = [16, 32],
    force_upcast = True,
    down_block_types = ["DownEncoderBlock2D"] * 2,
    up_block_types = ["UpDecoderBlock2D"] * 2,
)


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv(C.ENV_DIR, str(tmp_path / "cache"))
    monkeypatch.delenv(CC.CAST_CACHE_ENV, raising = False)
    monkeypatch.delenv(C.ENV_MODE, raising = False)


def _candidate(tmp_path, vae):
    from safetensors.torch import save_file

    source = tmp_path / "vae"
    source.mkdir()
    shard = source / "m.safetensors"
    src = diffusers.AutoencoderKL(**_CFG)
    save_file({k: v.contiguous() for k, v in src.state_dict().items()}, str(shard))
    (source / "config.json").write_text("{}", encoding = "utf-8")
    vae.load_state_dict(src.state_dict())
    cand = CC.CastCandidate(
        component = "vae",
        library = "diffusers",
        class_name = "AutoencoderKL",
        cls = diffusers.AutoencoderKL,
        dtype = torch.float16,
        source = str(source),
        shards = [str(shard)],
        config_path = str(source / "config.json"),
        key = "testkey",
        parts = {
            "kind": "cast",
            "layout": CC.CAST_LAYOUT_VERSION,
            "component": "vae",
            "library": "diffusers",
            "class": "AutoencoderKL",
            "dtype": "float16",
            "keep_in_fp32": [],
        },
    )
    return cand


def test_a_vae_rescaled_for_fp16_decode_is_not_cached(cache_root, tmp_path):
    vae = diffusers.AutoencoderKL(**_CFG)
    cand = _candidate(tmp_path, vae)
    vae = vae.half()
    cand.names = CC._state_names(vae)
    pipe = types.SimpleNamespace(vae = vae, unet = object())
    assert enable_fp16_vae_decode(pipe, types.SimpleNamespace(device = "cuda", backend = "cuda"))
    assert CC.save(cand, vae, logger = logging.getLogger("t")) is None


def test_autoencoder_keys_retire_entries_written_before_the_fix():
    import inspect
    assert '"vae_fp16_decode"' in inspect.getsource(CC)
