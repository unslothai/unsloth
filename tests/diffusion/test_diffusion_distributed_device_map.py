# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastDiffusionModel under torchrun: each rank loads one whole model on its own device, as
FastLanguageModel / FastModel do. With the default device_map = "auto" every rank spread the model
over all visible cards, and a 2-rank native FSDP2 launch died in transformers' infer_auto_device_map
(IndexError) before training."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "DiffusionGemmaForBlockDiffusion"):
    pytest.skip("transformers without DiffusionGemma", allow_module_level = True)

TINY = "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion"


class _Stop(Exception):
    pass


def _placement(monkeypatch, env, **kwargs):
    import unsloth.models.diffusion as diffusion

    for key in ("WORLD_SIZE", "LOCAL_RANK", "RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    seen = {}

    def resolve(device_map, *args, **kw):
        seen["device_map"] = device_map
        raise _Stop

    monkeypatch.setattr(diffusion, "resolve_unsloth_device_map", resolve)
    with pytest.raises(_Stop):
        diffusion.FastDiffusionModel.from_pretrained(TINY, dtype = torch.float32, **kwargs)
    return seen["device_map"]


RANK1 = {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "2"}


def test_each_rank_loads_on_its_own_device(monkeypatch):
    from unsloth.models.loader_utils import DEVICE_TYPE_TORCH
    assert _placement(monkeypatch, RANK1) == {"": f"{DEVICE_TYPE_TORCH}:1"}


def test_a_device_the_caller_named_is_kept(monkeypatch):
    assert _placement(monkeypatch, RANK1, device_map = "cpu") == "cpu"
    assert _placement(monkeypatch, RANK1, device_map = {"": "cpu"}) == {"": "cpu"}


def test_a_single_process_keeps_auto(monkeypatch):
    assert _placement(monkeypatch, {}) == "auto"
