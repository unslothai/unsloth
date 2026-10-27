# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Incomplete release caches are priced from the family bf16 table (CPU only)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from core.inference import diffusion as dmod
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_families import (
    detect_family,
    local_pipeline_components_are_complete,
)
from core.inference.diffusion_memory import OFFLOAD_NONE, DeviceMemory

MIRROR = "unsloth/FLUX.1-dev"
UPSTREAM = "black-forest-labs/FLUX.1-dev"
FINETUNE = "someone/flux-dev-finetune"
REV = "0" * 40
MIB = 1024 * 1024

_MODELS = {
    "text_encoder": ("transformers", "CLIPTextModel", "model.safetensors"),
    "text_encoder_2": ("transformers", "T5EncoderModel", "model.safetensors"),
    "transformer": ("diffusers", "FluxTransformer2DModel", "diffusion_pytorch_model.safetensors"),
    "vae": ("diffusers", "AutoencoderKL", "diffusion_pytorch_model.safetensors"),
}


def _safetensors(path: Path, mib: int) -> None:
    """A valid one-tensor safetensors file of about ``mib`` MiB."""
    numel = mib * MIB // 2
    header = json.dumps(
        {"w": {"dtype": "BF16", "shape": [numel], "data_offsets": [0, numel * 2]}}
    ).encode()
    header += b" " * (-len(header) % 8)
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "wb") as fh:
        fh.write(len(header).to_bytes(8, "little") + header)
        fh.truncate(8 + len(header) + numel * 2)


def _snapshot(
    hub: Path,
    repo_id: str,
    components: tuple[str, ...],
    mib: int = 2,
) -> Path:
    repo = hub / f"models--{repo_id.replace('/', '--')}"
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text(REV)
    snap = repo / "snapshots" / REV
    index = {"_class_name": "FluxPipeline", "_diffusers_version": "0.30.0"}
    index["scheduler"] = ["diffusers", "FlowMatchEulerDiscreteScheduler"]
    index["tokenizer"] = ["transformers", "CLIPTokenizer"]
    index["tokenizer_2"] = ["transformers", "T5TokenizerFast"]
    for name, (lib, cls, _f) in _MODELS.items():
        index[name] = [lib, cls]
    snap.mkdir(parents = True)
    (snap / "model_index.json").write_text(json.dumps(index))
    for folder, files in (
        ("scheduler", {"scheduler_config.json": "{}"}),
        ("tokenizer", {"tokenizer_config.json": "{}", "vocab.json": "{}", "merges.txt": "#v\n"}),
        ("tokenizer_2", {"tokenizer_config.json": "{}", "tokenizer.json": "{}"}),
    ):
        (snap / folder).mkdir()
        for name, text in files.items():
            (snap / folder / name).write_text(text)
    for name in components:
        (snap / name).mkdir()
        (snap / name / "config.json").write_text("{}")
        _safetensors(snap / name / _MODELS[name][2], mib)
    return snap


@pytest.fixture
def l4(monkeypatch, tmp_path):
    from huggingface_hub import constants as hf_constants

    live, other = tmp_path / "live-hub", tmp_path / "import-time-hub"
    live.mkdir()
    other.mkdir()
    monkeypatch.setattr(dmod, "hub_cache_dir", lambda: str(live))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(other))
    monkeypatch.setattr(
        dmod,
        "settled_snapshot_device_memory",
        lambda t: DeviceMemory("cuda", "cuda", "discrete_vram", 22_200, 23_034),
    )
    monkeypatch.setattr(dmod, "estimate_image_runtime_mib", lambda **kw: 1_000)
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)
    return types.SimpleNamespace(target = target, hub = live)


def _plan(
    target,
    repo_id,
    base = UPSTREAM,
):
    fam = detect_family(UPSTREAM)
    assert fam is not None
    return DiffusionBackend()._plan_memory(
        target, None, base, fam, None, False, kind = "pipeline", repo_id = repo_id, fetch_base = repo_id
    )


def _table_total_mib():
    fam = detect_family(UPSTREAM)
    table = dmod.family_bf16_components_gb(fam, UPSTREAM)
    return int(sum(table) * dmod._GB_TO_MIB)


def test_fixture_snapshots_read_as_complete_and_incomplete(l4):
    full = _snapshot(l4.hub, "a/full", tuple(_MODELS))
    part = _snapshot(l4.hub, "a/part", ("text_encoder", "text_encoder_2"))
    assert local_pipeline_components_are_complete(full, "model_index.json")
    assert not local_pipeline_components_are_complete(part, "model_index.json")


@pytest.mark.parametrize("repo_id", [MIRROR, UPSTREAM])
def test_text_encoders_only_cache_prices_the_release_from_its_table(l4, repo_id):
    """A cache holding only the text encoders must not price the pipeline at the encoders."""
    _snapshot(l4.hub, repo_id, ("text_encoder", "text_encoder_2"), mib = 64)
    plan = _plan(l4.target, repo_id)
    assert plan.estimates["model_dense_mib"] >= _table_total_mib(), plan.estimates
    assert plan.estimates["companion_dense_mib"] is not None
    assert plan.estimates["text_encoder_dense_mib"] is not None
    assert plan.offload_policy != OFFLOAD_NONE, plan.reasons


def test_an_empty_cache_prices_the_release_from_its_table(l4):
    plan = _plan(l4.target, MIRROR)
    assert plan.estimates["model_dense_mib"] >= _table_total_mib(), plan.estimates
    assert plan.offload_policy != OFFLOAD_NONE, plan.reasons


def test_a_complete_cache_keeps_its_own_bytes(l4):
    """A load whose cache already holds every component plans exactly from those bytes, as before."""
    _snapshot(l4.hub, MIRROR, tuple(_MODELS), mib = 4)
    plan = _plan(l4.target, MIRROR)
    assert plan.estimates["model_dense_mib"] < 64, plan.estimates


def test_a_finetune_with_a_partial_cache_keeps_its_own_bytes(l4):
    _snapshot(l4.hub, FINETUNE, ("text_encoder", "text_encoder_2"), mib = 8)
    plan = _plan(l4.target, FINETUNE, base = FINETUNE)
    assert plan.estimates["model_dense_mib"] < 64, plan.estimates


def test_floor_helper_contract():
    fam = detect_family(UPSTREAM)
    total = _table_total_mib()
    floored = dmod._incomplete_release_cache_floor(fam, MIRROR, False, 9318, 9318, 9318)
    assert floored is not None and floored[0] == total
    assert floored[1] >= 9318 and floored[2] >= 9318
    assert dmod._incomplete_release_cache_floor(fam, MIRROR, True, 9318, 9318, 9318) is None
    assert dmod._incomplete_release_cache_floor(fam, FINETUNE, False, 9318, 9318, 9318) is None
    assert dmod._incomplete_release_cache_floor(fam, None, False, None, None, None) is None
