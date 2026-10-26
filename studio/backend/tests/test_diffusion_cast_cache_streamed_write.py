# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cast-cache write that runs while the published pipeline renders reads block-streamed tensors from their offload
group's host copy, never from a device destination whose upload has not landed yet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_cast_cache as CC  # noqa: E402
from core.inference import diffusion_prequant_cache as C  # noqa: E402

FP16_DEFAULT = {"vae": torch.float32, "default": torch.float16}


@pytest.fixture(autouse = True)
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(C.ENV_DIR, str(root))
    monkeypatch.delenv(CC.CAST_CACHE_ENV, raising = False)
    monkeypatch.delenv(C.ENV_MODE, raising = False)
    monkeypatch.setattr(C, "_hub_source_stale", lambda shards, logger = None: False)
    with C._lock:
        C._REJECTED.clear()
    return root


@pytest.fixture
def pipe_dir(tmp_path):
    torch.manual_seed(0)
    model = diffusers.WanTransformer3DModel(
        patch_size = (1, 2, 2),
        num_attention_heads = 2,
        attention_head_dim = 16,
        in_channels = 4,
        out_channels = 4,
        text_dim = 32,
        freq_dim = 16,
        ffn_dim = 64,
        num_layers = 2,
        cross_attn_norm = True,
        qk_norm = "rms_norm_across_heads",
        eps = 1e-6,
        rope_max_seq_len = 32,
    ).to(torch.float32)
    root = tmp_path / "Wan2.1-T2V-tiny-Diffusers"
    model.save_pretrained(root / "transformer", max_shard_size = "60KB")
    (root / "model_index.json").write_text(
        json.dumps(
            {"_class_name": "WanPipeline", "transformer": ["diffusers", "WanTransformer3DModel"]}
        )
    )
    return root


def _cand(pipe_dir):
    return {c.component: c for c in CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT)}[
        "transformer"
    ]


def _dense(cand):
    kwargs = CC._load_kwargs(cand)
    kwargs.pop("output_loading_info")
    return cand.cls.from_pretrained(str(Path(cand.source) / cand.component), **kwargs)


def _bits(t):
    return t.reshape(-1).view(torch.uint8) if t.numel() else t


def _assert_hit_equals(cand, reference: dict):
    hit = CC.read(cand)
    assert hit is not None
    for k, v in hit.state_dict().items():
        assert v.dtype == reference[k].dtype and torch.equal(_bits(v), _bits(reference[k])), k


def _unsampled_cast_key(cand) -> str:
    """A float16 tensor the pre-write source comparison does not sample, so only the write itself can catch it."""
    names = {n: d for n, d, _s in cand.names}
    sampled = set()
    for shard in cand.shards:
        sampled.update(CC._sample_keys(CC._shard_header(shard) or {}, names))
    keys = sorted(
        n for n, d in names.items() if d == torch.float16 and n not in sampled and "weight" in n
    )
    assert keys
    return keys[0]


class _Group:
    def __init__(self, cpu_param_dict):
        self.cpu_param_dict = cpu_param_dict


class _Hook:
    def __init__(self, group):
        self.group = group


class _Registry:
    def __init__(self, hook):
        self.hooks = {"group_offloading": hook}


def test_streamed_tensor_is_written_from_its_host_copy(pipe_dir):
    cand = _cand(pipe_dir)
    dense = _dense(cand)
    reference = {k: v.clone() for k, v in dense.state_dict().items()}
    host = {t: t.data.clone() for t in [*dense.parameters(), *dense.buffers()]}
    dense._diffusers_hook = _Registry(_Hook(_Group(host)))
    CC.capture(type("P", (), {"transformer": dense})(), [cand])
    key = _unsampled_cast_key(cand)
    live = dense.get_parameter(key)
    live.data = torch.full_like(live.data, float("nan"))
    assert CC.save(cand, dense) is not None
    _assert_hit_equals(cand, reference)


def test_unstreamed_module_still_reads_its_live_tensors(pipe_dir):
    cand = _cand(pipe_dir)
    dense = _dense(cand)
    reference = {k: v.clone() for k, v in dense.state_dict().items()}
    CC.capture(type("P", (), {"transformer": dense})(), [cand])
    assert CC.save(cand, dense) is not None
    _assert_hit_equals(cand, reference)


def test_layout_version_retires_entries_written_by_the_racy_reader(pipe_dir):
    assert CC.CAST_LAYOUT_VERSION >= 2
    assert _cand(pipe_dir).parts["layout"] == CC.CAST_LAYOUT_VERSION


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_block_streamed_write_during_a_refill_is_bit_identical(pipe_dir, monkeypatch):
    """Before every read each group is re-pointed to fresh NaN device tensors whose upload has not landed."""
    from diffusers.hooks import apply_group_offloading

    monkeypatch.setattr(CC, "_ONLOAD_WAIT_SECONDS", 0.0)
    cand = _cand(pipe_dir)
    dense = _dense(cand)
    reference = {k: v.clone() for k, v in dense.state_dict().items()}
    apply_group_offloading(
        dense,
        onload_device = torch.device("cuda"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
    )
    CC.capture(type("P", (), {"transformer": dense})(), [cand])
    groups = []
    for m in dense.modules():
        for hook in getattr(getattr(m, "_diffusers_hook", None), "hooks", {}).values():
            g = getattr(hook, "group", None)
            if g is not None and all(g is not x for x in groups):
                groups.append(g)
    # a group holding no tensors (e.g. the unmatched-modules group) keeps an empty host copy
    groups = [g for g in groups if g.cpu_param_dict]
    assert groups

    def onload():
        for g in groups:
            with torch.cuda.stream(g.stream):
                for t, src in g.cpu_param_dict.items():
                    t.data = src.to("cuda", non_blocking = True)
        torch.cuda.synchronize()

    names = {n: d for n, d, _s in cand.names}
    sampled = sum(len(CC._sample_keys(CC._shard_header(sh) or {}, names)) for sh in cand.shards)
    state = {"calls": 0}

    def prefetch():
        # After the sampled reads (so the write itself is what goes wrong), as the prefetcher does: `.data`
        # re-pointed now, its upload queued behind work that has not finished.
        state["calls"] += 1
        if state["calls"] <= sampled:
            return
        moved = {t: torch.full_like(t.data, float("nan")) for g in groups for t in g.cpu_param_dict}
        torch.cuda.synchronize()
        for g in groups:
            for t in g.cpu_param_dict:
                t.data = moved[t]
            with torch.cuda.stream(g.stream):
                torch.cuda._sleep(50_000_000)
                for t, src in g.cpu_param_dict.items():
                    t.data.copy_(src, non_blocking = True)

    onload()
    assert CC.save(cand, dense, check = prefetch) is not None
    torch.cuda.synchronize()
    _assert_hit_equals(cand, reference)
