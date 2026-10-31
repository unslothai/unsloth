# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video small-host staging: routing, kill switch, bit-identical staged entry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_cast_cache as CC  # noqa: E402
from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference import video_small_host as VS  # noqa: E402

FP16_DEFAULT = {"vae": torch.float32, "default": torch.float16}
# Wan2.1-T2V-1.3B on a Colab T4: UMT5 13.4 GB loaded (wo kept fp32), DiT 2.8 GB; 12.7 GB host, ~9.9 GB available
WAN13 = {"text_encoder": 12748, "transformer": 2709}


@pytest.fixture(autouse = True)
def _clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv(C.ENV_DIR, str(tmp_path / "cache"))
    monkeypatch.delenv(CC.CAST_CACHE_ENV, raising = False)
    monkeypatch.delenv(C.ENV_MODE, raising = False)
    monkeypatch.delenv(VS.SMALL_HOST_ENV, raising = False)
    monkeypatch.setattr(C, "_hub_source_stale", lambda shards, logger = None: False)
    with C._lock:
        C._REJECTED.clear()


def test_big_host_is_unchanged():
    plan = VS.plan_stage(WAN13, available_mib = 48000, reserve_mib = 5300, device_free_mib = 22000)
    assert not plan.engaged and plan.stage == ()
    assert "fits" in plan.reason


def test_small_t4_host_stages_the_encoder_only():
    plan = VS.plan_stage(WAN13, available_mib = 10100, reserve_mib = 3072, device_free_mib = 14600)
    assert plan.engaged
    # the DiT's dense conversion (2.7 GB) fits once the encoder is off the host
    assert plan.stage == ("text_encoder",)
    assert plan.dense_mib == sum(WAN13.values())


def test_stages_largest_first_until_the_rest_fits():
    misses = {"text_encoder": 9000, "transformer": 6000, "transformer_2": 6000}
    plan = VS.plan_stage(misses, available_mib = 12000, reserve_mib = 3000, device_free_mib = 40000)
    assert plan.stage == ("text_encoder", "transformer")


def test_card_too_small_for_the_encoder_keeps_the_dense_load():
    plan = VS.plan_stage(WAN13, available_mib = 10100, reserve_mib = 3072, device_free_mib = 11000)
    # the encoder does not fit the card; the DiT alone does but cannot rescue the load: staged anyway, logged
    assert "text_encoder" in plan.skipped
    assert plan.stage == ("transformer",)


def test_kill_switch_and_force(monkeypatch):
    monkeypatch.setenv(VS.SMALL_HOST_ENV, "0")
    assert VS.stage_disabled()
    plan = VS.plan_stage(
        WAN13,
        available_mib = 100,
        reserve_mib = 3072,
        device_free_mib = 14600,
        disabled = VS.stage_disabled(),
    )
    assert not plan.engaged
    monkeypatch.setenv(VS.SMALL_HOST_ENV, "1")
    plan = VS.plan_stage(
        WAN13,
        available_mib = 48000,
        reserve_mib = 5300,
        device_free_mib = 22000,
        forced = VS.stage_forced(),
    )
    assert plan.stage == ("text_encoder", "transformer")


def test_unreadable_host_ram_changes_nothing():
    plan = VS.plan_stage(WAN13, available_mib = None, reserve_mib = 3072, device_free_mib = 14600)
    assert not plan.engaged


def test_small_candidates_are_never_staged():
    plan = VS.plan_stage({"vae": 200}, available_mib = 100, reserve_mib = 3072, device_free_mib = 14600)
    assert not plan.engaged and plan.skipped == {"vae": "small"}


def _tiny_umt5():
    torch.manual_seed(1)
    config = transformers.UMT5Config(
        vocab_size = 64,
        d_model = 32,
        d_kv = 8,
        d_ff = 64,
        num_layers = 2,
        num_heads = 4,
        relative_attention_num_buckets = 8,
    )
    return transformers.UMT5EncoderModel(config).to(torch.float32)


def _tiny_wan():
    torch.manual_seed(0)
    return diffusers.WanTransformer3DModel(
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


@pytest.fixture
def pipe_dir(tmp_path):
    root = tmp_path / "Wan2.1-T2V-tiny-Diffusers"
    _tiny_wan().save_pretrained(root / "transformer", max_shard_size = "60KB")
    _tiny_umt5().save_pretrained(root / "text_encoder")
    (root / "vae").mkdir()
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "WanPipeline",
                "transformer": ["diffusers", "WanTransformer3DModel"],
                "text_encoder": ["transformers", "UMT5EncoderModel"],
                "vae": ["diffusers", "AutoencoderKLWan"],
                "tokenizer": ["transformers", "T5TokenizerFast"],
                "scheduler": ["diffusers", "UniPCMultistepScheduler"],
            }
        )
    )
    return root


def _cands(pipe_dir):
    return {c.component: c for c in CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT)}


def _dense(cand):
    kwargs = CC._load_kwargs(cand)
    kwargs.pop("output_loading_info")
    return cand.cls.from_pretrained(str(Path(cand.source) / cand.component), **kwargs)


def _identical(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert sa[k].dtype == sb[k].dtype, k
        assert sa[k].device.type == "cpu" and sb[k].device.type == "cpu", k
        assert torch.equal(
            sa[k].reshape(-1).view(torch.uint8), sb[k].reshape(-1).view(torch.uint8)
        ), k


def test_loaded_mib_keeps_wo_wide(pipe_dir):
    cand = _cands(pipe_dir)["text_encoder"]
    assert cand.parts["keep_in_fp32"] == ["wo"]
    dense = _dense(cand)
    want = sum(t.numel() * t.element_size() for t in dense.state_dict().values())
    # tied embed_tokens is stored once; the estimate counts stored tensors at their loaded width
    got_bytes = 0
    for shard in cand.shards:
        for name, e in CC._shard_header(shard).items():
            b, end = e["data_offsets"]
            got_bytes += (end - b) if ".wo." in f".{name}" else (end - b) // 2
    assert VS.loaded_mib(cand) == -(-got_bytes // VS._MIB)
    assert got_bytes <= want


@pytest.mark.parametrize("component", ["text_encoder", "transformer"])
def test_staged_entry_reads_bit_identical_to_the_dense_load(pipe_dir, component):
    cand = _cands(pipe_dir)[component]
    assert VS.stage_on_device(cand, "cpu")
    hit = CC.read(_cands(pipe_dir)[component])
    assert hit is not None
    _identical(hit, _dense(cand))
    if component == "text_encoder":
        assert hit.encoder.block[0].layer[1].DenseReluDense.wo.weight.dtype == torch.float32
        assert hit.shared.weight.dtype == torch.float16


def _wire(
    monkeypatch,
    available,
    total = 13000,
    free = 14600,
):
    from core.inference import diffusion_small_host as SH

    calls = []
    real = VS.stage_on_device

    def fake_stage(cand, device, **kw):
        calls.append((cand.component, str(device)))
        return real(cand, "cpu", **kw)

    monkeypatch.setattr(VS, "stage_on_device", fake_stage)
    monkeypatch.setattr(VS, "_device_free_mib", lambda device: free)
    monkeypatch.setattr(VS, "_STAGE_MIN_MIB", 0)
    monkeypatch.setattr(SH, "host_ram_mib", lambda: (total, available))
    return calls


def test_video_reader_stages_then_serves_the_encoder_on_a_small_host(pipe_dir, monkeypatch):
    from core.inference import video as V

    sizes = {"text_encoder": 9000, "transformer": 2700}
    monkeypatch.setattr(VS, "loaded_mib", lambda cand: sizes[cand.component])
    calls = _wire(monkeypatch, available = 10000)
    pipe_kwargs = {"torch_dtype": dict(FP16_DEFAULT), "cache_dir": None, "local_files_only": True}
    missed = V._read_cast_cache(
        pipe_kwargs, source = str(pipe_dir), revision = None, base = None, device = "cuda"
    )
    assert calls == [("text_encoder", "cuda")]
    assert [c.component for c in missed] == ["transformer"]
    _identical(pipe_kwargs["text_encoder"], _dense(_cands(pipe_dir)["text_encoder"]))


def test_video_reader_on_a_big_host_never_stages(pipe_dir, monkeypatch):
    from core.inference import video as V

    calls = _wire(monkeypatch, available = 48000, total = 52000)
    pipe_kwargs = {"torch_dtype": dict(FP16_DEFAULT), "cache_dir": None, "local_files_only": True}
    missed = V._read_cast_cache(
        pipe_kwargs, source = str(pipe_dir), revision = None, base = None, device = "cuda"
    )
    assert calls == []
    assert sorted(c.component for c in missed) == ["text_encoder", "transformer"]
    assert "text_encoder" not in pipe_kwargs


def test_video_reader_kill_switch(pipe_dir, monkeypatch):
    from core.inference import video as V

    monkeypatch.setenv(VS.SMALL_HOST_ENV, "0")
    calls = _wire(monkeypatch, available = 10)
    pipe_kwargs = {"torch_dtype": dict(FP16_DEFAULT), "cache_dir": None, "local_files_only": True}
    V._read_cast_cache(pipe_kwargs, source = str(pipe_dir), revision = None, base = None, device = "cuda")
    assert calls == []


def test_non_cuda_device_never_stages(pipe_dir, monkeypatch):
    calls = _wire(monkeypatch, available = 10)
    assert VS.stage_misses(list(_cands(pipe_dir).values()), "cpu") == []
    assert calls == []
