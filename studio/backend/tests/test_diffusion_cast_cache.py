# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local float16 cast cache of fp32-shard components: candidates, bit-identical round trip, refusals, video wiring."""

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

FP16_DEFAULT = {"vae": torch.float32, "default": torch.float16}


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


@pytest.fixture(autouse = True)
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(C.ENV_DIR, str(root))
    monkeypatch.delenv(CC.CAST_CACHE_ENV, raising = False)
    monkeypatch.delenv(C.ENV_MODE, raising = False)
    # No Hub probe from a local directory source.
    monkeypatch.setattr(C, "_hub_source_stale", lambda shards, logger = None: False)
    with C._lock:
        C._REJECTED.clear()
    return root


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


def _dense(cand):
    kwargs = CC._load_kwargs(cand)
    kwargs.pop("output_loading_info")
    return cand.cls.from_pretrained(str(Path(cand.source) / cand.component), **kwargs)


class _Pipe:
    pass


def _save(cand, module):
    pipe = _Pipe()
    setattr(pipe, cand.component, module)
    writes = CC.defer_writes(pipe, [cand])
    assert writes is not None and writes.start()
    assert writes.wait(60)
    return writes.results.get(cand.component)


def _bits(t):
    return t.reshape(-1).view(torch.uint8) if t.numel() else t


def _identical(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert sa[k].dtype == sb[k].dtype, k
        assert torch.equal(_bits(sa[k]), _bits(sb[k])), k
    ba, bb = dict(a.named_buffers()), dict(b.named_buffers())
    assert ba.keys() == bb.keys()
    for k in ba:
        assert ba[k].dtype == bb[k].dtype and torch.equal(ba[k], bb[k]), k


def _by_name(cands):
    return {c.component: c for c in cands}


def test_candidates_are_fp32_shards_loaded_at_fp16(pipe_dir):
    cands = _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))
    assert set(cands) == {
        "transformer",
        "text_encoder",
    }  # the vae stays fp32; the rest hold no weights
    assert len(cands["transformer"].shards) > 1
    assert cands["text_encoder"].parts["keep_in_fp32"] == ["wo"]
    assert cands["transformer"].key != cands["text_encoder"].key


def test_no_candidates_at_float32(pipe_dir):
    assert CC.candidates(pipe_dir, torch_dtype = torch.float32) == []


def test_kill_switches(pipe_dir, monkeypatch):
    monkeypatch.setenv(CC.CAST_CACHE_ENV, "0")
    assert CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT) == []
    monkeypatch.delenv(CC.CAST_CACHE_ENV)
    monkeypatch.setenv(C.ENV_MODE, "0")
    assert CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT) == []


def test_skip_and_narrow_shards(pipe_dir):
    assert set(
        _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT, skip = ("text_encoder",)))
    ) == {"transformer"}
    _tiny_umt5().to(torch.float16).save_pretrained(pipe_dir / "text_encoder")
    assert set(_by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))) == {"transformer"}


@pytest.mark.parametrize("component", ["transformer", "text_encoder"])
def test_round_trip_is_bit_identical(pipe_dir, component):
    cand = _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))[component]
    assert CC.read(cand) is None  # cold: a miss
    dense = _dense(cand)
    path = _save(cand, dense)
    assert path is not None and C.path_in_cache(path)
    stored = CC._shard_header(path)
    assert {e["dtype"] for e in stored.values()} == {"F16", "F32"}  # fp32 keep modules stay fp32
    fresh = _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))[component]
    hit = CC.read(fresh)
    assert hit is not None
    _identical(dense, hit)
    _identical(_dense(fresh), hit)
    if component == "text_encoder":
        # the tied embedding is written once and re-tied by from_pretrained
        assert "encoder.embed_tokens.weight" not in stored
        assert hit.shared.weight.data_ptr() == hit.encoder.embed_tokens.weight.data_ptr()


def test_module_changed_in_place_is_not_cached(pipe_dir):
    cand = _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))["transformer"]
    dense = _dense(cand)
    pipe = _Pipe()
    pipe.transformer = dense
    writes = CC.defer_writes(pipe, [cand])
    with torch.no_grad():
        for p in dense.parameters():
            p.mul_(2)
    writes.start()
    writes.wait(60)
    assert writes.results["transformer"] is None
    assert C._entry(cand.key) is None


def test_module_restructured_after_load_is_not_cached(pipe_dir):
    cand = _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))["transformer"]
    dense = _dense(cand)
    pipe = _Pipe()
    pipe.transformer = dense
    writes = CC.defer_writes(pipe, [cand])
    dense.proj_out = torch.nn.Linear(4, 4)
    writes.start()
    writes.wait(60)
    assert writes.results["transformer"] is None


def test_new_source_bytes_change_the_key(pipe_dir):
    cand = _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))["text_encoder"]
    assert _save(cand, _dense(cand)) is not None
    model = _tiny_umt5()
    with torch.no_grad():
        model.shared.weight.add_(1)
    model.save_pretrained(pipe_dir / "text_encoder")
    fresh = _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))["text_encoder"]
    assert fresh.key != cand.key
    assert CC.read(fresh) is None


def test_corrupt_entry_is_a_miss(pipe_dir):
    cand = _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))["transformer"]
    path = Path(_save(cand, _dense(cand)))
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    (path.parent / C.VERIFIED_NAME).unlink()
    assert CC.read(cand) is None
    with C._lock:
        assert str(path.resolve()) in C._REJECTED


def test_video_reader_injects_hits_and_returns_misses(pipe_dir):
    from core.inference import video as V

    cand = _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))["text_encoder"]
    _save(cand, _dense(cand))
    pipe_kwargs = {"torch_dtype": dict(FP16_DEFAULT), "cache_dir": None, "local_files_only": True}
    missed = V._read_cast_cache(pipe_kwargs, source = str(pipe_dir), revision = None, base = None)
    assert [c.component for c in missed] == ["transformer"]
    assert isinstance(pipe_kwargs["text_encoder"], transformers.UMT5EncoderModel)
    assert "transformer" not in pipe_kwargs
    # an injected component is never a candidate
    again = V._read_cast_cache(dict(pipe_kwargs), source = str(pipe_dir), revision = None, base = None)
    assert [c.component for c in again] == ["transformer"]


def test_teardown_waits_then_clears(pipe_dir):
    from core.inference import video as V

    cand = _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))["transformer"]
    pipe = _Pipe()
    pipe.transformer = _dense(cand)
    writes = CC.defer_writes(pipe, [cand])
    pipe._unsloth_cast_writes = writes
    writes.start()
    V._finish_cast_writes(pipe)
    assert pipe._unsloth_cast_writes is None
    assert writes.done()
    assert C._entry(cand.key) is not None


def test_sample_keys_tolerate_one_kind_per_shard():
    names = {"a": torch.float16, "b": torch.float16, "w": torch.float32}
    entry = lambda n: {"dtype": "F32", "data_offsets": [0, n]}  # noqa: E731
    assert CC._sample_keys({"a": entry(8), "b": entry(16)}, names) == ["b"]
    assert CC._sample_keys({"w": entry(8)}, names) == ["w"]
    assert CC._sample_keys({"a": entry(CC._SAMPLE_MAX_BYTES + 1)}, names) == []


def test_save_when_kept_fp32_tensors_exceed_the_sample_size(pipe_dir, monkeypatch):
    cand = _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))["text_encoder"]
    dense = _dense(cand)
    wide = min(t.numel() * 4 for n, t in dense.state_dict().items() if t.dtype == torch.float32)
    monkeypatch.setattr(CC, "_SAMPLE_MAX_BYTES", wide - 1)
    assert _save(cand, dense) is not None
    _identical(dense, CC.read(cand))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_resident_device_module_is_read_back_bit_identical(pipe_dir):
    cand = _by_name(CC.candidates(pipe_dir, torch_dtype = FP16_DEFAULT))["transformer"]
    dense = _dense(cand)
    reference = {k: v.clone() for k, v in dense.state_dict().items()}
    dense.to("cuda")
    assert _save(cand, dense) is not None
    hit = CC.read(cand)
    for k, v in hit.state_dict().items():
        assert v.dtype == reference[k].dtype and torch.equal(_bits(v), _bits(reference[k])), k
