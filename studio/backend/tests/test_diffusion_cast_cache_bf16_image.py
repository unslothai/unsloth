# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cast cache beyond video float16: bfloat16 targets, the image pipeline wiring, the per-load budget, live-module
tolerances (a compiled wrapper, tensors added after the load) and the per-dtype / per-path switches."""

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

BF16 = torch.bfloat16
HALF = (torch.float16, torch.bfloat16)
IMAGE_ENV = "UNSLOTH_DIFFUSION_CAST_CACHE_IMAGE"


def _tiny_zimage():
    torch.manual_seed(0)
    return diffusers.ZImageTransformer2DModel(
        all_patch_size = (2,),
        all_f_patch_size = (1,),
        in_channels = 4,
        dim = 32,
        n_layers = 2,
        n_refiner_layers = 1,
        n_heads = 2,
        n_kv_heads = 2,
        cap_feat_dim = 16,
        axes_dims = [4, 6, 6],
        axes_lens = [64, 32, 32],
    ).to(torch.float32)


def _tiny_vae():
    torch.manual_seed(1)
    return diffusers.AutoencoderKL(
        in_channels = 3,
        out_channels = 3,
        down_block_types = ("DownEncoderBlock2D",),
        up_block_types = ("UpDecoderBlock2D",),
        block_out_channels = (8,),
        latent_channels = 4,
        norm_num_groups = 4,
    ).to(torch.float32)


def _tiny_umt5():
    torch.manual_seed(2)
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
    for env in (CC.CAST_CACHE_ENV, IMAGE_ENV, C.ENV_MODE, C.ENV_MAX_GB):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(C, "_hub_source_stale", lambda shards, logger = None: False)
    with C._lock:
        C._REJECTED.clear()
    return root


@pytest.fixture
def image_dir(tmp_path):
    root = tmp_path / "Z-Image-tiny"
    _tiny_zimage().save_pretrained(root / "transformer", max_shard_size = "40KB")
    _tiny_vae().save_pretrained(root / "vae")
    _tiny_umt5().save_pretrained(root / "text_encoder")
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "ZImagePipeline",
                "transformer": ["diffusers", "ZImageTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "UMT5EncoderModel"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            }
        )
    )
    return root


def _by_name(cands):
    return {c.component: c for c in cands}


def _dense(cand):
    kwargs = CC._load_kwargs(cand)
    kwargs.pop("output_loading_info")
    return cand.cls.from_pretrained(str(Path(cand.source) / cand.component), **kwargs)


class _Pipe:
    pass


def _write(pipe, cands):
    writes = CC.defer_writes(pipe, cands)
    assert writes is not None and writes.start()
    assert writes.wait(60)
    return writes.results


def _bits(t):
    return t.reshape(-1).view(torch.uint8) if t.numel() else t


def _identical(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert sa[k].dtype == sb[k].dtype, k
        assert torch.equal(_bits(sa[k]), _bits(sb[k])), k


@pytest.mark.parametrize("dtype", HALF)
def test_every_fp32_component_is_a_candidate_at_both_half_dtypes(image_dir, dtype):
    cands = _by_name(CC.candidates(image_dir, torch_dtype = dtype))
    assert set(cands) == {"transformer", "vae", "text_encoder"}
    assert {c.parts["dtype"] for c in cands.values()} == {str(dtype).replace("torch.", "")}
    assert all(c.nbytes > 0 for c in cands.values())


def test_fp16_and_bf16_entries_never_share_a_key(image_dir):
    a = _by_name(CC.candidates(image_dir, torch_dtype = torch.float16))
    b = _by_name(CC.candidates(image_dir, torch_dtype = BF16))
    assert all(a[k].key != b[k].key for k in a)


@pytest.mark.parametrize("dtype", HALF)
@pytest.mark.parametrize("component", ["transformer", "vae", "text_encoder"])
def test_round_trip_is_bit_identical(image_dir, dtype, component):
    cand = _by_name(CC.candidates(image_dir, torch_dtype = dtype))[component]
    assert CC.read(cand) is None
    dense = _dense(cand)
    pipe = _Pipe()
    setattr(pipe, component, dense)
    path = _write(pipe, [cand])[component]
    assert path is not None and C.path_in_cache(path)
    tag = {torch.float16: "F16", BF16: "BF16"}[dtype]
    assert tag in {e["dtype"] for e in CC._shard_header(path).values()}
    fresh = _by_name(CC.candidates(image_dir, torch_dtype = dtype))[component]
    hit = CC.read(fresh)
    assert hit is not None
    _identical(dense, hit)


def test_dtype_switch_keeps_one_target(image_dir, monkeypatch):
    monkeypatch.setenv(CC.CAST_CACHE_ENV, "fp16")
    assert CC.candidates(image_dir, torch_dtype = BF16) == []
    assert CC.candidates(image_dir, torch_dtype = torch.float16)
    monkeypatch.setenv(CC.CAST_CACHE_ENV, "bf16")
    assert CC.candidates(image_dir, torch_dtype = torch.float16) == []
    assert CC.candidates(image_dir, torch_dtype = BF16)
    monkeypatch.setenv(CC.CAST_CACHE_ENV, "0")
    assert CC.candidates(image_dir, torch_dtype = BF16) == []


def test_load_set_fits_the_budget(image_dir, monkeypatch):
    sizes = {c.component: c.nbytes for c in CC.candidates(image_dir, torch_dtype = BF16)}
    # room for the transformer alone: the others never get an entry they would evict on every load
    monkeypatch.setenv(C.ENV_MAX_GB, str((sizes["transformer"] + 1) / 2**30))
    names = [c.component for c in CC.candidates(image_dir, torch_dtype = BF16)]
    assert "transformer" in names and sum(sizes[n] for n in names) <= sizes["transformer"] + 1


def test_compiled_wrapper_and_added_tensors_still_write_the_load(image_dir):
    cand = _by_name(CC.candidates(image_dir, torch_dtype = BF16))["transformer"]
    dense = _dense(cand)
    reference = {k: v.clone() for k, v in dense.state_dict().items()}
    pipe = _Pipe()
    pipe.transformer = dense
    writes = CC.defer_writes(pipe, [cand])
    dense.register_parameter("extra_fused", torch.nn.Parameter(torch.zeros(3, dtype = BF16)))
    pipe.transformer = torch.compile(dense)
    assert writes.start() and writes.wait(60)
    assert writes.results["transformer"] is not None
    hit = CC.read(cand)
    assert set(hit.state_dict()) == set(reference)
    for k, v in hit.state_dict().items():
        assert torch.equal(_bits(v), _bits(reference[k])), k


def test_renamed_or_retyped_tensor_still_refuses(image_dir):
    cand = _by_name(CC.candidates(image_dir, torch_dtype = BF16))["vae"]
    dense = _dense(cand)
    pipe = _Pipe()
    pipe.vae = dense
    writes = CC.defer_writes(pipe, [cand])
    dense.to(
        torch.float32
    )  # an upcast after the load (or fp8 layerwise storage) is not what the load produced
    assert writes.start() and writes.wait(60)
    assert writes.results["vae"] is None


def test_image_helpers_inject_hits_and_skip_the_prefetch(image_dir, monkeypatch):
    from core.inference import diffusion as D

    fams = {f.name: f for f in D._IMAGE_REGISTRY}
    fam = fams["z-image"]
    cands = D._image_cast_candidates(fam, str(image_dir), dtype = BF16, base = None)
    assert {c.component for c in cands} == {"transformer", "vae", "text_encoder"}
    assert D._image_cast_cached(cands, None) == []
    tr = _by_name(cands)["transformer"]
    pipe = _Pipe()
    pipe.transformer = _dense(tr)
    assert _write(pipe, [tr])["transformer"] is not None
    cands = D._image_cast_candidates(fam, str(image_dir), dtype = BF16, base = None)
    assert D._image_cast_cached(cands, None) == ["transformer"]
    assert D._image_cast_cached(cands, "fp16") == []  # twins read: the prefetch must read them
    pipe_kwargs = {"torch_dtype": BF16, "local_files_only": True}
    missed = CC.inject(pipe_kwargs, cands)
    assert sorted(c.component for c in missed) == ["text_encoder", "vae"]
    _identical(pipe_kwargs["transformer"], pipe.transformer)
    # the per-path switch, and families assembled per component, are never wired
    monkeypatch.setenv(IMAGE_ENV, "0")
    assert D._image_cast_candidates(fam, str(image_dir), dtype = BF16, base = None) == []
    monkeypatch.delenv(IMAGE_ENV)
    for name in (D.KREA2_FAMILY_NAME, D.MOSS_IMAGE_FAMILY_NAME, "flux.3-klein"):
        assert D._image_cast_candidates(fams[name], str(image_dir), dtype = BF16, base = None) == []


def test_video_reader_serves_bf16(image_dir):
    from core.inference import video as V

    cand = _by_name(CC.candidates(image_dir, torch_dtype = BF16))["text_encoder"]
    pipe = _Pipe()
    pipe.text_encoder = _dense(cand)
    assert _write(pipe, [cand])["text_encoder"] is not None
    pipe_kwargs = {"torch_dtype": {"vae": torch.float32, "default": BF16}, "local_files_only": True}
    missed = V._read_cast_cache(pipe_kwargs, source = str(image_dir), revision = None, base = None)
    assert sorted(c.component for c in missed) == ["transformer"]
    _identical(pipe_kwargs["text_encoder"], pipe.text_encoder)


def test_pipeline_from_injected_components_matches_the_dense_load(image_dir):
    """The whole assembly: a pipeline loaded with the hits equals the dense bf16 pipeline load bit for bit."""
    cands = CC.candidates(image_dir, torch_dtype = BF16)
    dense = {c.component: _dense(c) for c in cands}
    pipe = _Pipe()
    for name, module in dense.items():
        setattr(pipe, name, module)
    assert all(_write(pipe, cands).values())
    kwargs = {}
    assert CC.inject(kwargs, CC.candidates(image_dir, torch_dtype = BF16)) == []
    for name, module in dense.items():
        _identical(module, kwargs[name])
