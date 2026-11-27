# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A native int8 cache miss builds the layers per tensor from the dense shards, bit-identical to the in-place build."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("safetensors")
pytest.importorskip("accelerate")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion as D  # noqa: E402
from core.inference import diffusion_native_cache as NC  # noqa: E402
from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402

FAMILY = "flux.1"
BASE = "black-forest-labs/FLUX.1-schnell"
CUDA = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


def _fam():
    return detect_family(BASE, override = FAMILY)


def _tiny(dtype = torch.bfloat16):
    torch.manual_seed(0)
    # 2 x 256 = 512 wide: every block Linear passes the 512 filter and g256 ConvRot divides it.
    return diffusers.FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 16,
        num_layers = 1,
        num_single_layers = 1,
        attention_head_dim = 256,
        num_attention_heads = 2,
        joint_attention_dim = 512,
        pooled_projection_dim = 64,
        axes_dims_rope = (32, 112, 112),
    ).to(dtype)


@pytest.fixture(params = ["single", "sharded"])
def base_dir(tmp_path, request):
    root = tmp_path / "FLUX.1-schnell"
    kwargs = {"max_shard_size": "2MB"} if request.param == "sharded" else {}
    _tiny().save_pretrained(str(root / "transformer"), safe_serialization = True, **kwargs)
    shards = sorted((root / "transformer").glob("*.safetensors"))
    assert (len(shards) > 1) == (request.param == "sharded")
    return str(root)


@pytest.fixture(autouse = True)
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(C.ENV_DIR, str(root))
    monkeypatch.delenv(C.ENV_MODE, raising = False)
    monkeypatch.delenv(C.ENV_ASYNC, raising = False)
    monkeypatch.delenv(NC.STREAM_BUILD_ENV, raising = False)
    monkeypatch.setattr(C, "_hub_source_stale", lambda shards, logger = None: False)
    return root


def _in_place(base_dir, act_int8):
    model = diffusers.FluxTransformer2DModel.from_pretrained(
        base_dir, subfolder = "transformer", torch_dtype = torch.bfloat16
    )
    view = types.SimpleNamespace(transformer = model)
    assert (
        tq._quantize_native(
            view,
            "int8",
            family = FAMILY,
            min_features = tq.DEFAULT_MIN_LINEAR_FEATURES,
            act_int8 = act_int8,
            logger = None,
        )
        == "int8"
    )
    return model


def _streamed(base_dir, act_int8, **kwargs):
    return NC.build_from_shards(
        diffusers.FluxTransformer2DModel,
        base_dir,
        NC.native_spec("int8", FAMILY, act_int8 = act_int8),
        family = FAMILY,
        dtype = kwargs.pop("dtype", torch.bfloat16),
        **kwargs,
    )


def _assert_same_module(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]), k
    ma, mb = dict(a.named_modules()), dict(b.named_modules())
    assert ma.keys() == mb.keys()
    for name in ma:
        assert type(ma[name]).__name__ == type(mb[name]).__name__, name
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and pa.requires_grad == pb.requires_grad, na
    ba = dict(a.named_buffers())
    bb = dict(b.named_buffers())
    assert ba.keys() == bb.keys()
    for k in ba:
        assert ba[k].dtype == bb[k].dtype and torch.equal(ba[k], bb[k]), k


@pytest.mark.parametrize("act_int8", [True, False])
def test_streamed_build_is_bit_identical_to_the_in_place_build(base_dir, act_int8):
    ref = _in_place(base_dir, act_int8)
    got = _streamed(base_dir, act_int8)
    assert got is not None
    assert got._unsloth_runtime_quant_info == {"scheme": "int8", "native": True}
    _assert_same_module(ref, got)
    from core.inference.diffusion_native_quant import is_native_linear

    layers = {n: m for n, m in got.named_modules() if is_native_linear(m)}
    assert layers
    for name, layer in ref.named_modules():
        if is_native_linear(layer):
            other = layers[name]
            assert (layer.scheme, layer.act_int8, layer.rot_group, layer.compute_dtype) == (
                other.scheme,
                other.act_int8,
                other.rot_group,
                other.compute_dtype,
            )
    x = torch.randn(1, 8, 16, dtype = torch.bfloat16)
    ctx = torch.randn(1, 4, 512, dtype = torch.bfloat16)
    kwargs = dict(
        hidden_states = x,
        encoder_hidden_states = ctx,
        pooled_projections = torch.randn(1, 64, dtype = torch.bfloat16),
        timestep = torch.tensor([0.5], dtype = torch.bfloat16),
        img_ids = torch.zeros(8, 3),
        txt_ids = torch.zeros(4, 3),
        return_dict = False,
    )
    with torch.no_grad():
        assert torch.equal(ref(**kwargs)[0], got(**kwargs)[0])


def test_cache_entry_is_byte_identical_to_the_in_place_entry(base_dir, tmp_path, monkeypatch):
    monkeypatch.setenv(C.ENV_ASYNC, "0")
    entries = []
    for arm, build in (("ref", _in_place), ("streamed", _streamed)):
        monkeypatch.setenv(C.ENV_DIR, str(tmp_path / f"cache_{arm}"))
        model = build(base_dir, True)
        out = NC.save_deferred(model, _fam(), base_repo = BASE, source_base = base_dir)
        assert isinstance(out, str) and Path(out).is_file()
        entries.append(Path(out))
    assert entries[0].parent.name == entries[1].parent.name
    assert entries[0].read_bytes() == entries[1].read_bytes()


def _mapped_shard_bytes(base_dir):
    total = 0
    for line in Path("/proc/self/maps").read_text().splitlines():
        if str(Path(base_dir).resolve()) in line:
            lo, hi = line.split()[0].split("-")
            total += int(hi, 16) - int(lo, 16)
    return total


def test_only_kept_tensors_stay_mapped(base_dir):
    import gc
    import mmap

    if not Path("/proc/self/maps").is_file():
        pytest.skip("needs /proc/self/maps")
    got = _streamed(base_dir, True)
    assert got is not None
    from core.inference.diffusion_native_quant import is_native_linear

    native = {
        id(t) for m in got.modules() if is_native_linear(m) for t in (m.weight_q, m.weight_scale)
    }
    kept = [t for t in got.state_dict().values() if id(t) not in native]
    dense = sum(t.numel() * t.element_size() for t in _tiny().state_dict().values())
    kept_bytes = sum(t.numel() * t.element_size() for t in kept)
    mapped = _mapped_shard_bytes(base_dir)
    # Each kept tensor maps its own bytes (plus page slack); no quantized weight leaves a mapping behind.
    assert 0 < mapped <= kept_bytes + len(kept) * 2 * mmap.ALLOCATIONGRANULARITY
    assert kept_bytes < dense
    del got, kept
    gc.collect()
    assert _mapped_shard_bytes(base_dir) == 0


def test_declines_leave_the_dense_load(base_dir, tmp_path, monkeypatch):
    # an fp16 load, a non-bf16 checkpoint, missing shards and the kill switch all fall back to the dense load
    assert _streamed(base_dir, True, dtype = torch.float16) is None
    fp32 = tmp_path / "fp32"
    _tiny(torch.float32).save_pretrained(str(fp32 / "transformer"), safe_serialization = True)
    assert _streamed(str(fp32), True) is None
    assert _streamed(str(tmp_path / "missing"), True) is None
    monkeypatch.setenv(NC.STREAM_BUILD_ENV, "0")
    assert _streamed(base_dir, True) is None


def test_a_cancel_is_a_none(base_dir):
    calls = []

    def cancel():
        calls.append(1)
        if len(calls) > 3:
            raise RuntimeError("cancelled")

    assert _streamed(base_dir, True, cancel_check = cancel) is None
    assert len(calls) == 4


def test_backend_builds_on_cuda_only_and_the_save_writes_it(base_dir, monkeypatch):
    monkeypatch.setenv(C.ENV_ASYNC, "0")
    build = lambda target, scheme = "int8": D.DiffusionBackend._native_quant_build_streamed(  # noqa: E731
        _fam(),
        diffusers.FluxTransformer2DModel,
        scheme,
        target,
        torch.bfloat16,
        source_base = base_dir,
    )
    assert build(types.SimpleNamespace(device = "mps", dtype = torch.bfloat16)) is None
    assert build(CUDA, None) is None
    got = build(CUDA)
    assert got is not None
    _assert_same_module(_in_place(base_dir, D.native_int8_act(CUDA)), got)
    out = D.DiffusionBackend._save_pipeline_quant_to_cache(
        types.SimpleNamespace(transformer = got),
        _fam(),
        [("transformer", got)],
        loras = None,
        base = BASE,
        source_base = base_dir,
        defer = True,
        native_cache = True,
    )
    assert isinstance(out, str) and Path(out).is_file()
    hit = D.DiffusionBackend._native_quant_from_cache(
        _fam(), diffusers.FluxTransformer2DModel, "int8", CUDA, base = BASE, source_base = base_dir
    )
    assert hit is not None
    _assert_same_module(got, hit)
