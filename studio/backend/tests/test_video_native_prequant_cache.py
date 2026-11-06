# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native prequant cache for offloaded two-expert video DiTs: round trip, keys, all-or-nothing reads, routing."""

from __future__ import annotations

import os

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

from core.inference import diffusion_native_cache as NC  # noqa: E402
from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference.video_families import detect_video_family  # noqa: E402

FAMILY = "wan2.2-t2v-a14b"
COMPONENTS = ("transformer", "transformer_2")


def _family():
    return detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers", override = FAMILY)


def _tiny_config() -> dict:
    # 2 x 256 = 512 wide: every attention / FFN Linear passes the 512 filter and the g256 ConvRot divides it.
    return dict(
        patch_size = (1, 2, 2),
        num_attention_heads = 2,
        attention_head_dim = 256,
        in_channels = 4,
        out_channels = 4,
        text_dim = 64,
        freq_dim = 32,
        ffn_dim = 1024,
        num_layers = 2,
        cross_attn_norm = True,
        qk_norm = "rms_norm_across_heads",
        eps = 1e-6,
        rope_max_seq_len = 64,
    )


@pytest.fixture
def base_dir(tmp_path):
    root = tmp_path / "Wan2.2-T2V-A14B-Diffusers"
    for seed, component in enumerate(COMPONENTS):
        torch.manual_seed(seed)
        model = diffusers.WanTransformer3DModel(**_tiny_config())
        model.save_pretrained(str(root / component), safe_serialization = True)
    return str(root)


@pytest.fixture(autouse = True)
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(C.ENV_DIR, str(root))
    monkeypatch.delenv(C.ENV_MODE, raising = False)
    monkeypatch.setattr(C, "_hub_source_stale", lambda shards, logger = None: False)
    return root


def _spec(act_int8 = True):
    return NC.native_spec("int8", FAMILY, act_int8 = act_int8)


def _fresh_build(
    base_dir: str,
    component: str,
    act_int8 = True,
):
    """What the offloaded load's runtime quant builds: dense bf16 from_pretrained, then the native pass."""
    model = diffusers.WanTransformer3DModel.from_pretrained(
        base_dir, subfolder = component, torch_dtype = torch.bfloat16
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


def _inputs():
    g = torch.Generator().manual_seed(7)
    hidden = torch.randn(1, 4, 3, 8, 8, generator = g).to(torch.bfloat16)
    text = torch.randn(1, 6, 64, generator = g).to(torch.bfloat16)
    t = torch.tensor([500])
    return hidden, t, text


def _assert_same_module(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert sa[k].dtype == sb[k].dtype, k
        assert torch.equal(sa[k], sb[k]), k
    ba, bb = (
        dict(a.named_buffers()),
        dict(b.named_buffers()),
    )  # non-persistent too (rope tables, rot_h)
    assert ba.keys() == bb.keys()
    for k in ba:
        assert ba[k].dtype == bb[k].dtype and torch.equal(ba[k], bb[k]), k
    for (na, ma), (nb, mb) in zip(a.named_modules(), b.named_modules()):
        assert na == nb and type(ma).__name__ == type(mb).__name__, na
        assert ma.extra_repr() == mb.extra_repr(), na
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys()
    for k in pa:
        assert pa[k].requires_grad == pb[k].requires_grad, k


def test_round_trip_is_bit_identical_per_expert(base_dir):
    fam = _family()
    spec = _spec()
    cls = diffusers.WanTransformer3DModel
    for component in COMPONENTS:
        fresh = _fresh_build(base_dir, component)
        assert NC.save(fresh, fam, spec, base_repo = base_dir, component = component) is not None
        path = NC.lookup(fam, spec, base_repo = base_dir, component = component)
        assert path is not None
        cached = NC.load(path, cls, base_dir, component = component, spec = spec)
        assert cached is not None
        assert cached._unsloth_runtime_quant_info["native"] is True
        _assert_same_module(fresh, cached)
        with torch.no_grad():
            out_a = fresh(*_inputs(), return_dict = False)[0]
            out_b = cached(*_inputs(), return_dict = False)[0]
        assert torch.equal(out_a, out_b)


def test_experts_have_distinct_keys_and_never_cross(base_dir):
    fam = _family()
    spec = _spec()
    fresh = _fresh_build(base_dir, "transformer")
    NC.save(fresh, fam, spec, base_repo = base_dir, component = "transformer")
    assert NC.lookup(fam, spec, base_repo = base_dir, component = "transformer") is not None
    # Expert 2 never resolves expert 1's entry: it would load fine and render with the wrong weights.
    assert NC.lookup(fam, spec, base_repo = base_dir, component = "transformer_2") is None
    k1 = NC._key_for(fam, spec, base_dir, "transformer")[0]
    k2 = NC._key_for(fam, spec, base_dir, "transformer_2")[0]
    assert k1 != k2


def test_key_tracks_build_and_source(base_dir):
    fam = _family()
    spec = _spec()
    NC.save(
        _fresh_build(base_dir, "transformer"),
        fam,
        spec,
        base_repo = base_dir,
        component = "transformer",
    )
    assert (
        NC.lookup(fam, _spec(act_int8 = False), base_repo = base_dir, component = "transformer") is None
    )
    assert _spec(act_int8 = False)["rot_group"] == 0 and _spec()["rot_group"] == 256
    assert (
        NC.lookup(
            fam, spec, base_repo = base_dir + "-other", source_base = base_dir, component = "transformer"
        )
        is None
    )
    torch.manual_seed(99)
    diffusers.WanTransformer3DModel(**_tiny_config()).save_pretrained(
        str(Path(base_dir) / "transformer"), safe_serialization = True
    )
    assert NC.lookup(fam, spec, base_repo = base_dir, component = "transformer") is None


def test_torchao_lookup_never_serves_a_native_entry(base_dir):
    fam = _family()
    NC.save(
        _fresh_build(base_dir, "transformer"),
        fam,
        _spec(),
        base_repo = base_dir,
        component = "transformer",
    )
    assert C.lookup(fam, "int8", base_repo = base_dir, component = "transformer") is None


def test_save_refuses_a_build_the_spec_does_not_describe(base_dir):
    fam = _family()
    weight_only = _fresh_build(base_dir, "transformer", act_int8 = False)
    assert (
        NC.save(weight_only, fam, _spec(act_int8 = True), base_repo = base_dir, component = "transformer")
        is None
    )
    dense = diffusers.WanTransformer3DModel.from_pretrained(
        base_dir, subfolder = "transformer", torch_dtype = torch.bfloat16
    )
    assert NC.save(dense, fam, _spec(), base_repo = base_dir, component = "transformer") is None


def test_corrupt_entry_is_a_miss(base_dir):
    fam = _family()
    spec = _spec()
    path = NC.save(
        _fresh_build(base_dir, "transformer"),
        fam,
        spec,
        base_repo = base_dir,
        component = "transformer",
    )
    (Path(path).parent / C.VERIFIED_NAME).unlink()
    with open(path, "r+b") as fh:
        fh.seek(-16, 2)
        fh.write(b"\x01" * 16)
    assert (
        NC.load(path, diffusers.WanTransformer3DModel, base_dir, component = "transformer", spec = spec)
        is None
    )


def test_a_rejected_entry_is_replaced_by_the_next_build(base_dir):
    fam = _family()
    spec = _spec()
    kw = dict(base_repo = base_dir, component = "transformer")
    path = NC.save(_fresh_build(base_dir, "transformer"), fam, spec, **kw)
    (Path(path).parent / C.VERIFIED_NAME).unlink()
    with open(path, "r+b") as fh:
        fh.seek(-16, 2)
        fh.write(b"\x01" * 16)
    load = lambda p: NC.load(
        p, diffusers.WanTransformer3DModel, base_dir, component = "transformer", spec = spec
    )
    assert load(path) is None
    # same key and size: without the rejection check this save was skipped and every later load missed
    replaced = NC.save(_fresh_build(base_dir, "transformer"), fam, spec, **kw)
    assert replaced is not None and load(replaced) is not None
    assert os.path.realpath(replaced) not in C._REJECTED


def test_video_read_is_all_or_nothing(base_dir):
    from core.inference import video as V

    fam = _family()
    spec = _spec()
    NC.save(
        _fresh_build(base_dir, "transformer"),
        fam,
        spec,
        base_repo = base_dir,
        component = "transformer",
    )
    kw = dict(base = base_dir, source = base_dir, config_kwargs = {})
    assert V._read_native_quant_from_cache(fam, spec, **kw) == {}
    NC.save(
        _fresh_build(base_dir, "transformer_2"),
        fam,
        spec,
        base_repo = base_dir,
        component = "transformer_2",
    )
    hits = V._read_native_quant_from_cache(fam, spec, **kw)
    assert set(hits) == set(COMPONENTS)
    assert V._read_native_quant_from_cache(fam, None, **kw) == {}


def test_video_save_writes_one_entry_per_expert(base_dir):
    from core.inference import video as V

    fam = _family()
    spec = _spec()
    pipe = types.SimpleNamespace(
        transformer = _fresh_build(base_dir, "transformer"),
        transformer_2 = _fresh_build(base_dir, "transformer_2"),
    )
    views = (pipe, types.SimpleNamespace(transformer = pipe.transformer_2))
    V._save_native_quant_to_cache(fam, views, spec, base = base_dir, source = base_dir)
    for component in COMPONENTS:
        assert NC.lookup(fam, spec, base_repo = base_dir, component = component) is not None
    hits = V._read_native_quant_from_cache(
        fam, spec, base = base_dir, source = base_dir, config_kwargs = {}
    )
    assert all(
        NC.save(m, fam, spec, base_repo = base_dir, component = c) is None for c, m in hits.items()
    )


@pytest.fixture
def nvidia(monkeypatch):
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)


OFFLOAD = types.SimpleNamespace(offload_policy = "group", stream_transformer = True)
RESIDENT = types.SimpleNamespace(offload_policy = "none")


@pytest.mark.parametrize(
    "quant, plan, kind, device, expect",
    [
        ("int8", OFFLOAD, "pipeline", "cuda", True),
        ("auto", OFFLOAD, "pipeline", "cuda", True),
        (None, OFFLOAD, "pipeline", "cuda", False),
        ("off", OFFLOAD, "pipeline", "cuda", False),
        (
            "int8",
            RESIDENT,
            "pipeline",
            "cuda",
            False,
        ),  # resident: torchao / hosted path, not native
        (
            "fp8",
            OFFLOAD,
            "pipeline",
            "cuda",
            False,
        ),  # fp8 has no torchao-free offload build on NVIDIA
        ("int8", OFFLOAD, "gguf", "cuda", False),
        ("int8", OFFLOAD, "pipeline", "mps", False),
    ],
)
def test_offload_routing_gate(nvidia, quant, plan, kind, device, expect):
    from core.inference import video as V

    target = types.SimpleNamespace(device = device, dtype = torch.bfloat16)
    spec = V._video_native_cache_spec(
        _family(), target = target, transformer_quant = quant, plan = plan, kind = kind, source_dir = None
    )
    if not expect:
        assert spec is None
        return
    assert spec == NC.native_spec("int8", FAMILY, act_int8 = tq.native_int8_act(target))
    assert spec["act_int8"] is True and spec["rot_group"] == 256
    assert spec == V._native_spec_for("int8", _family(), True)


def test_hosted_int8_is_not_the_offload_build():
    """The hosted A14B INT8 is a torchao v1 W8A8 checkpoint without ConvRot; the offload build rotates (g256)."""
    from core.inference.diffusion_transformer_quant import int8_convrot_enabled

    assert not int8_convrot_enabled(FAMILY)
    assert _spec()["rot_group"] == 256
