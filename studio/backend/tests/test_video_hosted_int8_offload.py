# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted A14B INT8 served to an offloaded load as torchao-free W8A8 layers: decode, build, gates, routing."""

from __future__ import annotations

import contextlib
import enum
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("accelerate")
pytest.importorskip("torchao")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_prequant as pq  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference import prequant_legacy_int8 as li  # noqa: E402
from core.inference import video_hosted_int8_native as hn  # noqa: E402
from core.inference.diffusion_native_quant import is_native_linear  # noqa: E402
from core.inference.video_families import detect_video_family  # noqa: E402

FAMILY = "wan2.2-t2v-a14b"
BASE = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
COMPONENTS = ("transformer", "transformer_2")

needs_decoder = pytest.mark.skipif(
    not (li.legacy_int8_classes_missing() and li.legacy_int8_decode_supported()),
    reason = "needs torchao >= 0.18 (the legacy int8 pickle decoder)",
)


def _family():
    return detect_video_family(BASE, override = FAMILY)


def tiny_config() -> dict:
    # 512 wide: every attention / FFN Linear passes the 512 filter, as in the native cache tests.
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


@contextlib.contextmanager
def v1_int8_classes():
    """Stand-ins for torchao's deleted v1 int8 classes under their pickled names, as torchao 0.17 wrote them."""

    class _Base(torch.Tensor):
        __torch_function__ = torch._C._disabled_torch_function_impl

        @staticmethod
        def __new__(cls, shape, dtype):
            return torch.Tensor._make_wrapper_subclass(cls, shape, dtype = dtype)

        @classmethod
        def __torch_dispatch__(
            cls,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            raise RuntimeError("stand-in")

        def __reduce_ex__(self, proto):
            args = (
                type(self),
                self.dtype,
                tuple(self.size()),
                self.stride(),
                self.storage_offset(),
                self.layout,
                self.device,
                False,
            )
            return (
                torch._tensor._rebuild_from_type_v2,
                (torch._utils._rebuild_wrapper_subclass, type(self), args, dict(self.__dict__)),
            )

    names = {
        "torchao.quantization.linear_activation_quantized_tensor": "LinearActivationQuantizedTensor",
        "torchao.dtypes.affine_quantized_tensor": "AffineQuantizedTensor",
        "torchao.dtypes.uintx.plain_layout": "PlainAQTTensorImpl",
    }
    mods, made = {}, {}
    for mod, name in names.items():
        cls = type(name, (_Base,), {"__module__": mod, "__qualname__": name})
        mods[mod] = types.ModuleType(mod)
        setattr(mods[mod], name, cls)
        made[name] = cls
    layout = type(
        "PlainLayout", (), {"__module__": "torchao.dtypes.utils", "__qualname__": "PlainLayout"}
    )
    mods["torchao.dtypes.utils"] = types.ModuleType("torchao.dtypes.utils")
    mods["torchao.dtypes.utils"].PlainLayout = layout

    def _int8_symm_per_token_reduced_range_quant(x):
        raise RuntimeError("never called")

    _int8_symm_per_token_reduced_range_quant.__module__ = "torchao.quantization.quant_api"
    _int8_symm_per_token_reduced_range_quant.__qualname__ = (
        "_int8_symm_per_token_reduced_range_quant"
    )
    real_api = sys.modules.get("torchao.quantization.quant_api")
    api = types.ModuleType("torchao.quantization.quant_api")
    if real_api is not None:
        api.__dict__.update(real_api.__dict__)
    api._int8_symm_per_token_reduced_range_quant = _int8_symm_per_token_reduced_range_quant
    mods["torchao.quantization.quant_api"] = api
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        yield made, layout, _int8_symm_per_token_reduced_range_quant
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def v1_int8_quantise(weight):
    """torchao v1 ``Int8DynamicActivationInt8WeightConfig`` weight maths: symmetric per row over [-128, 127],
    scale ``amax / 127.5`` in the weight's dtype."""
    w = weight.detach().float()
    amax = w.abs().amax(dim = 1)
    scale = (amax / 127.5).clamp(min = torch.finfo(torch.float32).eps).to(weight.dtype)
    codes = torch.round(w / scale.float()[:, None]).clamp(-128, 127).to(torch.int8)
    return codes, scale


def write_hosted_v1(
    model,
    path,
    *,
    component,
    base = BASE,
    metadata = None,
    zero_point = None,
):
    """A hosted-format INT8 ``.pt`` of ``model`` (torchao v1 W8A8 pickle, the published A14B layout). Returns
    ``{fqn: (codes, scale)}`` of the quantised layers."""
    from torchao.quantization.quant_primitives import ZeroPointDomain

    keep = tq.make_filter_fn(
        tq.DEFAULT_MIN_LINEAR_FEATURES,
        exclude_name_tokens = tq.exclude_tokens_for_scheme("int8", FAMILY),
    )
    quantised = {name: mod for name, mod in model.named_modules() if name and keep(mod, name)}
    written = {}
    with v1_int8_classes() as (made, layout, act):
        state = {}
        for key, value in model.state_dict().items():
            fqn = key[: -len(".weight")] if key.endswith(".weight") else None
            if fqn not in quantised:
                state[key] = value.detach().clone()
                continue
            codes, scale = v1_int8_quantise(value)
            n, k = codes.shape
            impl = made["PlainAQTTensorImpl"]((n, k), torch.int8)
            impl.__dict__.update(
                int_data = codes, scale = scale, zero_point = zero_point, _layout = layout()
            )
            aqt = made["AffineQuantizedTensor"]((n, k), value.dtype)
            aqt.__dict__.update(
                tensor_impl = impl,
                block_size = (1, k),
                quant_min = None,
                quant_max = None,
                zero_point_domain = ZeroPointDomain.NONE,
            )
            laqt = made["LinearActivationQuantizedTensor"]((n, k), value.dtype)
            laqt.__dict__.update(original_weight_tensor = aqt, input_quant_func = act, quant_kwargs = {})
            state[key] = laqt
            written[fqn] = (codes, scale)
        meta = {
            "scheme": "int8",
            "base_model_id": base,
            "family": FAMILY,
            "min_features": tq.DEFAULT_MIN_LINEAR_FEATURES,
            "component": component,
            **(metadata or {}),
        }
        torch.save({"format": pq.PREQUANT_FORMAT, "metadata": meta, "state_dict": state}, str(path))
    return written


def dense_expert(seed: int):
    torch.manual_seed(seed)
    return diffusers.WanTransformer3DModel(**tiny_config()).to(torch.bfloat16).eval()


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    """One hosted-format file per expert, served by the resolver without the Hub."""
    files, written, dense = {}, {}, {}
    for seed, component in enumerate(COMPONENTS):
        dense[component] = dense_expert(seed)
        path = tmp_path / f"{component}.pt"
        written[component] = write_hosted_v1(dense[component], path, component = component)
        files[component] = str(path)
    config = dict(tiny_config())
    monkeypatch.setattr(pq, "hosted_prequant_refusal", lambda *a, **k: None)
    monkeypatch.setattr(
        pq,
        "_resolve_checkpoint_path",
        lambda source, *a, **k: files[getattr(source, "filename", None) or "transformer"],
    )
    monkeypatch.setattr(pq, "_load_transformer_config", lambda *a, **k: dict(config))
    return types.SimpleNamespace(files = files, written = written, dense = dense)


def _source(component):
    return pq.PrequantSource(
        kind = "repo", location = "unsloth/Wan2.2-T2V-A14B-FP8", filename = component
    )


def _load(component):
    return hn.load_hosted_int8_native(
        diffusers.WanTransformer3DModel, BASE, _source(component), component = component
    )


@needs_decoder
@pytest.mark.parametrize("component", COMPONENTS)
def test_hosted_codes_and_scales_are_held_bit_for_bit(hosted, component):
    model = _load(component)
    assert model is not None
    layers = {n: m for n, m in model.named_modules() if is_native_linear(m)}
    assert set(layers) == set(hosted.written[component])
    for fqn, (codes, scale) in hosted.written[component].items():
        layer = layers[fqn]
        assert layer.act_int8 and layer.rot_group == 0 and layer.scheme == "int8"
        assert torch.equal(layer.weight_q, codes)
        assert torch.equal(layer.weight_scale.view(torch.float32), scale.float())
    dense = hosted.dense[component].state_dict()
    for key, value in model.state_dict().items():
        if key in dense:
            assert torch.equal(value, dense[key]), key
    assert model._unsloth_runtime_quant_info["native"] is True
    assert all(t.device.type == "cpu" for t in (*model.parameters(), *model.buffers()))


@needs_decoder
def test_the_experts_stay_distinct(hosted):
    a, b = _load("transformer"), _load("transformer_2")
    qa = dict(a.named_buffers())["blocks.0.attn1.to_q.weight_q"]
    qb = dict(b.named_buffers())["blocks.0.attn1.to_q.weight_q"]
    assert not torch.equal(qa, qb)


@needs_decoder
def test_forward_tracks_the_dense_expert(hosted):
    model = _load("transformer")
    g = torch.Generator().manual_seed(7)
    hidden = torch.randn(1, 4, 3, 8, 8, generator = g).to(torch.bfloat16)
    text = torch.randn(1, 6, 64, generator = g).to(torch.bfloat16)
    t = torch.tensor([500])
    with torch.no_grad():
        ref = hosted.dense["transformer"](hidden, t, text, return_dict = False)[0].float()
        out = model(hidden, t, text, return_dict = False)[0].float()
    assert torch.isfinite(out).all()
    rel = float((out - ref).norm() / ref.norm())
    assert rel < 0.05, rel


@needs_decoder
def test_a_rotated_checkpoint_is_refused(hosted, tmp_path, monkeypatch):
    from core.inference import diffusion_convrot

    monkeypatch.setattr(diffusion_convrot, "declares_rotation", lambda meta: True)
    monkeypatch.setattr(pq, "_validate_checkpoint", lambda *a, **k: True)
    assert _load("transformer") is None


@needs_decoder
def test_a_wrong_base_is_refused(hosted, tmp_path):
    path = Path(hosted.files["transformer"])
    write_hosted_v1(hosted.dense["transformer"], path, component = "transformer", base = "someone/else")
    assert _load("transformer") is None


@needs_decoder
def test_an_asymmetric_weight_fails_the_load(hosted, tmp_path):
    path = Path(hosted.files["transformer"])
    n = hosted.dense["transformer"].blocks[0].attn1.to_q.out_features
    write_hosted_v1(
        hosted.dense["transformer"],
        path,
        component = "transformer",
        zero_point = torch.ones(n, dtype = torch.int8),
    )
    assert _load("transformer") is None


def test_a_local_path_source_is_never_served():
    src = pq.PrequantSource(kind = "path", location = "/tmp/x.pt", filename = None)
    assert (
        hn.load_hosted_int8_native(
            diffusers.WanTransformer3DModel, BASE, src, component = "transformer"
        )
        is None
    )


def _int8_tensor(
    n = 8,
    k = 32,
    **extra,
):
    from core.inference.prequant_legacy_int8 import _int8_tensor_api, _to_int8_tensor

    t = _to_int8_tensor(
        torch.randint(-128, 128, (n, k), dtype = torch.int8),
        torch.rand(n).to(torch.bfloat16) + 0.01,
        torch.bfloat16,
        _int8_tensor_api(),
    )
    for key, value in extra.items():
        setattr(t, key, value)
    return t


@needs_decoder
def test_weight_parts_read_int8tensor_and_refuse_what_they_cannot_describe():
    assert hn.int8_weight_parts("w", torch.ones(2, 2)) is None
    codes, scale, dtype = hn.int8_weight_parts("w", _int8_tensor())
    assert codes.dtype == torch.int8 and scale.shape == (8,) and dtype == torch.bfloat16
    for extra in (
        {"act_quant_kwargs": None},
        {"act_pre_scale": torch.ones(32)},
        {"act_quant_scale": torch.ones(1)},
        {"zero_point": torch.ones(8, 1, dtype = torch.int8)},
        {"block_size": [1, 16]},
    ):
        with pytest.raises(ValueError):
            hn.int8_weight_parts("w", _int8_tensor(**extra))


# ── gates ────────────────────────────────────────────────────────────────────


@pytest.fixture
def nvidia(monkeypatch):
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.delenv(hn.HOSTED_INT8_OFFLOAD_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_NATIVE_INT8_ACT", raising = False)


CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


@pytest.mark.parametrize(
    "family, scheme, target, env, act, expect",
    [
        (FAMILY, "int8", CUDA_BF16, None, None, True),
        (FAMILY, "INT8", CUDA_BF16, None, None, True),
        (FAMILY, "int8", CUDA_BF16, "0", None, False),  # kill switch
        (FAMILY, "int8", CUDA_BF16, "off", None, False),
        (FAMILY, "int8", CUDA_BF16, None, "0", False),  # weight-only native is another build
        (FAMILY, "fp8", CUDA_BF16, None, None, False),
        (FAMILY, "nvfp4", CUDA_BF16, None, None, False),
        ("wan2.2-ti2v-5b", "int8", CUDA_BF16, None, None, False),
        (
            FAMILY,
            "int8",
            types.SimpleNamespace(device = "cuda", dtype = torch.float16),
            None,
            None,
            False,
        ),
        (
            FAMILY,
            "int8",
            types.SimpleNamespace(device = "mps", dtype = torch.bfloat16),
            None,
            None,
            False,
        ),
    ],
)
def test_eligibility(nvidia, monkeypatch, family, scheme, target, env, act, expect):
    if env is not None:
        monkeypatch.setenv(hn.HOSTED_INT8_OFFLOAD_ENV, env)
    if act is not None:
        monkeypatch.setenv("UNSLOTH_NATIVE_INT8_ACT", act)
    fam = detect_video_family(BASE, override = FAMILY)
    fam = types.SimpleNamespace(
        name = family, modular_workflow = getattr(fam, "modular_workflow", None)
    )
    assert hn.hosted_int8_offload_eligible(fam, scheme, target) is expect


def test_rocm_is_not_eligible(monkeypatch):
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: True)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.setattr(tq, "native_quant_host", lambda target: False)
    assert not hn.hosted_int8_offload_eligible(_family(), "int8", CUDA_BF16)


def test_a_full_native_cache_keeps_the_cached_build(nvidia, monkeypatch):
    from core.inference import diffusion_native_cache as NC

    hits = {"transformer": "/c/a", "transformer_2": "/c/b"}
    monkeypatch.setattr(NC, "lookup", lambda fam, spec, component, **k: hits.get(component))
    assert hn.native_cache_covers(_family(), CUDA_BF16, BASE)
    assert not hn.hosted_int8_offload_ok(_family(), "int8", CUDA_BF16, BASE)
    hits.pop("transformer_2")  # a partial cache is no cache: the hosted pair wins
    assert not hn.native_cache_covers(_family(), CUDA_BF16, BASE)
    assert hn.hosted_int8_offload_ok(_family(), "int8", CUDA_BF16, BASE)


# ── routing ──────────────────────────────────────────────────────────────────


def _planner(monkeypatch, *, offloads: bool, eligible: bool):
    from core.inference import video as V

    target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, ordinal = 0)
    monkeypatch.setattr(V, "resolve_diffusion_device_target", lambda **k: target)
    monkeypatch.setattr(V, "_video_auto_denoiser_scheme", lambda fam, **k: "int8")
    monkeypatch.setattr(V, "_auto_seed_yields_to_resident_bf16", lambda *a: False)
    monkeypatch.setattr(V, "_video_seed_stays_resident", lambda fam, **k: not offloads)
    monkeypatch.setattr(hn, "hosted_int8_offload_ok", lambda *a: eligible)
    backend = V.VideoBackend.__new__(V.VideoBackend)
    backend._state = None
    return backend._video_planned_auto_denoiser_scheme(
        _family(), base = BASE, kind = "pipeline", transformer_quant = "int8", speed_mode = None
    )


@pytest.mark.parametrize(
    "offloads, eligible, expect",
    [
        (True, True, "int8"),  # the hosted pair is fetched, the dense experts are not
        (True, False, "__declined__"),  # kill switch / cache hit / other host: main's behaviour
        (False, True, "int8"),  # resident: torchao seed, unchanged
        (False, False, "int8"),
    ],
)
def test_planner_routing(monkeypatch, offloads, eligible, expect):
    from core.inference import video as V
    got = _planner(monkeypatch, offloads = offloads, eligible = eligible)
    assert got == (V.DENOISER_SEED_DECLINED if expect == "__declined__" else expect)


@needs_decoder
def test_pipe_kwargs_seed_native_experts_all_or_nothing(hosted, monkeypatch):
    from core.inference import video_denoiser_prequant as vdp

    sources = {c: _source(c) for c in COMPONENTS}
    monkeypatch.setattr(vdp, "denoiser_prequant_sources", lambda *a, **k: dict(sources))
    seeded = vdp.denoiser_prequant_pipe_kwargs(
        _family(),
        BASE,
        scheme = "int8",
        dtype = torch.bfloat16,
        device = "cuda",
        target = CUDA_BF16,
        native_offload = True,
    )
    assert set(seeded) == set(COMPONENTS)
    assert all(any(is_native_linear(m) for m in mod.modules()) for mod in seeded.values())
    Path(hosted.files["transformer_2"]).write_bytes(b"not a checkpoint")
    assert (
        vdp.denoiser_prequant_pipe_kwargs(
            _family(),
            BASE,
            scheme = "int8",
            dtype = torch.bfloat16,
            device = "cuda",
            target = CUDA_BF16,
            native_offload = True,
        )
        == {}
    )
