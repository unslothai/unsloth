# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An offloaded Wan2.2-T2V-A14B int8 load reads the hosted torchao W8A8 pickle pair as native Linears through
main's hosted reader (``diffusion_native_hosted``) instead of fetching and quantising both dense experts."""

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
from core.inference import diffusion_native_hosted as NH  # noqa: E402
from core.inference import video as V  # noqa: E402
from core.inference.diffusion_native_cache import native_spec  # noqa: E402
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


def _spec():
    return native_spec("int8", FAMILY, act_int8 = True)


@pytest.fixture
def hosted(tmp_path):
    files, written, dense = {}, {}, {}
    cfg = tmp_path / "base"
    for seed, component in enumerate(COMPONENTS):
        dense[component] = dense_expert(seed)
        dense[component].save_config(str(cfg / component))
        path = tmp_path / f"{component}.pt"
        written[component] = write_hosted_v1(dense[component], path, component = component)
        files[component] = str(path)
    return types.SimpleNamespace(files = files, written = written, dense = dense, cfg = str(cfg))


def _load(
    hosted,
    component,
    *,
    base = BASE,
    spec = None,
):
    return NH.load_native_from_hosted(
        hosted.files[component],
        diffusers.WanTransformer3DModel,
        hosted.cfg,
        spec = spec or _spec(),
        family = FAMILY,
        component = component,
        base = base,
    )


@needs_decoder
@pytest.mark.parametrize("component", COMPONENTS)
def test_hosted_codes_and_scales_are_held_bit_for_bit(hosted, component):
    model = _load(hosted, component)
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
    a, b = _load(hosted, "transformer"), _load(hosted, "transformer_2")
    qa = dict(a.named_buffers())["blocks.0.attn1.to_q.weight_q"]
    qb = dict(b.named_buffers())["blocks.0.attn1.to_q.weight_q"]
    assert not torch.equal(qa, qb)


@needs_decoder
def test_forward_tracks_the_dense_expert(hosted):
    model = _load(hosted, "transformer")
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
def test_a_rotated_pickle_is_refused(hosted):
    from core.inference.diffusion_convrot import CONVROT_KIND, ROTATION_KEY
    write_hosted_v1(
        hosted.dense["transformer"],
        Path(hosted.files["transformer"]),
        component = "transformer",
        metadata = {ROTATION_KEY: CONVROT_KIND},
    )
    assert _load(hosted, "transformer") is None


@needs_decoder
def test_a_wrong_base_or_component_is_refused(hosted):
    path = Path(hosted.files["transformer"])
    write_hosted_v1(hosted.dense["transformer"], path, component = "transformer", base = "someone/else")
    assert _load(hosted, "transformer") is None
    write_hosted_v1(hosted.dense["transformer"], path, component = "transformer_2")
    assert _load(hosted, "transformer") is None


@needs_decoder
def test_an_asymmetric_weight_is_refused(hosted):
    path = Path(hosted.files["transformer"])
    n = hosted.dense["transformer"].blocks[0].attn1.to_q.out_features
    write_hosted_v1(
        hosted.dense["transformer"],
        path,
        component = "transformer",
        zero_point = torch.ones(n, dtype = torch.int8),
    )
    assert _load(hosted, "transformer") is None


@needs_decoder
def test_a_layer_set_other_than_the_native_builds_is_refused(hosted):
    spec = dict(_spec())
    spec["exclude_name_tokens"] = sorted(set(spec["exclude_name_tokens"]) | {"ffn"})
    assert _load(hosted, "transformer", spec = spec) is None


@needs_decoder
def test_a_weight_only_native_build_or_stubbed_torchao_is_refused(hosted, monkeypatch):
    assert _load(hosted, "transformer", spec = native_spec("int8", FAMILY, act_int8 = False)) is None
    import core._torchao_stub as stub

    monkeypatch.setattr(stub, "is_stubbed", lambda name: True)
    assert _load(hosted, "transformer") is None


needs_v1_classes = pytest.mark.skipif(
    li.legacy_int8_classes_missing(),
    reason = "needs torchao <= 0.17 (the v1 int8 classes)",
)


@needs_v1_classes
def test_a_real_torchao_v1_pickle_reads_natively(tmp_path):
    """torchao <= 0.17 unpickles the hosted file as real v1 tensors: the same codes and scales come out."""
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    dense = dense_expert(0)
    dense.save_config(str(tmp_path / "base" / "transformer"))
    quantised = dense_expert(0)
    keep = tq.make_filter_fn(
        tq.DEFAULT_MIN_LINEAR_FEATURES,
        exclude_name_tokens = tq.exclude_tokens_for_scheme("int8", FAMILY),
    )
    quantize_(quantised, Int8DynamicActivationInt8WeightConfig(version = 1), filter_fn = keep)
    state = {k: v for k, v in quantised.state_dict().items()}
    meta = {
        "scheme": "int8",
        "base_model_id": BASE,
        "family": FAMILY,
        "min_features": tq.DEFAULT_MIN_LINEAR_FEATURES,
        "component": "transformer",
    }
    path = tmp_path / "transformer.pt"
    torch.save({"format": pq.PREQUANT_FORMAT, "metadata": meta, "state_dict": state}, str(path))
    model = NH.load_native_from_hosted(
        str(path),
        diffusers.WanTransformer3DModel,
        str(tmp_path / "base"),
        spec = _spec(),
        family = FAMILY,
        component = "transformer",
        base = BASE,
    )
    assert model is not None
    layers = {n: m for n, m in model.named_modules() if is_native_linear(m)}
    assert layers
    for fqn, layer in layers.items():
        impl = state[fqn + ".weight"].original_weight_tensor.tensor_impl
        assert layer.rot_group == 0 and layer.act_int8
        assert torch.equal(layer.weight_q, impl.int_data)
        assert torch.equal(layer.weight_scale.view(torch.float32), impl.scale.float().reshape(-1))


@needs_decoder
def test_the_pickle_is_mapped_not_read_into_memory(hosted, monkeypatch):
    """Both experts read in full would sit in anonymous host memory (~29 GB) before offload pins its copies."""
    seen = []
    real = pq._load_prequant_checkpoint

    def spy(path, **kwargs):
        seen.append(bool(kwargs.get("mmap")))
        return real(path, **kwargs)

    monkeypatch.setattr(pq, "_load_prequant_checkpoint", spy)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv(pq._PREQUANT_MMAP_ENV, raising = False)
    model = _load(hosted, "transformer")
    assert model is not None and seen and seen[0] is True
    codes, _scale = hosted.written["transformer"]["blocks.0.attn1.to_q"]
    assert torch.equal(dict(model.named_buffers())["blocks.0.attn1.to_q.weight_q"], codes)


@needs_decoder
def test_the_exports_fp32_modules_are_kept_and_other_fp32_is_refused(hosted):
    """The real hosted A14B pickle stores time_embedder, scale_shift_table and norm2 in fp32 (125 tensors)."""
    dense = hosted.dense["transformer"]
    path = Path(hosted.files["transformer"])
    for name, p in dense.named_parameters():
        if {"time_embedder", "scale_shift_table", "norm2"} & set(name.split(".")):
            p.data = p.data.float()
    write_hosted_v1(dense, path, component = "transformer")
    model = _load(hosted, "transformer")
    assert model is not None
    assert model.scale_shift_table.dtype == torch.float32
    assert model.condition_embedder.time_embedder.linear_1.weight.dtype == torch.float32
    assert model.patch_embedding.weight.dtype == torch.bfloat16
    dense.patch_embedding.weight.data = dense.patch_embedding.weight.data.float()
    write_hosted_v1(dense, path, component = "transformer")
    assert _load(hosted, "transformer") is None


def test_an_unreadable_pickle_never_raises(tmp_path):
    bad = tmp_path / "x.pt"
    bad.write_bytes(b"not a checkpoint")
    assert (
        NH.load_native_from_hosted(
            str(bad),
            diffusers.WanTransformer3DModel,
            str(tmp_path),
            spec = _spec(),
            family = FAMILY,
            component = "transformer",
            base = BASE,
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
    assert NH.int8_weight_parts("w", torch.ones(2, 2)) is None
    codes, scale = NH.int8_weight_parts("w", _int8_tensor())
    assert codes.dtype == torch.int8 and scale.shape == (8,)
    for extra in (
        {"act_quant_kwargs": None},
        {"act_pre_scale": torch.ones(32)},
        {"act_quant_scale": torch.ones(1)},
        {"zero_point": torch.ones(8, 1, dtype = torch.int8)},
        {"block_size": [1, 16]},
    ):
        with pytest.raises(ValueError):
            NH.int8_weight_parts("w", _int8_tensor(**extra))


@needs_decoder
def test_the_loader_reads_both_experts_all_or_nothing(hosted, monkeypatch):
    from core.inference import video_denoiser_prequant as vdp

    sources = {c: _source(c) for c in COMPONENTS}
    monkeypatch.setattr(vdp, "denoiser_prequant_sources", lambda *a, **k: dict(sources))
    monkeypatch.setattr(
        pq, "_resolve_checkpoint_path", lambda src, *a, **k: hosted.files[src.filename]
    )
    out = V._read_native_quant_from_hosted(
        _family(), _spec(), base = BASE, source = hosted.cfg, config_kwargs = {}
    )
    assert set(out) == set(COMPONENTS)
    assert all(any(is_native_linear(m) for m in mod.modules()) for mod in out.values())
    Path(hosted.files["transformer_2"]).write_bytes(b"not a checkpoint")
    assert (
        V._read_native_quant_from_hosted(
            _family(), _spec(), base = BASE, source = hosted.cfg, config_kwargs = {}
        )
        == {}
    )


def _source(component):
    return pq.PrequantSource(
        kind = "repo", location = "unsloth/Wan2.2-T2V-A14B-FP8", filename = component
    )


@pytest.fixture
def nvidia(monkeypatch):
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.delenv(NH.NATIVE_HOSTED_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_NATIVE_INT8_ACT", raising = False)


CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


def test_a14b_opts_in_as_a_pickle_family():
    fam = _family()
    assert NH.native_hosted_family(fam) and NH.native_hosted_pickle(fam)
    ltx = detect_video_family("Lightricks/LTX-2.5-Diffusers", override = "ltx-2.5")
    assert ltx is not None and NH.native_hosted_family(ltx) and not NH.native_hosted_pickle(ltx)
    ti2v = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers", override = "wan2.2-ti2v-5b")
    assert ti2v is not None and not NH.native_hosted_family(ti2v)


@pytest.mark.parametrize(
    "scheme, target, env, act, expect",
    [
        ("int8", CUDA_BF16, None, None, "int8"),
        ("int8", CUDA_BF16, "0", None, None),
        ("int8", CUDA_BF16, None, "0", None),
        ("fp8", CUDA_BF16, None, None, None),
        ("nvfp4", CUDA_BF16, None, None, None),
        ("int8", types.SimpleNamespace(device = "cuda", dtype = torch.float16), None, None, None),
        ("int8", types.SimpleNamespace(device = "mps", dtype = torch.bfloat16), None, None, None),
    ],
)
def test_the_hosted_scheme_gate(nvidia, monkeypatch, scheme, target, env, act, expect):
    if env is not None:
        monkeypatch.setenv(NH.NATIVE_HOSTED_ENV, env)
    if act is not None:
        monkeypatch.setenv("UNSLOTH_NATIVE_INT8_ACT", act)
    assert V._video_native_hosted_scheme(_family(), scheme, target) == expect


def test_convrot_off_keeps_the_unrotated_pickle_read_but_not_ltx(nvidia, monkeypatch):
    from core.inference.diffusion_native_quant import NATIVE_INT8_ROT_ENV

    monkeypatch.setenv(NATIVE_INT8_ROT_ENV, "0")
    assert V._video_native_hosted_scheme(_family(), "int8", CUDA_BF16) == "int8"
    ltx = detect_video_family("Lightricks/LTX-2.5-Diffusers", override = "ltx-2.5")
    assert V._video_native_hosted_scheme(ltx, "int8", CUDA_BF16) is None
    monkeypatch.delenv(NATIVE_INT8_ROT_ENV)
    assert V._video_native_hosted_scheme(ltx, "int8", CUDA_BF16) == "int8"


@pytest.mark.parametrize("rocm, stubbed", [(True, False), (False, True)])
@pytest.mark.parametrize("act", [None, "1"])
def test_rocm_and_stubbed_torchao_do_not_read_the_pickle(monkeypatch, rocm, stubbed, act):
    # native_quant_host accepts these hosts (UNSLOTH_NATIVE_INT8_ACT=1 makes them W8A8); the pickle needs torchao.
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: rocm)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: stubbed)
    monkeypatch.delenv(NH.NATIVE_HOSTED_ENV, raising = False)
    if act is None:
        monkeypatch.delenv("UNSLOTH_NATIVE_INT8_ACT", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_NATIVE_INT8_ACT", act)
    assert tq.native_quant_host(CUDA_BF16)
    assert V._video_native_hosted_scheme(_family(), "int8", CUDA_BF16) is None


@pytest.mark.parametrize(
    "offloads, expect, hosted", [(True, "declined", ["int8"]), (False, "int8", [])]
)
def test_planner_stages_the_hosted_pair_under_offload(
    nvidia, monkeypatch, offloads, expect, hosted
):
    target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, ordinal = 0)
    monkeypatch.setattr(V, "resolve_diffusion_device_target", lambda **k: target)
    monkeypatch.setattr(V, "_video_auto_denoiser_scheme", lambda fam, **k: "int8")
    monkeypatch.setattr(V, "_auto_seed_yields_to_resident_bf16", lambda *a: False)
    monkeypatch.setattr(V, "_video_seed_stays_resident", lambda fam, **k: not offloads)
    backend = V.VideoBackend.__new__(V.VideoBackend)
    backend._state = None
    out: list = []
    got = backend._video_planned_auto_denoiser_scheme(
        _family(),
        base = BASE,
        kind = "pipeline",
        transformer_quant = "int8",
        speed_mode = None,
        native_hosted = out,
    )
    assert got == (V.DENOISER_SEED_DECLINED if expect == "declined" else expect)
    assert out == hosted
