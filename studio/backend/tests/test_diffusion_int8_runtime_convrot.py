# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime ConvRot on the int8 quantize path (Qwen-Image-2.1): which Linears rotate, that the dense model is
unchanged by it, and that it happens before quantize_ and only for the families that declare it."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn

import core.inference.diffusion_transformer_quant as tq
from core.inference.diffusion_convrot import CONVROT_ATTR, is_rotated_linear


class _Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias = False)])

    def forward(self, x):
        return self.to_out[0](self.to_q(x) * self.to_k(x).sigmoid() + self.to_v(x))


class _Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_layer = nn.Linear(dim, 3 * dim, bias = False)
        self.out = nn.Linear(3 * dim, dim, bias = False)

    def forward(self, x):
        return self.out(torch.nn.functional.silu(self.gate_layer(x)))


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _Attn(dim)
        self.img_mlp = _Mlp(dim)
        self.extra = nn.Linear(
            dim, dim, bias = False
        )  # quantized, but not in the family's rotation spec

    def forward(self, x):
        x = x + self.attn(x)
        return self.extra(x + self.img_mlp(x))


class _Tiny(nn.Module):
    def __init__(self, dim = 512):
        super().__init__()
        self.txt_in = nn.Linear(dim, dim, bias = False)
        self.small = nn.Linear(dim, 64, bias = False)
        self.odd = nn.Linear(640, dim, bias = False)
        self.transformer_blocks = nn.ModuleList([_Block(dim) for _ in range(2)])

    def forward(self, x, y):
        h = self.txt_in(x) + self.odd(y)
        for b in self.transformer_blocks:
            h = b(h)
        return h, self.small(h)


_ROTATED = {
    f"transformer_blocks.{i}.{n}"
    for i in range(2)
    for n in (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "img_mlp.gate_layer",
        "img_mlp.out",
    )
}


def _filter(family):
    return tq.make_filter_fn(
        512, exclude_name_tokens = tq.exclude_tokens_for_scheme(tq.TQ_INT8, family) + ("lora_",)
    )


def test_convrot_spec_only_for_int8_on_declared_families():
    group, suffixes = tq.convrot_spec_for_scheme(tq.TQ_INT8, "qwen-image-2.1")
    assert group == 256 and set(suffixes) == {
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "img_mlp.gate_layer",
        "img_mlp.proj",
        "img_mlp.out",
    }
    assert tq.convrot_spec_for_scheme(tq.TQ_INT8, " Qwen-Image-2.1 ")[0] == 256
    for scheme in (tq.TQ_FP8, tq.TQ_NVFP4, tq.TQ_MXFP8):
        assert tq.convrot_spec_for_scheme(scheme, "qwen-image-2.1") == (0, ())
    for family in (None, "qwen-image", "qwen-image-edit", "flux.1", "minimax-h3"):
        assert tq.convrot_spec_for_scheme(tq.TQ_INT8, family) == (0, ())


def test_int8_artifact_names_the_rotated_build_first_and_keeps_the_plain_one():
    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_prequant import candidate_filenames_of, resolve_prequant_source

    fam = detect_family("Qwen/Qwen-Image-2.1", override = "qwen-image-2.1")
    names = candidate_filenames_of(resolve_prequant_source(fam, "int8"))
    assert names[:2] == (
        "Qwen-Image-2.1-INT8-ConvRot.safetensors",
        "Qwen-Image-2.1-INT8.safetensors",
    )
    assert (
        candidate_filenames_of(resolve_prequant_source(fam, "fp8"))[0]
        == "Qwen-Image-2.1-FP8.safetensors"
    )


def test_convrot_does_not_touch_the_exclusion_set():
    # the published plain INT8 artifact validates against exclude_tokens_for_scheme; rotation must not move it
    assert tq.exclude_tokens_for_scheme(
        tq.TQ_INT8, "qwen-image-2.1"
    ) == tq._INT8_EXCLUDE_NAME_TOKENS + ("txt_in",)


def test_runtime_convrot_rotates_the_quantized_set_and_keeps_the_model_exact():
    torch.manual_seed(0)
    model = _Tiny().float()
    x, y = torch.randn(4, 512), torch.randn(4, 640)
    with torch.no_grad():
        ref = model(x, y)
        rotated = tq.apply_runtime_convrot(
            model, tq.TQ_INT8, "qwen-image-2.1", _filter("qwen-image-2.1")
        )
        got = model(x, y)
    assert set(rotated) == _ROTATED
    assert not is_rotated_linear(model.txt_in)  # family exclusion
    assert not is_rotated_linear(model.small)  # below min_features
    assert not is_rotated_linear(model.odd)  # 640 not divisible by 256
    assert not any(is_rotated_linear(blk.extra) for blk in model.transformer_blocks)
    assert getattr(model, CONVROT_ATTR)["linears"] == 12
    for a, b in zip(ref, got):
        torch.testing.assert_close(a, b, rtol = 1e-4, atol = 1e-4)


def test_runtime_convrot_is_inert_elsewhere():
    model = _Tiny()
    assert tq.apply_runtime_convrot(model, tq.TQ_INT8, "qwen-image", _filter("qwen-image")) == ()
    assert (
        tq.apply_runtime_convrot(model, tq.TQ_FP8, "qwen-image-2.1", _filter("qwen-image-2.1"))
        == ()
    )
    assert not any(is_rotated_linear(m) for m in model.modules())
    assert not hasattr(model, CONVROT_ATTR)


def _stub_torchao(monkeypatch, seen):
    tqz = types.ModuleType("torchao.quantization")

    def quantize_(
        module,
        config,
        filter_fn = None,
    ):
        seen.append(
            sorted(n for n, m in module.named_modules() if is_rotated_linear(m) and filter_fn(m, n))
        )

    tqz.quantize_ = quantize_
    tqz.Int8DynamicActivationInt8WeightConfig = lambda **kw: "int8-cfg"
    tqz.Float8DynamicActivationFloat8WeightConfig = lambda **kw: "fp8-cfg"
    tqz.PerRow = lambda: "per-row"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    monkeypatch.setattr(tq, "_make_quant_config", lambda scheme, fast_accum = None: f"{scheme}-cfg")


@pytest.mark.parametrize(
    "family, expect_rotated", [("qwen-image-2.1", 12), ("qwen-image", 0), (None, 0)]
)
def test_quantize_transformer_rotates_before_quantize(monkeypatch, family, expect_rotated):
    seen = []
    _stub_torchao(monkeypatch, seen)
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda target, mode, family = None: tq.TQ_INT8
    )
    pipe = types.SimpleNamespace(transformer = _Tiny())
    assert tq.quantize_transformer(pipe, object(), mode = "int8", family = family) == tq.TQ_INT8
    assert len(seen) == 1 and len(seen[0]) == expect_rotated


def test_quantize_transformer_leaves_fp8_unrotated(monkeypatch):
    seen = []
    _stub_torchao(monkeypatch, seen)
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda target, mode, family = None: tq.TQ_FP8
    )
    pipe = types.SimpleNamespace(transformer = _Tiny())
    assert tq.quantize_transformer(pipe, object(), mode = "fp8", family = "qwen-image-2.1") == tq.TQ_FP8
    assert seen == [[]]


def test_runtime_convrot_warms_the_hadamard_for_the_target_device():
    # the forward looks the matrix up in a module-level cache; if the first compile has to build it, dynamo guards on
    # its absence and the block recompiles on the second call
    from core.inference import diffusion_convrot as cr

    cr._HADAMARD_CACHE.clear()
    model = _Tiny()
    # the indexed torch_device wins over the bare "cuda" a DiffusionDeviceTarget keeps in .device
    target = types.SimpleNamespace(device = "cuda", torch_device = "cpu", dtype = torch.bfloat16)
    tq.apply_runtime_convrot(
        model, tq.TQ_INT8, "qwen-image-2.1", _filter("qwen-image-2.1"), target = target
    )
    assert (256, "cpu", torch.bfloat16) in cr._HADAMARD_CACHE


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "compiles a CUDA graph")
def test_runtime_convrot_adds_no_recompile():
    from torch._dynamo.utils import counters

    from core.inference import diffusion_convrot as cr

    cr._HADAMARD_CACHE.clear()
    torch._dynamo.reset()
    counters.clear()
    model = _Tiny().to(torch.bfloat16)
    target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    tq.apply_runtime_convrot(
        model, tq.TQ_INT8, "qwen-image-2.1", _filter("qwen-image-2.1"), target = target
    )
    block = torch.compile(model.cuda().transformer_blocks[0], fullgraph = True)
    x = torch.randn(4, 512, device = "cuda", dtype = torch.bfloat16)
    for _ in range(3):
        block(x)
    assert counters["stats"]["unique_graphs"] == 1


def test_unreachable_hub_still_loads_the_plain_artifact_already_cached(monkeypatch, tmp_path):
    # the ConvRot name now comes first; a user holding only the plain file from before must keep it offline
    from huggingface_hub.errors import LocalEntryNotFoundError

    from core.inference import diffusion_prequant as pq
    from core.inference.diffusion_families import detect_family

    plain = tmp_path / "Qwen-Image-2.1-INT8.safetensors"
    plain.write_bytes(b"weights")
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir = None, **k: str(tmp_path / filename)
        if (tmp_path / filename).is_file()
        else None,
    )
    asked = []

    def _dl(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
        local_files_only = False,
    ):
        asked.append(filename)
        raise LocalEntryNotFoundError("connection error")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _dl)
    source = pq.resolve_prequant_source(
        detect_family("Qwen/Qwen-Image-2.1", override = "qwen-image-2.1"), "int8"
    )
    assert pq._resolve_checkpoint_path(source, None, None) == str(plain)
    assert asked == ["Qwen-Image-2.1-INT8-ConvRot.safetensors"]
    plain.unlink()
    with pytest.raises(
        LocalEntryNotFoundError
    ):  # nothing cached: the connection error is still the answer
        pq._resolve_checkpoint_path(source, None, None)


def test_builder_publishes_rotated_and_plain_int8_under_different_names():
    import importlib.util
    from pathlib import Path

    from core.inference.diffusion_families import detect_family

    script = Path(__file__).resolve().parents[3] / "scripts" / "build_prequant_checkpoint.py"
    spec = importlib.util.spec_from_file_location("_build_prequant_for_convrot_test", script)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    fam = detect_family("Qwen/Qwen-Image-2.1", override = "qwen-image-2.1")
    repo = "unsloth/Qwen-Image-2.1-FP8"
    dest = lambda rotated: build.upload_destination(
        fam, "int8", rotated = rotated, safetensors = True, upload_repo = repo
    )
    assert dest(True) == "Qwen-Image-2.1-INT8-ConvRot.safetensors"
    assert dest(False) == "Qwen-Image-2.1-INT8.safetensors"
    assert (
        build.upload_destination(fam, "fp8", rotated = False, safetensors = True, upload_repo = repo)
        == "Qwen-Image-2.1-FP8.safetensors"
    )
