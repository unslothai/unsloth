# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVFP4 image builds in the local pre-quant cache: a cached NVFP4 load used to quantize the dense transformer again
on every load (no entry was ever written). Image families only, plain builds only; the key carries the weight
quantiser and the per-layer policy. The CPU tests hold plain bf16 tensors (the cache never looks inside them); the
CUDA test needs an sm100+ card and checks a hit is bit-identical to a fresh quantize, weights and output."""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("safetensors")

BACKEND = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for p in (BACKEND, TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference import prequant_safetensors as PS  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402
from core.inference.diffusion_nvfp4_weight_quant import WEIGHT_QUANT_ENV  # noqa: E402
from core.inference.diffusion_transformer_quant import exclude_tokens_for_scheme  # noqa: E402

from test_diffusion_prequant_cache import Tiny, _write_base, env  # noqa: E402,F401

NVFP4 = "nvfp4"


@pytest.fixture(autouse = True)
def _nvfp4_on(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    monkeypatch.delenv(WEIGHT_QUANT_ENV, raising = False)


def _klein(base: Path):
    """The real FLUX.2 klein image family (no NVFP4 policy), pointed at a local base directory."""
    fam = detect_family("black-forest-labs/FLUX.2-klein-4B")
    assert fam is not None and fam.name == "flux.2-klein"
    return dataclasses.replace(fam, transformer_class = "Tiny", base_repo = str(base))


def _stamp(
    model,
    family: str,
    method: str = "4over6",
):
    exclude = exclude_tokens_for_scheme(NVFP4, family) + ("lora_",)
    model._unsloth_runtime_quant_info = {
        "scheme": NVFP4,
        "family": family,
        "min_features": 512,
        "exclude_name_tokens": list(exclude),
        "require_bf16": False,
        "require_divisible": 16,
        "fast_accum": None,
        "convrot_group": None,
        "convrot_fqns": [],
        "dtype": str(torch.bfloat16),
        "nvfp4_weight_quant": method,
    }
    return model


def test_an_image_nvfp4_build_is_saved_and_read_back(env, tmp_path):
    base, model = _write_base(tmp_path)
    fam = _klein(base)
    _stamp(model, fam.name)
    path = C.save_runtime_quant(model, fam, base_repo = str(base))
    assert path is not None and Path(path).is_file()
    assert C.lookup(fam, NVFP4, base_repo = str(base)) == path
    meta = PS.load_prequant_safetensors(path, mmap = True)["metadata"]
    assert meta["scheme"] == NVFP4 and meta["nvfp4_weight_quant"] == "4over6"
    assert meta["family"] == "flux.2-klein"

    from core.inference.diffusion_prequant import _validate_checkpoint

    ckpt = PS.load_prequant_safetensors(path, mmap = True)
    assert _validate_checkpoint(
        ckpt, NVFP4, str(base), None, min_features = 512, component = "transformer"
    )


def test_the_weight_quantiser_is_part_of_the_key(env, tmp_path, monkeypatch):
    base, model = _write_base(tmp_path)
    fam = _klein(base)
    assert C.save_runtime_quant(_stamp(model, fam.name), fam, base_repo = str(base))
    assert C.lookup(fam, NVFP4, base_repo = str(base))
    monkeypatch.setenv(WEIGHT_QUANT_ENV, "max6")
    assert C.lookup(fam, NVFP4, base_repo = str(base)) is None
    # A build quantized with another method than the key describes is never written under it.
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None
    monkeypatch.delenv(WEIGHT_QUANT_ENV)
    assert C.lookup(fam, NVFP4, base_repo = str(base))


def test_a_build_stamped_without_its_quantiser_is_not_saved(env, tmp_path):
    base, model = _write_base(tmp_path)
    fam = _klein(base)
    _stamp(model, fam.name)
    del model._unsloth_runtime_quant_info["nvfp4_weight_quant"]
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None


def test_policy_bases_are_never_cached(env, tmp_path, monkeypatch):
    import core.inference.diffusion_nvfp4_policy as P

    base, model = _write_base(tmp_path)
    fam = _klein(base)
    _stamp(model, fam.name)
    policy = types.SimpleNamespace(policy_id = "test_policy", version = 1)
    monkeypatch.setattr(P, "resolve_policy", lambda family, base_repo = None: policy)
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None
    assert C.lookup(fam, NVFP4, base_repo = str(base)) is None


def test_a_policy_appearing_for_a_cached_base_is_a_miss(env, tmp_path, monkeypatch):
    import core.inference.diffusion_nvfp4_policy as P

    base, model = _write_base(tmp_path)
    fam = _klein(base)
    assert C.save_runtime_quant(_stamp(model, fam.name), fam, base_repo = str(base))
    policy = types.SimpleNamespace(policy_id = "new_policy", version = 1)
    monkeypatch.setattr(P, "resolve_policy", lambda family, base_repo = None: policy)
    assert C.lookup(fam, NVFP4, base_repo = str(base)) is None


def test_video_and_unknown_families_still_do_not_cache_nvfp4(env, tmp_path):
    base, model = _write_base(tmp_path)
    plain = types.SimpleNamespace(
        name = "flux.2-klein", transformer_class = "Tiny", base_repo = str(base)
    )
    _stamp(model, "flux.2-klein")
    assert not C.cacheable_scheme(NVFP4, plain)
    assert C.save_runtime_quant(model, plain, base_repo = str(base)) is None
    assert C.lookup(plain, NVFP4, base_repo = str(base)) is None
    from core.inference.video_families import VideoFamily

    assert not issubclass(VideoFamily, type(_klein(base)))


def test_int8_and_fp8_keys_are_unchanged():
    for scheme in ("int8", "fp8"):
        parts = C.key_parts(
            family = "flux.2-klein",
            transformer_class = "Tiny",
            scheme = scheme,
            component = "transformer",
            source_fp = "x",
            base = "black-forest-labs/FLUX.2-klein-4B",
        )
        assert "nvfp4_weight_quant" not in parts and "nvfp4_policy" not in parts
    assert C.CACHEABLE_SCHEMES == ("int8", "fp8")


def test_quantize_transformer_stamps_the_nvfp4_weight_quantiser(monkeypatch):
    import torchao.quantization as TQ

    import core.inference.diffusion_transformer_quant as Q

    monkeypatch.setattr(Q, "select_transformer_quant_scheme", lambda *a, **k: NVFP4)
    monkeypatch.setattr(Q, "native_quant_scheme", lambda *a, **k: None)
    monkeypatch.setattr(Q, "_make_quant_config", lambda *a, **k: None)
    monkeypatch.setattr(TQ, "quantize_", lambda *a, **k: None)
    monkeypatch.setenv(WEIGHT_QUANT_ENV, "max6")
    model = Tiny().to(torch.bfloat16)
    pipe = types.SimpleNamespace(transformer = model)
    target = types.SimpleNamespace(dtype = torch.bfloat16, device = "cpu")
    assert Q.quantize_transformer(pipe, target, mode = NVFP4, family = "flux.2-klein") == NVFP4
    assert model._unsloth_runtime_quant_info["nvfp4_weight_quant"] == "max6"


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


@pytest.mark.skipif(not _sm100(), reason = "NVFP4 quantize needs an sm100+ CUDA device")
def test_a_cache_hit_is_bit_identical_to_a_fresh_nvfp4_quantize(env, tmp_path):
    import core.inference.diffusion_transformer_quant as Q
    from core.inference.diffusion_prequant import PrequantSource, load_prequantized_transformer

    base, dense = _write_base(tmp_path)
    fam = _klein(base)
    reference = {k: v.clone() for k, v in dense.state_dict().items()}
    target = types.SimpleNamespace(dtype = torch.bfloat16, device = "cuda")

    def fresh():
        m = Tiny().to(torch.bfloat16)
        m.load_state_dict(reference)
        m = m.cuda()
        scheme = Q.quantize_transformer(
            types.SimpleNamespace(transformer = m),
            target,
            mode = NVFP4,
            family = fam.name,
            base_repo = str(base),
        )
        assert scheme == NVFP4
        return m

    built = fresh()
    assert Q.transformer_is_quantised(built)
    path = C.save_runtime_quant(built, fam, base_repo = str(base))
    assert path is not None
    assert C.lookup(fam, NVFP4, base_repo = str(base)) == path

    into = Tiny().to(torch.bfloat16)
    into.load_state_dict(reference)
    into = into.cuda()
    calls = []
    real_quantize = Q.quantize_transformer
    Q.quantize_transformer = lambda *a, **k: calls.append(1) or real_quantize(*a, **k)
    try:
        loaded = load_prequantized_transformer(
            Tiny,
            str(base),
            PrequantSource(kind = "path", location = path, filename = None),
            device = "cuda",
            dtype = torch.bfloat16,
            scheme = NVFP4,
            min_features = 512,
            local_files_only = True,
            family = fam.name,
            into = into,
        )
    finally:
        Q.quantize_transformer = real_quantize
    assert loaded is not None and calls == []

    again = fresh()
    from test_diffusion_prequant_cache import _flat_equal

    assert _flat_equal(again.state_dict(), loaded.state_dict()) == []
    for name, mod in again.named_modules():
        other = loaded.get_submodule(name)
        assert type(other) is type(mod), name
        w = getattr(mod, "weight", None)
        if w is not None:
            assert type(getattr(other, "weight", None)) is type(w), name

    torch.manual_seed(0)
    x = torch.randn(4, 77, 512, device = "cuda", dtype = torch.bfloat16)

    def run(m):
        out = x
        with torch.no_grad():
            for block in m.blocks:
                out = block.norm(block.to_q(out))
            return block.ff(out)

    a, b = run(again), run(loaded)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
