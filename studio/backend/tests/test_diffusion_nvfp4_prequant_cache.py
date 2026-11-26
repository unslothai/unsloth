# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The local prequant cache holds whole-model NVFP4 builds: key, round trip, policy families, weight quantiser.

Hermetic: a hand-built bf16 module quantized to NVFP4 on the CPU (torchao's sm100 assert is lifted for the weight
pass only), a base "repo" that is a local directory with one safetensors shard, and a private cache root per test.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("safetensors")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_nvfp4_weight_quant as WQ  # noqa: E402
from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference import prequant_safetensors as PS  # noqa: E402

FAMILY = "tinyfam-nvfp4-cache-test"


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(512, 512)
        self.ff = torch.nn.Linear(512, 1024)
        self.norm = torch.nn.LayerNorm(512)


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(2)])


@pytest.fixture
def env(tmp_path, monkeypatch):
    if PS._torchao_helpers() is None:
        pytest.skip("torchao safetensors helpers unavailable")
    try:
        import torchao.prototype.mx_formats.inference_workflow as iw
    except Exception:  # noqa: BLE001
        pytest.skip("torchao has no NVFP4 inference workflow")
    # The weight pass is plain torch; only the dynamic-activation GEMM needs sm100.
    monkeypatch.setattr(iw, "is_sm_at_least_100", lambda: True, raising = False)
    # _quantize runs the real quantize_transformer, which rewrites NVFP4Tensor.to_nvfp4 for the
    # whole process; pinning the current one here puts torchao's back at teardown.
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    monkeypatch.setattr(NVFP4Tensor, "to_nvfp4", NVFP4Tensor.__dict__["to_nvfp4"])
    root = tmp_path / "cache"
    monkeypatch.setenv(C.ENV_DIR, str(root))
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    monkeypatch.delenv(C.ENV_MODE, raising = False)
    monkeypatch.delenv(C.ENV_MAX_GB, raising = False)
    monkeypatch.delenv(WQ.WEIGHT_QUANT_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH", raising = False)
    C._HEADER_MEMO.clear()
    getattr(C, "_UNREACHABLE", {}).clear()
    return root


def _fam(base: Path, name: str = FAMILY) -> types.SimpleNamespace:
    return types.SimpleNamespace(name = name, transformer_class = "Tiny", base_repo = str(base))


def _write_base(tmp_path: Path, seed: int = 0) -> tuple[Path, Tiny]:
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    model = Tiny().to(torch.bfloat16)
    base = tmp_path / "base"
    (base / "transformer").mkdir(parents = True)
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        str(base / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    return base, model


def _quantize(model: Tiny, monkeypatch) -> Tiny:
    """The real runtime path (quantize_transformer), so the cache sees exactly what a load stamps."""
    import core.inference.diffusion_transformer_quant as Q

    monkeypatch.setattr(Q, "select_transformer_quant_scheme", lambda *a, **k: "nvfp4")
    monkeypatch.setattr(Q, "native_quant_scheme", lambda *a, **k: None)
    pipe = types.SimpleNamespace(transformer = model)
    target = types.SimpleNamespace(dtype = torch.bfloat16, device = "cpu")
    assert Q.quantize_transformer(pipe, target, mode = "nvfp4", family = FAMILY) == "nvfp4"
    assert type(model.blocks[0].to_q.weight).__name__ == "NVFP4Tensor"
    return model


def _flat_bytes(state_dict: dict) -> dict:
    flat, _ = C._flatten(state_dict)
    out = {}
    for k, v in flat.items():
        v = v.tensor() if isinstance(v, C._FlatRef) else v
        out[k] = (v.dtype, tuple(v.shape), v.reshape(-1).contiguous().view(torch.uint8).clone())
    return out


def _same_bytes(a: dict, b: dict) -> list:
    fa, fb = _flat_bytes(a), _flat_bytes(b)
    assert set(fa) == set(fb)
    return [k for k in fa if fa[k][:2] != fb[k][:2] or not torch.equal(fa[k][2], fb[k][2])]


def test_quantize_stamps_the_weight_quantiser(env, tmp_path, monkeypatch):
    _base, model = _write_base(tmp_path)
    _quantize(model, monkeypatch)
    info = model._unsloth_runtime_quant_info
    assert info["scheme"] == "nvfp4" and info["require_divisible"] == 16
    assert info["nvfp4_weight_quant"] == WQ.resolve_method() == WQ.DEFAULT_METHOD


def test_a_whole_model_nvfp4_build_round_trips_bit_identically(env, tmp_path, monkeypatch):
    base, model = _write_base(tmp_path)
    _quantize(model, monkeypatch)
    fam = _fam(base)
    path = C.save_runtime_quant(model, fam, base_repo = str(base), nvfp4_readback = True)
    assert path is not None and Path(path).is_file()
    assert C.lookup(fam, "nvfp4", base_repo = str(base), quant_device = "cpu") == path

    ckpt = PS.load_prequant_safetensors(path, mmap = True)
    assert _same_bytes(model.state_dict(), ckpt["state_dict"]) == []
    meta = ckpt["metadata"]
    assert meta["scheme"] == "nvfp4" and meta["family"] == FAMILY
    assert meta["nvfp4_weight_quant"] == WQ.DEFAULT_METHOD
    assert meta["nvfp4_quantizer"] == WQ.QUANTIZER_VERSION

    from core.inference.diffusion_prequant import _validate_checkpoint

    assert _validate_checkpoint(
        ckpt, "nvfp4", str(base), None, min_features = 512, component = "transformer"
    )


def test_a_returning_load_reads_the_nvfp4_build_into_the_dense_module(env, tmp_path, monkeypatch):
    """The cached-load path a second Images load takes: the dense module gets the first load's exact bytes."""
    from core.inference import diffusion as D

    base, model = _write_base(tmp_path)
    fresh = Tiny().to(torch.bfloat16)
    fresh.load_state_dict(model.state_dict())
    _quantize(model, monkeypatch)
    fam = _fam(base)
    assert C.save_runtime_quant(model, fam, base_repo = str(base), nvfp4_readback = True) is not None

    monkeypatch.setattr(D, "select_transformer_quant_scheme", lambda *a, **k: "nvfp4")
    monkeypatch.setattr(D, "native_quant_scheme", lambda *a, **k: None)
    pipe = types.SimpleNamespace(transformer = fresh)
    got = D.DiffusionBackend._pipeline_quant_from_cache(
        pipe,
        fam,
        [("transformer", fresh)],
        target = types.SimpleNamespace(dtype = torch.bfloat16, device = "cpu"),
        transformer_quant = "nvfp4",
        fast_accum = None,
        loras = [],
        base = str(base),
        fetch_base = str(base),
        source_base = str(base),
        hf_token = None,
    )
    assert got == "nvfp4"
    assert pipe.transformer is fresh and fresh._unsloth_runtime_quant == "nvfp4"
    assert type(fresh.blocks[1].ff.weight).__name__ == "NVFP4Tensor"
    assert _same_bytes(model.state_dict(), fresh.state_dict()) == []


def test_the_key_carries_the_weight_quantiser_and_its_version(env, tmp_path, monkeypatch):
    base, model = _write_base(tmp_path)
    _quantize(model, monkeypatch)
    fam = _fam(base)
    assert C.save_runtime_quant(model, fam, base_repo = str(base), nvfp4_readback = True)
    assert C.lookup(fam, "nvfp4", base_repo = str(base), quant_device = "cpu")

    monkeypatch.setenv(WQ.WEIGHT_QUANT_ENV, "max6")
    assert C.lookup(fam, "nvfp4", base_repo = str(base), quant_device = "cpu") is None
    # A build stamped with one quantiser is never filed under the other.
    assert C.save_runtime_quant(model, fam, base_repo = str(base), nvfp4_readback = True) is None
    monkeypatch.delenv(WQ.WEIGHT_QUANT_ENV)
    assert C.lookup(fam, "nvfp4", base_repo = str(base), quant_device = "cpu")

    monkeypatch.setattr(WQ, "QUANTIZER_VERSION", WQ.QUANTIZER_VERSION + 1)
    assert C.lookup(fam, "nvfp4", base_repo = str(base), quant_device = "cpu") is None


def test_int8_and_fp8_keys_are_unchanged():
    parts = C.key_parts(
        family = FAMILY,
        transformer_class = "Tiny",
        scheme = "fp8",
        component = "transformer",
        source_fp = "x",
    )
    assert not any(k.startswith("nvfp4") for k in parts)
    parts = C.key_parts(
        family = FAMILY,
        transformer_class = "Tiny",
        scheme = "nvfp4",
        component = "transformer",
        source_fp = "x",
    )
    assert parts["nvfp4_weight_quant"] and parts["nvfp4_quantizer"] == WQ.QUANTIZER_VERSION


def test_policy_families_and_the_kill_switch_never_key(env, tmp_path, monkeypatch):
    base, _model = _write_base(tmp_path)
    assert C._key_for(_fam(base), "nvfp4", str(base), "transformer", quant_device = "cpu") is not None
    # z-image's base takes a per-layer nvfp4 / fp8 / bf16 policy: that build is not the cached whole-model one.
    policy = _fam(base, "z-image")
    assert (
        C._key_for(
            policy,
            "nvfp4",
            str(base),
            "transformer",
            base_id = "Tongyi-MAI/Z-Image-Turbo",
            quant_device = "cpu",
        )
        is None
    )
    assert (
        C._key_for(policy, "fp8", str(base), "transformer", base_id = "Tongyi-MAI/Z-Image-Turbo")
        is not None
    )
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "0")
    assert C._key_for(_fam(base), "nvfp4", str(base), "transformer", quant_device = "cpu") is None


def test_mxfp8_is_still_never_cached(env, tmp_path):
    base, _ = _write_base(tmp_path)
    assert "mxfp8" not in C.CACHEABLE_SCHEMES
    assert C.lookup(_fam(base), "mxfp8", base_repo = str(base)) is None


def test_quantize_stamps_where_the_weights_were_quantized(env, tmp_path, monkeypatch):
    _base, model = _write_base(tmp_path)
    _quantize(model, monkeypatch)
    assert model._unsloth_runtime_quant_info["nvfp4_quant_device"] == "cpu"


def test_the_key_carries_the_quantize_device_and_a_split_build_is_not_saved(
    env, tmp_path, monkeypatch
):
    """4over6 compares fp32 reductions, so a CPU build and a CUDA build of the same weights differ in a few blocks:
    a hit must come from a quantize on the device type this load would quantize on."""
    base, model = _write_base(tmp_path)
    _quantize(model, monkeypatch)
    fam = _fam(base)
    info = model._unsloth_runtime_quant_info
    info["nvfp4_quant_device"] = None  # weights split across CPU and CUDA
    assert C.save_runtime_quant(model, fam, base_repo = str(base), nvfp4_readback = True) is None
    info["nvfp4_quant_device"] = "cpu"
    # A load path that never reads nvfp4 back does not write it either.
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None
    assert C.save_runtime_quant(model, fam, base_repo = str(base), nvfp4_readback = True)
    assert C.lookup(fam, "nvfp4", base_repo = str(base), quant_device = "cpu")
    assert C.lookup(fam, "nvfp4", base_repo = str(base), quant_device = "cuda") is None
    assert C.lookup(fam, "nvfp4", base_repo = str(base)) is None
    # int8 / fp8 keys never see the device.
    assert C._key_for(fam, "fp8", str(base), "transformer") is not None


def test_a_build_without_baked_activation_scales_skips_the_flashinfer_preflight(monkeypatch):
    """Only baked activation scales move to flashinfer; a runtime build (what the cache holds) never probes it."""
    from core.inference import diffusion_nvfp4_ops as ops
    from core.inference import diffusion_prequant as P

    calls = []
    monkeypatch.setattr(
        ops, "select_nvfp4_backend", lambda device = None: calls.append(device) or "torchao"
    )
    assert P._engage_nvfp4_backend(Tiny(), {"scheme": "nvfp4"}, "cuda") == 0
    assert calls == []
