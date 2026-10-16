# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image families default to a hosted pre-quantized text encoder; ``none`` still pins the bf16 one.

The scheme per family is the ComfyUI template's where ours holds up against the bf16 encoder, else the best hosted
scheme that does (measured numbers in each family's comment). FLUX.2 klein, HiDream-I1 and Ideogram 4 keep bf16 by
default (nothing hosted came close enough), SDXL's CLIPs are too small to bother.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_te_prequant as tpq  # noqa: E402
from core.inference.diffusion_families import _FAMILIES  # noqa: E402
from core.inference.diffusion_precision import (  # noqa: E402
    effective_te_quant,
    resolve_te_quant_request,
)

CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)

_T5_INT8 = ("unsloth/FLUX.1-schnell-FP8", "FLUX.1-schnell-text_encoder_2-INT8-ConvRot.safetensors")
_T5_FP8 = ("unsloth/FLUX.1-schnell-FP8", "FLUX.1-schnell-text_encoder_2-FP8.safetensors")
_QWEN25VL_INT8 = ("unsloth/Qwen-Image-FP8", "Qwen-Image-text_encoder-INT8-ConvRot.safetensors")
_QWEN25VL_FP8 = ("unsloth/Qwen-Image-FP8", "Qwen-Image-text_encoder-FP8.safetensors")

# family -> (scheme an unset request resolves to, component, hosted (repo, filename), fp8 (repo, filename) behind it)
EXPECTED = {
    "flux.1-kontext": ("fp8", "text_encoder_2", _T5_FP8, None),
    "flux.2-dev": (
        "int8",
        "text_encoder",
        ("unsloth/FLUX.2-dev-FP8", "FLUX.2-dev-text_encoder-INT8-ConvRot.safetensors"),
        None,
    ),
    # Families with a torchao keep-bf16 int8 schedule have no fp8 file behind the int8 one: a missing file takes the
    # on-the-fly int8 cast, as before the file existed.
    "qwen-image": ("int8", "text_encoder", _QWEN25VL_INT8, None),
    "qwen-image-edit": ("int8", "text_encoder", _QWEN25VL_INT8, None),
    "qwen-image-layered": ("fp8", "text_encoder", _QWEN25VL_FP8, None),
    "hunyuanimage-2.1": ("int8", "text_encoder", _QWEN25VL_INT8, _QWEN25VL_FP8),
    "krea-2": (
        "int8",
        "text_encoder",
        ("unsloth/Krea-2-Turbo-FP8", "Krea-2-Turbo-text_encoder-INT8-ConvRot.safetensors"),
        ("unsloth/Krea-2-Turbo-FP8", "Krea-2-Turbo-text_encoder-FP8.safetensors"),
    ),
}
# Measured and deliberately left on the bf16 encoder by default: a small encoder (about 5B or less) stays bf16 unless
# ComfyUI's template quantizes it (FLUX.1's T5, Z-Image, Lumina-2, klein), and HiDream / Ideogram 4 missed the bar.
KEEP_BF16 = ("flux.1", "z-image", "lumina-2", "flux.2-klein", "hidream-i1", "ideogram-4", "sdxl")


def _fam(name):
    return next(f for f in _FAMILIES if f.name == name)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_unset_request_resolves_to_the_hosted_scheme(name):
    fam = _fam(name)
    scheme, component, hosted, fp8 = EXPECTED[name]
    for value in (None, "", "auto"):
        assert resolve_te_quant_request(value, fam.te_quant_auto) == (scheme, True)
    source = tpq.te_prequant_sources_for_base(
        fam, fam.base_repo, te_quant_mode = scheme, target = CUDA_BF16
    )[component]
    assert (source.location, source.filename) == hosted
    if fp8 is not None:
        # The int8 file's fallback is the fp8 file in the same repo, never the dense shards.
        assert source.fallback_filenames[0] == fp8[1]
    # The pre-load gate asks about fp8 support only: neither hosted file needs torchao.
    assert effective_te_quant(scheme, name) == "fp8"


@pytest.mark.parametrize("name", sorted(EXPECTED))
@pytest.mark.parametrize("value", ["none", "off", "None", "OFF"])
def test_none_still_pins_the_bf16_encoder(name, value):
    fam = _fam(name)
    # Paired with the default: the same family resolves an unset request to a hosted scheme.
    assert resolve_te_quant_request(None, fam.te_quant_auto)[0] is not None
    assert resolve_te_quant_request(value, fam.te_quant_auto) == (None, False)
    assert tpq.te_prequant_sources_for_base(fam, fam.base_repo, te_quant_mode = None, target = CUDA_BF16) == {}


def test_every_image_family_is_either_defaulted_or_measured_and_kept_bf16():
    undefaulted = {f.name for f in _FAMILIES if f.te_quant_auto is None}
    # FLUX.3 Klein and Qwen-Image-2.1 default to int8 ConvRot on their own; everything else is listed above.
    assert undefaulted == set(KEEP_BF16)
    for name in KEEP_BF16:
        assert resolve_te_quant_request(None, _fam(name).te_quant_auto) == (None, False)


@pytest.mark.parametrize(
    "name, base",
    [
        ("qwen-image", "Qwen/Qwen-Image-2512"),
        ("qwen-image-edit", "Qwen/Qwen-Image-Edit"),
        ("qwen-image-edit", "Qwen/Qwen-Image-Edit-2509"),
        ("flux.1-kontext", "black-forest-labs/FLUX.1-Kontext-dev"),
        ("krea-2", "krea/Krea-2-Raw"),
    ],
)
def test_every_release_of_the_same_encoder_gets_the_default_file(name, base):
    fam = _fam(name)
    scheme, component, hosted, _fp8 = EXPECTED[name]
    source = tpq.te_prequant_sources_for_base(fam, base, te_quant_mode = scheme, target = CUDA_BF16)[component]
    assert (source.location, source.filename) == hosted


# ── an automatic default never fails a load ───────────────────────────────────

MPS_BF16 = types.SimpleNamespace(device = "mps", dtype = torch.bfloat16)


def _missing_file(monkeypatch):
    """The hosted file cannot be loaded (not uploaded yet, offline, refused) and the dense shards are on disk."""
    import core.inference.media_locality as ml

    monkeypatch.setattr(tpq, "load_prequant_text_encoder", lambda *a, **k: None)
    monkeypatch.setattr(tpq, "_held_locally", lambda *a, **k: False)
    monkeypatch.setattr(ml, "_hosted_component_cached", lambda *a, **k: True)


@pytest.mark.parametrize("name", sorted(n for n, row in EXPECTED.items() if row[0] == "int8"))
def test_auto_int8_with_the_file_missing_on_mps_loads_dense_without_an_error(name, monkeypatch):
    _missing_file(monkeypatch)
    fam = _fam(name)
    with tpq.auto_te_request(True):
        out = tpq.te_prequant_pipe_kwargs(
            fam, fam.base_repo, te_quant_mode = fam.te_quant_auto, target = MPS_BF16, dtype = None,
            dense_source = "/somewhere/staged",
        )
    assert not any(k.startswith("text_encoder") for k in out)


def test_explicit_int8_with_the_file_missing_on_mps_is_still_refused(monkeypatch):
    _missing_file(monkeypatch)
    fam = _fam("qwen-image")
    with tpq.auto_te_request(False), pytest.raises(RuntimeError, match = "could not be loaded"):
        tpq.te_prequant_pipe_kwargs(fam, fam.base_repo, te_quant_mode = "int8", target = MPS_BF16, dtype = None)


def test_auto_with_the_dense_shards_left_out_fetches_them_instead_of_failing(monkeypatch, tmp_path):
    import core.inference.media_locality as ml

    monkeypatch.setattr(tpq, "load_prequant_text_encoder", lambda *a, **k: None)
    monkeypatch.setattr(tpq, "_held_locally", lambda *a, **k: False)
    monkeypatch.setattr(ml, "_hosted_component_cached", lambda *a, **k: False)
    fetched = []
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo, **k: fetched.append((repo, k["allow_patterns"]))
    )
    fam = _fam("krea-2")
    snap = tmp_path / "models--krea--Krea-2-Turbo" / "snapshots" / "abc"
    snap.mkdir(parents = True)
    snap = str(snap)
    with tpq.auto_te_request(True):
        out = tpq.te_prequant_pipe_kwargs(
            fam, fam.base_repo, te_quant_mode = "int8", target = CUDA_BF16, dtype = None, dense_source = snap
        )
    assert out == {} and fetched == [("krea/Krea-2-Turbo", ["text_encoder/*"])]
    # offline: nothing to fall back to, so the load explains itself as before
    with tpq.auto_te_request(True), pytest.raises(RuntimeError):
        tpq.te_prequant_pipe_kwargs(
            fam, fam.base_repo, te_quant_mode = "int8", target = CUDA_BF16, dtype = None, dense_source = snap,
            local_files_only = True,
        )


@pytest.mark.parametrize("device", ["cpu", "xpu", "cuda"])
def test_auto_on_a_device_that_cannot_run_the_scheme_keeps_the_dense_encoder(device, monkeypatch):
    import core.inference.diffusion_precision as prec

    monkeypatch.setattr(prec, "te_quant_supported", lambda *a, **k: False)
    target = types.SimpleNamespace(device = device, dtype = torch.bfloat16)
    for name in sorted(EXPECTED):
        fam = _fam(name)
        # nothing pre-cast is planned or loaded ...
        assert tpq.te_prequant_sources_for_base(fam, fam.base_repo, te_quant_mode = fam.te_quant_auto, target = target) == {}
        # ... and the in-place pass reports it unsupported instead of raising, leaving the encoder dense
        pipe = types.SimpleNamespace(text_encoder = torch.nn.Linear(4, 4))
        out = prec.quantize_text_encoders(pipe, target, mode = fam.te_quant_auto, family = name)
        assert out.mode is None and pipe.text_encoder.weight.dtype == torch.float32


def test_the_auto_flag_follows_the_raw_request():
    from core.inference.diffusion import _with_supplied_components

    seen = []

    @_with_supplied_components
    def load(self, repo_id, **kwargs):
        seen.append(tpq.te_request_is_auto())
        return {}

    for value, auto in ((None, True), ("auto", True), ("int8", False), ("none", False)):
        load(object(), "r", text_encoder_quant = value)
        assert seen.pop() is auto
    assert tpq.te_request_is_auto() is False


# ── explicit int8 with the file missing: never worse than before the file existed ─


@pytest.mark.parametrize("name", ["qwen-image", "qwen-image-edit", "qwen-image-layered", "flux.2-dev"])
def test_explicit_int8_without_the_file_still_takes_the_torchao_int8_cast(name, monkeypatch):
    import core.inference.diffusion_precision as prec

    # no fp8 file behind the int8 one: a missing file means the dense encoder, then the old on-the-fly int8
    src = tpq.resolve_te_prequant_source(_fam(name), "text_encoder", "int8")
    assert src.filename.endswith("INT8-ConvRot.safetensors") and src.fallback_filenames == ()
    monkeypatch.setattr(prec, "te_quant_supported", lambda *a, **k: True)
    monkeypatch.setattr(prec, "torchao_quantize_importable", lambda: True)
    calls = []
    monkeypatch.setattr(prec, "_cast_int8_selective", lambda enc, tgt, f, l: calls.append((f, l)))
    pipe = types.SimpleNamespace(text_encoder = torch.nn.Linear(4, 4))
    out = prec.quantize_text_encoders(pipe, CUDA_BF16, mode = "int8", family = name)
    assert out.mode == "int8" and calls == [prec._TE_INT8_SKIP[name]]
    # without torchao it is the fp8 cast, never a refusal
    monkeypatch.setattr(prec, "torchao_quantize_importable", lambda: False)
    fp8 = []
    monkeypatch.setattr(prec, "_cast_fp8", lambda enc, tgt: fp8.append(enc))
    out = prec.quantize_text_encoders(pipe, CUDA_BF16, mode = "int8", family = name)
    assert out.mode == "fp8" and fp8 == [pipe.text_encoder]


# ── an automatic int8 default without its file is the dense encoder, never fp8 or torchao ─


INT8_DEFAULTS = sorted(n for n, row in EXPECTED.items() if row[0] == "int8")


@pytest.mark.parametrize("name", INT8_DEFAULTS)
def test_the_automatic_int8_default_never_plans_the_fp8_file_behind_it(name):
    fam = _fam(name)
    mode, auto = resolve_te_quant_request(None, fam.te_quant_auto)
    assert auto and mode == "int8"
    component = EXPECTED[name][1]
    # outside a load (download plan, memory budget): the resolved mode itself carries the automatic pick
    source = tpq.te_prequant_sources_for_base(fam, fam.base_repo, te_quant_mode = mode, target = CUDA_BF16)[component]
    assert "INT8" in source.filename and source.fallback_filenames == ()
    # inside a load: the raw request does
    with tpq.auto_te_request(True):
        source = tpq.te_prequant_sources_for_base(fam, fam.base_repo, te_quant_mode = "int8", target = CUDA_BF16)[
            component
        ]
    assert source.fallback_filenames == ()


@pytest.mark.parametrize("name", INT8_DEFAULTS)
def test_the_automatic_int8_default_without_its_file_keeps_the_dense_encoder(name, monkeypatch):
    import core.inference.diffusion_precision as prec

    monkeypatch.setattr(prec, "te_quant_supported", lambda *a, **k: True)
    monkeypatch.setattr(prec, "torchao_quantize_importable", lambda: True)
    monkeypatch.setattr(prec, "_cast_fp8", lambda *a, **k: pytest.fail("fp8 cast of an automatic int8 default"))
    monkeypatch.setattr(prec, "_cast_int8_selective", lambda *a, **k: pytest.fail("torchao cast of an automatic default"))
    mode, _ = resolve_te_quant_request(None, _fam(name).te_quant_auto)
    pipe = types.SimpleNamespace(text_encoder = torch.nn.Linear(4, 4))
    out = prec.quantize_text_encoders(pipe, CUDA_BF16, mode = mode, family = name)
    assert out.mode is None and out.status == prec.RESOLVED_FELL_BACK
    assert pipe.text_encoder.weight.dtype == torch.float32


def test_an_automatic_int8_file_that_will_not_build_is_not_retried_as_fp8(monkeypatch):
    calls = []

    def fake_load(base, component, source, **kwargs):
        calls.append(source.filename)
        return None

    monkeypatch.setattr(tpq, "load_prequant_text_encoder", fake_load)
    monkeypatch.setattr(tpq, "_held_locally", lambda *a, **k: True)
    import core.inference.media_locality as ml

    monkeypatch.setattr(ml, "_hosted_component_cached", lambda *a, **k: True)
    fam = _fam("krea-2")
    with tpq.auto_te_request(True):
        tpq.te_prequant_pipe_kwargs(
            fam, fam.base_repo, te_quant_mode = "int8", target = CUDA_BF16, dtype = None, dense_source = "/d"
        )
    assert calls == ["Krea-2-Turbo-text_encoder-INT8-ConvRot.safetensors"]
    calls.clear()
    # explicit int8 keeps the int8 -> fp8 chain
    with tpq.auto_te_request(False):
        tpq.te_prequant_pipe_kwargs(
            fam, fam.base_repo, te_quant_mode = "int8", target = CUDA_BF16, dtype = None, dense_source = "/d"
        )
    assert calls == [
        "Krea-2-Turbo-text_encoder-INT8-ConvRot.safetensors",
        "Krea-2-Turbo-text_encoder-FP8.safetensors",
    ]
