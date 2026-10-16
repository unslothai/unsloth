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
    assert (
        tpq.te_prequant_sources_for_base(fam, fam.base_repo, te_quant_mode = None, target = CUDA_BF16)
        == {}
    )


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
    source = tpq.te_prequant_sources_for_base(fam, base, te_quant_mode = scheme, target = CUDA_BF16)[
        component
    ]
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
            fam,
            fam.base_repo,
            te_quant_mode = fam.te_quant_auto,
            target = MPS_BF16,
            dtype = None,
            dense_source = "/somewhere/staged",
        )
    assert not any(k.startswith("text_encoder") for k in out)


def test_explicit_int8_with_the_file_missing_on_mps_is_still_refused(monkeypatch):
    _missing_file(monkeypatch)
    fam = _fam("qwen-image")
    with tpq.auto_te_request(False), pytest.raises(RuntimeError, match = "could not be loaded"):
        tpq.te_prequant_pipe_kwargs(
            fam, fam.base_repo, te_quant_mode = "int8", target = MPS_BF16, dtype = None
        )


def test_auto_with_the_dense_shards_left_out_fetches_them_instead_of_failing(monkeypatch, tmp_path):
    import core.inference.media_locality as ml

    monkeypatch.setattr(tpq, "load_prequant_text_encoder", lambda *a, **k: None)
    monkeypatch.setattr(tpq, "_held_locally", lambda *a, **k: False)
    monkeypatch.setattr(ml, "_hosted_component_cached", lambda *a, **k: False)
    fetched = []
    import huggingface_hub

    fam = _fam("krea-2")
    snap = tmp_path / "models--krea--Krea-2-Turbo" / "snapshots" / "abc"
    snap.mkdir(parents = True)
    snap = str(snap)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, **k: fetched.append((repo, k["allow_patterns"])) or snap,
    )
    with tpq.auto_te_request(True):
        out = tpq.te_prequant_pipe_kwargs(
            fam,
            fam.base_repo,
            te_quant_mode = "int8",
            target = CUDA_BF16,
            dtype = None,
            dense_source = snap,
        )
    assert out == {} and fetched == [("krea/Krea-2-Turbo", ["text_encoder/*"])]
    # offline: nothing to fall back to, so the load explains itself as before
    with tpq.auto_te_request(True), pytest.raises(RuntimeError):
        tpq.te_prequant_pipe_kwargs(
            fam,
            fam.base_repo,
            te_quant_mode = "int8",
            target = CUDA_BF16,
            dtype = None,
            dense_source = snap,
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
        assert (
            tpq.te_prequant_sources_for_base(
                fam, fam.base_repo, te_quant_mode = fam.te_quant_auto, target = target
            )
            == {}
        )
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


@pytest.mark.parametrize(
    "name", ["qwen-image", "qwen-image-edit", "qwen-image-layered", "flux.2-dev"]
)
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
    source = tpq.te_prequant_sources_for_base(
        fam, fam.base_repo, te_quant_mode = mode, target = CUDA_BF16
    )[component]
    assert "INT8" in source.filename and source.fallback_filenames == ()
    # inside a load: the raw request does
    with tpq.auto_te_request(True):
        source = tpq.te_prequant_sources_for_base(
            fam, fam.base_repo, te_quant_mode = "int8", target = CUDA_BF16
        )[component]
    assert source.fallback_filenames == ()


@pytest.mark.parametrize("name", INT8_DEFAULTS)
def test_the_automatic_int8_default_without_its_file_keeps_the_dense_encoder(name, monkeypatch):
    import core.inference.diffusion_precision as prec

    monkeypatch.setattr(prec, "te_quant_supported", lambda *a, **k: True)
    monkeypatch.setattr(prec, "torchao_quantize_importable", lambda: True)
    monkeypatch.setattr(
        prec, "_cast_fp8", lambda *a, **k: pytest.fail("fp8 cast of an automatic int8 default")
    )
    monkeypatch.setattr(
        prec,
        "_cast_int8_selective",
        lambda *a, **k: pytest.fail("torchao cast of an automatic default"),
    )
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
            fam,
            fam.base_repo,
            te_quant_mode = "int8",
            target = CUDA_BF16,
            dtype = None,
            dense_source = "/d",
        )
    assert calls == ["Krea-2-Turbo-text_encoder-INT8-ConvRot.safetensors"]
    calls.clear()
    # explicit int8 keeps the int8 -> fp8 chain
    with tpq.auto_te_request(False):
        tpq.te_prequant_pipe_kwargs(
            fam,
            fam.base_repo,
            te_quant_mode = "int8",
            target = CUDA_BF16,
            dtype = None,
            dense_source = "/d",
        )
    assert calls == [
        "Krea-2-Turbo-text_encoder-INT8-ConvRot.safetensors",
        "Krea-2-Turbo-text_encoder-FP8.safetensors",
    ]


# ---- memory budget: an automatic default budgets dense until its hosted file is confirmed ------------------------


def _budget(
    monkeypatch,
    mode,
    *,
    held = False,
    listed = (),
    offline = False,
    hub_raises = False,
):
    from core.inference.diffusion_precision import AutoTeQuant  # noqa: F401

    fam = _fam("qwen-image")
    tpq._confirmed_cache.clear()
    monkeypatch.setattr(tpq, "_held_locally", lambda repo, name, token: held)
    monkeypatch.setattr(tpq, "_hub_offline", lambda: offline)
    calls = []

    class _Api:
        def model_info(
            self,
            repo,
            files_metadata = False,
        ):
            calls.append(repo)
            if hub_raises:
                raise OSError("Hub unreachable")
            sib = [types.SimpleNamespace(rfilename = n, size = 9_000_000_000) for n in listed]
            return types.SimpleNamespace(siblings = sib)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: _Api())
    scale = tpq.te_prequant_budget_scale(
        fam, te_quant_mode = mode, target = CUDA_BF16, base = fam.base_repo
    )
    return scale, calls


def _auto_int8():
    from core.inference.diffusion_precision import AutoTeQuant
    return AutoTeQuant("int8")


def test_automatic_default_budgets_dense_while_the_file_is_not_on_the_hub(monkeypatch):
    scale, calls = _budget(monkeypatch, _auto_int8(), listed = ())
    assert scale == 1.0
    assert calls == [_QWEN25VL_INT8[0]]


def test_automatic_default_budgets_dense_offline_without_the_file_cached(monkeypatch):
    scale, calls = _budget(monkeypatch, _auto_int8(), offline = True, listed = (_QWEN25VL_INT8[1],))
    assert scale == 1.0
    assert calls == []


def test_automatic_default_budgets_dense_when_the_hub_cannot_answer(monkeypatch):
    scale, _ = _budget(monkeypatch, _auto_int8(), hub_raises = True)
    assert scale == 1.0


def test_automatic_default_budgets_int8_once_the_hub_lists_the_file(monkeypatch):
    scale, _ = _budget(monkeypatch, _auto_int8(), listed = (_QWEN25VL_INT8[1],))
    explicit, _ = _budget(monkeypatch, "int8")
    assert scale == explicit < 1.0


def test_automatic_default_budgets_int8_when_the_file_is_cached_without_asking_the_hub(monkeypatch):
    scale, calls = _budget(monkeypatch, _auto_int8(), held = True, offline = True)
    assert scale < 1.0
    assert calls == []


def test_automatic_default_inside_a_load_budgets_dense_without_the_file(monkeypatch):
    with tpq.auto_te_request(True):
        scale, _ = _budget(monkeypatch, "int8", listed = ())
    assert scale == 1.0


def test_explicit_request_budget_is_unchanged_by_the_confirmation(monkeypatch):
    scale, calls = _budget(monkeypatch, "int8", listed = ())
    assert scale < 1.0
    assert calls == []


# ---- review round 1 ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["flux.1-kontext", "qwen-image-layered"])
def test_the_automatic_fp8_default_without_its_file_keeps_the_dense_encoder(name, monkeypatch):
    import core.inference.diffusion_precision as prec

    monkeypatch.setattr(prec, "te_quant_supported", lambda *a, **k: True)
    monkeypatch.setattr(
        prec, "_cast_fp8", lambda *a, **k: pytest.fail("runtime fp8 cast of an automatic default")
    )
    mode, auto = resolve_te_quant_request(None, _fam(name).te_quant_auto)
    assert auto and mode == "fp8"
    pipe = types.SimpleNamespace(text_encoder = torch.nn.Linear(4, 4))
    out = prec.quantize_text_encoders(pipe, CUDA_BF16, mode = mode, family = name)
    assert out.mode is None and out.status == prec.RESOLVED_FELL_BACK
    with tpq.auto_te_request(True):
        out = prec.quantize_text_encoders(pipe, CUDA_BF16, mode = "fp8", family = name)
    assert out.mode is None


def test_the_automatic_fp8_default_with_its_file_is_applied(monkeypatch):
    import core.inference.diffusion_precision as prec

    monkeypatch.setattr(prec, "te_quant_supported", lambda *a, **k: True)
    cast = []
    monkeypatch.setattr(prec, "_cast_fp8", lambda enc, tgt: cast.append(enc))
    encoder = torch.nn.Linear(4, 4)
    setattr(encoder, tpq.TE_PREQUANT_SCHEME_ATTR, "fp8")
    mode, _ = resolve_te_quant_request(None, _fam("flux.1-kontext").te_quant_auto)
    out = prec.quantize_text_encoders(
        types.SimpleNamespace(text_encoder = encoder), CUDA_BF16, mode = mode, family = "flux.1-kontext"
    )
    assert out.mode == "fp8"


def test_an_explicit_fp8_request_still_casts_a_dense_encoder(monkeypatch):
    import core.inference.diffusion_precision as prec

    monkeypatch.setattr(prec, "te_quant_supported", lambda *a, **k: True)
    cast = []
    monkeypatch.setattr(prec, "_cast_fp8", lambda enc, tgt: cast.append(enc))
    out = prec.quantize_text_encoders(
        types.SimpleNamespace(text_encoder = torch.nn.Linear(4, 4)),
        CUDA_BF16,
        mode = "fp8",
        family = "flux.1-kontext",
    )
    assert out.mode == "fp8" and len(cast) == 1


def test_companion_cleanup_reads_four_field_rows_and_int8_convrot_files():
    from hub.utils.companion_assets import _te_prequant_names

    flux1 = _te_prequant_names("unsloth/FLUX.1-schnell-FP8")
    assert "flux.1-schnell-text_encoder_2-fp8.safetensors" in flux1
    assert "flux.1-schnell-text_encoder_2-int8-convrot.safetensors" in flux1
    for repo, filename in (
        _QWEN25VL_INT8,
        ("unsloth/FLUX.2-klein-9B-FP8", "FLUX.2-klein-9B-text_encoder-INT8-ConvRot.safetensors"),
        ("unsloth/HiDream-I1-Full-FP8", "HiDream-I1-Full-text_encoder_4-INT8-ConvRot.safetensors"),
        ("unsloth/HiDream-I1-Full-FP8", "HiDream-I1-Full-text_encoder_4-FP8.safetensors"),
    ):
        assert filename.lower() in _te_prequant_names(repo), filename
    # names stay per repo: another repo's ConvRot file is not a companion here
    assert "qwen-image-text_encoder-int8-convrot.safetensors" not in flux1


def test_a_repo_holding_only_an_int8_convrot_encoder_is_not_an_installed_checkpoint(monkeypatch):
    import hub.utils.companion_assets as ca

    repo = types.SimpleNamespace(repo_id = _QWEN25VL_INT8[0])
    monkeypatch.setattr(ca, "_snapshot_relative_names", lambda r: [_QWEN25VL_INT8[1]])
    assert ca.repo_holds_denoiser(repo) is False


def _hidream_pipe(
    te3_scheme,
    te4_scheme = None,
    te4_runtime = None,
):
    from core.inference.diffusion_precision import TE_RUNTIME_SCHEME_ATTR

    te3, te4 = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    setattr(te3, tpq.TE_PREQUANT_SCHEME_ATTR, te3_scheme)
    if te4_scheme:
        setattr(te4, tpq.TE_PREQUANT_SCHEME_ATTR, te4_scheme)
    if te4_runtime:
        setattr(te4, TE_RUNTIME_SCHEME_ATTR, te4_runtime)
    return types.SimpleNamespace(text_encoder_3 = te3, text_encoder_4 = te4)


@pytest.mark.parametrize(
    "te4_scheme, te4_runtime, held",
    [
        ("fp8", None, "text_encoder_4 is fp8"),
        (None, "fp8", "text_encoder_4 is fp8"),
        (None, None, "dense bf16"),
    ],
)
def test_hidream_reports_a_fourth_encoder_off_the_int8_scheme(
    monkeypatch, te4_scheme, te4_runtime, held
):
    import core.inference.diffusion_precision as prec

    monkeypatch.setattr(prec, "te_quant_supported", lambda *a, **k: True)
    out = prec.quantize_text_encoders(
        _hidream_pipe("int8", te4_scheme, te4_runtime), CUDA_BF16, mode = "int8", family = "hidream-i1"
    )
    assert out.mode == "int8" and out.status == prec.RESOLVED_FELL_BACK and held in out.reason


def test_hidream_int8_on_both_encoders_is_applied(monkeypatch):
    import core.inference.diffusion_precision as prec

    monkeypatch.setattr(prec, "te_quant_supported", lambda *a, **k: True)
    out = prec.quantize_text_encoders(
        _hidream_pipe("int8", "int8"), CUDA_BF16, mode = "int8", family = "hidream-i1"
    )
    assert out.mode == "int8" and out.status == prec.RESOLVED_APPLIED


MPS_BF16 = types.SimpleNamespace(device = "mps", dtype = torch.bfloat16, backend = "mps")


@pytest.mark.parametrize("confirmed, priced", [(False, False), (True, True)])
def test_mps_automatic_int8_prices_the_file_only_once_it_is_confirmed(
    monkeypatch, confirmed, priced
):
    from core.inference import diffusion_prequant as pq
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(pq, "prefer_cached_pickle_twins", lambda repo, names, **k: names)
    monkeypatch.setattr(pq, "first_cached_as_resolved", lambda *a, **k: None)
    monkeypatch.setattr(tpq, "te_prequant_sources_confirmed", lambda sources: confirmed)
    fam = _fam("flux.2-dev")
    mode, _ = resolve_te_quant_request(None, fam.te_quant_auto)
    got = DiffusionBackend._precast_text_encoder_mib(fam, fam.base_repo, MPS_BF16, mode)
    # None leaves the scanned / table dense encoder as the plan's encoder term
    assert (got is not None) is priced
    # an explicit request keeps pricing the file it either loads or refuses on
    assert (
        DiffusionBackend._precast_text_encoder_mib(fam, fam.base_repo, MPS_BF16, "int8") is not None
    )


# ---- review round 2 ---------------------------------------------------------------------------------------------


def test_dense_fallback_fills_the_staged_snapshot_in_its_own_cache_root(monkeypatch, tmp_path):
    import huggingface_hub

    import core.inference.media_locality as ml

    rev = "a" * 40
    snap = tmp_path / "secondary" / "models--Qwen--Qwen-Image" / "snapshots" / rev
    snap.mkdir(parents = True)
    monkeypatch.setattr(ml, "_hosted_component_cached", lambda *a, **k: False)
    calls = []

    def fake_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    assert tpq._fetch_dense_components(
        str(snap), ["text_encoder"], hf_token = None, local_files_only = False
    )
    repo_id, kwargs = calls[0]
    assert repo_id == "Qwen/Qwen-Image"
    assert kwargs["cache_dir"] == str(tmp_path / "secondary")
    assert kwargs["revision"] == rev
    assert kwargs["allow_patterns"] == ["text_encoder/*"]


def test_dense_fallback_into_another_snapshot_is_not_reported_as_filled(monkeypatch, tmp_path):
    import huggingface_hub

    import core.inference.media_locality as ml

    snap = tmp_path / "models--Qwen--Qwen-Image" / "snapshots" / ("b" * 40)
    snap.mkdir(parents = True)
    monkeypatch.setattr(ml, "_hosted_component_cached", lambda *a, **k: False)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path / "elsewhere")
    )
    assert not tpq._fetch_dense_components(
        str(snap), ["text_encoder"], hf_token = None, local_files_only = False
    )


@pytest.mark.parametrize("name", INT8_DEFAULTS)
def test_a_kept_lm_head_keeps_the_automatic_int8_default_dense(name, monkeypatch):
    import core.inference.diffusion_precision as prec
    from core.inference.diffusion_text_encoder_trim import KEEP_LM_HEAD_ENV

    monkeypatch.setenv(KEEP_LM_HEAD_ENV, "1")
    fam = _fam(name)
    mode, _ = resolve_te_quant_request(None, fam.te_quant_auto)
    assert (
        tpq.te_prequant_sources_for_base(fam, fam.base_repo, te_quant_mode = mode, target = CUDA_BF16)
        == {}
    )
    with tpq.auto_te_request(True):
        assert (
            tpq.te_prequant_sources_for_base(
                fam, fam.base_repo, te_quant_mode = "int8", target = CUDA_BF16
            )
            == {}
        )
    # an explicit int8 still takes the fp8 file, which carries the head
    explicit = tpq.te_prequant_sources_for_base(
        fam, fam.base_repo, te_quant_mode = "int8", target = CUDA_BF16
    )
    assert all("FP8" in s.filename for s in explicit.values()) and explicit
    monkeypatch.setattr(prec, "te_quant_supported", lambda *a, **k: True)
    out = prec.quantize_text_encoders(
        types.SimpleNamespace(text_encoder = torch.nn.Linear(4, 4)), CUDA_BF16, mode = mode, family = name
    )
    assert out.mode is None and KEEP_LM_HEAD_ENV in out.reason


MPS_TARGET = types.SimpleNamespace(
    device = "mps", dtype = torch.bfloat16, backend = "mps", supports_model_cpu_offload = False
)


def _preflight(fam, te_mode):
    from core.inference.diffusion import DiffusionBackend
    DiffusionBackend()._assert_precision_for_target(
        fam,
        MPS_TARGET,
        model_kind = "pipeline",
        pinned = None,
        te_mode = te_mode,
        memory_mode = None,
        cpu_offload = False,
        base_repo = fam.base_repo,
    )


@pytest.mark.parametrize("name", ["qwen-image", "flux.2-dev", "krea-2", "z-image"])
def test_mps_preflight_accepts_explicit_int8_on_a_family_hosting_the_int8_file(name, monkeypatch):
    monkeypatch.setattr(tpq, "te_prequant_sources_confirmed", lambda sources: True)
    _preflight(_fam(name), "int8")


def test_mps_preflight_still_refuses_explicit_fp8():
    with pytest.raises(RuntimeError):
        _preflight(_fam("qwen-image"), "fp8")


# ---- review round 3: one answer to "will this explicit int8 load the hosted int8 ConvRot file here" --------------

_DEVICES = {
    "cuda": CUDA_BF16,
    "mps": MPS_TARGET,
    "cpu": types.SimpleNamespace(
        device = "cpu", dtype = torch.bfloat16, backend = "cpu", supports_model_cpu_offload = False
    ),
}
_HOSTING = sorted(n for n in tpq.TE_INT8_CONVROT_FILES if any(f.name == n for f in _FAMILIES))


def _table_env(monkeypatch, device, present, keep_head):
    import core.inference.diffusion_precision as prec
    from core.inference.diffusion_text_encoder_trim import KEEP_LM_HEAD_ENV

    if keep_head:
        monkeypatch.setenv(KEEP_LM_HEAD_ENV, "1")
    else:
        monkeypatch.delenv(KEEP_LM_HEAD_ENV, raising = False)
    import core.inference.diffusion as D

    # fp8 runs on the CUDA card only; MPS and CPU do not. diffusion.py imports the helper by name, so patch both.
    supported = lambda target, mode: getattr(target, "device", None) == "cuda"
    monkeypatch.setattr(prec, "te_quant_supported", supported)
    monkeypatch.setattr(D, "te_quant_supported", supported)
    monkeypatch.setattr(tpq, "te_prequant_sources_confirmed", lambda sources: present)
    return _DEVICES[device]


@pytest.mark.parametrize("keep_head", [False, True])
@pytest.mark.parametrize("present", [True, False])
@pytest.mark.parametrize("device", sorted(_DEVICES))
@pytest.mark.parametrize("name", _HOSTING)
def test_explicit_int8_takes_the_hosted_file_table(name, device, present, keep_head, monkeypatch):
    target = _table_env(monkeypatch, device, present, keep_head)
    fam = _fam(name)
    got = tpq.explicit_int8_takes_hosted_file(fam, fam.base_repo, target)
    assert got is (device in ("cuda", "mps") and present and not keep_head)
    import core.inference.diffusion_precision as prec

    checked = prec.preflight_te_quant("int8", fam, fam.base_repo, target)
    if got:
        assert checked is None
    else:
        assert checked in ("int8", "fp8")
    # a supplied encoder for a hosted component replaces the file
    comp = next(iter(tpq.TE_INT8_CONVROT_FILES[name]))
    assert (
        tpq.explicit_int8_takes_hosted_file(fam, fam.base_repo, target, supplied = (comp,)) is False
    )


def test_explicit_int8_on_a_custom_base_does_not_take_the_hosted_file(monkeypatch):
    target = _table_env(monkeypatch, "mps", True, False)
    assert (
        tpq.explicit_int8_takes_hosted_file(_fam("qwen-image"), "someone/custom-qwen-image", target)
        is False
    )


@pytest.mark.parametrize("present", [True, False])
def test_mps_preflight_refuses_explicit_int8_unless_the_file_will_load(present, monkeypatch):
    target = _table_env(monkeypatch, "mps", present, False)
    from core.inference.diffusion import DiffusionBackend

    fam = _fam("krea-2")

    def run():
        DiffusionBackend()._assert_precision_for_target(
            fam,
            target,
            model_kind = "pipeline",
            pinned = None,
            te_mode = "int8",
            memory_mode = None,
            cpu_offload = False,
            base_repo = fam.base_repo,
        )

    if present:
        run()
    else:
        with pytest.raises(RuntimeError):
            run()


def test_mps_preflight_refuses_explicit_int8_with_a_kept_lm_head(monkeypatch):
    target = _table_env(monkeypatch, "mps", True, True)
    with pytest.raises(RuntimeError):
        _preflight(_fam("qwen-image"), "int8")


@pytest.mark.parametrize("present", [True, False])
def test_explicit_int8_under_offload_checks_torchao_when_the_file_will_not_load(
    present, monkeypatch
):
    import core.inference.diffusion as D
    import core.inference.diffusion_precision as prec

    target = _table_env(monkeypatch, "cuda", present, False)
    monkeypatch.setattr(prec, "torchao_quantize_importable", lambda: True)
    monkeypatch.setattr(D, "torchao_quantize_importable", lambda: True)
    fam = _fam("qwen-image")
    assert prec.preflight_te_quant("int8", fam, fam.base_repo, target) == (
        None if present else "int8"
    )

    def run():
        D.DiffusionBackend()._assert_precision_for_target(
            fam,
            target,
            model_kind = "pipeline",
            pinned = None,
            te_mode = "int8",
            memory_mode = "balanced",
            cpu_offload = True,
            base_repo = fam.base_repo,
        )

    if present:
        run()
    else:
        with pytest.raises(RuntimeError, match = "offload"):
            run()


def test_hidream_loads_the_hosted_int8_te4_on_mps(monkeypatch):
    import core.inference.diffusion_hidream as dh
    import core.inference.diffusion_precision as prec
    import transformers

    monkeypatch.delenv("UNSLOTH_TE_KEEP_LM_HEAD", raising = False)
    monkeypatch.setattr(prec, "te_quant_supported", lambda target, mode: False)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: "tok4")
    )
    monkeypatch.setattr(
        transformers.LlamaForCausalLM,
        "from_pretrained",
        staticmethod(lambda *a, **k: pytest.fail("dense TE4")),
    )
    loaded = []
    sentinel = object()

    def fake_load(base, component, source, **kwargs):
        loaded.append((source.filename, kwargs.get("scheme")))
        return sentinel

    monkeypatch.setattr(tpq, "load_prequant_text_encoder", fake_load)
    out = dh.hidream_te4_kwargs(
        None, None, fam = _fam("hidream-i1"), te_quant_mode = "int8", target = MPS_TARGET
    )
    assert out["text_encoder_4"] is sentinel
    assert loaded == [("HiDream-I1-Full-text_encoder_4-INT8-ConvRot.safetensors", "int8")]


def _hidream_mps(monkeypatch, *, load):
    import core.inference.diffusion_precision as prec
    import transformers

    monkeypatch.delenv("UNSLOTH_TE_KEEP_LM_HEAD", raising = False)
    monkeypatch.setattr(prec, "te_quant_supported", lambda target, mode: False)
    monkeypatch.setattr(prec, "_cast_fp8", lambda *a, **k: pytest.fail("fp8 cast on MPS"))
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: "tok4")
    )
    dense = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        transformers.LlamaForCausalLM, "from_pretrained", staticmethod(lambda *a, **k: dense)
    )
    monkeypatch.setattr(tpq, "load_prequant_text_encoder", lambda *a, **k: load)
    return dense


def test_hidream_explicit_int8_on_mps_refuses_when_the_te4_file_fails(monkeypatch):
    import core.inference.diffusion_hidream as dh
    _hidream_mps(monkeypatch, load = None)
    with pytest.raises(RuntimeError, match = "text_encoder_4"):
        dh.hidream_te4_kwargs(
            None, None, fam = _fam("hidream-i1"), te_quant_mode = "int8", target = MPS_TARGET
        )


def test_hidream_none_on_mps_still_loads_the_dense_te4(monkeypatch):
    import core.inference.diffusion_hidream as dh

    dense = _hidream_mps(monkeypatch, load = None)
    out = dh.hidream_te4_kwargs(
        None, None, fam = _fam("hidream-i1"), te_quant_mode = None, target = MPS_TARGET
    )
    assert out["text_encoder_4"] is dense


# ---- review round 4: both cache roots --------------------------------------------------------------------------


def _cache_file(
    root,
    repo,
    name,
    rev = "c" * 40,
):
    repo_dir = root / ("models--" + repo.replace("/", "--"))
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text(rev)
    snap = repo_dir / "snapshots" / rev
    snap.mkdir(parents = True, exist_ok = True)
    (snap / name).parent.mkdir(parents = True, exist_ok = True)
    (snap / name).write_bytes(b"x")
    return snap


def _two_roots(monkeypatch, tmp_path):
    from huggingface_hub import constants

    import utils.hf_cache_settings as hcs

    live, other = tmp_path / "live", tmp_path / "import-time"
    live.mkdir()
    other.mkdir()
    monkeypatch.setattr(hcs, "active_hf_hub_cache", lambda: str(live))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(other))
    import core.inference.diffusion as D

    monkeypatch.setattr(D, "hub_cache_dir", lambda: str(live))
    monkeypatch.delenv(tpq.TE_PREQUANT_MIRROR_ENV, raising = False)
    return live, other


@pytest.mark.parametrize("where", ["live", "import-time"])
def test_a_hosted_encoder_in_either_cache_root_is_confirmed_offline(where, monkeypatch, tmp_path):
    live, other = _two_roots(monkeypatch, tmp_path)
    _cache_file(live if where == "live" else other, *_QWEN25VL_INT8)
    monkeypatch.setattr(tpq, "_hub_offline", lambda: True)
    tpq._confirmed_cache.clear()
    assert tpq._held_locally(*_QWEN25VL_INT8, None)
    fam = _fam("qwen-image")
    sources = tpq.te_prequant_sources_for_base(
        fam, fam.base_repo, te_quant_mode = "int8", target = CUDA_BF16
    )
    assert tpq.te_prequant_sources_confirmed(sources)


def test_a_hosted_encoder_in_neither_root_is_not_confirmed_offline(monkeypatch, tmp_path):
    _two_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(tpq, "_hub_offline", lambda: True)
    assert not tpq._held_locally(*_QWEN25VL_INT8, None)


def test_dense_shards_in_the_import_time_root_count_as_cached(monkeypatch, tmp_path):
    live, other = _two_roots(monkeypatch, tmp_path)
    import core.inference.media_locality as ml

    monkeypatch.setattr(ml, "_hosted_component_cached", lambda *a, **k: False)
    _cache_file(other, "Qwen/Qwen-Image", "text_encoder/model.safetensors")
    _cache_file(other, "Qwen/Qwen-Image", "text_encoder/config.json")
    assert tpq._dense_component_cached("Qwen/Qwen-Image", "text_encoder")
    assert not tpq._dense_component_cached("Qwen/Qwen-Image", "vae")
