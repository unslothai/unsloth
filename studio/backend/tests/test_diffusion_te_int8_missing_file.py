# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Explicit int8 TE quant with an unpublished int8 file: CUDA falls back to fp8 and says so; Apple Silicon refuses
up front, naming the file. Network-free."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_precision as prec  # noqa: E402
import core.inference.diffusion_te_prequant as tpq  # noqa: E402
from core.inference.diffusion_families import _FAMILIES  # noqa: E402
from core.inference.video_families import _FAMILIES as _VIDEO_FAMILIES  # noqa: E402

CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, backend = "cuda", index = 0)
MPS_BF16 = types.SimpleNamespace(device = "mps", dtype = torch.bfloat16, backend = "mps", index = 0)

# Files the Hub did not list on 2026-11-11 (FLUX.2-klein-9B-FP8 is private).
UNPUBLISHED = {
    "FLUX.1-schnell-text_encoder_2-INT8-ConvRot.safetensors",
    "FLUX.2-klein-4B-text_encoder-INT8-ConvRot.safetensors",
    "FLUX.2-klein-9B-text_encoder-INT8-ConvRot.safetensors",
    "Z-Image-Turbo-text_encoder-INT8-ConvRot.safetensors",
    "HiDream-I1-Full-text_encoder_4-INT8-ConvRot.safetensors",
    "ideogram-4-text_encoder-INT8-ConvRot.safetensors",
    "Lumina-Image-2.0-text_encoder-INT8-ConvRot.safetensors",
    "Wan2.2-TI2V-5B-text_encoder-INT8-ConvRot.safetensors",
}
_ALL = {f.name: f for f in tuple(_FAMILIES) + tuple(_VIDEO_FAMILIES)}
AFFECTED = sorted(
    name
    for name, comps in tpq.TE_INT8_CONVROT_FILES.items()
    if any(entry[1] in UNPUBLISHED for entry in comps.values())
)


@pytest.fixture
def hub(monkeypatch):
    """The Hub lists every named file except the unpublished ones; nothing is cached."""
    from core.inference.diffusion_text_encoder_trim import KEEP_LM_HEAD_ENV

    monkeypatch.delenv(KEEP_LM_HEAD_ENV, raising = False)
    monkeypatch.setattr(tpq, "_confirmed_cache", {})
    monkeypatch.setattr(tpq, "_pending_logged", set())
    monkeypatch.setattr(tpq, "_held_locally", lambda *a, **k: False)
    monkeypatch.setattr(tpq, "_hub_offline", lambda: False)

    def hub_files(pending, _api):
        return {c: s.filename for c, s in pending.items() if s.filename not in UNPUBLISHED}

    monkeypatch.setattr(tpq, "te_prequant_hub_files", hub_files)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: object())


def _cuda(monkeypatch):
    monkeypatch.setattr(prec, "te_quant_supported", lambda target, mode: target.device == "cuda")


def test_the_affected_set_is_the_survey_list():
    assert set(AFFECTED) == {
        "flux.1",
        "flux.1-kontext",
        "flux.2-klein",
        "hidream-i1",
        "ideogram-4",
        "lumina-2",
        "wan2.1-t2v-1.3b",
        "wan2.1-t2v-14b",
        "wan2.2-t2v-a14b",
        "wan2.2-ti2v-5b",
        "z-image",
    }


@pytest.mark.parametrize("name", AFFECTED)
def test_cuda_explicit_int8_has_the_hosted_fp8_file_behind_the_missing_one(name, hub, monkeypatch):
    _cuda(monkeypatch)
    fam = _ALL[name]
    sources = tpq.te_prequant_sources_for_base(
        fam,
        fam.base_repo,
        te_quant_mode = "int8",
        target = CUDA_BF16,
        components = tuple(c for c in tpq.TE_INT8_CONVROT_FILES[name] if c != "text_encoder_4"),
    )
    assert sources
    for source in sources.values():
        chain = tpq.te_candidate_filenames(source)
        assert not set(chain) & UNPUBLISHED, chain
        assert chain and all("fp8" in n.lower() for n in chain), chain
    assert prec.preflight_te_quant("int8", fam, fam.base_repo, CUDA_BF16) == "fp8"


@pytest.mark.parametrize("name", AFFECTED)
def test_the_unavailable_reason_names_the_missing_file(name, hub, monkeypatch):
    _cuda(monkeypatch)
    fam = _ALL[name]
    reason = tpq.explicit_int8_unavailable_reason(fam, fam.base_repo, MPS_BF16)
    assert reason is not None and "is not published on the Hub" in reason
    assert any(f in reason for f in UNPUBLISHED)


@pytest.mark.parametrize(
    "name", ["qwen-image", "qwen-image-2.1", "krea-2", "flux.2-dev", "moss-image"]
)
def test_a_published_file_has_no_unavailable_reason(name, hub, monkeypatch):
    _cuda(monkeypatch)
    fam = _ALL[name]
    assert tpq.explicit_int8_unavailable_reason(fam, fam.base_repo, MPS_BF16) is None
    assert tpq.explicit_int8_unavailable_reason(fam, fam.base_repo, CUDA_BF16) is None


def test_a_family_naming_no_file_has_no_reason(hub):
    assert tpq.explicit_int8_unavailable_reason(_ALL["ltx-2"], "Lightricks/LTX-2", MPS_BF16) is None


@pytest.mark.parametrize("name", ["z-image", "lumina-2", "flux.1", "ideogram-4"])
def test_image_mps_refusal_names_the_missing_file(name, hub, monkeypatch):
    import core.inference.diffusion as D

    fam = _ALL[name]
    with pytest.raises(RuntimeError) as err:
        D.DiffusionBackend()._assert_precision_for_target(
            fam,
            MPS_BF16,
            model_kind = "pipeline",
            pinned = None,
            te_mode = "int8",
            memory_mode = None,
            cpu_offload = False,
            base_repo = fam.base_repo,
        )
    text = str(err.value)
    assert "INT8-ConvRot.safetensors" in text and "is not published on the Hub" in text
    assert "fp8" in text


@pytest.mark.parametrize("name", ["wan2.2-ti2v-5b", "wan2.1-t2v-1.3b"])
def test_video_mps_explicit_int8_with_the_file_missing_is_refused_before_the_load(name, hub):
    from core.inference.video import _assert_video_precision_for_target
    with pytest.raises(
        RuntimeError, match = "Wan2.2-TI2V-5B-text_encoder-INT8-ConvRot.safetensors.*not published"
    ):
        _assert_video_precision_for_target(
            _ALL[name], MPS_BF16, model_kind = "pipeline", text_encoder_quant = "int8"
        )


def test_video_mps_explicit_int8_passes_once_the_file_is_published(hub, monkeypatch):
    from core.inference.video import _assert_video_precision_for_target
    monkeypatch.setattr(
        tpq,
        "te_prequant_hub_files",
        lambda pending, _api: {c: s.filename for c, s in pending.items()},
    )
    _assert_video_precision_for_target(
        _ALL["wan2.2-ti2v-5b"], MPS_BF16, model_kind = "pipeline", text_encoder_quant = "int8"
    )


def test_cuda_video_explicit_int8_still_passes_the_gate(hub, monkeypatch):
    import core.inference.video as V
    monkeypatch.setattr(V, "te_quant_supported", lambda target, mode: True)
    V._assert_video_precision_for_target(
        _ALL["wan2.2-ti2v-5b"], CUDA_BF16, model_kind = "pipeline", text_encoder_quant = "int8"
    )


def test_cuda_load_reports_the_fp8_fallback_for_an_explicit_int8(monkeypatch):
    monkeypatch.setattr(prec, "te_quant_supported", lambda target, mode: True)
    monkeypatch.setattr(prec, "_cast_fp8", lambda enc, tgt: None)
    encoder = types.SimpleNamespace(**{tpq.TE_PREQUANT_SCHEME_ATTR: "fp8"})
    pipe = types.SimpleNamespace(text_encoder = encoder)
    outcome = prec.quantize_text_encoders(pipe, CUDA_BF16, mode = "int8", family = "z-image")
    assert outcome.mode == "fp8"
    assert outcome.status == prec.RESOLVED_FELL_BACK
    assert "int8 ConvRot text encoder is not available" in outcome.reason
    assert "fp8 was used instead" in outcome.reason


def _table_entries():
    rows = [e for comps in tpq.TE_INT8_CONVROT_FILES.values() for e in comps.values()]
    rows += [
        e
        for bases in tpq.TE_INT8_CONVROT_VARIANT_FILES.values()
        for comps in bases.values()
        for e in comps.values()
    ]
    return {(str(e[0]).lower(), e[1]) for e in rows}


def test_every_table_row_is_published_or_pending():
    entries = _table_entries()
    published = set(tpq.TE_INT8_CONVROT_PUBLISHED)
    pending = {e for e in entries if e[1] in UNPUBLISHED}
    # A new row must be added to the published list (after its upload) or to UNPUBLISHED here.
    assert entries == published | pending
    assert not published & pending
    assert published <= entries
    assert {e[1] for e in pending} == UNPUBLISHED


@pytest.mark.parametrize("entry", sorted(tpq.TE_INT8_CONVROT_PUBLISHED))
def test_a_published_row_needs_no_hub_call(entry, monkeypatch):
    monkeypatch.setattr(
        tpq, "_held_locally", lambda *a, **k: pytest.fail("published rows are not probed")
    )
    monkeypatch.setattr(
        tpq, "te_prequant_hub_files", lambda *a, **k: pytest.fail("published rows are not probed")
    )
    assert tpq.te_int8_convrot_published(*entry)


def test_a_pending_row_is_hidden_and_logged_once(hub, monkeypatch, caplog):
    calls = []

    def hub_files(pending, _api):
        calls.append(sorted(s.filename for s in pending.values()))
        return {}

    monkeypatch.setattr(tpq, "te_prequant_hub_files", hub_files)
    messages = []
    monkeypatch.setattr(
        tpq,
        "_module_logger",
        lambda: types.SimpleNamespace(info = lambda *a: messages.append(a[0] % a[1:])),
    )
    fam = _ALL["z-image"]
    for _ in range(3):
        assert tpq.family_te_int8_convrot(fam, "text_encoder") is None
    assert tpq.family_te_int8_convrot(fam, "text_encoder", include_pending = True) is not None
    assert calls == [["Z-Image-Turbo-text_encoder-INT8-ConvRot.safetensors"]]
    assert len(messages) == 1 and "not published yet" in messages[0] and "fp8" in messages[0]


def test_a_pending_row_is_never_asked_of_the_hub_offline(hub, monkeypatch):
    monkeypatch.setattr(tpq, "_hub_offline", lambda: True)
    monkeypatch.setattr(
        tpq, "te_prequant_hub_files", lambda *a, **k: pytest.fail("offline: no Hub call")
    )
    assert tpq.family_te_int8_convrot(_ALL["lumina-2"], "text_encoder") is None


def test_a_pending_row_engages_once_the_hub_lists_it(hub, monkeypatch):
    _cuda(monkeypatch)
    monkeypatch.setattr(
        tpq,
        "te_prequant_hub_files",
        lambda pending, _api: {c: s.filename for c, s in pending.items()},
    )
    fam = _ALL["z-image"]
    source = tpq.resolve_te_prequant_source(fam, "text_encoder", "int8")
    assert source.filename == "Z-Image-Turbo-text_encoder-INT8-ConvRot.safetensors"
    assert any("fp8" in n.lower() for n in source.fallback_filenames)


def test_a_pending_row_engages_once_it_is_cached_locally(hub, monkeypatch):
    _cuda(monkeypatch)
    monkeypatch.setattr(tpq, "_held_locally", lambda repo, name, token: "INT8-ConvRot" in str(name))
    monkeypatch.setattr(
        tpq, "te_prequant_hub_files", lambda *a, **k: pytest.fail("cached: no Hub call")
    )
    fam = _ALL["wan2.2-ti2v-5b"]
    source = tpq.resolve_te_prequant_source(fam, "text_encoder", "int8")
    assert source.filename == "Wan2.2-TI2V-5B-text_encoder-INT8-ConvRot.safetensors"


@pytest.mark.parametrize("name", ["z-image", "wan2.2-ti2v-5b", "flux.1"])
def test_cuda_load_of_a_pending_row_downloads_only_the_fp8_file(name, hub, monkeypatch):
    import core.inference.diffusion_prequant as dpq

    _cuda(monkeypatch)
    fam = _ALL[name]
    component = next(iter(tpq.TE_INT8_CONVROT_FILES[name]))
    source = tpq.te_prequant_sources_for_base(
        fam, fam.base_repo, te_quant_mode = "int8", target = CUDA_BF16, components = (component,)
    )[component]
    asked = []

    def download(src, filename, *a, **k):
        asked.append(filename)
        return f"/fake/{filename}"

    monkeypatch.setattr(dpq, "_download_checkpoint_name", download)
    path = tpq._resolve_checkpoint_path(source, None, cache_dir = "/nonexistent")
    assert asked and not set(asked) & UNPUBLISHED
    assert path.endswith(asked[-1]) and "fp8" in asked[-1].lower()


def test_a_cache_only_preflight_never_asks_the_hub_for_a_pending_row(hub, monkeypatch):
    _cuda(monkeypatch)
    monkeypatch.setattr(
        tpq, "te_prequant_hub_files", lambda *a, **k: pytest.fail("cache-only: no Hub call")
    )
    fam = _ALL["lumina-2"]
    takes = tpq.explicit_int8_hosted_components(
        fam, fam.base_repo, CUDA_BF16, local_files_only = True
    )
    assert takes and not any(takes.values())
