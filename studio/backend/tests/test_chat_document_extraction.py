# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Tests for the chat document extractor + VLM capability probe.

Probe tests run regardless of the extraction backend because they only
shape-check :mod:`core.chat.vlm_capability`. Backend-backed tests skip
cleanly when the optional deps (pymupdf / pymupdf4llm / mammoth) are
missing.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from core.chat.vlm_capability import (
    VlmCapability,
    detect_loaded_vlm,
    extract_self_base_url,
)


# ---------------------------------------------------------------------- #
# VlmCapability dataclass                                                #
# ---------------------------------------------------------------------- #


def test_vlm_capability_none_factory_is_safe_default() -> None:
    cap = VlmCapability.none()
    assert cap.is_vlm is False
    assert cap.endpoint_url is None
    assert cap.model_name is None
    assert cap.source == "none"
    assert cap.reason  # non-empty


def test_vlm_capability_to_dict_round_trips_fields() -> None:
    cap = VlmCapability(
        is_vlm = True,
        endpoint_url = "http://127.0.0.1:8080",
        model_name = "qwen2-vl",
        source = "gguf",
        reason = None,
    )
    assert cap.to_dict() == {
        "is_vlm": True,
        "endpoint_url": "http://127.0.0.1:8080",
        "model_name": "qwen2-vl",
        "source": "gguf",
        "reason": None,
    }


# ---------------------------------------------------------------------- #
# detect_loaded_vlm() across backend shapes                              #
# ---------------------------------------------------------------------- #


class _FakeLlama:
    def __init__(
        self,
        *,
        loaded: bool,
        vision: bool = False,
        base_url: str = "http://127.0.0.1:8080",
        model_id: str = "fake-gguf",
    ) -> None:
        self.is_loaded = loaded
        self.is_vision = vision
        self.base_url = base_url
        self.model_identifier = model_id


class _FakeInferenceBackend:
    def __init__(
        self,
        *,
        active: Optional[str],
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.active_model_name = active
        self.models: Dict[str, Dict[str, Any]] = (
            {active: info or {}} if active else {}
        )


def _patch_probes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    llama: Optional[_FakeLlama],
    inference: Optional[_FakeInferenceBackend],
) -> None:
    from core.chat import vlm_capability as vc

    if llama is None:
        monkeypatch.setattr(vc, "_probe_gguf", lambda _llama = None: None)
    else:
        def probe_gguf(llama_backend = None):
            backend = llama_backend or llama
            if not backend.is_loaded:
                return None
            is_vision = bool(backend.is_vision)
            return VlmCapability(
                is_vlm = is_vision,
                endpoint_url = backend.base_url,
                model_name = backend.model_identifier,
                source = "gguf",
                reason = None if is_vision else "loaded GGUF is not vision-capable",
            )

        monkeypatch.setattr(vc, "_probe_gguf", probe_gguf)

    if inference is None:
        monkeypatch.setattr(vc, "_probe_transformers", lambda _u: None)
    else:
        def probe_tf(self_base_url):
            name = inference.active_model_name
            if not name:
                return None
            info = inference.models.get(name) or {}
            is_vision = bool(info.get("is_vision", False))
            source = "unsloth" if info.get("is_lora") else "transformers"
            if not self_base_url:
                return VlmCapability(
                    is_vlm = False,
                    endpoint_url = None,
                    model_name = name,
                    source = source,
                    reason = "cannot self-loopback: request base URL unavailable",
                )
            return VlmCapability(
                is_vlm = is_vision,
                endpoint_url = self_base_url.rstrip("/"),
                model_name = name,
                source = source,
                reason = None if is_vision else "loaded model is not vision-capable",
            )

        monkeypatch.setattr(vc, "_probe_transformers", probe_tf)


def test_detect_returns_none_when_no_model_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_probes(monkeypatch, llama = None, inference = None)
    cap = detect_loaded_vlm()
    assert cap.source == "none"
    assert cap.is_vlm is False


def test_detect_gguf_vision_returns_llama_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llama = _FakeLlama(loaded = True, vision = True, base_url = "http://127.0.0.1:9999")
    _patch_probes(monkeypatch, llama = llama, inference = None)
    cap = detect_loaded_vlm("http://studio.local")
    assert cap.source == "gguf"
    assert cap.is_vlm is True
    assert cap.endpoint_url == "http://127.0.0.1:9999"  # GGUF ignores self_base_url
    assert cap.reason is None


def test_detect_gguf_vision_accepts_injected_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.chat import vlm_capability as vc

    llama = _FakeLlama(loaded = True, vision = True, base_url = "http://127.0.0.1:9999")
    monkeypatch.setattr(vc, "_probe_transformers", lambda _u: None)

    cap = detect_loaded_vlm(
        "http://127.0.0.1:8000",
        llama_backend = llama,
    )

    assert cap.source == "gguf"
    assert cap.is_vlm is True
    assert cap.endpoint_url == "http://127.0.0.1:9999"


def test_detect_gguf_vision_uses_core_llama_accessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The implicit GGUF fallback must use the core-owned singleton path."""
    from core.chat import vlm_capability as vc
    from core.inference import llama_cpp

    llama = _FakeLlama(loaded = True, vision = True, base_url = "http://127.0.0.1:9999")
    assert hasattr(llama_cpp, "get_llama_cpp_backend")
    monkeypatch.setattr(llama_cpp, "_llama_cpp_backend", llama)
    monkeypatch.setattr(vc, "_probe_transformers", lambda _u: None)

    cap = detect_loaded_vlm("http://127.0.0.1:8000")

    assert cap.source == "gguf"
    assert cap.is_vlm is True
    assert cap.endpoint_url == "http://127.0.0.1:9999"


def test_detect_gguf_non_vision_surfaces_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llama = _FakeLlama(loaded = True, vision = False)
    _patch_probes(monkeypatch, llama = llama, inference = None)
    cap = detect_loaded_vlm()
    assert cap.source == "gguf"
    assert cap.is_vlm is False
    assert cap.reason and "vision" in cap.reason.lower()


def test_detect_transformers_vision_uses_self_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ib = _FakeInferenceBackend(
        active = "Qwen2-VL-7B", info = {"is_vision": True, "is_lora": False},
    )
    _patch_probes(monkeypatch, llama = None, inference = ib)
    cap = detect_loaded_vlm("http://127.0.0.1:8000/")
    assert cap.source == "transformers"
    assert cap.is_vlm is True
    assert cap.endpoint_url == "http://127.0.0.1:8000"
    assert cap.model_name == "Qwen2-VL-7B"


def test_detect_unsloth_lora_vision_reports_unsloth_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ib = _FakeInferenceBackend(
        active = "my-qwen-vl-lora", info = {"is_vision": True, "is_lora": True},
    )
    _patch_probes(monkeypatch, llama = None, inference = ib)
    cap = detect_loaded_vlm("http://studio.local:8000")
    assert cap.source == "unsloth"
    assert cap.is_vlm is True


def test_detect_falls_through_when_gguf_is_loaded_but_endpoint_data_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A half-initialised llama-server (is_loaded=True but base_url/model
    missing) must not suppress the transformers fallback path — otherwise
    a misleading non-vision GGUF result hides an active transformers VLM.
    """
    from core.chat import vlm_capability as vc

    fake_llama_cpp = ModuleType("core.inference.llama_cpp")
    fake_llama_cpp.get_llama_cpp_backend = lambda: _FakeLlama(
        loaded = True, base_url = "", model_id = "",
    )
    fake_inference = ModuleType("core.inference")
    fake_inference.__path__ = []  # type: ignore[attr-defined]
    fake_inference.llama_cpp = fake_llama_cpp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "core.inference", fake_inference)
    monkeypatch.setitem(sys.modules, "core.inference.llama_cpp", fake_llama_cpp)

    ib = _FakeInferenceBackend(
        active = "Qwen2-VL-7B", info = {"is_vision": True, "is_lora": False},
    )
    monkeypatch.setattr(
        vc,
        "_probe_transformers",
        lambda self_base_url: VlmCapability(
            is_vlm = True,
            endpoint_url = self_base_url.rstrip("/") if self_base_url else None,
            model_name = ib.active_model_name,
            source = "transformers",
            reason = None,
        ),
    )

    cap = detect_loaded_vlm("http://127.0.0.1:8000")
    assert cap.source == "transformers"
    assert cap.is_vlm is True


def test_detect_transformers_without_self_url_reports_missing_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ib = _FakeInferenceBackend(
        active = "Qwen2-VL-7B", info = {"is_vision": True, "is_lora": False},
    )
    _patch_probes(monkeypatch, llama = None, inference = ib)
    cap = detect_loaded_vlm(None)
    assert cap.is_vlm is False
    assert cap.reason and "loopback" in cap.reason.lower()


# ---------------------------------------------------------------------- #
# extract_self_base_url — request base-URL extraction                    #
# ---------------------------------------------------------------------- #


class _FakeState:
    def __init__(self, server_port: Optional[int] = None) -> None:
        if server_port is not None:
            self.server_port = server_port


class _FakeApp:
    def __init__(self, server_port: Optional[int] = None) -> None:
        self.state = _FakeState(server_port)


class _FakeRequest:
    def __init__(
        self,
        base_url: str,
        *,
        server_port: Optional[int] = None,
        scope_server: Optional[tuple[str, int]] = None,
    ) -> None:
        self.base_url = base_url
        self.app = _FakeApp(server_port)
        self.scope = {"server": scope_server} if scope_server else {}


def test_extract_self_base_url_strips_trailing_slash() -> None:
    assert (
        extract_self_base_url(_FakeRequest("http://127.0.0.1:8000/"))
        == "http://127.0.0.1:8000"
    )


def test_extract_self_base_url_prefers_trusted_server_port() -> None:
    assert (
        extract_self_base_url(
            _FakeRequest(
                "http://attacker.invalid:9999/",
                server_port = 7777,
                scope_server = ("127.0.0.1", 6666),
            )
        )
        == "http://127.0.0.1:7777"
    )
    assert (
        extract_self_base_url(
            _FakeRequest(
                "http://attacker.invalid:9999/",
                scope_server = ("127.0.0.1", 6666),
            )
        )
        == "http://127.0.0.1:6666"
    )


def test_extract_self_base_url_ignores_host_header() -> None:
    assert (
        extract_self_base_url(_FakeRequest("http://studio.local:8000/"))
        == "http://127.0.0.1:8000"
    )
    assert (
        extract_self_base_url(_FakeRequest("https://example.com:9443/"))
        == "http://127.0.0.1:9443"
    )


def test_extract_self_base_url_none_when_empty() -> None:
    assert extract_self_base_url(_FakeRequest("")) is None


def test_extract_self_base_url_none_on_missing_attribute() -> None:
    assert extract_self_base_url(object()) is None


# ---------------------------------------------------------------------- #
# extract_document orchestration — backend-agnostic (monkey-patched)     #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_max_figures_zero_sets_describe_skipped_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_figures=0 must skip description with a specific diagnostic even
    when a VLM is available."""
    from core.chat import document_extractor as de

    def fake_extract(_fb, _fn, _opts, _ct = ""):
        return "# Smoke\n", [], 1, 0, 0

    monkeypatch.setattr(de, "DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(de, "_run_extract_sync", fake_extract)

    result = await de.extract_document(
        b"# Smoke\n",
        "sample.md",
        describe_images = True,
        max_figures = 0,
        capability = VlmCapability(
            is_vlm = True,
            endpoint_url = "http://127.0.0.1:8000",
            model_name = "vlm",
            source = "transformers",
        ),
    )

    assert result.describe_skipped_reason == (
        "figure description disabled because max_figures is 0"
    )
    assert result.markdown == "# Smoke\n"
    assert result.figures == []


@pytest.mark.asyncio
async def test_run_extract_sync_seam_receives_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The test seam path (monkeypatched _run_extract_sync) must be invoked
    with the content_type so dispatch-by-content-type can be exercised in
    tests, not only by filename suffix."""
    from core.chat import document_extractor as de

    received: dict[str, str] = {}

    def fake_extract(_fb, _fn, _opts, ct = ""):
        received["content_type"] = ct
        return "ok", [], 0, 0, 0

    monkeypatch.setattr(de, "DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(de, "_run_extract_sync", fake_extract)

    await de.extract_document(
        b"hello",
        "no-suffix-file",
        content_type = "text/plain",
        describe_images = False,
    )
    assert received["content_type"] == "text/plain"


@pytest.mark.asyncio
async def test_describe_image_via_vlm_sends_auth_header_and_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.chat import document_extractor as de

    captured: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "A chart."}}]}

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    fake_httpx = ModuleType("httpx")
    fake_httpx.AsyncClient = FakeAsyncClient
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    caption, error = await de._describe_image_via_vlm(
        image_base64 = "abc",
        image_mime = "image/jpeg",
        endpoint_url = "http://127.0.0.1:8000",
        model_name = "vlm",
        authorization_header = "Bearer token",
        timeout_seconds = 7,
    )

    assert caption == "A chart."
    assert error is None
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer token"
    assert captured["json"]["max_tokens"] == 512
    assert "max_completion_tokens" not in captured["json"]


# ---------------------------------------------------------------------- #
# Backend dispatch — real _run_extract_sync (requires pymupdf/mammoth)   #
# ---------------------------------------------------------------------- #


_BACKEND_INSTALLED = (
    importlib.util.find_spec("pymupdf") is not None
    and importlib.util.find_spec("pymupdf4llm") is not None
    and importlib.util.find_spec("mammoth") is not None
)


def test_run_extract_sync_rejects_pptx_with_value_error() -> None:
    """PPTX was dropped in the PyMuPDF4LLM migration. _run_extract_sync
    must raise ValueError so the route can map it to HTTP 415."""
    if not _BACKEND_INSTALLED:
        pytest.skip("extraction backend not installed")
    from core.chat import document_extractor as de

    with pytest.raises(ValueError):
        de._run_extract_sync(
            b"PK\x03\x04",
            "deck.pptx",
            {"max_figures": 0, "extract_images": False, "use_vlm_ocr": False},
        )


def test_run_extract_sync_text_path_decodes_utf8() -> None:
    """TXT / MD paths must not require PDF/DOCX parser dependencies."""
    from core.chat import document_extractor as de

    md, figs, pages, trunc, seen = de._run_extract_sync(
        "# Héllo\n".encode("utf-8"),
        "notes.md",
        {"max_figures": 0, "extract_images": False, "use_vlm_ocr": False},
    )
    assert md == "# Héllo\n"
    assert figs == []
    assert pages == 0 and trunc == 0 and seen == 0


def test_run_extract_sync_html_converts_to_markdown_without_parser_deps() -> None:
    """HTML must be cleaned before prompt injection and not depend on PDF/DOCX deps."""
    from core.chat import document_extractor as de

    md, figs, pages, trunc, seen = de._run_extract_sync(
        b"<html><head><style>.x{}</style></head><body><h1>Title</h1><script>x()</script><p>Hello <b>world</b></p></body></html>",
        "page.html",
        {"max_figures": 0, "extract_images": False, "use_vlm_ocr": False},
    )
    assert "# Title" in md
    assert "**world**" in md
    assert "<script>" not in md
    assert figs == []
    assert pages == 0 and trunc == 0 and seen == 0


# ---------------------------------------------------------------------- #
# Multi-figure encoding cap, partial VLM failure, timeout                #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_multi_figure_extraction_encoded_visuals_capped_at_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only _MAX_ENCODED_VISUALS (3) figures may have image_base64 set;
    remaining figures beyond the cap must have image_base64=None."""
    from core.chat import document_extractor as de
    from core.chat.document_extractor import ExtractedFigure

    def fake_extract(_fb, _fn, _opts, _ct = ""):
        figs = [
            ExtractedFigure(
                id=f"fig-{i}",
                page=i + 1,
                caption=None,
                kind="figure",
                image_mime="image/jpeg" if i < de._MAX_ENCODED_VISUALS else None,
                image_base64="b64" if i < de._MAX_ENCODED_VISUALS else None,
                image_width=10,
                image_height=10,
            )
            for i in range(5)
        ]
        return "# Multi\n", figs, 5, 0, 5

    monkeypatch.setattr(de, "DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(de, "_run_extract_sync", fake_extract)

    result = await de.extract_document(
        b"dummy",
        "doc.pdf",
        describe_images=False,
        max_figures=10,
        capability=VlmCapability.none(),
    )

    encoded = [f for f in result.figures if f.image_base64 is not None]
    assert len(encoded) <= de._MAX_ENCODED_VISUALS
    assert len(result.figures) == 5
    assert any("first 3 visual payloads" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_multi_figure_extraction_respects_configured_visual_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller can raise the image-byte cap up to the server safety maximum."""
    from core.chat import document_extractor as de
    from core.chat.document_extractor import ExtractedFigure

    def fake_extract(_fb, _fn, opts, _ct = ""):
        max_visuals = opts["max_visual_payloads"]
        figs = [
            ExtractedFigure(
                id=f"fig-{i}",
                page=i + 1,
                caption=None,
                kind="figure",
                image_mime="image/jpeg" if i < max_visuals else None,
                image_base64="b64" if i < max_visuals else None,
                image_width=10,
                image_height=10,
            )
            for i in range(6)
        ]
        return "# Multi\n", figs, 6, 0, 6

    monkeypatch.setattr(de, "DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(de, "_run_extract_sync", fake_extract)

    result = await de.extract_document(
        b"dummy",
        "doc.pdf",
        describe_images=False,
        max_figures=10,
        max_visual_payloads=5,
        capability=VlmCapability.none(),
    )

    encoded = [f for f in result.figures if f.image_base64 is not None]
    assert len(encoded) == 5
    assert any("first 5 visual payloads" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_partial_vlm_failure_records_per_figure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When one describe call raises, only the failing figure gets an
    error; the others still receive captions."""
    from core.chat import document_extractor as de
    from core.chat.document_extractor import ExtractedFigure

    def fake_extract(_fb, _fn, _opts, _ct = ""):
        figs = [
            ExtractedFigure(
                id=f"fig-{i}",
                page=i + 1,
                caption=None,
                kind="figure",
                image_mime="image/jpeg",
                image_base64="b64",
                image_width=10,
                image_height=10,
            )
            for i in range(3)
        ]
        return "# Doc\n", figs, 3, 0, 3

    call_idx: Dict[str, int] = {"n": 0}

    async def fake_describe(
        *,
        image_base64,
        image_mime,
        endpoint_url,
        model_name,
        authorization_header,
        timeout_seconds,
    ):
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx == 1:
            raise RuntimeError("VLM exploded on figure 1")
        return f"caption-{idx}", None

    monkeypatch.setattr(de, "DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(de, "_run_extract_sync", fake_extract)
    monkeypatch.setattr(de, "_describe_image_via_vlm", fake_describe)

    cap = VlmCapability(
        is_vlm=True,
        endpoint_url="http://127.0.0.1:9999",
        model_name="vlm",
        source="gguf",
        reason=None,
    )
    result = await de.extract_document(
        b"dummy",
        "doc.pdf",
        describe_images=True,
        max_figures=10,
        capability=cap,
    )

    figs = [f for f in result.figures if f.kind == "figure"]
    assert len(figs) == 3

    errored = [f for f in figs if f.error is not None]
    assert len(errored) == 1
    assert "RuntimeError" in errored[0].error or "VLM" in errored[0].error

    captioned = [f for f in figs if f.error is None and f.caption is not None]
    assert len(captioned) == 2


@pytest.mark.asyncio
async def test_local_vlm_captioning_serializes_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from core.chat import document_extractor as de
    from core.chat.document_extractor import ExtractedFigure

    def fake_extract(_fb, _fn, _opts, _ct = ""):
        figs = [
            ExtractedFigure(
                id=f"fig-{i}",
                page=i + 1,
                caption=None,
                kind="figure",
                image_mime="image/jpeg",
                image_base64="b64",
                image_width=10,
                image_height=10,
            )
            for i in range(3)
        ]
        return "# Doc\n", figs, 3, 0, 3

    active = 0
    max_active = 0

    async def fake_describe(**_kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return "caption", None

    monkeypatch.setattr(de, "DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(de, "_run_extract_sync", fake_extract)
    monkeypatch.setattr(de, "_describe_image_via_vlm", fake_describe)

    result = await de.extract_document(
        b"dummy",
        "doc.pdf",
        describe_images=True,
        max_figures=10,
        capability=VlmCapability(
            is_vlm=True,
            endpoint_url="http://127.0.0.1:8000",
            model_name="vlm",
            source="transformers",
            reason=None,
        ),
    )

    assert max_active == 1
    assert all(figure.caption == "caption" for figure in result.figures)


@pytest.mark.asyncio
async def test_local_vlm_captioning_respects_configured_visual_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.chat import document_extractor as de
    from core.chat.document_extractor import ExtractedFigure

    def fake_extract(_fb, _fn, opts, _ct = ""):
        max_visuals = opts["max_visual_payloads"]
        figs = []
        for i in range(5):
            has_payload = i < max_visuals
            figs.append(
                ExtractedFigure(
                    id=f"fig-{i}",
                    page=i + 1,
                    caption=None,
                    kind="figure",
                    image_mime="image/jpeg" if has_payload else None,
                    image_base64="b64" if has_payload else None,
                    image_width=10 if has_payload else None,
                    image_height=10 if has_payload else None,
                )
            )
        return "# Doc\n", figs, 5, 0, 5

    async def fake_describe(**_kwargs):
        return "caption", None

    monkeypatch.setattr(de, "DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(de, "_run_extract_sync", fake_extract)
    monkeypatch.setattr(de, "_describe_image_via_vlm", fake_describe)

    result = await de.extract_document(
        b"dummy",
        "doc.pdf",
        describe_images=True,
        max_figures=5,
        max_visual_payloads=5,
        capability=VlmCapability(
            is_vlm=True,
            endpoint_url="http://127.0.0.1:8000",
            model_name="vlm",
            source="transformers",
            reason=None,
        ),
    )

    captioned = [figure for figure in result.figures if figure.caption]
    assert len(captioned) == 5
    assert not any("Local VLM captioning is limited" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_extraction_timeout_raises_document_extraction_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _run_extract_sync exceeds the wall-clock limit,
    DocumentExtractionTimeout must be raised — not raw asyncio.TimeoutError."""
    import asyncio as _asyncio
    from core.chat import document_extractor as de
    from core.chat.document_extractor import DocumentExtractionTimeout

    def fake_extract(_fb, _fn, _opts, _ct = ""):
        return "# Doc\n", [], 0, 0, 0

    async def fake_wait_for(coro, timeout):
        try:
            coro.close()
        except Exception:
            pass
        raise _asyncio.TimeoutError()

    monkeypatch.setattr(de, "DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(de, "_run_extract_sync", fake_extract)
    monkeypatch.setattr(_asyncio, "wait_for", fake_wait_for)

    with pytest.raises(DocumentExtractionTimeout):
        await de.extract_document(
            b"dummy",
            "doc.pdf",
            describe_images=False,
            capability=VlmCapability.none(),
        )


# ---------------------------------------------------------------------- #
# Format dispatch via extract_document (DOCX / TXT)                      #
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_docx_path_uses_mammoth_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DOCX route must return whatever mammoth produces, with no figures."""
    from core.chat import document_extractor as de

    def fake_extract(_fb, filename, _opts, _ct = ""):
        assert filename.endswith(".docx")
        return "**bold** text", [], 0, 0, 0

    monkeypatch.setattr(de, "DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(de, "_run_extract_sync", fake_extract)

    result = await de.extract_document(
        b"PK\x03\x04",
        "notes.docx",
        describe_images=False,
        capability=VlmCapability.none(),
    )
    assert result.markdown == "**bold** text"
    assert result.figures == []


@pytest.mark.asyncio
async def test_use_vlm_ocr_emits_warning_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """use_vlm_ocr=True is accepted for API compatibility but this build
    ships no OCR engine — the extractor must surface a warning."""
    from core.chat import document_extractor as de

    def fake_extract(_fb, _fn, _opts, _ct = ""):
        return "# Doc\n", [], 1, 0, 0

    monkeypatch.setattr(de, "DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(de, "_run_extract_sync", fake_extract)

    result = await de.extract_document(
        b"dummy",
        "scan.pdf",
        describe_images=False,
        use_vlm_ocr=True,
        capability=VlmCapability.none(),
    )
    assert any("OCR" in w for w in result.warnings)
