# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import io
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason = "route helper tests require FastAPI")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.datastructures import Headers  # noqa: E402
import core.chat.document_extractor as extractor  # noqa: E402
from core.chat.vlm_capability import VlmCapability  # noqa: E402
from models.inference import ChatMessage  # noqa: E402
from routes import inference as route  # noqa: E402


class _ChunkedUpload:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, _size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _FakeRequest:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class _FakeStreamingRequest:
    def __init__(
        self,
        chunks: list[bytes],
        headers: Headers | None = None,
    ) -> None:
        self._chunks = list(chunks)
        self.headers = headers or Headers({})

    async def stream(self):
        for chunk in self._chunks:
            yield chunk


def test_reject_oversized_content_length_allows_missing_header() -> None:
    route._reject_oversized_content_length(_FakeRequest({}))


def test_reject_oversized_content_length_rejects_large_request() -> None:
    max_request_bytes = (
        route._EXTRACT_MAX_BYTES + route._EXTRACT_MULTIPART_OVERHEAD_BYTES + 1
    )
    with pytest.raises(HTTPException) as exc_info:
        route._reject_oversized_content_length(
            _FakeRequest({"content-length": str(max_request_bytes)})
        )
    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_read_upload_limited_rejects_streaming_overflow() -> None:
    upload = _ChunkedUpload([b"a" * 4, b"b" * 4, b"c"])
    with pytest.raises(HTTPException) as exc_info:
        await route._read_upload_limited(upload, max_bytes = 8)
    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_read_multipart_form_limited_rejects_streaming_overflow() -> None:
    boundary = "studio-boundary"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="doc.md"\r\n'
            "Content-Type: text/markdown\r\n"
            "\r\n"
        ).encode()
        + b"a" * 32
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = _FakeStreamingRequest(
        [body[:16], body[16:]],
        Headers({"Content-Type": f"multipart/form-data; boundary={boundary}"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        await route._read_multipart_form_limited(request, max_bytes = 16)

    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_read_json_body_limited_rejects_streaming_overflow() -> None:
    request = _FakeStreamingRequest([b'{"a":', b'"bc"}'])
    with pytest.raises(HTTPException) as exc_info:
        await route._read_json_body_limited(request, max_bytes = 7)
    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_read_json_body_limited_reports_bad_json() -> None:
    request = _FakeStreamingRequest([b"{bad"])
    with pytest.raises(HTTPException) as exc_info:
        await route._read_json_body_limited(request, max_bytes = 100)
    assert exc_info.value.status_code == 400
    assert "Invalid JSON body" in exc_info.value.detail


@pytest.mark.asyncio
async def test_read_json_body_limited_accepts_empty_body() -> None:
    request = _FakeStreamingRequest([])
    assert await route._read_json_body_limited(request, max_bytes = 100) == {}


def test_document_extraction_exports_are_available_to_routes() -> None:
    assert route._DOCUMENT_EXTRACTION_AVAILABLE is True
    assert route._extract_document is not None
    assert route._DOCUMENT_EXTRACT_CONCURRENCY >= 1
    assert route._DOC_SUFFIX_OK
    assert ".pdf" in route._DOC_SUFFIX_OK
    assert route._drain_doc_future_exception is extractor._drain_future_exception


def test_chat_body_limit_covers_document_visual_payload_budget() -> None:
    expected_image_slots = max(
        1,
        min(
            route._OPENAI_CHAT_MAX_IMAGES,
            route._MAX_DOCUMENT_VISUAL_PAYLOADS
            or route._DEFAULT_DOCUMENT_VISUAL_PAYLOADS
            or 1,
        ),
    )
    assert route._OPENAI_CHAT_BODY_IMAGE_SLOTS == expected_image_slots
    assert route._OPENAI_CHAT_BODY_MAX_BYTES >= (
        route._OPENAI_CHAT_MAX_IMAGE_BASE64_CHARS * expected_image_slots
    )


def test_extract_process_zero_queue_wait_admits_available_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQueue:
        def __init__(self, *, maxsize: int) -> None:
            assert maxsize == 1

        def get(self, *, timeout: float):
            assert timeout > 0
            return ("ok", ("plain text", [], 0, 0, 0))

        def close(self) -> None:
            pass

        def join_thread(self) -> None:
            pass

    class FakeProcess:
        exitcode = 0

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return False

        def join(self, _timeout: float) -> None:
            pass

        def terminate(self) -> None:
            raise AssertionError("process should not be terminated")

        def kill(self) -> None:
            raise AssertionError("process should not be killed")

    class FakeContext:
        def Queue(self, *, maxsize: int) -> FakeQueue:  # noqa: N802 - mirrors mp API
            return FakeQueue(maxsize = maxsize)

        def Process(self, *, target, args, daemon: bool) -> FakeProcess:  # noqa: N802
            assert target is extractor._run_extract_worker
            assert args[1] == b"plain text"
            assert args[2] == "sample.txt"
            assert daemon is True
            return FakeProcess()

    monkeypatch.setattr(extractor, "_EXTRACT_QUEUE_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(
        extractor,
        "_EXTRACT_SEMAPHORE",
        threading.BoundedSemaphore(1),
    )
    monkeypatch.setattr(
        extractor.multiprocessing,
        "get_context",
        lambda _method: FakeContext(),
    )

    assert extractor._run_extract_process_sync(
        b"plain text",
        "sample.txt",
        {"extract_images": False},
        "text/plain",
        5,
    ) == ("plain text", [], 0, 0, 0)


def test_openai_chat_completions_rejects_oversized_body_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.dependency_overrides[route.get_current_subject] = lambda: "test-user"
    app.include_router(route.router, prefix = "/v1")
    monkeypatch.setattr(route, "_OPENAI_CHAT_BODY_MAX_BYTES", 20)

    response = TestClient(app).post(
        "/v1/chat/completions",
        content = b'{"messages":[{"role":"user","content":"' + b"x" * 64 + b'"}]}',
        headers = {"Content-Type": "application/json"},
    )

    assert response.status_code == 413


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("", True),
        ("yes", True),
        ("OFF", False),
        ("0", False),
    ],
)
def test_parse_bool_form_accepts_known_tokens(value, expected) -> None:
    assert route._parse_bool_form(value, default = True, field = "flag") is expected


def test_describe_images_form_field_missing_defaults_to_off() -> None:
    """When describe_images is absent/empty the server default must be False."""
    assert route._parse_bool_form(None, default = False, field = "describe_images") is False
    assert route._parse_bool_form("", default = False, field = "describe_images") is False


def test_parse_bool_form_rejects_unknown_token() -> None:
    with pytest.raises(HTTPException) as exc_info:
        route._parse_bool_form("bogus", default = True, field = "describe_images")
    assert exc_info.value.status_code == 400
    assert "describe_images" in exc_info.value.detail


def test_truncate_markdown_caps_returned_payload() -> None:
    markdown = "word " * 2000
    clipped, tokens_est, warning = route._truncate_markdown_to_token_budget(
        markdown,
        token_budget = 1000,
        original_tokens_est = len(markdown) // 4,
    )
    assert len(clipped) < len(markdown)
    assert tokens_est == len(clipped) // 4
    assert warning and "truncated" in warning


def test_parse_int_form_defaults_invalid_and_clamps_bounds() -> None:
    assert route._parse_int_form("bogus", default = 40, lo = 0, hi = 200) == 40
    assert route._parse_int_form("-1", default = 40, lo = 0, hi = 200) == 0
    assert route._parse_int_form("999", default = 40, lo = 0, hi = 200) == 200
    assert route._parse_int_form("999999", default = 40, lo = 0) == 999999


def test_extract_content_parts_preserves_multiple_image_parts() -> None:
    message = ChatMessage(
        role = "user",
        content = [
            {"type": "text", "text": "Explain these."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,one"},
            },
            {"type": "text", "text": "Second:"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,two"},
            },
        ],
    )

    system_prompt, chat_messages, image_b64s = route._extract_content_parts([message])

    assert system_prompt == ""
    assert chat_messages == [
        {"role": "user", "content": "Explain these.\nSecond:"},
    ]
    assert image_b64s == ["one", "two"]


def test_preflight_pdf_page_count_uses_pypdf(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePdfReader:
        def __init__(self, _stream, *, strict: bool) -> None:
            assert strict is False
            self.is_encrypted = False
            self.pages = [object(), object(), object()]

    fake_pypdf = ModuleType("pypdf")
    fake_pypdf.PdfReader = FakePdfReader
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    assert (
        route._preflight_pdf_page_count(
            b"%PDF",
            "paper.pdf",
            "application/pdf",
        )
        == 3
    )


def test_preflight_pdf_page_count_falls_back_to_pymupdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenPdfReader:
        def __init__(self, _stream, *, strict: bool) -> None:
            raise ValueError("xref is odd")

    class FakeDocument:
        is_encrypted = False
        needs_pass = False

        def __len__(self) -> int:
            return 4

        def close(self) -> None:
            pass

    fake_pypdf = ModuleType("pypdf")
    fake_pypdf.PdfReader = BrokenPdfReader
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)
    fake_pymupdf = ModuleType("pymupdf")
    fake_pymupdf.open = lambda *, stream, filetype: FakeDocument()
    monkeypatch.setitem(sys.modules, "pymupdf", fake_pymupdf)

    assert (
        route._preflight_pdf_page_count(
            b"%PDF",
            "paper.pdf",
            "application/pdf",
        )
        == 4
    )


def test_preflight_pdf_page_count_skips_non_pdf() -> None:
    assert route._preflight_pdf_page_count(b"text", "notes.md", "text/markdown") is None


def test_validate_model_returns_trc_requirement_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.dependency_overrides[route.get_current_subject] = lambda: "test-user"
    app.include_router(route.router, prefix = "/api/inference")

    def fake_defaults(model_name: str) -> dict:
        assert model_name == "deepseek-ai/DeepSeek-OCR"
        return {
            "model": {"display_name": "DeepSeek-OCR", "is_vision": True},
            "inference": {"trust_remote_code": True},
        }

    def fail_probe(*_args, **_kwargs):
        raise AssertionError("validation should not probe custom code before opt-in")

    monkeypatch.setattr(route, "load_model_defaults", fake_defaults)
    monkeypatch.setattr(route.ModelConfig, "from_identifier", fail_probe)

    response = TestClient(app).post(
        "/api/inference/validate",
        json = {"model_path": "deepseek-ai/DeepSeek-OCR"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["requires_trust_remote_code"] is True
    assert body["is_vision"] is True


def test_legacy_generate_stream_registers_client_cancel_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.dependency_overrides[route.get_current_subject] = lambda: "test-user"
    app.include_router(route.router, prefix = "/api/inference")
    seen: dict[str, set[str]] = {}

    class FakeBackend:
        active_model_name = "test-model"
        models = {"test-model": {"is_vision": False}}

        def generate_chat_response(self, **kwargs):
            cancel_event = kwargs["cancel_event"]
            with route._CANCEL_LOCK:
                seen["keys"] = {
                    key
                    for key, bucket in route._CANCEL_REGISTRY.items()
                    if cancel_event in bucket
                }
            yield "hello"

        def reset_generation_state(self) -> None:
            pass

    with route._CANCEL_LOCK:
        route._CANCEL_REGISTRY.clear()
        route._PENDING_CANCELS.clear()
    monkeypatch.setattr(route, "get_inference_backend", lambda: FakeBackend())

    response = TestClient(app).post(
        "/api/inference/generate/stream",
        json = {
            "messages": [{"role": "user", "content": "Hello"}],
            "cancel_id": "cancel-1",
            "session_id": "session-1",
        },
    )

    assert response.status_code == 200
    assert "completion_id" in response.text
    assert "hello" in response.text
    assert {"cancel-1", "session-1"}.issubset(seen["keys"])
    assert any(key.startswith("legacy-") for key in seen["keys"])
    with route._CANCEL_LOCK:
        assert route._CANCEL_REGISTRY == {}


def test_extract_document_endpoint_streams_ndjson_with_caption_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the client sends `Accept: application/x-ndjson`, the
    endpoint streams progress events plus a final `{stage:"result"}`."""
    import json as _json

    app = FastAPI()
    app.dependency_overrides[route.get_current_subject] = lambda: "test-user"
    app.include_router(route.studio_router, prefix = "/api/inference")

    async def fake_extract_document(*_args, **kwargs):
        # Emit a parsing event then two captioning events to simulate
        # per-figure progress, then return a minimal result.
        progress_cb = kwargs.get("progress_cb")
        if progress_cb is not None:
            await progress_cb({"stage": "parsing"})
            await progress_cb(
                {
                    "stage": "captioning",
                    "current": 1,
                    "total": 2,
                    "page": 1,
                    "total_pages": 3,
                }
            )
            await progress_cb(
                {
                    "stage": "captioning",
                    "current": 2,
                    "total": 2,
                    "page": 2,
                    "total_pages": 3,
                }
            )
        return SimpleNamespace(
            markdown = "# Stream\n",
            page_count = 3,
            tokens_est = 5,
            figures = [],
            describe_skipped_reason = None,
            vlm_source = "none",
            vlm_model = None,
            warnings = [],
        )

    monkeypatch.setattr(route, "_DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(route, "_extract_document", fake_extract_document)
    monkeypatch.setattr(
        route,
        "_extract_self_base_url",
        lambda _request: "http://127.0.0.1:8000",
    )
    monkeypatch.setattr(
        route,
        "_detect_loaded_vlm",
        lambda *_args, **_kwargs: VlmCapability.none("no model loaded"),
    )

    client = TestClient(app)
    response = client.post(
        "/api/inference/chat/extract-document",
        headers = {
            "Authorization": "Bearer test-token",
            "Accept": "application/x-ndjson",
        },
        data = {"describe_images": "false"},
        files = {"file": ("sample.md", b"# Stream\n", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [_json.loads(line) for line in response.text.splitlines() if line.strip()]
    stages = [e.get("stage") for e in events]
    assert "parsing" in stages
    captioning_events = [e for e in events if e.get("stage") == "captioning"]
    assert len(captioning_events) >= 2
    assert captioning_events[0]["current"] == 1
    assert captioning_events[0]["total"] == 2
    assert captioning_events[0]["page"] == 1
    assert captioning_events[0]["total_pages"] == 3
    assert events[-1]["stage"] == "result"
    assert events[-1]["data"]["markdown"] == "# Stream\n"
    assert events[-1]["data"]["page_count"] == 3


def test_extract_document_endpoint_accepts_multipart_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.dependency_overrides[route.get_current_subject] = lambda: "test-user"
    app.include_router(route.studio_router, prefix = "/api/inference")

    captured: dict[str, object] = {}

    async def fake_extract_document(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            markdown = "# Smoke\n",
            page_count = 1,
            tokens_est = 2,
            figures = [],
            describe_skipped_reason = None,
            vlm_source = "none",
            vlm_model = None,
            warnings = [],
        )

    monkeypatch.setattr(route, "_DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(route, "_extract_document", fake_extract_document)
    monkeypatch.setattr(
        route,
        "_extract_self_base_url",
        lambda _request: "http://127.0.0.1:8000",
    )
    monkeypatch.setattr(
        route,
        "_detect_loaded_vlm",
        lambda *_args, **_kwargs: VlmCapability.none("no model loaded"),
    )

    client = TestClient(app)
    response = client.post(
        "/api/inference/chat/extract-document",
        headers = {"Authorization": "Bearer test-token"},
        data = {
            "describe_images": "false",
            "max_figures": "12345",
            "max_visual_payloads": "222",
        },
        files = {"file": ("sample.md", b"# Smoke\n", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["markdown"] == "# Smoke\n"
    assert response.json()["truncated"] is False
    assert captured["authorization_header"] == "Bearer test-token"
    assert captured["content_type"] == "text/markdown"
    assert captured["max_figures"] == 12345
    assert captured["max_visual_payloads"] == 222


def test_extract_document_endpoint_does_not_globally_gate_on_pdf_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_extract_document(*_args, **_kwargs):
        return SimpleNamespace(
            markdown = "# Text\n",
            page_count = 1,
            tokens_est = 2,
            figures = [],
            describe_skipped_reason = None,
            vlm_source = "none",
            vlm_model = None,
            warnings = [],
        )

    client = _make_app(monkeypatch, fake_extract = fake_extract_document)
    monkeypatch.setattr(route, "_DOCUMENT_EXTRACTION_AVAILABLE", False)
    response = client.post(
        "/api/inference/chat/extract-document",
        files = {"file": ("sample.md", b"# Text\n", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["markdown"] == "# Text\n"


def test_extract_document_endpoint_uses_llama_api_key_for_gguf_captions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.dependency_overrides[route.get_current_subject] = lambda: "test-user"
    app.include_router(route.studio_router, prefix = "/api/inference")

    captured: dict[str, object] = {}

    async def fake_extract_document(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            markdown = "# Smoke\n",
            page_count = 1,
            tokens_est = 2,
            figures = [],
            describe_skipped_reason = None,
            vlm_source = "gguf",
            vlm_model = "vision.gguf",
            warnings = [],
        )

    llama_backend = SimpleNamespace(api_key = "llama-secret")
    monkeypatch.setattr(route, "_extract_document", fake_extract_document)
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: llama_backend)
    monkeypatch.setattr(
        route,
        "_extract_self_base_url",
        lambda _request: "http://127.0.0.1:8000",
    )
    monkeypatch.setattr(
        route,
        "_detect_loaded_vlm",
        lambda *_args, **_kwargs: VlmCapability(
            is_vlm = True,
            endpoint_url = "http://127.0.0.1:8080",
            model_name = "vision.gguf",
            source = "gguf",
        ),
    )

    client = TestClient(app)
    response = client.post(
        "/api/inference/chat/extract-document",
        headers = {"Authorization": "Bearer studio-token"},
        data = {"describe_images": "true"},
        files = {"file": ("sample.md", b"# Smoke\n", "text/markdown")},
    )

    assert response.status_code == 200
    assert captured["authorization_header"] == "Bearer llama-secret"


def test_extract_document_endpoint_maps_busy_worker_to_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.dependency_overrides[route.get_current_subject] = lambda: "test-user"
    app.include_router(route.studio_router, prefix = "/api/inference")

    async def busy_extract_document(*_args, **_kwargs):
        raise route._DocumentExtractionBusy("document extraction is busy")

    monkeypatch.setattr(route, "_DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(route, "_extract_document", busy_extract_document)
    monkeypatch.setattr(
        route,
        "_extract_self_base_url",
        lambda _request: "http://127.0.0.1:8000",
    )
    monkeypatch.setattr(
        route,
        "_detect_loaded_vlm",
        lambda *_args, **_kwargs: VlmCapability.none("no model loaded"),
    )

    client = TestClient(app)
    response = client.post(
        "/api/inference/chat/extract-document",
        files = {"file": ("sample.md", b"# Smoke\n", "text/markdown")},
    )

    assert response.status_code == 503


def test_extract_document_endpoint_maps_value_error_to_415(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.dependency_overrides[route.get_current_subject] = lambda: "test-user"
    app.include_router(route.studio_router, prefix = "/api/inference")

    async def fake_extract_document(*_args, **_kwargs):
        raise ValueError("Unsupported file type: upload.bin")

    monkeypatch.setattr(route, "_DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(route, "_extract_document", fake_extract_document)
    monkeypatch.setattr(
        route,
        "_extract_self_base_url",
        lambda _request: "http://127.0.0.1:8000",
    )
    monkeypatch.setattr(
        route,
        "_detect_loaded_vlm",
        lambda *_args, **_kwargs: VlmCapability.none("no model loaded"),
    )

    client = TestClient(app)
    response = client.post(
        "/api/inference/chat/extract-document",
        files = {"file": ("upload.bin", b"hello", "text/plain")},
    )

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_extract_document_endpoint_maps_parse_value_error_to_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_extract_document(*_args, **_kwargs):
        raise ValueError("Could not parse document")

    client = _make_app(monkeypatch, fake_extract = fake_extract_document)
    response = client.post(
        "/api/inference/chat/extract-document",
        files = {"file": ("upload.md", b"# hello", "text/markdown")},
    )

    assert response.status_code == 400
    assert "Could not parse document" in response.json()["detail"]


def test_extract_document_endpoint_reports_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_extract_document(*_args, **_kwargs):
        return SimpleNamespace(
            markdown = "word " * 2000,
            page_count = 1,
            tokens_est = 2500,
            figures = [],
            describe_skipped_reason = None,
            vlm_source = "none",
            vlm_model = None,
            warnings = [],
        )

    client = _make_app(monkeypatch, fake_extract_document)
    response = client.post(
        "/api/inference/chat/extract-document",
        data = {"token_budget": "1000"},
        files = {"file": ("sample.md", b"# Smoke\n", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["truncated"] is True
    assert any("truncated" in w.lower() for w in response.json()["warnings"])


def test_extract_document_endpoint_sanitizes_extract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.dependency_overrides[route.get_current_subject] = lambda: "test-user"
    app.include_router(route.studio_router, prefix = "/api/inference")

    async def fake_extract_document(*_args, **_kwargs):
        raise RuntimeError("local path C:/secret/model/cache leaked")

    monkeypatch.setattr(route, "_DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(route, "_extract_document", fake_extract_document)
    monkeypatch.setattr(
        route,
        "_extract_self_base_url",
        lambda _request: "http://127.0.0.1:8000",
    )
    monkeypatch.setattr(
        route,
        "_detect_loaded_vlm",
        lambda *_args, **_kwargs: VlmCapability.none("no model loaded"),
    )

    client = TestClient(app)
    response = client.post(
        "/api/inference/chat/extract-document",
        files = {"file": ("sample.md", b"# Smoke\n", "text/markdown")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Extraction failed"


def _make_app(monkeypatch: pytest.MonkeyPatch, fake_extract = None):
    """Helper: create a FastAPI test app with extraction stubs applied."""
    app = FastAPI()
    app.dependency_overrides[route.get_current_subject] = lambda: "test-user"
    app.include_router(route.studio_router, prefix = "/api/inference")
    monkeypatch.setattr(route, "_DOCUMENT_EXTRACTION_AVAILABLE", True)
    monkeypatch.setattr(
        route,
        "_extract_self_base_url",
        lambda _request: "http://127.0.0.1:8000",
    )
    monkeypatch.setattr(
        route,
        "_detect_loaded_vlm",
        lambda *_args, **_kwargs: VlmCapability.none("no model loaded"),
    )
    if fake_extract is not None:
        monkeypatch.setattr(route, "_extract_document", fake_extract)
    return TestClient(app)


def test_document_support_reports_format_parser_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_app(monkeypatch)
    monkeypatch.setattr(
        route,
        "_document_parser_support",
        lambda: {"pdf": False, "docx": True, "text": True},
    )
    monkeypatch.setattr(
        route,
        "_document_parser_unavailable_reasons",
        lambda: {"pdf": "PDF extraction requires pymupdf and pymupdf4llm."},
    )

    response = client.get("/api/inference/chat/document-support")

    assert response.status_code == 200
    body = response.json()
    assert body["extraction_available"] is True
    assert body["max_extract_concurrency"] == route._DOCUMENT_EXTRACT_CONCURRENCY
    assert body["format_support"]["pdf"] is False
    assert body["format_support"]["text"] is True
    assert "pymupdf" in body["unavailable_formats"]["pdf"]


def test_document_support_maps_vlm_probe_bug_to_no_vlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_app(monkeypatch)
    monkeypatch.setattr(
        route,
        "_detect_loaded_vlm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.get("/api/inference/chat/document-support")

    assert response.status_code == 200
    body = response.json()
    assert body["extraction_available"] is True
    assert body["vlm"]["is_vlm"] is False
    assert "RuntimeError" in body["vlm"]["reason"]


def test_endpoint_rejects_unavailable_pdf_parser_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_extract(*_args, **_kwargs):
        raise AssertionError("unavailable parser should be rejected before extraction")

    client = _make_app(monkeypatch, fake_extract = fail_extract)
    monkeypatch.setattr(route, "_document_parser_support", lambda: {"pdf": False})
    monkeypatch.setattr(
        route,
        "_document_parser_unavailable_reasons",
        lambda: {"pdf": "PDF extraction requires pymupdf and pymupdf4llm."},
    )

    response = client.post(
        "/api/inference/chat/extract-document",
        files = {"file": ("paper.pdf", b"%PDF", "application/pdf")},
    )

    assert response.status_code == 501
    assert "pymupdf" in response.json()["detail"]


def test_413_message_does_not_mention_roadmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 413 detail must not promise background job support."""
    monkeypatch.setattr(route, "_EXTRACT_MAX_PAGES_INLINE", 1)

    class FakePdfReader:
        def __init__(self, _stream, *, strict: bool) -> None:
            self.is_encrypted = False
            self.pages = [object(), object(), object()]  # 3 pages > cap of 1

    fake_pypdf = ModuleType("pypdf")
    fake_pypdf.PdfReader = FakePdfReader
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    client = _make_app(monkeypatch)
    response = client.post(
        "/api/inference/chat/extract-document",
        files = {"file": ("paper.pdf", b"%PDF", "application/pdf")},
    )

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "roadmap" not in detail.lower()
    assert "split" in detail.lower() or "smaller" in detail.lower()


def test_figures_are_serialized_via_pydantic_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ExtractedFigureModel(**asdict(fig)) must be used so a field-name
    mismatch in the dataclass surfaces as a validation error, not a
    silently-wrong response."""
    from core.chat.document_extractor import ExtractedFigure

    async def fake_extract(*_args, **_kwargs):
        return SimpleNamespace(
            markdown = "# Doc\n",
            page_count = 1,
            tokens_est = 3,
            figures = [
                ExtractedFigure(
                    id = "fig-0",
                    page = 1,
                    caption = "A chart",
                    error = None,
                    kind = "figure",
                    image_mime = None,
                    image_base64 = None,
                    image_width = None,
                    image_height = None,
                )
            ],
            describe_skipped_reason = None,
            vlm_source = "none",
            vlm_model = None,
            warnings = [],
        )

    client = _make_app(monkeypatch, fake_extract = fake_extract)
    response = client.post(
        "/api/inference/chat/extract-document",
        data = {"describe_images": "false"},
        files = {"file": ("doc.md", b"# Doc\n", "text/markdown")},
    )

    assert response.status_code == 200
    figs = response.json()["figures"]
    assert len(figs) == 1
    assert figs[0]["id"] == "fig-0"
    assert figs[0]["caption"] == "A chart"


def test_extraction_timeout_returns_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.chat.document_extractor import DocumentExtractionTimeout

    async def fake_extract(*_args, **_kwargs):
        raise DocumentExtractionTimeout("timed out")

    monkeypatch.setattr(
        route,
        "_DocumentExtractionTimeout",
        DocumentExtractionTimeout,
    )
    client = _make_app(monkeypatch, fake_extract = fake_extract)
    response = client.post(
        "/api/inference/chat/extract-document",
        data = {"describe_images": "false"},
        files = {"file": ("doc.md", b"# Doc\n", "text/markdown")},
    )

    assert response.status_code == 504
    assert "120" in response.json()["detail"]


def test_encrypted_extraction_returns_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_extract(*_args, **_kwargs):
        raise route._DocumentExtractionEncrypted("Encrypted PDF")

    client = _make_app(monkeypatch, fake_extract = fake_extract)
    response = client.post(
        "/api/inference/chat/extract-document",
        data = {"describe_images": "false"},
        files = {"file": ("doc.md", b"# Doc\n", "text/markdown")},
    )

    assert response.status_code == 422
    assert "Encrypted PDF" in response.json()["detail"]


def test_real_encrypted_pdf_preflight_returns_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width = 72, height = 72)
    writer.encrypt("secret")
    encrypted = io.BytesIO()
    writer.write(encrypted)

    async def fail_extract(*_args, **_kwargs):
        raise AssertionError("encrypted PDFs should fail during preflight")

    client = _make_app(monkeypatch, fake_extract = fail_extract)
    response = client.post(
        "/api/inference/chat/extract-document",
        data = {"describe_images": "false"},
        files = {
            "file": ("locked.pdf", encrypted.getvalue(), "application/pdf"),
        },
    )

    assert response.status_code == 422
    assert "Encrypted PDF" in response.json()["detail"]


def test_cancelled_extraction_returns_499(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_extract(*_args, **_kwargs):
        raise route._DocumentExtractionCancelled("cancelled")

    client = _make_app(monkeypatch, fake_extract = fake_extract)
    response = client.post(
        "/api/inference/chat/extract-document",
        data = {"describe_images": "false"},
        files = {"file": ("doc.md", b"# Doc\n", "text/markdown")},
    )

    assert response.status_code == 499
    assert response.json()["detail"] == "Client closed request"


def test_endpoint_returns_415_for_unsupported_mime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_app(monkeypatch)
    response = client.post(
        "/api/inference/chat/extract-document",
        files = {"file": ("image.png", b"\x89PNG", "image/png")},
    )
    assert response.status_code == 415


def test_endpoint_returns_400_for_empty_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_app(monkeypatch)
    response = client.post(
        "/api/inference/chat/extract-document",
        files = {"file": ("empty.md", b"", "text/markdown")},
    )
    assert response.status_code == 400


def test_endpoint_returns_501_when_extraction_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.chat.document_extractor import DocumentExtractionUnavailable

    async def fake_extract(*_args, **_kwargs):
        raise DocumentExtractionUnavailable("document extraction is not installed")

    monkeypatch.setattr(
        route,
        "_DocumentExtractionUnavailable",
        DocumentExtractionUnavailable,
    )
    client = _make_app(monkeypatch, fake_extract = fake_extract)
    response = client.post(
        "/api/inference/chat/extract-document",
        data = {"describe_images": "false"},
        files = {"file": ("doc.md", b"# Doc\n", "text/markdown")},
    )
    assert response.status_code == 501


def test_endpoint_returns_415_for_pptx(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_app(monkeypatch)
    response = client.post(
        "/api/inference/chat/extract-document",
        files = {
            "file": (
                "deck.pptx",
                b"PK\x03\x04",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
        },
    )
    assert response.status_code == 415
