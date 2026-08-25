# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The /audio/stt/download route must validate a custom Transformers repo before
snapshot_download pulls it into the shared HF cache.

Regression for a Codex finding: the Transformers engine accepts arbitrary
`owner/model` repos, so an authenticated caller could make Studio download a
large non-STT repository before load-time validation ever ran. Whisper-
compatibility is now enforced (metadata-only, no weights) before the background
download starts. The GGUF engine only accepts curated ids, so it is not gated.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import core.inference.stt_ggml_sidecar as ggml_module  # noqa: E402
import core.inference.stt_sidecar as stt_module  # noqa: E402
import routes.inference as ri  # noqa: E402
from core.inference.stt_sidecar import SttModelCompatibilityError  # noqa: E402
from models.inference import SttLoadRequest  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def test_custom_non_whisper_repo_is_rejected_before_download(monkeypatch):
    started: list = []
    validated: list = []

    def fake_validate(model, hf_token = None):
        validated.append(model)
        raise SttModelCompatibilityError(
            f"STT model '{model}' is not a compatible Transformers Whisper model."
        )

    def fake_download(model, hf_token = None):
        started.append(model)

    monkeypatch.setattr(stt_module, "validate_remote_model", fake_validate)
    monkeypatch.setattr(stt_module, "start_model_download", fake_download)

    with pytest.raises(HTTPException) as excinfo:
        _run(
            ri.stt_download(
                SttLoadRequest(model = "owner/chat-model", engine = "transformers"),
                current_subject = "tester",
                hf_token = None,
            )
        )

    assert excinfo.value.status_code == 422
    assert validated == ["owner/chat-model"]
    # The download never starts for a repo that failed the Whisper check.
    assert started == []


def test_validated_transformers_repo_downloads(monkeypatch):
    started: list = []
    revision = "a" * 40

    monkeypatch.setattr(
        stt_module,
        "validate_remote_model",
        lambda model, hf_token = None: {"model": model, "revision": revision},
    )
    monkeypatch.setattr(
        stt_module,
        "start_model_download",
        lambda model, hf_token = None, revision = None: started.append((model, revision)),
    )
    monkeypatch.setattr(stt_module, "download_status", lambda: {"downloading": True})

    resp = _run(
        ri.stt_download(
            SttLoadRequest(model = "owner/real-whisper", engine = "transformers"),
            current_subject = "tester",
            hf_token = None,
        )
    )

    assert resp.status_code == 200
    assert started == [("owner/real-whisper", revision)]


def test_gguf_engine_skips_the_transformers_repo_check(monkeypatch):
    started: list = []

    def fail_if_called(model, hf_token = None):
        raise AssertionError("GGUF downloads must not run the Transformers repo check")

    # whisper-server present, so the GGUF request stays on the GGUF engine.
    monkeypatch.setattr(ggml_module, "is_available", lambda: True)
    monkeypatch.setattr(stt_module, "validate_remote_model", fail_if_called)
    monkeypatch.setattr(
        ggml_module, "start_model_download", lambda model, hf_token = None: started.append(model)
    )
    monkeypatch.setattr(ggml_module, "download_status", lambda: {"downloading": True})

    resp = _run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "gguf"),
            current_subject = "tester",
            hf_token = None,
        )
    )

    assert resp.status_code == 200
    assert started == ["small"]


def test_resolve_serving_stt_engine_falls_back_when_whisper_server_absent(monkeypatch):
    # A curated GGUF request downgrades to Transformers when whisper-server is not
    # installed (both engines serve curated ids), but stays GGUF when it is.
    monkeypatch.setattr(ggml_module, "is_available", lambda: False)
    assert ri._resolve_serving_stt_engine("gguf") == "transformers"
    monkeypatch.setattr(ggml_module, "is_available", lambda: True)
    assert ri._resolve_serving_stt_engine("gguf") == "gguf"
    # Transformers is unaffected by whisper-server availability.
    monkeypatch.setattr(ggml_module, "is_available", lambda: False)
    assert ri._resolve_serving_stt_engine("transformers") == "transformers"


def test_gguf_download_falls_back_to_transformers_when_server_absent(monkeypatch):
    """Selecting the default curated model on a host without whisper-server must
    download through the Transformers engine, not 501/dead-end on GGUF."""
    gguf_started: list = []
    tf_started: list = []

    monkeypatch.setattr(ggml_module, "is_available", lambda: False)  # no whisper-server
    # validate_remote_model no-ops curated ids in production; keep it a no-op here.
    monkeypatch.setattr(
        stt_module, "validate_remote_model", lambda model, hf_token = None: {"model": model}
    )
    monkeypatch.setattr(
        stt_module,
        "start_model_download",
        lambda model, hf_token = None, revision = None: tf_started.append(model),
    )
    monkeypatch.setattr(stt_module, "download_status", lambda: {"downloading": True})
    monkeypatch.setattr(
        ggml_module,
        "start_model_download",
        lambda model, hf_token = None: gguf_started.append(model),
    )

    resp = _run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "gguf"),
            current_subject = "tester",
            hf_token = None,
        )
    )

    assert resp.status_code == 200
    assert tf_started == ["small"]  # served by Transformers instead of dead-ending on GGUF
    assert gguf_started == []
