# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The /audio/stt/download route must validate a custom Transformers repo before
snapshot_download pulls it into the shared HF cache.

Regression for a Codex finding: the Transformers engine accepts arbitrary
`owner/model` repos, so an authenticated caller could make Unsloth download a
large non-STT repository before load-time validation ever ran. Whisper-
compatibility is now enforced (metadata-only, no weights) before the background
download starts. Curated ids short-circuit the check, including the ones a retired
whisper.cpp (``gguf``) request now downloads through Transformers.
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

import core.inference.stt_audiocpp_sidecar as audiocpp_module  # noqa: E402
import core.inference.stt_ggml_sidecar as ggml_module  # noqa: E402
import core.inference.stt_mtmd_sidecar as mtmd_module  # noqa: E402
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


def test_gguf_download_goes_through_transformers_with_whisper_server_installed(monkeypatch):
    """whisper.cpp is retired, so even a healthy whisper-server never downloads a .bin."""
    gguf_started: list = []
    tf_started: list = []
    validated: list = []

    monkeypatch.setattr(ggml_module, "is_available", lambda: True)
    monkeypatch.setattr(stt_module, "is_available", lambda: True)
    monkeypatch.setattr(
        stt_module,
        "validate_remote_model",
        lambda model, hf_token = None: validated.append(model) or {"model": model},
    )
    monkeypatch.setattr(
        stt_module,
        "start_model_download",
        lambda model, hf_token = None, revision = None: tf_started.append(model),
    )
    monkeypatch.setattr(stt_module, "download_status", lambda: {"downloading": True})
    monkeypatch.setattr(
        ggml_module, "start_model_download", lambda model, hf_token = None: gguf_started.append(model)
    )

    resp = _run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "gguf"),
            current_subject = "tester",
            hf_token = None,
        )
    )

    assert resp.status_code == 200
    assert validated == ["small"]
    assert tf_started == ["small"]
    assert gguf_started == []


def test_resolve_serving_stt_engine_never_serves_whisper_cpp(monkeypatch):
    # The retired alias goes to Transformers when PyTorch is here, else audio.cpp, else
    # llama.cpp, else Transformers' 501; whisper-server being installed never matters.
    monkeypatch.setattr(ggml_module, "is_available", lambda: True)
    monkeypatch.setattr(stt_module, "is_available", lambda: True)
    monkeypatch.setattr(audiocpp_module, "is_available", lambda: True)
    monkeypatch.setattr(mtmd_module, "is_available", lambda: True)
    assert ri._resolve_serving_stt_engine("gguf") == "transformers"
    monkeypatch.setattr(stt_module, "is_available", lambda: False)
    assert ri._resolve_serving_stt_engine("gguf") == "audiocpp"
    monkeypatch.setattr(audiocpp_module, "is_available", lambda: False)
    assert ri._resolve_serving_stt_engine("gguf") == "mtmd"
    monkeypatch.setattr(mtmd_module, "is_available", lambda: False)
    assert ri._resolve_serving_stt_engine("gguf") == "transformers"
    # Transformers is unaffected by any of it.
    assert ri._resolve_serving_stt_engine("transformers") == "transformers"


def test_gguf_download_falls_back_to_transformers_when_server_absent(monkeypatch):
    """Selecting the default curated model on a host without whisper-server must
    download through the Transformers engine, not 501/dead-end on GGUF."""
    gguf_started: list = []
    tf_started: list = []

    monkeypatch.setattr(ggml_module, "is_available", lambda: False)  # no whisper-server
    monkeypatch.setattr(stt_module, "is_available", lambda: True)
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
