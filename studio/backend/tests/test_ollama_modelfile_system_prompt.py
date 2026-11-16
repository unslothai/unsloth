# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
import routes.inference as inference_route
from .llama_backend_double import FakeLlamaCppBackend

_SYSTEM = "You are Captain Hook. Answer every question as a pirate."


def _ollama_ref_with_system(tmp_path: Path, monkeypatch) -> str:
    from hub.services.models import ollama

    root = tmp_path / "ollama"
    blobs = root / "blobs"
    blobs.mkdir(parents = True)
    (blobs / f"sha256-{'a' * 64}").write_bytes(b"GGUF-not-really")
    (blobs / f"sha256-{'b' * 64}").write_text(_SYSTEM, encoding = "utf-8")
    tag_file = root / "manifests" / "registry.ollama.ai" / "library" / "pirate" / "latest"
    tag_file.parent.mkdir(parents = True)
    tag_file.write_text(
        json.dumps(
            {
                "config": {},
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": f"sha256:{'a' * 64}",
                    },
                    {
                        "mediaType": "application/vnd.ollama.image.system",
                        "digest": f"sha256:{'b' * 64}",
                    },
                ],
            }
        ),
        encoding = "utf-8",
    )
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    (row,) = ollama.scan_ollama_dir(root)
    return row.load_id


def _sent_messages(tmp_path, monkeypatch, messages):
    sent = {}

    class _OllamaBackend(FakeLlamaCppBackend):
        model_identifier = _ollama_ref_with_system(tmp_path, monkeypatch)

        def generate_chat_completion(self, **kwargs):
            sent["messages"] = kwargs["messages"]
            yield "Arr"

    backend = _OllamaBackend()
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: False)

    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    response = TestClient(app).post(
        "/chat/completions", json = {"messages": messages, "stream": False}
    )
    assert response.status_code == 200, response.text
    return sent["messages"]


def test_modelfile_system_prompt_reaches_the_model(tmp_path, monkeypatch):
    messages = _sent_messages(tmp_path, monkeypatch, [{"role": "user", "content": "Hi"}])

    assert messages[0]["role"] == "system"
    assert _SYSTEM in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Hi"}


def test_request_system_prompt_replaces_modelfile_system_prompt(tmp_path, monkeypatch):
    messages = _sent_messages(
        tmp_path,
        monkeypatch,
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
    )

    system = [m["content"] for m in messages if m["role"] == "system"]
    assert len(system) == 1
    assert "Be brief." in system[0]
    assert _SYSTEM not in system[0]
