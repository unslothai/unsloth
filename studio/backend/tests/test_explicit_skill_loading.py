# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The reporter's mention must reach generation even when the model calls no tools."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import skills
from core.inference.tools import READ_SKILL_TOOL
from routes import inference as api
from .llama_backend_double import FakeLlamaCppBackend


class CapturingBackend(FakeLlamaCppBackend):
    supports_tools = True
    context_length = 8192

    def __init__(self):
        self.requests = []

    def count_chat_tokens(self, *args, **kwargs):
        return 100

    def generate_chat_completion_with_tools(self, **kwargs):
        self.requests.append(kwargs)
        yield {"type": "content", "text": "No model read call."}
        yield {"type": "metadata", "finish_reason": "stop"}

    def generate_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        yield "No model read call."
        yield {"type": "metadata", "finish_reason": "stop"}


@pytest.fixture
def mention_client(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    manifest = root / "skill-creator" / "SKILL.md"
    manifest.parent.mkdir(parents = True)
    manifest.write_text(
        "---\nname: skill-creator\ndescription: Test instructions.\n---\n"
        "Complete manifest sentinel: do not execute or create anything.\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: (("agents", root),))
    monkeypatch.setattr(skills, "studio_root", lambda: tmp_path / "studio")
    monkeypatch.setattr(api, "_AGENT_SKILLS_CACHE", {})
    backend = CapturingBackend()
    monkeypatch.setattr(api, "get_llama_cpp_backend", lambda: backend)
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app), backend, manifest


@pytest.mark.parametrize(
    "prompt",
    [
        "@skill-creator what did i just paste?",
        "wait so @skill-creator didnt load it for you?",
    ],
)
def test_reporter_manifest_is_in_first_generation_without_model_read(mention_client, prompt):
    client, backend, manifest = mention_client
    response = client.post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "enable_tools": True,
            "enabled_tools": ["read_skill"],
            "permission_mode": "auto",
            "max_tokens": 256,
        },
        headers = {"X-Unsloth-Events": "1"},
    )
    assert response.status_code == 200, response.text
    assert backend.requests, response.text
    first_context = json.dumps(backend.requests[0]["messages"], ensure_ascii = False)
    assert manifest.read_text() in "\n".join(
        message.get("content", "") for message in backend.requests[0]["messages"]
    ), f"complete SKILL.md absent from first generation: {first_context}"
    assert '"type": "skill_load"' in response.text
    assert '"status": "loaded"' in response.text
    assert '"type": "tool_start"' not in response.text
    assert [t["function"]["name"] for t in backend.requests[0]["tools"]] == [
        READ_SKILL_TOOL["function"]["name"]
    ]


@pytest.mark.parametrize("gate", ["code_off", "unsupported", "disabled"])
def test_actual_local_request_gates_do_not_load(mention_client, gate):
    client, backend, manifest = mention_client
    if gate == "unsupported":
        backend.supports_tools = False
    if gate == "disabled":
        skills.set_skill_enabled("skill-creator", False)
    response = client.post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "@skill-creator what did i just paste?"}],
            "stream": True,
            "enable_tools": gate != "code_off",
            "enabled_tools": [] if gate == "code_off" else ["read_skill"],
            "permission_mode": "auto",
        },
        headers = {"X-Unsloth-Events": "1"},
    )
    assert response.status_code == 200, response.text
    assert backend.requests
    assert manifest.read_text() not in "\n".join(
        m.get("content", "") for m in backend.requests[0]["messages"]
    )
    assert '"status": "loaded"' not in response.text


@pytest.mark.parametrize("code_on", [False, True])
def test_code_off_mention_loads_and_says_scripts_need_code(mention_client, code_on):
    client, backend, manifest = mention_client
    response = client.post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "@skill-creator what did i just paste?"}],
            "stream": True,
            "enable_tools": True,
            # a Code-off mention sends read_skill alone.
            "enabled_tools": ["read_skill", "python", "terminal"] if code_on else ["read_skill"],
            "permission_mode": "auto",
        },
        headers = {"X-Unsloth-Events": "1"},
    )
    assert response.status_code == 200, response.text
    context = "\n".join(m.get("content", "") for m in backend.requests[0]["messages"])
    assert manifest.read_text() in context
    assert '"status": "loaded"' in response.text
    names = {t["function"]["name"] for t in backend.requests[0]["tools"]}
    assert ("python" in names) is code_on
    assert "create_skill" not in names
    note = "bundled scripts cannot run"
    assert (note in context) is not code_on
    if not code_on:
        assert "Unsloth Studio's local Code tool" in context
