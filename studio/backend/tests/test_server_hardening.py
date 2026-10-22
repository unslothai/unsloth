# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Server hardening: login limiter key size, session log modes, vision probe argv, owner-only
llama-server Hub flags, video data URI payloads and managed embeddings input."""

import asyncio
import os
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend not in sys.path:
    sys.path.insert(0, _backend)


# Login limiter key


def test_login_bucket_key_is_bounded_for_long_usernames():
    from routes import auth as auth_route

    huge = "A" * (1 << 20)
    ip, name = auth_route._bucket_key(None, huge)
    assert len(name) < 128
    # Same name, same bucket; different names, different buckets.
    assert auth_route._bucket_key(None, huge.lower()) == (ip, name)
    assert auth_route._bucket_key(None, huge + "b")[1] != name
    # Normal usernames keep their casefolded key.
    assert auth_route._bucket_key(None, "Alice") == ("_unknown", "alice")
    assert auth_route._bucket_key(None, "x" * 256) == ("_unknown", "x" * 256)


# Session log modes


@pytest.mark.skipif(os.name == "nt", reason = "POSIX file modes")
def test_server_session_log_is_owner_only(tmp_path, monkeypatch):
    import faulthandler

    import run as run_mod

    home = tmp_path / "studio"
    old_dir = home / "logs" / "server"
    old_dir.mkdir(parents = True)
    os.chmod(old_dir, 0o755)
    old_log = old_dir / "server-20200101-000000-pid1.log"
    old_log.write_text("old session\n")
    os.chmod(old_log, 0o644)

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.delenv("UNSLOTH_STUDIO_NO_FILE_LOG", raising = False)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    previous = os.umask(0o022)
    try:
        log_path = run_mod._setup_server_disk_logging()
        # Idempotent: a second start leaves the same modes.
        second = run_mod._setup_server_disk_logging()
    finally:
        os.umask(previous)
        faulthandler.disable()
    assert log_path is not None and second is not None
    assert log_path.parent == old_dir
    for path in (log_path, second, old_log):
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, path
    assert stat.S_IMODE(old_dir.stat().st_mode) & 0o077 == 0
    assert old_log.read_text() == "old session\n"


# Vision probe argv


def test_vision_probe_passes_the_token_in_env_not_argv(monkeypatch):
    from utils.models import model_config

    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = list(command)
        seen["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(command, 0, stdout = '{"is_vision": false}', stderr = "")

    monkeypatch.setattr(model_config.subprocess, "run", fake_run)
    model_config._is_vision_model_subprocess("org/some-model", hf_token = "hf_secret_value_123")
    assert seen, "probe did not run"
    assert not any("hf_secret_value_123" in str(arg) for arg in seen["command"])
    # The argv slot stays so the child script's layout is unchanged.
    assert seen["command"][6] == ""
    assert seen["env"].get("HF_TOKEN") == "hf_secret_value_123"


# llama-server Hub drafter and preset flags


@pytest.mark.parametrize(
    "args",
    [
        ["-hfd", "owner/private-draft-GGUF"],
        ["-hfrd", "owner/private-draft-GGUF"],
        ["--hf-repo-draft=owner/private-draft-GGUF"],
        ["--spec-draft-hf", "owner/private-draft-GGUF"],
        ["--fim-qwen-1.5b-default"],
        ["--embd-gemma-default"],
        ["--gpt-oss-20b-default"],
        ["--vision-gemma-4b-default"],
        ["--fim-qwen-7b-spec"],
    ],
)
def test_hub_drafter_and_preset_flags_are_owner_only(args, monkeypatch):
    from core.inference import llama_server_args as lsa

    assert lsa.owner_only_path_flags(args), args

    inference_route = pytest.importorskip("routes.inference")
    monkeypatch.setattr(inference_route.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(inference_route, "_owner_chosen_launch", lambda *a, **k: set())
    with pytest.raises(HTTPException) as exc:
        inference_route._refuse_managed_custom_projector(args, "org/model")
    assert exc.value.status_code == 403


def test_spec_default_and_tuning_flags_stay_open():
    from core.inference import llama_server_args as lsa
    assert lsa.owner_only_path_flags(["--spec-default", "--ctx-size", "4096", "-ngl", "99"]) == []


# Video data URI payload


def test_video_data_uri_must_carry_payload_not_a_url():
    inference_route = pytest.importorskip("routes.inference")
    reject = inference_route._video_scheme_rejection
    for clip in (
        "data:video/mp4;base64,http://127.0.0.1:9/x.mp4",
        "data:video/mp4;base64,HTTPS://10.0.0.1/x.mp4",
        "data:video/mp4;base64,data:video/mp4;base64,http://127.0.0.1:9/x",
        "data:video/mp4;base64,file:///etc/passwd",
        "data:video/mp4;" + "x" * 300 + ",http://127.0.0.1:9/x.mp4",
    ):
        rejection = reject(clip)
        assert rejection is not None and rejection[0] == 400, clip
    # A real clip and the shapes handled elsewhere are unchanged.
    assert reject("data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28y" + "A" * 64) is None
    assert reject("") is None
    assert reject("http://example.com/x.mp4") is None


# Managed embeddings input


class _Request:
    def __init__(self, body):
        self._body = body
        self.method = "POST"
        self.url = SimpleNamespace(path = "/v1/embeddings")
        self.state = SimpleNamespace(skip_api_monitor = True)
        self.scope = {"type": "http", "path": "/v1/embeddings"}
        self.headers = {}

    async def json(self):
        return self._body

    async def is_disconnected(self):
        return False


def _embeddings_proxy(monkeypatch, managed):
    import httpx

    inference_route = pytest.importorskip("routes.inference")
    posted = []

    class _Client:
        async def post(
            self,
            url,
            json = None,
            **_kwargs,
        ):
            posted.append(json)
            return httpx.Response(200, json = {"data": [{"embedding": [0.5]}]})

        async def aclose(self):
            return None

    async def passthrough(request, current_subject, **_kwargs):
        return await request.json()

    async def no_studio_request(request):
        return None

    monkeypatch.setattr(inference_route.account_access, "managed_account", lambda: managed)
    monkeypatch.setattr(inference_route, "_should_validate_before_switch", lambda: False)
    monkeypatch.setattr(inference_route, "_studio_embedder_request_body", no_studio_request)
    monkeypatch.setattr(inference_route, "_auto_switch_from_request_body", passthrough)
    monkeypatch.setattr(inference_route, "_cancelable_nonstreaming_client", _Client)
    monkeypatch.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,
            is_embedding_gguf = True,
            base_url = "http://llama.test",
            context_length = 512,
            model_identifier = "org/E-GGUF",
        ),
    )
    monkeypatch.setattr(inference_route, "_resident_serves_embeddings", lambda backend: True)
    return inference_route, posted


_IMAGE_ITEM = {
    "input": [
        {"content": [{"type": "image_url", "image_url": {"url": "http://127.0.0.1:9/a.png"}}]}
    ],
    "model": "org/E-GGUF",
}


def test_managed_embeddings_refuse_multimodal_items(monkeypatch):
    inference_route, posted = _embeddings_proxy(monkeypatch, managed = True)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_embeddings(_Request(dict(_IMAGE_ITEM)), "member"))
    assert exc.value.status_code == 400
    assert posted == []
    # Text still goes through for a managed account.
    asyncio.run(
        inference_route.openai_embeddings(
            _Request({"input": ["alpha"], "model": "org/E-GGUF"}), "member"
        )
    )
    assert posted and posted[-1]["input"] == ["alpha"]


def test_owner_embeddings_keep_multimodal_items(monkeypatch):
    inference_route, posted = _embeddings_proxy(monkeypatch, managed = False)
    asyncio.run(inference_route.openai_embeddings(_Request(dict(_IMAGE_ITEM)), "owner"))
    assert posted and posted[-1]["input"] == _IMAGE_ITEM["input"]
