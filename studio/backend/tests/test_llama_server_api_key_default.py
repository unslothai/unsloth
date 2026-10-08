# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server runs with a per-launch API key by default, and every Studio request to it sends the key."""

import ast
import os
import stat
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import llama_cpp  # noqa: E402
from core.inference import llama_stats  # noqa: E402


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, True),
        ({"UNSLOTH_LLAMA_SERVER_API_KEY": "0"}, False),
        ({"UNSLOTH_LLAMA_SERVER_API_KEY": "1"}, True),
        ({"UNSLOTH_LLAMA_SERVER_API_KEY": "0", "UNSLOTH_DIRECT_STREAM": "1"}, True),
        ({"UNSLOTH_DIRECT_STREAM": "1"}, True),
    ],
)
def test_key_is_on_by_default(monkeypatch, env, expected):
    monkeypatch.delenv("UNSLOTH_LLAMA_SERVER_API_KEY", raising = False)
    monkeypatch.delenv("UNSLOTH_DIRECT_STREAM", raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert llama_cpp._llama_server_api_key_enabled() is expected


def test_key_file_is_per_backend_private_and_rewritten_on_relaunch(monkeypatch, tmp_path):
    import utils.paths.storage_roots as roots

    monkeypatch.setattr(roots, "auth_root", lambda: tmp_path)
    first = llama_cpp._write_direct_stream_key("one")
    second = llama_cpp._write_direct_stream_key("two", first)
    other = llama_cpp._write_direct_stream_key("three")
    assert second == first and other != first
    assert second.read_text() == "two" and other.read_text() == "three"
    assert second.name.startswith("llama_api_key_")
    if os.name == "posix":
        assert stat.S_IMODE(second.stat().st_mode) == 0o600


def test_tool_guard_still_refuses_the_per_launch_key_file():
    from core.inference import tools

    assert tools._references_studio_credential("cat /tmp/x/llama_api_key_0123abcd")
    assert tools._references_studio_credential("cat /tmp/x/llama_api_key")
    assert not tools._references_studio_credential("print('llama_api_key_x')")


def test_stats_scrape_sends_the_key(monkeypatch):
    seen = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""

    def _urlopen(request, timeout = None):
        seen["auth"] = request.get_header("Authorization")
        return _Resp()

    monkeypatch.setattr(llama_stats.urllib.request, "urlopen", _urlopen)
    logger = llama_stats.LlamaServerStatsLogger(
        "http://127.0.0.1:1", None, headers = {"Authorization": "Bearer k"}
    )
    logger._scrape()
    assert seen["auth"] == "Bearer k"


def _llama_requests(tree):
    """Every build_request / post in routes/inference.py whose URL is a llama-server target."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("build_request", "post"):
            continue
        names = {a.id for a in node.args if isinstance(a, ast.Name)}
        if names & {"target_url", "url"} and not names & {"provider_url"}:
            yield node


def test_every_llama_request_in_inference_routes_sends_auth():
    src = (BACKEND / "routes" / "inference.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    calls = list(_llama_requests(tree))
    assert len(calls) >= 10, "the call-site scan matched too few requests to mean anything"
    missing = []
    for call in calls:
        headers = next((kw.value for kw in call.keywords if kw.arg == "headers"), None)
        text = ast.unparse(headers) if headers is not None else ""
        if "upstream_headers" not in text and "_openai_passthrough_upstream_headers" not in text:
            missing.append(f"line {call.lineno}: {ast.unparse(call)[:120]}")
    assert not missing, "llama-server requests without the backend auth header:\n" + "\n".join(
        missing
    )
