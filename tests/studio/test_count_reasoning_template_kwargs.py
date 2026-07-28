# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The recount must render the template in the selected reasoning mode (#7453).

llama-server merges the load-time ``--chat-template-kwargs`` under whatever a
request omits, so ``/apply-template`` called with only messages and tools renders
the mode the model was LAUNCHED in. Studio launches every reasoning GGUF with a
fixed default (thinking on/off by family and size, ``preserve_thinking`` always
false) and then lets the user move both from the composer, forwarding
``enable_thinking`` / ``reasoning_effort`` / ``preserve_thinking`` as
``chat_template_kwargs`` on the completion. A count that drops them therefore
prices a prompt the next request never sends.

The stub llama-server below reproduces the branch the shipped Qwen3.6 template
takes (``unsloth/Qwen3.6-35B-A3B/chat_template.jinja``)::

    {%- if (preserve_thinking is defined and preserve_thinking is true)
           or (loop.index0 > ns.last_query_index) %}
        {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content
            + '\\n</think>\\n\\n' + content }}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\\n' + content }}
    {%- endif %}

and the launch-default merge llama.cpp performs in
``oaicompat_chat_params_parse``, which ``/apply-template`` shares with
``/v1/chat/completions``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]
BACKEND = WORKDIR / "studio/backend"
ADAPTER = WORKDIR / "studio/frontend/src/features/chat/api/chat-adapter.ts"
CAPABILITIES = WORKDIR / "studio/frontend/src/features/chat/provider-capabilities.ts"
TEMP = WORKDIR / "temp" / "count_reasoning_template_kwargs"

# Studio's launch flag for a preserve_thinking-capable reasoning model, verbatim from
# the llama-server command line it builds (see llama_cpp.py "--chat-template-kwargs").
LAUNCH_DEFAULTS = {"enable_thinking": True, "preserve_thinking": False}

# A thread whose assistant turns carry reasoning. Only the turn after the last user
# message renders its <think> block unconditionally; the earlier ones are gated on
# preserve_thinking.
THREAD = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "<think> a b c d e f g h </think> first answer"},
    {"role": "user", "content": "second question"},
    {"role": "assistant", "content": "<think> i j k l m n o p </think> second answer"},
    {"role": "user", "content": "third question"},
]


class _StubLlamaServer:
    """llama-server stand-in with the Qwen3.6 preserve_thinking branch.

    ``/tokenize`` counts whitespace-separated words. ``/apply-template`` merges the
    request's ``chat_template_kwargs`` over ``LAUNCH_DEFAULTS``, exactly as
    llama.cpp merges them over ``--chat-template-kwargs``.
    """

    def __init__(self):
        stub = self
        self.bodies = []

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body):
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                if self.path == "/tokenize":
                    self._send(200, {"tokens": body.get("content", "").split()})
                    return
                if self.path == "/apply-template":
                    stub.bodies.append(body)
                    self._send(200, {"prompt": stub.render(body)})
                    return
                self._send(404, {"error": "unknown route"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"

    @staticmethod
    def render(body) -> str:
        kwargs = dict(LAUNCH_DEFAULTS)
        kwargs.update(body.get("chat_template_kwargs") or {})
        messages = body.get("messages", [])
        last_user = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            default = -1,
        )
        out = []
        for index, message in enumerate(messages):
            content = str(message.get("content", ""))
            reasoning = ""
            if message.get("role") == "assistant" and "</think>" in content:
                reasoning = content.split("</think>")[0].split("<think>")[-1].strip()
                content = content.split("</think>")[-1].strip()
            keep_thinking = bool(kwargs.get("preserve_thinking")) or index > last_user
            head = f"<|im_start|> {message.get('role', '')} "
            if message.get("role") == "assistant" and keep_thinking:
                out.append(f"{head}<think> {reasoning} </think> {content} <|im_end|>")
            else:
                out.append(f"{head}{content} <|im_end|>")
        out.append(
            "<|im_start|> assistant <think>"
            if kwargs.get("enable_thinking")
            else "<|im_start|> assistant <think> </think>"
        )
        return " ".join(out)

    def __enter__(self):
        self._thread = threading.Thread(target = self._server.serve_forever, daemon = True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout = 5)
        return False


def _route():
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    from routes import inference
    return inference


def _reasoning_backend(server):
    """Backend stub running the real count_chat_tokens + _request_reasoning_kwargs."""
    import types

    from core.inference.llama_cpp import LlamaCppBackend

    class _Backend:
        is_loaded = True
        supports_tools = False
        supports_tool_passthrough = False
        model_identifier = "unsloth/Qwen3.6-35B-A3B-GGUF"
        hf_variant = None
        _openai_advertised_id = None
        # Flags detect_reasoning_flags derives from the Qwen3.6 template.
        _supports_reasoning = True
        _reasoning_style = "enable_thinking"
        _reasoning_always_on = False
        _reasoning_effort_levels: list = []
        _supports_preserve_thinking = True
        _architecture = None

    backend = _Backend()
    backend.base_url = server.base_url
    backend._auth_headers = {}
    backend.count_chat_tokens = types.MethodType(LlamaCppBackend.count_chat_tokens, backend)
    backend._request_reasoning_kwargs = types.MethodType(
        LlamaCppBackend._request_reasoning_kwargs, backend
    )
    return backend


def _counted(server, **payload_fields) -> int:
    """Total /chat/count_tokens reports for THREAD with these reasoning fields."""
    inference = _route()
    from models.inference import ChatCountTokensRequest

    backend = _reasoning_backend(server)
    original = inference.get_llama_cpp_backend
    inference.get_llama_cpp_backend = lambda: backend
    try:
        response = asyncio.run(
            inference.chat_count_tokens(
                ChatCountTokensRequest(model = "m", messages = THREAD, **payload_fields),
                current_subject = "tester",
            )
        )
    finally:
        inference.get_llama_cpp_backend = original
    return json.loads(response.body)["input_tokens"]


def test_preserve_thinking_is_priced_by_the_recount():
    """Turning the Preserve Thinking pill on keeps every past turn's <think> block in
    the real prompt. Counting without the kwarg renders the launch default (off), so
    the bar under-reports by the whole reasoning history."""
    with _StubLlamaServer() as server:
        default_mode = _counted(server)
        preserved = _counted(server, preserve_thinking = True)
        # What the completion will really render with the pill on.
        generated = len(
            _StubLlamaServer.render(
                {"messages": THREAD, "chat_template_kwargs": {"preserve_thinking": True}}
            ).split()
        )

    assert preserved == generated, "the count must match the prompt the completion renders"
    assert preserved > default_mode, (
        "preserve_thinking must add the gated <think> blocks; "
        f"got {preserved} against the launch default {default_mode}"
    )
    # Two gated assistant turns, 8 reasoning words plus the <think></think> pair each.
    assert preserved - default_mode == 20


def test_thinking_off_is_priced_by_the_recount():
    """The Think toggle moves off the launch default with one click and persists, so a
    count that omits enable_thinking prices the launched mode instead."""
    with _StubLlamaServer() as server:
        default_mode = _counted(server)
        thinking_off = _counted(server, enable_thinking = False)
        generated = len(
            _StubLlamaServer.render(
                {"messages": THREAD, "chat_template_kwargs": {"enable_thinking": False}}
            ).split()
        )

    assert thinking_off == generated
    assert thinking_off != default_mode


def test_a_request_without_reasoning_fields_still_counts():
    """Raw callers send no reasoning fields; those must keep inheriting the launch
    default rather than being forced into a mode the completion would not use."""
    with _StubLlamaServer() as server:
        total = _counted(server)
        sent = server.bodies[-1]
    assert total > 0
    assert "chat_template_kwargs" not in sent


# ── Frontend: the recount payload carries the selected settings ────────────


def _require_node() -> None:
    if shutil.which("node") is None:
        pytest.skip("node not available")
    for path in (ADAPTER, CAPABILITIES):
        if not path.exists():
            pytest.skip("studio chat sources not present")
    probe = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 30,
    )
    if probe.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")


def _read(path: Path) -> str:
    return path.read_text(encoding = "utf-8")


def _slice_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start + len(start_marker))
    return text[start:end]


def _count_extras_source() -> str:
    """buildLocalTokenCountExtras and its neighbours in the adapter, verbatim."""
    return _slice_between(
        _read(ADAPTER),
        "function outboundMessagesIncludeImage(",
        "async function resolveUseAdapter(",
    )


def _clamp_source() -> str:
    return _slice_between(
        _read(CAPABILITIES),
        "export function clampReasoningEffortToLevels(",
        "/**\n * Fallback cap for unknown providers",
    )


HARNESS_PRELUDE = """
// @ts-nocheck
// Fixtures the sliced adapter source reads through. Everything below the PRELUDE
// marker is copied verbatim out of the studio sources.
const state: any = {
  supportsTools: false,
  toolsEnabled: false,
  codeToolsEnabled: false,
  artifactsEnabled: false,
  mcpEnabledForChat: false,
  ragEnabled: false,
  ragSource: { type: "kb", kbId: null },
  ragMode: "hybrid",
  ragTopK: 5,
  ragAutoInject: "off",
  ragAutoInjectMinScore: 0,
  autoHealToolCalls: true,
  ggufContextLength: 8192,
  params: { checkpoint: "unsloth/Qwen3.6-35B-A3B-GGUF", maxSeqLength: 8192 },
  supportsReasoning: false,
  reasoningEnabled: false,
  reasoningStyle: "enable_thinking",
  reasoningEffort: "high",
  reasoningEffortLevels: [],
  supportsPreserveThinking: false,
  preserveThinking: false,
};

export const useChatRuntimeStore: any = { getState: () => state };

export function seed(patch: any): void {
  Object.assign(state, patch);
}

async function resolveProjectId(_threadId: string | undefined): Promise<string | null> {
  return null;
}

async function projectHasSources(_projectId: string): Promise<boolean> {
  return false;
}

function resolveAutoInject(mode: string, _checkpoint: string): string {
  return mode;
}

// ---- PRELUDE ENDS: verbatim studio source follows ----
"""


def _harness_source() -> str:
    return HARNESS_PRELUDE + _clamp_source() + "\n" + _count_extras_source()


def _run(script: str) -> dict:
    _require_node()
    TEMP.mkdir(parents = True, exist_ok = True)
    workdir = Path(tempfile.mkdtemp(prefix = "run", dir = str(TEMP)))
    (workdir / "harness.ts").write_text(_harness_source(), encoding = "utf-8")
    (workdir / "run.mts").write_text(script, encoding = "utf-8")
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", "run.mts"],
        cwd = str(workdir),
        capture_output = True,
        text = True,
        timeout = 60,
        env = dict(os.environ, NODE_NO_WARNINGS = "1"),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_count_payload_carries_the_selected_reasoning_settings():
    """The extras builder feeds /chat/count_tokens. A model with tools off is the
    plain case: it used to return an empty object, so the count fell back to the
    launch default while the completion sent the pills the user set."""
    out = _run(
        textwrap.dedent(
            """
            // @ts-nocheck
            import { buildLocalTokenCountExtras, seed } from "./harness.ts";

            seed({
              supportsReasoning: true,
              reasoningEnabled: false,
              reasoningStyle: "enable_thinking",
              supportsPreserveThinking: true,
              preserveThinking: true,
            });
            const withoutTools = await buildLocalTokenCountExtras(undefined, []);

            seed({ supportsTools: true, toolsEnabled: true });
            const withTools = await buildLocalTokenCountExtras(undefined, []);

            seed({
              supportsTools: false,
              toolsEnabled: false,
              reasoningStyle: "reasoning_effort",
              reasoningEnabled: true,
              reasoningEffort: "max",
              reasoningEffortLevels: ["low", "medium", "high"],
              supportsPreserveThinking: false,
            });
            const effortStyle = await buildLocalTokenCountExtras(undefined, []);

            seed({ supportsReasoning: false, reasoningStyle: "enable_thinking" });
            const noReasoning = await buildLocalTokenCountExtras(undefined, []);

            console.log(JSON.stringify({ withoutTools, withTools, effortStyle, noReasoning }));
            """
        )
    )
    # .get(), not [], so dropping the fields reddens on the VALUE the count would
    # send rather than on a KeyError that a renamed key would also raise.
    assert out["withoutTools"].get("enable_thinking") is False
    assert out["withoutTools"].get("preserve_thinking") is True
    # Tools on must not displace the reasoning fields.
    assert out["withTools"].get("enable_thinking") is False
    assert out["withTools"].get("preserve_thinking") is True
    assert out["withTools"].get("enable_tools") is True
    # A stale "max" is clamped to a level this template actually branches on, exactly
    # as the completion request clamps it (first advertised level).
    assert out["effortStyle"].get("reasoning_effort") == "low"
    assert "enable_thinking" not in out["effortStyle"]
    assert "preserve_thinking" not in out["effortStyle"]
    # A model without reasoning controls sends nothing, as before.
    assert out["noReasoning"] == {}
