# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from starlette.requests import Request

from core.inference import cloud_compaction
from core.inference import studio_tool_loop
from storage import studio_db

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_external_auto_compaction import _is_summary_call, _long_chat, _proxy  # noqa: E402
from test_external_provider_sampling_over_the_wire import _Handler  # noqa: E402

_TASK = "Run the six steps with the python tool, one call per step."
_STEPS = 6


def _reply(delta: dict) -> bytes:
    return (
        "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n\n"
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if "tool_calls" in delta else "stop",
                    }
                ]
            }
        )
        + "\n\n"
        "data: [DONE]\n\n"
    ).encode()


class _StepModel(_Handler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.recorded.append(body)  # type: ignore[attr-defined]
        messages = body["messages"]
        if _is_summary_call(body):
            payload = _reply({"content": "## Goal\nEarlier questions about words."})
        else:
            asked = next(
                (
                    i
                    for i, m in enumerate(messages)
                    if m["role"] == "user" and str(m["content"]).endswith(_TASK)
                ),
                None,
            )
            seen = [
                int(step)
                for m in messages[asked or 0 :]
                if m["role"] == "tool"
                for step in re.findall(r"^STEP (\d+)", str(m["content"]))
            ]
            if asked is None:
                payload = _reply({"content": "LOST TASK"})
            elif len(seen) < _STEPS:
                call = {
                    "index": 0,
                    "id": f"call_{len(messages)}",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": f"step({len(seen) + 1})"}),
                    },
                }
                payload = _reply({"tool_calls": [call]})
            else:
                payload = _reply({"content": "DONE " + ",".join(map(str, seen))})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(autouse = True)
def _no_remembered_summaries():
    cloud_compaction._memory_summaries.clear()
    cloud_compaction._cut_off_summaries.clear()
    yield
    cloud_compaction._memory_summaries.clear()
    cloud_compaction._cut_off_summaries.clear()


def _run_task(monkeypatch, mode, thread_id):
    runs: list[int] = []

    def execute(name, arguments, **kwargs):
        step = int(re.search(r"\d+", arguments["code"]).group())
        runs.append(step)
        return f"STEP {step} " + "harbor ledger canal " * 120

    async def connected(self) -> bool:
        return False

    monkeypatch.setattr(studio_tool_loop, "execute_tool", execute)
    monkeypatch.setattr(Request, "is_disconnected", connected)
    if thread_id:
        studio_db.upsert_chat_thread(
            {
                "id": thread_id,
                "title": "t",
                "modelType": "base",
                "modelId": "a-model",
                "createdAt": 1,
            }
        )
    recorded: list[dict] = []

    class Recording(_StepModel):
        def do_POST(self) -> None:  # noqa: N802
            self.server.recorded = recorded  # type: ignore[attr-defined]
            super().do_POST()

    messages = _long_chat(turns = 3, turn = "word " * 300)
    messages[-1] = {"role": "user", "content": _TASK}
    chunks, _ = _proxy(
        "openrouter",
        messages,
        handler = Recording,
        thread_id = thread_id,
        enable_tools = True,
        enabled_tools = ["python"],
        permission_mode = "off",
        context_overflow = "truncate_oldest",
        compaction_threshold = 3_000,
        context_window = 4_096,
        compaction_mode = mode,
    )
    reply = "".join(
        (choice.get("delta") or {}).get("content") or ""
        for chunk in chunks
        for choice in chunk.get("choices") or ()
    )
    calls = "".join("S" if _is_summary_call(body) else "C" for body in recorded)
    chats = [body for body in recorded if not _is_summary_call(body)]
    return reply, runs, calls, chats


@pytest.mark.parametrize("mode", ["summary", "rolling", "checkpoint"])
@pytest.mark.parametrize("thread_id", [None, "tool-loop-thread"])
def test_a_tool_loop_keeps_its_request_and_results_when_the_window_fills(
    monkeypatch, mode, thread_id
):
    reply, runs, calls, chats = _run_task(monkeypatch, mode, thread_id)
    assert runs == list(range(1, _STEPS + 1))
    assert reply.endswith("DONE 1,2,3,4,5,6")
    for body in chats:
        assert any(m["role"] == "user" and m["content"].endswith(_TASK) for m in body["messages"])
    # One compaction per reply: its summary calls all come before the same chat request.
    assert "C" not in calls.strip("C")


@pytest.mark.parametrize("nudged", [False, True])
def test_a_tool_loop_whose_call_arguments_fill_the_window_still_evicts_older_exchanges(nudged):
    import routes.inference as ri
    from models.inference import ChatCompletionRequest

    task = "Write the six files with edit_file."
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": task}]
    for n in range(6):
        arguments = json.dumps({"path": f"f{n}.py", "content": "x = compute(a, b)\n" * 120})
        fn = {"name": "edit_file", "arguments": arguments}
        call = {"id": f"c{n}", "type": "function", "function": fn}
        messages += [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"c{n}", "content": "ok"},
        ]
    if nudged:
        messages.append({"role": "user", "content": "You are out of tool calls. Answer now."})
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": task}],
        provider_type = "openrouter",
        model = "a-model",
        context_overflow = "truncate_oldest",
        compaction_threshold = 3_000,
        context_window = 4_096,
        max_tokens = 512,
    )

    fitted, truncation, _ = ri._fit_external_context(
        messages, payload, branch = [{"role": "user", "content": task}], tool_loop = True
    )
    assert truncation["fits"] and truncation["dropped_messages"] > 0
    assert any(m.get("content") == task for m in fitted)


_OVERFLOW_CHARS = 9_000


class _RefusesPastALimit(_Handler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        recorded = self.server.recorded  # type: ignore[attr-defined]
        recorded.append(body)
        if len(json.dumps(body["messages"])) > _OVERFLOW_CHARS:
            error = {
                "message": "This model's maximum context length is 8192 tokens. "
                "However, your messages resulted in 9000 tokens.",
                "code": "context_length_exceeded",
            }
            payload = json.dumps({"error": error}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
        else:
            sent = sum(
                1 for b in recorded[:-1] if len(json.dumps(b["messages"])) <= _OVERFLOW_CHARS
            )
            fn = {"name": "python", "arguments": json.dumps({"code": f"step({sent})"})}
            call = {"index": 0, "id": f"c{sent}", "type": "function", "function": fn}
            payload = _reply({"tool_calls": [call]} if sent < 4 else {"content": "DONE"})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.parametrize(
    "mode, repeats",
    [
        ("summary", 150),
        ("rolling", 150),
        ("checkpoint", 150),
        ("rolling", 600),
        ("checkpoint", 600),
    ],
)
def test_an_overflow_inside_the_only_turn_resends_with_its_results_shortened(
    monkeypatch, mode, repeats
):
    async def connected(self) -> bool:
        return False

    def execute(name, arguments, **kwargs):
        return "RESULT " + "harbor ledger canal " * repeats

    monkeypatch.setattr(studio_tool_loop, "execute_tool", execute)
    monkeypatch.setattr(Request, "is_disconnected", connected)
    recorded: list[dict] = []

    class Recording(_RefusesPastALimit):
        def do_POST(self) -> None:  # noqa: N802
            self.server.recorded = recorded  # type: ignore[attr-defined]
            super().do_POST()

    chunks, _ = _proxy(
        "openrouter",
        [{"role": "user", "content": "Run four python steps."}],
        handler = Recording,
        enable_tools = True,
        enabled_tools = ["python"],
        permission_mode = "off",
        context_overflow = "truncate_oldest",
        compaction_threshold = 200_000,
        compaction_mode = mode,
    )
    assert not [c for c in chunks if "error" in c]
    assert len(json.dumps(recorded[-1]["messages"])) <= _OVERFLOW_CHARS


def test_an_overflow_after_a_replayed_summary_resends_with_its_results_shortened(monkeypatch):
    async def connected(self) -> bool:
        return False

    def execute(name, arguments, **kwargs):
        return "RESULT " + "harbor ledger canal " * 150

    monkeypatch.setattr(studio_tool_loop, "execute_tool", execute)
    monkeypatch.setattr(Request, "is_disconnected", connected)
    thread_id = "replayed-summary-thread"
    studio_db.upsert_chat_thread(
        {"id": thread_id, "title": "t", "modelType": "base", "modelId": "a-model", "createdAt": 1}
    )
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "Run four python steps."},
    ]
    body = cloud_compaction.body_messages(messages)
    cloud_compaction.save_summary(
        thread_id, covered = body[:-1], summary = "## Goal\nEarlier.", model = "a-model", manual = False
    )
    recorded: list[dict] = []

    class Recording(_RefusesPastALimit):
        def do_POST(self) -> None:  # noqa: N802
            self.server.recorded = recorded  # type: ignore[attr-defined]
            super().do_POST()

    chunks, _ = _proxy(
        "openrouter",
        messages,
        handler = Recording,
        thread_id = thread_id,
        enable_tools = True,
        enabled_tools = ["python"],
        permission_mode = "off",
        context_overflow = "truncate_oldest",
        compaction_threshold = 200_000,
        compaction_mode = "summary",
    )
    assert not [c for c in chunks if "error" in c]
    assert len(json.dumps(recorded[-1]["messages"])) <= _OVERFLOW_CHARS


def test_a_cjk_tool_result_keeps_the_longest_excerpt_that_fits():
    import routes.inference as ri
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "q"}], provider_type = "openrouter", model = "m"
    )
    count = ri._external_token_counter(payload, None)
    call = {"id": "c", "type": "function", "function": {"name": "python", "arguments": "{}"}}

    def chat(result):
        return [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c", "content": result},
        ]

    text = "港口账本运河" * 1_400
    target = count(chat(text[:5_100]))
    messages = chat(text)
    assert ri._clip_turn_tool_results(messages, 1, set(), target, count)
    assert count(messages) <= target
    assert len(messages[-1]["content"]) > 4_500
