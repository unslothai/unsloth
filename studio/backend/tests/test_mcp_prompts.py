# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mcp_server import create_studio_mcp
from studio_mcp import prompts
from studio_mcp.outputs import (
    ChatModel,
    ChatSlot,
    GpuDevice,
    HardwareSlot,
    ModelEntry,
    ModelList,
    StudioStatus,
)
from studio_mcp.tools import models, status
from .mcp_harness import MCP_HEADERS, TEST_TOKEN, fake_studio, served

ARGUMENTS = {
    "make_short_film": [("idea", True), ("length_seconds", False)],
    "build_dataset_and_train": [("goal", True), ("base_model", False)],
}
TITLES = {
    "make_short_film": "Make a short film",
    "build_dataset_and_train": "Build a dataset and train",
}
# 24 GiB card with 4 GiB used: 20 GiB free is 21.5 decimal GB.
STATUS = StudioStatus(
    chat = ChatSlot(loaded = [ChatModel(id = "unsloth/Qwen3-0.6B")]),
    hardware = HardwareSlot(
        available = True, devices = [GpuDevice(index = 0, vram_used_gb = 4.0, vram_total_gb = 24.0)]
    ),
)


def _row(
    id,
    kind,
    size_gb,
    fits_now,
    pages = None,
):
    # Catalog fields come from list_models(source="catalog"); duck-typed so the test does not pin ModelEntry.
    return SimpleNamespace(
        id = id,
        kind = kind,
        loaded = False,
        size_bytes = int(size_gb * 1e9),
        fits_now = fits_now,
        audio_workflows = pages,
    )


CATALOG = {
    "video": [
        _row("unsloth/Wan-big", "video", 60.0, False),
        _row("unsloth/LTX-small", "video", 12.0, True),
        _row("unsloth/Wan-mid", "video", 18.5, True),
    ],
    "audio": [
        _row("unsloth/wake-word", "audio", 0.1, True, []),
        _row("unsloth/ACE-Step", "audio", 8.0, True, ["music"]),
    ],
    "tts": [_row(f"unsloth/tts-{i}", "tts", 1.0 + i, True, ["speak"]) for i in range(7)]
    + [_row("unsloth/voice-clone-only", "tts", 0.5, True, ["clone"])],
}
LLMS = [
    ModelEntry(id = "unsloth/Qwen3-0.6B", kind = "llm", loaded = True),
    ModelEntry(id = "unsloth/gemma-3-1b-it", kind = "llm"),
]


@pytest.fixture
def stubbed(monkeypatch):
    calls = []

    async def fake_status():
        return STATUS

    async def fake_list(
        kind = None,
        source = "local",
        **_,
    ):
        calls.append((kind, source))
        if source == "catalog":
            return SimpleNamespace(models = CATALOG.get(kind, []))
        return ModelList(models = [m for m in LLMS if m.kind == kind])

    monkeypatch.setattr(status, "studio_status", fake_status)
    monkeypatch.setattr(models, "list_models", fake_list)
    return calls


def _render(name, args):
    result = asyncio.run(create_studio_mcp().render_prompt(name, args))
    return "".join(message.content.text for message in result.messages)


def _tool_names():
    return {tool.name for tool in asyncio.run(create_studio_mcp().list_tools())}


def test_prompts_are_listed_with_titles_and_plain_arguments():
    listed = {p.name: p for p in asyncio.run(create_studio_mcp().list_prompts())}
    assert set(listed) == set(ARGUMENTS)
    for name, prompt in listed.items():
        assert prompt.title == TITLES[name]
        assert "Unsloth Studio" in prompt.description
        assert [(a.name, a.required) for a in prompt.arguments] == ARGUMENTS[name]
        for argument in prompt.arguments:
            assert argument.description, (name, argument.name)
            assert "JSON string" not in argument.description, (name, argument.name)


def test_film_prompt_shows_fitting_models_and_free_gpu(stubbed):
    text = _render("make_short_film", {"idea": "a fox at dawn", "length_seconds": "20"})
    assert len(text) <= prompts.MAX_CHARS
    assert "a fox at dawn" in text and "about 20 seconds" in text
    assert "GPU 0 21.5 GB free of 25.8 GB" in text
    assert "unsloth/Qwen3-0.6B" in text
    # Fitting rows only, smallest first, at most five per kind.
    assert "unsloth/Wan-big" not in text
    assert text.index("unsloth/LTX-small (12.0 GB)") < text.index("unsloth/Wan-mid (18.5 GB)")
    # Audio rows are only the ones a film can use: music for the bed, speak for the voice-over.
    assert "- audio (music): unsloth/ACE-Step (8.0 GB)" in text
    assert "unsloth/wake-word" not in text and "unsloth/voice-clone-only" not in text
    assert "unsloth/tts-4" in text and "unsloth/tts-5" not in text
    assert sorted(stubbed) == [("audio", "catalog"), ("tts", "catalog"), ("video", "catalog")]


def test_train_prompt_lists_downloaded_llms(stubbed):
    text = _render(
        "build_dataset_and_train",
        {"goal": "answer support tickets", "base_model": "unsloth/gemma-3-1b-it"},
    )
    assert "answer support tickets" in text
    assert "unsloth/Qwen3-0.6B (loaded)" in text and "unsloth/gemma-3-1b-it" in text
    assert 'list_models(model="unsloth/gemma-3-1b-it")' in text
    assert "21.5 GB free" in text
    # There are no LLM catalog rows, so the train prompt reads the downloaded list.
    assert stubbed == [("llm", "local")]


@pytest.mark.parametrize(
    "name, args", [("make_short_film", {"idea": "x"}), ("build_dataset_and_train", {"goal": "x"})]
)
def test_prompts_fall_back_without_live_data(monkeypatch, name, args):
    async def boom(*_, **__):
        raise RuntimeError("No active HTTP request found.")

    monkeypatch.setattr(status, "studio_status", boom)
    monkeypatch.setattr(models, "list_models", boom)
    text = _render(name, args)
    assert prompts.FALLBACK in text
    assert "Steps:" in text


@pytest.mark.parametrize(
    "name, args", [("make_short_film", {"idea": "x"}), ("build_dataset_and_train", {"goal": "x"})]
)
def test_prompts_render_in_process_without_stubs(name, args):
    # No HTTP request means no caller: the real tools raise and the prompt still renders.
    assert prompts.FALLBACK in _render(name, args)


def test_one_failed_listing_does_not_hide_the_rest(stubbed, monkeypatch):
    real = models.list_models

    async def no_audio(kind = None, **kwargs):
        if kind == "audio":
            raise RuntimeError("down")
        return await real(kind = kind, **kwargs)

    monkeypatch.setattr(models, "list_models", no_audio)
    text = _render("make_short_film", {"idea": "x"})
    assert '- audio: unknown; call list_models(kind="audio") yourself.' in text
    assert "unsloth/LTX-small" in text and "21.5 GB free" in text


@pytest.mark.parametrize(
    "name, args", [("make_short_film", {"idea": "x"}), ("build_dataset_and_train", {"goal": "x"})]
)
def test_rendered_calls_name_registered_tools(stubbed, name, args):
    tools = _tool_names()
    called = set(re.findall(r"\b([a-z_]+)\(", _render(name, args)))
    assert called and called <= tools, called - tools


def test_long_input_stays_under_the_cap(stubbed):
    text = _render("make_short_film", {"idea": "x" * 10_000, "length_seconds": "20"})
    assert len(text) <= prompts.MAX_CHARS
    assert "Steps:" in text


def test_every_mention_says_unsloth_studio(stubbed):
    for name, args in (
        ("make_short_film", {"idea": "x"}),
        ("build_dataset_and_train", {"goal": "x"}),
    ):
        text = _render(name, args)
        assert not re.search(r"(?<!Unsloth )Studio", text), name


def _rpc(http, method, params):
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    headers = {**MCP_HEADERS, "Authorization": f"Bearer {TEST_TOKEN}"}
    response = http.post("/mcp/", json = body, headers = headers)
    assert response.status_code == 200, response.text
    data = [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]
    return data[-1]["result"]


def test_prompts_over_http(stubbed, monkeypatch):
    with TestClient(served(create_studio_mcp(), fake_studio(), monkeypatch = monkeypatch)) as http:
        listed = {p["name"]: p for p in _rpc(http, "prompts/list", {})["prompts"]}
        got = _rpc(http, "prompts/get", {"name": "make_short_film", "arguments": {"idea": "a fox"}})
    assert {name: p["title"] for name, p in listed.items()} == TITLES
    for name, prompt in listed.items():
        assert [(a["name"], a.get("required", False)) for a in prompt["arguments"]] == ARGUMENTS[
            name
        ]
    assert "JSON string" not in json.dumps(listed)
    text = got["messages"][0]["content"]["text"]
    assert "unsloth/LTX-small (12.0 GB)" in text and "GPU 0 21.5 GB free" in text
