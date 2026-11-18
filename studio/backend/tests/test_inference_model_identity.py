# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import os

import pytest

from core.inference import model_identity as mi
from studio_mcp import loading, model_downloads, placement
from studio_mcp.outputs import SlotModel

AUDIO = "audio-cpp/audio.cpp-gguf/Qwen3-TTS"
NT = os.name == "nt"


@pytest.mark.parametrize(
    "name, local, split",
    [
        ("unsloth/Qwen3-GGUF:Q4_K_M", False, ("unsloth/Qwen3-GGUF", "Q4_K_M")),
        ("unsloth/Qwen3-GGUF:", False, ("unsloth/Qwen3-GGUF", None)),
        (f"{AUDIO}:Q8_0", False, (AUDIO, "Q8_0")),
        ("org/repo:sub/dir", False, ("org/repo:sub/dir", None)),
        ("ab:x", False, ("ab", "x")),
        ("1:x", False, ("1", "x")),
        ("~/m/a.gguf", True, ("~/m/a.gguf", None)),
        ("C:\\models\\a.gguf", True, ("C:\\models\\a.gguf", None)),
        ("C:/models/a.gguf", True, ("C:/models/a.gguf", None)),
        ("C:x", True, ("C:x", None)),
        ("m\\a.gguf", True, ("m\\a.gguf", None)),
        ("ref:0123abcd", True, ("ref:0123abcd", None)),
    ],
)
def test_names(monkeypatch, name, local, split):
    assert mi.is_local(name) is placement.is_local(name) is local
    assert model_downloads._looks_local(name) is local
    assert mi.split_variant(name) == split
    if local or split[1]:
        assert placement._split(name, None) == model_downloads._split(name, None) == split
    asked = []

    async def route_json(method, path, **kwargs):
        asked.append(path)
        return {}

    monkeypatch.setattr(loading, "route_json", route_json)
    asyncio.run(loading._llm_poll(None, name)())
    # Only a Hub name has download progress to read.
    assert ("/api/models/download-progress" in asked) is not local


@pytest.mark.parametrize(
    "left, right, model, repo",
    [
        ("unsloth/Qwen3-0.6B-GGUF", "UNSLOTH/qwen3-0.6b-gguf", True, True),
        (" unsloth/Qwen3 ", "unsloth/qwen3", True, True),
        ("unsloth/Qwen3-GGUF:Q4_K_M", "unsloth/qwen3-gguf", False, True),
        ("unsloth/Qwen3-GGUF:q4_k_m", "unsloth/Qwen3-GGUF:Q4_K_M", True, True),
        ("org/repo:Q4_K_M-3.97bpw", "org/repo:Q4_K_M-4.19bpw", False, True),
        ("unsloth/Qwen3-GGUF", "unsloth/Gemma-GGUF", False, False),
        (AUDIO, AUDIO.lower(), True, True),
        ("/models/Foo.gguf", "/models/foo.gguf", NT, NT),
        ("/models/Foo/", "/models/Foo", True, True),
        ("./m/Foo", "./m/foo", NT, NT),
        ("C:\\models\\a.gguf", "C:/models/a.gguf", True, True),
        ("C:\\models\\a.gguf", "C:\\models\\b.gguf", False, False),
        ("/models/a:b.gguf", "/models/a:c.gguf", False, False),
        ("ref:ABC", "ref:ABC", True, True),
        ("ref:ABC", "ref:abc", NT, NT),
        (None, "unsloth/Qwen3", False, False),
    ],
)
def test_same_model_and_repo(left, right, model, repo):
    for a, b in ((left, right), (right, left)):
        assert mi.same_model(a, b) is mi.matches_any(a, (None, "", b)) is model
        assert mi.same_repo(a, b) is repo


@pytest.mark.parametrize(
    "variant, filename, settled, result",
    [
        ("Q4_K_M", "model-Q4_K_M.gguf", True, True),
        ("q4_k_m", "model-Q4_K_M.gguf", False, True),
        ("Q8_0", "model-Q4_K_M.gguf", True, False),
        # Status drops the bpw suffix: mid-load a sibling build cannot be told apart.
        ("IQ4_XS", "model-IQ4_XS-3.97bpw.gguf", True, True),
        ("IQ4_XS", "model-IQ4_XS-3.97bpw.gguf", False, False),
        ("Q4_K_M", "model.safetensors", True, None),
    ],
)
def test_resident_quant_matches(variant, filename, settled, result):
    assert mi.resident_quant_matches(variant, filename, settled = settled) is result


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "Foo.gguf").write_bytes(b"")


def _predict(
    kind,
    model,
    resident,
    mode = "always",
):
    slots = [SlotModel(kind = kind, model = resident)]
    args = dict(device = None, alongside = None, mode = mode, status = {}, slots = slots)
    return placement.predict(kind = kind, model = model, **args)


@pytest.mark.parametrize(
    "left, right",
    [
        ("models/Foo.gguf", "models/foo.gguf"),
        ("unsloth/Qwen3-GGUF:Q4_K_M", "UNSLOTH/qwen3-gguf"),
        ("/m/Foo", "/m/foo"),
        ("C:\\m\\a.gguf", "C:\\m\\b.gguf"),
        (AUDIO, AUDIO.upper()),
        ("org/repo:sub/dir", "org/repo"),
    ],
)
def test_load_and_plan_agree(models_dir, left, right):
    assert _predict("llm", left, right)[2] is not loading._same_id(left, right)


def test_an_existing_relative_path_keeps_its_case(models_dir):
    assert mi.is_local("models/Foo.gguf", probe = True) and not mi.is_local("models/Foo.gguf")
    assert not mi.is_local("unsloth/Qwen3", probe = True)
    assert mi.same_model("models/foo.gguf", "models/Foo.gguf", probe = True) is NT
    assert mi.same_model("models/Foo.gguf", "models/foo.gguf")
    others, exact, _ = _predict("stt", "models/Foo.gguf", "models/foo.gguf", mode = None)
    # Windows reads them as one file; elsewhere they are two.
    assert NT or ([s.model for s in others], exact) == (["models/foo.gguf"], False)
    assert model_downloads.keys_for(None, "llm", "~/m/a.gguf", None) == []


def test_route_and_status_keep_their_old_rules():
    import routes.inference as route

    # Not vouched for as a path there: a dot-relative or forward-slash drive name.
    for name in ("/m", "~/m", "m\\a", "./m", "C:/m", "o/r"):
        assert route._looks_like_local_path(name) is (name in ("/m", "~/m", "m\\a")), name
    # Status may spell the resident path differently from the request.
    assert loading._resident_matches({"loaded": True, "repo_id": "/m/foo"}, "/m/Foo")
    assert not loading._resident_matches({"loaded": True, "repo_id": "/m/bar"}, "/m/Foo")
