# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""The decision model and SentenceTransformer pushes refuse private=True into an existing public repo
before any upload, like save.py (#17752). Hub calls are mocked: nothing reaches the network."""

import os

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from types import SimpleNamespace

import huggingface_hub
import pytest


def _fake_api(public, calls):
    class FakeApi:
        def __init__(self, *a, **kw):
            pass

        def create_repo(self, repo_id, **kw):
            calls.append("create_repo")
            return SimpleNamespace(repo_id = repo_id)

        def repo_info(self, repo_id, **kw):
            return SimpleNamespace(private = not public)

        def upload_folder(self, **kw):
            calls.append("upload")

        def upload_file(self, **kw):
            calls.append("upload")

        def whoami(self, *a, **kw):
            return {"name": "me"}

    return FakeApi


def _decision_clef(monkeypatch):
    from unsloth.models import decision
    model = SimpleNamespace(save_pretrained = lambda folder, tokenizer = None: None)
    return lambda private: decision.push_to_hub_clef(model, "me/repo", private = private)


def _decision_merged(monkeypatch):
    from unsloth.models import decision
    model = SimpleNamespace(save_pretrained_merged = lambda folder, tokenizer, method: None)
    return lambda private: decision.push_to_hub_merged(model, "me/repo", private = private)


def _embedding_gguf(monkeypatch):
    from unsloth.models import sentence_transformer as st

    monkeypatch.setattr(st, "HfApi", huggingface_hub.HfApi, raising = False)
    monkeypatch.setattr(st, "_save_pretrained_gguf", lambda *a, **kw: {"gguf_files": []})
    model = SimpleNamespace(tokenizer = None)
    return lambda private: st._push_to_hub_gguf(model, "me/repo", private = private, token = "t")


PUSHES = [_decision_clef, _decision_merged, _embedding_gguf]


@pytest.mark.parametrize("push", PUSHES, ids = lambda p: p.__name__)
def test_private_push_into_public_repo_refuses_before_upload(monkeypatch, push):
    calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api(True, calls))
    with pytest.raises(RuntimeError, match = "already exists and is public"):
        push(monkeypatch)(True)
    assert "upload" not in calls


@pytest.mark.parametrize("push", [_decision_clef, _decision_merged], ids = lambda p: p.__name__)
@pytest.mark.parametrize("public, private", [(True, False), (False, True)])
def test_other_pushes_still_upload(monkeypatch, push, public, private):
    calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api(public, calls))
    push(monkeypatch)(private)
    assert "upload" in calls
