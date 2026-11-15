# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""The decision model and SentenceTransformer pushes refuse private=True into an existing public repo
before any upload, like save.py (#17752). Hub calls are mocked: nothing reaches the network."""

import importlib.util
import os
import sys
from pathlib import Path

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


def _mlx(monkeypatch):
    # By path, as unsloth/__init__.py loads it on Apple Silicon: no torch, so no save.py.
    for name in ("unsloth._decision_common", "_decision_mlx_under_test"):
        monkeypatch.delitem(sys.modules, name, raising = False)
    path = Path(__file__).resolve().parents[2] / "unsloth" / "models" / "decision_mlx.py"
    spec = importlib.util.spec_from_file_location("_decision_mlx_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mlx_clef(monkeypatch):
    model = SimpleNamespace(save_pretrained = lambda folder, tokenizer = None: None)
    push = _mlx(monkeypatch)._clef_push
    return lambda private: push(model, "me/repo", private = private)


def _mlx_merged(monkeypatch):
    model = SimpleNamespace(save_pretrained_merged = lambda folder, tokenizer, method: None)
    push = _mlx(monkeypatch)._decision_push_merged
    return lambda private: push(model, "me/repo", private = private)


def _mlx_gguf(monkeypatch, tmp_path):
    gguf = _mlx(monkeypatch)._decision_gguf()
    (tmp_path / gguf._contract().EXPORT_DIR).mkdir()
    monkeypatch.setattr(gguf.tempfile, "TemporaryDirectory", lambda: _Folder(tmp_path))
    model = SimpleNamespace(save_pretrained_gguf = lambda *a, **kw: {})
    return lambda private: gguf.push_to_hub_gguf(model, "me/repo", private = private)


class _Folder:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return str(self.path)

    def __exit__(self, *exc):
        return False


PUSHES = [_decision_clef, _decision_merged, _embedding_gguf, _mlx_clef, _mlx_merged]


@pytest.mark.parametrize("push", PUSHES, ids = lambda p: p.__name__)
def test_private_push_into_public_repo_refuses_before_upload(monkeypatch, push):
    calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api(True, calls))
    with pytest.raises(RuntimeError, match = "already exists and is public"):
        push(monkeypatch)(True)
    assert "upload" not in calls


@pytest.mark.parametrize(
    "push", [_decision_clef, _decision_merged, _mlx_clef, _mlx_merged], ids = lambda p: p.__name__
)
@pytest.mark.parametrize("public, private", [(True, False), (False, True)])
def test_other_pushes_still_upload(monkeypatch, push, public, private):
    calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api(public, calls))
    push(monkeypatch)(private)
    assert "upload" in calls


@pytest.mark.parametrize(
    "public, private", [(True, True), (True, False), (False, True), (None, True)]
)
def test_mlx_twin_matches_save_guard(monkeypatch, public, private):
    from unsloth.save import _refuse_private_push_to_public_repo as save_guard

    twin = _mlx(monkeypatch)._refuse_private_push_to_public_repo

    class Api:
        def repo_info(self, **kw):
            if public is None:
                raise OSError("offline")
            return SimpleNamespace(private = not public)

    def outcome(guard):
        try:
            guard("me/repo", private, api = Api())
        except RuntimeError as exc:
            return str(exc)

    assert outcome(twin) == outcome(save_guard)


@pytest.mark.parametrize("public, private", [(True, True), (True, False), (False, True)])
def test_mlx_gguf_push(monkeypatch, tmp_path, public, private):
    # decision_mlx loads decision_gguf by path as unsloth._decision_gguf: no relative import of save.py.
    calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api(public, calls))
    push = _mlx_gguf(monkeypatch, tmp_path)
    if public and private:
        with pytest.raises(RuntimeError, match = "already exists and is public"):
            push(private)
        assert "upload" not in calls
    else:
        push(private)
        assert "upload" in calls
