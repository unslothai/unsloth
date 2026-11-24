# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""SentenceTransformer save_pretrained_gguf(push_to_hub=True) must not turn a failed Hub
upload into a successful save: a create_repo or upload_folder error has to propagate."""

import os

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from types import SimpleNamespace

import pytest

import unsloth.models.sentence_transformer as st_mod


class _ST:
    """Stand-in SentenceTransformer: only self[0].auto_model, tokenizer and save_pretrained are read."""

    def __init__(self):
        self.tokenizer = None
        self._inner = SimpleNamespace(auto_model = SimpleNamespace())

    def __getitem__(self, index):
        return self._inner

    def save_pretrained(self, save_directory):
        pass


class _Api:
    """Fake HfApi; `fail` names the call that raises."""

    def __init__(self, fail):
        self.fail = fail
        self.calls = []

    def create_repo(self, *a, **kw):
        self.calls.append("create_repo")
        if self.fail == "create_repo":
            raise RuntimeError("boom create_repo")
        return SimpleNamespace(repo_id = "user/repo")

    def upload_folder(self, *a, **kw):
        self.calls.append("upload_folder")
        raise RuntimeError("boom upload_folder")


def _save(monkeypatch, tmp_path, api):
    monkeypatch.setattr(st_mod, "HfApi", lambda *a, **kw: api)
    monkeypatch.setattr(
        st_mod, "unsloth_save_pretrained_gguf", lambda *a, **kw: {"gguf_files": []}
    )
    return st_mod._save_pretrained_gguf(
        _ST(),
        save_directory = str(tmp_path / "user_repo"),
        push_to_hub = True,
        token = "tok",
    )


def test_create_repo_failure_propagates(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match = "boom create_repo"):
        _save(monkeypatch, tmp_path, _Api(fail = "create_repo"))


def test_upload_folder_failure_propagates(monkeypatch, tmp_path):
    api = _Api(fail = "upload_folder")
    with pytest.raises(RuntimeError, match = "boom upload_folder"):
        _save(monkeypatch, tmp_path, api)
    assert api.calls == ["create_repo", "upload_folder"]
