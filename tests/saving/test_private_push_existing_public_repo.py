# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""create_repo(private=True) leaves an existing public repo public, so a private=True push must stop before
any upload. Hub calls are mocked: nothing reaches the network."""

import os

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from types import SimpleNamespace

import huggingface_hub
import pytest

import unsloth.save as save_mod

UPLOADS = {"upload_file", "upload_folder", "create_commit", "card_push", "model_push", "save"}


class _Hub:
    """Fake HfApi + create_repo + ModelCard recording every call; `existing` = None means a new repo."""

    def __init__(self, existing):
        self.existing = existing
        self.calls = []
        hub = self

        class FakeApi:
            def __init__(self, *a, **kw):
                pass

            def repo_info(
                self,
                repo_id,
                repo_type = None,
                **kw,
            ):
                hub.calls.append("repo_info")
                if hub.existing == "error":
                    raise OSError("network down")
                return SimpleNamespace(private = hub.private_now)

            def create_repo(
                self,
                repo_id,
                private = None,
                exist_ok = False,
                **kw,
            ):
                hub.create(private, exist_ok)

            def __getattr__(self, name):
                def record(*a, **kw):
                    hub.calls.append(name)
                    return SimpleNamespace(pr_url = None)

                return record

        class FakeCard:
            def __init__(self, content):
                self.data = SimpleNamespace(datasets = None, tags = None)

            def push_to_hub(self, *a, **kw):
                hub.calls.append("card_push")

        self.Api, self.Card = FakeApi, FakeCard
        self.private_now = None

    def create(self, private, exist_ok):
        self.calls.append("create_repo")
        if self.existing in (None, "error"):
            self.private_now = bool(private)
            self.existing_after = True
            return
        self.private_now = self.existing == "private"
        if not exist_ok:
            raise RuntimeError("409 repo already exists")

    def create_repo_fn(
        self,
        repo_id = None,
        token = None,
        repo_type = None,
        exist_ok = False,
        private = None,
    ):
        self.create(private, exist_ok)

    def uploads(self):
        return [c for c in self.calls if c in UPLOADS]


@pytest.fixture
def hub(monkeypatch):
    def make(existing):
        h = _Hub(existing)
        monkeypatch.setattr(save_mod, "HfApi", h.Api)
        monkeypatch.setattr(huggingface_hub, "HfApi", h.Api)
        monkeypatch.setattr(huggingface_hub, "create_repo", h.create_repo_fn)
        monkeypatch.setattr(huggingface_hub, "ModelCard", h.Card)
        monkeypatch.setattr(
            huggingface_hub, "metadata_update", lambda *a, **kw: None, raising = False
        )
        monkeypatch.setattr(huggingface_hub, "whoami", lambda *a, **kw: {"name": "u"})
        return h

    return make


def _model(h):
    def push_to_hub(*a, **kw):
        h.calls.append("model_push")

    return SimpleNamespace(
        config = SimpleNamespace(_name_or_path = "unsloth/tiny", model_type = "llama"),
        push_to_hub = push_to_hub,
    )


# (private, existing repo state) -> refused?
CASES = [
    (True, "public", True),
    (True, "private", False),
    (True, None, False),
    (False, "public", False),
    (None, "public", False),
    (True, "error", False),  # lookup failure keeps the old behaviour
]


def _check(h, refused, run):
    if refused:
        with pytest.raises(RuntimeError, match = "already exists and is public"):
            run()
        assert h.uploads() == [], h.calls
    else:
        run()
    assert "update_repo_settings" not in h.calls


@pytest.mark.parametrize("private, existing, refused", CASES)
def test_upload_to_huggingface(hub, tmp_path, monkeypatch, private, existing, refused):
    h = hub(existing)
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "w.bin"
    weights.write_bytes(b"x")
    run = lambda: save_mod.upload_to_huggingface(
        _model(h), "u/repo", "t", "finetuned", file_location = str(weights), private = private
    )
    _check(h, refused, run)
    if not refused:
        assert "upload_file" in h.calls


@pytest.mark.parametrize("private, existing, refused", CASES)
def test_create_huggingface_repo(hub, private, existing, refused):
    h = hub(existing)
    _check(
        h,
        refused,
        lambda: save_mod.create_huggingface_repo(_model(h), "u/repo", token = "t", private = private),
    )


@pytest.mark.parametrize("private, existing, refused", CASES)
def test_unsloth_save_model_lora_push(hub, private, existing, refused):
    h = hub(existing)
    model = _model(h)
    run = lambda: save_mod.unsloth_save_model(
        model, None, "u/repo", save_method = "lora", push_to_hub = True, token = "t", private = private
    )
    _check(h, refused, run)
    if not refused:
        assert "model_push" in h.calls


def _fake_gguf_convert(**kwargs):
    directory = kwargs["save_directory"]
    os.makedirs(directory, exist_ok = True)
    path = os.path.join(directory, os.path.basename(directory) + ".Q8_0.gguf")
    open(path, "wb").write(b"GGUF")
    return {
        "gguf_files": [path],
        "modelfile_location": None,
        "want_full_precision": False,
        "is_vlm": False,
        "fix_bos_token": False,
        "save_directory": directory,
    }


@pytest.mark.parametrize("private, existing, refused", CASES)
def test_unsloth_push_to_hub_gguf(hub, monkeypatch, tmp_path, private, existing, refused):
    h = hub(existing)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(save_mod, "unsloth_save_pretrained_gguf", _fake_gguf_convert)
    run = lambda: save_mod.unsloth_push_to_hub_gguf(
        object(), "u/repo", tokenizer = object(), token = "t", private = private
    )
    _check(h, refused, run)
    if not refused:
        assert "create_commit" in h.calls


@pytest.mark.parametrize("private, existing, refused", CASES)
def test_push_with_mtp_head(hub, monkeypatch, private, existing, refused):
    h = hub(existing)

    class _Reached(Exception):
        pass

    def fake_save(**kw):
        h.calls.append("save")
        raise _Reached

    monkeypatch.setattr(save_mod, "unsloth_save_model", fake_save)
    save_kwargs = dict(
        is_main_process = True,
        token = "t",
        save_directory = "u/repo",
        private = private,
        revision = None,
        create_pr = False,
    )
    if refused:
        _check(h, True, lambda: save_mod._push_with_mtp_head(save_kwargs))
    else:
        with pytest.raises(_Reached):
            save_mod._push_with_mtp_head(save_kwargs)
