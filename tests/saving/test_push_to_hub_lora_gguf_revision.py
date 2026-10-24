# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import contextlib
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub.errors import HfHubHTTPError

import unsloth.save as save_mod


class _LoraModel:
    pass


class _RecordingApi:
    calls = []
    branch_status = None

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def record(*args, **kwargs):
            _RecordingApi.calls.append((name, kwargs))
            if name == "create_branch" and _RecordingApi.branch_status:
                raise HfHubHTTPError(
                    "forbidden",
                    response = SimpleNamespace(
                        status_code = _RecordingApi.branch_status, headers = {}, request = None
                    ),
                )

        return record


@contextlib.contextmanager
def _fake_converter(cmd, **kwargs):
    open(cmd[cmd.index("--outfile") + 1], "wb").write(b"GGUF")
    yield SimpleNamespace(stdout = [], returncode = 0, wait = lambda: 0, args = cmd)


@pytest.fixture
def push(monkeypatch, tmp_path):
    _RecordingApi.calls = []
    _RecordingApi.branch_status = None
    (tmp_path / "convert_lora_to_gguf.py").write_text("")
    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingApi)
    monkeypatch.setattr(save_mod, "PeftModel", _LoraModel)
    monkeypatch.setattr(save_mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path))
    monkeypatch.setattr(save_mod, "install_llama_cpp", lambda **kwargs: None)
    monkeypatch.setattr(save_mod, "save_lora_to_custom_dir", lambda *args: None)
    monkeypatch.setattr(save_mod, "_lora_base_model_id", lambda model: "u/base")
    monkeypatch.setattr(save_mod, "_loaded_via_remote_code", lambda model: False)
    monkeypatch.setattr(save_mod.subprocess, "Popen", _fake_converter)
    monkeypatch.setattr(save_mod, "raise_if_merging_mistral_format_view", lambda *args: None)

    def run(**kwargs):
        save_mod.unsloth_push_to_hub_gguf(
            _LoraModel(), "u/my-lora", tokenizer = object(), save_method = "lora", token = "t", **kwargs
        )
        return [name for name, _ in _RecordingApi.calls]

    return run


def test_new_revision_is_created_before_the_upload(push):
    names = push(revision = "v2")
    assert names.index("create_branch") < names.index("upload_folder")
    branch = next(kw for name, kw in _RecordingApi.calls if name == "create_branch")
    assert branch["branch"] == "v2" and branch["exist_ok"] is True
    upload = next(kw for name, kw in _RecordingApi.calls if name == "upload_folder")
    assert upload["revision"] == "v2"


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"create_pr": True}, {"revision": "refs/pr/3"}, {"revision": "v2", "create_pr": True}],
)
def test_no_branch_is_created_without_a_new_revision(push, kwargs):
    names = push(**kwargs)
    assert "create_branch" not in names and "upload_folder" in names


def test_branch_creation_errors_are_not_swallowed(push):
    _RecordingApi.branch_status = 403
    with pytest.raises(HfHubHTTPError):
        push(revision = "v2")


@pytest.mark.parametrize(
    "kwargs, url",
    [
        ({}, "https://huggingface.co/u/my-lora\n"),
        ({"revision": "v2"}, "https://huggingface.co/u/my-lora/tree/v2\n"),
        ({"revision": "refs/pr/3"}, "https://huggingface.co/u/my-lora/discussions/3\n"),
    ],
)
def test_done_message_points_at_the_revision(push, capsys, kwargs, url):
    push(**kwargs)
    assert capsys.readouterr().out.endswith(f"Unsloth: Done. Uploaded to {url}")
