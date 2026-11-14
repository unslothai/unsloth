# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hub reads an MLX load starts early: the same requests, the same answers, less waiting."""

from __future__ import annotations

import threading
import urllib.error
from types import SimpleNamespace

import pytest

from utils import transformers_version as tv
from utils.models import model_config as mc


@pytest.fixture
def hub(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setattr(tv, "_hf_json_ahead", {})
    reads = []
    gate = threading.Barrier(3, timeout = 10)

    def read(url, token):
        reads.append((url, token))
        gate.wait()
        if url.endswith("adapter_config.json"):
            raise urllib.error.HTTPError(url, 404, "missing", None, None)
        return {"read": len(reads), "url": url}

    monkeypatch.setattr(tv, "_read_hf_json", read)
    return reads


def test_reads_run_together_and_each_answer_serves_one_read(hub, monkeypatch):
    tv.read_activation_files_ahead("org/model", "tok")

    config = tv._hf_raw_url("org/model", "config.json")
    tokenizer = tv._hf_raw_url("org/model", "tokenizer_config.json")
    assert tv._hf_json(config, "tok")["url"] == config
    assert tv._hf_json(tokenizer, "tok")["url"] == tokenizer
    assert sorted(token for _, token in hub) == ["tok"] * 3

    monkeypatch.setattr(tv, "_read_hf_json", lambda url, token: "read again")
    assert tv._hf_json(config, "tok") == "read again"


def test_a_404_reaches_the_caller_as_it_would_have(hub):
    tv.read_activation_files_ahead("org/model", None)
    url = tv._hf_raw_url("org/model", "adapter_config.json")
    with pytest.raises(urllib.error.HTTPError) as caught:
        tv._hf_json(url, None)
    assert caught.value.code == 404


def test_another_credential_does_not_read_this_answer(hub, monkeypatch):
    tv.read_activation_files_ahead("org/model", "tok")
    monkeypatch.setattr(tv, "_read_hf_json", lambda url, token: f"own read {token}")
    url = tv._hf_raw_url("org/model", "config.json")
    assert tv._hf_json(url, False) == "own read False"


@pytest.mark.parametrize(
    ("repo", "offline"),
    [("org/model", True), ("/local/model", False)],
)
def test_nothing_starts_offline_or_for_a_local_path(hub, monkeypatch, repo, offline):
    if offline:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    tv.read_activation_files_ahead(repo, None)
    assert hub == [] and tv._hf_json_ahead == {}


def test_an_existing_relative_checkpoint_is_not_read_from_the_hub(hub, monkeypatch, tmp_path):
    (tmp_path / "outputs" / "private-model").mkdir(parents = True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mc, "_env_offline", lambda: False)
    calls = []
    monkeypatch.setattr("huggingface_hub.model_info", lambda repo, **kwargs: calls.append(repo))
    tv.read_activation_files_ahead("outputs/private-model", "tok")
    with mc.shared_hub_model_info():
        assert mc.read_hub_model_info_ahead("outputs/private-model", None) is None
    assert hub == [] and tv._hf_json_ahead == {} and calls == []


def test_model_info_ahead_fills_the_open_scope(monkeypatch):
    calls = []
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setattr(mc, "_env_offline", lambda: False)
    monkeypatch.setattr(
        "huggingface_hub.model_info",
        lambda repo, **kwargs: calls.append(repo) or SimpleNamespace(id = repo),
    )
    with mc.shared_hub_model_info():
        mc.read_hub_model_info_ahead("org/model", None).join()
        assert mc._hub_model_info("org/model", None).id == "org/model"
    assert calls == ["org/model"]


def test_a_failed_read_ahead_leaves_the_answer_to_the_probe(monkeypatch):
    outcomes = [OSError("reset"), SimpleNamespace(id = "second")]

    def model_info(repo, **kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(mc, "_env_offline", lambda: False)
    monkeypatch.setattr("huggingface_hub.model_info", model_info)
    with mc.shared_hub_model_info():
        mc.read_hub_model_info_ahead("org/model", None).join()
        assert mc._hub_model_info("org/model", None).id == "second"
