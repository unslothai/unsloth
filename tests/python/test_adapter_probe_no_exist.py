# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A base repo the cache already records as having no adapter_config.json at the probed commit skips the PeftConfig HEAD."""

import os
from types import SimpleNamespace

import pytest
import transformers

import unsloth  # noqa: F401
import peft
from huggingface_hub import constants as hub_constants
from unsloth.models import loader as loader_module
from unsloth.models import loader_utils

REPO = "org/base-model"
SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(key, raising = False)
    return cache


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("unexpected Hub request")

    try:
        import requests
        monkeypatch.setattr(requests.Session, "request", refuse)
    except ImportError:
        pass
    try:
        import httpx
        monkeypatch.setattr(httpx.Client, "send", refuse)
    except ImportError:
        pass


def _repo_dir(cache, repo = REPO):
    return cache / ("models--" + repo.replace("/", "--"))


def _mark_missing(
    cache,
    commit = SHA,
    repo = REPO,
    filename = "adapter_config.json",
):
    marker = _repo_dir(cache, repo) / ".no_exist" / commit / filename
    marker.parent.mkdir(parents = True, exist_ok = True)
    marker.touch()


def _probe(
    config,
    revision = None,
    **kwargs,
):
    return loader_utils._cached_missing_adapter_error(REPO, config, revision, None, **kwargs)


def test_marker_for_the_probed_commit_gives_peft_error_without_a_request(hub_cache, no_network):
    _mark_missing(hub_cache)
    error = _probe(SimpleNamespace(_commit_hash = SHA))
    assert isinstance(error, ValueError)
    with pytest.raises(ValueError) as real:
        peft.PeftConfig.from_pretrained(REPO, local_files_only = True)
    assert str(error) == str(real.value)


def test_falls_through_when_the_cache_does_not_settle_it(
    hub_cache, tmp_path, no_network, monkeypatch
):
    config = SimpleNamespace(_commit_hash = SHA)
    assert _probe(config) is None  # no marker
    _mark_missing(hub_cache, commit = OTHER_SHA)
    assert _probe(config) is None  # marker for another commit
    _mark_missing(hub_cache, filename = "config.json")
    assert _probe(config) is None  # marker for another file
    _mark_missing(hub_cache)
    assert _probe(None) is None  # config probe failed
    assert _probe(SimpleNamespace()) is None  # no known commit
    assert _probe(SimpleNamespace(_commit_hash = "main")) is None
    assert _probe(config, local_files_only = True) is None
    local = tmp_path / "org" / "base-model"
    local.mkdir(parents = True)
    monkeypatch.chdir(tmp_path)
    assert _probe(config) is None  # a local folder of the same name
    monkeypatch.chdir(hub_cache)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert _probe(config) is None


def test_cached_adapter_config_still_runs_the_probe(hub_cache, no_network):
    snapshot = _repo_dir(hub_cache) / "snapshots" / SHA
    snapshot.mkdir(parents = True)
    (snapshot / "adapter_config.json").write_text("{}")
    assert _probe(SimpleNamespace(_commit_hash = SHA)) is None


def test_pinned_sha_revision_is_used_without_a_config_hash(hub_cache, no_network):
    _mark_missing(hub_cache)
    assert isinstance(_probe(SimpleNamespace(), revision = SHA), ValueError)
    assert _probe(SimpleNamespace(), revision = OTHER_SHA) is None


@pytest.mark.skipif(
    loader_utils._ResolvedRevision is None, reason = "huggingface_hub without ResolvedRevision"
)
def test_resolved_revision_from_the_config_probe_is_reused(hub_cache, no_network, monkeypatch):
    calls = []

    def resolve(
        name,
        revision = None,
        **kwargs,
    ):
        calls.append(name)
        return loader_utils._ResolvedRevision(resolved = SHA, initial = revision, repo_id = name)

    monkeypatch.setattr(loader_utils, "_hub_resolve_revision", resolve)
    _mark_missing(hub_cache)

    def load():
        loader_utils._resolved_revision(REPO, None, None)  # what the config probe does
        return _probe(SimpleNamespace())

    assert isinstance(loader_utils._load_probe_memo_scope(load)(), ValueError)
    assert calls == [REPO]


class _ReachedWeights(Exception):
    pass


def _save_tiny(tmp_path):
    config = transformers.LlamaConfig(
        vocab_size = 128,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
    )
    path = tmp_path / "tiny_llama"
    config.save_pretrained(str(path))
    return str(path), config


@pytest.fixture
def wired(monkeypatch):
    seen = {"helper": [], "peft": 0, "dispatch": []}

    def helper(model_name, model_config, revision, token, local_files_only):
        seen["helper"].append(model_config)
        return ValueError(f"Can't find 'adapter_config.json' at '{model_name}'")

    peftconfig = peft.config.PeftConfigMixin.from_pretrained.__func__

    def counted(cls, *args, **kwargs):
        seen["peft"] += 1
        return peftconfig(cls, *args, **kwargs)

    def stop(*args, **kwargs):
        seen["dispatch"].append(kwargs.get("model_name"))
        raise _ReachedWeights()

    from unsloth.models.llama import FastLlamaModel

    monkeypatch.setattr(loader_module, "_cached_missing_adapter_error", helper)
    monkeypatch.setattr(peft.config.PeftConfigMixin, "from_pretrained", classmethod(counted))
    monkeypatch.setattr(loader_module.FastBaseModel, "from_pretrained", staticmethod(stop))
    monkeypatch.setattr(FastLlamaModel, "from_pretrained", staticmethod(stop))
    return seen


@pytest.mark.skipif(
    getattr(unsloth, "_IS_MLX", False), reason = "the MLX loader needs real safetensors"
)
@pytest.mark.parametrize("entry", ["FastLanguageModel", "FastModel"])
def test_known_missing_adapter_skips_peft_and_loads_the_base(tmp_path, wired, entry):
    path, _ = _save_tiny(tmp_path)
    with pytest.raises(_ReachedWeights):
        getattr(unsloth, entry).from_pretrained(path, max_seq_length = 64, load_in_4bit = False)
    assert wired["peft"] == 0
    assert wired["helper"] and all(c is not None for c in wired["helper"])
    assert wired["dispatch"] and wired["dispatch"][-1] == path


@pytest.mark.skipif(
    getattr(unsloth, "_IS_MLX", False), reason = "the MLX loader needs real safetensors"
)
def test_fastmodel_user_config_never_vouches_for_the_repo(tmp_path, wired):
    path, config = _save_tiny(tmp_path)
    with pytest.raises(_ReachedWeights):
        unsloth.FastModel.from_pretrained(
            path, max_seq_length = 64, load_in_4bit = False, config = config
        )
    assert wired["helper"] and all(c is None for c in wired["helper"])
