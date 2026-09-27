# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF repo whose Hub listing cannot be read must not be routed to Transformers (#11551).

``detect_gguf_model_remote`` returned None both for "no GGUF in this repo" and for "the
Hub refused or never answered", so a GGUF-only repo fell through to the Transformers tier
and failed with "Both AutoConfig and PeftConfig loading failed", naming neither the
format nor the Hub failure. No GPU, no network.
"""

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

import core.inference.llama_cpp as llama_cpp
import utils.hf_cache_settings as hf_cache_settings
import utils.models.model_config as mc
from utils.models.model_config import (
    GgufRepoUnreadableError,
    GgufVariantInfo,
    ModelConfig,
    _looks_like_gguf_repo,
    detect_gguf_model_remote,
)

REPO = "unsloth/LFM2.5-VL-1.6B-GGUF"
GGUF = "LFM2.5-VL-1.6B-UD-Q4_K_XL.gguf"


class RepositoryNotFoundError(Exception):
    """Matched by type name, as detect_gguf_model_remote does."""

    def __init__(self, message, status_code = 401):
        super().__init__(message)
        self.response = SimpleNamespace(status_code = status_code)


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setattr(
        hf_cache_settings, "get_hf_cache_paths", lambda: SimpleNamespace(hub_cache = tmp_path)
    )
    monkeypatch.setattr(mc.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "_find_llama_server_binary",
        staticmethod(lambda *, include_denied = False: "/fake/llama-server"),
    )
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kwargs: None)
    monkeypatch.setattr(
        mc,
        "list_gguf_variants",
        lambda identifier, hf_token = None: (
            [GgufVariantInfo(filename = GGUF, quant = "UD-Q4_K_XL", size_bytes = 1)],
            True,
        ),
    )
    # The Transformers fall-through, so a misroute returns a config instead of hanging.
    monkeypatch.setattr(mc, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mc, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mc, "is_model_cached", lambda *a, **k: False)
    return tmp_path


def _hub(behaviour):
    def model_info(repo_id, **kwargs):
        if isinstance(behaviour, BaseException):
            raise behaviour
        return SimpleNamespace(siblings = [SimpleNamespace(rfilename = f) for f in behaviour])

    return model_info


def _cache(root: Path, repo: str, name: str) -> None:
    snapshot = root / f"models--{repo.replace('/', '--')}" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True, exist_ok = True)
    (snapshot / name).write_bytes(b"cached")


def test_healthy_hub_still_detects_gguf(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "model_info", _hub([GGUF, "README.md"]))
    config = ModelConfig.from_identifier(REPO)
    assert config.is_gguf and config.gguf_hf_repo == REPO
    assert config.is_vision


def test_hub_401_on_gguf_repo_raises_clear_error(monkeypatch):
    monkeypatch.setattr(
        huggingface_hub, "model_info", _hub(RepositoryNotFoundError("401 Client Error"))
    )
    with pytest.raises(GgufRepoUnreadableError) as exc_info:
        ModelConfig.from_identifier(REPO)
    msg = str(exc_info.value)
    assert REPO in msg
    assert "RepositoryNotFoundError (HTTP 401)" in msg
    assert "token in Settings" in msg and "HF_ENDPOINT" in msg
    assert "AutoConfig" not in msg
    # A ValueError, so /load and /validate answer 400 with this text.
    assert isinstance(exc_info.value, ValueError)


def test_hub_401_is_not_served_from_a_stale_cache(monkeypatch, _isolated):
    _cache(_isolated, REPO, GGUF)
    monkeypatch.setattr(
        huggingface_hub, "model_info", _hub(RepositoryNotFoundError("401 Client Error"))
    )
    with pytest.raises(GgufRepoUnreadableError):
        ModelConfig.from_identifier(REPO)


def test_repeated_timeouts_on_uncached_gguf_repo_raise_clear_error(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "model_info", _hub(TimeoutError("read timed out")))
    with pytest.raises(GgufRepoUnreadableError, match = "TimeoutError: read timed out"):
        ModelConfig.from_identifier(REPO)


def test_repeated_timeouts_with_cached_snapshot_still_load_from_cache(monkeypatch, _isolated):
    _cache(_isolated, REPO, GGUF)
    monkeypatch.setattr(huggingface_hub, "model_info", _hub(TimeoutError("read timed out")))
    config = ModelConfig.from_identifier(REPO)
    assert config.is_gguf


def test_explicit_variant_marks_an_unsuffixed_repo_as_gguf(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "model_info", _hub(OSError("proxy refused")))
    with pytest.raises(GgufRepoUnreadableError, match = "OSError: proxy refused"):
        ModelConfig.from_identifier("org/some-model", gguf_variant = "Q4_K_M")


def test_offline_uncached_gguf_repo_raises_clear_error(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub, "model_info", _hub(AssertionError("no API offline")))
    with pytest.raises(GgufRepoUnreadableError, match = "offline"):
        ModelConfig.from_identifier(REPO)


@pytest.mark.parametrize(
    "failure", [RepositoryNotFoundError("401 Client Error"), TimeoutError("read timed out")]
)
def test_non_gguf_repo_with_hub_failure_keeps_todays_behaviour(monkeypatch, failure):
    monkeypatch.setattr(huggingface_hub, "model_info", _hub(failure))
    config = ModelConfig.from_identifier("unsloth/Qwen3-0.6B")
    assert config is not None and not config.is_gguf


def test_gguf_repo_without_gguf_files_keeps_todays_behaviour(monkeypatch):
    # A successful listing with no GGUF is a real answer, not a Hub failure.
    monkeypatch.setattr(huggingface_hub, "model_info", _hub(["config.json", "model.safetensors"]))
    config = ModelConfig.from_identifier("org/odd-GGUF")
    assert config is not None and not config.is_gguf


def test_public_detect_signature_unchanged_outside_from_identifier(monkeypatch):
    monkeypatch.setattr(
        huggingface_hub, "model_info", _hub(RepositoryNotFoundError("401 Client Error"))
    )
    assert detect_gguf_model_remote(REPO) is None


@pytest.mark.parametrize(
    "repo_id,expected",
    [
        ("unsloth/LFM2.5-VL-1.6B-GGUF", True),
        ("prism-ml/Ternary-Bonsai-27B-gguf", True),
        ("org/gguf-models", True),
        ("org/model_GGUF.v2", True),
        ("unsloth/Qwen3-0.6B", False),
        ("org/ggufmaker", False),
        ("gguf-org/regular-model", False),
    ],
)
def test_looks_like_gguf_repo(repo_id, expected):
    assert _looks_like_gguf_repo(repo_id) is expected
