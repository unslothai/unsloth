# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression guards for the model-picker per-model-config feature (the set of
bugs that got the predecessor PR reverted). Pure-function / validation checks
only, so they run on CPU in the backend pytest job with no model download.

Covers, at the backend layer:
  - infra-model hiding: the RAG embedder (bge-small-en-v1.5) and the llama.cpp
    install-validation probe (ggml-org/models / stories260K) stay hidden, while
    normal chat repos are not hidden;
  - the HF token is honored from the dedicated header with the query string as a
    fallback, never the other way around;
  - the chat-template byte caps reject oversized overrides (both the char-count
    fast path and the UTF-8 byte path) and the sidecar reader is size-bounded.
"""

from __future__ import annotations

import sys
import types

import pytest

# Keep this test runnable without the optional structlog dependency (mirrors
# tests/test_cached_gguf_routes.py), since importing routes.models pulls it in.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route
from core.rag import config as rag_config
from hub.dependencies import get_hf_token
from models.inference import LoadRequest
from picker.schemas import MAX_CHAT_TEMPLATE_BYTES
from picker.service import _read_bounded_text
from utils.hidden_models import is_hidden_model


@pytest.fixture(autouse = True)
def _pin_default_embedder(monkeypatch):
    """Pin the effective embedder to Studio's static default so hiding is
    deterministic and cannot depend on ambient RAG config / env."""
    default = "unsloth/bge-small-en-v1.5"
    monkeypatch.setattr(rag_config, "EMBEDDING_MODEL", default, raising = False)
    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: default)
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: default)
    monkeypatch.setattr(rag_config, "default_gguf_repo", lambda: default)


# --------------------------------------------------------------------------- #
# Infra-model hiding (the "infra models resurfaced in the picker" regression)  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value",
    [
        "ggml-org/models",  # the probe repo id
        "unsloth/bge-small-en-v1.5",  # the RAG embedder repo
        "unsloth/bge-small-en-v1.5-GGUF",  # its GGUF companion
        "/root/.cache/huggingface/hub/x/stories260K.gguf",  # probe on disk
        "/root/.cache/x/Stories260K.GGUF",  # case-insensitive
        r"C:\\models\\stories260K.gguf",  # windows-style path
        "/opt/models/bge-small-en-v1.5",  # embedder basename folder
        "/opt/models/bge-small-en-v1.5-Q8_0.gguf",  # suffixed local weight
    ],
)
def test_infra_models_are_hidden(value):
    assert is_hidden_model(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "unsloth/gemma-3-270m-it-GGUF",  # a normal small chat GGUF
        "unsloth/Qwen3-0.6B",  # a normal non-GGUF chat model
        "user/stories260K-finetune-GGUF",  # repo id merely contains "stories260k"
        "user/model-chat",  # generic repo must not be hidden
        "meta-llama/Llama-3.1-8B-Instruct",
    ],
)
def test_normal_models_are_not_hidden(value):
    assert is_hidden_model(value) is False


def test_is_hidden_model_ignores_empty_values():
    assert is_hidden_model(None) is False
    assert is_hidden_model("") is False
    assert is_hidden_model(None, "", "unsloth/gemma-3-270m-it-GGUF") is False


def test_hidden_model_matchers_expose_probe_needles():
    needles, exact_ids, _exact_paths = models_route.hidden_model_matchers()
    lowered = [n.lower() for n in needles]
    assert "ggml-org/models" in lowered
    assert "stories260k.gguf" in lowered
    # The configured embedder is exposed as an exact repo id, never as a
    # basename needle that would substring-hide unrelated chat models.
    assert "bge-small-en-v1.5" not in lowered
    assert "unsloth/bge-small-en-v1.5" in exact_ids


def test_hidden_model_matchers_custom_repo_publishes_exact_ids(monkeypatch):
    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "org/model")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/model-GGUF")
    needles, exact_ids, exact_paths = models_route.hidden_model_matchers()
    assert needles == ["ggml-org/models", "stories260k.gguf"]
    assert "org/model" in exact_ids
    assert "org/model-gguf" in exact_ids
    assert exact_paths == []


def test_hidden_model_matchers_local_owner_name_path_is_exact_path(monkeypatch, tmp_path):
    # A local embedder shaped like owner/name that exists on disk must be an
    # exact resolved path, not a Hub repo id (mirroring is_hidden_model), so the
    # local row stays hidden instead of showing as a chat model.
    (tmp_path / "models" / "embedder").mkdir(parents = True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "models/embedder")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "ggml-org/models")
    _needles, exact_ids, exact_paths = models_route.hidden_model_matchers()
    resolved = str((tmp_path / "models" / "embedder").resolve()).lower()
    assert resolved in exact_paths
    assert "models/embedder" not in exact_ids


# --------------------------------------------------------------------------- #
# HF token via header, query string only as a fallback (the token-leak fix)    #
# --------------------------------------------------------------------------- #


def test_get_hf_token_strips_and_returns():
    assert get_hf_token("  hf_abc ") == "hf_abc"


@pytest.mark.parametrize("value", [None, "", "   ", "\n\t"])
def test_get_hf_token_blank_is_none(value):
    assert get_hf_token(value) is None


@pytest.mark.parametrize(
    "value,expected",
    [("  hf_x ", "hf_x"), ("", None), ("   ", None), (None, None), (1234, None)],
)
def test_normalize_hf_token(value, expected):
    assert models_route._normalize_hf_token(value) == expected


def test_header_token_wins_over_query():
    header, query = "hf_header", "hf_query"
    resolved = models_route._normalize_hf_token(header) or models_route._normalize_hf_token(query)
    assert resolved == "hf_header"


def test_query_token_is_fallback_when_header_absent():
    resolved = models_route._normalize_hf_token(None) or models_route._normalize_hf_token(
        "hf_query"
    )
    assert resolved == "hf_query"


# --------------------------------------------------------------------------- #
# Chat-template byte caps (the unbounded-template hardening)                    #
# --------------------------------------------------------------------------- #


def _load_request(**overrides):
    data = {"model_path": "unsloth/test-model-GGUF", "gguf_variant": "Q4_K_M"}
    data.update(overrides)
    return LoadRequest.model_validate(data)


def test_blank_chat_template_override_normalizes_to_none():
    assert _load_request(chat_template_override = "   \n\t").chat_template_override is None


def test_nonblank_chat_template_override_preserved_verbatim():
    template = "  {{ messages }}  "
    assert _load_request(chat_template_override = template).chat_template_override == template


def test_chat_template_at_byte_limit_is_accepted():
    template = "a" * MAX_CHAT_TEMPLATE_BYTES  # exactly the limit, 1 byte/char
    assert (
        len(_load_request(chat_template_override = template).chat_template_override)
        == MAX_CHAT_TEMPLATE_BYTES
    )


def test_chat_template_over_char_limit_is_rejected():
    with pytest.raises(Exception):  # pydantic ValidationError wrapping ValueError
        _load_request(chat_template_override = "a" * (MAX_CHAT_TEMPLATE_BYTES + 1))


def test_chat_template_over_byte_limit_is_rejected():
    # Char count stays under the limit but UTF-8 bytes exceed it (3 bytes/char),
    # so only the byte-count branch can catch this.
    multibyte = "€" * (MAX_CHAT_TEMPLATE_BYTES // 2)  # euro sign, 3 bytes each
    assert len(multibyte) <= MAX_CHAT_TEMPLATE_BYTES
    assert len(multibyte.encode("utf-8")) > MAX_CHAT_TEMPLATE_BYTES
    with pytest.raises(Exception):
        _load_request(chat_template_override = multibyte)


def test_read_bounded_text_reads_within_limit(tmp_path):
    p = tmp_path / "t.json"
    p.write_text("hello", encoding = "utf-8")
    assert _read_bounded_text(p, 16) == "hello"


def test_read_bounded_text_rejects_over_limit(tmp_path):
    p = tmp_path / "big.json"
    p.write_bytes(b"x" * 100)
    assert _read_bounded_text(p, 50) is None


def test_read_bounded_text_at_limit_is_read(tmp_path):
    p = tmp_path / "exact.json"
    p.write_bytes(b"x" * 50)
    assert _read_bounded_text(p, 50) == "x" * 50


def test_read_bounded_text_missing_file_is_none(tmp_path):
    assert _read_bounded_text(tmp_path / "nope.json", 50) is None
