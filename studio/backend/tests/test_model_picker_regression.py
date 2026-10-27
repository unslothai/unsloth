# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure checks: infra models stay hidden, and the HF header token beats the query-string fallback."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest

# Stub optional structlog: importing routes.models pulls it in.
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
    """Pin the effective embedder to Unsloth's static default so hiding is
    deterministic and cannot depend on ambient RAG config / env."""
    default = "unsloth/bge-small-en-v1.5"
    monkeypatch.setattr(rag_config, "EMBEDDING_MODEL", default, raising = False)
    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: default)
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: default)
    monkeypatch.setattr(rag_config, "default_gguf_repo", lambda: default)


@pytest.mark.parametrize(
    "value",
    [
        "ggml-org/models",
        "unsloth/bge-small-en-v1.5",
        "unsloth/bge-small-en-v1.5-GGUF",
        "/root/.cache/huggingface/hub/x/stories260K.gguf",
        "/root/.cache/x/Stories260K.GGUF",
        r"C:\\models\\stories260K.gguf",
        "/opt/models/bge-small-en-v1.5",
        "/opt/models/bge-small-en-v1.5-Q8_0.gguf",
    ],
)
def test_infra_models_are_hidden(value):
    assert is_hidden_model(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "unsloth/gemma-3-270m-it-GGUF",
        "unsloth/Qwen3-0.6B",
        "user/stories260K-finetune-GGUF",
        "user/model-chat",
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
    (tmp_path / "models" / "embedder").mkdir(parents = True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "models/embedder")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "ggml-org/models")
    _needles, exact_ids, exact_paths = models_route.hidden_model_matchers()
    resolved = str((tmp_path / "models" / "embedder").resolve()).lower()
    assert resolved in exact_paths
    assert "models/embedder" not in exact_ids


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
    template = "a" * MAX_CHAT_TEMPLATE_BYTES
    assert (
        len(_load_request(chat_template_override = template).chat_template_override)
        == MAX_CHAT_TEMPLATE_BYTES
    )


def test_chat_template_over_char_limit_is_rejected():
    with pytest.raises(Exception):
        _load_request(chat_template_override = "a" * (MAX_CHAT_TEMPLATE_BYTES + 1))


def test_chat_template_over_byte_limit_is_rejected():
    # Under the char limit but over in UTF-8 bytes (3 bytes/char).
    multibyte = "€" * (MAX_CHAT_TEMPLATE_BYTES // 2)
    assert len(multibyte) <= MAX_CHAT_TEMPLATE_BYTES
    assert len(multibyte.encode("utf-8")) > MAX_CHAT_TEMPLATE_BYTES
    with pytest.raises(Exception):
        _load_request(chat_template_override = multibyte)


def _large_community_template(target_bytes: int) -> str:
    # Shaped like the large community templates in #14841: one long chain of role branches.
    branch = (
        "{%- elif message['role'] == 'persona_{i}' -%}"
        "{{ '<|im_start|>persona_{i}\\n' + message['content'] | trim + '<|im_end|>\\n' }}\n"
    )
    body, i = [], 0
    while sum(map(len, body)) < target_bytes:
        body.append(branch.replace("{i}", str(i)))
        i += 1
    return (
        "{%- for message in messages -%}\n"
        "{%- if message['role'] == 'system' -%}{{ message['content'] }}\n"
        + "".join(body)
        + "{%- endif -%}{%- endfor -%}"
    )


def test_large_community_chat_template_is_accepted_end_to_end():
    from picker.schemas import ValidateChatTemplateRequest
    from picker.service import validate_chat_template
    from utils import openai_auto_switch_settings as settings

    template = _large_community_template(230 * 1024)
    assert len(template.encode("utf-8")) > 230 * 1024
    assert _load_request(chat_template_override = template).chat_template_override == template
    assert ValidateChatTemplateRequest(template = template).template == template
    assert validate_chat_template(template).valid
    stored = settings.normalize_model_override({"chat_template_override": template})
    assert stored["chat_template_override"] == template


def test_frontend_chat_template_limit_matches_backend():
    import re

    from utils import openai_auto_switch_settings as settings

    ts = (
        Path(__file__).resolve().parents[2]
        / "frontend/src/features/model-picker/model-config/per-model-config.ts"
    ).read_text(encoding = "utf-8")
    match = re.search(r"export const MAX_CHAT_TEMPLATE_BYTES = ([0-9_ *]+);", ts)
    assert match is not None
    frontend_limit = math.prod(int(part) for part in match.group(1).replace("_", "").split("*"))
    assert frontend_limit == MAX_CHAT_TEMPLATE_BYTES
    assert settings.MAX_CHAT_TEMPLATE_OVERRIDE_BYTES == MAX_CHAT_TEMPLATE_BYTES


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
