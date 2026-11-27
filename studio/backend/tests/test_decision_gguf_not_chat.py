# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decision-only GGUFs (e.g. lfm3-d1) are refused for chat, and llama-server's 501 keeps its message."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

import core.inference.diffusion_compat as diffusion_compat
import core.inference.llama_cpp as llama_cpp_module
from core.inference.llama_cpp import LlamaCppBackend

GGUF_MAGIC = 0x46554747


def _kv(key: str, value: str) -> bytes:
    k, v = key.encode(), value.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v


def _gguf_bytes(
    arch: str,
    decision_type: str | None,
    *,
    declared_kv: int | None = None,
) -> bytes:
    kvs = [("general.architecture", arch), ("general.type", "model")]
    if decision_type is not None:
        kvs.append((f"{arch}.decision.type", decision_type))
    body = b"".join(_kv(k, v) for k, v in kvs)
    kv_count = declared_kv if declared_kv is not None else len(kvs)
    return struct.pack("<II", GGUF_MAGIC, 3) + struct.pack("<QQ", 1, kv_count) + body


def _local_refusal(tmp_path: Path, arch: str, decision_type: str | None):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(_gguf_bytes(arch, decision_type))
    return LlamaCppBackend._non_chat_gguf_refusal_for_path(str(gguf), "owner/model")


@pytest.mark.parametrize("decision_type", ["lfm3-d1", "lfm2-d1", "clef", "openjev"])
def test_a_decision_only_gguf_is_refused_for_chat(tmp_path, decision_type):
    message = _local_refusal(tmp_path, "lfm3", decision_type)
    assert message is not None
    assert "decision model" in message
    assert "Decision API" in message


@pytest.mark.parametrize("decision_type", [None, "granite-switch", "someorg-future"])
def test_chat_capable_ggufs_still_load(tmp_path, decision_type):
    assert _local_refusal(tmp_path, "lfm3", decision_type) is None


def test_a_remote_header_prefix_is_enough(monkeypatch):
    # The range read is truncated: more KVs declared than present.
    header = _gguf_bytes("lfm3", "lfm3-d1", declared_kv = 40)
    monkeypatch.setattr(
        llama_cpp_module, "_resolve_variant_gguf_files", lambda *_a, **_k: ("d1-Q4_K_M.gguf", [])
    )
    monkeypatch.setattr(llama_cpp_module, "cached_gguf_for_load", lambda *_a, **_k: None)
    monkeypatch.setattr(diffusion_compat, "_read_gguf_header", lambda *_a, **_k: header)
    message = LlamaCppBackend._remote_non_chat_gguf_refusal(
        hf_repo = "AtomicChat/d1-2B-GGUF",
        hf_variant = "Q4_K_M",
        hf_token = None,
        model_identifier = "AtomicChat/d1-2B-GGUF",
    )
    assert message is not None and "decision model" in message


def test_llama_server_501_reaches_the_user():
    import routes.inference as routes_inference

    exc = RuntimeError(
        'llama-server returned 501: {"error":{"code":501,"message":"This model is a decision '
        'model without text generation, use /v1/systemone","type":"not_supported_error"}}'
    )
    expected = (
        "llama-server error: This model is a decision model without text generation, "
        "use /v1/systemone"
    )
    assert routes_inference._friendly_error(exc) == expected
    assert routes_inference._openai_stream_error_chunk(exc)["error"]["message"] == expected


def test_other_5xx_stay_generic():
    import routes.inference as routes_inference
    exc = RuntimeError('llama-server returned 500: {"error":{"message":"boom"}}')
    assert routes_inference._friendly_error(exc) == "An internal error occurred"
