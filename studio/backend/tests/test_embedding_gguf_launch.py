# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend contract for serving embedding GGUFs.

llama-server answers ``/v1/embeddings`` with a 501 ("This server does not
support embeddings. Start it with `--embeddings`") unless it was launched with
``--embedding``; nothing in llama.cpp turns that on from the model itself. These
tests pin the header probe that detects an embedding GGUF (``<arch>.pooling_type``,
the only place the flag can be decided before launch) and the ``load_model``
emission it gates.
"""

from __future__ import annotations

import inspect
import io
import struct
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Same external-dep stubs as the other llama_cpp unit tests so importing
# the backend doesn't drag in structlog / httpx / loggers.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

import httpx  # noqa: F401

from core.inference import llama_cpp as llama_cpp_module
from core.inference.llama_cpp import LlamaCppBackend

# llama_pooling_type, include/llama.h
POOLING_NONE = 0
POOLING_MEAN = 1
POOLING_CLS = 2
POOLING_LAST = 3
POOLING_RANK = 4

_VTYPE_UINT32 = 4
_VTYPE_STRING = 8


def _write_kv(buf: io.BytesIO, key: str, value, vtype: int) -> None:
    key_bytes = key.encode("utf-8")
    buf.write(struct.pack("<Q", len(key_bytes)))
    buf.write(key_bytes)
    buf.write(struct.pack("<I", vtype))
    if vtype == _VTYPE_UINT32:
        buf.write(struct.pack("<I", value))
    elif vtype == _VTYPE_STRING:
        val_bytes = value.encode("utf-8")
        buf.write(struct.pack("<Q", len(val_bytes)))
        buf.write(val_bytes)
    else:
        raise ValueError(f"Unsupported vtype in test helper: {vtype}")


def _make_gguf(
    tmp_path: Path,
    arch: str,
    *,
    pooling_type: int | None = None,
    pooling_first: bool = False,
    filename: str = "test.gguf",
) -> str:
    """Header-only GGUF v3 carrying the architecture and optional pooling type."""
    entries: list[tuple[str, object, int]] = []
    if pooling_type is not None and pooling_first:
        entries.append((f"{arch}.pooling_type", pooling_type, _VTYPE_UINT32))
    entries.append(("general.architecture", arch, _VTYPE_STRING))
    entries.append((f"{arch}.block_count", 12, _VTYPE_UINT32))
    if pooling_type is not None and not pooling_first:
        entries.append((f"{arch}.pooling_type", pooling_type, _VTYPE_UINT32))

    buf = io.BytesIO()
    buf.write(struct.pack("<I", 0x46554747))  # GGUF magic
    buf.write(struct.pack("<I", 3))  # version 3
    buf.write(struct.pack("<Q", 0))  # tensor count
    buf.write(struct.pack("<Q", len(entries)))
    for key, value, vtype in entries:
        _write_kv(buf, key, value, vtype)

    path = tmp_path / filename
    path.write_bytes(buf.getvalue())
    return str(path)


@pytest.fixture
def backend():
    with patch.object(LlamaCppBackend, "_kill_orphaned_servers"):
        with patch("atexit.register"):
            return LlamaCppBackend()


class TestIsEmbeddingGguf:
    def test_false_on_fresh_backend(self, backend):
        assert backend._pooling_type is None
        assert backend.is_embedding_gguf is False

    def test_false_on_minimal_backend_without_path_state(self):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._pooling_type = None
        assert backend.is_embedding_gguf is False

    @pytest.mark.parametrize("pooling_type", [POOLING_MEAN, POOLING_CLS, POOLING_LAST])
    def test_true_for_every_sequence_pooling_mode(self, tmp_path, backend, pooling_type):
        backend._read_gguf_metadata(_make_gguf(tmp_path, "bert", pooling_type = pooling_type))
        assert backend._pooling_type == pooling_type
        assert backend.is_embedding_gguf is True

    def test_pooling_before_architecture_is_detected(self, tmp_path, backend):
        backend._read_gguf_metadata(
            _make_gguf(tmp_path, "bert", pooling_type = POOLING_CLS, pooling_first = True)
        )
        assert backend._pooling_type == POOLING_CLS
        assert backend.is_embedding_gguf is True

    def test_false_when_the_header_pools_nothing(self, tmp_path, backend):
        # Pooling NONE returns per-token vectors, which /v1/embeddings cannot shape.
        backend._read_gguf_metadata(_make_gguf(tmp_path, "bert", pooling_type = POOLING_NONE))
        assert backend._pooling_type == POOLING_NONE
        assert backend.is_embedding_gguf is False

    def test_false_for_a_reranker(self, tmp_path, backend):
        # send_embedding would read n_embd_out floats from a RANK head's n_cls_out buffer.
        backend._read_gguf_metadata(_make_gguf(tmp_path, "qwen3", pooling_type = POOLING_RANK))
        assert backend._pooling_type == POOLING_RANK
        assert backend.is_embedding_gguf is False

    def test_false_for_a_chat_gguf(self, tmp_path, backend):
        backend._read_gguf_metadata(_make_gguf(tmp_path, "llama"))
        assert backend._pooling_type is None
        assert backend.is_embedding_gguf is False

    def test_true_for_dedicated_embedding_arch_without_pooling_type(self, tmp_path, backend):
        # nomic-bert and similar encoder GGUFs often omit pooling_type in the header.
        backend._read_gguf_metadata(_make_gguf(tmp_path, "nomic-bert-moe"))
        assert backend._pooling_type is None
        assert backend.is_embedding_gguf is True

    def test_true_for_embedding_name_hint_without_pooling_type(self, tmp_path, backend):
        backend._model_identifier = "unsloth/Qwen3-Embedding-4B"
        backend._read_gguf_metadata(
            _make_gguf(tmp_path, "qwen3", filename = "Qwen3-Embedding-4B-Q4_K_M.gguf")
        )
        assert backend._pooling_type is None
        assert backend.is_embedding_gguf is True

    def test_resets_between_parses(self, tmp_path, backend):
        backend._read_gguf_metadata(
            _make_gguf(tmp_path, "bert", pooling_type = POOLING_CLS, filename = "embed.gguf")
        )
        assert backend.is_embedding_gguf is True
        backend._read_gguf_metadata(_make_gguf(tmp_path, "llama", filename = "chat.gguf"))
        assert backend.is_embedding_gguf is False

    def test_false_after_unload(self, tmp_path, backend):
        # A stale pooling type would report an unloaded backend as an embedding server.
        backend._read_gguf_metadata(_make_gguf(tmp_path, "bert", pooling_type = POOLING_CLS))
        assert backend.is_embedding_gguf is True
        backend.unload_model()
        assert backend._pooling_type is None
        assert backend.is_embedding_gguf is False

    def test_probe_reads_the_arch_prefixed_key_only(self, tmp_path, backend):
        # A pooling_type under the wrong arch prefix is another model's key.
        backend._read_gguf_metadata(_make_gguf(tmp_path, "bert", pooling_type = POOLING_CLS))
        assert backend.is_embedding_gguf is True
        buf = io.BytesIO()
        buf.write(struct.pack("<I", 0x46554747))
        buf.write(struct.pack("<I", 3))
        buf.write(struct.pack("<Q", 0))
        buf.write(struct.pack("<Q", 2))
        _write_kv(buf, "general.architecture", "llama", _VTYPE_STRING)
        _write_kv(buf, "bert.pooling_type", POOLING_CLS, _VTYPE_UINT32)
        mismatched = tmp_path / "mismatched.gguf"
        mismatched.write_bytes(buf.getvalue())
        backend._read_gguf_metadata(str(mismatched))
        assert backend.is_embedding_gguf is False


class TestLoadModelEmitsTheFlag:
    """load_model is too large to drive here, so pin its source, as the
    GPU-memory-mode and batch-size suites do for the same command block."""

    def test_embedding_flag_is_gated_on_the_header_probe(self):
        src = inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)
        guard = src.find("if self.is_embedding_gguf:")
        assert guard != -1, "load_model must decide --embedding from the GGUF header"
        emit = src.find('cmd.append("--embedding")', guard)
        assert emit != -1 and emit - guard < 120, "--embedding must sit under that guard"

    def test_the_flag_is_never_unconditional(self):
        src = inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)
        base_start = src.find("cmd = [")
        base_end = src.find("\n                ]", base_start)
        assert '"--embedding"' not in src[base_start:base_end], (
            "--embedding restricts llama-server to embeddings, so it must never be "
            "in the base command every chat model launches with"
        )

    def test_slots_are_clamped_to_the_micro_batch(self):
        # The slots follow the micro-batch down, or --embedding aborts the load.
        src = inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)
        guard = src.find("_effective_ubatch < n_parallel")
        assert guard != -1, "load_model must compare the micro-batch against the slot count"
        block = src[guard : guard + 900]
        assert "n_parallel = _embedding_slots" in block, "slots must clamp to the micro-batch"
        assert (
            "max(1, _effective_ubatch)" in block
        ), "the clamp must floor at one slot; --parallel 0 is rejected at arg parse"
        assert "allow-slot-clamp:" in block, "the clamp needs the lint marker and a reason"
        assert (
            "_effective_ubatch = _ubatch_for_slots(n_parallel)" in block
        ), "the micro-batch must be re-derived at the reduced slot count"
        assert (
            src.find("self.is_embedding_gguf", guard - 400, guard) != -1
        ), "the clamp must be gated on the embedding probe"
        assert guard < src.find("cmd = ["), "the clamp must land before the fit and the launch"

    def test_pooling_is_left_at_the_model_default(self):
        src = inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)
        assert '"--pooling"' not in src, (
            "the GGUF's own pooling type is correct; pinning one here would "
            "override rerank (RANK) and mean-pooled models"
        )

    def test_inherited_pooling_cannot_override_the_header_probe(self):
        src = inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)
        for name in ("LLAMA_ARG_POOLING", "LLAMA_ARG_RERANKING", "LLAMA_ARG_EMBEDDINGS"):
            assert f'"{name}"' in src


@pytest.mark.parametrize("flag", ["--embedding", "--embeddings", "--pooling"])
def test_user_extra_args_still_cannot_pass_the_flag(flag):
    # The denylist keeps a user-supplied --embedding off the chat server; the
    # header probe is the only thing allowed to turn it on.
    from core.inference.llama_server_args import is_managed_flag, validate_extra_args
    assert is_managed_flag(flag) is True
    with pytest.raises(ValueError, match = "managed by Unsloth Studio"):
        validate_extra_args([flag])
