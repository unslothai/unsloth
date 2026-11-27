# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF is the publisher's file: the counts and lengths its header declares must not size what
Studio builds in memory while reading it."""

import struct
import tracemalloc

import pytest

from core.inference.llama_cpp import LlamaCppBackend

PEAK_LIMIT = 48 * 1024 * 1024


def _kv_string(key: str, value: bytes) -> bytes:
    k = key.encode()
    return (
        struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value
    )


def _kv_u32(key: str, value: int) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_u32_array(key: str, count: int) -> bytes:
    k = key.encode()
    return (
        struct.pack("<Q", len(k))
        + k
        + struct.pack("<I", 9)
        + struct.pack("<I", 4)
        + struct.pack("<Q", count)
        + b"\x00" * (4 * count)
    )


def _write_gguf(path, *kvs: bytes):
    with open(path, "wb") as handle:
        handle.write(struct.pack("<IIQQ", 0x46554747, 3, 0, len(kvs)))
        for kv in kvs:
            handle.write(kv)


def _read(path):
    probe = object.__new__(LlamaCppBackend)
    probe._model_identifier = "publisher/model"
    tracemalloc.start()
    try:
        probe._read_gguf_metadata(str(path))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return probe, peak


@pytest.fixture(autouse = True)
def _offline(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_OFFLINE", "1")


def test_declared_layer_count_does_not_size_per_layer_lists(tmp_path):
    path = tmp_path / "m.gguf"
    _write_gguf(
        path,
        _kv_string("general.architecture", b"llama"),
        _kv_u32("llama.block_count", 50_000_000),
        _kv_u32("llama.attention.sliding_window_pattern", 2),
        _kv_u32("llama.context_length", 4096),
    )
    probe, peak = _read(path)
    assert peak < PEAK_LIMIT, f"reading the header held {peak / 2**20:.0f} MiB"
    assert probe._architecture == "llama"
    assert probe._context_length == 4096


def test_declared_array_length_does_not_size_a_list(tmp_path):
    path = tmp_path / "m.gguf"
    _write_gguf(
        path,
        _kv_string("general.architecture", b"llama"),
        _kv_u32_array("llama.attention.head_count_kv", 5_000_000),
        _kv_u32("llama.context_length", 4096),
    )
    probe, peak = _read(path)
    assert peak < PEAK_LIMIT, f"reading the header held {peak / 2**20:.0f} MiB"
    # The read stays aligned past the skipped array.
    assert probe._context_length == 4096


def test_huge_chat_template_is_skipped(tmp_path):
    path = tmp_path / "m.gguf"
    _write_gguf(
        path,
        _kv_string("general.architecture", b"llama"),
        _kv_string("tokenizer.chat_template", b"x" * (32 * 1024 * 1024)),
        _kv_u32("llama.context_length", 4096),
    )
    probe, peak = _read(path)
    assert peak < PEAK_LIMIT, f"reading the header held {peak / 2**20:.0f} MiB"
    assert probe._chat_template is None
    assert probe._context_length == 4096


def test_ordinary_header_reads_as_before(tmp_path):
    path = tmp_path / "m.gguf"
    _write_gguf(
        path,
        _kv_string("general.architecture", b"llama"),
        _kv_u32("llama.block_count", 32),
        _kv_u32("llama.attention.sliding_window_pattern", 4),
        _kv_string("tokenizer.chat_template", b"{{ messages }}"),
        _kv_u32_array("llama.attention.head_count_kv", 32),
    )
    probe, _ = _read(path)
    assert probe._n_layers == 32
    assert probe._chat_template == "{{ messages }}"
    assert probe._sliding_window_pattern == [(i + 1) % 4 != 0 for i in range(32)]
    assert probe._n_kv_heads_by_layer == [0] * 32
