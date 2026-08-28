# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for charging the tied-embedding output duplicate to the VRAM budget.

A model that ties its embeddings ships no ``output.weight``; llama.cpp
re-creates it from ``token_embd`` as TENSOR_DUPLICATED and a second vocabulary
matrix is really allocated. Sizing the load from the GGUF file alone therefore
UNDER-counts, which is the dangerous direction: it leaves the context search
believing there is VRAM the load will consume.

Anchored on measurement. gemma-4-E2B-it UD-Q4_K_XL sums to 3021.88 MiB of
tensors, and llama-server reported 3285.89 MiB of model buffers for it -- the
difference is 264.01 MiB against a ``token_embd`` of exactly 264.00 MiB, and the
two copies land on DIFFERENT devices (the original in CPU_Mapped, the duplicate
in CUDA0), which is why the duplicate is a VRAM cost and not merely a RAM one.

Pure: no GPU, no network, no subprocess. GGUFs are synthesised in a tmp_path.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# ---------------------------------------------------------------------------
# A minimal but real GGUF, so the probe is exercised through the same reader the
# product uses rather than a mock that could agree with a wrong implementation.
# ---------------------------------------------------------------------------

_GGUF_MAGIC = 0x46554747
_TYPE_F32 = 0


def _write_gguf(path: Path, tensors: "list[tuple[str, tuple[int, ...]]]") -> None:
    """Write a GGUF v3 with `tensors` as (name, shape), all f32, no KV pairs."""
    blobs: list[bytes] = []
    infos = bytearray()
    offset = 0
    for name, shape in tensors:
        raw = name.encode()
        infos += struct.pack("<Q", len(raw)) + raw
        infos += struct.pack("<I", len(shape))
        for dim in shape:
            infos += struct.pack("<Q", dim)
        infos += struct.pack("<I", _TYPE_F32)
        infos += struct.pack("<Q", offset)
        nbytes = 4
        for dim in shape:
            nbytes *= dim
        blobs.append(b"\0" * nbytes)
        offset += nbytes

    header = struct.pack("<II", _GGUF_MAGIC, 3) + struct.pack("<QQ", len(tensors), 1)
    # One KV pair: general.alignment, so the reader has a well-defined alignment
    # and the data section starts where the tensor offsets say it does.
    key = b"general.alignment"
    kv = struct.pack("<Q", len(key)) + key + struct.pack("<I", 4) + struct.pack("<I", 32)

    body = header + kv + bytes(infos)
    pad = (-len(body)) % 32
    with open(path, "wb") as fh:
        fh.write(body)
        fh.write(b"\0" * pad)
        for blob in blobs:
            fh.write(blob)


@pytest.fixture(scope = "module")
def backend():
    pytest.importorskip("gguf")
    from core.inference.llama_cpp import LlamaCppBackend
    return LlamaCppBackend


@pytest.fixture
def tied_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "tied.gguf"
    _write_gguf(
        path,
        [
            ("token_embd.weight", (8, 64)),  # 2048 bytes at f32
            ("blk.0.attn_q.weight", (8, 8)),
            ("blk.0.ffn_down.weight", (8, 8)),
        ],
    )
    return path


@pytest.fixture
def untied_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "untied.gguf"
    _write_gguf(
        path,
        [
            ("token_embd.weight", (8, 64)),
            ("output.weight", (8, 64)),
            ("blk.0.attn_q.weight", (8, 8)),
        ],
    )
    return path


def test_a_tied_model_is_charged_one_more_embedding_matrix(backend, tied_gguf):
    # 8 * 64 * 4 bytes. The duplicate is the WHOLE matrix, not a fraction of it.
    assert backend._tied_output_bytes(str(tied_gguf)) == 8 * 64 * 4


def test_a_model_shipping_its_own_output_is_charged_nothing(backend, untied_gguf):
    # The file already contains both tensors, so the file size covers the load
    # and adding anything would over-count. This is the Qwen3.6 / Qwen3.8 case.
    assert backend._tied_output_bytes(str(untied_gguf)) == 0


def test_the_charge_is_the_embedding_size_not_a_constant(backend, tmp_path):
    """Two tied models of different vocabulary sizes must differ.

    Guards against a fixed fudge factor, which would be right for one model and
    wrong for every other: the real spread across the shipped gemma quants is
    264 MiB (E2B UD-Q4_K_XL) to 924 MiB (31B UD-Q4_K_XL).
    """
    small = tmp_path / "small.gguf"
    large = tmp_path / "large.gguf"
    _write_gguf(small, [("token_embd.weight", (8, 16))])
    _write_gguf(large, [("token_embd.weight", (8, 64))])
    assert backend._tied_output_bytes(str(large)) == 4 * backend._tied_output_bytes(str(small))


def test_a_split_gguf_abstains(backend, tmp_path):
    """A shard cannot see its siblings, so absence of output.weight proves nothing.

    GGUFReader maps only the path it is given. On a split model ``output.weight``
    may live in a later shard, which would read as "tied" here and add a
    duplicate that is never allocated -- an over-count, on exactly the models
    large enough to be split. Abstaining costs the old behaviour; guessing costs
    a wrongly shrunken context.
    """
    path = tmp_path / "model-00001-of-00003.gguf"
    _write_gguf(path, [("token_embd.weight", (8, 64))])
    assert backend._tied_output_bytes(str(path)) == 0


def test_an_unreadable_file_costs_the_old_budget_rather_than_the_launch(backend, tmp_path):
    """The budget must never be the reason a load fails.

    A truncated or non-GGUF file returns 0, which is exactly the behaviour
    before this change, instead of propagating out of the context search.
    """
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf at all")
    assert backend._tied_output_bytes(str(junk)) == 0
    assert backend._tied_output_bytes(str(tmp_path / "missing.gguf")) == 0


def test_the_probe_is_cached_on_file_identity_not_path(backend, tmp_path):
    """A model replaced in place must not serve the previous answer.

    The context search calls this once per candidate context, so it has to be
    cached; keying on the path alone would make a re-downloaded or re-quantised
    file keep its predecessor's charge.
    """
    path = tmp_path / "swapped.gguf"
    _write_gguf(path, [("token_embd.weight", (8, 64)), ("output.weight", (8, 64))])
    assert backend._tied_output_bytes(str(path)) == 0

    # Same name, different contents: now tied, and larger.
    _write_gguf(path, [("token_embd.weight", (8, 128))])
    assert backend._tied_output_bytes(str(path)) == 8 * 128 * 4


def test_the_budget_adds_the_duplicate_to_the_gguf_size(backend, tied_gguf):
    """The call site adds it; the helper alone proves nothing.

    Reads the source of the context-budget block rather than driving a full
    load, which needs a GPU and a binary. What must hold is that `model_size`
    is not the bare file size any more.
    """
    import inspect

    src = inspect.getsource(backend)
    assert (
        "model_size = gguf_size + mmproj_size + self._tied_output_bytes(model_path)" in src
    ), "the context budget no longer charges the tied-embedding duplicate"
