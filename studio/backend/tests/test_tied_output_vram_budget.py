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

import os
import struct
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# ---------------------------------------------------------------------------
# A minimal but real GGUF, so the probe is exercised against the byte layout the
# product reads rather than a mock that could agree with a wrong implementation.
# ---------------------------------------------------------------------------

_GGUF_MAGIC = 0x46554747
_TYPE_F32 = 0
_TYPE_Q4_K = 12  # 256-element blocks of 144 bytes


def _write_gguf(
    path: Path,
    tensors: "list[tuple[str, tuple[int, ...]]]",
    *,
    architecture: "str | None" = None,
    ggml_type: int = _TYPE_F32,
    pad: int = 0,
) -> Path:
    """Write a GGUF v3 with `tensors` as (name, shape).

    ``pad`` appends filler past the tensor data, which no reader of the header
    looks at: it exists so two files of different content can be given the same
    byte length for the cache-identity test.
    """
    block, block_bytes = (1, 4) if ggml_type == _TYPE_F32 else (256, 144)
    blobs: list[bytes] = []
    infos = bytearray()
    offset = 0
    for name, shape in tensors:
        raw = name.encode()
        infos += struct.pack("<Q", len(raw)) + raw
        infos += struct.pack("<I", len(shape))
        for dim in shape:
            infos += struct.pack("<Q", dim)
        infos += struct.pack("<I", ggml_type)
        infos += struct.pack("<Q", offset)
        elements = 1
        for dim in shape:
            elements *= dim
        nbytes = elements // block * block_bytes
        blobs.append(b"\0" * nbytes)
        offset += nbytes

    def _string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    # general.alignment, so the reader has a well-defined alignment and the data
    # section starts where the tensor offsets say it does.
    kv = _string("general.alignment") + struct.pack("<II", 4, 32)
    n_kv = 1
    if architecture is not None:
        kv += _string("general.architecture") + struct.pack("<I", 8) + _string(architecture)
        n_kv += 1

    body = struct.pack("<II", _GGUF_MAGIC, 3) + struct.pack("<QQ", len(tensors), n_kv)
    body += kv + bytes(infos)
    with open(path, "wb") as fh:
        fh.write(body)
        fh.write(b"\0" * ((-len(body)) % 32))
        for blob in blobs:
            fh.write(blob)
        fh.write(b"\0" * pad)
    return path


@pytest.fixture(scope = "module")
def backend():
    pytest.importorskip("gguf")
    from core.inference.llama_cpp import LlamaCppBackend
    return LlamaCppBackend


@pytest.fixture
def tied_gguf(tmp_path: Path) -> Path:
    return _write_gguf(
        tmp_path / "tied.gguf",
        [
            ("token_embd.weight", (8, 64)),  # 2048 bytes at f32
            ("blk.0.attn_q.weight", (8, 8)),
            ("blk.0.ffn_down.weight", (8, 8)),
        ],
    )


@pytest.fixture
def untied_gguf(tmp_path: Path) -> Path:
    return _write_gguf(
        tmp_path / "untied.gguf",
        [
            ("token_embd.weight", (8, 64)),
            ("output.weight", (8, 64)),
            ("blk.0.attn_q.weight", (8, 8)),
        ],
    )


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
    small = _write_gguf(tmp_path / "small.gguf", [("token_embd.weight", (8, 16))])
    large = _write_gguf(tmp_path / "large.gguf", [("token_embd.weight", (8, 64))])
    assert backend._tied_output_bytes(str(large)) == 4 * backend._tied_output_bytes(str(small))


def test_a_quantised_embedding_is_charged_its_blocks_not_its_elements(backend, tmp_path):
    """Every shipped tied model is quantised, so element count is not byte count.

    Q4_K stores 256 elements in 144 bytes. Reading the shape and multiplying by
    an f32 element size would over-charge that matrix by 7.1x.
    """
    path = _write_gguf(
        tmp_path / "q4k.gguf",
        [("token_embd.weight", (256, 64))],
        ggml_type = _TYPE_Q4_K,
    )
    assert backend._tied_output_bytes(str(path)) == 256 * 64 // 256 * 144


def test_a_quant_type_the_pinned_gguf_package_predates_is_still_charged(backend, tmp_path):
    """Studio runs a user-supplied llama.cpp; `gguf` is pinned in pyproject.

    A GGUF quantised with a type added after the pinned package is one the
    selected llama-server loads fine, so aborting the probe on the unknown enum
    would silently restore the whole under-count. The data layout bounds the
    tensor without the table: the next tensor's offset ends the previous one.
    """
    path = tmp_path / "future-quant.gguf"
    _write_gguf(
        path,
        [("token_embd.weight", (256, 64)), ("blk.0.attn_q.weight", (256, 8))],
        ggml_type = _TYPE_Q4_K,
    )
    expected = 256 * 64 // 256 * 144
    assert backend._tied_output_bytes(str(path)) == expected

    # The same file with the quant enum bumped past everything the table knows.
    from gguf.constants import GGML_QUANT_SIZES

    unknown = max(GGML_QUANT_SIZES) + 1
    raw = bytearray(path.read_bytes())
    assert raw.count(struct.pack("<I", _TYPE_Q4_K)) >= 1
    raw = raw.replace(struct.pack("<I", _TYPE_Q4_K), struct.pack("<I", unknown))
    (tmp_path / "bumped.gguf").write_bytes(raw)

    charged = backend._tied_output_bytes(str(tmp_path / "bumped.gguf"))
    assert charged >= expected, "an unknown quant type must not silently cost 0"
    assert charged < expected + 32, "and it must not over-count by more than the alignment"

    # And when token_embd is the last tensor there is no next offset, so the
    # data section itself has to bound it.
    only = tmp_path / "only.gguf"
    _write_gguf(only, [("token_embd.weight", (256, 64))], ggml_type = _TYPE_Q4_K)
    raw = bytearray(only.read_bytes())
    only.write_bytes(bytes(raw).replace(struct.pack("<I", _TYPE_Q4_K), struct.pack("<I", unknown)))
    last = backend._tied_output_bytes(str(only))
    assert expected <= last < expected + 32


def test_an_encoder_only_architecture_is_charged_nothing(backend, tmp_path):
    """BERT ships token_embd and no output.weight, yet nothing is duplicated.

    src/models/bert.cpp creates tok_embd and never an output tensor: the model
    produces no vocabulary logits at all, so its missing output.weight is not
    tying. Studio launches these through is_embedding_gguf, so they reach this
    budget, and charging them shrinks the context for VRAM nobody allocates.
    """
    path = _write_gguf(
        tmp_path / "bert.gguf",
        [("token_embd.weight", (8, 64))],
        architecture = "bert",
    )
    assert backend._tied_output_bytes(str(path)) == 0


def test_an_unrecognised_architecture_is_still_charged(backend, tmp_path):
    """The exemption is a blocklist, and it fails towards charging.

    A new decoder arch that is not listed over-counts by one embedding matrix;
    a new one wrongly exempted would under-count by one, which is what makes the
    search promise VRAM the load then takes.
    """
    path = _write_gguf(
        tmp_path / "future.gguf",
        [("token_embd.weight", (8, 64))],
        architecture = "some-arch-from-2027",
    )
    assert backend._tied_output_bytes(str(path)) == 8 * 64 * 4


def test_a_split_model_is_charged_when_no_shard_ships_an_output(backend, tmp_path):
    """The largest tied models are the split ones, so abstaining costs the most.

    token_embd is in shard 1 and output.weight, if the model had one, would be
    in a later shard, so shard 1 alone cannot answer. Every shard is scanned.
    """
    _write_gguf(tmp_path / "m-00001-of-00002.gguf", [("token_embd.weight", (8, 64))])
    _write_gguf(tmp_path / "m-00002-of-00002.gguf", [("blk.1.attn_q.weight", (8, 8))])
    assert backend._tied_output_bytes(str(tmp_path / "m-00001-of-00002.gguf")) == 8 * 64 * 4


def test_a_split_model_finds_its_output_weight_in_a_later_shard(backend, tmp_path):
    """Reading only the shard it was handed would call this model tied.

    That is the over-count, on exactly the models big enough to be split.
    """
    _write_gguf(tmp_path / "m-00001-of-00002.gguf", [("token_embd.weight", (8, 64))])
    _write_gguf(tmp_path / "m-00002-of-00002.gguf", [("output.weight", (8, 64))])
    assert backend._tied_output_bytes(str(tmp_path / "m-00001-of-00002.gguf")) == 0


def test_a_split_model_with_a_shard_missing_costs_the_old_budget(backend, tmp_path):
    """An unreadable sibling proves nothing, so it must not be read as tied."""
    _write_gguf(tmp_path / "m-00001-of-00003.gguf", [("token_embd.weight", (8, 64))])
    assert backend._tied_output_bytes(str(tmp_path / "m-00001-of-00003.gguf")) == 0


def test_an_unreadable_file_costs_the_old_budget_rather_than_the_launch(backend, tmp_path):
    """The budget must never be the reason a load fails.

    A truncated or non-GGUF file returns 0, which is exactly the behaviour
    before this change, instead of propagating out of the context search.
    """
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf at all")
    assert backend._tied_output_bytes(str(junk)) == 0
    assert backend._tied_output_bytes(str(tmp_path / "missing.gguf")) == 0


def test_the_probe_reads_the_header_without_building_a_gguf_reader(backend, tied_gguf):
    """gguf.GGUFReader materialises every KV value, the tokenizer vocabulary included.

    Measured in the studio venv: 12.1 s on gemma-4-E2B-it UD-Q4_K_XL and 6.2 s
    on the 240 MiB gemma-3-270m-it, against 30-90 ms for the streaming read --
    the cost tracks vocabulary size, not file size, so a small tied model pays it
    too. This runs under the backend lock before llama-server is spawned, so it
    would be a multi-second stall on every llama.cpp load.
    """
    import gguf

    def explode(*_args, **_kwargs):
        raise AssertionError("the tied-output probe must not construct a GGUFReader")

    original = gguf.GGUFReader
    gguf.GGUFReader = explode
    try:
        backend._tied_output_bytes_cached.cache_clear()
        assert backend._tied_output_bytes(str(tied_gguf)) == 8 * 64 * 4
    finally:
        gguf.GGUFReader = original


def test_the_probe_cache_is_keyed_on_the_inode_not_the_path_size_and_mtime(backend, tmp_path):
    """A model rebuilt in place must not serve its predecessor's answer.

    Path, size and mtime do not identify a file: an atomic replacement that
    restores the timestamp matches on all three. The backend's own
    _gguf_load_source_identity already keys on device and inode, and this probe
    reuses it, so the swap below is seen.
    """
    path = tmp_path / "swapped.gguf"
    _write_gguf(path, [("token_embd.weight", (8, 64)), ("output.weight", (8, 64))])
    assert backend._tied_output_bytes(str(path)) == 0

    before = path.stat()
    replacement = tmp_path / "replacement.gguf"
    _write_gguf(replacement, [("token_embd.weight", (8, 64))])
    # Same length and same nanosecond timestamp, different inode.
    pad = before.st_size - replacement.stat().st_size
    assert pad >= 0
    _write_gguf(replacement, [("token_embd.weight", (8, 64))], pad = pad)
    os.utime(replacement, ns = (before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, path)
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    assert after.st_ino != before.st_ino

    assert backend._tied_output_bytes(str(path)) == 8 * 64 * 4


def test_every_site_that_prices_the_weights_carries_the_duplicate(backend):
    """The charge must survive the vision path, not just the first assignment.

    Driving this through a real load needs a GPU and a binary, and the projector
    pin is a closure with no seam, so the placement behaviour is covered in
    test_mmproj_placement_policy.py and what is checked here is that no site
    re-derives the footprint from the bare file size again. `weights_size` is the
    one name that carries the correction; `gguf_size` is the file.
    """
    import inspect

    src = inspect.getsource(backend.load_model)
    assert "weights_size = gguf_size + self._tied_output_bytes(model_path)" in src
    assert "model_size = weights_size + mmproj_size" in src
    # The projector CPU pin removes the projector, not the duplicate.
    assert "model_size = gguf_size" not in src
    # And the probe that decides that pin prices what the placement prices.
    assert "_mm_need = (\n                            weights_size" in src
