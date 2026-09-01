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


def test_a_separate_tied_drafter_charges_its_duplicated_output(backend, tied_gguf):
    instance = backend()
    embedding = 8 * 64 * 4
    raw_size = instance._get_gguf_size_bytes(str(tied_gguf))

    assert instance._separate_drafter_weight_vram_bytes(str(tied_gguf), embedding) == raw_size
    assert instance._separate_drafter_weight_vram_bytes(str(tied_gguf), 0) == raw_size + embedding


def test_the_charge_is_the_embedding_size_not_a_constant(backend, tmp_path):
    """Two tied models of different vocabulary sizes must differ.

    Guards against a fixed fudge factor, which would be right for one model and
    wrong for every other: the real spread across the shipped gemma quants is
    264 MiB (E2B UD-Q4_K_XL) to 924 MiB (31B UD-Q4_K_XL).
    """
    small = _write_gguf(tmp_path / "small.gguf", [("token_embd.weight", (8, 16))])
    large = _write_gguf(tmp_path / "large.gguf", [("token_embd.weight", (8, 64))])
    assert backend._tied_output_bytes(str(large)) == 4 * backend._tied_output_bytes(str(small))


def test_a_split_gguf_is_read_across_every_shard(backend, tmp_path):
    """The probe must inspect every shard before inferring a tie or discount."""
    one = tmp_path / "m-00001-of-00002.gguf"
    two = tmp_path / "m-00002-of-00002.gguf"
    _write_gguf(one, [("token_embd.weight", (8, 64))])
    _write_gguf(two, [("output.weight", (8, 64)), ("blk.0.ffn_up.weight", (8, 8))])
    # output.weight in shard 2 makes this model untied.
    assert backend._tied_output_bytes(str(one)) == 0

    # Without output.weight in any shard, the embedding is tied and charged.
    three = tmp_path / "n-00001-of-00002.gguf"
    four = tmp_path / "n-00002-of-00002.gguf"
    _write_gguf(three, [("token_embd.weight", (8, 64))])
    _write_gguf(four, [("blk.0.ffn_up.weight", (8, 8))])
    assert backend._tied_output_bytes(str(three)) == 8 * 64 * 4

    # Ignore stale files outside the declared 1..N launch set.
    stale = tmp_path / "n-00003-of-00002.gguf"
    _write_gguf(stale, [("output.weight", (1, 1))])
    assert [p.name for p in backend._gguf_shard_paths(str(three))] == [three.name, four.name]
    assert backend._tied_output_bytes(str(three)) == 8 * 64 * 4

    # A partial split cannot answer either correction safely.
    partial = tmp_path / "q-00001-of-00002.gguf"
    _write_gguf(partial, [("token_embd.weight", (8, 64))])
    assert backend._tied_output_bytes(str(partial)) == 0
    assert backend._host_pinned_weight_bytes(str(partial)) == 0


def test_the_per_layer_embedding_is_counted_from_a_later_shard(backend, tmp_path):
    """The largest host-pinned tensor is not required to be in shard 1."""
    one = tmp_path / "p-00001-of-00002.gguf"
    two = tmp_path / "p-00002-of-00002.gguf"
    _write_gguf(one, [("token_embd.weight", (8, 64))])
    _write_gguf(two, [("per_layer_token_embd.weight", (16, 64)), ("output.weight", (8, 64))])
    assert backend._host_pinned_weight_bytes(str(one)) == (8 * 64 * 4) + (16 * 64 * 4)


def test_host_pinned_covers_both_embedding_families(backend, tied_gguf, tmp_path):
    # token_embd alone on a model without per-layer embeddings.
    assert backend._host_pinned_weight_bytes(str(tied_gguf)) == 8 * 64 * 4

    ple = tmp_path / "ple.gguf"
    _write_gguf(
        ple,
        [
            ("token_embd.weight", (8, 64)),
            ("per_layer_token_embd.weight", (32, 64)),
            ("blk.0.ffn_up.weight", (8, 8)),
        ],
    )
    assert backend._host_pinned_weight_bytes(str(ple)) == (8 * 64 * 4) + (32 * 64 * 4)


def test_host_pinned_is_zero_for_an_unreadable_file(backend, tmp_path):
    junk = tmp_path / "junk2.gguf"
    junk.write_bytes(b"nope")
    assert backend._host_pinned_weight_bytes(str(junk)) == 0
    assert backend._host_pinned_weight_bytes(str(tmp_path / "gone.gguf")) == 0


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
        backend._host_pinned_weight_bytes_cached.cache_clear()
        backend._host_pinned_weight_items_cached.cache_clear()
        assert backend._tied_output_bytes(str(tied_gguf)) == 8 * 64 * 4
        assert backend._host_pinned_weight_bytes(str(tied_gguf)) == 8 * 64 * 4
        assert backend._host_pinned_weight_items(str(tied_gguf)) == (
            ("token_embd.weight", 8 * 64 * 4),
        )
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


def test_the_host_pinned_cache_is_keyed_on_file_identity(backend, tmp_path):
    path = tmp_path / "host-swapped.gguf"
    _write_gguf(path, [("token_embd.weight", (8, 64))])
    assert backend._host_pinned_weight_bytes(str(path)) == 8 * 64 * 4

    before = path.stat()
    replacement = tmp_path / "host-replacement.gguf"
    _write_gguf(replacement, [("token_embd.weight", (8, 16))])
    pad = before.st_size - replacement.stat().st_size
    assert pad >= 0
    _write_gguf(replacement, [("token_embd.weight", (8, 16))], pad = pad)
    os.utime(replacement, ns = (before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, path)
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    assert after.st_ino != before.st_ino

    assert backend._host_pinned_weight_bytes(str(path)) == 8 * 16 * 4
    assert backend._host_pinned_weight_items(str(path)) == (("token_embd.weight", 8 * 16 * 4),)


def test_advanced_spec_drafter_does_not_inherit_the_main_cpu_device():
    from core.inference import llama_cpp

    advanced = [
        "--spec-type",
        "draft-simple",
        "--model-draft",
        "draft.gguf",
        "--device",
        "none",
    ]
    assert llama_cpp._extra_args_draft_device(advanced) is None
    assert llama_cpp._extra_args_draft_offloaded_to_cpu(advanced, {}) is False
    assert llama_cpp._extra_args_effective_draft_device_pin(advanced) is None

    generated = ["--model-draft", "draft.gguf", "--device", "none"]
    assert llama_cpp._extra_args_draft_offloaded_to_cpu(generated, {}) is True


def test_the_budget_sizes_from_what_lands_in_vram(backend, tied_gguf):
    """The pure budget seam keeps discrete and shared-memory arithmetic distinct."""
    import inspect

    expected = 8 * 64 * 4
    assert (
        backend._host_pinned_vram_discount(str(tied_gguf), [], env = {}, shared_memory = False)
        == expected
    )
    assert backend._host_pinned_vram_discount(str(tied_gguf), [], env = {}, shared_memory = True) == 0
    assert (
        backend._host_pinned_vram_discount(
            str(tied_gguf),
            [],
            env = {"LLAMA_ARG_OVERRIDE_TENSOR": "token_embd.weight=CUDA0"},
            shared_memory = False,
        )
        == 0
    )

    src = inspect.getsource(backend)
    assert (
        "+ self._tied_output_bytes(model_path)" in src
    ), "the context budget no longer charges the tied-embedding duplicate"
    assert "- _host_pinned" in src, "the context budget no longer discounts host-pinned embeddings"
    assert "_host_pinned_vram_discount(" in src
    assert "env = os.environ" in src
    assert "shared_memory = False" in src
    assert "_host_pinned = 0 if _shared_memory else _host_pinned_candidate" in src
    assert "_candidate_targets_proved_discrete" in src
    assert "_shared_gpu_ids | _unclassified_gpu_ids" in src


def test_a_draft_override_owns_only_the_drafter_discount(backend, tied_gguf):
    expected = 8 * 64 * 4
    draft_args = ["-otd", "token_embd.weight=CUDA0"]
    main_args = ["-ot", "token_embd.weight=CUDA0"]

    assert (
        backend._host_pinned_vram_discount(
            str(tied_gguf),
            draft_args,
            env = {},
            shared_memory = False,
            draft_model = True,
        )
        == 0
    )
    assert (
        backend._host_pinned_vram_discount(
            str(tied_gguf),
            main_args,
            env = {},
            shared_memory = False,
            draft_model = True,
        )
        == expected
    )
    assert backend._override_moves_host_pinned(draft_args, env = {}) is False


def test_vulkan_igpu_is_shared_memory_and_unknown_is_conservative(backend, monkeypatch):
    monkeypatch.setattr(
        backend,
        "_run_vulkan_probe",
        staticmethod(
            lambda _binary = None: [
                {"index": 0, "is_igpu": True},
                {"index": 1, "is_igpu": False},
            ]
        ),
    )
    assert backend._vulkan_targets_are_igpus("server", [0]) is True
    assert backend._vulkan_targets_are_igpus("server", [1]) is False
    assert backend._vulkan_targets_are_igpus("server", [2], conservative_on_unknown = True) is True
    assert backend._vulkan_targets_are_igpus("server", [1, 2]) is False
    assert backend._vulkan_targets_are_igpus("server", [1, 2], conservative_on_unknown = True) is True

    monkeypatch.setattr(backend, "_run_vulkan_probe", staticmethod(lambda _binary = None: []))
    assert backend._vulkan_targets_are_igpus("server", conservative_on_unknown = True) is True


def test_one_vulkan_snapshot_drives_memory_and_shared_classification(backend, monkeypatch):
    rows = [
        {"index": 0, "free_mib": 8192, "total_mib": 16384, "is_igpu": False},
        {"index": 1, "free_mib": 4096, "total_mib": 8192, "is_igpu": True},
    ]
    monkeypatch.setattr(
        backend,
        "_run_vulkan_probe",
        staticmethod(lambda *_a, **_kw: pytest.fail("snapshot was probed twice")),
    )
    assert backend._vulkan_rows_target_igpus(rows, [0]) is False
    assert backend._vulkan_rows_target_igpus(rows, [1]) is True
    memory = backend._get_gpu_free_memory_vulkan("server", rows = rows)
    assert memory[0] == (0, 8192, 16384)
    assert memory[1][0] == 1
    assert memory[1][2] == 0


def test_unreadable_cuda_properties_are_not_proved_discrete(backend, monkeypatch):
    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_properties(_ordinal):
            raise OSError("property probe failed")

    class _Version:
        hip = None

    class _Torch:
        cuda = _Cuda()
        version = _Version()
        __version__ = "2.9.0"

    monkeypatch.setitem(sys.modules, "torch", _Torch())
    monkeypatch.setattr(backend, "_resolve_visible_physical_ids", staticmethod(lambda: None))
    assert backend._torch_unified_memory_classification_known([0]) is False


def test_a_classified_discrete_cuda_device_is_known(backend, monkeypatch):
    class _Props:
        is_integrated = False

    class _Cuda:
        is_available = staticmethod(lambda: True)
        device_count = staticmethod(lambda: 1)
        get_device_properties = staticmethod(lambda _ordinal: _Props())

    class _Version:
        hip = None

    class _Torch:
        cuda = _Cuda()
        version = _Version()
        __version__ = "2.9.0"

    monkeypatch.setitem(sys.modules, "torch", _Torch())
    monkeypatch.setattr(backend, "_resolve_visible_physical_ids", staticmethod(lambda: None))
    assert backend._torch_unified_memory_classification_known([0]) is True


@pytest.mark.parametrize("arch", ["gfx90c", "gfx1103"])
def test_an_unclassified_rocm_arch_is_not_proved_discrete(backend, monkeypatch, arch):
    class _Props:
        gcnArchName = arch
        is_integrated = 0

    class _Cuda:
        is_available = staticmethod(lambda: True)
        device_count = staticmethod(lambda: 1)
        get_device_properties = staticmethod(lambda _ordinal: _Props())

    class _Version:
        hip = "6.2.0"

    class _Torch:
        cuda = _Cuda()
        version = _Version()
        __version__ = "2.9.0+rocm"

    monkeypatch.setitem(sys.modules, "torch", _Torch())
    monkeypatch.setattr(backend, "_resolve_visible_physical_ids", staticmethod(lambda: None))
    assert backend._torch_unified_memory_classification_known([0]) is False


@pytest.mark.parametrize(
    "arch",
    [
        "gfx803",
        "gfx900",
        "gfx906",
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx950",
        "gfx1010",
        "gfx1011",
        "gfx1012",
        "gfx1031",
        "gfx1100",
    ],
)
def test_a_known_discrete_rocm_arch_is_proved_discrete(backend, monkeypatch, arch):
    class _Props:
        gcnArchName = arch
        is_integrated = 0

    class _Cuda:
        is_available = staticmethod(lambda: True)
        device_count = staticmethod(lambda: 1)
        get_device_properties = staticmethod(lambda _ordinal: _Props())

    class _Version:
        hip = "6.2.0"

    class _Torch:
        cuda = _Cuda()
        version = _Version()
        __version__ = "2.9.0+rocm"

    monkeypatch.setitem(sys.modules, "torch", _Torch())
    monkeypatch.setattr(backend, "_resolve_visible_physical_ids", staticmethod(lambda: None))
    assert backend._torch_unified_memory_classification_known([0]) is True


def test_an_hsa_override_cannot_spoof_an_apu_into_a_discrete_discount(backend, monkeypatch):
    """The runtime arch is valid for kernel selection under the override, but it is
    not evidence about the underlying memory topology used by the VRAM budget."""

    class _Props:
        gcnArchName = "gfx1030"
        # Older ROCm wheels omit/zero this even for the gfx1035 laptop APU that is
        # commonly presented as gfx1030 through HSA_OVERRIDE_GFX_VERSION.
        is_integrated = 0

    class _Cuda:
        is_available = staticmethod(lambda: True)
        device_count = staticmethod(lambda: 1)
        get_device_properties = staticmethod(lambda _ordinal: _Props())

    class _Version:
        hip = "6.4"

    class _Torch:
        cuda = _Cuda()
        version = _Version()
        __version__ = "2.9.0"

    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    monkeypatch.setitem(sys.modules, "torch", _Torch())
    monkeypatch.setattr(backend, "_resolve_visible_physical_ids", staticmethod(lambda: None))

    assert backend._torch_unified_memory_classification_known([0]) is False
    # The override invalidates arch-only DISCRETE proof, not a positive unified-memory
    # signal from the device properties.
    _Props.is_integrated = 1
    assert backend._torch_unified_memory_classification_known([0]) is True


def test_a_user_override_to_a_gpu_buffer_cancels_the_discount(backend):
    """An explicit device override outranks llama.cpp's host fallback."""
    assert backend._override_moves_host_pinned(["-ot", "token_embd.weight=CUDA0"], env = {}) is True
    assert (
        backend._override_moves_host_pinned(
            ["-ot", r"^per_layer_token_embd\.weight$=CUDA0"], env = {}
        )
        is True
    )
    assert (
        backend._override_moves_host_pinned(
            ["--override_tensor", "token_embd.weight=CUDA0"], env = {}
        )
        is True
    )
    assert (
        backend._override_moves_host_pinned(["--override-tensor=token_embd.weight=CUDA0"], env = {})
        is True
    )
    assert backend._override_moves_host_pinned(["-ot=token_embd.weight=CUDA0"], env = {}) is True
    assert backend._override_moves_host_pinned(["-ot", r".*embd.*=CUDA0"], env = {}) is True
    assert backend._override_moves_host_pinned(["-ot", "token_embd=CUDA0"], env = {}) is True
    assert backend._override_moves_host_pinned(["-ot", "embd=CUDA0"], env = {}) is True
    assert backend._override_moves_host_pinned(["-ot", r"per_.*47$=CUDA0"], env = {}) is True
    # The family is open-ended, so even apparently unrelated device mappings
    # fail closed instead of relying on incomplete regex representatives.
    assert backend._override_moves_host_pinned(["-ot", r"^blk\.0=CUDA0"], env = {}) is True
    assert backend._override_moves_host_pinned(["-ot", r".*=CUDA0"], env = {}) is True
    assert (
        backend._override_moves_host_pinned(
            ["-ot", "token_embd.weight=CUDA0,blk.0.ffn_down.weight=CPU"], env = {}
        )
        is True
    )
    assert (
        backend._override_moves_host_pinned(
            [], env = {"LLAMA_ARG_OVERRIDE_TENSOR": "token_embd.weight=CUDA0"}
        )
        is True
    )
    assert (
        backend._override_moves_host_pinned([], env = {"LLAMA_ARG_OVERRIDE_TENSOR": "embd=CUDA0"})
        is True
    )


def test_cpu_only_or_absent_overrides_keep_the_discount(backend):
    # Sending them to CPU is where llama.cpp puts them anyway.
    assert backend._override_moves_host_pinned(["-ot", "token_embd.weight=CPU"], env = {}) is False
    assert (
        backend._override_moves_host_pinned(["--override_tensor=token_embd.weight=CPU"], env = {})
        is False
    )
    # Our own planner's patterns move FFN tensors, never the embeddings.
    assert (
        backend._override_moves_host_pinned(["-ot", r"^blk\.(1|2)\.ffn_down\.weight$=CPU"], env = {})
        is False
    )
    assert (
        backend._override_moves_host_pinned(
            ["-ot", "token_embd.weight=CPU,blk.0.ffn_down.weight=CUDA0"], env = {}
        )
        is True
    )
    assert backend._override_moves_host_pinned([], env = {}) is False
    assert backend._override_moves_host_pinned(None, env = {}) is False
    # A bare flag with no value must not crash the budget.
    assert backend._override_moves_host_pinned(["-ot"], env = {}) is False


def test_every_site_that_prices_the_weights_carries_both_corrections(backend):
    """The tied charge and host discount must survive the vision path.

    Driving this through a real load needs a GPU and a binary, and the projector
    pin is a closure with no seam, so the placement behaviour is covered in
    test_mmproj_placement_policy.py and what is checked here is that no site
    re-derives the footprint from the bare file size again.
    """
    import inspect

    src = inspect.getsource(backend.load_model)
    assert "weights_size = gguf_size + self._tied_output_bytes(model_path)" in src
    assert "_model_weight_vram_bytes = max(0, weights_size - _host_pinned)" in src
    assert "model_size = _model_weight_vram_bytes + mmproj_size" in src
    # The projector CPU pin removes only the projector, not either correction.
    assert "model_size = gguf_size" not in src
    # And the probe that decides that pin prices what the placement prices.
    assert "_mm_base_need = (\n                            _model_weight_vram_bytes" in src


@pytest.fixture
def tied_gguf_with_ple(tmp_path: Path) -> Path:
    """A tied model that also ships per-layer embeddings, like the gemma family."""
    return _write_gguf(
        tmp_path / "tied_ple.gguf",
        [
            ("token_embd.weight", (8, 64)),  # 2048 bytes at f32
            ("per_layer_token_embd.weight", (4, 64)),  # 1024 bytes
            ("blk.0.attn_q.weight", (8, 8)),
            ("blk.0.ffn_down.weight", (8, 8)),
        ],
    )


def test_the_two_corrections_only_cancel_as_a_set(backend, tied_gguf_with_ple):
    """Charging the tied duplicate and discounting host-pinned embeddings are HALVES.

    Reviewed separately each looks wrong, and the review of #9929 said so
    correctly: on its own, charging the tied duplicate double-counts, because
    ``gguf_size`` already carries one ``token_embd`` and llama.cpp only ever has
    one vocabulary matrix in VRAM. What that argument assumes is that the
    ``token_embd`` inside ``gguf_size`` stays there to stand in for the
    duplicate. The host-pinned discount removes it, because llama.cpp pins the
    input layer to the CPU (llama-model.cpp:1481-1483, "there is very little
    benefit to offloading the input layer"), and both embeddings are input-layer
    tensors (llama-arch.cpp:709 and :900). The one exception is the tied
    duplicate, which llama-model-loader.cpp:1118-1119 remaps to
    LLM_TENSOR_OUTPUT and does place on the GPU.

    So the three terms are only correct together, and each half alone is wrong
    by exactly one vocabulary matrix in the OPPOSITE direction. This test states
    that as arithmetic rather than as prose, and it is the reason neither half
    should be merged without the other.
    """
    path = str(tied_gguf_with_ple)
    gguf = backend._get_gguf_size_bytes(path)
    embd = 8 * 64 * 4  # token_embd, the matrix llama.cpp duplicates into VRAM
    ple = 4 * 64 * 4  # per_layer_token_embd, host-pinned and never duplicated

    assert backend._tied_output_bytes(path) == embd
    assert backend._host_pinned_weight_bytes(path) == embd + ple

    # What llama.cpp really puts in VRAM: everything in the file except the two
    # host-pinned embeddings, plus the duplicate of one of them.
    truth = gguf - (embd + ple) + embd
    assert truth == gguf - ple

    # Unbound, passing the class as self: the seam only reaches static helpers,
    # and going through it rather than re-deriving the formula here is the point
    # -- a test that recomputed the arithmetic would agree with a wrong shipped
    # implementation.
    shipped = backend._separate_drafter_weight_vram_bytes(
        backend, path, host_pinned_bytes = backend._host_pinned_weight_bytes(path)
    )
    assert shipped == truth

    # And each half on its own, priced from the same primitives.
    main_today = gguf
    tied_charge_alone = gguf + embd
    host_discount_alone = gguf - (embd + ple)

    assert main_today - truth == ple, "main over-charges by the per-layer embeddings"
    assert tied_charge_alone - truth == embd + ple, "the tied charge alone over-charges"
    assert truth - host_discount_alone == embd, (
        "the host-pinned discount alone UNDER-charges by one vocabulary matrix, "
        "which is the direction that fails a launch"
    )


@pytest.fixture
def mib_embd_pair(tmp_path: Path) -> "tuple[Path, Path]":
    """A tied and an untied model whose embedding matrix is exactly 1 MiB."""
    rows = (256, 1024)  # 262144 f32 elements
    return (
        _write_gguf(
            tmp_path / "tied_mib.gguf",
            [("token_embd.weight", rows), ("blk.0.attn_q.weight", (8, 8))],
        ),
        _write_gguf(
            tmp_path / "untied_mib.gguf",
            [
                ("token_embd.weight", rows),
                ("output.weight", rows),
                ("blk.0.attn_q.weight", (8, 8)),
            ],
        ),
    )


def test_the_host_shortfall_prices_the_tied_duplicate(backend, mib_embd_pair):
    """The duplicate is the one weight that cannot page back to disk.

    The rest of an oversized load survives because it is mmap'd, which is also
    what the pageable-load override is protecting. The duplicate is built from
    ``token_embd`` rather than read from the file, so it is an anonymous
    allocation: a host within one embedding matrix of the requirement is
    OOM-killed rather than run slowly, and must be warned.

    Priced at the boundary, so the assertion is the term and not a margin: 20 GiB
    of weights against 4 GiB of VRAM spills exactly 16 GiB, and available RAM is
    set to exactly that plus the reserved headroom.
    """
    from core.inference.llama_cpp import _HOST_RAM_HEADROOM_MIB

    tied, untied = mib_embd_pair
    assert backend._tied_output_bytes(str(tied)) == 1024 * 1024
    assert backend._tied_output_bytes(str(untied)) == 0

    instance = object.__new__(backend)
    instance._get_gguf_size_bytes = lambda _path: 20 * 1024**3
    avail_mib = 16 * 1024 + _HOST_RAM_HEADROOM_MIB

    def priced(path: Path):
        return instance._launch_host_shortfall_message(
            ["llama-server", "-m", str(path)],
            [(0, 4 * 1024)],
            avail_mib = avail_mib,
        )

    assert priced(tied) is not None, "the tied duplicate is missing from the host floor"
    assert priced(untied) is None, "a model shipping its own output must not be charged twice"
