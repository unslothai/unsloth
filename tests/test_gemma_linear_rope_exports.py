# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gemma / Gemma 2 trained with the loader's linear RoPE must export with it: the merged config.json
on an unsloth_zoo without the RoPE carry-over, and the GGUF, whose llama.cpp converters write no
RoPE scaling for these two architectures."""

from __future__ import annotations

import ast
import contextlib
import gc
import json
import logging
import os
import types
from pathlib import Path

import pytest
import torch

_SAVE_PY = Path(__file__).resolve().parents[1] / "unsloth" / "save.py"
_TREE = ast.parse(_SAVE_PY.read_text(encoding = "utf-8"))


def _load(*names):
    nodes = [n for n in _TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), f"missing from save.py: {set(names)}"
    namespace = {
        "os": os,
        "json": json,
        "gc": gc,
        "logger": logging.getLogger("unsloth-test"),
        # save.py swaps in llama.cpp's gguf-py here; the installed gguf is the same format.
        "use_local_gguf": contextlib.nullcontext,
    }
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(_SAVE_PY), "exec"), namespace)
    return namespace


def test_the_repaired_merged_config_reloads_scaled_in_stock_transformers(tmp_path):
    """What the fallback writes must be what the installed transformers reads as linear RoPE."""
    from transformers import AutoConfig, GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

    write = _load("_write_trained_gemma_rope")["_write_trained_gemma_rope"]
    GemmaConfig(head_dim = 16, max_position_embeddings = 8192).save_pretrained(tmp_path)
    trained = GemmaConfig(head_dim = 16, max_position_embeddings = 16384)
    trained.rope_scaling = {"type": "linear", "factor": 2.0, "rope_theta": 10000.0}
    write(types.SimpleNamespace(config = trained), str(tmp_path), 2.0)

    reloaded = AutoConfig.from_pretrained(tmp_path)
    assert reloaded.max_position_embeddings == 16384
    unscaled = GemmaRotaryEmbedding(GemmaConfig(head_dim = 16)).inv_freq
    rotary = GemmaRotaryEmbedding(reloaded)
    torch.testing.assert_close(rotary.inv_freq, unscaled / 2.0)
    assert rotary.attention_scaling == 1.0


def _tiny_gguf(
    path,
    arch,
    quantized = False,
):
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_context_length(16384)
    writer.add_block_count(1)
    writer.add_name("tiny")
    writer.add_token_list(["<pad>", "<eos>", "a"])
    writer.add_tensor("token_embd.weight", np.arange(64, dtype = np.float32).reshape(2, 32))
    weight = np.linspace(-1, 1, 64, dtype = np.float32).reshape(2, 32)
    if quantized:
        kind = gguf.GGMLQuantizationType.Q8_0
        writer.add_tensor("blk.0.attn_q.weight", gguf.quants.quantize(weight, kind), raw_dtype = kind)
    else:
        writer.add_tensor("blk.0.attn_q.weight", weight.astype(np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _read(path):
    import gguf

    reader = gguf.GGUFReader(str(path))
    fields = {
        name: field.contents()
        for name, field in reader.fields.items()
        if not name.startswith("GGUF.")
    }
    tensors = {t.name: (int(t.tensor_type), bytes(t.data.tobytes())) for t in reader.tensors}
    return fields, tensors


def _config(directory, model_type, rope):
    directory.mkdir(exist_ok = True)
    config = {"model_type": model_type, "max_position_embeddings": 16384, **rope}
    (directory / "config.json").write_text(json.dumps(config))
    return str(directory)


_LINEAR = {
    "rope_scaling": {"rope_scaling": {"type": "linear", "factor": 2.0}},
    "rope_parameters": {
        "rope_parameters": {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}
    },
}


@pytest.mark.parametrize("quantized", [False, True], ids = ["f16", "q8_0"])
@pytest.mark.parametrize("layout", sorted(_LINEAR))
@pytest.mark.parametrize("arch", ["gemma", "gemma2"])
def test_a_scaled_gemma_gguf_records_linear_rope_for_llama_cpp(tmp_path, arch, layout, quantized):
    pytest.importorskip("gguf")
    record = _load("_record_gemma_linear_rope_in_gguf")["_record_gemma_linear_rope_in_gguf"]
    path = tmp_path / f"{arch}.gguf"
    _tiny_gguf(path, arch, quantized = quantized)
    before_fields, before_tensors = _read(path)
    assert f"{arch}.rope.scaling.type" not in before_fields

    record(_config(tmp_path / "model", arch, _LINEAR[layout]), [str(path)])

    fields, tensors = _read(path)
    # llama.cpp reads these for every architecture: freq_scale_train = 1 / factor.
    assert fields.pop(f"{arch}.rope.scaling.type") == "linear"
    assert fields.pop(f"{arch}.rope.scaling.factor") == pytest.approx(2.0)
    assert fields == before_fields
    assert tensors == before_tensors
    assert sorted(os.listdir(tmp_path)) == sorted(["model", f"{arch}.gguf"])


@pytest.mark.parametrize(
    "model_type, rope",
    [
        ("gemma", {}),
        ("gemma2", {"rope_scaling": None}),
        ("gemma", {"rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}}),
        ("gemma", {"rope_scaling": {"type": "linear", "factor": 1.0}}),
        # The other converters write rope.scaling.* themselves.
        ("llama", {"rope_scaling": {"type": "linear", "factor": 2.0}}),
    ],
    ids = ["absent", "none", "default", "factor_one", "not_gemma"],
)
def test_an_unscaled_or_non_gemma_gguf_is_left_byte_identical(tmp_path, model_type, rope):
    pytest.importorskip("gguf")
    record = _load("_record_gemma_linear_rope_in_gguf")["_record_gemma_linear_rope_in_gguf"]
    path = tmp_path / "model.gguf"
    _tiny_gguf(path, "gemma" if model_type != "llama" else "llama")
    before = path.read_bytes()
    record(_config(tmp_path / "model", model_type, rope), [str(path)])
    assert path.read_bytes() == before


def test_a_gguf_the_scaling_cannot_be_added_to_fails_unless_opted_out(tmp_path, monkeypatch):
    """Never hand back a GGUF that runs unscaled without saying so."""
    record = _load("_record_gemma_linear_rope_in_gguf")["_record_gemma_linear_rope_in_gguf"]
    path = tmp_path / "broken.gguf"
    path.write_bytes(b"not a gguf")
    model = _config(tmp_path / "model", "gemma", _LINEAR["rope_scaling"])
    with pytest.raises(RuntimeError, match = "UNSLOTH_GGUF_ALLOW_UNSCALED_ROPE"):
        record(model, [str(path)])
    monkeypatch.setenv("UNSLOTH_GGUF_ALLOW_UNSCALED_ROPE", "1")
    record(model, [str(path)])
    assert path.read_bytes() == b"not a gguf"
    assert sorted(os.listdir(tmp_path)) == ["broken.gguf", "model"]


def test_save_to_gguf_records_the_scaling_before_quantizing():
    """llama-quantize copies every key, so the converter's output is the one place to add it."""
    save_to_gguf = next(
        n for n in _TREE.body if isinstance(n, ast.FunctionDef) and n.name == "save_to_gguf"
    )
    lines = {}
    for node in ast.walk(save_to_gguf):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            lines.setdefault(node.func.id, []).append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_quantize_one":
            lines.setdefault("def _quantize_one", []).append(node)
    (record,) = lines.get("_record_gemma_linear_rope_in_gguf", [None])
    assert record is not None, "save_to_gguf no longer records Gemma's linear RoPE"
    assert [ast.unparse(a) for a in record.args] == [
        "model_directory",
        "initial_files",
        "gemma_linear_rope",
    ]
    (convert,) = lines["_convert_into_gguf_directory"]
    (quantize,) = lines["def _quantize_one"]
    assert convert.lineno < record.lineno < quantize.lineno


@pytest.mark.parametrize("arch", ["gemma", "gemma2"])
def test_the_live_factor_wins_over_an_unscaled_staged_config(tmp_path, arch):
    """A reused checkpoint stages the base's config.json, unscaled; the live rotary is what ran."""
    pytest.importorskip("gguf")
    record = _load("_record_gemma_linear_rope_in_gguf")["_record_gemma_linear_rope_in_gguf"]
    path = tmp_path / f"{arch}.gguf"
    _tiny_gguf(path, arch)
    before_fields, before_tensors = _read(path)
    unscaled = _config(tmp_path / "model", arch, {"rope_scaling": None})

    record(unscaled, [str(path)], 4.0)

    fields, tensors = _read(path)
    assert fields.pop(f"{arch}.rope.scaling.type") == "linear"
    assert fields.pop(f"{arch}.rope.scaling.factor") == pytest.approx(4.0)
    assert fields == before_fields
    assert tensors == before_tensors

    # And a live model that applied none leaves a scaled-looking config's GGUF alone.
    other = tmp_path / "other.gguf"
    _tiny_gguf(other, arch)
    raw = other.read_bytes()
    record(_config(tmp_path / "scaled", arch, _LINEAR["rope_scaling"]), [str(other)], 1.0)
    assert other.read_bytes() == raw


@pytest.mark.parametrize("tensor_mib", [1, 68], ids = ["one_chunk", "two_chunks"])
def test_recording_the_scaling_never_holds_a_second_copy_of_the_weights(
    tmp_path, monkeypatch, tensor_mib
):
    """The preflight budgets one intermediate GGUF, so the edit may not stage a full copy of it."""
    gguf = pytest.importorskip("gguf")
    import numpy as np

    record = _load("_record_gemma_linear_rope_in_gguf")["_record_gemma_linear_rope_in_gguf"]
    path = tmp_path / "out" / "gemma.gguf"
    path.parent.mkdir()
    writer = gguf.GGUFWriter(str(path), "gemma")
    writer.add_context_length(16384)
    writer.add_tensor("token_embd.weight", np.arange(64, dtype = np.float32).reshape(2, 32))
    if tensor_mib:
        count = tensor_mib * 1024 * 1024 // 4
        writer.add_tensor("blk.0.ffn_up.weight", np.arange(count, dtype = np.float32) % 251)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size = path.stat().st_size
    _, before_tensors = _read(path)

    peak = [0]
    real_replace, real_collect = os.replace, gc.collect

    def _measure():
        total = sum(p.stat().st_size for p in path.parent.iterdir())
        peak[0] = max(peak[0], total)

    def replace(src, dst):
        _measure()
        return real_replace(src, dst)

    def collect(*args):
        _measure()
        return real_collect(*args)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(gc, "collect", collect)
    record(_config(tmp_path / "model", "gemma", _LINEAR["rope_scaling"]), [str(path)])
    _measure()

    assert peak[0] <= size + 4096, f"{peak[0] - size} bytes beyond the one GGUF"
    fields, tensors = _read(path)
    assert fields["gemma.rope.scaling.type"] == "linear"
    assert tensors == before_tensors
    assert sorted(os.listdir(path.parent)) == ["gemma.gguf"]


@pytest.mark.parametrize("arch", ["gemma", "gemma2"])
def test_a_split_gguf_records_the_scaling_in_the_first_shard_only(tmp_path, arch):
    """Past the converter's shard limit (Gemma 2 27B at f16), only the first shard carries
    general.architecture and the hparams; llama.cpp and llama-quantize read them from there."""
    gguf = pytest.importorskip("gguf")
    import numpy as np

    record = _load("_record_gemma_linear_rope_in_gguf")["_record_gemma_linear_rope_in_gguf"]
    writer = gguf.GGUFWriter(str(tmp_path / f"{arch}.gguf"), arch, split_max_tensors = 1)
    writer.add_context_length(16384)
    writer.add_block_count(1)
    writer.add_tensor("token_embd.weight", np.arange(64, dtype = np.float32).reshape(2, 32))
    writer.add_tensor("blk.0.attn_q.weight", np.ones((2, 32), dtype = np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    first, second = sorted(str(p) for p in tmp_path.glob(f"{arch}-*-of-*.gguf"))
    before_first, before_second = _read(first), Path(second).read_bytes()
    assert gguf.Keys.General.ARCHITECTURE not in _read(second)[0]

    record(_config(tmp_path / "model", arch, _LINEAR["rope_scaling"]), [first, second])

    fields, tensors = _read(first)
    assert fields.pop(f"{arch}.rope.scaling.type") == "linear"
    assert fields.pop(f"{arch}.rope.scaling.factor") == pytest.approx(2.0)
    assert (fields, tensors) == before_first
    assert Path(second).read_bytes() == before_second


def test_shards_without_any_model_metadata_fail_unless_opted_out(tmp_path, monkeypatch):
    gguf = pytest.importorskip("gguf")
    import numpy as np

    record = _load("_record_gemma_linear_rope_in_gguf")["_record_gemma_linear_rope_in_gguf"]
    writer = gguf.GGUFWriter(str(tmp_path / "gemma.gguf"), "gemma", split_max_tensors = 1)
    writer.add_tensor("token_embd.weight", np.zeros((2, 32), dtype = np.float32))
    writer.add_tensor("blk.0.attn_q.weight", np.zeros((2, 32), dtype = np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    second = str(sorted(tmp_path.glob("gemma-*-of-*.gguf"))[1])
    raw = Path(second).read_bytes()
    model = _config(tmp_path / "model", "gemma", _LINEAR["rope_scaling"])
    with pytest.raises(RuntimeError, match = "general.architecture"):
        record(model, [second])
    monkeypatch.setenv("UNSLOTH_GGUF_ALLOW_UNSCALED_ROPE", "1")
    record(model, [second])
    assert Path(second).read_bytes() == raw
