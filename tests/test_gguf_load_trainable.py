# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A `gguf_file` load must give a trainable model.

From transformers 5.17 a GGUF load goes through `GgufHfQuantizer`. By default it leaves the quantizer on the
model with `is_trainable = False` (and keeps Qwen3.5 / LFM2 weights packed where its ggml kernel resolves),
so Trainer refuses to fine-tune. The loaders pass `GgufConfig(dequantize = True)` to get the dense model a
GGUF load gave before 5.17.
"""

import ast
import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from unsloth.models._utils import gguf_dequantize_kwargs, gguf_drop_bitsandbytes

try:
    from transformers import GgufConfig
except ImportError:
    GgufConfig = None

needs_gguf_config = pytest.mark.skipif(
    GgufConfig is None or "dequantize" not in inspect.signature(GgufConfig.__init__).parameters,
    reason = "transformers < 5.17 has no GGUF quantizer",
)
MODELS = Path(__file__).resolve().parents[1] / "unsloth" / "models"


def test_no_gguf_file_is_untouched():
    kwargs = {"revision": "main"}
    assert gguf_dequantize_kwargs(kwargs) == {"revision": "main"}
    assert gguf_dequantize_kwargs({"gguf_file": None}) == {"gguf_file": None}


@needs_gguf_config
def test_gguf_file_gets_a_dequantizing_config():
    kwargs = gguf_dequantize_kwargs({"gguf_file": "model-Q4_K_M.gguf"})
    config = kwargs["quantization_config"]
    assert isinstance(config, GgufConfig)
    assert config.dequantize is True


@needs_gguf_config
def test_a_caller_quantization_config_wins():
    mine = GgufConfig(dequantize = False)
    kwargs = gguf_dequantize_kwargs({"gguf_file": "model.gguf", "quantization_config": mine})
    assert kwargs["quantization_config"] is mine
    assert mine.dequantize is False


def test_old_transformers_is_a_no_op(monkeypatch):
    import transformers

    monkeypatch.delattr(transformers, "GgufConfig", raising = False)
    real_import = __import__

    def fake_import(
        name,
        globals = None,
        locals = None,
        fromlist = (),
        level = 0,
    ):
        if name == "transformers" and fromlist and "GgufConfig" in fromlist:
            raise ImportError("cannot import name 'GgufConfig'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert gguf_dequantize_kwargs({"gguf_file": "model.gguf"}) == {"gguf_file": "model.gguf"}


@pytest.mark.parametrize(
    "filename, dtype_call",
    [
        ("llama.py", "add_dtype_kwargs(dtype, kwargs)"),
        ("vision.py", "add_dtype_kwargs(torch_dtype, kwargs)"),
    ],
)
def test_both_loaders_apply_it_before_from_pretrained(filename, dtype_call):
    source = (MODELS / filename).read_text(encoding = "utf-8")
    tree = ast.parse(source)
    lines = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            lines.setdefault(ast.unparse(node.value), []).append(node.lineno)
    assert lines.get(dtype_call), f"{filename}: {dtype_call} moved"
    assert lines.get(
        "gguf_dequantize_kwargs(kwargs)"
    ), f"{filename} no longer calls gguf_dequantize_kwargs"
    assert min(lines["gguf_dequantize_kwargs(kwargs)"]) > min(lines[dtype_call])


def test_gguf_drops_bitsandbytes_with_a_notice(capsys):
    assert gguf_drop_bitsandbytes("model.gguf", True, False) == (False, False)
    assert "ignored" in capsys.readouterr().out
    assert gguf_drop_bitsandbytes("model.gguf", False, True) == (False, False)
    assert gguf_drop_bitsandbytes("model.gguf", False, False) == (False, False)
    assert capsys.readouterr().out.count("ignored") == 1
    # Not a GGUF load: untouched, silent.
    assert gguf_drop_bitsandbytes(None, True, False) == (True, False)
    assert gguf_drop_bitsandbytes("", False, True) == (False, True)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("class_name", ["FastLanguageModel", "FastModel"])
def test_both_entry_points_drop_bitsandbytes_for_gguf(class_name):
    tree = ast.parse((MODELS / "loader.py").read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "gguf_drop_bitsandbytes"
    ]
    assert len(calls) == 1, f"{class_name}.from_pretrained must call gguf_drop_bitsandbytes once"
    assert ast.unparse(calls[0].targets[0]) == "(load_in_4bit, load_in_8bit)"
    # Before anything reads load_in_4bit to pick a checkpoint or build a bitsandbytes config.
    first_read = min(
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Name)
        and n.id == "load_in_4bit"
        and isinstance(n.ctx, ast.Load)
        and n.lineno != calls[0].lineno
        and not (calls[0].lineno <= n.lineno <= calls[0].end_lineno)
    )
    assert calls[0].lineno < first_read


def _write_tiny_llama_gguf(path, gguf):
    import numpy as np

    hidden, inter, vocab, layers, heads = 16, 32, 32, 1, 2
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_context_length(64)
    writer.add_embedding_length(hidden)
    writer.add_block_count(layers)
    writer.add_feed_forward_length(inter)
    writer.add_head_count(heads)
    writer.add_head_count_kv(heads)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_dimension_count(hidden // heads)
    writer.add_vocab_size(vocab)
    writer.add_tokenizer_model("llama")
    writer.add_token_list([f"t{i}" for i in range(vocab)])
    writer.add_token_scores([0.0] * vocab)
    writer.add_token_types([1] * vocab)
    rng = np.random.default_rng(0)

    def add(name, *shape):
        writer.add_tensor(name, (rng.standard_normal(shape) * 0.02).astype(np.float32))

    add("token_embd.weight", vocab, hidden)
    add("output.weight", vocab, hidden)
    writer.add_tensor("output_norm.weight", np.ones(hidden, dtype = np.float32))
    for i in range(layers):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(hidden, dtype = np.float32))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(hidden, dtype = np.float32))
        for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
            add(f"blk.{i}.{name}.weight", hidden, hidden)
        add(f"blk.{i}.ffn_gate.weight", inter, hidden)
        add(f"blk.{i}.ffn_up.weight", inter, hidden)
        add(f"blk.{i}.ffn_down.weight", hidden, inter)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@needs_gguf_config
@pytest.mark.parametrize("dequantize", [False, True])
def test_gguf_model_passes_the_trainer_guard(tmp_path, dequantize):
    gguf = pytest.importorskip("gguf")
    from transformers import AutoModelForCausalLM
    from transformers.trainer_utils import validate_quantization_for_training

    _write_tiny_llama_gguf(tmp_path / "tiny.gguf", gguf)
    kwargs = {"gguf_file": "tiny.gguf"}
    if dequantize:
        kwargs = gguf_dequantize_kwargs(kwargs)
    model = AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype = torch.float32, **kwargs)
    if not dequantize:
        # The default load, which is what the loaders passed before: the quantizer stays and Trainer refuses.
        assert getattr(model, "hf_quantizer", None) is not None
        with pytest.raises(ValueError, match = "purely quantized|do not support training"):
            validate_quantization_for_training(model)
        return
    assert getattr(model, "hf_quantizer", None) is None
    assert not model.is_quantized
    validate_quantization_for_training(model)
    assert all(p.is_floating_point() for p in model.parameters())
    ids = torch.randint(0, 32, (1, 8))
    model.train()
    model(input_ids = ids, labels = ids).loss.backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
