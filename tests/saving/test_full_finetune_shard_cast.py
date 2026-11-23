# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A full fine-tune saved as 16-bit (or exported as float32) must cast one output shard at a
time, not hold a second, cast copy of the whole state dict: that copy sits on the GPU for an
fp32 master saved as 16-bit and on the host for a float32 export. The files written must be
the ones the eager cast wrote. Sliced with `ast`: importing `unsloth.save` needs a GPU."""

import ast
import contextlib
import json
import os
import re
import types
import typing
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors")

SAVE_PY = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"
_SOURCE = SAVE_PY.read_text(encoding = "utf-8")
_WANTED = {
    "_normalize_safe_serialization",
    "_is_adapter_save_method",
    "_honours_safe_serialization",
    "_refuse_unsaveable_text_core",
    "_cast_state_dict_16bit",
    "_requests_float32",
    "_stamp_saved_config_float32",
    "_collect_garbage",
    "unsloth_generic_save",
    "_planned_save_dtypes",
    "_restamp_index_total_size",
    "_CastShardsOnWrite",
    "_cast_hook_stack",
    "_dispatch_cast_save",
}
_WANTED_ASSIGNS = {"_CAST_HOOK_LOCK", "_CAST_HOOK_STATE", "_CAST_HOOK_LOCAL"}


def _generic_save():
    namespace = {n: getattr(typing, n) for n in typing.__all__}
    namespace.update(
        os = os,
        json = json,
        torch = torch,
        re = re,
        threading = __import__("threading"),
        nullcontext = contextlib.nullcontext,
        contextmanager = contextlib.contextmanager,
        PeftModel = type("PeftModel", (), {}),
        PreTrainedTokenizerBase = type("Tokenizer", (), {}),
        ProcessorMixin = type("Processor", (), {}),
        patch_saving_functions = lambda t: t,
        get_token = lambda: None,
        _defer_to_main_process = lambda is_main_process: not is_main_process,
        get_model_name = lambda n: n,
        _prewarm_base_model_hub_cache = lambda *a, **k: None,
        _is_qwen3_5_vlm = lambda m: False,
        _refuse_mtp_variant = lambda m, v: None,
        _attached_mtp_head = lambda m: None,
        _internal_mtp_save = contextlib.nullcontext,
        _stitch_mtp_head = lambda m, d: None,
        _mtp_head_tensor_names = lambda m: [],
        _release_layout_for_merged_save = lambda m, d, s: False,
        _push_lfm3_audio_release_config = lambda *a, **k: False,
        logger = types.SimpleNamespace(warning_once = lambda *a, **k: None),
    )
    for node in ast.parse(_SOURCE).body:
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _WANTED) or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in _WANTED_ASSIGNS for t in node.targets)
        ):
            exec(compile(ast.get_source_segment(_SOURCE, node), str(SAVE_PY), "exec"), namespace)
    return torch.inference_mode()(namespace["unsloth_generic_save"]), namespace


def _model(dtype, tied):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size = 512,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        tie_word_embeddings = tied,
    )
    return LlamaForCausalLM(config).to(dtype)


def _read(directory):
    from safetensors import safe_open

    out = {}
    for name in sorted(os.listdir(directory)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(directory, name), "pt") as f:
                for key in f.keys():
                    out[key] = f.get_tensor(key)
    return out


def _expected(namespace, model, export_dtype):
    if export_dtype == "float32":
        return {k: v.float() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    target = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return namespace["_cast_state_dict_16bit"](model, model.state_dict(), target)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize(
    "model_dtype, export_dtype", [(torch.float32, None), (torch.bfloat16, "float32")]
)
def test_cast_per_shard_writes_what_the_eager_cast_wrote(tmp_path, model_dtype, export_dtype, tied):
    generic_save, namespace = _generic_save()
    model = _model(model_dtype, tied)
    expected = _expected(namespace, model, export_dtype)
    handed = {}
    real_save = model.save_pretrained

    def save_pretrained(directory, **kwargs):
        handed.update(kwargs["state_dict"])
        return real_save(directory, **kwargs)

    model.save_pretrained = save_pretrained
    generic_save(
        model,
        None,
        save_directory = str(tmp_path),
        save_method = "merged_16bit",
        max_shard_size = "100KB",
        dtype = export_dtype,
    )
    # The dictionary transformers plans shards from still holds the model's own storage.
    params = {p.data_ptr() for p in model.parameters()}
    uncast = [k for k, v in handed.items() if v.data_ptr() in params and v.dtype == model_dtype]
    assert len(uncast) >= len(handed) - 1, "a whole cast copy was built before the write"

    written = _read(tmp_path)
    if not tied or expected["lm_head.weight"].dtype == model_dtype:
        assert set(written) <= set(expected)
    else:
        # Like the eager cast: once the dtype changes, each alias is its own tensor on disk.
        assert set(written) == set(expected)
    for key, value in written.items():
        assert value.dtype == expected[key].dtype, key
        assert torch.equal(value, expected[key]), key
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(written)
    assert index["metadata"]["total_size"] == sum(
        v.numel() * v.element_size() for v in written.values()
    )


def test_a_transformers_that_writes_elsewhere_falls_back_to_the_eager_cast(tmp_path, monkeypatch):
    """If save_pretrained never reaches safe_save_file, the uncast write is redone eagerly."""
    generic_save, namespace = _generic_save()
    model = _model(torch.float32, False)
    expected = _expected(namespace, model, None)
    import transformers.modeling_utils as modeling_utils
    from safetensors.torch import save_file

    monkeypatch.setattr(modeling_utils, "safe_save_file", None, raising = False)
    real_save = model.save_pretrained

    def save_pretrained(directory, **kwargs):
        # Stands in for a writer that bypasses the module-level function.
        state_dict = kwargs["state_dict"]
        save_file(
            {k: v.contiguous() for k, v in state_dict.items()},
            os.path.join(directory, "model.safetensors"),
        )

    model.save_pretrained = save_pretrained
    generic_save(model, None, save_directory = str(tmp_path), save_method = "merged_16bit")
    written = _read(tmp_path)
    for key, value in written.items():
        assert value.dtype == expected[key].dtype and torch.equal(value, expected[key]), key


def test_the_eager_cast_is_still_one_env_var_away(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_SAVE_EAGER_CAST", "1")
    generic_save, namespace = _generic_save()
    model = _model(torch.float32, False)
    handed = {}
    real_save = model.save_pretrained

    def save_pretrained(directory, **kwargs):
        handed.update(kwargs["state_dict"])
        return real_save(directory, **kwargs)

    model.save_pretrained = save_pretrained
    generic_save(model, None, save_directory = str(tmp_path), save_method = "merged_16bit")
    assert all(v.dtype != torch.float32 for v in handed.values())


def test_mixed_cast_ratios_never_plan_past_max_shard_size(monkeypatch):
    # Retained fp32 tensors next to downcast ones: the limit is scaled by the largest per-tensor ratio (1.0 here),
    # so no written shard can pass max_shard_size; a uniform downcast still doubles it (the eager layout).
    _, ns = _generic_save()
    tensors = {f"keep{i}": torch.zeros(5 * 2**20 // 4 * 4, dtype = torch.float32) for i in range(4)}
    tensors.update({f"cast{i}": torch.zeros(10 * 2**20, dtype = torch.float32) for i in range(10)})
    mixed = {k: (torch.float32 if k.startswith("keep") else torch.bfloat16) for k in tensors}
    monkeypatch.setitem(
        ns, "_planned_save_dtypes", lambda model, t, dtype: {k: mixed[k] for k in t}
    )
    assert (
        ns["_CastShardsOnWrite"](None, tensors, torch.bfloat16, 50 * 2**20).max_shard_size
        == 50 * 2**20
    )
    monkeypatch.setitem(
        ns, "_planned_save_dtypes", lambda model, t, dtype: {k: torch.bfloat16 for k in t}
    )
    assert (
        ns["_CastShardsOnWrite"](None, tensors, torch.bfloat16, 50 * 2**20).max_shard_size
        == 100 * 2**20
    )


def test_overlapping_saves_each_write_through_their_own_caster(monkeypatch):
    import threading
    import transformers.modeling_utils as modeling_utils

    _, ns = _generic_save()
    written = []

    def original(tensors, filename, *args, **kwargs):
        written.append((filename, {k: v.dtype for k, v in tensors.items()}))

    monkeypatch.setattr(modeling_utils, "safe_save_file", original)
    plan = {"a": torch.bfloat16, "b": torch.float16}
    monkeypatch.setitem(
        ns, "_planned_save_dtypes", lambda model, t, dtype: {k: plan[dtype] for k in t}
    )
    t = {"w": torch.zeros(4, dtype = torch.float32)}
    first = ns["_CastShardsOnWrite"](None, t, "a", "1GB")
    second = ns["_CastShardsOnWrite"](None, t, "b", "1GB")
    ready, done = threading.Event(), threading.Event()

    def other_thread():
        with second:
            ready.set()
            done.wait(5)
            modeling_utils.safe_save_file(dict(t), "second.safetensors")

    worker = threading.Thread(target = other_thread)
    first.__enter__()
    worker.start()
    ready.wait(5)
    modeling_utils.safe_save_file(dict(t), "first.safetensors")
    first.__exit__(None, None, None)  # the first save closes while the second is still open
    assert modeling_utils.safe_save_file is not original
    done.set()
    worker.join(5)
    assert modeling_utils.safe_save_file is original  # restored once the last save closes
    assert dict(written) == {
        "first.safetensors": {"w": torch.bfloat16},
        "second.safetensors": {"w": torch.float16},
    }
    assert first.engaged == 1 and second.engaged == 1
