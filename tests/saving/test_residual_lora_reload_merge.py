# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A merged save after reloading a PiSSA / OLoRA adapter must keep the trained weights (CPU, no `import unsloth`).

Runs the loader's own adapter load blocks, then the merge the way `unsloth_generic_save` runs it."""

import ast
import contextlib
import copy
import functools
import pathlib
import threading
import types

import pytest
import torch

pytest.importorskip("peft")
saving_utils = pytest.importorskip("unsloth_zoo.saving_utils")

from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

_MODELS = pathlib.Path(__file__).resolve().parents[2] / "unsloth" / "models"
_KEY = "model.layers.0.self_attn.q_proj.weight"


def _utils():
    names = {
        "RESIDUAL_INIT_LORA_WEIGHTS",
        "CONVERT_ON_SAVE_LORA_WEIGHTS",
        "_RESIDUAL_LOAD",
        "_RESIDUAL_LOAD_HOOK",
    }
    functions = {
        "_lora_factors",
        "snapshot_residual_lora_init",
        "snapshot_residual_lora_load",
        "lora_relative_to_original_base",
    }
    tree = ast.parse((_MODELS / "_utils.py").read_text(encoding = "utf-8"))
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names)
    ]
    namespace = {
        "torch": torch,
        "contextlib": contextlib,
        "functools": functools,
        "threading": threading,
    }
    exec(compile(ast.Module(body, []), "_utils.py", "exec"), namespace)
    return namespace


def _load_blocks():
    tree = ast.parse((_MODELS / "loader.py").read_text(encoding = "utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With) and "PeftModel.from_pretrained(" in ast.unparse(node)
    ]


def _effective(model):
    return copy.deepcopy(model).merge_and_unload().state_dict()[_KEY]


def test_both_loader_paths_are_covered():
    assert len(_load_blocks()) == 2


@pytest.mark.parametrize("init", [True, "pissa", "olora"])
@pytest.mark.parametrize("path", [0, 1])
def test_merged_save_after_reload_keeps_trained_weights(tmp_path, monkeypatch, path, init):
    # CPU merge math on every host: the check is the arithmetic, not an accelerator copy.
    monkeypatch.setattr(saving_utils, "_active_merge_device", lambda: "cpu", raising = False)
    monkeypatch.setattr(
        saving_utils, "DEVICE_TYPE", "cpu", raising = False
    )  # a CUDA torch with no visible GPU
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 4,
    )
    base_dir, adapter_dir, merged_dir = (str(tmp_path / n) for n in ("base", "adapter", "merged"))
    LlamaForCausalLM(config).save_pretrained(base_dir)

    lora = LoraConfig(r = 4, lora_alpha = 4, target_modules = ["q_proj"], init_lora_weights = init)
    trained = get_peft_model(LlamaForCausalLM.from_pretrained(base_dir), lora)
    with torch.no_grad():
        for name, param in trained.named_parameters():
            if "lora_" in name:
                param.add_(0.02 * torch.randn_like(param))
    want = _effective(trained)
    trained.save_pretrained(adapter_dir)

    utils = _utils()
    namespace = {
        **utils,
        "PeftModel": PeftModel,
        "model": LlamaForCausalLM.from_pretrained(base_dir),
        "old_model_name": adapter_dir,
        "load_config": PeftConfig.from_pretrained(adapter_dir),
        "peft_load_kwargs": {},
        "token": None,
        "revision": None,
        "local_files_only": True,
        "trust_remote_code": False,
        "fast_pissa": False,
        "fast_lora_init": None,
        "block_swap_shaped": lambda *args, **kwargs: contextlib.nullcontext(),
    }
    exec(compile(ast.Module([_load_blocks()[path]], []), "loader.py", "exec"), namespace)
    reloaded = namespace["model"]
    torch.testing.assert_close(_effective(reloaded), want)

    with utils["lora_relative_to_original_base"](reloaded):
        saving_utils.merge_and_overwrite_lora(
            lambda name, load_in_4bit = False: name,
            model = reloaded,
            save_directory = merged_dir,
            save_method = "merged_16bit",
        )
    merged = load_file(str(tmp_path / "merged" / "model.safetensors"))[_KEY]
    torch.testing.assert_close(merged, want, atol = 1e-5, rtol = 1e-5)
    # The live model is left as it was loaded.
    torch.testing.assert_close(_effective(reloaded), want)


def test_load_hook_is_thread_scoped_and_restored(monkeypatch):
    utils = _utils()
    calls = []
    monkeypatch.setitem(
        utils, "snapshot_residual_lora_init", lambda model, init: calls.append((model, init))
    )
    stub = lambda model, *args, **kwargs: None
    monkeypatch.setattr(PeftModel, "load_adapter", stub)
    config = types.SimpleNamespace(init_lora_weights = "olora")
    entered, release = threading.Event(), threading.Event()

    def other_load():
        with utils["snapshot_residual_lora_load"](config):
            entered.set()
            release.wait(5)
            PeftModel.load_adapter("other")

    worker = threading.Thread(target = other_load)
    with utils["snapshot_residual_lora_load"](config):
        worker.start()
        entered.wait(5)
        unrelated = threading.Thread(target = lambda: PeftModel.load_adapter("unrelated"))
        unrelated.start()
        unrelated.join(5)
        PeftModel.load_adapter("first")
        PeftModel.load_adapter("second")
    # Exits interleave: this load leaves first, the other one still needs the hook.
    release.set()
    worker.join(5)
    assert PeftModel.load_adapter is stub
    assert calls == [("first", "olora"), ("other", "olora")]
