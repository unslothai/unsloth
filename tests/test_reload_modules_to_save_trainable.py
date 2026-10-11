# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A reloaded adapter keeps training its modules_to_save after kbit prep froze them (#3108)."""

import ast
import pathlib

import pytest
import torch

peft = pytest.importorskip("peft")
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils.other import ModulesToSaveWrapper


def _load_helper():
    """Exec the helper from source: importing llama.py would pull hardware probes."""
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "unsloth" / "models" / "llama.py").read_text(encoding = "utf-8")
    fn = next(
        node
        for node in ast.parse(src).body
        if isinstance(node, ast.FunctionDef) and node.name == "_unfreeze_modules_to_save"
    )
    ns = {"torch": torch, "ModulesToSaveWrapper": ModulesToSaveWrapper}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body = [fn], type_ignores = [])), "<llama>", "exec"
        ),
        ns,
    )
    return ns["_unfreeze_modules_to_save"]


unfreeze = _load_helper()


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.q_proj = torch.nn.Linear(8, 8)
        self.lm_head = torch.nn.Linear(8, 32, bias = False)

    def forward(self, x):
        return self.lm_head(self.q_proj(self.embed_tokens(x)))


def _freeze_like_kbit_prep(model):
    # prepare_model_for_training(full_finetuning=False): only LoRA weights keep requires_grad.
    for name, param in model.named_parameters():
        param.requires_grad_(".lora_A." in name or ".lora_B." in name)


def _trainable(model):
    return sorted(n for n, p in model.named_parameters() if p.requires_grad)


def _peft(dtype = torch.float32):
    torch.manual_seed(0)
    config = LoraConfig(r = 4, target_modules = ["q_proj"], modules_to_save = ["embed_tokens", "lm_head"])
    return get_peft_model(_Tiny().to(dtype), config)


def test_reloaded_adapter_trains_modules_to_save(tmp_path):
    fresh = _peft()
    expected = _trainable(fresh)
    assert any("modules_to_save.default" in n for n in expected)
    fresh.save_pretrained(tmp_path)

    torch.manual_seed(0)
    reloaded = PeftModel.from_pretrained(_Tiny(), tmp_path, is_trainable = True, torch_device = "cpu")
    _freeze_like_kbit_prep(reloaded)
    assert _trainable(reloaded) != expected  # the #3108 state before the fix

    unfreeze(reloaded)
    assert _trainable(reloaded) == expected
    assert not any("original_module" in n for n in _trainable(reloaded))


def test_float16_copy_is_upcast_and_original_untouched():
    model = _peft(torch.float16)
    _freeze_like_kbit_prep(model)
    unfreeze(model)
    for module in model.modules():
        if isinstance(module, ModulesToSaveWrapper):
            assert module.modules_to_save["default"].weight.dtype == torch.float32
            assert module.modules_to_save["default"].weight.requires_grad
            assert module.original_module.weight.dtype == torch.float16
            assert not module.original_module.weight.requires_grad


def test_float16_composite_saved_module_is_upcast():
    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Sequential(torch.nn.Linear(8, 8)))
    model = get_peft_model(
        base.to(torch.float16), LoraConfig(r = 4, target_modules = ["0"], modules_to_save = ["1"])
    )
    _freeze_like_kbit_prep(model)
    unfreeze(model)
    saved = [p for n, p in model.named_parameters() if "modules_to_save" in n]
    assert saved and all(p.dtype == torch.float32 and p.requires_grad for p in saved)


def test_idempotent_and_skips_disabled_adapters():
    model = _peft()
    _freeze_like_kbit_prep(model)
    unfreeze(model)
    once = _trainable(model)
    unfreeze(model)
    assert _trainable(model) == once

    _freeze_like_kbit_prep(model)
    with model.disable_adapter():
        unfreeze(model)
        assert not any("modules_to_save" in n for n in _trainable(model))
