# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Continuing from a saved adapter keeps the adapter's own LoRA shape (#13140).

The run's requested rank, alpha and CPT embedding modules cannot be applied to an
existing adapter, so the trainer must name what it ignored instead of dropping it.
"""

import ast
import os
from types import SimpleNamespace

TRAINER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "core",
    "training",
    "trainer.py",
)


def _helper():
    # trainer.py imports torch and unsloth; load only the pure helper.
    tree = ast.parse(open(TRAINER, encoding = "utf-8").read())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_ignored_adapter_settings"
    )
    namespace = {}
    exec(compile(ast.Module(body = [fn], type_ignores = []), TRAINER, "exec"), namespace)
    return namespace["_ignored_adapter_settings"]


def _adapter(
    r = 8,
    lora_alpha = 16,
    modules_to_save = None,
    **extra,
):
    fields = {
        "r": r,
        "lora_alpha": lora_alpha,
        "modules_to_save": modules_to_save,
        "lora_dropout": 0.0,
        "use_dora": False,
        "use_rslora": False,
        "target_modules": {"q_proj", "v_proj"},
    }
    fields.update(extra)
    return {"default": SimpleNamespace(**fields)}


def test_matching_request_ignores_nothing():
    ignored = _helper()
    assert ignored(_adapter(), 8, 16, None) == []
    assert ignored(_adapter(modules_to_save = ["embed_tokens", "lm_head"]), 8, 16, ["lm_head"]) == []


def test_reports_rank_and_alpha_kept_from_adapter():
    ignored = _helper()(_adapter(r = 8, lora_alpha = 16), 16, 32, None)
    assert ignored == ["rank 16 (adapter has 8)", "alpha 32 (adapter has 16)"]


def test_reports_cpt_embedding_modules_the_adapter_does_not_train():
    ignored = _helper()(_adapter(), 8, 16, ["embed_tokens", "model.lm_head"])
    assert ignored == ["training embed_tokens, model.lm_head (adapter does not train them)"]


def test_unknown_peft_config_shape_is_ignored():
    ignored = _helper()
    assert ignored(None, 8, 16, None) == []
    assert ignored({}, 8, 16, None) == []


def test_reports_targets_dropout_and_variant_kept_from_adapter():
    ignored = _helper()(
        _adapter(),
        8,
        16,
        None,
        target_modules = ["q_proj", "k_proj", "v_proj"],
        lora_dropout = 0.1,
        use_dora = True,
    )
    assert ignored == [
        "dropout 0.1 (adapter has 0.0)",
        "DoRA on",
        "target modules k_proj, q_proj, v_proj (adapter has q_proj, v_proj)",
    ]


def test_matching_targets_and_all_linear_are_not_reported():
    ignored = _helper()
    assert ignored(_adapter(), 8, 16, None, target_modules = ["model.q_proj", "v_proj"]) == []
    assert ignored(_adapter(), 8, 16, None, target_modules = ["all-linear"]) == []
    assert (
        ignored(_adapter(target_modules = "all-linear"), 8, 16, None, target_modules = ["q_proj"]) == []
    )


def _tokenizer_loader(fake_auto, base):
    import sys
    import types

    tree = ast.parse(open(TRAINER, encoding = "utf-8").read())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_load_adapter_aware_tokenizer"
    )
    namespace = {}
    exec(compile(ast.Module(body = [fn], type_ignores = []), TRAINER, "exec"), namespace)
    transformers = types.SimpleNamespace(AutoTokenizer = fake_auto)
    model_config = types.SimpleNamespace(
        get_base_model_from_lora_identifier = lambda name, token: base
    )
    return namespace["_load_adapter_aware_tokenizer"], {
        "transformers": transformers,
        "utils.models.model_config": model_config,
    }


class _FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        cls.calls.append((name, kwargs))
        if name.endswith("adapter"):
            raise OSError("no tokenizer files")
        return f"tokenizer:{name}"


def test_adapter_without_tokenizer_loads_the_base_tokenizer(monkeypatch):
    import sys

    _FakeAutoTokenizer.calls = []
    load, modules = _tokenizer_loader(_FakeAutoTokenizer, "org/base")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert (
        load("/runs/adapter", "tok", revision = "abc", local_files_only = False) == "tokenizer:org/base"
    )
    assert _FakeAutoTokenizer.calls[-1] == ("org/base", {"token": "tok", "local_files_only": False})


def test_non_adapter_tokenizer_failure_is_not_masked(monkeypatch):
    import sys

    import pytest

    _FakeAutoTokenizer.calls = []
    load, modules = _tokenizer_loader(_FakeAutoTokenizer, None)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.raises(OSError):
        load("/models/broken-adapter", None)
