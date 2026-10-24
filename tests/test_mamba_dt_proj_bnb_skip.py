# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Mamba QLoRA must keep mixer.dt_proj unquantized: transformers 5's Mamba slow path runs
``self.dt_proj.weight @ time_step``, which fails on a packed Params4bit."""

import torch


def test_mamba_skips_dt_proj():
    from unsloth.models.vision import _architecture_skip_modules
    assert "dt_proj" in _architecture_skip_modules(["mamba"])


def test_falcon_mamba_and_others_unchanged():
    from unsloth.models.vision import _architecture_skip_modules

    assert "dt_proj" not in _architecture_skip_modules(["falcon_mamba"])
    assert "dt_proj" not in _architecture_skip_modules(["mamba2"])
    assert _architecture_skip_modules(["llama"]) == []


def test_user_bnb_config_gets_mamba_dt_proj():
    from transformers import BitsAndBytesConfig
    from unsloth.models.vision import _with_architecture_skip_modules

    user = BitsAndBytesConfig(load_in_4bit = True, llm_int8_skip_modules = ["lm_head"])
    out = _with_architecture_skip_modules(user, ["mamba"])
    assert out.llm_int8_skip_modules == ["lm_head", "dt_proj"]
    assert user.llm_int8_skip_modules == ["lm_head"]


def test_skip_key_keeps_tiny_mamba_dt_proj_out_of_bnb():
    import pytest

    pytest.importorskip("bitsandbytes")
    from transformers import MambaConfig, MambaForCausalLM
    from transformers.integrations import bitsandbytes as tbnb
    from unsloth.models.vision import _architecture_skip_modules
    from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES

    config = MambaConfig(vocab_size = 64, hidden_size = 16, state_size = 4, num_hidden_layers = 2, expand = 2)
    with torch.device("meta"):
        model = MambaForCausalLM(config)
    skip = list(SKIP_QUANTIZATION_MODULES) + _architecture_skip_modules(["mamba"])
    from transformers import BitsAndBytesConfig

    qc = BitsAndBytesConfig(load_in_4bit = True, llm_int8_skip_modules = skip)
    try:
        model = tbnb.replace_with_bnb_linear(
            model, modules_to_not_convert = skip, quantization_config = qc
        )
    except TypeError:
        model = tbnb.replace_with_bnb_linear(
            model, modules_to_not_convert = skip, quantization_config = qc, pre_quantized = False
        )
    import bitsandbytes as bnb

    kinds = {
        name: type(m).__name__
        for name, m in model.named_modules()
        if name.endswith(("dt_proj", "in_proj", "x_proj"))
    }
    assert all(
        not isinstance(model.get_submodule(n), bnb.nn.Linear4bit)
        for n in kinds
        if n.endswith("dt_proj")
    ), kinds
    assert any(
        isinstance(model.get_submodule(n), bnb.nn.Linear4bit)
        for n in kinds
        if n.endswith("in_proj")
    ), kinds
