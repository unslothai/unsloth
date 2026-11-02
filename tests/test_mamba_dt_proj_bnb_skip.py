# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Mamba QLoRA must keep mixer.dt_proj unquantized: transformers 5's Mamba slow path runs
``self.dt_proj.weight @ time_step``, which fails on a packed Params4bit."""

import importlib.util

import torch


def _mamba_ssm(monkeypatch, present):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: (object() if present else None)
        if name == "mamba_ssm"
        else real(name, *a, **k),
    )


def test_mamba_skips_dt_proj():
    from unsloth.models.vision import _architecture_skip_modules
    assert "dt_proj" in _architecture_skip_modules(["mamba"])


def test_mamba_skips_fused_kernel_dense_weights(monkeypatch):
    _mamba_ssm(monkeypatch, True)
    from unsloth.models.vision import _architecture_skip_modules

    skip = _architecture_skip_modules(["mamba"])
    assert "x_proj" in skip and "out_proj" in skip
    assert "in_proj" not in skip


def test_falcon_mamba_and_others_unchanged(monkeypatch):
    _mamba_ssm(monkeypatch, False)
    from unsloth.models.vision import _architecture_skip_modules

    assert "dt_proj" not in _architecture_skip_modules(["falcon_mamba"])
    assert "dt_proj" not in _architecture_skip_modules(["mamba2"])
    assert _architecture_skip_modules(["llama"]) == []


def test_user_bnb_config_gets_mamba_dt_proj(monkeypatch):
    _mamba_ssm(monkeypatch, True)
    from transformers import BitsAndBytesConfig
    from unsloth.models.vision import _with_architecture_skip_modules

    user = BitsAndBytesConfig(load_in_4bit = True, llm_int8_skip_modules = ["lm_head"])
    out = _with_architecture_skip_modules(user, ["mamba"])
    assert out.llm_int8_skip_modules == ["lm_head", "dt_proj", "x_proj", "out_proj"]
    assert user.llm_int8_skip_modules == ["lm_head"]


def test_skip_key_keeps_tiny_mamba_dt_proj_out_of_bnb(monkeypatch):
    _mamba_ssm(monkeypatch, True)
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
        if name.endswith(("dt_proj", "in_proj", "x_proj", "out_proj"))
    }
    assert all(
        not isinstance(model.get_submodule(n), bnb.nn.Linear4bit)
        for n in kinds
        if n.endswith(("dt_proj", "x_proj", "out_proj"))
    ), kinds
    assert any(
        isinstance(model.get_submodule(n), bnb.nn.Linear4bit)
        for n in kinds
        if n.endswith("in_proj")
    ), kinds


def test_fused_mamba_kernel_trains_with_skip_list():
    """Real mamba_ssm training step on a random NF4 Mamba built with the skip list."""
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("mamba_ssm")
    pytest.importorskip("causal_conv1d")
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig, MambaConfig, MambaForCausalLM
    from transformers.integrations import bitsandbytes as tbnb
    from transformers.models.mamba import modeling_mamba
    from unsloth.models.vision import _architecture_skip_modules

    config = MambaConfig(vocab_size = 64, hidden_size = 64, state_size = 8, num_hidden_layers = 2, expand = 2)
    torch.manual_seed(0)
    model = MambaForCausalLM(config)
    state = {n: p.detach().clone() for n, p in model.named_parameters()}
    skip = ["lm_head"] + _architecture_skip_modules(["mamba"])
    qc = BitsAndBytesConfig(
        load_in_4bit = True, bnb_4bit_compute_dtype = torch.float32, llm_int8_skip_modules = skip
    )
    try:
        model = tbnb.replace_with_bnb_linear(
            model, modules_to_not_convert = skip, quantization_config = qc
        )
    except TypeError:
        model = tbnb.replace_with_bnb_linear(
            model, modules_to_not_convert = skip, quantization_config = qc, pre_quantized = False
        )
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name + ".weight" in state:
            module.weight = torch.nn.Parameter(state[name + ".weight"])
        if isinstance(module, bnb.nn.Linear4bit):
            module.weight = bnb.nn.Params4bit(
                state[name + ".weight"], requires_grad = False, quant_type = "nf4"
            )
    model = model.cuda()
    assert isinstance(model.backbone.layers[0].mixer.in_proj, bnb.nn.Linear4bit)

    calls = {"n": 0}
    original = modeling_mamba.mamba_inner_fn

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    modeling_mamba.mamba_inner_fn = counting
    try:
        model.train()
        ids = torch.randint(0, 64, (2, 16), device = "cuda")
        loss = model(input_ids = ids, labels = ids, use_cache = False).loss
        loss.backward()
    finally:
        modeling_mamba.mamba_inner_fn = original
    assert calls["n"] > 0
    assert torch.isfinite(loss)
