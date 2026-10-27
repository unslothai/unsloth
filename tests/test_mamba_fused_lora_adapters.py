# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""LoRA on a Mamba projection the fused training kernel reads raw (x_proj) must still train.

transformers' fused training branch passes `self.x_proj.weight` to `mamba_inner_fn`, so an adapter on
x_proj is bypassed. On CPU the fused path is simulated with a recording stub that, like the real
kernel, only sees the raw weights; the mixer must route to its non-fused path when x_proj carries an
active adapter, and keep the fused path otherwise.
"""

import ast
import importlib
import importlib.util
import pathlib

import pytest
import torch
import torch.nn.functional as F

peft = pytest.importorskip("peft")

ARCHS = {
    "mamba": ("MambaConfig", "MambaForCausalLM", "MambaMixer"),
    "falcon_mamba": ("FalconMambaConfig", "FalconMambaForCausalLM", "FalconMambaMixer"),
}


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_FIX = _ROOT / "unsloth" / "models" / "_mamba_lora.py"


def _patch_fn():
    # Loaded by path (it only needs torch + peft), so no GPU / unsloth_zoo import is needed. A no-op
    # on trees without the fix, so the behavioural assertions, not an ImportError, decide.
    if not _FIX.is_file():
        return lambda model: model
    spec = importlib.util.spec_from_file_location("_unsloth_mamba_lora_under_test", _FIX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.patch_mamba_fused_lora


def _fake_fused(xz, out_proj_weight):
    # The real kernel's output depends only on raw weights; any function of them shows the bypass.
    return F.linear(xz.chunk(2, dim = 1)[0].transpose(1, 2), out_proj_weight)


def _build(monkeypatch, arch, targets):
    import transformers

    config_name, model_name, mixer_name = ARCHS[arch]
    modeling = importlib.import_module(f"transformers.models.{arch}.modeling_{arch}")
    mixer_cls = getattr(modeling, mixer_name)
    calls = []

    # Both transformers lines call the module-level mamba_inner_fn in training without a cache;
    # transformers 5 treats a None return as "no kernel", 4.x returns its result directly.
    def mamba_inner_fn(
        xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, out_proj_weight, *a, **k
    ):
        calls.append("mamba_inner_fn")
        return _fake_fused(xz, out_proj_weight) + 0 * x_proj_weight.sum()

    monkeypatch.setattr(modeling, "mamba_inner_fn", mamba_inner_fn)
    monkeypatch.setattr(mixer_cls, "forward", mixer_cls.forward)  # undo what the fix installs
    if hasattr(mixer_cls, "cuda_kernels_forward"):
        # transformers 4.x dispatches to cuda_kernels_forward only on CUDA with mamba_ssm; force it.
        def forward(
            self,
            hidden_states,
            cache_params = None,
            cache_position = None,
            attention_mask = None,
        ):
            return self.cuda_kernels_forward(
                hidden_states, cache_params, cache_position, attention_mask
            )

        monkeypatch.setattr(mixer_cls, "forward", forward)

    config = getattr(transformers, config_name)(
        vocab_size = 64,
        hidden_size = 16,
        state_size = 4,
        num_hidden_layers = 2,
        intermediate_size = 32,
        time_step_rank = 4,
        conv_kernel = 4,
        use_cache = False,
    )
    torch.manual_seed(0)
    model = getattr(transformers, model_name)(config)
    _patch_fn()(model)  # what FastModel.from_pretrained does at load
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(r = 4, lora_alpha = 8, lora_dropout = 0.0, target_modules = targets),
    )
    _patch_fn()(model)  # and get_peft_model afterwards
    model.train()
    return model, calls


def _loss(model, ids):
    return model(input_ids = ids, labels = ids, use_cache = False).loss


def _lora_param(model, leaf, kind):
    return [p for n, p in model.named_parameters() if f".{leaf}.lora_{kind}." in n]


@pytest.mark.parametrize("arch", list(ARCHS))
def test_x_proj_adapter_leaves_fused_path_and_trains(monkeypatch, arch):
    model, calls = _build(monkeypatch, arch, ["in_proj", "x_proj"])
    ids = torch.randint(0, 64, (2, 8))

    loss = _loss(model, ids)
    loss.backward()
    assert calls == [], f"fused kernel ran with an x_proj adapter active: {calls}"
    for leaf in ("in_proj", "x_proj"):
        grads = [p.grad for p in _lora_param(model, leaf, "B")]
        assert grads and all(g is not None and g.abs().sum() > 0 for g in grads), leaf

    # Changing x_proj lora_B must change the output, as it does for any applied adapter.
    model.zero_grad(set_to_none = True)
    with torch.no_grad():
        before = model(input_ids = ids, use_cache = False).logits
        for p in _lora_param(model, "x_proj", "B"):
            p.normal_(std = 0.5)
    model.train()
    with torch.no_grad():
        after_train = model(input_ids = ids, use_cache = False).logits
    assert (after_train - before).abs().max() > 1e-4


@pytest.mark.parametrize("arch", list(ARCHS))
def test_fused_path_kept_without_raw_weight_adapters(monkeypatch, arch):
    model, calls = _build(monkeypatch, arch, ["in_proj"])
    ids = torch.randint(0, 64, (2, 8))
    _loss(model, ids).backward()
    assert len(calls) == 2  # one per layer
    grads = [p.grad for p in _lora_param(model, "in_proj", "B")]
    assert grads and all(g is not None for g in grads)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_disabled_adapter_keeps_fused_path(monkeypatch, arch):
    model, calls = _build(monkeypatch, arch, ["in_proj", "x_proj"])
    ids = torch.randint(0, 64, (2, 8))
    with model.disable_adapter():
        _loss(model, ids)
    assert len(calls) == 2
    calls.clear()
    _loss(model, ids)
    assert calls == []


def test_post_patch_model_installs_the_routing():
    # FastModel.from_pretrained and get_peft_model both end in FastBaseModel.post_patch_model.
    tree = ast.parse((_ROOT / "unsloth" / "models" / "vision.py").read_text())
    post = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "post_patch_model"
    )
    called = {
        node.func.id
        for node in ast.walk(post)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "patch_mamba_fused_lora" in called
