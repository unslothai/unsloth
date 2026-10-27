# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""LoRA on a Mamba projection the fused training kernel reads raw (x_proj / dt_proj / out_proj) must train.

transformers' fused training branch passes `self.x_proj.weight` (and dt_proj / out_proj) to
`mamba_inner_fn`, so an adapter on those modules is bypassed. On CPU the fused kernel is simulated by
a stub that, like the real kernel, only sees the weights it is handed: plain LoRA must keep the fused
call and hand it `W + delta`; other adapters must route the mixer to its non-fused path.
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
# out_proj is read raw too, but PEFT refuses LoRA on it for these model types.
RAW = ("x_proj", "dt_proj")


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


def _fake_fused(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, out_proj_weight, *a, **k
):
    # The real kernel's output depends only on the weights it is handed; any function of them shows
    # whether an adapter reached it.
    out = F.linear(xz.chunk(2, dim = 1)[0].transpose(1, 2), out_proj_weight)
    return out + 1e-3 * (x_proj_weight.sum() + delta_proj_weight.sum())


def _reference_fused(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    B = None,
    C = None,
    D = None,
    delta_bias = None,
    delta_softplus = True,
    **kwargs,
):
    # mamba_ssm's mamba_inner_fn math (mamba_inner_ref) on the weights it is handed.
    seq_len = xz.shape[-1]
    x, z = xz.chunk(2, dim = 1)
    x = F.conv1d(
        x,
        conv1d_weight,
        conv1d_bias,
        padding = conv1d_weight.shape[-1] - 1,
        groups = x.shape[1],
    )
    x = F.silu(x[..., :seq_len])
    rank, n = delta_proj_weight.shape[1], A.shape[1]
    dt, Bm, Cm = torch.split(F.linear(x.transpose(1, 2), x_proj_weight), [rank, n, n], dim = -1)
    delta = delta_proj_weight @ dt.transpose(1, 2)
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = xz.new_zeros(x.shape[0], x.shape[1], n)
    ys = []
    for t in range(seq_len):
        state = (
            torch.exp(delta[:, :, t, None] * A) * state
            + delta[:, :, t, None] * Bm[:, t, None, :] * x[:, :, t, None]
        )
        ys.append((state * Cm[:, t, None, :]).sum(-1))
    y = (torch.stack(ys, dim = -1) + x * D[None, :, None]) * F.silu(z)
    return F.linear(y.transpose(1, 2), out_proj_weight, out_proj_bias)


def _build(
    monkeypatch,
    arch,
    targets,
    kernel = _fake_fused,
    lora_dropout = 0.0,
    fused = True,
    patch = True,
):
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
        calls.append(
            {
                "x_proj": x_proj_weight,
                "dt_proj": delta_proj_weight,
                "out_proj": out_proj_weight,
            }
        )
        if not fused:
            return None
        return kernel(
            xz,
            conv1d_weight,
            conv1d_bias,
            x_proj_weight,
            delta_proj_weight,
            out_proj_weight,
            *a,
            **k,
        )

    monkeypatch.setattr(modeling, "mamba_inner_fn", mamba_inner_fn)
    monkeypatch.setattr(mixer_cls, "forward", mixer_cls.forward)  # undo what the fix installs
    if hasattr(mixer_cls, "cuda_kernels_forward"):
        # transformers 4.x dispatches to cuda_kernels_forward only on CUDA with mamba_ssm; force it,
        # or its torch path for the non-fused reference.
        def forward(
            self,
            hidden_states,
            cache_params = None,
            cache_position = None,
            attention_mask = None,
        ):
            impl = self.cuda_kernels_forward if fused else self.slow_forward
            return impl(hidden_states, cache_params, cache_position, attention_mask)

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
    if patch:
        _patch_fn()(model)  # what FastModel.from_pretrained does at load
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(r = 4, lora_alpha = 8, lora_dropout = lora_dropout, target_modules = targets),
    )
    if patch:
        _patch_fn()(model)  # and get_peft_model afterwards
    model.train()
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_B." in name:
                p.normal_(std = 0.3)
    return model, calls


def _loss(model, ids):
    return model(input_ids = ids, labels = ids, use_cache = False).loss


def _lora_param(model, leaf, kind):
    return [p for n, p in model.named_parameters() if f".{leaf}.lora_{kind}." in n]


def _mixers(model):
    return [layer.mixer for layer in model.base_model.model.backbone.layers]


def _legacy(arch):
    modeling = importlib.import_module(f"transformers.models.{arch}.modeling_{arch}")
    return hasattr(getattr(modeling, ARCHS[arch][2]), "slow_forward")


@pytest.mark.parametrize("arch", list(ARCHS))
def test_plain_lora_keeps_fused_kernel_with_merged_weights(monkeypatch, arch):
    model, calls = _build(monkeypatch, arch, ["in_proj", *RAW])
    ids = torch.randint(0, 64, (2, 8))

    loss = _loss(model, ids)
    assert len(calls) == 2, "fused kernel must stay on for plain LoRA"
    for mixer, seen in zip(_mixers(model), calls):
        for leaf in RAW:
            module = mixer._modules[leaf]
            expected = module.get_base_layer().weight + module.get_delta_weight("default")
            assert seen[leaf] is not module.get_base_layer().weight, leaf
            torch.testing.assert_close(seen[leaf], expected)
    loss.backward()
    for leaf in ("in_proj", *RAW):
        for kind in ("A", "B"):
            grads = [p.grad for p in _lora_param(model, leaf, kind)]
            assert grads and all(g is not None and g.abs().sum() > 0 for g in grads), (leaf, kind)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, name

    # Changing a raw-weight lora_B must change the output, as it does for any applied adapter.
    for leaf in RAW:
        with torch.no_grad():
            before = model(input_ids = ids, use_cache = False).logits
            for p in _lora_param(model, leaf, "B"):
                p.mul_(2.0)
            after = model(input_ids = ids, use_cache = False).logits
        assert (after - before).abs().max() > 1e-5, leaf


def test_merged_weight_grads_match_non_fused_path(monkeypatch):
    # transformers 5's torch path reads dt_proj.weight raw too, so it is only a reference on 4.x.
    targets = ["in_proj", "x_proj"] + (["dt_proj"] if _legacy("mamba") else [])
    ids = torch.randint(0, 64, (2, 8), generator = torch.Generator().manual_seed(3))
    results = {}
    for fused in (True, False):
        model, calls = _build(
            monkeypatch,
            "mamba",
            targets,
            kernel = _reference_fused,
            fused = fused,
            patch = fused,
        )
        loss = _loss(model, ids)
        loss.backward()
        assert len(calls) == (2 if fused else (0 if _legacy("mamba") else 2))
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if ".lora_" in n}
        results[fused] = (loss.detach(), grads)
        monkeypatch.undo()
    (loss_f, grads_f), (loss_ref, grads_ref) = results[True], results[False]
    torch.testing.assert_close(loss_f, loss_ref, rtol = 1e-5, atol = 1e-6)
    assert grads_f.keys() == grads_ref.keys()
    for leaf in targets:
        names = [n for n in grads_ref if f".{leaf}.lora_" in n]
        assert names, leaf
        for n in names:
            assert grads_ref[n].abs().sum() > 0, n
            torch.testing.assert_close(grads_f[n], grads_ref[n], rtol = 1e-4, atol = 1e-6)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_dropout_routes_to_non_fused_path(monkeypatch, arch):
    model, calls = _build(monkeypatch, arch, ["in_proj", "x_proj"], lora_dropout = 0.1)
    ids = torch.randint(0, 64, (2, 8))
    _loss(model, ids).backward()
    assert calls == [], "dropout on x_proj cannot be merged into the weight"
    for kind in ("A", "B"):
        grads = [p.grad for p in _lora_param(model, "x_proj", kind)]
        assert grads and all(g is not None and g.abs().sum() > 0 for g in grads), kind


def test_dt_proj_adapter_on_transformers_5_torch_path_warns(monkeypatch):
    # transformers 5's torch path computes dt_proj.weight @ time_step, so a dt_proj adapter that
    # cannot be merged into the fused call is not applied there; say so instead of training silently.
    if _legacy("mamba"):
        pytest.skip("transformers 4.x slow_forward calls dt_proj as a module")
    model, calls = _build(monkeypatch, "mamba", ["in_proj", "dt_proj"], lora_dropout = 0.1)
    with pytest.warns(UserWarning, match = "dt_proj"):
        _loss(model, torch.randint(0, 64, (2, 8)))
    assert calls == []


@pytest.mark.parametrize("arch", list(ARCHS))
def test_multiple_active_adapters_route_to_non_fused_path(monkeypatch, arch):
    model, calls = _build(monkeypatch, arch, ["in_proj", "x_proj"])
    model.add_adapter(
        "second", peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["in_proj", "x_proj"])
    )
    model.base_model.set_adapter(["default", "second"])
    model.train()
    _loss(model, torch.randint(0, 64, (2, 8))).backward()
    assert calls == []
    grads = [p.grad for p in _lora_param(model, "x_proj", "A")]
    assert grads and all(g is not None for g in grads)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_fused_path_kept_without_raw_weight_adapters(monkeypatch, arch):
    model, calls = _build(monkeypatch, arch, ["in_proj"])
    ids = torch.randint(0, 64, (2, 8))
    _loss(model, ids).backward()
    assert len(calls) == 2  # one per layer
    for mixer, seen in zip(_mixers(model), calls):
        for leaf in RAW:
            assert seen[leaf] is mixer._modules[leaf].weight, leaf
    grads = [p.grad for p in _lora_param(model, "in_proj", "B")]
    assert grads and all(g is not None for g in grads)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_disabled_adapter_keeps_raw_weights(monkeypatch, arch):
    model, calls = _build(monkeypatch, arch, ["in_proj", "x_proj"])
    ids = torch.randint(0, 64, (2, 8))
    with model.disable_adapter():
        _loss(model, ids)
    assert len(calls) == 2
    for mixer, seen in zip(_mixers(model), calls):
        assert seen["x_proj"] is mixer.x_proj.get_base_layer().weight
    calls.clear()
    _loss(model, ids)
    assert len(calls) == 2
    for mixer, seen in zip(_mixers(model), calls):
        assert seen["x_proj"] is not mixer.x_proj.get_base_layer().weight


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
