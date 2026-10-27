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
import inspect
import pathlib
import threading
import warnings

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
    # Loaded by path (torch + peft only); a no-op on trees without the fix.
    if not _FIX.is_file():
        return lambda model: model
    spec = importlib.util.spec_from_file_location("_unsloth_mamba_lora_under_test", _FIX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.patch_mamba_fused_lora


def _fake_fused(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, out_proj_weight, *a, **k
):
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
    forward_kwargs = False,
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

        if forward_kwargs:
            # transformers 5.5 / 5.10: the dispatching forward also takes **kwargs, slow_forward does not.
            def forward(
                self,
                hidden_states,
                cache_params = None,
                cache_position = None,
                attention_mask = None,
                **kwargs,
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

    for leaf in RAW:
        with torch.no_grad():
            before = model(input_ids = ids, use_cache = False).logits
            for p in _lora_param(model, leaf, "B"):
                p.mul_(2.0)
            after = model(input_ids = ids, use_cache = False).logits
        assert (after - before).abs().max() > 1e-5, leaf


@pytest.mark.parametrize("arch", list(ARCHS))
@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_dispatching_forward_with_trailing_kwargs_is_patched(monkeypatch, arch, dropout):
    # transformers 5.5 / 5.10: forward(..., **kwargs) over slow_forward / cuda_kernels_forward without it.
    if not _legacy(arch):
        pytest.skip("this transformers has no slow_forward / cuda_kernels_forward split")
    model, calls = _build(
        monkeypatch, arch, ["in_proj", "x_proj"], lora_dropout = dropout, forward_kwargs = True
    )
    ids = torch.randint(0, 64, (2, 8))
    _loss(model, ids).backward()
    assert len(calls) == (0 if dropout else 2)
    for mixer in _mixers(model):
        assert mixer(torch.randn(2, 8, 16), unused_by_slow_forward = 1).shape == (2, 8, 16)
    for kind in ("A", "B"):
        grads = [p.grad for p in _lora_param(model, "x_proj", kind)]
        assert grads and all(g is not None and g.abs().sum() > 0 for g in grads), kind


def _assert_fused_with_merged_x_proj(model, calls, ids):
    calls.clear()
    _loss(model, ids).backward()
    assert len(calls) == 2, "fused kernel must stay on for plain LoRA"
    for mixer, seen in zip(_mixers(model), calls):
        assert seen["x_proj"] is not mixer.x_proj.get_base_layer().weight
    grads = [p.grad for p in _lora_param(model, "x_proj", "B")]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_kernel_global_reassigned_after_patching_is_gated_again(monkeypatch, arch):
    # transformers 5.5 / 5.10 MambaMixer.__init__ reassigns the module's mamba_inner_fn, e.g. when a
    # second model is built after the first was patched.
    model, calls = _build(monkeypatch, arch, ["in_proj", "x_proj"])
    modeling = importlib.import_module(f"transformers.models.{arch}.modeling_{arch}")
    modeling.mamba_inner_fn = inspect.unwrap(modeling.mamba_inner_fn)
    _assert_fused_with_merged_x_proj(model, calls, torch.randint(0, 64, (2, 8)))


@pytest.mark.parametrize("arch", list(ARCHS))
def test_routing_state_is_per_thread(monkeypatch, arch):
    # nn.DataParallel replicas run concurrently: another thread's frame or force-slow must not leak in.
    model, calls = _build(monkeypatch, arch, ["in_proj", "x_proj"])
    routing = type(_mixers(model)[0]).forward.__globals__["_routing"]
    routing.force_slow += 1
    routing.stack.append(None)
    errors = []

    def run():
        try:
            _assert_fused_with_merged_x_proj(model, calls, torch.randint(0, 64, (2, 8)))
        except BaseException as e:
            errors.append(e)

    try:
        thread = threading.Thread(target = run)
        thread.start()
        thread.join(120)
    finally:
        routing.force_slow -= 1
        routing.stack.pop()
    assert not thread.is_alive()
    if errors:
        raise errors[0]


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


def _slow_calls(arch, leaf):
    modeling = importlib.import_module(f"transformers.models.{arch}.modeling_{arch}")
    mixer_cls = getattr(modeling, ARCHS[arch][2])
    slow = mixer_cls.slow_forward if _legacy(arch) else mixer_cls.forward
    return f"self.{leaf}(" in inspect.getsource(inspect.unwrap(slow))


@pytest.mark.parametrize("arch", list(ARCHS))
def test_dt_proj_lora_trains_on_fused_kernel(monkeypatch, arch):
    model, calls = _build(monkeypatch, arch, ["in_proj", "dt_proj"])
    ids = torch.randint(0, 64, (2, 8))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        loss = _loss(model, ids)
    assert not any("not applied" in str(w.message) for w in caught)
    assert len(calls) == 2
    for mixer, seen in zip(_mixers(model), calls):
        expected = mixer.dt_proj.get_base_layer().weight + mixer.dt_proj.get_delta_weight("default")
        torch.testing.assert_close(seen["dt_proj"], expected)
        assert seen["x_proj"] is mixer.x_proj.weight
    loss.backward()
    for kind in ("A", "B"):
        grads = [p.grad for p in _lora_param(model, "dt_proj", kind)]
        assert grads and all(g is not None and g.abs().sum() > 0 for g in grads), kind


@pytest.mark.parametrize("arch", list(ARCHS))
def test_unmergeable_dt_proj_routes_only_where_the_slow_path_calls_it(monkeypatch, arch):
    # transformers 5's MambaMixer torch path computes dt_proj.weight @ time_step, so leaving the
    # fused kernel for a dt_proj adapter with dropout would cost speed and still train nothing:
    # stay fused and warn. 4.x's slow_forward calls self.dt_proj(...), so there it routes and trains.
    model, calls = _build(monkeypatch, arch, ["in_proj", "dt_proj"], lora_dropout = 0.1)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _loss(model, torch.randint(0, 64, (2, 8))).backward()
    warned = any("dt_proj" in str(w.message) for w in caught)
    grads = [p.grad for p in _lora_param(model, "dt_proj", "B")]
    trained = bool(grads) and all(g is not None and g.abs().sum() > 0 for g in grads)
    if _slow_calls(arch, "dt_proj"):
        assert calls == []
    else:
        assert len(calls) == 2, "left the fused kernel for an adapter the slow path ignores"
    if _legacy(arch):
        assert trained and not warned
    assert trained != warned, (trained, warned)


def test_dt_proj_lora_without_kernel_on_transformers_5_warns(monkeypatch):
    # No fused kernel (the entry point returns None): transformers 5 falls through to its torch path,
    # which reads dt_proj.weight raw, so even plain LoRA there cannot train.
    if _legacy("mamba"):
        pytest.skip("transformers 4.x slow_forward calls dt_proj as a module")
    model, calls = _build(monkeypatch, "mamba", ["in_proj", "x_proj", "dt_proj"], fused = False)
    with pytest.warns(UserWarning, match = r"MambaMixer\.dt_proj") as caught:
        _loss(model, torch.randint(0, 64, (2, 8))).backward()
    assert not any("x_proj" in str(w.message) for w in caught)
    assert len(calls) == 2
    grads = [p.grad for p in _lora_param(model, "x_proj", "B")]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


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
