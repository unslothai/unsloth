# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A trained LoRA on Mamba's dt_proj must also apply in eval, cached decode and generate.

transformers' Mamba torch path (transformers 5 always, 4.x ``cuda_kernels_forward`` outside fused
training) computes ``self.dt_proj.weight @ time_step``, which reads the base weight and skips the
adapter. Unmerged eval / generate must equal the ``merge_and_unload`` model, ``disable_adapter()``
must give the base model, and the fused training path must stay compile friendly.
"""

import copy
import importlib
import importlib.util
import pathlib
import sys
import threading
import warnings

import pytest
import torch
import torch.nn.functional as F

peft = pytest.importorskip("peft")

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_mamba_fused_lora_helpers", _HERE / "test_mamba_fused_lora_adapters.py"
)
helpers = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = helpers  # torch.compile resolves the fake kernel's globals by module
_spec.loader.exec_module(helpers)

ARCHS = list(helpers.ARCHS)
ALL = ["in_proj", "x_proj", "dt_proj"]


# transformers 4.x: the test models always dispatch to cuda_kernels_forward, which outside fused
# training reads dt_proj.weight raw. Its mamba_ssm / causal_conv1d ops get their reference math here
# so prefill and cached decode run on CPU.
def _causal_conv1d_ref(
    x,
    weight,
    bias = None,
    activation = None,
    **kwargs,
):
    seq_len = x.shape[-1]
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding = weight.shape[-1] - 1, groups = x.shape[1])
    out = out[..., :seq_len]
    return F.silu(out) if activation in ("silu", "swish") else out


def _selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D = None,
    z = None,
    delta_bias = None,
    delta_softplus = False,
    return_last_state = False,
):
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1], dtype = torch.float32)
    ys = []
    for t in range(u.shape[-1]):
        state = (
            torch.exp(delta[:, :, t, None] * A) * state
            + delta[:, :, t, None] * B[:, None, :, t] * u[:, :, t, None]
        )
        ys.append((state * C[:, None, :, t]).sum(-1))
    y = torch.stack(ys, dim = -1)
    if D is not None:
        y = y + u * D[None, :, None]
    if z is not None:
        y = y * F.silu(z)
    y = y.to(u.dtype)
    return (y, state) if return_last_state else y


def _causal_conv1d_update_ref(
    x,
    conv_state,
    weight,
    bias = None,
    activation = None,
    **kwargs,
):
    conv_state.copy_(torch.roll(conv_state, shifts = -1, dims = -1))
    conv_state[:, :, -1] = x
    out = (conv_state * weight).sum(-1)
    if bias is not None:
        out = out + bias
    return F.silu(out) if activation in ("silu", "swish") else out


def _selective_state_update_ref(
    state,
    x,
    dt,
    A,
    B,
    C,
    D = None,
    z = None,
    dt_bias = None,
    dt_softplus = False,
    **kwargs,
):
    dt = dt.float()
    if dt_bias is not None:
        dt = dt + dt_bias.float()
    if dt_softplus:
        dt = F.softplus(dt)
    new = torch.exp(dt[..., None] * A) * state + dt[..., None] * B[:, None, :] * x[..., None]
    state.copy_(new)
    y = (new * C[:, None, :]).sum(-1)
    if D is not None:
        y = y + x * D
    if z is not None:
        y = y * F.silu(z)
    return y.to(x.dtype)


def _install_legacy_reference_ops(monkeypatch, arch):
    modeling = importlib.import_module(f"transformers.models.{arch}.modeling_{arch}")
    ops = {
        "selective_scan_fn": _selective_scan_ref,
        "selective_state_update": _selective_state_update_ref,
        "_lazy_load_causal_conv1d": lambda: (_causal_conv1d_update_ref, _causal_conv1d_ref),
        # transformers < 4.56 binds the causal_conv1d ops as module globals instead of lazily.
        "causal_conv1d_fn": _causal_conv1d_ref,
        "causal_conv1d_update": _causal_conv1d_update_ref,
    }
    for name, fn in ops.items():
        monkeypatch.setattr(modeling, name, fn, raising = False)


def _trained(monkeypatch, arch, targets, **kwargs):
    """A model whose adapters took one step through the fused training path, then put in eval."""
    model, calls = helpers._build(
        monkeypatch, arch, targets, kernel = helpers._reference_fused, **kwargs
    )
    ids = torch.randint(0, 64, (2, 8), generator = torch.Generator().manual_seed(5))
    helpers._loss(model, ids).backward()
    assert calls, "training did not take the fused path"
    grads = [p.grad for p in helpers._lora_param(model, "dt_proj", "B")]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is not None:
                p -= 10.0 * p.grad
    model.zero_grad(set_to_none = True)
    model.eval()
    if helpers._legacy(arch):
        _install_legacy_reference_ops(monkeypatch, arch)
    return model


def _merged(model):
    return copy.deepcopy(model).merge_and_unload().eval()


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids = ids, use_cache = False).logits


def _ids(seed = 7):
    return torch.randint(0, 64, (2, 12), generator = torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("targets", [["dt_proj"], ALL], ids = ["dt_proj", "all"])
def test_eval_logits_match_merged(monkeypatch, arch, targets):
    model = _trained(monkeypatch, arch, targets)
    ids = _ids()
    unmerged, merged = _logits(model, ids), _logits(_merged(model), ids)
    with model.disable_adapter():
        base = _logits(model, ids)
    assert (merged - base).abs().max() > 1e-6, "the trained adapter has no effect"
    if targets == ["dt_proj"]:
        # dt_proj is only read as a weight, so W + delta is the merged weight bit for bit.
        assert torch.equal(unmerged, merged), (unmerged - merged).abs().max()
    else:
        torch.testing.assert_close(unmerged, merged, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("arch", ARCHS)
def test_greedy_generate_matches_merged(monkeypatch, arch):
    model = _trained(monkeypatch, arch, ALL)
    merged = _merged(model)
    ids = _ids(11)[:1, :6]
    kwargs = dict(
        max_new_tokens = 8,
        do_sample = False,
        use_cache = True,
        output_scores = True,
        return_dict_in_generate = True,
        pad_token_id = 0,
    )
    with torch.no_grad():
        out_u = model.generate(input_ids = ids, **kwargs)
        out_m = merged.generate(input_ids = ids, **kwargs)
    assert torch.equal(out_u.sequences, out_m.sequences)
    for su, sm in zip(out_u.scores, out_m.scores):
        torch.testing.assert_close(su, sm, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("arch", ARCHS)
def test_disable_adapter_gives_base_model(monkeypatch, arch):
    model = _trained(monkeypatch, arch, ALL)
    ids = _ids()
    base = _logits(copy.deepcopy(model).unload().eval(), ids)
    with model.disable_adapter():
        assert torch.equal(_logits(model, ids), base)
    assert not torch.equal(_logits(model, ids), base)


@pytest.mark.parametrize("arch", ARCHS)
def test_cached_weight_follows_updates_and_adapter_switches(monkeypatch, arch):
    model = _trained(monkeypatch, arch, ["dt_proj"])
    ids = _ids()
    first = _logits(model, ids)
    # An in-place optimizer-style update must invalidate the no-grad cache.
    with torch.no_grad():
        for p in helpers._lora_param(model, "dt_proj", "B"):
            p.mul_(-1.5)
    assert torch.equal(_logits(model, ids), _logits(_merged(model), ids))
    assert not torch.equal(_logits(model, ids), first)

    model.add_adapter("second", peft.LoraConfig(r = 2, lora_alpha = 4, target_modules = ["dt_proj"]))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_B.second." in name:
                p.normal_(std = 0.5)
    model.eval()
    for active in ("second", "default"):
        model.set_adapter(active)
        model.eval()
        assert torch.equal(_logits(model, ids), _logits(_merged(model), ids)), active
    model.base_model.set_adapter(["default", "second"])
    model.eval()
    assert torch.equal(_logits(model, ids), _logits(_merged(model), ids))

    # merge_adapter() folds the delta into the base weight: no double counting, and unmerge restores.
    unmerged = _logits(model, ids)
    model.merge_adapter()
    torch.testing.assert_close(_logits(model, ids), unmerged, rtol = 1e-5, atol = 1e-6)
    model.unmerge_adapter()
    torch.testing.assert_close(_logits(model, ids), unmerged, rtol = 1e-5, atol = 1e-6)


def test_concurrent_eval_threads_each_see_the_adapter(monkeypatch):
    # nn.DataParallel replicas and serving threads run forwards concurrently: the effective weight
    # is armed per thread, so one thread finishing cannot disarm another mid-forward.
    model = _trained(monkeypatch, "mamba", ["dt_proj"])
    ids = _ids()
    expected = _logits(_merged(model), ids)
    results, errors = [], []

    def run():
        try:
            for _ in range(5):
                results.append(torch.equal(_logits(model, ids), expected))
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target = run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    if errors:
        raise errors[0]
    assert len(results) == 20 and all(results)


def test_unfoldable_adapter_warns_on_raw_read(monkeypatch):
    if helpers._legacy("mamba"):
        pytest.skip("transformers 4.x eval on CPU calls dt_proj as a module")
    model, _ = helpers._build(monkeypatch, "mamba", ["dt_proj"], fused = False)
    model = peft.get_peft_model(
        model.unload(),
        peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["dt_proj"], use_dora = True),
    )
    helpers._patch_fn()(model)
    model.eval()
    with pytest.warns(UserWarning, match = r"MambaMixer\.dt_proj is not applied"):
        _logits(model, _ids())


@pytest.mark.parametrize("arch", ARCHS)
def test_fused_training_mixer_compiles_fullgraph(monkeypatch, arch):
    model, calls = helpers._build(monkeypatch, arch, ALL, kernel = helpers._reference_fused)
    mixer = helpers._mixers(model)[0]
    hidden = torch.randn(2, 8, 16, generator = torch.Generator().manual_seed(2))
    params = [p for n, p in mixer.named_parameters() if ".lora_" in n]

    def run(fn):
        for p in params:
            p.grad = None
        out = fn(hidden)
        out.float().pow(2).sum().backward()
        return out.detach(), [p.grad.clone() for p in params]

    out_e, grads_e = run(mixer)
    torch._dynamo.reset()
    compiled = torch.compile(mixer, backend = "aot_eager", fullgraph = True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_c, grads_c = run(compiled)
        run(compiled)
    torch._dynamo.reset()
    torch.testing.assert_close(out_c, out_e, rtol = 1e-5, atol = 1e-6)
    for gc, ge in zip(grads_c, grads_e):
        assert ge.abs().sum() > 0
        torch.testing.assert_close(gc, ge, rtol = 1e-4, atol = 1e-6)
