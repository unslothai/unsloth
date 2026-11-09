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
import pickle
import subprocess
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
def test_eval_weight_follows_updates_and_adapter_switches(monkeypatch, arch):
    model = _trained(monkeypatch, arch, ["dt_proj"])
    ids = _ids()
    first = _logits(model, ids)
    # An in-place optimizer-style update must show in the next eval forward.
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


@pytest.mark.parametrize("arch", ARCHS)
def test_eval_weight_follows_data_reassignment(monkeypatch, arch):
    # `p.data = t` (custom LoRA inits, dtype casts at save time) keeps id(p) and p._version.
    model = _trained(monkeypatch, arch, ["dt_proj"])
    ids = _ids()
    first = _logits(model, ids)
    for p in helpers._lora_param(model, "dt_proj", "B"):
        p.data = p.data * -1.5
    assert torch.equal(_logits(model, ids), _logits(_merged(model), ids))
    assert not torch.equal(_logits(model, ids), first)


@pytest.mark.parametrize("arch", ARCHS)
def test_eval_weight_follows_dtype_conversion(monkeypatch, arch):
    # model.double() / .bfloat16() keep the parameter objects and their version counters.
    model = _trained(monkeypatch, arch, ["dt_proj"])
    ids = _ids()
    _logits(model, ids)
    model.double()
    assert all(p.dtype == torch.float64 for p in helpers._lora_param(model, "dt_proj", "B"))
    assert torch.equal(_logits(model, ids), _logits(_merged(model), ids))


@pytest.mark.parametrize("arch", ARCHS)
def test_eval_weight_follows_unversioned_optimizer_step(monkeypatch, arch):
    # FlashAdamW's Triton update writes the parameters in place without bumping their version
    # counters (as an in-place op through `p.data` does), so eval -> train step -> eval must not
    # serve the W + delta of the first eval.
    model = _trained(monkeypatch, arch, ["dt_proj"])
    ids = _ids()
    first = _logits(model, ids)
    model.train()
    helpers._loss(model, ids).backward()
    for p in helpers._lora_param(model, "dt_proj", "B"):
        version = p._version
        p.data.add_(p.grad.data, alpha = -10.0)
        p.data.mul_(-1.5)
        assert p._version == version
    model.zero_grad(set_to_none = True)
    model.eval()
    assert torch.equal(_logits(model, ids), _logits(_merged(model), ids))
    assert not torch.equal(_logits(model, ids), first)


@pytest.mark.parametrize("arch", ARCHS)
def test_eval_holds_no_effective_weight_copy(monkeypatch, arch):
    # W + delta is rebuilt per read, never stored on the leaf, so model.cpu() / .to() leave no
    # stale copy behind and every later read sees the converted weights.
    model = _trained(monkeypatch, arch, ["dt_proj"])
    ids = _ids()
    _logits(model, ids)
    leaves = [m for m in model.modules() if type(m).__name__ == "Linear" and hasattr(m, "lora_A")]
    assert leaves
    for leaf in leaves:
        assert not any(isinstance(v, torch.Tensor) for v in leaf.__dict__.values())
        assert not any(
            isinstance(v, tuple) and any(isinstance(t, torch.Tensor) for t in v)
            for v in leaf.__dict__.values()
        )
    model.to(torch.float64)
    assert torch.equal(_logits(model, ids), _logits(_merged(model), ids))


@pytest.mark.parametrize("arch", ARCHS)
def test_autocast_eval_does_not_leak_into_full_precision(monkeypatch, arch):
    # W + delta is built outside autocast: no bf16-rounded delta from an earlier autocast forward.
    model = _trained(monkeypatch, arch, ["dt_proj"])
    ids = _ids()
    with torch.autocast("cpu", dtype = torch.bfloat16):
        _logits(model, ids)
    assert torch.equal(_logits(model, ids), _logits(_merged(model), ids))


@pytest.mark.parametrize("arch", ARCHS)
def test_mixed_adapter_batch_routes_each_row(monkeypatch, arch):
    # PEFT's adapter_names= picks an adapter per row ("__base__" = none) through LoRA pre-hooks that a
    # raw weight read bypasses: each row must still get exactly its own adapter.
    model = _trained(monkeypatch, arch, ["dt_proj"])
    ids = _ids()
    adapted = _logits(model, ids)
    with model.disable_adapter():
        base = _logits(model, ids)
    assert (adapted - base).abs().max() > 1e-6, "the trained adapter has no effect"
    with torch.no_grad():
        mixed = model(input_ids = ids, use_cache = False, adapter_names = ["__base__", "default"]).logits
    torch.testing.assert_close(mixed[0], base[0], rtol = 1e-6, atol = 1e-6)
    torch.testing.assert_close(mixed[1], adapted[1], rtol = 1e-6, atol = 1e-6)
    # The mixed batch must not poison the plain path afterwards.
    assert torch.equal(_logits(model, ids), adapted)


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


def test_pickled_model_loads_without_unsloth(monkeypatch, tmp_path):
    # torch.save(model) / pickle / deepcopy must stay loadable in a process that never ran the patch.
    model = _trained(monkeypatch, "mamba", ["dt_proj"])
    ids = _ids()
    expected = _logits(model, ids)
    leaf = model.base_model.model.backbone.layers[0].mixer.dt_proj
    data = pickle.dumps(model)
    assert b"_RawRead_" not in data
    # copy / deepcopy keep the effective-weight class (see the compiled deepcopy test below).
    assert type(copy.deepcopy(leaf)) is type(leaf)
    assert torch.equal(_logits(pickle.loads(data), ids), expected)
    path = tmp_path / "model.pkl"
    path.write_bytes(data)
    code = (
        "import pickle, sys; m = pickle.load(open(sys.argv[1], 'rb')); "
        "print(type(m.base_model.model.backbone.layers[0].mixer.dt_proj).__qualname__)"
    )
    out = subprocess.run([sys.executable, "-c", code, str(path)], capture_output = True, text = True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == type(leaf).__mro__[1].__qualname__


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


def _hotswap_fn():
    try:
        from peft.utils.hotswap import hotswap_adapter
    except Exception:
        return None
    return hotswap_adapter


@pytest.mark.parametrize("arch", ARCHS)
def test_hotswapped_adapter_applies_in_eval(monkeypatch, arch, tmp_path):
    # PEFT's hotswap_adapter replaces same-shaped LoRA weights through `old.data.copy_(new)`, which
    # keeps each parameter's identity, storage and version counter; eval must serve the new adapter.
    model = _trained(monkeypatch, arch, ["dt_proj"])
    ids = _ids()
    other = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in other.named_parameters():
            if ".lora_" in name:
                p.normal_(std = 2.0, generator = torch.Generator().manual_seed(3))
    expected = _logits(_merged(other), ids)
    old = _logits(model, ids)  # an eval forward before the swap
    assert (expected - old).abs().max() > 1e-3, "the replacement adapter has no effect"
    hotswap = _hotswap_fn()
    if hotswap is not None:
        other.save_pretrained(tmp_path)
        hotswap(model, str(tmp_path), adapter_name = "default", torch_device = "cpu")
    else:
        new = dict(other.named_parameters())
        for name, p in model.named_parameters():
            if ".lora_" in name:
                p.data.copy_(new[name].data)
    # No model.eval() / train() here: a hot swap in a serving process calls neither.
    assert torch.equal(_logits(model, ids), expected)


@pytest.mark.parametrize("arch", ARCHS)
def test_eval_mixer_compiles_fullgraph_under_no_grad(monkeypatch, arch):
    # A no-grad eval forward reads dt_proj.weight raw; under torch.compile it must trace in one
    # graph (no cache lookups or mutations inside it) and match eager.
    model = _trained(monkeypatch, arch, ["dt_proj"])
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_" in name:
                p.normal_(std = 2.0, generator = torch.Generator().manual_seed(3))
    mixer = helpers._mixers(model)[0]
    hidden = torch.randn(2, 8, 16, generator = torch.Generator().manual_seed(2))
    with torch.no_grad():
        eager = mixer(hidden)
        with model.disable_adapter():
            base = mixer(hidden)
    assert (eager - base).abs().max() > 1e-5, "the adapter has no effect"
    torch._dynamo.reset()
    try:
        compiled = torch.compile(mixer, backend = "aot_eager", fullgraph = True)
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            out = compiled(hidden)
            again = compiled(hidden)
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(out, eager, rtol = 1e-5, atol = 1e-6)
    assert torch.equal(out, again)


def _adapted_mixer(monkeypatch, arch):
    model = _trained(monkeypatch, arch, ["dt_proj"])
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_" in name:
                p.normal_(std = 2.0, generator = torch.Generator().manual_seed(3))
    return model, helpers._mixers(model)[0]


def _compiled_no_grad(
    mixer,
    hidden,
    fullgraph = True,
):
    torch._dynamo.reset()
    try:
        compiled = torch.compile(mixer, backend = "aot_eager", fullgraph = fullgraph)
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            out = compiled(hidden)
            again = compiled(hidden)
    finally:
        torch._dynamo.reset()
    assert torch.equal(out, again)
    return out


def _eager_and_base(model, mixer, hidden):
    with torch.no_grad():
        eager = mixer(hidden)
        with model.disable_adapter():
            base = mixer(hidden)
    assert (eager - base).abs().max() > 1e-5, "the adapter has no effect"
    return eager


@pytest.mark.parametrize("arch", ARCHS)
def test_deepcopied_eval_mixer_compiles_with_its_adapter(monkeypatch, arch):
    # A deep copy compiled before any eager forward must still serve W + delta, not the base weight.
    model, mixer = _adapted_mixer(monkeypatch, arch)
    copied = copy.deepcopy(mixer)
    hidden = torch.randn(2, 8, 16, generator = torch.Generator().manual_seed(2))
    out = _compiled_no_grad(copied, hidden)
    torch.testing.assert_close(out, _eager_and_base(model, mixer, hidden), rtol = 1e-5, atol = 1e-6)
    assert type(copied.dt_proj) is type(mixer.dt_proj)


@pytest.mark.parametrize("arch", ARCHS)
def test_adapter_attached_after_patch_compiles_with_it(monkeypatch, arch):
    # A dt_proj adapter added after the patch, compiled before any eager forward, must apply.
    model, _ = helpers._build(monkeypatch, arch, ["x_proj"], kernel = helpers._reference_fused)
    if helpers._legacy(arch):
        _install_legacy_reference_ops(monkeypatch, arch)
    model.add_adapter("late", peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["dt_proj"]))
    model.set_adapter("late")
    model.eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_" in name and ".late." in name:
                p.normal_(std = 2.0, generator = torch.Generator().manual_seed(3))
    mixer = helpers._mixers(model)[0]
    hidden = torch.randn(2, 8, 16, generator = torch.Generator().manual_seed(2))
    out = _compiled_no_grad(mixer, hidden)
    torch.testing.assert_close(out, _eager_and_base(model, mixer, hidden), rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("arch", ARCHS)
def test_unprepared_adapter_never_compiles_silently(monkeypatch, arch):
    # A leaf that lost its effective-weight class: fullgraph fails loudly, otherwise the frame runs
    # eagerly (which installs it) and every call applies the adapter.
    model, mixer = _adapted_mixer(monkeypatch, arch)
    hidden = torch.randn(2, 8, 16, generator = torch.Generator().manual_seed(2))
    eager = _eager_and_base(model, mixer, hidden)
    plain = type(mixer.dt_proj).__mro__[1]
    mixer.dt_proj.__class__ = plain
    with pytest.raises(Exception, match = r"dt_proj has an active LoRA adapter"):
        _compiled_no_grad(mixer, hidden)
    mixer.dt_proj.__class__ = plain
    out = _compiled_no_grad(mixer, hidden, fullgraph = False)
    torch.testing.assert_close(out, eager, rtol = 1e-5, atol = 1e-6)


def _row_logits(model, sequences, prompt_len, names):
    """Per-step logits of each generated row, recomputed with that row's adapter alone."""
    out = []
    for row, name in zip(sequences, names):
        if name == "__base__":
            with model.disable_adapter():
                logits = _logits(model, row[None])
        else:
            logits = _logits(model, row[None])
        out.append(logits[0, prompt_len - 1 : -1])
    return torch.stack(out)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize(
    "names", [["__base__", "default"], ["default", "default"]], ids = ["mixed", "same"]
)
@pytest.mark.parametrize("greedy", [False, True], ids = ["sample", "greedy"])
def test_mixed_adapter_generate_num_return_sequences(monkeypatch, arch, names, greedy):
    # generate expands each prompt num_return_sequences times (repeat_interleave), while PEFT's
    # adapter_names hook keeps one name per prompt: every expanded row keeps its prompt's adapter.
    # transformers refuses num_return_sequences without sampling, so greedy is top_k = 1 sampling.
    model = _trained(monkeypatch, arch, ["dt_proj"])
    ids = _ids(13)[:, :4]
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.generate(
            input_ids = ids,
            adapter_names = names,
            do_sample = True,
            top_k = 1 if greedy else 0,
            num_return_sequences = 2,
            max_new_tokens = 3,
            use_cache = True,
            output_logits = True,
            return_dict_in_generate = True,
            pad_token_id = 0,
        )
    assert out.sequences.shape[0] == 4
    row_names = [name for name in names for _ in range(2)]
    assert torch.equal(out.sequences[:, :4], ids.repeat_interleave(2, dim = 0))
    got = torch.stack(out.logits, dim = 1)
    expected = _row_logits(model, out.sequences, 4, row_names)
    torch.testing.assert_close(got, expected, rtol = 1e-5, atol = 1e-5)
    if greedy:
        for i, name in enumerate(names):
            kwargs = dict(max_new_tokens = 3, do_sample = False, use_cache = True, pad_token_id = 0)
            with torch.no_grad():
                if name == "__base__":
                    with model.disable_adapter():
                        alone = model.generate(input_ids = ids[i : i + 1], **kwargs)
                else:
                    alone = model.generate(input_ids = ids[i : i + 1], **kwargs)
            assert torch.equal(out.sequences[2 * i], alone[0])
            assert torch.equal(out.sequences[2 * i + 1], alone[0])


@pytest.mark.parametrize("arch", ARCHS)
def test_mixed_adapter_generate_beam_search(monkeypatch, arch):
    # PEFT repeats adapter_names num_beams times itself; they must be used as given.
    model = _trained(monkeypatch, arch, ["dt_proj"])
    ids = _ids(17)[:, :4]
    names = ["__base__", "default"]
    kwargs = dict(
        max_new_tokens = 3,
        num_beams = 2,
        num_return_sequences = 2,
        do_sample = False,
        use_cache = True,
        pad_token_id = 0,
    )
    with torch.no_grad():
        out = model.generate(input_ids = ids, adapter_names = names, **kwargs)
        assert out.shape[0] == 4
        for i, name in enumerate(names):
            if name == "__base__":
                with model.disable_adapter():
                    alone = model.generate(input_ids = ids[i : i + 1], **kwargs)
            else:
                alone = model.generate(input_ids = ids[i : i + 1], **kwargs)
            assert torch.equal(out[2 * i : 2 * i + 2], alone)


def test_mixed_adapter_names_not_dividing_batch_raise(monkeypatch):
    model = _trained(monkeypatch, "mamba", ["dt_proj"])
    ids = torch.randint(0, 64, (3, 6), generator = torch.Generator().manual_seed(3))
    with torch.no_grad(), pytest.raises(ValueError, match = "adapter_names"):
        model(input_ids = ids, use_cache = False, adapter_names = ["__base__", "default"])
