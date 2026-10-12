# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Gated-delta hybrid packing on by default: the mode switch, the exactness probe and the
unsloth-zoo Triton conv acceptance, on small CPU stand-ins for the GatedDeltaNet mixer."""

import functools
import inspect
import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import unsloth.utils.packing as packing_module
from unsloth.utils.packing import patch_hybrid_linear_attention_varlen

ENV = "UNSLOTH_EXPERIMENTAL_HYBRID_PACKING"


def _segments(T, seq_idx):
    if seq_idx is None:
        return [(0, T)]
    ids = seq_idx.reshape(-1)
    cuts = (torch.nonzero(ids[1:] != ids[:-1]).flatten() + 1).tolist()
    bounds = [0] + cuts + [T]
    return list(zip(bounds[:-1], bounds[1:]))


def conv_exact(x, weight, bias = None, activation = None, seq_idx = None, **kwargs):
    # x (B, D, T); each segment is convolved alone (left zero padding per segment).
    import torch.nn.functional as F

    outs = []
    for s, e in _segments(x.shape[-1], seq_idx):
        seg = x[:, :, s:e]
        y = F.conv1d(seg, weight.unsqueeze(1), bias, padding = weight.shape[-1] - 1, groups = x.shape[1])
        outs.append(y[:, :, : e - s])
    out = torch.cat(outs, dim = -1)
    return F.silu(out) if activation else out


def conv_leaky(x, weight, bias = None, activation = None, seq_idx = None, **kwargs):
    # Names seq_idx (passes the static check) but ignores it.
    return conv_exact(x, weight, bias, activation, None)


def scan_exact(x, cu_seqlens = None, **kwargs):
    # A decayed running sum that restarts at every cu_seqlens boundary.
    T = x.shape[1]
    starts = set(cu_seqlens.tolist()[:-1]) if cu_seqlens is not None else {0}
    out, state = [], torch.zeros_like(x[:, 0])
    for t in range(T):
        if t in starts:
            state = torch.zeros_like(state)
        state = 0.7 * state + x[:, t]
        out.append(state)
    return torch.stack(out, dim = 1)


def scan_leaky(x, cu_seqlens = None, **kwargs):
    return scan_exact(x, None)


class FakeGatedDeltaNet(torch.nn.Module):
    """transformers < 5.16 shape: instance kernels, the conv called with seq_idx=None."""

    def __init__(self, conv = conv_exact, scan = scan_exact, hidden = 8, width = 4):
        super().__init__()
        if hidden is not None:
            self.hidden_size = hidden
        self.conv1d = torch.nn.Conv1d(8, 8, width, groups = 8, bias = False)
        self.causal_conv1d_fn = conv
        self.chunk_gated_delta_rule = scan

    def forward(self, hidden_states, cache_params = None, attention_mask = None):
        x = hidden_states.transpose(1, 2)
        x = self.causal_conv1d_fn(
            x = x, weight = self.conv1d.weight.squeeze(1), bias = None, activation = "silu", seq_idx = None
        )
        return self.chunk_gated_delta_rule(x.transpose(1, 2), cu_seqlens = None)


class FakeHybrid(torch.nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.config = SimpleNamespace(layer_types = ["linear_attention", "full_attention"])
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(32, 8)
        self.linear_attn = mixer

    def forward(self, input_ids = None, position_ids = None, packed_seq_lengths = None, use_cache = None, **kw):
        return self.linear_attn(self.embed(input_ids))


@pytest.fixture(autouse = True)
def _fresh_warnings(monkeypatch):
    packing_module._HYBRID_WARNED.clear()
    # The stand-ins live on CPU; default-on is CUDA-only (tested on its own below).
    monkeypatch.setattr(packing_module, "_auto_hybrid_device_ok", lambda module: True)
    yield
    packing_module._HYBRID_WARNED.clear()


def _untouched(model, conv, scan):
    mixer = model.linear_attn
    return (
        mixer.causal_conv1d_fn is conv
        and mixer.chunk_gated_delta_rule is scan
        and "forward" not in mixer.__dict__
        and not getattr(mixer, "_unsloth_varlen_wrapped", False)
        and not getattr(model, "_unsloth_varlen_forward_wrapped", False)
    )


@pytest.mark.parametrize(
    "value,mode",
    [(None, "auto"), ("", "auto"), ("maybe", "auto"), ("1", "on"), ("TRUE", "on"), ("yes", "on"),
     ("on", "on"), ("0", "off"), ("false", "off"), ("No", "off"), (" off ", "off")],
)
def test_mode_parsing(monkeypatch, value, mode):
    if value is None:
        monkeypatch.delenv(ENV, raising = False)
    else:
        monkeypatch.setenv(ENV, value)
    assert packing_module._hybrid_packing_mode() == mode
    assert packing_module._hybrid_packing_enabled() is (mode == "on")


def test_default_packs_exact_gated_delta(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    model = FakeHybrid(FakeGatedDeltaNet())
    rng = torch.get_rng_state()
    assert patch_hybrid_linear_attention_varlen(model) is True
    assert torch.equal(rng, torch.get_rng_state())  # the probe uses its own generator
    assert not packing_module._HYBRID_WARNED
    ids = torch.randint(0, 32, (1, 9), generator = torch.Generator().manual_seed(1))
    lengths = [4, 1, 4]
    packed = model(input_ids = ids, packed_seq_lengths = torch.tensor(lengths, dtype = torch.int32), use_cache = False)
    assert model._unsloth_varlen_handshake_done
    alone = torch.cat([model(input_ids = ids[:, s:s + n]) for s, n in zip((0, 4, 5), lengths)], dim = 1)
    torch.testing.assert_close(packed, alone)
    # A cached / plain forward is not packed.
    plain = model(input_ids = ids, use_cache = True)
    assert not torch.allclose(plain, packed)


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_kill_switch_keeps_padded_path(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    model = FakeHybrid(FakeGatedDeltaNet())
    assert patch_hybrid_linear_attention_varlen(model) is False
    assert _untouched(model, conv_exact, scan_exact)


@pytest.mark.parametrize("mode", [None, "1"])
@pytest.mark.parametrize("conv,scan", [(conv_leaky, scan_exact), (conv_exact, scan_leaky)])
def test_probe_rejects_a_leaky_kernel_and_rolls_back(monkeypatch, caplog, mode, conv, scan):
    # Flip test for the probe: kernels that name the boundary argument but ignore it.
    if mode is None:
        monkeypatch.delenv(ENV, raising = False)
    else:
        monkeypatch.setenv(ENV, mode)
    model = FakeHybrid(FakeGatedDeltaNet(conv = conv, scan = scan))
    with caplog.at_level(logging.WARNING, logger = "unsloth.hybrid_packing"):
        assert patch_hybrid_linear_attention_varlen(model) is False
    assert "exactness probe" in caplog.text and "differs" in caplog.text
    assert _untouched(model, conv, scan)


def test_probe_failure_to_run_is_a_rejection(monkeypatch, caplog):
    monkeypatch.delenv(ENV, raising = False)

    def conv_raises(x, weight, bias = None, activation = None, seq_idx = None, **kwargs):
        raise RuntimeError("kernel exploded")

    model = FakeHybrid(FakeGatedDeltaNet(conv = conv_raises))
    with caplog.at_level(logging.WARNING, logger = "unsloth.hybrid_packing"):
        assert patch_hybrid_linear_attention_varlen(model) is False
    assert "kernel exploded" in caplog.text
    assert _untouched(model, conv_raises, scan_exact)


def test_probe_requires_dispatch(monkeypatch, caplog):
    monkeypatch.delenv(ENV, raising = False)

    class BypassGatedDeltaNet(FakeGatedDeltaNet):
        def forward(self, hidden_states, cache_params = None, attention_mask = None):
            return conv_exact(hidden_states.transpose(1, 2), self.conv1d.weight.squeeze(1)).transpose(1, 2)

    mixer = BypassGatedDeltaNet()
    model = FakeHybrid(mixer)
    with caplog.at_level(logging.WARNING, logger = "unsloth.hybrid_packing"):
        assert patch_hybrid_linear_attention_varlen(model) is False
    assert "not dispatched" in caplog.text


def test_default_needs_a_probeable_mixer(monkeypatch, caplog):
    monkeypatch.delenv(ENV, raising = False)
    model = FakeHybrid(FakeGatedDeltaNet(hidden = None))
    with caplog.at_level(logging.WARNING, logger = "unsloth.hybrid_packing"):
        assert patch_hybrid_linear_attention_varlen(model) is False
    assert "probe" in caplog.text
    assert _untouched(model, conv_exact, scan_exact)
    # The explicit opt-in keeps its runtime handshake for mixers the probe cannot drive.
    monkeypatch.setenv(ENV, "1")
    assert patch_hybrid_linear_attention_varlen(FakeHybrid(FakeGatedDeltaNet(hidden = None))) is True


def test_default_rejects_torch_fallback(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)

    def torch_chunk_gated_delta_rule(x, cu_seqlens = None, **kwargs):
        return scan_exact(x, cu_seqlens)

    model = FakeHybrid(FakeGatedDeltaNet(scan = torch_chunk_gated_delta_rule))
    assert patch_hybrid_linear_attention_varlen(model) is False


# transformers >= 5.16: forward(**kwargs) splats into module-level hub kernels.


def _hub_wrapped(torch_function, implementation):
    # Same closure shape as transformers' use_kernel_func_from_hub_with_fallback.
    applicable_params = tuple(inspect.signature(implementation).parameters)

    @functools.wraps(torch_function)
    def wrapped(*args, **kwargs):
        if implementation is None:  # transformers references torch_function here too (export path)
            return torch_function(*args, **kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k in applicable_params}
        return implementation(*args, **kwargs)

    return wrapped


def _zoo_style_dispatch(torch_function, name_seq_idx = True):
    # unsloth-zoo gdn_causal_conv1d._make_hub_dispatch: marked, own signature naming seq_idx.
    if name_seq_idx:
        @functools.wraps(torch_function)
        def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, seq_idx = None, **kwargs):
            return conv_exact(hidden_states, weight, bias, activation, seq_idx)
    else:
        @functools.wraps(torch_function)
        def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, **kwargs):
            return conv_exact(hidden_states, weight, bias, activation)
    causal_conv1d_fn._unsloth_triton_causal_conv1d = True
    causal_conv1d_fn.__signature__ = inspect.signature(causal_conv1d_fn, follow_wrapped = False)
    return causal_conv1d_fn


def _kwargs_model(name, conv_impl):
    def conv_ref(hidden_states, weight, bias = None, activation = None, **kwargs):
        return conv_exact(hidden_states, weight, bias, activation)

    def scan_ref(x, **kwargs):
        return scan_exact(x)

    def scan_fast(x, cu_seqlens = None, **kwargs):
        return scan_exact(x, cu_seqlens)

    modeling = types.ModuleType(name)
    impl = conv_impl(conv_ref)
    # Without the causal_conv1d package the hub binds the (zoo-wrapped) torch function itself.
    modeling.causal_conv1d_fn = _hub_wrapped(impl, impl)
    modeling.torch_chunk_gated_delta_rule = _hub_wrapped(scan_ref, scan_fast)
    exec(
        "import torch\n"
        "class FakeKwargsGatedDeltaNet(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.hidden_size = 8\n"
        "        self.conv1d = torch.nn.Conv1d(8, 8, 4, groups = 8, bias = False)\n"
        "    def forward(self, hidden_states, cache_params = None, attention_mask = None, **kwargs):\n"
        "        x = causal_conv1d_fn(hidden_states.transpose(1, 2), self.conv1d.weight.squeeze(1), None,\n"
        "                             activation = 'silu', **kwargs).transpose(1, 2)\n"
        "        return torch_chunk_gated_delta_rule(x, cu_seqlens = kwargs.pop('cu_seq_lens_q', None), **kwargs)\n",
        modeling.__dict__,
    )
    modeling.FakeKwargsGatedDeltaNet.__module__ = name
    sys.modules[name] = modeling
    return FakeHybrid(modeling.FakeKwargsGatedDeltaNet())


@pytest.mark.parametrize("mode", [None, "1"])
def test_zoo_triton_conv_is_accepted(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv(ENV, raising = False)
    else:
        monkeypatch.setenv(ENV, mode)
    name = "fake_modeling_zoo_conv_gdn"
    try:
        model = _kwargs_model(name, _zoo_style_dispatch)
        assert patch_hybrid_linear_attention_varlen(model) is True
        ids = torch.randint(0, 32, (1, 9), generator = torch.Generator().manual_seed(2))
        packed = model(input_ids = ids, packed_seq_lengths = torch.tensor([4, 1, 4], dtype = torch.int32), use_cache = False)
        alone = torch.cat([model(input_ids = ids[:, s:e]) for s, e in ((0, 4), (4, 5), (5, 9))], dim = 1)
        torch.testing.assert_close(packed, alone)
        assert model.linear_attn._unsloth_varlen_kwargs_hit
    finally:
        sys.modules.pop(name, None)


def test_zoo_triton_conv_without_seq_idx_is_rejected(monkeypatch, caplog):
    # An older unsloth-zoo dispatch (marked, no seq_idx) would be filtered by the hub wrapper.
    monkeypatch.delenv(ENV, raising = False)
    name = "fake_modeling_zoo_conv_old"
    try:
        model = _kwargs_model(name, functools.partial(_zoo_style_dispatch, name_seq_idx = False))
        with caplog.at_level(logging.WARNING, logger = "unsloth.hybrid_packing"):
            assert patch_hybrid_linear_attention_varlen(model) is False
        assert "causal_conv1d_fn" in caplog.text
    finally:
        sys.modules.pop(name, None)


def test_unmarked_torch_conv_is_rejected(monkeypatch, caplog):
    monkeypatch.delenv(ENV, raising = False)
    name = "fake_modeling_torch_conv"
    try:
        model = _kwargs_model(name, lambda fn: fn)
        with caplog.at_level(logging.WARNING, logger = "unsloth.hybrid_packing"):
            assert patch_hybrid_linear_attention_varlen(model) is False
        assert "has no accelerated kernel" in caplog.text
    finally:
        sys.modules.pop(name, None)


def test_real_zoo_dispatch_is_recognised():
    gcc = pytest.importorskip("unsloth_zoo.temporary_patches.gdn_causal_conv1d")
    if "seq_idx" not in inspect.signature(gcc._legacy_causal_conv1d_fn).parameters:
        pytest.skip("unsloth-zoo without the seq_idx Triton conv")

    def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, **kwargs):
        return hidden_states

    assert packing_module._is_unsloth_seq_idx_conv(gcc._make_hub_dispatch(causal_conv1d_fn))
    assert packing_module._is_unsloth_seq_idx_conv(gcc._legacy_causal_conv1d_fn)
    assert not packing_module._is_unsloth_seq_idx_conv(causal_conv1d_fn)


def test_default_on_is_cuda_only(monkeypatch, caplog):
    monkeypatch.delenv(ENV, raising = False)
    monkeypatch.undo()  # the real device check
    monkeypatch.delenv(ENV, raising = False)
    mixer = FakeGatedDeltaNet()
    assert not packing_module._auto_hybrid_device_ok(mixer)  # CPU
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert not packing_module._auto_hybrid_device_ok(mixer)
    monkeypatch.setattr(torch.version, "hip", None)
    model = FakeHybrid(mixer)
    with caplog.at_level(logging.WARNING, logger = "unsloth.hybrid_packing"):
        assert patch_hybrid_linear_attention_varlen(model) is False
    assert "only on CUDA" in caplog.text
    assert _untouched(model, conv_exact, scan_exact)
    # The explicit opt-in still runs the exactness probe off CUDA.
    monkeypatch.setenv(ENV, "1")
    assert patch_hybrid_linear_attention_varlen(FakeHybrid(FakeGatedDeltaNet())) is True


def test_engagement_counters(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    stats = dict(packing_module.HYBRID_PACKING_STATS)
    assert patch_hybrid_linear_attention_varlen(FakeHybrid(FakeGatedDeltaNet())) is True
    assert patch_hybrid_linear_attention_varlen(FakeHybrid(FakeGatedDeltaNet(conv = conv_leaky))) is False
    now = packing_module.HYBRID_PACKING_STATS
    assert now["probe_runs"] == stats["probe_runs"] + 2
    assert now["probe_passed"] == stats["probe_passed"] + 1
    assert now["probe_failed"] == stats["probe_failed"] + 1
    assert now["patched"] == stats["patched"] + 1


def test_hub_kernel_behind_compile_disable_wrapper(monkeypatch):
    # unsloth_compiled_module_* wrap the hub kernels in torch_compiler_disable_unless_decode
    # (Qwen3-Next on transformers 5.x); the gate must look through it to the hub closure.
    monkeypatch.delenv(ENV, raising = False)
    name = "fake_unsloth_compiled_module_gdn"

    def disable_unless_decode(func):
        @functools.wraps(func)
        def forward(*args, **kwargs):
            return func(*args, **kwargs)

        forward._unsloth_undisabled = func
        return forward

    try:
        model = _kwargs_model(name, _zoo_style_dispatch)
        modeling = sys.modules[name]
        modeling.torch_chunk_gated_delta_rule = disable_unless_decode(modeling.torch_chunk_gated_delta_rule)
        assert "implementation" in packing_module._hub_closure(modeling.torch_chunk_gated_delta_rule)
        assert patch_hybrid_linear_attention_varlen(model) is True
    finally:
        sys.modules.pop(name, None)
