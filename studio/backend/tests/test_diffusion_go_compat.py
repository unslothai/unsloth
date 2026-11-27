# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's group-offload helpers follow whichever wrapper-tensor API the installed diffusers has: the torchao-only
helpers of diffusers <= 0.42, or the generic wrapper-subclass ones of huggingface/diffusers#15383."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")
go = pytest.importorskip("diffusers.hooks.group_offloading")

from core.inference import diffusion_go_compat as go_compat  # noqa: E402
from core.inference import diffusion_memory as mem  # noqa: E402
from core.inference import diffusion_prequant as pq  # noqa: E402

two = pytest.importorskip("torch.testing._internal.two_tensor")


def _int8_linear():
    quant = pytest.importorskip("torchao.quantization")
    config = getattr(quant, "Int8WeightOnlyConfig", None)
    if config is None:
        pytest.skip("this torchao has no Int8WeightOnlyConfig")
    m = torch.nn.Sequential(torch.nn.Linear(32, 32, bias = False)).to(torch.bfloat16)
    quant.quantize_(m, config())
    names = getattr(m[0].weight, "__tensor_flatten__", lambda: ([], None))()[0]
    if not names:
        pytest.skip("this torchao's int8 weight is not a wrapper subclass")
    return m, names


def _legacy_go():
    calls = []
    ns = types.SimpleNamespace(
        _is_torchao_tensor = lambda t: isinstance(t, two.TwoTensor),
        _swap_torchao_tensor = lambda p, s: calls.append(("swap", p, s)),
        _restore_torchao_tensor = lambda p, s: calls.append(("restore", p, s)),
        _record_stream_torchao_tensor = lambda p, st: calls.append(("record", p, st)),
    )
    return ns, calls


def _generic_go():
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    def _apply(tensor, fn):
        names, ctx = tensor.__tensor_flatten__()
        inner = {n: fn(getattr(tensor, n)) for n in names}
        return type(tensor).__tensor_unflatten__(inner, ctx, tensor.shape, tensor.stride())

    recorded = []
    ns = types.SimpleNamespace(
        is_traceable_wrapper_subclass = is_traceable_wrapper_subclass,
        _apply_to_inner_tensors = _apply,
        _swap_wrapper_tensor = torch.utils.swap_tensors,
        _record_stream_wrapper_tensor = lambda p, st: recorded.append((p, st)),
    )
    return ns, recorded


def test_installed_diffusers_resolves_one_api():
    """Every diffusers Studio supports has one of the two APIs; a third shape must fail here, not silently."""
    assert go_compat.is_legacy(go) != go_compat.is_generic(go)
    assert callable(go_compat.is_wrapper_fn(go)) and callable(go_compat.swap_fn(go))
    assert callable(go_compat.restore_fn(go)) and callable(go_compat.record_stream_fn(go))


def test_legacy_names_are_returned_untouched():
    ns, calls = _legacy_go()
    assert go_compat.is_wrapper_fn(ns) is ns._is_torchao_tensor
    assert go_compat.swap_fn(ns) is ns._swap_torchao_tensor
    assert go_compat.restore_fn(ns) is ns._restore_torchao_tensor
    assert go_compat.record_stream_fn(ns) is ns._record_stream_torchao_tensor
    assert go_compat.swap_name(ns) == "_swap_torchao_tensor"
    assert go_compat.restore_name(ns) == "_restore_torchao_tensor"
    assert go_compat.record_name(ns) == "_record_stream_torchao_tensor"
    # legacy wins even when the generic names also exist
    ns._swap_wrapper_tensor = lambda p, s: None
    assert go_compat.swap_name(ns) == "_swap_torchao_tensor"


def test_generic_api_moves_any_wrapper_subclass():
    ns, recorded = _generic_go()
    is_wrapper = go_compat.is_wrapper_fn(ns)
    w = torch.randn(4, 4)
    t = two.TwoTensor(w, w.clone())
    assert is_wrapper(t) and not is_wrapper(w) and not is_wrapper(torch.nn.Parameter(w))
    assert go_compat.swap_name(ns) == "_swap_wrapper_tensor"
    assert go_compat.restore_name(ns) is None  # inlined in diffusers after #15383
    go_compat.record_stream(t, "s", ns)
    assert recorded == [(t, "s")]


def test_generic_restore_points_param_at_the_host_copy_without_mutating_it():
    ns, _ = _generic_go()
    host_a, host_b = torch.ones(4), torch.full((4,), 2.0)
    host = two.TwoTensor(host_a, host_b)
    param = two.TwoTensor(torch.zeros(4), torch.zeros(4))
    go_compat.restore(param, host, ns)
    assert param.a is host_a and param.b is host_b
    assert host.a is host_a and host.b is host_b


def test_no_group_offloading_api_resolves_nothing():
    ns = types.SimpleNamespace()
    assert go_compat.is_wrapper_fn(ns) is None and go_compat.swap_fn(ns) is None
    assert go_compat.is_wrapper_or_false(ns)(two.TwoTensor(torch.ones(1), torch.ones(1))) is False
    assert not go_compat.wrapper_swap_supported(ns)
    with pytest.raises(AttributeError):
        go_compat.swap(None, None, ns)


def test_torchao_streaming_capability_holds_on_the_installed_diffusers():
    """diffusers main dropped _is_torchao_tensor: the old probe turned streamed INT8 / prequant off there."""
    assert pq.torchao_group_offload_supported() is True


def test_capability_probe_accepts_the_generic_api(monkeypatch):
    ns, _ = _generic_go()
    hooks = types.ModuleType("diffusers.hooks")
    hooks.apply_group_offloading = lambda *a, **k: None
    fake = types.ModuleType("diffusers.hooks.group_offloading")
    for k, v in vars(ns).items():
        setattr(fake, k, v)
    hooks.group_offloading = fake
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.group_offloading", fake)
    assert pq.torchao_group_offload_supported() is True


def test_return_module_tensors_moves_torchao_inner_tensors():
    """On diffusers main the old code took the .data path for an Int8Tensor and set_data raised."""
    m, names = _int8_linear()
    assert go_compat.is_wrapper_or_false(go)(m[0].weight)
    mem._return_module_tensors(m, torch.device("meta"))
    w = m[0].weight
    assert w.device.type == "meta"
    assert all(getattr(w, n).device.type == "meta" for n in names)


def test_swap_retry_wraps_the_helper_diffusers_calls(monkeypatch):
    name = go_compat.swap_name(go)
    assert name is not None
    monkeypatch.setattr(go, name, getattr(go, name))
    assert mem.install_group_offload_torchao_swap_retry() is True
    assert getattr(getattr(go, name), "_unsloth_swap_retry", False)
    assert mem.install_group_offload_torchao_swap_retry() is False


def test_swap_retry_on_generic_api_retries_a_weakref_refusal(monkeypatch):
    ns, _ = _generic_go()
    attempts = []

    def _swap(p, s):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("Cannot swap t1 because it has weakref associated with it")

    ns._swap_wrapper_tensor = _swap
    hooks = types.ModuleType("diffusers.hooks")
    fake = types.ModuleType("diffusers.hooks.group_offloading")
    for k, v in vars(ns).items():
        setattr(fake, k, v)
    hooks.group_offloading = fake
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.group_offloading", fake)
    assert mem.install_group_offload_torchao_swap_retry() is True
    fake._swap_wrapper_tensor("p", "s")
    assert len(attempts) == 2
    assert not hasattr(fake, "_swap_torchao_tensor")


def _fake_v1_torchao(monkeypatch):
    """Stand-ins for torchao <= 0.17's v1 int8 classes (the real ones need torchao 0.17)."""

    def _cls(name):
        return type(
            name,
            (),
            {"_ATEN_OP_TABLE": {}, "implements": classmethod(lambda cls, ops: (lambda fn: fn))},
        )

    aqt_mod = types.ModuleType("torchao.dtypes.affine_quantized_tensor")
    aqt_mod.AffineQuantizedTensor = _cls("AffineQuantizedTensor")
    laqt_mod = types.ModuleType("torchao.quantization.linear_activation_quantized_tensor")
    laqt_mod.LinearActivationQuantizedTensor = _cls("LinearActivationQuantizedTensor")
    dtypes = types.ModuleType("torchao.dtypes")
    dtypes.affine_quantized_tensor = aqt_mod
    monkeypatch.setitem(sys.modules, "torchao.dtypes", dtypes)
    monkeypatch.setitem(sys.modules, "torchao.dtypes.affine_quantized_tensor", aqt_mod)
    monkeypatch.setitem(
        sys.modules, "torchao.quantization.linear_activation_quantized_tensor", laqt_mod
    )


def test_v1_int8_pin_ops_install_on_the_installed_diffusers(monkeypatch):
    """diffusers main has no _restore_torchao_tensor to wrap (it rebuilds nested v1 wrappers itself): the pin ops
    still install, so torchao 0.17 v1 int8 keeps the copy stream instead of the synchronous fallback."""
    pytest.importorskip("torchao.quantization")
    _fake_v1_torchao(monkeypatch)
    monkeypatch.setattr(mem, "_V1_INT8_PIN_OPS_INSTALLED", False)
    monkeypatch.delenv(mem.INT8_STREAM_TORCHAO17_ENV, raising = False)
    for name in (go_compat.LEGACY_RESTORE, go_compat.LEGACY_RECORD):
        if hasattr(go, name):
            monkeypatch.setattr(go, name, getattr(go, name))
    assert mem.install_torchao_v1_int8_pin_ops() is True
    if go_compat.is_legacy(go):
        assert getattr(go._restore_torchao_tensor, "_unsloth_v1_int8", False)
        assert getattr(go._record_stream_torchao_tensor, "_unsloth_v1_int8", False)
    else:
        assert not hasattr(go, go_compat.LEGACY_RESTORE)
        assert not hasattr(go, go_compat.LEGACY_RECORD)
