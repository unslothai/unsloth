# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted INT8 ``.pt`` checkpoints on a torchao that deleted the v1 int8 classes (0.18+).

The hermetic tests pin the routing (when the rebuild engages, what planning is told). The file
tests write a checkpoint with the exact pickle layout torchao <= 0.17 produced and read it back
through the loader; they need a torchao where the rebuild applies and are skipped elsewhere.
"""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.diffusion_prequant as pq
import core.inference.prequant_legacy_int8 as li


# ---------------------------------------------------------------------------------------- routing


def test_only_the_deleted_int8_names_trigger_the_rebuild():
    refused = (
        "Weights only load failed ... Unsupported global: GLOBAL "
        "torchao.quantization.linear_activation_quantized_tensor.LinearActivationQuantizedTensor "
        "was not an allowed global by default."
    )
    assert li.names_legacy_int8_class(RuntimeError(refused))
    assert not li.names_legacy_int8_class(RuntimeError("Unsupported global: GLOBAL os.system"))


def test_a_torchao_that_still_ships_the_classes_keeps_the_old_path(monkeypatch):
    monkeypatch.setattr(li, "_resolve", lambda name: object())
    assert not li.legacy_int8_classes_missing()


def test_int8_support_is_answered_by_the_rebuild_when_the_classes_are_gone(monkeypatch):
    """torchao 0.18: the int8 constructors cannot be allowlisted, so without the rebuild planning
    is told no, the hosted checkpoint is skipped and the dense bf16 transformer is downloaded."""
    monkeypatch.setattr(pq, "_register_prequant_safe_globals", lambda: True)
    monkeypatch.setattr(pq, "_RESOLVED_SAFE_GLOBALS", set(pq._SCHEME_REQUIRED_GLOBALS["fp8"]))
    monkeypatch.setattr(li, "legacy_int8_decode_supported", lambda: False)
    assert not pq.restricted_prequant_load_supported("int8", "X-INT8.pt")
    monkeypatch.setattr(li, "legacy_int8_decode_supported", lambda: True)
    assert pq.restricted_prequant_load_supported("int8", "X-INT8.pt")
    assert pq.restricted_prequant_load_supported("int8")
    # fp8 never depended on it, and an unknown scheme gets no free pass.
    assert pq.restricted_prequant_load_supported("fp8", "X-FP8.pt")
    monkeypatch.setattr(pq, "_RESOLVED_SAFE_GLOBALS", {"torch.torch_version.TorchVersion"})
    assert not pq.restricted_prequant_load_supported("fp8", "X-FP8.pt")


def test_a_refusal_for_any_other_global_is_not_retried(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    def _load(*a, **k):
        raise RuntimeError("Unsupported global: GLOBAL os.system was not an allowed global")

    monkeypatch.setattr(pq, "_register_prequant_safe_globals", lambda: True)
    monkeypatch.setattr(torch, "load", _load)
    monkeypatch.setattr(li, "legacy_int8_decode_supported", lambda: True)
    monkeypatch.setattr(li, "load_legacy_int8_pickle", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match = "os.system"):
        pq._torch_load_prequant("/x.pt", map_location = "cpu")
    assert calls == []


def test_unreadable_hosted_checkpoint_is_named_for_status(monkeypatch):
    from core.inference.diffusion_families import DiffusionFamily

    fam = DiffusionFamily(
        name = "test-fam",
        base_repo = "org/base",
        pipeline_class = "P",
        transformer_class = "T",
        prequant_repos = (("int8", "org/Model-FP8"),),
    )
    monkeypatch.setattr(pq, "cached_checkpoint_path", lambda *a, **k: None)
    monkeypatch.setattr(pq, "restricted_prequant_load_supported", lambda scheme, name = None: False)
    monkeypatch.setattr(
        pq, "_unreadable_why", lambda scheme: "torchao 9.9 no longer ships 5 class(es)"
    )
    note = pq.prequant_unreadable_reason(fam, "int8")
    assert "org/Model-FP8" in note and "no longer ships" in note
    # A derived safetensors name alone is a guess, not a readable artifact.
    monkeypatch.setattr(
        pq,
        "restricted_prequant_load_supported",
        lambda scheme, name = None: bool(name) and name.endswith(".safetensors"),
    )
    assert pq.prequant_unreadable_reason(fam, "int8") is not None
    monkeypatch.setattr(pq, "restricted_prequant_load_supported", lambda scheme, name = None: True)
    assert pq.prequant_unreadable_reason(fam, "int8") is None
    assert pq.prequant_unreadable_reason(fam, "fp8") is None  # nothing hosted for it
    assert pq.prequant_unreadable_reason(fam, None) is None


def test_the_dense_fast_path_reason_says_why_the_hosted_checkpoint_was_skipped(monkeypatch):
    import core.inference.diffusion as d

    monkeypatch.setattr(
        d,
        "prequant_unreadable_reason",
        lambda *a, **k: "the hosted int8 checkpoint (r) cannot be read here: x",
    )
    reason = d._dense_fast_path_reason(object(), "int8", "org/base", "pipeline", None)
    assert "cannot be read here" in reason and "dense bf16 transformer" in reason
    # A local override or a GGUF pick never had a hosted checkpoint to skip.
    assert (
        d._dense_fast_path_reason(object(), "int8", "org/base", "pipeline", "/p.pt")
        == "engaged on the dense fast path"
    )
    assert (
        d._dense_fast_path_reason(object(), "int8", "org/base", "gguf", None)
        == "engaged on the dense fast path"
    )
    # A LoRA bake always takes the dense transformer, readable checkpoint or not.
    assert (
        d._dense_fast_path_reason(
            object(), "int8", "org/base", "pipeline", None, [("org/lora", 1.0)]
        )
        == "engaged on the dense fast path"
    )


# ------------------------------------------------------------------------------------ real files


_needs_rebuild = pytest.mark.skipif(
    not li.legacy_int8_decode_supported(),
    reason = "needs a torchao without the v1 int8 classes and with Int8Tensor (0.18+)",
)


@pytest.fixture(autouse = True)
def _no_foreign_legacy_registrations():
    """Other test files register placeholder objects under these names process-wide; a real 0.18
    install has nothing there, which is the situation under test."""
    torch = pytest.importorskip("torch")
    saved = torch.serialization.get_safe_globals()
    legacy = {*li.LEGACY_INT8_CLASS_NAMES, li._ACT_QUANT}
    torch.serialization.clear_safe_globals()
    torch.serialization.add_safe_globals(
        [g for g in saved if not (isinstance(g, tuple) and g[1] in legacy)]
    )
    yield
    torch.serialization.clear_safe_globals()
    torch.serialization.add_safe_globals(saved)


def _legacy_modules():
    """Classes and a function under the module paths torchao <= 0.17 pickled them from."""
    import enum

    import torch

    class _Base(torch.Tensor):
        __torch_function__ = torch._C._disabled_torch_function_impl

        @staticmethod
        def __new__(cls, shape, dtype):
            return torch.Tensor._make_wrapper_subclass(cls, shape, dtype = dtype)

        @classmethod
        def __torch_dispatch__(
            cls,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            raise RuntimeError("fake")

        def __reduce_ex__(self, proto):
            args = (
                type(self),
                self.dtype,
                tuple(self.size()),
                self.stride(),
                self.storage_offset(),
                self.layout,
                self.device,
                False,
            )
            return (
                torch._tensor._rebuild_from_type_v2,
                (torch._utils._rebuild_wrapper_subclass, type(self), args, dict(self.__dict__)),
            )

    spec = {
        "torchao.quantization.linear_activation_quantized_tensor": "LinearActivationQuantizedTensor",
        "torchao.dtypes.affine_quantized_tensor": "AffineQuantizedTensor",
        "torchao.dtypes.uintx.plain_layout": "PlainAQTTensorImpl",
    }
    mods, made = {}, {}
    for mod, name in spec.items():
        cls = type(name, (_Base,), {"__module__": mod, "__qualname__": name})
        m = types.ModuleType(mod)
        setattr(m, name, cls)
        mods[mod] = m
        made[name] = cls
    layout = type(
        "PlainLayout", (), {"__module__": "torchao.dtypes.utils", "__qualname__": "PlainLayout"}
    )
    mods["torchao.dtypes.utils"] = types.ModuleType("torchao.dtypes.utils")
    mods["torchao.dtypes.utils"].PlainLayout = layout

    def _int8_symm_per_token_reduced_range_quant(x):
        raise RuntimeError("never called")

    _int8_symm_per_token_reduced_range_quant.__module__ = "torchao.quantization.quant_api"
    _int8_symm_per_token_reduced_range_quant.__qualname__ = (
        "_int8_symm_per_token_reduced_range_quant"
    )
    real_api = sys.modules.get("torchao.quantization.quant_api")
    api = types.ModuleType("torchao.quantization.quant_api")
    if real_api is not None:
        api.__dict__.update(real_api.__dict__)
    api._int8_symm_per_token_reduced_range_quant = _int8_symm_per_token_reduced_range_quant
    mods["torchao.quantization.quant_api"] = api

    class ZeroPointDomain(enum.Enum):
        INT = 1
        FLOAT = 2
        NONE = 3

    return mods, made, layout, _int8_symm_per_token_reduced_range_quant, ZeroPointDomain


def _write_legacy_checkpoint(
    monkeypatch,
    path,
    *,
    zero_point = None,
    extra = None,
):
    import torch

    from torchao.quantization.quant_primitives import ZeroPointDomain

    mods, made, layout, act, _ = _legacy_modules()
    torch.manual_seed(0)
    n, k = 8, 16
    qdata = torch.randint(-128, 128, (n, k), dtype = torch.int8)
    scale = (torch.rand(n) / 50 + 1e-3).to(torch.bfloat16)
    with monkeypatch.context() as m:
        for name, mod in mods.items():
            m.setitem(sys.modules, name, mod)
        impl = made["PlainAQTTensorImpl"]((n, k), torch.int8)
        impl.__dict__.update(int_data = qdata, scale = scale, zero_point = zero_point, _layout = layout())
        aqt = made["AffineQuantizedTensor"]((n, k), torch.bfloat16)
        aqt.__dict__.update(
            tensor_impl = impl,
            block_size = (1, k),
            quant_min = None,
            quant_max = None,
            zero_point_domain = ZeroPointDomain.NONE,
        )
        laqt = made["LinearActivationQuantizedTensor"]((n, k), torch.bfloat16)
        laqt.__dict__.update(original_weight_tensor = aqt, input_quant_func = act, quant_kwargs = {})
        state_dict = {
            "blk.proj.weight": laqt,
            "blk.norm.weight": torch.ones(k, dtype = torch.bfloat16),
        }
        if extra:
            state_dict.update(extra)
        torch.save(
            {
                "format": pq.PREQUANT_FORMAT,
                "metadata": {"scheme": "int8"},
                "state_dict": state_dict,
            },
            str(path),
        )
    return qdata, scale


@_needs_rebuild
def test_a_legacy_int8_pickle_loads_as_int8tensor_with_the_same_weights(monkeypatch, tmp_path):
    import torch

    path = tmp_path / "Model-INT8.pt"
    qdata, scale = _write_legacy_checkpoint(monkeypatch, path)
    # The plain load is what 0.18 refuses; the loader must not.
    with pytest.raises(
        Exception, match = "LinearActivationQuantizedTensor|was not an allowed global"
    ):
        torch.load(str(path), weights_only = True)
    ckpt = pq._load_prequant_checkpoint(str(path), map_location = "cpu")
    w = ckpt["state_dict"]["blk.proj.weight"]
    assert type(w).__name__ == "Int8Tensor"
    assert torch.equal(w.qdata, qdata)
    assert w.scale.dtype == torch.bfloat16 and torch.equal(w.scale.flatten(), scale)
    assert w.dtype == torch.bfloat16 and tuple(w.shape) == (8, 16)
    assert type(w.act_quant_kwargs.granularity).__name__ == "PerRow"
    assert torch.equal(w.dequantize(), qdata.to(torch.bfloat16) * scale.reshape(8, 1))
    assert type(ckpt["state_dict"]["blk.norm.weight"]) is torch.Tensor
    # The stand-ins were registered for that load only.
    assert not any(
        isinstance(g, tuple) and g[1] == li._LAQT for g in torch.serialization.get_safe_globals()
    )


@_needs_rebuild
def test_the_rebuild_does_not_open_the_door_to_other_globals(monkeypatch, tmp_path):
    import os

    import torch

    path = tmp_path / "Model-INT8.pt"
    _write_legacy_checkpoint(monkeypatch, path, extra = {"zz": os.getcwd})
    with pytest.raises(Exception, match = "getcwd|posix|was not an allowed global"):
        pq._load_prequant_checkpoint(str(path), map_location = "cpu")
    del torch


@_needs_rebuild
def test_an_asymmetric_legacy_weight_is_refused(monkeypatch, tmp_path):
    import torch

    path = tmp_path / "Model-INT8.pt"
    _write_legacy_checkpoint(monkeypatch, path, zero_point = torch.ones(8, dtype = torch.int8))
    with pytest.raises(ValueError, match = "zero point"):
        pq._load_prequant_checkpoint(str(path), map_location = "cpu")


@_needs_rebuild
def test_a_plain_load_that_overlaps_a_legacy_one_is_rebuilt_too(monkeypatch, tmp_path):
    """While one thread's legacy load has the stand-ins registered, another thread's plain
    weights_only load of an int8 pickle succeeds against them. It must not hand inert stand-ins
    to load_state_dict."""
    import torch

    path = tmp_path / "Model-INT8.pt"
    qdata, _ = _write_legacy_checkpoint(monkeypatch, path)
    standins = li._standins()
    pairs = [(obj, name) for name, obj in standins.items() if li._resolve(name) is None]
    with torch.serialization.safe_globals(pairs):
        raw = torch.load(str(path), weights_only = True)
    assert isinstance(raw["state_dict"]["blk.proj.weight"], standins[li._LAQT])
    monkeypatch.setattr(torch, "load", lambda *a, **k: raw)
    monkeypatch.setattr(pq, "_register_prequant_safe_globals", lambda: True)
    w = pq._torch_load_prequant(str(path), map_location = "cpu")["state_dict"]["blk.proj.weight"]
    assert type(w).__name__ == "Int8Tensor" and torch.equal(w.qdata, qdata)
