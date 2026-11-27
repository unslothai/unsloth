# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A weight change re-merges only the quantised layers of the adapters it changes, bit-equal to a full re-merge."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")

import core.inference.diffusion_comfy_quant as cq  # noqa: E402
from core.inference import diffusion_lora_files as lf  # noqa: E402
from core.inference import diffusion_lora_quant_merge as qm  # noqa: E402
from core.inference.diffusion_convrot import _install_rotation, quantize_convrot_int8_weight  # noqa: E402

GROUP = 16


@pytest.fixture(autouse = True)
def _roomy_host(monkeypatch):
    from core.inference import diffusion_memory
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 1 << 20)
    monkeypatch.setattr(lf, "_refresh_group_offload", lambda roots: None)


class _Pipe:
    def __init__(self, net):
        self.transformer = net
        self.unloaded = 0

    @property
    def components(self):
        return {"transformer": self.transformer}

    def unload_lora_weights(self):
        self.unloaded += 1


INCREMENTAL_ENV = "UNSLOTH_DIFFUSION_LORA_INCREMENTAL_REMERGE"


class _TwoLayerNet(torch.nn.Module):
    """Two int8 ConvRot layers, each with its own LoKR: adapter "a" on q, "b" on k (and "c" on both)."""

    def __init__(self, shared = False):
        super().__init__()
        self.proj = torch.nn.ModuleDict()
        for i, name in enumerate(("q", "k")):
            torch.manual_seed(10 + i)
            dense = torch.randn(32, 64) * 0.05
            layer = torch.nn.Linear(64, 32, bias = False)
            codes, scale = quantize_convrot_int8_weight(dense, GROUP)
            layer.weight = torch.nn.Parameter(
                cq._int8_tensor(f"proj.{name}.weight", codes, scale, torch.bfloat16),
                requires_grad = False,
            )
            _install_rotation(layer, GROUP)
            self.proj[name] = layer
        self.pipe = _Pipe(self)
        reg = self.pipe.__dict__.setdefault(lf.KRON_ATTR, {})
        torch.manual_seed(3)
        for adapter, name in (("a", "q"), ("b", "k")) + (
            (("c", "q"), ("c", "k")) if shared else ()
        ):
            w1, w2 = torch.randn(4, 4) * 0.1, torch.randn(8, 16) * 0.1
            reg.setdefault(adapter, []).append(lf.KronAdapter(self, self.proj[name], w1, w2, 1.0))
            self.pipe.__dict__.setdefault(lf.KRON_ONLY_ATTR, set()).add(adapter)

    def codes(self):
        out = {}
        for name in ("q", "k"):
            w = self.proj[name].weight
            for k, t in qm._tensors(w, qm._weight_kind(w)).items():
                out[f"{name}.{k}"] = t.clone()
        return out


def _count_dequant(monkeypatch):
    calls = []
    real = qm._dequant

    def counted(module, *a, **k):
        calls.append(module)
        return real(module, *a, **k)

    monkeypatch.setattr(qm, "_dequant", counted)
    return calls


@pytest.mark.parametrize("shared", [False, True])
def test_a_weight_change_re_merges_only_the_layers_of_the_changed_adapters(monkeypatch, shared):
    monkeypatch.delenv(INCREMENTAL_ENV, raising = False)
    weights = {"a": 1.0, "b": 0.5, **({"c": 0.25} if shared else {})}
    net = _TwoLayerNet(shared)
    assert qm.merge(net.pipe, [net], weights, "cpu")
    k_layer = net.proj["k"]
    k_before = {k: t.clone() for k, t in qm._tensors(k_layer.weight, "int8").items()}
    calls = _count_dequant(monkeypatch)
    assert qm.set_weights(net.pipe, dict(weights, a = 0.3))
    assert calls == [net.proj["q"]]  # b's layer is neither restored nor re-quantised
    assert all(torch.equal(t, k_before[k]) for k, t in qm._tensors(k_layer.weight, "int8").items())
    incremental = net.codes()

    # reference: a full re-merge
    monkeypatch.setenv(INCREMENTAL_ENV, "0")
    ref = _TwoLayerNet(shared)
    assert qm.merge(ref.pipe, [ref], weights, "cpu")
    assert qm.set_weights(ref.pipe, dict(weights, a = 0.3))
    full = ref.codes()
    assert incremental.keys() == full.keys()
    assert all(torch.equal(incremental[k], full[k]) for k in full)


def test_a_change_to_a_shared_adapter_re_merges_every_layer_it_touches(monkeypatch):
    monkeypatch.delenv(INCREMENTAL_ENV, raising = False)
    net = _TwoLayerNet(shared = True)
    assert qm.merge(net.pipe, [net], {"a": 1.0, "b": 0.5, "c": 0.25}, "cpu")
    calls = _count_dequant(monkeypatch)
    assert qm.set_weights(net.pipe, {"a": 1.0, "b": 0.5})  # c omitted = 0
    assert calls == [net.proj["q"], net.proj["k"]]


def test_the_incremental_switch_re_merges_every_layer(monkeypatch):
    net = _TwoLayerNet()
    assert qm.merge(net.pipe, [net], {"a": 1.0, "b": 0.5}, "cpu")
    monkeypatch.setenv(INCREMENTAL_ENV, "0")
    calls = _count_dequant(monkeypatch)
    assert qm.set_weights(net.pipe, {"a": 0.3, "b": 0.5})
    assert calls == [net.proj["q"], net.proj["k"]]
    assert qm.INCREMENTAL_ENV == INCREMENTAL_ENV


def test_a_failed_incremental_re_merge_leaves_every_layer_at_its_base_codes(monkeypatch):
    monkeypatch.delenv(INCREMENTAL_ENV, raising = False)
    net = _TwoLayerNet()
    original = net.codes()
    assert qm.merge(net.pipe, [net], {"a": 1.0, "b": 0.5}, "cpu")

    def boom(*a, **k):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(qm, "_requant_", boom)
    with pytest.raises(RuntimeError):
        qm.set_weights(net.pipe, {"a": 0.3, "b": 0.5})
    assert all(torch.equal(net.codes()[k], original[k]) for k in original)
    monkeypatch.undo()
    monkeypatch.delenv(INCREMENTAL_ENV, raising = False)
    # the next change merges both again from the base codes: b too, though its weight did not move
    calls = _count_dequant(monkeypatch)
    assert qm.set_weights(net.pipe, {"a": 0.3, "b": 0.5})
    assert len(calls) == 2
