# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Adapters merged into a ComfyUI-quantized file's weights: int8 ConvRot, fp8 per row / per tensor, dense."""

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
    # The host's own free RAM can sit below the backup reserve (7 GB CI runners).
    from core.inference import diffusion_memory
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 1 << 20)
    # Resident merges have no group offloading to re-apply; the refresh imports diffusers, absent on CPU CI.
    monkeypatch.setattr(lf, "_refresh_group_offload", lambda roots: None)


class _Net(torch.nn.Module):
    def __init__(
        self,
        kind: str,
        out_features = 32,
        in_features = 64,
    ):
        super().__init__()
        torch.manual_seed(0)
        dense = torch.randn(out_features, in_features) * 0.05
        self.dense = dense
        self.proj = torch.nn.ModuleDict(
            {"q": torch.nn.Linear(in_features, out_features, bias = False)}
        )
        layer = self.proj["q"]
        if kind == "int8":
            codes, scale = quantize_convrot_int8_weight(dense, GROUP)
            weight = cq._int8_tensor("proj.q.weight", codes, scale, torch.bfloat16)
            _install_rotation(layer, GROUP)
        elif kind == "fp8":
            scale = dense.abs().amax() / 448.0
            codes = (dense / scale).to(torch.float8_e4m3fn)
            weight = cq._fp8_tensor("proj.q.weight", codes, scale.reshape(()), torch.bfloat16)
        else:
            weight = dense.to(torch.bfloat16)
        layer.weight = torch.nn.Parameter(weight, requires_grad = False)


class _Pipe:
    def __init__(self, net):
        self.transformer = net
        self.unloaded = 0

    @property
    def components(self):
        return {"transformer": self.transformer}

    def unload_lora_weights(self):
        self.unloaded += 1


def _lokr(
    net,
    pipe,
    name = "k",
    rows = None,
):
    torch.manual_seed(1)
    w1, w2 = torch.randn(4, 4) * 0.1, torch.randn(8, 16) * 0.1
    adapter = lf.KronAdapter(net, net.proj["q"], w1, w2, 1.0, rows)
    pipe.__dict__.setdefault(lf.KRON_ATTR, {})[name] = [adapter]
    pipe.__dict__.setdefault(lf.KRON_ONLY_ATTR, set()).add(name)
    return w1, w2


def _weight(net):
    return qm._dequant(
        net.proj["q"], qm._weight_kind(net.proj["q"].weight), qm._group(net.proj["q"]), "cpu"
    )


def _codes(net):
    w = net.proj["q"].weight
    kind = qm._weight_kind(w)
    return {k: t.clone() for k, t in qm._tensors(w, kind).items()}


@pytest.mark.parametrize("kind", ["int8", "fp8", "dense"])
def test_a_lokr_merges_into_the_weight_and_its_layer_is_removed(kind):
    net = _Net(kind)
    pipe = _Pipe(net)
    base = _weight(net)
    w1, w2 = _lokr(net, pipe)
    assert qm.merge(pipe, [net], {"k": 0.7}, "cpu")
    assert isinstance(net.proj["q"], torch.nn.Linear) and not isinstance(
        net.proj["q"], lf.kron_linear_class()
    )
    assert not pipe.__dict__.get(lf.KRON_ATTR) and pipe.unloaded == 1
    want = base + 0.7 * torch.kron(w1, w2)
    assert (_weight(net) - want).norm() / want.norm() < {"int8": 0.02, "fp8": 0.05, "dense": 0.01}[
        kind
    ]


@pytest.mark.parametrize("kind", ["int8", "fp8", "dense"])
def test_weight_zero_restores_the_original_codes_and_a_weight_merges_the_same_twice(kind):
    net = _Net(kind)
    pipe = _Pipe(net)
    original = _codes(net)
    _lokr(net, pipe)
    assert qm.merge(pipe, [net], {"k": 1.0}, "cpu")
    once = _codes(net)
    assert any(not torch.equal(once[k], original[k]) for k in once)
    assert qm.set_weights(pipe, {})
    assert all(torch.equal(_codes(net)[k], original[k]) for k in original)
    assert qm.set_weights(pipe, {"k": 1.0})
    assert all(torch.equal(_codes(net)[k], once[k]) for k in once)


def test_the_merge_keeps_each_layers_tensors_in_place():
    """Compiled code and captured graphs read the qdata / scale tensors: a weight change must write into them."""
    net = _Net("int8")
    pipe = _Pipe(net)
    _lokr(net, pipe)
    qdata, scale = net.proj["q"].weight.qdata, net.proj["q"].weight.scale
    ptrs = (qdata.data_ptr(), scale.data_ptr())
    assert qm.merge(pipe, [net], {"k": 1.0}, "cpu")
    assert qm.set_weights(pipe, {"k": 0.3})
    w = net.proj["q"].weight
    assert w.qdata is qdata and w.scale is scale and (qdata.data_ptr(), scale.data_ptr()) == ptrs


def test_a_row_slice_lokr_merges_only_its_rows():
    """MOSS-Image's fused qkv LoKR: each of q / k / v takes its own rows of the Kronecker product."""
    net = _Net("dense", out_features = 16)
    pipe = _Pipe(net)
    base = _weight(net)
    w1, w2 = _lokr(net, pipe, rows = (16, 32))
    assert qm.merge(pipe, [net], {"k": 1.0}, "cpu")
    want = base + torch.kron(w1, w2)[16:32]
    assert (_weight(net) - want).abs().max() <= want.abs().max() / 128


def test_an_adapter_outside_the_denoisers_keeps_the_adapter_layers():
    net = _Net("int8")
    pipe = _Pipe(net)
    _lokr(net, pipe)
    other = torch.nn.Linear(4, 4)
    assert not qm.merge(pipe, [other], {"k": 1.0}, "cpu")
    assert isinstance(net.proj["q"], lf.kron_linear_class())


def test_an_unsupported_weight_format_keeps_the_adapter_layers():
    net = _Net("dense")
    pipe = _Pipe(net)
    _lokr(net, pipe)
    net.proj["q"].get_base_layer().weight = torch.nn.Parameter(
        torch.zeros(32, 64, dtype = torch.int32), requires_grad = False
    )
    assert not qm.merge(pipe, [net], {"k": 1.0}, "cpu")
    assert isinstance(net.proj["q"], lf.kron_linear_class())


def test_the_kill_switch_keeps_the_adapter_layers(monkeypatch):
    monkeypatch.setenv(qm.MERGE_ENV, "0")
    net = _Net("int8")
    pipe = _Pipe(net)
    _lokr(net, pipe)
    assert not qm.merge(pipe, [net], {"k": 1.0}, "cpu")
    assert not qm.set_weights(pipe, {"k": 1.0})


def test_a_peft_lora_merges_at_its_alpha_scale():
    peft = pytest.importorskip("peft")
    net = _Net("int8")
    base = _weight(net)

    class _PeftPipe(_Pipe):
        def set_adapters(
            self,
            names,
            adapter_weights = None,
        ):
            for m in net.modules():
                if hasattr(m, "set_scale"):
                    for n, w in zip(names, adapter_weights):
                        m.set_scale(n, w)

        def unload_lora_weights(self):
            super().unload_lora_weights()
            from peft.tuners.tuners_utils import BaseTunerLayer
            for parent in list(net.modules()):
                for attr, child in list(parent.named_children()):
                    if isinstance(child, BaseTunerLayer):
                        setattr(parent, attr, child.get_base_layer())

    pipe = _PeftPipe(net)
    peft.inject_adapter_in_model(
        peft.LoraConfig(r = 2, lora_alpha = 4, target_modules = ["q"], init_lora_weights = False), net, "l"
    )
    layer = net.proj["q"]
    a, b = layer.lora_A["l"].weight.detach().clone(), layer.lora_B["l"].weight.detach().clone()
    assert qm.merge(pipe, [net], {"l": 0.5}, "cpu")
    assert not hasattr(net.proj["q"], "lora_A")
    want = base + 0.5 * 2.0 * (b @ a)
    assert (_weight(net) - want).norm() / want.norm() < 0.02


def test_a_wrapped_linear_keeps_its_adapter_layers():
    """A wrapper exposing ``weight`` as a computed view: a write there would never reach the forward."""
    net = _Net("dense")
    pipe = _Pipe(net)
    inner = net.proj["q"]

    class _Wrapper(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        @property
        def weight(self):
            return self.layer.weight.detach().clone()

        def forward(self, x):
            return self.layer(x)

    net.proj["q"] = _Wrapper(inner)
    adapter = lf.KronAdapter(net, net.proj["q"], torch.randn(4, 4), torch.randn(8, 16), 1.0)
    pipe.__dict__[lf.KRON_ATTR] = {"k": [adapter]}
    assert qm.plan(pipe, [net]) is None


def test_no_room_for_the_backups_keeps_the_adapter_layers(monkeypatch):
    from core.inference import diffusion_memory

    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 1024)
    net = _Net("int8")
    pipe = _Pipe(net)
    _lokr(net, pipe)
    assert not qm.merge(pipe, [net], {"k": 1.0}, "cpu")
    assert isinstance(net.proj["q"], lf.kron_linear_class())


def test_max_speed_never_fuses_qkv_over_merged_weights():
    """A fused to_qkv is a copy of to_q / to_k / to_v: a weight change re-merges those, and the copy would keep the old
    adapter weight (at weight 0 the adapter would still run)."""
    from core.inference import diffusion_speed as ds

    net = _Net("int8")
    calls = []

    class _FusingPipe(_Pipe):
        def fuse_qkv_projections(self):
            calls.append("pipe")

    net.fuse_qkv_projections = lambda: calls.append("dit")
    pipe = _FusingPipe(net)
    _lokr(net, pipe)
    assert qm.merge(pipe, [net], {"k": 1.0}, "cpu")
    assert ds._fuse_qkv(pipe, None) is False and calls == []


def test_a_failed_backup_allocation_keeps_the_adapter_layers(monkeypatch):
    net = _Net("int8")
    pipe = _Pipe(net)
    original = _codes(net)
    _lokr(net, pipe)
    real_to = torch.Tensor.to

    def to(self, *args, **kwargs):
        if args and args[0] == "cpu" and kwargs.get("copy"):
            raise RuntimeError("DefaultCPUAllocator: not enough memory")
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    assert not qm.merge(pipe, [net], {"k": 1.0}, "cpu")
    monkeypatch.undo()
    assert isinstance(net.proj["q"], lf.kron_linear_class()) and pipe.unloaded == 0
    assert all(torch.equal(_codes(net)[k], original[k]) for k in original)
    assert not qm.merged(pipe)


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

    # the reference: every layer restored and re-merged at the same weights
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
