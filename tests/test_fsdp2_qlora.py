# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP2 + bitsandbytes 4-bit (QLoRA): GPU-free checks of unsloth/models/fsdp2_qlora.py.

The module is loaded from its file (importing unsloth needs a GPU). The distributed tests shard real
bitsandbytes Linear4bit layers (CPU backend) across 2 gloo ranks with FSDP2's fully_shard.
"""

import importlib.util
import os
import socket
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = Path(
    os.environ.get("UNSLOTH_FSDP2_QLORA_PATH", _ROOT / "unsloth" / "models" / "fsdp2_qlora.py")
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_unsloth_fsdp2_qlora_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bnb = pytest.importorskip("bitsandbytes")
fully_shard_mod = pytest.importorskip("torch.distributed.fsdp")
if not hasattr(fully_shard_mod, "fully_shard"):
    pytest.skip("torch has no FSDP2 fully_shard", allow_module_level = True)


def _linear4bit(in_features, out_features, storage, seed):
    torch.manual_seed(seed)
    lin = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias = False,
        compute_dtype = torch.bfloat16,
        quant_type = "nf4",
        quant_storage = storage,
    )
    lin.weight = bnb.nn.Params4bit(
        torch.randn(out_features, in_features) * 0.05,
        requires_grad = False,
        quant_type = "nf4",
        quant_storage = storage,
        blocksize = 64,
        module = lin,
    )
    return lin.to("cpu")


class _Block(torch.nn.Module):
    def __init__(self, storage, seed):
        super().__init__()
        self.up = _linear4bit(128, 256, storage, seed)
        self.down = _linear4bit(256, 128, storage, seed + 1)
        torch.manual_seed(seed + 2)
        # Trainable fp32 adapter, like LoRA's A matrix.
        self.adapter = torch.nn.Linear(128, 8, bias = False)
        # Zero, so the sharded output can be compared bit for bit; its gradient is still nonzero.
        torch.nn.init.zeros_(self.adapter.weight)

    def forward(self, x):
        h = self.down(torch.nn.functional.silu(self.up(x)))
        return h + self.adapter(x.to(self.adapter.weight.dtype)).sum(-1, keepdim = True).to(h.dtype)


class _Model(torch.nn.Module):
    def __init__(self, storage):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(storage, 10 * i) for i in range(2)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_quant_storage_dtype():
    m = _load_module()
    assert m.fsdp2_quant_storage_dtype(torch.float16) is torch.float16
    assert m.fsdp2_quant_storage_dtype("bfloat16") is torch.bfloat16
    assert m.fsdp2_quant_storage_dtype(torch.uint8) is None
    assert m.fsdp2_quant_storage_dtype(None) is None


def test_set_quant_storage_on_bnb_config():
    m = _load_module()
    from transformers import BitsAndBytesConfig

    cfg = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = torch.float16)
    assert m.set_fsdp2_quant_storage(cfg, torch.float16)
    assert cfg.bnb_4bit_quant_storage == torch.float16
    # A caller's explicit float storage stays.
    cfg = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_quant_storage = torch.float32)
    assert not m.set_fsdp2_quant_storage(cfg, torch.bfloat16)
    assert cfg.bnb_4bit_quant_storage == torch.float32
    # 8-bit has no packed 4-bit storage.
    cfg = BitsAndBytesConfig(load_in_8bit = True)
    assert not m.set_fsdp2_quant_storage(cfg, torch.bfloat16)
    assert not m.set_fsdp2_quant_storage(None, torch.bfloat16)


@pytest.mark.parametrize("storage", [torch.bfloat16, torch.float16])
def test_uint8_to_float_storage_is_a_view(storage):
    m = _load_module()
    lin = _linear4bit(128, 64, torch.uint8, 0)
    weight = lin.weight
    before = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
    raw = weight.data.clone()
    model = torch.nn.Sequential(lin)
    assert m.params4bit_to_float_storage(model, storage) == 1
    assert lin.weight is weight
    assert weight.dtype == storage and weight.quant_storage == storage
    assert lin.quant_storage == storage
    assert weight.shape == (raw.numel() // 2, 1)
    assert torch.equal(weight.data.view(torch.uint8).reshape(-1), raw.reshape(-1))
    after = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
    assert torch.equal(before, after)
    x = torch.randn(3, 128, dtype = torch.bfloat16)
    torch.testing.assert_close(lin(x), x @ before.t().to(x.dtype), rtol = 0, atol = 0)
    # Idempotent; float storage is left alone.
    assert m.params4bit_to_float_storage(model, storage) == 0
    assert m.params4bit_to_float_storage(model, torch.uint8) == 0


def _prepare_that_upcasts(accelerator, model):
    # accelerate < 1.13 ends its FSDP2 wrap like this when mixed precision is on.
    model = model.to(torch.float32)
    return model


def _prepare_that_keeps_dtypes(accelerator, model):
    return model


def test_float_storage_only_when_the_wrap_does_not_upcast():
    m = _load_module()
    assert m.float_storage_survives_wrap(_prepare_that_upcasts) is False
    assert m.float_storage_survives_wrap(_prepare_that_keeps_dtypes) is True
    # The installed accelerate's own function is read through any wrapper.
    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

    assert m.float_storage_survives_wrap() is m.float_storage_survives_wrap(fsdp2_prepare_model)


def test_upcasting_wrap_keeps_uint8_and_guards_parameter_creation():
    m = _load_module()
    lin = _linear4bit(128, 64, torch.uint8, 0)
    model = torch.nn.Sequential(lin)
    with m.fsdp2_qlora_wrap(model, _prepare_that_upcasts) as info:
        assert info["params4bit"] == 1 and info["converted"] == 0
        assert info["storage"] == ["uint8"]
        # torch < 2.12 builds the sharded uint8 parameter with requires_grad = True first.
        assert torch.nn.Parameter(torch.zeros(2, dtype = torch.uint8)).requires_grad is False
        # What accelerate < 1.13 then does is harmless to uint8 bytes.
        before = lin.weight.data.clone()
        model.to(torch.float32)
        assert lin.weight.dtype == torch.uint8 and torch.equal(lin.weight.data, before)
    with pytest.raises(RuntimeError):
        torch.nn.Parameter(torch.zeros(2, dtype = torch.uint8))


def test_wrap_is_a_noop_without_4bit_weights():
    m = _load_module()
    model = torch.nn.Linear(4, 4)
    with m.fsdp2_qlora_wrap(model, _prepare_that_keeps_dtypes) as info:
        assert info is None
    assert m.prepare_fsdp2_qlora(model)["params4bit"] == 0


def test_anchor_check_refuses_unknown_fsdp_param():
    m = _load_module()

    class _Old:
        def _init_sharded_param(self, tensor, device):
            pass

        def init_unsharded_param(self):
            self._unsharded_param = None

    missing = m._check_fsdp_param_anchors(_Old)
    assert "FSDPParam.init_dtype_attrs" in missing

    class _Renamed(_Old):
        def init_dtype_attrs(self, mp_policy):
            self.param_dtype = None

    assert m._check_fsdp_param_anchors(_Renamed) == [
        "FSDPParam._init_sharded_param(self, param, ...)"
    ]


def test_anchors_present_on_installed_torch():
    m = _load_module()
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    assert m._check_fsdp_param_anchors(FSDPParam) == []


def test_nonfloat_guard_restores_parameter_new():
    m = _load_module()
    before = torch.nn.Parameter.__dict__.get("__new__")
    with m.nonfloat_param_guard():
        p = torch.nn.Parameter(torch.zeros(4, dtype = torch.uint8))
        assert p.requires_grad is False
        assert torch.nn.Parameter(torch.zeros(4)).requires_grad is True
    assert torch.nn.Parameter.__dict__.get("__new__") is before
    with pytest.raises(RuntimeError):
        torch.nn.Parameter(torch.zeros(4, dtype = torch.uint8))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world_size, port, case, queue):
    try:
        os.environ.update(
            MASTER_ADDR = "127.0.0.1",
            MASTER_PORT = str(port),
            RANK = str(rank),
            WORLD_SIZE = str(world_size),
        )
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        dist.init_process_group("gloo", rank = rank, world_size = world_size)
        m = _load_module()
        storage = torch.uint8 if case.startswith("uint8") else torch.bfloat16
        model = _Model(storage)
        reference = _Model(storage)
        if case.endswith("uniform"):
            # Every all-gather input of a block in one float dtype: FSDP2's foreach-copy path.
            for net in (model, reference):
                for layer in net.layers:
                    layer.adapter.to(torch.bfloat16)
        torch.manual_seed(1234)
        x = torch.randn(4, 128, dtype = torch.bfloat16)
        expected = reference(x).detach()

        if case.startswith("unpatched"):
            info = {"patched": False}
        else:
            info = m.prepare_fsdp2_qlora(
                model, torch.bfloat16 if case == "uint8_converted" else None
            )
            if case == "uint8_converted":
                assert info["converted"] == 4, info
        # The dtype each all-gather moves its inputs in, as foreach_all_gather decided it.
        from torch.distributed.fsdp._fully_shard import _fsdp_collectives

        gather_dtypes = []
        decide = _fsdp_collectives._get_all_gather_input_metadatas

        def _get_all_gather_input_metadatas(*args, **kwargs):
            out = decide(*args, **kwargs)
            gather_dtypes.append(str(out[2]))
            return out

        _fsdp_collectives._get_all_gather_input_metadatas = _get_all_gather_input_metadatas
        packed_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters() if m.is_params4bit(p)
        )
        mesh = init_device_mesh("cpu", (world_size,))
        if case.endswith("uniform"):
            mp = MixedPrecisionPolicy()
        else:
            # fp16 all-gather: on bf16 storage this cast would scramble the packed bytes.
            mp = MixedPrecisionPolicy(
                param_dtype = torch.float16, reduce_dtype = torch.float32, cast_forward_inputs = False
            )
        for layer in model.layers:
            fully_shard(layer, mesh = mesh, mp_policy = mp)
        fully_shard(model, mesh = mesh, mp_policy = mp)

        local_bytes = 0
        for name, p in model.named_parameters():
            if "adapter" in name:
                continue
            local = p.to_local()
            local_bytes += local.numel() * local.element_size()
        seen = {}

        def hook(module, args):
            w = module.weight
            seen[id(module)] = (getattr(w, "quant_state", None) is not None, w.dtype)

        for layer in model.layers:
            layer.up.register_forward_pre_hook(hook)
            layer.down.register_forward_pre_hook(hook)
        out = model(x)
        out.float().sum().backward()
        grad_ok = all(p.grad is not None for n, p in model.named_parameters() if "adapter" in n)
        sharded_has_qs = all(
            getattr(p, "quant_state", None) is not None
            for n, p in model.named_parameters()
            if "adapter" not in n
        )
        result = dict(
            info = {k: v for k, v in info.items() if k != "reason"},
            max_diff = (out.float() - expected.float()).abs().max().item(),
            packed_bytes = packed_bytes,
            local_bytes = local_bytes,
            weight_quant_state = [v[0] for v in seen.values()],
            grad_ok = grad_ok,
            sharded_has_qs = sharded_has_qs,
            gather_dtypes = sorted(set(gather_dtypes)),
        )
        queue.put((rank, result))
        dist.destroy_process_group()
    except BaseException as error:
        import traceback
        queue.put((rank, {"error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"}))


def _old_torch_worker(rank, world_size, port, case, queue):
    """torch < 2.8 (forced): uint8 weights must stay out of every FSDP2 unit, whose trainable float
    adapter would otherwise share it with them (`FSDP expects uniform original parameter dtype`)."""
    try:
        os.environ.update(
            MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port), RANK = str(rank), WORLD_SIZE = str(world_size)
        )
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.tensor import DTensor

        dist.init_process_group("gloo", rank = rank, world_size = world_size)
        m = _load_module()
        m._fsdp2_checks_frozen_dtypes = lambda: True
        model, reference = _Model(torch.uint8), _Model(torch.uint8)
        torch.manual_seed(1234)
        x = torch.randn(4, 128, dtype = torch.bfloat16)
        expected = reference(x).detach()
        mesh = init_device_mesh("cpu", (world_size,))
        with m.fsdp2_qlora_wrap(model, _prepare_that_keeps_dtypes) as info:
            for layer in model.layers:
                fully_shard(layer, mesh = mesh)
            fully_shard(model, mesh = mesh)
        unit_dtypes = []
        for module in model.modules():
            state = getattr(module, "_get_fsdp_state", None)
            group = state()._fsdp_param_group if state is not None else None
            if group is not None:
                unit_dtypes.append(sorted({str(fp.sharded_param.dtype) for fp in group.fsdp_params}))
        out = model(x)
        out.float().sum().backward()
        queue.put((rank, dict(
            storage = info["storage"],
            unit_dtypes = unit_dtypes,
            uint8_sharded = sum(isinstance(p, DTensor) for p in model.parameters() if not p.is_floating_point()),
            max_diff = (out.float() - expected.float()).abs().max().item(),
            grad_ok = all(p.grad is not None for n, p in model.named_parameters() if "adapter" in n),
        )))
        dist.destroy_process_group()
    except BaseException as error:
        import traceback

        queue.put((rank, {"error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"}))


def _run(case, world_size = 2, target = None):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target = target or _worker, args = (r, world_size, port, case, queue))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = {}
    for _ in procs:
        rank, result = queue.get(timeout = 240)
        results[rank] = result
    for p in procs:
        p.join(timeout = 60)
    return results


_RUN_DIST = pytest.mark.skipif(
    sys.platform == "win32"
    or not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason = "needs torch.distributed with gloo",
)


@_RUN_DIST
@pytest.mark.parametrize("case", ["bf16", "uint8_converted", "bf16_uniform"])
def test_fsdp2_shards_4bit_and_keeps_quant_state(case):
    results = _run(case)
    for rank, r in results.items():
        assert "error" not in r, r.get("error")
        assert r["info"]["patched"], r
        # The packed 4-bit bytes are split across the 2 ranks, not replicated.
        assert r["local_bytes"] * 2 == r["packed_bytes"], r
        # FSDP2's all-gathered weight still carries bitsandbytes' QuantState in forward.
        assert r["weight_quant_state"] and all(r["weight_quant_state"]), r
        assert r["sharded_has_qs"], r
        # Bit-exact vs the unsharded model: no fp16 cast touched the packed bytes.
        assert r["max_diff"] == 0.0, r
        assert r["grad_ok"], r
        # Packed bytes always travel as bytes: CUDA's float foreach copy rewrites NaN bit patterns.
        assert r["gather_dtypes"] == ["torch.uint8"], r


@_RUN_DIST
def test_unpatched_uniform_group_gathers_as_float():
    # Control: without the patch a block whose inputs share bf16 is gathered as bf16.
    results = _run("unpatched_uniform")
    for rank, r in results.items():
        assert "error" not in r, r.get("error")
        assert "torch.bfloat16" in r["gather_dtypes"], r


@_RUN_DIST
def test_fsdp2_mixed_precision_corrupts_unpatched_bf16_storage():
    # Control for the test above: without the patches, FSDP2 casts the packed bf16 bytes to fp16.
    results = _run("unpatched")
    for rank, r in results.items():
        print(rank, r)
        assert "error" not in r, r.get("error")
        assert r["max_diff"] > 0.0, r


def test_patch_is_idempotent_across_imports():
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

    assert _load_module().patch_fsdp2_params4bit() == (True, None)
    first = FSDPParam.init_dtype_attrs
    # A second copy of the module (another import path) must not wrap the wrapper.
    assert _load_module().patch_fsdp2_params4bit() == (True, None)
    assert FSDPParam.init_dtype_attrs is first
    assert getattr(first._unsloth_original, "_unsloth_original", None) is None


def test_no_float_storage_when_fsdp2_checks_frozen_dtypes():
    # torch < 2.8 asserts one dtype per unit, frozen weights included.
    m = _load_module()
    assert m._fsdp2_checks_frozen_dtypes() is False  # installed torch >= 2.8
    m._fsdp2_checks_frozen_dtypes = lambda: True
    assert m.float_storage_survives_wrap(_prepare_that_keeps_dtypes) is False
    from transformers import BitsAndBytesConfig

    cfg = BitsAndBytesConfig(load_in_4bit = True)
    assert not m.set_fsdp2_quant_storage(cfg, torch.float16)


@_RUN_DIST
def test_old_torch_leaves_uint8_weights_out_of_the_units():
    results = _run("uint8_old_torch", target = _old_torch_worker)
    for rank, r in results.items():
        assert "error" not in r, r.get("error")
        assert r["storage"] == ["uint8"], r
        assert r["unit_dtypes"] and all(d == ["torch.float32"] for d in r["unit_dtypes"]), r
        assert r["uint8_sharded"] == 0, r
        assert r["max_diff"] == 0.0, r
        assert r["grad_ok"], r


@pytest.mark.parametrize("why", ["upcasting_wrap", "old_torch", "unpatched"])
def test_caller_float_storage_falls_back_to_uint8_when_it_cannot_survive(monkeypatch, why):
    m = _load_module()
    lin = _linear4bit(128, 64, torch.bfloat16, 0)
    weight = lin.weight
    before = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
    raw = weight.data.view(torch.uint8).clone()
    prepare = _prepare_that_keeps_dtypes
    if why == "upcasting_wrap":
        prepare = _prepare_that_upcasts
    elif why == "old_torch":
        monkeypatch.setattr(m, "_fsdp2_checks_frozen_dtypes", lambda: True)
    else:
        monkeypatch.setattr(m, "patch_fsdp2_params4bit", lambda: (False, "no anchors"))
        monkeypatch.setattr(m, "_missing_fsdp2_anchors", lambda: ["x"])
    model = torch.nn.Sequential(lin)
    with m.fsdp2_qlora_wrap(model, prepare) as info:
        assert info["reverted"] == 1 and info["storage"] == ["uint8"], info
        model.to(torch.float32)
    assert lin.weight is weight and weight.dtype == torch.uint8 and lin.quant_storage == torch.uint8
    assert torch.equal(weight.data.reshape(-1), raw.reshape(-1))
    assert torch.equal(bnb.functional.dequantize_4bit(weight.data, weight.quant_state), before)


def test_float_storage_is_kept_when_it_survives():
    m = _load_module()
    lin = _linear4bit(128, 64, torch.bfloat16, 0)
    with m.fsdp2_qlora_wrap(torch.nn.Sequential(lin), _prepare_that_keeps_dtypes) as info:
        assert info["reverted"] == 0 and info["storage"] == ["bfloat16"], info


@pytest.mark.parametrize("storage", [torch.bfloat16, torch.float32])
def test_size_based_wrap_counts_packed_weights_as_uint8(storage):
    m = _load_module()
    from accelerate.utils import fsdp_utils
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    assert m.patch_size_based_wrap_policy()
    uint8_block = _Block(torch.uint8, 0)
    packed_block = _Block(torch.uint8, 0)
    assert m.params4bit_to_float_storage(packed_block, storage) == 2
    uint8_numel = sum(p.numel() for p in uint8_block.parameters())
    plugin = type("Plugin", (), {})()
    plugin.auto_wrap_policy = size_based_auto_wrap_policy
    plugin.min_num_params = uint8_numel - 1
    plugin.transformer_cls_names_to_wrap = None
    policy = fsdp_utils.fsdp2_prepare_auto_wrap_policy(plugin, packed_block)
    assert policy(uint8_block) and policy(packed_block)
    original = fsdp_utils.fsdp2_prepare_auto_wrap_policy._unsloth_original
    assert not original(plugin, packed_block)(packed_block)
