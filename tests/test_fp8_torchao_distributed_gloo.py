# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""load_in_fp8 weights (torchao Float8Tensor) under native FSDP2 and DDP, gloo world of 2, CPU only.

Float8Tensor reports bfloat16 / float32 as its dtype and implements only the ops its linear needs:
fully_shard fails on it (aten.new_zeros) and DDP's start-up broadcast fails on it (reshape to [-1]).
The frozen FP8 base is loaded identically on every rank, so it stays replicated: out of fully_shard
(ignored_params) in every layout, and out of DDP (_ddp_params_and_buffers_to_ignore), both via the load
time hook and via the trainer hook. The trainable parameters still shard / all-reduce, and the gradients
match one process run over the same rows. A trainable Float8Tensor is refused under both."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import ast  # noqa: E402
import copy  # noqa: E402
import datetime  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402
from test_fsdp2_native_guards_gloo import _tiny_llama  # noqa: E402

MODELS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth", "models"
)
_FAKE_PKG = "_fp8_torchao_ddp_pkg"


def _extract(path, names):
    src = open(path, encoding = "utf-8").read()
    out = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names:
            out.append(ast.get_source_segment(src, node))
    return "\n\n".join(out)


def _ddp_helpers():
    """The DDP ignore-list helpers of loader_utils.py / _utils.py, without importing unsloth (GPU)."""
    lu = types.ModuleType(f"{_FAKE_PKG}.loader_utils")
    lu.__dict__.update(torch = torch, is_distributed = lambda: True)
    exec(
        _extract(
            os.path.join(MODELS, "loader_utils.py"),
            {
                "_ROTARY_INV_FREQ_BUFFER_NAMES",
                "_is_tensor_subclass_weight",
                "_holds_float8",
                "_ddp_can_flatten",
                "_frozen_tensor_subclass_names_for_ddp",
                "_exclude_rope_inv_freq_from_ddp",
            },
        ),
        lu.__dict__,
    )
    pkg = types.ModuleType(_FAKE_PKG)
    pkg.__path__ = []
    sys.modules[_FAKE_PKG] = pkg
    sys.modules[lu.__name__] = lu
    ut = {"__name__": f"{_FAKE_PKG}._utils", "__package__": _FAKE_PKG, "torch": torch, "os": os}
    exec(_extract(os.path.join(MODELS, "_utils.py"), {"_tensor_subclass_weights_for_ddp"}), ut)
    return lu, ut


def _fp8_llama(trainable_fp8 = False):
    from torchao.quantization import Float8Tensor, PerRow

    torch.manual_seed(0)
    model = _tiny_llama()
    for name, p in model.named_parameters():
        p.requires_grad_("norm" in name)  # the trainable set beside the frozen FP8 base
    for layer in model.model.layers:
        for mod in (layer.self_attn.q_proj, layer.mlp.up_proj, layer.mlp.down_proj):
            mod.weight = torch.nn.Parameter(
                Float8Tensor.from_hp(mod.weight.detach().clone(), granularity = PerRow()),
                requires_grad = trainable_fp8,
            )
    return model


def _fp8_weights(model):
    return [p for p in model.parameters() if type(p).__name__ == "Float8Tensor"]


def _rows(rank):
    return torch.randint(1, 100, (1, 8), generator = torch.Generator().manual_seed(100 + rank))


def _reference_grads(model):
    """Gradients of one process over both ranks' rows (mean of the two per-row losses)."""
    ref = copy.deepcopy(model)
    for rank in (0, 1):
        ids = _rows(rank)
        (ref(input_ids = ids, labels = ids).loss / 2).backward()
    return {n: p.grad.clone() for n, p in ref.named_parameters() if p.requires_grad}


def _grad_diff(
    model,
    want,
    canon = lambda n: n,
):
    from torch.distributed.tensor import DTensor

    worst = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
        worst = max(worst, (g - want[canon(n)]).abs().max().item())
    return worst


def _fsdp_case(native, rank, shard_frozen):
    from torch.distributed.tensor import DTensor

    model = _fp8_llama()
    want = _reference_grads(model)
    cfg = native.FSDP2Config(param_dtype = torch.float32, shard_frozen = shard_frozen)
    try:
        model = native.shard_model(model, cfg)
    except Exception as exc:
        return {"wrap": f"{type(exc).__name__}: {str(exc)[:300]}"}
    weights = _fp8_weights(model)
    res = {
        "wrap": "ok",
        "fp8": len(weights),
        "fp8_dtensor": sum(isinstance(w, DTensor) for w in weights),
        "other_dtensor": sum(isinstance(p, DTensor) for p in model.parameters()),
    }
    ids = _rows(rank)
    model(input_ids = ids, labels = ids).loss.backward()
    res["grad_diff"] = _grad_diff(model, want)
    return res


def _other_torchao_llama(kind):
    """A 256 wide llama whose MLP weights are torchao subclasses that store no float8: frozen NF4 (QLoRA)
    or trainable int8 mixed precision training weights. Both implement FSDP2's all-gather hooks."""
    import transformers

    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        hidden_size = 256,
        intermediate_size = 256,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        vocab_size = 128,
    )
    model = transformers.LlamaForCausalLM(config)
    for name, p in model.named_parameters():
        p.requires_grad_("norm" in name)
    mlp = model.model.layers[0].mlp
    for mod in (mlp.up_proj, mlp.down_proj):
        w = mod.weight.detach().clone()
        if kind == "nf4":
            from torchao.quantization.quantize_.workflows.nf4.nf4_tensor import to_nf4
            mod.weight = torch.nn.Parameter(to_nf4(w), requires_grad = False)
        else:
            from torchao.prototype.quantized_training.int8_mixed_precision import (
                Int8MixedPrecisionTrainingConfig,
                Int8MixedPrecisionTrainingLinearWeight,
            )
            mod.weight = torch.nn.Parameter(
                Int8MixedPrecisionTrainingLinearWeight(w, Int8MixedPrecisionTrainingConfig()),
                requires_grad = True,
            )
    return model


def _fsdp_other_case(native, rank, kind):
    from torch.distributed.tensor import DTensor

    model = _other_torchao_llama(kind)
    cfg = native.FSDP2Config(param_dtype = torch.float32, shard_frozen = True)
    try:
        model = native.shard_model(model, cfg)
    except Exception as exc:
        return {"wrap": f"{type(exc).__name__}: {str(exc)[:300]}"}
    mlp = model.model.layers[0].mlp
    weights = [mlp.up_proj.weight, mlp.down_proj.weight]
    local = [w.to_local() if isinstance(w, DTensor) else w for w in weights]
    if kind == "nf4":
        nbytes = sum(t.quantized_data.numel() * t.quantized_data.element_size() for t in local)
    else:
        nbytes = sum(t._data.numel() * t._data.element_size() for t in local)
    ids = _rows(rank)
    model(input_ids = ids, labels = ids).loss.backward()
    grads = [w.grad for w in weights if w.requires_grad]
    return {
        "wrap": "ok",
        "dtensor": sum(isinstance(w, DTensor) for w in weights),
        "local_bytes": nbytes,
        "grads": len(grads),
        "finite": all(
            torch.isfinite(g.full_tensor() if isinstance(g, DTensor) else g).all().item()
            for g in grads
        ),
    }


def _ddp_case(lu, ut, rank, via):
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = _fp8_llama(trainable_fp8 = via == "trainable")
    want = None if via == "trainable" else _reference_grads(model)
    try:
        if via == "load":
            lu._exclude_rope_inv_freq_from_ddp(model)  # from_pretrained / get_peft_model
        elif via in ("trainer", "trainable"):
            ut["_tensor_subclass_weights_for_ddp"](types.SimpleNamespace(model = model))
        elif via == "model_init":  # Trainer.call_model_init marks the rebuilt model it returns
            ut["_tensor_subclass_weights_for_ddp"](types.SimpleNamespace(model = None), model)
        ddp = DDP(model)
    except Exception as exc:
        return {"wrap": f"{type(exc).__name__}: {str(exc)[:300]}"}
    ids = _rows(rank)
    ddp(input_ids = ids, labels = ids).loss.backward()
    return {
        "wrap": "ok",
        "fp8": len(_fp8_weights(model)),
        "ignored": len(getattr(model, "_ddp_params_and_buffers_to_ignore", ())),
        "grad_diff": _grad_diff(model, want),
    }


def _worker(rank, world, port, out, kind, arg):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    if kind.startswith("fsdp"):
        os.environ["UNSLOTH_FSDP2"] = "1"
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 30)
    )
    try:
        if kind == "fsdp":
            res = _fsdp_case(_load_native(), rank, arg)
        elif kind == "fsdp_other":
            res = _fsdp_other_case(_load_native(), rank, arg)
        else:
            res = _ddp_case(*_ddp_helpers(), rank, arg)
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(
    kind,
    arg,
    timeout = 150,
):
    import torch.multiprocessing as mp

    out = mp.get_context("spawn").Queue()
    procs = mp.start_processes(
        _worker, args = (2, _free_port(), out, kind, arg), nprocs = 2, join = False, start_method = "spawn"
    )
    deadline = time.time() + timeout
    results = {}
    try:
        while not procs.join(timeout = 2):
            while not out.empty():
                rank, r = out.get(timeout = 5)
                results[rank] = r
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail(
                    f"fp8 {kind} {arg}: the ranks did not finish (hung collective) {results}"
                )
    except mp.ProcessExitedException as exc:
        pytest.fail(f"fp8 {kind} {arg}: a rank died ({exc}) {results}")
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("FP8_RESULT", kind, arg, rank, r)
        assert "crash" not in r, r["crash"]
    return results


@pytest.fixture(autouse = True)
def _needs_torchao():
    pytest.importorskip("transformers")
    pytest.importorskip("torchao.quantization").Float8Tensor  # noqa: B018


@pytest.mark.parametrize("shard_frozen", [True, None], ids = ["sharded", "auto_fit"])
def test_fsdp2_keeps_float8tensor_replicated_and_matches_one_process(shard_frozen):
    for rank, r in _run("fsdp", shard_frozen).items():
        assert r["wrap"] == "ok", (rank, r)
        assert r["fp8"] > 0 and r["fp8_dtensor"] == 0, (rank, r)
        assert r["other_dtensor"] > 0, (rank, r)  # the trainable norms are still sharded
        assert r["grad_diff"] < 1e-5, (rank, r)


@pytest.mark.parametrize("via", ["load", "trainer", "model_init"])
def test_ddp_skips_float8tensor_broadcast_and_matches_one_process(via):
    for rank, r in _run("ddp", via).items():
        assert r["wrap"] == "ok", (rank, r)
        assert r["ignored"] >= r["fp8"] > 0, (rank, r)
        assert r["grad_diff"] < 1e-5, (rank, r)


def test_trainable_float8tensor_is_refused():
    for rank, r in _run("ddp", "trainable").items():
        assert r["wrap"].startswith("NotImplementedError") and "Float8Tensor" in r["wrap"], (
            rank,
            r,
        )
    native = _load_native()
    with pytest.raises(NotImplementedError, match = "trainable float8"):
        native._float8_params(_fp8_llama(trainable_fp8 = True))


def test_float8_training_weights_stay_with_ddp():
    # torchao.float8's training weight implements the flatten DDP's broadcast needs, and plain DDP
    # trains it: neither left out of DDP nor refused, as fsdp2_native leaves it to fully_shard.
    fsdp_utils = pytest.importorskip("torchao.float8.fsdp_utils")
    from torchao.float8.float8_training_tensor import LinearMMConfig

    lu, _ = _ddp_helpers()
    lin = torch.nn.Linear(8, 8, bias = False)
    lin.weight = torch.nn.Parameter(
        fsdp_utils.WeightWithDynamicFloat8CastTensor(
            lin.weight.detach().clone(), LinearMMConfig(), torch.float8_e4m3fn
        )
    )
    assert lu._is_tensor_subclass_weight(lin.weight) is False
    assert lu._frozen_tensor_subclass_names_for_ddp(lin, refuse_trainable = True) == []
    from torchao.quantization import Float8Tensor, PerRow

    frozen = torch.nn.Parameter(
        Float8Tensor.from_hp(torch.randn(8, 8), granularity = PerRow()), requires_grad = False
    )
    assert lu._is_tensor_subclass_weight(frozen) is True


def test_fsdp2_still_shards_frozen_nf4_weights():
    # NF4Tensor stores uint8 / int8, not float8, and implements FSDP2's hooks: with shard_frozen each rank
    # holds half the packed bytes (256 x 256 nf4 = 32768 bytes per weight), not a replicated copy.
    pytest.importorskip("torchao.quantization.quantize_.workflows.nf4.nf4_tensor")
    for rank, r in _run("fsdp_other", "nf4").items():
        assert r["wrap"] == "ok", (rank, r)
        assert r["dtensor"] == 2, (rank, r)
        assert r["local_bytes"] == 2 * 32768 // 2, (rank, r)


def test_fsdp2_shards_trainable_int8_mixed_precision_weights():
    pytest.importorskip("torchao.prototype.quantized_training.int8_mixed_precision")
    for rank, r in _run("fsdp_other", "int8").items():
        assert r["wrap"] == "ok", (rank, r)
        assert r["dtensor"] == 2 and r["grads"] == 2 and r["finite"], (rank, r)


def test_ddp_keeps_trainable_non_float8_torchao_weights():
    # Only float8 storage is refused for DDP training: the int8 mixed precision training weight
    # implements the flatten DDP's broadcast needs, and frozen NF4 is simply left out of the broadcast.
    nf4 = pytest.importorskip("torchao.quantization.quantize_.workflows.nf4.nf4_tensor")
    int8 = pytest.importorskip("torchao.prototype.quantized_training.int8_mixed_precision")
    lu, _ = _ddp_helpers()
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256, bias = False), torch.nn.Linear(256, 256, bias = False)
    )
    model[0].weight = torch.nn.Parameter(
        int8.Int8MixedPrecisionTrainingLinearWeight(
            model[0].weight.detach().clone(), int8.Int8MixedPrecisionTrainingConfig()
        ),
        requires_grad = True,
    )
    model[1].weight = torch.nn.Parameter(
        nf4.to_nf4(model[1].weight.detach().clone()), requires_grad = False
    )
    assert lu._frozen_tensor_subclass_names_for_ddp(model, refuse_trainable = True) == ["1.weight"]
    native = _load_native()
    assert native._float8_params(model) == set()


def test_frozen_int8_inference_weights_stay_replicated_and_trainable_ones_refused():
    # torchao's Int8Tensor (Int8WeightOnlyConfig, the portable INT8 export) stores int8, not float8, but
    # implements no FSDP2 all-gather extension: fully_shard fails on it, so frozen ones stay replicated as
    # under #15001, and DDP's broadcast cannot flatten it, so a trainable one is refused up front.
    Q = pytest.importorskip("torchao.quantization")
    if not hasattr(Q, "Int8Tensor"):
        pytest.skip("torchao without Int8Tensor")
    lin = torch.nn.Linear(64, 64, bias = False)
    Q.quantize_(lin, Q.Int8WeightOnlyConfig(version = 2))
    assert type(lin.weight).__name__ == "Int8Tensor"
    lin.weight.requires_grad_(False)
    native = _load_native()
    assert native._float8_params(lin) == {lin.weight}
    lu, _ = _ddp_helpers()
    assert lu._frozen_tensor_subclass_names_for_ddp(lin, refuse_trainable = True) == ["weight"]
    lin.weight.requires_grad_(True)
    with pytest.raises(NotImplementedError, match = "native FSDP2 cannot shard"):
        native._float8_params(lin)
    with pytest.raises(NotImplementedError, match = "DDP cannot train"):
        lu._frozen_tensor_subclass_names_for_ddp(lin, refuse_trainable = True)
