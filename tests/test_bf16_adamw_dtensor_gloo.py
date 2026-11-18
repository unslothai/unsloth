# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""AdamWBF16 on dim-0 sharded DTensors (gloo, world 2): bitwise equal to the unsharded run."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_layout_agree_gloo import _init, _spawn  # noqa: E402

SHAPES = [(7, 5), (64, 33), (3,), (1, 4), (130,)]
STEPS = 3
RETAIN = 20


def _api():
    path = Path(__file__).resolve().parents[1] / "unsloth/optimizers/bf16_adamw.py"
    spec = importlib.util.spec_from_file_location("_test_bf16_adamw_dtensor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _initial(i, shape):
    g = torch.Generator().manual_seed(10 + i)
    return (torch.randn(shape, generator = g) * 0.05).to(torch.bfloat16)


def _grad(i, step, shape):
    g = torch.Generator().manual_seed(1000 * step + i)
    return (torch.randn(shape, generator = g) * 1e-3).to(torch.bfloat16)


def _reference(api, lr):
    params = [torch.nn.Parameter(_initial(i, s)) for i, s in enumerate(SHAPES)]
    opt = api.AdamWBF16(params, lr = lr, weight_decay = 0.01, seed = 11)
    for step in range(STEPS):
        for i, p in enumerate(params):
            p.grad = _grad(i, step, p.shape)
        opt.step()
    return params, opt


def _sharded(
    api,
    mesh,
    lr,
    steps,
    state = None,
):
    from torch.distributed.tensor import Shard, distribute_tensor

    params = [
        torch.nn.Parameter(distribute_tensor(_initial(i, s), mesh, [Shard(0)]))
        for i, s in enumerate(SHAPES)
    ]
    opt = api.AdamWBF16(params, lr = lr, weight_decay = 0.01, seed = 11)
    if state is not None:
        with torch.no_grad():
            for p, value in zip(params, state["params"]):
                p.copy_(distribute_tensor(value, mesh, [Shard(0)]))
        opt.load_state_dict(state["optimizer"])
    for step in steps:
        for i, p in enumerate(params):
            p.grad = distribute_tensor(_grad(i, step, p.shape), mesh, [Shard(0)])
        opt.step()
    return params, opt


def _local_run(full, p):
    offset = _api()._shard_offset(p)
    return full.reshape(-1)[offset : offset + p.to_local().numel()].view(p.to_local().shape)


def _same(sharded, opt, ref, ref_opt):
    for i, (p, r) in enumerate(zip(sharded, ref)):
        assert torch.equal(p.to_local(), _local_run(r.detach(), p)), f"param {i}"
        for name in ("exp_avg", "exp_avg_sq"):
            mine = opt.state[p][name]
            assert type(mine).__name__ == "DTensor", (i, name)
            assert torch.equal(mine.to_local(), _local_run(ref_opt.state[r][name], p)), (i, name)
        assert int(opt.state[p]["step"]) == int(ref_opt.state[r]["step"])


def _case_parity(rank):
    from torch.distributed.device_mesh import init_device_mesh

    api = _api()
    mesh = init_device_mesh("cpu", (2,))
    ref, ref_opt = _reference(api, lr = 1e-3)
    sharded, opt = _sharded(api, mesh, 1e-3, range(STEPS))
    _same(sharded, opt, ref, ref_opt)
    # The (1, 4) parameter leaves rank 1 an empty shard; the update must still run on rank 0.
    assert sharded[3].to_local().numel() == (4 if rank == 0 else 0)
    # Resume: a whole saved state (one GPU's checkpoint) continues bitwise on the shards.
    params, part_opt = _reference_partial(api)
    state = {"params": [p.detach() for p in params], "optimizer": part_opt.state_dict()}
    resumed, resumed_opt = _sharded(api, mesh, 1e-3, range(1, STEPS), state = state)
    _same(resumed, resumed_opt, ref, ref_opt)
    again, again_opt = _sharded(api, mesh, 1e-3, [])
    again_opt.load_state_dict(opt.state_dict())
    for p, q in zip(again, sharded):
        for name in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(again_opt.state[p][name].to_local(), opt.state[q][name].to_local())
    return "ok"


def _reference_partial(api):
    params = [torch.nn.Parameter(_initial(i, s)) for i, s in enumerate(SHAPES)]
    opt = api.AdamWBF16(params, lr = 1e-3, weight_decay = 0.01, seed = 11)
    for i, p in enumerate(params):
        p.grad = _grad(i, 0, p.shape)
    opt.step()
    return params, opt


def _case_native(rank):
    """At lr 1e-5 stochastic rounding keeps the fp32-master move that plain bf16 AdamW rounds away."""
    from test_fsdp2_native_units import _load_native

    native = _load_native()
    api = _api()
    import transformers

    config = transformers.LlamaConfig(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        tie_word_embeddings = False,
    )
    out = {}
    for name in ("adamw", "adamw_bf16"):
        torch.manual_seed(0)
        model = transformers.LlamaForCausalLM(config).to(torch.bfloat16)
        native.shard_model(model, native.FSDP2Config(param_dtype = torch.bfloat16, fit_layout = False))
        params = [p for p in model.parameters() if p.requires_grad]
        dtensors = [p for p in params if type(p).__name__ == "DTensor"]
        assert dtensors and all(p.dtype == torch.bfloat16 for p in dtensors)
        if name == "adamw":
            opt = torch.optim.AdamW(native.split_param_groups(params), lr = 1e-5, weight_decay = 0.0)
        else:
            opt = api.AdamWBF16(params, lr = 1e-5, weight_decay = 0.0)
        ids = torch.randint(3, 256, (2, 16), generator = torch.Generator().manual_seed(rank))
        model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
        grads = {p: p.grad.detach().clone() for p in params}
        before = [p.detach().to_local().clone() for p in dtensors]
        masters = [torch.nn.Parameter(b.float()) for b in before]
        ref = torch.optim.AdamW(masters, lr = 1e-5, weight_decay = 0.0)
        for _ in range(RETAIN):
            for p in params:
                p.grad = grads[p].clone()
            for m, p in zip(masters, dtensors):
                m.grad = grads[p].to_local().float()
            opt.step()
            ref.step()
        moved = torch.cat(
            [
                (p.detach().to_local().float() - b.float()).flatten()
                for p, b in zip(dtensors, before)
            ]
        )
        wanted = torch.cat([(m.detach() - b.float()).flatten() for m, b in zip(masters, before)])
        out[name] = (moved.double() @ wanted.double() / (wanted.double() @ wanted.double())).item()
    assert out["adamw"] < 0.5, out
    assert abs(out["adamw_bf16"] - 1) < 0.1, out
    return out


def _worker(rank, world, port, out, case):
    dist = _init(rank, world, port)
    try:
        result = {"parity": _case_parity, "native": _case_native}[case](rank)
        out.put({"rank": rank, "result": result})
    except Exception as exc:
        import traceback
        out.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["parity", "native"])
def test_adamw_bf16_steps_dtensor_shards(case):
    if case == "native":
        pytest.importorskip("transformers")
    results = _spawn(_worker, case)
    assert len(results) == 2, results
    for r in results:
        assert "error" not in r, r["error"]
    print("ADAMW_BF16_DTENSOR", case, results)


def _has_real_cuda():
    try:
        from real_accelerator import has_real_cuda
        return has_real_cuda()
    except Exception:
        return torch.cuda.is_available()


@pytest.mark.gpu
@pytest.mark.skipif(not _has_real_cuda(), reason = "requires real CUDA")
@pytest.mark.parametrize("compiled", [False, True])
def test_gpu_shard_offset_matches_unsharded_run(compiled):
    api = _api()
    if api.triton is None:
        pytest.skip("needs Triton")
    sizes = (1000, 70000)
    cut = (300, 40001)
    full = [torch.randn(n, device = "cuda").mul(0.05).bfloat16() for n in sizes]
    grads = [torch.randn(n, device = "cuda").mul(1e-3).bfloat16() for n in sizes]

    def run(params, gs, offsets):
        m = [torch.zeros_like(p, dtype = torch.float32) for p in params]
        v = [torch.zeros_like(p, dtype = torch.float32) for p in params]
        steps = [torch.ones((), device = "cuda", dtype = torch.int64) for _ in params]
        seeds = [torch.tensor(17 + (i << 32), device = "cuda") for i in range(len(params))]
        lr = torch.tensor(1e-3, device = "cuda")
        args = (params, gs, m, v, steps, lr, seeds, offsets, 0.9, 0.999, 1e-8, 0.01)
        if compiled:
            torch.compile(lambda *a: api._cuda_update(*a), fullgraph = True)(*args)
        else:
            api._launch(*args)
        return params, m

    ref, ref_m = run([p.clone() for p in full], grads, [0, 0])
    head, head_m = run(
        [p[:c].clone() for p, c in zip(full, cut)], [g[:c] for g, c in zip(grads, cut)], [0, 0]
    )
    tail, tail_m = run(
        [p[c:].clone() for p, c in zip(full, cut)], [g[c:] for g, c in zip(grads, cut)], list(cut)
    )
    for i, c in enumerate(cut):
        assert torch.equal(torch.cat([head[i], tail[i]]), ref[i]), i
        assert torch.equal(torch.cat([head_m[i], tail_m[i]]), ref_m[i]), i
    # Without the offset the tail would reuse the head's draws: a different rounding.
    wrong, _ = run(
        [p[c:].clone() for p, c in zip(full, cut)], [g[c:] for g, c in zip(grads, cut)], [0, 0]
    )
    assert any(not torch.equal(w, t) for w, t in zip(wrong, tail))
