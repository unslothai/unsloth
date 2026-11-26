# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Expert parallel collectives (gloo CPU, 2 and 4 ranks): no rank skips a collective its peers enter."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib.util  # noqa: E402
import inspect  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_shared"))


def _epn():
    from unsloth_models_loader import load_models_module
    return load_models_module("ep_native")


def _tb():
    name = "_unsloth_token_budget_ep"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(REPO, "unsloth", "utils", "token_budget.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class _Experts(torch.nn.Module):
    """A fused experts module (per-slot routing weights)."""

    def __init__(self, experts, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(experts, hidden, hidden))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = hidden_states.new_zeros(hidden_states.shape)
        for k in range(top_k_index.shape[1]):
            e = top_k_index[:, k]
            w = top_k_weights[:, k : k + 1]
            out = out + w * torch.einsum("th,thg->tg", hidden_states, self.weight[e])
        return out


def _tagged(epn, tensor):
    p = torch.nn.Parameter(tensor)
    setattr(p, epn._EP, ("test",))
    return p


def _case_empty_rank(rank, world, tmp):
    """Even ranks hold expert gradients, odd ranks none (expert-only training)."""
    epn = _epn()
    g = torch.Generator().manual_seed(rank)
    params = [_tagged(epn, torch.zeros(3, 4)), _tagged(epn, torch.zeros(5))]
    if rank % 2 == 0:
        for p in params:
            p.grad = torch.randn(p.shape, generator = g)
    norm = float(epn.ep_clip_grad_norm_(params, 1e9))
    holder = torch.nn.Module()
    holder.a, holder.b = params
    epn._sync_nonfinite(holder, None)
    return {"norm": norm, "file": epn.__file__}


def _case_fp16_clip(rank, world, tmp, max_norm):
    epn = _epn()
    g = torch.Generator().manual_seed(10 + rank)
    expert = _tagged(epn, torch.zeros(5000, dtype = torch.float16))
    expert.grad = (torch.rand(5000, generator = g) * 5e3 + 5e3).half()
    shared = torch.nn.Parameter(torch.zeros(64, dtype = torch.float16))
    shared.grad = (torch.rand(64, generator = torch.Generator().manual_seed(99)) * 5e3 + 5e3).half()
    before = (expert.grad.double().clone(), shared.grad.double().clone())
    norm = float(epn.ep_clip_grad_norm_([expert, shared], max_norm))
    return {
        "norm": norm,
        "before": before,
        "after": (expert.grad.clone(), shared.grad.clone()),
        "file": epn.__file__,
    }


def _ep_shard(epn, world, rank, E, H):
    import torch.distributed as dist

    per_rank = E // world
    full = _Experts(E, H)  # same seed on every rank
    local = _Experts(per_rank, H)
    with torch.no_grad():
        local.weight.copy_(full.weight[rank * per_rank : (rank + 1) * per_rank])
    shard = types.SimpleNamespace(
        block = None,
        leaf = local,
        group = dist.group.WORLD,
        ep = world,
        calls = 0,
        experts = E,
        per_rank = per_rank,
        lo = rank * per_rank,
        hi = (rank + 1) * per_rank,
        swaps = [],
        frozen = {},
    )
    return full, local, shard


def _case_zero_token(rank, world, tmp):
    """Odd ranks route zero tokens; every rank's output and the expert grads match one process."""
    epn = _epn()
    E, H, K = 2 * world, 4, 2
    torch.manual_seed(0)
    full, local, shard = _ep_shard(epn, world, rank, E, H)
    rows = []
    for r in range(world):
        gen = torch.Generator().manual_seed(100 + r)
        T = 0 if r % 2 else 5 + r
        x = torch.randn(T, H, generator = gen)
        probs = torch.randn(T, E, generator = gen).softmax(-1)
        index = probs.topk(K, dim = -1).indices
        rows.append((x, index, probs.gather(1, index)))
    x, index, w = rows[rank]
    x = x.clone().requires_grad_(True)
    y = epn._ep_experts(shard, local.forward, x, index, w)
    y.square().sum().backward()
    for xr, ir, wr in rows:
        full(xr, ir, wr).square().sum().backward()
    want_y = full(*rows[rank]).detach()
    lo, hi = shard.lo, shard.hi
    return {
        "y": (y.detach(), want_y),
        "dw": (local.weight.grad, full.weight.grad[lo:hi]),
        "file": epn.__file__,
    }


class _Accelerator:
    distributed_type = "MULTI_GPU"
    sync_gradients = True

    def __init__(
        self,
        world,
        scaler = None,
    ):
        self.num_processes = world
        self.device = torch.device("cpu")
        self.scaler = scaler

    def backward(self, loss, **kwargs):
        loss.backward()

    def prepare_model(self, model, *args, **kwargs):
        return model

    def no_sync(self, model):
        return model.no_sync()

    def unscale_gradients(self):
        pass


class _Trainer:
    model_init = None
    model_accepts_loss_kwargs = True

    def __init__(self, model, accelerator, world, tmp):
        self.model, self.accelerator, self.world = model, accelerator, world
        self.args = types.SimpleNamespace(
            optim = "adamw_torch",
            neftune_noise_alpha = None,
            load_best_model_at_end = False,
            save_only_model = False,
            output_dir = tmp,
        )
        self.compute_loss_func = lambda *a, **k: None

    def save_model(self, *a, **k):
        pass

    def _save_optimizer_and_scheduler(self, *a):
        pass

    def _load_optimizer_and_scheduler(self, *a):
        pass

    def _load_from_checkpoint(self, *a, **k):
        pass

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch = None,
    ):
        from test_ep_moe_families_gloo import _sum_loss

        loss = _sum_loss(model, inputs["input_ids"], inputs["labels"]) * self.world
        loss = loss / num_items_in_batch
        self.accelerator.backward(loss)
        return loss.detach()


def _case_budget_replicas(rank, world, tmp, ep):
    """ep < world replicas: a step split by UNSLOTH_TOKEN_BUDGET gives the unsplit step's gradients."""
    from test_ep_moe_families_gloo import FAMILIES, _build, _modules, _rows
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel

    build, tpn, epn = _modules()
    tb = _tb()
    mesh = init_device_mesh("cpu", (world // ep, ep), mesh_dim_names = ("ep_rep", "ep"))
    model = _build(build, FAMILIES["olmoe"])
    epn.apply_expert_parallel(model, mesh)
    group = epn._mesh(model).get_group()
    ddp = DistributedDataParallel(model)
    trainer = _Trainer(model, _Accelerator(world), world, tmp)
    assert epn.ep_trainer_setup(trainer)
    vocab = tpn._text_config(model).vocab_size
    rows = [_rows(vocab, r) for r in range(world)]
    n = sum(int((lab[:, 1:] != -100).sum()) for _, lab in rows)
    ids, labels = rows[rank]
    inputs = {"input_ids": ids, "labels": labels}

    def grads():
        out = {}
        for k, p in model.named_parameters():
            if p.grad is not None:
                g = epn._gather_expert(p, group, p.grad) if getattr(p, epn._EP, None) else p.grad
                out[k] = g.detach().clone()
        model.zero_grad(set_to_none = True)
        return out

    trainer.training_step(ddp, dict(inputs), n)
    unsplit = grads()
    os.environ["UNSLOTH_TOKEN_BUDGET"] = str(ids.shape[1])  # one row per chunk: 2 chunks
    os.environ["UNSLOTH_TOKEN_BALANCE"] = "0"
    tb.install_token_budget(trainer)
    trainer.training_step(ddp, dict(inputs), n)
    split = grads()
    return {
        "splits": trainer._unsloth_token_budget_splits,
        "grads": (split, unsplit),
        "file": epn.__file__,
        "tb_file": tb.__file__,
    }


def _case_nonfinite_scaler(rank, world, tmp, ep):
    """Rank 1 holds an inf expert gradient: poisoned everywhere only under an enabled GradScaler."""
    from test_ep_moe_families_gloo import FAMILIES, _build, _modules
    from torch.distributed.device_mesh import init_device_mesh

    build, tpn, epn = _modules()
    mesh = init_device_mesh("cpu", (world // ep, ep), mesh_dim_names = ("ep_rep", "ep"))
    model = _build(build, FAMILIES["olmoe"])
    epn.apply_expert_parallel(model, mesh)
    out = {"file": epn.__file__}
    for name, scaler in (
        ("none", None),
        ("enabled", types.SimpleNamespace(is_enabled = lambda: True)),
    ):
        trainer = _Trainer(model, _Accelerator(world, scaler), world, tmp)
        assert epn.ep_trainer_setup(trainer)
        for _, p in epn._ep_params(model):
            p.grad = torch.zeros_like(p)
        if rank == 1:
            epn._ep_params(model)[-1][1].grad.view(-1)[3] = float("inf")
        trainer.accelerator.backward(torch.zeros((), requires_grad = True) * 1)
        out[name] = all(bool(torch.isfinite(p.grad).all()) for _, p in epn._ep_params(model))
    return out


def _case_nonfinite_no_expert_grads(rank, world, tmp):
    """fp16 GradScaler: rank 0 has an inf expert gradient, rank 1 no expert gradient at all (none of its experts
    was routed to) and a finite shared one. Every rank must skip the step, or the shared weights diverge."""
    epn = _epn()
    holder = torch.nn.Module()
    holder.expert = _tagged(epn, torch.zeros(3, 4))
    holder.shared = torch.nn.Parameter(torch.ones(5))
    if rank == 0:
        holder.expert.grad = torch.zeros(3, 4)
        holder.expert.grad[1, 2] = float("inf")
    holder.shared.grad = torch.full((5,), 0.5)
    optimizer = torch.optim.SGD(holder.parameters(), lr = 1.0)
    scaler = torch.amp.GradScaler("cpu", init_scale = 1.0)
    scaler.scale(torch.ones(()))  # creates the scale, as scaler.scale(loss) does before backward
    epn._sync_nonfinite(holder, None)
    scaler.step(optimizer)
    scaler.update()
    return {
        "stepped": not torch.equal(holder.shared.detach(), torch.ones(5)),
        "scale": float(scaler.get_scale()),
        "file": epn.__file__,
    }


def _worker(rank, world, tmp, case, timeout, kwargs):
    os.environ.update(
        RANK = str(rank), WORLD_SIZE = str(world), LOCAL_RANK = str(rank), LOCAL_WORLD_SIZE = str(world)
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo",
        store = dist.FileStore(os.path.join(tmp, "rendezvous"), world),
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = timeout),
    )
    try:
        res = globals()[case](rank, world, tmp, **kwargs)
    except BaseException as exc:  # a rank must report, not vanish
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-3000:]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _spawn(
    case,
    world = 2,
    timeout = 300,
    **kwargs,
):
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, tmp, case, timeout, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        deadline = time.monotonic() + 600
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            time.sleep(0.2)
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(30)
        out = []
        for r in range(world):
            path = os.path.join(tmp, f"result_{r}.pt")
            out.append(torch.load(path, weights_only = False) if os.path.exists(path) else None)
        return out


def _ok(results):
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:  # each arm imports its own tree
        assert r["file"].startswith(os.path.join(REPO, "unsloth")), (r["file"], REPO)


@pytest.mark.parametrize("world", [2, 4])
def test_rank_without_expert_grads_enters_the_collectives(world):
    results = _spawn("_case_empty_rank", world = world, timeout = 20)
    _ok(results)
    sq = 0.0
    for r in range(0, world, 2):
        g = torch.Generator().manual_seed(r)
        sq += sum(
            float(torch.randn(s, generator = g).double().square().sum()) for s in ((3, 4), (5,))
        )
    for r in results:
        assert r["norm"] == pytest.approx(sq**0.5, rel = 1e-6), [x["norm"] for x in results]


def test_fp16_clip_coefficient_does_not_underflow():
    max_norm = 1e-2
    results = _spawn("_case_fp16_clip", world = 2, max_norm = max_norm)
    _ok(results)
    shared = results[0]["before"][1]
    total = (
        sum(float(r["before"][0].square().sum()) for r in results) + float(shared.square().sum())
    ) ** 0.5
    assert total / max_norm > 1e7
    coef = min(1.0, max_norm / (total + 1e-6))
    for r in results:
        assert r["norm"] == pytest.approx(total, rel = 1e-5)
        for before, after in zip(r["before"], r["after"]):
            assert after.dtype == torch.float16
            assert bool((after != 0).all()), "clipped fp16 gradient rounded to 0"
            torch.testing.assert_close(after.double(), before * coef, rtol = 2e-3, atol = 0)


@pytest.mark.parametrize("world", [2, 4])
def test_zero_token_sender_matches_one_process(world):
    results = _spawn("_case_zero_token", world = world)
    _ok(results)
    for r in results:
        got, want = r["y"]
        assert got.shape == want.shape, (got.shape, want.shape)
        torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-5)
        got, want = r["dw"]
        assert got is not None
        torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-5)


def test_token_budget_chunks_sum_replicas_once():
    results = _spawn("_case_budget_replicas", world = 4, ep = 2)
    _ok(results)
    for r in results:
        assert r["splits"] == 1, r["splits"]
        assert r["tb_file"].startswith(REPO), r["tb_file"]
        split, unsplit = r["grads"]
        assert set(split) == set(unsplit)
        assert any(".experts." in k for k in split)
        for k in unsplit:
            torch.testing.assert_close(split[k], unsplit[k], rtol = 1e-5, atol = 1e-6, msg = k)


def test_nonfinite_sync_only_under_an_enabled_grad_scaler():
    results = _spawn("_case_nonfinite_scaler", world = 4, ep = 2)
    _ok(results)
    assert [r["none"] for r in results] == [True, False, True, False], results
    assert [r["enabled"] for r in results] == [False] * 4, results


def test_nonfinite_sync_skips_the_step_on_a_rank_without_expert_grads():
    results = _spawn("_case_nonfinite_no_expert_grads", world = 2, timeout = 20)
    _ok(results)
    assert [r["stepped"] for r in results] == [False, False], results
    assert results[0]["scale"] == results[1]["scale"] == 0.5, results  # both scalers backed off


@pytest.mark.parametrize("tokens", [0, 1, 7, 64])
@pytest.mark.parametrize("ep", [2, 3, 8])
def test_destination_mask_is_the_one_hot_mask(tokens, ep):
    epn = _epn()
    for k in (1, 2, 4):
        owner = torch.randint(0, ep, (tokens, k), generator = torch.Generator().manual_seed(k))
        want = torch.nn.functional.one_hot(owner, ep).amax(1).bool()
        got = epn._destinations(owner, ep)
        assert got.dtype == torch.bool and got.shape == want.shape
        assert torch.equal(got, want)


def test_ep_experts_dispatch_is_compile_disabled():
    # CPU proxy for the CUDA graph-break proof: Dynamo's disable marker.
    epn = _epn()
    assert getattr(epn._ep_experts, "_torchdynamo_disable", False) is True
    assert epn._ep_experts.__name__ == "_ep_experts"


def test_tp_zero_token_call(monkeypatch):
    from test_tp_native_oracle import tpn

    monkeypatch.setattr(tpn, "_experts_copy", lambda t, group: t)
    monkeypatch.setattr(tpn, "_experts_reduce", lambda t, group: t)
    torch.manual_seed(0)
    E, H, K = 4, 4, 2
    local = _Experts(2, H)
    shard = types.SimpleNamespace(
        block = None, leaf = local, group = None, lo = 0, hi = 2, per_rank = 2, experts = E, swaps = [], frozen = {}
    )
    x = torch.zeros(1, 0, H, requires_grad = True)
    index = torch.zeros(1, 0, K, dtype = torch.long)
    w = torch.zeros(1, 0, K)
    y = tpn._sharded_experts(shard, local.forward, x, index, w)
    assert y.shape == (1, 0, H)
    (y.sum() + 0).backward()
    assert local.weight.grad is not None and not local.weight.grad.any()


def _level():
    class Level:
        def generate(
            self,
            input_ids,
            max_new_tokens = 1,
            **kwargs,
        ):
            """generate docstring"""
            return kwargs

    return Level()


def test_ep_wrappers_keep_the_wrapped_signature():
    epn = _epn()
    level = _level()
    want = inspect.signature(level.generate)
    epn._sync_generate(level)
    assert level.generate.__name__ == "generate"
    assert inspect.signature(level.generate) == want
    assert level.generate(1)["synced_gpus"] is True

    class Block(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            attention_mask = None,
        ):
            return hidden_states

    block = Block()
    want = inspect.signature(block.forward)
    epn._guard_bypass(block, types.SimpleNamespace(calls = 0))
    assert inspect.signature(block.forward) == want
