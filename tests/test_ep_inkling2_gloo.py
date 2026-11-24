# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Inkling 2 expert parallel (gloo CPU, 2 ranks, fp32) vs one process on the concatenated batch.

Each rank trains its own rows (rank 1 a shorter sequence, so the all-to-all splits are uneven) through torch
DDP; the routed experts are sharded by index (2 of 4 per rank) and every experts call exchanges tokens over the
ranks. Inkling-2-Mini (inkling2_loop: layers 6-11 looped three times with message gates) and Inkling-2-Small
(inkling2), LoRA on the attention and on the routed experts; then bitsandbytes 4-bit expert stacks quantized on
load where each rank keeps only its own experts. Losses, LoRA gradients (expert ones gathered), the clipping
norm and the parameters after one SGD step match one process; the gathered adapter state equals one process'."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_tp_inkling2_gloo as ink  # noqa: E402

_SEQ = (12, 8)  # rank 0 / rank 1 sequence length


def _rows(rank):
    g = torch.Generator().manual_seed(10 + rank)
    ids = torch.randint(1, 250, (2, _SEQ[rank]), generator = g)
    labels = ids.clone()
    labels[0, :3] = -100
    return ids, labels


def _sum_loss(model, ids, labels):
    logits = model(input_ids = ids, use_cache = False).logits
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].flatten(0, 1).float(),
        labels[:, 1:].flatten(),
        ignore_index = -100,
        reduction = "sum",
    )


def _tokens(world):
    return sum(int((_rows(r)[1][:, 1:] != -100).sum()) for r in range(world))


def _ref(
    model,
    world,
    steps = 2,
):
    """One process: the token mean over every rank's rows (one backward per rank's rows, summed)."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr = 0.05)
    n = _tokens(world)
    losses, grads, after, norm = [], None, None, None
    for step in range(steps):
        total = 0.0
        for r in range(world):
            loss = _sum_loss(model, *_rows(r)) / n
            loss.backward()
            total += float(loss.detach())
        if step == 0:
            grads = {
                k: p.grad.detach().clone().double()
                for k, p in model.named_parameters()
                if p.grad is not None
            }
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1e9))
        opt.step()
        opt.zero_grad(set_to_none = True)
        if step == 0:
            after = {
                k: p.detach().clone().double()
                for k, p in model.named_parameters()
                if p.requires_grad
            }
        losses.append(total)
    return {"losses": losses, "grads": grads, "after": after, "norm": norm}


def _ep_train(
    epn,
    model,
    rank,
    world,
    steps = 2,
):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    group = epn._mesh(model).get_group()
    ddp = DistributedDataParallel(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr = 0.05)
    n = _tokens(world)

    def full(p, value):
        return epn._gather_expert(p, group, value) if getattr(p, epn._EP, None) else value

    losses, grads, after, norm = [], None, None, None
    for step in range(steps):
        # Each rank's loss scaled as the Trainer's (sum over its tokens * world / all tokens); DDP averages.
        loss = _sum_loss(ddp, *_rows(rank)) * world / n
        loss.backward()
        value = loss.detach() / world
        dist.all_reduce(value)
        if step == 0:
            grads = {
                k: full(p, p.grad).clone().double()
                for k, p in model.named_parameters()
                if p.grad is not None
            }
            norm = float(epn.ep_clip_grad_norm_(params, 1e9, group = group))
        opt.step()
        opt.zero_grad(set_to_none = True)
        if step == 0:
            after = {
                k: full(p, p).clone().double()
                for k, p in model.named_parameters()
                if p.requires_grad
            }
        losses.append(float(value))
    return {"losses": losses, "grads": grads, "after": after, "norm": norm}


def _experts(model):
    return [m for m in model.modules() if type(m).__name__.endswith("Experts")]


def _case_lora(rank, world, tmp, loop):
    from torch.distributed.device_mesh import init_device_mesh

    ink._load("tp_native")
    epn = ink._load("ep_native")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ref_model = ink.build(loop)
    ref = _ref(ref_model, world)
    model = ink.build(loop)
    whole = {k: p.detach().clone() for k, p in model.named_parameters()}
    epn.apply_expert_parallel(model, mesh)
    # Each rank keeps its own slice (DDP's start broadcast must not overwrite it with rank 0's).
    per = _experts(model)[0].gate_up_proj.shape[0]
    sliced = all(
        torch.equal(p, whole[k][rank * per : (rank + 1) * per])
        for k, p in model.named_parameters()
        if k.endswith(("experts.gate_up_proj", "experts.down_proj"))
    )
    state = epn.ep_full_adapter_state_dict(model)
    state_equal = all(torch.equal(state[k], w) for k, w in whole.items() if "lora_" in k)
    got = _ep_train(epn, model, rank, world)
    return {
        "ref": ref,
        "ep": got,
        "per_rank": per,
        "sliced": sliced,
        "state_equal": state_equal,
        "ignored": len(model._ddp_params_and_buffers_to_ignore),
    }


def _case_four_bit_on_load(rank, world, tmp, loop):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    pytest.importorskip("bitsandbytes")
    import test_tp_inkling2_qlora_experts_gloo as q4

    ink._load("tp_native")
    epn = ink._load("ep_native")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ckpt = os.path.join(tmp, "ckpt")
    if rank == 0:
        source = ink.build(loop, lora = False, text_overrides = q4.TEXT)
        source.generation_config.pad_token_id = None
        source.save_pretrained(ckpt, safe_serialization = True)
    dist.barrier()
    q4._quantize_stacks_on_load()
    full = q4._load_4bit(ckpt)
    full_bytes = q4._held_4bit(full)
    want = {
        k: q4._dense(p)
        for k, p in full.named_parameters()
        if k.endswith(("experts.gate_up_proj", "experts.down_proj"))
    }
    ref = _ref(q4._lora(full), world)
    with epn.expert_parallel_loading(mesh) as state:
        model = q4._load_4bit(ckpt)
    stats = dict(model.__dict__.get("_unsloth_tp_load_stats") or {})
    held = q4._held_4bit(model)
    model = q4._lora(model)
    epn.apply_expert_parallel(model, mesh)
    per = None
    sliced = True
    for k, p in model.named_parameters():
        if k.endswith(("experts.gate_up_proj", "experts.down_proj")):
            per = p.quant_state.shape[0]
            key = k.replace("base_model.model.", "")
            sliced &= torch.equal(q4._dense(p), want[key][rank * per : (rank + 1) * per])
    got = _ep_train(epn, model, rank, world)
    return {
        "ref": ref,
        "ep": got,
        "models": len(state["models"]),
        "stats": stats,
        "bytes": (full_bytes, held),
        "per_rank": per,
        "sliced": sliced,
        "compared": len(want),
    }


def _case_env_trainer(rank, world, tmp):
    """UNSLOTH_PARALLEL="ep=2" (set before `import unsloth`) through transformers' Trainer under Unsloth's
    Trainer.__init__: DDP over 2 ranks of 1 row each vs one process on 2 rows, clipping on, then save_model."""
    from test_tp_native_gloo import _import_checkout_unsloth, _Rows, _trainer_args

    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import transformers
    from peft import LoraConfig, get_peft_model

    from unsloth.models import ep_native, tp_env
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)

    class SequentialTrainer(transformers.Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    def peft(model):
        targets = [
            n
            for n, m in model.named_modules()
            if ".language_model.layers." in n and n.endswith(ink.ATTENTION)
        ]
        config = LoraConfig(
            r = 4,
            lora_alpha = 8,
            target_modules = targets,
            target_parameters = ink.EXPERTS,
            init_lora_weights = False,
        )
        torch.manual_seed(1)
        return get_peft_model(model, config)

    model = tp_env.tensor_parallel_env_peft(peft)(ink.build(True, lora = False))
    g = torch.Generator().manual_seed(1)
    data = _Rows(list(torch.randint(1, 250, (6, 9), generator = g)))
    out = os.path.join(tmp, "out")
    # adam eps 1e-4 as test_tp_env_gloo: the default 1e-8 turns fp32 reduction-order noise into weight drift.
    args = _trainer_args(
        out, 2 // world, gradient_accumulation_steps = 1, max_grad_norm = 0.05, adam_epsilon = 1e-4
    )
    trainer = SequentialTrainer(model = model, args = args, train_dataset = data)
    trainer.train()
    trainer.save_model(out)
    logs = [h for h in trainer.state.log_history if "loss" in h]
    from safetensors.torch import load_file

    return {
        "is_ep": ep_native.is_expert_parallel(trainer.model),
        "wrapped": type(trainer.model_wrapped).__name__,
        "losses": [h["loss"] for h in logs],
        "grad_norms": [h["grad_norm"] for h in logs],
        "adapter": load_file(os.path.join(out, "adapter_model.safetensors")),
    }


def _worker(
    rank,
    world,
    tmp,
    case,
    kwargs,
    env = None,
):
    os.environ.update(
        RANK = str(rank), WORLD_SIZE = str(world), LOCAL_RANK = str(rank), LOCAL_WORLD_SIZE = str(world)
    )
    os.environ.update(env or {})
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    if "MASTER_PORT" not in (
        env or {}
    ):  # else torchrun's environment: Unsloth creates the group itself
        dist.init_process_group(  # a file rendezvous: no port to lose to another process
            "gloo",
            store = dist.FileStore(os.path.join(tmp, "rendezvous"), world),
            rank = rank,
            world_size = world,
            timeout = datetime.timedelta(seconds = 300),
        )
    try:
        res = globals()[case](rank, world, tmp, **kwargs)
    except Exception as exc:
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-6000:]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    if dist.is_initialized():
        dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
    env = None,
    **kwargs,
):
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, tmp, case, kwargs, env))
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


def _check(ref, got, tol):
    assert got["losses"][0] == pytest.approx(ref["losses"][0], rel = 1e-6, abs = 1e-6)
    assert got["losses"][1] == pytest.approx(ref["losses"][1], rel = 1e-3)
    assert ink._worst(ref["grads"], got["grads"]) <= tol
    assert ink._worst(ref["after"], got["after"]) <= tol * 1e-1
    assert got["norm"] == pytest.approx(ref["norm"], rel = 1e-5)


@pytest.mark.parametrize("loop", [True, False], ids = ["mini_loop", "small"])
def test_ep_lora_matches_one_process(loop):
    results = _spawn("_case_lora", loop = loop)
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["per_rank"] == 2 and r["sliced"] and r["state_equal"], r
        assert r["ignored"] > 0
        _check(r["ref"], r["ep"], 1e-5)
        assert any(".experts." in k and "lora_" in k for k in r["ref"]["grads"])
        assert any(".q_proj.linear." in k for k in r["ref"]["grads"])
    # Every rank ends with the same replicated weights and the same expert LoRA.
    assert ink._worst(results[0]["ep"]["after"], results[1]["ep"]["after"]) == 0


@pytest.mark.parametrize("loop", [True, False], ids = ["mini_loop", "small"])
def test_ep_four_bit_experts_quantized_on_load(loop):
    results = _spawn("_case_four_bit_on_load", loop = loop)
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        stats = r["stats"]
        # Only the 12 blocks x 2 expert stacks are cut; the attention loads whole.
        assert stats.get("quantized_on_load") and stats["sharded_expert_stacks"] == 24, stats
        assert stats["sharded_weights"] == 0, stats
        full_bytes, held = r["bytes"]
        assert held < full_bytes, r["bytes"]
        assert r["per_rank"] == 2 and r["sliced"] and r["compared"] == 24, r
        # 4-bit Mini: the looped backward amplifies fp32 GEMM-shape rounding (the experts see both ranks' pairs at
        # once) up to ~2e-4 on some runs; a broken exchange or DDP ignore list moves the step-1 loss by >= 1e-3.
        _check(r["ref"], r["ep"], 5e-4)


def test_ep_env_trainer_matches_one_process():
    (ref,) = _spawn("_case_env_trainer", world = 1)
    if "skip" in ref:
        pytest.skip(f"import unsloth failed on this host: {ref['skip']}")
    assert "error" not in ref, ref
    from tests._rendezvous_port import free_port

    # As under torchrun: no process group yet at get_peft_model (Unsloth creates it from the environment).
    env = {"UNSLOTH_PARALLEL": "ep=2", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(free_port())}
    results = _spawn("_case_env_trainer", env = env)
    assert all(r is not None and "error" not in r for r in results), results
    assert not ref["is_ep"]
    for r in results:
        assert r["is_ep"] and "DistributedDataParallel" in r["wrapped"], r["wrapped"]
        assert r["losses"] == pytest.approx(ref["losses"], rel = 1e-5)
        # max_grad_norm 0.05 clips every step: the norm counts each expert once over the ranks.
        assert r["grad_norms"] == pytest.approx(ref["grad_norms"], rel = 1e-5)
        assert min(r["grad_norms"]) > 0.05
    # save_model wrote the 1-GPU adapter: expert LoRA gathered, the same keys, shapes and values.
    want, got = ref["adapter"], results[0]["adapter"]
    assert sorted(got) == sorted(want) and any(".experts." in k for k in got)
    assert (
        ink._worst(
            {k: v.double() for k, v in want.items()}, {k: v.double() for k, v in got.items()}
        )
        < 1e-5
    )


def test_parallel_flag_accepts_ep_over_every_rank():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ep_parallel_env", os.path.join(ink.REPO, "unsloth", "parallel_env.py")
    )
    pe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pe)
    assert pe.translate(pe.parse_parallel_spec("ep=2"), {"WORLD_SIZE": "2"}) == {
        "UNSLOTH_EP_SIZE": "2"
    }
    assert pe.translate(pe.parse_parallel_spec("ep_size=2,dp=2"), {"WORLD_SIZE": "2"}) == {
        "UNSLOTH_EP_SIZE": "2"
    }
    for raw, world, match in (
        ("ep=2,tp=2", "4", "cannot be combined with tp=2"),
        ("ep=2,fsdp=1", "2", "cannot be combined with fsdp=1"),
        ("ep=2", "4", "needs dp = WORLD_SIZE = 2"),
    ):
        with pytest.raises(ValueError, match = match):
            pe.translate(pe.parse_parallel_spec(raw), {"WORLD_SIZE": world})
    with pytest.raises(ValueError, match = "efsdp"):
        pe.parse_parallel_spec("efsdp=2")
