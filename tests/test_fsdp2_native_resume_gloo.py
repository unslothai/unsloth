# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 checkpoint resume on a gloo CPU world of 2 (fsdp2_native.py loaded on its own, no GPU):
train 2 steps, save the model (as Trainer.save_model writes it) and the optimizer with
native_save_optimizer, resume a freshly built and sharded copy with native_load_model_checkpoint +
native_load_optimizer, then 2 more steps on both: losses and parameters must be identical. Negative
control: rank 0's plain `optimizer.state_dict()` (the Trainer's own save) holds only rank 0's shards."""

import os
import sys

import pytest
import torch

# Spawned gloo ranks re-import this module: keep them on the CPU (deterministic on GPU hosts; CUDA
# reads the mask at its lazy first use, after this).
if __import__("multiprocessing").parent_process() is not None or getattr(
    __import__("multiprocessing").current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

REPO_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _model(lora, seed):
    import transformers

    config = transformers.AutoConfig.from_pretrained(REPO_ID)
    config.pad_token_id = None  # tiny repo's -1 fails save_pretrained's config check
    torch.manual_seed(0)  # identical base weights on every build
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = None
    if lora:
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(seed)
        model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear"))
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed)
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.02, generator = g)
    elif seed:
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed)
            for p in model.parameters():
                p.add_(torch.randn(p.shape, generator = g) * 0.01)
    return model


def _batch(step, rank):
    g = torch.Generator().manual_seed(100 * step + rank)
    ids = torch.randint(3, 1000, (2, 16), generator = g)
    return {"input_ids": ids, "labels": ids.clone(), "use_cache": False}


def _step(model, opt, step, rank):
    loss = model(**_batch(step, rank)).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none = True)
    return loss.detach().clone()


def _full_params(model):
    from torch.distributed.tensor import DTensor
    return {
        n: (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def _worker(rank, world, port, lora, tmp, out):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
    )
    import torch.distributed as dist

    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        native = _load_native()
        cfg = native.FSDP2Config(param_dtype = torch.float32)

        def _new(seed):
            m = native.shard_model(_model(lora, seed), cfg)
            o = torch.optim.AdamW(
                [p for p in m.parameters() if p.requires_grad], lr = 1e-2, foreach = False
            )
            return m, o

        model, opt = _new(1)
        for step in (1, 2):
            _step(model, opt, step, rank)
        ckpt = os.path.join(tmp, "ckpt")
        state = native.native_full_state_dict(model, trainable_only = lora)
        opt_path = os.path.join(ckpt, "optimizer.pt")
        plain_path = os.path.join(ckpt, "optimizer_rank0_plain.pt")
        if rank == 0:
            os.makedirs(ckpt, exist_ok = True)
            model.save_pretrained(ckpt, state_dict = state, safe_serialization = True)
            torch.save(opt.state_dict(), plain_path)  # what Trainer saves without the hook
        native.native_save_optimizer(model, opt, opt_path, should_save = rank == 0)
        dist.barrier()
        want = [_step(model, opt, step, rank).item() for step in (3, 4)]
        want_params = _full_params(model)

        resumed, ropt = _new(2)  # different fresh trainable values: the load must replace them
        native.native_load_model_checkpoint(resumed, ckpt)
        native.native_load_optimizer(resumed, ropt, opt_path)
        got = [_step(resumed, ropt, step, rank).item() for step in (3, 4)]
        got_params = _full_params(resumed)
        param_diff = max((got_params[n] - want_params[n]).abs().max().item() for n in want_params)

        control, copt = _new(2)
        native.native_load_model_checkpoint(control, ckpt)
        try:
            copt.load_state_dict(torch.load(plain_path, weights_only = False))
            ctrl = [_step(control, copt, step, rank).item() for step in (3, 4)]
            ctrl_error = None
        except Exception as exc:
            ctrl, ctrl_error = None, f"{type(exc).__name__}: {str(exc)[:200]}"
        ctrl_flag = torch.tensor([0 if ctrl is None else int(ctrl != want)], dtype = torch.int64)
        dist.all_reduce(ctrl_flag)
        out.put(
            (
                rank,
                {
                    "want": want,
                    "got": got,
                    "param_diff": param_diff,
                    "ctrl": ctrl,
                    "ctrl_error": ctrl_error,
                    "ctrl_differs_any_rank": int(ctrl_flag.item()),
                },
            )
        )
    except Exception as exc:
        import traceback
        out.put((rank, {"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("lora", [True, False], ids = ["lora", "full"])
def test_native_resume_replays_exactly(lora, tmp_path):
    pytest.importorskip("transformers")
    if lora:
        pytest.importorskip("peft")
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target = _worker, args = (r, 2, port, lora, str(tmp_path), out)) for r in range(2)
    ]
    for p in procs:
        p.start()
    results = dict(out.get(timeout = 600) for _ in procs)
    for p in procs:
        p.join(timeout = 60)
    for rank, r in results.items():
        print("RESUME_RESULT", rank, r)
        assert "error" not in r, r["error"]
        assert r["got"] == r["want"], (rank, r)
        assert r["param_diff"] == 0.0, (rank, r)
        # Rank 0's plain optimizer.state_dict() is not a valid resume for a sharded model.
        assert r["ctrl_error"] is not None or r["ctrl_differs_any_rank"] > 0, (rank, r)
