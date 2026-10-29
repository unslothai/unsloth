# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 fit layout + gradient checkpointing, on a gloo CPU world of 2: the backward recompute must
not fire the explicit forward prefetch (it re-gathered block i+1 right after its backward resharded it).
Same gradients bitwise; fewer all-gathers under reentrant checkpointing (Unsloth's), unchanged counts under
non-reentrant (FSDP already skips its pre-forward there); the
UNSLOTH_FSDP2_RECOMPUTE_PREFETCH=1 kill switch restores torch's behaviour (the counter can fail)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_layout_agree_gloo import REPO_ID, _init, _spawn  # noqa: E402
from test_fsdp2_native_units import _build, _load_native  # noqa: E402

STEPS = 2


def _run(native, counts, reentrant, kill_switch):
    from torch.distributed.tensor import DTensor

    if kill_switch:
        os.environ["UNSLOTH_FSDP2_RECOMPUTE_PREFETCH"] = "1"
    else:
        os.environ.pop("UNSLOTH_FSDP2_RECOMPUTE_PREFETCH", None)
    rank = int(os.environ["RANK"])
    torch.manual_seed(0)
    model = _build(REPO_ID, "AutoModelForCausalLM")  # full fine-tuning, 2 blocks
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": reentrant})
    model.train()
    if reentrant:  # reentrant checkpointing only recomputes when its input requires grad
        model.enable_input_require_grads()
    native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32, fit_layout = True))
    layout = model._unsloth_fsdp2_layout
    opt = torch.optim.SGD(model.parameters(), lr = 0.1)
    per_step, grads = [], None
    for step in range(STEPS):
        before = counts["ag"]
        ids = torch.randint(
            3, 1000, (2, 16), generator = torch.Generator().manual_seed(10 * rank + step)
        )
        model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
        per_step.append(counts["ag"] - before)
        if step == 0:
            grads = {
                n: (p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad).clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        opt.step()
        opt.zero_grad(set_to_none = True)
    params = {
        n: (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone()
        for n, p in model.named_parameters()
    }
    return {
        "per_step": per_step,
        "grads": grads,
        "params": params,
        "fits": layout["fits"],
        "prefetch": layout["prefetch"],
    }


def _worker(rank, world, port, out, path):
    dist = _init(rank, world, port)
    try:
        native = _load_native()
        importlib.import_module("unsloth.models.fsdp2_perf")
        fpg = importlib.import_module("torch.distributed.fsdp._fully_shard._fsdp_param_group")
        counts = {"ag": 0}
        orig_ag = fpg.foreach_all_gather

        def ag(*a, **k):
            counts["ag"] += 1
            return orig_ag(*a, **k)

        fpg.foreach_all_gather = ag
        res = {
            (reentrant, kill): _run(native, counts, reentrant, kill)
            for reentrant in (False, True)
            for kill in (True, False)
        }
        if rank == 0:  # a file: a large queued payload blocks the child's exit (join deadlock)
            torch.save(res, path)
            out.put("ok")
    except Exception as exc:
        import traceback
        out.put({"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        dist.destroy_process_group()


def test_recompute_does_not_refire_forward_prefetch(tmp_path):
    pytest.importorskip("transformers")
    path = str(tmp_path / "recompute_prefetch.pt")
    (payload,) = _spawn(_worker, path)
    assert payload == "ok", payload["error"]
    res = torch.load(path)
    for reentrant in (False, True):
        base, head = res[(reentrant, True)], res[(reentrant, False)]
        print(
            "RECOMPUTE_PREFETCH",
            "reentrant" if reentrant else "nonreentrant",
            "base",
            base["per_step"],
            "head",
            head["per_step"],
        )
        assert base["fits"] and base["prefetch"], base
        if reentrant:
            # Reentrant checkpointing runs the forward under no_grad (no FSDP pre-backward hook), so the
            # recompute's pre-forward runs in full and prefetches the block after it. The kill switch
            # reproduces that waste (the counter can fail); the default removes it.
            assert all(h < b for h, b in zip(head["per_step"], base["per_step"])), (
                base["per_step"],
                head["per_step"],
            )
        else:
            # Non-reentrant: FSDP's pre-forward already returns early in PRE_BACKWARD; nothing to remove.
            assert head["per_step"] == base["per_step"], (base["per_step"], head["per_step"])
        assert base["grads"].keys() == head["grads"].keys() and base["grads"]
        for n in base["grads"]:
            assert torch.equal(base["grads"][n], head["grads"][n]), n
        for n in base["params"]:
            assert torch.equal(base["params"][n], head["params"][n]), n
