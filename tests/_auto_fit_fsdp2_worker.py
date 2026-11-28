# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""One gloo rank for test_auto_fit_fsdp2_gloo.py: the in-backward optimizer step on a fully_shard model."""

from __future__ import annotations

import importlib.util
import os
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = 5
LR = 1e-2
MAX_NORM = 0.05  # well under this model's gradient norm: clipping engages on every step
LAYERS = 3


def load_auto_fit():
    spec = importlib.util.spec_from_file_location(
        "_fsdp2_auto_fit", ROOT / "unsloth/optimizers/auto_fit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fsdp_group_sizes(model):
    """Parameters per fully_shard group, read from FSDP2's own state."""
    sizes = []
    for module in model.modules():
        get_state = getattr(module, "_get_fsdp_state", None)
        if get_state is None:
            continue
        state = get_state()
        groups = getattr(state, "_fsdp_param_groups", None)
        if groups is None:
            groups = [getattr(state, "_fsdp_param_group", None)]
        sizes += [len(g.fsdp_params) for g in groups if g is not None]
    return sizes


def tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = LAYERS,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 32,
        tie_word_embeddings = False,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config).float()


def batches(world, micro = 1):
    """STEPS x micro micro-batches per rank: [rank][step][micro] -> input ids (2 x 8)."""
    import torch

    g = torch.Generator().manual_seed(1)
    data = torch.randint(0, 64, (STEPS, micro, world, 2, 8), generator = g)
    return [[[data[s, m, r] for m in range(micro)] for s in range(STEPS)] for r in range(world)]


def shard(model, mp = False):
    import torch
    from torch.distributed.fsdp import fully_shard

    kwargs = {}
    if mp:
        from torch.distributed.fsdp import MixedPrecisionPolicy
        kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype = torch.bfloat16, reduce_dtype = torch.float32
        )
    for layer in model.model.layers:
        fully_shard(layer, **kwargs)
    fully_shard(model, **kwargs)  # root group: embeddings, final norm, lm_head
    return model


def loss_of(model, ids):
    return model(input_ids = ids, labels = ids).loss


def full_state(model):
    out = {}
    for name, p in model.named_parameters():
        t = p.full_tensor() if hasattr(p, "full_tensor") else p
        out[name] = t.detach().clone()
    return out


def _group_of(name):
    """The FSDP2 group a parameter is reduced with, in reduce order (last layer first, the root group last)."""
    for i in range(LAYERS):
        if f"layers.{i}." in name:
            return LAYERS - 1 - i
    return LAYERS


def reference(data, ga, clip, calibrate):
    """Single process over the whole global batch (FSDP2 averages over ranks, each rank's loss a mean over the same
    number of tokens). Clipping applies auto_fit's rule per FSDP2 group, in reduce order, in float64: S = this step's
    squares so far, P = the previous step's squares of the same tensors, T = the previous total."""
    import torch

    model = tiny_llama()
    opt = torch.optim.AdamW(model.parameters(), lr = LR, foreach = True)
    world = len(data)
    named = list(model.named_parameters())
    prev_sq, prev_total, norms, losses = None, None, [], []
    for s in range(STEPS):
        total = 0.0
        for m in range(ga):
            for r in range(world):
                loss = loss_of(model, data[r][s][m]) / (world * ga)
                loss.backward()
                total += float(loss)
        sq = {n: float(p.grad.double().square().sum()) for n, p in named}
        full = sum(sq.values())
        norms.append(full**0.5)
        if clip:
            seen = seen_prev = 0.0
            for g in range(LAYERS + 1):
                members = [(n, p) for n, p in named if _group_of(n) == g]
                seen += sum(sq[n] for n, _ in members)
                if prev_sq is not None:
                    seen_prev += sum(prev_sq[n] for n, _ in members)
                for n, p in members:
                    if clip == "per_tensor":
                        norm2 = sq[n]
                    elif clip == "exact" or (prev_total is None and calibrate):
                        norm2 = full
                    elif prev_total is None:
                        norm2 = seen
                    elif clip == "lagged":
                        norm2 = max(seen, prev_total)
                    else:
                        norm2 = max(seen, seen * prev_total / seen_prev) if seen_prev > 0 else seen
                    p.grad.mul_(min(1.0, MAX_NORM / (norm2**0.5 + 1e-6)))
        prev_sq, prev_total = sq, full
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(total)
    return full_state(model), norms, losses


def fsdp_arm(
    af,
    data,
    rank,
    mode,
    ga = 1,
    clip = None,
    sync = "no_sync",
    mp = False,
):
    """mode 'standard' (FSDP2 + optimizer.step(), exact global clip), 'in_backward', 'in_backward_unarmed' (hooks
    inactive, as when FSDP2 skips them under compiled autograd) or 'in_backward_swap' (the optimizer is built over the
    unsharded model and its parameters swapped for the shards afterwards, as accelerate's FSDP2 prepare does). sync
    'no_sync' sets set_requires_gradient_sync(False) on all but the last micro-batch (accelerate / HF Trainer);
    'every' reduces every micro-batch and gates the step with sync_gate (Unsloth's native GA mode 'each')."""
    import torch

    if mode == "in_backward_swap":
        model = tiny_llama()
        base = torch.optim.AdamW(model.parameters(), lr = LR, foreach = True)
    else:
        model = shard(tiny_llama(), mp)
        base = torch.optim.AdamW(model.parameters(), lr = LR, foreach = True)
    stats = dict(peak_live = 0, live_after_backward = [], norms = [], losses = [])
    flag = {"sync": True}
    opt = base
    if mode.startswith("in_backward"):
        kwargs = dict(sync_gate = lambda: flag["sync"]) if sync == "every" else {}
        opt = af.InBackwardOptimizer(
            base,
            max_grad_norm = MAX_NORM if clip else 0.0,
            model = model,
            clip = clip or "estimate",
            **kwargs,
        )
        if ga > 1 and clip and clip != "per_tensor":
            # As the trainer path: a multi-micro-batch step cannot be measured twice, so the first step (every
            # step for exact) is stepped at optimizer.step() with the exact norm.
            opt.calibrate = False
            opt.defer = "all" if clip == "exact" else "first"
        if mode == "in_backward_swap":
            names = {id(p): n for n, p in model.named_parameters()}
            groups = [[names[id(p)] for p in g["params"]] for g in base.param_groups]
            shard(model, mp)
            sharded = dict(model.named_parameters())
            for g, ns in zip(base.param_groups, groups):
                g["params"] = [sharded[n] for n in ns]
    params = list(model.parameters())

    def observe(_p):
        live = [q for q in params if q.grad is not None]
        stats["peak_live"] = max(stats["peak_live"], len(live))

    def run(micro, armed):
        if sync == "no_sync" and ga > 1:
            model.set_requires_gradient_sync(micro == ga - 1)
        flag["sync"] = micro == ga - 1
        with armed:
            loss = loss_of(model, data[rank][s][micro]) / ga
            loss.backward()
        return float(loss)

    hooked = False
    for s in range(STEPS):
        if mode.startswith("in_backward") and not hooked and mode != "in_backward_unarmed":
            opt._refresh()  # the swapped parameters get their hooks (armed() does this too)
            params = list(model.parameters())
            for (
                p
            ) in params:  # registered after the optimizer's hooks: sees the state after each flush
                p.register_post_accumulate_grad_hook(observe)
            hooked = True
        total = 0.0
        if mode == "in_backward" or mode == "in_backward_swap":
            if opt.wants_measure():
                run(0, opt.measuring())
                assert all(q.grad is None for q in params)
            for m in range(ga):
                total += run(m, opt.armed())
        else:
            for m in range(ga):
                total += run(m, _null())
        stats["live_after_backward"].append(sum(1 for q in params if q.grad is not None))
        if mode == "standard":
            local = torch.stack(
                [q.grad._local_tensor.double().square().sum() for q in params]
            ).sum()
            torch.distributed.all_reduce(local)
            norm = local.sqrt()
            if clip:
                coef = torch.clamp(MAX_NORM / (norm + 1e-6), max = 1.0).float()
                torch._foreach_mul_([q.grad._local_tensor for q in params], coef)
            stats["norms"].append(float(norm))
            opt.step()
        else:
            opt.flush()
            stats["norms"].append(float(opt.total_norm()))
            opt.step()
            stats["late_steps"] = opt.late_steps
            stats["measured_steps"] = opt.measured_steps
        opt.zero_grad(set_to_none = True)
        t = torch.tensor(total)
        torch.distributed.all_reduce(t)
        stats["losses"].append(float(t) / torch.distributed.get_world_size())
    stats["total_params"] = len(params)
    stats["groups"] = fsdp_group_sizes(model)
    return full_state(model), stats


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def max_diff(a, b):
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


def run_rank(rank, port, queue, cases):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        LOCAL_RANK = str(rank),
        WORLD_SIZE = "2",
        CUDA_VISIBLE_DEVICES = "",
    )
    import torch

    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", rank = rank, world_size = 2)
    out = {}
    af = load_auto_fit()
    try:
        # The planner's whole-weight budget read from FSDP2's state: the root group stays whole (never resharded
        # after the forward), a block that reshards is whole one at a time (+ the prefetched one), one that does
        # not reshard stays whole too.
        from torch.distributed.fsdp import fully_shard

        model = tiny_llama()
        full = {n: p.numel() * p.element_size() for n, p in model.named_parameters()}
        fully_shard(model.model.layers[0], reshard_after_forward = False)
        for layer in model.model.layers[1:]:
            fully_shard(layer)
        fully_shard(model)
        block = sum(v for n, v in full.items() if "layers.1." in n)
        out["_gathered"] = dict(
            got = af._fsdp_gathered(None, model),
            want = (sum(v for n, v in full.items() if "layers." not in n) + block, 2 * block),
        )
    except AttributeError as exc:
        if "record_event" not in str(exc):
            raise
    # Every rank plans with the smallest free memory of any rank, so all pick the same tier.
    out["_min_over_ranks"] = af._min_over_ranks(1000 + 500 * rank, torch.device("cpu"))
    for name, (ga, clip, sync, mp, arm) in cases.items():
        try:
            data = batches(2, ga)
            calibrate = (
                True  # measured (GA = 1) or stepped at optimizer.step() (GA > 1): exact first step
            )
            std, std_stats = fsdp_arm(
                af, data, rank, "standard", ga, clip if clip == "exact" else None, sync, mp
            )
            inb, inb_stats = fsdp_arm(af, data, rank, arm, ga, clip, sync, mp)
            result = dict(
                inb_vs_std = max_diff(inb, std),
                loss_vs_std = max(
                    abs(a - b) for a, b in zip(inb_stats["losses"], std_stats["losses"])
                ),
                norm_vs_std = max(
                    abs(a - b) / b for a, b in zip(inb_stats["norms"], std_stats["norms"])
                ),
                std_stats = {k: v for k, v in std_stats.items() if k != "norms"},
                inb_stats = inb_stats,
            )
            if not mp:
                ref, ref_norms, ref_losses = reference(data, ga, clip, calibrate)
                result.update(
                    inb_vs_ref = max_diff(inb, ref),
                    moved = max_diff(ref, full_state(tiny_llama())),
                    norm_diff = max(abs(a - b) / b for a, b in zip(inb_stats["norms"], ref_norms)),
                    loss_diff = max(abs(a - b) for a, b in zip(inb_stats["losses"], ref_losses)),
                )
                if clip in (None, "exact"):
                    result["std_vs_ref"] = max_diff(std, ref)
            if name == "adamw":
                late, late_stats = fsdp_arm(af, data, rank, "in_backward_unarmed", ga, clip, sync)
                result.update(
                    unarmed_vs_ref = max_diff(late, ref), unarmed_late_steps = late_stats["late_steps"]
                )
            out[name] = result
        except AttributeError as exc:
            if "record_event" not in str(exc):
                out[name] = {"error": traceback.format_exc()}
                continue
            # Older torch's FSDP2 needs accelerator streams (no CPU support): the standard arm fails alike.
            out = {"skip": f"FSDP2 on CPU unsupported by torch {torch.__version__}"}
            break
        except Exception:
            out[name] = {"error": traceback.format_exc()}
    queue.put((rank, out))
    torch.distributed.destroy_process_group()
