# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 gradient accumulation modes (UNSLOTH_FSDP2_GA_SYNC), on a gloo CPU world of 2.

Four micro-batches, the first three under the no_sync context the Trainer enters:
* "skip" (default): no reduce-scatter on the non-final micro-batches (one per unit per step);
* "defer": the same gradients bitwise, and fewer all-gathers (none in the fit layout) after the first;
* "each": a reduce-scatter on every micro-batch, the same gradients up to fp32 summation order;
* "auto" (default): "defer" for the fit layout, "skip" for the sharded layout.
For the sharded layout (full fine-tuning) and the fit layout (LoRA, root unit)."""

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

MICRO = 4


def _run(native, perf, counts, layout, mode):
    from torch.distributed.tensor import DTensor

    os.environ["UNSLOTH_FSDP2_GA_SYNC"] = mode  # read on every no_sync entry
    rank = int(os.environ["RANK"])
    torch.manual_seed(0)
    model = _build(REPO_ID, "AutoModelForCausalLM")
    if layout == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear"))
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.02, generator = g)
        cfg = native.FSDP2Config(param_dtype = torch.float32, fit_layout = True)
    else:
        cfg = native.FSDP2Config(param_dtype = torch.float32, fit_layout = False)
    native.shard_model(model, cfg)
    from torch.distributed.fsdp import FSDPModule

    units = sum(isinstance(m, FSDPModule) for m in model.modules())  # blocks + root
    per_micro = []
    for i in range(MICRO):
        before = dict(counts)
        ids = torch.randint(
            3, 1000, (2, 16), generator = torch.Generator().manual_seed(100 * rank + i)
        )
        ctx = native._no_grad_sync(model) if i < MICRO - 1 else torch.enable_grad()
        with ctx:
            model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
        per_micro.append({k: counts[k] - before[k] for k in counts})
    grads = {
        n: (p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad).clone()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    return {
        "units": units,
        "per_micro": per_micro,
        "grads": grads,
        "fits": model._unsloth_fsdp2_layout["fits"],
    }


def _worker(rank, world, port, out, path):
    dist = _init(rank, world, port)
    try:
        native = _load_native()
        perf = importlib.import_module("unsloth.models.fsdp2_perf")
        fpg = importlib.import_module("torch.distributed.fsdp._fully_shard._fsdp_param_group")
        counts = {"ag": 0, "rs": 0}
        orig_ag, orig_rs = fpg.foreach_all_gather, fpg.foreach_reduce

        def ag(*a, **k):
            counts["ag"] += 1
            return orig_ag(*a, **k)

        def rs(*a, **k):
            counts["rs"] += 1
            return orig_rs(*a, **k)

        fpg.foreach_all_gather, fpg.foreach_reduce = ag, rs
        res = {
            (layout, mode): _run(native, perf, counts, layout, mode)
            for layout in ("full", "lora")
            for mode in ("skip", "defer", "each", "auto")
        }
        if rank == 0:  # a file: a large queued payload blocks the child's exit (join deadlock)
            torch.save(res, path)
            out.put("ok")
    except Exception as exc:
        import traceback
        out.put({"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        dist.destroy_process_group()


def test_ga_sync_modes(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    path = str(tmp_path / "ga_sync.pt")
    (payload,) = _spawn(_worker, path)
    assert payload == "ok", payload["error"]
    allres = torch.load(path)
    for layout in ("full", "lora"):
        res = {m: allres[(layout, m)] for m in ("skip", "defer", "each", "auto")}
        for m, r in res.items():
            print("GA_SYNC", layout, m, r["fits"], r["units"], r["per_micro"])
        _check(layout, res)


def _check(layout, res):
    units = res["skip"]["units"]
    assert units >= 1
    if layout == "full":
        assert units > 2 and res["skip"]["fits"] is False, res["skip"]["units"]
    skip, defer, each = (res[m]["per_micro"] for m in ("skip", "defer", "each"))
    units = skip[-1]["rs"]  # units holding trainable parameters (one reduce-scatter each)
    assert units >= (2 if layout == "full" else 1), skip
    # skip: reduce only on the final micro-batch; every micro-batch gathers again.
    assert [m["rs"] for m in skip] == [0] * (MICRO - 1) + [units], skip
    assert all(m["ag"] >= units for m in skip), skip
    # defer: same reductions; after the first micro-batch no unit is gathered for forward (kept from
    # the previous backward). Units resharded after forward (sharded layout) still gather for backward.
    assert [m["rs"] for m in defer] == [0] * (MICRO - 1) + [units], defer
    assert defer[0]["ag"] == skip[0]["ag"], (defer, skip)
    assert all(d["ag"] < s["ag"] for d, s in zip(defer[1:], skip[1:])), (defer, skip)
    if layout == "lora":  # fit layout: kept gathered after forward too, so no gather in between
        assert all(m["ag"] == 0 for m in defer[1:-1]), defer
    # each: one reduce per unit per micro-batch.
    assert [m["rs"] for m in each] == [units] * MICRO, each
    # auto (the default): defer for the fit layout, skip for the sharded one.
    assert res["auto"]["per_micro"] == (defer if layout == "lora" else skip), res["auto"][
        "per_micro"
    ]
    g_skip, g_defer, g_each = (res[m]["grads"] for m in ("skip", "defer", "each"))
    assert g_skip and g_skip.keys() == g_defer.keys() == g_each.keys()
    for n in g_skip:
        assert torch.equal(g_skip[n], g_defer[n]), n
        torch.testing.assert_close(g_each[n], g_skip[n], rtol = 1e-5, atol = 1e-7, msg = n)
