# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 layout decisions every rank must share, on a gloo CPU world of 2.

* Unequal device memory: rank 0 "fits", rank 1 does not. The ranks agree on fits = False, and the
  reshard_after_forward derived from it must be the same on both (a rank keeping its blocks gathered
  while the other reshards issues different backward collectives: a hang on GPUs).
* torch without fully_shard(ignored_params) (2.6): the automatic fit layout falls back to sharding the
  frozen weights; an explicit replication request keeps the capability error."""

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
from test_fsdp2_native_units import _build, _load_native  # noqa: E402

REPO_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _init(
    rank,
    world,
    port,
    extra_env = None,
):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
        **(extra_env or {}),
    )
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    return dist


def _ids(seed):
    return torch.randint(3, 1000, (2, 16), generator = torch.Generator().manual_seed(seed))


def _unequal_worker(rank, world, port, out):
    dist = _init(rank, world, port)
    try:
        native = _load_native()
        perf = importlib.import_module("unsloth.models.fsdp2_perf")
        # Rank 0 sees a huge device (the model fits), rank 1 a tiny one (it does not).
        big = 1 << 40 if rank == 0 else 1 << 10
        perf._device_bytes = lambda device = None: big
        local = perf.plan_fsdp2_layout(_build(REPO_ID, "AutoModelForCausalLM")).as_dict()
        model = _build(REPO_ID, "AutoModelForCausalLM")  # full fine-tuning
        native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32))
        layout = model._unsloth_fsdp2_layout
        reshards = [u._unsloth_reshard_after_forward for u in model._unsloth_fsdp2_units]
        mine = {
            "local_fits": local["fits"],
            "fits": layout["fits"],
            "reshard_after_forward": layout["reshard_after_forward"],
            "prefetch": layout["prefetch"],
            "unit_reshards": reshards,
        }
        everyone = [None] * world
        dist.all_gather_object(everyone, mine)
        consistent = all(e == {**everyone[0], "local_fits": e["local_fits"]} for e in everyone)
        step = None
        if consistent:  # mismatched layouts would hang the backward: only run it when they agree
            ids = _ids(rank)
            model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
            step = "ok"
        if rank == 0:
            out.put({"ranks": everyone, "consistent": consistent, "step": step})
    except Exception as exc:
        import traceback
        out.put({"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        dist.destroy_process_group()


def _spawn(worker, *args):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        worker, args = (2, _free_port(), out, *args), nprocs = 2, join = True, start_method = "spawn"
    )
    results = []
    while not out.empty():
        results.append(out.get(timeout = 5))
    return results


def test_unequal_capacity_ranks_agree_on_reshard():
    pytest.importorskip("transformers")
    (r,) = _spawn(_unequal_worker)
    assert "error" not in r, r["error"]
    print("AGREE_RESULT", r)
    ranks = r["ranks"]
    assert [x["local_fits"] for x in ranks] == [True, False], r  # the setup really differs per rank
    assert all(x["fits"] is False for x in ranks), r
    assert r["consistent"], r
    # Not fitting everywhere: blocks reshard after forward (all but the last, which never does).
    assert all(x["reshard_after_forward"] is True for x in ranks), r
    assert all(x["prefetch"] is False for x in ranks), r
    assert r["step"] == "ok", r


def _no_ignored_worker(rank, world, port, out, mode):
    extra = {"UNSLOTH_FSDP2_REPLICATE_FROZEN": "1"} if mode == "env" else {}
    dist = _init(rank, world, port, extra)
    try:
        native = _load_native()
        from peft import LoraConfig, get_peft_model
        from torch.distributed.tensor import DTensor

        native._supports_ignored_params = lambda fully_shard: False  # torch 2.6's fully_shard
        model = get_peft_model(
            _build(REPO_ID, "AutoModelForCausalLM"),
            LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear"),
        )
        cfg = native.FSDP2Config(
            param_dtype = torch.float32,
            fit_layout = True,
            shard_frozen = False if mode == "config" else None,
        )
        try:
            native.shard_model(model, cfg)
        except RuntimeError as exc:
            out.put((rank, {"raised": str(exc)}))
            return
        layout = model._unsloth_fsdp2_layout
        ids = _ids(rank)
        model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
        out.put(
            (
                rank,
                {
                    "layout": layout,
                    "frozen_sharded": all(
                        isinstance(p, DTensor) for p in model.parameters() if not p.requires_grad
                    ),
                    "lora_grads": all(
                        p.grad is not None for p in model.parameters() if p.requires_grad
                    ),
                },
            )
        )
    except Exception as exc:
        import traceback
        out.put((rank, {"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["auto", "config", "env"])
def test_fit_layout_without_ignored_params(mode):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    results = dict(_spawn(_no_ignored_worker, mode))
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("NO_IGNORED_RESULT", mode, rank, r)
        assert "error" not in r, r["error"]
        if mode == "auto":
            assert "raised" not in r, r
            assert r["layout"]["fits"] is True, r
            assert r["layout"]["replicate_frozen"] is False, r
            assert r["layout"]["root_only"] is False, r
            assert r["frozen_sharded"] and r["lora_grads"], r
        else:  # explicit replication: the capability error stays
            assert "ignored_params" in r.get("raised", ""), r
