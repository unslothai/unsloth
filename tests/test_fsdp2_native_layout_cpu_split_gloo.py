# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 shard layout (gloo world of 2):

* a large root embedding table split into its own unit (Gemma4 `embed_tokens_per_layer`,
  _large_root_embeddings) is its own dtype group: a 16-bit table next to float32 root parameters
  leaves both homogeneous, with nothing replicated or upcast;
* a CPU mesh (TrainingArguments(use_cpu = True) -> FSDP2Config(device = "cpu")) on a CUDA host plans
  without the GPU's capacity: the per-block sharded layout, as on a CPU-only host."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import datetime  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402
from test_fsdp2_native_trainer_args_gloo import _lora  # noqa: E402
from test_fsdp2_native_guards_gloo import _tiny_llama  # noqa: E402


def _case_split(native):
    torch.manual_seed(0)
    model = _tiny_llama()  # float32 root: embed_tokens, norm, lm_head
    for layer in model.model.layers:
        layer.to(torch.bfloat16)
    hidden = model.config.hidden_size
    model.model.embed_tokens_per_layer = torch.nn.Embedding(4096, hidden).to(torch.bfloat16)
    table = model.model.embed_tokens_per_layer.weight
    root = [model.model.embed_tokens.weight, model.model.norm.weight, model.lm_head.weight]
    cfg = native.FSDP2Config(param_dtype = torch.float32, split_large_root_embeddings = 0.01)
    model = native.shard_model(model, cfg)
    return {
        "extra_units": len(model._unsloth_fsdp2_extra_units),
        "replicated": model._unsloth_fsdp2_replicated,
        "table_dtype": str(model.model.embed_tokens_per_layer.weight.dtype),
        "table_dtensor": type(model.model.embed_tokens_per_layer.weight).__name__,
        "root_dtypes": sorted({str(p.dtype) for p in root}),
        "table_was": str(table.dtype),
    }


def _case_cpu_mesh(native):
    from torch.distributed.device_mesh import init_device_mesh

    model = _lora(1)
    # Built before CUDA is faked below: on a CPU-only torch build a faked CUDA makes new_group ask
    # for NCCL. shard_model's own default mesh is the same CPU mesh (device = "cpu").
    mesh = init_device_mesh("cpu", (2,))
    # A CUDA host whose GPU would fit the model, but the Trainer runs on the CPU.
    torch.cuda.is_available = lambda: True
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_properties = lambda device = None: types.SimpleNamespace(
        total_memory = 80 << 30
    )
    agreed = {}
    original = native._agree_over_ranks

    def record(*args, **kwargs):
        agreed["flags"] = original(*args, **kwargs)
        return agreed["flags"]

    native._agree_over_ranks = record
    cfg = native.FSDP2Config(param_dtype = torch.float32, device = "cpu", mesh = mesh)
    model = native.shard_model(model, cfg)
    fits, replicate_frozen, root_only, prefetch = agreed["flags"]
    return {
        "fits": fits,
        "replicate_frozen": replicate_frozen,
        "root_only": root_only,
        "blocks": model._unsloth_fsdp2_blocks,
        "replicated": model._unsloth_fsdp2_replicated,
    }


def _worker(rank, world, port, out, case):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 30)
    )
    try:
        native = _load_native()
        res = _case_split(native) if case == "split" else _case_cpu_mesh(native)
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(case, timeout = 150):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker, args = (2, _free_port(), out, case), nprocs = 2, join = False, start_method = "spawn"
    )
    deadline = time.time() + timeout
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail(
                    f"{case}: the ranks did not finish within {timeout} s (hung collective)"
                )
    except mp.ProcessExitedException as exc:
        pytest.fail(f"{case}: a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("ROUND13_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_split_root_table_is_its_own_dtype_group():
    pytest.importorskip("transformers")
    for rank, r in _run("split").items():
        assert r["extra_units"] == 1 and r["table_was"] == "torch.bfloat16", (rank, r)
        assert r["replicated"] == 0, (rank, r)  # neither the table nor the float32 root
        assert r["table_dtype"] == "torch.bfloat16" and r["table_dtensor"] == "DTensor", (rank, r)
        assert r["root_dtypes"] == ["torch.float32"], (rank, r)


def test_cpu_mesh_on_a_cuda_host_plans_the_sharded_layout():
    pytest.importorskip("peft")
    for rank, r in _run("cpu_mesh").items():
        assert r["fits"] is False and r["replicate_frozen"] is False, (rank, r)
        assert r["root_only"] is False and r["blocks"] > 0 and r["replicated"] == 0, (rank, r)
