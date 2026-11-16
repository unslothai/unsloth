# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 keeps large frozen, untied input / output embeddings out of the root unit (which never
reshards, so its gather of both OOMed a T4 on Qwen3.8-27B 4-bit)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

transformers = pytest.importorskip("transformers")

VOCAB = 1024
HIDDEN = 32


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _model(tied = False):
    config = transformers.Qwen3Config(
        vocab_size = VOCAB,
        hidden_size = HIDDEN,
        intermediate_size = 64,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        tie_word_embeddings = tied,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config).float()
    for name, p in model.named_parameters():
        # LoRA-like: only the MLP projections train; embeddings, head, attention and norms are frozen.
        p.requires_grad_(".mlp." in name)
    return model


def _group_bytes(module):
    """Unsharded bytes of `module`'s FSDP2 parameter group."""
    state = module._get_fsdp_state()
    group = getattr(state, "_fsdp_param_group", None)
    total = 0
    for fp in getattr(group, "fsdp_params", None) or ():
        dtype = getattr(fp, "param_dtype", None) or fp.sharded_param.dtype
        total += fp._orig_size.numel() * torch.empty((), dtype = dtype).element_size()
    return total


def test_frozen_untied_io_tables_are_picked_by_size():
    native = _load_native()
    model = _model()
    units = native.find_shard_units(model)
    nbytes = 2 * VOCAB * HIDDEN * 4
    embed, head = model.get_input_embeddings(), model.get_output_embeddings()
    assert native._frozen_root_io_tables(model, units, nbytes, 0.5) == [embed, head]
    assert (
        native._frozen_root_io_tables(model, units, nbytes * 10, 0.5) == []
    )  # small next to the device
    assert native._frozen_root_io_tables(model, units, 0, 0.5) == []  # no device size (CPU mesh)
    assert native._frozen_root_io_tables(model, units, nbytes, 0) == []


def test_trainable_or_tied_io_tables_stay_in_the_root(monkeypatch):
    native = _load_native()
    monkeypatch.setenv("UNSLOTH_FSDP2_REPLICATE_IO_TABLES", "1")
    tied = _model(tied = True)
    assert native._frozen_root_io_tables(tied, native.find_shard_units(tied), 0, 0.15) == []
    trained = _model()
    trained.lm_head.weight.requires_grad_(True)
    assert native._frozen_root_io_tables(trained, native.find_shard_units(trained), 0, 0.15) == []
    monkeypatch.setenv("UNSLOTH_FSDP2_REPLICATE_IO_TABLES", "0")
    model = _model()
    assert native._frozen_root_io_tables(model, native.find_shard_units(model), 1, 1e-9) == []


def test_cpu_lookup_returns_on_the_ids_device():
    native = _load_native()
    table = torch.nn.Embedding(16, 4)
    native._serve_lookups_from_cpu(table)
    ids = torch.tensor([[1, 2, 3]])
    out = table(ids)
    assert table.weight.device.type == "cpu" and out.device == ids.device
    assert torch.equal(out, torch.nn.functional.embedding(ids, table.weight))


def _worker(
    rank,
    world,
    port,
    out,
    replicate,
    gather = "float32",
):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
        UNSLOTH_FSDP2_REPLICATE_IO_TABLES = replicate,
    )
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 300)
    )
    try:
        from torch.distributed.tensor import DTensor

        native = _load_native()
        ref, model = _model(), _model()
        config = native.FSDP2Config(
            param_dtype = getattr(torch, gather), device = "cpu", fit_layout = False
        )
        native.shard_model(model, config)
        units = list(model._unsloth_fsdp2_units)
        unit_bytes = [_group_bytes(u) for u in units]
        root_bytes = _group_bytes(model)
        embed, head = model.get_input_embeddings(), model.get_output_embeddings()
        sharded = {
            "embed": isinstance(embed.weight, DTensor),
            "head": isinstance(head.weight, DTensor),
        }
        torch.manual_seed(1)
        ids = torch.randint(0, VOCAB, (2, 16))
        if gather != "float32":
            # float32 tables, 16-bit gathers: the head must still see what the root unit's cast gave it.
            logits = model(input_ids = ids, use_cache = False).logits
            out.put(
                {
                    "rank": rank,
                    "sharded": sharded,
                    "logits_dtype": str(logits.dtype),
                    "finite": bool(torch.isfinite(logits.float()).all()),
                }
            )
            return
        loss = model(input_ids = ids, labels = ids, use_cache = False).loss
        loss.backward()
        ref_loss = ref(input_ids = ids, labels = ids, use_cache = False).loss
        ref_loss.backward()
        ref_params = dict(ref.named_parameters())
        diff = 0.0
        missing = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                missing.append(name)
                continue
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            diff = max(diff, float((g - ref_params[name].grad).abs().max()))
        out.put(
            {
                "rank": rank,
                "unit_bytes": unit_bytes,
                "root_bytes": root_bytes,
                "sharded": sharded,
                "loss_diff": abs(float(loss) - float(ref_loss)),
                "grad_diff": diff,
                "missing": missing,
            }
        )
    except Exception as exc:
        import traceback
        out.put({"error": f"rank {rank} {type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        dist.destroy_process_group()


def _run(replicate, gather = "float32"):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        _worker,
        args = (2, _free_port(), out, replicate, gather),
        nprocs = 2,
        join = True,
        start_method = "spawn",
    )
    results = [out.get(timeout = 5) for _ in range(2)]
    for r in results:
        assert "error" not in r, r["error"]
    return results


def test_root_gathers_no_io_table_when_replicated_and_matches_one_process():
    tables = 2 * VOCAB * HIDDEN * 4
    for r in _run("1"):
        assert r["sharded"] == {"embed": False, "head": False}, r
        # The root keeps only the final norm; the largest gather is now a block.
        assert r["root_bytes"] < tables // 8, r
        assert max([r["root_bytes"], *r["unit_bytes"]]) < tables // 2, r
        assert not r["missing"], r
        assert r["loss_diff"] < 1e-5 and r["grad_diff"] < 1e-5, r


def test_root_still_gathers_io_tables_when_disabled():
    for r in _run("0"):
        assert r["sharded"] == {"embed": True, "head": True}, r
        assert r["root_bytes"] >= 2 * VOCAB * HIDDEN * 4, r
        assert r["loss_diff"] < 1e-5 and r["grad_diff"] < 1e-5, r


def _gather_worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    import datetime
    import importlib

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        from torch.distributed.device_mesh import init_device_mesh

        _load_native()
        load = importlib.import_module("unsloth.models.fsdp2_load")
        load._GATHER_SLICE_BYTES = 3 * 5 * 2  # 3 rows of a (7, 5) bfloat16 tensor per slice
        mesh = init_device_mesh("cpu", (world,))
        full = torch.arange(7 * 5, dtype = torch.float32).reshape(7, 5).to(torch.bfloat16)
        size = -(-7 // world)
        local = full[rank * size : (rank + 1) * size].clone()
        info = {"shape": (7, 5), "world": world}
        got = load._full_from_chunks(local, info, mesh, torch.float32, torch.device("cpu"))
        same = load._full_from_chunks(local, info, mesh, torch.bfloat16)
        out.put(
            {
                "exact": bool(torch.equal(got, full.float())) and bool(torch.equal(same, full)),
                "dtype": str(got.dtype),
            }
        )
    except Exception as exc:
        import traceback
        out.put({"error": f"rank {rank} {type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        dist.destroy_process_group()


def test_sliced_chunk_gather_is_exact():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        _gather_worker, args = (2, _free_port(), out), nprocs = 2, join = True, start_method = "spawn"
    )
    for r in [out.get(timeout = 5) for _ in range(2)]:
        assert "error" not in r, r["error"]
        assert r["exact"] and r["dtype"] == "torch.float32", r


def test_io_tables_stay_in_the_root_when_the_gather_casts_them():
    # Replicating would skip the root's cast and feed bfloat16 hidden states into a float32 lm_head.
    for r in _run("1", "bfloat16"):
        assert r["sharded"] == {"embed": True, "head": True}, r
        assert r["logits_dtype"] == "torch.bfloat16" and r["finite"], r
