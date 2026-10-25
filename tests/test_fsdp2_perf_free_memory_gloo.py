# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The FSDP2 fit layout checks the memory a rank can still use, not only the GPU's total.

mem_get_info is faked per rank on a gloo CPU world of 2: with plenty free on both ranks the decision
is the total-memory one (fit); when one rank's GPU is crowded (CUDA context, NCCL, another process)
that rank does not fit and both ranks agree on the sharded layout."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_layout_agree_gloo import REPO_ID, _ids, _init, _spawn  # noqa: E402
from test_fsdp2_native_units import _build, _load_native  # noqa: E402


def _worker(rank, world, port, out, free_by_rank):
    dist = _init(rank, world, port)
    try:
        native = _load_native()
        perf = importlib.import_module("unsloth.models.fsdp2_perf")
        perf._device_bytes = lambda device = None: 16 << 30  # a 16 GiB GPU on every rank
        free = free_by_rank[rank]
        perf._device_free_bytes = lambda device = None: free  # faked mem_get_info + cache
        local = perf.plan_fsdp2_layout(_build(REPO_ID, "AutoModelForCausalLM")).as_dict()
        model = _build(REPO_ID, "AutoModelForCausalLM")
        native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32))
        layout = model._unsloth_fsdp2_layout
        mine = {
            "local_fits": local["fits"],
            "fits": layout["fits"],
            "reshard_after_forward": layout["reshard_after_forward"],
            "unit_reshards": [u._unsloth_reshard_after_forward for u in model._unsloth_fsdp2_units],
            "reasons": local["reasons"],
        }
        everyone = [None] * world
        dist.all_gather_object(everyone, mine)
        strip = lambda e: {k: v for k, v in e.items() if k not in ("local_fits", "reasons")}
        consistent = all(strip(e) == strip(everyone[0]) for e in everyone)
        step = None
        if consistent:
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


def test_plenty_free_keeps_the_total_memory_decision():
    pytest.importorskip("transformers")
    (r,) = _spawn(_worker, [15 << 30, 15 << 30])
    assert "error" not in r, r["error"]
    assert [x["local_fits"] for x in r["ranks"]] == [True, True], r
    assert all(x["fits"] is True for x in r["ranks"]), r
    assert r["consistent"] and r["step"] == "ok", r


def test_one_crowded_rank_shards_every_rank():
    pytest.importorskip("transformers")
    # Rank 1: a few KiB left (the tiny model is a few MiB), so model > 0.5 x usable there.
    (r,) = _spawn(_worker, [15 << 30, 1 << 10])
    assert "error" not in r, r["error"]
    ranks = r["ranks"]
    assert [x["local_fits"] for x in ranks] == [True, False], r
    assert any("usable on this GPU" in reason for reason in ranks[1]["reasons"]), r
    assert all(x["fits"] is False for x in ranks), r
    assert all(x["reshard_after_forward"] is True for x in ranks), r
    assert r["consistent"] and r["step"] == "ok", r


def test_device_free_bytes_reads_mem_get_info_plus_cache(monkeypatch):
    perf = importlib.import_module("unsloth.models.fsdp2_perf")
    fake = types.SimpleNamespace(
        mem_get_info = lambda device = None: (3 << 30, 16 << 30),
        memory_reserved = lambda device = None: 2 << 30,
        memory_allocated = lambda device = None: 1 << 30,
        current_device = lambda: 0,
    )
    monkeypatch.setattr(perf.torch, "cuda", fake)
    assert perf._device_free_bytes("cuda:0") == (3 << 30) + (1 << 30)
    assert perf._device_free_bytes("cpu") is None
    fake.mem_get_info = lambda device = None: (_ for _ in ()).throw(RuntimeError("no driver"))
    assert perf._device_free_bytes("cuda:0") is None


def test_raised_fit_fraction_keeps_an_idle_card_fit(monkeypatch):
    # UNSLOTH_FSDP2_FIT_FRACTION = 0.6 and an idle card (85% usable): the free-memory rule scales with
    # the fit fraction, so a model at 55% of the card stays in the fit layout.
    pytest.importorskip("transformers")
    perf = importlib.import_module("unsloth.models.fsdp2_perf")
    model = _build(REPO_ID, "AutoModelForCausalLM")
    probe = perf.plan_fsdp2_layout(model)
    model_bytes = probe.frozen_bytes + probe.trainable_bytes
    device = int(model_bytes / 0.55)
    monkeypatch.setenv("UNSLOTH_FSDP2_FIT_FRACTION", "0.6")
    monkeypatch.delenv("UNSLOTH_FSDP2_FREE_FIT_FRACTION", raising = False)
    monkeypatch.setattr(perf, "_device_bytes", lambda device_ = None: device)
    monkeypatch.setattr(perf, "_device_free_bytes", lambda device_ = None: int(device * 0.85))
    assert perf.plan_fsdp2_layout(model).fits is True
