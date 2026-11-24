# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Parallel preflight guards, on CPU (gloo ranks or no process group):

- several pipeline stages per rank over a root mesh with a data parallel dimension under any name (dp,
  dp_shard, dp_replicate) is refused before the model is cut and before pipeline_sharded_loading reads;
- the native FSDP2 and pipeline parallel loaders refuse transformers' own tp_plan / tp_size / ... as the
  tensor parallel loader does, before any load runs;
- eval_loss on a V schedule (ZBV, DualPipeV) runs forward only and matches the one-process loss;
- the launcher's mesh refuses a CUDA layout on a PyTorch build without NCCL before init_process_group.

Legal layouts (pp x dp with one stage per rank, pp 2 x dp 1 with two stages, pp x tp / pp x cp) pass the
same guards unchanged."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pp_native_gloo import (  # noqa: E402
    LLAMA,
    REPO,
    _free_port,
    _load,
    _torch_pipelining_error,
    batch,
    build,
    ppn,
)

V_SCHEDULES = ("zbvzerobubble", "dualpipev")


def _uncut(model):
    layers = ppn._find_parts(model).layers
    return not ppn.is_pipeline_parallel(model) and not any(
        type(layer).__name__ == "_PPPassThrough" for layer in layers
    )


def _error(fn):
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        return str(exc)
    return None


def _case_mesh4(rank, world, tmp):
    """4 ranks: pp 2 x a data parallel dimension of 2 under the launcher's names."""
    from torch.distributed.device_mesh import init_device_mesh

    ppl = _load("pp_load")
    out = {}
    for dp_name in ("dp_shard", "dp_replicate", "dp"):
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names = ("pp", dp_name))
        model = build(LLAMA, layers = 4)
        out[dp_name] = _error(
            lambda: ppn.apply_pipeline_parallel(
                model, mesh["pp"], n_microbatches = 4, schedule = "interleaved1f1b"
            )
        )
        out[dp_name + ":uncut"] = _uncut(model)

        def enter():
            with ppl.pipeline_sharded_loading(mesh["pp"], schedule = "interleaved1f1b"):
                pass

        out[dp_name + ":load"] = _error(enter)
        # One stage per rank composes with dp: accepted and cut.
        single = ppn.apply_pipeline_parallel(
            build(LLAMA, layers = 4), mesh["pp"], n_microbatches = 2, schedule = "gpipe"
        )
        out[dp_name + ":single"] = single.stages_per_rank
    # Model parallel dimensions beside pp are not data parallel: two stages per rank are cut as before.
    for other in ("tp", "cp"):
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names = ("pp", other))
        model = build(LLAMA, layers = 4)
        out[other] = _error(
            lambda: ppn.apply_pipeline_parallel(
                model, mesh["pp"], n_microbatches = 4, schedule = "interleaved1f1b"
            )
        )
        out[other + ":stages"] = getattr(ppn.pipeline_stage_of(model), "stages_per_rank", None)
    return out


def _case_eval2(rank, world, tmp):
    """2 ranks: eval_loss on V schedules and on pp 2 x dp 1 with two stages per rank."""
    from torch.distributed.device_mesh import init_device_mesh

    out = {}
    ref = build(LLAMA, layers = 4)
    ids, labels = batch(ref)
    with torch.no_grad():
        out["ref"] = float(ref(input_ids = ids, labels = labels).loss)
    layouts = [(s, ("pp",), (world,)) for s in V_SCHEDULES]
    layouts.append(("interleaved1f1b", ("pp", "dp"), (world, 1)))
    for schedule, names, shape in layouts:
        key = f"{schedule}:{'x'.join(names)}"
        mesh = init_device_mesh("cpu", shape, mesh_dim_names = names)
        model = build(LLAMA, layers = 4)
        try:
            pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 4, schedule = schedule)
        except (ValueError, RuntimeError) as exc:
            out[key] = {"apply": str(exc)}
            continue
        res = {"apply": None}
        try:
            res["eval"] = float(pp.eval_loss(ids, labels))
            res["train"] = float(pp.step(ids, labels))
            res["eval_after"] = float(pp.eval_loss(ids, labels))
            res["error"] = None
        except Exception as exc:  # noqa: BLE001 - torch asserts with AssertionError
            res["error"] = f"{type(exc).__name__}: {exc}"
        out[key] = res
    return out


CASES = {"mesh4": _case_mesh4, "eval2": _case_eval2}


def _worker(rank, world, port, case, tmp):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        res = CASES[case](rank, world, tmp)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(case, world):
    if _torch_pipelining_error():
        pytest.skip(_torch_pipelining_error())
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp)) for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


def test_multi_stage_pp_x_dp_refused_under_every_dp_name_before_cut_and_load():
    for res in _spawn("mesh4", 4):
        for dp_name in ("dp_shard", "dp_replicate", "dp"):
            assert "several stages per rank is not supported" in (res[dp_name] or ""), (
                dp_name,
                res,
            )
            assert repr(dp_name) in res[dp_name], res[dp_name]
            assert res[dp_name + ":uncut"], (dp_name, res)
            assert "several stages per rank is not supported" in (res[dp_name + ":load"] or ""), (
                dp_name,
                res,
            )
            assert res[dp_name + ":single"] == 1, (dp_name, res)
        for other in ("tp", "cp"):
            assert res[other] is None and res[other + ":stages"] == 2, (other, res)


def _v_supported():
    return importlib.import_module("unsloth.models.pp_schedules").v_schedules_supported()


def test_v_schedule_eval_runs_forward_only_and_pp2_x_dp1_multi_stage_is_legal():
    results = _spawn("eval2", 2)
    ref = results[0]["ref"]
    for res in results:
        for schedule in (*V_SCHEDULES, "interleaved1f1b"):
            names = "pp" if schedule in V_SCHEDULES else "ppxdp"
            got = res[f"{schedule}:{names}"]
            if schedule in V_SCHEDULES and not _v_supported():
                assert "(torch >= 2.12)" in (got["apply"] or ""), got
                continue
            assert got["apply"] is None, (schedule, got)
            assert got["error"] is None, (schedule, got)
            # Forward only, the same token mean loss as the model in one process, every rank.
            assert got["eval"] == pytest.approx(ref, rel = 1e-5, abs = 1e-6), (schedule, got, ref)
            assert got["train"] == pytest.approx(ref, rel = 1e-5, abs = 1e-6), (schedule, got, ref)
            # step leaves the parameters as they were (no optimizer step): eval after training matches.
            assert got["eval_after"] == pytest.approx(got["eval"], rel = 1e-6, abs = 1e-7), (
                schedule,
                got,
            )


# ------------------------------------------------------------------- transformers' own distributed kwargs


class _Mesh1D:
    ndim = 1


def _spy():
    calls = []

    def load(*args, **kwargs):
        calls.append(args)
        return "loaded"

    return calls, load


@pytest.mark.parametrize("key", ["tp_plan", "tp_size", "device_mesh", "distributed_config"])
def test_fsdp2_sharded_loading_refuses_hf_distributed_kwargs_before_loading(monkeypatch, key):
    from transformers import LlamaForCausalLM

    fl = _load("fsdp2_load")
    calls, load = _spy()
    monkeypatch.setattr(fl, "bnb4_on_load", lambda kwargs: True)  # the 4-bit quantize-on-load path
    monkeypatch.setattr(fl, "_load_quantizing", load)
    kwargs = {key: "auto" if key == "tp_plan" else 2, "quantization_config": {"load_in_4bit": True}}
    with fl.fsdp2_sharded_loading(_Mesh1D()):
        with pytest.raises(ValueError, match = f"cannot be combined with transformers' own {key}"):
            LlamaForCausalLM.from_pretrained("unused", **kwargs)
    assert calls == []


def test_fsdp2_sharded_loading_without_hf_distributed_kwargs_still_loads(monkeypatch):
    from transformers import LlamaForCausalLM

    fl = _load("fsdp2_load")
    calls, load = _spy()
    monkeypatch.setattr(fl, "bnb4_on_load", lambda kwargs: True)
    monkeypatch.setattr(fl, "_load_quantizing", load)
    with fl.fsdp2_sharded_loading(_Mesh1D()):
        got = LlamaForCausalLM.from_pretrained(
            "unused", tp_plan = None, quantization_config = {"load_in_4bit": True}
        )
    assert got == "loaded" and len(calls) == 1


@pytest.mark.parametrize("quantize", [True, False])
def test_pipeline_sharded_loading_refuses_hf_distributed_kwargs_before_loading(
    monkeypatch, quantize
):
    from transformers import LlamaForCausalLM

    ppl = _load("pp_load")
    calls, load = _spy()
    monkeypatch.setattr(ppl, "_bnb4_on_load", lambda cls, name, kwargs: quantize)
    monkeypatch.setattr(ppl, "_quantize_sharded", lambda live, name, kwargs: True)
    monkeypatch.setattr(ppl, "_load_quantizing", load)
    with ppl.pipeline_sharded_loading(_Mesh1D(), schedule = "gpipe"):
        with pytest.raises(ValueError, match = "cannot be combined with transformers' own tp_plan"):
            LlamaForCausalLM.from_pretrained(
                "unused", tp_plan = "auto", quantization_config = {"load_in_4bit": True}
            )
        if quantize:
            # Without the competing kwarg the quantize-on-load path still runs.
            assert (
                LlamaForCausalLM.from_pretrained(
                    "unused", quantization_config = {"load_in_4bit": True}
                )
                == "loaded"
            )
    assert len(calls) == int(quantize)


# ------------------------------------------------------------------------------------------- NCCL backend


def _parallel_env():
    _load("pp_native")  # the unsloth / unsloth.models stubs of this tree
    env = importlib.import_module("unsloth.parallel_env")
    assert os.path.dirname(env.__file__) == os.path.join(REPO, "unsloth"), env.__file__
    return env


def _fake_cuda(monkeypatch, nccl):
    import torch.distributed as dist

    inits = []

    def init_process_group(
        backend = None,
        *args,
        **kwargs,
    ):
        inits.append(backend)
        raise RuntimeError("init_process_group reached")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: nccl)
    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    return inits


def test_cuda_layout_without_nccl_gives_a_clear_error_before_init_process_group(monkeypatch):
    env = _parallel_env()
    inits = _fake_cuda(monkeypatch, nccl = False)
    layout = env.ParallelLayout(2, 1, 2, False)
    with pytest.raises(RuntimeError, match = "has no NCCL backend"):
        layout.mesh()
    assert inits == []


def test_cuda_layout_with_nccl_still_initializes_nccl(monkeypatch):
    env = _parallel_env()
    inits = _fake_cuda(monkeypatch, nccl = True)
    layout = env.ParallelLayout(2, 1, 2, False)
    with pytest.raises(RuntimeError, match = "init_process_group reached"):
        layout.mesh()
    assert inits == ["nccl"]


def test_cpu_layout_without_nccl_still_uses_gloo(monkeypatch):
    import torch.distributed as dist

    env = _parallel_env()
    inits = _fake_cuda(monkeypatch, nccl = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    layout = env.ParallelLayout(2, 1, 2, False)
    with pytest.raises(RuntimeError, match = "init_process_group reached"):
        layout.mesh()
    assert inits == ["gloo"]
    assert dist.is_nccl_available() is False
