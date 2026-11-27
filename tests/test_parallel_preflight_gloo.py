# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Parallel preflight guards, on CPU (gloo ranks or no process group):

- several pipeline stages per rank over a root mesh with an FSDP2 data parallel dimension (dp, dp_shard) is
  refused before the model is cut and before pipeline_sharded_loading reads (a mesh factory too);
- the native FSDP2, pipeline parallel and launcher (UNSLOTH_TP_SIZE / UNSLOTH_PP_SIZE) loaders refuse
  transformers' own tp_plan / tp_size / ... (any value but None) as the tensor parallel loader does, before any
  load runs, and the launcher's whole-load fallback does not swallow that refusal;
- eval_loss with adjacent stages on one rank (V schedules, loop schedules at pp 1) runs forward only, matches
  the one-process loss and frees its forward caches;
- the launcher's mesh refuses a CUDA or ROCm layout on a PyTorch build without NCCL before init_process_group.

Legal layouts (pp x dp with one stage per rank, pp 2 x dp 1 with two stages, several stages over a replicated
dp_replicate axis, pp x tp / pp x cp under any axis names) pass the same guards unchanged."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

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

    from transformers import LlamaForCausalLM

    ppl = _load("pp_load")
    out = {}
    for pp_name, dp_name in (("pp", "dp_shard"), ("pp", "dp"), ("pipeline", "dp_shard")):
        key = dp_name if pp_name == "pp" else f"{pp_name}x{dp_name}"
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names = (pp_name, dp_name))
        model = build(LLAMA, layers = 4)
        out[key] = _error(
            lambda: ppn.apply_pipeline_parallel(
                model, mesh[pp_name], n_microbatches = 4, schedule = "interleaved1f1b"
            )
        )
        out[key + ":uncut"] = _uncut(model)

        def enter():
            with ppl.pipeline_sharded_loading(mesh[pp_name], schedule = "interleaved1f1b"):
                pass

        out[key + ":load"] = _error(enter)

        def enter_factory():
            # A mesh factory is resolved and checked at the first from_pretrained, before any read.
            with ppl.pipeline_sharded_loading(lambda: mesh[pp_name], schedule = "interleaved1f1b"):
                LlamaForCausalLM.from_pretrained(
                    tmp
                )  # an empty local folder: a load would fail there

        out[key + ":factory"] = _error(enter_factory)
        # One stage per rank composes with dp: accepted and cut.
        single = ppn.apply_pipeline_parallel(
            build(LLAMA, layers = 4), mesh[pp_name], n_microbatches = 2, schedule = "gpipe"
        )
        out[key + ":single"] = single.stages_per_rank
    # Model parallel dimensions beside pp (under any name, pp's own included) and a replicated dp axis are not
    # FSDP2 data parallel: two stages per rank are cut as before.
    for names in (
        ("pp", "tp"),
        ("pp", "cp"),
        ("pp", "dp_replicate"),
        ("pipeline", "tp"),
        ("pp", "tensor"),
    ):
        key = "x".join(names)
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names = names)
        model = build(LLAMA, layers = 4)
        out[key] = _error(
            lambda: ppn.apply_pipeline_parallel(
                model, mesh[names[0]], n_microbatches = 4, schedule = "interleaved1f1b"
            )
        )
        out[key + ":stages"] = getattr(ppn.pipeline_stage_of(model), "stages_per_rank", None)
    return out


LOOP_SCHEDULES = ("interleaved1f1b", "loopedbfs", "interleavedzerobubble")


def _fwd_cached(pp):
    stages = [s for sched in pp._shapes.values() for s in _schedule_stages(sched)]
    return sum(len(s.fwd_cache) for s in stages)


def _schedule_stages(schedule):
    return getattr(schedule, "_stages", None) or (schedule._stage,)


def _case_eval1(rank, world, tmp):
    """1 rank (pp 1): loop schedules put every stage, consecutive ones included, on this rank."""
    from torch.distributed.device_mesh import init_device_mesh

    out = {}
    ref = build(LLAMA, layers = 4)
    ids, labels = batch(ref)
    with torch.no_grad():
        out["ref"] = float(ref(input_ids = ids, labels = labels).loss)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names = ("pp",))
    for schedule in LOOP_SCHEDULES:
        res = {}
        try:
            pp = ppn.apply_pipeline_parallel(
                build(LLAMA, layers = 4), mesh["pp"], n_microbatches = 4, schedule = schedule
            )
            res["eval"] = float(pp.eval_loss(ids, labels))
            res["cached"] = _fwd_cached(pp)
            res["train"] = float(pp.step(ids, labels))
            res["eval_after"] = float(pp.eval_loss(ids, labels))
            res["error"] = None
        except Exception as exc:  # noqa: BLE001 - torch asserts with AssertionError
            res["error"] = f"{type(exc).__name__}: {exc}"
        out[schedule] = res
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
            res["cached"] = _fwd_cached(pp)
            res["train"] = float(pp.step(ids, labels))
            res["eval_after"] = float(pp.eval_loss(ids, labels))
            res["error"] = None
        except Exception as exc:  # noqa: BLE001 - torch asserts with AssertionError
            res["error"] = f"{type(exc).__name__}: {exc}"
        out[key] = res
    return out


CASES = {"mesh4": _case_mesh4, "eval2": _case_eval2, "eval1": _case_eval1}


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
        deadline = time.monotonic() + 300  # one budget for every rank
        try:
            for p in procs:
                p.join(max(deadline - time.monotonic(), 0))
        finally:
            for p in procs:
                if p.is_alive():  # a hung rank must not outlive the result directory
                    p.terminate()
                    p.join(10)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


def test_multi_stage_pp_x_fsdp2_dp_refused_before_cut_and_load_other_axes_legal():
    for res in _spawn("mesh4", 4):
        for key, dp_name in (
            ("dp_shard", "dp_shard"),
            ("dp", "dp"),
            ("pipelinexdp_shard", "dp_shard"),
        ):
            for when in ("", ":load", ":factory"):
                assert "several stages per rank is not supported" in (res[key + when] or ""), (
                    key + when,
                    res,
                )
            assert repr(dp_name) in res[key], res[key]
            assert res[key + ":uncut"], (key, res)
            assert res[key + ":single"] == 1, (key, res)
        for key in ("ppxtp", "ppxcp", "ppxdp_replicate", "pipelinextp", "ppxtensor"):
            assert res[key] is None and res[key + ":stages"] == 2, (key, res)


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
            assert got["cached"] == 0, (schedule, got)  # eval frees its micro-batch forward caches


def test_loop_schedule_eval_at_pp1_runs_forward_only():
    (res,) = _spawn("eval1", 1)
    for schedule in LOOP_SCHEDULES:
        got = res[schedule]
        assert got["error"] is None, (schedule, got)
        assert got["eval"] == pytest.approx(res["ref"], rel = 1e-5, abs = 1e-6), (schedule, got)
        assert got["train"] == pytest.approx(res["ref"], rel = 1e-5, abs = 1e-6), (schedule, got)
        assert got["eval_after"] == pytest.approx(got["eval"], rel = 1e-6, abs = 1e-7), (schedule, got)
        assert got["cached"] == 0, (schedule, got)


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
            assert (
                LlamaForCausalLM.from_pretrained(
                    "unused", quantization_config = {"load_in_4bit": True}
                )
                == "loaded"
            )
    assert len(calls) == int(quantize)


@pytest.mark.parametrize(
    "kwargs",
    [{"tp_plan": {}}, {"tp_size": 1}, {"tp_plan": "auto"}],
    ids = ["empty_plan", "tp_size_1", "auto"],
)
def test_hf_distributed_kwargs_refused_by_value_not_truthiness(kwargs):
    # transformers reads tp_plan / tp_size as given when they are not None (tp_plan = {} is a plan).
    tpn = _load("tp_native")
    with pytest.raises(tpn.HFDistributedKwargsError, match = "cannot be combined"):
        tpn.refuse_hf_distributed_kwargs(kwargs, "tensor parallel", "apply_tensor_parallel")
    tpn.refuse_hf_distributed_kwargs(
        {"tp_plan": None, "device_mesh": None}, "tensor parallel", "apply_tensor_parallel"
    )


class _EnvLayout:
    env_vars = "UNSLOTH_TP_SIZE=2"
    pp, tp, dp = 1, 2, 1

    def __init__(self, sharded_loading):
        self.sharded_loading = sharded_loading
        self.tp_mesh = _Mesh1D()


def _env_loader(monkeypatch, layout, inner_kwargs):
    """A launcher-wrapped FastLanguageModel.from_pretrained stand-in that calls transformers itself."""
    from transformers import LlamaForCausalLM

    tpe = _load("tp_env")
    monkeypatch.setattr(tpe, "env_layout", lambda: layout)
    monkeypatch.setattr(tpe, "_rank0", lambda: False)
    calls = []

    def from_pretrained(model_name, **kwargs):
        calls.append(kwargs)
        return LlamaForCausalLM.from_pretrained(model_name, **inner_kwargs, **kwargs)

    return tpe.tensor_parallel_env_load(from_pretrained), calls


@pytest.mark.parametrize("sharded", [True, False], ids = ["sharded", "whole_load"])
def test_launcher_loader_refuses_hf_distributed_kwargs_on_every_branch(
    monkeypatch, tmp_path, sharded
):
    tpn = _load("tp_native")
    load, calls = _env_loader(monkeypatch, _EnvLayout(sharded), {})
    with pytest.raises(tpn.HFDistributedKwargsError, match = "transformers' own tp_plan"):
        load(str(tmp_path), tp_plan = "auto")
    assert calls == []  # before transformers sees the kwarg


def test_launcher_sharded_load_refusal_is_not_retried_as_a_whole_load(monkeypatch, tmp_path):
    # The loader adds tp_plan itself: the sharded path refuses it, and the whole-load retry would forward it.
    tpn = _load("tp_native")
    load, calls = _env_loader(monkeypatch, _EnvLayout(True), {"tp_plan": "auto"})
    with pytest.raises(tpn.HFDistributedKwargsError, match = "transformers' own tp_plan"):
        load(str(tmp_path))
    assert len(calls) == 1


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


def test_rocm_layout_without_rccl_gives_a_clear_error_before_init_process_group(monkeypatch):
    env = _parallel_env()
    inits = _fake_cuda(monkeypatch, nccl = False)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising = False)
    layout = env.ParallelLayout(2, 1, 2, False)
    with pytest.raises(RuntimeError, match = "RCCL on ROCm GPUs.*has no NCCL backend"):
        layout.mesh()
    assert inits == []


def test_non_gpu_accelerator_build_without_nccl_is_not_refused(monkeypatch):
    env = _parallel_env()
    inits = _fake_cuda(monkeypatch, nccl = False)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    layout = env.ParallelLayout(2, 1, 2, False)
    with pytest.raises(RuntimeError, match = "init_process_group reached"):
        layout.mesh()
    assert inits == ["nccl"]
