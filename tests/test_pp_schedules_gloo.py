# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Multi-stage pipeline schedules (Interleaved1F1B, LoopedBFS, InterleavedZeroBubble, ZBVZeroBubble, DualPipeV)
of apply_pipeline_parallel on a gloo CPU world (2 / 4 ranks, 2 stages per rank) vs the same model in one
process, fp32; refusals before the model is cut."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import weakref  # noqa: E402
import gc as gc_module  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pp_native_gloo import (  # noqa: E402
    GEMMA3,
    LLAMA,
    MISTRAL,
    QWEN3,
    _enable_reentrant_gc,
    _free_port,
    _read_adapter,
    _reference,
    _torch_pipelining_error,
    batch,
    build,
    ppn,
)

ARCHS = {"llama": LLAMA, "qwen3": QWEN3, "gemma3": GEMMA3, "mistral": MISTRAL}
MULTI = ["interleaved1f1b", "loopedbfs", "interleavedzerobubble", "zbvzerobubble", "dualpipev"]
V_SCHEDULES = ("zbvzerobubble", "dualpipev")
N_LAYERS = 8


def _v_supported():
    import importlib
    return importlib.import_module("unsloth.models.pp_schedules").v_schedules_supported()


def _skip_v(schedule):
    if schedule in V_SCHEDULES and not _v_supported():
        pytest.skip(
            f"V schedules need torch >= 2.12 (refused on {torch.__version__}, see the refusal test)"
        )


def _stage_map(
    pp,
    schedule,
    v = 2,
):
    """{global stage: pp rank} as the schedule places them."""
    out = {}
    for r in range(pp):
        if schedule in ("zbvzerobubble", "dualpipev"):
            ks = (r, 2 * pp - 1 - r)
        else:
            ks = tuple(r + k * pp for k in range(v))
        for k in ks:
            out[k] = r
    return out


def _case_multi(
    rank,
    world,
    tmp,
    arch,
    schedule,
    n_mb = 4,
    layers = N_LAYERS,
    gc = None,
    steps = 3,
    tie = None,
    bsz = 4,
    compile_layers = False,
):
    from torch.distributed.device_mesh import init_device_mesh

    repo = ARCHS[arch]
    ref = build(repo, layers = layers, tie = tie)
    ids, labels = batch(ref, bsz = bsz)
    if gc == "reentrant":
        _enable_reentrant_gc(ref)
    elif gc == "non_reentrant":
        ref.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    ref_res = _reference(ref, ids, labels, steps)
    if rank == 0:
        ref.save_pretrained(os.path.join(tmp, "ref"))
    model = build(repo, layers = layers, tie = tie)
    if gc == "reentrant":
        _enable_reentrant_gc(model)
    elif gc == "non_reentrant":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    originals = {
        "embed": weakref.ref(model.get_input_embeddings()),
        "head": weakref.ref(model.get_output_embeddings()),
    }
    if compile_layers:
        # Unsloth compiles each decoder block (torch.compile regions): AOTAutograd's compiled backward donates
        # buffers, which a split (zero bubble) backward with retain_graph cannot run.
        from torch import _dynamo
        _dynamo.config.cache_size_limit = 64
        for layer in ppn._find_parts(model).layers:
            layer.compile(dynamic = False)
    from torch._functorch import config as functorch_config

    donated_before = getattr(functorch_config, "donated_buffer", None)
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = n_mb, schedule = schedule)
    donated_after = getattr(functorch_config, "donated_buffer", None)
    gc_module.collect()
    alive = {k: r() is not None for k, r in originals.items()}
    outs = {k: [] for k in pp.local_stages}
    handles = [
        module.register_forward_hook(
            lambda m, a, kw, o, k = k: outs[k].append(o.detach().clone()), with_kwargs = True
        )
        for k, module in zip(pp.local_stages, pp.stage_modules)
    ]
    calls = {}
    for i, layer in enumerate(pp.parts.layers):
        if type(layer).__name__ != "_PPPassThrough":
            handles.append(
                layer.register_forward_pre_hook(
                    lambda m, a, i = i: calls.__setitem__(i, calls.get(i, 0) + 1)
                )
            )
    opt = torch.optim.AdamW(pp.parameters(), lr = 1e-2, foreach = False)
    losses, norms, grads = [], [], None
    for step in range(steps):
        loss = pp.step(ids, labels)
        if step == 0:
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            for h in handles:
                h.remove()
        norms.append(float(pp.clip_grad_norm_(1.0)))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
    built = next(iter(pp._shapes.values()))
    torch_map = dict(getattr(built, "stage_index_to_group_rank", {}) or {})
    pp.save_adapter(os.path.join(tmp, "pp"))
    # Every layer module of the model is back in place after the stage forwards (no swap leaked).
    layer_types = [type(layer).__name__ for layer in pp.parts.layers]
    direct_error = None
    try:
        model(input_ids = ids, labels = labels)
    except RuntimeError as exc:
        direct_error = str(exc)
    return {
        "ref": {k: v for k, v in ref_res.items() if k != "hidden"},
        "hidden": ref_res["hidden"],
        "losses": losses,
        "norms": norms,
        "grads": grads,
        "outs": outs,
        "bounds": pp.bounds,
        "local_stages": pp.local_stages,
        "owned": {n for n, p in model.named_parameters() if p.requires_grad},
        "trained": {
            n.replace(".default.", "."): p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        },
        "alive": alive,
        "layer_types": layer_types,
        "direct_error": direct_error,
        "info": dict(model.__dict__.get("_unsloth_pp") or {}),
        "repr": repr(pp),
        "torch_map": torch_map,
        "layer_calls": calls,
        "donated": (donated_before, donated_after),
        "gc": gc,
        "n_mb": n_mb,
    }


def _check_multi(
    results,
    saved,
    pp,
    schedule,
    layers = N_LAYERS,
    v = 2,
):
    total = pp * v
    placement = _stage_map(pp, schedule, v)
    ref = results[0]["ref"]
    union = set()
    for rank, res in enumerate(results):
        bounds = res["bounds"]
        assert len(bounds) == total and bounds[0][0] == 0 and bounds[-1][1] == layers, bounds
        assert all(bounds[k][1] == bounds[k + 1][0] for k in range(total - 1)), bounds
        assert set(res["local_stages"]) == {k for k, r in placement.items() if r == rank}
        # torch's schedule places the stages where this rank built them.
        assert res["torch_map"] == placement, (res["torch_map"], placement)
        assert res["info"]["stages_per_rank"] == v and tuple(res["info"]["local_stages"]) == tuple(
            res["local_stages"]
        )
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-5, abs = 1e-6), (
            rank,
            res["losses"],
            ref["losses"],
        )
        assert res["norms"][0] == pytest.approx(ref["norms"][0], rel = 1e-5), (rank, res["norms"])
        assert res["norms"] == pytest.approx(ref["norms"], rel = 1e-4), (
            rank,
            res["norms"],
            ref["norms"],
        )
        assert set(res["grads"]) == res["owned"], sorted(set(res["grads"]) ^ res["owned"])[:6]
        assert not union & set(res["grads"]), "a trainable parameter lives on two ranks"
        union |= set(res["grads"])
        for name, grad in res["grads"].items():
            torch.testing.assert_close(grad, ref["grads"][name], rtol = 1e-4, atol = 1e-6, msg = name)
        # Each local stage's output (micro-batch by micro-batch) = the one-process hidden state entering the
        # next stage's first layer; the last stage returns the loss.
        for k, chunks in res["outs"].items():
            assert len(chunks) == res["n_mb"], (k, len(chunks))
            if k < total - 1:
                torch.testing.assert_close(
                    torch.cat(chunks), res["hidden"][bounds[k][1]], rtol = 1e-5, atol = 1e-5, msg = str(k)
                )
            else:
                assert all(o.dim() == 0 for o in chunks)
        owns_first, owns_last = 0 in res["local_stages"], total - 1 in res["local_stages"]
        assert res["alive"] == {"embed": owns_first, "head": owns_last}, (rank, res["alive"])
        owned_layers = {i for k in res["local_stages"] for i in range(*bounds[k])}
        if res["gc"] is None:
            # Each owned layer runs once per micro-batch: a stage never runs this rank's other stages' layers.
            assert res["layer_calls"] == {i: res["n_mb"] for i in owned_layers}, (
                rank,
                res["layer_calls"],
            )
        for i, name in enumerate(res["layer_types"]):
            assert (name == "_PPPassThrough") == (i not in owned_layers), (rank, i, name)
        assert "split for pipeline parallelism" in (res["direct_error"] or "")
        # Zero bubble (split backward) turns AOTAutograd's donated buffers off; other schedules leave it alone.
        before, after = res["donated"]
        if before is not None:
            split = schedule in ("interleavedzerobubble", "zbvzerobubble", "dualpipev")
            assert after == (False if split else before), (schedule, res["donated"])
    assert union == set(ref["grads"]), sorted(union ^ set(ref["grads"]))[:6]
    assert set(saved["pp"]) == set(saved["ref"]), sorted(set(saved["pp"]) ^ set(saved["ref"]))[:6]
    trained = {}
    for res in results:
        trained.update(res["trained"])
    for key, value in saved["pp"].items():
        assert torch.equal(value, trained[key]), key
    for key, value in saved["ref"].items():
        torch.testing.assert_close(saved["pp"][key], value, rtol = 1e-3, atol = 2e-3, msg = key)


def _case_refusals(rank, world, tmp):
    """Every refusal leaves the model uncut: no pass-through layer, the embedding and lm_head in place."""
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    out = {}

    def attempt(key, model, **kwargs):
        embed, head = model.get_input_embeddings(), model.get_output_embeddings()
        try:
            ppn.apply_pipeline_parallel(model, mesh, **kwargs)
            out[key] = None
        except (ValueError, RuntimeError) as exc:
            out[key] = str(exc)
        layers = ppn._find_parts(model).layers
        out[key + ":uncut"] = (
            not any(type(layer).__name__ == "_PPPassThrough" for layer in layers)
            and model.get_input_embeddings() is embed
            and model.get_output_embeddings() is head
            and not ppn.is_pipeline_parallel(model)
        )

    attempt("unknown", build(LLAMA), schedule = "pipedream")
    attempt(
        "gpipe_two_stages",
        build(LLAMA),
        schedule = "gpipe",
        config = ppn.PPConfig(stages_per_rank = 2),
    )
    attempt(
        "zbv_three_stages",
        build(LLAMA, layers = 8),
        schedule = "zbvzerobubble",
        n_microbatches = 4,
        config = ppn.PPConfig(stages_per_rank = 3),
    )
    # DualPipeV needs at least as many micro-batches as stages (4 here): torch's own check, run up front.
    attempt("dualpipev_few_microbatches", build(LLAMA), schedule = "dualpipev", n_microbatches = 2)
    # Interleaved 1F1B: micro-batches a multiple of the rounds (5 // 2 = 2 rounds).
    attempt("interleaved_rounds", build(LLAMA), schedule = "interleaved1f1b", n_microbatches = 5)
    attempt("too_many_stages", build(LLAMA, layers = 3), schedule = "interleaved1f1b", n_microbatches = 4)

    reentrant = build(LLAMA)
    _enable_reentrant_gc(reentrant)
    attempt("zb_reentrant_gc", reentrant, schedule = "interleavedzerobubble", n_microbatches = 4)
    attempt("v_schedule", build(LLAMA), schedule = "zbvzerobubble", n_microbatches = 4)

    def apply_lora_qkv(self, x):  # stands in for unsloth.kernels.fast_lora's fused kernel
        raise AssertionError("not called")

    apply_lora_qkv.__module__ = "unsloth.kernels.fast_lora"
    fused = build(LLAMA)
    for layer in ppn._find_parts(fused).layers:
        layer.self_attn.apply_qkv = apply_lora_qkv
    attempt("zb_fused_lora", fused, schedule = "interleavedzerobubble", n_microbatches = 4)
    fused_ok = build(LLAMA)
    for layer in ppn._find_parts(fused_ok).layers:
        layer.self_attn.apply_qkv = apply_lora_qkv
    attempt("interleaved_fused_lora", fused_ok, schedule = "interleaved1f1b", n_microbatches = 4)

    fake = types.ModuleType("unsloth_zoo.gradient_checkpointing")  # pre-#2071 zoo: no slot leases
    saved = sys.modules.get("unsloth_zoo.gradient_checkpointing")
    sys.modules["unsloth_zoo.gradient_checkpointing"] = fake
    try:
        old_zoo = build(LLAMA)
        _enable_reentrant_gc(old_zoo)
        attempt("interleaved_old_zoo", old_zoo, schedule = "interleaved1f1b", n_microbatches = 4)
        bfs = build(LLAMA)
        _enable_reentrant_gc(bfs)
        # LoopedBFS runs every forward of a step before its backwards: smart checkpointing is safe there.
        attempt("loopedbfs_old_zoo", bfs, schedule = "loopedbfs", n_microbatches = 4)
    finally:
        if saved is None:
            sys.modules.pop("unsloth_zoo.gradient_checkpointing", None)
        else:
            sys.modules["unsloth_zoo.gradient_checkpointing"] = saved
    return out


def _case_dp_refusal(rank, world, tmp):
    """pp 2 x dp 2 mesh: several stages per rank do not compose with native FSDP2 over dp (refused before the cut);
    the same mesh with one stage per rank is still accepted."""
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names = ("pp", "dp"))
    model = build(LLAMA)
    out = {}
    try:
        ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 4, schedule = "interleaved1f1b")
        out["multi"] = None
    except ValueError as exc:
        out["multi"] = str(exc)
    out["uncut"] = not ppn.is_pipeline_parallel(model)
    pp = ppn.apply_pipeline_parallel(build(LLAMA), mesh["pp"], n_microbatches = 2, schedule = "gpipe")
    out["single"] = pp.stages_per_rank
    return out


def _case_compiled_refusal(rank, world, tmp, schedule):
    """A split-backward schedule over compiled blocks on a torch whose split backward cannot run them."""
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    model = build(LLAMA)
    for layer in ppn._find_parts(model).layers:
        layer.compile(dynamic = False)
    embed, head = model.get_input_embeddings(), model.get_output_embeddings()
    out = {}
    try:
        ppn.apply_pipeline_parallel(model, mesh, n_microbatches = 4, schedule = schedule)
        out["error"] = None
    except (ValueError, RuntimeError) as exc:
        out["error"] = str(exc)
    out["uncut"] = (
        not any(type(layer).__name__ == "_PPPassThrough" for layer in ppn._find_parts(model).layers)
        and model.get_input_embeddings() is embed
        and model.get_output_embeddings() is head
        and not ppn.is_pipeline_parallel(model)
    )
    return out


CASES = {
    "multi": _case_multi,
    "refusals": _case_refusals,
    "dp_refusal": _case_dp_refusal,
    "compiled_refusal": _case_compiled_refusal,
}


def _worker(rank, world, port, case, tmp, kwargs):
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

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 180)
    )
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
    keep = None,
    **kwargs,
):
    if _torch_pipelining_error():
        pytest.skip(_torch_pipelining_error())
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
        if keep is not None:
            keep["ref"] = _read_adapter(os.path.join(tmp, "ref"))
            keep["pp"] = _read_adapter(os.path.join(tmp, "pp"))
        return results


@pytest.mark.parametrize("schedule", MULTI)
def test_pp2_multi_stage_schedules_match_one_process(schedule):
    _skip_v(schedule)
    saved = {}
    results = _spawn("multi", 2, keep = saved, arch = "llama", schedule = schedule)
    _check_multi(results, saved, 2, schedule)


@pytest.mark.parametrize("schedule", ["interleaved1f1b", "zbvzerobubble"])
def test_pp4_multi_stage_schedules_match_one_process(schedule):
    _skip_v(schedule)
    saved = {}
    results = _spawn("multi", 4, keep = saved, arch = "mistral", schedule = schedule, n_mb = 4)
    _check_multi(results, saved, 4, schedule)


@pytest.mark.parametrize("arch", ["qwen3", "gemma3"])
def test_pp2_interleaved1f1b_other_archs_match_one_process(arch):
    saved = {}
    results = _spawn("multi", 2, keep = saved, arch = arch, schedule = "interleaved1f1b", n_mb = 2, bsz = 4)
    _check_multi(results, saved, 2, "interleaved1f1b")


def test_pp2_dualpipev_tied_embeddings_both_ends_on_rank0():
    # V placement puts the first and the last stage on rank 0: the tied embedding / lm_head stay there only.
    _skip_v("dualpipev")
    saved = {}
    results = _spawn("multi", 2, keep = saved, arch = "gemma3", schedule = "dualpipev", tie = True)
    _check_multi(results, saved, 2, "dualpipev")
    assert results[0]["alive"] == {"embed": True, "head": True}
    assert results[1]["alive"] == {"embed": False, "head": False}


@pytest.mark.parametrize("schedule", ["interleaved1f1b", "loopedbfs"])
def test_pp2_multi_stage_reentrant_gradient_checkpointing_matches_one_process(schedule):
    saved = {}
    results = _spawn("multi", 2, keep = saved, arch = "qwen3", schedule = schedule, gc = "reentrant")
    _check_multi(results, saved, 2, schedule)


@pytest.mark.parametrize("schedule", ["interleavedzerobubble", "zbvzerobubble"])
def test_pp2_zero_bubble_with_non_reentrant_checkpointing_matches_one_process(schedule):
    _skip_v(schedule)
    saved = {}
    results = _spawn("multi", 2, keep = saved, arch = "llama", schedule = schedule, gc = "non_reentrant")
    _check_multi(results, saved, 2, schedule)


@pytest.mark.parametrize("schedule", ["interleavedzerobubble", "zbvzerobubble"])
def test_pp2_zero_bubble_with_compiled_blocks_matches_one_process(schedule):
    sched = sys.modules["unsloth.models.pp_schedules"]
    if not sched.split_backward_keeps_function_nodes():
        # torch <= 2.12: split backward reads _input_metadata of compiled (AOTAutograd) nodes whose graph root
        # is already released; refused before the model is cut.
        # V schedules on torch < 2.12 hit the earlier V-placement refusal first.
        v_refused = schedule in V_SCHEDULES and not _v_supported()
        for res in _spawn("compiled_refusal", 2, schedule = schedule):
            if v_refused:
                assert "(torch >= 2.12)" in (res["error"] or ""), res
            else:
                assert "cannot run over torch.compile'd blocks" in (res["error"] or ""), res
                assert schedule in res["error"] and "interleaved1f1b" in res["error"], res
            assert res["uncut"], res
        return
    _skip_v(schedule)
    saved = {}
    results = _spawn("multi", 2, keep = saved, arch = "llama", schedule = schedule, compile_layers = True)
    _check_multi(results, saved, 2, schedule)


def test_pp4_interleaved1f1b_x_tp2_matches_one_process():
    from test_pp_fsdp2_gloo import _check, _spawn as compose_spawn
    dims = {"pp": 2, "tp": 2}
    _check(
        compose_spawn(
            "compose", world = 4, arch = "llama", dims = dims, schedule = "interleaved1f1b", n_mb = 4
        ),
        dims,
    )


def test_multi_stage_refusals_leave_the_model_uncut():
    res = _spawn("refusals", 2)[0]
    for key in [k for k in res if not k.endswith(":uncut")]:
        if key in ("loopedbfs_old_zoo", "interleaved_fused_lora") or (
            key == "v_schedule" and _v_supported()
        ):
            continue
        assert res[key + ":uncut"], (key, res[key])
    assert "pipeline schedule must be one of" in res["unknown"]
    assert "runs one stage per rank" in res["gpipe_two_stages"]
    assert "exactly 2 stages per rank" in res["zbv_three_stages"]
    assert (
        "DualPipeV requires at least as many microbatches as stages"
        if _v_supported()
        else "(torch >= 2.12)"
    ) in res["dualpipev_few_microbatches"]
    assert "multiple of the number of rounds" in res["interleaved_rounds"]
    assert "cannot split 3 decoder layers into 4 pipeline stages" in res["too_many_stages"]
    assert "splits each backward into an input and a weight pass" in res["zb_reentrant_gc"]
    assert "interleaves forwards and backwards" in res["interleaved_old_zoo"]
    assert res["loopedbfs_old_zoo"] is None
    assert (
        "runs the backward of Unsloth's fused LoRA kernels (apply_lora_qkv) twice"
        in res["zb_fused_lora"]
    )
    assert res["interleaved_fused_lora"] is None  # full-backward schedules keep the fused kernels
    if _v_supported():
        assert res["v_schedule"] is None
    else:
        assert "(torch >= 2.12)" in res["v_schedule"]


def test_multi_stage_on_a_pp_x_dp_mesh_is_refused_before_the_cut():
    res = _spawn("dp_refusal", 4)
    for r in res:
        assert "several stages per rank is not supported" in (r["multi"] or ""), r
        assert r["uncut"] and r["single"] == 1


def test_schedule_names_normalize():
    sched = sys.modules["unsloth.models.pp_schedules"]
    assert sched.schedule_spec("Interleaved1F1B").name == "interleaved1f1b"
    assert sched.schedule_spec("ScheduleDualPipeV").name == "dualpipev"
    assert sched.schedule_spec("zbv").name == "zbvzerobubble"
    assert sched.schedule_spec("looped_bfs").name == "loopedbfs"
    assert sched.schedule_spec("GPipe").name == "gpipe"


def test_split_backward_probe_matches_torch_node_ownership():
    # THPFunction owns its node (c10::intrusive_ptr) from torch 2.13; up to 2.12 it is a std::weak_ptr.
    sched = sys.modules["unsloth.models.pp_schedules"]
    major, minor = (int(v) for v in torch.__version__.split(".")[:2])
    assert sched.split_backward_keeps_function_nodes() == ((major, minor) >= (2, 13))


def test_compiled_blocks_refused_for_split_backward_on_affected_torch(monkeypatch):
    sched = sys.modules["unsloth.models.pp_schedules"]
    split = {"interleaved": True, "split_backward": True}
    full = {"interleaved": True, "split_backward": False}

    def model_with(install):
        block = torch.nn.Linear(2, 2)
        install(block)
        return torch.nn.Sequential(torch.nn.Linear(2, 2), block)

    def marked(block):
        cls = type("Block", (torch.nn.Linear,), {})
        forward = lambda self, x: torch.nn.Linear.forward(self, x)  # noqa: E731
        forward._unsloth_block_compiled = True
        cls.forward = forward
        block.__class__ = cls

    installs = {
        "module_compile": lambda b: b.compile(dynamic = False),
        "block_compile": marked,
        "compiled_forward": lambda b: setattr(b, "forward", torch.compile(b.forward)),
    }
    assert sched.compiled_blocks(torch.nn.Sequential(torch.nn.Linear(2, 2))) == []
    assert sched.compiled_blocks(torch.compile(torch.nn.Linear(2, 2)))
    for schedule in ("interleavedzerobubble", "zbvzerobubble", "dualpipev"):
        spec = sched.schedule_spec(schedule)
        for kind, install in installs.items():
            assert sched.compiled_blocks(model_with(install)) == ["1"], kind
            monkeypatch.setattr(ppn, "split_backward_keeps_function_nodes", lambda: False)
            with pytest.raises(ValueError, match = "cannot run over torch.compile'd blocks") as exc:
                ppn._check_schedule(model_with(install), spec, split)
            assert schedule in str(exc.value) and "UNSLOTH_BLOCK_COMPILE=0" in str(exc.value)
            ppn._check_schedule(model_with(install), spec, full)  # full backward: not affected
            monkeypatch.setattr(ppn, "split_backward_keeps_function_nodes", lambda: True)
            ppn._check_schedule(model_with(install), spec, split)  # torch >= 2.13: allowed
    monkeypatch.setattr(ppn, "split_backward_keeps_function_nodes", lambda: False)
    ppn._check_schedule(torch.nn.Sequential(torch.nn.Linear(2, 2)), spec, split)  # nothing compiled


def test_forward_calling_compiled_function_refused_for_split_backward(monkeypatch):
    # unsloth_zoo's generated unsloth_compiled_module_* files keep the class forward eager and call a module-level
    # compiled function from it (Gemma3MLP.forward -> Gemma3MLP_forward), through torch_compile_with_fallback's
    # functools.wraps wrapper. Its backward is a CompiledFunctionBackward node: torch <= 2.12's split backward fails
    # on it after the cut ("_input_metadata is invalid"), as on a compiled block.
    import functools

    sched = sys.modules["unsloth.models.pp_schedules"]
    split = {"interleaved": True, "split_backward": True}

    def eager(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

    compiled = torch.compile(eager, dynamic = False)

    @functools.wraps(compiled)
    def dispatch(*args, **kwargs):  # unsloth_zoo _eager_during_decode's shape
        return compiled(*args, **kwargs)

    dispatch.__wrapped__ = eager
    dispatch._unsloth_compiled_func = compiled

    def block_calling(target):
        scope = {"Block_forward": target}
        exec("def forward(self, x):\n    return Block_forward(self, x)\n", scope)
        cls = type("Block", (torch.nn.Linear,), {"forward": scope["forward"]})
        block = torch.nn.Linear(2, 2)
        block.__class__ = cls
        return torch.nn.Sequential(torch.nn.Linear(2, 2), block)

    for kind, target in {"torch_compile": compiled, "zoo_wrapper": dispatch}.items():
        assert sched.compiled_blocks(block_calling(target)) == ["1"], kind
        spec = sched.schedule_spec("interleavedzerobubble")
        monkeypatch.setattr(ppn, "split_backward_keeps_function_nodes", lambda: False)
        with pytest.raises(ValueError, match = "cannot run over torch.compile'd blocks"):
            ppn._check_schedule(block_calling(target), spec, split)
        monkeypatch.setattr(ppn, "split_backward_keeps_function_nodes", lambda: True)
        ppn._check_schedule(block_calling(target), spec, split)  # torch >= 2.13: allowed

    # A compiled helper captured from an enclosing scope (a closure, not a global) is found too.
    def closure_forward(self, x):
        return compiled(self, x)

    closure_block = torch.nn.Linear(2, 2)
    closure_block.__class__ = type("Block", (torch.nn.Linear,), {"forward": closure_forward})
    assert sched.compiled_blocks(torch.nn.Sequential(torch.nn.Linear(2, 2), closure_block)) == ["1"]
    # An eager helper called the same way is not a compiled block.
    assert sched.compiled_blocks(block_calling(eager)) == []


def test_single_backward_kernels_unwraps_partials():
    # patch_peft_model binds Cohere's MLP kernel as MethodType(partial(apply_lora_mlp, inplace=False), mlp).
    import functools

    sched = sys.modules["unsloth.models.pp_schedules"]

    def apply_lora_mlp_swiglu(
        self,
        X,
        inplace = True,
    ):  # stands in for fast_lora's fused MLP kernel
        raise AssertionError("not called")

    apply_lora_mlp_swiglu.__module__ = "unsloth.kernels.fast_lora"
    cohere = functools.partial(apply_lora_mlp_swiglu, inplace = False)
    installs = {
        "plain": lambda m: types.MethodType(apply_lora_mlp_swiglu, m),
        "partial": lambda m: types.MethodType(cohere, m),
        "nested_partial": lambda m: types.MethodType(functools.partial(cohere), m),
        "unbound_partial": lambda m: cohere,
    }
    for attr in ("forward", "_unsloth_forward"):
        for kind, install in installs.items():
            mlp = torch.nn.Linear(2, 2)
            setattr(mlp, attr, install(mlp))
            model = torch.nn.Sequential(mlp)
            assert sched.single_backward_kernels(model) == ["apply_lora_mlp_swiglu"], (attr, kind)
    assert sched.single_backward_kernels(torch.nn.Sequential(torch.nn.Linear(2, 2))) == []
