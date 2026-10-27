# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pipeline parallel composed with native FSDP2 data parallel and / or tensor parallel on a gloo CPU world
(pp 2 x dp 2, pp 2 x tp 2, pp 2 x dp 2 x tp 2) vs the same model in one process on the global batch, fp32."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import socket  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pp_native_gloo import _clip_fp64, ppn  # noqa: E402
from test_tp_native_oracle import tpn  # noqa: E402

LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
QWEN3 = "trl-internal-testing/tiny-Qwen3ForCausalLM"
GEMMA3 = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
# Overrides so the head counts split over tp = 2 (as the TP suite).
ARCHS = {
    "llama": (LLAMA, {}),
    "qwen3": (QWEN3, {}),
    "gemma3": (GEMMA3, {"num_attention_heads": 4, "num_key_value_heads": 2}),
}
N_LAYERS = 6
STEPS = 3
# Step-0 grad norm tolerance vs the one-process reference. Gemma3's own fp32 grad norm sits 1.0e-4 from fp64 on one
# process (Llama 7.8e-7), so any change of summation order (TP splits, CPU ISA) moves it by that much.
STEP0_NORM_REL = {"llama": 1e-5, "qwen3": 1e-5, "gemma3": 3e-4}
# Step-0 grad tolerances: per element relative to the tensor's largest entry, and relative L2 over all
# grads. Gemma3's fp32 reordering noise under tensor parallel depends on the CPU's kernels: worst
# element / relative L2 2.8e-5 / 2.4e-5 with this host's AVX-512 MKL, 1.6e-4 / 1.4e-4 with MKL and
# oneDNN held to AVX2, and 2e-4 per element on a CI runner, against 2.6e-6 for llama and 5.8e-6 for
# qwen3 on the same AVX2 run (pp x dp gemma3 stays near 4e-7). llama and qwen3 keep the strict bounds
# on the same tensor parallel code; a gemma3-specific fault (a wrong shard, a missed reduction) is O(1).
GRAD_ATOL_REL = {"llama": 2e-5, "qwen3": 2e-5, "gemma3": 1e-3}
GRAD_L2_REL = {"llama": 5e-5, "qwen3": 5e-5, "gemma3": 5e-4}


def _native():
    module = sys.modules.get("unsloth.models.fsdp2_native")
    if module is None:
        from test_fsdp2_native_units import _load_native
        module = _load_native()
    return module


def build(
    repo,
    overrides = None,
    seed = 0,
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    probe = AutoConfig.from_pretrained(repo)
    values = probe.to_dict()
    values["num_hidden_layers"] = N_LAYERS
    values.pop("layer_types", None)
    config = type(probe).from_dict(values)
    text = config.get_text_config()
    for key, value in (overrides or {}).items():
        setattr(text, key, value)
    for key in ("attention_dropout", "hidden_dropout", "dropout"):
        if isinstance(getattr(text, key, None), float):
            setattr(text, key, 0.0)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.add_(torch.randn_like(p) * 0.1)
    torch.manual_seed(seed + 1)
    lora = LoraConfig(
        r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False, lora_dropout = 0.0
    )
    return get_peft_model(model, lora).train()


def global_batch(
    model,
    bsz,
    seq = 8,
    seed = 1,
):
    vocab = ppn._text_config(model).vocab_size
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (bsz, seq), generator = g)
    labels = ids.clone()
    labels[
        0, :3
    ] = -100  # uneven token counts per dp replica / micro-batch: the global count matters
    labels[-1, -2:] = -100
    labels[1, -1] = -100
    return ids, labels


def _reference(model, ids, labels, steps):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-2, foreach = False)
    losses, norms, grads = [], [], None
    for step in range(steps):
        out = model(input_ids = ids, labels = labels, use_cache = False)
        out.loss.backward()
        if step == 0:
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        norms.append(_clip_fp64(params, 1.0))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss))
    return {"losses": losses, "norms": norms, "grads": grads}


def _trainable_param_groups(model, trainable = True):
    from torch.distributed.fsdp import FSDPModule

    groups = []
    for m in model.modules():
        if isinstance(m, FSDPModule):
            state = m._get_fsdp_state()
            found = getattr(state, "_fsdp_param_groups", None)
            if found is None:
                group = getattr(state, "_fsdp_param_group", None)
                found = [group] if group is not None else []
            for group in found:
                if not trainable or any(fp.sharded_param.requires_grad for fp in group.fsdp_params):
                    groups.append(group)
    return groups


def _count_reduces():
    """Wrap FSDP2's gradient reduce (one call per param group per synced backward)."""
    import torch.distributed.fsdp._fully_shard._fsdp_param_group as pg

    calls = [0]
    original = pg.foreach_reduce

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    pg.foreach_reduce = counted
    return calls


def _raised(fn):
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        return str(exc)
    return None


def _case_compose(
    rank,
    world,
    tmp,
    arch,
    dims,
    schedule = "gpipe",
    n_mb = 2,
    fit_layout = None,
    rows = 4,
):
    """apply_pipeline_parallel over mesh['pp'], then apply_tensor_parallel over mesh['tp'] and / or shard_model
    over mesh['dp']. Each dp replica trains on its rows of the global batch."""
    import torch.distributed as dist
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import init_device_mesh

    native = _native()
    repo, overrides = ARCHS[arch]
    names = tuple(dims)
    mesh = init_device_mesh("cpu", tuple(dims.values()), mesh_dim_names = names)
    dp = dims.get("dp", 1)
    dp_rank = mesh["dp"].get_local_rank() if "dp" in dims else 0
    ref = build(repo, overrides)
    ids, labels = global_batch(ref, rows * dp)
    ref_res = _reference(ref, ids, labels, STEPS)
    if rank == 0:
        ref.save_pretrained(os.path.join(tmp, "ref"))
    mine = slice(rows * dp_rank, rows * (dp_rank + 1))
    model = build(repo, overrides)
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = n_mb, schedule = schedule)
    refusals = {}
    if (
        world != dims["pp"]
    ):  # a pp mesh smaller than the world: the other dimensions must be applied first
        refusals["step_before_all_dims"] = _raised(lambda: pp.step(ids[mine], labels[mine]))
    if "tp" in dims:
        tpn.apply_tensor_parallel(model, mesh["tp"])
    if "dp" in dims:
        refusals["shard_model_pp_mesh"] = _raised(
            lambda: native.shard_model(
                model, native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["pp"])
            )
        )
        refusals["shard_model_no_mesh"] = _raised(
            lambda: native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32))
        )
        refusals["shard_model_inner_level"] = _raised(
            lambda: native.shard_model(
                model.base_model, native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"])
            )
        )
        refusals["shard_model_cpu_offload"] = _raised(
            lambda: native.shard_model(
                model,
                native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"], cpu_offload = True),
            )
        )
        refusals["fsdp_after_refusals"] = sum(
            type(m).__name__.startswith("FSDP") for m in model.modules()
        )
        native.shard_model(
            model,
            native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"], fit_layout = fit_layout),
        )
    from torch.distributed.fsdp import FSDPModule

    layouts = tpn._param_layouts(model)
    groups = _trainable_param_groups(model)
    calls = _count_reduces()
    params = pp.parameters()
    opt = torch.optim.AdamW(params, lr = 1e-2, foreach = False)
    losses, norms, reduces, grads = [], [], [], None
    for step in range(STEPS):
        before = calls[0]
        loss = pp.step(ids[mine], labels[mine])
        reduces.append(calls[0] - before)
        if step == 0:
            grads = {
                n: tpn._full(p.grad, layout = layouts.get(id(p))).detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        norms.append(float(pp.clip_grad_norm_(1.0)))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
    pp.save_adapter(os.path.join(tmp, "pp"))
    dist.barrier()
    placements = {
        n: tuple(str(pl) for pl in p.placements)
        for n, p in model.named_parameters()
        if p.requires_grad and tpn._is_dtensor(p)
    }
    return {
        "ref": ref_res,
        "losses": losses,
        "norms": norms,
        "grads": grads,
        "owned": {n for n, p in model.named_parameters() if p.requires_grad},
        "reduces": reduces,
        "groups": len(groups),
        # torchtitan's PP default: no unit reshards after forward (each micro-batch would all-gather again).
        "reshard_after_forward": [
            g.post_forward_mesh_info is not None
            for g in _trainable_param_groups(model, trainable = False)
        ],
        "stage_root_fsdp": isinstance(pp.module, FSDPModule),
        "placements": placements,
        "refusals": refusals,
        "stage": pp.stage_index,
        "saved_ref": load_file(os.path.join(tmp, "ref", "adapter_model.safetensors"))
        if rank == 0
        else None,
        "saved_pp": load_file(os.path.join(tmp, "pp", "adapter_model.safetensors"))
        if rank == 0
        else None,
    }


def _case_scaler(rank, world, tmp):
    """pp 2 x dp 2: an inf on one rank's (last stage, dp replica 1) gradient skips the step on every rank."""
    from torch.distributed.device_mesh import init_device_mesh

    native = _native()
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names = ("pp", "dp"))
    model = build(LLAMA)
    ids, labels = global_batch(model, 8)
    dp_rank = mesh["dp"].get_local_rank()
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 2)
    native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"]))
    scaler = torch.amp.GradScaler("cpu", init_scale = 1024.0)
    opt = torch.optim.SGD(pp.parameters(), lr = 1.0)
    pp.step(
        ids[4 * dp_rank : 4 * dp_rank + 4], labels[4 * dp_rank : 4 * dp_rank + 4], scaler = scaler
    )
    if rank == world - 1:
        # One rank (last stage, dp replica 1) finds an inf in its unscale (as a replicated gradient would);
        # set on its found_inf directly: an inf in a DTensor shard is already reduced over dp by GradScaler.
        unscale = scaler.unscale_

        def unscale_with_inf(optimizer):
            unscale(optimizer)
            for t in scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"].values():
                t.fill_(1.0)

        scaler.unscale_ = unscale_with_inf
    before = [tpn._local(p).detach().clone() for p in pp.parameters()]
    pp.unscale_(scaler, opt)
    scaler.step(opt)
    scaler.update()
    after = [tpn._local(p).detach().clone() for p in pp.parameters()]
    return {
        "unchanged": all(torch.equal(a, b) for a, b in zip(before, after)),
        "scale": scaler.get_scale(),
    }


def _case_order_refusals(rank, world, tmp):
    """TP or FSDP2 before the pipeline split, and a non-composable pp mesh, are refused before any change."""
    from torch.distributed.device_mesh import init_device_mesh

    native = _native()
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names = ("pp", "dp"))
    out = {}
    model = build(LLAMA)
    tpn.apply_tensor_parallel(model, mesh["dp"])
    layers = len(model.get_base_model().model.layers)
    out["tp_first"] = _raised(lambda: ppn.apply_pipeline_parallel(model, mesh["pp"]))
    out["tp_first_unchanged"] = not ppn.is_pipeline_parallel(model) and layers == N_LAYERS
    model = build(LLAMA)
    native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"]))
    out["fsdp_first"] = _raised(lambda: ppn.apply_pipeline_parallel(model, mesh["pp"]))
    out["fsdp_first_unchanged"] = not ppn.is_pipeline_parallel(model)
    # pp mesh over all 4 ranks = world: a plain v1 split; a 2-D mesh is refused.
    model = build(LLAMA)
    out["mesh_2d"] = _raised(lambda: ppn.apply_pipeline_parallel(model, mesh))
    # pp x dp: TP after shard_model is refused.
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"])
    native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"]))
    out["tp_after_fsdp"] = _raised(lambda: tpn.apply_tensor_parallel(model, mesh["dp"]))
    out["shard_twice_is_noop"] = native.shard_model(model) is model
    out["composed"] = pp.composed
    return out


def _case_pp_tp_mesh_refusal(rank, world, tmp):
    """pp 2 x tp 2: a tp mesh that is the pp mesh itself is refused before anything is sharded."""
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names = ("pp", "tp"))
    model = build(LLAMA)
    ppn.apply_pipeline_parallel(model, mesh["pp"])
    err = _raised(lambda: tpn.apply_tensor_parallel(model, mesh["pp"]))
    return {"err": err, "tp": tpn.is_tensor_parallel(model)}


CASES = {
    "compose": _case_compose,
    "scaler": _case_scaler,
    "order": _case_order_refusals,
    "pp_tp_mesh": _case_pp_tp_mesh_refusal,
}


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 240)
    )
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(
    case,
    world = 4,
    **kwargs,
):
    try:
        ppn._check_torch_pipelining()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    native = _native()
    if not native._supports_ignored_params(native._fsdp_api()[2]):
        pytest.skip("native FSDP2 composition needs fully_shard(ignored_params) (torch 2.7+)")
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
            p.join(900)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


def _check(
    results,
    dims,
    n_mb = 2,
    norm_rel = 1e-5,
    grad_rel = 2e-5,
    l2_rel = 5e-5,
):
    ref = results[0]["ref"]
    union = set()
    for rank, res in enumerate(results):
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-5, abs = 1e-6), (
            rank,
            res["losses"],
            ref["losses"],
        )
        assert res["norms"][0] == pytest.approx(ref["norms"][0], rel = norm_rel), (
            rank,
            res["norms"],
            ref["norms"],
        )
        # Later norms follow 1-2 AdamW steps (lr 1e-2), whose first updates are ~lr * sign(grad): fp32 summation
        # order on near-zero grads moves them up to ~5e-4 (Gemma3; the pp-only split alone moves it 2.6e-4).
        assert res["norms"] == pytest.approx(ref["norms"], rel = 2e-3), (
            rank,
            res["norms"],
            ref["norms"],
        )
        assert set(res["grads"]) == res["owned"], sorted(set(res["grads"]) ^ res["owned"])[:6]
        num = den = 0.0
        for name, grad in res["grads"].items():
            want = ref["grads"][name]
            # Tensor parallel row-parallel sums reorder fp32 additions: tolerance relative to the tensor's scale.
            atol = max(1e-6, grad_rel * float(want.abs().max()))
            torch.testing.assert_close(
                grad, want, rtol = 1e-4, atol = atol, msg = lambda m, name = name: f"{name}: {m}"
            )
            num += float((grad.double() - want.double()).square().sum())
            den += float(want.double().square().sum())
        assert (num / den) ** 0.5 < l2_rel, (rank, (num / den) ** 0.5)
        union |= set(res["grads"])
        refusals = res["refusals"]
        if len(results) != dims["pp"]:
            assert "Apply the other mesh dimensions" in (
                refusals["step_before_all_dims"] or ""
            ), refusals
        if "dp" in dims:
            assert res["stage_root_fsdp"]
            # FSDP2 shards the trainable weights over dp; each group reduces once per step, not per micro-batch.
            assert res["placements"] and all(
                any("S(" in pl or "Shard" in pl for pl in pls) for pls in res["placements"].values()
            ), res["placements"]
            assert res["groups"] >= 1 and res["reduces"] == [res["groups"]] * STEPS, (
                res["reduces"],
                res["groups"],
            )
            assert res["reshard_after_forward"] and not any(res["reshard_after_forward"]), res[
                "reshard_after_forward"
            ]
            assert "meeting only at this rank" in (refusals["shard_model_pp_mesh"] or ""), refusals
            assert "needs FSDP2Config(mesh = mesh['dp'])" in (
                refusals["shard_model_no_mesh"] or ""
            ), refusals
            assert "not a level inside it" in (refusals["shard_model_inner_level"] or ""), refusals
            assert "cpu_offload is not supported on a pipeline parallel model" in (
                refusals["shard_model_cpu_offload"] or ""
            ), refusals
            assert refusals["fsdp_after_refusals"] == 0, refusals
            if dims["dp"] == 1:
                continue
        else:
            assert res["reduces"] == [0] * STEPS
        if "tp" in dims:
            assert res["placements"], "no tensor parallel DTensors"
    assert union == set(ref["grads"]), sorted(union ^ set(ref["grads"]))[:6]
    saved_ref, saved_pp = results[0]["saved_ref"], results[0]["saved_pp"]
    assert set(saved_pp) == set(saved_ref), sorted(set(saved_pp) ^ set(saved_ref))[:6]
    for key, value in saved_ref.items():
        assert saved_pp[key].shape == value.shape, key
        # 3 AdamW steps (lr 1e-2) carry fp32 summation order differences (as the pp-only suite).
        torch.testing.assert_close(saved_pp[key], value, rtol = 1e-3, atol = 2e-3, msg = key)


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_pp2_x_fsdp2_dp2_matches_one_process(arch):
    dims = {"pp": 2, "dp": 2}
    _check(
        _spawn("compose", arch = arch, dims = dims),
        dims,
        norm_rel = STEP0_NORM_REL[arch],
        grad_rel = GRAD_ATOL_REL[arch],
        l2_rel = GRAD_L2_REL[arch],
    )


def test_pp2_x_fsdp2_dp2_1f1b_sharded_layout_matches_one_process():
    # 1F1B, 4 micro-batches, the sharded layout (one FSDP2 unit per owned decoder layer, frozen weights sharded).
    dims = {"pp": 2, "dp": 2}
    _check(
        _spawn("compose", arch = "llama", dims = dims, schedule = "1f1b", n_mb = 4, fit_layout = False),
        dims,
    )


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_pp2_x_tp2_matches_one_process(arch):
    dims = {"pp": 2, "tp": 2}
    _check(
        _spawn("compose", arch = arch, dims = dims),
        dims,
        norm_rel = STEP0_NORM_REL[arch],
        grad_rel = GRAD_ATOL_REL[arch],
        l2_rel = GRAD_L2_REL[arch],
    )


@pytest.mark.parametrize("other", ["dp", "tp"])
def test_pp2_x_size_one_dp_or_tp_matches_one_process(other):
    # (pp 2, dp 1) / (pp 2, tp 1) on 2 ranks: FSDP2 / TP DTensors on a 1-rank mesh, the shape the 2-GPU proof runs.
    dims = {"pp": 2, other: 1}
    _check(_spawn("compose", world = 2, arch = "llama", dims = dims), dims)


def test_pp2_x_dp2_x_tp2_matches_one_process():
    dims = {"pp": 2, "dp": 2, "tp": 2}
    _check(_spawn("compose", world = 8, arch = "llama", dims = dims), dims)


def test_pp2_x_dp2_grad_scaler_skip_is_global():
    results = _spawn("scaler")
    for res in results:
        assert res["unchanged"], res
        assert res["scale"] == 512.0, res


def test_pp_composition_order_and_meshes_are_refused_before_any_change():
    for res in _spawn("order"):
        assert "must come before apply_tensor_parallel" in (res["tp_first"] or ""), res
        assert res["tp_first_unchanged"], res
        assert "must come before FSDP2" in (res["fsdp_first"] or ""), res
        assert res["fsdp_first_unchanged"], res
        assert "1-D pp mesh" in (res["mesh_2d"] or ""), res
        assert res["tp_after_fsdp"], res
        assert res["shard_twice_is_noop"], res
        assert res["composed"], res
    for res in _spawn("pp_tp_mesh"):
        assert "does not compose with Unsloth pipeline parallelism on these meshes" in (
            res["err"] or ""
        ), res
        assert not res["tp"], res
