# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PPConfig.text_only: gloo pp 2 vs one process, fp32 (losses, grads, saved adapter); media batches refused."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _tiny_mm as tiny  # noqa: E402
from test_pp_native_gloo import _load  # noqa: E402

pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("transformers.models.gemma4")

ppn = _load("pp_native")
STEPS = 4
LR = 0.01  # SGD: Adam's per-element normalisation turns ~1e-8 gradient differences into whole steps


def _clip_fp64(params, max_norm):
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.stack([g.double().square().sum() for g in grads]).sum().sqrt()
    coef = torch.clamp(max_norm / (total + 1e-6), max = 1.0)
    for g in grads:
        g.mul_(coef.to(g.dtype))
    return float(total)


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def reference(
    model,
    ids,
    labels,
    steps = STEPS,
):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr = LR)
    losses, grads = [], []
    for _ in range(steps):
        out = model(input_ids = ids, labels = labels, use_cache = False)
        out.loss.backward()
        grads.append(_grads(model))
        _clip_fp64(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss))
    return losses, grads


def pipeline(
    model,
    pp,
    ids,
    labels,
    steps = STEPS,
):
    opt = torch.optim.SGD(pp.parameters(), lr = LR)
    losses, grads = [], []
    for _ in range(steps):
        loss = pp.step(ids, labels)
        grads.append(_grads(model))
        pp.clip_grad_norm_(1.0)
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
    return losses, grads


def _towers_left(model):
    inner = ppn._unwrap(model)
    out = []
    for name, module in inner.named_modules():
        if type(module).__name__ == "_PPTowerStub":
            out.append((module._pp_tower, sum(p.numel() for p in module.parameters())))
    return sorted(out)


def _case_parity(
    rank,
    world,
    tmp,
    arch,
    keep_replica = False,
    layers = 4,
):
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import init_device_mesh

    ref = tiny.build(arch, layers)
    ids, labels = tiny.batch()
    ref_res = reference(ref, ids, labels)
    if rank == 0:
        ref.save_pretrained(os.path.join(tmp, "ref"))
    model = tiny.build(arch, layers)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    config = ppn.PPConfig(text_only = True, keep_replica = keep_replica)
    import gc
    import weakref

    before = {n: weakref.ref(m) for n, m in ppn._unwrap(model).named_modules()}
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 2, config = config)
    towers = _towers_left(model)
    gc.collect()
    stubs = [n for n, m in ppn._unwrap(model).named_modules() if type(m).__name__ == "_PPTowerStub"]
    towers_alive = [n for n in stubs if n in before and before[n]() is not None]
    tower_params_in_stage = [
        n for n, _ in model.named_parameters() if "tower" in n or ".visual." in n
    ]
    media_error = None
    try:
        pp.step(ids, labels, pixel_values = torch.zeros(1, 3, 8, 8))
    except ValueError as exc:
        media_error = str(exc)
    pp_res = pipeline(model, pp, ids, labels)
    pp.save_adapter(os.path.join(tmp, "pp"))
    out = {
        "ref": ref_res,
        "pp": pp_res,
        "bounds": pp.bounds,
        "towers": towers,
        "tower_params_in_stage": tower_params_in_stage,
        "media_error": media_error,
        "kept": len(pp._kept or ()),
        "towers_alive": towers_alive,
    }
    if rank == 0:
        out["saved_ref"] = load_file(os.path.join(tmp, "ref", "adapter_model.safetensors"))
        out["saved_pp"] = load_file(os.path.join(tmp, "pp", "adapter_model.safetensors"))
    if keep_replica:
        # What merged / GGUF exports read: the whole model back in place, towers included, every stage's training.
        _load("pp_env").gather_replica(model, "save_pretrained_merged")
        full, want = model.state_dict(), ref.state_dict()
        out["replica_keys"] = (sorted(full), sorted(want))
        out["replica_diff"] = max(
            float((full[k].float() - want[k].float()).abs().max()) for k in want
        )
    return out


def _case_refusals(rank, world, tmp, arch):
    from torch.distributed.device_mesh import init_device_mesh

    model = tiny.build(arch)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    errors = {}
    try:
        ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 2)
    except ValueError as exc:
        errors["no_flag"] = str(exc)
    errors["untouched"] = not ppn.is_pipeline_parallel(model) and not _towers_left(model)
    try:
        ppn.check_pipeline_model(model)  # at load: the text-only decision waits for the split
        errors["at_load"] = None
    except ValueError as exc:
        errors["at_load"] = str(exc)
    return errors


CASES = {"parity": _case_parity, "refusals": _case_refusals}


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
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    finished = False
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
        finished = True
    finally:
        dist.destroy_process_group()
    if finished:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def spawn(
    case,
    world = 2,
    **kwargs,
):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
        codes = [p.exitcode for p in procs]
        for p in procs:
            if p.is_alive():
                p.kill()
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


def grad_error(got, want):
    """Largest |difference| over the tensor's largest |entry| (fp32 reductions in another order)."""
    assert set(got) == set(want) and want, sorted(set(got) ^ set(want))[:6]
    return max(
        (float((got[n] - want[n]).abs().max() / want[n].abs().max().clamp_min(1e-12)), n)
        for n in want
    )


def check_parity(
    results,
    grad_tol = 1e-3,
    loss_tol = 1e-5,
):
    """Losses and every owned gradient within tolerance of one process at every step; returns the worst error."""
    ref_losses, ref_grads = results[0]["ref"]
    worst = (0.0, None)
    for step, want in enumerate(ref_grads):
        got = {}
        for res in results:
            for name, g in res["pp"][1][step].items():
                assert name not in got, f"{name} has a gradient on two stages"
                got[name] = g
        worst = max(worst, grad_error(got, want))
    assert worst[0] < grad_tol, worst
    for res in results:
        assert res["pp"][0] == pytest.approx(ref_losses, rel = 0, abs = loss_tol), (
            res["pp"][0],
            ref_losses,
        )
    assert len(set(ref_losses)) == len(ref_losses)  # the steps train
    return worst


def check_saved(results):
    saved_ref, saved_pp = results[0]["saved_ref"], results[0]["saved_pp"]
    assert set(saved_pp) == set(saved_ref), sorted(set(saved_pp) ^ set(saved_ref))[:6]
    for key in saved_ref:
        torch.testing.assert_close(saved_pp[key], saved_ref[key], rtol = 1e-4, atol = 1e-5, msg = key)


@pytest.mark.parametrize(
    "arch", ["qwen3_5", "gemma4", "gemma4_moe", "gemma4_unified", "muse_glimmer"]
)
def test_pp2_text_only_matches_one_process(arch):
    results = spawn("parity", arch = arch)
    print(arch, "worst gradient error", check_parity(results))
    check_saved(results)
    # The tower entries of the all-linear adapter are saved, unchanged (no tower runs in a text-only step).
    assert any(
        ".lora_A." in k and ("tower" in k or ".visual." in k or "embed_vision" in k)
        for k in results[0]["saved_ref"]
    )
    for res in results:
        assert res["towers"] and all(count == 0 for _, count in res["towers"]), res["towers"]
        assert not res["tower_params_in_stage"], res["tower_params_in_stage"][:4]
        assert "not dropped" in res["media_error"], res["media_error"]
        assert res["kept"] == 0
        assert not res["towers_alive"], res["towers_alive"]  # dropped towers free their memory
    assert results[0]["bounds"][0][0] == 0 and results[1]["bounds"][-1][1] == 4


def test_pp2_text_only_keep_replica_keeps_towers_on_the_cpu():
    results = spawn("parity", arch = "gemma4", keep_replica = True)
    check_parity(results)
    check_saved(results)
    for res in results:
        assert res["kept"] >= 2  # the towers besides the other stage's modules
        assert res["towers_alive"]
        keys, want = res["replica_keys"]
        assert keys == want and any("vision_tower" in k for k in keys)
        assert res["replica_diff"] < 1e-4, res[
            "replica_diff"
        ]  # every weight as one process trained it


@pytest.mark.parametrize("arch", ["qwen3_5", "gemma4"])
def test_multimodal_without_text_only_is_refused_untouched(arch):
    for res in spawn("refusals", arch = arch):
        assert "PPConfig(text_only = True)" in res["no_flag"], res["no_flag"]
        assert res["untouched"]
        assert res["at_load"] is None, res["at_load"]


def test_towers_are_found_beside_the_language_model():
    expected = {
        "qwen3_5": {"visual"},
        "gemma4": {"vision_tower", "embed_vision"},
        "gemma4_unified": {"embed_vision"},
        "muse_glimmer": {"vision_tower", "vision_adapter", "vision_projection"},
    }
    for arch, names in expected.items():
        parts = ppn._find_parts(tiny.build(arch, lora = False))
        assert {name for _, name, _ in parts.towers} == names, (arch, parts.towers)
        assert type(parts.decoder).__name__.endswith("TextModel"), type(parts.decoder)


def test_layer_loop_found_through_a_closure_wrapper():
    # Unsloth Zoo wraps Gemma4TextModel.forward in a closure without __wrapped__ (KV-sharing / checkpointing fix).
    model = tiny.build("gemma4", lora = False)
    decoder = ppn._find_parts(model).decoder
    original = type(decoder).forward

    def outer(orig):
        def forward(self, *args, **kwargs):
            return orig(self, *args, **kwargs)

        return forward

    decoder.forward = outer(original).__get__(decoder)
    assert ppn._layer_call_format(decoder, "layers") == "tensor"


def _pp_env():
    return _load("pp_env")


def test_trainer_text_only_detection():
    env = _pp_env()
    import datasets

    text = datasets.Dataset.from_dict({"messages": [[{"role": "user", "content": "hi"}]]})
    chat = datasets.Dataset.from_dict(
        {
            "messages": [
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}]
            ]
        }
    )
    images = datasets.Dataset.from_dict({"images": [[]], "text": ["x"]})
    assert env.media_evidence(text) is None
    assert "'image' part" in env.media_evidence(chat)
    assert "column 'images'" in env.media_evidence(None, {"test": images})

    class UnslothVisionDataCollator:
        pass

    assert "UnslothVisionDataCollator" in env.media_evidence(
        text, None, UnslothVisionDataCollator()
    )


def test_trainer_batch_with_media_is_refused():
    env = _pp_env()

    class Stage:
        env_layout = None
        num_stages = 2

    with pytest.raises(NotImplementedError, match = "not dropped"):
        env._batch(
            Stage(),
            {
                "input_ids": torch.zeros(1, 2),
                "labels": torch.zeros(1, 2),
                "pixel_values": torch.zeros(1),
            },
        )
