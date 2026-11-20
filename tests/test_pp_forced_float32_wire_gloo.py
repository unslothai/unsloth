# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PP over a forced float32 model: stages get back their sender's 16-bit dtype off the float32 wire (gpt-oss
style upcasting layers keep float32). gloo world of 2 vs single-process grad accumulation."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pp_native_gloo as base  # noqa: E402

ppn = base.ppn
REPOS = {"qwen3": base.QWEN3, "qwen3_moe": base.QWEN3_MOE, "llama": base.LLAMA}
N_MB = 2


def _model(arch, upcast):
    model = base.build(REPOS[arch], layers = 4).to(torch.bfloat16)
    model._unsloth_forced_float32 = True  # what from_pretrained stamps on a FORCE_FLOAT32 family
    seen = []
    layers = ppn._find_parts(model).layers
    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(
            lambda m, args, kwargs, i = i: seen.append(
                (i, str((args[0] if args else kwargs["hidden_states"]).dtype))
            ),
            with_kwargs = True,
        )
    if upcast:
        # gpt-oss style: layers compute in 16-bit and return float32.
        def to_bf16(module, args, kwargs):
            if args:
                return (args[0].to(torch.bfloat16), *args[1:]), kwargs
            return args, dict(kwargs, hidden_states = kwargs["hidden_states"].to(torch.bfloat16))

        def to_float32(module, args, out):
            return (out[0].float(), *out[1:]) if isinstance(out, tuple) else out.float()

        for layer in layers:
            layer.register_forward_pre_hook(to_bf16, with_kwargs = True)
            layer.register_forward_hook(to_float32)
        model.get_output_embeddings().register_forward_pre_hook(to_bf16, with_kwargs = True)
    return model, seen


def _case_forced(rank, world, tmp, arch, upcast, schedule):
    from torch.distributed.device_mesh import init_device_mesh

    ref, ref_seen = _model(arch, upcast)
    ids, labels = base.batch(ref, bsz = 2 * N_MB)
    n_items = int((labels[:, 1:] != -100).sum())
    ref_loss = 0.0
    for chunk_ids, chunk_labels in zip(ids.chunk(N_MB), labels.chunk(N_MB)):
        out = ref(
            input_ids = chunk_ids, labels = chunk_labels, num_items_in_batch = n_items, use_cache = False
        )
        out.loss.backward()
        ref_loss += float(out.loss)
    ref_grads = {n: p.grad.float() for n, p in ref.named_parameters() if p.grad is not None}
    ref.eval()
    with torch.no_grad():
        ref_eval = float(ref(input_ids = ids, labels = labels, use_cache = False).loss)

    model, seen = _model(arch, upcast)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = N_MB, schedule = schedule)
    try:
        loss = float(pp.step(ids, labels))
        error = None
    except Exception as exc:  # the failure on main: float32 input against bf16 weights
        loss, error = None, f"{type(exc).__name__}: {exc}"[:400]
    grads = {n: p.grad.float() for n, p in model.named_parameters() if p.grad is not None}
    eval_loss = None
    if error is None:
        model.eval()
        eval_loss = float(pp.eval_loss(ids, labels))
    return {
        "ref_loss": ref_loss,
        "ref_eval": ref_eval,
        "ref_grads": ref_grads,
        "ref_seen": sorted(set(ref_seen)),
        "loss": loss,
        "eval_loss": eval_loss,
        "error": error,
        "grads": grads,
        "seen": sorted(set(seen)),
        "wire": str(pp.activation_dtype),
        "restores": getattr(pp.ctx, "wire_dtypes", None) is not None,
        "bounds": pp.bounds,
    }


def _worker(rank, world, port, kwargs, tmp):
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
        torch.save(_case_forced(rank, world, tmp, **kwargs), os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(world = 2, **kwargs):
    if base._torch_pipelining_error():
        pytest.skip(base._torch_pipelining_error())
    ctx = multiprocessing.get_context("spawn")
    port = base._free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, kwargs, tmp)) for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


@pytest.mark.parametrize(
    "arch, upcast, schedule",
    [
        ("qwen3", False, "gpipe"),
        ("qwen3_moe", False, "1f1b"),
        ("llama", True, "gpipe"),
    ],
)
def test_pp2_forced_float32_matches_one_process(arch, upcast, schedule):
    results = _spawn(arch = arch, upcast = upcast, schedule = schedule)
    errors = [res["error"] for res in results]
    assert errors == [None] * len(results), errors
    union = set()
    for rank, res in enumerate(results):
        assert res["wire"] == "torch.float32", res["wire"]
        assert res["restores"] or upcast, rank  # float32 hidden states already arrive as they are
        start, end = res["bounds"][rank]
        assert res["seen"] == [s for s in res["ref_seen"] if start <= s[0] < end], (
            rank,
            res["seen"],
        )
        assert {d for _, d in res["ref_seen"]} == (
            {"torch.bfloat16", "torch.float32"} if upcast else {"torch.bfloat16"}
        )
        assert res["loss"] == pytest.approx(res["ref_loss"], rel = 1e-3), (
            rank,
            res["loss"],
            res["ref_loss"],
        )
        assert res["eval_loss"] == pytest.approx(res["ref_eval"], rel = 1e-3), (
            rank,
            res["eval_loss"],
            res["ref_eval"],
        )
        assert res["grads"], rank
        for name, grad in res["grads"].items():
            ref = res["ref_grads"][name]
            scale = float(ref.abs().max()) or 1.0
            diff = float((grad - ref).abs().max())
            assert diff <= 2e-2 * scale, (rank, name, diff, scale)
        union |= set(res["grads"])
    assert union == set(results[0]["ref_grads"]), sorted(union ^ set(results[0]["ref_grads"]))[:6]


def test_pp2_bf16_model_without_force_does_not_restore(monkeypatch):
    """A 16-bit model that is not forced sends its own dtype: no store round trip, nothing to restore."""
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising = False)
    model = base.build(base.QWEN3, layers = 2).to(torch.bfloat16)
    parts = ppn._find_parts(model)
    assert ppn._default_activation_dtype(model, parts) == torch.bfloat16
    model._unsloth_forced_float32 = True
    assert ppn._default_activation_dtype(model, parts) == torch.float32
