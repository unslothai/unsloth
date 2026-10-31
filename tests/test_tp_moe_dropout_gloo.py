# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallel on a gloo CPU world vs one process: mixture-of-experts blocks run replicated (attention and
dense MLPs still sharded), and dropout draws the masks one process would draw."""

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
from test_tp_native_gloo import _free_port, _train  # noqa: E402
from test_tp_native_oracle import LLAMA, QWEN3_MOE, _batch, build, tpn  # noqa: E402

MOE_OVERRIDES = {"mlp_only_layers": [1]}
MOE_LORA = dict(target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])


def _seeded_train(model, ids, steps, clip, seed):
    torch.manual_seed(seed)  # the same RNG state on every rank and in the one-process reference
    model.train()
    return _train(model, ids, steps, clip)


class _FusedO:
    """o_proj through Unsloth's fused LoRA_W (apply_lora_o, as llama.py's attention forward calls it): its
    lora_dropout runs inside the fused Function, on this rank's row shard of the input."""

    def __init__(self, attn):
        self.attn, self.inner = attn, attn.o_proj.forward
        self.busy = False
        attn.o_proj.forward = self

    def __call__(self, X, *args, **kwargs):
        if self.busy:  # apply_lora_o's own fallback: the projection's forward
            return self.inner(X, *args, **kwargs)
        from unsloth.kernels.fast_lora import apply_lora_o

        self.busy = True
        try:
            return apply_lora_o(self.attn, X)
        finally:
            self.busy = False


def _case_parity(
    rank,
    world,
    tmp,
    repo,
    overrides,
    lora_kwargs,
    dropout_seed = None,
    reentrant_checkpointing = False,
    fused_lora_o = False,
):
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import init_device_mesh

    if fused_lora_o:
        from test_tp_native_gloo import _import_checkout_unsloth

        error = _import_checkout_unsloth()
        assert error is None, error
        from unsloth.models import (
            tp_native as T,
        )  # one tp_native for the fused kernels and tp_replica
    else:
        T = tpn
    ref = build(repo, overrides, **lora_kwargs)
    model = build(repo, overrides, **lora_kwargs)
    for m in (ref, model):
        if reentrant_checkpointing:
            # The first forward of a reentrant checkpoint runs under no_grad, the recompute with grad.
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
            m.enable_input_require_grads()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    T.apply_tensor_parallel(model, mesh["tp"])
    if fused_lora_o:
        for m in (ref, model):
            for layer in T._unwrap(m).model.layers:
                _FusedO(layer.self_attn)
    plan = model._unsloth_tp_plan
    generated = model.generate(
        input_ids = torch.zeros(1, 2, dtype = torch.long), max_new_tokens = 2, pad_token_id = 0
    )
    ids = _batch(ref)
    if dropout_seed is None:
        ref_out = _train(ref, ids, 3, torch.nn.utils.clip_grad_norm_)
        tp_out = _train(model, ids, 3, T.tp_clip_grad_norm_)
    else:
        ref_out = _seeded_train(ref, ids, 3, torch.nn.utils.clip_grad_norm_, dropout_seed)
        tp_out = _seeded_train(model, ids, 3, T.tp_clip_grad_norm_, dropout_seed)
        other = build(repo, overrides, **lora_kwargs)
        other_out = _seeded_train(other, ids, 1, torch.nn.utils.clip_grad_norm_, dropout_seed + 1)
    ref_dir, tp_dir = os.path.join(tmp, f"ref{rank}"), os.path.join(tmp, "tp")
    ref.save_pretrained(ref_dir)
    T.save_tp_adapter(model, tp_dir)
    return {
        "ref": ref_out,
        "tp": tp_out,
        "other_loss": None if dropout_seed is None else other_out[2][0],
        "regions": [(r.path, r.kind) for r in plan.regions],
        "replicated_moe": plan.replicated_moe,
        "moe_synced": all(b.__dict__.get("_unsloth_tp_size") == world for b in plan.moe_blocks),
        "generated": tuple(generated.shape),
        "saved_ref": load_file(os.path.join(ref_dir, "adapter_model.safetensors")),
        "saved_tp": load_file(os.path.join(tp_dir, "adapter_model.safetensors"))
        if rank == 0
        else None,
    }


def _case_mean_half(rank, world, tmp, dtype):
    import torch.distributed as dist

    out = {}
    for value in (40000.0, 3.0):
        x = torch.full((4,), value, dtype = getattr(torch, dtype))
        out[value] = tpn._MeanFromTP.apply(x, dist.group.WORLD, world)
    return out


CASES = {"parity": _case_parity, "mean_half": _case_mean_half}


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
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
    **kwargs,
):
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
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


def _check(results):
    for rank, res in enumerate(results):
        ref_logits, ref_grads, ref_losses, ref_norms = res["ref"]
        logits, grads, losses, norms = res["tp"]
        torch.testing.assert_close(logits, ref_logits, rtol = 1e-5, atol = 1e-5)
        assert set(grads) == set(ref_grads), sorted(set(grads) ^ set(ref_grads))[:6]
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol = 1e-4, atol = 1e-6, msg = name)
        assert losses == pytest.approx(ref_losses, rel = 1e-5, abs = 1e-6), (rank, losses, ref_losses)
        assert norms == pytest.approx(ref_norms, rel = 1e-5), (rank, norms, ref_norms)
        assert res["generated"] == (1, 4)
    saved_tp, saved_ref = results[0]["saved_tp"], results[0]["saved_ref"]
    assert set(saved_tp) == set(saved_ref)
    for key in saved_ref:
        # 3 AdamW steps (eps 1e-8) turn fp32 reduction-order noise on tiny expert grads into ~5e-5 here; a
        # missed or doubled reduction moves weights by ~lr = 1e-2.
        torch.testing.assert_close(saved_tp[key], saved_ref[key], rtol = 1e-3, atol = 2e-4, msg = key)
    return results[0]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_tp2_replicated_mean_does_not_overflow_half(dtype):
    # 40000 is finite in fp16 but two replicas summed in fp16 are inf: the mean must not overflow.
    for res in _spawn("mean_half", 2, dtype = dtype):
        for value, y in res.items():
            assert y.dtype == getattr(torch, dtype)
            assert torch.equal(y, torch.full_like(y, value)), (value, y)


def test_tp2_moe_runs_replicated_and_matches_one_process():
    res = _check(_spawn("parity", 2, repo = QWEN3_MOE, overrides = MOE_OVERRIDES, lora_kwargs = MOE_LORA))
    assert res["replicated_moe"] == 1 and res["moe_synced"]
    assert sorted(k for _, k in res["regions"]) == ["attention", "attention", "mlp"], res["regions"]
    assert any("experts" in k for k in res["saved_tp"]), sorted(res["saved_tp"])[:8]


@pytest.mark.parametrize(
    "overrides,lora_kwargs",
    [
        ({}, {"lora_dropout": 0.25}),  # lora_dropout: column input replicated, row input sharded
        (
            {"attention_dropout": 0.25},
            {"lora_dropout": 0.25},
        ),  # + eager attention probabilities, per head
    ],
    ids = ["lora_dropout", "attention_and_lora_dropout"],
)
def test_tp2_dropout_draws_one_process_masks(overrides, lora_kwargs):
    results = _spawn(
        "parity", 2, repo = LLAMA, overrides = overrides, lora_kwargs = lora_kwargs, dropout_seed = 1234
    )
    res = _check(results)
    assert res["other_loss"] != pytest.approx(res["ref"][2][0], rel = 1e-6), (
        res["other_loss"],
        res["ref"][2],
    )


def test_tp2_attention_dropout_in_a_fused_kernel_is_refused():
    model = build(LLAMA, {"attention_dropout": 0.25})
    tpn._text_config(model)._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match = "eager attention only"):
        tpn.derive_tp_plan(model, 2)


@pytest.mark.parametrize(
    "overrides,kwargs",
    [
        # Reentrant gradient checkpointing: the attention dropout masks of the no-grad first forward and of the
        # recompute must both be one process's masks.
        ({"attention_dropout": 0.25}, {"reentrant_checkpointing": True}),
        # One KV head < tp 2: the replicated key / value projections read the whole (replicated) input.
        ({"num_attention_heads": 4, "num_key_value_heads": 1}, {}),
        # Unsloth's fused LoRA o_proj (LoRA_W) with lora_dropout: its input is sharded over tp.
        ({}, {"fused_lora_o": True}),
    ],
    ids = ["reentrant_checkpointing", "kv_replicated_lora_dropout", "fused_lora_o_dropout"],
)
def test_tp2_dropout_paths_draw_one_process_masks(overrides, kwargs):
    results = _spawn(
        "parity",
        2,
        repo = LLAMA,
        overrides = overrides,
        lora_kwargs = {"lora_dropout": 0.25},
        dropout_seed = 1234,
        **kwargs,
    )
    res = _check(results)
    assert res["other_loss"] != pytest.approx(res["ref"][2][0], rel = 1e-6)
