# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The MoE router auxiliary (load balancing) loss under Ulysses context parallel, through SFTTrainer on a gloo CPU
world, vs the same script in one process: losses (CE + coef x aux), grad norms, every gradient of the first
optimizer step (router, attention and expert LoRA, or every weight) and the weights after the run.

The loss is E x sum_e f_e P_e over the whole micro-batch (f: routed fraction, P: mean routing probability, both
over every layer and valid token). Each cp rank sees only its shard's router logits, so context parallel sums
the per-expert counts, probability sums and row counts over the cp group and forms the loss from those
(context_parallel._cp_load_balancing_loss). Harness: tests/test_parallel_cp_compose_gloo.py (production env,
mesh, sharding, token count and gradient reduction; nothing patched open)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_parallel_cp_compose_gloo import _close, _same_rows, _spawn  # noqa: E402

_AUX = dict(arch = "qwen3_moe", aux_coef = 0.5)
_NO_FUSED_FORWARD = {"UNSLOTH_FUSED_FORWARD": "0"}


@pytest.mark.parametrize(
    "mode, world, env, kw",
    [
        ("lora", 2, {}, {}),
        ("lora", 2, {}, {"padding": "right"}),
        ("lora", 2, {}, {"padding": "short"}),
        ("lora", 2, {}, {"padding": "odd"}),
        ("lora", 2, {}, {"gc": "unsloth_nonre", "padding": "right"}),
        ("lora", 4, {}, {"padding": "right"}),
        ("lora", 4, {"UNSLOTH_FSDP2": "1"}, {}),
        ("full", 2, {}, {"padding": "right"}),
        ("lora", 2, {}, {"hf_loss": True, "padding": "right"}),
        ("lora", 2, _NO_FUSED_FORWARD, {"hf_loss": True, "padding": "right"}),
        ("lora", 2, {}, {"hf_loss": True, "zoo_compile": True, "padding": "right"}),
    ],
    ids = [
        "lora-cp2",
        "lora-cp2-right_pad",
        "lora-cp2-empty_shard",
        "lora-cp2-cp_pad",
        "lora-cp2-unsloth_nonre_gc",
        "lora-dp2_cp2-right_pad",
        "lora-fsdp2_cp2",
        "full-cp2-right_pad",
        "lora-cp2-forward_aux_scaled",
        "lora-cp2-forward_aux_plain",
        "lora-cp2-zoo_compiled_forward",
    ],
)
def test_router_aux_loss_under_cp_matches_one_process(mode, world, env, kw):
    """cp 2 (alone, beside DDP dp 2, or native FSDP2 over dp x cp) equals one process accumulating the same
    micro-batches (the aux loss is per micro-batch, so dp replicas count as accumulation steps) with the router
    aux loss on (coef 0.5): padded rows, an empty shard and a cp-appended token included. The aux enters through
    stock TRL's compute_loss, or (hf_loss, the GPU route) the model forward, rewritten by zoo or plain."""
    cp, dp = 2, world // 2
    ref_kw = {k: v for k, v in kw.items() if k != "gc"}  # checkpointing does not change the math
    ref_env = {k: v for k, v in env.items() if k == "UNSLOTH_FUSED_FORWARD"}
    ref_args = dict(bs = 2, ga = 2 * dp, mode = mode, **ref_kw)
    (ref,) = _spawn("train", 1, ref_env, router_aux = True, **_AUX, **ref_args)
    (off,) = _spawn("train", 1, ref_env, **{**_AUX, "aux_coef": None}, **ref_args)
    assert "error" not in off, off
    # The aux term moves the loss: a run that dropped it could not match.
    assert ref["losses"] != pytest.approx(off["losses"], rel = 1e-4), (ref["losses"], off["losses"])
    results = _spawn("train", world, env, bs = 2, cp = cp, mode = mode, router_aux = True, **_AUX, **kw)
    _same_rows(results, ref, what = (mode, world, env, kw))
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res.get("error"), res.get("trace"))
        # Engaged: the cp statistics ran, scaled for the route the aux loss enters by.
        plain = kw.get("hf_loss") and not res["aux_scaler"]
        assert res["cp_aux_calls"] > 0, res["cp_aux_calls"]
        assert res["router_aux_scale"] == (1 / cp if plain else 1.0), (
            res["router_aux_scale"],
            plain,
        )
        if env.get("UNSLOTH_FUSED_FORWARD") == "0":
            assert not res["aux_scaler"]
        routers = [k for k in res["grads"] if ".mlp.gate." in k or k.endswith(".mlp.gate.weight")]
        assert routers and all(res["grads"][k].abs().sum() > 0 for k in routers), routers
        if kw.get("gc"):
            assert res["memory"] == "unsloth_offload" and res["gc"]["checkpoint_calls"] > 0, res[
                "gc"
            ]
        _close(
            res, ref, what = (mode, world, env, kw, rank), state_atol = 1e-5 if mode == "full" else 1e-6
        )


def test_router_aux_loss_under_cp_with_reentrant_checkpointing_is_refused():
    """Reentrant checkpointing runs the layers under no_grad, so the router logits carry no graph: refused at the
    first forward instead of training without the aux gradient."""
    for res in _spawn("train", 2, bs = 2, cp = 2, router_aux = True, gc = "unsloth", **_AUX):
        error = res.get("error") or ""
        assert "reentrant checkpointing would drop its gradient" in error, error or res["losses"]
