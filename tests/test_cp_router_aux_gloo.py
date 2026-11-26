# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MoE router aux loss under Ulysses cp (statistics summed over the cp group) vs one process, via SFTTrainer on gloo."""

import os
import re
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
    """The aux loss is per micro-batch, so dp replicas count as accumulation steps in the reference."""
    cp, dp = 2, world // 2
    ref_kw = {k: v for k, v in kw.items() if k != "gc"}  # checkpointing does not change the math
    ref_env = {k: v for k, v in env.items() if k == "UNSLOTH_FUSED_FORWARD"}
    ref_args = dict(bs = 2, ga = 2 * dp, mode = mode, **ref_kw)
    (ref,) = _spawn("train", 1, ref_env, router_aux = True, **_AUX, **ref_args)
    (off,) = _spawn("train", 1, ref_env, **{**_AUX, "aux_coef": None}, **ref_args)
    assert "error" not in off, off
    assert ref["losses"] != pytest.approx(off["losses"], rel = 1e-4), (ref["losses"], off["losses"])
    results = _spawn("train", world, env, bs = 2, cp = cp, mode = mode, router_aux = True, **_AUX, **kw)
    _same_rows(results, ref, what = (mode, world, env, kw))
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res.get("error"), res.get("trace"))
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
    """Reentrant checkpointing runs under no_grad, so the aux loss would get no gradient."""
    for res in _spawn("train", 2, bs = 2, cp = 2, router_aux = True, gc = "unsloth", **_AUX):
        error = res.get("error") or ""
        assert "reentrant checkpointing would drop its gradient" in error, error or res["losses"]


_ROUTES = {
    "trl": ({}, {}),
    "forward_aux_scaled": ({}, {"hf_loss": True}),
    "forward_aux_plain": (_NO_FUSED_FORWARD, {"hf_loss": True}),
}


@pytest.mark.parametrize("padding", ["right", "odd"], ids = ["right_pad", "cp_pad"])
@pytest.mark.parametrize("route", list(_ROUTES))
def test_router_aux_eval_loss_under_cp_matches_one_process(route, padding):
    """evaluate() under cp 2 reports one process's eval loss (token CE + coef x aux) for each route the aux term
    enters by, with right padded rows or a cp-appended token and no attention_mask."""
    env, kw = _ROUTES[route]
    args = dict(bs = 2, ga = 1, steps = 2, eval_only = True, padding = padding, **_AUX, **kw)
    (ref,) = _spawn("train", 1, env, router_aux = True, **args)
    (off,) = _spawn("train", 1, env, **{**args, "aux_coef": None})
    assert "error" not in ref and "error" not in off, (ref, off)
    assert ref["eval_loss"] != pytest.approx(off["eval_loss"], rel = 1e-4), (ref, off)
    for rank, res in enumerate(_spawn("train", 2, env, cp = 2, router_aux = True, **args)):
        assert "error" not in res, (rank, res.get("error"), res.get("trace"))
        assert res["cp_aux_calls"] > 0 and res["router_aux_scale"] is not None, res
        assert res["rows_read"] == ref["rows_read"], (res["rows_read"], ref["rows_read"])
        assert res["eval_loss"] == pytest.approx(ref["eval_loss"], rel = 1e-5), (
            route,
            padding,
            rank,
            res["eval_loss"],
            ref["eval_loss"],
            off["eval_loss"],
        )


@pytest.mark.parametrize("padding", ["right", "odd"], ids = ["right_pad", "cp_pad"])
def test_router_aux_eval_loss_under_cp_inside_no_grad_matches_one_process(padding):
    """evaluate() called under torch.no_grad() still drops the cp-appended token from the router statistics."""
    args = dict(bs = 2, ga = 1, steps = 2, eval_only = True, padding = padding, **_AUX)
    (ref,) = _spawn("train", 1, {}, router_aux = True, eval_no_grad = True, **args)
    assert "error" not in ref, ref
    for rank, res in enumerate(
        _spawn("train", 2, {}, cp = 2, router_aux = True, eval_no_grad = True, **args)
    ):
        assert "error" not in res, (rank, res.get("error"), res.get("trace"))
        assert res["cp_aux_calls"] > 0, res
        assert res["eval_loss"] == pytest.approx(ref["eval_loss"], rel = 1e-5), (
            padding,
            rank,
            res["eval_loss"],
            ref["eval_loss"],
        )


# Families beside qwen3_moe whose router aux loss cp forms from summed statistics (tiny random configs in
# test_parallel_cp_compose_gloo._MOE_CONFIGS). gpt_oss has no sdpa attention in transformers: eager.
_FAMILIES = {
    "mixtral": {},
    "qwen2_moe": {},
    "olmoe": {},
    "qwen3_5_moe": {},
    "qwen3_next": {},
    "gpt_oss": {"attn": "eager"},
}
_FAMILY_CELLS = [("lora", None), ("lora", "right"), ("lora", "odd"), ("full", None)]


@pytest.mark.parametrize(
    "mode, padding",
    _FAMILY_CELLS,
    ids = ["lora-plain", "lora-right_pad", "lora-cp_pad", "full-plain"],
)
@pytest.mark.parametrize("arch", list(_FAMILIES))
def test_family_router_aux_loss_under_cp_matches_one_process(arch, mode, padding):
    """Per family, Ulysses cp 2 vs one process with the aux loss on (and changing the loss): losses, first-step
    gradients (1e-6) and the trained state. Full finetuning also checks the routers' own gradients."""
    kw = dict(arch = arch, aux_coef = 0.5, mode = mode, bs = 2, **_FAMILIES[arch])
    if padding:
        kw["padding"] = padding
    (ref,) = _spawn("train", 1, router_aux = True, ga = 2, **kw)
    (off,) = _spawn("train", 1, ga = 2, **{**kw, "aux_coef": None})
    assert "error" not in ref and "error" not in off, (ref.get("error"), off.get("error"))
    assert ref["losses"] != pytest.approx(off["losses"], rel = 1e-4), (ref["losses"], off["losses"])
    results = _spawn("train", 2, cp = 2, router_aux = True, **kw)
    _same_rows(results, ref, what = (arch, mode, padding))
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res.get("error"), res.get("trace"))
        assert res["cp_aux_calls"] > 0 and res["router_aux_scale"] == 1.0, res["cp_aux_calls"]
        if mode == "full":
            routers = [k for k in res["grads"] if re.search(r"\.mlp\.(gate|router)\.weight$", k)]
            assert routers and all(res["grads"][k].abs().sum() > 0 for k in routers), routers
        # Two Adam steps (eps 1e-4) carry first-step gradient differences of ~1e-7 into the state at ~1e-6.
        _close(res, ref, what = (arch, mode, padding, rank), state_atol = 1e-5)


@pytest.mark.parametrize("arch", list(_FAMILIES))
def test_family_router_aux_eval_loss_under_cp_matches_one_process(arch):
    """evaluate() under cp 2 with a cp-appended token equals one process's eval loss, per family."""
    args = dict(bs = 2, ga = 1, steps = 2, eval_only = True, padding = "odd", arch = arch, aux_coef = 0.5)
    args.update(_FAMILIES[arch])
    (ref,) = _spawn("train", 1, router_aux = True, **args)
    (off,) = _spawn("train", 1, **{**args, "aux_coef": None})
    assert "error" not in ref and "error" not in off, (ref, off)
    assert ref["eval_loss"] != pytest.approx(off["eval_loss"], rel = 1e-4), (ref, off)
    for rank, res in enumerate(_spawn("train", 2, cp = 2, router_aux = True, **args)):
        assert "error" not in res, (rank, res.get("error"), res.get("trace"))
        assert res["cp_aux_calls"] > 0, res
        assert res["eval_loss"] == pytest.approx(ref["eval_loss"], rel = 1e-5), (arch, rank, res, ref)


def test_phimoe_router_aux_loss_under_cp_is_refused():
    """PhiMoE's SparseMixer samples its training routes from the torch RNG: cp keeps refusing its aux loss."""
    for res in _spawn(
        "train", 2, bs = 2, cp = 2, mode = "full", router_aux = True, arch = "phimoe", aux_coef = 0.5
    ):
        error = res.get("error") or ""
        assert "router" in error and "context parallel" in error, error or res.get("losses")
