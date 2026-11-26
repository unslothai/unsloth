# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Expert parallel x Ulysses cp via SFTTrainer on gloo vs one process; expert grads keep DDP's 1 / world scale."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_parallel_cp_compose_gloo import SEQ, _close, _ref, _same_rows, _spawn  # noqa: E402

import transformers  # noqa: E402

# Expert parallel shards fused expert stacks: transformers 4.x keeps Qwen3-MoE experts as per-expert modules.
_FUSED = pytest.mark.skipif(
    int(transformers.__version__.split(".")[0]) < 5,
    reason = "expert parallel needs fused MoE expert stacks (transformers >= 5)",
)


@_FUSED
@pytest.mark.parametrize(
    "world, ep, aux",
    [(2, 2, False), (2, 2, True), (4, 2, False), (4, 4, False)],
    ids = ["ep2_cp2", "ep2_cp2-router_aux", "dp2_cp2-ep2_replicas", "dp2_cp2-ep4"],
)
def test_ep_x_cp_qwen3_moe_lora_matches_one_process(world, ep, aux):
    cp = 2
    kw = dict(arch = "qwen3_moe", mode = "lora")
    if aux:
        kw.update(router_aux = True, aux_coef = 0.5)
    # The aux loss is per micro-batch, so dp replicas count as accumulation steps in the reference.
    ref = (
        _spawn("train", 1, bs = 2, ga = 2 * (world // cp), **kw)[0]
        if aux
        else _ref(2 * world // cp, **kw)
    )
    results = _spawn("train", world, {"UNSLOTH_EP_SIZE": str(ep)}, bs = 2, cp = cp, **kw)
    _same_rows(results, ref, what = (world, ep, aux))
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res.get("error"), res.get("trace"))
        assert res["is_ep"] and res["cp_backend"] == "ulysses", res
        assert res["seen_lengths"] == [SEQ // cp] and res["ulysses_calls"] >= 2, res
        experts = [k for k in res["grads"] if ".experts." in k and "lora_" in k]
        assert experts and all(res["grads"][k].abs().sum() > 0 for k in experts), experts
        if aux:
            assert res["cp_aux_calls"] > 0, res["cp_aux_calls"]
        _close(res, ref, what = (world, ep, aux, rank))


@_FUSED
def test_ep_x_cp_from_the_launch_variables_matches_one_process():
    """UNSLOTH_PARALLEL="ep=2,cp=2" passes the registry check and trains like one process."""
    kw = dict(arch = "qwen3_moe", mode = "lora")
    ref = _ref(2, **kw)
    env = {"UNSLOTH_EP_SIZE": "2", "UNSLOTH_CP_SIZE": "2", "UNSLOTH_CP_BACKEND": "ulysses"}
    results = _spawn("train", 2, env, bs = 2, cp = 2, sft_cp = False, **kw)
    _same_rows(results, ref, what = "env ep2 x cp2")
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res.get("error"), res.get("trace"))
        assert (
            res["is_ep"] and res["cp_backend"] == "ulysses" and res["seen_lengths"] == [SEQ // 2]
        ), res
        _close(res, ref, what = ("env ep2 x cp2", rank))


@_FUSED
def test_ep_x_ring_cp_is_refused():
    """Only Ulysses has ep x cp evidence: ring attention under expert parallel is refused before training."""
    env = {"UNSLOTH_EP_SIZE": "2", "UNSLOTH_CP_SIZE": "2", "UNSLOTH_CP_BACKEND": "ring"}
    for res in _spawn("train", 2, env, bs = 2, cp = 2, sft_cp = False, arch = "qwen3_moe", mode = "lora"):
        assert "no supported cell covers it" in (res.get("error") or ""), res.get(
            "error"
        ) or res.get("losses")


def test_unsloth_parallel_ep_with_cp_translates():
    from unsloth.parallel_env import parse_parallel_spec, translate

    out = translate(parse_parallel_spec("ep=2,cp=2"), {"WORLD_SIZE": "2"})
    assert out["UNSLOTH_EP_SIZE"] == "2" and out["UNSLOTH_CP_SIZE"] == "2", out
    out = translate(parse_parallel_spec("ep=2,cp=2"), {"WORLD_SIZE": "4"})
    assert out["UNSLOTH_EP_SIZE"] == "2" and out["UNSLOTH_CP_SIZE"] == "2", out
    with pytest.raises(ValueError, match = "must divide the number of ranks"):
        translate(parse_parallel_spec("ep=4,cp=2"), {"WORLD_SIZE": "2"})
    for flag in ("ep=2,tp=2", "ep=2,pp=2", "ep=2,cp=2,fsdp=1"):
        with pytest.raises(ValueError, match = "cannot be combined"):
            translate(parse_parallel_spec(flag), {"WORLD_SIZE": "4"})
