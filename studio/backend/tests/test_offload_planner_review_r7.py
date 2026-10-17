# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fitter arm a declined spill really launches (the caller's slot count, its cache and its
micro-batch), GLM-5.3-Flash's unpriced indexer cache, and the lazy mode a CPU recovery inherits."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference.offload_layout import BlockLayout, ModelLayout  # noqa: E402
from core.inference.offload_planner import (  # noqa: E402
    ContextPolicy,
    PlanOptions,
    plan_placement,
)

GIB = 1024**3
MIB = 1024**2


def _dense(state_gib: float) -> ModelLayout:
    """32 dense blocks of 0.45 GiB and a per-slot recurrent state: fewer slots free real VRAM."""
    blocks = tuple(BlockLayout(i, int(0.40 * GIB), int(0.05 * GIB)) for i in range(32))
    return ModelLayout(
        arch = "llama",
        n_layers = 32,
        n_attention_layers = 32,
        blocks = blocks,
        lm_head_bytes = int(0.5 * GIB),
        token_embd_bytes = int(0.5 * GIB),
        other_resident_bytes = int(0.01 * GIB),
        kv_bytes_per_token_f16 = 2 * 128 * 2 * 2 * 32,
        n_ctx_train = 131072,
        recurrent_bytes = int(state_gib * GIB),
        complete = True,
    )


def _opts(**kw) -> PlanOptions:
    base = dict(
        require_cost_win = True,
        context_policy = ContextPolicy.NEVER_REDUCE,
        n_parallel = 4,
        min_parallel = 1,
        kv_unified = True,
    )
    base.update(kw)
    return PlanOptions(**base)


# ------------------------------------------------ findings 1 / 2: the fitter keeps the caller's slots


def test_the_spill_gate_prices_the_fitter_at_the_slots_a_decline_launches():
    """Rung 1 cuts 4 slots to 1 and the rest still spills. Declining launches --parallel 4, which
    holds four recurrent states, but the fitter arm was priced with one: 8542 ms against the
    spill's 8355, a decline. At the four it launches the fitter costs 10026 ms and the spill
    wins."""
    layout = _dense(0.3)
    opts = _opts()
    plan = plan_placement(layout, [int(12.5 * GIB)], 128 * GIB, 32768, opts = opts)
    assert not plan.declined_by_gate, plan.reason
    assert plan.n_parallel == 1 and plan.spilled_blocks, plan.reason
    assert plan.predicted_fit_request_ms > plan.predicted_request_ms / (
        1 - opts.min_penalty_reduction
    )


# ------------------------------------------------ findings 3 / 4: each arm at its own micro-batch


@pytest.mark.parametrize(
    "card_gib, ubatch, declined",
    [
        # The cut plan runs a 4096 micro-batch, the 4-slot fitter 512: scoring the fitter at 4096
        # under-charges its prefill and declined a spill that wins.
        (12.5, {4: 512, 1: 4096}, False),
        # The mirror: the fitter's own 4096 makes it cheaper than the 512 the cut plan runs.
        (11.5, {4: 4096, 1: 512}, True),
    ],
)
def test_the_spill_gate_scores_the_fitter_at_the_micro_batch_it_launches(
    card_gib, ubatch, declined
):
    opts = _opts(n_ubatch_by_parallel = ubatch, workload_prompt_tokens = 2048)
    plan = plan_placement(_dense(0.05), [int(card_gib * GIB)], 128 * GIB, 32768, opts = opts)
    assert plan.declined_by_gate is declined, plan.reason
    if not declined:
        assert plan.n_parallel == 1 and plan.spilled_blocks


def test_the_fitter_score_does_not_follow_the_cut_plans_micro_batch():
    """The fitter launches the caller's 4 slots, so mapping ONLY the cut count to a bigger
    micro-batch must leave its score alone."""
    plain = plan_placement(_dense(0.3), [int(12.5 * GIB)], 128 * GIB, 32768, opts = _opts())
    mapped = plan_placement(
        _dense(0.3),
        [int(12.5 * GIB)],
        128 * GIB,
        32768,
        opts = _opts(n_ubatch_by_parallel = {1: 4096, 4: 512}),
    )
    assert plain.predicted_fit_request_ms > 0
    assert mapped.predicted_fit_request_ms == plain.predicted_fit_request_ms
    assert mapped.predicted_request_ms < plain.predicted_request_ms


# ------------------------------------------------ finding 5: GLM-5.3-Flash's indexer cache


def test_glm5next_abstains_while_its_indexer_cache_is_unpriced():
    from test_offload_planner import (
        _StubReader,
        _kimi_k3_fields,
        _layout_from_reader,
        _shard_tensors,
    )

    heads = [1 if (i % 4) == 3 else 0 for i in range(46)]
    heads[45] = 1
    glm = _layout_from_reader(
        _StubReader(
            {
                "general.architecture": "glm5next",
                "glm5next.block_count": 46,
                "glm5next.nextn_predict_layers": 1,
                "glm5next.attention.head_count_kv": heads,
                "glm5next.attention.head_count": 64,
                "glm5next.embedding_length": 4096,
                "glm5next.attention.key_length": 512,
                "glm5next.attention.value_length": 512,
                "glm5next.attention.key_length_mla": 256,
                "glm5next.attention.value_length_mla": 256,
                "glm5next.attention.indexer.key_length": 128,
                "glm5next.ssm.conv_kernel": 4,
                "glm5next.kda.head_dim": 128,
                "glm5next.context_length": 1048576,
            },
            _shard_tensors(range(46)),
        )
    )
    assert not glm.complete
    # Kimi-K3 has no indexer and keeps its priced KDA state.
    kimi = _layout_from_reader(_StubReader(_kimi_k3_fields(), _shard_tensors(range(93))))
    assert kimi.complete and round(kimi.recurrent_bytes / MIB) == 443


# ------------------------------------------------ finding 6: the CPU recovery's lazy mode


@pytest.mark.parametrize(
    "emitted, extras, expected",
    [
        ("off", None, None),  # Studio's device-specific choice leaves with the devices
        ("on", None, None),
        (None, ["-lzm", "off"], "off"),  # the user's own flag stays
        (None, ["-lzm", "on"], "on"),
    ],
)
def test_the_cpu_recovery_hands_studios_lazy_mode_back_to_auto(
    tmp_path, monkeypatch, emitted, extras, expected
):
    from test_llama_cpp_placement import _launch_with_vulkan_cpu_replay, _vulkan_cpu_replay_backend

    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 2.0, free_mib = 24 * 1024, avail_mib = 64 * 1024
    )
    backend._studio_lazy_mode = lambda **_kw: emitted
    # Emitted only on a launch the planner may own (the seam itself still declines).
    backend._planner_may_run = lambda *a, **k: True
    kwargs = {"extra_args": tuple(extras)} if extras else {}
    captured = _launch_with_vulkan_cpu_replay(backend, gguf, **kwargs)
    first, replay = captured["cmds"][0], captured["cmds"][-1]
    if emitted is not None:
        assert first[first.index("-lzm") + 1] == emitted, "the control must emit Studio's mode"
    modes = [replay[i + 1] for i, tok in enumerate(replay) if tok == "-lzm"]
    assert modes == ([] if expected is None else [expected]), replay
