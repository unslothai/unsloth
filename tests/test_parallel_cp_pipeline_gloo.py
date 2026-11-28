# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pipeline parallel MoE router aux loss and pipeline eval on a gloo CPU world, vs the same script in one process.

Split out of test_parallel_cp_compose_gloo.py, whose worker, checkpoints and spawn harness these reuse, so that each
file fits its CI shard (distributed-cpu-ci.yml gloo-cp and gloo-cp-pipeline)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_parallel_cp_compose_gloo import (  # noqa: E402
    _close,
    _same_rows,
    _spawn,
)


# torch < 2.13 single-stage schedules refuse fewer micro-batches than stages; 2.13 keeps it in 1F1B.
_ONE_MICROBATCH_REFUSED = "pipeline schedule cannot run 1 micro-batches over 2 stages"


def _skip_if_one_microbatch_refused(results):
    from unsloth.models import pp_env

    errors = [res.get("error") or "" for res in results]
    # Where GPipe pads a single micro-batch up to the stage count, a batch of one splits too and is refused.
    padded = pp_env._gpipe_needs_stage_count() and all(
        "splits each Trainer batch into 2" in e for e in errors
    )
    if errors and (padded or all(_ONE_MICROBATCH_REFUSED in e for e in errors)):
        import torch as _torch
        pytest.skip(
            f"torch {_torch.__version__}'s GPipe schedule refuses 1 micro-batch over 2 stages (torch < 2.13), and "
            "the pipeline takes the router aux loss over one micro-batch only: " + errors[0][:200]
        )


_PP_AUX_SPLIT = "only the GPipe schedule does that"


@pytest.mark.parametrize("zoo_compile", [False, True], ids = ["forward_install", "zoo_compiled"])
@pytest.mark.parametrize("aux", [True, False], ids = ["aux_on", "aux_off"])
@pytest.mark.parametrize("ga", [1, 2, 4])
def test_pp_moe_router_aux_loss_with_accumulation_matches_one_process(
    ga, aux, zoo_compile, tmp_path
):
    """PP=2 Qwen3-MoE GA 1/2/4 matches one process: the aux term is token-share weighted (full = ~ga x larger)."""
    kw = dict(bs = 2, ga = ga, steps = 3, arch = "qwen3_moe", router_aux = aux, aux_coef = 0.5 if aux else None)
    base = {}
    if zoo_compile:
        kw["zoo_compile"] = True
        base = {
            "UNSLOTH_FUSED_FORWARD": "0",
            "UNSLOTH_COMPILE_DISABLE": "1",
            "UNSLOTH_COMPILE_LOCATION": str(tmp_path / "unsloth_compiled_cache"),
        }
    (ref,) = _spawn("train", 1, base, **kw)
    assert ref["output_router_logits"] is aux, ref
    env = {
        "UNSLOTH_PP_SIZE": "2",
        "UNSLOTH_PP_MICROBATCHES": "1",
        "UNSLOTH_PP_SCHEDULE": "gpipe",
        **base,
    }
    results = _spawn("train", 2, env, **kw)
    _skip_if_one_microbatch_refused(results)
    _same_rows(results, ref, what = (ga, aux))
    assert sorted(res.get("stage") for res in results) == [0, 1], results
    for rank, res in enumerate(results):
        # transformers 4.x drifts ~1.4e-4 / ~5e-6 at GA 4 even aux off; the guarded error is 13-42% of the loss
        _close(res, ref, rel = 5e-4, state_atol = 1e-5, what = (ga, aux, rank))


@pytest.mark.parametrize(
    "arch,aux",
    [("llama", False), ("qwen3_moe", False), ("qwen3_moe", True)],
    ids = ["dense", "moe_aux_off", "moe_aux_on"],
)
def test_cpu_pipeline_eval_matches_one_process(monkeypatch, arch, aux):
    monkeypatch.setenv("UNSLOTH_FORCE_CPU", "1")
    kw = dict(
        bs = 2,
        ga = 1,
        steps = 3,
        arch = arch,
        eval_only = True,
        router_aux = aux,
        aux_coef = 0.5 if aux else None,
    )
    (ref,) = _spawn("train", 1, **kw)
    assert "error" not in ref, ref
    results = _spawn(
        "train",
        2,
        {"UNSLOTH_PP_SIZE": "2", "UNSLOTH_PP_MICROBATCHES": "1", "UNSLOTH_PP_SCHEDULE": "gpipe"},
        **kw,
    )
    _skip_if_one_microbatch_refused(results)
    assert sorted(res.get("stage", -1) for res in results) == [0, 1], results
    for res in [ref, *results]:
        assert "error" not in res, res
        assert res["cpu_compute_loss"]
        assert res["cpu_chunked_ce"] is (arch == "llama")
        assert res["rows_read"] == ref["rows_read"]
        assert res["eval_loss"] == pytest.approx(ref["eval_loss"], rel = 1e-5)


@pytest.mark.parametrize(
    "how",
    [
        dict(router_aux = True, aux_coef = 0.5),
        dict(trl_aux = True, aux_coef = 0.5),
    ],
    ids = ["config_flag", "trl_coef"],
)
@pytest.mark.parametrize(
    "env, bs",
    [
        ({"UNSLOTH_PP_MICROBATCHES": "2", "UNSLOTH_PP_SCHEDULE": "gpipe"}, 2),
        ({}, 4),  # the default schedule: 4 micro-batches (2 x pp), run as GPipe
    ],
    ids = ["gpipe_mb2", "default_mb4"],
)
@pytest.mark.parametrize("ga", [1, 2])
def test_pp_moe_router_aux_loss_split_over_micro_batches_matches_one_process(how, env, bs, ga):
    """A Trainer batch split into micro-batches forms the router aux loss over the whole batch, as one process
    does: summed statistics, the whole-batch gradient handed to every micro-batch after all forwards (GPipe).
    Failing before: refused at the Trainer init (the pipeline summed one aux loss per micro-batch)."""
    if how.get("trl_aux"):
        from trl import SFTConfig
        if "router_aux_loss_coef" not in getattr(SFTConfig, "__dataclass_fields__", {}):
            pytest.skip(
                "TRL < 1.6 has no SFTConfig.router_aux_loss_coef: the trainer never turns the aux loss on"
            )
    kw = dict(bs = bs, ga = ga, steps = 3, arch = "qwen3_moe", **how)
    (ref,) = _spawn("train", 1, **kw)
    assert ref["output_router_logits"] or how.get("trl_aux"), ref
    results = _spawn("train", 2, {"UNSLOTH_PP_SIZE": "2", **env}, **kw)
    _same_rows(results, ref, what = (env, ga, how))
    assert sorted(res.get("stage") for res in results) == [0, 1], results
    # Engaged on the last stage for every Trainer batch (ga per optimizer step), never on the first.
    assert sorted(res["carried_aux_steps"] for res in results) == [0, 3 * ga], results
    for rank, res in enumerate(results):
        _close(res, ref, rel = 5e-4, state_atol = 1e-5, what = (env, ga, how, rank))


@pytest.mark.parametrize(
    "how",
    [
        dict(router_aux = True, aux_coef = 0.5),
        dict(trl_aux = True, aux_coef = 0.5),
    ],
    ids = ["config_flag", "trl_coef"],
)
@pytest.mark.parametrize("env", [{"UNSLOTH_PP_SCHEDULE": "1f1b"}], ids = ["1f1b"])
def test_pp_moe_router_aux_loss_split_over_micro_batches_is_refused_at_trainer_init(how, env):
    """An explicit 1F1B schedule runs a backward before the last forward: the whole-batch aux loss is refused."""
    env = {"UNSLOTH_PP_SIZE": "2", **env}
    for rank, res in enumerate(
        _spawn("train", 2, env, bs = 2, ga = 1, steps = 1, arch = "qwen3_moe", probe_refusal = True, **how)
    ):
        assert _PP_AUX_SPLIT in (res.get("error") or ""), (rank, res)
        assert res["phase"] == "init", (rank, res)
        assert not res["split"] and res["untouched"], (rank, res)


@pytest.mark.parametrize(
    "how, env",
    [
        (dict(), {}),
        (dict(router_aux = True, args_coef = 0.0), {}),
        (
            dict(router_aux = True, aux_coef = 0.5, ga = 2),
            {"UNSLOTH_PP_MICROBATCHES": "1", "UNSLOTH_PP_SCHEDULE": "gpipe"},
        ),
        (dict(router_aux = True, aux_coef = 0.5, bs = 1), {"UNSLOTH_PP_SCHEDULE": "gpipe"}),
    ],
    ids = ["aux_off", "config_flag_coef0", "aux_on_one_microbatch", "aux_on_batch_of_one"],
)
def test_pp_moe_router_aux_loss_allowed_layouts_still_train(how, env):
    kw = dict(bs = 2, ga = 1, steps = 2, arch = "qwen3_moe", probe_refusal = True)
    kw.update(how)
    results = _spawn("train", 2, {"UNSLOTH_PP_SIZE": "2", **env}, **kw)
    _skip_if_one_microbatch_refused(results)
    assert sorted(res.get("stage") for res in results) == [0, 1], results
    for rank, res in enumerate(results):
        assert res["phase"] is None and "error" not in res, (rank, res)
        assert len(res["losses"]) == 2 and all(l == l for l in res["losses"]), (rank, res)
