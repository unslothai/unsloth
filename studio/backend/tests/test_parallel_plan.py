# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""parallel_plan.py on spoofed hosts and models, with this checkout's capability table."""

from __future__ import annotations

import os

import types

import pytest

from core.training import parallel_plan as pp

_REPO_UNSLOTH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "unsloth")
)


@pytest.fixture(autouse = True)
def _core_from_this_checkout(monkeypatch):
    """Read Unsloth's parallel modules from this repository, not whatever unsloth is installed."""

    def _core_file(*parts):
        path = os.path.join(_REPO_UNSLOTH, *parts)
        return path if os.path.isfile(path) else None

    monkeypatch.setattr(pp, "_core_file", _core_file)
    monkeypatch.setattr(pp, "_CORE_MODULES", {})
    yield


def _t4(index, free = 14.5):
    return {"index": index, "name": "Tesla T4", "total_gb": 14.7, "free_gb": free}


def _host(
    n = 2,
    *,
    ep = False,
    reason = None,
    platform = "linux",
    device = "cuda",
    gpus = None,
):
    core = {"ddp": True, "fsdp2": True, "tp": True, "pp": True, "cp": True, "ep": ep}
    return {
        "platform": platform,
        "device": device,
        "distributed_backend": None if reason else ("rccl" if device == "rocm" else "nccl"),
        "multi_process_reason": reason,
        "gpus": gpus if gpus is not None else [_t4(i) for i in range(n)],
        "core": core,
        "torch": (2, 9),
        "sm": 7,
    }


def _model(
    need = 6.0,
    shard = None,
    moe = False,
    mode = "lora",
    family = "qwen3",
    hybrid = False,
):
    per = {
        1: need,
        2: shard if shard is not None else need / 2 + 1,
        4: (shard or need / 2 + 1) / 2 + 0.5,
    }
    return {
        "required_gb": need,
        "per_gpu_gb": per,
        "is_moe": moe,
        "family": family,
        "multimodal": False,
        "hybrid": hybrid,
        "mode": mode,
        "trainer": "sft",
    }


def test_core_axes_read_this_checkout():
    axes = pp.core_axes()
    assert axes["ddp"] and axes["fsdp2"] and axes["tp"] and axes["pp"] and axes["cp"]


def test_one_gpu_is_single_and_unpinned():
    plan = pp.plan({"parallel_mode": "auto"}, _host(1), _model())
    assert plan["mode"] == "single"
    assert plan["launcher"] == "single_process"
    assert plan["world_size"] == 1 and plan["spec"] is None
    assert plan["gpu_ids"] is None  # start-time auto selection still picks, as before


def test_fits_each_gpu_is_ddp_over_all():
    plan = pp.plan({"parallel_mode": "auto"}, _host(2), _model(need = 6.0))
    assert plan["mode"] == "ddp"
    assert plan["spec"] == "dp=2"
    assert plan["world_size"] == 2 and plan["gpu_ids"] == [0, 1]
    assert plan["launcher"] == "multi_process" and plan["distributed_backend"] == "nccl"


def test_does_not_fit_one_gpu_but_fits_sharded_is_fsdp2():
    plan = pp.plan({"parallel_mode": "auto"}, _host(2), _model(need = 20.0, shard = 11.0))
    assert plan["mode"] == "fsdp2"
    assert plan["spec"] == "dp=2,fsdp=1"


def test_does_not_fit_sharded_is_legacy_layer_split():
    plan = pp.plan({"parallel_mode": "auto"}, _host(2), _model(need = 40.0, shard = 22.0))
    assert plan["mode"] == "layer_split"
    assert plan["launcher"] == "single_process"
    assert plan["gpu_ids"] is None


def test_auto_never_picks_expert_parallel():
    without = pp.plan({"parallel_mode": "auto"}, _host(2, ep = False), _model(need = 6.0, moe = True))
    assert without["mode"] == "ddp"
    # Automatic never picks expert parallel; it stays an explicit choice that Core checks at load.
    with_ep = pp.plan({"parallel_mode": "auto"}, _host(2, ep = True), _model(need = 6.0, moe = True))
    assert with_ep["mode"] == "ddp" and with_ep["spec"] == "dp=2", with_ep
    big = pp.plan(
        {"parallel_mode": "auto"}, _host(2, ep = True), _model(need = 20.0, shard = 11.0, moe = True)
    )
    assert big["mode"] == "fsdp2", big
    explicit = pp.plan({"parallel_mode": "ep"}, _host(2, ep = True), _model(need = 6.0, moe = True))
    assert explicit["mode"] == "ep" and explicit["spec"] == "ep=2", explicit


def test_auto_falls_back_when_ep_cannot_shard_the_family():
    # JetMoE's experts take (layer_input,), not the routed experts call: ep cannot shard them.
    ep_host = _host(2, ep = True)
    fits = pp.plan({"parallel_mode": "auto"}, ep_host, _model(need = 6.0, moe = True, family = "jetmoe"))
    assert fits["mode"] == "ddp" and fits["spec"] == "dp=2", fits
    sharded = pp.plan(
        {"parallel_mode": "auto"},
        ep_host,
        _model(need = 20.0, shard = 11.0, moe = True, family = "jetmoe"),
    )
    assert sharded["mode"] == "fsdp2" and sharded["spec"] == "dp=2,fsdp=1", sharded
    rows = {r["mode"]: r for r in sharded["modes"]}
    assert not rows["ep"]["available"] and "experts forward takes" in rows["ep"]["reason"], rows[
        "ep"
    ]
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "ep"}, ep_host, _model(moe = True, family = "jetmoe"))
    payload = err.value.payload
    assert payload["code"] == "parallel_unsupported", payload
    assert (
        not payload["message"].startswith("Unsloth:") and " x model:" not in payload["message"]
    ), payload
    assert payload["alternative"] and "drop ep" in payload["alternative"], payload


def test_busy_gpu_counts_its_free_memory():
    host = _host(gpus = [_t4(0, free = 14.5), _t4(1, free = 4.0)])
    plan = pp.plan({"parallel_mode": "auto"}, host, _model(need = 6.0, shard = 3.4))
    assert plan["mode"] == "single" and plan["gpu_ids"] == [0], plan
    both_busy = _host(gpus = [_t4(0, free = 4.0), _t4(1, free = 4.0)])
    assert (
        pp.plan({"parallel_mode": "auto"}, both_busy, _model(need = 6.0, shard = 3.4))["mode"]
        == "fsdp2"
    )


def test_unknown_size_keeps_one_process():
    plan = pp.plan({"parallel_mode": "auto"}, _host(2), {"required_gb": None, "per_gpu_gb": {}})
    assert plan["launcher"] == "single_process"


@pytest.mark.parametrize(
    "reason",
    [
        "Windows has no NCCL, so multi-process training ...",
        "This PyTorch build has no NCCL / RCCL ...",
    ],
)
def test_no_multi_process_auto_falls_back_to_today(reason):
    host = _host(2, reason = reason, platform = "windows" if "Windows" in reason else "linux")
    fits = pp.plan({"parallel_mode": "auto"}, host, _model(need = 6.0))
    assert fits["mode"] == "single" and fits["launcher"] == "single_process"
    big = pp.plan({"parallel_mode": "auto"}, host, _model(need = 20.0, shard = 11.0))
    assert big["mode"] == "layer_split"


def test_windows_refuses_multi_process_clearly():
    host = _host(2, reason = "Windows has no NCCL, so multi-process training ...", platform = "windows")
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "ddp"}, host, _model())
    assert err.value.payload["code"] == "parallel_platform"
    assert "layer_split" in err.value.payload["alternative"]
    assert pp.plan({"parallel_mode": "layer_split"}, host, _model())["mode"] == "layer_split"


def test_linux_without_nccl_is_no_backend():
    host = _host(
        2, reason = "This PyTorch build has no NCCL / RCCL, which multi-process training needs."
    )
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "fsdp2"}, host, _model())
    assert err.value.payload["code"] == "parallel_no_backend"


def test_rocm_uses_rccl():
    plan = pp.plan({"parallel_mode": "ddp"}, _host(2, device = "rocm"), _model())
    assert plan["distributed_backend"] == "rccl" and plan["spec"] == "dp=2"


def test_ep_before_core_supports_it():
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "ep"}, _host(2), _model(moe = True))
    assert err.value.payload["code"] == "parallel_mode_unavailable"
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "custom", "parallel_spec": "ep=2"}, _host(2), _model(moe = True))
    assert err.value.payload["code"] == "parallel_mode_unavailable"


def test_core_refusal_is_parallel_unsupported_with_alternative():
    # parallel_capabilities RULES: pipeline parallel trains LoRA adapters only.
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "pp"}, _host(2), _model(mode = "full"))
    payload = err.value.payload
    assert payload["code"] == "parallel_unsupported" and payload["mode"] == "pp"
    assert "pipeline parallel" in payload["message"]
    assert payload["alternative"]


def test_explicit_modes_build_their_spec():
    for mode, spec in (
        ("ddp", "dp=2"),
        ("fsdp2", "dp=2,fsdp=1"),
        ("tp", "tp=2"),
        ("pp", "pp=2"),
        ("cp", "cp=2"),
    ):
        plan = pp.plan({"parallel_mode": mode}, _host(2), _model())
        assert (plan["mode"], plan["spec"], plan["world_size"]) == (mode, spec, 2)


def test_custom_spec_checks_the_world():
    four = _host(4)
    plan = pp.plan({"parallel_mode": "custom", "parallel_spec": "dp=2,tp=2"}, four, _model())
    assert plan["spec"] == "dp=2,tp=2" and plan["world_size"] == 4
    plan = pp.plan({"parallel_mode": "custom", "parallel_spec": "tp=2"}, four, _model())
    assert plan["world_size"] == 4  # dp fills the world
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "custom", "parallel_spec": "dp=2,tp=2"}, _host(2), _model())
    assert err.value.payload["code"] == "parallel_world_mismatch"
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "custom", "parallel_spec": "bogus"}, _host(2), _model())
    assert err.value.payload["code"] == "parallel_bad_spec"
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "ddp", "parallel_spec": "tp=2"}, _host(2), _model())
    assert err.value.payload["code"] == "parallel_bad_spec"


def test_single_process_features():
    for extra in (
        {"offload_layers": "auto"},
        {"is_embedding": True},
        {"batch_members": True},
        {"is_dataset_audio": True},
    ):
        auto = pp.plan({"parallel_mode": "auto", **extra}, _host(2), _model())
        assert auto["launcher"] == "single_process", extra
        with pytest.raises(pp.ParallelPlanError) as err:
            pp.plan({"parallel_mode": "ddp", **extra}, _host(2), _model())
        assert err.value.payload["code"] == "parallel_feature", extra


def test_gpu_ids():
    plan = pp.plan({"parallel_mode": "auto", "gpu_ids": [1, 2]}, _host(4), _model())
    assert plan["mode"] == "ddp" and plan["gpu_ids"] == [1, 2]
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "single", "gpu_ids": [0, 1]}, _host(2), _model())
    assert err.value.payload["code"] == "parallel_gpu_ids"
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "ddp", "gpu_ids": [7]}, _host(2), _model())
    assert err.value.payload["code"] == "parallel_gpu_ids"
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "ddp", "gpu_ids": [0]}, _host(2), _model())
    assert err.value.payload["code"] == "parallel_not_enough_gpus"
    single = pp.plan({"parallel_mode": "single"}, _host(gpus = [_t4(0, 3.0), _t4(1, 14.0)]), _model())
    assert single["gpu_ids"] == [1]  # the freest card


def test_mixed_gpus_warn():
    host = _host(
        gpus = [_t4(0), {"index": 1, "name": "NVIDIA L4", "total_gb": 22.5, "free_gb": 22.0}]
    )
    plan = pp.plan({"parallel_mode": "ddp"}, host, _model())
    assert any("data parallel runs at the pace" in n for n in plan["notes"]), plan
    assert not plan["warnings"], plan
    tp = pp.plan({"parallel_mode": "tp"}, host, _model())
    assert tp["notes"] and not any("data parallel" in n for n in tp["notes"]), tp["notes"]
    big = pp.plan({"parallel_mode": "ddp"}, host, _model(need = 20.0))
    assert any("may not fit" in w for w in big["warnings"]), big


def test_capabilities_rows():
    caps = pp.capabilities(_host(2))
    rows = {r["mode"]: r for r in caps["modes"]}
    assert caps["multi_process"] and caps["identical_gpus"]
    assert rows["ddp"]["available"] and rows["fsdp2"]["available"]
    assert not rows["ep"]["available"] and "expert parallel" in rows["ep"]["reason"]
    win = pp.capabilities(_host(2, reason = "Windows has no NCCL ...", platform = "windows"))
    rows = {r["mode"]: r for r in win["modes"]}
    assert not win["multi_process"]
    assert not rows["ddp"]["available"] and rows["layer_split"]["available"]
    mac = pp.capabilities(
        _host(0, reason = "Apple Silicon trains on one device.", platform = "darwin", device = "mlx")
    )
    rows = {r["mode"]: r for r in mac["modes"]}
    assert (
        rows["single"]["available"]
        and not rows["layer_split"]["available"]
        and not rows["ddp"]["available"]
    )
    one = pp.capabilities(_host(1))
    rows = {r["mode"]: r for r in one["modes"]}
    assert not rows["ddp"]["available"] and "at least 2" in rows["ddp"]["reason"]


def test_model_rows_reflect_core_refusals():
    plan = pp.plan({"parallel_mode": "auto"}, _host(2), _model(mode = "full"))
    rows = {r["mode"]: r for r in plan["modes"]}
    assert not rows["pp"]["available"]
    assert rows["ddp"]["available"]
    assert not rows["ep"]["available"]


def test_unknown_mode():
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "zero3"}, _host(2), _model())
    assert err.value.payload["code"] == "parallel_bad_spec"


def test_duplicate_gpu_ids_are_refused():
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "ddp", "gpu_ids": [0, 0]}, _host(2), _model())
    assert err.value.payload["code"] == "parallel_gpu_ids"


def test_explicit_gpus_keep_layer_split_without_multi_process():
    # Today an explicit [0, 1] splits layers across both even when one would do; auto keeps that off Linux.
    host = _host(2, reason = "Windows has no NCCL ...", platform = "windows")
    plan = pp.plan({"parallel_mode": "auto", "gpu_ids": [0, 1]}, host, _model(need = 6.0))
    assert plan["mode"] == "layer_split" and plan["gpu_ids"] == [0, 1]


def test_custom_spec_carries_backend_and_schedule_to_core():
    # Ring attention is refused below compute capability 8.0 (T4); Ulysses is not.
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan(
            {"parallel_mode": "custom", "parallel_spec": "cp=2,cp_backend=ring"}, _host(2), _model()
        )
    assert (
        err.value.payload["code"] == "parallel_unsupported"
        and "ring" in err.value.payload["message"]
    )
    assert pp.plan(
        {"parallel_mode": "custom", "parallel_spec": "cp=2,cp_backend=ulysses"}, _host(2), _model()
    )["spec"]
    assert pp.plan(
        {"parallel_mode": "custom", "parallel_spec": "pp=2,pp_schedule=gpipe"}, _host(2), _model()
    )["spec"]
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "custom", "parallel_spec": ","}, _host(2), _model())
    assert err.value.payload["code"] == "parallel_bad_spec"


def test_custom_spec_checks_installed_axes():
    host = _host(2)
    host["core"]["tp"] = False
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "custom", "parallel_spec": "tp=2"}, host, _model())
    assert err.value.payload["code"] == "parallel_mode_unavailable"


def test_compute_capability_comes_from_the_run_gpus():
    # A T4 (sm 7) beside two sm 8 cards: a run on the sm 8 cards is judged on sm 8.
    gpus = [dict(_t4(0), sm = 7), dict(_t4(1), sm = 8), dict(_t4(2), sm = 8)]
    host = _host(gpus = gpus)
    host["sm"] = None
    spec = {"parallel_mode": "custom", "parallel_spec": "cp=2,cp_backend=ring"}
    with pytest.raises(pp.ParallelPlanError):
        pp.plan({**spec, "gpu_ids": [0, 1]}, host, _model())
    assert pp.plan({**spec, "gpu_ids": [1, 2]}, host, _model())["spec"] == "cp=2,cp_backend=ring"


def test_hybrid_model_traits_reach_core():
    # Core refuses cp x tp on hybrid (linear attention) models; the planner must forward the trait.
    four = _host(4)
    pp.plan({"parallel_mode": "custom", "parallel_spec": "tp=2,cp=2"}, four, _model())
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan(
            {"parallel_mode": "custom", "parallel_spec": "tp=2,cp=2"}, four, _model(hybrid = True)
        )
    assert err.value.payload["code"] == "parallel_unsupported"
    assert "hybrid" in err.value.payload["message"]


def test_config_traits_match_core():
    config = types.SimpleNamespace(
        model_type = "qwen3_next",
        layer_types = ["linear_attention", "full_attention"],
        sliding_window = None,
    )
    traits = pp._config_traits(config)
    assert traits == {
        "family": "qwen3_next",
        "hybrid": True,
        "multimodal": False,
        "windowed": False,
    }


def _mixed_host(**extra):
    gpus = [
        {"index": 0, "name": "NVIDIA RTX 6000 Ada Generation", "total_gb": 47.99, "free_gb": 44.9},
        {"index": 1, "name": "NVIDIA GeForce RTX 3090", "total_gb": 24.0, "free_gb": 23.9},
    ]
    return dict(_host(gpus = gpus), **extra)


def test_mixed_gpus_train_alone_on_the_card_that_holds_the_run():
    # RTX 6000 Ada 48 GB + RTX 3090 24 GB, Qwen3-4B full fine-tune (~31.7 GB): FSDP2 ran out of memory on
    # the 3090 at step 1; the 48 GB card trains it alone.
    plan = pp.plan(
        {"parallel_mode": "auto"}, _mixed_host(), _model(need = 31.7, shard = 16.8, mode = "full")
    )
    assert plan["mode"] == "single" and plan["gpu_ids"] == [0], plan
    assert (
        "GPU 0" in plan["reason"] and "not every GPU has room for a full copy" in plan["reason"]
    ), plan
    gpus = _mixed_host()["gpus"] + [
        {"index": 2, "name": "NVIDIA RTX 6000 Ada Generation", "total_gb": 47.99, "free_gb": 44.0}
    ]
    three = pp.plan({"parallel_mode": "auto"}, _host(gpus = gpus), _model(need = 31.7, shard = 11.0))
    assert three["mode"] == "single" and "only" not in three["reason"], three


def test_one_card_is_not_pinned_on_a_mask_that_cannot_be_narrowed():
    # GPUs visible by UUID: /start refuses a plan that pins a subset, so Automatic does not pick one card.
    plan = pp.plan(
        {"parallel_mode": "auto"}, _mixed_host(pins_gpu_ids = False), _model(need = 31.7, shard = 16.8)
    )
    assert not (plan["mode"] == "single" and plan["gpu_ids"]), plan
    assert plan["mode"] == "fsdp2" and "cannot keep a run to one card" in plan["reason"], plan
    assert "any one GPU" not in plan["reason"], plan


def test_fsdp2_reshards_when_the_run_does_not_fit_one_gpu():
    host = _host(2)
    big = pp.plan({"parallel_mode": "auto"}, host, _model(need = 20.0, shard = 11.0, mode = "full"))
    assert big["mode"] == "fsdp2" and big["env"] == {"UNSLOTH_FSDP2_LAYOUT": "sharded"}, big
    small = pp.plan({"parallel_mode": "fsdp2"}, host, _model(need = 6.0))
    assert small["mode"] == "fsdp2" and small["env"] == {}, small
    custom = pp.plan(
        {"parallel_mode": "custom", "parallel_spec": "dp=2,fsdp=1"},
        host,
        _model(need = 20.0, shard = 11.0),
    )
    assert custom["env"] == {"UNSLOTH_FSDP2_LAYOUT": "sharded"}, custom
    assert pp.plan({"parallel_mode": "ddp"}, host, _model(need = 20.0))["env"] == {}


def test_core_rule_prefix_is_dropped():
    assert (
        pp._plain_core_message(
            "Unsloth: pp x packing: pipeline parallel with packed rows is not supported."
        )
        == "Pipeline parallel with packed rows is not supported."
    )
    assert pp._plain_core_message("Unsloth: tp x model x lora: no.") == "No."
    assert pp._plain_core_message("Plain reason.") == "Plain reason."


def test_custom_hint_is_a_layout_this_run_accepts():
    with pytest.raises(pp.ParallelPlanError) as err:
        pp.plan({"parallel_mode": "custom", "parallel_spec": ""}, _host(2), _model())
    assert '"dp=2"' in err.value.payload["message"], err.value.payload
    assert pp._spec_example(4) == "dp=4" and pp._spec_example(3) == "dp=3"
    assert (
        pp.plan({"parallel_mode": "custom", "parallel_spec": "dp=2"}, _host(2), _model())["error"]
        is None
    )


def test_hunyuan_v1_moe_takes_ep_when_picked_and_auto_keeps_ddp():
    # Its default config has num_experts = 1; real Hunyuan-A13B (64 experts) shards under ep.
    host = _host(2, ep = True)
    model = _model(need = 6.0, moe = True, family = "hunyuan_v1_moe")
    explicit = pp.plan({"parallel_mode": "ep"}, host, model)
    assert explicit["mode"] == "ep" and explicit["spec"] == "ep=2", explicit
    rows = {r["mode"]: r for r in explicit["modes"]}
    assert rows["ep"]["available"], rows["ep"]
    # Automatic never picks expert parallel.
    auto = pp.plan({"parallel_mode": "auto"}, host, model)
    assert auto["mode"] == "ddp" and auto["spec"] == "dp=2", auto


def test_gpt_oss_qlora_plans_ep_like_16bit():
    ep_host = _host(2, ep = True)
    lora = pp.plan(
        {"parallel_mode": "ep", "load_in_4bit": False},
        ep_host,
        _model(need = 6.0, moe = True, family = "gpt_oss"),
    )
    assert lora["mode"] == "ep" and lora["spec"] == "ep=2", lora
    qlora = pp.plan(
        {"parallel_mode": "ep"},
        ep_host,
        _model(need = 6.0, moe = True, family = "gpt_oss", mode = "qlora"),
    )
    assert qlora["mode"] == "ep" and qlora["spec"] == "ep=2", qlora
    # A pre-quantized repo name is 4-bit even when the request leaves load_in_4bit off.
    named = pp.plan(
        {"parallel_mode": "ep", "model_name": "unsloth/gpt-oss-20b-unsloth-bnb-4bit"},
        ep_host,
        _model(need = 6.0, moe = True, family = "gpt_oss"),
    )
    assert named["mode"] == "ep" and named["spec"] == "ep=2", named
    fits = pp.plan(
        {"parallel_mode": "auto"},
        ep_host,
        _model(need = 6.0, moe = True, family = "gpt_oss", mode = "qlora"),
    )
    assert fits["mode"] == "ddp" and fits["spec"] == "dp=2", fits
    sharded = pp.plan(
        {"parallel_mode": "auto"},
        ep_host,
        _model(need = 20.0, shard = 11.0, moe = True, family = "gpt_oss", mode = "qlora"),
    )
    assert sharded["mode"] == "fsdp2" and sharded["spec"] == "dp=2,fsdp=1", sharded
    rows = {r["mode"]: r for r in sharded["modes"]}
    assert rows["ep"]["available"], rows["ep"]
