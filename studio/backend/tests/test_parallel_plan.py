# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-GPU plan (core/training/parallel_plan.py) on spoofed hosts and models: the auto default, explicit modes,
and the refusals, with Unsloth's own capability table from this checkout."""

from __future__ import annotations

import os

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


def test_moe_uses_ep_only_when_core_has_it():
    without = pp.plan({"parallel_mode": "auto"}, _host(2, ep = False), _model(need = 6.0, moe = True))
    assert without["mode"] == "ddp"
    with_ep = pp.plan({"parallel_mode": "auto"}, _host(2, ep = True), _model(need = 6.0, moe = True))
    assert with_ep["mode"] == "ep" and with_ep["spec"] == "ep=2"


def test_busy_gpu_counts_its_free_memory():
    host = _host(gpus = [_t4(0, free = 14.5), _t4(1, free = 4.0)])
    plan = pp.plan({"parallel_mode": "auto"}, host, _model(need = 6.0, shard = 3.4))
    assert plan["mode"] == "fsdp2"


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
    # Layer split still runs there.
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
    assert any("differ" in w for w in plan["warnings"])


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
