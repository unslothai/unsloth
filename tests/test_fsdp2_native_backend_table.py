# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 against every accelerate distributed backend (fsdp2_launch.NATIVE_BACKEND_TABLE):
the table names every accelerate DistributedType (accelerate/utils/dataclasses.py DistributedType),
and each "refuse" row, switched on the way accelerate itself would see it (launch env, plugin
TrainingArguments, installed torch_xla, an available vendor device, SageMaker env, ParallelismConfig
env), is refused at the launch step (apply_unsloth_fsdp2_launch_env, env / package / device rows) and
at Trainer setup (every row), before the model is touched."""

import importlib.machinery
import re
import sys
import types

import pytest
import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

_CLEAN = (
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_USE_MEGATRON_LM",
    "ACCELERATE_USE_FSDP",
    "ACCELERATE_USE_SAGEMAKER",
    "ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE",
    "ACCELERATE_USE_PARALLELISM_CONFIG",
    "PARALLELISM_CONFIG_TP_SIZE",
    "ACCELERATE_USE_CPU",
    "ACCELERATE_MIXED_PRECISION",
    "USE_TORCH_XLA",
    "UNSLOTH_FSDP2_BACKEND",
)


@pytest.fixture
def native(monkeypatch):
    for key in _CLEAN:
        monkeypatch.delenv(key, raising = False)
    for key, value in dict(UNSLOTH_FSDP2 = "1", WORLD_SIZE = "2", RANK = "0").items():
        monkeypatch.setenv(key, value)
    return _load_native()


def _fake_module(monkeypatch, name):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _switch_on(row, monkeypatch):
    """Turn the row's backend on as accelerate would detect it; returns TrainingArguments."""
    args = types.SimpleNamespace(save_on_each_node = False)
    if row.get("env"):
        monkeypatch.setenv(row["env"][0], "true")
    elif row.get("args"):
        setattr(args, row["args"][0], object())
    elif row.get("package"):
        _fake_module(monkeypatch, row["package"])
    elif row.get("device"):
        package, attr, before_cuda = row["device"]
        if package is not None:
            parts = package.split(".")
            for i in range(len(parts)):
                _fake_module(monkeypatch, ".".join(parts[: i + 1]))
        monkeypatch.setattr(
            torch, attr, types.SimpleNamespace(is_available = lambda: True), raising = False
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    elif row.get("env_pair"):
        monkeypatch.setenv(row["env_pair"][0], "true")
        monkeypatch.setenv(row["env_pair"][1], "DATA_PARALLEL")
    elif row.get("sizes"):
        monkeypatch.setenv("ACCELERATE_USE_PARALLELISM_CONFIG", "true")
        monkeypatch.setenv(row["sizes"] + "TP_SIZE", "2")
    else:
        raise AssertionError(f"row {row['type']} has no detector")
    return args


def test_table_covers_every_accelerate_distributed_type(native):
    dataclasses = pytest.importorskip("accelerate.utils.dataclasses")
    launch = sys.modules["unsloth.fsdp2_launch"]
    names = {row["type"] for row in launch.NATIVE_BACKEND_TABLE}
    missing = {t.name for t in dataclasses.DistributedType} - names
    assert not missing, missing
    for row in launch.NATIVE_BACKEND_TABLE:
        assert row["action"] in ("handle", "refuse"), row
        assert row["action"] == "handle" or row.get("label"), row


def test_every_refuse_row_is_refused_at_both_entry_points(native, monkeypatch):
    launch = sys.modules["unsloth.fsdp2_launch"]
    # Explicit native: on a torch build without CUDA / ROCm (CPU CI) the generic flag routes a vendor
    # device to accelerate's FSDP2 instead (non_cuda_install), which is not refused.
    monkeypatch.setenv("UNSLOTH_FSDP2_BACKEND", "native")
    rows = [r for r in launch.NATIVE_BACKEND_TABLE if r["action"] == "refuse"]
    assert len(rows) >= 10
    for row in rows:
        with monkeypatch.context() as m:
            args = _switch_on(row, m)
            model = torch.nn.Linear(2, 2)
            weight = model.weight
            with pytest.raises(RuntimeError, match = "native FSDP2") as exc:
                native.native_fsdp2_trainer_setup(model = model, phase = "pre", training_args = args)
            assert row["label"] in str(exc.value), (row["type"], str(exc.value))
            assert model.weight is weight and not native.is_native_fsdp2(model), row["type"]
            if not row.get("args"):  # the launch step sees only the env / installed packages
                with pytest.raises(RuntimeError, match = re.escape(row["label"])):
                    launch.apply_unsloth_fsdp2_launch_env()
    # Nothing on: neither entry point refuses.
    assert launch.native_backend_conflicts() == []
    assert launch.apply_unsloth_fsdp2_launch_env() is False


def test_megatron_lm_launch_is_refused(native, monkeypatch):
    """accelerate launch --use_megatron_lm exports ACCELERATE_USE_MEGATRON_LM
    (accelerate/utils/launch.py), which AcceleratorState turns into DistributedType.MEGATRON_LM."""
    launch = sys.modules["unsloth.fsdp2_launch"]
    monkeypatch.setenv("ACCELERATE_USE_MEGATRON_LM", "true")
    with pytest.raises(RuntimeError, match = "Megatron-LM"):
        launch.apply_unsloth_fsdp2_launch_env()
    with pytest.raises(RuntimeError, match = "Megatron-LM"):
        native.native_fsdp2_trainer_setup(model = torch.nn.Linear(2, 2), phase = "pre")


def test_xla_off_switch_and_cpu_runs_are_not_refused(native, monkeypatch):
    launch = sys.modules["unsloth.fsdp2_launch"]
    _fake_module(monkeypatch, "torch_xla")
    monkeypatch.setenv("USE_TORCH_XLA", "0")  # accelerate's own off switch
    assert launch.native_backend_conflicts() == []
    monkeypatch.setattr(torch, "xpu", types.SimpleNamespace(is_available = lambda: True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert launch.native_backend_conflicts(uses_cpu = True) == []
    assert [t for t, _, _ in launch.native_backend_conflicts()] == ["MULTI_XPU"]
    # A ParallelismConfig env that is a plain shard over every rank is what native FSDP2 does.
    monkeypatch.setenv("ACCELERATE_USE_PARALLELISM_CONFIG", "true")
    monkeypatch.setenv("PARALLELISM_CONFIG_DP_SHARD_SIZE", "2")
    assert "PARALLELISM_CONFIG" not in [t for t, _, _ in launch.native_backend_conflicts()]
