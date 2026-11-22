# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_FSDP2=1 shards natively inside the Trainer only. A custom loop's own Accelerator
(accelerator.prepare(model, optimizer) on an unsharded model) keeps accelerate's FSDP2, as before
native FSDP2, instead of a plain DDP wrap: fsdp2_launch.install_raw_accelerator_fsdp2_fallback
exports the accelerate FSDP2 launch env for it. A Trainer's accelerator, an explicit
UNSLOTH_FSDP2_BACKEND, an explicit plugin, a CPU run, an already initialized AcceleratorState and an
already natively sharded model are left alone. CPU only: accelerate is replaced by a recorder."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

_ENV = (
    "ACCELERATE_USE_FSDP",
    "FSDP_VERSION",
    "FSDP_AUTO_WRAP_POLICY",
    "FSDP_STATE_DICT_TYPE",
    "FSDP_RESHARD_AFTER_FORWARD",
    "ACCELERATE_USE_CPU",
    "ACCELERATE_USE_DEEPSPEED",
    "UNSLOTH_FSDP2_BACKEND",
    "UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD",
)


@pytest.fixture
def launch(monkeypatch):
    for key in _ENV:
        # setenv first so teardown also drops what the fallback exports
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    for key, value in dict(UNSLOTH_FSDP2 = "1", WORLD_SIZE = "2", RANK = "0").items():
        monkeypatch.setenv(key, value)
    native = _load_native()
    monkeypatch.setattr(native, "_SHARDED_ANY", False)
    seen = []

    class Accelerator:
        def __init__(self, *args, **kwargs):
            seen.append(os.environ.get("ACCELERATE_USE_FSDP"))

    class AcceleratorState:
        _shared_state = {}

    accelerate = types.ModuleType("accelerate")
    accelerate.Accelerator = Accelerator
    state = types.ModuleType("accelerate.state")
    state.AcceleratorState = AcceleratorState
    accelerate.state = state
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.state", state)
    module = sys.modules["unsloth.fsdp2_launch"]
    monkeypatch.setattr(module, "_accelerate_fsdp2_support", lambda: True)
    assert module.install_raw_accelerator_fsdp2_fallback() is True
    return types.SimpleNamespace(
        module = module, native = native, Accelerator = Accelerator, State = AcceleratorState, seen = seen
    )


def test_custom_loop_accelerator_gets_accelerate_fsdp2(launch):
    assert launch.module.unsloth_fsdp2_native_requested()
    launch.Accelerator()
    assert launch.seen == ["true"]
    assert os.environ["FSDP_VERSION"] == "2"
    assert os.environ["FSDP_AUTO_WRAP_POLICY"] == "TRANSFORMER_BASED_WRAP"
    assert os.environ["FSDP_STATE_DICT_TYPE"] == "FULL_STATE_DICT"
    # From here on the process is the accelerate FSDP2 backend, also for a later Trainer.
    assert not launch.module.unsloth_fsdp2_native_requested()


def test_trainer_accelerator_stays_native(launch):
    class Trainer:
        args = object()

        def create_accelerator_and_postprocess(self):
            self.accelerator = launch.Accelerator()

    Trainer().create_accelerator_and_postprocess()
    assert launch.seen == [None]
    assert launch.module.unsloth_fsdp2_native_requested()


@pytest.mark.parametrize(
    "setup",
    [
        "backend_native",
        "backend_accelerate",
        "plugin",
        "tp_plugin",
        "parallelism_config",
        "cpu",
        "state",
        "sharded",
        "single",
    ],
)
def test_left_alone(launch, monkeypatch, setup):
    kwargs = {}
    if setup == "backend_native":
        monkeypatch.setenv("UNSLOTH_FSDP2_BACKEND", "native")
    elif setup == "backend_accelerate":
        monkeypatch.setenv("UNSLOTH_FSDP2_BACKEND", "accelerate")
    elif setup == "plugin":
        kwargs["fsdp_plugin"] = object()
    elif setup == "tp_plugin":
        kwargs["torch_tp_plugin"] = object()
    elif setup == "parallelism_config":
        kwargs["parallelism_config"] = object()
    elif setup == "cpu":
        kwargs["cpu"] = True
    elif setup == "state":
        monkeypatch.setattr(launch.State, "_shared_state", {"distributed_type": "MULTI_GPU"})
    elif setup == "sharded":
        monkeypatch.setattr(launch.native, "_SHARDED_ANY", True)
    elif setup == "single":
        monkeypatch.setenv("WORLD_SIZE", "1")
    launch.Accelerator(**kwargs)
    assert launch.seen == [None]


def test_install_is_idempotent(launch):
    patched = launch.Accelerator.__init__
    assert launch.module.install_raw_accelerator_fsdp2_fallback() is True
    assert launch.Accelerator.__init__ is patched


def test_positional_options_count_as_explicit(launch, monkeypatch):
    # Accelerator(True, False, None, 1, True) selects the CPU positionally.
    seen = []

    class Accelerator:
        def __init__(
            self,
            device_placement = None,
            split_batches = None,
            mixed_precision = None,
            gradient_accumulation_steps = 1,
            cpu = False,
            deepspeed_plugin = None,
            fsdp_plugin = None,
        ):
            seen.append(os.environ.get("ACCELERATE_USE_FSDP"))

    monkeypatch.setattr(sys.modules["accelerate"], "Accelerator", Accelerator)
    assert launch.module.install_raw_accelerator_fsdp2_fallback() is True
    Accelerator(True, False, None, 1, True)
    Accelerator(None, None, None, 1, False, None, object())
    assert seen == [None, None]
    Accelerator(True)
    assert seen == [None, None, "true"]
