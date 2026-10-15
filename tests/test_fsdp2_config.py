# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""What Unsloth fills into accelerate's FSDP2 config (unsloth/models/fsdp2.py). No GPU."""

from __future__ import annotations

import contextlib
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")
fsdp = pytest.importorskip("torch.distributed.fsdp")
MixedPrecisionPolicy = getattr(fsdp, "MixedPrecisionPolicy", None)
if MixedPrecisionPolicy is None:
    pytest.skip("torch without FSDP2", allow_module_level = True)

from unsloth.models import fsdp2  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for key in ("UNSLOTH_FSDP2_ACCELERATE_DEFAULTS", "UNSLOTH_FSDP2_FUSED_LORA"):
        monkeypatch.delenv(key, raising = False)


def _accelerator(policy, mixed_precision = "fp16", version = 2, auto_wrap_policy = object()):
    plugin = types.SimpleNamespace(
        fsdp_version = version,
        mixed_precision_policy = policy,
        auto_wrap_policy = auto_wrap_policy,
        reshard_after_forward = True,
    )
    return types.SimpleNamespace(
        state = types.SimpleNamespace(fsdp_plugin = plugin), mixed_precision = mixed_precision
    )


@pytest.mark.parametrize("dtype,name", [(torch.float16, "fp16"), (torch.bfloat16, "bf16")])
def test_accelerates_auto_policy_reduces_in_fp32(dtype, name):
    acc = _accelerator(MixedPrecisionPolicy(param_dtype = dtype, reduce_dtype = dtype, output_dtype = dtype), name)
    policy = fsdp2.fill_fsdp2_mixed_precision(acc)
    assert policy.param_dtype == dtype
    assert policy.reduce_dtype == torch.float32
    assert policy.output_dtype is None
    assert acc.state.fsdp_plugin.mixed_precision_policy is policy


def test_a_policy_the_user_wrote_is_left_alone():
    user = MixedPrecisionPolicy(param_dtype = torch.float16, reduce_dtype = torch.float16)
    acc = _accelerator(user, "bf16")
    assert fsdp2.fill_fsdp2_mixed_precision(acc) is user
    user = MixedPrecisionPolicy(param_dtype = None, reduce_dtype = torch.float32)
    acc = _accelerator(user, "fp16")
    assert fsdp2.fill_fsdp2_mixed_precision(acc) is user


def test_no_mixed_precision_and_the_opt_out(monkeypatch):
    auto = MixedPrecisionPolicy(param_dtype = torch.float16, reduce_dtype = torch.float16, output_dtype = torch.float16)
    assert fsdp2.fill_fsdp2_mixed_precision(_accelerator(None, "no")) is None
    monkeypatch.setenv("UNSLOTH_FSDP2_ACCELERATE_DEFAULTS", "1")
    assert fsdp2.fill_fsdp2_mixed_precision(_accelerator(auto, "fp16")) is auto


def test_trainer_setup_keeps_fused_lora_only_for_a_per_block_fsdp2_wrap(monkeypatch):
    installed = []
    monkeypatch.setattr(fsdp2, "install_fsdp2_prepare_hook", lambda: installed.append("prepare"))
    monkeypatch.setattr(fsdp2, "install_fsdp2_no_sync", lambda: installed.append("no_sync"))

    def trainer(**kw):
        return types.SimpleNamespace(accelerator = _accelerator(None, **kw))

    assert fsdp2.fsdp2_trainer_setup(trainer(version = 1)) == (False, None)
    assert installed == []
    assert fsdp2.fsdp2_trainer_setup(trainer()) == (True, None)
    assert installed == ["prepare", "no_sync"]
    keep, reason = fsdp2.fsdp2_trainer_setup(trainer(auto_wrap_policy = None))
    assert keep is False and "root" in reason
    monkeypatch.setenv("UNSLOTH_FSDP2_FUSED_LORA", "0")
    keep, reason = fsdp2.fsdp2_trainer_setup(trainer())
    assert keep is False and "UNSLOTH_FSDP2_FUSED_LORA" in reason


def test_old_accelerate_no_sync_gets_fsdp2_gradient_sync(monkeypatch):
    """accelerate < 1.11 only looked for `model.no_sync`, which FSDP2 modules lack."""
    from accelerate import Accelerator

    @contextlib.contextmanager
    def old_no_sync(self, model):
        context = getattr(model, "no_sync", contextlib.nullcontext)
        with context():
            yield

    monkeypatch.setattr(Accelerator, "no_sync", old_no_sync)
    assert fsdp2.install_fsdp2_no_sync() is True
    assert Accelerator.no_sync is not old_no_sync
    assert fsdp2.install_fsdp2_no_sync() is True  # idempotent

    calls = []
    model = types.SimpleNamespace(set_requires_gradient_sync = calls.append)
    fake = types.SimpleNamespace(is_fsdp2 = True)
    with Accelerator.no_sync(fake, model):
        assert calls == [False]
    assert calls == [False, True]
    fake.is_fsdp2 = False
    with Accelerator.no_sync(fake, model):
        pass
    assert calls == [False, True]


def test_current_accelerate_no_sync_is_not_touched(monkeypatch):
    from accelerate import Accelerator
    import inspect

    original = Accelerator.no_sync
    if "set_requires_gradient_sync" not in inspect.getsource(original):
        pytest.skip("accelerate < 1.11 in this environment")
    assert fsdp2.install_fsdp2_no_sync() is True
    assert Accelerator.no_sync is original


def test_the_trainer_init_wrapper_routes_fsdp_through_the_setup():
    import inspect

    from unsloth.models import _utils

    source = inspect.getsource(_utils.patch_gradient_accumulation_fix)
    assert "fsdp2_trainer_setup" in source and "_decline_fused_lora_for_fsdp" in source
