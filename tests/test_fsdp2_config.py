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


def _accelerator(
    policy,
    mixed_precision = "fp16",
    version = 2,
    auto_wrap_policy = object(),
    auto = True,
    seen = True,
):
    plugin = types.SimpleNamespace(
        fsdp_version = version,
        mixed_precision_policy = policy,
        auto_wrap_policy = auto_wrap_policy,
        reshard_after_forward = True,
    )
    if auto and seen:
        setattr(plugin, fsdp2._AUTO_POLICY, policy)
    if seen:
        setattr(plugin, fsdp2._POLICY_SEEN, True)
    return types.SimpleNamespace(
        state = types.SimpleNamespace(fsdp_plugin = plugin), mixed_precision = mixed_precision
    )


@pytest.mark.parametrize("dtype,name", [(torch.float16, "fp16"), (torch.bfloat16, "bf16")])
def test_accelerates_auto_policy_reduces_in_fp32(dtype, name):
    acc = _accelerator(
        MixedPrecisionPolicy(param_dtype = dtype, reduce_dtype = dtype, output_dtype = dtype), name
    )
    policy = fsdp2.fill_fsdp2_mixed_precision(acc)
    # float16 gathers the stored dtype and leaves the casting to autocast, like the unsharded run.
    assert policy.param_dtype == (None if dtype == torch.float16 else dtype)
    assert policy.reduce_dtype == torch.float32
    assert policy.output_dtype is None
    assert acc.state.fsdp_plugin.mixed_precision_policy is policy


def test_a_policy_the_user_wrote_is_left_alone():
    user = MixedPrecisionPolicy(param_dtype = torch.float16, reduce_dtype = torch.float16)
    acc = _accelerator(user, "bf16", auto = False)
    assert fsdp2.fill_fsdp2_mixed_precision(acc) is user
    user = MixedPrecisionPolicy(param_dtype = None, reduce_dtype = torch.float32)
    acc = _accelerator(user, "fp16", auto = False)
    assert fsdp2.fill_fsdp2_mixed_precision(acc) is user
    # Same dtypes as accelerate's automatic policy, but written by the user.
    user = MixedPrecisionPolicy(
        param_dtype = torch.float16, reduce_dtype = torch.float16, output_dtype = torch.float16
    )
    acc = _accelerator(user, "fp16", auto = False)
    assert fsdp2.fill_fsdp2_mixed_precision(acc) is user


def test_an_accelerator_built_before_unsloth_import_is_not_guessed(monkeypatch):
    """Without provenance, an fp16 policy shaped like accelerate's automatic one is refused, not rewritten."""
    shaped = dict(param_dtype = torch.float16, reduce_dtype = torch.float16, output_dtype = torch.float16)
    with pytest.raises(RuntimeError, match = "before `import unsloth`"):
        fsdp2.fill_fsdp2_mixed_precision(
            _accelerator(MixedPrecisionPolicy(**shaped), "fp16", seen = False)
        )
    bf16 = MixedPrecisionPolicy(
        param_dtype = torch.bfloat16, reduce_dtype = torch.bfloat16, output_dtype = torch.bfloat16
    )
    assert fsdp2.fill_fsdp2_mixed_precision(_accelerator(bf16, "bf16", seen = False)) is bf16
    other = MixedPrecisionPolicy(param_dtype = None, reduce_dtype = torch.float32)
    assert fsdp2.fill_fsdp2_mixed_precision(_accelerator(other, "fp16", seen = False)) is other
    monkeypatch.setenv("UNSLOTH_FSDP2_ACCELERATE_DEFAULTS", "1")
    kept = MixedPrecisionPolicy(**shaped)
    assert fsdp2.fill_fsdp2_mixed_precision(_accelerator(kept, "fp16", seen = False)) is kept


def test_provenance_tags_only_the_policy_accelerate_builds():
    from accelerate.utils import FullyShardedDataParallelPlugin

    assert fsdp2.install_fsdp2_policy_provenance()
    plugin = types.SimpleNamespace(fsdp_version = 2, mixed_precision_policy = None)
    FullyShardedDataParallelPlugin.set_mixed_precision(plugin, "fp16")
    assert getattr(plugin, fsdp2._AUTO_POLICY) is plugin.mixed_precision_policy
    user = MixedPrecisionPolicy(
        param_dtype = torch.float16, reduce_dtype = torch.float16, output_dtype = torch.float16
    )
    plugin = types.SimpleNamespace(fsdp_version = 2, mixed_precision_policy = user)
    FullyShardedDataParallelPlugin.set_mixed_precision(plugin, "fp16")
    assert plugin.mixed_precision_policy is user
    assert not hasattr(plugin, fsdp2._AUTO_POLICY)
    assert getattr(plugin, fsdp2._POLICY_SEEN) is True


def test_no_mixed_precision_and_the_opt_out(monkeypatch):
    auto = MixedPrecisionPolicy(
        param_dtype = torch.float16, reduce_dtype = torch.float16, output_dtype = torch.float16
    )
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


def test_bf16_full_finetuning_gets_one_trainable_dtype_per_unit():
    """Unsloth's bfloat16 full finetuning: no autocast, bf16 linears, fp32 norms (FSDP2 refuses mixed units)."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 4).to(torch.bfloat16), torch.nn.LayerNorm(4))
    frozen = torch.nn.Linear(4, 4).to(torch.bfloat16).requires_grad_(False)
    model.append(frozen)
    acc = _accelerator(None, "no")
    assert fsdp2.unify_trainable_dtypes(acc, model) is True
    assert {p.dtype for p in model.parameters() if p.requires_grad} == {torch.float32}
    assert frozen.weight.dtype == torch.bfloat16
    policy = acc.state.fsdp_plugin.mixed_precision_policy
    assert policy.param_dtype == torch.bfloat16 and policy.reduce_dtype == torch.float32


@pytest.mark.parametrize("explicit", [False, True])
def test_fp16_full_finetuning_gets_one_trainable_dtype_per_unit(explicit):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4).to(torch.float16), torch.nn.LayerNorm(4))
    user = (
        MixedPrecisionPolicy(param_dtype = torch.float16, reduce_dtype = torch.float32)
        if explicit
        else None
    )
    acc = _accelerator(user, "no", auto = False)
    assert fsdp2.unify_trainable_dtypes(acc, model) is True
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    policy = acc.state.fsdp_plugin.mixed_precision_policy
    if explicit:
        assert policy is user
    assert policy.param_dtype == torch.float16 and policy.reduce_dtype == torch.float32


def test_uniform_or_autocast_runs_are_not_upcast():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4).to(torch.bfloat16))
    assert fsdp2.unify_trainable_dtypes(_accelerator(None, "no"), model) is False
    assert model[0].weight.dtype == torch.bfloat16
    uniform = torch.nn.Sequential(torch.nn.Linear(4, 4).to(torch.bfloat16))
    acc = _accelerator(None, "bf16")
    policy = acc.state.fsdp_plugin.mixed_precision_policy
    # Under autocast a uniform unit is left to accelerate's own upcast.
    assert fsdp2.unify_trainable_dtypes(acc, uniform) is False
    assert uniform[0].weight.dtype == torch.bfloat16
    assert acc.state.fsdp_plugin.mixed_precision_policy is policy


def test_mixed_precision_masters_are_unified_when_accelerate_does_not():
    """accelerate 1.13's upcast is `param = param.to(torch.float32)`, which leaves the model alone."""
    mixed = torch.nn.Sequential(
        torch.nn.Linear(4, 4).to(torch.bfloat16),
        torch.nn.LayerNorm(4),
        torch.nn.Linear(4, 4).to(torch.bfloat16).requires_grad_(False),
    )
    acc = _accelerator(None, "bf16")
    policy = acc.state.fsdp_plugin.mixed_precision_policy
    assert fsdp2.unify_trainable_dtypes(acc, mixed) is True
    assert {p.dtype for p in mixed.parameters() if p.requires_grad} == {torch.float32}
    assert mixed[2].weight.dtype == torch.bfloat16
    assert acc.state.fsdp_plugin.mixed_precision_policy is policy


def test_install_fsdp2_hooks_patches_accelerate_without_a_trainer():
    import accelerate.accelerator as accelerator_module
    from accelerate import Accelerator
    from accelerate.utils import FullyShardedDataParallelPlugin

    fsdp2.install_fsdp2_hooks()
    assert getattr(accelerator_module.fsdp2_prepare_model, fsdp2._PATCHED, False)
    assert getattr(FullyShardedDataParallelPlugin.set_mixed_precision, fsdp2._PATCHED, False)
    assert hasattr(Accelerator, "no_sync")
    import inspect

    import accelerate.utils.fsdp_utils as fsdp_utils

    get = fsdp_utils._get_model_state_dict
    assert getattr(get, fsdp2._PATCHED, False) or "full_state_dict" in inspect.getsource(get)


def test_an_explicit_policy_still_gets_float32_master_weights():
    """Accelerator mixed_precision="no" with a launcher policy: the masters still need one dtype."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 4).to(torch.bfloat16), torch.nn.LayerNorm(4))
    user = MixedPrecisionPolicy(param_dtype = torch.bfloat16, reduce_dtype = torch.bfloat16)
    acc = _accelerator(user, "no", auto = False)
    assert fsdp2.unify_trainable_dtypes(acc, model) is True
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert acc.state.fsdp_plugin.mixed_precision_policy is user


def _lora_like():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4).to(torch.bfloat16), torch.nn.Linear(4, 4))
    model[0].requires_grad_(False)
    return model


def test_frozen_weights_are_upcast_only_where_torch_checks_them():
    policy = MixedPrecisionPolicy(param_dtype = torch.bfloat16, reduce_dtype = torch.float32)
    model = _lora_like()
    assert (
        fsdp2.unify_frozen_dtypes(_accelerator(policy, "bf16"), model, frozen_checked = False)
        is False
    )
    assert model[0].weight.dtype == torch.bfloat16
    assert (
        fsdp2.unify_frozen_dtypes(_accelerator(policy, "bf16"), model, frozen_checked = True) is True
    )
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert not model[0].weight.requires_grad
    with pytest.raises(RuntimeError, match = "torch >= 2.8"):
        fsdp2.unify_frozen_dtypes(_accelerator(None, "no"), _lora_like(), frozen_checked = True)
