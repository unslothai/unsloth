# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPOConfig / DPOConfig / SFTConfig(context_parallel_size = N) with no UNSLOTH_CP_SIZE: if the context
parallel wrapper never reached the trainer class (patch_rl_trainer raised, which rl.py only logs), every rank
trained on the whole sequence as plain data parallel. Trainer init now refuses, as it does for the env request."""

import types

import pytest

import unsloth  # noqa: F401


@pytest.fixture
def no_parallel_env(monkeypatch):
    for name in (
        "UNSLOTH_PARALLEL",
        "UNSLOTH_TP_SIZE",
        "UNSLOTH_PP_SIZE",
        "UNSLOTH_CP_SIZE",
        "UNSLOTH_ALLOW_CPU",
        "UNSLOTH_FSDP2",
    ):
        monkeypatch.delenv(name, raising = False)
    return monkeypatch


def _grpo_like(wrapped):
    """The generated GRPO trainer as it is left when patch_rl_trainer raised (wrapped = False)."""
    import trl

    class UnslothGRPOTrainer(trl.GRPOTrainer):
        pass

    UnslothGRPOTrainer.__unsloth_context_parallel__ = wrapped
    return UnslothGRPOTrainer


def _pre(
    trainer_cls,
    cp,
    positional = False,
):
    from unsloth.models import _utils

    trainer = object.__new__(trainer_cls)
    config = types.SimpleNamespace(context_parallel_size = cp)
    if positional:

        def init(
            self,
            model = None,
            args = None,
        ):
            pass

        return _utils._parallel_capabilities_pre(trainer, init, (None, config), {}, None)
    return _utils._parallel_capabilities_pre(
        trainer, trainer_cls.__init__, (), {"args": config}, None
    )


@pytest.mark.parametrize("positional", [False, True], ids = ["kwarg", "positional"])
def test_config_cp_without_the_wrapper_is_refused(no_parallel_env, positional):
    with pytest.raises(NotImplementedError) as info:
        _pre(_grpo_like(False), 2, positional = positional)
    message = str(info.value)
    assert "context_parallel_size = 2" in message, message
    assert "no context parallel wrapper" in message, message


@pytest.mark.parametrize("cp", [1, None, True])
def test_no_cp_request_is_untouched(no_parallel_env, cp):
    assert _pre(_grpo_like(False), cp) is None


def test_wrapped_trainer_keeps_its_config_cp(no_parallel_env):
    assert _pre(_grpo_like(True), 2) is None
