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
    if positional == "generic":
        # Trainer.__init__ wrapped without functools.wraps: the signature hides `args`.
        def init(self, *args, **kwargs):
            pass

        return _utils._parallel_capabilities_pre(trainer, init, (None, config), {}, None)
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


@pytest.mark.parametrize(
    "positional", [False, True, "generic"], ids = ["kwarg", "positional", "generic_signature"]
)
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


def test_refusal_names_the_context_parallel_patch_error(no_parallel_env):
    import sys

    from unsloth.models import _utils

    cls = _grpo_like(False)
    cls._unsloth_cp_patch_error = "RuntimeError: injected"
    no_parallel_env.delitem(sys.modules, "unsloth._cpu_init", raising = False)
    message = _utils._cp_trainer_refusal(cls, 2, "context_parallel_size = 2")
    assert (
        "could not add context parallelism to UnslothGRPOTrainer (RuntimeError: injected)"
        in message
    )
    assert "import unsloth before trl" not in message, message


_PUBLIC_CONSTRUCTOR = """
import types
import unsloth
import transformers
from unsloth.models import _utils
# The scenario: no Unsloth loader / launch flag ran, so Trainer.__init__ carries no Unsloth hook.
assert not getattr(transformers.Trainer, "_unsloth_init_wrapped_for_accelerate_gas", False)
class GRPOTrainer(transformers.Trainer):
    pass
class UnslothGRPOTrainer(GRPOTrainer):
    def __init__(self, model = None, reward_funcs = None, args = None, **kwargs):
        self.reached_super = True
        super().__init__(model = model, args = args, **kwargs)
_utils.guard_config_cp_after_patch_failure(UnslothGRPOTrainer, RuntimeError("injected"))
assert UnslothGRPOTrainer._unsloth_cp_patch_error == "RuntimeError: injected"
for call in ("kwarg", "positional"):
    config = types.SimpleNamespace(context_parallel_size = 2)
    trainer = object.__new__(UnslothGRPOTrainer)
    try:
        if call == "kwarg":
            UnslothGRPOTrainer.__init__(trainer, None, None, args = config)
        else:
            UnslothGRPOTrainer.__init__(trainer, None, None, config)
    except NotImplementedError as exc:
        assert "context_parallel_size = 2" in str(exc), exc
        assert not getattr(trainer, "reached_super", False)
    else:
        raise AssertionError(call + ": config-only context parallel was not refused")
# cp 1 is untouched: the class's own __init__ runs.
trainer = object.__new__(UnslothGRPOTrainer)
try:
    UnslothGRPOTrainer.__init__(trainer, None, None, args = types.SimpleNamespace(context_parallel_size = 1))
except Exception:
    pass
assert trainer.reached_super
print("OK")
"""


def test_patch_failure_guards_the_public_constructor_without_the_trainer_hook():
    import os
    import subprocess
    import sys

    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "UNSLOTH_PARALLEL",
            "UNSLOTH_TP_SIZE",
            "UNSLOTH_PP_SIZE",
            "UNSLOTH_CP_SIZE",
            "UNSLOTH_FSDP2",
            "UNSLOTH_DISABLE_CPU_FALLBACK",  # conftest's: a GPU-less host imports the CPU backend here
        )
    }
    out = subprocess.run(
        [sys.executable, "-c", _PUBLIC_CONSTRUCTOR],
        capture_output = True,
        text = True,
        env = env,
        timeout = 600,
    )
    assert out.returncode == 0 and "OK" in out.stdout, out.stdout[-2000:] + out.stderr[-4000:]


def test_rl_patch_failure_installs_the_guard():
    # rl.py's handler for a raising patch_rl_trainer: logged, then the class guard (not only the log line).
    import inspect

    from unsloth.models import rl

    source = inspect.getsource(rl)
    handler = source[source.index("Could not add context parallelism to") :][:600]
    assert "guard_config_cp_after_patch_failure" in handler, handler
