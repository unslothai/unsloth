# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pre-Volta NVIDIA GPUs (sm < 7.0) train with torch.compile off (#1998).

Inductor raises GPUTooOldForTriton on the first compiled call there, which killed GRPO,
vision and MoE training mid-run on GTX 10xx / P100 / Maxwell cards. Decision-table checks, no GPU.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from unsloth import device_type


@pytest.mark.parametrize(
    "major, applied",
    [
        (5, True),  # Maxwell: Quadro M4000 / M6000 in the issue
        (6, True),  # Pascal: GTX 1070 / 1080, Kaggle P100
        (7, False),  # Volta / Turing: Inductor's own floor, T4 keeps compiling
        (8, False),
        (9, False),
        (10, False),
        (12, False),
        (None, False),  # unreadable capability: never guess
    ],
)
def test_only_pre_volta_turns_compile_off(major, applied):
    env, config = {}, SimpleNamespace(disable = False)
    assert device_type.apply_pre_volta_compile_workaround(major, env, config) is applied
    assert config.disable is applied
    if applied:
        assert env == {
            "TORCHDYNAMO_DISABLE": "1",
            "TORCH_COMPILE_DISABLE": "1",
            "UNSLOTH_COMPILE_DISABLE": "1",
        }
    else:
        assert env == {}


def test_dynamo_config_is_set_not_just_the_env():
    """Regions zoo wrapped at import never reread TORCHDYNAMO_DISABLE; env-only left MoE failing."""
    config = SimpleNamespace(disable = False)
    device_type.apply_pre_volta_compile_workaround(6, {}, config)
    assert config.disable is True


def test_an_explicit_user_choice_is_left_alone():
    env, config = {"TORCHDYNAMO_DISABLE": "0"}, SimpleNamespace(disable = False)
    assert device_type.apply_pre_volta_compile_workaround(6, env, config) is False
    assert env == {"TORCHDYNAMO_DISABLE": "0"}
    assert config.disable is False


def test_other_compile_switches_the_user_set_are_kept():
    env = {"UNSLOTH_COMPILE_DISABLE": "partial"}
    device_type.apply_pre_volta_compile_workaround(6, env, SimpleNamespace(disable = False))
    assert env["UNSLOTH_COMPILE_DISABLE"] == "partial"
