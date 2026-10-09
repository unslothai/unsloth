# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth#3551: Unsloth's adamw_8bit default cannot step FSDP2's DTensor parameters."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("trl")

_PROBE = textwrap.dedent(
    """
    import json, os
    from unsloth import FastLanguageModel  # noqa: F401, patches the TRL configs
    from trl import GRPOConfig, SFTConfig

    def optim(cls, env, **kwargs):
        for name in ("FSDP_VERSION", "ACCELERATE_USE_FSDP"):
            os.environ.pop(name, None)
        os.environ.update(env)
        value = cls(output_dir = os.environ["PROBE_OUT"], **kwargs).optim
        return str(getattr(value, "value", value))

    fsdp2_args = {"fsdp": "full_shard", "fsdp_config": {"fsdp_version": 2}}
    fsdp2_json = os.environ["PROBE_OUT"] + "_cfg.json"
    with open(fsdp2_json, "w", encoding = "utf-8") as f:
        json.dump({"fsdp_version": 2}, f)
    print("PROBE " + json.dumps({
        "grpo_plain": optim(GRPOConfig, {}),
        "sft_plain": optim(SFTConfig, {}),
        "grpo_fsdp1": optim(GRPOConfig, {"ACCELERATE_USE_FSDP": "true", "FSDP_VERSION": "1"}),
        "grpo_fsdp2": optim(GRPOConfig, {"ACCELERATE_USE_FSDP": "true", "FSDP_VERSION": "2"}),
        "sft_fsdp2": optim(SFTConfig, {"ACCELERATE_USE_FSDP": "true", "FSDP_VERSION": "2"}),
        "grpo_fsdp2_args": optim(GRPOConfig, {}, **fsdp2_args),
        "grpo_fsdp2_json": optim(GRPOConfig, {}, fsdp = "full_shard", fsdp_config = fsdp2_json),
        "grpo_fsdp2_explicit_torch": optim(GRPOConfig, {"FSDP_VERSION": "2"}, optim = "adamw_torch"),
    }))
    """
)


@pytest.fixture(scope = "module")
def probe(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fsdp2_optim")
    env = dict(os.environ)
    env.update(UNSLOTH_COMPILE_LOCATION = str(tmp / "cache"), PROBE_OUT = str(tmp / "out"))
    run = subprocess.run(
        [sys.executable, "-c", _PROBE], env = env, capture_output = True, text = True, timeout = 900
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith("PROBE ")]
    assert lines, f"probe failed:\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}"
    return json.loads(lines[-1][len("PROBE ") :])


def test_without_fsdp2_the_default_is_unchanged(probe):
    assert probe["grpo_plain"] == "adamw_8bit"
    assert probe["sft_plain"] == "adamw_8bit"
    assert probe["grpo_fsdp1"] == "adamw_8bit"


@pytest.mark.parametrize("case", ["grpo_fsdp2", "sft_fsdp2", "grpo_fsdp2_args", "grpo_fsdp2_json"])
def test_fsdp2_gets_an_optimizer_that_steps_dtensors(probe, case):
    assert probe[case] == "adamw_torch_fused"


def test_an_explicit_non_bnb_choice_is_kept(probe):
    assert probe["grpo_fsdp2_explicit_torch"] == "adamw_torch"
