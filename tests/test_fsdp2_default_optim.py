# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth#3551: Unsloth's adamw_8bit default under FSDP2. The generated configs used to switch it to
adamw_torch_fused; FSDP2 (native and accelerate's) now steps bitsandbytes Adam / AdamW on each rank's
local shard, so the configs keep it (tests/test_fsdp2_bnb_optim_kept.py)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from real_accelerator import has_real_cuda

pytest.importorskip("trl")
# The probe imports unsloth in a fresh process, outside the conftest CPU spoof.
pytestmark = pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")

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
        "sft_fsdp2_paged": optim(SFTConfig, {"FSDP_VERSION": "2"}, optim = "paged_adamw_8bit"),
        "grpo_fsdp_unversioned": optim(GRPOConfig, {}, fsdp = "full_shard"),
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
    assert "Switching optim" not in run.stdout
    return json.loads(lines[-1][len("PROBE ") :])


def test_without_fsdp2_the_default_is_unchanged(probe):
    assert probe["grpo_plain"] == "adamw_8bit"
    assert probe["sft_plain"] == "adamw_8bit"
    assert probe["grpo_fsdp1"] == "adamw_8bit"


@pytest.mark.parametrize("case", ["grpo_fsdp2", "sft_fsdp2", "grpo_fsdp2_args", "grpo_fsdp2_json"])
def test_fsdp2_keeps_adamw_8bit(probe, case):
    # Stepped on the local shards (fsdp2_bnb_optim), not silently switched to torch's AdamW.
    assert probe[case] == "adamw_8bit"


def test_fsdp_without_a_version_keeps_adamw_8bit(probe):
    assert probe["grpo_fsdp_unversioned"] == "adamw_8bit"


def test_explicit_choices_are_kept(probe):
    assert probe["grpo_fsdp2_explicit_torch"] == "adamw_torch"
    assert probe["sft_fsdp2_paged"] == "paged_adamw_8bit"
