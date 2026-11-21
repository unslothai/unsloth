# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastModel passes load_in_4bit = False with a quantization_config that still quantizes: no 16bit notice then."""

import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "bitsandbytes needs a GPU")

NOTICE = "Switching to 16bit LoRA"

SCRIPT = textwrap.dedent(
    """
    import sys
    from unsloth import FastModel
    import bitsandbytes as bnb
    import torch
    from transformers import BitsAndBytesConfig

    kwargs = dict(max_seq_length = 128, load_in_4bit = False)
    if sys.argv[1] == "config":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16,
        )
    model, _ = FastModel.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", **kwargs)
    n = sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
    print("LINEAR4BIT", n)
    """
)


def _load(mode):
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, mode],
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]
    count = int(out.stdout.split("LINEAR4BIT")[-1].split()[0])
    return out.stdout + out.stderr, count


def test_an_explicit_4bit_config_loads_4bit_without_the_16bit_notice():
    log, n_4bit = _load("config")
    assert n_4bit > 0
    assert NOTICE not in log


def test_a_plain_16bit_load_still_prints_the_notice():
    log, n_4bit = _load("plain")
    assert n_4bit == 0
    assert NOTICE in log
