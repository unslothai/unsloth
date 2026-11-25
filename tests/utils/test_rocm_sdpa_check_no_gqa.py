# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""ROCm SDPA self-check on a torch without enable_gqa (the 2.4 floor): the flash child must not count the
TypeError of an argument Unsloth never sends there as a kernel failure, which would refuse flash for every call.

CPU: the real child script runs in a subprocess with CUDA calls mapped to CPU (torch's CPU flash kernel) and
F.scaled_dot_product_attention replaced by one that rejects enable_gqa, as torch 2.4's operator does."""

import json
import subprocess
import sys

import pytest
import torch

check = pytest.importorskip("unsloth.utils.rocm_sdpa_check")

_CPU_SHIM = r"""
import torch
import torch.nn.functional as F

_device = torch.device
torch.device = lambda *args, **kwargs: _device("cpu")
torch.cuda.set_device = lambda index: None
torch.cuda.synchronize = lambda *args, **kwargs: None
torch.cuda.is_bf16_supported = lambda *args, **kwargs: True
_sdpa = F.scaled_dot_product_attention

def _sdpa_without_gqa(*args, **kwargs):
    if "enable_gqa" in kwargs:
        raise TypeError("scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'")
    return _sdpa(*args, **kwargs)

_sdpa_without_gqa.__doc__ = "scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)"
F.scaled_dot_product_attention = _sdpa_without_gqa
"""


def _child_report(backend):
    done = subprocess.run(
        [sys.executable, "-c", _CPU_SHIM + check._CHILD, "0", backend],
        capture_output = True,
        text = True,
        timeout = 900,
    )
    line = next(
        (l for l in done.stdout.splitlines() if l.startswith("UNSLOTH_ROCM_SDPA_CHECK ")), None
    )
    assert line is not None, (done.returncode, done.stderr[-2000:])
    return json.loads(line.split(" ", 1)[1])


def test_flash_child_skips_enable_gqa_where_torch_has_none():
    report = _child_report("flash")
    assert report["ok"] is True, report
    # The expanded K / V segments still ran: fp16 + bf16 at head dims 64 and 128.
    assert report["ran"] == 4, report
    assert not any("enable_gqa" in d for d in report["detail"]), report
