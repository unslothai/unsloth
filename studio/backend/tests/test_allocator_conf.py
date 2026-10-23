# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""covers early PyTorch allocator-config normalization."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from utils.allocator_conf import ALLOCATOR_CONF_ENV_VARS, normalize_allocator_conf


_BACKEND_DIR = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", ALLOCATOR_CONF_ENV_VARS)
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("expandable_segments:false", "expandable_segments:False"),
        ("expandable_segments:true", "expandable_segments:True"),
        ("expandable_segments:FALSE", "expandable_segments:False"),
        ("expandable_segments : tRuE", "expandable_segments : True"),
        (
            "max_split_size_mb:128,expandable_segments:false,pinned_use_cuda_host_register:true",
            "max_split_size_mb:128,expandable_segments:False,pinned_use_cuda_host_register:True",
        ),
        (
            "expandable_segments:false,roundup_power2_divisions:[32:256,64:128,>:32]",
            "expandable_segments:False,roundup_power2_divisions:[32:256,64:128,>:32]",
        ),
    ],
)
def test_noncanonical_booleans_are_capitalized(name, raw, expected):
    env = {name: raw}

    assert normalize_allocator_conf(env) == [(name, raw, expected)]
    assert env == {name: expected}


@pytest.mark.parametrize(
    "raw",
    [
        "expandable_segments:True",
        "expandable_segments:False,max_split_size_mb:128",
        "roundup_power2_divisions:[32:256,64:128,256:64,>:32]",
        "backend:cudaMallocAsync",
        "garbage_collection_threshold:0.6",
        # non-case errors remain for PyTorch to report.
        "expandable_segments:1",
        "expandable_segments:falsey",
        "",
    ],
)
def test_values_without_a_noncanonical_boolean_are_untouched(raw):
    env = {"PYTORCH_ALLOC_CONF": raw}

    assert normalize_allocator_conf(env) == []
    assert env == {"PYTORCH_ALLOC_CONF": raw}


def test_unrelated_variables_are_untouched():
    env = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "false", "UNSLOTH_FLAG": "expandable_segments:false"}
    snapshot = dict(env)

    assert normalize_allocator_conf(env) == []
    assert env == snapshot


def test_uses_os_environ_and_is_idempotent(monkeypatch):
    for name in ALLOCATOR_CONF_ENV_VARS:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:false")

    assert normalize_allocator_conf() == [
        ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:false", "expandable_segments:False")
    ]
    assert normalize_allocator_conf() == []


# `import main` must normalize before CUDA without importing torch so workers inherit the fixed value.
def test_import_main_normalizes_the_allocator_config():
    env = {k: v for k, v in os.environ.items() if k not in ALLOCATOR_CONF_ENV_VARS}
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:false"
    env["UNSLOTH_STUDIO_DISABLE_TORCH_WARM"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sys, main; "
            "print('ALLOC', os.environ['PYTORCH_ALLOC_CONF'], 'torch' in sys.modules)",
        ],
        cwd = str(_BACKEND_DIR),
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert "ALLOC expandable_segments:False False" in proc.stdout
    assert (
        "PYTORCH_ALLOC_CONF='expandable_segments:false' has noncanonical boolean casing, "
        "which PyTorch rejects; using 'expandable_segments:False'."
    ) in proc.stderr
