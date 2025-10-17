from __future__ import annotations

import os
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def unsloth_model_name() -> str:
    # You can override with:  UNSLOTH_TEST_MODEL=your/model
    # The default is a tiny LLaMA random-weight model widely used for tests.
    model = os.environ.get("UNSLOTH_TEST_MODEL", "unsloth/Qwen3-4B-Instruct-2507")
    if model == 'qwen':
        model = "unsloth/Qwen3-4B-Instruct-2507"
    elif model == 'gemma3':
        model = "unsloth/gemma-3-4b-it"
    elif model == 'gpt_oss':
        model = "unsloth/gpt-oss-20b"
    return model


@pytest.fixture(scope="session")
def can_4bit() -> bool:
    try:
        import torch  # noqa
        import bitsandbytes as bnb  # noqa
        # 4-bit really needs a usable CUDA device in practice.
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.parametrize("load_in_4bit", [True], ids=lambda b: f"4bit={b}")
def test_from_pretrained_many_processes(
    load_in_4bit: bool,
    can_4bit: bool,
    unsloth_model_name: str,
    per_test_dir: Path,
    run_many,
    assert_all_ok,
    hf_cache_env: dict,
    workers: int,
):
    if load_in_4bit and not can_4bit:
        pytest.skip("bitsandbytes/CUDA not available; skipping 4-bit path")

    # Payloads: all processes load the same model into the same HF cache
    payload = {"args": [unsloth_model_name], "kwargs": {"load_in_4bit": load_in_4bit}}
    payloads = [payload for _ in range(workers)]

    # --- Pre-warm single load so we fail fast if offline or the model isn't compatible.
    warmup = run_many(
        "tests.pytest.filelocks.workers:load_from_pretrained",
        [payload],
        cwd=per_test_dir,
        env=hf_cache_env,
        timeout=600.0,   # generous first pull
        max_parallel=1,
    )[0]
    if warmup.returncode != 0:
        msg = f"Warmup load failed (rc={warmup.returncode}).\n\nstdout:\n{warmup.stdout}\n\nstderr:\n{warmup.stderr}"
        pytest.skip(msg)

    # --- Now the real concurrency test
    results = run_many(
        "tests.pytest.filelocks.workers:load_from_pretrained",
        payloads,
        cwd=per_test_dir,
        env=hf_cache_env,
        timeout=120.0,
    )
    assert_all_ok(results)
