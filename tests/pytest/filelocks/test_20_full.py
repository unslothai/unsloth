from __future__ import annotations

import os
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def unsloth_model_name() -> str:
    model = os.environ.get("UNSLOTH_TEST_MODEL", "unsloth/Qwen3-4B-Instruct-2507")
    if model == 'qwen':
        model = "unsloth/Qwen3-4B-Instruct-2507"
    elif model == 'gemma3':
        model = "unsloth/gemma-3-4b-it"
    elif model == 'gpt_oss':
        model = "unsloth/gpt-oss-20b"
    return model



@pytest.fixture(scope="session")
def chat_template_name() -> str:
    return os.environ.get("UNSLOTH_CHAT_TEMPLATE", "qwen3")


@pytest.fixture(scope="session")
def can_4bit() -> bool:
    try:
        import torch  # noqa: F401
        import bitsandbytes  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def test_full_end_to_end_clobber(
    per_test_dir: Path,
    project_root: Path,
    hf_cache_env: dict,
    run_many,
    assert_all_ok,
    workers: int,
    unsloth_model_name: str,
    chat_template_name: str,
    can_4bit: bool,
):
    if os.environ.get("UNSLOTH_RUN_FULL", "0") != "1":
        pytest.skip("Set UNSLOTH_RUN_FULL=1 to enable this full clobber test")

    load_in_4bit = (os.environ.get("UNSLOTH_FULL_4BIT", "1") == "1")
    if load_in_4bit and not can_4bit:
        pytest.skip("4-bit requested but no CUDA/bitsandbytes; skipping")

    # Shared caches to maximize lock contention (models + datasets)
    env = dict(hf_cache_env)
    hf_home = Path(env["HF_HOME"])
    env["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    (hf_home / "datasets").mkdir(parents=True, exist_ok=True)

    # Everyone writes into the SAME target dir on purpose
    artifacts = per_test_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    barrier_base = per_test_dir / "barriers"

    enable_push = bool(os.environ.get("UNSLOTH_PUSH_TOKEN"))
    payload_common = {
        "load_in_4bit": load_in_4bit,
        "chat_template": chat_template_name,
        "barrier_base": str(barrier_base),
        "nprocs": workers,
        "dataset_slice": os.environ.get("UNSLOTH_DATASET_SLICE", "train[:1000]"),
        "artifacts_dir": str(artifacts), # shared by all workers
        "enable_push": enable_push,
        "hf_repo": os.environ.get("UNSLOTH_HF_REPO", ""),
        "push_token": os.environ.get("UNSLOTH_PUSH_TOKEN", ""),
        "enable_gguf": os.environ.get("UNSLOTH_ENABLE_GGUF", "1") != "0",
    }

    # Fan out
    payloads = [{"args": [unsloth_model_name], "kwargs": payload_common} for _ in range(workers)]
    results = run_many(
        "tests.pytest.filelocks.workers:run_full",
        payloads,
        cwd=project_root,
        env=env,
        timeout=float(os.environ.get("UNSLOTH_FULL_TIMEOUT", "3600")),  # protects against full hangs
    )
    assert_all_ok(results)

    model_dir = artifacts / "model"
    assert model_dir.exists() and any(model_dir.iterdir()), "model dir missing or empty"

    check = run_many(
        "tests.pytest.filelocks.workers:validate_local_model",
        [{"args": [str(model_dir)], "kwargs": {"load_in_4bit": False}}],
        cwd=project_root,
        env=env,
        timeout=180.0,
        max_parallel=1,
    )
    assert_all_ok(check)
