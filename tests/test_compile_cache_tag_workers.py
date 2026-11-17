# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Inductor's compile workers must key autotune results with the same cache_key_tag as this
process. A worker reads the tag only from TORCH_COMPILE_CACHE_KEY_TAG, so a tag set here alone
made workers save each .best_config under the empty tag while warm runs looked it up under the
full one, and every warm run re-benchmarked those kernels. CPU only."""

import ast
import importlib.util
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")

ROOT = Path(__file__).resolve().parents[1]
_CONFIG = getattr(torch.compiler, "config", None)
if not hasattr(_CONFIG, "cache_key_tag"):
    pytest.skip("torch has no compile cache key tag", allow_module_level = True)
_ENV = "TORCH_COMPILE_CACHE_KEY_TAG"


def _load():
    spec = importlib.util.spec_from_file_location(
        "unsloth_triton_launch_tag_under_test", ROOT / "unsloth" / "kernels" / "triton_launch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_tag(monkeypatch):
    previous = _CONFIG.cache_key_tag
    monkeypatch.delenv(_ENV, raising = False)
    _CONFIG.cache_key_tag = ""
    try:
        yield
    finally:
        _CONFIG.cache_key_tag = previous


def _kernel(tmp_path):
    kernel = tmp_path / f"fake_kernel_{uuid.uuid4().hex[:8]}.py"
    kernel.write_text(f"# {uuid.uuid4()}\n")
    return str(kernel)


def _worker_autotune_key(env):
    """The autotune key a fresh interpreter (as an Inductor compile worker is) computes."""
    code = (
        "from torch._inductor.runtime.autotune_cache import AutotuneCache;"
        "print(AutotuneCache._prepare_key('/x/ckernel.py'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env = env,
        capture_output = True,
        text = True,
        check = True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_tag_reaches_the_environment(clean_tag, tmp_path):
    launch = _load()
    launch.tag_compile_cache(_kernel(tmp_path))
    tags = _CONFIG.cache_key_tag.split(",")
    assert launch._WORKER_TAG in tags
    assert any(t.startswith("unsloth/fake_kernel_") for t in tags)
    assert os.environ[_ENV] == _CONFIG.cache_key_tag


def test_tagging_twice_adds_nothing(clean_tag, tmp_path):
    launch = _load()
    kernel = _kernel(tmp_path)
    launch.tag_compile_cache(kernel)
    first = _CONFIG.cache_key_tag
    launch.tag_compile_cache(kernel)
    assert _CONFIG.cache_key_tag == first
    assert first.count(launch._WORKER_TAG) == 1


def test_sync_keeps_a_tag_set_elsewhere(clean_tag):
    # unsloth_zoo adds its own tag through torch config only; the final sync forwards it too.
    launch = _load()
    _CONFIG.cache_key_tag = "unsloth/inductor-rand4x-off"
    assert launch.sync_compile_cache_tag_to_workers() == "unsloth/inductor-rand4x-off"
    assert os.environ[_ENV] == "unsloth/inductor-rand4x-off"


def test_empty_tag_leaves_the_environment_alone(clean_tag):
    launch = _load()
    assert launch.sync_compile_cache_tag_to_workers() == ""
    assert _ENV not in os.environ


def test_worker_and_parent_autotune_keys_match(clean_tag, tmp_path):
    from torch._inductor.runtime.autotune_cache import AutotuneCache

    launch = _load()
    launch.tag_compile_cache(_kernel(tmp_path))
    parent = AutotuneCache._prepare_key("/x/ckernel.py")
    env = dict(os.environ)
    assert _worker_autotune_key(env) == parent
    # Without the environment copy (the old behaviour) the worker keys under the empty tag.
    env.pop(_ENV)
    assert _worker_autotune_key(env) != parent


def test_init_syncs_after_the_kernel_imports():
    tree = ast.parse((ROOT / "unsloth" / "__init__.py").read_text(encoding = "utf-8"))
    lines = {
        node.func.id: node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    # At the end of the package, after every kernel module (each calls tag_compile_cache).
    assert "_sync_tag" in lines
    assert lines["_sync_tag"] < lines["_restore_gc_threshold"]
