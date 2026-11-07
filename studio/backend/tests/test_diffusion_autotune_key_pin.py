# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The parent looks up a kernel's tuned config under the key a fresh compile worker saved it with."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
autotune_cache = pytest.importorskip("torch._inductor.runtime.autotune_cache")

import torch.compiler.config as cconfig  # noqa: E402

from core.inference import diffusion_aot_cache as aot  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
PARENT_TAG = "unsloth/rope_embedding.py:0123456789abcdef"
_WORKER = """
import json, sys
from torch._inductor.runtime.autotune_cache import AutotuneCache
cache = AutotuneCache.create({}, sys.argv[1], "h")
print(json.dumps(cache.local_cache[1]))
"""

if not hasattr(autotune_cache.AutotuneCache, "_prepare_key"):
    pytest.skip(
        reason = "this torch has no AutotuneCache._prepare_key: pin_autotune_key returns False, nothing to pin",
        allow_module_level = True,
    )


@pytest.fixture(autouse = True)
def _restore(monkeypatch):
    raw = autotune_cache.AutotuneCache.__dict__["_prepare_key"]
    if getattr(
        raw.__func__, "_unsloth_worker_tag", False
    ):  # pinned by an earlier test in this process
        autotune_cache.AutotuneCache._prepare_key = staticmethod(raw.__func__.__wrapped__)
    tag = cconfig.cache_key_tag
    monkeypatch.delenv("TORCH_COMPILE_CACHE_KEY_TAG", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_AUTOTUNE_KEY_PIN", raising = False)
    yield
    autotune_cache.AutotuneCache._prepare_key = raw
    cconfig.cache_key_tag = tag


def _kernel(tmp_path: Path) -> str:
    path = tmp_path / "ab" / "cabcdefghijklmnop.py"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text("# kernel\n")
    return str(path)


def _parent_path(filename: str) -> str:
    return autotune_cache.AutotuneCache.create({}, filename, "h").local_cache[1]


def _worker_path(filename: str, env_tag = None) -> str:
    """What a compile worker computes: a fresh interpreter, torch's default tag."""
    env = {k: v for k, v in os.environ.items() if k != "TORCH_COMPILE_CACHE_KEY_TAG"}
    if env_tag is not None:
        env["TORCH_COMPILE_CACHE_KEY_TAG"] = env_tag
    out = subprocess.run(
        [sys.executable, "-c", _WORKER, filename],
        capture_output = True,
        text = True,
        env = env,
        check = True,
        timeout = 300,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_tagged_parent_reads_what_the_worker_saved(tmp_path):
    kernel = _kernel(tmp_path)
    worker = _worker_path(kernel)
    cconfig.cache_key_tag = PARENT_TAG
    assert _parent_path(kernel) != worker  # the defect: every returning start missed
    assert aot.pin_autotune_key() is True
    assert _parent_path(kernel) == worker
    # The parent's own tag is untouched: FX graph cache keys still carry it.
    assert cconfig.cache_key_tag == PARENT_TAG


def test_untagged_parent_key_unchanged(tmp_path):
    kernel = _kernel(tmp_path)
    before = _parent_path(kernel)
    assert aot.pin_autotune_key() is True
    assert _parent_path(kernel) == before == _worker_path(kernel)


def test_env_tag_is_the_workers_tag(tmp_path, monkeypatch):
    kernel = _kernel(tmp_path)
    monkeypatch.setenv("TORCH_COMPILE_CACHE_KEY_TAG", "site")
    worker = _worker_path(kernel, env_tag = "site")
    cconfig.cache_key_tag = "site," + PARENT_TAG
    assert aot.pin_autotune_key() is True
    assert _parent_path(kernel) == worker


def test_idempotent(tmp_path):
    assert aot.pin_autotune_key() is True
    first = autotune_cache.AutotuneCache.__dict__["_prepare_key"]
    assert aot.pin_autotune_key() is True
    assert autotune_cache.AutotuneCache.__dict__["_prepare_key"] is first


def test_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_AUTOTUNE_KEY_PIN", "0")
    raw = autotune_cache.AutotuneCache.__dict__["_prepare_key"]
    assert aot.pin_autotune_key() is False
    assert autotune_cache.AutotuneCache.__dict__["_prepare_key"] is raw


def test_missing_hook_is_a_no_op(monkeypatch):
    monkeypatch.delattr(autotune_cache.AutotuneCache, "_prepare_key")
    assert aot.pin_autotune_key() is False


def test_unknown_key_formula_is_left_alone(monkeypatch):
    other = staticmethod(lambda filename: "not-the-known-formula")
    monkeypatch.setattr(autotune_cache.AutotuneCache, "_prepare_key", other)
    assert aot.pin_autotune_key() is False
    assert autotune_cache.AutotuneCache.__dict__["_prepare_key"] is other


def test_torch_that_hands_workers_its_tag_is_left_alone(monkeypatch):
    raw = autotune_cache.AutotuneCache.__dict__["_prepare_key"]
    monkeypatch.setattr(aot, "_workers_get_parent_tag", lambda: True)
    assert aot.pin_autotune_key() is False
    assert autotune_cache.AutotuneCache.__dict__["_prepare_key"] is raw


def _calls(path: Path, func: str) -> set:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            return {
                getattr(c.func, "attr", getattr(c.func, "id", None))
                for c in ast.walk(node)
                if isinstance(c, ast.Call)
            }
    raise AssertionError(func)


def test_wired_into_both_compile_entry_points():
    inference = BACKEND / "core" / "inference"
    assert "pin_autotune_key" in _calls(
        inference / "diffusion_speed.py", "_compile_repeated_blocks"
    )
    assert "pin_autotune_key" in _calls(inference / "diffusion_compile_cache.py", "begin")
