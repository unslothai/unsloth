# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""torch >= 2.12 caches get_hash() and save_config_portable() per thread (ContextVars). A direct config
write mirrored to every thread by fix_dynamo_config_thread_visibility must also drop those caches, or a
worker thread that already hashed the config (autograd / checkpoint recompute) keeps a stale Inductor
FX-graph cache key. CPU only."""

import importlib.util
import threading
from pathlib import Path

import pytest
import torch
from packaging.version import Version

if Version(torch.__version__.split("+")[0]) < Version("2.12.0"):
    pytest.skip("config is process-wide before torch 2.12", allow_module_level = True)

import torch._inductor.config as inductor_config
from torch.utils._config_module import ConfigModule

ROOT = Path(__file__).resolve().parents[1]


def _apply_fix():
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_under_test", ROOT / "unsloth" / "import_fixes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve().is_relative_to(ROOT)
    module.fix_dynamo_config_thread_visibility()
    if not getattr(ConfigModule.__setattr__, "__unsloth_patched__", False):
        pytest.skip("this torch has no per-thread config layout")


def _view(name):
    return inductor_config.save_config_portable(ignore_private_configs = False, readonly_values = True)[
        name
    ]


def test_worker_thread_hash_follows_mirrored_write():
    _apply_fix()
    name = "max_autotune"
    before = getattr(inductor_config, name)
    first, second, seen = threading.Event(), threading.Event(), {}

    def worker():
        seen["hash0"] = inductor_config.get_hash()
        seen["view0"] = _view(name)
        first.set()
        second.wait()
        seen["value1"] = getattr(inductor_config, name)
        seen["hash1"] = inductor_config.get_hash()
        seen["view1"] = _view(name)

    thread = threading.Thread(target = worker)
    thread.start()
    first.wait()
    try:
        setattr(inductor_config, name, not before)
        main_hash = inductor_config.get_hash()
        second.set()
        thread.join()
        assert seen["value1"] is (not before)
        assert seen["hash1"] != seen["hash0"]
        assert seen["hash1"] == main_hash
        assert seen["view0"] is before
        assert seen["view1"] is (not before)
    finally:
        second.set()
        thread.join()
        setattr(inductor_config, name, before)
    assert inductor_config.get_hash() != main_hash
