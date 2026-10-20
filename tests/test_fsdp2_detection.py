# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP2 detection in every launch mode: `accelerate launch` (ACCELERATE_USE_FSDP + FSDP_VERSION),
FSDP_VERSION by hand, UNSLOTH_FSDP2=1 under plain torchrun, and a built Accelerator. No GPU."""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_LAUNCH = _ROOT / "unsloth" / "fsdp2_launch.py"

_ENV = (
    "UNSLOTH_FSDP2",
    "UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD",
    "UNSLOTH_FSDP2_FUSED_LORA",
    "UNSLOTH_FORCE_FUSED_LORA",
    "ACCELERATE_USE_FSDP",
    "ACCELERATE_USE_DEEPSPEED",
    "FSDP_VERSION",
    "FSDP_AUTO_WRAP_POLICY",
    "FSDP_RESHARD_AFTER_FORWARD",
    "FSDP_STATE_DICT_TYPE",
    "WORLD_SIZE",
)


def _load_launch():
    # By path: the module needs neither torch nor the unsloth package.
    spec = importlib.util.spec_from_file_location("_unsloth_fsdp2_launch", _LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    # apply_unsloth_fsdp2_launch_env writes os.environ directly, which monkeypatch does not undo for
    # keys that were absent; restore the whole environment so nothing leaks into later test files.
    saved = dict(os.environ)
    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    # These tests cover the accelerate FSDP2 launch; UNSLOTH_FSDP2=1 alone now selects Unsloth's native
    # fully_shard wrap (test_native_backend_exports_nothing).
    monkeypatch.setenv("UNSLOTH_FSDP2_BACKEND", "accelerate")
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_native_backend_exports_nothing(monkeypatch, launch):
    monkeypatch.delenv("UNSLOTH_FSDP2_BACKEND")
    monkeypatch.setenv("UNSLOTH_FSDP2", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert launch.unsloth_fsdp2_native_requested()
    assert launch.fsdp_version_from_env() == 2
    assert launch.apply_unsloth_fsdp2_launch_env() is False
    assert "ACCELERATE_USE_FSDP" not in os.environ and "FSDP_VERSION" not in os.environ


@pytest.fixture
def launch():
    return _load_launch()


def test_no_launcher_env_is_no_fsdp(launch):
    assert launch.fsdp_version_from_env() == 0
    assert launch.apply_unsloth_fsdp2_launch_env() is False
    assert "ACCELERATE_USE_FSDP" not in os.environ


@pytest.mark.parametrize(
    "use_fsdp,version,expected",
    [
        ("true", "2", 2),
        ("true", "1", 1),
        ("true", "", 1),  # accelerate builds FSDP1 when no version is named
        ("", "2", 2),  # torchrun users export FSDP_VERSION by hand
        ("", "0", 0),
        ("false", "", 0),
    ],
)
def test_accelerate_launch_env(monkeypatch, launch, use_fsdp, version, expected):
    monkeypatch.setenv("ACCELERATE_USE_FSDP", use_fsdp)
    monkeypatch.setenv("FSDP_VERSION", version)
    assert launch.fsdp_version_from_env() == expected


def test_unsloth_fsdp2_needs_more_than_one_process(monkeypatch, launch):
    monkeypatch.setenv("UNSLOTH_FSDP2", "1")
    assert launch.fsdp_version_from_env() == 0
    assert launch.apply_unsloth_fsdp2_launch_env() is False
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert launch.apply_unsloth_fsdp2_launch_env() is False
    assert "FSDP_VERSION" not in os.environ


def test_unsloth_fsdp2_exports_the_accelerate_fsdp2_launch(monkeypatch, launch):
    monkeypatch.setenv("UNSLOTH_FSDP2", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert launch.fsdp_version_from_env() == 2
    assert launch.apply_unsloth_fsdp2_launch_env() is True
    assert os.environ["ACCELERATE_USE_FSDP"] == "true"
    assert os.environ["FSDP_VERSION"] == "2"
    assert os.environ["FSDP_AUTO_WRAP_POLICY"] == "TRANSFORMER_BASED_WRAP"
    assert os.environ["FSDP_STATE_DICT_TYPE"] == "FULL_STATE_DICT"
    assert "FSDP_RESHARD_AFTER_FORWARD" not in os.environ
    # Idempotent.
    assert launch.apply_unsloth_fsdp2_launch_env() is True


def test_explicit_fsdp_settings_win(monkeypatch, launch):
    monkeypatch.setenv("UNSLOTH_FSDP2", "true")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("FSDP_AUTO_WRAP_POLICY", "SIZE_BASED_WRAP")
    monkeypatch.setenv("FSDP_STATE_DICT_TYPE", "SHARDED_STATE_DICT")
    launch.apply_unsloth_fsdp2_launch_env()
    assert os.environ["FSDP_AUTO_WRAP_POLICY"] == "SIZE_BASED_WRAP"
    assert os.environ["FSDP_STATE_DICT_TYPE"] == "SHARDED_STATE_DICT"


@pytest.mark.parametrize("value,expected", [("0", "false"), ("false", "false"), ("1", "true")])
def test_reshard_after_forward_switch(monkeypatch, launch, value, expected):
    monkeypatch.setenv("UNSLOTH_FSDP2", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD", value)
    launch.apply_unsloth_fsdp2_launch_env()
    assert os.environ["FSDP_RESHARD_AFTER_FORWARD"] == expected


def _fake_accelerate(monkeypatch, launch, tmp_path, source):
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils" / "fsdp_utils.py").write_text(source, encoding = "utf-8")
    spec = importlib.util.spec_from_file_location("accelerate", tmp_path / "__init__.py")
    spec.submodule_search_locations = [str(tmp_path)]
    monkeypatch.setattr(launch.importlib.util, "find_spec", lambda name: spec)


@pytest.mark.parametrize(
    "source,ok",
    [
        ("def fsdp_prepare_model(): pass\n", False),  # accelerate < 1.6: no FSDP2
        ("def fsdp2_prepare_model(): pass\n", False),  # 1.6: no FSDP2 state dict options
        ("def fsdp2_prepare_model(): pass\ndef _prepare_sd_options(): pass\n", True),
    ],
)
def test_unsloth_fsdp2_needs_an_accelerate_with_fsdp2(monkeypatch, launch, tmp_path, source, ok):
    _fake_accelerate(monkeypatch, launch, tmp_path, source)
    monkeypatch.setenv("UNSLOTH_FSDP2", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    if ok:
        assert launch.apply_unsloth_fsdp2_launch_env() is True
    else:
        with pytest.raises(RuntimeError, match = "accelerate >= 1.7"):
            launch.apply_unsloth_fsdp2_launch_env()
        assert "FSDP_VERSION" not in os.environ


def test_the_installed_accelerate_is_read_without_importing_it(launch):
    pytest.importorskip("accelerate")
    assert launch._accelerate_fsdp2_support() is True


def test_conflicting_launches_are_refused(monkeypatch, launch):
    monkeypatch.setenv("UNSLOTH_FSDP2", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("FSDP_VERSION", "1")
    with pytest.raises(ValueError, match = "FSDP_VERSION=1"):
        launch.apply_unsloth_fsdp2_launch_env()
    monkeypatch.delenv("FSDP_VERSION")
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    with pytest.raises(RuntimeError, match = "DeepSpeed"):
        launch.apply_unsloth_fsdp2_launch_env()


def test_the_env_becomes_accelerates_fsdp2_plugin(monkeypatch, launch):
    """What accelerate builds from the exported env: FSDP2, one unit per `_no_split_modules` block."""
    accelerate_utils = pytest.importorskip("accelerate.utils")
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    monkeypatch.setenv("UNSLOTH_FSDP2", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD", "0")
    launch.apply_unsloth_fsdp2_launch_env()
    plugin = accelerate_utils.FullyShardedDataParallelPlugin()
    assert plugin.fsdp_version == 2
    assert plugin.auto_wrap_policy is transformer_auto_wrap_policy
    assert plugin.reshard_after_forward is False
    assert "FULL_STATE_DICT" in str(plugin.state_dict_type)


def test_unsloth_init_applies_the_env_before_transformers():
    """The call must sit in unsloth/__init__.py ahead of any import that pulls in transformers."""
    source = (_ROOT / "unsloth" / "__init__.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    call_line = None
    first_heavy = None
    for node in tree.body:
        text = ast.unparse(node)
        if call_line is None and "_apply_fsdp2_env()" in text:
            call_line = node.lineno
        if first_heavy is None and isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(x in text for x in ("transformers", "accelerate", "unsloth_zoo", ".models")):
                first_heavy = node.lineno
    assert call_line is not None
    assert first_heavy is None or call_line < first_heavy


def test_loader_utils_versions(monkeypatch):
    loader_utils = pytest.importorskip("unsloth.models.loader_utils")
    accelerate_state = pytest.importorskip("accelerate.state")
    shared = accelerate_state.AcceleratorState._shared_state
    monkeypatch.delitem(shared, "fsdp_plugin", raising = False)
    monkeypatch.delitem(shared, "distributed_type", raising = False)

    assert loader_utils.fsdp_version_requested() == 0
    assert loader_utils.fsdp2_keeps_fused_lora() is False

    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    monkeypatch.setenv("FSDP_VERSION", "1")
    assert loader_utils.fsdp_will_wrap() is True
    assert loader_utils.fsdp2_keeps_fused_lora() is False

    monkeypatch.setenv("FSDP_VERSION", "2")
    assert loader_utils.fsdp2_requested() is True
    assert loader_utils.fsdp2_keeps_fused_lora() is True
    monkeypatch.setenv("UNSLOTH_FSDP2_FUSED_LORA", "0")
    assert loader_utils.fsdp2_keeps_fused_lora() is False
    monkeypatch.delenv("UNSLOTH_FSDP2_FUSED_LORA")

    # UNSLOTH_FSDP2=1 under torchrun, before any env was exported.
    monkeypatch.delenv("ACCELERATE_USE_FSDP")
    monkeypatch.delenv("FSDP_VERSION")
    monkeypatch.setenv("UNSLOTH_FSDP2", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert loader_utils.fsdp2_requested() is True


def test_a_built_accelerator_is_authoritative(monkeypatch):
    """An FSDP1 plugin wins over a stale FSDP_VERSION=2 in the env, and vice versa."""
    import types

    loader_utils = pytest.importorskip("unsloth.models.loader_utils")
    accelerate_state = pytest.importorskip("accelerate.state")
    shared = accelerate_state.AcceleratorState._shared_state
    monkeypatch.setenv("FSDP_VERSION", "2")
    monkeypatch.setitem(shared, "fsdp_plugin", types.SimpleNamespace(fsdp_version = 1))
    assert loader_utils.fsdp_version_requested() == 1
    assert loader_utils.fsdp2_keeps_fused_lora() is False
    monkeypatch.delenv("FSDP_VERSION")
    monkeypatch.setitem(shared, "fsdp_plugin", types.SimpleNamespace(fsdp_version = 2))
    assert loader_utils.fsdp2_requested() is True


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _rank_worker(rank, port, out_dir):
    import json

    import torch.distributed as dist

    os.environ.update(
        RANK = str(rank),
        LOCAL_RANK = str(rank),
        WORLD_SIZE = "2",
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        UNSLOTH_FSDP2 = "1",
    )
    launch = _load_launch()
    launch.apply_unsloth_fsdp2_launch_env()
    dist.init_process_group("gloo", rank = rank, world_size = 2)
    try:
        mine = (
            launch.fsdp_version_from_env(),
            os.environ.get("ACCELERATE_USE_FSDP"),
            os.environ.get("FSDP_AUTO_WRAP_POLICY"),
        )
        everyone = [None, None]
        dist.all_gather_object(everyone, mine)
        if rank == 0:
            with open(os.path.join(out_dir, "ranks.json"), "w", encoding = "utf-8") as f:
                json.dump(everyone, f)
    finally:
        dist.destroy_process_group()


def test_every_rank_reads_the_same_fsdp2_launch(tmp_path):
    import json

    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend unavailable")
    import torch.multiprocessing as mp

    mp.spawn(_rank_worker, args = (_free_port(), str(tmp_path)), nprocs = 2, join = True)
    ranks = json.loads((tmp_path / "ranks.json").read_text(encoding = "utf-8"))
    assert ranks[0] == ranks[1] == [2, "true", "TRANSFORMER_BASED_WRAP"]


def test_unsloth_init_installs_the_fsdp2_hooks_without_a_launcher_env():
    """`Accelerator(fsdp_plugin = FullyShardedDataParallelPlugin(fsdp_version = 2))` in a raw loop
    sets no env before unsloth is imported, so the install must not be gated on it."""
    import ast

    source = (_ROOT / "unsloth" / "__init__.py").read_text(encoding = "utf-8")
    gated = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If)
        and "fsdp_version_from_env" in ast.unparse(node.test)
        and "_install_fsdp2_hooks" in ast.unparse(node)
    ]
    assert not gated, "install_fsdp2_hooks is gated on the launcher env"
    assert "_install_fsdp2_hooks()" in source
