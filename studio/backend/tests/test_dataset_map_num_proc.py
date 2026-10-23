# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""None, not 1, disables workers: datasets >= 4.1 builds a pool for any num_proc >= 1."""

from __future__ import annotations

import sys
import types

import pytest

import utils.hardware.hardware as hw

try:
    # Before the fixture spoofs sys.platform: multiprocess picks its contexts at import time.
    import multiprocess  # noqa: F401
except ImportError:
    pass

# The real one, read before anything can lie about it.
_HOST_PLATFORM = sys.platform


@pytest.fixture(autouse = True)
def _fork_platform(monkeypatch):
    """Pin Linux: dataset_map_num_proc returns None on win32 and darwin, so count assertions need it."""
    monkeypatch.setattr(sys, "platform", "linux")


@pytest.fixture(autouse = True)
def _memory_headroom(monkeypatch):
    """Pin the affordable-worker count, since free RAM would otherwise clamp every asserted count."""
    policy = hw._shared_policy()
    if policy is None:
        return
    monkeypatch.setattr(policy, "_affordable_workers", lambda: 64)


def _require_fork(multiprocess):
    """Skip without fork: spawn pools fail for unrelated reasons (WinError 10038, missing os.WNOHANG)."""
    if _HOST_PLATFORM == "win32" or "fork" not in multiprocess.get_all_start_methods():
        pytest.skip("needs fork to build a real worker pool")


def _policy_or_skip():
    """Use this repo's policy, not unsloth_zoo.dataset_num_proc: that import skipped every case on CI."""
    policy = hw._shared_policy()
    if policy is None:
        pytest.skip("no dataset_num_proc policy on this installation")
    return policy


def _patch_device(
    monkeypatch,
    device,
    *,
    visible_gpus: int = 1,
):
    monkeypatch.setattr(hw, "get_device", lambda: device)
    monkeypatch.setattr(hw, "get_visible_gpu_count", lambda: visible_gpus)


def _torch_module(monkeypatch):
    """A stand-in torch is needed: ImportError would read as untouched runtime and no-op the XPU guard."""
    try:
        import torch
        return torch
    except ImportError:
        stub = types.ModuleType("torch")
        monkeypatch.setitem(sys.modules, "torch", stub)
        return stub


def _patch_runtime(monkeypatch, name, *, is_initialized):
    """Fake torch.<name>.is_initialized(): a bool, or a callable that raises to model a failing probe."""
    torch = _torch_module(monkeypatch)

    if callable(is_initialized):
        probe = is_initialized
    else:
        probe = lambda: is_initialized  # noqa: E731

    monkeypatch.setattr(torch, name, types.SimpleNamespace(is_initialized = probe), raising = False)


def test_dataset_map_num_proc_parallelizes_on_initialized_cuda(monkeypatch):
    """Initialized CUDA must not disable workers: 300 forked map() runs showed no failures."""
    _patch_device(monkeypatch, hw.DeviceType.CUDA)
    _patch_runtime(monkeypatch, "cuda", is_initialized = True)
    assert hw.dataset_map_num_proc(4) == 4


def test_dataset_map_num_proc_cuda_respects_multi_gpu_cap(monkeypatch):
    _patch_device(monkeypatch, hw.DeviceType.CUDA)
    _patch_runtime(monkeypatch, "cuda", is_initialized = True)
    monkeypatch.setattr(hw, "get_visible_gpu_count", lambda: 2)
    assert hw.dataset_map_num_proc(16) == 4


def test_dataset_map_num_proc_none_after_xpu_init(monkeypatch):
    _patch_device(monkeypatch, hw.DeviceType.XPU)
    _patch_runtime(monkeypatch, "xpu", is_initialized = True)
    assert hw.dataset_map_num_proc(4) is None


def test_dataset_map_num_proc_parallel_before_xpu_init(monkeypatch):
    _patch_device(monkeypatch, hw.DeviceType.XPU)
    _patch_runtime(monkeypatch, "xpu", is_initialized = False)
    assert hw.dataset_map_num_proc(4) == 4


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_dataset_map_num_proc_none_on_spawn_platforms(monkeypatch, platform):
    # Must be None and never 1: datasets >= 4.1 builds a Pool(1) for num_proc=1.
    monkeypatch.setattr(sys, "platform", platform)
    assert hw.dataset_map_num_proc(4) is None


def test_dataset_map_num_proc_cpu_host_parallelizes(monkeypatch):
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    assert hw.dataset_map_num_proc(4) == 4


def test_none_builds_no_pool_but_a_count_does(monkeypatch):
    """None must mean no pool in the installed datasets, while a count builds one."""
    datasets = pytest.importorskip("datasets")
    multiprocess = pytest.importorskip("multiprocess")
    _require_fork(multiprocess)

    pools_built = []
    real_pool = multiprocess.Pool

    def _spy_pool(*args, **kwargs):
        pools_built.append((args, kwargs))
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(multiprocess, "Pool", _spy_pool)
    monkeypatch.setattr(datasets.arrow_dataset, "Pool", _spy_pool, raising = False)

    dataset = datasets.Dataset.from_dict({"text": [f"row {i}" for i in range(8)]})
    _count = lambda batch: {"n": [len(t) for t in batch["text"]]}  # noqa: E731

    mapped = dataset.map(_count, batched = True, num_proc = None)
    assert len(mapped) == 8
    assert pools_built == [], f"Dataset.map built a worker pool: {pools_built}"

    dataset.map(_count, batched = True, num_proc = 2)
    assert len(pools_built) == 1, "Pool spy never fired; the no-pool check is vacuous"


def test_num_proc_one_is_not_a_disable_sentinel():
    """Only None is in-process on both datasets 3.x and 4.x; num_proc=1 builds a Pool(1) on 4.x."""
    datasets = pytest.importorskip("datasets")
    multiprocess = pytest.importorskip("multiprocess")
    _require_fork(multiprocess)
    from packaging.version import Version

    # Spy on the Pool class: datasets 5.x calls mp.Pool(), so arrow_dataset.Pool may not exist.
    import multiprocess.pool

    pools_built = []
    real_init = multiprocess.pool.Pool.__init__

    def _spy_init(self, *args, **kwargs):
        pools_built.append((args, kwargs))
        return real_init(self, *args, **kwargs)

    dataset = datasets.Dataset.from_dict({"text": [f"row {i}" for i in range(8)]})
    _count = lambda batch: {"n": [len(t) for t in batch["text"]]}  # noqa: E731

    multiprocess.pool.Pool.__init__ = _spy_init
    try:
        dataset.map(_count, batched = True, num_proc = None)
        assert pools_built == [], "num_proc=None must always run in-process"

        dataset.map(_count, batched = True, num_proc = 1)
        built_for_one = len(pools_built)
    finally:
        multiprocess.pool.Pool.__init__ = real_init

    if Version(datasets.__version__) >= Version("4.1.0"):
        assert built_for_one == 1, (
            f"datasets {datasets.__version__} was expected to build a Pool(1); "
            "if that changed, dataset_map_num_proc's docstring needs updating"
        )
    else:
        assert built_for_one == 0


def test_a_low_memory_host_gets_no_workers(monkeypatch):
    """Direct callers need the shared policy: a 2GB container with 8 cores otherwise gets 8 workers."""
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    policy = _policy_or_skip()
    monkeypatch.setattr(policy, "_affordable_workers", lambda: 0)
    monkeypatch.setattr(policy, "multiprocessing_start_method", lambda: "fork")
    assert hw.dataset_map_num_proc(8) is None


def test_the_memory_clamp_reduces_rather_than_refuses(monkeypatch):
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    policy = _policy_or_skip()
    monkeypatch.setattr(policy, "_affordable_workers", lambda: 3)
    monkeypatch.setattr(policy, "multiprocessing_start_method", lambda: "fork")
    assert hw.dataset_map_num_proc(8) == 3


def test_the_env_override_reaches_these_callers(monkeypatch):
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    policy = _policy_or_skip()
    monkeypatch.setattr(policy, "multiprocessing_start_method", lambda: "fork")
    policy.reset_warning_state()

    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", "2")
    assert hw.dataset_map_num_proc(8) == 2

    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", "0")
    assert hw.dataset_map_num_proc(8) is None


def test_an_older_unsloth_zoo_keeps_the_previous_behaviour(monkeypatch):
    # Lazy, guarded import: hardware detection must not depend on the training package.
    import builtins

    real_import = builtins.__import__

    def _no_policy(name, *args, **kwargs):
        if name == "unsloth_zoo.dataset_num_proc":
            raise ImportError("older unsloth_zoo")
        return real_import(name, *args, **kwargs)

    _patch_device(monkeypatch, hw.DeviceType.CPU)
    monkeypatch.setattr(builtins, "__import__", _no_policy)
    assert hw.dataset_map_num_proc(4) == 4


def test_the_cap_no_longer_advertises_an_override_it_cannot_honour():
    """safe_num_proc cannot express in-process, so it must not advertise UNSLOTH_DATASET_NUM_PROC."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(hw.safe_num_proc))
    logged = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).startswith("logger.")
    ]
    assert logged, "the cap stopped logging; re-check what this is guarding"
    assert not any(
        "UNSLOTH_DATASET_NUM_PROC" in line for line in logged
    ), "safe_num_proc tells the user to set an override it never reads"


def test_a_serial_request_survives_the_config_round_trip(monkeypatch):
    """The audio paths ask for 1; the config layer must still see a request for 1."""
    _policy_or_skip()
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    assert hw.dataset_map_num_proc(1, serial_as_none = False) == 1
    assert hw.dataset_map_num_proc(1) is None


def test_the_config_value_is_still_in_process_after_the_layer_reads_it(monkeypatch):
    """End to end: what Unsloth stores, read back the way the SFT config layer reads it."""
    policy = _policy_or_skip()
    _patch_device(monkeypatch, hw.DeviceType.CPU)

    stored = hw.dataset_map_num_proc(1, serial_as_none = False)
    from_config = policy.get_dataset_num_proc(stored, serial_as_none = False)
    at_the_map_site = policy.get_dataset_num_proc(from_config)
    assert (stored, from_config, at_the_map_site) == (1, 1, None)


def test_xpu_initialized_stays_serial_through_a_config(monkeypatch):
    """Initialized XPU must stay serial via config: forking corrupts the Level-Zero context."""
    _patch_device(monkeypatch, hw.DeviceType.XPU)
    _patch_runtime(monkeypatch, "xpu", is_initialized = True)
    assert hw.dataset_map_num_proc(4, serial_as_none = False) == 1
    assert hw.dataset_map_num_proc(4) is None


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_spawn_platforms_keep_none_at_either_layer(monkeypatch, platform):
    """Store None, not 1: a Pool(1) on spawn platforms re-imports the user's __main__ in its child."""
    monkeypatch.setattr(sys, "platform", platform)
    assert hw.dataset_map_num_proc(4, serial_as_none = False) is None


def test_every_other_caller_keeps_the_map_site_default(monkeypatch):
    """Only the config boundary opts in; the seven map-site callers must not."""
    _policy_or_skip()
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    assert hw.dataset_map_num_proc(1) is None
    assert hw.dataset_map_num_proc(4) == 4


def test_the_trainer_config_asks_for_the_config_sentinel():
    """SFTConfig must pass serial_as_none = False; dropping it silently gives audio paths workers."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(source)

    config_calls = [
        value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant)
        and key.value == "dataset_num_proc"
        and isinstance(value, ast.Call)
        and ast.unparse(value.func) == "dataset_map_num_proc"
    ]
    assert config_calls, "the SFTConfig dataset_num_proc entry moved; re-check this guard"
    for call in config_calls:
        keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert keywords.get("serial_as_none") == "False", (
            "a config-boundary dataset_map_num_proc call lost serial_as_none = False, "
            "so a serial request will be read back as 'auto-size me': "
            f"{ast.unparse(call)}"
        )


def _unexpected_auto_sizing(desired = None):
    if desired is None:
        raise AssertionError("the auto request was materialized before the policy saw it")
    return desired


def test_an_auto_request_is_sized_by_the_policy_not_by_the_host_cpu_count(monkeypatch):
    """Leave auto requests to the policy: cpu_count ignores the affinity mask and cgroup quota."""
    policy = _policy_or_skip()
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    monkeypatch.setattr(policy, "multiprocessing_start_method", lambda: "fork")
    monkeypatch.setattr(policy, "_usable_cpus", lambda: 2)
    monkeypatch.setattr(policy, "_affordable_workers", lambda: 64)
    monkeypatch.setattr(hw, "safe_num_proc", _unexpected_auto_sizing)

    assert hw.dataset_map_num_proc() == 2


def test_studio_caps_still_apply_to_a_policy_chosen_count(monkeypatch):
    """The multi-GPU fork-deadlock cap is knowledge the policy does not have."""
    policy = _policy_or_skip()
    _patch_device(monkeypatch, hw.DeviceType.CUDA, visible_gpus = 2)
    monkeypatch.setattr(policy, "multiprocessing_start_method", lambda: "fork")
    monkeypatch.setattr(policy, "_usable_cpus", lambda: 64)
    monkeypatch.setattr(policy, "_affordable_workers", lambda: 64)
    assert hw.dataset_map_num_proc() == 4


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_the_override_is_honoured_on_spawn_platforms(monkeypatch, platform):
    """An explicit UNSLOTH_DATASET_NUM_PROC overrides the spawn-platform veto, or the remedy is a no-op."""
    policy = _policy_or_skip()
    policy.reset_warning_state()
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", "2")
    assert hw.dataset_map_num_proc(8) == 2

    monkeypatch.delenv("UNSLOTH_DATASET_NUM_PROC")
    assert hw.dataset_map_num_proc(8) is None


def test_the_override_is_not_capped_by_the_studio_heuristics(monkeypatch):
    """Uncapped by contract, including by the multi-GPU cap Unsloth adds after."""
    policy = _policy_or_skip()
    policy.reset_warning_state()
    _patch_device(monkeypatch, hw.DeviceType.CUDA, visible_gpus = 4)
    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", "16")
    assert hw.dataset_map_num_proc(2) == 16


def test_an_older_zoo_falls_back_to_the_unsloth_copy(monkeypatch):
    """Use unsloth's copy only if already imported; importing it here would patch torch."""
    import builtins

    calls = []
    stub = types.ModuleType("unsloth.dataset_num_proc")
    stub.NUM_PROC_ENV_VAR = "UNSLOTH_DATASET_NUM_PROC"
    stub.get_dataset_num_proc = lambda desired = None, *, serial_as_none = True: (
        calls.append((desired, serial_as_none)) or 3
    )
    package = types.ModuleType("unsloth")
    package.dataset_num_proc = stub

    monkeypatch.setitem(sys.modules, "unsloth", package)
    monkeypatch.setitem(sys.modules, "unsloth.dataset_num_proc", stub)

    real_import = builtins.__import__

    def _no_zoo_policy(name, *args, **kwargs):
        if name == "unsloth_zoo.dataset_num_proc":
            raise ImportError("older unsloth_zoo")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_zoo_policy)
    _patch_device(monkeypatch, hw.DeviceType.CPU)

    assert hw.dataset_map_num_proc(8) == 3
    assert calls == [(8, True)], calls


def test_no_policy_anywhere_keeps_the_previous_behaviour(monkeypatch):
    """Neither module importable: the pre-policy Unsloth count, not a crash."""
    import builtins

    monkeypatch.delitem(sys.modules, "unsloth", raising = False)
    real_import = builtins.__import__

    def _no_policy(name, *args, **kwargs):
        if name.endswith("dataset_num_proc"):
            raise ImportError("no policy here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_policy)
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    assert hw.dataset_map_num_proc(4) == 4


def test_the_override_is_honoured_after_xpu_init(monkeypatch):
    """The override is honoured after XPU init too: the user has accepted the fork risk it names."""
    policy = _policy_or_skip()
    policy.reset_warning_state()
    _patch_device(monkeypatch, hw.DeviceType.XPU)
    _patch_runtime(monkeypatch, "xpu", is_initialized = True)

    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", "2")
    assert hw.dataset_map_num_proc(8) == 2

    monkeypatch.delenv("UNSLOTH_DATASET_NUM_PROC")
    assert hw.dataset_map_num_proc(8) is None
    assert hw.dataset_map_num_proc(8, serial_as_none = False) == 1


@pytest.mark.parametrize("raw", ["-1", "not-a-number"])
def test_an_ignored_override_does_not_skip_the_studio_caps(monkeypatch, raw):
    """Ignored override values must not skip the multi-GPU fork-deadlock cap; only valid ones do."""
    policy = _policy_or_skip()
    policy.reset_warning_state()
    monkeypatch.setattr(policy, "multiprocessing_start_method", lambda: "fork")
    monkeypatch.setattr(policy, "_affordable_workers", lambda: 64)
    _patch_device(monkeypatch, hw.DeviceType.CUDA, visible_gpus = 4)

    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", raw)
    assert hw.dataset_map_num_proc(16) == 4


def test_the_trainer_leaves_the_ordinary_case_to_the_policy():
    """Pass None, not a cpu_count-derived number: any integer skips the policy's CPU affinity and quota."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py").read_text(
        encoding = "utf-8"
    )

    calls = [
        value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant)
        and key.value == "dataset_num_proc"
        and isinstance(value, ast.Call)
        and ast.unparse(value.func) == "dataset_map_num_proc"
    ]
    assert calls, "the SFTConfig dataset_num_proc entry moved; re-check this guard"
    for call in calls:
        requested = ast.unparse(call.args[0])
        assert "cpu_count" not in requested, (
            "the ordinary case is being sized from the host CPU count before the "
            f"policy can see it: {requested}"
        )
        assert requested.rstrip().endswith("None"), requested
