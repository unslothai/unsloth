# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``dataset_map_num_proc()``.

``None``, not ``1``, is the disable sentinel: on ``datasets`` >= 4.1 (Unsloth pins
4.3.0) ``map()`` takes the pool branch for any ``num_proc >= 1``, while 3.x runs
``1`` in-process, so that split is asserted per installed release.

There is deliberately no CUDA-initialized guard; see ``dataset_map_num_proc``'s
docstring. The XPU guard is pre-existing and covered so it cannot regress.

The device probes, the platform and torch itself are substituted, so this runs
identically on a GPU box, a CPU-only Linux runner, and the macOS and Windows
legs where the function short-circuits to None.
"""

from __future__ import annotations

import sys
import types

import pytest

import utils.hardware.hardware as hw

try:
    # Import before the fixture below spoofs sys.platform: multiprocess picks its
    # contexts at import time, so a Windows runner would get POSIX fork contexts
    # and the pool would then die reaching for os.WNOHANG.
    import multiprocess  # noqa: F401
except ImportError:
    pass

# The real one, read before anything can lie about it.
_HOST_PLATFORM = sys.platform


@pytest.fixture(autouse = True)
def _fork_platform(monkeypatch):
    """Pin a platform where workers are possible.

    ``dataset_map_num_proc`` returns None outright on win32 and darwin, so every
    assertion expecting a count is really about Linux. The parametrised platform
    test sets its own value after this, which wins.
    """
    monkeypatch.setattr(sys, "platform", "linux")


@pytest.fixture(autouse = True)
def _memory_headroom(monkeypatch):
    """Pin the memory ceiling the shared policy applies.

    Every count these tests assert is bounded by free RAM, so on a small runner
    ``dataset_map_num_proc(4) == 4`` quietly becomes a test of the clamp. The
    cases that are about the clamp pin their own value afterwards, which wins.
    """
    policy = hw._shared_policy()
    if policy is None:
        return
    monkeypatch.setattr(policy, "_affordable_workers", lambda: 64)


def _require_fork(multiprocess):
    """Skip when this host cannot fork.

    The two tests below build real workers to observe whether ``datasets`` takes
    its pool branch -- a claim about the num_proc guard, not about any start
    method, and under spawn inside pytest the pool fails for unrelated reasons
    (WinError 10038, missing os.WNOHANG).
    """
    if _HOST_PLATFORM == "win32" or "fork" not in multiprocess.get_all_start_methods():
        pytest.skip("needs fork to build a real worker pool")


def _policy_or_skip():
    """The policy module the production code will actually consult.

    Not `importorskip("unsloth_zoo.dataset_num_proc")`: that module only exists
    in the companion zoo PR, so on CI -- which clones unsloth_zoo main -- every
    case that reached the policy skipped, and the ones that stayed exercised the
    pre-PR path. `_shared_policy` finds this repo's own fallback copy, and
    returning the same object it uses is also what makes monkeypatching it bite.
    """
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
    """The real torch, or a stand-in when the runner has none.

    ``dataset_map_num_proc`` imports torch to read ``<device>.is_initialized()``
    and reads an ImportError as "runtime not touched yet", which on a torch-less
    runner would turn the XPU guard into a no-op.
    """
    try:
        import torch
        return torch
    except ImportError:
        stub = types.ModuleType("torch")
        monkeypatch.setitem(sys.modules, "torch", stub)
        return stub


def _patch_runtime(monkeypatch, name, *, is_initialized):
    """Install a fake ``torch.<name>`` whose is_initialized() we control.

    ``is_initialized`` may be a bool or a callable that raises, to model a probe
    that fails rather than answering.
    """
    torch = _torch_module(monkeypatch)

    if callable(is_initialized):
        probe = is_initialized
    else:
        probe = lambda: is_initialized  # noqa: E731

    monkeypatch.setattr(torch, name, types.SimpleNamespace(is_initialized = probe), raising = False)


# ---------- CUDA: initialization must NOT disable workers ----------


def test_dataset_map_num_proc_parallelizes_on_initialized_cuda(monkeypatch):
    """An initialized CUDA context must not disable dataset workers.

    The map child only runs the tokenizer, and 300 forced-fork map() runs on an
    initialized context produced no failures. Since detect_hardware() always
    initializes CUDA, a guard would serialize every CUDA run. Pinned so it is
    not added back without new evidence.
    """
    _patch_device(monkeypatch, hw.DeviceType.CUDA)
    _patch_runtime(monkeypatch, "cuda", is_initialized = True)
    assert hw.dataset_map_num_proc(4) == 4


def test_dataset_map_num_proc_cuda_respects_multi_gpu_cap(monkeypatch):
    # CUDA still routes through safe_num_proc, which caps to 4 on multi-GPU.
    _patch_device(monkeypatch, hw.DeviceType.CUDA)
    _patch_runtime(monkeypatch, "cuda", is_initialized = True)
    monkeypatch.setattr(hw, "get_visible_gpu_count", lambda: 2)
    assert hw.dataset_map_num_proc(16) == 4


# ---------- XPU (regression guard: the pre-existing behaviour must not move) ----------


def test_dataset_map_num_proc_none_after_xpu_init(monkeypatch):
    _patch_device(monkeypatch, hw.DeviceType.XPU)
    _patch_runtime(monkeypatch, "xpu", is_initialized = True)
    assert hw.dataset_map_num_proc(4) is None


def test_dataset_map_num_proc_parallel_before_xpu_init(monkeypatch):
    _patch_device(monkeypatch, hw.DeviceType.XPU)
    _patch_runtime(monkeypatch, "xpu", is_initialized = False)
    assert hw.dataset_map_num_proc(4) == 4


# ---------- platform ----------


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_dataset_map_num_proc_none_on_spawn_platforms(monkeypatch, platform):
    # Must be None and never 1: datasets >= 4.1 builds a Pool(1) for num_proc=1.
    monkeypatch.setattr(sys, "platform", platform)
    assert hw.dataset_map_num_proc(4) is None


def test_dataset_map_num_proc_cpu_host_parallelizes(monkeypatch):
    # No accelerator to corrupt, so preprocessing may use workers.
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    assert hw.dataset_map_num_proc(4) == 4


# ---------- the value actually reaches datasets as "no pool" ----------


def test_none_builds_no_pool_but_a_count_does(monkeypatch):
    """The disable sentinel must actually reach ``datasets`` as "no pool".

    The property the whole module rests on, so assert it against the installed
    ``datasets`` rather than trusting the docstring.
    """
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

    # Control: the spy is live, so the assertion above means something.
    dataset.map(_count, batched = True, num_proc = 2)
    assert len(pools_built) == 1, "Pool spy never fired; the no-pool check is vacuous"


def test_num_proc_one_is_not_a_disable_sentinel():
    """Pin the reason ``dataset_map_num_proc`` returns ``None`` and never ``1``.

    ``datasets`` 3.x runs ``num_proc=1`` in-process; 4.x (Unsloth pins 4.3.0)
    builds a ``Pool(1)``. Only ``None`` is in-process on both, so assert per
    installed version rather than hard-coding one.
    """
    datasets = pytest.importorskip("datasets")
    multiprocess = pytest.importorskip("multiprocess")
    _require_fork(multiprocess)
    from packaging.version import Version

    # Counted at the Pool class rather than at datasets.arrow_dataset.Pool:
    # 3.x and 4.x do `from multiprocess import Pool`, but 5.x calls `mp.Pool()`
    # and a spawn context instead, so the module attribute is simply absent
    # there and patching it is an AttributeError. Every one of those routes
    # constructs this class.
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


# ---------- the value that reaches Dataset.map is bounded like any other ----------


def test_a_low_memory_host_gets_no_workers(monkeypatch):
    """format_conversion.py and chat_templates.py call this directly.

    Without the shared policy a 2GB container with eight cores still handed eight
    tokenizer workers to Dataset.map, which is the OOM the policy exists to stop.
    """
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
    # The cap's log line used to advertise this on a path that never read it.
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    policy = _policy_or_skip()
    monkeypatch.setattr(policy, "multiprocessing_start_method", lambda: "fork")
    policy.reset_warning_state()

    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", "2")
    assert hw.dataset_map_num_proc(8) == 2

    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", "0")
    assert hw.dataset_map_num_proc(8) is None


def test_an_older_unsloth_zoo_keeps_the_previous_behaviour(monkeypatch):
    # The policy is a lazy, guarded import: hardware detection must not start
    # depending on the training package.
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
    """safe_num_proc returns an int >= 1, so it cannot express in-process.

    Naming the variable there told users to set something that path never read.
    """
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


# ---------- the config boundary spells "in-process" as 1, not None ----------
#
# trainer.py writes this value into SFTConfig.dataset_num_proc rather than
# handing it to map(). Every downstream reader treats a config None as
# "auto-size me", so a serial request stored as None comes back out as a full
# worker set: dataset_map_num_proc(1) -> None -> SFTConfig -> 8.


def test_a_serial_request_survives_the_config_round_trip(monkeypatch):
    """The audio paths ask for 1; the config layer must still see a request for 1."""
    # The map-site half of this needs the policy: without it
    # _bounded_by_the_shared_policy returns the count unchanged by design, so a
    # runner with no unsloth_zoo reads 1 rather than None.
    _policy_or_skip()
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    assert hw.dataset_map_num_proc(1, serial_as_none = False) == 1
    # The map-site default is what turns it back into "no pool" at the call.
    assert hw.dataset_map_num_proc(1) is None


def test_the_config_value_is_still_in_process_after_the_layer_reads_it(monkeypatch):
    """End to end: what Unsloth stores, read back the way the SFT config layer reads it."""
    policy = _policy_or_skip()
    _patch_device(monkeypatch, hw.DeviceType.CPU)

    stored = hw.dataset_map_num_proc(1, serial_as_none = False)
    # rl.py generates serial_as_none = False for sft_trainer, then the map-site
    # rewrite converts the config value for the actual Dataset.map call.
    from_config = policy.get_dataset_num_proc(stored, serial_as_none = False)
    at_the_map_site = policy.get_dataset_num_proc(from_config)
    assert (stored, from_config, at_the_map_site) == (1, 1, None)


def test_xpu_initialized_stays_serial_through_a_config(monkeypatch):
    """The one guard where a config None is actively dangerous.

    Unlike the spawn platforms, forking still works here, so an auto-sizer
    reading a config None would fork the corrupted Level-Zero context this
    guard exists to protect.
    """
    _patch_device(monkeypatch, hw.DeviceType.XPU)
    _patch_runtime(monkeypatch, "xpu", is_initialized = True)
    assert hw.dataset_map_num_proc(4, serial_as_none = False) == 1
    assert hw.dataset_map_num_proc(4) is None


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_spawn_platforms_keep_none_at_either_layer(monkeypatch, platform):
    """None is safe to store here: no reader can inflate it.

    Workers are unusable on a spawn platform, so every auto-sizer vetoes. A 1
    would be worse: only the SFT map site is rewritten, so DPO/KTO/CPO/ORPO and
    friends would hand that 1 to Dataset.map and get a Pool(1) whose child
    re-imports the user's __main__ (#3211 / #3397).
    """
    monkeypatch.setattr(sys, "platform", platform)
    assert hw.dataset_map_num_proc(4, serial_as_none = False) is None


def test_every_other_caller_keeps_the_map_site_default(monkeypatch):
    """Only the config boundary opts in; the seven map-site callers must not."""
    _policy_or_skip()
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    assert hw.dataset_map_num_proc(1) is None
    assert hw.dataset_map_num_proc(4) == 4


def test_the_trainer_config_asks_for_the_config_sentinel():
    """The call that builds SFTConfig must pass serial_as_none = False.

    Dropping it is silent: the run still trains, just with a worker set where
    the audio paths asked for none.
    """
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


# ---------- the request reaches the policy as the caller wrote it ----------


def _unexpected_auto_sizing(desired = None):
    if desired is None:
        raise AssertionError("the auto request was materialized before the policy saw it")
    return desired


def test_an_auto_request_is_sized_by_the_policy_not_by_the_host_cpu_count(monkeypatch):
    """``safe_num_proc(None)`` reads ``os.cpu_count()``; the policy reads this process.

    Materializing the auto request before the policy saw it hid the affinity
    mask and the cgroup quota, so a 2-core container on a 64-core box asked for
    ``cpu_count // 3`` workers and was bounded only by memory.
    """
    policy = _policy_or_skip()
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    monkeypatch.setattr(policy, "multiprocessing_start_method", lambda: "fork")
    monkeypatch.setattr(policy, "_usable_cpus", lambda: 2)
    monkeypatch.setattr(policy, "_affordable_workers", lambda: 64)
    monkeypatch.setattr(hw, "safe_num_proc", _unexpected_auto_sizing)

    # min(max(2 // 2, 2), cap) = 2, against os.cpu_count() // 3 on the host.
    assert hw.dataset_map_num_proc() == 2


def test_studio_caps_still_apply_to_a_policy_chosen_count(monkeypatch):
    """The multi-GPU fork-deadlock cap is knowledge the policy does not have."""
    policy = _policy_or_skip()
    _patch_device(monkeypatch, hw.DeviceType.CUDA, visible_gpus = 2)
    monkeypatch.setattr(policy, "multiprocessing_start_method", lambda: "fork")
    monkeypatch.setattr(policy, "_usable_cpus", lambda: 64)
    monkeypatch.setattr(policy, "_affordable_workers", lambda: 64)
    assert hw.dataset_map_num_proc() == 4


# ---------- the escape hatch ----------


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_the_override_is_honoured_on_spawn_platforms(monkeypatch, platform):
    """The policy calls it unvetoed, so the platform veto must not swallow it.

    A user who has read the dead-worker message and set this has accepted spawn
    workers; silently ignoring it makes the documented remedy a no-op.
    """
    policy = _policy_or_skip()
    policy.reset_warning_state()
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", "2")
    assert hw.dataset_map_num_proc(8) == 2

    # Unset, the veto stands.
    monkeypatch.delenv("UNSLOTH_DATASET_NUM_PROC")
    assert hw.dataset_map_num_proc(8) is None


def test_the_override_is_not_capped_by_the_studio_heuristics(monkeypatch):
    """Uncapped by contract, including by the multi-GPU cap Unsloth adds after."""
    policy = _policy_or_skip()
    policy.reset_warning_state()
    _patch_device(monkeypatch, hw.DeviceType.CUDA, visible_gpus = 4)
    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", "16")
    assert hw.dataset_map_num_proc(2) == 16


# ---------- the local fallback copy ----------


def test_an_older_zoo_falls_back_to_the_unsloth_copy(monkeypatch):
    """unsloth.dataset_num_proc is the same policy; Unsloth should use it too.

    Only when unsloth is already imported: importing it from here would make
    hardware detection patch torch and pull in the model stack.
    """
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
    """Same contract as the spawn platforms: the hatch is unvetoed.

    The XPU guard exists because fork corrupts the Level-Zero context, but a
    user who has read the dead-worker message and set this has accepted that.
    """
    policy = _policy_or_skip()
    policy.reset_warning_state()
    _patch_device(monkeypatch, hw.DeviceType.XPU)
    _patch_runtime(monkeypatch, "xpu", is_initialized = True)

    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", "2")
    assert hw.dataset_map_num_proc(8) == 2

    # Unset, the veto stands at both layers.
    monkeypatch.delenv("UNSLOTH_DATASET_NUM_PROC")
    assert hw.dataset_map_num_proc(8) is None
    assert hw.dataset_map_num_proc(8, serial_as_none = False) == 1


@pytest.mark.parametrize("raw", ["-1", "not-a-number"])
def test_an_ignored_override_does_not_skip_the_studio_caps(monkeypatch, raw):
    """The policy warns and ignores these, so they are not the hatch.

    Treating any non-empty value as an active override let a typo skip the
    multi-GPU fork-deadlock cap while contributing nothing in its place.
    """
    policy = _policy_or_skip()
    policy.reset_warning_state()
    monkeypatch.setattr(policy, "multiprocessing_start_method", lambda: "fork")
    monkeypatch.setattr(policy, "_affordable_workers", lambda: 64)
    _patch_device(monkeypatch, hw.DeviceType.CUDA, visible_gpus = 4)

    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", raw)
    assert hw.dataset_map_num_proc(16) == 4


def test_the_trainer_leaves_the_ordinary_case_to_the_policy():
    """The non-audio branch must be None, not a host-derived count.

    ``get_dataset_num_proc`` reads any integer as an explicit request and skips
    its own auto path, which is the only one that consults this process's CPU
    affinity and cgroup quota. A count computed from ``os.cpu_count()`` here
    therefore hides the container from the policy.
    """
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
