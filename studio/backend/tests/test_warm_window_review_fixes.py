# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The warm window's remaining sharp edges, one test per edge.

Deferring the ML stack creates an interval between the socket binding and the
stack being importable. Several separate things went wrong in that interval, and
each is cheap to assert once it is named:

  * the generation counter advanced on the cached path, so ordinary get_device()
    traffic looked like a hardware re-detection;
  * a failed forced re-detect left half a verdict published;
  * building the orchestrator on the warm thread turned its ranking fetch into an
    unprompted outbound request at boot;
  * two sync helpers reached the inference singleton from the event-loop thread
    ahead of the offloads meant to cover them;
  * the torch-warm kill switch also disabled MLX self-heal;
  * purging a half-imported package raced a request retrying the same import;
  * the post-warm worker outlived its lifespan, then starved the next one;
  * /api/health published a verdict read mid-re-detect, and treated a torn
    event-set/DEVICE-None state as a settled answer.
"""

from __future__ import annotations

import ast
import builtins
import sys
import threading
from unittest import mock
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.hardware.hardware as hw  # noqa: E402


class _NeverStarts:
    """Stand-in for threading.Thread that records the call and runs nothing."""

    def start(self) -> None:
        return None


# ---------------------------------------------------------------- generation


def test_the_cached_path_does_not_look_like_a_redetect(monkeypatch):
    """get_device() on a warm process must not advance the generation counter.

    InferenceOrchestrator._refresh_static_models_if_stale() reads a change as
    "hardware was re-detected" and rebuilds the curated defaults, so counting
    cache hits made any GPU or export helper trigger a rebuild plus a false log.
    """
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU, raising = False)
    before = hw.DETECTION_GENERATION
    for _ in range(5):
        hw.ensure_hardware_detected()
    assert hw.DETECTION_GENERATION == before, (
        "the cached path advanced DETECTION_GENERATION; every get_device() would "
        "make the next model list rebuild its curated defaults"
    )


def test_a_real_detection_still_advances_the_generation(monkeypatch):
    """Negative control: the counter must still move when detection runs."""
    monkeypatch.setattr(hw, "DEVICE", None, raising = False)
    calls: list[int] = []

    def _fake_locked():
        calls.append(1)
        hw.DEVICE = hw.DeviceType.CPU
        return hw.DEVICE

    monkeypatch.setattr(hw, "_detect_hardware_locked", _fake_locked)
    before = hw.DETECTION_GENERATION
    hw.ensure_hardware_detected()
    assert calls == [1]
    assert hw.DETECTION_GENERATION == before + 1


def test_the_completion_event_is_published_on_the_cached_path_too(monkeypatch):
    """The event is not the counter: a late waiter must still find it set."""
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU, raising = False)
    hw.DETECTION_COMPLETE.clear()
    try:
        hw.ensure_hardware_detected()
        assert hw.DETECTION_COMPLETE.is_set()
    finally:
        hw.DETECTION_COMPLETE.set()


# ------------------------------------------------------- failed re-detection


def test_a_failed_redetect_restores_the_whole_published_verdict(monkeypatch):
    """A raise must not leave a half-written verdict as what health serves.

    _detect_hardware_locked() resets CHAT_ONLY / CHAT_ONLY_REASON / IS_ROCM on
    entry, and the MLX autorepair path catches the exception, so without a restore
    the reason is gone -- and the sidebar recovery poll only continues while it
    reads "mlx_unavailable".
    """
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU, raising = False)
    monkeypatch.setattr(hw, "CHAT_ONLY", True, raising = False)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "mlx_unavailable", raising = False)
    monkeypatch.setattr(hw, "IS_ROCM", True, raising = False)

    def _boom():
        # Exactly what the real body does before it can fail.
        hw.CHAT_ONLY = True
        hw.CHAT_ONLY_REASON = None
        hw.IS_ROCM = False
        hw.DEVICE = hw.DeviceType.MLX
        raise RuntimeError("probe blew up mid-pass")

    monkeypatch.setattr(hw, "_detect_hardware_locked", _boom)
    hw.DETECTION_COMPLETE.set()

    with pytest.raises(RuntimeError):
        hw.detect_hardware()

    assert hw.DEVICE is hw.DeviceType.CPU
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable", (
        "the failed re-detect dropped the chat_only reason; the frontend recovery "
        "poll stops as soon as it reads a reply without it"
    )
    assert hw.IS_ROCM is True
    assert hw.DETECTION_COMPLETE.is_set()


def test_a_successful_redetect_publishes_the_new_verdict(monkeypatch):
    """Negative control: the restore must not undo a pass that worked."""
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU, raising = False)
    monkeypatch.setattr(hw, "CHAT_ONLY", True, raising = False)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "mlx_unavailable", raising = False)

    def _ok():
        hw.CHAT_ONLY = False
        hw.CHAT_ONLY_REASON = None
        hw.DEVICE = hw.DeviceType.MLX
        return hw.DEVICE

    monkeypatch.setattr(hw, "_detect_hardware_locked", _ok)
    before = hw.DETECTION_GENERATION
    assert hw.detect_hardware() is hw.DeviceType.MLX
    assert hw.CHAT_ONLY is False
    assert hw.CHAT_ONLY_REASON is None
    assert hw.DETECTION_GENERATION == before + 1


# ------------------------------------------------------------- boot silence


def test_building_the_orchestrator_makes_no_outbound_request():
    """Construction moved onto the warm thread, so it must not fetch anything.

    Starting the ranking fetch from __init__ meant every boot reached
    huggingface.co before anyone signed in, on a host that may never serve a
    request. Asserted on the source: importing the orchestrator here would pull
    the whole inference stack.
    """
    tree = ast.parse(
        (_BACKEND / "core" / "inference" / "orchestrator.py").read_text(encoding = "utf-8")
    )
    # Scope to the class: the module defines more than one __init__, and ast.walk
    # would otherwise assert on a different constructor.
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "InferenceOrchestrator"
    )
    init = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    offenders = [
        sub
        for sub in ast.walk(init)
        if isinstance(sub, ast.Attribute) and sub.attr == "_fetch_top_models"
    ]
    assert not offenders, (
        "the orchestrator constructor references _fetch_top_models again; building "
        "it on the warm thread then reaches huggingface.co on every boot, before "
        "anyone signs in"
    )


def test_the_ranking_fetch_is_started_by_the_first_reader():
    """...but it must still be reachable, or the extra choices never load."""
    tree = ast.parse(
        (_BACKEND / "core" / "inference" / "orchestrator.py").read_text(encoding = "utf-8")
    )
    prop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "default_models"
    )
    called = {
        sub.func.attr
        for sub in ast.walk(prop)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
    }
    assert "_start_top_models_fetch" in called


@pytest.mark.parametrize(
    "env",
    [
        {"HF_HUB_OFFLINE": "1"},
        {"HF_HUB_OFFLINE": "true"},
        {"HF_HUB_OFFLINE": "on"},
        {"TRANSFORMERS_OFFLINE": "1"},
    ],
)
def test_the_ranking_fetch_starts_no_thread_when_offline(monkeypatch, env):
    """It is a raw httpx.get, so the offline variables do not reach it by themselves.

    Driven through the real method rather than the source: every spelling the
    rest of the backend accepts as offline has to leave the first model list
    network-silent, not just the one the guard happens to compare against.
    """
    from core.inference import orchestrator as orch

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    started = []
    monkeypatch.setattr(
        orch.threading,
        "Thread",
        lambda *a, **kw: started.append(kw.get("name")) or _NeverStarts(),
        raising = True,
    )

    instance = object.__new__(orch.InferenceOrchestrator)
    instance._top_models_started = False
    orch.InferenceOrchestrator._start_top_models_fetch(instance)

    assert not started, f"{env} still put up the top-models fetch thread"


def test_the_ranking_fetch_still_runs_when_online(monkeypatch):
    """Negative control: nothing above may have disabled the fetch outright."""
    from core.inference import orchestrator as orch

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)

    started = []
    monkeypatch.setattr(
        orch.threading,
        "Thread",
        lambda *a, **kw: started.append(kw.get("name")) or _NeverStarts(),
        raising = True,
    )

    instance = object.__new__(orch.InferenceOrchestrator)
    instance._top_models_started = False
    orch.InferenceOrchestrator._start_top_models_fetch(instance)

    assert started == ["top-models"], "an online host no longer fetches the ranking"


# ------------------------------------------------- offloads at the call site


def _async_offloaded_names(path: Path, function: str) -> set[str]:
    """Names this async function hands to asyncio.to_thread."""
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == function
    )
    out: set[str] = set()
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Call):
            continue
        target = sub.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
        if name != "to_thread" or not sub.args:
            continue
        first = sub.args[0]
        out.add(getattr(first, "id", None) or getattr(first, "attr", ""))
    return out


@pytest.mark.parametrize("function", ["_openai_catalog_objects", "openai_retrieve_model"])
def test_the_openai_model_listing_reaches_the_singleton_off_loop(function):
    """_openai_model_objects() is sync and calls get_inference_backend().

    Left inline, an early GET /v1/models held the event-loop thread for the rest
    of the torch import, and the offload further down that module could not help
    because the call had already happened.
    """
    path = _BACKEND / "routes" / "inference.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)}
    if function not in names:
        pytest.skip(f"{function} is not an async handler in this tree")
    assert "_openai_model_objects" in _async_offloaded_names(path, function)


def test_the_model_config_capability_block_runs_off_loop():
    """is_vision_model() reaches _detection_sets(), so it cannot run inline.

    Which nested helper carries it is main's business -- it wraps the whole
    resolution in one now -- so this pins the property instead of the name: the
    handler body itself must not call the probes, and whatever the worker runs
    must.
    """
    path = _BACKEND / "routes" / "models.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_model_config"
    )
    nested = {
        sub
        for helper in ast.walk(fn)
        if isinstance(helper, ast.FunctionDef)
        for sub in ast.walk(helper)
    }
    inline = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and getattr(sub.func, "id", None) in {"is_vision_model", "is_embedding_model"}
        and sub not in nested
    ]
    assert not inline, (
        "the capability detection block is inline again; is_vision_model() can "
        "import transformers on the event-loop thread"
    )
    offloaded = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "to_thread"
    ]
    assert offloaded, "nothing in the config handler goes to a worker thread any more"
    reached = {
        call.func.id
        for helper in ast.walk(fn)
        if isinstance(helper, ast.FunctionDef)
        for call in ast.walk(helper)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "is_vision_model" in reached, (
        "no offloaded helper reaches is_vision_model; the probe moved somewhere "
        "this test no longer covers"
    )


# ------------------------------------------------------------- kill switch


def test_the_torch_kill_switch_leaves_mlx_selfheal_running():
    """The switch is about torch; MLX autorepair has its own opt-out.

    Gating autorepair on it meant an Apple Silicon host with a broken MLX stack
    stayed chat-only for good, where before the change autorepair ran in the
    lifespan no matter what.
    """
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_post_warm_background_work"
    )
    body = fn.body

    def _index_of(predicate) -> int:
        for i, stmt in enumerate(body):
            if predicate(ast.dump(stmt)):
                return i
        return -1

    mlx_at = _index_of(lambda d: "start_mlx_autorepair_if_needed" in d)
    gate_at = _index_of(lambda d: "DISABLE_ENV_VAR" in d and "Return" in d)
    rag_at = _index_of(lambda d: "_warm_rag_embedder" in d)

    assert mlx_at >= 0 and gate_at >= 0 and rag_at >= 0
    assert mlx_at < gate_at, (
        "the kill-switch return precedes MLX autorepair, so setting "
        "UNSLOTH_STUDIO_DISABLE_TORCH_WARM leaves a broken MLX host chat-only"
    )
    assert gate_at < rag_at, (
        "the RAG warm is no longer gated; it pulls sentence-transformers and "
        "torch, which is what the kill switch exists to prevent"
    )


def test_the_purge_rechecks_before_touching_sys_modules():
    """A racing retry republishes the parent; do not strip modules under it."""
    src = (_BACKEND / "utils" / "torch_warmup.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "purge_partial_import"
    )
    # `package in sys.modules` is a Compare/In whose comparator is the sys.modules
    # attribute; a dumped-text search misses it (the dump spells it attr='modules').
    checks = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Compare)
        and any(isinstance(op, ast.In) for op in sub.ops)
        and any(
            isinstance(c, ast.Attribute)
            and c.attr == "modules"
            and getattr(c.value, "id", None) == "sys"
            for c in sub.comparators
        )
    ]
    assert len(checks) >= 3, (
        f"expected the entry guard plus a re-check before and during the pops, "
        f"found {len(checks)} membership tests"
    )


def test_the_purge_stops_when_the_parent_reappears(monkeypatch):
    """Behavioural half of the guard above, not just its shape."""
    import utils.torch_warmup as warmup

    popped: list[str] = []
    fake = {"pkg.a": object(), "pkg.b": object(), "pkg.c": object()}

    class _Watcher(dict):
        def pop(
            self,
            key,
            default = None,
        ):
            popped.append(key)
            # A retry publishes the parent right after the first pop.
            if len(popped) == 1:
                self["pkg"] = object()
            return super().pop(key, default)

    watcher = _Watcher(fake)
    monkeypatch.setattr(warmup.sys, "modules", watcher)
    monkeypatch.setattr(warmup, "_is_extension_module", lambda name: False)

    warmup.purge_partial_import("pkg")
    assert len(popped) == 1, f"kept purging after the parent came back: popped {popped}"


def test_the_purge_still_cleans_an_uncontested_failure(monkeypatch):
    """Negative control: the guards must not disable the purge outright."""
    import utils.torch_warmup as warmup

    fake = {"pkg.a": object(), "pkg.b": object()}
    monkeypatch.setattr(warmup.sys, "modules", fake)
    monkeypatch.setattr(warmup, "_is_extension_module", lambda name: False)

    removed = warmup.purge_partial_import("pkg")
    assert sorted(removed) == ["pkg.a", "pkg.b"]
    assert fake == {}


def test_threading_import_is_present_for_the_lazy_fetch():
    """The lazy start puts up its own thread; keep the import it needs."""
    src = (_BACKEND / "core" / "inference" / "orchestrator.py").read_text(encoding = "utf-8")
    assert "import threading" in src
    assert isinstance(threading.Lock(), type(threading.Lock()))


# --------------------------------------------------- post-warm thread lifetime


def test_shutdown_stands_the_post_warm_thread_down(monkeypatch):
    """Work must not start for an application that has already stopped.

    The worker spends its life parked in join_background_warm(), so a shutdown
    landing before the warm finishes cannot reach it any other way: it wakes up
    later and goes on to load the embedder, which can fall back to spawning a
    llama-server.
    """
    import main as main_mod

    ran: list[str] = []
    released = threading.Event()

    monkeypatch.setattr(main_mod, "join_background_warm", lambda *a, **k: released.wait(30))
    monkeypatch.setattr(main_mod, "_warm_rag_embedder", lambda: ran.append("rag"))
    import utils.mlx_repair as mlx_mod

    monkeypatch.setattr(mlx_mod, "start_mlx_autorepair_if_needed", lambda: ran.append("mlx"))

    assert main_mod._start_post_warm_thread() is True
    worker = main_mod._post_warm_thread
    try:
        # Shutdown while the worker is still inside the join.
        main_mod._stop_post_warm_thread()
    finally:
        released.set()
        worker.join(30)

    assert ran == [], (
        f"post-warm work ran after shutdown: {ran}. The RAG warm can load an "
        f"embedder or start a llama-server for a stopped application."
    )


def test_the_post_warm_thread_still_works_without_a_shutdown(monkeypatch):
    """Negative control: standing down must not become never running."""
    import main as main_mod
    import utils.mlx_repair as mlx_mod

    ran: list[str] = []
    monkeypatch.setattr(main_mod, "join_background_warm", lambda *a, **k: None)
    monkeypatch.setattr(mlx_mod, "start_mlx_autorepair_if_needed", lambda: ran.append("mlx"))
    monkeypatch.setattr(main_mod, "_warm_rag_embedder", lambda: ran.append("rag"))

    assert main_mod._start_post_warm_thread() is True
    main_mod._post_warm_thread.join(30)
    assert ran == ["mlx", "rag"]


def test_a_restart_gets_its_own_worker_while_the_old_one_is_parked(monkeypatch):
    """The handoff case: the replacement must do the work, not inherit a refusal.

    Refusing to start while the previous worker was alive meant a restart got
    nothing at all -- the old worker was parked in the join, so the start declined,
    and then the old worker read the shutdown and exited. A Mac with a broken MLX
    stack stayed chat-only for the whole restart.
    """
    import main as main_mod
    import utils.mlx_repair as mlx_mod

    ran: list[str] = []
    release_old = threading.Event()
    release_new = threading.Event()
    joins: list[int] = []

    def _join(*_a, **_k):
        joins.append(1)
        (release_old if len(joins) == 1 else release_new).wait(30)

    monkeypatch.setattr(main_mod, "join_background_warm", _join)
    monkeypatch.setattr(mlx_mod, "start_mlx_autorepair_if_needed", lambda: ran.append("mlx"))
    monkeypatch.setattr(main_mod, "_warm_rag_embedder", lambda: ran.append("rag"))

    # Lifespan 1 starts a worker, then shuts down while it is still parked.
    assert main_mod._start_post_warm_thread() is True
    old = main_mod._post_warm_thread
    main_mod._stop_post_warm_thread()

    # Lifespan 2 starts immediately, with the old worker still alive.
    assert (
        main_mod._start_post_warm_thread() is True
    ), "the restart was refused a worker because the retired one was still parked"
    new = main_mod._post_warm_thread
    assert new is not old

    release_old.set()
    old.join(30)
    release_new.set()
    new.join(30)

    assert ran == ["mlx", "rag"], f"the restarted lifespan did not get its deferred work: {ran}"


def test_only_the_current_generation_does_the_work(monkeypatch):
    """Two parked workers, one retired: exactly one must proceed."""
    import main as main_mod
    import utils.mlx_repair as mlx_mod

    ran: list[str] = []
    gate = threading.Event()
    monkeypatch.setattr(main_mod, "join_background_warm", lambda *a, **k: gate.wait(30))
    monkeypatch.setattr(mlx_mod, "start_mlx_autorepair_if_needed", lambda: ran.append("mlx"))
    monkeypatch.setattr(main_mod, "_warm_rag_embedder", lambda: ran.append("rag"))

    main_mod._start_post_warm_thread()
    first = main_mod._post_warm_thread
    main_mod._start_post_warm_thread()
    second = main_mod._post_warm_thread

    gate.set()
    first.join(30)
    second.join(30)

    assert ran == ["mlx", "rag"], f"expected one worker to do the work once, got {ran}"


def test_shutdown_does_not_wait_for_the_post_warm_thread():
    """Retiring must not join: a join would hold shutdown for the rest of the ML
    stack import, which is the stall this path exists to avoid."""
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_stop_post_warm_thread"
    )
    joins = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call) and getattr(sub.func, "attr", None) == "join"
    ]
    assert not joins, "shutdown joins the post-warm thread; that can block for the import"


def test_the_lifespan_stops_the_post_warm_thread_on_shutdown():
    """Guard the wiring: the signal has to actually be sent."""
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )
    called = {
        sub.func.id
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
    }
    assert "_stop_post_warm_thread" in called
    assert "_start_post_warm_thread" in called


# ------------------------------------------------- authoritative health verdict


def test_health_will_not_publish_a_verdict_mid_redetect(monkeypatch):
    """A forced re-detect must not be reported as a settled answer.

    frontend/src/config/env.ts caches the first reply carrying device_type as
    authoritative, and the sidebar recovery poll only continues while it reads
    chat_only_reason == "mlx_unavailable". One chat-only reply with a null reason,
    taken mid-pass, therefore hides Train for the rest of the SPA session.
    """
    import main as main_mod

    hw_mod = main_mod._hw_module
    monkeypatch.setattr(hw_mod, "DEVICE", hw_mod.DeviceType.CPU, raising = False)
    monkeypatch.setattr(hw_mod, "CHAT_ONLY", True, raising = False)
    monkeypatch.setattr(hw_mod, "CHAT_ONLY_REASON", None, raising = False)
    hw_mod.DETECTION_COMPLETE.clear()
    try:
        assert (
            main_mod._hardware_snapshot() is None
        ), "a cleared completion event still produced a publishable verdict"
    finally:
        hw_mod.DETECTION_COMPLETE.set()


def test_health_snapshot_rejects_a_torn_read(monkeypatch):
    """Event set but DEVICE gone is the shutdown/detector race, not an answer."""
    import main as main_mod

    hw_mod = main_mod._hw_module
    monkeypatch.setattr(hw_mod, "DEVICE", None, raising = False)
    hw_mod.DETECTION_COMPLETE.set()
    assert main_mod._hardware_snapshot() is None


def test_health_snapshot_returns_a_settled_verdict(monkeypatch):
    """Negative control: a settled verdict must still be publishable."""
    import main as main_mod

    hw_mod = main_mod._hw_module
    monkeypatch.setattr(hw_mod, "DEVICE", hw_mod.DeviceType.CPU, raising = False)
    monkeypatch.setattr(hw_mod, "CHAT_ONLY", True, raising = False)
    monkeypatch.setattr(hw_mod, "CHAT_ONLY_REASON", "mlx_unavailable", raising = False)
    hw_mod.DETECTION_COMPLETE.set()
    assert main_mod._hardware_snapshot() == (True, "mlx_unavailable")


def test_health_rereads_the_verdict_after_authentication():
    """The bearer check is an await, so the pre-auth answer must be revalidated."""
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "health_check"
    )
    snapshots = [
        sub.lineno
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == "_hardware_snapshot"
    ]
    assert len(snapshots) >= 2, (
        f"health_check takes {len(snapshots)} hardware snapshot(s); the verdict "
        f"published to an authed caller must be re-read after the auth await"
    )
    awaits = [sub.lineno for sub in ast.walk(fn) if isinstance(sub, ast.Await)]
    assert any(
        any(a > snapshots[0] and a < s for a in awaits) for s in snapshots[1:]
    ), "the second snapshot does not come after an await, so it revalidates nothing"


def test_detection_wait_requires_a_device_not_just_the_event():
    """Event-set-with-DEVICE-None must send the caller to a fresh detection."""
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_await_hardware_detection"
    )
    # The docstring names DEVICE, so match the executable form (DEVICE compared to
    # None); a dumped-text search would pass on prose.
    device_tests = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Compare)
        and isinstance(sub.left, ast.Attribute)
        and sub.left.attr == "DEVICE"
        and any(isinstance(op, (ast.IsNot, ast.Is)) for op in sub.ops)
    ]
    assert len(device_tests) >= 2, (
        f"found {len(device_tests)} DEVICE comparison(s); both the fast path and "
        f"the poll loop must require a device, or a torn event-set/DEVICE-None "
        f"state is reported as detected and nothing re-detects"
    )


# ------------------------------------------------- cached-model delete guard


def test_the_delete_guard_runs_off_the_event_loop():
    """Both load-state guards must go to a worker, in one hop.

    _inference_backend_blocks_delete() is sync and reaches
    get_inference_backend(), whose cold build waits on hardware detection. Inline
    in the coroutine, an authed DELETE /api/models/delete-cached arriving during
    the warm held the event-loop thread for the rest of the torch import.
    """
    path = _BACKEND / "hub" / "services" / "models" / "deletion.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "delete_cached_model_response"
    )

    # The guards belong to the nested helper handed to the worker, not the
    # coroutine body.
    nested = {node.name for node in ast.walk(fn) if isinstance(node, ast.FunctionDef)}
    assert "_load_state_blocks_delete" in nested, (
        "the load-state guards are called inline again; get_inference_backend() "
        "then blocks the event loop for the remaining torch import"
    )

    offloaded = set()
    for sub in ast.walk(fn):
        if (
            isinstance(sub, ast.Call)
            and getattr(sub.func, "attr", None) == "to_thread"
            and sub.args
        ):
            offloaded.add(getattr(sub.args[0], "id", None))
    assert "_load_state_blocks_delete" in offloaded


def test_the_delete_guard_keeps_its_short_circuit_and_fail_closed():
    """The offload must not change what the guard decides.

    The `or` matters: when a GGUF model is loaded the first guard answers and the
    second never runs. And an unreadable load state must still raise rather than
    fall through to unlinking weights under a live process.
    """
    path = _BACKEND / "hub" / "services" / "models" / "deletion.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_load_state_blocks_delete"
    )
    bool_ops = [sub for sub in ast.walk(helper) if isinstance(sub, ast.BoolOp)]
    assert bool_ops and isinstance(bool_ops[0].op, ast.Or), (
        "the guard no longer short-circuits; the inference lookup would run even "
        "when llama.cpp has already answered"
    )

    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "delete_cached_model_response"
    )

    # Bind to the specific Try, not any Try whose dump mentions to_thread: a
    # decoy try/to_thread/raise elsewhere in the handler satisfied that, so this
    # passed with the real fail-closed guard deleted.
    def _offloads_the_delete_guard(node: ast.Try) -> bool:
        return any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "to_thread"
            and any(
                isinstance(a, ast.Name) and a.id == "_load_state_blocks_delete" for a in call.args
            )
            for stmt in node.body
            for call in ast.walk(stmt)
        )

    guarded = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Try) and sub.handlers and _offloads_the_delete_guard(sub)
    ]
    assert guarded, "the offloaded guard is no longer inside a try/except"
    # The raise must be in a handler. In the body it is whatever the guard itself
    # raises on success, which says nothing about an unreadable load state.
    raises = [
        sub
        for handler in guarded[0].handlers
        for sub in ast.walk(handler)
        if isinstance(sub, ast.Raise)
    ]
    assert raises, "an unreadable load state no longer raises; delete would proceed"


# ------------------------------------------------- the kill switch and health
def test_health_does_not_kick_detection_when_the_warm_is_off(monkeypatch):
    """The torch-warm switch has to survive the automatic health probe.

    The desktop preflight and the frontend's first fetch both hit /api/health
    without being asked to, so a kick from there imports torch on exactly the
    hosts whose owner set the switch because that import is broken or expensive.
    """
    import asyncio as _asyncio

    import main as main_mod

    kicks = []
    monkeypatch.setattr(
        main_mod, "start_background_detection", lambda: kicks.append(1), raising = True
    )
    monkeypatch.setenv(main_mod.DISABLE_ENV_VAR, "1")

    hw_mod = main_mod._hw_module
    monkeypatch.setattr(hw_mod, "DEVICE", None, raising = False)
    was_complete = hw_mod.DETECTION_COMPLETE.is_set()
    hw_mod.DETECTION_COMPLETE.clear()
    try:
        detected = _asyncio.run(main_mod._await_hardware_detection(0.2))
    finally:
        if was_complete:
            hw_mod.DETECTION_COMPLETE.set()

    assert not kicks, "the health path started a torch import the kill switch forbids"
    assert detected is False, "health claimed a verdict nothing had measured"


def test_health_still_kicks_detection_when_the_warm_is_on(monkeypatch):
    """Negative control: the switch is what stops the kick, not the fix itself."""
    import asyncio as _asyncio

    import main as main_mod

    kicks = []
    monkeypatch.setattr(
        main_mod, "start_background_detection", lambda: kicks.append(1), raising = True
    )
    monkeypatch.delenv(main_mod.DISABLE_ENV_VAR, raising = False)

    hw_mod = main_mod._hw_module
    monkeypatch.setattr(hw_mod, "DEVICE", None, raising = False)
    was_complete = hw_mod.DETECTION_COMPLETE.is_set()
    hw_mod.DETECTION_COMPLETE.clear()
    try:
        _asyncio.run(main_mod._await_hardware_detection(0.05))
    finally:
        if was_complete:
            hw_mod.DETECTION_COMPLETE.set()

    assert kicks, "nothing filled DEVICE in; health would report detecting forever"


def test_the_switch_still_reports_a_verdict_it_already_has(monkeypatch):
    """Not kicking is not the same as not answering.

    Once something else has detected -- the first training or export call -- the
    switch must not keep health on its provisional answer.
    """
    import asyncio as _asyncio

    import main as main_mod

    monkeypatch.setenv(main_mod.DISABLE_ENV_VAR, "1")
    hw_mod = main_mod._hw_module
    monkeypatch.setattr(hw_mod, "DEVICE", hw_mod.DeviceType.CPU, raising = False)
    was_complete = hw_mod.DETECTION_COMPLETE.is_set()
    hw_mod.DETECTION_COMPLETE.set()
    try:
        assert _asyncio.run(main_mod._await_hardware_detection(0.05)) is True
    finally:
        if not was_complete:
            hw_mod.DETECTION_COMPLETE.clear()


# ------------------------------------------------------- the vision probe hop
def test_the_standalone_vision_probe_runs_off_the_event_loop():
    """GET /api/models/check-vision must not build the registry sets inline.

    The sets behind is_vision_model() are built lazily now, so the first call
    either imports transformers or waits on _DETECTION_SETS_LOCK while the warm
    thread holds it -- both of which park uvicorn for the rest of that import.
    """
    path = _BACKEND / "routes" / "models.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "check_vision_model"
    )

    # Nested helpers are the offload, so only the handler's own body counts as
    # inline.
    nested = {
        sub
        for helper in ast.walk(fn)
        if isinstance(helper, ast.FunctionDef)
        for sub in ast.walk(helper)
    }
    direct = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "is_vision_model"
        and sub not in nested
    ]
    assert not direct, (
        "is_vision_model() is called inline from the handler again; the lazy "
        "registry build then blocks the event loop"
    )

    offloaded = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "to_thread"
    ]
    assert offloaded, "the vision probe is no longer handed to a worker thread"
    reached = {
        call.func.id
        for helper in ast.walk(fn)
        if isinstance(helper, ast.FunctionDef)
        for call in ast.walk(helper)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "is_vision_model" in reached, (
        "nothing handed to the worker calls is_vision_model; the probe moved "
        "somewhere this test no longer covers"
    )


# ------------------------------------------------------ offline is not just 1
def test_the_ranking_fetch_uses_the_shared_offline_check():
    """HF_HUB_OFFLINE=true and TRANSFORMERS_OFFLINE are offline here too.

    Every other offline read in this backend goes through hf_env_offline(); a
    literal "1" comparison leaves a boot that set any of the other accepted
    spellings making a raw outbound httpx.get to Hugging Face.
    """
    path = _BACKEND / "core" / "inference" / "orchestrator.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_start_top_models_fetch"
    )

    calls = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "hf_env_offline"
    ]
    assert calls, "the ranking guard no longer asks the shared offline helper"

    literal = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Constant) and sub.value == "HF_HUB_OFFLINE"
    ]
    assert not literal, (
        "the guard reads HF_HUB_OFFLINE directly again; true/yes/on and "
        "TRANSFORMERS_OFFLINE would go back to being treated as online"
    )


def test_the_shared_offline_check_accepts_the_other_spellings(monkeypatch):
    """The helper the guard now uses has to be the broader one, not a rename."""
    from utils.utils import hf_env_offline

    for var, value in (
        ("HF_HUB_OFFLINE", "true"),
        ("HF_HUB_OFFLINE", "yes"),
        ("HF_HUB_OFFLINE", "on"),
        ("TRANSFORMERS_OFFLINE", "1"),
    ):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
        monkeypatch.setenv(var, value)
        assert hf_env_offline(), f"{var}={value} was treated as online"

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    assert not hf_env_offline(), "a host with neither variable set was called offline"


# --------------------------------------------- shutdown resets the whole verdict
def test_shutdown_resets_chat_only_with_the_detection_event():
    """Clearing DEVICE without CHAT_ONLY leaves a stale capability published.

    Health falls back to a bare CHAT_ONLY read while the completion event is
    clear, so a second lifespan after a GPU run would answer chat_only: false
    before anything re-measured it. config/env.ts stores that even with no
    device_type, which shows Train and Export and lets the route guard through.
    """
    import asyncio as _asyncio

    from utils import lifespan_shutdown as shutdown_mod

    hw_mod = _FakeHardwareModule()
    hw_mod.DEVICE = "cuda"
    hw_mod.CHAT_ONLY = False
    hw_mod.CHAT_ONLY_REASON = None
    hw_mod.IS_ROCM = True
    hw_mod.DETECTION_COMPLETE.set()

    _asyncio.run(_run_shutdown(shutdown_mod, hw_mod))

    assert hw_mod.DEVICE is None
    assert not hw_mod.DETECTION_COMPLETE.is_set()
    assert hw_mod.CHAT_ONLY is True, (
        "a torn-down GPU verdict is still published as chat_only: false; the "
        "next lifespan offers Train and Export before detection has run"
    )
    assert hw_mod.CHAT_ONLY_REASON is None
    assert hw_mod.IS_ROCM is False, "a stale ROCm flag would mislabel the next host"


class _FakeHardwareModule:
    """The minimal hardware surface run_lifespan_shutdown touches."""

    def __init__(self) -> None:
        self.DEVICE = None
        self.CHAT_ONLY = True
        self.CHAT_ONLY_REASON = None
        self.IS_ROCM = False
        self.DETECTION_COMPLETE = threading.Event()


async def _run_shutdown(shutdown_mod, hw_mod) -> None:
    """Drive run_lifespan_shutdown with everything but the hardware reset stubbed."""
    await shutdown_mod.run_lifespan_shutdown(
        terminate_downloads = lambda: None,
        clear_compiled_cache = lambda: None,
        hw_module = hw_mod,
    )


# ------------------------------------------- the post-warm worker rechecks
def test_the_post_warm_worker_rechecks_before_each_action():
    """One check after the join leaves a window shutdown can land in.

    The join is the long wait, but the MLX autorepair and the RAG warm each take
    their own time, and the RAG warm can go as far as spawning a llama-server.
    A generation read before each action is what keeps a stopped lifespan from
    starting either.
    """
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_post_warm_background_work"
    )
    checks = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "_post_warm_retired"
    ]
    assert len(checks) >= 3, (
        "the post-warm worker checks retirement fewer than once per action; a "
        "shutdown landing after the join can still start MLX autorepair or a "
        "llama-server for a lifespan that has stopped"
    )


def test_the_retirement_check_reads_the_live_generation(monkeypatch):
    """It has to compare against the current counter, not a captured one."""
    import main as main_mod

    monkeypatch.setattr(main_mod, "_post_warm_current_generation", lambda: 7, raising = True)
    assert main_mod._post_warm_retired(7) is False
    assert main_mod._post_warm_retired(6) is True
    assert (
        main_mod._post_warm_retired(None) is False
    ), "a direct call with no generation must still run; that is the test path"


# --------------------------------------- the saved GPU override goes off-loop
def test_the_saved_gpu_override_check_runs_off_the_event_loop():
    """The auto-switch path reaches this before any explicit-gpu_ids offload.

    resolve_requested_gpu_ids() calls get_device() and get_physical_gpu_count()
    itself, so both wait on the detection lock while the warm imports torch.
    """
    path = _BACKEND / "routes" / "inference.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_override_gpu_ids_still_resolve"
    )
    nested = {
        sub
        for helper in ast.walk(fn)
        if isinstance(helper, ast.FunctionDef)
        for sub in ast.walk(helper)
    }
    inline = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id in {"get_device", "resolve_requested_gpu_ids"}
        and sub not in nested
    ]
    assert not inline, (
        "get_device() or resolve_requested_gpu_ids() is called inline again; the "
        "first auto-switch to a model with a stored pin then holds the event loop "
        "for the whole torch import"
    )
    reached = {
        call.func.id
        for helper in ast.walk(fn)
        if isinstance(helper, ast.FunctionDef)
        for call in ast.walk(helper)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert {"get_device", "resolve_requested_gpu_ids"} <= reached, (
        "the offloaded helper no longer does the device resolution; the check "
        "moved somewhere this test does not cover"
    )


# ------------------------------------------ a retired detection cannot publish
def test_a_detection_retired_by_shutdown_does_not_publish(monkeypatch):
    """Shutdown clears the verdict; a detector inside the torch import must not
    put it back.

    Shutdown cannot take _DETECT_LOCK to stop it -- that parks teardown behind the
    whole import, the stall this startup path exists to remove -- so it retires
    the epoch instead. Without this, the detector republishes a complete,
    settled-looking verdict over the reset, and the next lifespan reads a non-None
    DEVICE and skips detection entirely, serving the retired run's answer.
    """
    monkeypatch.setattr(hw, "DEVICE", None, raising = False)
    monkeypatch.setattr(hw, "CHAT_ONLY", True, raising = False)

    def _detect_and_get_retired():
        # Stands in for the torch import: shutdown lands while we are inside it.
        hw.DEVICE = hw.DeviceType.CUDA
        hw.CHAT_ONLY = False
        hw.invalidate_detection()
        return hw.DeviceType.CUDA

    monkeypatch.setattr(hw, "_detect_hardware_locked", _detect_and_get_retired)
    hw.DETECTION_COMPLETE.clear()
    hw.detect_hardware()

    assert hw.DEVICE is None, (
        "a retired detection published its verdict; the next lifespan skips "
        "detection and serves hardware the previous one measured"
    )
    assert hw.CHAT_ONLY is True
    assert not hw.DETECTION_COMPLETE.is_set()


def test_a_detection_that_is_not_retired_still_publishes(monkeypatch):
    """Negative control: the epoch gate must not block ordinary detection."""
    monkeypatch.setattr(hw, "DEVICE", None, raising = False)
    monkeypatch.setattr(hw, "CHAT_ONLY", True, raising = False)

    def _ok():
        hw.DEVICE = hw.DeviceType.CUDA
        hw.CHAT_ONLY = False
        return hw.DeviceType.CUDA

    monkeypatch.setattr(hw, "_detect_hardware_locked", _ok)
    hw.DETECTION_COMPLETE.clear()
    try:
        assert hw.detect_hardware() is hw.DeviceType.CUDA
        assert hw.DEVICE is hw.DeviceType.CUDA
        assert hw.DETECTION_COMPLETE.is_set()
    finally:
        hw.DETECTION_COMPLETE.set()


def test_shutdown_retires_the_detection_epoch():
    """run_lifespan_shutdown must move the epoch, not just clear the globals."""
    src = (_BACKEND / "utils" / "lifespan_shutdown.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_lifespan_shutdown"
    )
    names = {
        sub.value
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    }
    assert "invalidate_detection" in names, (
        "shutdown no longer retires the detection epoch; a detector in the torch "
        "import republishes over the reset"
    )


# ------------------------------------------- the MCP status tool stays off-loop
def test_the_mcp_status_tool_reads_hardware_off_the_event_loop():
    """get_gpu_utilization() reaches detection, which blocks on the warm import."""
    path = _BACKEND / "mcp_server.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    called_directly = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_gpu_utilization"
    ]
    assert not called_directly, (
        "the MCP status tool calls get_gpu_utilization() inline again; it blocks "
        "the event loop on the warm's torch import"
    )
    offloaded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to_thread"
        and any(isinstance(a, ast.Name) and a.id == "get_gpu_utilization" for a in node.args)
    ]
    assert offloaded, "the hardware read is no longer handed to a worker thread"


# -------------------------------- an authed reply is not both settled and not
def test_an_authed_reply_drops_the_provisional_marker():
    """base is built before the bearer await, so its marker can be out of date."""
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "health_check"
    )
    pops = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "pop"
        and any(isinstance(a, ast.Constant) and a.value == "hardware_detecting" for a in sub.args)
    ]
    assert pops, (
        "an authed reply can still carry hardware_detecting beside the measured "
        "verdict it qualifies; a client may believe either one"
    )


# ----------------------------------------- deferred detection is not "in progress"
def test_health_marks_a_deferred_detection_as_deferred(monkeypatch):
    """With the warm off nothing settles, so a poller must be told to stop."""
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "health_check"
    )
    keys = {
        sub.slice.value
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Subscript)
        and isinstance(sub.slice, ast.Constant)
        and isinstance(sub.slice.value, str)
    }
    assert "hardware_detection_deferred" in keys, (
        "health does not distinguish a deferred detection from one in flight; a "
        "client waiting for a measured verdict stalls every load for its whole "
        "budget, including /login"
    )


# ------------------------------------ the warm stops once its lifespan is retired
def test_the_warm_stops_after_a_stage_its_lifespan_no_longer_owns():
    """Later stages build the orchestrator, which starts a fresh detection.

    Discarding the hardware stage's verdict is not enough on its own: the
    inference_backend stage reaches get_device() and would republish DEVICE
    after teardown cleared it.
    """
    tree = ast.parse((_BACKEND / "utils" / "torch_warmup.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_warm"
    )
    loops = [sub for sub in ast.walk(fn) if isinstance(sub, ast.For)]
    assert loops, "the warm no longer iterates its stages"
    checks = [
        sub
        for sub in ast.walk(loops[0])
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "_detection_epoch"
    ]
    assert checks, (
        "the stage loop does not re-check the detection epoch; a warm whose "
        "lifespan ended still runs the stages that rebuild the verdict"
    )


# -------------------------------- describing what is loaded must not build it
def test_the_monitor_context_read_does_not_construct_the_backend():
    """It is called inline from the OpenAI, Responses and Anthropic paths."""
    tree = ast.parse((_BACKEND / "routes" / "inference.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_monitor_context_length"
    )
    constructing = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "get_inference_backend"
    ]
    assert not constructing, (
        "_monitor_context_length constructs the singleton again; the first API "
        "request during the warm then waits on the torch import on the loop"
    )
    peeking = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "_peek_inference_backend"
    ]
    assert peeking, "the helper no longer reads the backend at all"


def test_the_peek_never_constructs():
    """Behavioural: the peek must return None rather than build one."""
    from core.inference.orchestrator import peek_inference_backend
    import core.inference.orchestrator as orch

    before = orch._inference_backend
    try:
        orch._inference_backend = None
        assert peek_inference_backend() is None
        assert orch._inference_backend is None, "the peek constructed an orchestrator"
    finally:
        orch._inference_backend = before


def test_the_metadata_cleanup_does_not_construct_the_backend():
    """A metadata-only cleanup has no reason to import torch."""
    tree = ast.parse((_BACKEND / "routes" / "models.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "discard_remote_code_download"
    )
    constructing = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "get_inference_backend"
    ]
    assert (
        not constructing
    ), "the cleanup path constructs the ML backend again for a metadata-only decision"
    assert any(
        isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "peek_inference_backend"
        for sub in ast.walk(fn)
    ), "the cleanup path no longer checks the loaded model at all"


def test_the_research_probes_do_not_construct_the_backend():
    """A durable research run resumes on the loop as soon as the port is recorded.

    _process() and _wait_for_local_model() call these two sync probes inline, so a cold
    get_inference_backend() there parks uvicorn on the torch import for the rest of the
    warm: login, liveness and the deadline-bound health probe all stall behind it.
    """
    tree = ast.parse((_BACKEND / "core" / "research_runs.py").read_text(encoding = "utf-8"))
    for name in ("_loaded_context_length", "_local_model_ready"):
        fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert not [
            sub
            for sub in ast.walk(fn)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "get_inference_backend"
        ], f"{name} constructs the singleton inline; a resumed run then blocks the loop"
        assert any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "_peek_inference_backend"
            for sub in ast.walk(fn)
        ), f"{name} no longer reads the native backend at all"


def test_the_research_peek_answers_cold_without_constructing():
    """Behavioural: with no orchestrator built, both probes answer and build nothing."""
    import core.inference.orchestrator as orch
    from core import research_runs

    before = orch._inference_backend
    try:
        orch._inference_backend = None
        assert research_runs._loaded_context_length() is None
        assert research_runs._local_model_ready() is False, (
            "no orchestrator means no native model is loaded; that is an answer, not a "
            "failed probe, so the run must not be told to go ahead"
        )
        assert orch._inference_backend is None, "a probe constructed an orchestrator"
    finally:
        orch._inference_backend = before


class _DeferredThread:
    """A Thread whose start() only schedules: run() is called by the test, later.

    That gap is the race. The spawner has returned, shutdown has retired the epoch,
    and only then does the worker body run.
    """

    instances: list = []

    def __init__(
        self,
        target = None,
        args = (),
        **_kw,
    ):
        self._target, self._args = target, args
        self._started = False
        _DeferredThread.instances.append(self)

    def start(self):
        self._started = True

    def is_alive(self):
        return self._started

    def run_now(self):
        self._target(*self._args)


def test_a_warm_delayed_past_shutdown_is_already_retired(monkeypatch):
    """The epoch must be bound at spawn, not by the thread itself.

    Thread.start() releases the GIL and the child may not run for a while. A shutdown
    in that gap retires the lifespan; a thread that read the epoch itself would read
    the post-shutdown value, match it against itself forever, and warm on -- rebuilding
    DEVICE and the orchestrator after teardown cleared them.
    """
    import utils.torch_warmup as warm
    from utils.hardware import hardware as hw

    ran: list = []
    stages = tuple((name, lambda name = name: ran.append(name)) for name in ("hardware", "later"))
    monkeypatch.setattr(warm, "_STAGES", stages)
    monkeypatch.setattr(warm.threading, "Thread", _DeferredThread)
    monkeypatch.setattr(warm, "_thread", None)
    # Own the latch and the reported status: both are module state a real start
    # mutates, and leaving them set makes the next test see a warm already running.
    monkeypatch.setattr(warm, "_status", {"started": False, "finished": False, "stages": {}})
    monkeypatch.delenv(warm.DISABLE_ENV_VAR, raising = False)
    _DeferredThread.instances.clear()

    assert warm.start_background_warm() is True
    hw.invalidate_detection()  # the shutdown that lands before the thread runs
    _DeferredThread.instances[-1].run_now()

    assert ran == [], f"a warm retired before it ran did its stages anyway: {ran}"


def test_a_warm_that_owns_its_epoch_still_runs(monkeypatch):
    """Negative control: the guard must not stop an ordinary warm."""
    import utils.torch_warmup as warm

    ran: list = []
    stages = tuple((name, lambda name = name: ran.append(name)) for name in ("hardware", "later"))
    monkeypatch.setattr(warm, "_STAGES", stages)
    monkeypatch.setattr(warm.threading, "Thread", _DeferredThread)
    monkeypatch.setattr(warm, "_thread", None)
    # Own the latch and the reported status: both are module state a real start
    # mutates, and leaving them set makes the next test see a warm already running.
    monkeypatch.setattr(warm, "_status", {"started": False, "finished": False, "stages": {}})
    monkeypatch.delenv(warm.DISABLE_ENV_VAR, raising = False)
    _DeferredThread.instances.clear()

    assert warm.start_background_warm() is True
    _DeferredThread.instances[-1].run_now()

    assert ran == ["hardware", "later"], f"a live warm was stopped: {ran}"


def test_a_detection_delayed_past_shutdown_does_not_publish(monkeypatch):
    """Same race on the other spawner: start_background_detection()'s worker."""
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "DEVICE", None)
    monkeypatch.setattr(hw, "_DETECT_THREAD", None)
    monkeypatch.setattr(hw.threading, "Thread", _DeferredThread)
    monkeypatch.setattr(hw, "_detect_hardware_locked", lambda: setattr(hw, "DEVICE", "cuda"))
    # A fresh Event, so clearing it cannot leak the "not detected yet" state into a
    # later test that reads the real one.
    monkeypatch.setattr(hw, "DETECTION_COMPLETE", threading.Event())
    _DeferredThread.instances.clear()

    hw.start_background_detection()
    hw.invalidate_detection()
    _DeferredThread.instances[-1].run_now()

    assert hw.DEVICE is None, "a retired detection published a verdict after teardown"
    assert not hw.DETECTION_COMPLETE.is_set(), "a retired detection announced itself as settled"


def test_both_spawners_read_the_epoch_before_start():
    """AST: the epoch has to be an argument to the thread, not read inside the target."""
    for rel, spawner, target in (
        (("utils", "torch_warmup.py"), "start_background_warm", "_warm"),
        (
            ("utils", "hardware", "hardware.py"),
            "start_background_detection",
            "ensure_hardware_detected",
        ),
    ):
        tree = ast.parse(_BACKEND.joinpath(*rel).read_text(encoding = "utf-8"))
        fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == spawner
        )
        call = next(
            sub
            for sub in ast.walk(fn)
            if isinstance(sub, ast.Call)
            and any(
                kw.arg == "target" and isinstance(kw.value, ast.Name) and kw.value.id == target
                for kw in sub.keywords
            )
        )
        args_kw = next((kw for kw in call.keywords if kw.arg == "args"), None)
        assert args_kw is not None and isinstance(
            args_kw.value, ast.Tuple
        ), f"{spawner} starts {target} with no epoch bound at spawn time"
        assert any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id in {"_detection_epoch", "current_detection_epoch"}
            for sub in ast.walk(args_kw.value)
        ), f"{spawner} passes args to {target} but not the detection epoch"


def test_a_failed_forced_redetect_does_not_restore_a_retired_verdict():
    """detect_hardware()'s except path must honour the epoch too.

    It saves the published verdict, clears DETECTION_COMPLETE, then re-detects. If
    shutdown retires the pass and the probe then raises, restoring puts back exactly
    what shutdown cleared, and the next lifespan reads a non-None DEVICE and skips its
    own detection. The success path checks; the failure path did not.
    """
    from utils.hardware import hardware as hw

    saved = (hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM)
    was_complete = hw.DETECTION_COMPLETE.is_set()
    try:
        hw.DEVICE = hw.DeviceType.MLX
        hw.CHAT_ONLY = True
        hw.CHAT_ONLY_REASON = "mlx_unavailable"
        hw.DETECTION_COMPLETE.set()

        def _shutdown_then_fail():
            hw.invalidate_detection()  # the shutdown, mid-pass
            raise RuntimeError("probe blew up")

        with mock.patch.object(hw, "_detect_hardware_locked", _shutdown_then_fail):
            with pytest.raises(RuntimeError):
                hw.detect_hardware()

        assert hw.DEVICE is None, "a retired pass restored the verdict shutdown cleared"
        assert hw.CHAT_ONLY_REASON is None
        assert not hw.DETECTION_COMPLETE.is_set(), "a retired pass re-announced itself as settled"
    finally:
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM = saved
        hw.DETECTION_COMPLETE.set() if was_complete else hw.DETECTION_COMPLETE.clear()


def test_a_failed_redetect_inside_its_own_lifespan_still_restores():
    """Negative control: without a shutdown, the rollback must still happen."""
    from utils.hardware import hardware as hw

    saved = (hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM)
    was_complete = hw.DETECTION_COMPLETE.is_set()
    try:
        hw.DEVICE = hw.DeviceType.MLX
        hw.CHAT_ONLY = True
        hw.CHAT_ONLY_REASON = "mlx_unavailable"
        hw.DETECTION_COMPLETE.set()

        def _just_fail():
            raise RuntimeError("probe blew up")

        with mock.patch.object(hw, "_detect_hardware_locked", _just_fail):
            with pytest.raises(RuntimeError):
                hw.detect_hardware()

        assert hw.DEVICE is hw.DeviceType.MLX, "a live lifespan lost its verdict to a failed probe"
        assert (
            hw.CHAT_ONLY_REASON == "mlx_unavailable"
        ), "losing the reason stops the sidebar's MLX recovery poll for good"
        assert hw.DETECTION_COMPLETE.is_set()
    finally:
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM = saved
        hw.DETECTION_COMPLETE.set() if was_complete else hw.DETECTION_COMPLETE.clear()


def test_a_broken_torch_install_is_not_reported_as_a_host_without_a_gpu():
    """An installed torch whose import raises is a detection failure, not "no GPU".

    Widening _has_torch() to swallow every exception is what keeps the warm thread
    from making each later request retry the same failing import. Reporting it as
    no_gpu, though, tells a GPU box it has no GPU and sends export_capability down
    the "install PyTorch" branch for a PyTorch that is installed.
    """
    from utils.hardware import hardware as hw

    saved_error = hw.TORCH_IMPORT_ERROR
    real_import = builtins.__import__

    def _broken(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise OSError("libcudart.so.13: cannot open shared object file")
        return real_import(name, *a, **k)

    try:
        with mock.patch.object(builtins, "__import__", _broken):
            assert hw._has_torch() is False, "a broken torch must not count as importable"
        assert hw.TORCH_IMPORT_ERROR is not None, "the import failure was not recorded"
        assert "libcudart" in hw.TORCH_IMPORT_ERROR
    finally:
        hw.TORCH_IMPORT_ERROR = saved_error


def test_an_absent_torch_is_still_just_absent():
    """Negative control: ImportError is "not installed", not a broken install."""
    from utils.hardware import hardware as hw

    saved_error = hw.TORCH_IMPORT_ERROR
    real_import = builtins.__import__

    def _missing(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *a, **k)

    try:
        hw.TORCH_IMPORT_ERROR = "stale"
        with mock.patch.object(builtins, "__import__", _missing):
            assert hw._has_torch() is False
        assert (
            hw.TORCH_IMPORT_ERROR is None
        ), "a --no-torch install would be reported as a detection failure"
    finally:
        hw.TORCH_IMPORT_ERROR = saved_error


def test_the_default_model_list_is_stamped_before_it_is_built():
    """AST: read the generation first, or a stale list is tagged current forever.

    get_default_models() settles detection and reads CHAT_ONLY. A re-detection landing
    between that and the stamp would mark the pre-repair list as belonging to the
    post-repair generation, and _refresh_static_models_if_stale would never fire again.
    """
    tree = ast.parse(
        (_BACKEND / "core" / "inference" / "orchestrator.py").read_text(encoding = "utf-8")
    )

    def _line_of(fn, predicate):
        return next(
            sub.lineno for sub in ast.walk(fn) if isinstance(sub, ast.Assign) and predicate(sub)
        )

    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "InferenceOrchestrator"
    )
    for name in ("__init__", "_refresh_static_models_if_stale"):
        fn = next(
            node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        stamp = _line_of(
            fn,
            lambda a: any(
                isinstance(t, ast.Attribute) and t.attr == "_static_models_generation"
                for t in a.targets
            ),
        )
        build = _line_of(
            fn,
            lambda a: any(
                isinstance(t, ast.Attribute) and t.attr == "_static_models" for t in a.targets
            ),
        )
        assert stamp < build, (
            f"InferenceOrchestrator.{name} stamps the generation after building the "
            "list, so a "
            "re-detection in between makes the stale list look current forever"
        )


def test_a_redetect_during_the_bearer_await_leaves_the_reply_provisional():
    """AST: the authed branch must mark the reply when the second snapshot is None.

    base carries no chat_only_reason, so an unmarked reply is read as measured and
    stores chat_only with reason null, which stops the sidebar's mlx_unavailable poll.
    """
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "health_check"
    )
    branch = next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and any(
            isinstance(sub, ast.Subscript)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "authed"
            and getattr(sub.slice, "value", None) == "device_type"
            for sub in ast.walk(node)
        )
    )
    assert branch.orelse, "no else branch: a mid-await re-detect ships an unmarked reply"
    assert any(
        isinstance(sub, ast.Subscript)
        and isinstance(sub.value, ast.Name)
        and sub.value.id == "authed"
        and getattr(sub.slice, "value", None) == "hardware_detecting"
        for node in branch.orelse
        for sub in ast.walk(node)
    ), "the else branch does not mark the reply provisional"


def test_the_unload_eviction_checks_are_offloaded():
    """AST: both _unload_may_evict() calls reach the singleton, so neither runs inline."""
    tree = ast.parse((_BACKEND / "routes" / "inference.py").read_text(encoding = "utf-8"))
    inline = [
        sub.lineno
        for sub in ast.walk(tree)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "_unload_may_evict"
    ]
    assert not inline, (
        f"_unload_may_evict is called inline at {inline}; it reaches "
        "get_inference_backend(), so an unload during the warm stalls the loop"
    )
    offloaded = [
        sub.lineno
        for sub in ast.walk(tree)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "to_thread"
        and any(isinstance(a, ast.Name) and a.id == "_unload_may_evict" for a in sub.args)
    ]
    assert len(offloaded) == 2, f"expected both eviction checks offloaded, found {offloaded}"
