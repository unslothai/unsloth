# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The warm window's remaining sharp edges, one test per edge.

Deferring the ML stack creates an interval between the socket binding and the stack
being importable. Several separate things went wrong in that interval:

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
import os
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

    _refresh_static_models_if_stale() reads a change as "hardware was re-detected" and
    rebuilds the curated defaults, so counting cache hits made any GPU or export helper
    trigger a rebuild plus a false log.
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

    The pass resets CHAT_ONLY / CHAT_ONLY_REASON / IS_ROCM on entry and the MLX
    autorepair path catches the exception, so without a restore the reason is gone, and
    the sidebar recovery poll only continues while it reads "mlx_unavailable".
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

    Starting the ranking fetch from __init__ meant every boot reached huggingface.co
    before anyone signed in. Asserted on the source: importing the orchestrator here
    would pull the whole inference stack.
    """
    tree = ast.parse(
        (_BACKEND / "core" / "inference" / "orchestrator.py").read_text(encoding = "utf-8")
    )
    # Scope to the class: the module defines more than one __init__.
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

    Driven through the real method: every spelling the rest of the backend accepts as
    offline has to leave the first model list network-silent, not just the one the
    guard happens to compare against.
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

    Left inline, an early GET /v1/models held the event-loop thread for the rest of the
    torch import, and the offload further down could not help: the call had happened.
    """
    path = _BACKEND / "routes" / "inference.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)}
    if function not in names:
        pytest.skip(f"{function} is not an async handler in this tree")
    assert "_openai_model_objects" in _async_offloaded_names(path, function)


def test_the_model_config_capability_block_runs_off_loop():
    """is_vision_model() reaches _detection_sets(), so it cannot run inline.

    Pins the property rather than a helper name: the handler body must not call the
    probes, and whatever the worker runs must.
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

    Gating autorepair on it left an Apple Silicon host with a broken MLX stack
    chat-only for good, where before it ran in the lifespan no matter what.
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
    # `package in sys.modules` is a Compare/In; a dumped-text search misses it, since
    # the dump spells it attr='modules'.
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

    The worker spends its life parked in join_background_warm(), so a shutdown landing
    before the warm finishes cannot reach it any other way: it wakes up later and loads
    the embedder, which can fall back to spawning a llama-server.
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

    Refusing while the previous worker was alive meant a restart got nothing at all:
    the old worker was parked in the join so the start declined, then it read the
    shutdown and exited.
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

    config/env.ts caches the first reply carrying device_type as authoritative, and the
    sidebar recovery poll continues only while it reads chat_only_reason ==
    "mlx_unavailable", so one mid-pass chat-only reply with a null reason hides Train
    for the rest of the SPA session.
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
    # Match the executable form (DEVICE compared to None); the docstring names DEVICE
    # too, so a dumped-text search would pass on prose.
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

    _inference_backend_blocks_delete() is sync and reaches get_inference_backend(),
    whose cold build waits on detection, so inline an authed DELETE arriving during the
    warm held the event-loop thread for the rest of the torch import.
    """
    path = _BACKEND / "hub" / "services" / "models" / "deletion.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "delete_cached_model_response"
    )

    # The guards belong to the nested helper handed to the worker, not the body.
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

    The `or` matters: with a GGUF model loaded the first guard answers and the second
    never runs. And an unreadable load state must still raise rather than fall through
    to unlinking weights under a live process.
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

    # Bind to the specific Try, not any Try whose dump mentions to_thread: a decoy
    # try/to_thread/raise elsewhere in the handler satisfies that even with the real
    # fail-closed guard deleted.
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
    # The raise must be in a handler: in the body it is whatever the guard raises on
    # success, which says nothing about an unreadable load state.
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

    The desktop preflight and the frontend's first fetch both hit /api/health unasked,
    so a kick from there imports torch on exactly the hosts whose owner set the switch
    because that import is broken or expensive.
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
    """Not kicking is not the same as not answering: once something else has detected
    (the first training or export call), health must stop answering provisionally."""
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

    The sets behind is_vision_model() are lazy now, so the first call either imports
    transformers or waits on _DETECTION_SETS_LOCK while the warm holds it, both of
    which park uvicorn for the rest of that import.
    """
    path = _BACKEND / "routes" / "models.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "check_vision_model"
    )

    # Nested helpers are the offload, so only the handler's own body counts as inline.
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

    Every other offline read here goes through hf_env_offline(); a literal "1"
    comparison leaves a boot that set any other accepted spelling making a raw outbound
    httpx.get to Hugging Face.
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

    Health falls back to a bare CHAT_ONLY read while the event is clear, so a second
    lifespan after a GPU run would answer chat_only: false before anything re-measured
    it. config/env.ts stores that even with no device_type, showing Train and Export.
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

    The MLX autorepair and the RAG warm each take their own time, and the RAG warm can
    spawn a llama-server, so a generation read before each action is what keeps a
    stopped lifespan from starting either.
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
    """The auto-switch path reaches this before any explicit-gpu_ids offload, and
    resolve_requested_gpu_ids() calls get_device() itself, so both wait on the
    detection lock while the warm imports torch."""
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
    """A detector inside the torch import must not put back what shutdown cleared.

    Shutdown cannot take _DETECT_LOCK to stop it: that parks teardown behind the whole
    import. It retires the epoch instead. Otherwise the detector republishes a
    settled-looking verdict over the reset, and the next lifespan reads a non-None
    DEVICE, skips detection, and serves the retired run's answer.
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

    Discarding the hardware stage's verdict is not enough: the inference_backend stage
    reaches get_device() and would republish DEVICE after teardown cleared it.
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
    """A durable research run resumes on the loop as soon as the port is recorded, and
    calls these two sync probes inline, so a cold get_inference_backend() there parks
    uvicorn on the torch import for the rest of the warm."""
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
    """A Thread whose start() only schedules; the test calls run() later.

    That gap is the race: the spawner has returned and shutdown has retired the epoch
    before the worker body runs.
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

    The child may not run for a while after start(), and a shutdown in that gap retires
    the lifespan. A thread reading the epoch itself would read the post-shutdown value
    and warm on, rebuilding DEVICE and the orchestrator after teardown cleared them.
    """
    import utils.torch_warmup as warm
    from utils.hardware import hardware as hw

    ran: list = []
    stages = tuple((name, lambda name = name: ran.append(name)) for name in ("hardware", "later"))
    monkeypatch.setattr(warm, "_STAGES", stages)
    monkeypatch.setattr(warm.threading, "Thread", _DeferredThread)
    monkeypatch.setattr(warm, "_thread", None)
    # Own the latch and the status: both are module state a real start mutates, and
    # leaving them set makes the next test see a warm already running.
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
    # Own the latch and the status: both are module state a real start mutates, and
    # leaving them set makes the next test see a warm already running.
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
    """AST: the epoch is read by the spawner and handed over, never read in the thread.

    The target may be chosen indirectly (start_background_warm picks a successor when a
    retired warm is still running), so match the Thread call by the spawner it lives in
    rather than by a literal target name.
    """
    readers = {"_detection_epoch", "current_detection_epoch"}
    for rel, spawner in (
        (("utils", "torch_warmup.py"), "start_background_warm"),
        (("utils", "hardware", "hardware.py"), "start_background_detection"),
    ):
        tree = ast.parse(_BACKEND.joinpath(*rel).read_text(encoding = "utf-8"))
        fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == spawner
        )
        call = next(
            (
                sub
                for sub in ast.walk(fn)
                if isinstance(sub, ast.Call) and any(kw.arg == "target" for kw in sub.keywords)
            ),
            None,
        )
        assert call is not None, f"{spawner} no longer starts a thread"
        args_kw = next((kw for kw in call.keywords if kw.arg == "args"), None)
        assert args_kw is not None, f"{spawner} starts its worker with no epoch handed over"

        # Every name the args expression can carry, following one level of indirection
        # through a local assignment (args = (thread, epoch) bound to a name).
        def _names(node):
            out = {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}
            for assign in ast.walk(fn):
                if isinstance(assign, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in out for t in assign.targets
                ):
                    out |= {s.id for s in ast.walk(assign.value) if isinstance(s, ast.Name)}
                if isinstance(assign, ast.Assign) and isinstance(assign.targets[0], ast.Tuple):
                    tgt = {t.id for t in assign.targets[0].elts if isinstance(t, ast.Name)}
                    if tgt & out:
                        out |= {s.id for s in ast.walk(assign.value) if isinstance(s, ast.Name)}
            return out

        inline = any(
            isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in readers
            for sub in ast.walk(args_kw.value)
        )
        carried = _names(args_kw.value)
        bound = any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in carried for t in node.targets)
            and any(
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in readers
                for sub in ast.walk(node.value)
            )
            for node in ast.walk(fn)
        )
        assert (
            inline or bound
        ), f"{spawner} hands its worker args that never carry the detection epoch"


def test_a_new_lifespan_warms_even_when_the_retired_one_is_still_running():
    """Declining to a stale live warm leaves the new lifespan with no warm at all.

    A shutdown landing mid-warm cannot reset the latch, and the stale worker stops at
    its next stage boundary. Nothing retries, so without a hand-off the restart serves
    with the inference backend and the remaining imports cold, which is the stall this
    module exists to remove.
    """
    import utils.torch_warmup as warm
    from utils.hardware import hardware as hw

    saved_thread, saved_epoch = warm._thread, warm._thread_epoch
    saved_status = dict(warm._status)
    disable = os.environ.pop(warm.DISABLE_ENV_VAR, None)
    release, entered, ran = threading.Event(), threading.Event(), []
    try:
        warm._thread, warm._thread_epoch = None, None
        warm._status.update({"started": False, "finished": False, "stages": {}})

        def _slow():
            entered.set()
            release.wait(30)
            ran.append("stale")

        with mock.patch.object(warm, "_STAGES", (("slow", _slow),)):
            assert warm.start_background_warm() is True
            assert entered.wait(30), "the first warm never started"
            hw.invalidate_detection()  # shutdown, with the warm still inside a stage
            # reset_background_warm() declines here, exactly as it does in the lifespan.
            assert warm.reset_background_warm() is False
            with mock.patch.object(warm, "_STAGES", (("fresh", lambda: ran.append("fresh")),)):
                assert warm.start_background_warm() is True, (
                    "the new lifespan was refused a warm because the retired one was "
                    "still running"
                )
                release.set()
                assert warm.join_background_warm(60) is True
        assert ran == [
            "stale",
            "fresh",
        ], f"expected the successor to wait out the retired warm, got {ran}"
    finally:
        release.set()
        warm.join_background_warm(30)
        if disable is not None:
            os.environ[warm.DISABLE_ENV_VAR] = disable
        warm._thread, warm._thread_epoch = saved_thread, saved_epoch
        warm._status.clear()
        warm._status.update(saved_status)


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
    """Negative control: a genuinely missing package is not a broken install.

    Only ModuleNotFoundError named "torch" is narrow enough to trust; plain ImportError is
    what a broken wheel raises.
    """
    from utils.hardware import hardware as hw

    saved_error = hw.TORCH_IMPORT_ERROR
    real_import = builtins.__import__

    def _missing(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'", name = "torch")
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


@pytest.mark.parametrize(
    "exc",
    [
        # Linux: raised inside torch's own __init__ when a native dependency does not resolve.
        ImportError("libcudart.so.12: cannot open shared object file"),
        # Windows.
        OSError("[WinError 126] The specified module could not be found"),
        # A missing submodule still means torch itself is installed.
        ModuleNotFoundError("No module named 'torch._C'", name = "torch._C"),
    ],
    ids = ["import_error_native_lib", "os_error_windows", "missing_submodule"],
)
def test_a_broken_torch_is_never_mistaken_for_an_absent_one(exc):
    """ImportError is not a synonym for "not installed".

    Keying on the class reports a wheel with unresolved CUDA libraries as a host with no
    GPU, and sends export_capability() down the "install PyTorch" branch for one already
    installed, re-running the failing import on every check.
    """
    from utils.hardware import hardware as hw

    saved_error = hw.TORCH_IMPORT_ERROR
    real_import = builtins.__import__

    def _broken(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise exc
        return real_import(name, *a, **k)

    try:
        hw.TORCH_IMPORT_ERROR = None
        with mock.patch.object(builtins, "__import__", _broken):
            assert hw._has_torch() is False
        assert (
            hw.TORCH_IMPORT_ERROR is not None
        ), f"{type(exc).__name__} from an installed torch was read as absent"
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
    # __init__ is single-threaded, so ordering is enough there: stamping first leaves a
    # racing re-detection ahead of the stamp, which is the safe direction.
    init = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    stamp = _line_of(
        init,
        lambda a: any(
            isinstance(t, ast.Attribute) and t.attr == "_static_models_generation"
            for t in a.targets
        ),
    )
    build = _line_of(
        init,
        lambda a: any(
            isinstance(t, ast.Attribute) and t.attr == "_static_models" for t in a.targets
        ),
    )
    assert stamp < build, (
        "InferenceOrchestrator.__init__ stamps the generation after building the list, "
        "so a re-detection in between makes the stale list look current forever"
    )

    # The refresh path has concurrent readers, where ordering alone is not enough: two
    # of them on different generations can commit out of order. It must capture the
    # generation before building and commit only while that is still the newest.
    fn = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_refresh_static_models_if_stale"
    )
    capture = _line_of(
        fn,
        lambda a: any(isinstance(t, ast.Name) and t.id == "generation" for t in a.targets),
    )
    build_call = next(
        sub.lineno
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "get_default_models"
    )
    assert (
        capture < build_call
    ), "_refresh_static_models_if_stale reads the generation after building the list"
    guarded = next(
        (
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.With)
            and any(
                isinstance(sub, ast.Attribute) and sub.attr == "_static_models_lock"
                for item in node.items
                for sub in ast.walk(item.context_expr)
            )
        ),
        None,
    )
    assert guarded is not None, "the commit is not serialized against concurrent refreshes"
    assert any(
        isinstance(sub, ast.Compare)
        and any(isinstance(n, ast.Name) and n.id == "generation" for n in ast.walk(sub))
        for sub in ast.walk(guarded)
    ), "the commit does not re-check the generation it built against"
    assert all(
        _line_of(
            guarded,
            lambda a, attr = attr: any(
                isinstance(t, ast.Attribute) and t.attr == attr for t in a.targets
            ),
        )
        for attr in ("_static_models", "_static_models_generation")
    ), "the stamp and the value are not committed together under the lock"


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


def test_a_stale_waiter_does_not_discard_the_new_lifespan_verdict():
    """Only discard what this call produced.

    A detection worker spawned by the previous lifespan can still be blocked on
    _DETECT_LOCK when shutdown retires its epoch. The new lifespan's warm takes the
    lock first and publishes. The stale worker then enters, finds DEVICE already set
    so runs no detection, and must not wipe the verdict it did not produce: doing so
    leaves the restarted app provisional until some request kicks detection again.
    """
    from utils.hardware import hardware as hw

    saved = (hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM)
    was_complete = hw.DETECTION_COMPLETE.is_set()
    try:
        stale_epoch = hw.current_detection_epoch()
        hw.invalidate_detection()  # the shutdown the worker lost to
        # The new lifespan's verdict, already published while the stale worker waited.
        hw.DEVICE = hw.DeviceType.CUDA
        hw.CHAT_ONLY = False
        hw.CHAT_ONLY_REASON = None
        hw.DETECTION_COMPLETE.set()

        def _must_not_run():
            raise AssertionError("the stale waiter re-ran detection over a live verdict")

        with mock.patch.object(hw, "_detect_hardware_locked", _must_not_run):
            hw.ensure_hardware_detected(stale_epoch)

        assert (
            hw.DEVICE is hw.DeviceType.CUDA
        ), "a stale waiter discarded the new lifespan's verdict"
        assert hw.CHAT_ONLY is False
        assert hw.DETECTION_COMPLETE.is_set(), "the restart was left reporting as unsettled"
    finally:
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM = saved
        hw.DETECTION_COMPLETE.set() if was_complete else hw.DETECTION_COMPLETE.clear()


def test_a_retired_pass_that_did_detect_still_discards():
    """Negative control: the discard must still fire for a verdict this call produced."""
    from utils.hardware import hardware as hw

    saved = (hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM)
    was_complete = hw.DETECTION_COMPLETE.is_set()
    try:
        hw.DEVICE = None
        hw.DETECTION_COMPLETE.clear()
        epoch = hw.current_detection_epoch()

        def _detect_then_shutdown():
            hw.DEVICE = hw.DeviceType.CUDA
            hw.CHAT_ONLY = False
            hw.invalidate_detection()  # shutdown lands inside the pass

        with mock.patch.object(hw, "_detect_hardware_locked", _detect_then_shutdown):
            hw.ensure_hardware_detected(epoch)

        assert hw.DEVICE is None, "a retired pass published its own verdict anyway"
        assert not hw.DETECTION_COMPLETE.is_set()
    finally:
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM = saved
        hw.DETECTION_COMPLETE.set() if was_complete else hw.DETECTION_COMPLETE.clear()


def test_the_warm_hands_its_epoch_to_detection():
    """The pre-stage check is not enough on its own.

    Shutdown can land between that check and _DETECT_LOCK, inside the torch import the
    hardware stage is running, so detection reading the epoch itself would bind to the
    retirement it should lose to and publish for the lifespan that ended.
    """
    import utils.torch_warmup as warm
    from utils.hardware import hardware as hw_mod

    # Patch the MODULE function, not the name the stage imports: utils.hardware exposes a
    # hand-written wrapper, and patching the bound name hops over it. That is how a wrapper
    # dropping the argument, and so raising into _run_stage on every boot, went unnoticed.
    seen: list = []
    with mock.patch.object(hw_mod, "ensure_hardware_detected", lambda e = None: seen.append(e)):
        warm._warm_hardware(41)
    assert seen == [41], f"the hardware stage dropped its epoch: {seen}"


def test_every_public_hardware_wrapper_accepts_what_it_delegates_to():
    """utils.hardware wraps rather than re-exports, so signatures can drift apart.

    A wrapper narrower than its target raises TypeError at the call site, which _run_stage
    catches and logs, so the stage is skipped and nothing detects the hardware.
    """
    import inspect
    import utils.hardware as pkg
    from utils.hardware import hardware as hw_mod

    for name in (
        "ensure_hardware_detected",
        "detect_hardware",
        "start_background_detection",
        "export_capability",
        "get_device",
    ):
        wrapper = getattr(pkg, name, None)
        target = getattr(hw_mod, name, None)
        if wrapper is None or target is None or wrapper is target:
            continue  # a genuine re-export cannot drift
        w = inspect.signature(wrapper).parameters
        t = inspect.signature(target).parameters
        assert set(t) <= set(w), (
            f"utils.hardware.{name} accepts {sorted(w)} but delegates to one taking "
            f"{sorted(t)}; a caller passing the extra argument raises TypeError"
        )


def test_the_warm_loop_passes_the_epoch_to_the_real_stage_only():
    """_warm() must bind the epoch onto the real hardware stage, and only that one.

    A patched _STAGES entry is a different object, so it keeps the zero-argument contract
    every other stage and every test stub relies on.
    """
    import utils.torch_warmup as warm
    from utils.hardware import hardware as hw

    got: list = []

    def _hardware(epoch = None):
        got.append(("hardware", epoch))

    zero_arg_calls: list = []
    stages = (("hardware", _hardware), ("later", lambda: zero_arg_calls.append("later")))
    with mock.patch.object(warm, "_STAGES", stages):
        warm._warm(hw.current_detection_epoch())
    # _hardware is not warm._warm_hardware, so it is called with no arguments.
    assert got == [("hardware", None)], got
    assert zero_arg_calls == ["later"]

    # The real stage does get it: assert on the call the loop builds.
    tree = ast.parse((_BACKEND / "utils" / "torch_warmup.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_warm"
    )
    assert any(
        isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "partial"
        and any(isinstance(a, ast.Name) and a.id == "epoch" for a in sub.args)
        for sub in ast.walk(fn)
    ), "_warm no longer binds the epoch onto the hardware stage"


def test_deleting_a_cached_model_does_not_construct_the_backend():
    """A metadata-only delete has no reason to import torch.

    Off-loop already, so this is not a stall; it is the kill switch defeated, and the warm
    window paying a torch import to answer "nothing loaded".
    """
    tree = ast.parse(
        (_BACKEND / "hub" / "services" / "models" / "deletion.py").read_text(encoding = "utf-8")
    )
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_inference_backend_blocks_delete"
    )
    assert not [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "get_inference_backend"
    ], "the delete guard constructs the ML backend to learn that nothing is loaded"
    assert any(
        isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "peek_inference_backend"
        for sub in ast.walk(fn)
    ), "the delete guard no longer checks the loaded model at all"


def test_the_delete_guard_lets_a_delete_through_when_nothing_is_loaded():
    """Behavioural: no orchestrator means no standard model, so nothing blocks."""
    import core.inference.orchestrator as orch
    from hub.services.models import deletion

    before = orch._inference_backend
    try:
        orch._inference_backend = None
        assert deletion._inference_backend_blocks_delete("unsloth/Qwen3.5-2B") is False
        assert orch._inference_backend is None, "the delete guard constructed an orchestrator"
    finally:
        orch._inference_backend = before


def test_the_warm_epoch_is_retired_before_any_shutdown_await():
    """The coordinated warm has to be stopped at shutdown entry too.

    run_lifespan_shutdown() invalidates, but only after several awaits. A warm running
    through those keeps building the inference backend and importing transformers,
    datasets and unsloth_zoo for a lifespan that has stopped.
    """
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )
    yield_line = next(sub.lineno for sub in ast.walk(fn) if isinstance(sub, ast.Yield))
    awaits_after = sorted(
        sub.lineno for sub in ast.walk(fn) if isinstance(sub, ast.Await) and sub.lineno > yield_line
    )
    assert awaits_after, "the shutdown path no longer awaits anything; re-derive this"
    # The call goes through a getattr-bound name, so match the binding, not an attribute call.
    retire_lines = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "getattr"
            and any(
                isinstance(a, ast.Constant) and a.value == "invalidate_detection" for a in sub.args
            )
            for sub in ast.walk(node.value)
        )
        and node.lineno > yield_line
    ]
    assert retire_lines, "shutdown never retires the warm's detection epoch itself"
    assert min(retire_lines) < awaits_after[0], (
        f"the epoch is retired at line {min(retire_lines)}, after the first shutdown "
        f"await at {awaits_after[0]}; the warm keeps importing through that gap"
    )


def test_the_post_warm_worker_is_retired_before_any_shutdown_await():
    """It has to stop first, not merely early.

    Everything the post-warm worker does next loads part of the ML stack, including starting
    a llama-server. A warm finishing during a later shutdown await would still read the
    lifespan as current and go ahead, on one that is already tearing down.
    """
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )
    body = fn.body

    def _lineno(pred):
        return next(sub.lineno for sub in ast.walk(fn) if pred(sub))

    yield_line = _lineno(lambda n: isinstance(n, ast.Yield))
    stop_line = _lineno(
        lambda n: isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_stop_post_warm_thread"
    )
    awaits_after_yield = sorted(
        sub.lineno for sub in ast.walk(fn) if isinstance(sub, ast.Await) and sub.lineno > yield_line
    )
    assert stop_line > yield_line, "_stop_post_warm_thread must run on shutdown"
    assert awaits_after_yield, "the shutdown path no longer awaits anything; re-derive this"
    assert stop_line < awaits_after_yield[0], (
        f"_stop_post_warm_thread runs at line {stop_line}, after the first shutdown "
        f"await at {awaits_after_yield[0]}; a warm finishing in that gap starts "
        "post-warm work on a lifespan that is already shutting down"
    )
    assert body  # the walk above only makes sense on a non-empty body


def test_a_finished_warm_still_holds_the_latch_inside_its_own_lifespan():
    """Repeat calls stay no-ops however fast the warm ran.

    Treating any finished thread as absent is timing-dependent: with a trivial stage the
    warm can finish between two calls and the second starts a whole second warm (green on
    Linux, red on macOS). Only a warm whose epoch shutdown has retired is stale.
    """
    import utils.torch_warmup as warm

    saved_thread, saved_epoch = warm._thread, warm._thread_epoch
    saved_status = dict(warm._status)
    disable = os.environ.pop(warm.DISABLE_ENV_VAR, None)
    try:
        warm._thread, warm._thread_epoch = None, None
        warm._status.update({"started": False, "finished": False, "stages": {}})
        with mock.patch.object(warm, "_STAGES", (("noop", lambda: None),)):
            assert warm.start_background_warm() is True
            assert warm.join_background_warm(60) is True
            # Finished, same lifespan: still latched.
            assert (
                warm.start_background_warm() is False
            ), "a second warm started beside a completed one in the same lifespan"
            # Shutdown retires the epoch, as run_lifespan_shutdown does.
            from utils.hardware import hardware as hw

            hw.invalidate_detection()
            assert (
                warm.start_background_warm() is True
            ), "the next lifespan skipped its warm over hardware state shutdown cleared"
            assert warm.join_background_warm(60) is True
    finally:
        if disable is not None:
            os.environ[warm.DISABLE_ENV_VAR] = disable
        warm._thread, warm._thread_epoch = saved_thread, saved_epoch
        warm._status.clear()
        warm._status.update(saved_status)


def test_an_offline_first_read_does_not_retire_the_ranking_fetch():
    """Claiming the latch before the offline check disables the fetch for the process.

    A boot that happened to be offline, or a temporary force_hf_offline() scope, would
    then never pick the remote ranking up again however long the host stays online.
    """
    tree = ast.parse(
        (_BACKEND / "core" / "inference" / "orchestrator.py").read_text(encoding = "utf-8")
    )
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_start_top_models_fetch"
    )
    offline_line = next(
        sub.lineno
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "hf_env_offline"
    )
    claim_line = next(
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "_top_models_started" for t in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
    )
    assert offline_line < claim_line, (
        f"the latch is claimed at line {claim_line}, before the offline check at "
        f"{offline_line}; an offline first read then retires the fetch permanently"
    )


def test_an_offline_read_leaves_the_fetch_available():
    """Behavioural: offline must not consume the one-shot start."""
    import core.inference.orchestrator as orch

    started = []
    backend = orch.InferenceOrchestrator.__new__(orch.InferenceOrchestrator)
    backend._top_models_started = False
    with mock.patch.object(orch, "hf_env_offline", lambda: True):
        backend._start_top_models_fetch()
    assert (
        backend._top_models_started is False
    ), "an offline read consumed the latch, so the ranking can never be fetched again"
    with mock.patch.object(orch, "hf_env_offline", lambda: False):
        with mock.patch.object(orch.threading, "Thread") as thread:
            thread.side_effect = lambda **kw: started.append(kw) or mock.MagicMock()
            backend._start_top_models_fetch()
    assert (
        backend._top_models_started is True and started
    ), "coming back online did not start the fetch"


def test_a_slow_refresh_cannot_overwrite_a_newer_default_list():
    """Two readers refreshing different generations must not commit out of order.

    An MLX repair followed by a repeated lifespan gives several generations. A reader
    that started earlier can finish later, and storing its older list under the newer
    stamp leaves that stale GGUF-only list looking current for the rest of the process.
    """
    import core.inference.orchestrator as orch
    import utils.hardware.hardware as hw_mod

    backend = orch.InferenceOrchestrator.__new__(orch.InferenceOrchestrator)
    backend._static_models_lock = threading.Lock()
    backend._static_models = ["gen0"]
    backend._static_models_generation = 0

    saved_generation = hw_mod.DETECTION_GENERATION
    try:
        # The newer reader has already committed generation 2.
        hw_mod.DETECTION_GENERATION = 2
        with mock.patch("core.inference.defaults.get_default_models", lambda: ["gen2"]):
            backend._refresh_static_models_if_stale()
        assert backend._static_models == ["gen2"]
        assert backend._static_models_generation == 2

        # Now the slow reader from generation 1 finishes and tries to commit.
        hw_mod.DETECTION_GENERATION = 1
        backend._static_models_generation_before = backend._static_models_generation
        with mock.patch("core.inference.defaults.get_default_models", lambda: ["gen1"]):
            backend._refresh_static_models_if_stale()
        assert backend._static_models == [
            "gen2"
        ], "an older reader overwrote the newer default list"
        assert backend._static_models_generation == 2
    finally:
        hw_mod.DETECTION_GENERATION = saved_generation


def test_a_refresh_whose_generation_moved_mid_build_does_not_commit():
    """A list built against a generation that has since advanced is already stale."""
    import core.inference.orchestrator as orch
    import utils.hardware.hardware as hw_mod

    backend = orch.InferenceOrchestrator.__new__(orch.InferenceOrchestrator)
    backend._static_models_lock = threading.Lock()
    backend._static_models = ["old"]
    backend._static_models_generation = 1

    saved_generation = hw_mod.DETECTION_GENERATION
    try:
        hw_mod.DETECTION_GENERATION = 2

        def _slow_build():
            hw_mod.DETECTION_GENERATION = 3  # a re-detection lands mid-build
            return ["built-against-2"]

        with mock.patch("core.inference.defaults.get_default_models", _slow_build):
            backend._refresh_static_models_if_stale()
        assert backend._static_models == ["old"], "a list built against a retired generation landed"
        assert (
            backend._static_models_generation == 1
        ), "the stamp moved without the value, so the next read would not refresh"
    finally:
        hw_mod.DETECTION_GENERATION = saved_generation


def test_an_ordinary_refresh_still_happens():
    """Negative control: a single reader on a newer generation must still refresh."""
    import core.inference.orchestrator as orch
    import utils.hardware.hardware as hw_mod

    backend = orch.InferenceOrchestrator.__new__(orch.InferenceOrchestrator)
    backend._static_models_lock = threading.Lock()
    backend._static_models = ["old"]
    backend._static_models_generation = 1

    saved_generation = hw_mod.DETECTION_GENERATION
    try:
        hw_mod.DETECTION_GENERATION = 2
        with mock.patch("core.inference.defaults.get_default_models", lambda: ["new"]):
            backend._refresh_static_models_if_stale()
        assert backend._static_models == ["new"]
        assert backend._static_models_generation == 2
    finally:
        hw_mod.DETECTION_GENERATION = saved_generation


def test_a_retired_worker_does_not_probe_before_being_discarded():
    """Discarding after the probe still pays for the probe.

    A health-triggered detection thread can reach _DETECT_LOCK after shutdown retired
    its epoch. Running the full torch probe there imports the ML stack for a lifespan
    that has stopped, and the next lifespan's warm queues on the same lock behind it
    only to detect again, delaying the verdict this PR exists to deliver sooner.
    """
    from utils.hardware import hardware as hw

    saved = (hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM)
    was_complete = hw.DETECTION_COMPLETE.is_set()
    try:
        hw.DEVICE = None
        hw.DETECTION_COMPLETE.clear()
        stale_epoch = hw.current_detection_epoch()
        hw.invalidate_detection()  # shutdown, before the worker gets the lock

        probed = []
        with mock.patch.object(hw, "_detect_hardware_locked", lambda: probed.append(1)):
            hw.ensure_hardware_detected(stale_epoch)

        assert probed == [], "a retired worker imported the ML stack anyway"
        assert hw.DEVICE is None
        assert not hw.DETECTION_COMPLETE.is_set()
    finally:
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM = saved
        hw.DETECTION_COMPLETE.set() if was_complete else hw.DETECTION_COMPLETE.clear()


def test_a_live_worker_still_probes():
    """Negative control: an owner of the current epoch must detect as before."""
    from utils.hardware import hardware as hw

    saved = (hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM)
    was_complete = hw.DETECTION_COMPLETE.is_set()
    try:
        hw.DEVICE = None
        hw.DETECTION_COMPLETE.clear()
        probed = []

        def _probe():
            probed.append(1)
            hw.DEVICE = hw.DeviceType.CUDA

        with mock.patch.object(hw, "_detect_hardware_locked", _probe):
            hw.ensure_hardware_detected(hw.current_detection_epoch())

        assert probed == [1], "a live worker was refused its probe"
        assert hw.DETECTION_COMPLETE.is_set()
    finally:
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM = saved
        hw.DETECTION_COMPLETE.set() if was_complete else hw.DETECTION_COMPLETE.clear()
