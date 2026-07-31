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
import sys
import threading
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
    # Scope to the class: this module defines more than one __init__, and picking
    # whichever ast.walk yields first made this assert a different constructor.
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
    # `package in sys.modules` is a Compare with an In op whose comparator is the
    # sys.modules attribute; matching the dumped text for "sys.modules" finds
    # nothing, because the dump spells it Attribute(..., attr='modules').
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
    # The docstring names DEVICE, so match the executable form: a comparison of
    # the DEVICE attribute against None. A dumped-text search passes on prose.
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

    # The guards must not be called directly from the coroutine body; they belong
    # to the nested helper that is handed to the worker.
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
    guarded = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Try)
        and any("to_thread" in ast.dump(h) for h in [sub])
        and sub.handlers
    ]
    assert guarded, "the offloaded guard is no longer inside a try/except"
    raises = [sub for sub in ast.walk(guarded[0]) if isinstance(sub, ast.Raise)]
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
    # inline. The helper the worker runs may call is_vision_model freely.
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
