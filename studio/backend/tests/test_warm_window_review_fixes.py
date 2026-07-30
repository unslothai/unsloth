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


def test_the_ranking_fetch_honours_hf_hub_offline():
    """It is a raw httpx.get, so HF_HUB_OFFLINE does not reach it by itself."""
    src = (_BACKEND / "core" / "inference" / "orchestrator.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_start_top_models_fetch"
    )
    # The name appears in the docstring explaining why the check is needed, so
    # match the executable form: a string constant "HF_HUB_OFFLINE" passed to a
    # call. A docstring is a bare Constant, never a Call argument.
    reads = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and any(isinstance(a, ast.Constant) and a.value == "HF_HUB_OFFLINE" for a in sub.args)
    ]
    assert reads, (
        "the lazy start does not read HF_HUB_OFFLINE, so an offline host still "
        "reaches huggingface.co on its first model list"
    )


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
    """is_vision_model() reaches _detection_sets(), so it cannot run inline."""
    path = _BACKEND / "routes" / "models.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_model_config"
    )
    dumped = ast.dump(fn)
    assert "_capabilities" in dumped, (
        "the capability detection block is inline again; is_vision_model() can "
        "import transformers on the event-loop thread"
    )
    # And the inline calls must be gone from the handler body proper.
    inline = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == "is_vision_model"
    ]
    assert len(inline) == 1, "is_vision_model is called somewhere other than the offloaded helper"


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
