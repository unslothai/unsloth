# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Classify every main-thread task by ORIGIN, read rather than inferred.

"A scheduler task took 8 ms" is not a finding. "A `MessageChannel` port message
posted from Blink's mojo connector ran 8 ms of page script, and there were 119
of them" is. The difference is that the second one is read out of the trace.

There is deliberately NO catch-all class. Buckets called "other" are exactly the
failure this tool exists to remove, so a task that no rule explains is counted
as `unclassified` and raises `unclassified_task_pct`; above the threshold the
cell FAILS instead of quietly reporting a smaller, cleaner-looking table.

Three evidence sources, in strict order of authority:

1. **Blink's own task type**, from the `scheduler` trace category. That category
   adds no events; it attaches TYPED ARGS to the `toplevel`
   `ThreadControllerImpl::RunTask` slice:
   `args.renderer_main_thread_task_execution.task_type` (a `TaskType` enum such
   as `TASK_TYPE_JAVASCRIPT_TIMER_DELAYED_HIGH_NESTING` or
   `TASK_TYPE_*POSTED_MESSAGE`) and `args.sequence_manager_task.queue_name`
   (`FRAME_THROTTLEABLE_TQ`, `COMPOSITOR_TQ`, `INPUT_TQ`, `V8_TQ`, ...). This is
   the scheduler stating what it thinks it is running. It is not an inference at
   all, and it is why `scheduler` is in the category list.
2. `src_file` / `src_func` on the same slice: the C++ that called `PostTask`.
3. The `devtools.timeline` events nested inside the task (`TimerFire`,
   `FireAnimationFrame`, `EventDispatch`, resource events), which say what the
   task then DID.

The ordering matters for the one class this tool most needs to get right.
Blink implements `MessagePort` on top of a mojo message pipe, so a React
scheduler callback and an inbound IPC arrive as the SAME
`connector.cc / PostDispatchNextMessageFromPipe` task and `src_file` cannot
separate them. The task type can: on a real capture, 120 `MessageChannel` round
trips came back labelled as posted-message tasks, exactly 120 of them, while the
`src_file` evidence lumped them in with unrelated IPC.

Note that `sequence_manager` as a category does NOT carry queue names, which is
the natural but wrong assumption; it contributes only the DoWork and DoIdleWork
scoping slices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from . import CellFailure
from .traceparse import Task, Thread, Trace, join_posted_from, walk_within_task

INPUT = "input"
TIMER = "timer"
MESSAGE_CHANNEL = "message_channel"
NETWORK = "network"
RAF = "raf"
IDLE = "idle"
GC = "gc"
BROWSER_INTERNAL = "browser_internal"
# Browser-to-renderer IPC that ran page script. In a harnessed run this is dominated by the
# devtools pipe delivering `Runtime.evaluate`, which is OUR OWN cost and belongs in a column with
# its own name so it can be watched for growth with the treatment.
AGENT_IPC = "agent_ipc"
UNCLASSIFIED = "unclassified"

ORIGINS = (
    INPUT,
    TIMER,
    MESSAGE_CHANNEL,
    NETWORK,
    RAF,
    IDLE,
    GC,
    BROWSER_INTERNAL,
    AGENT_IPC,
    UNCLASSIFIED,
)

# Origins that are the harness observing, not the app working. Reported, never subtracted: a
# correction you cannot see is a correction you cannot check.
HARNESS_ORIGINS = (AGENT_IPC,)

# Above this share of main-thread task time in the unclassified bucket, the cell is not quotable.
# The point of the tool is that we can name things.
UNCLASSIFIED_FAIL_PCT = 2.0

_JS_EVENTS = frozenset(
    {
        "FunctionCall",
        "v8.callFunction",
        "EvaluateScript",
        "v8.run",
        "v8.compile",
        "V8.Execute",
    }
)

_RESOURCE_EVENTS = frozenset(
    {
        "ResourceSendRequest",
        "ResourceReceiveResponse",
        "ResourceReceivedData",
        "ResourceFinish",
        "ResourceMarkAsCached",
        "XHRReadyStateChange",
        "XHRLoad",
    }
)

_RAF_EVENTS = frozenset(
    {
        "FireAnimationFrame",
        "AnimationFrame::Render",
        "AnimationFrame::StyleAndLayout",
        "BeginMainThreadFrame",
    }
)

_INPUT_EVENT_TYPES = frozenset(
    {
        "keydown",
        "keyup",
        "keypress",
        "beforeinput",
        "input",
        "compositionstart",
        "compositionupdate",
        "compositionend",
        "textInput",
        "mousedown",
        "mouseup",
        "mousemove",
        "click",
        "dblclick",
        "wheel",
        "pointerdown",
        "pointerup",
        "pointermove",
        "pointercancel",
        "touchstart",
        "touchend",
        "touchmove",
        "gesturescrollbegin",
        "gesturescrollupdate",
    }
)

# Network-ish streaming: an SSE frame arrives as a `message` event on an EventSource, and a
# fetch-based stream as a resource event.
_NETWORK_EVENT_TYPES = frozenset({"message", "readystatechange", "load", "error", "open"})

_POSTED_FROM_RULES: tuple[tuple[str, str, str], ...] = (
    # (src_file substring, src_func substring or "", origin)
    ("core/scheduler/dom_timer.cc", "", TIMER),
    ("blink/renderer/core/timing", "", TIMER),
    ("widget/input/main_thread_event_queue.cc", "", INPUT),
    ("blink/renderer/core/input", "", INPUT),
    ("cc/trees/proxy_impl.cc", "ScheduledActionSendBeginMainFrame", RAF),
    ("cc/trees/proxy_main.cc", "", RAF),
    ("v8/src/heap/", "", GC),
    ("blink/renderer/platform/heap/", "", GC),
    ("v8/src/tasks/", "", GC),
)

# Tasks the browser posts to itself. Only ever applied to a task that ran NO page script, so this
# can never swallow a JS bottleneck.
_BROWSER_INTERNAL_FILES = (
    "web_frame_widget_impl.cc",
    "main_thread_scheduler_impl.cc",
    "trace_session_observer.cc",
    "ipc_mojo_bootstrap.cc",
    "simple_watcher.cc",
    "render_media_client.cc",
    "sandboxed_process_thread_type_handler.cc",
    "memory_usage_monitor.cc",
    "gpu.cc",
    "interface_endpoint_client.cc",
    "paint_timing.cc",
    "layout_shift_tracker.cc",
    "cc/trees/",
    "perfetto_task_runner.cc",
    "compositor_frame_reporting_controller.cc",
    "frame_scheduler_impl.cc",
    "page_scheduler_impl.cc",
)

_IDLE_MARKERS = ("DoIdleWork", "RunIdleTask", "IdleTask", "ScheduleIdleTask")

# Blink `TaskType` enum names as they appear in
# `args.renderer_main_thread_task_execution.task_type`. Matched by SUBSTRING on the part after
# `TASK_TYPE_`, so both `..._TIMER_DELAYED_HIGH_NESTING` and `..._TIMER_IMMEDIATE` land on
# `timer`.
_TASK_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("JAVASCRIPT_TIMER", TIMER),
    ("POSTED_MESSAGE", MESSAGE_CHANNEL),
    ("UNSHIPPED_PORT_MESSAGE", MESSAGE_CHANNEL),
    ("POST_MESSAGE_FORWARDING", MESSAGE_CHANNEL),
    ("USER_INTERACTION", INPUT),
    ("INPUT_BLOCKING", INPUT),
    ("MAIN_THREAD_TASK_QUEUE_INPUT", INPUT),
    ("NETWORKING", NETWORK),
    ("WEB_SOCKET", NETWORK),
    ("MAIN_THREAD_TASK_QUEUE_COMPOSITOR", RAF),
    ("COMPOSITOR_THREAD_TASK_QUEUE", RAF),
    ("IDLE_TASK", IDLE),
    ("MAIN_THREAD_TASK_QUEUE_IDLE", IDLE),
    ("MAIN_THREAD_TASK_QUEUE_V8", GC),
    ("MICROTASK", MESSAGE_CHANNEL),
)

# `args.sequence_manager_task.queue_name`, used when the task type is absent.
_QUEUE_NAME_RULES: tuple[tuple[str, str], ...] = (
    ("INPUT_TQ", INPUT),
    ("UI_USER_INPUT_TQ", INPUT),
    ("COMPOSITOR_TQ", RAF),
    ("FRAME_THROTTLEABLE_TQ", TIMER),
    ("FRAME_LOADING_TQ", NETWORK),
    ("FRAME_LOADING_CONTROL_TQ", NETWORK),
    ("IDLE_TQ", IDLE),
    ("V8_TQ", GC),
    ("V8_USER_VISIBLE_TQ", GC),
    ("V8_BEST_EFFORT_TQ", GC),
)


def scheduler_labels(task: Task) -> dict[str, str]:
    """Read Blink's own labels for a task, if the `scheduler` category was on.

    Returns `{}` when the category was absent, which is a legitimate state:
    classification then falls back to `src_file` and nested evidence and the
    result carries lower-authority evidence strings.
    """
    for t in walk_within_task(task):
        if t.name != "ThreadControllerImpl::RunTask":
            continue
        a = t.args or {}
        exec_args = a.get("renderer_main_thread_task_execution") or {}
        sm_args = a.get("sequence_manager_task") or {}
        out: dict[str, str] = {}
        if isinstance(exec_args, dict) and exec_args.get("task_type"):
            out["task_type"] = str(exec_args["task_type"])
        if isinstance(sm_args, dict) and sm_args.get("queue_name"):
            out["queue_name"] = str(sm_args["queue_name"])
        if isinstance(sm_args, dict) and sm_args.get("priority"):
            out["priority"] = str(sm_args["priority"])
        if out:
            return out
    return {}


@dataclass
class ClassifiedTask:
    task: Task
    origin: str
    evidence: str
    src_file: str = ""
    src_func: str = ""
    ran_js: bool = False
    task_type: str = ""
    queue_name: str = ""

    @property
    def dur_us(self) -> int:
        return self.task.dur

    def as_row(self) -> dict[str, Any]:
        return {
            "ts": self.task.ts,
            "dur_us": self.task.dur,
            "origin": self.origin,
            "evidence": self.evidence,
            "src_file": self.src_file,
            "src_func": self.src_func,
            "ran_js": self.ran_js,
            "task_type": self.task_type,
            "queue_name": self.queue_name,
        }


@dataclass
class Classification:
    tasks: list[ClassifiedTask] = field(default_factory = list)
    by_origin_us: dict[str, int] = field(default_factory = dict)
    by_origin_count: dict[str, int] = field(default_factory = dict)
    total_us: int = 0

    @property
    def unclassified_pct(self) -> float:
        if self.total_us <= 0:
            return 0.0
        return 100.0 * self.by_origin_us.get(UNCLASSIFIED, 0) / self.total_us

    def assert_named(self, threshold_pct: float = UNCLASSIFIED_FAIL_PCT) -> None:
        pct = self.unclassified_pct
        if pct > threshold_pct:
            worst = sorted(
                (c for c in self.tasks if c.origin == UNCLASSIFIED),
                key = lambda c: -c.dur_us,
            )[:5]
            detail = "; ".join(
                f"{c.dur_us / 1000:.1f}ms from {c.src_file or '?'}::{c.src_func or '?'}"
                f" [task_type={c.task_type or 'absent'} queue={c.queue_name or 'absent'}]"
                for c in worst
            )
            raise CellFailure(
                "unclassified_task_pct",
                f"{pct:.2f}% of main-thread task time has no known origin "
                f"(limit {threshold_pct}%). Worst: {detail}",
            )

    def summary(self) -> dict[str, Any]:
        return {
            "total_task_ms": self.total_us / 1000.0,
            "unclassified_task_pct": round(self.unclassified_pct, 4),
            "by_origin_ms": {k: round(v / 1000.0, 3) for k, v in sorted(self.by_origin_us.items())},
            "by_origin_count": dict(sorted(self.by_origin_count.items())),
        }

    def tasks_of(self, origin: str) -> list[ClassifiedTask]:
        return [c for c in self.tasks if c.origin == origin]

    def windows_of(self, origin: str) -> list[tuple[int, int]]:
        """(start, end) microsecond windows, for slicing a CPU profile."""
        return [(c.task.ts, c.task.end) for c in self.tasks_of(origin)]


def _nested_names(task: Task) -> set[str]:
    return {t.name for t in walk_within_task(task) if t is not task}


def _event_dispatch_types(task: Task) -> set[str]:
    out: set[str] = set()
    for t in walk_within_task(task):
        if t.name == "EventDispatch":
            data = (t.args or {}).get("data") or {}
            typ = data.get("type")
            if isinstance(typ, str):
                out.add(typ)
    return out


def classify_task(task: Task) -> ClassifiedTask:
    posted = join_posted_from(task)
    src_file = str(posted.get("src_file", "") or "")
    src_func = str(posted.get("src_func", "") or "")
    names = _nested_names(task)
    types = _event_dispatch_types(task)
    ran_js = bool(names & _JS_EVENTS)

    labels = scheduler_labels(task)
    task_type = labels.get("task_type", "")
    queue_name = labels.get("queue_name", "")

    def done(origin: str, evidence: str) -> ClassifiedTask:
        return ClassifiedTask(
            task = task,
            origin = origin,
            evidence = evidence,
            src_file = src_file,
            src_func = src_func,
            ran_js = ran_js,
            task_type = task_type,
            queue_name = queue_name,
        )

    # 0. Blink's own label, when the `scheduler` category recorded one. It outranks everything below
    # because it is the scheduler stating what it dispatched. Two exceptions are handled first: a V8
    # task queue that ran no JS is GC or compilation bookkeeping while one that DID run JS is a real
    # JS task, and a compositor-queue task with a nested input dispatch is input.
    if task_type:
        if types & _INPUT_EVENT_TYPES:
            return done(
                INPUT,
                f"task_type:{task_type}+EventDispatch:{sorted(types & _INPUT_EVENT_TYPES)[0]}",
            )
        for needle, origin in _TASK_TYPE_RULES:
            if needle in task_type:
                if origin == GC and ran_js:
                    continue
                return done(origin, f"task_type:{task_type}")

    # 1. Input first: a keystroke that also fires a timer is still a keystroke, and input latency is what a user feels.
    if types & _INPUT_EVENT_TYPES:
        return done(INPUT, f"EventDispatch:{sorted(types & _INPUT_EVENT_TYPES)[0]}")
    if "main_thread_event_queue.cc" in src_file:
        return done(INPUT, "posted_from:main_thread_event_queue")

    # 2. GC before everything else that could contain it, because a major GC posted from the
    # incremental marking job is not a timer task even when it runs inside one.
    gc_names = {
        n
        for n in names
        if n.startswith("V8.GC") or n in ("MajorGC", "MinorGC", "BlinkGC.AtomicPhase")
    }
    if any(marker in src_file for marker in ("v8/src/heap/", "platform/heap/")):
        return done(GC, f"posted_from:{src_file.rsplit('/', 1)[-1]}")
    if gc_names and not ran_js:
        return done(GC, f"nested:{sorted(gc_names)[0]}")

    # 3. Rendering lifecycle.
    if names & _RAF_EVENTS or "ScheduledActionSendBeginMainFrame" in src_func:
        return done(RAF, "frame_lifecycle")

    # 4. Idle.
    if any(m in src_func or m in task.name for m in _IDLE_MARKERS) or "RunIdleTask" in names:
        return done(IDLE, "idle_queue")

    # 5. Timers.
    if "TimerFire" in names:
        return done(TIMER, "nested:TimerFire")
    if "dom_timer.cc" in src_file:
        return done(TIMER, "posted_from:dom_timer")

    # 6. Network / SSE. Checked before message-channel because both arrive over a mojo pipe and only
    # the resource events tell them apart.
    if names & _RESOURCE_EVENTS:
        return done(NETWORK, f"nested:{sorted(names & _RESOURCE_EVENTS)[0]}")
    if types & _NETWORK_EVENT_TYPES and "connector.cc" in src_file:
        return done(NETWORK, f"EventDispatch:{sorted(types & _NETWORK_EVENT_TYPES)[0]}")

    # 7. Message channel, i.e. the React scheduler. Blink dispatches a `MessagePort` message as a
    # mojo connector task, so the discriminator is that page script ran and nothing network-shaped
    # happened.
    if ran_js and (
        "connector.cc" in src_file
        or "message_port" in src_file
        or "interface_endpoint_client.cc" in src_file
    ):
        return done(MESSAGE_CHANNEL, "mojo_pipe_running_page_script")

    # 8. The devtools / browser IPC channel running page script: what `Runtime.evaluate` and `Page.*`
    # look like from inside the renderer, so it is measurement cost. Named so it can be watched for
    # correlation with the treatment; never subtracted.
    if ran_js and any(
        f in src_file for f in ("ipc_mojo_bootstrap.cc", "simple_watcher.cc", "mojo_bootstrap")
    ):
        return done(AGENT_IPC, f"posted_from:{src_file.rsplit('/', 1)[-1]}")

    # 9. Browser bookkeeping, and ONLY when no page script ran inside. This guard is what stops the
    # class becoming the residual under a new name.
    if not ran_js and any(f in src_file for f in _BROWSER_INTERNAL_FILES):
        return done(BROWSER_INTERNAL, f"posted_from:{src_file.rsplit('/', 1)[-1]}")

    for needle, func_needle, origin in _POSTED_FROM_RULES:
        if needle in src_file and (not func_needle or func_needle in src_func):
            return done(origin, f"posted_from:{needle}")

    # 10. Last resort before giving up: the queue the task sat on. Weaker than the task type, because
    # a queue carries a mix, but still read from the scheduler rather than guessed.
    for needle, origin in _QUEUE_NAME_RULES:
        if needle == queue_name:
            if origin == GC and ran_js:
                continue
            return done(origin, f"queue_name:{queue_name}")

    return done(
        UNCLASSIFIED,
        f"no rule for {src_file or 'unknown'}::{src_func or 'unknown'}"
        + (" (ran JS)" if ran_js else ""),
    )


def classify_thread(trace: Trace, thread: Thread | None = None) -> Classification:
    th = thread if thread is not None else trace.renderer_main()
    result = Classification()
    for task in trace.run_tasks(th):
        c = classify_task(task)
        result.tasks.append(c)
        result.by_origin_us[c.origin] = result.by_origin_us.get(c.origin, 0) + c.dur_us
        result.by_origin_count[c.origin] = result.by_origin_count.get(c.origin, 0) + 1
        result.total_us += c.dur_us
    return result


def cross_check_task_duration(
    classification: Classification,
    cdp_task_duration_s: float,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Summed trace `RunTask` must agree with `Performance.getMetrics.TaskDuration`.

    Two independent accountings of the same quantity. If they disagree the trace
    is missing tasks or the metrics window did not line up with the trace
    window, and either way nothing derived from either is safe to quote.
    """
    trace_s = classification.total_us / 1e6
    if cdp_task_duration_s <= 0:
        raise CellFailure("task_duration_zero", "Performance.getMetrics reported TaskDuration <= 0")
    drift = abs(trace_s - cdp_task_duration_s) / cdp_task_duration_s
    report = {
        "trace_run_task_s": trace_s,
        "cdp_task_duration_s": cdp_task_duration_s,
        "drift": drift,
        "tolerance": tolerance,
    }
    if drift > tolerance:
        raise CellFailure(
            "task_duration_mismatch",
            f"trace RunTask total {trace_s * 1000:.1f} ms vs CDP TaskDuration "
            f"{cdp_task_duration_s * 1000:.1f} ms = {drift * 100:.1f}% apart, "
            f"above {tolerance * 100:.0f}%",
        )
    return report


def origin_windows(classification: Classification, origins: Iterable[str]) -> list[tuple[int, int]]:
    wanted = set(origins)
    return [(c.task.ts, c.task.end) for c in classification.tasks if c.origin in wanted]
