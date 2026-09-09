# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parse a Chrome trace into a per-thread task tree.

All timestamps in a Chrome JSON trace are MICROSECONDS on the monotonic clock
(`ts`), with `dur` also in microseconds for complete (`ph: "X"`) events. Nothing
in this module rescales; if a caller wants milliseconds it divides, so that a
unit mistake is a visible division and not a hidden constant.

Two facts about real traces that this module encodes because both were observed
in a captured trace rather than assumed:

1. `RunTask` (category `disabled-by-default-devtools.timeline`) has EMPTY args.
   It tells you a task ran and how long it took, and nothing whatsoever about
   where it came from. The origin lives on a SIBLING event,
   `ThreadControllerImpl::RunTask` (category `toplevel`), which carries
   `src_file` / `src_func` / `src_line` naming the code that POSTED the task.
   The two nest, they are 1:1 on a thread, and their `ts` values differ by a
   microsecond or two, so they must be joined by interval containment and not by
   timestamp equality. A naive equality join matched 79% of tasks on a real
   capture, and the 21% it dropped is not a random sample.

2. `ProfileChunk` events are emitted on the V8 profiler's own thread
   (`v8:ProfEvntProc`), NOT on the thread being profiled. The profiled thread is
   named by the `Profile` event's own `pid`/`tid`. Filtering chunks by the
   renderer main thread id yields zero samples and looks exactly like "the CPU
   profiler was not enabled".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

from . import CellFailure

# Complete-duration events are the only phase that forms the task tree. Async (`b`/`e`/`n`), flow
# (`s`/`t`/`f`), instant (`I`/`R`), sample (`P`), counter (`C`) and metadata (`M`) events are
# carried alongside but never nested, since their timestamps do not describe a stack.
_PHASE_COMPLETE = "X"
_PHASE_BEGIN = "B"
_PHASE_END = "E"

# When two events share both `ts` and `dur` the trace gives no ordering. On the main thread the
# devtools view (`RunTask`) is conceptually the outer frame and the scheduler view
# (`ThreadControllerImpl::RunTask`) the inner one, so pin that order rather than letting dict
# ordering decide.
_OUTERMOST_FIRST = {
    "RunTask": 0,
    "ThreadControllerImpl::RunTask": 1,
}


@dataclass
class Task:
    """One complete-duration trace event plus the events nested inside it."""

    name: str
    cat: str
    ts: int
    dur: int
    pid: int
    tid: int
    args: dict[str, Any] = field(default_factory = dict)
    children: list["Task"] = field(default_factory = list)
    parent: "Task | None" = field(default = None, repr = False)

    @property
    def end(self) -> int:
        return self.ts + self.dur

    @property
    def self_dur(self) -> int:
        """Duration minus the time attributed to nested events.

        Children of a complete event do not overlap each other in a well-formed
        trace, so a plain sum is correct. Clamped at zero because a malformed
        trace can report a child longer than its parent and a negative self time
        would poison every downstream sum silently.
        """
        return max(0, self.dur - sum(c.dur for c in self.children))

    def walk(self) -> Iterator["Task"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def descendants_named(self, *names: str) -> list["Task"]:
        wanted = set(names)
        return [t for t in self.walk() if t is not self and t.name in wanted]

    def find_child_task(self, name: str) -> "Task | None":
        for t in self.walk():
            if t is not self and t.name == name:
                return t
        return None


@dataclass
class Thread:
    pid: int
    tid: int
    name: str
    roots: list[Task] = field(default_factory = list)
    # Every event on this thread, including phases that do not nest.
    events: list[dict[str, Any]] = field(default_factory = list)

    def tasks_named(self, name: str) -> list[Task]:
        return [t for r in self.roots for t in r.walk() if t.name == name]

    def wall_span_us(self) -> tuple[int, int]:
        if not self.roots:
            return (0, 0)
        return (self.roots[0].ts, max(r.end for r in self.roots))


class Trace:
    """A loaded Chrome trace, indexed by thread."""

    def __init__(
        self,
        events: Sequence[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.events: list[dict[str, Any]] = list(events)
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._thread_names: dict[tuple[int, int], str] = {}
        for e in self.events:
            if e.get("name") == "thread_name" and e.get("ph") == "M":
                key = (e.get("pid"), e.get("tid"))
                name = (e.get("args") or {}).get("name")
                if isinstance(name, str) and None not in key:
                    self._thread_names[key] = name  # type: ignore[index]
        self._threads: dict[tuple[int, int], Thread] = {}

    # loading

    @classmethod
    def from_json_text(cls, text: str) -> "Trace":
        """Load the exact wire format `Tracing` emits.

        `transferMode: ReturnAsStream` with `streamFormat: json` produces an
        OBJECT, `{"traceEvents": [...], "metadata": {...}}`, not the bare array
        that the Trace Event Format also permits. Both are accepted here because
        traces saved by the DevTools UI use the array form.
        """
        text = text.strip()
        if not text:
            raise CellFailure("trace_empty", "trace stream contained no bytes")
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            # A truncated JSON document is the signature of a drained stream that was cut short: a failed cell,
            # never a short trace.
            raise CellFailure(
                "trace_truncated",
                f"trace JSON did not parse ({exc}); {len(text)} bytes drained",
            ) from exc
        if isinstance(doc, list):
            return cls(doc, {})
        if isinstance(doc, dict) and isinstance(doc.get("traceEvents"), list):
            return cls(doc["traceEvents"], doc.get("metadata") or {})
        raise CellFailure(
            "trace_shape",
            f"unrecognised trace document keys: {sorted(doc)[:8] if isinstance(doc, dict) else type(doc)}",
        )

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> "Trace":
        """Load a trace from disk, transparently gunzipping a `.gz`.

        Checked-in fixtures are gzipped because a trace of any useful length is
        megabytes of highly repetitive JSON.
        """
        p = str(path)
        if p.endswith(".gz"):
            import gzip
            with gzip.open(p, "rt", encoding = "utf-8") as fh:
                return cls.from_json_text(fh.read())
        with open(p, "r", encoding = "utf-8") as fh:
            return cls.from_json_text(fh.read())

    # ---------------------------------------------------------------- threads

    def thread_name(self, pid: int, tid: int) -> str:
        return self._thread_names.get((pid, tid), "")

    def thread(self, pid: int, tid: int) -> Thread:
        key = (pid, tid)
        cached = self._threads.get(key)
        if cached is not None:
            return cached
        own = [e for e in self.events if e.get("pid") == pid and e.get("tid") == tid]
        th = Thread(pid = pid, tid = tid, name = self.thread_name(pid, tid), events = own)
        th.roots = build_tree(own)
        self._threads[key] = th
        return th

    def profiled_thread(self) -> tuple[int, int]:
        """The thread the V8 CPU profiler attached to, read from `Profile`.

        This is the correct anchor for the renderer main thread whenever the CPU
        profiler category is on, because it is the thread whose stacks we have.
        Falls back to the thread named `CrRendererMain`.
        """
        for e in self.events:
            if e.get("name") == "Profile" and e.get("cat") == "disabled-by-default-v8.cpu_profiler":
                return (int(e["pid"]), int(e["tid"]))
        for (pid, tid), name in self._thread_names.items():
            if name == "CrRendererMain":
                return (pid, tid)
        raise CellFailure(
            "no_renderer_thread",
            "trace has neither a v8 Profile event nor a CrRendererMain thread_name",
        )

    def renderer_main(self) -> Thread:
        pid, tid = self.profiled_thread()
        return self.thread(pid, tid)

    # ------------------------------------------------------------------ joins

    def run_tasks(self, thread: Thread | None = None) -> list[Task]:
        """Top-level `RunTask` events on a thread, outermost only."""
        th = thread if thread is not None else self.renderer_main()
        out: list[Task] = []
        for root in th.roots:
            for t in root.walk():
                if t.name == "RunTask" and not _has_runtask_ancestor(t):
                    out.append(t)
        out.sort(key = lambda t: t.ts)
        return out

    def total_run_task_ms(self, thread: Thread | None = None) -> float:
        return sum(t.dur for t in self.run_tasks(thread)) / 1000.0


def _has_runtask_ancestor(task: Task) -> bool:
    p = task.parent
    while p is not None:
        if p.name == "RunTask":
            return True
        p = p.parent
    return False


def build_tree(events: Iterable[dict[str, Any]]) -> list[Task]:
    """Nest complete-duration events on ONE thread into a forest.

    `B`/`E` pairs are folded into synthetic complete events first so that a
    trace which uses the begin/end encoding parses identically. Unmatched `B`
    events are dropped rather than guessed at, and an unmatched `E` is ignored,
    because inventing an end timestamp would invent duration.
    """
    complete: list[Task] = []
    open_stack: list[dict[str, Any]] = []
    for e in events:
        ph = e.get("ph")
        if ph == _PHASE_COMPLETE:
            dur = e.get("dur")
            if dur is None:
                # A complete event without `dur` is a zero-width marker.
                dur = 0
            complete.append(
                Task(
                    name = str(e.get("name", "")),
                    cat = str(e.get("cat", "")),
                    ts = int(e["ts"]),
                    dur = int(dur),
                    pid = int(e.get("pid", 0)),
                    tid = int(e.get("tid", 0)),
                    args = dict(e.get("args") or {}),
                )
            )
        elif ph == _PHASE_BEGIN:
            open_stack.append(e)
        elif ph == _PHASE_END:
            if not open_stack:
                continue
            b = open_stack.pop()
            args = dict(b.get("args") or {})
            args.update(e.get("args") or {})
            complete.append(
                Task(
                    name = str(b.get("name", "")),
                    cat = str(b.get("cat", "")),
                    ts = int(b["ts"]),
                    dur = max(0, int(e["ts"]) - int(b["ts"])),
                    pid = int(b.get("pid", 0)),
                    tid = int(b.get("tid", 0)),
                    args = args,
                )
            )

    complete.sort(key = lambda t: (t.ts, -t.dur, _OUTERMOST_FIRST.get(t.name, 50)))

    roots: list[Task] = []
    stack: list[Task] = []
    for t in complete:
        while stack and t.ts >= stack[-1].end:
            stack.pop()
        # An event that starts inside its would-be parent but ends after it is not nested; the trace is
        # inconsistent there, so treat it as a sibling rather than corrupting self-time arithmetic for the
        # whole subtree.
        while stack and t.end > stack[-1].end:
            stack.pop()
        if stack:
            t.parent = stack[-1]
            stack[-1].children.append(t)
        else:
            roots.append(t)
        stack.append(t)
    return roots


def join_posted_from(task: Task) -> dict[str, Any]:
    """Return the `src_file`/`src_func`/`src_line` that POSTED this task.

    Joined by interval containment against the nested
    `ThreadControllerImpl::RunTask`, since the timestamps differ by a couple of
    microseconds and an equality join loses a fifth of all tasks. Returns an
    empty dict when the `toplevel` category was not recorded, which is a
    legitimate state and not an error: the caller then classifies on nested
    evidence alone and reports lower confidence.
    """
    for t in walk_within_task(task):
        if t.name == "ThreadControllerImpl::RunTask":
            a = t.args or {}
            if "src_file" in a or "src_func" in a:
                return {
                    "src_file": a.get("src_file", ""),
                    "src_func": a.get("src_func", ""),
                    "src_line": a.get("src_line"),
                }
    return {}


def walk_within_task(task: Task) -> Iterator[Task]:
    """Walk a task's subtree, stopping at any nested `RunTask` boundary.

    Without the boundary a nested task's scheduler frame or its `TimerFire`
    would be attributed to the outer task, which is how one long task swallows
    the origin of every task it contains.
    """
    yield task
    stack = list(task.children)
    while stack:
        node = stack.pop()
        if node.name == "RunTask":
            continue
        yield node
        stack.extend(node.children)
