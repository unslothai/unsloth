# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Join RunTask to ThreadControllerImpl::RunTask by interval containment, not timestamp equality."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

from . import CellFailure

# Only complete (`X`) events form the task tree; other phases are carried but never nested.
_PHASE_COMPLETE = "X"
_PHASE_BEGIN = "B"
_PHASE_END = "E"

# Equal `ts` and `dur` give no ordering, so pin `RunTask` outside `ThreadControllerImpl::RunTask`.
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
        """Clamped at zero: a child longer than its parent in a malformed trace would make self time
        negative."""
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

    @classmethod
    def from_json_text(cls, text: str) -> "Trace":
        """Tracing emits an object with traceEvents; DevTools-saved traces are bare arrays, so both
        parse."""
        text = text.strip()
        if not text:
            raise CellFailure("trace_empty", "trace stream contained no bytes")
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            # A truncated document is a cut-short stream: a failed cell, never a short trace.
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
        """Transparently gunzips .gz paths, since checked-in traces are megabytes of repetitive JSON."""
        p = str(path)
        if p.endswith(".gz"):
            import gzip
            with gzip.open(p, "rt", encoding = "utf-8") as fh:
                return cls.from_json_text(fh.read())
        with open(p, "r", encoding = "utf-8") as fh:
            return cls.from_json_text(fh.read())

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
        """Taken from the Profile event, the thread whose stacks exist; falls back to CrRendererMain."""
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
    """Unmatched B is dropped and unmatched E ignored: an invented end timestamp would invent duration."""
    complete: list[Task] = []
    open_stack: list[dict[str, Any]] = []
    for e in events:
        ph = e.get("ph")
        if ph == _PHASE_COMPLETE:
            dur = e.get("dur")
            if dur is None:
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
        # An improperly nested event is treated as a sibling to keep self-time arithmetic sound.
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
    """Joined by interval containment, not timestamp equality; empty when toplevel was not recorded."""
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
    """Stops at nested RunTask so an outer task does not absorb the origin of tasks it contains."""
    yield task
    stack = list(task.children)
    while stack:
        node = stack.pop()
        if node.name == "RunTask":
            continue
        yield node
        stack.extend(node.children)
