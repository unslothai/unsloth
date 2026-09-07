# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drive every registered surface once and take a parity digest of each.

THE ONE RULE. A surface that was not reached records a REASON. Never a bare zero, never a missing
row, and never a digest of whatever happened to be on screen instead. The failure this whole tool
is built against is a check that cannot tell "did not run" from "passed", and a sweep is unusually
good at producing exactly that: navigate to /hub, fail to render, digest the chat page that is
still mounted underneath, and report a pass. So every row carries `reached`, `reason`, the settle
observation that decided it, and the root the digest was actually taken from.

WHY THE SWEEP RUNS AGAINST AN EMPTY CHAT. The known state is a fresh `/chat`. Several surface
roots -- the shell, the sidebar, the active route container's siblings -- contain the keep-alive
chat page, so a sweep run against a loaded thread would carry that thread into the digest of every
other surface and any thread difference would flip all of them at once. An empty chat makes the
surface digests about the surfaces.

WHAT THE DIGEST IS. `window.__sb.parity.capture()`, the same function and the same normalisation
the film uses at the close of every action window, pointed at the surface's own root by
surfaces.js. Same function on purpose: a surface digest and an action digest that were produced by
two implementations could not be compared, and comparing them is the point.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from . import surfaces as registry

#: How long a settle condition is given before the surface is recorded as unreached. Generous: the
#: settings panels are lazily imported chunks and the hub's first paint waits on a network round
#: trip, and a sweep that calls those failures is a sweep nobody believes.
SETTLE_TIMEOUT_MS = 8000
SETTLE_POLL_MS = 150

#: A per-surface ceiling covering reach, settle and capture together, so a reach that navigates
#: into a redirect loop cannot hold the sweep open indefinitely.
SURFACE_BUDGET_MS = 25_000


def _log_noop(_msg: str) -> None:
    pass


class _Driver:
    """The step interpreter. One place that touches the browser, so the registry stays declarative
    and the unit tests can read it with no browser installed."""

    def __init__(self, page: Any, base_url: str, log: Callable[[str], None]) -> None:
        self.page = page
        self.base_url = base_url.rstrip("/")
        self.log = log

    def run(self, steps: tuple) -> Optional[str]:
        """Execute steps. Returns None on success, or the reason the first one failed."""
        for step in steps:
            verb, args = step[0], step[1:]
            try:
                self._step(verb, args)
            except Exception as exc:  # noqa: BLE001
                return (
                    f"step {verb!r}{list(args)!r} failed: "
                    f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:200]}"
                )
        return None

    def _step(self, verb: str, args: tuple) -> None:
        page = self.page
        if verb == "goto":
            page.goto(f"{self.base_url}{args[0]}", wait_until = "domcontentloaded", timeout = 60_000)
        elif verb == "click":
            # A REAL mouse click through the driver, not element.click(): Radix menus open on pointerdown and
            # a synthetic click never opens them, which reads downstream as a menu that opened in zero
            # milliseconds. The same trap dom.js documents for the film's actions.
            page.click(args[0], timeout = 6000)
        elif verb == "click_if":
            el = page.query_selector(args[0])
            if el is not None:
                el.click(timeout = 6000)
        elif verb == "hover":
            page.hover(args[0], timeout = 6000)
        elif verb == "press":
            page.keyboard.press(args[0])
        elif verb == "fill":
            page.fill(args[0], args[1], timeout = 6000)
        elif verb == "wait":
            page.wait_for_timeout(int(args[0]))
        else:  # pragma: no cover
            raise ValueError(f"unknown step verb {verb!r}")

    def settle(self, spec: Optional[dict], deadline: float) -> dict:
        """Poll the settle condition. Returns what was observed, never a bare boolean."""
        if spec is None:
            return {"settled": True, "detail": "no settle condition declared"}
        last = {"ok": False, "detail": "never evaluated"}
        while time.monotonic() < deadline:
            try:
                last = self.page.evaluate("(spec) => window.__sb.surfaces.settled(spec)", spec)
            except Exception as exc:  # noqa: BLE001
                last = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
            if last.get("ok"):
                return {"settled": True, "detail": last.get("detail"), "spec": spec}
            self.page.wait_for_timeout(SETTLE_POLL_MS)
        return {"settled": False, "detail": last.get("detail"), "spec": spec}

    def capture(self, root: tuple) -> dict:
        try:
            return self.page.evaluate("(root) => window.__sb.surfaces.capture(root)", list(root))
        except Exception as exc:  # noqa: BLE001
            return {
                "parity_attempted": False,
                "reason": f"the digest could not be taken: {type(exc).__name__}: {exc}",
            }

    def facts(self, root: tuple) -> dict:
        try:
            return self.page.evaluate("(root) => window.__sb.surfaces.facts(root)", list(root))
        except Exception as exc:  # noqa: BLE001
            return {"facts_attempted": False, "reason": f"{type(exc).__name__}: {exc}"}

    def is_clean(self) -> dict:
        try:
            return self.page.evaluate("() => window.__sb.surfaces.isClean()")
        except Exception as exc:  # noqa: BLE001
            return {"clean_attempted": False, "reason": f"{type(exc).__name__}: {exc}"}


def _row(surface: registry.Surface, cell_id: Optional[str]) -> dict:
    """A surface row with every mandatory field already present and honest.

    Built up front rather than assembled on the success path, so an exception anywhere below
    leaves a row that says it did not run instead of a row that is missing.
    """
    return {
        "row_type": "surface",
        "cell_id": cell_id,
        "surface": surface.id,
        "group": surface.group,
        "title": surface.title,
        "reached": False,
        "reason": "the sweep did not get to this surface",
        "conditional": surface.conditional,
        "also_in_film": surface.also_in_film,
        # Carried on the ROW, not only in the registry, so a reader of a payload can tell a digest that is
        # a parity signal from one that moves on its own without this source tree in front of them.
        "volatile": surface.volatile,
        "parity": {"parity_attempted": False, "reason": "not captured"},
        "settle": None,
        "facts": None,
        "restored": None,
        "sweep_attempted": True,
        "reach_ms": None,
    }


def sweep(
    page: Any,
    base_url: str,
    log: Callable[[str], None] = _log_noop,
    only: Optional[list] = None,
    cell_id: Optional[str] = None,
    recorder: Any = None,
    settle_timeout_ms: Optional[int] = None,
    surface_budget_ms: Optional[int] = None,
) -> tuple[list[dict], dict]:
    """Drive every registered surface once. Returns `(rows, manifest)`.

    `recorder`, when given, receives each row as it is produced, so a sweep that dies halfway
    still leaves the surfaces it did reach in the payload.

    The two timeouts are arguments rather than constants only so the unit tests can drive the
    unreached path without paying eight real seconds per surface for a page that was never going
    to settle. Every caller in the tool uses the defaults.
    """
    settle_ms = SETTLE_TIMEOUT_MS if settle_timeout_ms is None else settle_timeout_ms
    budget_ms = SURFACE_BUDGET_MS if surface_budget_ms is None else surface_budget_ms
    registry.validate_registry()
    driver = _Driver(page, base_url, log)
    entries = registry.surfaces()
    if only:
        wanted = set(only)
        missing = sorted(wanted - set(registry.surface_ids()))
        if missing:
            raise registry.RegistryError(f"no such surface(s): {missing}")
        entries = [s for s in entries if s.id in wanted]

    # Does parity.capture() honour a moved root? If not, every surface digest is the same page-wide
    # reading and the sweep's forty passes mean nothing. Asked ONCE, up front, and the answer is
    # carried on every row.
    home = driver.run(registry.HOME)
    scoping = {
        "scoped": False,
        "scoping_attempted": False,
        "reason": "the known state could not be reached, so scoping was never probed",
    }
    if home is None:
        try:
            scoping = page.evaluate("() => window.__sb.surfaces.probeScoping()")
        except Exception as exc:  # noqa: BLE001
            scoping = {
                "scoped": False,
                "scoping_attempted": False,
                "reason": f"the scoping probe raised {type(exc).__name__}: {exc}",
            }
    if not scoping.get("scoped"):
        log(f"  surface digests are NOT scoped to their surface: {scoping.get('reason')}")

    rows: list[dict] = []
    for surface in entries:
        row = _row(surface, cell_id)
        row["scoping"] = scoping
        started = time.monotonic()
        deadline = started + budget_ms / 1000

        # From the KNOWN STATE, never from wherever the previous surface left the app: a reach that works
        # only because the last surface left a menu open breaks the first time the order changes, and
        # silently.
        reset = driver.run(registry.HOME)
        if reset is not None:
            row["reason"] = f"the known state could not be restored before this surface: {reset}"
            rows.append(row)
            _emit(recorder, row)
            log(f"    {surface.id}: NOT REACHED -- {row['reason']}")
            continue

        failed = driver.run(surface.reach)
        if failed is not None:
            row["reason"] = f"the reach failed: {failed}"
            # The facts are still recorded. Where the app ACTUALLY ended up is the first thing a reader of a
            # failed reach needs, and it is gone by the time anyone looks.
            row["facts"] = driver.facts(surface.root)
            row["reach_ms"] = round((time.monotonic() - started) * 1000, 1)
            rows.append(row)
            _emit(recorder, row)
            log(f"    {surface.id}: NOT REACHED -- {row['reason']}")
            _recover(driver, log)
            continue

        # Whichever runs out first: the settle window or what is left of the surface's budget.
        settled = driver.settle(surface.settle, min(deadline, time.monotonic() + settle_ms / 1000))
        row["settle"] = settled
        row["facts"] = driver.facts(surface.root)
        row["reach_ms"] = round((time.monotonic() - started) * 1000, 1)
        if not settled.get("settled"):
            row["reason"] = (
                f"the reach ran but the surface never settled: " f"{settled.get('detail')}"
            )
            rows.append(row)
            _emit(recorder, row)
            log(f"    {surface.id}: NOT REACHED -- {row['reason']}")
            _recover(driver, log)
            continue

        parity = driver.capture(surface.root)
        row["parity"] = parity
        if not parity.get("parity_attempted"):
            # Reached, but no digest. Recorded as unreached FOR PARITY PURPOSES, because a surface with no
            # digest contributes nothing to the parity claim and counting it as covered is the inflation this
            # file avoids.
            row["reason"] = f"the surface rendered but no digest was taken: {parity.get('reason')}"
            rows.append(row)
            _emit(recorder, row)
            log(f"    {surface.id}: NO DIGEST -- {parity.get('reason')}")
            _recover(driver, log)
            continue

        row["reached"] = True
        row["reason"] = None
        rows.append(row)
        log(
            f"    {surface.id}: digest {parity.get('digest')} "
            f"({parity.get('chars')} chars from {parity.get('root_selector')}, "
            f"{(row['facts'] or {}).get('root_elements')} elements)"
        )

        restored = driver.run(surface.restore)
        clean = driver.is_clean()
        row["restored"] = restored is None
        row["restore_state"] = clean
        if restored is not None:
            row["restore_reason"] = restored
            _recover(driver, log)
        elif (clean.get("open_dialogs") or 0) or (clean.get("open_menus") or 0):
            # The declared restore ran and the app is still dirty. Said out loud: the next surface would
            # otherwise be reached from a state nobody declared and its digest would include a leftover
            # overlay.
            row["restore_reason"] = (
                f"the restore ran but left {clean.get('open_dialogs')} dialog(s) and "
                f"{clean.get('open_menus')} menu(s) open"
            )
            row["restored"] = False
            _recover(driver, log)
        _emit(recorder, row)

    manifest = build_manifest(rows, entries, scoping)
    return rows, manifest


def _emit(recorder: Any, row: dict) -> None:
    if recorder is None:
        return
    try:
        recorder.emit(dict(row))
    except Exception:  # noqa: BLE001
        # A recorder that rejects a row must not cost the sweep the rest of its surfaces. The row is still
        # in the returned list and in the manifest.
        pass


def _recover(driver: _Driver, log: Callable[[str], None]) -> None:
    """Hard reset after a failure or a dirty restore."""
    driver.run(registry.ESCAPE_OUT + registry.HOME)
    clean = driver.is_clean()
    if (clean.get("open_dialogs") or 0) or (clean.get("open_menus") or 0):
        log(f"    the app is still not at the known state after a reset: {clean}")


def build_manifest(rows: list, entries: list, scoping: dict) -> dict:
    """The coverage manifest: what was swept, what was not, and what is out of reach by design."""
    reached = [r for r in rows if r.get("reached")]
    failed = [r for r in rows if not r.get("reached")]
    # A conditional surface that did not render is not a hole in the registry but a property of this
    # installation. Kept separate so the coverage figure is not quietly deflated by a host with no
    # GPU, nor inflated by counting it as covered.
    conditional_misses = [r for r in failed if r.get("conditional")]
    hard_misses = [r for r in failed if not r.get("conditional")]
    registered = len(entries)
    return {
        "manifest_attempted": True,
        "registered": registered,
        "swept": len(rows),
        "reached": len(reached),
        "not_reached": len(failed),
        "not_reached_conditional": len(conditional_misses),
        "not_reached_hard": len(hard_misses),
        # Against the surfaces that COULD have rendered on this host. Reported next to the raw count,
        # never instead of it.
        "coverage_pct": (
            round(100.0 * len(reached) / (registered - len(conditional_misses)), 1)
            if registered - len(conditional_misses) > 0
            else None
        ),
        "coverage_pct_of_registered": (
            round(100.0 * len(reached) / registered, 1) if registered else None
        ),
        "digests_scoped": bool(scoping.get("scoped")),
        "digests_scoped_reason": scoping.get("reason"),
        "failures": [
            {
                "id": r["surface"],
                "group": r["group"],
                "reason": r["reason"],
                "conditional": r.get("conditional"),
            }
            for r in failed
        ],
        "restore_failures": [
            {"id": r["surface"], "reason": r.get("restore_reason")}
            for r in rows
            if r.get("restored") is False
        ],
        # Reached, digested, and NOT a parity signal. Counted separately because what matters to somebody
        # comparing two arms is how many surfaces can carry a verdict, which is `comparable`, not
        # `reached`.
        "volatile": len([r for r in reached if r.get("volatile")]),
        "comparable": len([r for r in reached if not r.get("volatile")]),
        "volatile_surfaces": [
            {"id": r["surface"], "mechanism": r["volatile"]} for r in reached if r.get("volatile")
        ],
        "by_group": _by_group(rows),
        "known_uncovered": [dict(u) for u in registry.KNOWN_UNCOVERED],
        "known_uncovered_count": len(registry.KNOWN_UNCOVERED),
    }


def _by_group(rows: list) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        bucket = out.setdefault(
            r["group"], {"reached": 0, "not_reached": 0, "group_attempted": True}
        )
        bucket["reached" if r.get("reached") else "not_reached"] += 1
    return out


def render_manifest(manifest: dict) -> str:
    """The manifest as text. Reads top-down: the number, then everything that qualifies it."""
    lines = []
    lines.append("UI SURFACE COVERAGE")
    lines.append("=" * 78)
    pct = manifest.get("coverage_pct")
    lines.append(
        f"{manifest['reached']} of {manifest['registered']} registered surfaces reached"
        + (f" ({pct}% of those this host can render)" if pct is not None else "")
    )
    if not manifest.get("digests_scoped"):
        lines.append("")
        lines.append(
            "  WARNING: the digests are NOT scoped to their surface -- "
            f"{manifest.get('digests_scoped_reason')}"
        )
        lines.append(
            "  Every digest below is a whole-page reading, so two different surfaces can agree"
        )
        lines.append("  for reasons that have nothing to do with either of them.")
    lines.append(
        f"{manifest.get('comparable', 0)} of those carry a comparable digest; "
        f"{manifest.get('volatile', 0)} move between two runs of the same build"
    )
    lines.append("")
    lines.append(f"{'group':<12} {'reached':>8} {'not reached':>12}")
    lines.append("-" * 78)
    for group in sorted(manifest.get("by_group", {})):
        got = manifest["by_group"][group]
        lines.append(f"{group:<12} {got['reached']:>8} {got['not_reached']:>12}")
    if manifest.get("failures"):
        lines.append("")
        lines.append("NOT REACHED (each with the reason, never a silent gap)")
        lines.append("-" * 78)
        for f in manifest["failures"]:
            tag = " [conditional]" if f.get("conditional") else ""
            lines.append(f"  {f['id']}{tag}")
            lines.append(f"      {f['reason']}")
            if f.get("conditional"):
                lines.append(f"      precondition: {f['conditional']}")
    if manifest.get("restore_failures"):
        lines.append("")
        lines.append("RESTORE DID NOT RETURN THE APP TO THE KNOWN STATE")
        lines.append("-" * 78)
        for f in manifest["restore_failures"]:
            lines.append(f"  {f['id']}: {f['reason']}")
    if manifest.get("volatile_surfaces"):
        lines.append("")
        lines.append("REACHED BUT NOT COMPARABLE (the digest moves between two runs of one build)")
        lines.append("-" * 78)
        for entry in manifest["volatile_surfaces"]:
            lines.append(f"  {entry['id']}")
            lines.append(f"      {entry['mechanism']}")
    lines.append("")
    lines.append(
        f"KNOWN UNCOVERED ({manifest['known_uncovered_count']} surfaces, out of reach "
        "by mechanism, not by oversight)"
    )
    lines.append("-" * 78)
    for entry in manifest.get("known_uncovered", []):
        lines.append(f"  {entry['id']}: {entry['title']}")
        lines.append(f"      {entry['reason']}")
    return "\n".join(lines)


__all__ = ["sweep", "build_manifest", "render_manifest", "SETTLE_TIMEOUT_MS", "SURFACE_BUDGET_MS"]
