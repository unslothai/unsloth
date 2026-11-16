# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Base vs head: does a change make a long chat cost more than its base build does?

playwright_long_thread_gate.py compares a build with itself (40 vs 200 turns), so a cost both
lengths pay alike passes it: #17504 left the viewport blank while a reply streamed under a reader
who had scrolled away, and every within-build ratio stayed green. This runs areas of the area
matrix (_area_matrix.py) on production builds of both refs, ABBA on fresh pages in one browser. An
area fails when head's median main-thread or long-task time is over --max-ratio times base's with no
overlap, or (stream_scroll) when any head reading leaves the viewport blank and no base one does.

  python tests/studio/long_thread_ab.py --base-ref origin/main --head-ref HEAD
  python tests/studio/long_thread_ab.py --base DIST_A --head DIST_B      # prebuilt dists
  python tests/studio/long_thread_ab.py --base-ref X --head-ref X        # A/A noise

Exit 0 no regression, 1 regression, 2 void (a page or area that did not run), 3 not comparable (a
build failed, e.g. the base needs other node_modules).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _area_matrix as areas  # noqa: E402
from _playwright_robust import chromium_launch_args  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio" / "frontend"
SCRATCH = REPO / ".playwright-cli" / "long-thread-ab"
METRICS = (
    "main_ms",
    "blocked_ms",
    "long_tasks",
    "cpu_ms",
    "style_ms",
    "layout_ms",
    "elements",
    "longest_task_ms",
)
# Renderer main thread only (raster/compositor swing with the runner); floors in ms.
GATES = {"main_ms": 50.0, "blocked_ms": 100.0}


def _split_with_blocking(events: list) -> dict[str, dict]:
    """The area matrix's counters plus the part of each main-thread task over 50 ms (Long Tasks):
    total time can fall while a change bunches it into tasks the reader feels."""
    out = _split_trace(events)
    names, windows = {}, {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            names[(e["pid"], e["tid"])] = (e.get("args") or {}).get("name", "")
        elif e.get("name") == "TimeStamp":
            msg = ((e.get("args") or {}).get("data") or {}).get("message") or ""
            if msg.startswith("gate:"):
                _, edge, area = msg.split(":", 2)
                windows.setdefault(area, [None, None])[0 if edge == "begin" else 1] = e["ts"]
    spans = [(w[0], w[1], a) for a, w in windows.items() if None not in w and a in out]
    for _, _, a in spans:
        out[a]["blocked_ms"], out[a]["long_tasks"] = 0.0, 0
    for e in events:
        if (
            e.get("ph") != "X"
            or e.get("cat") != "toplevel"
            or e.get("name") not in ("ThreadControllerImpl::RunTask", "RunTask")
            or names.get((e.get("pid"), e.get("tid"))) != "CrRendererMain"
        ):
            continue
        dur = e.get("dur", 0) / 1000
        if dur <= 50:
            continue
        for b, en, a in spans:
            if b <= e["ts"] <= en:
                out[a]["blocked_ms"] += dur - 50
                out[a]["long_tasks"] += 1
                break
    return out


_split_trace = areas.split_trace
areas.split_trace = _split_with_blocking


def sh(cmd: list[str], cwd: Path) -> str:
    return subprocess.run(
        cmd, cwd = cwd, check = True, capture_output = True, text = True, shell = os.name == "nt"
    ).stdout.strip()


def build_ref(ref: str) -> tuple[Path, str]:
    """Production build of `ref` in a scratch worktree sharing this checkout's node_modules."""
    sha = sh(["git", "rev-parse", f"{ref}^{{commit}}"], REPO)
    out = SCRATCH / f"dist-{sha[:12]}"
    if (out / "index.html").exists():
        return out, sha
    wt = SCRATCH / f"wt-{sha[:12]}"
    if not wt.exists():
        sh(["git", "worktree", "add", "--detach", "--force", str(wt), sha], REPO)
    modules = wt / "studio" / "frontend" / "node_modules"
    if not modules.exists():
        modules.symlink_to(FRONTEND / "node_modules", target_is_directory = True)
    t = time.time()
    try:
        subprocess.run(
            [
                "npx",
                "vite",
                "build",
                "--config",
                "vite.long-thread.config.ts",
                "--outDir",
                str(out),
            ],
            cwd = wt / "studio" / "frontend",
            check = True,
            shell = os.name == "nt",
            stdout = subprocess.DEVNULL,
        )
    finally:
        modules.unlink(missing_ok = True)
        sh(["git", "worktree", "remove", "--force", str(wt)], REPO)
    print(f"built {ref} ({sha[:12]}) in {time.time() - t:.0f} s -> {out}", flush = True)
    return out, sha


def lengthen_stream(chars: int) -> None:
    """Stream about `chars` more characters: the gate's own reply is a few hundred."""
    short = areas.stream_frames

    def frames() -> list[dict]:
        out = short()
        body = [f for f in out if (f.get("choices") or [{}])[0].get("delta", {}).get("content")]
        end = out.index(body[-1])
        line = "Streaming prose for the long reply, sixty characters a frame. "
        extra = [
            {**body[0], "choices": [{**body[0]["choices"][0], "delta": {"content": line}}]}
            for _ in range(max(0, chars // len(line)))
        ]
        return out[:end] + extra + out[end:]

    areas.stream_frames = frames
    areas.STREAM_TIMEOUT_MS = max(
        areas.STREAM_TIMEOUT_MS, 30_000 + 4 * areas.STREAM_MS * len(frames())
    )


COVERAGE = """() => {
  const v = document.querySelector('.aui-thread-viewport');
  if (!v) return null;
  const f = v.getBoundingClientRect();
  const spans = [];
  for (const r of v.querySelectorAll('[data-message-id]')) {
    const b = r.getBoundingClientRect();
    const top = Math.max(b.top, f.top), bottom = Math.min(b.bottom, f.bottom);
    if (bottom > top) spans.push([top, bottom]);
  }
  spans.sort((x, y) => x[0] - y[0]);
  let covered = 0, end = f.top;
  for (const [t, b] of spans) { const s = Math.max(t, end); if (b > s) { covered += b - s; end = b; } }
  return covered / Math.max(1, f.height);
}"""
AFTER_WHEEL = (
    """async ([frames, samples, gap]) => {
  const cover = """
    + COVERAGE
    + """;
  const out = [];
  for (let i = 0; i < frames; i++) await new Promise((r) => requestAnimationFrame(r));
  for (let i = 0; i < samples; i++) {
    const c = cover();
    if (c !== null) out.push(c);
    await new Promise((r) => setTimeout(r, gap));
  }
  return out;
}"""
)
AFTER_FRAMES, AFTER_SAMPLES, AFTER_GAP_MS = 20, 15, 100
STREAM_DONE = (
    """() => !!window.__gateStreamDone && !document.querySelector('.aui-composer-cancel')"""
)
COVERAGE_MIN = 0.5
WHEEL_STEPS, WHEEL_PX, WHEEL_GAP_MS = 12, -600, 60

# #17504: every chunk restarted the window's transition commit. Runs on a page no stream has used.
areas.AREAS["stream_scroll"] = {
    "setup": ("compose", areas.STREAM_PROMPT),
    "open": ("stream_scroll", None),
    "ready": ("stream_done",),
    "close": None,
    "gone": ("settle",),
    "covers": [],
}


areas.AREAS["wheel_sweep"] = {
    "open": ("wheel_sweep", None),
    "ready": ("settle",),
    "close": None,
    "gone": ("settle",),
    "covers": [],
}
SWEEP_UP, SWEEP_DOWN, SWEEP_PX, SWEEP_GAP_MS = 250, 120, 400, 16


class ABPage(areas.Page):
    coverage: list[float]

    def act(self, action) -> bool:
        if action is not None and action[0] == "wheel_sweep":
            page = self.page
            box = page.locator(".aui-thread-viewport").bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            for k in range(SWEEP_UP + SWEEP_DOWN):
                page.mouse.wheel(0, -SWEEP_PX if k < SWEEP_UP else SWEEP_PX)
                page.wait_for_timeout(SWEEP_GAP_MS)
            self.hovered = True
            return True
        if action is None or action[0] != "stream_scroll":
            return super().act(action)
        page = self.page
        before = page.evaluate("() => window.__gateStreams || 0")
        page.keyboard.press("Enter")
        try:
            page.wait_for_function(
                "(n) => (window.__gateStreams || 0) > n", arg = before, timeout = 10_000
            )
        except Exception:  # noqa: BLE001 - no stream started, the area is NOT_RUN
            self.last_error = "Enter did not start a stream"
            return False
        page.wait_for_timeout(500)
        box = page.locator(".aui-thread-viewport").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        self.coverage = []
        for _ in range(WHEEL_STEPS):
            page.mouse.wheel(0, WHEEL_PX)
            page.wait_for_timeout(WHEEL_GAP_MS)
            c = page.evaluate(COVERAGE)
            if c is not None:
                self.coverage.append(c)
        self.coverage += page.evaluate(AFTER_WHEEL, [AFTER_FRAMES, AFTER_SAMPLES, AFTER_GAP_MS])
        self.still_streaming = page.evaluate("() => !window.__gateStreamDone")
        self.hovered = True
        return True

    def wait(self, pred, base_overlays: int) -> bool:
        if pred[0] != "stream_done":
            return super().wait(pred, base_overlays)
        try:
            self.page.wait_for_function(STREAM_DONE, timeout = areas.STREAM_TIMEOUT_MS, polling = 50)
            return True
        except Exception:  # noqa: BLE001 - a stream that never ends is NOT_RUN
            return False


def tall_turns(repeat: int) -> None:
    """Each reply's answer block (prose, code, table, math) `repeat` times: rows slower to render
    than the gap between two streamed chunks, as long agent replies are."""
    short = areas.turn

    def turn(tid: str, t: int, parent):
        pair = short(tid, t, parent)
        parts = pair[1]["content"]
        answer = next(p for p in parts if p["type"] == "text")
        answer["text"] = "\n\n".join(
            [answer["text"]]
            + [answer["text"].replace(f"Answer {t}.", f"Part {k}.") for k in range(1, repeat)]
        )
        return pair

    areas.turn = turn


def open_page(browser, base: str, turns: int, mode: dict):
    """One retry: a fresh browser's first page can miss the load timeout on a loaded runner."""
    try:
        return ABPage(browser, base, turns, mode, [])
    except Exception:  # noqa: BLE001 - the second failure is the one reported
        return ABPage(browser, base, turns, mode, [])


def measure(dists: dict[str, Path], turns: int, reps: int, wanted: list[str]) -> dict:
    from playwright.sync_api import sync_playwright

    mode = next(m for m in areas.app_modes() if m["name"] == "default")
    run = areas.with_deps(wanted)
    servers = {side: areas.serve_app(d) for side, d in dists.items()}
    readings: dict[str, dict[str, list[dict]]] = {s: {a: [] for a in wanted} for s in dists}
    errors: list[str] = []
    order = [("base", "head") if k % 2 == 0 else ("head", "base") for k in range(reps)]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args = chromium_launch_args())
            try:
                for k, pair in enumerate(order):
                    for side in pair:
                        page = None
                        try:
                            page = open_page(browser, servers[side][1], turns, mode)
                            cells = page.sweep(run)
                        except Exception as exc:  # noqa: BLE001 - reported, the verdict goes VOID
                            errors.append(f"{side} rep {k}: {type(exc).__name__}: {str(exc)[:200]}")
                            continue
                        finally:
                            if page is not None:
                                page.close()
                        for a in wanted:
                            cell = cells.get(a, {})
                            if cell.get("status") != "OK":
                                errors.append(f"{side} rep {k} {a}: {cell.get('status')}")
                                continue
                            got = {m: cell.get(m, 0.0) for m in METRICS}
                            if a == "stream_scroll":
                                cov = getattr(page, "coverage", None) or [0.0]
                                got["coverage_min"] = round(min(cov), 3)
                                got["coverage_blank_samples"] = sum(c < COVERAGE_MIN for c in cov)
                                if not getattr(page, "still_streaming", False):
                                    errors.append(
                                        f"{side} rep {k} {a}: the reply ended before the check"
                                    )
                                    continue
                            readings[side][a].append(got)
                        line = "  ".join(
                            f"{a} {cells.get(a, {}).get('main_ms', float('nan')):.0f}"
                            f" (blocked {cells.get(a, {}).get('blocked_ms', float('nan')):.0f}"
                            f" in {cells.get(a, {}).get('long_tasks', 0)})"
                            for a in wanted
                        )
                        print(f"  rep {k} {side:4s} main_ms: {line}", flush = True)
            finally:
                browser.close()
    finally:
        for server, _ in servers.values():
            server.shutdown()
    return {"readings": readings, "errors": errors}


def fmt(x: float) -> str:
    return f"{x:.2f}" if abs(x) < 10 and x != int(x) else f"{x:.0f}"


def judge(
    readings: dict, wanted: list[str], max_ratio: float, gated: list[str]
) -> tuple[list[dict], bool]:
    """Per area and gated metric: SLOWER when head's median is over max_ratio times base's (each
    floored) and every head reading is above every base reading."""
    rows, regressed = [], False
    for a in wanted:
        n_b, n_h = len(readings["base"][a]), len(readings["head"][a])
        if n_b < 2 or n_h < 2:
            rows.append({"area": a, "verdict": "VOID", "readings": [n_b, n_h]})
            continue
        row = {"area": a, "metrics": {}}
        slower = []
        extra = [
            k for k in ("coverage_min", "coverage_blank_samples") if k in readings["head"][a][0]
        ]
        for m in (*METRICS, *extra):
            b = [r[m] for r in readings["base"][a]]
            h = [r[m] for r in readings["head"][a]]
            mb, mh = statistics.median(b), statistics.median(h)
            info = {
                "base_median": round(mb, 1),
                "head_median": round(mh, 1),
                "base": [round(x, 1) for x in b],
                "head": [round(x, 1) for x in h],
            }
            if m in GATES:
                floor = GATES[m]
                info["ratio"] = round(max(mh, floor) / max(mb, floor), 2)
                info["separated"] = min(h) > max(b)
                if info["ratio"] > max_ratio and info["separated"]:
                    slower.append(m)
            row["metrics"][m] = info
        cov = row["metrics"].get("coverage_min")
        # One blank moment is visible, so any head reading counts; a base that blanks too is not the change's.
        if cov and min(cov["head"]) < COVERAGE_MIN <= min(cov["base"]):
            slower.append("coverage_min")
        verdict = f"SLOWER ({', '.join(slower)})" if slower else "ok"
        if slower and a in gated:
            regressed = True
        row["verdict"] = verdict if a in gated else f"{verdict}, advisory"
        rows.append(row)
    return rows, regressed


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    ap.add_argument("--base", type = Path, help = "prebuilt dist of the base")
    ap.add_argument("--head", type = Path, help = "prebuilt dist of the head")
    ap.add_argument("--base-ref", help = "git ref to build as the base")
    ap.add_argument("--head-ref", help = "git ref to build as the head")
    ap.add_argument("--turns", type = int, default = int(os.environ.get("LONG_THREAD_AB_TURNS", "200")))
    ap.add_argument("--reps", type = int, default = int(os.environ.get("LONG_THREAD_AB_REPS", "4")))
    ap.add_argument(
        "--areas", default = os.environ.get("LONG_THREAD_AB_AREAS", "stream,type,thread_scroll")
    )
    ap.add_argument(
        "--gated",
        default = os.environ.get("LONG_THREAD_AB_GATED", "stream,type,stream_scroll,wheel_sweep"),
        help = "areas whose SLOWER verdict fails the run; the rest are reported",
    )
    ap.add_argument(
        "--max-ratio", type = float, default = float(os.environ.get("LONG_THREAD_AB_MAX_RATIO", "1.5"))
    )
    ap.add_argument(
        "--stream-chars",
        type = int,
        default = int(os.environ.get("LONG_THREAD_AB_STREAM_CHARS", "0")),
        help = "lengthen the streamed reply by about this many characters (0: the gate's own reply)",
    )
    ap.add_argument(
        "--tall",
        type = int,
        default = int(os.environ.get("LONG_THREAD_AB_TALL", "1")),
        help = "repeat each seeded reply's answer block this many times (taller, heavier rows)",
    )
    ap.add_argument(
        "--stream-ms",
        type = int,
        default = int(os.environ.get("LONG_THREAD_AB_STREAM_MS", "0")),
        help = "gap between streamed chunks in ms (0: the area matrix's own)",
    )
    ap.add_argument("--json", type = Path)
    args = ap.parse_args()

    wanted = [a.strip() for a in args.areas.split(",") if a.strip()]
    unknown = [a for a in wanted if a not in areas.AREAS]
    if unknown:
        ap.error(f"unknown areas {unknown}; known: {', '.join(areas.AREAS)}")
    gated = [a.strip() for a in args.gated.split(",") if a.strip()]
    dists, shas = {}, {}
    try:
        for side, dist, ref in (
            ("base", args.base, args.base_ref),
            ("head", args.head, args.head_ref),
        ):
            if dist:
                dists[side], shas[side] = dist, None
            elif ref:
                dists[side], shas[side] = build_ref(ref)
            else:
                ap.error(f"give --{side} or --{side}-ref")
    except subprocess.CalledProcessError as exc:
        # Usually a dependency change: the base cannot build with this node_modules.
        print(f"NOT_COMPARABLE: build failed: {exc}", flush = True)
        return 3
    for side, d in dists.items():
        if not (d / "index.html").exists():
            print(f"VOID: no app build at {d / 'index.html'} ({side})", flush = True)
            return 2

    if "stream_scroll" in wanted and len(wanted) > 1:
        ap.error("stream_scroll runs alone: --areas stream_scroll")
    if args.stream_ms:
        areas.STREAM_MS = args.stream_ms
    if args.tall > 1:
        tall_turns(args.tall)
    if "stream_scroll" in wanted and not args.stream_chars:
        args.stream_chars = max(20000, 8000 // areas.STREAM_MS * 60)
    if args.stream_chars:
        lengthen_stream(args.stream_chars)
    started = time.time()
    print(
        f"long-thread A/B: {args.turns} turns, stream +{args.stream_chars} chars, {args.reps} reps per side, areas {wanted}",
        flush = True,
    )
    got = measure(dists, args.turns, args.reps, wanted)
    rows, regressed = judge(got["readings"], wanted, args.max_ratio, gated)
    print(
        "\n| area | metric | base median [readings] | head median [readings] | head/base | separated |"
    )
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r["verdict"] == "VOID":
            print(
                f"| {r['area']} | - | {r['readings'][0]} readings | {r['readings'][1]} readings | - | VOID |"
            )
            continue
        for m, i in r["metrics"].items():
            if m not in GATES and m not in (
                "long_tasks",
                "longest_task_ms",
                "elements",
                "coverage_min",
                "coverage_blank_samples",
            ):
                continue
            print(
                f"| {r['area']} | {m} | {fmt(i['base_median'])} {[fmt(x) for x in i['base']]} | "
                f"{fmt(i['head_median'])} {[fmt(x) for x in i['head']]} | {i.get('ratio', '')} | {i.get('separated', '')} |"
            )
        print(f"| {r['area']} | verdict | | | | {r['verdict']} |")
    for e in got["errors"]:
        print(f"  error: {e}")
    void = any(r["verdict"] == "VOID" and r["area"] in gated for r in rows)
    verdict = "VOID" if void else "REGRESSION" if regressed else "NO_REGRESSION"
    print(
        f"\nVERDICT {verdict}: gated {gated}, limit {args.max_ratio}x on median {'/'.join(GATES)} with no overlap "
        f"(base {shas.get('base') or dists['base']}, head {shas.get('head') or dists['head']}, "
        f"{time.time() - started:.0f} s)",
        flush = True,
    )
    if args.json:
        args.json.parent.mkdir(parents = True, exist_ok = True)
        args.json.write_text(
            json.dumps(
                {
                    "verdict": verdict,
                    "shas": shas,
                    "dists": {k: str(v) for k, v in dists.items()},
                    "turns": args.turns,
                    "reps": args.reps,
                    "stream_chars": args.stream_chars,
                    "tall": args.tall,
                    "stream_ms": args.stream_ms or areas.STREAM_MS,
                    "rows": rows,
                    "errors": got["errors"],
                    "readings": got["readings"],
                },
                indent = 1,
            )
        )
    return 2 if void else 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
