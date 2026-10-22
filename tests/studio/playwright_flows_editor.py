# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flows editor in a real browser against the in-memory backend of smoke-flows.html: gallery, story
graph, quick add, inspector, a simulated run, undo, and the 1000-node performance numbers."""

import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5431"))
BASE = f"http://127.0.0.1:{PORT}/smoke-flows.html"
SHOTS = Path(os.environ.get("FLOWS_SHOTS", Path(__file__).resolve().parent / "flows_shots"))
PERF_NODES = int(os.environ.get("FLOWS_PERF_NODES", "1000"))

failures: list[str] = []
metrics: dict[str, float] = {}


def check(ok: bool, label: str) -> None:
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        failures.append(label)


def node_count(page) -> int:
    return page.locator(".react-flow__node-flowNode").count()


def open_flow(
    page,
    flow: str,
    extra: str = "",
) -> float:
    t0 = time.perf_counter()
    page.goto(f"{BASE}?flow={flow}{extra}")
    page.wait_for_selector('[data-testid="flows-editor"]', timeout = 60_000)
    page.wait_for_selector(".react-flow__node-flowNode", timeout = 60_000)
    page.wait_for_timeout(300)
    return time.perf_counter() - t0


def pane_center(page) -> tuple[float, float]:
    box = page.locator(".react-flow__pane").bounding_box()
    return box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5


def pan_metrics(page, label: str) -> None:
    """Space-drag the canvas for ~2 s while a rAF loop records frame times."""
    box = page.locator(".react-flow__pane").bounding_box()
    # Start in the corner: fitView leaves padding there, so the drag grabs the pane, not a node.
    x, y = box["x"] + 230, box["y"] + 90
    transform = "() => document.querySelector('.react-flow__viewport').style.transform"
    before = page.evaluate(transform)
    page.evaluate(
        """() => { window.__frames = []; window.__stopFrames = false; const loop = (t) => {
        window.__frames.push(t); if (!window.__stopFrames) requestAnimationFrame(loop); };
        requestAnimationFrame(loop); }"""
    )
    page.mouse.move(x, y)
    page.keyboard.down("Space")
    page.mouse.down()
    for i in range(120):
        page.mouse.move(x + 3 * i, y + (i % 40), steps = 1)
    page.mouse.up()
    page.keyboard.up("Space")
    frames = page.evaluate("() => { window.__stopFrames = true; return window.__frames; }")
    check(page.evaluate(transform) != before, f"{label}: the drag actually panned the canvas")
    if len(frames) > 2:
        span = (frames[-1] - frames[0]) / 1000
        deltas = sorted(b - a for a, b in zip(frames, frames[1:]))
        metrics[f"{label}_fps"] = (len(frames) - 1) / span if span > 0 else 0
        metrics[f"{label}_p95_frame_ms"] = deltas[int(len(deltas) * 0.95) - 1]


def main() -> int:
    SHOTS.mkdir(parents = True, exist_ok = True)
    proc = start_vite(PORT)
    try:
        wait_for_smoke_page(BASE, "smoke-flows-main.tsx", proc = proc)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args = chromium_launch_args())
            page = browser.new_page(viewport = {"width": 1600, "height": 960})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            # Gallery
            page.goto(BASE)
            page.wait_for_selector('[data-testid="flows-gallery"]', timeout = 60_000)
            page.wait_for_selector('[data-testid="flows-card"]', timeout = 20_000)
            page.wait_for_timeout(400)
            check(
                page.locator('[data-testid="flows-card"]').count() >= 4,
                "gallery lists flows and examples",
            )
            page.screenshot(path = str(SHOTS / "editor_gallery.png"))

            # New pipeline from the gallery opens the editor
            page.click('[data-testid="flows-new-pipeline"]')
            page.wait_for_selector('[data-testid="flows-editor"]', timeout = 20_000)
            page.wait_for_selector(".react-flow__node-flowNode", timeout = 20_000)
            check(node_count(page) == 4, "new pipeline opens with its 4 template nodes")

            # Story graph
            open_flow(page, "story")
            metrics["story_time_to_nodes_ms"] = page.evaluate("() => performance.now()")
            check(node_count(page) == 8, f"story template renders 8 nodes (got {node_count(page)})")
            check(page.locator(".react-flow__edge").count() == 7, "story template renders 7 edges")
            page.screenshot(path = str(SHOTS / "editor_story.png"))

            # Quick add by double-clicking the canvas
            before = node_count(page)
            x, y = pane_center(page)
            page.mouse.move(x, y + 260)
            page.mouse.dblclick(x, y + 260)
            page.wait_for_selector('[data-testid="flows-quick-add"]', timeout = 5_000)
            page.keyboard.type("generate image")
            page.wait_for_timeout(150)
            page.screenshot(path = str(SHOTS / "editor_quickadd.png"))
            page.keyboard.press("Enter")
            page.wait_for_timeout(250)
            check(node_count(page) == before + 1, "quick add inserts a node")

            # Undo / redo
            mod = "Meta" if sys.platform == "darwin" else "Control"
            page.locator(".react-flow__pane").click(position = {"x": 20, "y": 20})
            page.keyboard.press(f"{mod}+z")
            page.wait_for_timeout(150)
            check(node_count(page) == before, "undo removes it")
            page.keyboard.press(f"{mod}+Shift+z")
            page.wait_for_timeout(150)
            check(node_count(page) == before + 1, "redo restores it")

            # Inspector
            header = page.locator('.react-flow__node-flowNode [data-node-id="s1"] .flow-node-drag')
            t0 = time.perf_counter()
            header.dblclick()
            page.wait_for_selector('[data-testid="flows-inspector"]', timeout = 5_000)
            metrics["story_inspector_open_ms"] = (time.perf_counter() - t0) * 1000
            check(
                page.locator('[data-testid="flows-inspector"] textarea').count() >= 2,
                "inspector renders scene fields",
            )
            page.locator('[data-testid="flows-inspector"] input').nth(1).fill("The misty shore")
            page.wait_for_timeout(1200)
            saved = page.evaluate("() => window.__flowsSmoke.ui.getState().saveState")
            check(saved == "saved", f"autosave settles to saved (got {saved})")
            page.screenshot(path = str(SHOTS / "editor_inspector.png"))

            # Simulated run on the pipeline
            open_flow(page, "pipeline")
            page.click('[data-testid="flows-run"]')
            page.wait_for_selector('[data-testid="flows-run-drawer"]', timeout = 5_000)
            page.wait_for_function(
                "() => window.__flowsSmoke.runtime.getState().runStatus === 'completed'",
                timeout = 30_000,
            )
            page.wait_for_timeout(500)
            done = page.locator('[data-status="done"]').count()
            check(done == 4, f"all 4 pipeline nodes finish (got {done})")
            check(
                page.locator(".react-flow__node-flowNode img").count() >= 1,
                "image output previews in its node",
            )
            page.screenshot(path = str(SHOTS / "editor_run.png"))

            # Perf: 1000 nodes / ~1500 edges
            open_flow(page, "perf", f"&nodes={PERF_NODES}")
            page.wait_for_function(
                f"() => document.querySelectorAll('.react-flow__node-flowNode').length >= {PERF_NODES}",
                timeout = 60_000,
            )
            # From navigation start, in the page clock: module load + catalog + first full paint of every node.
            metrics["perf_time_to_all_nodes_ms"] = page.evaluate("() => performance.now()")
            metrics["perf_doc_edges"] = page.evaluate(
                "() => window.__flowsSmoke.doc.getState().graph.edgeOrder.length"
            )
            page.wait_for_timeout(1500)
            metrics["perf_rendered_nodes"] = node_count(page)
            metrics["perf_rendered_edges"] = page.locator(".react-flow__edge").count()
            page.screenshot(path = str(SHOTS / f"editor_perf_{PERF_NODES}.png"))

            pan_metrics(page, "pan_all_1000_visible")
            # Zoom in to a working scale, then pan for 2 s while counting frames.
            page.evaluate("() => 0")
            x, y = pane_center(page)
            for _ in range(6):
                page.mouse.move(x, y)
                page.mouse.wheel(0, -400)
                page.wait_for_timeout(120)
            page.wait_for_timeout(500)
            metrics["perf_rendered_nodes_zoomed"] = node_count(page)
            pan_metrics(page, "pan_zoomed")

            # Inspector latency on the big graph
            first = page.locator(".react-flow__node-flowNode .flow-node-drag").first
            t0 = time.perf_counter()
            first.dblclick()
            page.wait_for_selector('[data-testid="flows-inspector"]', timeout = 5_000)
            metrics["perf_inspector_open_ms"] = (time.perf_counter() - t0) * 1000

            # Render isolation: progress for one node commits only that node's subtree.
            result = page.evaluate(
                """async () => {
                  const s = window.__flowsSmoke;
                  const target = document.querySelector('.react-flow__node-flowNode [data-node-id]').getAttribute('data-node-id');
                  const rt = s.runtime.getState();
                  rt.startRun('perf-run');
                  await new Promise(r => setTimeout(r, 300));
                  const before = s.commits.length;
                  let seq = 0;
                  for (let i = 0; i < 30; i++) {
                    s.runtime.getState().applyEvents([{ seq: ++seq, run_id: 'perf-run', type: 'node.progress',
                      node_id: target, ts: Date.now() / 1000, data: { fraction: i / 30 } }]);
                    await new Promise(r => requestAnimationFrame(() => r()));
                  }
                  await new Promise(r => setTimeout(r, 200));
                  const commits = s.commits.slice(before);
                  const total = commits.reduce((a, c) => a + c.actualDuration, 0);
                  const bar = document.querySelector(`[data-node-id="${target}"]`).getAttribute('data-status');
                  return { target, bar, commits: commits.length, avg_ms: commits.length ? total / commits.length : 0,
                           max_ms: Math.max(0, ...commits.map(c => c.actualDuration)) };
                }"""
            )
            print("progress target", result["target"], result["bar"])
            metrics["progress_event_commits"] = result["commits"]
            metrics["progress_event_avg_commit_ms"] = result["avg_ms"]
            metrics["progress_event_max_commit_ms"] = result["max_ms"]

            check(
                not [e for e in errors if "ResizeObserver" not in e],
                f"no page errors ({errors[:3]})",
            )
            browser.close()
    finally:
        stop_process(proc)

    print("METRICS " + json.dumps({k: round(v, 2) for k, v in metrics.items()}))
    print(f"screenshots: {SHOTS}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
