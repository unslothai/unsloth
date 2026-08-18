# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does the token cache change what the screen SHOWS? Byte-for-byte, on two trees at once.

A memory fix that quietly restyles code is not a memory fix, so this renders the same fences in
two working trees and compares the rendered HTML of the whole reply, including every Shiki inline
style. It drives smoke-shiki-retention.html, so it renders through the real chat markdown pipeline:
the production code plugin, the production themes, `stabilizeStreamingMarkdown`, the incremental
block cache and `<Streamdown mode="streaming">`.

Cases cover both paths through the plugin: fences above MIN_INCREMENTAL_CHARS (the per-fence slot
path a streaming reply uses) and below it (the shortcut straight to the highlighter), streamed and
delivered whole, plus prose with no fence at all.

The comparison is checked before it is trusted: a control case renders DIFFERENT sources on the
two trees and the run fails if that pair comes back identical, because a comparator that cannot
see a real difference would report every case as a pass.

    python probe_shiki_render_parity.py --base <worktree> --head <worktree>
"""

from __future__ import annotations

import argparse
import difflib
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

CASES = [
    # kind, chars, seed, ticks, tick_ms, label
    ("stream", 32768, 11, 48, 40, "streamed 32 KB fence"),
    ("stream", 8192, 12, 24, 40, "streamed 8 KB fence"),
    ("stream", 1200, 13, 12, 40, "streamed fence below MIN_INCREMENTAL_CHARS"),
    ("whole", 32768, 14, 1, 0, "32 KB fence delivered whole"),
    ("whole", 1200, 15, 1, 0, "small fence delivered whole"),
    ("prose", 16384, 16, 32, 20, "prose, no fence"),
]


def info(message: str) -> None:
    print(f"[render-parity] {message}", flush = True)


def render(page, kind: str, chars: int, seed: int, ticks: int, tick_ms: int) -> str:
    page.evaluate(
        "(spec) => window.__sd.runOne(spec)",
        {"kind": kind, "chars": chars, "seed": seed, "ticks": ticks, "tickMs": tick_ms},
    )
    # Highlighting lands asynchronously, and an early read would compare an unstyled frame against
    # a styled one and call the fix a regression. Wait for the page's own idle signal.
    page.wait_for_function(
        "() => window.__sd.counters().pending === 0", timeout = 120_000
    )
    page.wait_for_timeout(1500)
    page.wait_for_function(
        "() => window.__sd.counters().pending === 0", timeout = 120_000
    )
    # The tokens arrive on a callback, so the commit that paints them is one render after the
    # counter clears. Reading before it lands would compare a styled tree against an unstyled one.
    page.wait_for_timeout(400)
    html = page.evaluate("() => document.getElementById('root').innerHTML")
    if '<div data-smoke="reply"' not in html:
        raise SystemExit(f"nothing rendered for {kind}@{chars}")
    return html


def start_vite(tree: Path, port: int):
    """Vite in THIS tree, in its own process group.

    Deliberately not `tests/studio/_playwright_robust.start_vite`: that module pins the frontend
    directory at import time, so importing it once per tree would silently serve the first tree's
    build on both ports and report perfect parity between a tree and itself.
    """
    frontend = tree / "studio" / "frontend"
    if not (frontend / "smoke-shiki-retention.html").exists():
        raise SystemExit(f"{frontend} has no smoke-shiki-retention.html")
    proc = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", str(port), "--strictPort"],
        cwd = frontend,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        start_new_session = True,
    )
    deadline = time.monotonic() + 180
    url = f"http://127.0.0.1:{port}/smoke-shiki-retention.html"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"vite for {tree} exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout = 3.0) as response:
                body = response.read().decode("utf-8", errors = "replace")
                # Vite answers 200 with index.html for a path it cannot resolve, so match the
                # module specifier rather than the status.
                if response.status == 200 and "/smoke-shiki-retention-main.tsx" in body:
                    info(f"{url} ready")
                    return proc
        except Exception:
            pass
        time.sleep(1.0)
    raise SystemExit(f"vite for {tree} never served {url}")


def stop_vite(proc) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout = 10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def open_tree(pw, tree: Path, port: int, servers: list):
    servers.append(start_vite(tree, port))
    browser = pw.chromium.launch(headless = True, args = ["--no-sandbox", "--disable-gpu"])
    page = browser.new_page()
    page.goto(
        f"http://127.0.0.1:{port}/smoke-shiki-retention.html", wait_until = "domcontentloaded"
    )
    page.wait_for_function("() => window.__sd && window.__sd.ready", timeout = 120_000)
    return browser, page


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required = True, type = Path)
    parser.add_argument("--head", required = True, type = Path)
    parser.add_argument("--base-port", type = int, default = 5401)
    parser.add_argument("--head-port", type = int, default = 5402)
    args = parser.parse_args()

    servers: list = []
    failures: list[str] = []
    try:
        with sync_playwright() as pw:
            base_browser, base_page = open_tree(pw, args.base, args.base_port, servers)
            head_browser, head_page = open_tree(pw, args.head, args.head_port, servers)

            for kind, chars, seed, ticks, tick_ms, label in CASES:
                # The fixture is generated in-page from a seeded LCG. If the two trees disagree
                # about the source, comparing what they rendered is meaningless, so refuse.
                base_hash = base_page.evaluate(
                    "([k, c, s]) => window.__sd.fixtureHash(k, c, s)", [kind, chars, seed]
                )
                head_hash = head_page.evaluate(
                    "([k, c, s]) => window.__sd.fixtureHash(k, c, s)", [kind, chars, seed]
                )
                if base_hash != head_hash:
                    raise SystemExit(
                        f"fixture census failed for {label}: {base_hash} != {head_hash}"
                    )
                base_html = render(base_page, kind, chars, seed, ticks, tick_ms)
                head_html = render(head_page, kind, chars, seed, ticks, tick_ms)
                if base_html == head_html:
                    info(f"identical: {label} ({len(base_html)} chars of HTML)")
                else:
                    ratio = similarity(base_html, head_html)
                    failures.append(f"{label}: {ratio * 100:.2f}% identical")
                    info(f"DIFFERENT: {label} -> {ratio * 100:.2f}% identical")
                    Path("logs").mkdir(exist_ok = True)
                    Path(f"logs/render-parity-{kind}-{chars}-base.html").write_text(base_html)
                    Path(f"logs/render-parity-{kind}-{chars}-head.html").write_text(head_html)

            # Comparator self-check. Different sources must come back different; if they do not,
            # every "identical" above is worthless.
            control_base = render(base_page, "stream", 8192, 900, 24, 40)
            control_head = render(head_page, "stream", 8192, 901, 24, 40)
            if control_base == control_head:
                failures.append(
                    "control: two DIFFERENT sources rendered identical HTML, so the comparator "
                    "is not reading the rendered output"
                )
                info("CONTROL FAILED")
            else:
                info(
                    "control: different sources differ "
                    f"({similarity(control_base, control_head) * 100:.2f}% identical)"
                )
            base_browser.close()
            head_browser.close()
    finally:
        for proc in servers:
            stop_vite(proc)

    for failure in failures:
        info(f"FAILURE: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
