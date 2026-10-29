# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser smoke for the shipped MCP App bridge: the seeded view is served through its port, a
document the frame navigates to is not, and the size fallback lets a widget shrink."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

LEAF = Path(__file__).resolve().parents[2] / "studio/frontend/src/features/chat/mcp-apps/mcp-ui.ts"
TOKEN, ORIGIN = "test-token-2f8c41", "https://mcp-app.test"


def shipped() -> dict:
    js = (
        f"import({json.dumps(LEAF.as_uri())}).then((m) => console.log(JSON.stringify({{"
        f"shim: m.bridgeShim({json.dumps(TOKEN)}, {json.dumps(ORIGIN)}),"
        "insert: m.withBridgeShim.toString(), resize: m.RESIZE_FALLBACK})))"
    )
    cmd = ["node", "--experimental-strip-types", "--no-warnings", "-e", js]
    return json.loads(subprocess.run(cmd, check = True, capture_output = True, text = True).stdout)


# Only the token-carrying handshake is read off the window, as in Studio.
HOST = """<!doctype html><body><iframe id="f" sandbox="allow-scripts"></iframe><script>
  window.got = []; window.leaked = []; window.seen = []; let port = null;
  const f = document.getElementById("f");
  addEventListener("message", (e) => {
    window.seen.push(e.data);
    if (e.data && e.data.leaked) return window.leaked.push(e.data.leaked);
    if (e.source === f.contentWindow && e.data && e.data.__unslothMcpApp === TOKEN) {
      port = e.ports[0];
      port.onmessage = (ev) => window.got.push(ev.data);
      port.postMessage({jsonrpc: "2.0", id: 1, result: {secret: "FIRST"}});
    }
  });
  f.src = "PAGE";
  setTimeout(() => {  // replies that arrive after the frame moved on
    port.postMessage({jsonrpc: "2.0", id: 2, result: {secret: "PORT-REPLY"}});
    f.contentWindow.postMessage({jsonrpc: "2.0", id: 2, result: {secret: "WILDCARD"}}, "*");
  }, 900);
</script></body>"""

VIEW = """<!doctype html><html><head>SHIM</head><body><script>
  addEventListener("message", (e) => {
    const s = e.data && e.data.result && e.data.result.secret;
    if (s !== "FIRST") return;
    const ok = e.source === window.parent && e.origin === ORIGIN;
    parent.postMessage({report: ok ? "source-and-origin-ok" : "bad"}, "*");
    parent.postMessage({jsonrpc: "2.0", id: 1, method: "tools/call", params: {name: "refresh"}}, "*");
    setTimeout(() => location.replace("https://undeclared.test/other.html"), 150);
  });
</script></body></html>"""

NAVIGATED = """<!doctype html><script>
  window.parent.postMessage({jsonrpc: "2.0", id: 9, method: "tools/call", params: {name: "exfiltrate"}}, "*");
  addEventListener("message", (e) => {
    const s = e.data && e.data.result && e.data.result.secret;
    if (s) window.parent.postMessage({leaked: s}, "*");
  });
</script>other"""

RESIZE = """<!doctype html><body><iframe id="f" sandbox="allow-scripts" style="height:320px"></iframe><script>
  window.heights = [];
  addEventListener("message", (e) => { if (typeof e.data.mcpAppHeight === "number") window.heights.push(e.data.mcpAppHeight); });
</script></body>"""


def main() -> None:
    s = shipped()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless = True)
        page = browser.new_page()
        page.goto("about:blank")
        insert = f"([h, m]) => ({s['insert']})(h, m)"
        inserted = page.evaluate(
            insert, ["<!doctype html><html><!-- no <head> --><body>hi</body></html>", "MARK"]
        )
        bare = page.evaluate(insert, ["<p>a bare fragment</p>", "MARK"])
        assert (
            inserted.index("<head>") < inserted.index("MARK") < inserted.index("</head>")
        ), inserted
        assert not bare.lower().startswith(
            "<!doctype"
        ), f"a quirks-mode template gained a doctype: {bare}"

        for name, body in {
            "host.html": HOST.replace("TOKEN", json.dumps(TOKEN)).replace(
                "PAGE", f"{ORIGIN}/view.html"
            ),
            "view.html": VIEW.replace("SHIM", f"<script>{s['shim']}</script>").replace(
                "ORIGIN", json.dumps(ORIGIN)
            ),
            "other.html": NAVIGATED,
        }.items():
            page.route(
                f"**/{name}",
                (lambda b: lambda r: r.fulfill(status = 200, content_type = "text/html", body = b))(body),
            )
        page.goto(f"{ORIGIN}/host.html")
        page.wait_for_timeout(2_000)
        got, leaked, seen = (page.evaluate(f"window.{k}") for k in ("got", "leaked", "seen"))
        names = [m.get("params", {}).get("name") for m in got if isinstance(m, dict)]
        print(
            f"[mcp-app-bridge] port delivered {names} {[m.get('report') for m in got if 'report' in m]}; leaked {leaked}"
        )
        assert {"report": "source-and-origin-ok"} in got, f"the port broke an ordinary view: {got}"
        assert "refresh" in names, "the seeded view's own tools/call never arrived"
        assert any(
            isinstance(m, dict) and m.get("id") == 9 for m in seen
        ), "fixture: navigated page never posted"
        assert "exfiltrate" not in names, "REGRESSION: a navigated document reached the bridge"
        assert "WILDCARD" in leaked, "fixture: the navigated page was not listening"
        assert (
            "PORT-REPLY" not in leaked
        ), "REGRESSION: a port reply followed the frame to another page"

        page.set_content(RESIZE)
        view = f'<!doctype html><body><div id="c" style="height:90px"></div>{s["resize"]}'
        page.evaluate("(h) => { document.getElementById('f').srcdoc = h; }", view)
        page.wait_for_timeout(500)
        for px in (700, 40):  # awaiting a frame pumps rendering, which headless otherwise idles
            page.frames[1].evaluate(
                f"document.getElementById('c').style.height = '{px}px';"
                "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))"
            )
            page.wait_for_timeout(200)
        heights = page.evaluate("window.heights")
        browser.close()
    print(f"[mcp-app-bridge] fallback heights in a 320px frame: {heights}")
    assert heights and heights[0] < 150, f"a 90px widget did not report its own height: {heights}"
    assert (
        max(heights) >= 700 and heights[-1] < 100
    ), f"the widget could not grow then shrink: {heights}"


if __name__ == "__main__":
    sys.exit(main())
