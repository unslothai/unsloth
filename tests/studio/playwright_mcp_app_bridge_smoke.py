# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser smoke: an MCP App widget's bridge does not survive the frame navigating away.

A sandboxed iframe keeps one ``contentWindow`` and reports the opaque origin ``"null"``
whatever it loads, and the replacement document's inline scripts run BEFORE the iframe's
load event. So neither sender identity, nor origin, nor a load-counting flag can tell the
seeded widget from the page an ordinary in-widget link moved the frame to. The host stamps
a per-document token into the shim it seeds instead; this asserts the resulting behaviour
in a real browser rather than in the source.

Runs the SHIPPED shim: both helpers are read out of mcp-app-frame.tsx.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

FRAME_TSX = (
    Path(__file__).resolve().parents[2]
    / "studio/frontend/src/features/chat/mcp-apps/mcp-app-frame.tsx"
)
TOKEN = "test-token-2f8c41"


def shipped_shim() -> str:
    """The shim string mcp-app-frame.tsx builds, with the token substituted."""
    source = FRAME_TSX.read_text(encoding = "utf-8")
    body = re.search(
        r"export function bridgeShim\(token: string\): string \{\s*return `(.*?)`;\s*\}",
        source,
        re.S,
    )
    if not body:
        raise SystemExit("bridgeShim is no longer a single template literal in mcp-app-frame.tsx")
    return body.group(1).replace("${JSON.stringify(token)}", f'"{TOKEN}"')


HOST = """<!doctype html><html><body>
<iframe id="f" sandbox="allow-scripts"></iframe>
<script>
  window.__accepted = [];
  window.__oldAccepted = [];
  window.__seen = [];
  const f = document.getElementById("f");
  // The gate this smoke exists to retire: arm on the load we seeded, revoke on
  // any load after it. Scored alongside the real one so a green run shows what
  // the token buys rather than only that nothing broke.
  let oldArmed = false, seeded = false;
  f.addEventListener("load", () => { oldArmed = !seeded; seeded = true; });
  window.addEventListener("message", (e) => {
    window.__seen.push(e.data);
    if (e.source !== f.contentWindow) return;
    if (e.origin !== "null") return;
    const env = e.data;
    const payload = env && typeof env === "object" && "__unslothMcpApp" in env
      ? env.message : env;
    if (oldArmed) window.__oldAccepted.push(payload);
    if (!env || typeof env !== "object" || env.__unslothMcpApp !== TOKEN) return;
    window.__accepted.push(env.message);
  });
  f.src = "https://mcp-app.test/seeded.html";
</script></body></html>"""

SEEDED = """<!doctype html><html><head>SHIM</head><body>
<script>
  parent.postMessage({jsonrpc: "2.0", id: 1, method: "tools/call",
                      params: {name: "refresh"}}, "*");
  // An ordinary in-widget link moving the frame to an undeclared site.
  setTimeout(() => window.location.replace("https://undeclared.test/other.html"), 50);
</script></body></html>"""

# Posts while parsing, so it lands before this document's load event.
NAVIGATED = """<!doctype html><html><head><script>
  window.parent.postMessage({jsonrpc: "2.0", id: 99, method: "tools/call",
                             params: {name: "exfiltrate"}}, "*");
</script></head><body>other<img src="https://undeclared.test/slow.png"></body></html>"""


def main() -> None:
    shim = shipped_shim()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless = True)
        page = browser.new_page()
        page.route(
            "**/host.html",
            lambda r: r.fulfill(
                status = 200,
                content_type = "text/html",
                body = HOST.replace("TOKEN", f'"{TOKEN}"'),
            ),
        )
        page.route(
            "**/seeded.html",
            lambda r: r.fulfill(
                status = 200, content_type = "text/html", body = SEEDED.replace("SHIM", shim)
            ),
        )
        page.route(
            "**/other.html",
            lambda r: r.fulfill(status = 200, content_type = "text/html", body = NAVIGATED),
        )
        # Keeps the navigated document's load event pending while its script runs.
        page.route("**/slow.png", lambda r: r.abort())
        page.goto("https://mcp-app.test/host.html")
        page.wait_for_timeout(2_000)

        accepted = page.evaluate("window.__accepted")
        old_accepted = page.evaluate("window.__oldAccepted")
        seen = page.evaluate("window.__seen")
        browser.close()

    def names(messages) -> list:
        return [m.get("params", {}).get("name") for m in messages if isinstance(m, dict)]

    methods, old_methods = names(accepted), names(old_accepted)
    print(f"[mcp-app-bridge] messages reaching the host: {len(seen)}")
    print(f"[mcp-app-bridge] load-flag gate would accept: {old_methods}")
    print(f"[mcp-app-bridge] token gate accepts:          {methods}")

    if "exfiltrate" not in old_methods:
        raise SystemExit(
            "the navigated document did not beat the load event in this browser, so the "
            "run does not exercise the race this smoke is for; check the fixture"
        )

    if "refresh" not in methods:
        raise SystemExit("the seeded widget's own tools/call was refused; the shim is broken")
    if "exfiltrate" in methods:
        raise SystemExit(
            "REGRESSION: a document the frame navigated to still reached the tool bridge"
        )
    if len(seen) < 2:
        raise SystemExit(
            "the navigated document never posted, so this run proves nothing; check the fixture"
        )
    print("[mcp-app-bridge] the seeded widget is served and the navigated document is refused")


if __name__ == "__main__":
    sys.exit(main())
