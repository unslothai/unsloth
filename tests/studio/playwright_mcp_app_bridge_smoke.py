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
HOST_ORIGIN = "https://mcp-app.test"


def shipped_shim() -> str:
    """The shim string mcp-app-frame.tsx builds, with the token substituted."""
    source = FRAME_TSX.read_text(encoding = "utf-8")
    body = re.search(
        r"export function bridgeShim\(.*?\n  return `(.*?)`;\n\}",
        source,
        re.S,
    )
    if not body:
        raise SystemExit(
            "bridgeShim no longer returns a single template literal in mcp-app-frame.tsx"
        )
    shim = body.group(1)
    shim = shim.replace("${JSON.stringify(token)}", f'"{TOKEN}"')
    shim = shim.replace("${JSON.stringify(hostOrigin)}", f'"{HOST_ORIGIN}"')
    shim = shim.replace("\\`", "`")
    if "${" in shim:
        raise SystemExit(f"bridgeShim grew an interpolation this smoke does not substitute: {shim}")
    return shim


HOST = """<!doctype html><html><body>
<iframe id="f" sandbox="allow-scripts"></iframe>
<script>
  window.__accepted = [];
  window.__oldAccepted = [];
  window.__seen = [];
  window.__leaked = [];
  window.__port = null;
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
    if (env && env.leaked) { window.__leaked.push(env.leaked); return; }
    const payload = env && typeof env === "object" && "__unslothMcpApp" in env
      ? env.message : env;
    if (oldArmed) window.__oldAccepted.push(payload);
    if (!env || typeof env !== "object" || env.__unslothMcpApp !== TOKEN) return;
    if (env.__unslothMcpAppPort === true) { window.__port = e.ports[0]; return; }
    window.__accepted.push(env.message);
  });
  // The reply to the widget's tools/call, arriving after the frame moved on.
  // The port carries what the host actually sends; the wildcard post alongside it
  // is the retired path, scored so a green run shows the port is what stops the
  // leak rather than the fixture never reaching it.
  setTimeout(() => {
    const rpc = (secret) => ({jsonrpc: "2.0", id: 1, result: {secret}});
    if (window.__port) window.__port.postMessage(rpc("PORT-REPLY"));
    f.contentWindow.postMessage(rpc("WILDCARD-REPLY"), "*");
  }, 700);
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
  // Anything the host still sends this frame lands here.
  window.addEventListener("message", (e) => {
    const secret = e.data && e.data.result && e.data.result.secret;
    if (secret) window.parent.postMessage({leaked: secret}, "*");
  });
</script></head><body>other<img src="https://undeclared.test/slow.png"></body></html>"""


STAYING_HOST = """<!doctype html><html><body>
<iframe id="f" sandbox="allow-scripts"></iframe>
<script>
  window.__report = null;
  const f = document.getElementById("f");
  window.addEventListener("message", (e) => {
    const env = e.data;
    if (!env || typeof env !== "object" || env.__unslothMcpApp !== TOKEN) return;
    if (env.__unslothMcpAppPort === true) {
      e.ports[0].postMessage({jsonrpc: "2.0", id: 1, result: {secret: "PORT-REPLY"}});
      return;
    }
    if (env.message && env.message.report) window.__report = env.message.report;
  });
  f.src = "https://mcp-app.test/staying.html";
</script></body></html>"""

# No navigation: an ordinary view that listens on window, as the protocol says.
STAYING = """<!doctype html><html><head>SHIM</head><body>
<script>
  window.addEventListener("message", (e) => {
    const secret = e.data && e.data.result && e.data.result.secret;
    if (!secret) return;
    parent.postMessage({report: secret + " | origin=" + e.origin +
                        " | source is a window:" + (e.source !== null)}, "*");
  });
</script>staying</body></html>"""


def check_a_staying_view_still_gets_its_replies(browser, shim: str) -> None:
    """The port must not cost an ordinary view its replies."""
    page = browser.new_page()
    page.route(
        "**/staying-host.html",
        lambda r: r.fulfill(
            status = 200,
            content_type = "text/html",
            body = STAYING_HOST.replace("TOKEN", f'"{TOKEN}"'),
        ),
    )
    page.route(
        "**/staying.html",
        lambda r: r.fulfill(
            status = 200, content_type = "text/html", body = STAYING.replace("SHIM", shim)
        ),
    )
    page.goto("https://mcp-app.test/staying-host.html")
    page.wait_for_timeout(1_500)
    report = page.evaluate("window.__report")
    page.close()
    print(f"[mcp-app-bridge] a view that stays put received: {report}")
    if not report or not report.startswith("PORT-REPLY"):
        raise SystemExit(
            "a reply sent down the port never reached a view listening the ordinary "
            f"way, so the port broke every widget (report={report!r})"
        )
    if f"origin={HOST_ORIGIN}" not in report:
        raise SystemExit(f"the re-dispatched reply lost the host origin (report={report!r})")


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
        check_a_staying_view_still_gets_its_replies(browser, shim)

        page.goto("https://mcp-app.test/host.html")
        page.wait_for_timeout(2_000)

        leaked = page.evaluate("window.__leaked")
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
    print(f"[mcp-app-bridge] replies reaching the navigated document: {leaked}")

    if "WILDCARD-REPLY" not in leaked:
        raise SystemExit(
            "the wildcard reply never reached the navigated document, so this run "
            "does not exercise the leak window; check the fixture"
        )
    if "PORT-REPLY" in leaked:
        raise SystemExit(
            "REGRESSION: a reply sent down the seeded document's port still reached "
            "the page the frame navigated to"
        )

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
    print(
        "[mcp-app-bridge] the seeded widget is served, the navigated document is "
        "refused, and no reply follows the frame to it"
    )


if __name__ == "__main__":
    sys.exit(main())
