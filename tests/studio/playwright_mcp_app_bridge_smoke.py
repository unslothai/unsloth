# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser smoke: an MCP App widget's bridge does not survive the frame navigating away.

A sandboxed iframe keeps one ``contentWindow`` and reports the opaque origin ``"null"``
whatever it loads, and the replacement document's inline scripts run BEFORE the iframe's
load event. So neither sender identity, nor origin, nor a load-counting flag can tell the
seeded widget from the page an ordinary in-widget link moved the frame to. What can is a
MessageChannel port, which cannot outlive the document that made it, and which the shim
installs as that document's ``window.parent``.

Four things are checked in a real browser, against the SHIPPED shim and inserter read out
of mcp-app-frame.tsx:

  1. the shim lands where the browser will run it, not in a comment that mentions ``<head>``;
  2. a view that stays put still gets its replies, and ``event.source === window.parent``
     still holds for one that filters on it;
  3. a document the frame navigated to cannot reach the bridge;
  4. no in-flight reply follows the frame to that document.

Each of (3) and (4) also scores the path it replaced, so a green run shows the mechanism
working rather than a fixture that stopped reaching the window it is about.
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


def shipped_inserter() -> str:
    """withBridgeShim as plain JS, so the smoke inserts the way Studio does."""
    source = FRAME_TSX.read_text(encoding = "utf-8")
    start = source.index("export function withBridgeShim(")
    body = source[start : source.index("\n}\n", start) + 3].replace("export ", "", 1)
    stripped = re.sub(r":\s*string(?=[,)\s])", "", body)
    if re.search(r"\w:\s*[A-Z]\w*", stripped):
        raise SystemExit(f"withBridgeShim grew a type this smoke cannot strip:\n{stripped}")
    return stripped


# The host under test: takes the port from the token-checked handshake and reads
# everything else off it. The window listener also scores the retired paths.
HOST = """<!doctype html><html><body>
<iframe id="f" sandbox="allow-scripts"></iframe>
<script>
  window.__accepted = [];
  window.__oldAccepted = [];
  window.__leaked = [];
  window.__seen = [];
  window.__port = null;
  const f = document.getElementById("f");
  // The gate this smoke retired: arm on the load we seeded, revoke on any after.
  let oldArmed = false, seeded = false;
  f.addEventListener("load", () => { oldArmed = !seeded; seeded = true; });
  window.addEventListener("message", (e) => {
    window.__seen.push(e.data);
    const env = e.data;
    if (env && env.leaked) { window.__leaked.push(env.leaked); return; }
    if (e.source !== f.contentWindow) return;
    if (e.origin !== "null") return;
    if (env && env.__unslothMcpApp === TOKEN && env.__unslothMcpAppPort === true) {
      window.__port = e.ports[0];
      window.__port.onmessage = (ev) => window.__accepted.push(ev.data);
      return;
    }
    // Anything else on the window is a document with no port of its own.
    if (oldArmed) window.__oldAccepted.push(env);
  });
  f.src = "https://mcp-app.test/seeded.html";
  // The reply to the widget's tools/call, arriving after the frame moved on. The
  // port carries what the host actually sends; the wildcard post beside it is the
  // retired path, scored so a green run shows the port is what stops the leak.
  setTimeout(() => {
    const rpc = (secret) => ({jsonrpc: "2.0", id: 1, result: {secret}});
    if (window.__port) window.__port.postMessage(rpc("PORT-REPLY"));
    f.contentWindow.postMessage(rpc("WILDCARD-REPLY"), "*");
  }, 700);
</script></body></html>"""

# The seeded view: calls a tool the ordinary way, then follows a link out.
SEEDED = """<!doctype html><html><head>SHIM</head><body>
<script>
  parent.postMessage({jsonrpc: "2.0", id: 1, method: "tools/call",
                      params: {name: "refresh"}}, "*");
  setTimeout(() => window.location.replace("https://undeclared.test/other.html"), 120);
</script>seeded</body></html>"""

# The page the frame navigates to. No shim, so no port: it can only reach the
# window. It posts while parsing, before its own load event.
NAVIGATED = """<!doctype html><html><head><script>
  window.parent.postMessage({jsonrpc: "2.0", id: 99, method: "tools/call",
                             params: {name: "exfiltrate"}}, "*");
  window.addEventListener("message", (e) => {
    const secret = e.data && e.data.result && e.data.result.secret;
    if (secret) window.parent.postMessage({leaked: secret}, "*");
  });
</script></head><body>other<img src="https://undeclared.test/slow.png"></body></html>"""

STAYING_HOST = """<!doctype html><html><body>
<iframe id="f" sandbox="allow-scripts"></iframe>
<script>
  window.__report = null;
  window.addEventListener("message", (e) => {
    const env = e.data;
    if (!env || env.__unslothMcpApp !== TOKEN || env.__unslothMcpAppPort !== true) return;
    const port = e.ports[0];
    port.onmessage = (ev) => { if (ev.data && ev.data.report) window.__report = ev.data.report; };
    port.postMessage({jsonrpc: "2.0", id: 1, result: {secret: "PORT-REPLY"}});
  });
  document.getElementById("f").src = "https://mcp-app.test/staying.html";
</script></body></html>"""

# A defensive view: accepts a response only when it came from window.parent, which
# is the habit this design has to keep working.
STAYING = """<!doctype html><html><head>SHIM</head><body>
<script>
  window.addEventListener("message", (e) => {
    const secret = e.data && e.data.result && e.data.result.secret;
    if (!secret) return;
    if (e.source !== window.parent) {
      parent.postMessage({report: "REJECTED: event.source !== window.parent"}, "*");
      return;
    }
    e.source.postMessage({report: secret + " | origin=" + e.origin +
                                  " | source===parent"}, "*");
  });
</script>staying</body></html>"""


def _serve(page, name: str, body: str) -> None:
    page.route(
        f"**/{name}",
        lambda r: r.fulfill(status = 200, content_type = "text/html", body = body),
    )


def check_the_shim_lands_where_the_browser_runs_it(browser) -> None:
    """The first textual `<head>` can be inside a comment; a shim placed there
    never runs, which downstream looks like a view that just never initializes."""
    page = browser.new_page()
    page.goto("about:blank")
    call = f"([html, marker]) => {{ {shipped_inserter()} return withBridgeShim(html, marker); }}"
    inserted = page.evaluate(
        call,
        [
            "<!doctype html><html><!-- template has no <head> --><body>hi</body></html>",
            "BRIDGE_MARKER",
        ],
    )
    bare = page.evaluate(call, ["<p>a bare fragment</p>", "BRIDGE_MARKER"])
    page.close()

    head_at, head_end = inserted.find("<head>"), inserted.find("</head>")
    marker_at = inserted.find("BRIDGE_MARKER")
    comment_at, comment_end = inserted.find("<!--"), inserted.find("-->")
    print(
        f"[mcp-app-bridge] shim at {marker_at}, parsed head {head_at}..{head_end}, "
        f"comment mentioning a head at {comment_at}..{comment_end}"
    )
    if head_at < 0 or not head_at < marker_at < head_end:
        raise SystemExit(f"the shim did not land inside the parsed head:\n{inserted}")
    if comment_at < marker_at < comment_end:
        raise SystemExit(f"the shim landed inside the comment, where it never runs:\n{inserted}")
    if bare.lstrip().lower().startswith("<!doctype"):
        raise SystemExit(f"a template with no doctype gained one:\n{bare}")


def check_a_staying_view_still_gets_its_replies(browser, shim: str) -> None:
    """The port must not cost an ordinary view its replies, and must keep
    `event.source === window.parent` true for one that checks."""
    page = browser.new_page()
    _serve(page, "staying-host.html", STAYING_HOST.replace("TOKEN", f'"{TOKEN}"'))
    _serve(page, "staying.html", STAYING.replace("SHIM", shim))
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
    if "source===parent" not in report:
        raise SystemExit(f"event.source === window.parent no longer holds (report={report!r})")
    if f"origin={HOST_ORIGIN}" not in report:
        raise SystemExit(f"the re-dispatched reply lost the host origin (report={report!r})")


def main() -> None:
    # bridgeShim returns a bare body now; withBridgeShim is what wraps it in a real
    # script element, so these hand-built fixtures have to do the same.
    shim = f"<script>{shipped_shim()}</script>"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless = True)
        check_the_shim_lands_where_the_browser_runs_it(browser)
        check_a_staying_view_still_gets_its_replies(browser, shim)

        page = browser.new_page()
        _serve(page, "host.html", HOST.replace("TOKEN", f'"{TOKEN}"'))
        _serve(page, "seeded.html", SEEDED.replace("SHIM", shim))
        _serve(page, "other.html", NAVIGATED)
        # Keeps the navigated document's load event pending while its script runs.
        page.route("**/slow.png", lambda r: r.abort())
        page.goto("https://mcp-app.test/host.html")
        page.wait_for_timeout(2_500)

        accepted = page.evaluate("window.__accepted")
        old_accepted = page.evaluate("window.__oldAccepted")
        leaked = page.evaluate("window.__leaked")
        seen = page.evaluate("window.__seen")
        browser.close()

    def names(messages) -> list:
        return [
            m.get("params", {}).get("name")
            for m in messages
            if isinstance(m, dict) and isinstance(m.get("params"), dict)
        ]

    methods, old_methods = names(accepted), names(old_accepted)
    print(f"[mcp-app-bridge] messages reaching the host window: {len(seen)}")
    print(f"[mcp-app-bridge] a window+load-flag gate would accept: {old_methods}")
    print(f"[mcp-app-bridge] the port delivers:                    {methods}")
    print(f"[mcp-app-bridge] replies reaching the navigated document: {leaked}")

    if "refresh" not in methods:
        raise SystemExit("the seeded view's own tools/call never arrived; the shim is broken")
    if "exfiltrate" not in old_methods:
        raise SystemExit(
            "the navigated document did not beat the load event in this browser, so the "
            "run does not exercise the race this smoke is for; check the fixture"
        )
    if "exfiltrate" in methods:
        raise SystemExit("REGRESSION: a document the frame navigated to reached the bridge")
    if "WILDCARD-REPLY" not in leaked:
        raise SystemExit(
            "the wildcard reply never reached the navigated document, so this run does "
            "not exercise the leak window; check the fixture"
        )
    if "PORT-REPLY" in leaked:
        raise SystemExit(
            "REGRESSION: a reply sent down the seeded document's port still reached the "
            "page the frame navigated to"
        )
    print(
        "[mcp-app-bridge] the seeded view is served, the navigated document is refused, "
        "and no reply follows the frame to it"
    )


if __name__ == "__main__":
    sys.exit(main())
