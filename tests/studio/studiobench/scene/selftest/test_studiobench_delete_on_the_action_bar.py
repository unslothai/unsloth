# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`delete_message` reaches Delete where the app puts it: a button on the reply's action bar.

#12735 moved Delete off the assistant action bar into the More menu, and #13615 put it back on the
bar as an inline button. Each move left the scene looking in the old place, waiting out its bound
and reporting `delete_message: NOT RUN`, which fails the real-path session's liveness gate.

Two checks. The first runs the shipped `DELETE_JS` and `dom.js` in node against a shim of the
handful of DOM calls they make, so the lookup is exercised rather than re-implemented. The second
reads thread.tsx and requires every control name the scene asks for to still be rendered there, so
the next move of a control fails here, with its name, instead of as a NOT RUN in a browser job.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.scene.actions import DELETE_JS  # noqa: E402

SCENE = Path(__file__).resolve().parents[1]
DOM_JS = SCENE / "dom.js"
REPO = Path(__file__).resolve().parents[5]
THREAD_TSX = REPO / "studio" / "frontend" / "src" / "components" / "assistant-ui" / "thread.tsx"

HARNESS_JS = r"""
const fs = require("fs");
const domSrc = fs.readFileSync(process.argv[2], "utf8");
const deleteSrc = fs.readFileSync(process.argv[3], "utf8");
const barNames = JSON.parse(process.argv[4]);

const node = (sel, attrs, kids) => {
  const self = {
    sel: sel,
    kids: kids || [],
    isConnected: true,
    getAttribute: (k) => (attrs && k in attrs ? attrs[k] : null),
    textContent: (attrs && attrs.text) || "",
    dispatchEvent: (ev) => { if (attrs && attrs.on) attrs.on(ev); return true; },
    click: () => { if (attrs && attrs.click) attrs.click(); },
    closest: () => null,
  };
  const all = (s) => {
    const out = [];
    for (const k of self.kids) {
      if (k.sel.indexOf(s) >= 0) out.push(k);
      for (const g of k.querySelectorAll(s)) out.push(g);
    }
    return out;
  };
  self.querySelectorAll = all;
  self.querySelector = (s) => all(s)[0] || null;
  return self;
};

const state = { escapes: 0 };
const observers = [];
class MutationObserver {
  constructor(cb) { this.cb = cb; }
  observe() { observers.push(this); }
  disconnect() { const i = observers.indexOf(this); if (i >= 0) observers.splice(i, 1); }
}
const user = node(['[data-role]', '[data-role="user"]'], {});
let messages = [user];
let assistant = null;
const buttons = barNames.map((name) => node(["button"], {
  "aria-label": name,
  click: () => {
    if (name !== "Delete") return;
    assistant.isConnected = false;
    messages = messages.filter((m) => m !== assistant);
  },
}));
const bar = node([".aui-assistant-action-bar-root"], {}, buttons);
assistant = node(['[data-role]', '[data-role="assistant"]'], {}, [bar]);
messages.push(assistant);

const document = {
  body: { sel: ["body"] },
  querySelector: (s) => document.querySelectorAll(s)[0] || null,
  querySelectorAll: (s) => {
    if (s === "[data-role]") return messages;
    if (s === '[data-role="assistant"]') return messages.filter((m) => m === assistant);
    return [];
  },
  dispatchEvent: (ev) => {
    if (ev && ev.key === "Escape") state.escapes += 1;
    return true;
  },
};

class PointerEvent { constructor(type, init) { this.type = type; Object.assign(this, init || {}); } }
class KeyboardEvent { constructor(type, init) { this.type = type; Object.assign(this, init || {}); } }

const window = {};
window.__sbNextPaint = () => new Promise((r) => setTimeout(r, 5));
window.addEventListener = () => {};

(new Function("window", "document", "PointerEvent", "MutationObserver", domSrc))(
  window, document, PointerEvent, MutationObserver);

const run = (new Function(
  "window", "document", "PointerEvent", "KeyboardEvent", "performance",
  "return (" + deleteSrc + ")",
))(window, document, PointerEvent, KeyboardEvent, performance);

run({ timeoutMs: 2000, waitForButtonMs: 300 }).then((out) => {
  out.escapes = state.escapes;
  console.log(JSON.stringify(out));
  process.exit(0);
}, (err) => { console.error(String((err && err.stack) || err)); process.exit(1); });
"""


def _run(bar_names: list[str]) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.cjs"
        harness.write_text(HARNESS_JS, encoding = "utf-8")
        delete_js = Path(tmp) / "delete.js"
        delete_js.write_text(DELETE_JS.strip(), encoding = "utf-8")
        done = subprocess.run(
            [node, str(harness), str(DOM_JS), str(delete_js), json.dumps(bar_names)],
            capture_output = True,
            text = True,
            timeout = 60,
        )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_delete_clicks_the_bar_button_and_the_reply_leaves():
    out = _run(["Copy", "Edit response", "Delete", "More"])
    assert out["ran"] is True, out
    assert out["ms"] is not None, "the reply never left the document after Delete was clicked"
    assert (out["before"], out["after"]) == (2, 1), out
    assert out["escapes"] == 0, "nothing was opened, so nothing should be dismissed"


def test_a_bar_without_delete_is_reported_not_run():
    out = _run(["Copy", "Edit response", "More"])
    assert out["ran"] is False
    assert "no Delete button" in out["reason"], out["reason"]


def _names_the_scene_asks_for() -> tuple[set[str], set[str]]:
    text = (SCENE / "actions.py").read_text(encoding = "utf-8") + (DOM_JS).read_text(encoding = "utf-8")
    buttons = set(re.findall(r"\b(?:waitForActionButton|actionButton)\(\"([^\"]+)\"", text))
    items = set(re.findall(r"\bopenMenuAndFind\([^,]+,\s*\"([^\"]+)\"", text))
    return buttons, items


def _component_body(tsx: str, name: str) -> str | None:
    """The source of a component defined in thread.tsx, `const X: FC = ...` or `function X(`."""
    match = re.search(rf"^(?:const {name}\b[^\n]*=>\s*\{{|function {name}\()", tsx, re.M)
    if not match:
        return None
    end = tsx.find("\n}", match.end())
    return tsx[match.start() : end if end != -1 else len(tsx)]


def _assistant_bar_source(tsx: str) -> str:
    """Collects only what AssistantActionBar renders; a whole-file search would find the user's More too."""
    seen: dict[str, str] = {}
    queue = ["AssistantActionBar"]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        body = _component_body(tsx, name)
        if body is None:
            continue
        seen[name] = body
        queue.extend(re.findall(r"<([A-Z][A-Za-z0-9]*)\b(?!\.)", body))
    assert "AssistantActionBar" in seen, "could not find AssistantActionBar in thread.tsx"
    return "\n".join(seen.values())


def test_every_control_the_scene_asks_for_is_still_rendered():
    """Bar buttons by their tooltip, menu items by their text, read out of what the assistant
    message's action bar renders in thread.tsx."""
    tsx = _assistant_bar_source(THREAD_TSX.read_text(encoding = "utf-8"))
    buttons, items = _names_the_scene_asks_for()
    assert {
        "Copy",
        "Delete",
    } <= buttons, "parsed fewer control names than the scene uses; this check would pass on nothing"
    tooltips = set(re.findall(r"tooltip=\"([^\"]+)\"", tsx))
    # Attributes can hold arrow functions, so the opening tag cannot be skipped with [^>]*.
    rendered_items = set()
    for body in re.findall(
        r"<ActionBarMorePrimitive\.Item\b([\s\S]*?)</ActionBarMorePrimitive\.Item>", tsx
    ):
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if lines and not lines[-1].startswith(("<", "{", "/")):
            rendered_items.add(lines[-1])
    missing = sorted(buttons - tooltips) + sorted(items - rendered_items)
    assert not missing, (
        f"the studiobench scene asks for {missing}, which the assistant action bar in thread.tsx "
        "no longer renders as a tooltip or a More-menu item, so that action would report NOT RUN"
    )
