# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`delete_message` reaches Delete where the app puts it: the last item of the reply's More menu.

#12735 took Delete off the assistant action bar and made it the More menu's last item. The scene
still looked for a "Delete message" button on the bar, waited out `ACTION_BAR_WAIT_MS` for it, and
reported `delete_message: NOT RUN -- no Delete button`, which fails the real-path session's
liveness gate on every pull request.

Two checks. The first runs the shipped `DELETE_JS` and `dom.js` in node against a shim of the
handful of DOM calls they make, so the menu walk is exercised rather than re-implemented. The
second reads thread.tsx and requires every control name the scene asks for to still be rendered
there, so the next move of a control fails here, with its name, instead of as a NOT RUN in a
browser job.
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
const menuNames = JSON.parse(process.argv[4]);

// A node matches a selector when the selector is in its `sel` list: dom.js reads the app through
// a short fixed list of selectors, and a real matcher would only be a second thing to get wrong.
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

const state = { menuOpen: false, escapes: 0, menuLookups: 0 };
// The portal mounting is a change to body's children, delivered to observers as the app's would be.
const observers = [];
class MutationObserver {
  constructor(cb) { this.cb = cb; }
  observe() { observers.push(this); }
  disconnect() { const i = observers.indexOf(this); if (i >= 0) observers.splice(i, 1); }
}
const setMenu = (open) => {
  state.menuOpen = open;
  for (const o of observers.slice()) o.cb();
};
// The menu mounts this many paints after the trigger is pressed, as a portal does a commit later.
let paintsUntilMenu = 0;
const user = node(['[data-role]', '[data-role="user"]'], {});
let messages = [user];

// Radix: the trigger opens on pointerdown, an item selects on click.
const more = node(["button"], {
  "aria-label": "More",
  on: (ev) => { if (ev && ev.type === "pointerdown") paintsUntilMenu = 3; },
});
const bar = node([".aui-assistant-action-bar-root"], {}, [more]);
const assistant = node(['[data-role]', '[data-role="assistant"]'], {}, [bar]);
messages.push(assistant);

const items = menuNames.map((name) => node([".aui-action-bar-more-item"], {
  text: name,
  click: () => {
    if (name !== "Delete") return;
    setMenu(false);
    assistant.isConnected = false;
    messages = messages.filter((m) => m !== assistant);
  },
}));
const menu = node([".aui-action-bar-more-content"], {}, items);

const document = {
  body: { sel: ["body"] },
  querySelector: (s) => document.querySelectorAll(s)[0] || null,
  querySelectorAll: (s) => {
    if (s === "[data-role]") return messages;
    if (s === '[data-role="assistant"]') return messages.filter((m) => m === assistant);
    if (s === ".aui-action-bar-more-content") {
      state.menuLookups += 1;
      return state.menuOpen ? [menu] : [];
    }
    return [];
  },
  dispatchEvent: (ev) => {
    if (ev && ev.key === "Escape") { setMenu(false); state.escapes += 1; }
    return true;
  },
};

class PointerEvent { constructor(type, init) { this.type = type; Object.assign(this, init || {}); } }
class KeyboardEvent { constructor(type, init) { this.type = type; Object.assign(this, init || {}); } }

const window = {};
window.__sbNextPaint = () => new Promise((r) => setTimeout(() => {
  if (paintsUntilMenu > 0 && --paintsUntilMenu === 0) setMenu(true);
  r();
}, 5));
window.addEventListener = () => {};

(new Function("window", "document", "PointerEvent", "MutationObserver", domSrc))(
  window, document, PointerEvent, MutationObserver);

const run = (new Function(
  "window", "document", "PointerEvent", "KeyboardEvent", "performance",
  "return (" + deleteSrc + ")",
))(window, document, PointerEvent, KeyboardEvent, performance);

run({ timeoutMs: 2000, waitForButtonMs: 300 }).then((out) => {
  out.menuOpenAfter = state.menuOpen;
  out.escapes = state.escapes;
  out.menuLookups = state.menuLookups;
  console.log(JSON.stringify(out));
  process.exit(0);
}, (err) => { console.error(String((err && err.stack) || err)); process.exit(1); });
"""


def _run(menu_names: list[str]) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.cjs"
        harness.write_text(HARNESS_JS, encoding = "utf-8")
        delete_js = Path(tmp) / "delete.js"
        delete_js.write_text(DELETE_JS.strip(), encoding = "utf-8")
        done = subprocess.run(
            [node, str(harness), str(DOM_JS), str(delete_js), json.dumps(menu_names)],
            capture_output = True,
            text = True,
            timeout = 60,
        )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_delete_opens_the_more_menu_and_selects_its_delete_item():
    out = _run(["Edit response", "Fork in new chat", "Delete"])
    assert out["ran"] is True, out
    assert out["ms"] is not None, "the reply never left the document after Delete was selected"
    assert (out["before"], out["after"]) == (2, 1), out
    # The menu mounts three paints late. The document-wide lookup for it runs on the first look
    # and when body's children change, not once per paint while waiting.
    assert out["menuLookups"] <= 2, out


def test_a_menu_without_delete_is_reported_and_closed():
    out = _run(["Edit response", "Fork in new chat"])
    assert out["ran"] is False
    assert "no Delete item in the More menu" in out["reason"], out["reason"]
    assert (
        out["menuOpenAfter"] is False and out["escapes"] == 1
    ), "a menu the action opened and could not use must be closed, or it covers the next action"


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
    """AssistantActionBar and every thread.tsx component it renders, transitively.

    The scene acts on the last ASSISTANT message, so a control only counts if the assistant bar
    renders it: `More` and `Delete` also exist in the user message's menu and as reusable
    components, and a whole-file search would still find them after the assistant bar dropped them.
    """
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
    assert (
        {"More", "Copy"} <= buttons and "Delete" in items
    ), "parsed fewer control names than the scene uses; this check would pass on nothing"
    tooltips = set(re.findall(r"tooltip=\"([^\"]+)\"", tsx))
    # An item's label is its last line of text before the closing tag. Its attributes can hold
    # arrow functions, so the opening tag cannot be skipped with a `[^>]*` match.
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
