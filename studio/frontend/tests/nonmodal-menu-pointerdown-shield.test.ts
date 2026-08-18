// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `swallowDismissingClick` covers every control that acts on `click`. Two do not wait for one:
// Radix Slider commits in onPointerDown -> onSlideStart -> updateValues -> onValueChange, and
// Radix Select opens in onPointerDown. So on desktop, where the Run settings panel is a
// co-visible <aside> rather than an overlay, one press on a visible ParamSlider with a composer,
// action-bar or project menu open both dismissed the menu and wrote a new inference value:
// measured with real page.mouse, temperature 0.6 -> 1.7 on chromium and 0.6 -> 1.69 on firefox
// and webkit, reaching chat_settings in studio.db and surviving a reload. On the merge base the
// same press landed on HTML with body pointer-events: none and changed nothing.
//
// The fix is a document attribute driving a rule whose RIGHTMOST compound is the committing
// control, so what it invalidates is the sliders and not the thread. These tests pin the three
// halves that can rot independently: the ref-counted marker, the rule that reads it, and the
// rule that every non-modal menu mounts the guard which sets it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { readdirSync, statSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SRC = fileURLToPath(new URL("../src/", import.meta.url));

const read = (relative: string): string =>
  readFileSync(path.join(SRC, relative), "utf8");

// ---------------------------------------------------------------------------
// 1. The marker itself, ref-counted, against a fake document.
// ---------------------------------------------------------------------------

const attributes = new Set<string>();
const documentElement = {
  setAttribute: (name: string) => {
    attributes.add(name);
  },
  removeAttribute: (name: string) => {
    attributes.delete(name);
  },
  hasAttribute: (name: string) => attributes.has(name),
};
// menu-dismiss.ts touches `document` only from inside its functions, so a stub installed here is
// enough and no renderer is needed.
(globalThis as { document?: unknown }).document = { documentElement };

const { markNonModalMenuOpen, NON_MODAL_MENU_OPEN_ATTRIBUTE } = await import(
  "../src/lib/menu-dismiss.ts"
);

const marked = (): boolean => attributes.has(NON_MODAL_MENU_OPEN_ATTRIBUTE);

test("an open non-modal menu marks the document, and closing it unmarks", () => {
  assert.equal(marked(), false);
  const release = markNonModalMenuOpen();
  assert.equal(marked(), true, "the shield must be up while the menu is open");
  release();
  assert.equal(marked(), false, "and gone the moment the last menu closes");
});

test("two open menus need two releases", () => {
  const first = markNonModalMenuOpen();
  const second = markNonModalMenuOpen();
  first();
  assert.equal(
    marked(),
    true,
    "a submenu unmounting must not drop the shield while its parent is still open",
  );
  second();
  assert.equal(marked(), false);
});

test("a doubled release does not drop another menu's shield", () => {
  const first = markNonModalMenuOpen();
  const second = markNonModalMenuOpen();
  first();
  first();
  assert.equal(
    marked(),
    true,
    "React can run an effect cleanup twice under StrictMode; the second must be a no-op",
  );
  second();
  assert.equal(marked(), false);
});

// ---------------------------------------------------------------------------
// 2. The rule that reads the marker.
// ---------------------------------------------------------------------------

const CSS = read("index.css");

const SLIDER_SHIELD =
  /\[data-nonmodal-menu-open\][^{]*\[data-slot="slider"\][^{]*\{[^}]*pointer-events:\s*none/s;
const SELECT_SHIELD =
  /\[data-nonmodal-menu-open\][^{]*\[data-slot="select-trigger"\][^{]*\{[^}]*pointer-events:\s*none/s;
const POPPER_RESTORE =
  /\[data-nonmodal-menu-open\]\s*\[data-radix-popper-content-wrapper\]\s*\[data-slot="slider"\][^{]*\{[^}]*pointer-events:\s*auto/s;
const BODY_SHIELD = /\bbody\s*\{[^}]*pointer-events:\s*none/s;
const TAG_NAME = /^<([A-Za-z0-9_.]+)/;
const GUARD = /<MenuDismissGuard\s*\/>/;

test("the marker takes pointerdown-committing controls out of the hit test", () => {
  assert.match(
    CSS,
    SLIDER_SHIELD,
    "without this rule the dismissing press reaches the slider track and Radix commits a value " +
      "on pointerdown, which no later click swallow can undo",
  );
  assert.match(
    CSS,
    SELECT_SHIELD,
    "Select opens on pointerdown too, so a slider-only rule leaves the panel's selects exposed",
  );
});

test("a slider inside the open menu is still the user's to press", () => {
  assert.match(
    CSS,
    POPPER_RESTORE,
    "the shield is for controls UNDER the menu; one inside it must keep working",
  );
});

test("the branch does not put the body shield back", () => {
  assert.doesNotMatch(
    CSS,
    BODY_SHIELD,
    "shielding <body> is the inherited-property write this branch exists to remove",
  );
});

// ---------------------------------------------------------------------------
// 3. Every non-modal menu mounts the guard that sets the marker.
// ---------------------------------------------------------------------------

/** Every .tsx under src/, so a new non-modal menu anywhere is covered without a list. */
function sources(dir: string, found: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) {
      sources(full, found);
    } else if (entry.endsWith(".tsx")) {
      found.push(full);
    }
  }
  return found;
}

/**
 * The JSX element that carries `modal={false}` at `at`, by counting its own opening and closing
 * tags. A file-level "as many guards as menus" count would pass a file where one menu mounts two.
 */
function element(source: string, at: number): { tag: string; body: string } {
  const open = source.lastIndexOf("<", at);
  if (open === -1) {
    throw new Error("modal={false} outside any element");
  }
  const tag = TAG_NAME.exec(source.slice(open))?.[1];
  if (!tag) {
    throw new Error("no tag name at the element carrying modal={false}");
  }
  let depth = 0;
  let i = open;
  while (i < source.length) {
    const nextOpen = source.indexOf(`<${tag}`, i + 1);
    const nextClose = source.indexOf(`</${tag}>`, i + 1);
    if (nextClose === -1) {
      break;
    }
    if (nextOpen !== -1 && nextOpen < nextClose) {
      depth++;
      i = nextOpen;
      continue;
    }
    if (depth === 0) {
      return { tag, body: source.slice(open, nextClose) };
    }
    depth--;
    i = nextClose;
  }
  return { tag, body: source.slice(open) };
}

test("every non-modal menu in the tree mounts MenuDismissGuard", () => {
  const offenders: string[] = [];
  let menus = 0;
  for (const file of sources(SRC)) {
    const source = readFileSync(file, "utf8");
    let at = source.indexOf("modal={false}");
    while (at !== -1) {
      menus++;
      const { tag, body } = element(source, at);
      if (!GUARD.test(body)) {
        offenders.push(`${path.relative(SRC, file)} <${tag}>`);
      }
      at = source.indexOf("modal={false}", at + 1);
    }
  }
  // A sweep that resolves nothing would pass without measuring anything.
  assert.ok(
    menus >= 10,
    `only found ${menus} non-modal menus; the sweep is not reaching the tree`,
  );
  assert.deepEqual(
    offenders,
    [],
    `a non-modal menu with no guard has neither the click swallow nor the pointerdown shield: the sidebar's "More" destinations menu was exactly this, and a ParamSlider on the Images page was hittable underneath it. Offenders: ${offenders.join(", ")}`,
  );
});
