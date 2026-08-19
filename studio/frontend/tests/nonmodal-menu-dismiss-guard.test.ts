// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A modal Radix menu absorbs the press that dismisses it, because `pointer-events: none` on
// <body> takes everything underneath out of the hit test. `modal={false}` removes that write --
// which is the point, since it is an inherited property and invalidates style for the whole
// thread -- and it removes the absorption with it. Radix's outside handler dismisses the menu
// but never cancels the event, so one press on a control beside the menu both closes it and
// fires that control. In the assistant action bar that control is an unconfirmed
// "Delete message" two buttons from the trigger: measured on the heavy-thread smoke page,
// eleven separate dismissal gestures each removed a message on chromium and nine on webkit.
//
// lib/menu-dismiss.ts restores the absorption by swallowing exactly that click. It only helps a
// menu that mounts it, so this sweeps the tree for every layer that opts OUT of the modal one
// with `modal={false}`, and one added tomorrow has to be decided about rather than defaulting to
// unguarded. Radix Popover is non-modal by DEFAULT and carries no such prop, so it is outside
// this sweep and outside this change.
//
// tests/studio/probe_dismiss_guard.py carries the same question against a real engine with real
// input, and is the gate in CI.

import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SRC = fileURLToPath(new URL("../src/", import.meta.url));

const TAG_NAME = /^<([A-Za-z0-9_.]+)/;
const GUARD = /<MenuDismissGuard\s*\/>/;

/**
 * Non-modal menus that deliberately do not swallow their dismissing click, and why. Swallowing
 * costs the user a click, so a menu that can open without being asked for cannot afford it.
 */
const UNGUARDED = new Map<string, string>([
  [
    "components/app-sidebar.tsx <DropdownMenu>",
    "the sidebar's More flyout opens on POINTER ENTER and stays open for 180ms after the pointer " +
      "leaves, so an unconditional swallow eats an ordinary click on the nav row the pointer was " +
      "heading for. A press pins it open too, and that path could take one, but the flyout has " +
      "been non-modal since #6763 rather than since #8992, so it is a defect of its own.",
  ],
]);

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
 * Where `<tag` opens an element of exactly that name rather than one whose name merely starts
 * with it. `<DropdownMenu` is a prefix of `<DropdownMenuTrigger`, which never closes as
 * `</DropdownMenu>`, so counting prefixes leaves the depth permanently unbalanced and the body
 * below runs to end of file.
 */
function openingTagAt(source: string, tag: string, from: number): number {
  let at = source.indexOf(`<${tag}`, from);
  while (at !== -1) {
    const next = source[at + tag.length + 1] ?? "";
    if (next === "" || !/[A-Za-z0-9_.]/.test(next)) return at;
    at = source.indexOf(`<${tag}`, at + 1);
  }
  return -1;
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
    const nextOpen = openingTagAt(source, tag, i + 1);
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
  throw new Error(`unbalanced <${tag}> around the menu at offset ${at}`);
}

/** Every `modal={false}` menu in the tree, as `relative/path.tsx <Tag>`, with its own body. */
function nonModalMenus(): { id: string; body: string }[] {
  const menus: { id: string; body: string }[] = [];
  for (const file of sources(SRC)) {
    const source = readFileSync(file, "utf8");
    let at = source.indexOf("modal={false}");
    while (at !== -1) {
      const { tag, body } = element(source, at);
      menus.push({ id: `${path.relative(SRC, file)} <${tag}>`, body });
      at = source.indexOf("modal={false}", at + 1);
    }
  }
  return menus;
}

test("every non-modal menu either mounts MenuDismissGuard or is a listed exception", () => {
  const menus = nonModalMenus();
  // A sweep that resolves nothing would pass without checking anything.
  assert.ok(
    menus.length >= 2,
    `only found ${menus.length} non-modal menus; the sweep is not reaching the tree`,
  );
  const offenders = menus
    .filter(({ id, body }) => !GUARD.test(body) && !UNGUARDED.has(id))
    .map(({ id }) => id);
  assert.deepEqual(
    offenders,
    [],
    `a non-modal menu with no guard lets the press that dismisses it fire whatever is underneath. Add the guard, or list it in UNGUARDED with the reason. Offenders: ${offenders.join(", ")}`,
  );
});

test("the exception list does not outlive the menus it excuses", () => {
  const ids = new Set(nonModalMenus().map(({ id }) => id));
  for (const id of UNGUARDED.keys()) {
    assert.ok(
      ids.has(id),
      `${id} is excused from the guard but is no longer a non-modal menu; drop the entry`,
    );
  }
});

test("the element scan does not confuse a tag with one that merely starts the same", () => {
  // `<DropdownMenu>` against `<DropdownMenuTrigger>`: the real shape in app-sidebar.tsx, and the
  // one that silently returned the rest of the file as the menu's body.
  const source = [
    "<DropdownMenu modal={false}>",
    "  <DropdownMenuTrigger />",
    "</DropdownMenu>",
    "<MenuDismissGuard />",
  ].join("\n");
  const { tag, body } = element(source, source.indexOf("modal={false}"));
  assert.equal(tag, "DropdownMenu");
  assert.ok(
    !GUARD.test(body),
    "the scan ran past the menu's own closing tag and swallowed the next element",
  );
});

test("the guard component is what mounts the watcher", () => {
  const guard = readFileSync(
    path.join(SRC, "lib/menu-dismiss-guard.tsx"),
    "utf8",
  );
  assert.match(
    guard,
    /useDismissingClickGuard\(\)/,
    "MenuDismissGuard must install the document watcher, or every mount above is decoration",
  );
});
