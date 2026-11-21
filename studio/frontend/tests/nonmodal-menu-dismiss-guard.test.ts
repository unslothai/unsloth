// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SRC = fileURLToPath(new URL("../src/", import.meta.url));

const TAG_NAME = /^<([A-Za-z0-9_.]+)/;
const GUARD = /<MenuDismissGuard\b(?=[^>]*\btriggerRef=\{)[^>]*\/>/;
// Boundary-aware so `aria-modal={false}` and `data-modal={false}` do not count.
// Indentation-independent so a tab or a reflowed line is still scanned.
const MODAL_FALSE = /(?<![\w-])modal=\{false\}/g;

const UNGUARDED = new Map<string, string>([
  [
    "components/app-sidebar.tsx <DropdownMenu>",
    "the sidebar's More flyout opens on POINTER ENTER and stays open for 180ms after the pointer " +
      "leaves, so an unconditional swallow eats an ordinary click on the nav row the pointer was " +
      "heading for. A press pins it open too, and that path could take one, but the flyout has " +
      "been non-modal since #6763 rather than since #8992, so it is a defect of its own.",
  ],
  [
    "features/browser/browser-toggle.tsx <DropdownMenu>",
    "the chat header's new tab menu opens on POINTER ENTER and closes 200ms after the pointer " +
      "leaves, so a swallow would eat the click the pointer was heading for. A press on its " +
      "button opens a tab rather than the menu, so no press ever opens it.",
  ],
]);

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

function openingTagAt(source: string, tag: string, from: number): number {
  let at = source.indexOf(`<${tag}`, from);
  while (at !== -1) {
    const next = source[at + tag.length + 1] ?? "";
    if (next === "" || !/[A-Za-z0-9_.]/.test(next)) return at;
    at = source.indexOf(`<${tag}`, at + 1);
  }
  return -1;
}

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

const relativeId = (file: string): string =>
  path.relative(SRC, file).split(path.sep).join("/");

function elementsIn(
  source: string,
  idFor: (tag: string) => string = (tag) => `<fixture> <${tag}>`,
): { id: string; body: string }[] {
  const menus: { id: string; body: string }[] = [];
  const modalFalse = new RegExp(MODAL_FALSE.source, "g");
  while (true) {
    const match = modalFalse.exec(source);
    if (match === null) {
      break;
    }
    const { tag, body } = element(source, match.index);
    menus.push({ id: idFor(tag), body });
  }
  return menus;
}

function nonModalMenus(): { id: string; body: string }[] {
  const menus: { id: string; body: string }[] = [];
  for (const file of sources(SRC)) {
    const source = readFileSync(file, "utf8");
    menus.push(...elementsIn(source, (tag) => `${relativeId(file)} <${tag}>`));
  }
  return menus;
}

test("every non-modal menu either mounts MenuDismissGuard or is a listed exception", () => {
  const menus = nonModalMenus();
  // Require at least one menu so an empty sweep cannot pass.
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
  const source = [
    "<DropdownMenu modal={false}>",
    "  <DropdownMenuTrigger />",
    "</DropdownMenu>",
    "<MenuDismissGuard triggerRef={triggerRef} />",
  ].join("\n");
  const { tag, body } = element(source, source.indexOf("modal={false}"));
  assert.equal(tag, "DropdownMenu");
  assert.ok(
    !GUARD.test(body),
    "the scan ran past the menu's own closing tag and swallowed the next element",
  );
});

test("the non-modal scan ignores indentation and aria-modal", () => {
  const source = [
    "<DropdownMenu",
    "\tmodal={false}",
    ">",
    "  <MenuDismissGuard triggerRef={triggerRef} />",
    "</DropdownMenu>",
    '<div role="dialog" aria-modal={false} />',
    "<div data-modal={false} />",
  ].join("\n");
  const menus = elementsIn(source);
  assert.deepEqual(
    menus.map(({ id }) => id),
    ["<fixture> <DropdownMenu>"],
    "a tab-indented prop must be found; aria-modal and data-modal must not",
  );
  assert.match(menus[0].body, GUARD);
});

test("menu ids are separator-independent", () => {
  for (const { id } of nonModalMenus()) {
    assert.ok(!id.includes("\\"), `${id} carries a host separator`);
  }
});

test("the guard component is what mounts the watcher", () => {
  const guard = readFileSync(
    path.join(SRC, "lib/menu-dismiss-guard.tsx"),
    "utf8",
  );
  assert.match(
    guard,
    /useDismissingClickGuard\(triggerRef\)/,
    "MenuDismissGuard must give the document watcher its trigger, or every mount above loses focus restoration",
  );
});
