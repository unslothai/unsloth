// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

type Tag = { tagName: string; type: string; textContent: string; parentNode: Head | null };
type Head = { children: Tag[]; appendChild(tag: Tag): Tag; removeChild(tag: Tag): Tag };

function fakeDocument() {
  const head: Head = {
    children: [],
    appendChild(tag) { this.children.push(tag); tag.parentNode = this; return tag; },
    removeChild(tag) { this.children = this.children.filter((t) => t !== tag); tag.parentNode = null; return tag; },
  };
  const doc = { head, createElement: (tagName: string): Tag => ({ tagName, type: "", textContent: "", parentNode: null }) };
  (globalThis as { document?: unknown }).document = doc;
  return head;
}

test("the scroll-lock sheet stays in <head> after the last lock ends", async () => {
  const head = fakeDocument();
  const { stylesheetSingleton } = await import("../src/lib/style-singleton.ts");
  const sheet = stylesheetSingleton();
  sheet.add("body[data-scroll-locked] { overflow: hidden; }");
  sheet.add("body[data-scroll-locked] { overflow: hidden; }");
  sheet.remove();
  sheet.remove();
  assert.equal(head.children.length, 1);
  const tag = head.children[0];
  sheet.add("body[data-scroll-locked] { overflow: hidden; }");
  assert.equal(head.children.length, 1, "a later lock reuses the tag");
  assert.equal(head.children[0], tag);
  sheet.add("body[data-scroll-locked] { overflow: hidden; margin-right: 15px; }");
  assert.equal(head.children.length, 1);
  assert.match(tag.textContent, /margin-right: 15px/, "a changed sheet is updated in place");
});

test("every bundle resolves react-style-singleton to the keeping singleton", () => {
  const config = readFileSync(new URL("../vite.config.ts", import.meta.url), "utf8");
  assert.match(config, /"react-style-singleton": path\.resolve\(__dirname, "\.\/src\/lib\/style-singleton\.ts"\)/);
  for (const name of ["vite.long-thread.config.ts", "vite.reasoning.config.ts"]) {
    assert.match(readFileSync(new URL(`../${name}`, import.meta.url), "utf8"), /mergeConfig\(base,/, name);
  }
  const shim = readFileSync(new URL("../src/lib/style-singleton.ts", import.meta.url), "utf8");
  for (const name of ["styleSingleton", "styleHookSingleton", "stylesheetSingleton"]) {
    assert.match(shim, new RegExp(`export const ${name} =`));
  }
});
