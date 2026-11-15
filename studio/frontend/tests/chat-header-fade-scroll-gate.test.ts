// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Over a flat page the header fades round to a one-level seam (Firefox any space, WebKit sRGB), so
// they stay hidden until their own thread scrolls under the header.

import assert from "node:assert/strict";
import test from "node:test";

import postcss from "postcss";

import { readSrc } from "./helpers/kit.ts";

class FakeElement {
  attrs = new Set<string>();
  classList: { contains: (c: string) => boolean };
  scrollTop = 0;
  listeners = new Map<string, { fn: (e: Event) => void; capture: boolean }>();
  dock: boolean;
  constructor(classes: string[] = [], dock = false) {
    this.dock = dock;
    this.classList = { contains: (c) => classes.includes(c) };
  }
  closest(selector: string) {
    return selector === ".chat-full-view-dock" && this.dock ? this : null;
  }
  toggleAttribute(name: string, on: boolean) {
    if (on) this.attrs.add(name);
    else this.attrs.delete(name);
  }
  addEventListener(type: string, fn: (e: Event) => void, opts: { capture: boolean }) {
    this.listeners.set(type, { fn, capture: opts.capture });
  }
  removeEventListener(type: string) {
    this.listeners.delete(type);
  }
  fire(target: FakeElement) {
    const l = this.listeners.get("scroll");
    l?.fn.call(this, { target } as unknown as Event);
  }
}
(globalThis as { HTMLElement?: unknown }).HTMLElement = FakeElement;
const { trackThreadScroll, THREAD_SCROLLED_ATTR } = await import("../src/features/chat/thread-scroll-flag.ts");

test("the flag follows the thread viewport's scrollTop, captured on the fade's parent", () => {
  const host = new FakeElement();
  const cleanup = trackThreadScroll(host as unknown as HTMLElement);
  assert.equal(host.listeners.get("scroll")?.capture, true, "scroll does not bubble: capture phase");
  const viewport = new FakeElement(["aui-thread-viewport"]);
  viewport.scrollTop = 40;
  host.fire(viewport);
  assert.ok(host.attrs.has(THREAD_SCROLLED_ATTR));
  viewport.scrollTop = 0;
  host.fire(viewport);
  assert.ok(!host.attrs.has(THREAD_SCROLLED_ATTR), "back at the top: fade hidden again");
  cleanup?.();
  assert.equal(host.listeners.size, 0);
});

test("other scrollers and the full-view dock thread never set the flag", () => {
  const host = new FakeElement();
  trackThreadScroll(host as unknown as HTMLElement);
  const codeBlock = new FakeElement(["code-scroller"]);
  codeBlock.scrollTop = 99;
  host.fire(codeBlock);
  const dock = new FakeElement(["aui-thread-viewport"], true);
  dock.scrollTop = 99;
  host.fire(dock);
  assert.equal(host.attrs.size, 0);
});

test("both fades are hidden by default and shown only under a scrolled direct parent", () => {
  const rules: postcss.Rule[] = [];
  postcss.parse(readSrc("index.css")).walkRules((rule) => {
    if (/chat-header-fade, \.compare-pane-fade\)$/.test(rule.selector)) rules.push(rule);
  });
  assert.deepEqual(rules.map((r) => r.selector), [
    ":is(.chat-header-fade, .compare-pane-fade)",
    "[data-thread-scrolled] > :is(.chat-header-fade, .compare-pane-fade)",
  ]);
  assert.match(rules[0].toString(), /opacity: 0/);
  assert.match(rules[1].toString(), /opacity: 1/);
});

test("each fade's parent carries the scroll tracker, and the fades keep Oklab", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  for (const fade of ["compare-pane-fade", "chat-header-fade"]) {
    const at = page.indexOf(`className="${fade} `);
    assert.ok(at > 0, fade);
    const parentOpen = page.lastIndexOf("<div", page.lastIndexOf("<div", at) - 1);
    assert.match(page.slice(parentOpen, at), /ref=\{trackThreadScroll\}/, `${fade} parent`);
    const cls = page.slice(at, page.indexOf('"', at + 11));
    assert.match(cls, /bg-gradient-to-b from-background to-transparent/);
    assert.doesNotMatch(cls, /\/srgb/, "sRGB bands in WebKit");
  }
});
