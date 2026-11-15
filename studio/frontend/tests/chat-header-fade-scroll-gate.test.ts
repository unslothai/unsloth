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
  parent: FakeElement | null;
  dock: boolean;
  constructor(classes: string[] = [], parent: FakeElement | null = null, dock = false) {
    this.parent = parent;
    this.dock = dock;
    this.classList = { contains: (c) => classes.includes(c) };
  }
  closest(selector: string): FakeElement | null {
    for (let el: FakeElement | null = this; el; el = el.parent) {
      if (selector === ".chat-full-view-dock" ? el.dock : el.attrs.has(selector.slice(1, -1))) return el;
    }
    return null;
  }
  setAttribute(name: string) {
    this.attrs.add(name);
  }
  removeAttribute(name: string) {
    this.attrs.delete(name);
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
  // Capture phase: every tracked ancestor of the target sees the event, outermost first.
  fire(target: FakeElement) {
    const chain: FakeElement[] = [];
    for (let el: FakeElement | null = target; el; el = el.parent) chain.unshift(el);
    for (const el of chain) el.listeners.get("scroll")?.fn.call(el, { target } as unknown as Event);
  }
}
(globalThis as { HTMLElement?: unknown }).HTMLElement = FakeElement;
const { trackThreadScroll, THREAD_SCROLLED_ATTR } = await import("../src/features/chat/thread-scroll-flag.ts");

test("the flag follows the thread viewport's scrollTop, captured on the fade's parent", () => {
  const host = new FakeElement();
  const cleanup = trackThreadScroll(host as unknown as HTMLElement);
  assert.equal(host.listeners.get("scroll")?.capture, true, "scroll does not bubble: capture phase");
  const viewport = new FakeElement(["aui-thread-viewport"], host);
  viewport.scrollTop = 40;
  host.fire(viewport);
  assert.ok(host.attrs.has(THREAD_SCROLLED_ATTR));
  viewport.scrollTop = 0;
  host.fire(viewport);
  assert.ok(!host.attrs.has(THREAD_SCROLLED_ATTR), "back at the top: fade hidden again");
  cleanup?.();
  assert.equal(host.listeners.size, 0);
  assert.ok(!host.attrs.has("data-thread-scroll-host"));
});

test("other scrollers and the full-view dock thread never set the flag", () => {
  const host = new FakeElement();
  trackThreadScroll(host as unknown as HTMLElement);
  const codeBlock = new FakeElement(["code-scroller"], host);
  codeBlock.scrollTop = 99;
  host.fire(codeBlock);
  const dock = new FakeElement(["aui-thread-viewport"], host, true);
  dock.scrollTop = 99;
  host.fire(dock);
  assert.ok(!host.attrs.has(THREAD_SCROLLED_ATTR));
});

test("a compare pane's scroll moves only its own flag, not the backgrounded chat's", () => {
  const root = new FakeElement();
  trackThreadScroll(root as unknown as HTMLElement);
  const chat = new FakeElement(["aui-thread-viewport"], root);
  chat.scrollTop = 500;
  root.fire(chat);
  const pane = new FakeElement([], root);
  trackThreadScroll(pane as unknown as HTMLElement);
  const paneViewport = new FakeElement(["aui-thread-viewport"], pane);
  paneViewport.scrollTop = 80;
  root.fire(paneViewport);
  assert.ok(pane.attrs.has(THREAD_SCROLLED_ATTR));
  paneViewport.scrollTop = 0;
  root.fire(paneViewport);
  assert.ok(!pane.attrs.has(THREAD_SCROLLED_ATTR));
  assert.ok(root.attrs.has(THREAD_SCROLLED_ATTR), "the chat behind compare is still scrolled");
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
