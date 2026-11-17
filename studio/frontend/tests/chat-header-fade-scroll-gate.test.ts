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
  constructor(classes: string[] = [], parent: FakeElement | null = null, dock = false, attrs: string[] = []) {
    this.parent = parent;
    for (const a of attrs) this.attrs.add(a);
    this.dock = dock;
    this.classList = { contains: (c) => classes.includes(c) };
  }
  closest(selector: string): FakeElement | null {
    for (let el: FakeElement | null = this; el; el = el.parent) {
      if (selector === ".chat-full-view-dock" ? el.dock : el.attrs.has(selector.slice(1, -1))) return el;
    }
    return null;
  }
  matches(selector: string) {
    return selector.split(", ").some((s) => (s.startsWith(".") ? this.classList.contains(s.slice(1)) : this.attrs.has(s.slice(1, -1))));
  }
  static host(parent: FakeElement | null = null) {
    return new FakeElement([], parent, false, [HOST_ATTR]);
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
const { HOST_ATTR, syncThreadScroll, trackThreadScroll, THREAD_SCROLLED_ATTR } = await import("../src/features/chat/thread-scroll-flag.ts");

test("the flag follows the thread viewport's scrollTop, captured on the fade's parent", () => {
  const host = FakeElement.host();
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
});

test("other scrollers never set the flag; the full-view dock's thread still does", () => {
  const host = FakeElement.host();
  trackThreadScroll(host as unknown as HTMLElement);
  const codeBlock = new FakeElement(["code-scroller"], host);
  codeBlock.scrollTop = 99;
  host.fire(codeBlock);
  assert.ok(!host.attrs.has(THREAD_SCROLLED_ATTR));
  // The dock only toggles a class over the same thread, so leaving it reruns no ref.
  const dock = new FakeElement(["aui-thread-viewport"], new FakeElement([], host, true));
  dock.scrollTop = 99;
  host.fire(dock);
  assert.ok(host.attrs.has(THREAD_SCROLLED_ATTR));
  dock.scrollTop = 0;
  host.fire(dock);
  assert.ok(!host.attrs.has(THREAD_SCROLLED_ATTR));
});

test("the project landing list drives the shared header fade like a thread", () => {
  const root = FakeElement.host();
  trackThreadScroll(root as unknown as HTMLElement);
  const landing = new FakeElement([], root, false, ["data-header-fade-scroller"]);
  landing.scrollTop = 120;
  root.fire(landing);
  assert.ok(root.attrs.has(THREAD_SCROLLED_ATTR));
  const page = readSrc("features/chat/chat-page.tsx");
  const landingFn = page.slice(page.indexOf("function ProjectLanding("));
  assert.match(landingFn.slice(0, landingFn.indexOf("\nfunction ")), /data-header-fade-scroller=""/);
});

test("a replacement scroller mounting at the top clears the flag its predecessor left", () => {
  const root = FakeElement.host();
  trackThreadScroll(root as unknown as HTMLElement);
  const old = new FakeElement(["aui-thread-viewport"], root);
  old.scrollTop = 900;
  root.fire(old);
  assert.ok(root.attrs.has(THREAD_SCROLLED_ATTR));
  syncThreadScroll(new FakeElement(["aui-thread-viewport"], root) as unknown as HTMLElement);
  assert.ok(!root.attrs.has(THREAD_SCROLLED_ATTR), "new empty chat: no fade over the flat page");
  syncThreadScroll(null);
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const ref = thread.slice(thread.indexOf("const composedViewportRef"));
  assert.match(ref.slice(0, ref.indexOf("[viewportRef, runtimeThreadId]")), /syncThreadScroll\(node\)/);
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /ref=\{syncThreadScroll\}\s+data-wallpaper-reading-surface=""\s+data-header-fade-scroller=""/);
});

test("a compare pane's scroll moves only its own flag, not the backgrounded chat's", () => {
  const root = FakeElement.host();
  trackThreadScroll(root as unknown as HTMLElement);
  const chat = new FakeElement(["aui-thread-viewport"], root);
  chat.scrollTop = 500;
  root.fire(chat);
  // Mounting order: the pane viewport's ref syncs before the pane's own ref runs.
  const pane = FakeElement.host(root);
  const paneViewport0 = new FakeElement(["aui-thread-viewport"], pane);
  syncThreadScroll(paneViewport0 as unknown as HTMLElement);
  assert.ok(root.attrs.has(THREAD_SCROLLED_ATTR), "a mounting pane leaves the chat root alone");
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
    assert.match(page.slice(parentOpen, at), /ref=\{trackThreadScroll\}\s+data-thread-scroll-host=""/, `${fade} parent`);
    const cls = page.slice(at, page.indexOf('"', at + 11));
    assert.match(cls, /bg-gradient-to-b from-background to-transparent/);
    assert.doesNotMatch(cls, /\/srgb/, "sRGB bands in WebKit");
  }
});
