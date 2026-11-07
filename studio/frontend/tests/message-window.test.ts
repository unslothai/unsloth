// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";
import ts from "@typescript/typescript6";

const source = readFileSync(new URL("../src/components/assistant-ui/message-window-controller.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } });

type EventData = { target?: ElementFake; key?: string; ctrlKey?: boolean; metaKey?: boolean; pointerType?: string; button?: number; buttons?: number };
class Events {
  listeners = new Map<string, Set<(event: EventData) => void>>();
  addEventListener(name: string, callback: (event: EventData) => void) {
    if (!this.listeners.has(name)) this.listeners.set(name, new Set());
    this.listeners.get(name)!.add(callback);
  }
  removeEventListener(name: string, callback: (event: EventData) => void) { this.listeners.get(name)?.delete(callback); }
  fire(name: string, event: EventData = {}) { for (const callback of this.listeners.get(name) ?? []) callback(event); }
  count() { return [...this.listeners.values()].reduce((sum, set) => sum + set.size, 0); }
}
class Style {
  values = new Map<string, [string, string]>();
  setProperty(name: string, value: string, priority = "") { this.values.set(name, [value, priority]); }
  getPropertyValue(name: string) { return this.values.get(name)?.[0] ?? ""; }
  getPropertyPriority(name: string) { return this.values.get(name)?.[1] ?? ""; }
  removeProperty(name: string) { this.values.delete(name); }
  set height(value: string) { this.setProperty("height", value); }
  set minHeight(value: string) { this.setProperty("min-height", value); }
  set flexShrink(value: string) { this.setProperty("flex-shrink", value); }
}
class ElementFake extends Events {
  parent: ElementFake | null = null;
  children: ElementFake[] = [];
  attributes = new Map<string, string>();
  style = new Style();
  className = "";
  textContent = "original message";
  top = -5000;
  height = 200;
  clientWidth = 600;
  scrollWidth = 600;
  protectedMedia = false;
  viewport: ElementFake | null = null;
  append(child: ElementFake) { child.remove(); this.children.push(child); child.parent = this; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((node) => node !== this); this.parent = null; }
  contains(node: ElementFake | null): boolean { return node === this || this.children.some((child) => child.contains(node)); }
  closest(selector: string) { return selector === ".aui-thread-viewport" ? this.viewport : null; }
  querySelector() { return this.protectedMedia ? this : null; }
  getBoundingClientRect() { return { top: this.top, bottom: this.top + this.height, height: this.height }; }
  setAttribute(name: string, value: string) { this.attributes.set(name, value); }
  removeAttribute(name: string) { this.attributes.delete(name); }
  hasAttribute(name: string) { return this.attributes.has(name); }
}

function world() {
  const viewport = new ElementFake(); viewport.top = 0; viewport.height = 800;
  const html = new ElementFake();
  let selected = false;
  const doc = Object.assign(new Events(), {
    documentElement: html, activeElement: null as ElementFake | null,
    createElement: () => new ElementFake(), getSelection: () => ({ isCollapsed: !selected, rangeCount: 1, getRangeAt: () => ({ intersectsNode: () => true }) }),
  });
  const media = Object.assign(new Events(), { matches: false });
  const win = Object.assign(new Events(), { matchMedia: () => media });
  let intersections: (entries: Array<{ target: ElementFake; isIntersecting: boolean }>) => void = () => {};
  let mutation: () => void = () => {};
  let observers = 0, disconnected = 0, frameId = 0;
  const frames = new Map<number, () => void>();
  const exports: { observeMessageWindow?: (root: ElementFake, holder: ElementFake, content: ElementFake, adjust?: (deltaPx: number) => void) => () => void } = {};
  vm.runInNewContext(outputText, {
    exports, window: win, document: doc,
    requestAnimationFrame: (callback: () => void) => { frames.set(++frameId, callback); return frameId; },
    cancelAnimationFrame: (id: number) => frames.delete(id),
    IntersectionObserver: class {
      constructor(callback: typeof intersections) { intersections = callback; observers++; }
      observe() {} unobserve() {} disconnect() { disconnected++; }
    },
    MutationObserver: class {
      constructor(callback: () => void) { mutation = callback; }
      observe() {} disconnect() { disconnected++; }
    },
  });
  const flush = () => { while (frames.size) { const pending = [...frames.values()]; frames.clear(); pending.forEach((callback) => callback()); } };
  const add = (adjust?: (deltaPx: number) => void) => {
    const root = new ElementFake(), holder = new ElementFake(), content = new ElementFake(), child = new ElementFake();
    root.viewport = viewport; viewport.append(root); root.append(holder); holder.append(content); content.append(child);
    root.style.setProperty("height", "200px", "important");
    const cleanup = exports.observeMessageWindow!(root, holder, content, adjust);
    return { root, holder, content, child, cleanup, parked: () => root.hasAttribute("data-message-parked") };
  };
  return {
    add, viewport, doc, win, html, media, flush,
    near: (root: ElementFake, near: boolean, top = near ? 0 : -5000) => { root.top = top; intersections([{ target: root, isIntersecting: near }]); flush(); },
    select: (value: boolean) => { selected = value; doc.fire("selectionchange"); flush(); },
    find: (value: boolean) => { if (value) html.setAttribute("data-find-bar-open", ""); else html.removeAttribute("data-find-bar-open"); mutation(); flush(); },
    observers: () => observers, disconnected: () => disconnected,
  };
}

test("distant bodies keep their height and reconnect the same nodes and inline styles", () => {
  const w = world(), row = w.add();
  w.near(row.root, false);
  assert.equal(row.parked(), true);
  assert.equal(row.content.parent, null);
  assert.equal(row.root.style.getPropertyValue("height"), "200px");
  row.root.style.setProperty("min-height", "50px");
  assert.equal(row.holder.children[0].textContent, "original message");
  w.near(row.root, true);
  assert.equal(row.parked(), false);
  assert.equal(row.content.parent, row.holder);
  assert.equal(row.content.children[0], row.child);
  assert.equal(row.root.style.getPropertyPriority("height"), "important");
  assert.equal(row.root.style.getPropertyValue("min-height"), "50px", "do not overwrite the viewport slack helper");
  row.cleanup();
});

test("find, print and selection reconnect every body until the operation ends", () => {
  const w = world(), a = w.add(), b = w.add();
  w.near(a.root, false); w.near(b.root, false);
  for (const hold of [w.find, w.select]) {
    hold(true); assert.equal(a.parked() || b.parked(), false);
    hold(false); assert.equal(a.parked() && b.parked(), true);
  }
  w.win.fire("beforeprint"); assert.equal(a.parked() || b.parked(), false);
  w.win.fire("afterprint"); w.flush(); assert.equal(a.parked() && b.parked(), true);
  w.media.matches = true; w.media.fire("change"); assert.equal(a.parked() || b.parked(), false);
  a.cleanup(); b.cleanup();
});

test("focus and select-all reconnect before browser default actions; dragging does not detach", () => {
  const w = world(), row = w.add();
  w.viewport.fire("pointerdown", { target: row.child, pointerType: "mouse", button: 0 }); w.near(row.root, false); assert.equal(row.parked(), false);
  w.win.fire("pointerup"); w.flush(); assert.equal(row.parked(), true);
  w.viewport.fire("focusin", { target: row.root }); assert.equal(row.parked(), false);
  w.near(row.root, false); w.doc.fire("keydown", { key: "a", ctrlKey: true }); assert.equal(row.parked(), false);
  row.cleanup();
});

test("wide replies and embedded media stay connected; shared observers and listeners clean up", () => {
  const w = world(), a = w.add(), b = w.add();
  a.root.scrollWidth = 1200; b.content.protectedMedia = true;
  w.near(a.root, false); w.near(b.root, false);
  assert.equal(a.parked() || b.parked(), false);
  assert.equal(w.observers(), 1);
  assert.equal(w.viewport.hasAttribute("data-windowed-messages"), true);
  a.cleanup(); assert.equal(w.disconnected(), 0);
  b.cleanup(); assert.equal(w.disconnected(), 2);
  assert.equal(w.viewport.hasAttribute("data-windowed-messages"), false);
  assert.equal(w.doc.count() + w.win.count() + w.viewport.count() + w.media.count(), 0);
});

test("thumb dragging releases visited bodies and holds restored heights until release", () => {
  const w = world(), row = w.add();
  row.root.style.removeProperty("height");
  w.near(row.root, false);
  w.viewport.fire("pointerdown", { target: w.viewport, pointerType: "mouse", button: 0 });
  w.near(row.root, true);
  assert.equal(row.parked(), false);
  assert.equal(row.content.children[0], row.child);
  assert.equal(row.root.style.getPropertyValue("height"), "200px");
  w.near(row.root, false);
  assert.equal(row.parked(), true, "a held thumb must not accumulate visited bodies");
  w.near(row.root, true);
  w.win.fire("pointerup"); w.flush();
  assert.equal(row.root.style.getPropertyValue("height"), "", "release the original natural height");
  row.cleanup();
});

test("print and drag cancellation release frozen message heights", () => {
  for (const event of ["beforeprint", "pointercancel", "blur"]) {
    const w = world(), row = w.add();
    row.root.style.removeProperty("height");
    w.near(row.root, false);
    w.viewport.fire("pointerdown", { target: w.viewport, pointerType: "mouse", button: 0 });
    w.near(row.root, true);
    w.win.fire(event); w.flush();
    assert.equal(row.root.style.getPropertyValue("height"), "", event);
    row.cleanup();
  }
});

test("a thumb drag restores the viewport before the normal lookahead", () => {
  const w = world(), row = w.add();
  w.near(row.root, false);
  w.viewport.fire("pointerdown", { target: w.viewport, pointerType: "mouse", button: 0 });
  w.near(row.root, true, 1800);
  assert.equal(row.parked(), true, "do not render a message beyond the drag buffer");
  row.root.top = 700;
  w.viewport.fire("scroll"); w.flush();
  assert.equal(row.parked(), false, "scrolling into view must restore without another IO boundary");
  row.cleanup();
});

test("a missed mouseup releases heights on the next pointer move", () => {
  const w = world(), row = w.add();
  row.root.style.removeProperty("height");
  w.near(row.root, true);
  w.viewport.fire("pointerdown", { target: w.viewport, pointerType: "mouse", button: 0 });
  assert.equal(row.root.style.getPropertyValue("height"), "200px");
  w.win.fire("pointermove", { buttons: 0 }); w.flush();
  assert.equal(row.root.style.getPropertyValue("height"), "");
  row.cleanup();
});

test("wheel, keyboard and thumb jumps restore visible bodies before the next frame or IO delivery", () => {
  for (const thumb of [false, true]) {
    const w = world(), row = w.add();
    w.near(row.root, false);
    if (thumb) w.viewport.fire("pointerdown", { target: w.viewport, pointerType: "mouse", button: 0 });
    row.root.top = 100;
    w.viewport.fire("scroll");
    assert.equal(row.parked(), false, "restore synchronously even though the last IO record said offscreen");
    assert.equal(row.content.parent, row.holder);
    row.root.top = -5000;
    w.viewport.fire("scroll");
    assert.equal(row.parked(), true);
    row.cleanup();
  }
});

test("stale intersection records cannot detach content that moved into view after a prepend", () => {
  const w = world(), row = w.add();
  w.near(row.root, false, 100);
  assert.equal(row.parked(), false, "current geometry overrides the old intersection record");
  w.near(row.root, true, -5000);
  assert.equal(row.parked(), true);
  // Find/selection settling defers measurements for parking, but must not delay visibility.
  w.doc.fire("selectionchange");
  row.root.top = 100;
  w.viewport.fire("scroll");
  assert.equal(row.parked(), false);
  w.flush();
  row.cleanup();
});

test("small thumb reversals retain already restored overscan content", () => {
  const w = world(), row = w.add();
  w.near(row.root, false);
  w.viewport.fire("pointerdown", { target: w.viewport, pointerType: "mouse", button: 0 });
  row.root.top = 100;
  w.viewport.fire("scroll");
  assert.equal(row.parked(), false);
  row.root.top = 2000;
  w.viewport.fire("scroll");
  assert.equal(row.parked(), false, "keep connected content beyond the restore band");
  row.root.top = 4000;
  w.viewport.fire("scroll");
  assert.equal(row.parked(), true);
  row.cleanup();
});

test("restoring a reflowed body above the reader keeps the first visible row in place", () => {
  const w = world(), above = w.add(), visible = w.add();
  const scrolls: Array<{ top: number; behavior: string }> = [];
  Object.assign(w.viewport, {
    scrollTop: 1000,
    scrollTo(options: { top: number; behavior: string }) { scrolls.push(options); (this as unknown as { scrollTop: number }).scrollTop = options.top; },
  });
  w.near(above.root, false);
  assert.equal(above.parked(), true);
  // The chat narrowed while `above` was parked: reconnected, its body is 60px taller than the
  // placeholder height it kept, so everything below it moves down by that much.
  visible.root.top = 50;
  Object.defineProperty(visible.root, "isConnected", { value: true });
  visible.root.getBoundingClientRect = () => {
    const top = visible.root.top + (above.content.parent ? 60 : 0);
    return { top, bottom: top + visible.root.height, height: visible.root.height };
  };
  w.viewport.fire("scroll");
  assert.equal((w.viewport as unknown as { scrollTop: number }).scrollTop, 1000, "nothing reconnected, nothing corrected");
  above.root.top = -1500;
  w.viewport.fire("scroll");
  assert.equal(above.parked(), false);
  assert.equal((w.viewport as unknown as { scrollTop: number }).scrollTop, 1060, "the reflow above the reader is scrolled away");
  assert.equal(JSON.stringify(scrolls), JSON.stringify([{ top: 1060, behavior: "instant" }]), "instant despite the viewport's scroll-smooth class");
  above.cleanup();
  visible.cleanup();
});

test("with an autoscroll owner the correction goes through it instead of writing scrollTop", () => {
  const w = world(), deltas: number[] = [];
  const above = w.add((delta) => deltas.push(delta)), visible = w.add();
  Object.assign(w.viewport, { scrollTop: 1000, scrollTo() { throw new Error("must not write scrollTop directly"); } });
  w.near(above.root, false);
  visible.root.top = 50;
  Object.defineProperty(visible.root, "isConnected", { value: true });
  visible.root.getBoundingClientRect = () => {
    const top = visible.root.top + (above.content.parent ? 60 : 0);
    return { top, bottom: top + visible.root.height, height: visible.root.height };
  };
  w.viewport.fire("scroll");
  above.root.top = -1500;
  w.viewport.fire("scroll");
  assert.equal(JSON.stringify(deltas), "[60]", "the owner resyncs its follow intent and scrolls instantly");
  above.cleanup();
  visible.cleanup();
});

test("a body restored during a thumb drag is clipped until release, then keeps its own overflow", () => {
  const w = world(), row = w.add();
  row.root.style.setProperty("overflow", "visible");
  w.near(row.root, false);
  assert.equal(row.parked(), true);
  w.viewport.fire("pointerdown", { target: w.viewport, pointerType: "mouse", button: 0 });
  row.root.top = 100;
  w.viewport.fire("scroll");
  assert.equal(row.parked(), false);
  assert.equal(row.root.style.getPropertyValue("height"), "200px", "height frozen while the thumb is held");
  assert.equal(row.root.style.getPropertyValue("overflow"), "clip", "a reflowed body cannot paint over the next messages");
  w.win.fire("pointerup");
  assert.equal(row.root.style.getPropertyValue("overflow"), "visible", "release restores the original overflow");
  assert.equal(row.root.style.getPropertyPriority("height"), "important", "and the original height");
  row.cleanup();
});

test("a body parked and restored again in one thumb drag still releases its own overflow", () => {
  const w = world(), row = w.add();
  row.root.style.setProperty("overflow", "visible");
  w.near(row.root, false);
  w.viewport.fire("pointerdown", { target: w.viewport, pointerType: "mouse", button: 0 });
  for (const top of [100, 4000, 100]) {
    row.root.top = top;
    w.viewport.fire("scroll");
  }
  assert.equal(row.parked(), false);
  assert.equal(row.root.style.getPropertyValue("overflow"), "clip");
  w.win.fire("pointerup");
  assert.equal(row.root.style.getPropertyValue("overflow"), "visible", "the temporary clip is never saved as the original");
  row.cleanup();
  assert.equal(row.root.style.getPropertyValue("overflow"), "visible");
});
