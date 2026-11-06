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
  const exports: { observeMessageWindow?: (root: ElementFake, holder: ElementFake, content: ElementFake) => () => void } = {};
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
  const add = () => {
    const root = new ElementFake(), holder = new ElementFake(), content = new ElementFake(), child = new ElementFake();
    root.viewport = viewport; viewport.append(root); root.append(holder); holder.append(content); content.append(child);
    root.style.setProperty("height", "200px", "important");
    const cleanup = exports.observeMessageWindow!(root, holder, content);
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
