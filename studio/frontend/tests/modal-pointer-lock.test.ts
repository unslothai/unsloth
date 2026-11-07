// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Radix writes pointer-events: none on <body> for each modal layer. Rules keyed on that inline
// style (body[style*=...] X) made WebKit walk every element under <body>, the whole thread
// included, on each open and close. These pin that the modal rules key on attributes that never
// sit above a thread root, and that the watcher marks the page so the rules keep their meaning.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

type Init = { childList?: boolean; attributes?: boolean; subtree?: boolean; attributeFilter?: string[] };

class FakeElement {
  attrs = new Map<string, string>();
  children: FakeElement[] = [];
  parentElement: FakeElement | null = null;
  style = { pointerEvents: "" };
  name: string;
  constructor(name: string, className = "", role?: string) {
    this.name = name;
    if (className) this.attrs.set("class", className);
    if (role) this.attrs.set("role", role);
  }
  append(...kids: FakeElement[]): this {
    for (const kid of kids) {
      kid.parentElement = this;
      this.children.push(kid);
    }
    return this;
  }
  getAttribute(name: string): string | null {
    return this.attrs.get(name) ?? null;
  }
  setAttribute(name: string, value: string): void {
    this.attrs.set(name, value);
  }
  removeAttribute(name: string): void {
    this.attrs.delete(name);
  }
  hasAttribute(name: string): boolean {
    return this.attrs.has(name);
  }
  descendants(): FakeElement[] {
    return this.children.flatMap((kid) => [kid, ...kid.descendants()]);
  }
  contains(other: FakeElement): boolean {
    return other === this || this.descendants().includes(other);
  }
  getElementsByClassName(name: string): FakeElement[] {
    return this.descendants().filter((el) => (el.getAttribute("class") ?? "").split(" ").includes(name));
  }
  querySelectorAll(selector: string): FakeElement[] {
    assert.match(selector, /\[role="dialog"\]/);
    return this.descendants().filter((el) =>
      ["dialog", "alertdialog", "menu", "listbox"].includes(el.getAttribute("role") ?? ""),
    );
  }
}

const observers: { cb: (records: unknown[]) => void; targets: [FakeElement, Init][] }[] = [];
class FakeMutationObserver {
  entry: (typeof observers)[number];
  constructor(cb: (records: unknown[]) => void) {
    this.entry = { cb, targets: [] };
    observers.push(this.entry);
  }
  observe(target: FakeElement, init: Init) {
    this.entry.targets.push([target, init]);
  }
  disconnect() {
    this.entry.targets = [];
  }
}

function deliverStyle(target: FakeElement): void {
  for (const o of observers) {
    const hit = o.targets.some(
      ([t, init]) =>
        init.attributeFilter?.includes("style") && (t === target || (init.subtree && t.contains(target))),
    );
    if (hit) o.cb([{ type: "attributes", attributeName: "style", target }]);
  }
}

function deliverChildList(target: FakeElement): void {
  for (const o of observers) {
    if (o.targets.some(([t, init]) => t === target && init.childList)) {
      o.cb([{ type: "childList", target }]);
    }
  }
}

// body > [root > [sidebar > [row > action], shell > [header > toolbar, main > [threadRoot > viewport > msgs]]], portal]
const body = new FakeElement("body");
const root = new FakeElement("root");
const sidebar = new FakeElement("sidebar");
const rowAction = new FakeElement("rowAction", "sidebar-row-action");
const shell = new FakeElement("shell", "pointer-events-auto");
const header = new FakeElement("header");
const toolbar = new FakeElement("toolbar", "pointer-events-auto");
const main = new FakeElement("main");
const threadRoot = new FakeElement("threadRoot", "aui-root aui-thread-root relative");
const viewport = new FakeElement("viewport", "aui-thread-viewport");
const messages = Array.from({ length: 50 }, (_, i) => new FakeElement(`msg${i}`, "pointer-events-auto"));
const menuPortal = new FakeElement("menuPortal");
const nonModalMenu = new FakeElement("nonModalMenu", "", "menu");
body.append(
  root.append(
    sidebar.append(new FakeElement("row").append(rowAction)),
    shell.append(header.append(toolbar), main.append(threadRoot.append(viewport.append(...messages)))),
  ),
  menuPortal.append(nonModalMenu),
);

Object.assign(globalThis, { MutationObserver: FakeMutationObserver });
const doc = { body } as unknown as Document;
const lock = await import("../src/lib/modal-pointer-lock.ts");
const ATTRS = [
  lock.MODAL_LOCK_THREAD_ATTR,
  lock.MODAL_LOCK_PATH_ATTR,
  lock.MODAL_LOCK_REGION_ATTR,
  lock.MODAL_LOCK_LOWER_ATTR,
];

function marked(): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const el of body.descendants()) {
    for (const attr of ATTRS) if (el.hasAttribute(attr)) (out[attr] ??= []).push(el.name);
  }
  return out;
}

/** Inside one region, or a path element, or a thread root (whose subtree no rule reaches). */
function coveredOnce(el: FakeElement): boolean {
  let regions = 0;
  for (let node: FakeElement | null = el; node && node !== body; node = node.parentElement) {
    if (node.hasAttribute(lock.MODAL_LOCK_REGION_ATTR)) regions += 1;
    if (node.hasAttribute(lock.MODAL_LOCK_THREAD_ATTR)) return regions === 0;
  }
  return regions === 1 || (regions === 0 && el.hasAttribute(lock.MODAL_LOCK_PATH_ATTR));
}

const stop = lock.watchModalPointerLock(doc);

test("no modal rule keys on an inline style, and lock attributes above a thread are subject-only", () => {
  const css = readSrc("index.css");
  // Any [style*=...] selector makes every inline style flip on an ancestor walk its subtree in WebKit.
  assert.doesNotMatch(css, /\[style[*^$~|]?=/);
  for (const attr of ["data-modal-lock-thread", "data-modal-lock-path"]) {
    let uses = 0;
    for (let at = css.indexOf(`[${attr}]`); at >= 0; at = css.indexOf(`[${attr}]`, at + 1)) {
      uses += 1;
      // Rest of this compound: a combinator at paren depth 0 would make it an ancestor.
      let depth = 0;
      for (let i = at + attr.length + 2; i < css.length; i++) {
        const ch = css[i];
        if (ch === "(") depth += 1;
        else if (ch === ")") depth -= 1;
        if (depth < 0) depth = 0; // closed the :is() this attribute sits in
        if (depth > 0) continue;
        if (ch === "," || ch === "{") break;
        assert.ok(!/[\s>+~]/.test(ch) || /^\s*[,{]/.test(css.slice(i)), `${attr} used as an ancestor near ${css.slice(at, at + 80)}`);
      }
    }
    assert.ok(uses > 0, attr);
  }
  assert.match(readSrc("main.tsx"), /watchModalPointerLock\(document\);/);
});

test("nothing is marked until the body lock goes up", () => {
  assert.deepEqual(marked(), {});
});

test("the lock splits the page into regions around the thread and marks no message", () => {
  body.style.pointerEvents = "none";
  deliverStyle(body);
  assert.deepEqual(marked()[lock.MODAL_LOCK_THREAD_ATTR], ["threadRoot"]);
  assert.deepEqual(marked()[lock.MODAL_LOCK_PATH_ATTR]?.sort(), ["main", "root", "shell"]);
  assert.deepEqual(marked()[lock.MODAL_LOCK_REGION_ATTR]?.sort(), ["header", "menuPortal", "sidebar"]);
  for (const el of body.descendants()) {
    if (threadRoot.contains(el) && el !== threadRoot) {
      assert.equal(ATTRS.some((attr) => el.hasAttribute(attr)), false, `${el.name} inside the thread was marked`);
    } else {
      assert.ok(coveredOnce(el), `${el.name} is not covered by exactly one lock rule scope`);
    }
  }
});

test("a child mounted next to the thread while locked joins a region", () => {
  const banner = new FakeElement("banner", "pointer-events-auto");
  main.append(banner);
  deliverChildList(main);
  assert.ok(banner.hasAttribute(lock.MODAL_LOCK_REGION_ATTR));
  // A child inside a region needs nothing: the region's descendant rule reaches it.
  const late = new FakeElement("late", "pointer-events-auto");
  header.append(late);
  assert.ok(coveredOnce(late));
});

test("a stacked modal layer marks the layer under it as lower, and lifts it again", () => {
  const dialog = new FakeElement("dialog", "", "dialog");
  const dialogPortal = new FakeElement("dialogPortal").append(dialog);
  body.append(dialogPortal);
  deliverChildList(body);
  assert.ok(dialogPortal.hasAttribute(lock.MODAL_LOCK_REGION_ATTR));
  dialog.style.pointerEvents = "none";
  deliverStyle(dialog);
  assert.ok(dialog.hasAttribute(lock.MODAL_LOCK_LOWER_ATTR));
  dialog.style.pointerEvents = "auto";
  deliverStyle(dialog);
  assert.equal(dialog.hasAttribute(lock.MODAL_LOCK_LOWER_ATTR), false);
  // Inline pointer-events on something that is not a layer is not a lower layer.
  toolbar.style.pointerEvents = "none";
  deliverStyle(toolbar);
  assert.equal(toolbar.hasAttribute(lock.MODAL_LOCK_LOWER_ATTR), false);
  toolbar.style.pointerEvents = "";
});

test("unlocking clears every mark and stops watching the page", () => {
  body.style.pointerEvents = "";
  deliverStyle(body);
  assert.deepEqual(marked(), {});
  const active = observers.filter((o) => o.targets.length > 0);
  assert.equal(active.length, 1, "only the body style observer stays connected");
  assert.deepEqual(active[0].targets.map(([t]) => t.name), ["body"]);
});

test("a layer already lower when the lock goes up is marked at once", () => {
  nonModalMenu.style.pointerEvents = "none";
  body.style.pointerEvents = "none";
  deliverStyle(body);
  assert.ok(nonModalMenu.hasAttribute(lock.MODAL_LOCK_LOWER_ATTR));
  body.style.pointerEvents = "";
  deliverStyle(body);
  assert.deepEqual(marked(), {});
  nonModalMenu.style.pointerEvents = "";
});

test("a second watcher on the same document is a no-op", () => {
  const before = observers.length;
  const again = lock.watchModalPointerLock(doc);
  assert.equal(observers.length, before);
  again();
  stop();
  assert.equal(observers.filter((o) => o.targets.length > 0).length, 0);
});
