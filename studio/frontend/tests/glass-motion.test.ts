// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { mock, test } from "node:test";

import postcss from "postcss";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");
const MATERIAL = readSrc("features/chat-background/material.ts");

// --- a DOM just big enough for glass-motion.ts -------------------------------------------------------

type Listener = (event: { target: unknown }) => void;

// Tags stand for the selectors glass-motion.ts asks about: "surface" (GLASS_MOTION_SURFACES), "tooltip",
// "sidebar", "frame", "thread" (the chat viewport), "field", "huge" (must never be searched), "class:NAME".
class FakeElement {
  attrs = new Map<string, string>();
  children: FakeElement[] = [];
  parentElement: FakeElement | null = null;
  writes = 0;
  tags: string[];
  constructor(tags: string[] = []) {
    this.tags = tags;
  }
  append(...kids: FakeElement[]): this {
    for (const kid of kids) {
      kid.parentElement = this;
      this.children.push(kid);
    }
    return this;
  }
  is(tag: string): boolean {
    return this.tags.includes(tag);
  }
  get isConnected(): boolean {
    let el: FakeElement = this;
    while (el.parentElement) el = el.parentElement;
    return el.is("root");
  }
  matches(selector: string): boolean {
    if (selector === SURFACES) return this.is("surface");
    if (selector.includes("tooltip-compact")) return this.is("tooltip");
    if (selector.includes("sidebar-inner")) return this.is("sidebar");
    if (selector.includes("windows-glass-frame")) return this.is("frame");
    if (selector.includes("aui-thread-viewport")) return this.is("thread");
    if (selector.includes("textarea")) return this.is("field");
    throw new Error(`unexpected selector ${selector}`);
  }
  closest(selector: string): FakeElement | null {
    for (let el: FakeElement | null = this; el; el = el.parentElement) if (el.matches(selector)) return el;
    return null;
  }
  all(pred: (el: FakeElement) => boolean): FakeElement[] {
    if (this.is("huge")) throw new Error("searched the chat subtree");
    const out: FakeElement[] = [];
    for (const kid of this.children) {
      if (pred(kid)) out.push(kid);
      if (!kid.is("huge")) out.push(...kid.all(pred));
    }
    return out;
  }
  querySelectorAll(selector: string): FakeElement[] {
    return this.all((el) => el.matches(selector));
  }
  hasAttribute(name: string): boolean {
    return this.attrs.has(name);
  }
  getAttribute(name: string): string | null {
    return this.attrs.get(name) ?? null;
  }
  setAttribute(name: string, value: string): void {
    this.writes++;
    this.attrs.set(name, value);
  }
  removeAttribute(name: string): void {
    this.writes++;
    this.attrs.delete(name);
  }
}

class FakeDocument {
  listeners = new Map<string, Set<Listener>>();
  documentElement = new FakeElement(["root"]);
  body = new FakeElement();
  scrollingElement = new FakeElement();
  constructor() {
    this.documentElement.append(this.body);
  }
  querySelector(selector: string): FakeElement | null {
    return this.body.querySelectorAll(selector)[0] ?? null;
  }
  getElementsByClassName(name: string): FakeElement[] {
    // A live collection in a browser; the fake walks past "huge" nodes like the engine's indexed lookup would.
    return this.body.all((el) => el.is(`class:${name}`));
  }
  addEventListener(type: string, fn: Listener): void {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(fn);
  }
  removeEventListener(type: string, fn: Listener): void {
    this.listeners.get(type)?.delete(fn);
  }
  fire(type: string, target: unknown): void {
    for (const fn of this.listeners.get(type) ?? []) fn({ target });
  }
  count(): number {
    return [...this.listeners.values()].reduce((n, s) => n + s.size, 0);
  }
}

let observed: { cb: (records: unknown[]) => void; target: unknown; opts: unknown; live: boolean }[] = [];
class FakeMutationObserver {
  entry: (typeof observed)[number];
  constructor(cb: (records: unknown[]) => void) {
    this.entry = { cb, target: null, opts: null, live: false };
    observed.push(this.entry);
  }
  observe(target: unknown, opts: unknown): void {
    Object.assign(this.entry, { target, opts, live: true });
  }
  disconnect(): void {
    this.entry.live = false;
  }
}

Object.assign(globalThis, { Element: FakeElement, MutationObserver: FakeMutationObserver });
mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
let clock = 0;
Object.defineProperty(globalThis, "performance", { value: { now: () => clock }, configurable: true });
const tick = (ms: number) => {
  clock += ms;
  mock.timers.tick(ms);
};

const motion = await import("../src/features/chat-background/glass-motion.ts");
const SURFACES = motion.GLASS_MOTION_SURFACES;
const ATTR = motion.GLASS_MOVING_ATTR;
const SETTLING = motion.GLASS_SETTLING_ATTR;

/** A chat page: sidebar (in a Windows glass frame), thread viewport (huge) beside the composer dock. */
function page(glass = "liquid-3d") {
  observed = [];
  const doc = new FakeDocument();
  if (glass) doc.documentElement.setAttribute("data-glass", glass);
  const sidebar = new FakeElement(["sidebar"]);
  const frame = new FakeElement(["frame"]).append(sidebar);
  const composer = new FakeElement(["surface", "class:unsloth-composer-surface"]);
  const float = new FakeElement(["surface"]);
  const dock = new FakeElement().append(composer, float);
  const viewport = new FakeElement(["thread", "huge"]);
  doc.body.append(frame, new FakeElement().append(viewport, dock));
  const ctl = motion.startGlassMotion(doc as unknown as Document);
  return { doc, ctl, sidebar, frame, composer, float, viewport };
}
const moving = (...els: FakeElement[]) => els.map((el) => el.hasAttribute(ATTR));

// --- behaviour ------------------------------------------------------------------------------------

test("scrolling the chat pauses the glass over it and the standing glass, then the blur returns in one step", () => {
  const { doc, ctl, sidebar, frame, composer, float, viewport } = page();
  doc.fire("scroll", viewport);
  assert.deepEqual(moving(composer, float, sidebar, frame), [true, true, true, true]);
  assert.ok(!viewport.hasAttribute(ATTR), "the chat itself is never flagged");
  tick(motion.SETTLE_MS - 1);
  doc.fire("scroll", viewport);
  tick(motion.SETTLE_MS - 1);
  assert.ok(sidebar.hasAttribute(ATTR), "each scroll event extends the hold");
  tick(1);
  assert.deepEqual(moving(composer, float, sidebar, frame), [false, false, false, false]);
  assert.ok(composer.hasAttribute(SETTLING), "the blur returns with the surface's transitions off");
  tick(motion.SETTLING_MS);
  assert.ok(!composer.hasAttribute(SETTLING), "then its transitions are back");
  assert.equal(sidebar.writes, 4, "set, settle + clear, unsettle: however many scroll events");
  ctl.stop();
});

test("a scroll inside a dialog pauses it with the standing glass; typing in a field pauses nothing", () => {
  const { doc, ctl, sidebar, composer } = page();
  const pane = new FakeElement();
  const dialog = new FakeElement(["surface"]).append(pane);
  const field = new FakeElement(["field"]);
  doc.body.append(dialog, new FakeElement(["surface"]).append(field));
  doc.fire("scroll", field);
  assert.deepEqual(moving(sidebar, composer), [false, false], "text scrolling inside a field is typing");
  doc.fire("scroll", pane);
  assert.deepEqual(moving(dialog, sidebar, composer), [true, true, true]);
  doc.fire("scroll", doc);
  ctl.stop();
});

test("keys, presses and edits pause the standing glass and the surface they land in through the burst", () => {
  const { doc, ctl, sidebar, frame, composer, float } = page();
  const field = new FakeElement(["field"]);
  composer.append(field);
  for (const type of motion.INTERACTION_EVENTS) {
    doc.fire(type, field);
    assert.deepEqual(moving(sidebar, frame, composer), [true, true, true], type);
    assert.ok(!float.hasAttribute(ATTR), "glass the event did not land in and that does not stand stays blurred");
    tick(motion.SETTLE_MS - 1);
  }
  assert.ok(sidebar.hasAttribute(ATTR), "each key extends the hold, so a burst of typing stays on the fill");
  tick(1);
  assert.deepEqual(moving(sidebar, frame, composer), [false, false, false], "the blur returns once the keys stop");
  const button = new FakeElement();
  const dialog = new FakeElement(["surface"]).append(button);
  doc.body.append(dialog);
  doc.fire("pointerdown", button);
  assert.deepEqual(moving(dialog, sidebar, composer), [true, true, true], "a press in a dialog pauses the dialog too");
  doc.fire("keydown", doc);
  ctl.stop();
});

test("a menu mounting is paused from its first frame with the standing glass; a tooltip only pauses itself", () => {
  const { ctl, sidebar, composer } = page();
  const [mo] = observed;
  assert.deepEqual(mo.opts, { childList: true }, "body's own children only, never the chat subtree");
  const tip = new FakeElement(["surface", "tooltip"]);
  mo.cb([{ addedNodes: [new FakeElement().append(tip), "text node"] }]);
  assert.ok(tip.hasAttribute(ATTR));
  assert.deepEqual(moving(sidebar, composer), [false, false], "hovering must not flicker the sidebar");
  const menu = new FakeElement(["surface"]);
  mo.cb([{ addedNodes: [new FakeElement().append(menu)] }]);
  assert.deepEqual(moving(menu, sidebar, composer), [true, true, true]);
  tick(motion.MOUNT_HOLD_MS);
  assert.deepEqual(moving(menu, sidebar, composer), [false, false, false]);
  ctl.stop();
});

test("an animation holds its surface until it ends; the standing glass is never released early", () => {
  const { doc, ctl, sidebar } = page();
  const dialog = new FakeElement(["surface"]);
  doc.fire("animationstart", dialog);
  tick(motion.MOUNT_HOLD_MS + 10);
  assert.ok(dialog.hasAttribute(ATTR), "a running animation holds it past the mount hold");
  doc.fire("animationend", dialog);
  tick(motion.SETTLE_MS);
  assert.ok(!dialog.hasAttribute(ATTR));
  const long = new FakeElement(["surface"]);
  doc.fire("animationstart", long);
  doc.fire("scroll", new FakeElement());
  tick(motion.SETTLE_MS + 1);
  assert.ok(sidebar.hasAttribute(ATTR), "a short scroll does not cut the animation's hold on the sidebar");
  tick(motion.ANIMATION_HOLD_MS);
  assert.ok(!long.hasAttribute(ATTR), "an animation whose end never arrives is released by ANIMATION_HOLD_MS");
  const spinner = new FakeElement();
  doc.fire("animationstart", spinner);
  assert.ok(!spinner.hasAttribute(ATTR) && !sidebar.hasAttribute(ATTR), "only glass surfaces' animations count");
  ctl.stop();
});

test("with the native macOS material the sidebar keeps its look", () => {
  const { doc, ctl, sidebar, composer, viewport } = page("macos");
  doc.fire("scroll", viewport);
  assert.deepEqual(moving(sidebar, composer), [false, true]);
  ctl.stop();
});

test("the standing glass is found again once it leaves the page", () => {
  const { doc, ctl, composer, viewport } = page();
  const dock = composer.parentElement!;
  dock.children = dock.children.filter((el) => el !== composer);
  composer.parentElement = null;
  const fresh = new FakeElement(["surface", "class:unsloth-composer-surface"]);
  dock.append(fresh);
  doc.fire("scroll", new FakeElement());
  tick(motion.SETTLE_MS + motion.SETTLING_MS);
  doc.fire("scroll", viewport);
  assert.ok(fresh.hasAttribute(ATTR));
  ctl.stop();
});

test("stop() removes every listener, the observer and every flag, and is safe to call twice", () => {
  const { doc, ctl, sidebar } = page();
  assert.equal(doc.count(), 4 + motion.INTERACTION_EVENTS.length);
  const a = new FakeElement(["surface"]);
  const c = new FakeElement(["surface"]);
  ctl.hold(a as unknown as Element, 10_000);
  ctl.hold(c as unknown as Element, 10);
  doc.fire("scroll", new FakeElement());
  tick(10);
  assert.ok(c.hasAttribute(SETTLING));
  ctl.stop();
  assert.equal(doc.count(), 0);
  assert.equal(observed[0].live, false);
  assert.ok(!a.hasAttribute(ATTR) && !sidebar.hasAttribute(ATTR));
  assert.ok(!c.hasAttribute(SETTLING), "a settling surface gets its transitions back");
  assert.ok(!a.hasAttribute(SETTLING), "stop() restores the blur without a settle step");
  ctl.stop();
  tick(10_000);
  assert.equal(a.writes, 2, "the cancelled timer never fires a late write");
});

test("a script-driven animation holds its surface through the running controller, and nothing without one", () => {
  const panel = new FakeElement(["surface"]);
  motion.holdGlassMotion(panel as unknown as Element, 1000);
  assert.ok(!panel.hasAttribute(ATTR), "no controller: glass motion is off, nothing changes");
  const { ctl } = page();
  motion.holdGlassMotion(panel as unknown as Element, motion.ANIMATION_HOLD_MS);
  assert.ok(panel.hasAttribute(ATTR));
  motion.holdGlassMotion(panel as unknown as Element, motion.SETTLE_MS);
  tick(motion.SETTLE_MS);
  assert.ok(!panel.hasAttribute(ATTR), "the end of the animation releases it SETTLE_MS later");
  ctl.stop();
  const other = new FakeElement(["surface"]);
  motion.holdGlassMotion(other as unknown as Element, 1000);
  assert.ok(!other.hasAttribute(ATTR), "a stopped controller is no longer reachable");
  motion.holdGlassMotion(null, 1000);
});

// --- wiring and CSS contract -----------------------------------------------------------------------

test("glass motion runs only with software compositing, where glass blurs: never solid or out of scope, and Transparent only on Windows Desktop", () => {
  assert.match(MATERIAL, /import \{ startGlassMotion \} from "\.\/glass-motion";/);
  assert.match(
    MATERIAL,
    /const clearBlur =\s*material === "transparent" &&\s*root\.classList\.contains\("tauri"\) &&\s*root\.classList\.contains\("client-windows"\);\s*const on =\s*material !== "solid" &&\s*\(material !== "transparent" \|\| clearBlur\) &&\s*here &&\s*softwareCompositing\(\);\s*document\.documentElement\.toggleAttribute\("data-glass-motion", on\);\s*return on \? startGlassMotion\(\)\.stop : undefined;\s*\}, \[material, here\]\);/,
  );
});

const root = postcss.parse(CSS);
const movingRules: postcss.Rule[] = [];
root.walkRules((rule) => {
  if (/data-glass-(moving|settling)|\[data-slot\$="-content"\]\[data-state="closed"\]/.test(rule.selector)) movingRules.push(rule);
});

test("the motion flag is keyed on the surface, never on html or body, and never inside :has()", () => {
  assert.ok(movingRules.length >= 5);
  for (const rule of movingRules) {
    for (const sel of rule.selectors) {
      assert.doesNotMatch(sel, /html\[[^\]]*data-glass-moving|body\[[^\]]*data-glass-moving/, sel);
      assert.doesNotMatch(sel, /:has\(/, `${sel}: a :has() here would re-match on every DOM change`);
      assert.match(sel, /^html(\.dark)?(\[data-glass|:is\(\[data-glass\],\[data-glass-clear\]\)|\.tauri\.client-windows\[data-glass-clear\])/, `${sel}: only with Glass on (or Windows Transparent, which blurs)`);
      assert.match(sel, /^html[^ ]*\[data-glass-motion\]/, `${sel}: only with software compositing; with a GPU nothing changes`);
    }
    assert.equal(rule.parent?.type, "root", `${rule.selector}: unlayered, outside @layer and @media`);
  }
  for (const rule of movingRules) {
    for (const sel of rule.selectors) {
      // Chromium indexes invalidation by attribute name: a descendant rule on data-state would search the
      // subtree of every element whose data-state changes (the sidebar, collapsibles in the chat).
      assert.doesNotMatch(sel, /\[data-state[^\]]*\][^,]*\s+[^\s]/, `${sel}: no descendant keyed on data-state`);
      assert.ok(!/\[data-state/.test(sel) || /\[data-slot\$?="[^"]*-content"\]/.test(sel), `${sel}: only Radix content closing`);
    }
  }
  assert.doesNotMatch(readSrc("features/chat-background/glass-motion.ts"), /(documentElement|body)\??\.(set|toggle|remove)Attribute/);
});

test("a moving or closing surface loses its blur at once and takes a near-opaque fill", () => {
  const pause = movingRules.find((r) => r.selectors.includes("html:is([data-glass],[data-glass-clear])[data-glass-motion] [data-glass-moving]"));
  assert.ok(pause);
  for (const sel of [
    "html:is([data-glass],[data-glass-clear])[data-glass-motion] [data-glass-moving]::before",
    "html:is([data-glass],[data-glass-clear])[data-glass-motion] [data-glass-moving] .sticky",
    'html:is([data-glass],[data-glass-clear])[data-glass-motion] [data-slot$="-content"][data-state="closed"]',
    'html:is([data-glass],[data-glass-clear])[data-glass-motion] [data-slot$="-content"][data-state="closed"]::before',
  ]) assert.ok(pause.selectors.includes(sel), sel);
  const decl = (prop: string) => pause.nodes.find((n) => n.type === "decl" && n.prop === prop) as postcss.Declaration;
  assert.equal(decl("backdrop-filter").value, "none");
  assert.ok(decl("backdrop-filter").important);
  assert.ok(decl("-webkit-backdrop-filter").important);
  const still = movingRules.find(
    (r) => r.selector === 'html:is([data-glass],[data-glass-clear])[data-glass-motion] [data-slot$="-content"]:is([data-glass-moving], [data-glass-settling], [data-state="closed"])',
  );
  assert.ok(still, "Radix content pauses, settles and closes with its transitions off");
  assert.ok(
    still.nodes.some((n) => n.type === "decl" && n.prop === "transition" && n.value === "none" && n.important),
    "a running transition outranks !important, and transition-all would fade the blur over every frame saved",
  );
  assert.ok(!pause.nodes.some((n) => n.type === "decl" && n.prop === "transition"), "the sidebar keeps its own transitions");
  const fills = movingRules.filter((r) => r.nodes.some((n) => n.type === "decl" && n.prop === "background-color"));
  for (const surface of [".unsloth-composer-surface", '[data-slot$="menu-content"]', '[data-slot="dialog-content"]', ".tooltip-compact"]) {
    assert.ok(fills.some((r) => r.selector.includes(surface)), `${surface} has a motion fill`);
  }
  for (const rule of fills) {
    const bg = rule.nodes.find((n) => n.type === "decl" && n.prop === "background-color") as postcss.Declaration;
    assert.ok(bg.important, rule.selector);
    assert.match(bg.value, /96%|^var\(--(tooltip-bg|glass-page-popover)\)$/, `${rule.selector}: dense enough that nothing ghosts`);
  }
});

test("no later important blur can win the cascade back from a moving surface", () => {
  const at = CSS.indexOf("html:is([data-glass],[data-glass-clear])[data-glass-motion] [data-glass-moving],");
  assert.ok(at > 0);
  const tail = CSS.slice(at);
  assert.doesNotMatch(tail, /backdrop-filter:\s*blur[^;]*!important/);
});

test("in motion, Liquid 3D drops its wide blurred inner glows on dialogs and menus", () => {
  const rims = movingRules.filter((r) => r.selector.includes('[data-glass="liquid-3d"]'));
  assert.equal(rims.length, 4, "dialog and menus, light and dark");
  for (const rule of rims) {
    for (const node of rule.nodes) {
      if (node.type !== "decl" || !/^(box-shadow|--glass-rim)$/.test(node.prop)) continue;
      for (const shadow of node.value.split(/,(?![^(]*\))/)) {
        const blur = /inset\s+-?[\d.]+(?:px)?\s+-?[\d.]+(?:px)?\s+([\d.]+)px/.exec(shadow)?.[1];
        assert.ok(!blur || Number(blur) <= 2, `${rule.selector}: no wide blurred inset (${shadow.trim()})`);
      }
    }
  }
  const dialog = rims.find((r) => r.selector.includes("dialog-content") && !r.selector.includes(".dark"));
  assert.ok(dialog?.nodes.some((n) => n.type === "decl" && n.prop === "box-shadow" && n.important), "set on the dialog itself, not inherited by its subtree");
});

test("the API monitor's script fade and the Settings pane's scrolling skip per-frame blur and raster work", () => {
  const overlay = readSrc("features/api-monitor/api-monitor-overlay.tsx");
  assert.match(overlay, /onAnimationStart=\{\(\) => holdGlassMotion\(monitorRef\.current, ANIMATION_HOLD_MS\)\}/);
  assert.match(overlay, /onAnimationComplete=\{\(\) => holdGlassMotion\(monitorRef\.current, SETTLE_MS\)\}/);
  assert.match(overlay, /"menu-soft-surface /, "the panel is a glass surface the pause rules cover");
  assert.match(readSrc("features/settings/settings-dialog.tsx"), /ref=\{mainScrollRef\}\s+data-settings-main-scroll=""/);
  const rule = postcss.parse(CSS).nodes.find(
    (n): n is postcss.Rule => n.type === "rule" && n.selector === "html[data-glass][data-glass-motion] [data-settings-main-scroll]",
  );
  assert.ok(rule, "only with Glass under software compositing: GPU and the default look are unchanged");
  assert.deepEqual(
    rule.nodes.filter((n) => n.type === "decl").map((n) => `${(n as postcss.Declaration).prop}: ${(n as postcss.Declaration).value}`),
    ["will-change: scroll-position"],
  );
});
