// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/toast-resolver.mjs", import.meta.url);

type Listener = (event: { clientX: number; clientY: number; buttons: number }) => void;
const windowListeners = new Map<string, Set<Listener>>();
const frames: (() => void)[] = [];
const storage = new Map<string, string>();
const rect = (left: number, top: number, width: number, height: number) =>
  ({ left, top, width, height, right: left + width, bottom: top + height }) as DOMRect;

Object.assign(globalThis, {
  window: globalThis,
  localStorage: {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => void storage.set(key, value),
  },
  addEventListener: (type: string, listener: Listener) => {
    if (!windowListeners.has(type)) windowListeners.set(type, new Set());
    windowListeners.get(type)?.add(listener);
  },
  removeEventListener: (type: string, listener: Listener) => windowListeners.get(type)?.delete(listener),
  dispatchEvent: () => true,
  requestAnimationFrame: (callback: () => void) => frames.push(callback),
  cancelAnimationFrame: () => undefined,
  ResizeObserver: class {
    observe() {}
    disconnect() {}
  },
  document: {
    createElement: () => ({ style: {}, setAttribute() {}, remove() {} }),
    body: { appendChild() {} },
  },
});

// The chat: x 300..1300, y 0..900, its header 48 tall.
const header = { getBoundingClientRect: () => rect(300, 0, 1000, 48) };
const root = { querySelector: () => header };
let composer: DOMRect | null = null;
const composerForm = {
  offsetParent: {},
  closest: () => null,
  getBoundingClientRect: () => composer ?? rect(0, 0, 0, 0),
};
const split = {
  getBoundingClientRect: () => rect(300, 0, 1000, 900),
  closest: () => root,
  querySelector: () => composerForm,
  querySelectorAll: () => (composer ? [composerForm] : []),
};
let dockedWidth = 600;
const attributes = new Set<string>(["data-browser-floating"]);
const style: Record<string, string> = {};
const surface = {
  style: Object.assign(style, {
    getPropertyValue: (name: string) => style[name] ?? "",
    setProperty: (name: string, value: string) => void (style[name] = value),
    removeProperty: (name: string) => void delete style[name],
  }),
  closest: (selector: string) => (selector === ".chat-artifact-split" ? split : surface),
  getBoundingClientRect: () => rect(700, 0, dockedWidth, 900),
  hasAttribute: (name: string) => attributes.has(name),
  setAttribute: (name: string) => void attributes.add(name),
  removeAttribute: (name: string) => void attributes.delete(name),
};
const placed = () => ["left", "top", "width", "height"].map((key) => Number.parseFloat(style[key] ?? "NaN"));

const { floatSurface, startFloatingGesture } = await import("../src/features/browser/floating-preview.ts");

function drag(edge: Parameters<typeof startFloatingGesture>[1], from: [number, number], to: [number, number]) {
  const target = { closest: () => null };
  startFloatingGesture(
    {
      button: 0,
      clientX: from[0],
      clientY: from[1],
      target,
      currentTarget: { closest: () => surface },
      preventDefault() {},
    } as never,
    edge,
  );
  for (const listener of windowListeners.get("pointermove") ?? []) listener({ clientX: to[0], clientY: to[1], buttons: 1 });
  while (frames.length) frames.shift()?.();
  for (const listener of [...(windowListeners.get("pointerup") ?? [])]) listener({ clientX: to[0], clientY: to[1], buttons: 0 });
}

test("the preview opens at the chat's top right, below its header, and gives the surface back when it docks", () => {
  style.transform = "none";
  const release = floatSurface(surface as never);
  // 38% of the 976px room is 371; 12px in from the right and below the header.
  assert.deepEqual(placed(), [1300 - 12 - 371, 48 + 12, 371, Math.round(371 * 0.68)]);
  assert.equal(style.position, "fixed");
  assert.equal(style["--browser-floating-scale"], String(Math.round(((371 - 2) / 600) * 1000) / 1000));
  release();
  assert.equal(style["--browser-floating-scale"], undefined);
  assert.equal(style.position, undefined);
  assert.equal(style.transform, "none");
});

test("a press without a drag leaves the preview where it is, so a double-click on its header can dock it", () => {
  const release = floatSurface(surface as never);
  const before = placed();
  drag("move", [before[0] + 20, before[1] + 10], [before[0] + 21, before[1] + 11]);
  assert.deepEqual(placed(), before);
  assert.equal(storage.has("unsloth_browser_floating_preview"), false);
  release();
});

test("a drag that outlives the preview leaves the docked panel alone", () => {
  const release = floatSurface(surface as never);
  const target = { closest: () => null };
  startFloatingGesture(
    { button: 0, clientX: 900, clientY: 70, target, currentTarget: { closest: () => surface }, preventDefault() {} } as never,
    "move",
  );
  for (const listener of windowListeners.get("pointermove") ?? []) listener({ clientX: 500, clientY: 300, buttons: 1 });
  release();
  attributes.delete("data-browser-floating");
  for (const listener of [...(windowListeners.get("pointerup") ?? [])]) listener({ clientX: 500, clientY: 300, buttons: 0 });
  while (frames.length) frames.shift()?.();
  assert.equal(style.position, undefined);
  assert.equal(storage.has("unsloth_browser_floating_preview"), false);
  attributes.add("data-browser-floating");
});

test("a move stays in the chat, and one dropped at the right edge stays anchored there", () => {
  const release = floatSurface(surface as never);
  drag("move", [1000, 70], [-500, 2000]);
  const [left, top, width, height] = placed();
  assert.equal(left, 312);
  assert.equal(top + height, 900 - 12);
  assert.equal(width, 371);
  drag("move", [400, 700], [2400, 700]);
  assert.equal(JSON.parse(storage.get("unsloth_browser_floating_preview") ?? "{}").left, null);
  assert.equal(placed()[0] + placed()[2], 1300 - 12);
  release();
});

test("a resize from a corner keeps the opposite corner put and never goes under the minimum", () => {
  const release = floatSurface(surface as never);
  const [left, top, width, height] = placed();
  drag("nw", [left, top], [left - 200, top - 100]);
  const grown = placed();
  assert.equal(grown[0] + grown[2], left + width);
  assert.equal(grown[1] + grown[3], top + height);
  assert.equal(grown[2], width + 200);
  drag("nw", [grown[0], grown[1]], [grown[0] + 5000, grown[1] + 5000]);
  const shrunk = placed();
  assert.deepEqual([shrunk[2], shrunk[3]], [260, 180]);
  assert.equal(shrunk[0] + shrunk[2], left + width);
  release();
});

test("the window goes wherever it is dragged: over the composer and up over the chat header", () => {
  composer = rect(500, 760, 600, 120);
  dockedWidth = 0;
  const release = floatSurface(surface as never);
  const [left, top, width, height] = placed();
  drag("move", [left + 10, top + 10], [800, 790]);
  // Dropped onto the composer, it stays there, down at the chat's bottom edge.
  assert.equal(placed()[1] + placed()[3], 900 - 12);
  drag("move", [800, placed()[1]], [800, -500]);
  // Up to the chat's own edge, over its 48px header.
  assert.equal(placed()[1], 12);
  drag("s", [placed()[0] + 10, 12 + height], [placed()[0] + 10, 2000]);
  assert.deepEqual([placed()[1], placed()[1] + placed()[3], placed()[2]], [12, 900 - 12, width]);
  release();
  composer = null;
});
