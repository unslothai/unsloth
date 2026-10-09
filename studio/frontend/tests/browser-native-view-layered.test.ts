// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

// macOS: the page sits under the app's webview, so panel UI draws over it without a snapshot.
const frames: (() => void)[] = [];
const mutations: (() => void)[] = [];
const rect = (x: number, y: number, width: number, height: number) =>
  ({ left: x, top: y, right: x + width, bottom: y + height, width, height }) as DOMRect;
type Overlay = { getBoundingClientRect: () => DOMRect; closest: () => null; querySelector: () => null };
const overlay = (box: DOMRect): Overlay => ({
  getBoundingClientRect: () => box,
  closest: () => null,
  querySelector: () => null,
});
// overlays by selector: menus and dialogs block, toasts and the find bar take only their rects
const blocking: Overlay[] = [];
const clickable: Overlay[] = [];

function node(name: string, background: string, parent: FakeNode | null) {
  const vars = new Map<string, string>();
  const attributes = new Set<string>();
  return {
    name,
    background,
    parentElement: parent,
    className: name,
    isConnected: true,
    offsetParent: {},
    attributes,
    vars,
    style: {
      setProperty: (key: string, value: string) => vars.set(key, value),
      removeProperty: (key: string) => vars.delete(key),
      getPropertyValue: (key: string) => vars.get(key) ?? "",
    },
    setAttribute: (key: string) => attributes.add(key),
    removeAttribute: (key: string) => attributes.delete(key),
    getBoundingClientRect: () => rect(500, 100, 500, 600),
    closest: () => null,
  };
}
type FakeNode = ReturnType<typeof node>;
const html = node("html", "rgb(255, 255, 255)", null);
const section = node("section", "oklch(0.97 0 0)", html);
const wrapper = node("wrapper", "rgba(0, 0, 0, 0)", section);
const placeholder = node("placeholder", "oklch(1 0 0 / 1)", wrapper);

const noop = () => undefined;
Object.assign(globalThis, {
  window: Object.assign(globalThis, { innerWidth: 1000, addEventListener: noop, removeEventListener: noop }),
  document: {
    documentElement: Object.assign(html, { dataset: {} }),
    body: {},
    querySelector: (selector: string) => (selector.startsWith("[data-native-page") ? placeholder : null),
    querySelectorAll: (selector: string) =>
      selector.startsWith("[data-sonner-toast]") ? clickable : selector.startsWith("[data-radix") ? blocking : [],
  },
  getComputedStyle: (element: FakeNode) => ({ backgroundColor: element.background, backgroundImage: "none" }),
  CSS: { escape: (value: string) => value },
  DOMRect: class {
    constructor(x: number, y: number, width: number, height: number) {
      // biome-ignore lint/correctness/noConstructorReturn: a plain rect stands in for DOMRect
      return rect(x, y, width, height);
    }
  },
  requestAnimationFrame: (callback: () => void) => frames.push(callback),
  cancelAnimationFrame: noop,
  ResizeObserver: class {
    observe() {}
    unobserve() {}
    disconnect() {}
  },
  MutationObserver: class {
    constructor(callback: () => void) {
      mutations.push(callback);
    }
    observe() {}
    disconnect() {}
  },
});

type Call = { command: string; args?: Record<string, unknown> };
const calls: Call[] = [];
(globalThis as { nativeViewCall?: unknown }).nativeViewCall = (command: string, args?: Record<string, unknown>) => {
  calls.push({ command, args });
  return Promise.resolve(command === "browser_view_layered" ? true : undefined);
};

register("./helpers/browser-store-resolver.mjs", import.meta.url);
register("./helpers/native-view-resolver.mjs", import.meta.url);
const { useBrowserStore } = await import("../src/features/browser/store.ts");
const { startNativeViews } = await import("../src/features/browser/native-view.ts");

const settle = () => new Promise((resolve) => setTimeout(resolve, 5));
async function frame(): Promise<void> {
  for (const callback of mutations) callback();
  for (const callback of frames.splice(0)) callback();
  await settle();
}
const lastInput = () => calls.filter(({ command }) => command === "browser_view_input").at(-1)?.args;

test("a menu over a layered page neither captures nor hides it, and owns its input", async () => {
  useBrowserStore.getState().openUrl("https://example.com/");
  const stop = startNativeViews();
  try {
    await frame();
    await frame();
    const tabId = useBrowserStore.getState().activeTabId;
    assert.ok(calls.some(({ command, args }) => command === "browser_view_show" && args?.tabId === tabId));

    // every opaque background around the page skips its rect; transparent ones are left alone
    assert.equal(html.vars.get("--native-hole-x"), "500px");
    assert.equal(html.vars.get("--native-hole-h"), "600px");
    assert.deepEqual(
      [html, section, wrapper, placeholder].map((element) => element.attributes.has("data-native-hole")),
      [true, true, false, true],
    );
    assert.equal(section.vars.get("--native-hole-bg"), "oklch(0.97 0 0)");

    // a theme switch recolors them, though the page didn't move; read mid-switch, it tries again
    html.className = "html dark";
    for (const element of [html, section, placeholder]) element.background = "rgba(0, 0, 0, 0)";
    await frame();
    assert.equal(section.attributes.has("data-native-hole"), false);
    section.background = "oklch(0.2 0 0)";
    await frame();
    assert.equal(section.vars.get("--native-hole-bg"), "oklch(0.2 0 0)");

    const shows = calls.filter(({ command }) => command === "browser_view_show").length;
    blocking.push(overlay(rect(900, 100, 100, 200)));
    await frame();
    assert.deepEqual(lastInput(), { blocked: true, exclude: [] });
    assert.equal(calls.filter(({ command }) => command === "browser_view_show").length, shows, "the page stays put");
    assert.ok(!calls.some(({ command }) => command === "browser_capture"), "nothing is captured");

    // zoom from the menu reaches the live page at once
    if (tabId) useBrowserStore.getState().updateTab(tabId, { zoom: 1.5 });
    await frame();
    assert.ok(calls.some(({ command, args }) => command === "browser_view_zoom" && args?.zoom === 1.5));

    // a toast over the page takes input only where it is
    blocking.length = 0;
    clickable.push(overlay(rect(600, 600, 200.5, 50)));
    await frame();
    assert.deepEqual(lastInput(), {
      blocked: false,
      exclude: [{ x: 600, y: 600, width: 202, height: 51, viewportWidth: 1000 }],
    });
    clickable.length = 0;
    await frame();
    assert.deepEqual(lastInput(), { blocked: false, exclude: [] });
  } finally {
    stop();
  }
  assert.equal(placeholder.attributes.has("data-native-hole"), false, "closing the panel fills the hole");
  assert.equal(html.vars.has("--native-hole-x"), false);
});
