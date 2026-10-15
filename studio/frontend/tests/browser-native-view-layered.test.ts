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
type Overlay = {
  getBoundingClientRect: () => DOMRect;
  closest: () => null;
  querySelector: (selector: string) => object | null;
};
const overlay = (box: DOMRect): Overlay => ({
  getBoundingClientRect: () => box,
  closest: () => null,
  querySelector: () => null,
});
// overlays by selector: menus and dialogs block, toasts and the find bar take only their rects
const blocking: Overlay[] = [];
const clickable: Overlay[] = [];

function node(name: string, background: string, parent: object | null) {
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
const listeners = new Map<string, (event: { target: unknown }) => void>();
let panelFocused = true;
Object.assign(globalThis, {
  window: Object.assign(globalThis, { innerWidth: 1000, addEventListener: noop, removeEventListener: noop }),
  document: {
    documentElement: Object.assign(html, { dataset: {} }),
    body: {},
    activeElement: null,
    hasFocus: () => panelFocused,
    addEventListener: (type: string, listener: (event: { target: unknown }) => void) => listeners.set(type, listener),
    removeEventListener: (type: string) => listeners.delete(type),
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
const { keepInBackground, releaseBackground, startNativeViews } = await import("../src/features/browser/native-view.ts");

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
    // the rect goes on each holed node (the vars don't inherit), so moving it restyles only those
    assert.equal(placeholder.vars.get("--native-hole-y"), "100px");
    assert.equal(wrapper.vars.has("--native-hole-x"), false);

    // under the app's interface zoom the rect is divided by it: WebKit scales a fixed background's
    // lengths by the zoom on top of the layout they came from
    const zoomChanged = (globalThis as { nativeViewZoomChanged?: () => void }).nativeViewZoomChanged;
    (globalThis as { nativeViewZoom?: number }).nativeViewZoom = 0.8;
    zoomChanged?.();
    assert.equal(html.vars.get("--native-hole-x"), "625px");
    assert.equal(placeholder.vars.get("--native-hole-h"), "750px");
    (globalThis as { nativeViewZoom?: number }).nativeViewZoom = 1;
    zoomChanged?.();
    assert.equal(html.vars.get("--native-hole-x"), "500px");

    // a theme switch recolors them, though the page didn't move; read mid-switch, it tries again
    html.className = "html dark";
    for (const element of [html, section, placeholder]) element.background = "rgba(0, 0, 0, 0)";
    await frame();
    assert.equal(section.attributes.has("data-native-hole"), false);
    section.background = "oklch(0.2 0 0)";
    await frame();
    assert.equal(section.vars.get("--native-hole-bg"), "oklch(0.2 0 0)");

    // a split drag pins the page at its old width while the panel shrinks past it: the app covers
    // the overflow by cutting the hole no wider than the panel, and the page view isn't resized,
    // which would relayout the page every frame
    let panelLeft = 650;
    const panelBody = { getBoundingClientRect: () => rect(panelLeft, 0, 1000 - panelLeft, 800), closest: () => null };
    // the pinned outer page box keeps its old rect, the page's own
    const outerPage: { parentElement: object; closest: () => unknown; getBoundingClientRect: () => unknown } = {
      parentElement: panelBody,
      closest: () => outerPage,
      getBoundingClientRect: () => rect(500, 100, 500, 600),
    };
    const innerPage = { parentElement: outerPage, closest: () => innerPage };
    const unpinned = placeholder.closest;
    Object.assign(placeholder, { closest: () => innerPage });
    const showsBeforeDrag = calls.filter(({ command }) => command === "browser_view_show").length;
    for (const left of [650, 700, 750]) {
      panelLeft = left;
      await frame();
      assert.equal(section.vars.get("--native-hole-x"), `${left}px`);
      assert.equal(section.vars.get("--native-hole-w"), `${1000 - left}px`);
    }
    assert.equal(calls.filter(({ command }) => command === "browser_view_show").length, showsBeforeDrag, "the page view keeps its frame");
    // input over the covered strip goes to the app, not the page view still under it
    assert.deepEqual(lastInput(), {
      blocked: false,
      exclude: [{ x: 500, y: 100, width: 250, height: 600, viewportWidth: 1000 }],
    });
    Object.assign(placeholder, { closest: unpinned });
    await frame();
    assert.equal(section.vars.get("--native-hole-x"), "500px", "released: the whole page again");

    const shows = calls.filter(({ command }) => command === "browser_view_show").length;
    blocking.push(overlay(rect(900, 100, 100, 200)));
    await frame();
    assert.deepEqual(lastInput(), { blocked: true, exclude: [] });
    assert.equal(calls.filter(({ command }) => command === "browser_view_show").length, shows, "the page stays put");
    assert.ok(!calls.some(({ command }) => command === "browser_capture"), "nothing is captured");

    // zoom from the menu reaches the live page at once
    if (tabId) useBrowserStore.getState().setZoom(tabId, 1.5);
    await frame();
    assert.ok(calls.some(({ command, args }) => command === "browser_view_zoom" && args?.zoom === 1.5));

    // a menu away from the page owns it too, so a click there closes the menu; a hover card doesn't
    blocking.length = 0;
    blocking.push(overlay(rect(0, 600, 200, 100)));
    await frame();
    assert.deepEqual(lastInput(), { blocked: true, exclude: [] });
    blocking.length = 0;
    blocking.push({
      ...overlay(rect(0, 600, 200, 100)),
      querySelector: (selector: string) => (selector.includes("hover-card") ? {} : null),
    });
    await frame();
    assert.deepEqual(lastInput(), { blocked: false, exclude: [] });

    // the dialog the browser opens in (compare, projects, phones) holds the page: it isn't over it
    blocking.length = 0;
    blocking.push({
      ...overlay(rect(0, 0, 1000, 800)),
      querySelector: (selector: string) => (selector.startsWith("[data-native-page") ? placeholder : null),
    });
    await frame();
    assert.deepEqual(lastInput(), { blocked: false, exclude: [] });

    // a dialog the site raised while its page had the keys takes them, and gives them back on close
    blocking.length = 0;
    await frame();
    panelFocused = false;
    blocking.push(overlay(rect(600, 200, 300, 200)));
    await frame();
    const actions = () => calls.filter(({ command }) => command === "browser_view_action").map(({ args }) => args?.action);
    assert.deepEqual(actions().slice(-1), ["blur"]);
    panelFocused = true;
    blocking.length = 0;
    await frame();
    await frame();
    assert.deepEqual(actions().slice(-1), ["focus"]);

    // switched to another tab meanwhile: the hidden page doesn't get the keys back
    panelFocused = false;
    blocking.push(overlay(rect(600, 200, 300, 200)));
    await frame();
    useBrowserStore.getState().openUrl("https://example.org/other", { newTab: true });
    await frame();
    await frame();
    panelFocused = true;
    blocking.length = 0;
    await frame();
    await frame();
    assert.equal(
      calls.filter(({ command, args }) => command === "browser_view_action" && args?.action === "focus" && args?.tabId === tabId).length,
      1,
      "only the earlier hand-back focused the first tab",
    );

    // a toast over the page takes input only where it is
    blocking.length = 0;
    clickable.push(overlay(rect(600, 600, 200.5, 50)));
    await frame();
    assert.deepEqual(lastInput(), {
      blocked: false,
      exclude: [{ x: 600, y: 600, width: 202, height: 51, viewportWidth: 1000 }],
    });
    // a toast sliding in is measured as it moves, not on the next recheck
    clickable[0] = overlay(rect(600, 560, 200, 50));
    listeners.get("transitionrun")?.({ target: { closest: () => ({}) } });
    for (const callback of frames.splice(0)) callback();
    await settle();
    assert.equal((lastInput()?.exclude as { y: number }[])[0].y, 560);
    assert.ok(frames.length > 0, "and again next frame while it moves");
    clickable.length = 0;
    await frame();
    assert.deepEqual(lastInput(), { blocked: false, exclude: [] });
  } finally {
    stop();
  }
  assert.equal(placeholder.attributes.has("data-native-hole"), false, "closing the panel fills the hole");
  assert.equal(html.vars.has("--native-hole-x"), false);
});

test("a page at work in a chat off screen opens hidden, follows its tab, and outlives the panel", async () => {
  const store = useBrowserStore.getState();
  store.closePanel();
  store.showChat("single:shown");
  const tabId = store.openTabIn("single:away", "https://away.example/") ?? "";
  calls.length = 0;
  keepInBackground(tabId);
  await settle();
  const opened = calls.find(({ command }) => command === "browser_view_open_background");
  assert.deepEqual([opened?.args?.tabId, opened?.args?.url], [tabId, "https://away.example/"]);
  // The agent moves it on: the hidden page follows, and is never shown over this chat.
  store.navigate(tabId, { url: "https://away.example/next" }, { replace: false });
  await settle();
  assert.ok(calls.some(({ command, args }) => command === "browser_view_navigate" && args?.url === "https://away.example/next"));
  // A panel opening and closing over this chat leaves it open (closed pages go; a kept one is hidden).
  store.openUrl("https://mine.example/");
  const stop = startNativeViews();
  await frame();
  await frame();
  stop();
  assert.ok(!calls.some(({ command, args }) => command === "browser_view_show" && args?.tabId === tabId));
  assert.ok(!calls.some(({ command, args }) => command === "browser_view_close" && args?.tabId === tabId));
  assert.deepEqual(calls.at(-1), { command: "browser_view_show", args: { tabId: null } });
  // Let go with no panel up, it closes.
  releaseBackground(tabId);
  assert.ok(calls.some(({ command, args }) => command === "browser_view_close" && args?.tabId === tabId));
});
