// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

// Just enough DOM for native-view.ts: one page placeholder, and overlays the test puts over it.
const frames: (() => void)[] = [];
const mutations: (() => void)[] = [];
const overlays: {
  getBoundingClientRect: () => DOMRect;
  closest: () => null;
  querySelector: () => null;
}[] = [];
const rect = (x: number, y: number, width: number, height: number) =>
  ({
    left: x,
    top: y,
    right: x + width,
    bottom: y + height,
    width,
    height,
  }) as DOMRect;
const placeholder = {
  style: {} as Record<string, string> & {
    removeProperty: (name: string) => void;
  },
  offsetParent: {},
  isConnected: true,
  getBoundingClientRect: () => rect(500, 100, 500, 600),
  closest: () => null,
};
placeholder.style.removeProperty = (name) => {
  delete placeholder.style[
    name.replace(/-(\w)/g, (_, letter: string) => letter.toUpperCase())
  ];
};
const noop = () => undefined;
Object.assign(globalThis, {
  window: Object.assign(globalThis, {
    innerWidth: 1000,
    addEventListener: noop,
    removeEventListener: noop,
  }),
  document: {
    documentElement: { style: { setProperty: noop, removeProperty: noop } },
    body: {},
    querySelector: (selector: string) =>
      selector.startsWith("[data-native-page") ? placeholder : null,
    querySelectorAll: (selector: string) =>
      selector.startsWith(".chat-full-view-dock") ? [] : overlays,
  },
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
let captureDone: ((bytes: ArrayBuffer) => void) | null = null;
(globalThis as { nativeViewCall?: unknown }).nativeViewCall = (
  command: string,
  args?: Record<string, unknown>,
) => {
  calls.push({ command, args });
  if (command === "browser_capture")
    return new Promise((resolve) => (captureDone = resolve));
  return Promise.resolve();
};

register("./helpers/browser-store-resolver.mjs", import.meta.url);
register("./helpers/native-view-resolver.mjs", import.meta.url);
const { useBrowserStore } = await import("../src/features/browser/store.ts");
const { startNativeViews } = await import(
  "../src/features/browser/native-view.ts"
);

const settle = () => new Promise((resolve) => setTimeout(resolve, 5));
// An overlay mounting or unmounting, then the frame it schedules.
async function frame(): Promise<void> {
  for (const callback of mutations) callback();
  for (const callback of frames.splice(0)) callback();
  await settle();
}
const menu = {
  getBoundingClientRect: () => rect(900, 100, 100, 200),
  closest: () => null,
  querySelector: () => null,
};

test("a menu that closes and reopens while the page is captured keeps the snapshot", async () => {
  useBrowserStore.getState().openUrl("https://example.com/");
  const stop = startNativeViews();
  try {
    await frame();
    const tabId = useBrowserStore.getState().activeTabId;
    assert.ok(
      calls.some(
        ({ command, args }) =>
          command === "browser_view_show" && args?.tabId === tabId,
      ),
    );

    overlays.push(menu);
    await frame();
    assert.ok(captureDone, "the covered page is captured before it hides");
    // While the capture is pending, the menu closes and opens again.
    overlays.length = 0;
    await frame();
    overlays.push(menu);
    await frame();
    captureDone?.(new Uint8Array([137, 80, 78, 71]).buffer);
    await settle();
    await frame();

    assert.match(placeholder.style.backgroundImage ?? "", /^url\(blob:/);
    const shows = calls.filter(
      ({ command }) => command === "browser_view_show",
    );
    assert.equal(
      shows.at(-1)?.args?.tabId,
      null,
      "the page stays hidden under the menu",
    );

    overlays.length = 0;
    await frame();
    assert.equal(
      placeholder.style.backgroundImage,
      undefined,
      "the snapshot goes once the page shows",
    );
  } finally {
    stop();
  }
});
