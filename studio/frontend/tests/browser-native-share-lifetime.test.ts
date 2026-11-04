// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

const frames: (() => void)[] = [];
const mutations: (() => void)[] = [];
const rect = (x: number, y: number, width: number, height: number) =>
  ({ left: x, top: y, right: x + width, bottom: y + height, width, height }) as DOMRect;
type Overlay = {
  getBoundingClientRect: () => DOMRect;
  closest: () => null;
  querySelector: (selector: string) => object | null;
};
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
const panelFocused = true;
Object.assign(globalThis, {
  localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
  window: Object.assign(globalThis, { innerWidth: 1000, location: { protocol: "http:" }, addEventListener: noop, removeEventListener: noop }),
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
  return Promise.resolve(command === "browser_view_layered" ? process.env.TEST_LAYERED === "1" : undefined);
};

register("./helpers/native-share-resolver.mjs", import.meta.url);
const { useBrowserStore } = await import("../src/features/browser/store.ts");
const { startNativeViews, keepInBackground, releaseBackground, hasNativeView } = await import("../src/features/browser/native-view.ts");
const { shareAgentTab } = await import("../src/features/browser/agent/tab-sharing.ts");
const agent = await import("../src/features/browser/agent/agent-store.ts");
const settle = () => new Promise((resolve) => setTimeout(resolve, 5));
async function frame() { for (const callback of mutations) callback(); for (const callback of frames.splice(0)) callback(); await settle(); }
const count = (command: string, tabId: string) => calls.filter((call) => call.command === command && call.args?.tabId === tabId).length;
const loads = (tabId: string) => calls.filter(({ command, args }) => args?.tabId === tabId && (command === "browser_view_navigate" || command === "browser_view_open_background" || (command === "browser_view_show" && args.url))).length;

test("shared native page survives panel teardown before executor loads", async () => {
  const store = useBrowserStore.getState(); const chat = store.chat;
  const tab = store.openTabIn(chat, "https://example.com/native")!; store.activateTab(tab);
  const stop = startNativeViews(); await frame(); await frame();
  assert.ok(count("browser_view_show", tab) > 0);
  const originalLoads = loads(tab);
  shareAgentTab(chat, tab); shareAgentTab(chat, tab);
  const opened = count("browser_view_show", tab); const closed = count("browser_view_close", tab);
  stop(); await settle();
  assert.equal(count("browser_view_close", tab), closed);
  assert.equal(count("browser_view_show", tab), opened);
  assert.equal(loads(tab), originalLoads);
  assert.equal(agent.agentChat(chat).shared, true);
});

test("run end retains a shared page; switch, revoke and tab close release it", async () => {
  await import("../src/features/browser/agent/executor.ts");
  const store = useBrowserStore.getState(); const chat = store.chat;
  const first = store.activeTabId!;
  const stop = startNativeViews(); await frame(); await frame();
  const opened = count("browser_view_show", first);
  const originalLoads = loads(first);
  keepInBackground(first);
  agent.patchAgentChat(chat, { runKey: "native-shared-run" });
  stop(); await settle();
  const closed = count("browser_view_close", first);
  agent.endBrowserAgentRun("native-shared-run"); await settle();
  assert.equal(count("browser_view_close", first), closed);
  assert.equal(count("browser_view_show", first), opened);
  assert.equal(loads(first), originalLoads);
  const second = store.openTabIn(chat, "https://example.com/native2")!;
  store.activateTab(second); shareAgentTab(chat, second); await settle();
  assert.equal(count("browser_view_close", first), closed + 1);
  const secondClosed = count("browser_view_close", second);
  shareAgentTab(chat, null); await settle();
  assert.equal(count("browser_view_close", second), secondClosed + 1);
  store.activateTab(first); shareAgentTab(chat, first); await settle();
  const beforeClose = count("browser_view_close", first);
  store.closeTab(first); await settle();
  assert.equal(count("browser_view_close", first), beforeClose + 1);
});

test("an agent-owned page survives its run and closes when the tab is closed", async () => {
  const store = useBrowserStore.getState(); const chat = store.chat;
  const tab = store.openTabIn(chat, "https://example.com/agent-owned")!;
  store.activateTab(tab);
  agent.patchAgentChat(chat, { tabId: tab, shared: false, runKey: "native-owned-run" });
  keepInBackground(tab); await settle();
  const closed = count("browser_view_close", tab);
  agent.endBrowserAgentRun("native-owned-run"); await settle();
  assert.equal(count("browser_view_close", tab), closed);
  store.closeTab(tab); await settle();
  assert.equal(count("browser_view_close", tab), closed + 1);
});

for (const first of ["panel", "agent"] as const) {
  test(`${first}-first native creation is shared with the other opener`, async () => {
    const store = useBrowserStore.getState();
    const tab = store.openTabIn(store.chat, `https://example.com/race-${first}`)!;
    const bridge = globalThis as unknown as { nativeViewCall: (command: string, args?: Record<string, unknown>) => Promise<unknown> };
    const original = bridge.nativeViewCall;
    let release: (() => void) | undefined;
    let creating = false;
    let created = false;
    let duplicate = false;
    let stop: (() => void) | undefined;
    let otherTab: string | null = null;
    bridge.nativeViewCall = (command, args) => {
      if (args?.tabId === tab && ["browser_view_show", "browser_view_open_background"].includes(command) && !created) {
        if (creating) {
          duplicate = true;
          return Promise.reject(new Error(`a webview with label ${tab} already exists`));
        }
        creating = true;
        return new Promise((resolve) => {
          release = () => { created = true; creating = false; resolve(undefined); };
        });
      }
      return original(command, args);
    };
    try {
      if (first === "agent") {
        keepInBackground(tab);
        await settle();
      }
      stop = startNativeViews();
      await frame();
      if (first === "panel") keepInBackground(tab);
      await settle();
      assert.equal(duplicate, false, "foreground/background creation overlapped");
      assert.ok(release, "the native creation path did not run");
      otherTab = store.openTabIn("single:independent-native", "https://example.com/independent");
      assert.ok(otherTab);
      keepInBackground(otherTab);
      await settle();
      assert.equal(created, false);
      assert.ok(count("browser_view_open_background", otherTab) > 0, "another chat must not wait for this tab's creation");
      release();
      await frame();
      await frame();
      assert.equal(useBrowserStore.getState().tabs.find((item) => item.id === tab)?.nativeError ?? null, null);
      assert.equal(hasNativeView(tab), true);
      assert.equal(count("browser_view_navigate", tab), 0, "joining creation must preserve its document");
    } finally {
      release?.();
      await settle();
      stop?.();
      releaseBackground(tab);
      useBrowserStore.getState().closeTab(tab);
      if (otherTab) {
        releaseBackground(otherTab);
        useBrowserStore.getState().closeTab(otherTab);
      }
      bridge.nativeViewCall = original;
    }
  });
}
