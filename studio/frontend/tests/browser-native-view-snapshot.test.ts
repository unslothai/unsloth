// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

// minimal DOM for native-view.ts with one page placeholder and test-controlled overlays
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
let pageBox = rect(500, 100, 500, 600);
const placeholder = {
  style: {} as Record<string, string> & {
    removeProperty: (name: string) => void;
  },
  offsetParent: {},
  isConnected: true,
  getBoundingClientRect: () => pageBox,
  closest: () => null,
};
const rootVars = new Map<string, string>();
const rootStyle = {
  getPropertyValue: (name: string) => rootVars.get(name) ?? "",
  setProperty: (name: string, value: string) => rootVars.set(name, value),
  removeProperty: (name: string) => rootVars.delete(name),
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
    documentElement: { style: rootStyle },
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
const { currentEntry, useBrowserStore } = await import("../src/features/browser/store.ts");
const { startNativeViews } = await import(
  "../src/features/browser/native-view.ts"
);

const settle = () => new Promise((resolve) => setTimeout(resolve, 5));
// apply overlay mutations before their scheduled frame
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
    // close and reopen the menu before capture resolves
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

test("toasts move left of a page that sits beside the Run settings panel", async () => {
  const stop = startNativeViews();
  try {
    pageBox = rect(500, 100, 500, 600);
    await frame();
    assert.equal(rootVars.get("--studio-browser-page-inset"), "500px");

    // ignore pages away from the window edge because they do not constrain toasts
    pageBox = rect(400, 100, 300, 600);
    await frame();
    assert.equal(rootVars.has("--studio-browser-page-inset"), false);

    // include the 300px Run settings panel between the page and window edge
    rootVars.set("--studio-chat-settings-inset", "300px");
    await frame();
    assert.equal(rootVars.get("--studio-browser-page-inset"), "600px");

    // omit the inset when no toast column fits; an overlapping toast will hide the page instead
    pageBox = rect(200, 100, 500, 600);
    await frame();
    assert.equal(rootVars.has("--studio-browser-page-inset"), false);
  } finally {
    stop();
    rootVars.clear();
    pageBox = rect(500, 100, 500, 600);
  }
});

test("a finished download is listed and reported after its tab closed, and warns when it isn't marked", async () => {
  const stop = startNativeViews();
  try {
    useBrowserStore.getState().openUrl("https://example.org/", { newTab: true });
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    useBrowserStore.getState().closeTab(tabId);
    await frame();
    const seen: { level: string; message: string }[] = [];
    const g = globalThis as {
      nativeViewListener?: (event: { payload: unknown }) => void;
      nativeViewSeen?: unknown;
    };
    g.nativeViewSeen = seen;
    const done = {
      kind: "download",
      tabId,
      url: "https://example.com/a.zip",
      name: "a.zip",
      path: null,
      size: 3,
      done: true,
      success: true,
    };
    g.nativeViewListener?.({ payload: { ...done, downloadId: "d1", marked: true } });
    g.nativeViewListener?.({ payload: { ...done, downloadId: "d2", marked: false } });
    g.nativeViewListener?.({ payload: { ...done, downloadId: "d3", marked: null } });
    // A tab this page never opened: the account signed in before the last reload (an account switch) started it.
    g.nativeViewListener?.({ payload: { ...done, tabId: "tab-before-reload", downloadId: "d4", marked: true } });
    assert.deepEqual(seen, [
      { level: "history", message: "d1" },
      { level: "success", message: "browser.native.downloaded" },
      { level: "history", message: "d2" },
      { level: "warning", message: "browser.native.notMarked" },
      { level: "history", message: "d3" },
      { level: "success", message: "browser.native.downloaded" },
    ]);
  } finally {
    stop();
  }
});

test("an approved download runs on the Downloads button until it lands or its save is cancelled", async () => {
  const stop = startNativeViews();
  const g = globalThis as {
    nativeViewListener?: (event: { payload: unknown }) => void;
    nativeViewSeen?: unknown;
    nativeViewActivity?: string[];
    nativeViewApprove?: boolean;
    nativeViewDownloadsButton?: boolean;
  };
  g.nativeViewApprove = true;
  g.nativeViewDownloadsButton = true;
  try {
    useBrowserStore.getState().openUrl("https://example.org/", { newTab: true });
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    g.nativeViewSeen = [];
    g.nativeViewActivity = [];
    const prompt = { kind: "downloadPrompt", tabId, site: "https://example.org/", name: "a.zip" };
    g.nativeViewListener?.({ payload: { ...prompt, url: "https://example.org/a.zip", id: "p1" } });
    g.nativeViewListener?.({ payload: { ...prompt, url: "https://example.org/b.zip", id: "p2" } });
    await frame();
    g.nativeViewListener?.({
      payload: {
        kind: "download", tabId, url: "https://example.org/a.zip", name: "a.zip", path: null, size: 3,
        done: true, success: true, downloadId: "d1", marked: true, promptId: "p1",
      },
    });
    g.nativeViewListener?.({
      payload: { kind: "downloadCancelled", tabId, url: "https://example.org/b.zip", promptId: "p2" },
    });
    assert.deepEqual(g.nativeViewActivity, ["begin native:p1", "begin native:p2", "finish native:p1", "abandon native:p2"]);
    // The button shows it running, so no "downloading" toast.
    assert.deepEqual(g.nativeViewSeen, [{ level: "history", message: "d1" }]);
  } finally {
    g.nativeViewApprove = false;
    g.nativeViewDownloadsButton = false;
    stop();
  }
});

test("two downloads of one address run apart: the first to land doesn't end the other", async () => {
  const stop = startNativeViews();
  const g = globalThis as {
    nativeViewListener?: (event: { payload: unknown }) => void;
    nativeViewSeen?: unknown;
    nativeViewActivity?: string[];
    nativeViewApprove?: boolean;
  };
  g.nativeViewApprove = true;
  try {
    useBrowserStore.getState().openUrl("https://example.org/", { newTab: true });
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const url = "https://example.org/same.zip";
    g.nativeViewSeen = [];
    g.nativeViewActivity = [];
    const prompt = { kind: "downloadPrompt", tabId, url, site: "https://example.org/", name: "same.zip" };
    g.nativeViewListener?.({ payload: { ...prompt, id: "p1" } });
    g.nativeViewListener?.({ payload: { ...prompt, id: "p2" } });
    await frame();
    g.nativeViewListener?.({
      payload: {
        kind: "download", tabId, url, name: "same.zip", path: null, size: 3,
        done: true, success: true, downloadId: "d1", marked: true, promptId: "p1",
      },
    });
    assert.deepEqual(g.nativeViewActivity, ["begin native:p1", "begin native:p2", "finish native:p1"]);
  } finally {
    g.nativeViewApprove = false;
    stop();
  }
});

test("a download that finished while its prompt was open doesn't keep spinning", async () => {
  const stop = startNativeViews();
  const g = globalThis as {
    nativeViewListener?: (event: { payload: unknown }) => void;
    nativeViewSeen?: unknown;
    nativeViewActivity?: string[];
    nativeViewApprove?: boolean;
    nativeViewDecide?: () => void;
  };
  g.nativeViewApprove = true;
  try {
    useBrowserStore.getState().openUrl("https://example.org/", { newTab: true });
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const url = "https://example.org/big.zip";
    g.nativeViewSeen = [];
    g.nativeViewActivity = [];
    // Already in staging, so the app delivers it before the decide call returns.
    g.nativeViewDecide = () =>
      g.nativeViewListener?.({
        payload: {
          kind: "download", tabId, url, name: "big.zip", path: null, size: 3,
          done: true, success: true, downloadId: "d1", marked: true, promptId: "p1",
        },
      });
    g.nativeViewListener?.({
      payload: { kind: "downloadPrompt", tabId, url, site: "https://example.org/", name: "big.zip", id: "p1" },
    });
    await frame();
    assert.deepEqual(g.nativeViewActivity, ["begin native:p1", "finish native:p1"]);
    // No button on screen: it says it downloaded, and not then that it's downloading.
    assert.deepEqual(g.nativeViewSeen, [
      { level: "history", message: "d1" },
      { level: "success", message: "browser.native.downloaded" },
    ]);
  } finally {
    g.nativeViewApprove = false;
    g.nativeViewDecide = undefined;
    stop();
  }
});

test("with a Downloads button on screen, a download shows there, not as a toast", async () => {
  const stop = startNativeViews();
  const g = globalThis as {
    nativeViewListener?: (event: { payload: unknown }) => void;
    nativeViewSeen?: unknown;
    nativeViewDownloadsButton?: boolean;
  };
  g.nativeViewDownloadsButton = true;
  try {
    useBrowserStore.getState().openUrl("https://example.org/", { newTab: true });
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const seen: { level: string; message: string }[] = [];
    g.nativeViewSeen = seen;
    const event = { kind: "download", tabId, url: "https://example.com/b.zip", name: "b.zip", path: null, size: 3 };
    g.nativeViewListener?.({ payload: { ...event, done: false, success: false, downloadId: null } });
    g.nativeViewListener?.({ payload: { ...event, done: true, success: true, downloadId: "d1", marked: true } });
    g.nativeViewListener?.({ payload: { ...event, done: true, success: false, downloadId: null } });
    g.nativeViewListener?.({ payload: { ...event, done: true, success: true, downloadId: "d2", marked: false } });
    // The unmarked warning still toasts: the button only says the file finished.
    assert.deepEqual(seen, [
      { level: "history", message: "d1" },
      { level: "history", message: "d2" },
      { level: "warning", message: "browser.native.notMarked" },
    ]);
  } finally {
    g.nativeViewDownloadsButton = false;
    stop();
  }
});

test("a native page and its downloads begun beside a temporary chat stay temporary when they land after it", async () => {
  const { useChatRuntimeStore } = await import("@/features/chat");
  const stop = startNativeViews();
  const g = globalThis as {
    nativeViewListener?: (event: { payload: unknown }) => void;
    nativeViewSeen?: unknown;
    nativeViewVisits?: unknown;
    nativeViewApprove?: boolean;
  };
  try {
    useBrowserStore.getState().openUrl("https://example.org/", { newTab: true });
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const visits: { url: string; temporary: boolean }[] = [];
    const seen: { level: string; message: string; temporary?: boolean }[] = [];
    g.nativeViewVisits = visits;
    g.nativeViewSeen = seen;
    g.nativeViewApprove = true;
    const load = (url: string, loading: boolean) =>
      g.nativeViewListener?.({ payload: { kind: "load", tabId, url, loading } });
    const url = "https://example.org/a.zip";
    useChatRuntimeStore.getState().setIncognito(true);
    load("https://example.org/next", true);
    g.nativeViewListener?.({ payload: { kind: "downloadPrompt", tabId, url, site: "", name: "a.zip", id: "p1" } });
    g.nativeViewListener?.({ payload: { kind: "downloadPrompt", tabId, url, site: "", name: "a.zip", id: "p2" } });
    await settle();
    useChatRuntimeStore.getState().setIncognito(false);
    load("https://example.org/next", false);
    const done = { kind: "download", tabId, url, name: "a.zip", path: null, size: 3, done: true, success: true, marked: true };
    g.nativeViewListener?.({ payload: { ...done, downloadId: "d1", promptId: "p1" } });
    g.nativeViewListener?.({ payload: { ...done, downloadId: "d2", promptId: "p2" } });
    g.nativeViewListener?.({ payload: { ...done, downloadId: "d3", promptId: null } });
    load("https://example.org/later", true);
    load("https://example.org/later", false);
    assert.deepEqual(visits, [
      { url: "https://example.org/next", temporary: true },
      { url: "https://example.org/later", temporary: false },
    ]);
    assert.deepEqual(
      seen.filter((item) => item.level === "history"),
      [
        { level: "history", message: "d1", temporary: true },
        { level: "history", message: "d2", temporary: true },
        { level: "history", message: "d3" },
      ],
    );
  } finally {
    useChatRuntimeStore.getState().setIncognito(false);
    delete g.nativeViewApprove;
    stop();
  }
});

test("a native page begun beside a normal chat is kept in history though its tab opened beside a temporary chat", async () => {
  const { useChatRuntimeStore } = await import("@/features/chat");
  const stop = startNativeViews();
  const g = globalThis as { nativeViewListener?: (event: { payload: unknown }) => void; nativeViewVisits?: unknown };
  try {
    useChatRuntimeStore.getState().setIncognito(true);
    useBrowserStore.getState().openUrl("https://example.net/", { newTab: true });
    useChatRuntimeStore.getState().setIncognito(false);
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const visits: { url: string; temporary: boolean }[] = [];
    g.nativeViewVisits = visits;
    const load = (url: string, loading: boolean) =>
      g.nativeViewListener?.({ payload: { kind: "load", tabId, url, loading } });
    // The view's first page is the entry's, even when it starts loading after the chat turned normal.
    load("https://example.net/", true);
    load("https://example.net/", false);
    load("https://example.net/next", true);
    load("https://example.net/next", false);
    assert.deepEqual(visits, [
      { url: "https://example.net/", temporary: true },
      { url: "https://example.net/next", temporary: false },
    ]);
    const seen: { level: string; message: string; temporary?: boolean }[] = [];
    (globalThis as { nativeViewSeen?: unknown }).nativeViewSeen = seen;
    (globalThis as { nativeViewApprove?: boolean }).nativeViewApprove = true;
    const url = "https://example.net/b.zip";
    g.nativeViewListener?.({ payload: { kind: "downloadPrompt", tabId, url, site: "", name: "b.zip", id: "p3" } });
    await settle();
    g.nativeViewListener?.({
      payload: { kind: "download", tabId, url, name: "b.zip", path: null, size: 3, done: true, success: true, marked: true, downloadId: "d5", promptId: "p3" },
    });
    assert.deepEqual(seen.filter((item) => item.level === "history"), [{ level: "history", message: "d5" }]);
  } finally {
    useChatRuntimeStore.getState().setIncognito(false);
    delete (globalThis as { nativeViewApprove?: boolean }).nativeViewApprove;
    stop();
  }
});

test("a native page begun beside a temporary chat keeps it through a redirect and a clear", async () => {
  const { useChatRuntimeStore } = await import("@/features/chat");
  const stop = startNativeViews();
  const g = globalThis as {
    nativeViewListener?: (event: { payload: unknown }) => void;
    nativeViewVisits?: unknown;
    nativeViewsClosed?: () => void;
  };
  try {
    useBrowserStore.getState().openUrl("https://example.edu/", { newTab: true });
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const visits: { url: string; temporary: boolean }[] = [];
    g.nativeViewVisits = visits;
    const load = (url: string, loading: boolean) =>
      g.nativeViewListener?.({ payload: { kind: "load", tabId, url, loading } });
    load("https://example.edu/", true);
    load("https://example.edu/", false);
    useChatRuntimeStore.getState().setIncognito(true);
    load("https://example.edu/go", true);
    useChatRuntimeStore.getState().setIncognito(false);
    // The redirect starts again inside the same navigation.
    load("https://example.edu/landed", true);
    load("https://example.edu/landed", false);
    assert.deepEqual(visits.at(-1), { url: "https://example.edu/landed", temporary: true });
    useChatRuntimeStore.getState().setIncognito(true);
    load("https://example.edu/private", true);
    load("https://example.edu/private", false);
    useChatRuntimeStore.getState().setIncognito(false);
    g.nativeViewsClosed?.();
    await frame();
    const tab = useBrowserStore.getState().tabs.find((item) => item.id === tabId)!;
    const reached = currentEntry(tab);
    assert.ok(reached.kind === "web" && reached.temporary === true);
    load("https://example.edu/private", true);
    load("https://example.edu/private", false);
    assert.deepEqual(visits.at(-1), { url: "https://example.edu/private", temporary: true });
  } finally {
    useChatRuntimeStore.getState().setIncognito(false);
    stop();
  }
});
