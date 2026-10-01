// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Desktop app web pages: a real browser view per tab (src-tauri/src/browser_webview.rs) laid over
 * the tab's placeholder, so bot checks and web apps work. It sits above the DOM, so it hides while
 * a menu or dialog covers it.
 */

import { getLocale, translate } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import type { InterpolationValues } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { openExternalLink } from "@/lib/open-link";
import { toast } from "@/lib/toast";
import { hostOf } from "./address";
import { proxiedFavicon } from "./favicon";
import { useBrowserHistoryStore } from "./history-store";
import { type BrowserTab, currentEntry, entryKey, setNativeWebHistory, useBrowserStore } from "./store";

/** Native views in the desktop app; the proxy otherwise. */
export const nativeBrowser = isTauri;

if (nativeBrowser) setNativeWebHistory(true);

const EVENT = "unsloth-browser";
// Live views, hidden ones included.
const MAX_VIEWS = 4;
// Gap between a full-view page and the chat floating over it.
const DOCK_GAP = 8;
// Catches moves that resize nothing.
const RECHECK_MS = 300;

type NativeEvent =
  | { kind: "load"; tabId: string; url: string; loading: boolean }
  | { kind: "title"; tabId: string; title: string }
  | { kind: "url"; tabId: string; url: string }
  | { kind: "history"; tabId: string; canGoBack: boolean; canGoForward: boolean; icon: string | null }
  | { kind: "newTab"; tabId: string; url: string }
  | { kind: "external"; tabId: string; url: string }
  | {
      kind: "download";
      tabId: string;
      url: string;
      name: string;
      path: string | null;
      size: number | null;
      done: boolean;
      success: boolean;
    };

type Bounds = { x: number; y: number; width: number; height: number; viewportWidth: number };

const t = (key: TranslationKey, values?: InterpolationValues) => translate(key, values, getLocale());

async function call<T = void>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

// Tab id to the history entry its view last loaded.
const views = new Map<string, number>();
// Tab ids, most recently shown last.
let recency: string[] = [];
const zooms = new Map<string, number>();
const icons = new Map<string, string>();
// What each view really shows, to return to after a refused address.
const pages = new Map<string, { url: string; title: string; favicon: string | null }>();

function page(tabId: string) {
  const shown = pages.get(tabId) ?? { url: "", title: "", favicon: null };
  pages.set(tabId, shown);
  return shown;
}

function error(value: unknown): string {
  return value instanceof Error ? value.message : String(value);
}

function closeView(tabId: string): void {
  views.delete(tabId);
  zooms.delete(tabId);
  icons.delete(tabId);
  pages.delete(tabId);
  recency = recency.filter((id) => id !== tabId);
  void call("browser_view_close", { tabId }).catch(() => undefined);
}

let listening = false;

function listenOnce(): void {
  if (listening) return;
  listening = true;
  void import("@tauri-apps/api/event").then(({ listen }) =>
    listen<NativeEvent>(EVENT, (event) => onNativeEvent(event.payload)),
  );
}

function onNativeEvent(event: NativeEvent): void {
  const store = useBrowserStore.getState();
  const tab = store.tabs.find((candidate) => candidate.id === event.tabId);
  if (!tab || currentEntry(tab).kind !== "web") return;
  const history = useBrowserHistoryStore.getState();
  switch (event.kind) {
    case "load":
      store.updateTab(tab.id, { loading: event.loading, displayUrl: event.url });
      page(tab.id).url = event.url;
      if (!event.loading) history.recordVisit(event.url, tab.title);
      break;
    case "title":
      store.updateTab(tab.id, { title: event.title });
      page(tab.id).title = event.title;
      history.recordVisit(tab.displayUrl ?? currentEntryUrl(tab), event.title);
      break;
    case "url":
      store.updateTab(tab.id, { displayUrl: event.url });
      page(tab.id).url = event.url;
      break;
    case "history": {
      const next = { back: event.canGoBack, forward: event.canGoForward };
      if (tab.nativeHistory?.back !== next.back || tab.nativeHistory?.forward !== next.forward) {
        store.updateTab(tab.id, { nativeHistory: next });
      }
      // Through the backend, so a page can't point Studio at a local address.
      if (event.icon && icons.get(tab.id) !== event.icon) {
        const icon = event.icon;
        icons.set(tab.id, icon);
        void proxiedFavicon(icon).then((favicon) => {
          if (!favicon || icons.get(tab.id) !== icon) return;
          page(tab.id).favicon = favicon;
          useBrowserStore.getState().updateTab(tab.id, { favicon });
        });
      }
      break;
    }
    case "newTab":
      store.openUrl(event.url, { newTab: true });
      break;
    case "external":
      // Pages can ask without a click, so the user decides.
      toast(t("browser.native.externalPrompt", { host: hostOf(currentEntryUrl(tab)), url: event.url }), {
        action: { label: t("browser.native.open"), onClick: () => openExternalLink(event.url) },
      });
      break;
    case "download":
      if (!event.done) {
        toast(t("browser.native.downloading", { name: event.name }));
      } else if (event.success) {
        history.recordDownload({ name: event.name, url: event.url, size: event.size ?? 0, contentType: "" });
        toast.success(t("browser.native.downloaded", { name: event.name }));
      } else {
        toast.error(t("browser.native.downloadFailed", { name: event.name }));
      }
      break;
  }
}

function currentEntryUrl(tab: BrowserTab): string {
  const entry = currentEntry(tab);
  return entry.kind === "web" ? entry.url : "";
}

/** Back, forward, reload or stop a tab's native page. */
export function nativeAction(tabId: string, action: "back" | "forward" | "reload" | "stop"): void {
  if (!views.has(tabId)) return;
  void call("browser_view_action", { tabId, action }).catch(() => undefined);
}

/** Back from a refused address to the page still shown, without a reload. */
export function returnToNativePage(tabId: string): boolean {
  const shown = pages.get(tabId);
  const store = useBrowserStore.getState();
  const tab = store.tabs.find((candidate) => candidate.id === tabId);
  if (!shown?.url || !tab?.nativeError || !views.has(tabId)) return false;
  store.navigate(tabId, { url: shown.url }, { replace: true });
  const back = useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId);
  if (back) views.set(tabId, entryKey(currentEntry(back)));
  store.updateTab(tabId, { title: shown.title, favicon: shown.favicon, loading: false, displayUrl: shown.url });
  return true;
}

export function hasNativeView(tabId: string): boolean {
  return views.has(tabId);
}

/** Find in a tab's native page; whether it matched. */
export async function nativeFind(tabId: string, query: string, backwards: boolean): Promise<boolean> {
  if (!views.has(tabId)) return false;
  return call<boolean>("browser_view_find", { tabId, query, backwards }).catch(() => false);
}

/** Clear the native pages' own cookies, storage and cache. */
export function clearNativeBrowsingData(): void {
  if (!nativeBrowser) return;
  void call("browser_view_clear_data").catch(() => undefined);
}

// Studio UI that covers the panel. Not tooltips, or every hover would blank the page.
const OVERLAY_SELECTOR =
  '[data-radix-popper-content-wrapper], [role="dialog"], [role="alertdialog"], [data-slot$="-overlay"]';

function intersects(a: DOMRect, b: DOMRect): boolean {
  return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
}

function covered(rect: DOMRect): boolean {
  for (const element of document.querySelectorAll<HTMLElement>(OVERLAY_SELECTOR)) {
    if (element.closest("[data-native-page]")) continue;
    if (element.querySelector('[role="tooltip"]')) continue;
    const box = element.getBoundingClientRect();
    if (box.width > 0 && box.height > 0 && intersects(box, rect)) return true;
  }
  return false;
}

/** The page's rect, short of the chat floating over a full-view browser. */
function visibleRect(element: HTMLElement): DOMRect | null {
  const rect = element.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return null;
  let bottom = rect.bottom;
  for (const dock of document.querySelectorAll<HTMLElement>(".chat-full-view-dock, .chat-full-view-dock-minimized")) {
    const box = dock.getBoundingClientRect();
    if (box.height > 0 && intersects(box, rect)) bottom = Math.min(bottom, box.top - DOCK_GAP);
  }
  if (bottom - rect.top < 2) return null;
  return new DOMRect(rect.left, rect.top, rect.width, bottom - rect.top);
}

type Desired = { tabId: string; url: string; entry: number; zoom: number; bounds: Bounds } | null;

function desiredView(): Desired {
  const state = useBrowserStore.getState();
  if (!state.open) return null;
  const tab = state.tabs.find((candidate) => candidate.id === state.activeTabId);
  if (!tab || tab.nativeError) return null;
  const entry = currentEntry(tab);
  if (entry.kind !== "web") return null;
  const element = document.querySelector<HTMLElement>(`[data-native-page="${CSS.escape(tab.id)}"]`);
  if (!element || element.offsetParent === null) return null;
  const rect = visibleRect(element);
  if (!rect || covered(rect)) return null;
  return {
    tabId: tab.id,
    url: entry.url,
    entry: entryKey(entry),
    zoom: tab.zoom,
    bounds: {
      x: Math.round(rect.left),
      y: Math.round(rect.top),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      viewportWidth: window.innerWidth,
    },
  };
}

/** Close views of closed or non-web tabs, and the least recent past the cap. */
function pruneViews(shown: string | null): void {
  const tabs = new Map(useBrowserStore.getState().tabs.map((tab) => [tab.id, tab]));
  for (const tabId of [...views.keys()]) {
    const tab = tabs.get(tabId);
    if (!tab || currentEntry(tab).kind !== "web") closeView(tabId);
  }
  while (views.size > MAX_VIEWS) {
    const oldest = recency.find((id) => id !== shown && views.has(id));
    if (!oldest) break;
    closeView(oldest);
  }
}

async function applyView(desired: Desired): Promise<void> {
  if (!desired) {
    await call("browser_view_show", { tabId: null });
    return;
  }
  const { tabId, url, entry, zoom, bounds } = desired;
  const existed = views.has(tabId);
  const loaded = views.get(tabId);
  try {
    await call("browser_view_show", { tabId, url, bounds });
    views.set(tabId, entry);
    // A new address for an existing view.
    if (existed && loaded !== entry) await call("browser_view_navigate", { tabId, url });
    if ((zooms.get(tabId) ?? 1) !== zoom) {
      zooms.set(tabId, zoom);
      await call("browser_view_zoom", { tabId, zoom });
    }
  } catch (cause) {
    useBrowserStore.getState().updateTab(tabId, {
      loading: false,
      nativeError: /public web/i.test(error(cause)) ? t("browser.native.blocked") : error(cause),
    });
  }
  recency = [...recency.filter((id) => id !== tabId), tabId];
}

/** Keeps the active tab's view over its placeholder, or hidden. Mounted with the panel. */
export function startNativeViews(): () => void {
  listenOnce();
  let frame = 0;
  let sent = "";
  let queue: Promise<void> = Promise.resolve();
  let resized: HTMLElement | null = null;
  const resizeObserver = new ResizeObserver(() => schedule());

  const sync = () => {
    frame = 0;
    const desired = desiredView();
    pruneViews(desired?.tabId ?? null);
    const element = desired
      ? document.querySelector<HTMLElement>(`[data-native-page="${CSS.escape(desired.tabId)}"]`)
      : null;
    if (element !== resized) {
      if (resized) resizeObserver.unobserve(resized);
      if (element) resizeObserver.observe(element);
      resized = element;
    }
    const key = JSON.stringify(desired);
    if (key === sent) return;
    sent = key;
    // In order: a hide must not land after the show that followed it.
    queue = queue.then(() => applyView(desired)).catch(() => undefined);
  };
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(sync);
  };

  const unsubscribe = useBrowserStore.subscribe(schedule);
  // Menus and dialogs mount in portals on <body>.
  const overlays = new MutationObserver(schedule);
  overlays.observe(document.body, { childList: true });
  window.addEventListener("resize", schedule);
  const interval = window.setInterval(schedule, RECHECK_MS);
  schedule();

  return () => {
    cancelAnimationFrame(frame);
    unsubscribe();
    overlays.disconnect();
    resizeObserver.disconnect();
    window.removeEventListener("resize", schedule);
    window.clearInterval(interval);
    queue = queue.then(() => applyView(null)).catch(() => undefined);
  };
}
