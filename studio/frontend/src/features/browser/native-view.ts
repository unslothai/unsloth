// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** desktop web pages use per-tab native views for bot checks; native views cover the DOM, so overlays use snapshots. */

import { useChatRuntimeStore } from "@/features/chat";
import { getLocale, translate } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import type { InterpolationValues } from "@/i18n";
import { openExternalLink } from "@/lib/open-link";
import { toast } from "@/lib/toast";
import { BROWSER_PAGE_INSET_VAR, CHAT_SETTINGS_INSET_VAR } from "@/lib/toast-offset";
import { hostOf } from "./address";
import { approveDownload, downloadSiteOf } from "./download-approval-queue";
import { proxiedFavicon } from "./favicon";
import { useBrowserHistoryStore } from "./history-store";
import { decideNativeDownload } from "./native-downloads";
import { callNative as call, nativeClearing, onNativeViewsClosed } from "./native-support";
import { useBrowserPrefsStore } from "./prefs-store";
import { type BrowserEntry, type BrowserTab, currentEntry, entryKey, useBrowserStore } from "./store";

export { clearNativeBrowsingData, useNativeBrowser } from "./native-support";

const EVENT = "unsloth-browser";
const MAX_VIEWS = 4;
const DOCK_GAP = 8;
// keep the page visible during capture, but hide it without a snapshot after this timeout.
const SNAPSHOT_WAIT_MS = 250;
// Sonner toast width plus edge offsets, reserving a column beside the page.
const TOAST_COLUMN = 380;
// catch layout shifts that do not trigger resize observers.
const RECHECK_MS = 300;
// cap clickless tab creation across all pages, then require user confirmation.
const NEW_TABS_PER_WINDOW = 3;
const NEW_TAB_WINDOW_MS = 60_000;

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
      downloadId: string | null;
      /** False when the file couldn't be marked as downloaded from the internet; null where nothing marks. */
      marked?: boolean | null;
    }
  | { kind: "downloadPrompt"; tabId: string; url: string; site: string; name: string; id: string };

type Bounds = { x: number; y: number; width: number; height: number; viewportWidth: number };

const t = (key: TranslationKey, values?: InterpolationValues) => translate(key, values, getLocale());

const views = new Map<string, number>();
// Tabs this page has opened a view for. A download from any other tab started under the account
// signed in before the last reload (an account switch reloads), so it isn't this account's to list.
const openedTabs = new Set<string>();
let recency: string[] = [];
const zooms = new Map<string, number>();
const icons = new Map<string, string>();
// What each view really shows, to return to after a refused address.
const pages = new Map<string, { url: string; title: string; favicon: string | null }>();
// Where a closed view's page had got to, so it reopens there rather than at the entry's address.
const resume = new Map<string, { entry: number; url: string }>();
let newTabTimes: number[] = [];
// The view on screen, as last shown; menus and dialogs over the page hide it.
let shownView: string | null = null;
const shownWaiters = new Set<() => void>();
// Bumped when the panel unmounts, so a call still in flight leaves the closed views alone.
let generation = 0;

function page(tabId: string) {
  const shown = pages.get(tabId) ?? { url: "", title: "", favicon: null };
  pages.set(tabId, shown);
  return shown;
}

function error(value: unknown): string {
  return value instanceof Error ? value.message : String(value);
}

// The view's history closes with it: keep the page it reached as a tab entry, for Back.
function keepReachedPage(tabId: string): void {
  const store = useBrowserStore.getState();
  const tab = store.tabs.find((candidate) => candidate.id === tabId);
  const shown = pages.get(tabId);
  if (!tab || !shown?.url || currentEntry(tab).kind !== "web" || shown.url === currentEntryUrl(tab)) return;
  store.navigate(tabId, { url: shown.url, temporary: temporaryPages.get(tabId) }, { replace: false });
  store.updateTab(tabId, { title: shown.title, favicon: shown.favicon, loading: false });
}

function closeView(tabId: string): void {
  keepReachedPage(tabId);
  views.delete(tabId);
  viewBounds.delete(tabId);
  zooms.delete(tabId);
  icons.delete(tabId);
  pages.delete(tabId);
  temporaryPages.delete(tabId);
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

// Downloads asked for beside a temporary chat, counted per tab and address: they land later, often after the chat is gone.
const temporaryDownloads = new Map<string, number>();
const downloadKey = (tabId: string, url: string) => `${tabId}\n${url}`;

function takeTemporaryDownload(key: string): boolean {
  const count = temporaryDownloads.get(key) ?? 0;
  if (count > 1) temporaryDownloads.set(key, count - 1);
  else temporaryDownloads.delete(key);
  return count > 0;
}

// Per tab, whether its page began loading beside a temporary chat: in-page navigation makes no new entry.
const temporaryPages = new Map<string, boolean>();

function notePageStart(tabId: string): void {
  temporaryPages.set(tabId, useChatRuntimeStore.getState().incognito);
}

/** Always answered: an unanswered download would sit in staging until the app quits. */
function onDownloadPrompt(event: Extract<NativeEvent, { kind: "downloadPrompt" }>, tab: BrowserTab | undefined): void {
  const { id, url, site, name } = event;
  const entry = tab ? currentEntry(tab) : null;
  const key = downloadKey(event.tabId, url);
  if (useChatRuntimeStore.getState().incognito || (entry?.kind === "web" && entry.temporary)) {
    temporaryDownloads.set(key, (temporaryDownloads.get(key) ?? 0) + 1);
  }
  // The site asking is the page that started it, taken then (a later site's answer must not cover it); blob: counts as its creator. With no web origin yet, the opener or the address asked for.
  const asking = downloadSiteOf(site) ? site : entry?.kind === "web" ? entry.from || entry.url : "";
  const decided = entry?.kind === "web" ? approveDownload(url, name, asking) : Promise.resolve(false);
  void decided
    .then(async (allow) => {
      if (!allow) takeTemporaryDownload(key);
      await decideNativeDownload(id, allow, useBrowserPrefsStore.getState().askWhereToSave);
      if (allow) toast(t("browser.native.downloading", { name }));
    })
    .catch(() => undefined);
}

function onNativeEvent(event: NativeEvent): void {
  const store = useBrowserStore.getState();
  const tab = store.tabs.find((candidate) => candidate.id === event.tabId);
  if (event.kind === "downloadPrompt") {
    onDownloadPrompt(event, tab);
    return;
  }
  // A download outlives its page: it often lands after the tab closed or moved on, and still belongs in history.
  if (event.kind === "download") {
    if (openedTabs.has(event.tabId)) onDownload(event);
    return;
  }
  if (!tab) return;
  const entry = currentEntry(tab);
  if (entry.kind !== "web") return;
  const history = useBrowserHistoryStore.getState();
  if ((event.kind === "load" && event.loading) || event.kind === "url") notePageStart(tab.id);
  const temporary = temporaryPages.get(tab.id) ?? entry.temporary === true;
  switch (event.kind) {
    case "load":
      store.updateTab(tab.id, { loading: event.loading, displayUrl: event.url, ...leftOpenedPage(tab, event.url) });
      page(tab.id).url = event.url;
      remember(tab.id, event.url);
      if (!event.loading) history.recordVisit(event.url, tab.title, temporary);
      break;
    case "title":
      store.updateTab(tab.id, { title: event.title });
      page(tab.id).title = event.title;
      history.recordVisit(tab.displayUrl ?? currentEntryUrl(tab), event.title, temporary);
      break;
    case "url":
      store.updateTab(tab.id, { displayUrl: event.url, ...leftOpenedPage(tab, event.url) });
      page(tab.id).url = event.url;
      remember(tab.id, event.url);
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
    case "newTab": {
      const now = Date.now();
      newTabTimes = newTabTimes.filter((time) => now - time < NEW_TAB_WINDOW_MS);
      if (newTabTimes.length < NEW_TABS_PER_WINDOW) {
        newTabTimes.push(now);
        store.openUrl(event.url, { newTab: true, from: shownUrl(tab) });
      } else {
        prompt(t("browser.native.externalPrompt", { host: hostOf(shownUrl(tab)), url: event.url }), {
          label: t("browser.native.open"),
          onClick: () => store.openUrl(event.url, { newTab: true, from: shownUrl(tab) }),
        });
      }
      break;
    }
    case "external":
      prompt(t("browser.native.externalPrompt", { host: hostOf(shownUrl(tab)), url: event.url }), {
        label: t("browser.native.open"),
        onClick: () => openExternalLink(event.url),
      });
      break;
  }
}

function onDownload(event: Extract<NativeEvent, { kind: "download" }>): void {
  if (!event.done) {
    toast(t("browser.native.downloading", { name: event.name }));
    return;
  }
  const temporary = takeTemporaryDownload(downloadKey(event.tabId, event.url));
  if (event.success) {
    const item = {
      name: event.name,
      url: event.url,
      size: event.size ?? 0,
      contentType: "",
      nativeId: event.downloadId ?? undefined,
    };
    useBrowserHistoryStore.getState().recordDownload(item, temporary);
    if (event.marked === false) toast.warning(t("browser.native.notMarked", { name: event.name }));
    else toast.success(t("browser.native.downloaded", { name: event.name }));
  } else {
    toast.error(t("browser.native.downloadFailed", { name: event.name }));
  }
}

// Pages can ask in a loop: one prompt on screen, replaced at most once a second.
const PROMPT_ID = "browser-native-prompt";
const PROMPT_INTERVAL_MS = 1000;
let lastPrompt = Number.NEGATIVE_INFINITY;

function prompt(message: string, action: { label: string; onClick: () => void }): void {
  const now = Date.now();
  if (now - lastPrompt < PROMPT_INTERVAL_MS) return;
  lastPrompt = now;
  toast(message, { id: PROMPT_ID, action });
}

/** Keyed by the entry the view holds, not the tab's current one, which may not have loaded yet.
 *  Web addresses only: a blob: or data: page goes with the view that made it. */
function remember(tabId: string, url: string): void {
  const entry = views.get(tabId);
  if (entry !== undefined && /^https?:/i.test(url)) resume.set(tabId, { entry, url });
}

function shownUrl(tab: BrowserTab): string {
  return tab.displayUrl ?? currentEntryUrl(tab);
}

/** Once the view moves off the address the tab was opened for, opening that address again
 *  opens it rather than focusing this tab. */
function leftOpenedPage(tab: BrowserTab, url: string): { openKey?: null } {
  return tab.openKey?.startsWith("url:") && tab.openKey !== `url:${url}` ? { openKey: null } : {};
}

function currentEntryUrl(tab: BrowserTab): string {
  const entry = currentEntry(tab);
  return entry.kind === "web" ? entry.url : "";
}

export function nativeAction(tabId: string, action: "back" | "forward" | "reload" | "stop"): void {
  if (!views.has(tabId)) return;
  void call("browser_view_action", { tabId, action }).catch(() => undefined);
}

/** Gives key focus back to the panel's webview. */
export function focusPanel(tabId: string): Promise<void> {
  if (!views.has(tabId)) return Promise.resolve();
  return call("browser_view_action", { tabId, action: "blur" }).catch(() => undefined);
}

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

/** Last shown bounds of `tabId`'s view, in window coordinates. */
export function nativeViewBounds(tabId: string): Bounds | null {
  return viewBounds.get(tabId) ?? null;
}

export async function nativeFind(tabId: string, query: string, backwards: boolean): Promise<boolean> {
  if (!views.has(tabId)) return false;
  return call<boolean>("browser_view_find", { tabId, query, backwards }).catch(() => false);
}

// exclude upward-opening tooltips to avoid hiding the native page on every hover; overlapping toasts still cover it.
// `data-native-cover`: panel UI over the page, e.g. an annotation comment.
const OVERLAY_SELECTOR =
  '[data-radix-popper-content-wrapper], [role="dialog"], [role="alertdialog"], [data-slot$="-overlay"], [data-sonner-toast], [data-native-cover]';

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

function visibleRect(element: HTMLElement): DOMRect | null {
  let rect = element.getBoundingClientRect();
  // A native view isn't clipped by the DOM: trim it to the page area (overflowed while pinned).
  const area = element.closest("[data-browser-page]")?.parentElement?.getBoundingClientRect();
  if (area) {
    const left = Math.max(rect.left, area.left);
    rect = new DOMRect(left, rect.top, Math.min(rect.right, area.right) - left, rect.height);
  }
  if (rect.width < 2 || rect.height < 2) return null;
  let bottom = rect.bottom;
  // `data-native-inset`: panel UI below the page, e.g. the annotate bar.
  for (const dock of document.querySelectorAll<HTMLElement>(
    ".chat-full-view-dock, .chat-full-view-dock-minimized, [data-native-inset]",
  )) {
    const box = dock.getBoundingClientRect();
    if (box.height > 0 && intersects(box, rect)) bottom = Math.min(bottom, box.top - DOCK_GAP);
  }
  if (bottom - rect.top < 2) return null;
  return new DOMRect(rect.left, rect.top, rect.width, bottom - rect.top);
}

type Desired =
  | { tabId: string; url: string; entry: number; zoom: number; bounds: Bounds }
  | { tabId: string; covered: true }
  | null;

function placeholder(tabId: string): HTMLElement | null {
  return document.querySelector<HTMLElement>(`[data-native-page="${CSS.escape(tabId)}"]`);
}

let toastInset: string | null = null;

// native views cover DOM toasts, so reserve a left column when the page reaches the right edge or Run settings.
function insetToasts(rect: DOMRect | null): void {
  const style = document.documentElement.style;
  // read the inline value set by watchChatSettingsInset; it is absent when the panel is closed or the row is narrow.
  const settings = Number.parseFloat(style.getPropertyValue(CHAT_SETTINGS_INSET_VAR)) || 0;
  const room = rect && rect.right >= window.innerWidth - settings - 2 && rect.left >= TOAST_COLUMN;
  const next = room ? `${Math.round(window.innerWidth - rect.left)}px` : null;
  if (next === toastInset) return;
  toastInset = next;
  if (next) style.setProperty(BROWSER_PAGE_INSET_VAR, next);
  else style.removeProperty(BROWSER_PAGE_INSET_VAR);
}

function desiredView(): Desired {
  const page = pageRect();
  insetToasts(page?.rect ?? null);
  if (!page) return null;
  const { tab, entry, rect } = page;
  if (covered(rect)) return { tabId: tab.id, covered: true };
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

function pageRect(): { tab: BrowserTab; entry: Extract<BrowserEntry, { kind: "web" }>; rect: DOMRect } | null {
  const state = useBrowserStore.getState();
  if (!state.open) return null;
  const tab = state.tabs.find((candidate) => candidate.id === state.activeTabId);
  if (!tab || tab.nativeError) return null;
  const entry = currentEntry(tab);
  if (entry.kind !== "web") return null;
  const element = placeholder(tab.id);
  if (!element || element.offsetParent === null) return null;
  const rect = visibleRect(element);
  return rect ? { tab, entry, rect } : null;
}

function pruneViews(shown: string | null): void {
  const tabs = new Map(useBrowserStore.getState().tabs.map((tab) => [tab.id, tab]));
  for (const tabId of [...views.keys()]) {
    const tab = tabs.get(tabId);
    if (!tab || currentEntry(tab).kind !== "web") {
      closeView(tabId);
      resume.delete(tabId);
    }
  }
  while (views.size > MAX_VIEWS) {
    const oldest = recency.find((id) => id !== shown && views.has(id));
    if (!oldest) break;
    closeView(oldest);
  }
}

function setShownView(tabId: string | null): void {
  shownView = tabId;
  for (const wake of [...shownWaiters]) wake();
}

/** Resolves true once `tabId`'s view is on screen (a menu over it has closed), false on timeout. */
export function whenNativeViewShown(tabId: string, timeoutMs = 1500): Promise<boolean> {
  if (shownView === tabId) return Promise.resolve(true);
  return new Promise((resolve) => {
    const finish = (shown: boolean) => {
      shownWaiters.delete(wake);
      clearTimeout(timer);
      resolve(shown);
    };
    const wake = () => shownView === tabId && finish(true);
    const timer = setTimeout(() => finish(false), timeoutMs);
    shownWaiters.add(wake);
  });
}

// keep the last frame on the placeholder so overlays do not leave the panel blank.
let snapshot: { tabId: string; element: HTMLElement; url: string } | null = null;
let shownBounds: Bounds | null = null;
const viewBounds = new Map<string, Bounds>();

function clearSnapshot(): void {
  if (!snapshot) return;
  const { style } = snapshot.element;
  for (const property of ["background-image", "background-position", "background-size", "background-repeat"]) {
    style.removeProperty(property);
  }
  URL.revokeObjectURL(snapshot.url);
  snapshot = null;
}

async function paintSnapshot(tabId: string): Promise<void> {
  const bounds = shownBounds;
  const element = placeholder(tabId);
  if (shownView !== tabId || !bounds || !element) return;
  const started = generation;
  const png = await Promise.race([
    call<ArrayBuffer>("browser_capture", { tabId }).catch(() => null),
    new Promise<null>((resolve) => setTimeout(() => resolve(null), SNAPSHOT_WAIT_MS)),
  ]);
  if (!png?.byteLength || !element.isConnected || started !== generation) return;
  const url = URL.createObjectURL(new Blob([png], { type: "image/png" }));
  // align snapshots to the visible native bounds because the chat dock can shorten the view.
  const box = element.getBoundingClientRect();
  element.style.backgroundImage = `url(${url})`;
  element.style.backgroundPosition = `${bounds.x - box.left}px ${bounds.y - box.top}px`;
  element.style.backgroundSize = `${bounds.width}px ${bounds.height}px`;
  element.style.backgroundRepeat = "no-repeat";
  snapshot = { tabId, element, url };
}

async function applyView(desired: Desired): Promise<void> {
  if (!desired || "covered" in desired) {
    // keep the snapshot if an overlay reopens during capture.
    if (snapshot?.tabId !== desired?.tabId) clearSnapshot();
    if (desired) await paintSnapshot(desired.tabId);
    setShownView(null);
    await call("browser_view_show", { tabId: null });
    return;
  }
  const { tabId, url, entry, zoom, bounds } = desired;
  const existed = views.has(tabId);
  const loaded = views.get(tabId);
  const resumed = resume.get(tabId);
  const started = generation;
  const stale = () => {
    if (started === generation) return false;
    void call("browser_view_close", { tabId }).catch(() => undefined);
    return true;
  };
  try {
    openedTabs.add(tabId);
    await call("browser_view_show", { tabId, url: resumed?.entry === entry ? resumed.url : url, bounds });
    if (stale()) return;
    clearSnapshot();
    shownBounds = bounds;
    viewBounds.set(tabId, bounds);
    setShownView(tabId);
    // record the entry after navigation succeeds so failures remain retryable.
    if (existed && loaded !== entry) {
      await call("browser_view_navigate", { tabId, url });
      if (stale()) return;
    }
    views.set(tabId, entry);
    if ((zooms.get(tabId) ?? 1) !== zoom) {
      zooms.set(tabId, zoom);
      await call("browser_view_zoom", { tabId, zoom });
      if (stale()) return;
    }
  } catch (cause) {
    if (stale()) return;
    useBrowserStore.getState().updateTab(tabId, {
      loading: false,
      nativeError: /public web/i.test(error(cause)) ? t("browser.native.blocked") : error(cause),
    });
  }
  recency = [...recency.filter((id) => id !== tabId), tabId];
}

// One call in flight, across mounts; meanwhile only the newest state waits, so a drag can't queue
// a backlog of stale bounds for the native view to replay.
let running = false;
let pending: { desired: Desired } | null = null;

function pump(): void {
  if (running || !pending || nativeClearing()) return;
  const { desired } = pending;
  pending = null;
  running = true;
  void applyView(desired)
    .catch(() => undefined)
    .finally(() => {
      running = false;
      pump();
    });
}

function apply(desired: Desired): void {
  pending = { desired };
  pump();
}

// A clear closed every page: forget them, keeping where each tab got to, and show them again.
let epoch = 0;
onNativeViewsClosed(() => {
  for (const tabId of [...views.keys()]) keepReachedPage(tabId);
  setShownView(null);
  views.clear();
  viewBounds.clear();
  zooms.clear();
  icons.clear();
  pages.clear();
  resume.clear();
  recency = [];
  epoch += 1;
  pump();
});

export function startNativeViews(): () => void {
  listenOnce();
  let frame = 0;
  let sent = "";
  let resized: HTMLElement | null = null;
  const resizeObserver = new ResizeObserver(() => schedule());

  const sync = () => {
    frame = 0;
    const desired = desiredView();
    pruneViews(desired?.tabId ?? null);
    const element = desired ? placeholder(desired.tabId) : null;
    if (element !== resized) {
      if (resized) resizeObserver.unobserve(resized);
      if (element) resizeObserver.observe(element);
      resized = element;
    }
    const key = JSON.stringify([epoch, desired]);
    if (key === sent) return;
    sent = key;
    apply(desired);
  };
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(sync);
  };

  const unsubscribe = useBrowserStore.subscribe(schedule);
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
    // close pages because hidden native views keep scripts and media running.
    generation += 1;
    pending = null;
    setShownView(null);
    clearSnapshot();
    insetToasts(null);
    for (const tabId of [...views.keys()]) closeView(tabId);
  };
}
