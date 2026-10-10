// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** desktop web pages use per-tab native views for bot checks. macOS puts each page under the app's webview, so
 *  overlays draw over the live page; elsewhere native views cover the DOM, so overlays use snapshots. */

import { useChatRuntimeStore } from "@/features/chat";
import { getLocale, translate } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import type { InterpolationValues } from "@/i18n";
import { openExternalLink } from "@/lib/open-link";
import { toast } from "@/lib/toast";
import { BROWSER_PAGE_INSET_VAR, CHAT_SETTINGS_INSET_VAR } from "@/lib/toast-offset";
import { hostOf } from "./address";
import { approveChosenDownload, approveDownload, downloadSiteOf } from "./download-approval-queue";
import { abandonDownload, beginDownload, finishDownload, useDownloadActivity } from "./download-activity";
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
      /** The downloadPrompt `id` it was asked under; null when refused before asking. */
      promptId?: string | null;
    }
  | { kind: "downloadPrompt"; tabId: string; url: string; site: string; name: string; id: string; saveAs: boolean }
  | { kind: "downloadCancelled"; tabId: string; url: string; promptId: string };

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
  pageEntries.delete(tabId);
  loadingPages.delete(tabId);
  recency = recency.filter((id) => id !== tabId);
  if (parkedView === tabId) parkedView = null;
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

// Prompts asked for beside a temporary chat: their downloads land later, often after the chat is gone.
const temporaryDownloads = new Set<string>();

// Per tab, whether its page began loading beside a temporary chat: in-page navigation makes no new entry.
// A page's first load is its entry's, which may start long after the entry was made (a background tab).
const temporaryPages = new Map<string, boolean>();
const pageEntries = new Map<string, BrowserEntry>();
// Tabs mid-load: a redirect starts again within the same navigation, which keeps its state.
const loadingPages = new Set<string>();

function notePageStart(tabId: string, entry: Extract<BrowserEntry, { kind: "web" }>): void {
  const fresh = pageEntries.get(tabId) !== entry;
  pageEntries.set(tabId, entry);
  const kept = fresh ? entry.temporary === true : loadingPages.has(tabId) && temporaryPages.get(tabId) === true;
  temporaryPages.set(tabId, useChatRuntimeStore.getState().incognito || kept);
}

function pageTemporary(tabId: string, entry: BrowserEntry): boolean {
  return temporaryPages.get(tabId) ?? (entry.kind === "web" && entry.temporary === true);
}

/** One running download in the Downloads button, from approval until it ends. */
const downloadKey = (promptId: string) => `native:${promptId}`;

/** Always answered: an unanswered download would sit in staging until the app quits. */
function onDownloadPrompt(event: Extract<NativeEvent, { kind: "downloadPrompt" }>, tab: BrowserTab | undefined): void {
  const { id, url, site, name, saveAs } = event;
  const entry = tab ? currentEntry(tab) : null;
  if (useChatRuntimeStore.getState().incognito || (tab && entry && pageTemporary(tab.id, entry))) temporaryDownloads.add(id);
  // The site asking is the page that started it, taken then (a later site's answer must not cover it); blob: counts as its creator. With no web origin yet, the opener or the address asked for.
  const asking = downloadSiteOf(site) ? site : entry?.kind === "web" ? entry.from || entry.url : "";
  // Picked from the context menu: the save dialog is the prompt, whatever the site's remembered answer.
  const decided =
    entry?.kind !== "web"
      ? Promise.resolve(false)
      : saveAs
        ? approveChosenDownload(url, name)
        : approveDownload(url, name, asking);
  const key = downloadKey(id);
  void decided
    .then(async (allow) => {
      if (!allow) temporaryDownloads.delete(id);
      // Begun before deciding: a file that finished while the prompt was open lands at once.
      if (allow) beginDownload(key, name);
      await decideNativeDownload(id, allow, saveAs || useBrowserPrefsStore.getState().askWhereToSave);
      // Still running: one that landed during the decide call has already said so.
      const { active, buttons } = useDownloadActivity.getState();
      if (allow && buttons === 0 && key in active) toast(t("browser.native.downloading", { name }));
    })
    .catch(() => {
      temporaryDownloads.delete(id);
      abandonDownload(key);
    });
}


function onNativeEvent(event: NativeEvent): void {
  const store = useBrowserStore.getState();
  const tab = store.tabs.find((candidate) => candidate.id === event.tabId);
  if (event.kind === "downloadPrompt") {
    onDownloadPrompt(event, tab);
    return;
  }
  if (event.kind === "downloadCancelled") {
    temporaryDownloads.delete(event.promptId);
    abandonDownload(downloadKey(event.promptId));
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
  if ((event.kind === "load" && event.loading) || event.kind === "url") notePageStart(tab.id, entry);
  if (event.kind === "load") {
    if (event.loading) loadingPages.add(tab.id);
    else loadingPages.delete(tab.id);
  }
  const temporary = pageTemporary(tab.id, entry);
  switch (event.kind) {
    case "load":
      store.updateTab(tab.id, { loading: event.loading, displayUrl: event.url, ...leftOpenedPage(tab, event.url) });
      page(tab.id).url = event.url;
      remember(tab.id, event.url);
      if (!event.loading) history.recordVisit(event.url, tab.title, temporary);
      // A page that loads while parked would otherwise show its first (blank) frame.
      if (!event.loading) refreshCoveredPage(tab.id);
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

/** Shown on the toolbar's Downloads button; toasts only when none is on screen. */
function onDownload(event: Extract<NativeEvent, { kind: "download" }>): void {
  // Refused before asking, it never ran: there's nothing to end, only a result to show.
  const key = event.promptId ? downloadKey(event.promptId) : `native:${event.tabId}:${event.url}`;
  if (!event.done) return;
  const temporary = event.promptId ? temporaryDownloads.delete(event.promptId) : false;
  const item = { name: event.name, url: event.url, size: event.size ?? 0, contentType: "", nativeId: event.downloadId ?? undefined };
  const historyId = event.success ? useBrowserHistoryStore.getState().recordDownload(item, temporary) : undefined;
  const shown = finishDownload(key, {
    name: event.name,
    size: event.size ?? 0,
    contentType: "",
    url: event.url,
    nativeId: event.downloadId ?? undefined,
    historyId,
    failed: !event.success,
  });
  if (event.success && event.marked === false) toast.warning(t("browser.native.notMarked", { name: event.name }));
  else if (!shown && event.success) toast.success(t("browser.native.downloaded", { name: event.name }));
  else if (!shown) toast.error(t("browser.native.downloadFailed", { name: event.name }));
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

/** Gives key focus to the page, e.g. as the find bar it was lent to closes. */
export function focusPage(tabId: string): void {
  if (!views.has(tabId)) return;
  void call("browser_view_action", { tabId, action: "focus" }).catch(() => undefined);
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

// Snapshot mode: any of these over the page covers it.
// `data-native-cover`: panel UI over the page, e.g. an annotation comment.
const OVERLAY_SELECTOR =
  '[data-radix-popper-content-wrapper], [role="dialog"], [role="alertdialog"], [data-slot$="-overlay"], [data-sonner-toast], [data-native-cover], .find-bar-surface';
// Layered mode: menus and dialogs take all page input, so an outside click closes them.
const BLOCKING_SELECTOR =
  '[data-radix-popper-content-wrapper], [role="dialog"], [role="alertdialog"], [data-slot$="-overlay"], [data-native-cover]';
const MENU_SELECTOR = "[data-radix-popper-content-wrapper]";
// Layered mode: these take input only within their own rect. `data-native-clickable`: lasting panels
// (monitors); `data-native-rail`: a rail of lasting cards (update, downloads), each its own rect. Both
// stay out of snapshot mode, which would freeze the page while they show.
const CLICKABLE_SELECTOR =
  "[data-sonner-toast], .find-bar-surface, [data-native-clickable], [data-native-rail] > *";

// Whether pages sit under the app's webview (macOS); null until the backend answers.
let layered: boolean | null = null;
let layeredAsked: Promise<boolean> | null = null;

function askLayered(): Promise<boolean> {
  layeredAsked ??= call<boolean>("browser_view_layered")
    .catch(() => false)
    .then((answer) => (layered = answer === true));
  return layeredAsked;
}

function intersects(a: DOMRect, b: DOMRect): boolean {
  return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
}

const TOOLTIP = '[role="tooltip"]';
// Hover cards close on their own once the pointer leaves, like tooltips.
const HOVER_ONLY = '[role="tooltip"], [data-slot="hover-card-content"]';

/** Boxes of `selector` over `rect` (anywhere when null), minus those holding `skip`. */
function overlays(selector: string, rect: DOMRect | null, skip: string | null = TOOLTIP): DOMRect[] {
  const boxes: DOMRect[] = [];
  for (const element of document.querySelectorAll<HTMLElement>(selector)) {
    if (element.closest("[data-native-page]")) continue;
    if (skip && element.querySelector(skip)) continue;
    const box = element.getBoundingClientRect();
    if (box.width > 0 && box.height > 0 && (!rect || intersects(box, rect))) boxes.push(box);
  }
  return boxes;
}

function covering(rect: DOMRect): "none" | "tooltip" | "other" {
  if (overlays(OVERLAY_SELECTOR, rect).length > 0) return "other";
  return overlays(OVERLAY_SELECTOR, rect, null).length > 0 ? "tooltip" : "none";
}

// A page covered only by a tooltip stays parked this long after it closes (or until the pointer
// reaches the page), so hovering along the toolbar captures once, not per button.
const TOOLTIP_HOLD_MS = 250;
let tooltipHold = 0;

function coveredNow(rect: DOMRect): boolean {
  const cover = covering(rect);
  if (cover === "tooltip") tooltipHold = performance.now() + TOOLTIP_HOLD_MS;
  else if (cover === "other") tooltipHold = 0;
  return cover !== "none" || performance.now() < tooltipHold;
}

type Input = { blocked: boolean; exclude: Bounds[] };

const NO_INPUT: Input = { blocked: false, exclude: [] };

function panelInput(rect: DOMRect): Input {
  // A menu away from the page owns it too, so a click on the page closes the menu, as anywhere else.
  if (overlays(BLOCKING_SELECTOR, rect).length > 0 || overlays(MENU_SELECTOR, null, HOVER_ONLY).length > 0) {
    return { blocked: true, exclude: [] };
  }
  const exclude = overlays(CLICKABLE_SELECTOR, rect).map((box) => ({
    x: Math.floor(box.left),
    y: Math.floor(box.top),
    width: Math.ceil(box.width) + 1,
    height: Math.ceil(box.height) + 1,
    viewportWidth: window.innerWidth,
  }));
  return { blocked: false, exclude };
}

// Empty, so the first sync after a reload clears a menu's block the backend kept.
let sentInput = "";

// Sent directly, not queued, so a menu owns the page as soon as it opens.
function sendInput(input: Input): void {
  const key = JSON.stringify(input);
  if (key === sentInput) return;
  const wasBlocked = sentInput !== "" && (JSON.parse(sentInput) as Input).blocked;
  sentInput = key;
  void call("browser_view_input", input).catch(() => undefined);
  if (input.blocked !== wasBlocked) lendKeys(input.blocked);
}

// A menu or dialog over a page that holds the keys (one the site raised, or a chord) takes them, so
// Escape and Enter reach it; they go back when it closes unless focus moved on meanwhile.
let lentFrom: { tabId: string; active: Element | null } | null = null;

function lendKeys(blocked: boolean): void {
  if (blocked) {
    if (!shownView || document.hasFocus()) return;
    lentFrom = { tabId: shownView, active: document.activeElement };
    void focusPanel(shownView);
    return;
  }
  const lent = lentFrom;
  lentFrom = null;
  if (!lent) return;
  requestAnimationFrame(() => {
    const active = document.activeElement;
    if (active === null || active === document.body || active === lent.active) focusPage(lent.tabId);
  });
}

// Backgrounds around the page skip its rect so the page shows through.
const HOLE_ATTRIBUTE = "data-native-hole";
const HOLE_VARS = ["--native-hole-x", "--native-hole-y", "--native-hole-w", "--native-hole-h"] as const;
let holed: HTMLElement[] = [];
let holeSignature = "";
let holeTab: string | null = null;
let holeBounds: Bounds | null = null;

function opaque(color: string): boolean {
  if (color === "transparent") return false;
  const alpha = /\/\s*([\d.]+)%?\s*\)$/.exec(color) ?? /^rgba\([^)]*,\s*([\d.]+)\s*\)$/.exec(color);
  return !alpha || Number.parseFloat(alpha[1]) > 0;
}

function openHole(tabId: string, bounds: Bounds): void {
  const moved = JSON.stringify(bounds) !== JSON.stringify(holeBounds);
  holeTab = tabId;
  holeBounds = bounds;
  if (!markHole() && moved) placeHole();
}

// Set on each holed node, not the root: the vars don't inherit (index.css), so a move restyles
// only those nodes instead of the whole document.
function placeHole(): void {
  if (!holeBounds) return;
  const values = [holeBounds.x, holeBounds.y, holeBounds.width, holeBounds.height];
  for (const node of holed) HOLE_VARS.forEach((name, index) => node.style.setProperty(name, `${values[index]}px`));
}

/** Re-reads background colors when the chain, its classes or the theme change; true if it did. */
function markHole(): boolean {
  const element = holeTab ? placeholder(holeTab) : null;
  if (!element) return false;
  const chain: HTMLElement[] = [];
  for (let node: HTMLElement | null = element; node; node = node.parentElement) chain.push(node);
  // Theme: html classes, palette and inline color vars, minus our own.
  const root = document.documentElement;
  const theme = (root.style.cssText ?? "").replace(/--native-hole-[^;]*;?/g, "");
  const signature = `${root.dataset.palette}|${theme}|${chain.map((node) => node.className).join("|")}`;
  if (signature === holeSignature && holed.every((node) => node.isConnected)) return false;
  unmarkHole();
  holeSignature = signature;
  const colors = chain.map((node) => getComputedStyle(node));
  chain.forEach((node, index) => {
    const computed = colors[index];
    if (computed.backgroundImage === "none" && !opaque(computed.backgroundColor)) return;
    node.style.setProperty("--native-hole-bg", computed.backgroundColor);
    holed.push(node);
  });
  placeHole();
  for (const node of holed) node.setAttribute(HOLE_ATTRIBUTE, "");
  // Mid theme switch everything can read transparent: retry next sync.
  if (holed.length === 0) holeSignature = "";
  return true;
}

function unmarkHole(): void {
  for (const node of holed) {
    node.removeAttribute(HOLE_ATTRIBUTE);
    for (const name of ["--native-hole-bg", ...HOLE_VARS]) node.style.removeProperty(name);
  }
  holed = [];
  holeSignature = "";
}

function closeHole(): void {
  holeTab = null;
  holeBounds = null;
  unmarkHole();
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

type Shown = { tabId: string; url: string; entry: number; zoom: number; bounds: Bounds };
type Desired = Shown | (Shown & { covered: true }) | null;

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
  // Layered mode needs no toast column.
  if (layered) sendInput(page ? panelInput(page.rect) : NO_INPUT);
  else insetToasts(page?.rect ?? null);
  if (!page) return null;
  const { tab, entry, rect } = page;
  const shown: Shown = {
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
  return !layered && coveredNow(rect) ? { ...shown, covered: true } : shown;
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

let paintToken = 0;

/** Paints a fresh snapshot; true when the placeholder shows one for `tabId` (a fresh one if `fresh`).
 *  Only the newest call paints. */
async function paintSnapshot(tabId: string, fresh = false): Promise<boolean> {
  const shows = () => !fresh && snapshot?.tabId === tabId;
  const bounds = shownBounds;
  const element = placeholder(tabId);
  if ((shownView !== tabId && parkedView !== tabId) || !bounds || !element) return shows();
  const started = generation;
  const token = ++paintToken;
  const current = () => element.isConnected && started === generation && token === paintToken;
  const png = await Promise.race([
    call<ArrayBuffer>("browser_capture", { tabId }).catch(() => null),
    new Promise<null>((resolve) => setTimeout(() => resolve(null), SNAPSHOT_WAIT_MS)),
  ]);
  if (!png?.byteLength || !current()) return shows();
  const url = URL.createObjectURL(new Blob([png], { type: "image/png" }));
  // Decode before hiding the view, or the placeholder flashes blank.
  const image = new Image();
  image.src = url;
  if (!(await image.decode().then(() => true, () => false)) || !current()) {
    URL.revokeObjectURL(url);
    return shows();
  }
  // align snapshots to the visible native bounds because the chat dock can shorten the view.
  const box = element.getBoundingClientRect();
  element.style.backgroundImage = `url(${url})`;
  element.style.backgroundPosition = `${bounds.x - box.left}px ${bounds.y - box.top}px`;
  element.style.backgroundSize = `${bounds.width}px ${bounds.height}px`;
  element.style.backgroundRepeat = "no-repeat";
  if (snapshot && snapshot.url !== url) URL.revokeObjectURL(snapshot.url);
  snapshot = { tabId, element, url };
  return true;
}

// Covered views are parked off-window instead of hidden: a hidden view stops painting and returns blank.
let parkedView: string | null = null;

function parkable(tabId: string): boolean {
  return views.has(tabId) && shownBounds !== null && (shownView === tabId || parkedView === tabId);
}

// A failed capture keeps the page on screen and asks the next sync to try again, a few times.
const COVER_TRIES = 3;
let coverFails = 0;
let coverRetry = 0;

// A parked page whose snapshot missed a change (a zoom), until a capture lands.
let staleSnapshot: string | null = null;

function retryCover(): void {
  if (coverFails >= COVER_TRIES) return;
  coverFails += 1;
  coverRetry += 1;
}

async function coverView(tabId: string, zoom: number): Promise<void> {
  if (parkedView !== tabId) {
    // Parked without a snapshot, the placeholder would show blank.
    if (!(await paintSnapshot(tabId))) {
      retryCover();
      return;
    }
    await call("browser_view_show", { tabId, bounds: shownBounds, parked: true });
    parkedView = tabId;
    staleSnapshot = null;
    coverFails = 0;
    setShownView(null);
  }
  if ((zooms.get(tabId) ?? 1) !== zoom) {
    zooms.set(tabId, zoom);
    await call("browser_view_zoom", { tabId, zoom });
    staleSnapshot = tabId;
  }
  if (staleSnapshot !== tabId) return;
  if (await paintSnapshot(tabId, true)) {
    staleSnapshot = null;
    coverFails = 0;
  } else retryCover();
}

/** Re-snapshots a covered page after it changes, e.g. a find step. */
export function refreshCoveredPage(tabId: string): void {
  if (parkedView !== tabId) return;
  // Stale until a fresh capture lands; a miss retries through the sync like a zoom.
  staleSnapshot = tabId;
  void paintSnapshot(tabId, true).then((painted) => {
    if (!painted) retryCover();
    else if (staleSnapshot === tabId) {
      staleSnapshot = null;
      coverFails = 0;
    }
  });
}

async function applyView(desired: Desired): Promise<void> {
  if (!desired || "covered" in desired) {
    closeHole();
    // keep the snapshot if an overlay reopens during capture.
    if (snapshot?.tabId !== desired?.tabId) clearSnapshot();
    // Another tab under a lasting cover (the find bar): show it first so it can be captured, not left blank.
    if (desired && !parkable(desired.tabId)) {
      const { tabId, url, entry, zoom, bounds } = desired;
      await applyView({ tabId, url, entry, zoom, bounds });
    }
    if (desired && parkable(desired.tabId)) {
      await coverView(desired.tabId, desired.zoom);
      return;
    }
    parkedView = null;
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
    parkedView = null;
    staleSnapshot = null;
    coverFails = 0;
    clearSnapshot();
    // After the move, so hole and page update together.
    if (layered) openHole(tabId, bounds);
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
  parkedView = null;
  setShownView(null);
  views.clear();
  viewBounds.clear();
  zooms.clear();
  icons.clear();
  pages.clear();
  temporaryPages.clear();
  pageEntries.clear();
  loadingPages.clear();
  resume.clear();
  recency = [];
  epoch += 1;
  pump();
});

const MOTION_EVENTS = ["transitionrun", "transitionend", "animationstart", "animationend"] as const;
// Past Sonner's 400 ms slide.
const MOTION_MS = 500;
let motionUntil = 0;

export function startNativeViews(): () => void {
  listenOnce();
  let frame = 0;
  let sent = "";
  let resized: HTMLElement | null = null;
  const resizeObserver = new ResizeObserver(() => schedule());

  const sync = () => {
    frame = 0;
    if (layered === null) return;
    markHole();
    const desired = desiredView();
    pruneViews(desired?.tabId ?? null);
    const element = desired ? placeholder(desired.tabId) : null;
    if (element !== resized) {
      if (resized) resizeObserver.unobserve(resized);
      if (element) resizeObserver.observe(element);
      resized = element;
    }
    if (performance.now() < motionUntil) schedule();
    const key = JSON.stringify([epoch, coverRetry, desired]);
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
  // Theme switches recolor the backgrounds around the hole.
  overlays.observe(document.documentElement, { attributes: true, attributeFilter: ["class", "style", "data-palette"] });
  window.addEventListener("resize", schedule);
  // Toasts and the find bar mount without a body mutation and slide in: measure them every frame
  // while they move, so a click on a moving toast doesn't fall through to the page.
  const moved = (event: Event) => {
    if (!(event.target as Element | null)?.closest?.(CLICKABLE_SELECTOR)) return;
    motionUntil = performance.now() + MOTION_MS;
    schedule();
  };
  for (const type of MOTION_EVENTS) document.addEventListener(type, moved, true);
  const reachedPage = (event: Event) => {
    if (tooltipHold && (event.target as Element | null)?.closest?.("[data-native-page]")) {
      tooltipHold = 0;
      schedule();
    }
  };
  document.addEventListener("pointerover", reachedPage, true);
  // A dragged or resized panel ends where the pointer lets go.
  document.addEventListener("pointerup", schedule, true);
  const interval = window.setInterval(schedule, RECHECK_MS);
  void askLayered().then(schedule);
  schedule();

  return () => {
    cancelAnimationFrame(frame);
    unsubscribe();
    overlays.disconnect();
    resizeObserver.disconnect();
    window.removeEventListener("resize", schedule);
    for (const type of MOTION_EVENTS) document.removeEventListener(type, moved, true);
    document.removeEventListener("pointerover", reachedPage, true);
    document.removeEventListener("pointerup", schedule, true);
    window.clearInterval(interval);
    tooltipHold = 0;
    // close pages because hidden native views keep scripts and media running.
    generation += 1;
    pending = null;
    parkedView = null;
    setShownView(null);
    clearSnapshot();
    closeHole();
    sendInput(NO_INPUT);
    insetToasts(null);
    for (const tabId of [...views.keys()]) closeView(tabId);
  };
}
