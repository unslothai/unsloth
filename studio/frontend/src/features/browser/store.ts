// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type DocumentAnnotations, useChatRuntimeStore } from "@/features/chat";
import { create } from "zustand";
import { unwrapRedirect } from "./address";
import type { BrowserPage } from "./api";
import { PageCache, cacheLimits, reportedDeviceMemory } from "./page-cache";
import { defaultZoom } from "./prefs-store";
import { REACT_PREVIEW_KEY_PREFIX, REACT_PREVIEW_TYPE } from "./react-preview-type";

export type BrowserEntry =
  | { kind: "newtab" }
  | { kind: "internal"; page: InternalPage }
  | {
      kind: "web";
      url: string;
      method?: "GET" | "POST";
      body?: string;
      /** source page for link, form, script, or refresh; files reached here download on its behalf. */
      from?: string;
      /** Opened beside a temporary chat, so it stays out of history. */
      temporary?: true;
    }
  | {
      kind: "file";
      fileId: string;
      name: string;
      contentType: string;
      /** show document-extracted text as text even when named .html. */
      plainText?: boolean;
      /** the tab's openKey while this entry shows, so Back restores it. */
      openKey?: string;
      chatPage?: boolean;
    };

export type InternalPage = "history" | "downloads" | "bookmarks";

export type DeviceMode = "off" | "mobile" | "tablet";

export type ChatSide = "left" | "right";

export type ChatDock = "minimized" | "composer" | "expanded";

export type RequestEdits = (prompt: string) => void;

/** false keeps marks when the composer rejects them and explains why; `files` carries the annotation screenshot. */
export type SendAnnotations = (annotations: DocumentAnnotations, files?: File[]) => Promise<boolean>;

/** Stages a file in the chat's composer; false when it refused it (it says why). */
export type AttachToChat = (file: File) => Promise<boolean>;

export type FileViewMode = "preview" | "source";

export type FileViewState = {
  mode: FileViewMode;
  consoleOpen: boolean;
  errorCount: number;
  wrap: boolean;
};

export const DEFAULT_FILE_VIEW: FileViewState = { mode: "preview", consoleOpen: false, errorCount: 0, wrap: false };

export type BrowserTab = {
  id: string;
  history: BrowserEntry[];
  index: number;
  title: string;
  favicon: string | null;
  documentType: string | null;
  /** Address from pushState, shown instead of the loaded URL. */
  displayUrl: string | null;
  loading: boolean;
  reloadKey: number;
  /** What the tab was opened for; opening it again focuses the tab. */
  openKey: string | null;
  zoom: number;
  nativeHistory: { back: boolean; forward: boolean } | null;
  nativeError: string | null;
  /** The proxied page failed to load, so an error shows instead of its frame. */
  pageError?: boolean;
  /** The name the reader gave the tab; kept as it navigates. */
  customTitle: string | null;
  muted: boolean;
  /** The sidebar pin this tab shows (pinned-pages-store), kept as it navigates. */
  pinnedId: string | null;
};

export type OpenFileInput = {
  blob: Blob;
  name: string;
  contentType?: string;
  plainText?: boolean;
  key?: string;
};

// Blobs live outside the store; documents can be 50 MB.
const files = new Map<string, Blob>();

export function browserFile(fileId: string): Blob | undefined {
  return files.get(fileId);
}

type PageDownload = { blob: Blob; name: string; contentType: string };
const pageDownloads = new Map<string, PageDownload>();

export function setPageDownload(tabId: string, download: PageDownload | null): void {
  if (download) pageDownloads.set(tabId, download);
  else pageDownloads.delete(tabId);
}

export function pageDownload(tabId: string): PageDownload | undefined {
  return pageDownloads.get(tabId);
}

const isWeb = (url: string) => /^https?:\/\//i.test(url.trim());

let nativeWebHistory = false;

export function setNativeWebHistory(native: boolean): void {
  nativeWebHistory = native;
}

const MAX_HISTORY = 50;

// Loaded pages by history entry, so back and forward skip the fetch. Reload drops only its own entry.
// Fewer on a low-memory machine.
const cacheLimit = cacheLimits(reportedDeviceMemory());
const pageCache = new PageCache<BrowserEntry>(cacheLimit.maxPages, cacheLimit.maxTotalBytes);

// The entry each page-driven navigation left, while it is still the one before it.
const sentFrom = new WeakMap<BrowserEntry, BrowserEntry>();

const entryIds = new WeakMap<BrowserEntry, number>();
let nextEntryId = 0;

/** Stable id for a history entry; entries can share a URL. */
export function entryKey(entry: BrowserEntry): number {
  let id = entryIds.get(entry);
  if (id === undefined) {
    id = nextEntryId++;
    entryIds.set(entry, id);
  }
  return id;
}

export function cachedPage(entry: BrowserEntry): BrowserPage | undefined {
  return pageCache.get(entry);
}

export function cachePage(entry: BrowserEntry, page: BrowserPage): void {
  pageCache.set(entry, page);
}

export function clearPageCache(): void {
  pageCache.clear();
}

// Posted entries already sent: loading one again asks rather than resubmitting (tab-view.tsx).
export const sentPosts = new WeakSet<BrowserEntry>();

let nextId = 0;
const newId = (prefix: string) => `${prefix}-${Date.now().toString(36)}-${(nextId++).toString(36)}`;

function createTab(entry: BrowserEntry, openKey: string | null = null): BrowserTab {
  return {
    id: newId("tab"),
    history: [entry],
    index: 0,
    title: entry.kind === "file" ? entry.name : "",
    favicon: null,
    documentType: null,
    displayUrl: null,
    loading: false,
    reloadKey: 0,
    openKey,
    // Files open fitted; the default zoom is for web pages.
    zoom: entry.kind === "file" ? 1 : defaultZoom(),
    nativeHistory: null,
    nativeError: null,
    customTitle: null,
    muted: false,
    pinnedId: null,
  };
}

export const MAX_TAB_TITLE_CHARS = 120;

// A copy of each entry, so the copy keeps its own cached pages and native view.
function copyTab(tab: BrowserTab): BrowserTab {
  return {
    ...createTab({ kind: "newtab" }),
    history: tab.history.map((entry) => {
      const copy = { ...entry };
      // The original keeps its key; a copy going Back must not claim it.
      if (copy.kind === "file") delete copy.openKey;
      // A form result is not sent again unasked just because the tab was copied.
      if (entry.kind === "web" && entry.method === "POST") sentPosts.add(copy);
      return copy;
    }),
    index: tab.index,
    title: tab.title,
    favicon: tab.favicon,
    documentType: tab.documentType,
    zoom: tab.zoom,
    customTitle: tab.customTitle,
  };
}

export function currentEntry(tab: BrowserTab): BrowserEntry {
  return tab.history[tab.index] ?? { kind: "newtab" };
}

function webEntry(url: string, method?: "GET" | "POST", body?: string, from?: string, temporary?: boolean): BrowserEntry {
  const entry: Extract<BrowserEntry, { kind: "web" }> =
    method === "POST" ? { kind: "web", url, method, body } : { kind: "web", url: unwrapRedirect(url) };
  if (temporary ?? useChatRuntimeStore.getState().incognito) entry.temporary = true;
  return from ? { ...entry, from } : entry;
}

// Pending file refreshes per tab, run in order, and the blobs they will compare.
const refreshes = new Map<string, Promise<void>>();
const queuedFiles = new Set<string>();

const COMPARE_CHUNK_BYTES = 1024 * 1024;

// In slices, a word at a time: a 50 MB file never holds the UI thread or both copies whole.
async function sameBytes(a: Blob | undefined, b: Blob): Promise<boolean> {
  if (!a || a.size !== b.size) return false;
  for (let start = 0; start < b.size; start += COMPARE_CHUNK_BYTES) {
    const end = start + COMPARE_CHUNK_BYTES;
    const [x, y] = await Promise.all([a.slice(start, end).arrayBuffer(), b.slice(start, end).arrayBuffer()]);
    if (!sameBuffer(x, y)) return false;
  }
  return true;
}

function sameBuffer(x: ArrayBuffer, y: ArrayBuffer): boolean {
  const words = x.byteLength >> 2;
  const wx = new Uint32Array(x, 0, words);
  const wy = new Uint32Array(y, 0, words);
  for (let i = 0; i < words; i++) if (wx[i] !== wy[i]) return false;
  const bx = new Uint8Array(x);
  const by = new Uint8Array(y);
  for (let i = words << 2; i < bx.length; i++) if (bx[i] !== by[i]) return false;
  return true;
}

function releaseFiles(tabs: BrowserTab[]): void {
  const live = new Set<string>(queuedFiles);
  for (const tab of tabs) {
    for (const entry of tab.history) if (entry.kind === "file") live.add(entry.fileId);
  }
  for (const id of files.keys()) if (!live.has(id)) files.delete(id);
}

type BrowserState = {
  open: boolean;
  tabs: BrowserTab[];
  activeTabId: string | null;
  /** Bumped on open, so the panel re-expands after a drag shut. */
  openSequence: number;
  focusAddressSequence: number;
  /** Bumped by ⌘D, so the address bar's star bookmarks the page or opens its editor. */
  bookmarkSequence: number;
  device: DeviceMode;
  fullView: boolean;
  chatDock: ChatDock;
  chatSide: ChatSide;
  /** Stages a prompt in the chat's composer; set by the chat while it is shown. */
  requestEdits: RequestEdits | null;
  sendAnnotations: SendAnnotations | null;
  attachToChat: AttachToChat | null;
  annotateTabId: string | null;
  setAnnotating: (tabId: string | null) => void;
  fileViews: Record<string, FileViewState>;
  setFileView: (tabId: string, patch: Partial<FileViewState>) => void;
  openPanel: () => void;
  closePanel: () => void;
  togglePanel: () => void;
  newTab: () => void;
  newTabAfter: (tabId: string) => void;
  /** copies the tab and its history immediately after it. */
  duplicateTab: (tabId: string) => void;
  /** opens a pinned page in its own tab or focuses the tab already showing it. */
  openPinned: (pinnedId: string, url: string, title: string) => void;
  setTabPinned: (tabId: string, pinnedId: string | null) => void;
  renamingTabId: string | null;
  setRenamingTab: (tabId: string | null) => void;
  /** sets a tab name, or null to use the page title. */
  renameTab: (tabId: string, title: string | null) => void;
  setMuted: (tabId: string, muted: boolean) => void;
  closeOtherTabs: (tabId: string) => void;
  closeTabsToRight: (tabId: string) => void;
  closeChatPages: () => void;
  openUrl: (
    url: string,
    options?: { newTab?: boolean; background?: boolean; method?: "GET" | "POST"; body?: string; from?: string },
  ) => void;
  openFile: (input: OpenFileInput) => void;
  navigate: (
    tabId: string,
    request: { url: string; method?: "GET" | "POST"; body?: string; from?: string; temporary?: boolean },
    options?: { replace?: boolean },
  ) => void;
  /** removes a page-sent download from history and returns to its sending page unless the tab moved on. */
  leaveDownload: (tabId: string, entry: BrowserEntry) => void;
  goBack: (tabId: string) => void;
  goForward: (tabId: string) => void;
  reload: (tabId: string) => void;
  activateTab: (tabId: string) => void;
  closeTab: (tabId: string) => void;
  /** Moves a tab to `index` in the strip, as dragging it there does. */
  moveTab: (tabId: string, index: number) => void;
  updateTab: (
    tabId: string,
    patch: Partial<
      Pick<
        BrowserTab,
        "title" | "favicon" | "documentType" | "displayUrl" | "loading" | "nativeHistory" | "nativeError" | "pageError"
      >
    >,
  ) => void;
  focusAddress: () => void;
  bookmarkPage: () => void;
  openInternal: (page: InternalPage) => void;
  setZoom: (tabId: string, zoom: number) => void;
  setDevice: (device: DeviceMode) => void;
  setFullView: (fullView: boolean) => void;
  setChatDock: (dock: ChatDock) => void;
  splitWithChatOn: (side: ChatSide) => void;
};

const patchTab = (tabs: BrowserTab[], tabId: string, update: (tab: BrowserTab) => BrowserTab) =>
  tabs.map((tab) => (tab.id === tabId ? update(tab) : tab));

/** Zoom on entering `entry`, by opening it or going back or forward. A web page reached from a new
 *  tab, a history page or an unzoomed file starts at the default; a file reached from an unzoomed
 *  web page shows at 100%. A zoom the reader chose carries over, and Back/Forward (`traversal`)
 *  skip the new tab rule, so a page keeps the zoom its new tab entry carried. */
function zoomFor(tab: BrowserTab, entry: BrowserEntry, traversal = false): number {
  const from = tab.history[tab.index];
  if (!from) return tab.zoom;
  const preferred = defaultZoom();
  const at = (zoom: number) => Math.abs(tab.zoom - zoom) < 0.001;
  if (entry.kind === "web") {
    const unzoomedFile = from.kind === "file" && at(1);
    const blank = !traversal && (from.kind === "newtab" || from.kind === "internal");
    return blank || unzoomedFile ? preferred : tab.zoom;
  }
  if (entry.kind === "file" && from.kind === "web" && at(preferred)) return 1;
  return tab.zoom;
}

function pushEntry(tab: BrowserTab, entry: BrowserEntry, replace = false): BrowserTab {
  const zoom = zoomFor(tab, entry);
  const history = tab.history.slice(0, replace ? tab.index : tab.index + 1);
  history.push(entry);
  if (history.length > MAX_HISTORY) history.splice(0, history.length - MAX_HISTORY);
  return {
    ...tab,
    history,
    index: history.length - 1,
    title: entry.kind === "file" ? entry.name : "",
    favicon: null,
    documentType: null,
    displayUrl: null,
    loading: entry.kind === "web",
    openKey: null,
    nativeError: null,
    pageError: false,
    zoom,
  };
}

function moveTo(tab: BrowserTab, index: number): BrowserTab {
  const entry = tab.history[index] ?? { kind: "newtab" };
  return {
    ...tab,
    zoom: zoomFor(tab, entry, true),
    index,
    title: entry.kind === "file" ? entry.name : "",
    favicon: null,
    documentType: null,
    displayUrl: null,
    loading: entry.kind === "web",
    openKey: entry.kind === "file" ? (entry.openKey ?? null) : null,
    nativeError: null,
    pageError: false,
  };
}

export const useBrowserStore = create<BrowserState>((set, get) => {
  const openTab = (tab: BrowserTab, background = false, after?: string) => {
    set((state) => {
      const at = after ? state.tabs.findIndex((other) => other.id === after) : -1;
      const tabs = [...state.tabs];
      tabs.splice(at < 0 ? tabs.length : at + 1, 0, tab);
      return {
        open: true,
        tabs,
        activeTabId: background && state.activeTabId ? state.activeTabId : tab.id,
        openSequence: state.openSequence + 1,
      };
    });
  };
  const focusExisting = (openKey: string): boolean => {
    const existing = get().tabs.find((tab) => tab.openKey === openKey);
    if (!existing) return false;
    set((state) => ({ open: true, activeTabId: existing.id, openSequence: state.openSequence + 1 }));
    return true;
  };

  return {
    open: false,
    tabs: [],
    activeTabId: null,
    openSequence: 0,
    focusAddressSequence: 0,
    bookmarkSequence: 0,
    device: "off",
    fullView: false,
    chatDock: "composer",
    chatSide: "left",
    requestEdits: null,
    sendAnnotations: null,
    attachToChat: null,
    annotateTabId: null,
    setAnnotating: (annotateTabId) => set({ annotateTabId }),
    fileViews: {},
    setFileView: (tabId, patch) =>
      set((state) => {
        const current = state.fileViews[tabId] ?? DEFAULT_FILE_VIEW;
        const next = { ...current, ...patch };
        if ((Object.keys(next) as (keyof FileViewState)[]).every((key) => next[key] === current[key])) return state;
        return { fileViews: { ...state.fileViews, [tabId]: next } };
      }),
    openPanel: () => {
      if (get().tabs.length === 0) {
        get().newTab();
        return;
      }
      set((state) => ({ open: true, openSequence: state.openSequence + 1 }));
    },
    closePanel: () => set({ open: false, fullView: false, annotateTabId: null }),
    togglePanel: () => (get().open ? get().closePanel() : get().openPanel()),
    newTab: () => {
      openTab(createTab({ kind: "newtab" }));
      get().focusAddress();
    },
    newTabAfter: (tabId) => {
      openTab(createTab({ kind: "newtab" }), false, tabId);
      get().focusAddress();
    },
    duplicateTab: (tabId) => {
      const tab = get().tabs.find((candidate) => candidate.id === tabId);
      if (!tab) return;
      openTab(copyTab(tab), false, tabId);
    },
    openPinned: (pinnedId, url, title) => {
      const shown = get().tabs.find((tab) => tab.pinnedId === pinnedId);
      if (shown) {
        set((state) => ({ open: true, activeTabId: shown.id, openSequence: state.openSequence + 1 }));
        return;
      }
      if (!isWeb(url)) return;
      openTab({ ...createTab(webEntry(url)), pinnedId, title, loading: true });
    },
    setTabPinned: (tabId, pinnedId) =>
      set((state) => ({ tabs: patchTab(state.tabs, tabId, (tab) => ({ ...tab, pinnedId })) })),
    renamingTabId: null,
    setRenamingTab: (renamingTabId) => set({ renamingTabId }),
    renameTab: (tabId, title) => {
      const customTitle = title?.trim().slice(0, MAX_TAB_TITLE_CHARS) || null;
      set((state) => ({ tabs: patchTab(state.tabs, tabId, (tab) => ({ ...tab, customTitle })) }));
    },
    setMuted: (tabId, muted) =>
      set((state) => ({ tabs: patchTab(state.tabs, tabId, (tab) => (tab.muted === muted ? tab : { ...tab, muted })) })),
    closeOtherTabs: (tabId) => {
      for (const tab of get().tabs) if (tab.id !== tabId) get().closeTab(tab.id);
    },
    closeTabsToRight: (tabId) => {
      const { tabs } = get();
      const index = tabs.findIndex((tab) => tab.id === tabId);
      if (index < 0) return;
      for (const tab of tabs.slice(index + 1)) get().closeTab(tab.id);
    },
    closeChatPages: () => {
      for (const tab of get().tabs) {
        if (tab.history.some((entry) => entry.kind === "file" && entry.chatPage)) get().closeTab(tab.id);
      }
    },
    openUrl: (url, options) => {
      if (!isWeb(url)) return;
      if (options?.method === "POST") {
        openTab(createTab(webEntry(url, "POST", options.body ?? "", options.from)), options.background);
        return;
      }
      const target = unwrapRedirect(url);
      const openKey = `url:${target}`;
      const { activeTabId } = get();
      if (options?.newTab === false && activeTabId) {
        get().navigate(activeTabId, { url: target, from: options.from });
        set((state) => ({ open: true, openSequence: state.openSequence + 1 }));
        return;
      }
      if (options?.newTab === undefined && focusExisting(openKey)) return;
      openTab(createTab(webEntry(target, undefined, undefined, options?.from), openKey), options?.background);
    },
    openFile: ({ blob, name, contentType, plainText, key }) => {
      const openKey = key ? `file:${key}` : null;
      const fileId = newId("file");
      files.set(fileId, blob);
      const type = contentType || blob.type;
      const entry: BrowserEntry = {
        kind: "file",
        fileId,
        name: name || "Untitled",
        // Only a React preview opened from chat runs as one; a file that claims the type is text.
        contentType: type === REACT_PREVIEW_TYPE && !key?.startsWith(REACT_PREVIEW_KEY_PREFIX) ? "text/plain" : type,
        plainText,
        ...(openKey ? { openKey } : {}),
        ...(key?.startsWith("html:") ? { chatPage: true } : {}),
      };
      const existing = openKey ? get().tabs.find((tab) => tab.openKey === openKey) : undefined;
      if (openKey && existing) {
        focusExisting(openKey);
        // refresh changed files in order per tab so the last reopen wins.
        const previous = refreshes.get(existing.id) ?? Promise.resolve();
        queuedFiles.add(fileId);
        const next = previous.then(async () => {
          const tab = get().tabs.find((candidate) => candidate.id === existing.id);
          const shown = tab && currentEntry(tab);
          if (!tab || shown?.kind !== "file" || (await sameBytes(files.get(shown.fileId), blob))) {
            files.delete(fileId);
            return;
          }
          const tabs = patchTab(get().tabs, tab.id, (current) => ({
            ...current,
            history: current.history.map((item) => (item === shown ? entry : item)),
          }));
          set({ tabs });
          // Not releaseFiles: a later reopen's blob is stored but not referenced yet.
          if (!tabs.some((other) => other.history.some((item) => item.kind === "file" && item.fileId === shown.fileId))) {
            files.delete(shown.fileId);
          }
        });
        refreshes.set(existing.id, next);
        void next.finally(() => {
          queuedFiles.delete(fileId);
          if (refreshes.get(existing.id) === next) refreshes.delete(existing.id);
        });
        return;
      }
      openTab(createTab(entry, openKey));
    },
    navigate: (tabId, request, options) => {
      if (!isWeb(request.url)) return;
      set((state) => ({
        tabs: patchTab(state.tabs, tabId, (tab) => {
          const replace = options?.replace ?? (nativeWebHistory && currentEntry(tab).kind === "web");
          const entry = webEntry(request.url, request.method, request.body, request.from, request.temporary);
          if (request.from && !replace) sentFrom.set(entry, currentEntry(tab));
          return pushEntry(tab, entry, replace);
        }),
      }));
    },
    leaveDownload: (tabId, entry) =>
      set((state) => ({
        tabs: patchTab(state.tabs, tabId, (tab) => {
          const previous = tab.history[tab.index - 1];
          if (currentEntry(tab) !== entry || !previous || sentFrom.get(entry) !== previous) return tab;
          const history = tab.history.filter((candidate) => candidate !== entry);
          pageCache.delete(entry);
          return moveTo({ ...tab, history }, tab.index - 1);
        }),
      })),
    goBack: (tabId) =>
      set((state) => ({
        tabs: patchTab(state.tabs, tabId, (tab) => (tab.index > 0 ? moveTo(tab, tab.index - 1) : tab)),
      })),
    goForward: (tabId) =>
      set((state) => ({
        tabs: patchTab(state.tabs, tabId, (tab) =>
          tab.index < tab.history.length - 1 ? moveTo(tab, tab.index + 1) : tab,
        ),
      })),
    reload: (tabId) => {
      const tab = get().tabs.find((candidate) => candidate.id === tabId);
      if (tab) pageCache.delete(currentEntry(tab));
      set((state) => ({
        tabs: patchTab(state.tabs, tabId, (tab) => ({ ...tab, reloadKey: tab.reloadKey + 1 })),
      }));
    },
    activateTab: (tabId) => set({ activeTabId: tabId, annotateTabId: null }),
    moveTab: (tabId, index) =>
      set((state) => {
        const from = state.tabs.findIndex((tab) => tab.id === tabId);
        const to = Math.max(0, Math.min(index, state.tabs.length - 1));
        if (from < 0 || from === to) return state;
        const tabs = [...state.tabs];
        const [moved] = tabs.splice(from, 1);
        if (moved) tabs.splice(to, 0, moved);
        return { tabs };
      }),
    closeTab: (tabId) => {
      const { tabs, activeTabId } = get();
      const index = tabs.findIndex((tab) => tab.id === tabId);
      if (index < 0) return;
      const remaining = tabs.filter((tab) => tab.id !== tabId);
      for (const entry of tabs[index]?.history ?? []) pageCache.delete(entry);
      releaseFiles(remaining);
      pageDownloads.delete(tabId);
      const { [tabId]: _closed, ...fileViews } = get().fileViews;
      const annotateTabId = get().annotateTabId === tabId ? null : get().annotateTabId;
      const renamingTabId = get().renamingTabId === tabId ? null : get().renamingTabId;
      if (remaining.length === 0) {
        set({ tabs: [], activeTabId: null, open: false, fullView: false, annotateTabId: null, renamingTabId, fileViews });
        return;
      }
      const nextActive =
        activeTabId === tabId ? (remaining[Math.min(index, remaining.length - 1)]?.id ?? null) : activeTabId;
      set({ tabs: remaining, activeTabId: nextActive, annotateTabId, renamingTabId, fileViews });
    },
    updateTab: (tabId, patch) =>
      set((state) => {
        const tab = state.tabs.find((candidate) => candidate.id === tabId);
        const keys = Object.keys(patch) as (keyof typeof patch)[];
        if (!tab || keys.every((key) => tab[key] === patch[key])) return state;
        return { tabs: patchTab(state.tabs, tabId, (current) => ({ ...current, ...patch })) };
      }),
    focusAddress: () => set((state) => ({ focusAddressSequence: state.focusAddressSequence + 1 })),
    bookmarkPage: () => set((state) => ({ bookmarkSequence: state.bookmarkSequence + 1 })),
    openInternal: (page) => {
      const openKey = `internal:${page}`;
      if (focusExisting(openKey)) return;
      openTab(createTab({ kind: "internal", page }, openKey));
    },
    setZoom: (tabId, zoom) =>
      set((state) => ({
        tabs: patchTab(state.tabs, tabId, (tab) => ({ ...tab, zoom: Math.min(5, Math.max(0.25, zoom)) })),
      })),
    setDevice: (device) => set({ device }),
    setFullView: (fullView) => set({ fullView, chatDock: "composer" }),
    setChatDock: (chatDock) => set({ chatDock }),
    splitWithChatOn: (chatSide) => set({ fullView: false, chatSide }),
  };
});
