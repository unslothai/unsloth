// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type DocumentAnnotations, useChatArtifactsStore } from "@/features/chat";
import { create } from "zustand";
import { unwrapRedirect } from "./address";
import type { BrowserPage } from "./api";
import { PageCache } from "./page-cache";

export type BrowserEntry =
  | { kind: "newtab" }
  | { kind: "internal"; page: InternalPage }
  | { kind: "web"; url: string; method?: "GET" | "POST"; body?: string }
  | {
      kind: "file";
      fileId: string;
      name: string;
      contentType: string;
      /** Show as text even if named .html (text extracted from a document). */
      plainText?: boolean;
    };

export type InternalPage = "history" | "downloads";

export type DeviceMode = "off" | "mobile" | "tablet";

/** Which side of the browser the chat sits on when they are split. */
export type ChatSide = "left" | "right";

/** The chat floating over a full-view browser: hidden to a button, its composer, or the whole conversation. */
export type ChatDock = "minimized" | "composer" | "expanded";

export type RequestEdits = (prompt: string) => void;

export type SendAnnotations = (annotations: DocumentAnnotations) => void;

export type OpenInCanvas = (file: { title: string; code: string }) => void;

/** How a text file tab shows its file, like the canvas: the rendered page or its source. */
export type FileViewMode = "preview" | "source";

export type FileViewState = {
  mode: FileViewMode;
  /** The HTML console under the page. */
  consoleOpen: boolean;
  /** Errors the HTML page reported, for the console button's badge. */
  errorCount: number;
  /** Long lines wrap in the source and text views. */
  wrap: boolean;
};

export const DEFAULT_FILE_VIEW: FileViewState = { mode: "preview", consoleOpen: false, errorCount: 0, wrap: false };

export type BrowserTab = {
  id: string;
  history: BrowserEntry[];
  index: number;
  title: string;
  favicon: string | null;
  /** Content type of a document the web tab shows (a PDF link), for its icon. */
  documentType: string | null;
  /** Address from pushState, shown instead of the loaded URL. */
  displayUrl: string | null;
  loading: boolean;
  /** Bumped by reload to refetch. */
  reloadKey: number;
  /** What the tab was opened for; opening it again focuses the tab. */
  openKey: string | null;
  /** Page zoom, kept across navigations like a browser's per-tab zoom. */
  zoom: number;
  /** A native page's own history, used before the tab's. */
  nativeHistory: { back: boolean; forward: boolean } | null;
  /** Why a native page couldn't open, shown in its place. */
  nativeError: string | null;
};

export type OpenFileInput = {
  blob: Blob;
  name: string;
  contentType?: string;
  plainText?: boolean;
  /** Stable identity; reopening it focuses the existing tab. */
  key?: string;
};

// Blobs live outside the store; documents can be 50 MB.
const files = new Map<string, Blob>();

export function browserFile(fileId: string): Blob | undefined {
  return files.get(fileId);
}

type PageDownload = { blob: Blob; name: string; contentType: string };
// Document shown by a web tab (e.g. a PDF link), for Download.
const pageDownloads = new Map<string, PageDownload>();

export function setPageDownload(tabId: string, download: PageDownload | null): void {
  if (download) pageDownloads.set(tabId, download);
  else pageDownloads.delete(tabId);
}

export function pageDownload(tabId: string): PageDownload | undefined {
  return pageDownloads.get(tabId);
}

const isWeb = (url: string) => /^https?:\/\//i.test(url.trim());

// Set in the desktop app, where web pages are native views with their own history.
let nativeWebHistory = false;

export function setNativeWebHistory(native: boolean): void {
  nativeWebHistory = native;
}

// Caps history for pages that keep redirecting.
const MAX_HISTORY = 50;

// Loaded pages by history entry, so back and forward skip the fetch. Reload bumps reloadKey.
const pageCache = new PageCache<BrowserEntry>();

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

export function cachedPage(entry: BrowserEntry, reloadKey: number): BrowserPage | undefined {
  return pageCache.get(entry, reloadKey);
}

export function cachePage(entry: BrowserEntry, reloadKey: number, page: BrowserPage): void {
  pageCache.set(entry, reloadKey, page);
}

export function clearPageCache(): void {
  pageCache.clear();
}

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
    zoom: 1,
    nativeHistory: null,
    nativeError: null,
  };
}

export function currentEntry(tab: BrowserTab): BrowserEntry {
  return tab.history[tab.index] ?? { kind: "newtab" };
}

function webEntry(url: string, method?: "GET" | "POST", body?: string): BrowserEntry {
  return method === "POST" ? { kind: "web", url, method, body } : { kind: "web", url: unwrapRedirect(url) };
}

/** Drop blobs no remaining history entry points at. */
function releaseFiles(tabs: BrowserTab[]): void {
  const live = new Set<string>();
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
  /** Bumped to ask the address bar for focus (Cmd+L, a new tab). */
  focusAddressSequence: number;
  /** Find bar for the active tab, and whether its last search missed. */
  findOpen: boolean;
  findMiss: boolean;
  /** Device toolbar: pages shown at a phone or tablet width. */
  device: DeviceMode;
  /** Full view: the browser takes the whole width and the chat floats over it. */
  fullView: boolean;
  chatDock: ChatDock;
  chatSide: ChatSide;
  /** Whether the chat beside the panel has messages; the header toggle hides in a new chat. */
  chatHasMessages: boolean;
  /** Stages a prompt in the chat's composer; set by the chat while it is shown. */
  requestEdits: RequestEdits | null;
  /** Sends a file's annotations to the chat; set by the chat while it is shown. */
  sendAnnotations: SendAnnotations | null;
  /** Opens an HTML file in the chat's canvas; set by the chat while it is shown. */
  openInCanvas: OpenInCanvas | null;
  /** The file tab being annotated for Request edits. */
  annotateTabId: string | null;
  setAnnotating: (tabId: string | null) => void;
  /** Per file tab; absent means DEFAULT_FILE_VIEW. */
  fileViews: Record<string, FileViewState>;
  setFileView: (tabId: string, patch: Partial<FileViewState>) => void;
  openPanel: () => void;
  closePanel: () => void;
  togglePanel: () => void;
  newTab: () => void;
  openUrl: (
    url: string,
    options?: { newTab?: boolean; background?: boolean; method?: "GET" | "POST"; body?: string },
  ) => void;
  openFile: (input: OpenFileInput) => void;
  navigate: (
    tabId: string,
    request: { url: string; method?: "GET" | "POST"; body?: string },
    options?: { replace?: boolean },
  ) => void;
  goBack: (tabId: string) => void;
  goForward: (tabId: string) => void;
  reload: (tabId: string) => void;
  activateTab: (tabId: string) => void;
  closeTab: (tabId: string) => void;
  updateTab: (
    tabId: string,
    patch: Partial<
      Pick<
        BrowserTab,
        "title" | "favicon" | "documentType" | "displayUrl" | "loading" | "nativeHistory" | "nativeError"
      >
    >,
  ) => void;
  focusAddress: () => void;
  openInternal: (page: InternalPage) => void;
  setZoom: (tabId: string, zoom: number) => void;
  setFindOpen: (open: boolean) => void;
  setFindMiss: (miss: boolean) => void;
  setDevice: (device: DeviceMode) => void;
  setFullView: (fullView: boolean) => void;
  setChatDock: (dock: ChatDock) => void;
  /** Leaves full view with the chat on this side. */
  splitWithChatOn: (side: ChatSide) => void;
};

const patchTab = (tabs: BrowserTab[], tabId: string, update: (tab: BrowserTab) => BrowserTab) =>
  tabs.map((tab) => (tab.id === tabId ? update(tab) : tab));

function pushEntry(tab: BrowserTab, entry: BrowserEntry, replace = false): BrowserTab {
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
  };
}

function moveTo(tab: BrowserTab, index: number): BrowserTab {
  const entry = tab.history[index] ?? { kind: "newtab" };
  return {
    ...tab,
    index,
    title: entry.kind === "file" ? entry.name : "",
    favicon: null,
    documentType: null,
    displayUrl: null,
    loading: entry.kind === "web",
    nativeError: null,
  };
}

function showPanel(): void {
  // One context panel at a time.
  useChatArtifactsStore.getState().closeArtifactSurface();
}

export const useBrowserStore = create<BrowserState>((set, get) => {
  const openTab = (tab: BrowserTab, background = false) => {
    showPanel();
    set((state) => ({
      open: true,
      tabs: [...state.tabs, tab],
      activeTabId: background && state.activeTabId ? state.activeTabId : tab.id,
      openSequence: state.openSequence + 1,
    }));
  };
  const focusExisting = (openKey: string): boolean => {
    const existing = get().tabs.find((tab) => tab.openKey === openKey);
    if (!existing) return false;
    showPanel();
    set((state) => ({ open: true, activeTabId: existing.id, openSequence: state.openSequence + 1 }));
    return true;
  };

  return {
    open: false,
    tabs: [],
    activeTabId: null,
    openSequence: 0,
    focusAddressSequence: 0,
    findOpen: false,
    findMiss: false,
    device: "off",
    fullView: false,
    chatDock: "composer",
    chatSide: "left",
    chatHasMessages: false,
    requestEdits: null,
    sendAnnotations: null,
    openInCanvas: null,
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
      showPanel();
      set((state) => ({ open: true, openSequence: state.openSequence + 1 }));
    },
    closePanel: () => set({ open: false, fullView: false, annotateTabId: null }),
    togglePanel: () => (get().open ? get().closePanel() : get().openPanel()),
    newTab: () => {
      openTab(createTab({ kind: "newtab" }));
      get().focusAddress();
    },
    openUrl: (url, options) => {
      if (!isWeb(url)) return;
      // A form posted to a new tab (target=_blank) keeps its body.
      if (options?.method === "POST") {
        openTab(createTab(webEntry(url, "POST", options.body ?? "")), options.background);
        return;
      }
      const target = unwrapRedirect(url);
      const openKey = `url:${target}`;
      const { activeTabId } = get();
      if (options?.newTab === false && activeTabId) {
        get().navigate(activeTabId, { url: target });
        showPanel();
        set((state) => ({ open: true, openSequence: state.openSequence + 1 }));
        return;
      }
      if (options?.newTab === undefined && focusExisting(openKey)) return;
      openTab(createTab(webEntry(target), openKey), options?.background);
    },
    openFile: ({ blob, name, contentType, plainText, key }) => {
      const openKey = key ? `file:${key}` : null;
      if (openKey && focusExisting(openKey)) return;
      const fileId = newId("file");
      files.set(fileId, blob);
      openTab(
        createTab(
          { kind: "file", fileId, name: name || "Untitled", contentType: contentType || blob.type, plainText },
          openKey,
        ),
      );
    },
    navigate: (tabId, request, options) => {
      if (!isWeb(request.url)) return;
      set((state) => ({
        tabs: patchTab(state.tabs, tabId, (tab) =>
          pushEntry(
            tab,
            webEntry(request.url, request.method, request.body),
            // A native page keeps its own history.
            options?.replace || (nativeWebHistory && currentEntry(tab).kind === "web"),
          ),
        ),
      }));
    },
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
    reload: (tabId) =>
      set((state) => ({
        tabs: patchTab(state.tabs, tabId, (tab) => ({ ...tab, reloadKey: tab.reloadKey + 1 })),
      })),
    activateTab: (tabId) => set({ activeTabId: tabId, findOpen: false, findMiss: false, annotateTabId: null }),
    closeTab: (tabId) => {
      const { tabs, activeTabId } = get();
      const index = tabs.findIndex((tab) => tab.id === tabId);
      if (index < 0) return;
      const remaining = tabs.filter((tab) => tab.id !== tabId);
      for (const entry of tabs[index]?.history ?? []) pageCache.delete(entry);
      releaseFiles(remaining);
      pageDownloads.delete(tabId);
      const { [tabId]: _closed, ...fileViews } = get().fileViews;
      if (remaining.length === 0) {
        set({ tabs: [], activeTabId: null, open: false, fileViews });
        return;
      }
      const nextActive =
        activeTabId === tabId ? (remaining[Math.min(index, remaining.length - 1)]?.id ?? null) : activeTabId;
      set({ tabs: remaining, activeTabId: nextActive, fileViews });
    },
    updateTab: (tabId, patch) =>
      set((state) => {
        const tab = state.tabs.find((candidate) => candidate.id === tabId);
        // Skip no-op updates.
        const keys = Object.keys(patch) as (keyof typeof patch)[];
        if (!tab || keys.every((key) => tab[key] === patch[key])) return state;
        return { tabs: patchTab(state.tabs, tabId, (current) => ({ ...current, ...patch })) };
      }),
    focusAddress: () => set((state) => ({ focusAddressSequence: state.focusAddressSequence + 1 })),
    openInternal: (page) => {
      const openKey = `internal:${page}`;
      if (focusExisting(openKey)) return;
      openTab(createTab({ kind: "internal", page }, openKey));
    },
    setZoom: (tabId, zoom) =>
      set((state) => ({
        tabs: patchTab(state.tabs, tabId, (tab) => ({ ...tab, zoom: Math.min(5, Math.max(0.25, zoom)) })),
      })),
    setFindOpen: (findOpen) => set({ findOpen, findMiss: false }),
    setFindMiss: (findMiss) => set({ findMiss }),
    setDevice: (device) => set({ device }),
    setFullView: (fullView) => set({ fullView, chatDock: "composer" }),
    setChatDock: (chatDock) => set({ chatDock }),
    splitWithChatOn: (chatSide) => set({ fullView: false, chatSide }),
  };
});
