// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Open tabs and each chat's panel, kept in sessionStorage: for a reload of this window only.

import { accountDatabaseName } from "@/lib/account-transition";
import { isWebUrl } from "./address";
import type { BrowserEntry, BrowserTab, InternalPage } from "./store";

type ChatBrowser = { open: boolean; activeTabId: string | null; fullView: boolean; floating: boolean };

export type TabSessionSource = {
  tabs: BrowserTab[];
  chat: string;
  chats: Record<string, ChatBrowser>;
  open: boolean;
  activeTabId: string | null;
  fullView: boolean;
  floating: boolean;
  floatedForAgent?: boolean;
};

export type TabSession = {
  tabs: BrowserTab[];
  chats: Record<string, ChatBrowser>;
  /** The chat that was on screen, if it had tabs (store.ts `claimRestored`). */
  shown: string | null;
};

const VERSION = 1;
const MAX_TABS = 50;
const MAX_ENTRIES = 50;
const MAX_URL_CHARS = 8192;
const MAX_TITLE_CHARS = 200;
const MAX_ICON_CHARS = 2048;
const MAX_ID_CHARS = 200;
const INTERNAL_PAGES: readonly string[] = ["history", "downloads", "bookmarks"] satisfies InternalPage[];

// Per account: a write after an account switch lands under the account that made it.
export const TAB_SESSION_KEY = accountDatabaseName("unsloth_browser_tabs");

const EMPTY: TabSession = { tabs: [], chats: {}, shown: null };

function sessionStore(): Storage | null {
  try {
    return globalThis.sessionStorage ?? null;
  } catch {
    return null;
  }
}

function webUrl(value: unknown): string | null {
  if (typeof value !== "string" || value.length > MAX_URL_CHARS || !isWebUrl(value)) return null;
  try {
    new URL(value);
    return value;
  } catch {
    return null;
  }
}

const text = (value: unknown, max: number) => (typeof value === "string" ? value.slice(0, max) : "");
const id = (value: unknown) => (typeof value === "string" && value.length > 0 && value.length <= MAX_ID_CHARS ? value : null);

/** null for what a reload can't show: a file, a form post, a temporary chat's page. */
function keptEntry(entry: unknown): BrowserEntry | null {
  if (!entry || typeof entry !== "object") return null;
  const { kind, page, url, method, body, from, temporary } = entry as Record<string, unknown>;
  if (kind === "newtab") return { kind };
  if (kind === "internal") return typeof page === "string" && INTERNAL_PAGES.includes(page) ? { kind, page: page as InternalPage } : null;
  if (kind !== "web" || temporary || body !== undefined || (method !== undefined && method !== "GET")) return null;
  const target = webUrl(url);
  if (!target) return null;
  const source = webUrl(from);
  return source ? { kind, url: target, from: source } : { kind, url: target };
}

function keptHistory(entries: unknown[], at: number, displayUrl: unknown): { history: BrowserEntry[]; index: number } {
  let history: BrowserEntry[] = [];
  let index = 0;
  entries.forEach((item, position) => {
    const entry = keptEntry(item);
    if (!entry) return;
    // Where the page got to: native-view clicks and router moves update only displayUrl.
    const moved = position === at && entry.kind === "web" ? webUrl(displayUrl) : null;
    history.push(moved ? { kind: "web", url: moved } : entry);
    if (position <= at) index = history.length - 1;
  });
  if (history.length > MAX_ENTRIES) {
    const start = Math.max(0, Math.min(index - MAX_ENTRIES + 1, history.length - MAX_ENTRIES));
    history = history.slice(start, start + MAX_ENTRIES);
    index -= start;
  }
  return { history, index };
}

function keptTab(value: unknown): BrowserTab | null {
  if (!value || typeof value !== "object") return null;
  const tab = value as Record<string, unknown>;
  const tabId = id(tab.id);
  if (!tabId || typeof tab.chat !== "string" || tab.chat.length > MAX_ID_CHARS || !Array.isArray(tab.history)) return null;
  const last = tab.history.length - 1;
  const at = typeof tab.index === "number" && Number.isInteger(tab.index) ? Math.max(0, Math.min(tab.index, last)) : last;
  const { history, index } = keptHistory(tab.history, at, tab.displayUrl);
  const shown = history[index];
  if (!shown) return null;
  // Kept only while it still names the page shown, so opening that page again focuses this tab.
  const openKey = shown.kind === "web" ? `url:${shown.url}` : shown.kind === "internal" ? `internal:${shown.page}` : null;
  return {
    id: tabId,
    chat: tab.chat,
    history,
    index,
    title: text(tab.title, MAX_TITLE_CHARS),
    favicon: typeof tab.favicon === "string" && tab.favicon.length <= MAX_ICON_CHARS ? tab.favicon : null,
    documentType: null,
    displayUrl: null,
    loading: false,
    reloadKey: 0,
    openKey: openKey !== null && tab.openKey === openKey ? openKey : null,
    zoom: typeof tab.zoom === "number" && Number.isFinite(tab.zoom) ? Math.min(5, Math.max(0.25, tab.zoom)) : 1,
    nativeHistory: null,
    nativeError: null,
    customTitle: text(tab.customTitle, MAX_TITLE_CHARS) || null,
    muted: tab.muted === true,
    pinnedId: id(tab.pinnedId),
  };
}

function kept(tabList: unknown, panels: unknown, first?: string): TabSession {
  if (!Array.isArray(tabList)) return EMPTY;
  const seen = new Set<string>();
  const all = tabList.flatMap((value) => {
    const tab = keptTab(value);
    if (!tab || seen.has(tab.id)) return [];
    seen.add(tab.id);
    return [tab];
  });
  // Past the cap, the chat on screen keeps its tabs first.
  const budget = new Set([...all.filter((tab) => tab.chat === first), ...all.filter((tab) => tab.chat !== first)].slice(0, MAX_TABS));
  const tabs = all.filter((tab) => budget.has(tab));
  const chats: Record<string, ChatBrowser> = {};
  const saved = panels && typeof panels === "object" ? (panels as Record<string, unknown>) : {};
  for (const chat of new Set(tabs.map((tab) => tab.chat))) {
    const own = tabs.filter((tab) => tab.chat === chat);
    const panel = saved[chat] && typeof saved[chat] === "object" ? (saved[chat] as Record<string, unknown>) : null;
    if (!panel) continue;
    const open = panel.open === true;
    chats[chat] = {
      open,
      activeTabId: own.some((tab) => tab.id === panel.activeTabId) ? (panel.activeTabId as string) : (own.at(-1)?.id ?? null),
      fullView: open && panel.fullView === true,
      floating: panel.floating === true,
    };
  }
  return { tabs, chats, shown: typeof first === "string" && tabs.some((tab) => tab.chat === first) ? first : null };
}

export function serializeTabSession(state: TabSessionSource): string | null {
  const panels = {
    ...state.chats,
    [state.chat]: { open: state.open, activeTabId: state.activeTabId, fullView: state.fullView, floating: state.floating && !state.floatedForAgent },
  };
  const { tabs, chats, shown } = kept(state.tabs, panels, state.chat);
  if (tabs.length === 0) return null;
  const slim = tabs.map(({ id, chat, history, index, title, favicon, openKey, zoom, customTitle, muted, pinnedId }) => ({ id, chat, history, index, title, favicon, openKey, zoom, customTitle, muted, pinnedId }));
  return JSON.stringify({ version: VERSION, tabs: slim, chats, shown });
}

export function parseTabSession(raw: string | null): TabSession {
  if (!raw) return EMPTY;
  try {
    const saved = JSON.parse(raw) as { version?: unknown; tabs?: unknown; chats?: unknown; shown?: unknown } | null;
    if (!saved || typeof saved !== "object" || saved.version !== VERSION) return EMPTY;
    return kept(saved.tabs, saved.chats, typeof saved.shown === "string" ? saved.shown : undefined);
  } catch {
    return EMPTY;
  }
}

export function loadTabSession(key = TAB_SESSION_KEY, storage = sessionStore()): TabSession {
  try {
    return parseTabSession(storage?.getItem(key) ?? null);
  } catch {
    return EMPTY;
  }
}

let written: string | null | undefined;

export function saveTabSession(state: TabSessionSource, key = TAB_SESSION_KEY, storage = sessionStore()): void {
  try {
    const next = serializeTabSession(state);
    if (next === written) return;
    written = next;
    if (next === null) storage?.removeItem(key);
    else storage?.setItem(key, next);
  } catch {
    // Quota or private mode.
  }
}

/** Whether `chat` (a store chat key) was never saved, so no reload brings it back under that key. */
export async function chatIsUnsaved(chat: string): Promise<boolean> {
  // A project's new chat is keyed by a nonce made on mount (chat-page.tsx); saved, it is a single: chat.
  if (chat.startsWith("project:")) return true;
  if (!chat.startsWith("single:")) return false;
  try {
    const { getStoredChatThread } = await import("@/features/chat/utils/chat-history-storage");
    return !(await getStoredChatThread(chat.slice("single:".length)));
  } catch {
    return false;
  }
}

/** A project's unsaved chat hands its tabs only to that project's new chat. */
export function sameKindOfChat(from: string, to: string): boolean {
  const scope = (chat: string) => (chat.startsWith("project:") ? chat.slice(0, chat.lastIndexOf(":") + 1) : chat.slice(0, chat.indexOf(":") + 1));
  return scope(from) === scope(to);
}
