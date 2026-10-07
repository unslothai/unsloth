// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/**
 * One list, so the type and the persisted-tab check cannot drift: a tab added
 * to the union alone used to be rejected on reload and fall back to General.
 */
export const SETTINGS_TABS = [
  "general",
  "profile",
  "accounts",
  "appearance",
  "resources",
  "chat",
  "sandbox",
  "voice",
  "connections",
  "library",
  "data",
  "api-keys",
  "remote-lan",
  "agents",
  "keyboard-shortcuts",
  "browser",
  "debugging",
  "about",
] as const;

export type SettingsTab = (typeof SETTINGS_TABS)[number];

export type SettingsScrollTarget =
  | "about-updates"
  | "api-keys-audio-api"
  | "api-keys-decision-api"
  | "appearance-sidebar-nav"
  | "chat-composer"
  | "browser-html-network"
  | "general-hub"
  | "general-rag-embedding"
  /** Old name of sandbox-permissions, from when Permissions lived in General. */
  | "general-permissions"
  | "library-storage"
  | "resources-caches"
  | "sandbox-permissions";

export interface AudioApiRequest {
  workflow:
    | "speak"
    | "clone"
    | "edit"
    | "convert"
    | "music"
    | "separate"
    | "transcribe";
  model: string | null;
}

/** Which archive the Data tab should open straight into. */
export type ArchivedShelf = "chats" | "images" | "videos" | "audio";

interface OpenDialogOptions {
  scrollTarget?: SettingsScrollTarget;
  focusFallback?: HTMLElement | null;
  opener?: HTMLElement | null;
}

interface SettingsDialogState {
  open: boolean;
  activeTab: SettingsTab;
  scrollTarget: SettingsScrollTarget | null;
  // Element focused when openDialog() ran. Radix's FocusScope normally tracks this, but the
  // rAF-scheduled focus() in settings-dialog.tsx races its previous-focus capture, leaving focus on
  // <body> after close. We restore explicitly via onCloseAutoFocus.
  opener: HTMLElement | null;
  openerFallback: HTMLElement | null;
  // Set when something asks to jump straight to an archive listing (the archive
  // toast). DataTab uses it as its initial subpage, then clears it. See requestsFor
  // for how long it lives unconsumed.
  archivedRequested: ArchivedShelf | null;
  logFamilyRequested: string | null;
  logSourcePathRequested: string | null;
  /** Bumped per View logs click, so a repeated identical request still reads as new. */
  logRequestSeq: number;
  // Set when something asks for one connection's settings (the picker's Connected group gear).
  // ConnectionsTab hands it to the form, then clears it. Same lifetime as archivedRequested.
  connectionRequested: string | null;
  // The Audio API card applies it, then clears it. Same lifetime as archivedRequested.
  audioApiRequested: AudioApiRequest | null;
  openDialog: (tab?: SettingsTab, options?: OpenDialogOptions) => void;
  openArchivedChats: () => void;
  openArchivedMedia: (shelf: Exclude<ArchivedShelf, "chats">) => void;
  /** Open Connections with `providerId`'s edit form already up. */
  openConnectionSettings: (providerId: string) => void;
  openAudioApi: (request: AudioApiRequest) => void;
  consumeAudioApiRequest: () => void;
  consumeArchivedChatsRequest: () => void;
  openLogs: (family?: string, sourcePath?: string | null) => void;
  consumeLogFamilyRequest: () => void;
  consumeConnectionRequest: () => void;
  consumeScrollTarget: (target: SettingsScrollTarget) => void;
  closeDialog: () => void;
  setActiveTab: (tab: SettingsTab) => void;
}

function captureOpener(): HTMLElement | null {
  return typeof document !== "undefined" &&
    document.activeElement instanceof HTMLElement &&
    document.activeElement !== document.body
    ? document.activeElement
    : null;
}

function focusForOpen(
  state: SettingsDialogState,
  requestedFallback: HTMLElement | null = null,
  requestedOpener?: HTMLElement | null,
) {
  if (state.open) {
    return {
      opener: state.opener,
      openerFallback: state.openerFallback,
    };
  }
  // Handoff from a dialog that closed first (the command palette).
  if (requestedOpener !== undefined) {
    return { opener: requestedOpener, openerFallback: requestedFallback };
  }
  const opener = captureOpener();
  if (opener?.closest("[data-slot=dialog-content]")) {
    return {
      opener: state.opener,
      openerFallback: state.openerFallback,
    };
  }
  return { opener, openerFallback: requestedFallback };
}

const ACTIVE_TAB_KEY = "unsloth_settings_active_tab";

function loadInitialTab(): SettingsTab {
  if (typeof window === "undefined") return "general";
  let stored: string | null = null;
  try {
    stored = window.localStorage.getItem(ACTIVE_TAB_KEY);
  } catch {
    return "general";
  }
  return (SETTINGS_TABS as readonly string[]).includes(stored ?? "")
    ? (stored as SettingsTab)
    : "general";
}

/** The panel that delivers each scroll target, so a navigation elsewhere abandons it. */
const SCROLL_TARGET_TAB: Record<SettingsScrollTarget, SettingsTab> = {
  "chat-composer": "chat",
  "about-updates": "about",
  "api-keys-audio-api": "api-keys",
  "api-keys-decision-api": "api-keys",
  "appearance-sidebar-nav": "appearance",
  "browser-html-network": "browser",
  "general-hub": "general",
  "general-rag-embedding": "general",
  "general-permissions": "sandbox",
  "library-storage": "library",
  "resources-caches": "resources",
  "sandbox-permissions": "sandbox",
};

/** Permissions moved from General to the top of Sandbox; an old link still lands there. */
export function resolveScrollRequest(
  tab: SettingsTab | undefined,
  target: SettingsScrollTarget | undefined,
): { tab: SettingsTab | undefined; target: SettingsScrollTarget | undefined } {
  if (target === "general-permissions" || target === "sandbox-permissions") {
    return { tab: "sandbox", target: "sandbox-permissions" };
  }
  return { tab, target };
}

/**
 * The unconsumed deep-link requests that outlive a navigation landing on `tab`.
 *
 * Only the panel that performs a jump clears its request, and panels are fetched on first
 * view, so a navigation can move before the chunk arrives. A request therefore lives while
 * the dialog is open on the tab that reads it: reselecting keeps it, anything else drops
 * it, and closing (below) always does. Held wider, a stale request replays on a later
 * visit; held narrower, reselecting loses a deep-link still in flight.
 */
function requestsFor(state: SettingsDialogState, tab: SettingsTab) {
  return {
    scrollTarget:
      state.scrollTarget && SCROLL_TARGET_TAB[state.scrollTarget] === tab
        ? state.scrollTarget
        : null,
    archivedRequested: tab === "data" ? state.archivedRequested : null,
    logFamilyRequested: tab === "debugging" ? state.logFamilyRequested : null,
    logSourcePathRequested:
      tab === "debugging" ? state.logSourcePathRequested : null,
    connectionRequested:
      tab === "connections" ? state.connectionRequested : null,
    audioApiRequested: tab === "api-keys" ? state.audioApiRequested : null,
  };
}

export const NO_PENDING_LOG_REQUEST = "|";

export function pendingLogRequestKey(state: {
  logFamilyRequested: string | null;
  logSourcePathRequested: string | null;
  logRequestSeq?: number;
}): string {
  if (state.logFamilyRequested == null && state.logSourcePathRequested == null)
    return NO_PENDING_LOG_REQUEST;
  return `${state.logFamilyRequested ?? ""}|${state.logSourcePathRequested ?? ""}|${state.logRequestSeq ?? 0}`;
}


export const useSettingsDialogStore = create<SettingsDialogState>((set) => ({
  open: false,
  activeTab: loadInitialTab(),
  scrollTarget: null,
  opener: null,
  openerFallback: null,
  archivedRequested: null,
  logFamilyRequested: null,
  logSourcePathRequested: null,
  logRequestSeq: 0,
  connectionRequested: null,
  audioApiRequested: null,
  openDialog: (requestedTab, options) =>
    set((state) => {
      const { tab, target } = resolveScrollRequest(requestedTab, options?.scrollTarget);
      const next = tab ?? state.activeTab;
      const pending = requestsFor(state, next);
      return {
        open: true,
        activeTab: next,
        // A caller that names a target replaces whatever was still pending.
        scrollTarget: target ?? pending.scrollTarget,
        archivedRequested: pending.archivedRequested,
        logFamilyRequested: pending.logFamilyRequested,
        logSourcePathRequested: pending.logSourcePathRequested,
        connectionRequested: pending.connectionRequested,
        audioApiRequested: pending.audioApiRequested,
        ...focusForOpen(state, options?.focusFallback, options?.opener),
      };
    }),
  openArchivedChats: () =>
    set((state) => ({
      open: true,
      activeTab: "data",
      scrollTarget: null,
      archivedRequested: "chats",
      logFamilyRequested: null,
      logSourcePathRequested: null,
      connectionRequested: null,
      audioApiRequested: null,
      ...focusForOpen(state),
    })),
  openArchivedMedia: (shelf) =>
    set((state) => ({
      open: true,
      activeTab: "data",
      scrollTarget: null,
      archivedRequested: shelf,
      logFamilyRequested: null,
      logSourcePathRequested: null,
      connectionRequested: null,
      audioApiRequested: null,
      ...focusForOpen(state),
    })),
  openConnectionSettings: (providerId) =>
    set((state) => ({
      open: true,
      activeTab: "connections",
      scrollTarget: null,
      archivedRequested: null,
      logFamilyRequested: null,
      logSourcePathRequested: null,
      connectionRequested: providerId,
      audioApiRequested: null,
      ...focusForOpen(state),
    })),
  openAudioApi: (request) =>
    set((state) => ({
      open: true,
      activeTab: "api-keys",
      scrollTarget: "api-keys-audio-api",
      archivedRequested: null,
      logFamilyRequested: null,
      logSourcePathRequested: null,
      connectionRequested: null,
      audioApiRequested: request,
      ...focusForOpen(state),
    })),
  consumeAudioApiRequest: () => set({ audioApiRequested: null }),
  consumeArchivedChatsRequest: () => set({ archivedRequested: null }),
  openLogs: (family, sourcePath) =>
    set((state) => ({
      open: true,
      activeTab: "debugging",
      scrollTarget: null,
      archivedRequested: null,
      logFamilyRequested: family ?? null,
      logSourcePathRequested: sourcePath ?? null,
      logRequestSeq: state.logRequestSeq + 1,
      connectionRequested: null,
      audioApiRequested: null,
      ...focusForOpen(state),
    })),
  consumeLogFamilyRequest: () =>
    set({ logFamilyRequested: null, logSourcePathRequested: null }),
  consumeConnectionRequest: () => set({ connectionRequested: null }),
  consumeScrollTarget: (target) =>
    set((state) => ({
      scrollTarget: state.scrollTarget === target ? null : state.scrollTarget,
    })),
  // Do NOT clear `opener` here. onCloseAutoFocus runs on the next render
  // pass after `open: false` lands, so the opener must still be readable
  // from the store at that point. The next openDialog() overwrites it.
  closeDialog: () =>
    set({
      open: false,
      scrollTarget: null,
      archivedRequested: null,
      logFamilyRequested: null,
      logSourcePathRequested: null,
      connectionRequested: null,
      audioApiRequested: null,
    }),
  setActiveTab: (tab) => {
    try {
      window.localStorage.setItem(ACTIVE_TAB_KEY, tab);
    } catch {
      // ignore storage failures
    }
    set((state) => ({ activeTab: tab, ...requestsFor(state, tab) }));
  },
}));
