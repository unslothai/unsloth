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
  "appearance",
  "resources",
  "chat",
  "voice",
  "connections",
  "data",
  "api-keys",
  "remote-lan",
  "agents",
  "keyboard-shortcuts",
  "unforgettable",
  "debugging",
  "about",
] as const;

export type SettingsTab = (typeof SETTINGS_TABS)[number];

export type SettingsScrollTarget =
  | "about-updates"
  | "appearance-sidebar-nav"
  | "chat-canvas-network";

/** Which archive the Data tab should open straight into. */
export type ArchivedShelf = "chats" | "images" | "videos";

interface OpenDialogOptions {
  scrollTarget?: SettingsScrollTarget;
  focusFallback?: HTMLElement | null;
}

interface SettingsDialogState {
  open: boolean;
  activeTab: SettingsTab;
  scrollTarget: SettingsScrollTarget | null;
  // Element focused when openDialog() ran. Radix's FocusScope normally tracks
  // this, but the rAF-scheduled focus() in settings-dialog.tsx races its
  // previous-focus capture, leaving focus on <body> after close. We restore
  // explicitly via onCloseAutoFocus.
  opener: HTMLElement | null;
  openerFallback: HTMLElement | null;
  // Set when something asks to jump straight to an archive listing (the archive
  // toast). DataTab uses it as its initial subpage, then clears it. See requestsFor
  // for how long it lives unconsumed.
  archivedRequested: ArchivedShelf | null;
  openDialog: (tab?: SettingsTab, options?: OpenDialogOptions) => void;
  openArchivedChats: () => void;
  openArchivedMedia: (shelf: "images" | "videos") => void;
  consumeArchivedChatsRequest: () => void;
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
) {
  if (state.open) {
    return {
      opener: state.opener,
      openerFallback: state.openerFallback,
    };
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
  "about-updates": "about",
  "appearance-sidebar-nav": "appearance",
  "chat-canvas-network": "chat",
};

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
  };
}

export const useSettingsDialogStore = create<SettingsDialogState>((set) => ({
  open: false,
  activeTab: loadInitialTab(),
  scrollTarget: null,
  opener: null,
  openerFallback: null,
  archivedRequested: null,
  openDialog: (tab, options) =>
    set((state) => {
      const next = tab ?? state.activeTab;
      const pending = requestsFor(state, next);
      return {
        open: true,
        activeTab: next,
        // A caller that names a target replaces whatever was still pending.
        scrollTarget: options?.scrollTarget ?? pending.scrollTarget,
        archivedRequested: pending.archivedRequested,
        ...focusForOpen(state, options?.focusFallback),
      };
    }),
  openArchivedChats: () =>
    set((state) => ({
      open: true,
      activeTab: "data",
      scrollTarget: null,
      archivedRequested: "chats",
      ...focusForOpen(state),
    })),
  openArchivedMedia: (shelf) =>
    set((state) => ({
      open: true,
      activeTab: "data",
      scrollTarget: null,
      archivedRequested: shelf,
      ...focusForOpen(state),
    })),
  consumeArchivedChatsRequest: () => set({ archivedRequested: null }),
  consumeScrollTarget: (target) =>
    set((state) => ({
      scrollTarget: state.scrollTarget === target ? null : state.scrollTarget,
    })),
  // Do NOT clear `opener` here. onCloseAutoFocus runs on the next render
  // pass after `open: false` lands, so the opener must still be readable
  // from the store at that point. The next openDialog() overwrites it.
  closeDialog: () =>
    set({ open: false, scrollTarget: null, archivedRequested: null }),
  setActiveTab: (tab) => {
    try {
      window.localStorage.setItem(ACTIVE_TAB_KEY, tab);
    } catch {
      // ignore storage failures
    }
    set((state) => ({ activeTab: tab, ...requestsFor(state, tab) }));
  },
}));
