// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export type SettingsTab =
  | "general"
  | "profile"
  | "appearance"
  | "resources"
  | "chat"
  | "voice"
  | "connections"
  | "data"
  | "api-keys"
  | "agents"
  | "debugging"
  | "about";

export type SettingsScrollTarget = "about-updates" | "appearance-sidebar-nav";

/** Which archive the Data tab should open straight into. */
export type ArchivedShelf = "chats" | "images" | "videos";

interface OpenDialogOptions {
  scrollTarget?: SettingsScrollTarget;
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
  // Set when something asks to jump straight to an archive listing (the archive
  // toast). DataTab uses it as its initial subpage, then clears it. See
  // archivedFor for how long an unconsumed one lives.
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

const ACTIVE_TAB_KEY = "unsloth_settings_active_tab";

function loadInitialTab(): SettingsTab {
  if (typeof window === "undefined") return "general";
  let stored: string | null = null;
  try {
    stored = window.localStorage.getItem(ACTIVE_TAB_KEY);
  } catch {
    return "general";
  }
  const valid: SettingsTab[] = [
    "general",
    "profile",
    "appearance",
    "resources",
    "chat",
    "voice",
    "connections",
    "data",
    "api-keys",
    "agents",
    "debugging",
    "about",
  ];
  return valid.includes(stored as SettingsTab)
    ? (stored as SettingsTab)
    : "general";
}

/**
 * An unconsumed archive request, after a navigation that lands on `tab`.
 *
 * Only DataTab clears the request, and the panel is fetched on first view, so a dialog
 * closed or a tab left before it arrives strands the flag and the next ordinary visit to
 * Data opens an archive listing nobody asked for. The request is for Data, so it lives
 * exactly as long as the dialog is open on Data: reselecting Data goes nowhere and keeps
 * it, and closing (below) always drops it, since nothing is left to read it.
 */
function archivedFor(
  state: SettingsDialogState,
  tab: SettingsTab,
): ArchivedShelf | null {
  return tab === "data" ? state.archivedRequested : null;
}

export const useSettingsDialogStore = create<SettingsDialogState>((set) => ({
  open: false,
  activeTab: loadInitialTab(),
  scrollTarget: null,
  opener: null,
  archivedRequested: null,
  openDialog: (tab, options) =>
    set((state) => {
      const next = tab ?? state.activeTab;
      return {
        open: true,
        activeTab: next,
        scrollTarget: options?.scrollTarget ?? null,
        archivedRequested: archivedFor(state, next),
        opener: captureOpener(),
      };
    }),
  openArchivedChats: () =>
    set({
      open: true,
      activeTab: "data",
      scrollTarget: null,
      archivedRequested: "chats",
      opener: captureOpener(),
    }),
  openArchivedMedia: (shelf) =>
    set({
      open: true,
      activeTab: "data",
      scrollTarget: null,
      archivedRequested: shelf,
      opener: captureOpener(),
    }),
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
    set((state) => ({
      activeTab: tab,
      scrollTarget: null,
      archivedRequested: archivedFor(state, tab),
    }));
  },
}));
