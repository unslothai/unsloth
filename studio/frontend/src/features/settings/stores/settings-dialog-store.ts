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
  // toast). DataTab uses it as its initial subpage, then clears it.
  archivedRequested: ArchivedShelf | null;
  // Set when a failure elsewhere in the app offers "View logs". The Logs tab reads it
  // as its initial source family, then clears it. A FAMILY rather than a source id:
  // ids are a digest of the real path the frontend cannot compute, and at the moment
  // of a failure the newest file in the family is the attempt that just failed.
  logFamilyRequested: string | null;
  openDialog: (tab?: SettingsTab, options?: OpenDialogOptions) => void;
  openArchivedChats: () => void;
  openArchivedMedia: (shelf: "images" | "videos") => void;
  consumeArchivedChatsRequest: () => void;
  openLogs: (family?: string) => void;
  consumeLogFamilyRequest: () => void;
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

export const useSettingsDialogStore = create<SettingsDialogState>((set) => ({
  open: false,
  activeTab: loadInitialTab(),
  scrollTarget: null,
  opener: null,
  archivedRequested: null,
  logFamilyRequested: null,
  openDialog: (tab, options) =>
    set((state) => ({
      open: true,
      activeTab: tab ?? state.activeTab,
      scrollTarget: options?.scrollTarget ?? null,
      opener: captureOpener(),
    })),
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
  openLogs: (family) =>
    set({
      open: true,
      activeTab: "debugging",
      scrollTarget: null,
      logFamilyRequested: family ?? null,
      opener: captureOpener(),
    }),
  consumeLogFamilyRequest: () => set({ logFamilyRequested: null }),
  consumeScrollTarget: (target) =>
    set((state) => ({
      scrollTarget: state.scrollTarget === target ? null : state.scrollTarget,
    })),
  // Do NOT clear `opener` here. onCloseAutoFocus runs on the next render
  // pass after `open: false` lands, so the opener must still be readable
  // from the store at that point. The next openDialog() overwrites it.
  closeDialog: () => set({ open: false, scrollTarget: null }),
  setActiveTab: (tab) => {
    try {
      window.localStorage.setItem(ACTIVE_TAB_KEY, tab);
    } catch {
      // ignore storage failures
    }
    set({ activeTab: tab, scrollTarget: null });
  },
}));
