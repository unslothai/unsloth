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
  | "about";

export type SettingsScrollTarget = "about-updates";

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
  // Set when something asks to jump straight to the archived chats list (the
  // archive toast). DataTab uses it as its initial subpage, then clears it.
  archivedChatsRequested: boolean;
  openDialog: (tab?: SettingsTab, options?: OpenDialogOptions) => void;
  openArchivedChats: () => void;
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
  archivedChatsRequested: false,
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
      archivedChatsRequested: true,
      opener: captureOpener(),
    }),
  consumeArchivedChatsRequest: () => set({ archivedChatsRequested: false }),
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
