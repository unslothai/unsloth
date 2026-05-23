// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export type SettingsTab =
  | "general"
  | "profile"
  | "appearance"
  | "chat"
  | "knowledge-bases"
  | "connections"
  | "api-keys"
  | "about";

interface SettingsDialogState {
  open: boolean;
  activeTab: SettingsTab;
  // Element focused at the moment openDialog() ran. Radix's FocusScope
  // would normally track this, but the rAF-scheduled focus() in
  // settings-dialog.tsx races its previous-focus capture, leaving focus
  // on <body> after close. We restore explicitly via onCloseAutoFocus.
  opener: HTMLElement | null;
  openDialog: (tab?: SettingsTab) => void;
  closeDialog: () => void;
  setActiveTab: (tab: SettingsTab) => void;
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
    "chat",
    "knowledge-bases",
    "connections",
    "api-keys",
    "about",
  ];
  return valid.includes(stored as SettingsTab)
    ? (stored as SettingsTab)
    : "general";
}

export const useSettingsDialogStore = create<SettingsDialogState>((set) => ({
  open: false,
  activeTab: loadInitialTab(),
  opener: null,
  openDialog: (tab) =>
    set((state) => ({
      open: true,
      activeTab: tab ?? state.activeTab,
      opener:
        typeof document !== "undefined" &&
        document.activeElement instanceof HTMLElement &&
        document.activeElement !== document.body
          ? document.activeElement
          : null,
    })),
  // Do NOT clear `opener` here. onCloseAutoFocus runs on the next render
  // pass after `open: false` lands, so the opener must still be readable
  // from the store at that point. The next openDialog() overwrites it.
  closeDialog: () => set({ open: false }),
  setActiveTab: (tab) => {
    try {
      window.localStorage.setItem(ACTIVE_TAB_KEY, tab);
    } catch {
      // ignore storage failures
    }
    set({ activeTab: tab });
  },
}));
