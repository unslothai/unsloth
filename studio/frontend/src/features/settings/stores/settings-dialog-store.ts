// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export type SettingsTab =
  | "general"
  | "profile"
  | "appearance"
  | "chat"
  | "connections"
  | "api-keys"
  | "about";

interface SettingsDialogState {
  open: boolean;
  activeTab: SettingsTab;
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
  openDialog: (tab) =>
    set((state) => ({
      open: true,
      activeTab: tab ?? state.activeTab,
    })),
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
