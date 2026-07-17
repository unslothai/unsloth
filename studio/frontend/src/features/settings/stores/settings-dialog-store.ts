// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { captureFocusedElement } from "@/lib/focus";
import { create } from "zustand";

export type SettingsTab =
  | "general"
  | "profile"
  | "appearance"
  | "resources"
  | "chat"
  | "voice"
  | "connections"
  | "api-keys"
  | "about";

export type SettingsScrollTarget = "about-updates";

interface OpenDialogOptions {
  scrollTarget?: SettingsScrollTarget;
  // Focus-restore target overriding the activeElement capture. Needed when
  // the caller is itself about to unmount (command palette): activeElement
  // still points inside it, so capturing here would grab a dead element.
  opener?: HTMLElement | null;
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
  // archive toast). ChatTab consumes it to open the dialog, then clears it.
  archivedChatsRequested: boolean;
  openDialog: (tab?: SettingsTab, options?: OpenDialogOptions) => void;
  openArchivedChats: () => void;
  consumeArchivedChatsRequest: () => void;
  consumeScrollTarget: (target: SettingsScrollTarget) => void;
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
    "resources",
    "chat",
    "voice",
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
  scrollTarget: null,
  opener: null,
  archivedChatsRequested: false,
  openDialog: (tab, options) =>
    set((state) => ({
      open: true,
      activeTab: tab ?? state.activeTab,
      scrollTarget: options?.scrollTarget ?? null,
      opener:
        options?.opener !== undefined
          ? options.opener
          : captureFocusedElement(),
    })),
  openArchivedChats: () =>
    set({
      open: true,
      activeTab: "chat",
      scrollTarget: null,
      archivedChatsRequested: true,
      opener: captureFocusedElement(),
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
