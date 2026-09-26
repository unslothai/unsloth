// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type SettingsTab, settingsTabVisible, useSettingsDialogStore } from "@/features/settings";
import type { TranslationKey } from "@/i18n";
import { openLink } from "@/lib/open-link";
import {
  ActivityIcon,
  Book03Icon,
  KeyboardIcon,
  MessageNotification01Icon,
  NewReleasesIcon,
  Wrench01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

/** The Help menu: the sidebar's account menu and the desktop app's Help menu list the same items
 *  (src-tauri/src/app_menu.rs, HELP_ROWS), grouped the same way. */
export type HelpAction =
  | "help-documentation"
  | "help-keyboard-shortcuts"
  | "help-whats-new"
  | "help-troubleshooting"
  | "help-system-status"
  | "help-send-feedback";

export const HELP_GROUPS: HelpAction[][] = [
  ["help-documentation", "help-keyboard-shortcuts", "help-whats-new"],
  ["help-troubleshooting", "help-system-status", "help-send-feedback"],
];

export const HELP_ITEMS: Record<HelpAction, { label: TranslationKey; icon: IconSvgElement }> = {
  "help-documentation": { label: "shell.helpMenu.documentation", icon: Book03Icon },
  "help-keyboard-shortcuts": { label: "shell.helpMenu.keyboardShortcuts", icon: KeyboardIcon },
  "help-whats-new": { label: "shell.helpMenu.whatsNew", icon: NewReleasesIcon },
  "help-troubleshooting": { label: "shell.helpMenu.troubleshooting", icon: Wrench01Icon },
  "help-system-status": { label: "shell.helpMenu.systemStatus", icon: ActivityIcon },
  "help-send-feedback": { label: "shell.helpMenu.sendFeedback", icon: MessageNotification01Icon },
};

/** The Settings page an action opens. */
const HELP_SETTINGS_TABS: Partial<Record<HelpAction, SettingsTab>> = {
  "help-keyboard-shortcuts": "keyboard-shortcuts",
  // The server log, where a failed load or generation usually says why.
  "help-troubleshooting": "debugging",
  // Live hardware, memory and storage for this server.
  "help-system-status": "resources",
};

/** False when the action's Settings page is owner-only and this account cannot open it. */
export function helpActionAvailable(action: HelpAction, isOwner: boolean): boolean {
  const tab = HELP_SETTINGS_TABS[action];
  return !tab || settingsTabVisible(tab, isOwner);
}

export function runHelpAction(action: HelpAction): void {
  const tab = HELP_SETTINGS_TABS[action];
  if (tab) {
    useSettingsDialogStore.getState().openDialog(tab);
    return;
  }
  switch (action) {
    case "help-documentation":
      openLink("https://unsloth.ai/docs");
      return;
    case "help-whats-new":
      openLink("https://unsloth.ai/docs/new/changelog");
      return;
    case "help-send-feedback":
      openLink("https://github.com/unslothai/unsloth/issues");
      return;
  }
}
