// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { SettingsTab } from "./stores/settings-dialog-store";

// Tabs whose every backing route is owner-only (installation-wide settings), so a managed account
// would only see controls that fail with 403. Sandbox is not one: its Permissions section is per
// account, and only the OS sandbox sections below it are owner-only (SandboxTab skips them).
export const OWNER_ONLY_SETTINGS_TABS: ReadonlySet<SettingsTab> = new Set<SettingsTab>([
  "accounts",
  "resources",
  "remote-lan",
  "agents",
  "debugging",
]);

export function settingsTabVisible(tab: SettingsTab, isOwner: boolean): boolean {
  return isOwner || !OWNER_ONLY_SETTINGS_TABS.has(tab);
}

export function resolveSettingsTab(requested: SettingsTab, isOwner: boolean): SettingsTab {
  return settingsTabVisible(requested, isOwner) ? requested : "general";
}
