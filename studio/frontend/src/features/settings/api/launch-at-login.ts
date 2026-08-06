// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export async function loadLaunchAtLogin(): Promise<boolean> {
  const { isEnabled } = await import("@tauri-apps/plugin-autostart");
  return isEnabled();
}

export async function updateLaunchAtLogin(enabled: boolean): Promise<boolean> {
  const autostart = await import("@tauri-apps/plugin-autostart");
  if (enabled) {
    await autostart.enable();
  } else {
    await autostart.disable();
  }
  return autostart.isEnabled();
}
