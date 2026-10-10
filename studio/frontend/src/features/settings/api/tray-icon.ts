// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export async function loadTrayIconVisible(): Promise<boolean | null> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean | null>("get_tray_icon_visible");
}

export async function updateTrayIconVisible(visible: boolean): Promise<boolean> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean>("set_tray_icon_visible", { enabled: visible });
}
