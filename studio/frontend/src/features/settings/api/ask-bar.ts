// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** null off macOS, where the desktop app has no Ask bar. */
export async function loadAskBar(): Promise<boolean | null> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean | null>("get_ask_bar");
}

export async function updateAskBar(enabled: boolean): Promise<boolean> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean>("set_ask_bar", { enabled });
}
