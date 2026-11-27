// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";

/**
 * Forgets a saved "Don't allow" (no UI in WebView2); best effort: older runtimes lack the command.
 */
export async function resetMicrophonePermission(): Promise<void> {
  if (!isTauri) return;
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("reset_microphone_permission");
  } catch (error) {
    console.warn("Could not reset the saved microphone permission:", error);
  }
}
