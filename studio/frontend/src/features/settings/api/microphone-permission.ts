// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";

/**
 * Forget a saved "Don't allow": WebView2 keeps it with no site-settings UI. Best effort, since
 * browsers and older WebView2 runtimes lack the command but getUserMedia may still prompt.
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
