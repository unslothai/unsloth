// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";

/**
 * Forget a saved "Don't allow" so the next getUserMedia can prompt again.
 *
 * WebView2 keeps that answer in its profile and offers no site-settings UI, so without
 * this an accidental deny blocked dictation permanently (#9001). The command only clears
 * the stored answer; access still comes from the prompt.
 *
 * Best effort by design: a browser tab has no command to call, and an older WebView2
 * runtime has no permission API, but getUserMedia is still worth trying in both.
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
