// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type { SttDevice } from "../stores/voice-settings-store";

/**
 * Persist Voice Settings "Load into" on the server so a cold
 * `/v1/audio/transcriptions` load honours the same placement as the Settings
 * Load button (#15280). Soft-fails: localStorage remains the UI source of truth.
 */
export function syncSttDevicePreference(device: SttDevice): void {
  void authFetch("/api/settings/stt-device", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device }),
  }).catch(() => {});
}
