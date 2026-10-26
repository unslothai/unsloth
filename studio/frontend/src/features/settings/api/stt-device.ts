// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The module, not the barrel: the barrel pulls in page components the store tests cannot load.
import { authFetch } from "@/features/auth/api";
import type { SttDevice } from "../stores/voice-settings-store";

// Soft-fails: localStorage stays the UI's source of truth (#15280).
export function syncSttDevicePreference(device: SttDevice): void {
  void authFetch("/api/settings/stt-device", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device }),
  }).catch(() => {});
}
