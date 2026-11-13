// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The module, not the barrel: the barrel pulls in page components the store tests cannot load.
import { authFetch } from "@/features/auth/api";
import type { SttDevice } from "../stores/voice-settings-store";

type SttDeviceState = { device: SttDevice | "gpu"; saved?: boolean };

// Serialize saves so the last choice lands last.
let pending: Promise<unknown> = Promise.resolve();
// Bumped per save, so a read that started before one is not applied after it.
let revision = 0;

export function syncSttDevicePreference(
  device: SttDevice,
  onlyIfUnset = false,
  adopt?: (device: SttDevice) => void,
): void {
  const saving = ++revision;
  pending = pending
    .then(async () => {
      const response = await authFetch("/api/settings/stt-device", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          device,
          ...(onlyIfUnset ? { only_if_unset: true } : {}),
        }),
      });
      if (onlyIfUnset && response.ok) {
        const state = (await response.json()) as SttDeviceState;
        if (revision === saving && state.device !== device) {
          adopt?.(state.device === "cpu" ? "cpu" : "auto");
        }
      }
    })
    .catch(() => undefined);
}

// Adopt a saved server choice; otherwise migrate the local choice.
export async function reconcileSttDevicePreference(
  local: SttDevice,
  adopt: (device: SttDevice) => void,
): Promise<void> {
  const started = revision;
  try {
    await pending;
    const response = await authFetch("/api/settings/stt-device");
    if (!response.ok) {
      return;
    }
    const state = (await response.json()) as SttDeviceState;
    if (revision !== started) {
      return;
    }
    if (state.saved) {
      adopt(state.device === "cpu" ? "cpu" : "auto");
    } else if (local !== "auto") {
      syncSttDevicePreference(local, true, adopt);
    }
  } catch {
    // Offline or not the owner: the local choice still drives this browser's own loads.
  }
}
