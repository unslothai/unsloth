// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Applies persisted pill settings at startup so the hotkey works before settings is opened.

import { useEffect } from "react";
import { isTauri } from "@/lib/api-base";
import {
  fetchPillSettings,
  syncNativePillConfig,
  updatePillSettings,
  withNativeApplyLock,
} from "./api";
import { isMacPlatform, pillSetConfig, pillStatus } from "@/lib/pill-native";

// Throttle on last SUCCESS, so a failed sync does not swallow the port-triggered retry.
let lastSyncSucceededAt = 0;
let syncInFlight = false;
let syncQueued = false;

async function syncConfigToNative(): Promise<void> {
  // Queue a mid-attempt trigger: during startup it is often the only retry.
  if (syncInFlight) {
    syncQueued = true;
    return;
  }
  if (lastSyncSucceededAt && Date.now() - lastSyncSucceededAt < 30_000) return;
  syncInFlight = true;
  try {
    await withNativeApplyLock(async () => {
      let settings;
      try {
        settings = await fetchPillSettings();
      } catch {
        return;
      }
      try {
        await syncNativePillConfig(settings);
        lastSyncSucceededAt = Date.now();
      } catch {
        // Align the backend with native status, as in the settings tab.
        try {
          const status = await pillStatus();
          if (!status.supported || status.enabled === settings.enabled) return;
          await updatePillSettings({ enabled: status.enabled });
          // Also rewrite selection-pill.json: syncNativePillConfig would skip it, but init reads the file.
          await pillSetConfig({ enabled: status.enabled });
        } catch {
        }
      }
    });
  } finally {
    syncInFlight = false;
    if (syncQueued) {
      syncQueued = false;
      void syncConfigToNative();
    }
  }
}

export function PillConfigSync(): null {
  useEffect(() => {
    if (!isTauri || !isMacPlatform()) return;
    let disposed = false;
    let unlisten: (() => void) | undefined;

    const timer = setTimeout(() => void syncConfigToNative(), 3000);
    void import("@tauri-apps/api/event")
      .then(({ listen }) =>
        listen("server-port", () => {
          setTimeout(() => void syncConfigToNative(), 2000);
        }),
      )
      .then((cleanup) => {
        if (disposed) cleanup();
        else unlisten = cleanup;
      })
      .catch(() => undefined);

    return () => {
      disposed = true;
      clearTimeout(timer);
      unlisten?.();
    };
  }, []);

  return null;
}
