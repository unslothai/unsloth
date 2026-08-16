// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pushes backend-persisted pill settings to the Rust layer on startup so the
// hotkey works before the settings UI is ever opened.

import { useEffect } from "react";
import { isTauri } from "@/lib/api-base";
import {
  fetchPillSettings,
  syncNativePillConfig,
  updatePillSettings,
} from "./api";
import { isMacPlatform, pillSetConfig, pillStatus } from "@/lib/pill-native";

// server-port re-announcements can arrive in bursts; one sync per window. The
// throttle keys on the last SUCCESS, not the last attempt: a failed startup
// sync used to bank 30s of silence and swallow the retry the port triggers,
// leaving the hotkey stale for the window's lifetime.
let lastSyncSucceededAt = 0;
let syncInFlight = false;
let syncQueued = false;

async function syncConfigToNative(): Promise<void> {
  // A trigger arriving mid-attempt is remembered, not dropped: during startup
  // it is often the only retry there will be, and the attempt it overlaps is
  // the one most likely to fail on a backend that is not up yet.
  if (syncInFlight) {
    syncQueued = true;
    return;
  }
  if (lastSyncSucceededAt && Date.now() - lastSyncSucceededAt < 30_000) return;
  syncInFlight = true;
  try {
    // Read and apply are separated because they fail for different reasons: a
    // failed read means the backend is not up yet and the next trigger retries,
    // while a failed apply means the shortcut could not be taken.
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
      // Nothing here renders native status, so a backend left saying enabled
      // would keep the settings switch claiming a bar with no shortcut behind
      // it. Undo it, exactly as the settings tab does on the same failure.
      if (!settings.enabled) return;
      try {
        const reverted = await updatePillSettings({ enabled: false });
        // Write the disabled config to disk. Rust reports disabled after a
        // failed registration but selection-pill.json still says enabled, and
        // that file is what init reads: left alone it would re-register the
        // shortcut on a later launch, against a UI and backend now saying
        // disabled. syncNativePillConfig cannot do this, because the managed
        // status is already disabled and its equality check would skip the
        // write that is the entire point, so call the command directly.
        const status = await pillStatus();
        if (status.supported) {
          await pillSetConfig({
            enabled: false,
            hotkey: status.hotkey,
            excludedApps: reverted.excludedApps,
          });
        }
      } catch {
        // Could not undo it either; the next open reads the backend.
      }
    }
  } finally {
    syncInFlight = false;
    if (syncQueued) {
      syncQueued = false;
      // Re-enters the throttle above, so a successful attempt still collapses
      // the queued one instead of running it back to back.
      void syncConfigToNative();
    }
  }
}

export function PillConfigSync(): null {
  useEffect(() => {
    if (!isTauri || !isMacPlatform()) return;
    let disposed = false;
    let unlisten: (() => void) | undefined;

    // Give auth/backend startup a moment, then sync; also re-sync when the
    // backend port is (re)announced.
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
